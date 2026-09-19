"""Execute an idempotent sandbox invocation.

Resolve and upload assets outside this module. Logical slots share a cache while
the runtime serializes physical container use.
"""

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import PurePosixPath
import re
import tarfile
import threading
from typing import TYPE_CHECKING, Optional

from agent_sandbox.config import MAX_RETAINED_OUTPUT_BYTES, RuntimeResourceNames, runtime_resource_names
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef
from agent_sandbox.omp_runner import to_output_events
from agent_sandbox.profile import load_profile
from agent_sandbox.runtime import (
    AgentSandboxError,
    ManagedExecutionRequest,
    PromptImage,
    SandboxRuntime,
    _is_safe_archive_path,
    _normalize_prompt_images,
    _validate_restore_archive_member,
)
from agent_sandbox.runtime import run_python_script as _run_python_script

if TYPE_CHECKING:
    from agent_sandbox.outputs import OutputEvent
    from agent_sandbox.profile import Profile
    from agent_sandbox.runtime import ExecResult


class _InProcessRunStore:
    """Cache terminal runs and serialize callers of each logical slot."""

    def __init__(self) -> None:
        self._completed: dict[tuple[str, str], "SandboxRun"] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def get(self, slot_key: str, run_id: str) -> Optional["SandboxRun"]:
        return self._completed.get((slot_key, run_id))

    def put(self, slot_key: str, run_id: str, result: "SandboxRun") -> None:
        self._completed[(slot_key, run_id)] = result

    def discard(self, slot_key: str, run_id: str, result: "SandboxRun") -> None:
        """Evict a released result without deleting a later retry."""
        key = (slot_key, run_id)
        if self._completed.get(key) is result:
            del self._completed[key]

    def slot_lock(self, slot_key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(slot_key, threading.Lock())


_MAX_APPEND_SYSTEM_BYTES = 64 * 1024
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SandboxInvocation:
    """Describe one caller-requested sandbox run."""

    slot_key: str  # Identify the deterministic caller-owned sandbox slot.
    profile_name: str
    prompt: str
    run_id: str  # Reuse the upstream tool-call identity for idempotency.
    active_manifest: Optional[SandboxManifest] = None
    workspace_archive_path: Optional[str] = None  # Supply the caller-downloaded workspace snapshot.
    prompt_images: tuple[PromptImage | tuple[str, bytes], ...] = ()
    model: str | None = None
    thinking: str | None = None
    append_system: str | None = None
    lean: bool = False  # Skip OMP discovery the caller's workflow does not use.
    workspace_archive_sha256: str | None = None  # Verify a first-invocation archive without a prior manifest.

    def __post_init__(self) -> None:
        _require_nonempty_string("slot_key", self.slot_key)
        _require_nonempty_string("run_id", self.run_id)
        if not isinstance(self.profile_name, str) or not _PROFILE_NAME_PATTERN.fullmatch(self.profile_name):
            raise ValueError("profile_name must be a safe profile name")
        _require_nonempty_string("prompt", self.prompt)
        _require_optional_string("model", self.model)
        _require_optional_string("thinking", self.thinking)
        object.__setattr__(self, "prompt_images", _normalize_prompt_images(self.prompt_images))
        if self.workspace_archive_path is not None:
            _require_nonempty_string("workspace_archive_path", self.workspace_archive_path)
        if self.workspace_archive_sha256 is not None and (
            not isinstance(self.workspace_archive_sha256, str) or not _SHA256_PATTERN.fullmatch(self.workspace_archive_sha256)
        ):
            raise ValueError("workspace_archive_sha256 must be a 64-character lowercase hexadecimal digest")
        if self.active_manifest is not None and not isinstance(self.active_manifest, SandboxManifest):
            raise TypeError("active_manifest must be a SandboxManifest or None")
        if self.active_manifest is None:
            if self.workspace_archive_path is None and self.workspace_archive_sha256 is not None:
                raise ValueError("workspace_archive_sha256 requires workspace_archive_path")
            if self.workspace_archive_path is not None and self.workspace_archive_sha256 is None:
                raise ValueError("workspace_archive_path requires workspace_archive_sha256 without an active manifest")
        else:
            if self.workspace_archive_sha256 is not None and self.workspace_archive_sha256 != self.active_manifest.workspace.sha256:
                raise ValueError("workspace_archive_sha256 conflicts with active_manifest workspace sha256")
            if self.workspace_archive_path is None:
                raise ValueError("active_manifest requires workspace_archive_path")
        if self.append_system is None:
            return
        if not isinstance(self.append_system, str):
            raise TypeError("append_system must be a string or None")
        if len(self.append_system.encode("utf-8")) > _MAX_APPEND_SYSTEM_BYTES:
            raise ValueError("append_system must not exceed 64 KiB")


@dataclass(frozen=True)
class SandboxRun:
    status: str  # Record "completed" or "failed".
    summary: str  # Keep a bounded summary safe for an outer model.
    next_manifest: SandboxManifest
    outputs: tuple  # Preserve ordered output events.

    workspace_changed: bool
    capture_path: str
    _store: Optional["_InProcessRunStore"] = field(default=None, repr=False, compare=False)
    _slot_key: str = field(default="", repr=False, compare=False)
    _run_id: str = field(default="", repr=False, compare=False)

    def read_workspace_file(self, relative_path: str, max_bytes: int = MAX_RETAINED_OUTPUT_BYTES) -> bytes:
        """Read one bounded regular file from the retained workspace capture."""
        path = _normalize_workspace_file_path(relative_path)
        _require_workspace_read_limit(max_bytes)
        try:
            with tarfile.open(self.capture_path, mode="r:") as archive:
                match = None
                for member in archive:
                    _validate_restore_archive_member(member)
                    if member.name.removeprefix("./") != path:
                        continue
                    if match is not None:
                        raise AgentSandboxError(f"workspace capture contains duplicate member: {path!r}")
                    match = member
                if match is None:
                    raise AgentSandboxError(f"workspace capture does not contain {path!r}")
                if not match.isfile():
                    raise AgentSandboxError(f"workspace capture member is not a regular file: {path!r}")
                if match.size > max_bytes:
                    raise AgentSandboxError(f"workspace capture member exceeds {max_bytes} byte limit")
                source = archive.extractfile(match)
                if source is None:
                    raise AgentSandboxError(f"workspace capture member cannot be read: {path!r}")
                content = bytearray()
                while chunk := source.read(min(64 * 1024, max_bytes - len(content) + 1)):
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise AgentSandboxError(f"workspace capture member exceeds {max_bytes} byte limit")
                if len(content) != match.size:
                    raise AgentSandboxError(f"workspace capture member is truncated: {path!r}")
                return bytes(content)
        except AgentSandboxError:
            raise
        except (OSError, EOFError, tarfile.TarError) as exc:
            raise AgentSandboxError(f"Refusing workspace capture: {exc}") from exc

    def release(self) -> None:
        """Remove the local capture after its durable upload."""
        try:
            with suppress(FileNotFoundError):
                os.unlink(self.capture_path)
        finally:
            if self._store is not None:
                self._store.discard(self._slot_key, self._run_id, self)

    def __enter__(self) -> "SandboxRun":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


def _require_nonempty_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_optional_string(field: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    return value


def _normalize_workspace_file_path(relative_path: object) -> str:
    """Return one normalized relative POSIX workspace path."""
    if not isinstance(relative_path, str):
        raise TypeError("relative_path must be a string")
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
        or relative_path != str(path)
        or relative_path == "."
        or not _is_safe_archive_path(relative_path)
    ):
        raise ValueError("relative_path must be a normalized relative POSIX path")
    return relative_path


def _require_workspace_read_limit(max_bytes: object) -> int:
    """Require a positive integral workspace read cap."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return max_bytes


class Sandbox:
    """Own the application-lifetime state for managed sandbox work."""

    def __init__(self, namespace: str | None = None) -> None:
        self._store = _InProcessRunStore()
        self._resource_names = runtime_resource_names(namespace)

    def run(self, invocation: SandboxInvocation) -> SandboxRun:
        """Run an invocation using this sandbox's logical result cache."""
        return _run(invocation, self._store, self._resource_names)

    def run_python_script(self, interpreter: str, script: str, timeout: float, *, profile_name: str = "gigachad") -> str:
        """Run Python in a disposable profiled sandbox."""
        return _run_python_script(interpreter, script, timeout, profile_name=profile_name)


def _bounded_summary(exec_result: "ExecResult", status: str, limit: int = 400) -> str:
    """Return a bounded execution summary for outer-model context."""
    text = (exec_result.stdout.strip() or exec_result.stderr.strip() or "(no output)").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 1] + "\u2026"
    return f"[{status}] {text}"


def _build_manifest(
    profile: "Profile",
    invocation: SandboxInvocation,
    sha256: str,
    status: str,
    exit_code: int,
    runtime_fingerprint: str,
) -> SandboxManifest:
    """Build the next immutable manifest without host-local references."""
    return SandboxManifest(
        schema_version=1,
        manifest_id=f"{invocation.run_id}-{sha256[:16]}",
        profile=profile.name,
        runtime_fingerprint=runtime_fingerprint,
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=None)},
        execution=Execution(
            run_id=invocation.run_id,
            status=status,
            exit_code=exit_code,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def _run(invocation: SandboxInvocation, store: "_InProcessRunStore", resource_names: RuntimeResourceNames | None = None) -> SandboxRun:
    """Execute and cache one logical invocation."""

    existing = store.get(invocation.slot_key, invocation.run_id)
    if existing is not None:
        return existing

    with store.slot_lock(invocation.slot_key):
        existing = store.get(invocation.slot_key, invocation.run_id)  # Recheck after waiting for this slot.
        if existing is not None:
            return existing

        profile = load_profile(invocation.profile_name)
        runtime = SandboxRuntime(profile, resource_names=resource_names or runtime_resource_names())
        prior = invocation.active_manifest
        execution = runtime.execute(
            ManagedExecutionRequest(
                run_id=invocation.run_id,
                manifest_id=prior.manifest_id if prior else None,
                prior_runtime_fingerprint=prior.runtime_fingerprint if prior else None,
                workspace_archive_path=invocation.workspace_archive_path,
                expected_workspace_sha256=prior.workspace.sha256 if prior else invocation.workspace_archive_sha256,
                prompt=invocation.prompt,
                resuming=bool(prior and prior.omp_sessions),
                prompt_images=invocation.prompt_images,
                model=invocation.model,
                thinking=invocation.thinking,
                append_system=invocation.append_system,
                lean=invocation.lean,
            )
        )
        capture_path = execution.archive_path
        try:
            exec_result = execution.exec_result
            events: list[OutputEvent] = to_output_events(exec_result.exit_code, exec_result.stdout, exec_result.stderr)
            status = "completed" if exec_result.exit_code == 0 else "failed"
            previous_sha256 = prior.workspace.sha256 if prior else None

            result = SandboxRun(
                status=status,
                summary=_bounded_summary(exec_result, status),
                next_manifest=_build_manifest(
                    profile,
                    invocation,
                    execution.workspace_sha256,
                    status,
                    exec_result.exit_code,
                    execution.runtime_fingerprint or profile.runtime_fingerprint,
                ),
                outputs=tuple(events),
                workspace_changed=(execution.workspace_sha256 != previous_sha256),
                capture_path=capture_path,
                _store=store,
                _slot_key=invocation.slot_key,
                _run_id=invocation.run_id,
            )
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(capture_path)
            raise
        store.put(invocation.slot_key, invocation.run_id, result)
        return result


_default_sandbox = Sandbox()


def run(invocation: SandboxInvocation, store: Optional["_InProcessRunStore"] = None) -> SandboxRun:
    """Run an invocation through the default sandbox or a caller-supplied store."""
    if store is not None:
        return _run(invocation, store)
    return _default_sandbox.run(invocation)


def run_python_script(interpreter: str, script: str, timeout: float, *, profile_name: str = "gigachad") -> str:
    """Run Python through the default sandbox."""
    return _default_sandbox.run_python_script(interpreter, script, timeout, profile_name=profile_name)
