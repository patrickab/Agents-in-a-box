"""Run idempotent managed OMP invocations for caller-owned sandbox slots.

Resolve and upload durable assets outside this module. Logical slots share a
result cache while the runtime serializes physical container use.
"""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import threading
from typing import Literal

from agent_sandbox._validation import is_safe_name, require_nonempty_string
from agent_sandbox.archive import WorkspaceArchive, read_member
from agent_sandbox.config import MAX_RETAINED_OUTPUT_BYTES, runtime_resource_names
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef
from agent_sandbox.omp import OmpTask, OutputEvent, to_output_events
from agent_sandbox.profile import load_profile
from agent_sandbox.runtime import ExecResult, ManagedExecutionResult, SandboxRuntime, WorkspaceRestore

RunStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class SandboxInvocation:
    """Describe one caller-requested sandbox run."""

    slot_key: str  # Deterministic caller-owned sandbox slot; runs in one slot are serialized.
    profile_name: str
    run_id: str  # Upstream tool-call identity; repeated runs with it return the cached result.
    task: OmpTask
    active_manifest: SandboxManifest | None = None  # Resume this manifest's OMP session.
    workspace: WorkspaceArchive | None = None  # Caller-downloaded snapshot to restore.

    def __post_init__(self) -> None:
        require_nonempty_string("slot_key", self.slot_key)
        require_nonempty_string("run_id", self.run_id)
        if not is_safe_name(self.profile_name):
            raise ValueError("profile_name must be a safe profile name")
        if not isinstance(self.task, OmpTask):
            raise TypeError("task must be an OmpTask")
        if self.workspace is not None and not isinstance(self.workspace, WorkspaceArchive):
            raise TypeError("workspace must be a WorkspaceArchive or None")
        if self.active_manifest is None:
            return
        if not isinstance(self.active_manifest, SandboxManifest):
            raise TypeError("active_manifest must be a SandboxManifest or None")
        if self.workspace is None:
            raise ValueError("active_manifest requires workspace")
        if self.workspace.sha256 != self.active_manifest.workspace.sha256:
            raise ValueError("workspace sha256 conflicts with active_manifest workspace sha256")

    @property
    def restore(self) -> WorkspaceRestore | None:
        if self.workspace is None:
            return None
        prior_fingerprint = self.active_manifest.runtime_fingerprint if self.active_manifest else None
        return WorkspaceRestore(self.workspace, prior_fingerprint)


@dataclass(frozen=True)
class SandboxRun:
    """Hold one run's result and its transient local workspace capture until release."""

    status: RunStatus
    summary: str  # Bounded summary safe for an outer model.
    next_manifest: SandboxManifest
    outputs: tuple[OutputEvent, ...]
    workspace_changed: bool
    workspace: WorkspaceArchive  # Local capture; deleted by release().
    _on_release: Callable[[], None] = field(default=lambda: None, repr=False, compare=False)

    def read_workspace_file(self, relative_path: str, max_bytes: int = MAX_RETAINED_OUTPUT_BYTES) -> bytes:
        """Read one bounded regular file from the workspace capture."""
        return read_member(self.workspace.path, relative_path, max_bytes)

    def release(self) -> None:
        """Delete the local capture after its durable upload and allow the run ID to execute again."""
        try:
            with suppress(FileNotFoundError):
                os.unlink(self.workspace.path)
        finally:
            self._on_release()

    def __enter__(self) -> "SandboxRun":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


class Sandbox:
    """Own the application-lifetime result cache and runtime namespace for managed sandbox work."""

    def __init__(self, namespace: str | None = None) -> None:
        self._resource_names = runtime_resource_names(namespace)
        self._completed: dict[tuple[str, str], SandboxRun] = {}
        self._slot_locks: dict[str, threading.Lock] = {}
        self._slot_locks_guard = threading.Lock()

    def run(self, invocation: SandboxInvocation) -> SandboxRun:
        """Run an invocation once per (slot, run ID) until its result is released."""
        key = (invocation.slot_key, invocation.run_id)
        if (existing := self._completed.get(key)) is not None:
            return existing
        with self._slot_lock(invocation.slot_key):
            if (existing := self._completed.get(key)) is not None:  # Recheck after waiting for this slot.
                return existing
            result = self._execute(invocation, key)
            self._completed[key] = result
            return result

    def run_python_script(self, interpreter: str, script: str, timeout: float, *, profile_name: str) -> str:
        """Run Python in a disposable network-disabled profiled sandbox."""
        return SandboxRuntime(load_profile(profile_name)).execute_python(interpreter, script, timeout)

    def _slot_lock(self, slot_key: str) -> threading.Lock:
        with self._slot_locks_guard:
            return self._slot_locks.setdefault(slot_key, threading.Lock())

    def _evict(self, key: tuple[str, str], run: SandboxRun) -> None:
        """Evict a released result without dropping a later retry cached under the same key."""
        if self._completed.get(key) is run:
            del self._completed[key]

    def _execute(self, invocation: SandboxInvocation, key: tuple[str, str]) -> SandboxRun:
        profile = load_profile(invocation.profile_name)
        execution = SandboxRuntime(profile, self._resource_names).execute(invocation.task, invocation.restore)
        try:
            exec_result = execution.exec_result
            status: RunStatus = "completed" if exec_result.exit_code == 0 else "failed"
            prior = invocation.workspace
            run = SandboxRun(
                status=status,
                summary=_bounded_summary(exec_result, status),
                next_manifest=_build_manifest(profile.name, invocation.run_id, execution, status),
                outputs=to_output_events(exec_result.exit_code, exec_result.stdout, exec_result.stderr),
                workspace_changed=prior is None or prior.sha256 != execution.workspace.sha256,
                workspace=execution.workspace,
                _on_release=lambda: self._evict(key, run),
            )
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(execution.workspace.path)
            raise
        return run


def _bounded_summary(exec_result: ExecResult, status: RunStatus, limit: int = 400) -> str:
    """Return a bounded execution summary for outer-model context."""
    text = (exec_result.stdout.strip() or exec_result.stderr.strip() or "(no output)").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 1] + "\u2026"
    return f"[{status}] {text}"


def _build_manifest(profile_name: str, run_id: str, execution: ManagedExecutionResult, status: RunStatus) -> SandboxManifest:
    """Build the next immutable manifest without host-local references."""
    sha256 = execution.workspace.sha256
    return SandboxManifest(
        schema_version=1,
        manifest_id=f"{run_id}-{sha256[:16]}",
        profile=profile_name,
        runtime_fingerprint=execution.runtime_fingerprint,
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=None)},
        execution=Execution(
            run_id=run_id,
            status=status,
            exit_code=execution.exec_result.exit_code,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
