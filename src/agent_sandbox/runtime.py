"""Run profiled OMP work in short-lived rootless Docker containers.

Every managed execution gets a new named container solely so a namespace can
serialize and clean up a crashed predecessor.  Workspace continuity is carried
only by caller-supplied verified archives; neither a Docker volume nor an OMP
state marker survives a transaction.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import fcntl
import hashlib
import io
import math
import os
from pathlib import PurePosixPath
import posixpath
import socket
import tarfile
import tempfile
import threading

from docker.types import Mount as DockerMount

from agent_sandbox.config import (
    DOCKER_HOST,
    MANAGED_OMP_TIMEOUT_SECONDS,
    MAX_CAPTURED_WORKSPACE_BYTES,
    MAX_PROMPT_IMAGE_BYTES,
    MAX_RESTORE_ARCHIVE_BYTES,
    MAX_RETAINED_OUTPUT_BYTES,
    RUNTIME_TMPFS_SIZE,
    TMP_TMPFS_SIZE,
    RuntimeResourceNames,
    runtime_resource_names,
)
from agent_sandbox.omp_runner import build_omp_argv
from agent_sandbox.profile import Profile, load_profile
import docker

_DEFAULT_OUTPUT_LIMIT = object()

INTERNAL_WORKDIR = "/runtime/active"
_PROMPT_IMAGES_DIR = "/runtime/prompt-images"
_APPEND_SYSTEM_DIR = "/runtime/append-system"
_APPEND_SYSTEM_PATH = f"{_APPEND_SYSTEM_DIR}/APPEND_SYSTEM.md"
_OMP_HOME = "/runtime/omp-home"
_RESTORE_ARCHIVE_PATH = "/runtime/restore-archive.tar"
_DEFAULT_RUNTIME_RESOURCES = runtime_resource_names()
_RUNTIME_LOCK_PATH = _DEFAULT_RUNTIME_RESOURCES.lock_path
_runtime_thread_locks: dict[str, threading.Lock] = {}
_runtime_thread_locks_guard = threading.Lock()


@dataclass(frozen=True)
class PromptImage:
    """Hold one safe prompt-image filename and its bytes."""

    filename: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename or "\\" in self.filename or "\x00" in self.filename:
            raise ValueError("prompt image filename must be a safe basename")
        path = PurePosixPath(self.filename)
        if self.filename != path.name or self.filename in {".", ".."}:
            raise ValueError("prompt image filename must be a safe basename")
        if not isinstance(self.content, bytes):
            raise TypeError("prompt image content must be bytes")


def _normalize_prompt_images(prompt_images: object) -> tuple[PromptImage, ...]:
    """Normalize typed and legacy prompt images into immutable values."""
    if not isinstance(prompt_images, tuple):
        raise TypeError("prompt_images must be a tuple of (name, content) tuples")
    images = []
    total_bytes = 0
    for image in prompt_images:
        if isinstance(image, PromptImage):
            normalized = image
        else:
            if not isinstance(image, tuple) or len(image) != 2:
                raise ValueError("each prompt image must be a (name, content) tuple")
            normalized = PromptImage(filename=image[0], content=image[1])
        total_bytes += len(normalized.content)
        if total_bytes > MAX_PROMPT_IMAGE_BYTES:
            raise ValueError(f"prompt images must not exceed {MAX_PROMPT_IMAGE_BYTES} bytes")
        images.append(normalized)
    return tuple(images)


@dataclass(frozen=True)
class ManagedExecutionRequest:
    """Describe one complete managed-workspace OMP transaction."""

    run_id: str
    manifest_id: str | None
    workspace_archive_path: str | None
    expected_workspace_sha256: str | None
    prompt: str
    resuming: bool
    prompt_images: tuple[PromptImage, ...] = ()
    model: str | None = None
    thinking: str | None = None
    append_system: str | None = None
    lean: bool = False
    prior_runtime_fingerprint: str | None = None


@dataclass(frozen=True)
class ManagedExecutionResult:
    """Return command output and the transient workspace snapshot from one transaction."""

    exec_result: "ExecResult"
    archive_path: str
    workspace_sha256: str
    runtime_fingerprint: str = ""


@contextmanager
def _runtime_lease(lock_path: str = _RUNTIME_LOCK_PATH) -> Iterator[None]:
    """Serialize physical use of one namespaced Docker runtime across processes."""
    with _runtime_thread_locks_guard:
        thread_lock = _runtime_thread_locks.setdefault(lock_path, threading.Lock())
    with thread_lock, open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


class AgentSandboxError(RuntimeError):
    """Base error for caller-facing sandbox runtime failures."""


class SecurityEnvironmentError(AgentSandboxError):
    """Report missing rootless Docker, runsc, or image prerequisites."""


class ContainerRuntimeError(AgentSandboxError):
    """Report a container lifecycle or command failure."""


class PythonScriptError(AgentSandboxError):
    """Report a failed or timed-out disposable Python execution."""


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str




def _container_policy(network: str) -> dict:
    """Build the mandatory isolation and network policy for every container."""
    if network == "bridge":
        network_policy = {"network_mode": "bridge"}
    elif network == "disabled":
        network_policy = {"network_disabled": True}
    else:
        raise ValueError(f"unsupported container network policy: {network!r}")
    return {
        "runtime": "runsc",
        "user": "0:0",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        **network_policy,
    }


def _is_safe_archive_path(path: str) -> bool:
    return bool(path) and not path.startswith("/") and all(part != ".." for part in PurePosixPath(path).parts)


def _validate_restore_archive_member(member: tarfile.TarInfo) -> None:
    if not _is_safe_archive_path(member.name):
        raise ContainerRuntimeError(f"restore archive contains unsafe member path: {member.name!r}")
    if member.issym() or member.islnk():
        base = posixpath.dirname(member.name) if member.issym() else ""
        target = member.linkname
        resolved_target = posixpath.normpath(posixpath.join(base, target))
        if not target or target.startswith("/") or not _is_safe_archive_path(resolved_target):
            raise ContainerRuntimeError(f"restore archive contains unsafe link target: {target!r}")
    elif not member.isfile() and not member.isdir():
        raise ContainerRuntimeError(f"restore archive contains unsupported member type: {member.name!r}")


def _validate_restore_archive(path: str) -> str:
    """Validate and incrementally hash one bounded workspace tar before Docker mutation."""
    try:
        if os.path.getsize(path) > MAX_RESTORE_ARCHIVE_BYTES:
            raise ContainerRuntimeError(f"restore archive exceeds {MAX_RESTORE_ARCHIVE_BYTES} byte limit")
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                _validate_restore_archive_member(member)
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                size += len(chunk)
                if size > MAX_RESTORE_ARCHIVE_BYTES:
                    raise ContainerRuntimeError(f"restore archive exceeds {MAX_RESTORE_ARCHIVE_BYTES} byte limit")
                digest.update(chunk)
        return digest.hexdigest()
    except ContainerRuntimeError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ContainerRuntimeError(f"Refusing restore archive: {exc}") from exc


def _stream_container_exec(
    container: object,
    command: list[str],
    *,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    stdout_sink: Callable[[bytes], None] | None = None,
    stdout_limit: int | None = None,
    stderr_limit: int | None = None,
    total_limit: int | None | object = _DEFAULT_OUTPUT_LIMIT,
) -> ExecResult:
    """Execute and incrementally retain or consume demultiplexed container output."""
    stdout_limit = MAX_RETAINED_OUTPUT_BYTES if stdout_limit is None else stdout_limit
    stderr_limit = MAX_RETAINED_OUTPUT_BYTES if stderr_limit is None else stderr_limit
    total_limit = MAX_RETAINED_OUTPUT_BYTES if total_limit is _DEFAULT_OUTPUT_LIMIT else total_limit
    api = container.client.api
    created = api.exec_create(
        container.id,
        command,
        stdout=True,
        stderr=True,
        environment=environment,
        workdir=workdir,
    )
    exec_id = created["Id"] if isinstance(created, dict) else created
    if not isinstance(exec_id, str) or not exec_id:
        raise ContainerRuntimeError("Docker did not return an exec identifier")

    stdout = bytearray()
    stderr = bytearray()
    stdout_size = 0
    stderr_size = 0
    total_size = 0
    stream = api.exec_start(exec_id, stream=True, demux=True)
    try:
        for chunk in stream:
            if isinstance(chunk, tuple):
                stdout_chunk, stderr_chunk = chunk
            else:
                stdout_chunk, stderr_chunk = chunk, None
            for stream_name, payload in (("stdout", stdout_chunk), ("stderr", stderr_chunk)):
                if not payload:
                    continue
                if not isinstance(payload, bytes):
                    raise ContainerRuntimeError(f"Docker returned non-bytes {stream_name} output")
                total_size += len(payload)
                if total_limit is not None and total_size > total_limit:
                    raise ContainerRuntimeError(f"command output exceeds {total_limit} byte limit")
                if stream_name == "stdout":
                    stdout_size += len(payload)
                    if stdout_size > stdout_limit:
                        raise ContainerRuntimeError(f"captured workspace exceeds {stdout_limit} byte limit")
                    if stdout_sink is None:
                        stdout.extend(payload)
                    else:
                        stdout_sink(payload)
                else:
                    stderr_size += len(payload)
                    if stderr_size > stderr_limit:
                        raise ContainerRuntimeError(f"command diagnostics exceed {stderr_limit} byte limit")
                    stderr.extend(payload)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    inspection = api.exec_inspect(exec_id)
    exit_code = inspection.get("ExitCode") if isinstance(inspection, dict) else None
    if type(exit_code) is not int:
        raise ContainerRuntimeError("Docker did not report an exec exit code")
    return ExecResult(exit_code, bytes(stdout).decode(errors="replace"), bytes(stderr).decode(errors="replace"))


class SandboxRuntime:
    """Run isolated managed-workspace transactions for one profile."""

    def __init__(self, profile: Profile, resource_names: RuntimeResourceNames | None = None) -> None:
        self.profile = profile
        self._resource_names = resource_names or _DEFAULT_RUNTIME_RESOURCES
        try:
            self.client = docker.DockerClient(base_url=DOCKER_HOST)
        except docker.errors.DockerException as exc:
            raise SecurityEnvironmentError(f"Failed to connect to Docker at {DOCKER_HOST}: {exc}") from exc

    def _resources(self) -> RuntimeResourceNames:
        return getattr(self, "_resource_names", _DEFAULT_RUNTIME_RESOURCES)

    def run_probe(self, argv: list[str]) -> bytes:
        """Run a bridge-networked diagnostic container through a fail-closed lifecycle."""
        container = None
        try:
            container = self.client.containers.create(
                self.profile.image,
                argv,
                mounts=[
                    DockerMount(target=mount.target, source=mount.source, type="bind", read_only=(mount.mode == "ro"))
                    for mount in self.profile.mounts
                ],
                read_only=True,
                tmpfs=self._tmpfs(runtime=False),
                **_container_policy("bridge"),
            )
            container.start()
            result = container.wait()
            output = container.logs(stdout=True, stderr=True)
            exit_code = result.get("StatusCode")
            if exit_code != 0:
                raise docker.errors.ContainerError(
                    container,
                    exit_code,
                    argv,
                    self.profile.image,
                    output,
                )
            return output
        finally:
            if container is not None:
                container.remove(force=True)

    def _verify_environment(self) -> str:
        """Verify the daemon prerequisites and return the immutable resolved image ID."""
        try:
            info = self.client.info()
        except docker.errors.DockerException as exc:
            raise SecurityEnvironmentError(f"Docker daemon unreachable: {exc}") from exc
        if not any("rootless" in option.lower() for option in info.get("SecurityOptions", [])):
            raise SecurityEnvironmentError("Rootless Docker is not enabled. Daemon must run in rootless mode.")
        if "runsc" not in info.get("Runtimes", {}):
            raise SecurityEnvironmentError("gVisor 'runsc' runtime is not configured in Docker.")
        try:
            image = self.client.images.get(self.profile.image)
        except docker.errors.ImageNotFound:
            raise SecurityEnvironmentError(f"Image '{self.profile.image}' not found.")
        image_id = getattr(image, "id", None)
        if not isinstance(image_id, str) or not image_id:
            image_id = getattr(image, "attrs", {}).get("Id") if isinstance(getattr(image, "attrs", {}), dict) else None
        if not isinstance(image_id, str) or not image_id:
            raise SecurityEnvironmentError(f"Image '{self.profile.image}' did not expose a resolved Docker image ID.")
        return image_id

    def _effective_runtime_fingerprint(self, image_id: str) -> str:
        """Bind canonical profile policy to the image bytes Docker resolved for this run."""
        material = f"{self.profile.runtime_fingerprint}\x00{image_id}".encode("utf-8")
        return f"sha256:{hashlib.sha256(material).hexdigest()}"

    def _remove_stale_container(self) -> None:
        """Remove a stopped or leaked predecessor before allocating this namespace name."""
        try:
            stale = self.client.containers.get(self._resources().container)
        except docker.errors.NotFound:
            return
        stale.remove(force=True)

    def _rewrite_target(self, target: str) -> str:
        """Rewrite workspace-relative bind mount targets for the ephemeral workspace."""
        workdir = self.profile.workdir.rstrip("/")
        if target == workdir:
            return INTERNAL_WORKDIR
        if target.startswith(workdir + "/"):
            return INTERNAL_WORKDIR + target[len(workdir) :]
        return target

    def _managed_mounts(self) -> list[DockerMount]:
        return [
            DockerMount(target=self._rewrite_target(mount.target), source=mount.source, type="bind", read_only=(mount.mode == "ro"))
            for mount in self.profile.mounts
        ]

    def _python_mounts(self) -> list[DockerMount]:
        """Mount only profile locations that provide registered Python interpreters."""
        interpreter_paths = tuple(dict(self.profile.python_interpreters).values())
        return [
            DockerMount(target=mount.target, source=mount.source, type="bind", read_only=True)
            for mount in self.profile.mounts
            if any(
                mount.target == "/" or path == mount.target.rstrip("/") or path.startswith(mount.target.rstrip("/") + "/")
                for path in interpreter_paths
            )
        ]

    @staticmethod
    def _tmpfs(*, runtime: bool) -> dict[str, str]:
        tmpfs = {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"}
        if runtime:
            tmpfs["/runtime"] = f"rw,nosuid,nodev,size={RUNTIME_TMPFS_SIZE}"
        return tmpfs

    def _create_managed_container(self, image_id: str, restore_archive_path: str | None) -> object:
        mounts = self._managed_mounts()
        if restore_archive_path is not None:
            mounts.append(
                DockerMount(target=_RESTORE_ARCHIVE_PATH, source=restore_archive_path, type="bind", read_only=True)
            )
        return self.client.containers.create(
            image_id,
            name=self._resources().container,
            command=["sleep", "infinity"],
            read_only=True,
            tmpfs=self._tmpfs(runtime=True),
            mounts=mounts,
            environment=self._environment(),
            **_container_policy("bridge"),
        )

    def _environment(self) -> dict[str, str]:
        """Forward only allowlisted variables and declared bridged host services."""
        forwarded = {name: os.environ[name] for name in self.profile.env_passthrough if name in os.environ}
        if not self.profile.host_services:
            return forwarded
        host = self.host_address()
        return {**forwarded, **{name: f"http://{host}:{port}" for name, port in self.profile.host_services}}

    def host_address(self) -> str:
        """Return the routable host address for explicit bridge-networked services."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
        except OSError as exc:
            raise ContainerRuntimeError("cannot determine a host address for bridged host services") from exc
        finally:
            probe.close()

    def _omp_environment(self) -> dict[str, str]:
        bin_dirs = [str(PurePosixPath(path).parent) for _label, path in self.profile.python_interpreters]
        path = ":".join([*dict.fromkeys(bin_dirs), "/usr/local/bin", "/usr/bin", "/bin"])
        return {**self._environment(), "HOME": _OMP_HOME, "PATH": path}

    def _run_or_raise(self, container: object, command: str) -> None:
        result = _stream_container_exec(container, ["sh", "-c", command])
        if result.exit_code != 0:
            detail = result.stdout or result.stderr
            raise ContainerRuntimeError(f"container setup failed (exit {result.exit_code}): {command}\n{detail}")

    def _restore_workspace(self, container: object, archive_path: str | None) -> None:
        self._run_or_raise(container, f"mkdir -p {INTERNAL_WORKDIR}")
        if archive_path is not None:
            self._run_or_raise(
                container,
                f"tar --extract --file={_RESTORE_ARCHIVE_PATH} --directory={INTERNAL_WORKDIR} "
                "--no-same-owner --no-same-permissions",
            )


    def _materialize_append_system(self, container: object, append_system: str) -> None:
        self._run_or_raise(container, f"rm -rf {_APPEND_SYSTEM_DIR} && mkdir -p {_APPEND_SYSTEM_DIR}")
        content = append_system.encode("utf-8")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            entry = tarfile.TarInfo("APPEND_SYSTEM.md")
            entry.size = len(content)
            entry.mode = 0o600
            tar.addfile(entry, io.BytesIO(content))
        if not container.put_archive(_APPEND_SYSTEM_DIR, archive.getvalue()):
            raise ContainerRuntimeError("failed to materialize appended system prompt")

    def _materialize_prompt_images(self, container: object, prompt_images: tuple[PromptImage, ...]) -> None:
        self._run_or_raise(container, f"rm -rf {_PROMPT_IMAGES_DIR} && mkdir -p {_PROMPT_IMAGES_DIR}")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for image in prompt_images:
                entry = tarfile.TarInfo(image.filename)
                entry.size = len(image.content)
                entry.mode = 0o600
                tar.addfile(entry, io.BytesIO(image.content))
        if not container.put_archive(_PROMPT_IMAGES_DIR, archive.getvalue()):
            raise ContainerRuntimeError("failed to materialize prompt images")

    def _prepare_omp_home(self, container: object) -> None:
        """Create a writable OMP home in this transaction's runtime tmpfs."""
        self._run_or_raise(
            container,
            f"rm -rf {_OMP_HOME} && mkdir -p {_OMP_HOME}/.omp && "
            f"{{ [ ! -d /root/.omp ] || cp -a /root/.omp/. {_OMP_HOME}/.omp/; }}",
        )

    def _invoke_omp(
        self,
        container: object,
        argv: list[str],
        *,
        prompt_images: tuple[PromptImage, ...],
        append_system: str | None,
    ) -> ExecResult:
        self._prepare_omp_home(container)
        if append_system is not None:
            self._materialize_append_system(container, append_system)
        if prompt_images:
            self._materialize_prompt_images(container, prompt_images)
        result = _stream_container_exec(container, argv, workdir=INTERNAL_WORKDIR, environment=self._omp_environment())
        if result.exit_code in {124, 137}:
            raise ContainerRuntimeError(f"OMP execution exceeded {MANAGED_OMP_TIMEOUT_SECONDS}s limit")
        return result

    def _captured_exclusions(self) -> list[str]:
        """Return exact workspace-relative bind-mount paths to omit from a snapshot."""
        prefix = INTERNAL_WORKDIR + "/"
        return sorted(
            {
                target[len(prefix) :]
                for mount in self.profile.mounts
                if (target := self._rewrite_target(mount.target)).startswith(prefix)
            }
        )

    def _capture_workspace(self, container: object) -> tuple[str, str]:
        """Stream a bounded tar snapshot, then validate and hash it before publication."""
        exclusions = [f"--exclude=./{path}" for path in self._captured_exclusions()]
        command = [
            "tar",
            "-C",
            INTERNAL_WORKDIR,
            "--anchored",
            "--no-wildcards",
            *exclusions,
            "-cf",
            "-",
            ".",
        ]
        descriptor, archive_path = tempfile.mkstemp(suffix=".tar", prefix="agent-sandbox-snapshot-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as archive:
                result = _stream_container_exec(
                    container,
                    command,
                    stdout_sink=archive.write,
                    stdout_limit=MAX_CAPTURED_WORKSPACE_BYTES,
                    stderr_limit=MAX_RETAINED_OUTPUT_BYTES,
                    total_limit=None,
                )
            if result.exit_code != 0:
                raise ContainerRuntimeError(f"capture failed: tar exited {result.exit_code}: {result.stderr}")
            return archive_path, _validate_restore_archive(archive_path)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(archive_path)
            raise

    def _verified_restore_archive(self, request: ManagedExecutionRequest, effective_fingerprint: str) -> str | None:
        """Validate all restore preconditions before mutating a managed container."""
        if request.manifest_id is not None:
            if request.prior_runtime_fingerprint != effective_fingerprint:
                raise ContainerRuntimeError(f"Refusing restore for manifest {request.manifest_id}: runtime fingerprint mismatch")
            if request.expected_workspace_sha256 is None:
                raise ContainerRuntimeError(f"Refusing restore for manifest {request.manifest_id}: workspace hash is missing")
            if request.workspace_archive_path is None:
                raise ContainerRuntimeError(f"Refusing restore for manifest {request.manifest_id}: workspace archive is required")
        elif request.workspace_archive_path is None and request.expected_workspace_sha256 is not None:
            raise ContainerRuntimeError("workspace hash requires a workspace archive")
        elif request.workspace_archive_path is not None and request.expected_workspace_sha256 is None:
            raise ContainerRuntimeError("workspace archive requires a workspace hash")

        if request.workspace_archive_path is None:
            return None
        digest = _validate_restore_archive(request.workspace_archive_path)
        if digest != request.expected_workspace_sha256:
            raise ContainerRuntimeError(
                f"Refusing restore for manifest {request.manifest_id}: workspace hash mismatch "
                f"(expected {request.expected_workspace_sha256}, got {digest})"
            )
        return request.workspace_archive_path

    def execute(self, request: ManagedExecutionRequest) -> ManagedExecutionResult:
        """Run one fresh-container managed OMP transaction under the namespace lease."""
        image_paths = tuple(f"{_PROMPT_IMAGES_DIR}/{image.filename}" for image in request.prompt_images)
        with _runtime_lease(self._resources().lock_path):
            image_id = self._verify_environment()
            self._remove_stale_container()
            effective_fingerprint = self._effective_runtime_fingerprint(image_id)
            restore_archive = self._verified_restore_archive(request, effective_fingerprint)
            container = None
            capture_path: str | None = None
            operation_error: BaseException | None = None
            try:
                container = self._create_managed_container(image_id, restore_archive)
                container.start()
                self._restore_workspace(container, restore_archive)
                argv = build_omp_argv(
                    self.profile.omp_binary,
                    request.prompt,
                    session_dir=f"{INTERNAL_WORKDIR}/.omp-session",
                    resuming=request.resuming,
                    image_paths=image_paths,
                    model=request.model,
                    thinking=request.thinking,
                    append_system_path=_APPEND_SYSTEM_PATH if request.append_system is not None else None,
                    lean=request.lean,
                )
                exec_result = self._invoke_omp(
                    container,
                    ["timeout", "--signal=KILL", f"{MANAGED_OMP_TIMEOUT_SECONDS}s", *argv],
                    prompt_images=request.prompt_images,
                    append_system=request.append_system,
                )
                capture_path, workspace_sha256 = self._capture_workspace(container)
                return ManagedExecutionResult(exec_result, capture_path, workspace_sha256, effective_fingerprint)
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                if container is not None:
                    try:
                        container.remove(force=True)
                    except Exception as cleanup_error:
                        if capture_path is not None:
                            with suppress(FileNotFoundError):
                                os.unlink(capture_path)
                        if operation_error is not None:
                            raise ContainerRuntimeError(
                                f"managed execution failed ({operation_error}). "
                                f"Additionally failed to force-remove container: {cleanup_error}"
                            ) from cleanup_error
                        raise ContainerRuntimeError(f"failed to force-remove managed container: {cleanup_error}") from cleanup_error

    def execute_python(self, interpreter: str, script: str, timeout: float) -> str:
        """Run a script in a fresh network-disabled container and return bounded stdout."""
        interpreters = dict(self.profile.python_interpreters)
        interpreter_path = interpreters.get(interpreter)
        if interpreter_path is None:
            raise PythonScriptError(f"Unknown profile Python interpreter: {interpreter!r}.")
        if not isinstance(script, str):
            raise PythonScriptError("Python script must be text.")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise PythonScriptError("Python timeout must be a positive finite number.")

        container = None
        operation_error: PythonScriptError | None = None
        try:
            image_id = self._verify_environment()
            container = self.client.containers.create(
                image_id,
                command=["sleep", "infinity"],
                read_only=True,
                tmpfs=self._tmpfs(runtime=False),
                **_container_policy("disabled"),
                mounts=self._python_mounts(),
                environment={},
                working_dir="/",
            )
            container.start()
            result = _stream_container_exec(
                container,
                ["timeout", "--signal=KILL", f"{timeout:g}s", interpreter_path, "-c", script],
                workdir="/",
                environment={},
            )
            if result.exit_code in {124, 137}:
                raise PythonScriptError(f"Python script did not finish within {timeout:g}s.")
            if result.exit_code != 0:
                raise PythonScriptError(result.stderr.strip()[-4000:] or f"exited with status {result.exit_code}")
            return result.stdout
        except PythonScriptError as exc:
            operation_error = exc
            raise
        except Exception as exc:
            operation_error = PythonScriptError(f"Sandbox Python execution failed: {exc}")
            raise operation_error from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception as cleanup_error:
                    if operation_error is not None:
                        raise PythonScriptError(
                            f"Python execution failed ({operation_error}). "
                            f"Additionally failed to force-remove container: {cleanup_error}"
                        ) from cleanup_error
                    raise PythonScriptError(f"failed to force-remove Python container: {cleanup_error}") from cleanup_error


def run_python_script(interpreter: str, script: str, timeout: float, *, profile_name: str = "gigachad") -> str:
    """Run Python from a named profile in a fresh rootless Docker/gVisor sandbox."""
    try:
        return SandboxRuntime(load_profile(profile_name)).execute_python(interpreter, script, timeout)
    except PythonScriptError:
        raise
    except Exception as exc:
        raise PythonScriptError(str(exc)) from exc
