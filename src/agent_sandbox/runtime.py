"""Run profiled OMP work in short-lived rootless Docker/gVisor containers.

Every managed execution gets a fresh named container solely so a namespace can
serialize and clean up a crashed predecessor. Workspace continuity is carried
only by caller-supplied verified archives; nothing survives a transaction inside Docker.
"""

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import PurePosixPath
import tempfile
import threading

from docker.models.containers import Container
from docker.types import Mount as DockerMount

from agent_sandbox.archive import WorkspaceArchive, verify_archive
from agent_sandbox.config import (
    DOCKER_HOST,
    HOST_SERVICE_ADDRESS,
    MANAGED_OMP_TIMEOUT_SECONDS,
    MAX_RETAINED_OUTPUT_BYTES,
    MAX_WORKSPACE_ARCHIVE_BYTES,
    RUNTIME_TMPFS_SIZE,
    TMP_TMPFS_SIZE,
    RuntimeResourceNames,
    runtime_resource_names,
)
from agent_sandbox.errors import (
    AgentSandboxError,
    ArchiveError,
    ContainerRuntimeError,
    PythonScriptError,
    SecurityEnvironmentError,
)
from agent_sandbox.omp import OmpTask, build_omp_argv
from agent_sandbox.profile import Profile
import docker

INTERNAL_WORKDIR = "/runtime/active"
_SESSION_DIR = f"{INTERNAL_WORKDIR}/.omp-session"
_PROMPT_IMAGES_DIR = "/runtime/prompt-images"
_APPEND_SYSTEM_DIR = "/runtime/append-system"
_APPEND_SYSTEM_PATH = f"{_APPEND_SYSTEM_DIR}/APPEND_SYSTEM.md"
_OMP_HOME = "/runtime/omp-home"
_RESTORE_ARCHIVE_PATH = "/runtime/restore-archive.tar"
_TIMEOUT_EXIT_CODES = frozenset({124, 137})  # timeout(1) and SIGKILL.
_runtime_thread_locks: dict[str, threading.Lock] = {}
_runtime_thread_locks_guard = threading.Lock()


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class WorkspaceRestore:
    """Restore a pinned workspace archive, optionally continuing the OMP session inside it."""

    archive: WorkspaceArchive
    prior_runtime_fingerprint: str | None = None  # Set when resuming; must equal this run's effective fingerprint.

    @property
    def resuming(self) -> bool:
        return self.prior_runtime_fingerprint is not None


@dataclass(frozen=True)
class ManagedExecutionResult:
    """Return OMP output and the transient workspace capture from one transaction."""

    exec_result: ExecResult
    workspace: WorkspaceArchive
    runtime_fingerprint: str


@contextmanager
def _runtime_lease(lock_path: str) -> Iterator[None]:
    """Serialize physical use of one namespaced Docker runtime across threads and processes."""
    with _runtime_thread_locks_guard:
        thread_lock = _runtime_thread_locks.setdefault(lock_path, threading.Lock())
    with thread_lock, open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _bind(source: str, target: str, *, read_only: bool = True) -> DockerMount:
    return DockerMount(target=target, source=source, type="bind", read_only=read_only)


def _stream_container_exec(
    container: Container,
    command: list[str],
    *,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    workspace_sink: Callable[[bytes], None] | None = None,
) -> ExecResult:
    """Run one exec with bounded output.

    Retained stdout and stderr share one MAX_RETAINED_OUTPUT_BYTES budget. With a workspace
    sink, stdout streams into it under MAX_WORKSPACE_ARCHIVE_BYTES and only stderr is retained.
    """
    api = container.client.api
    exec_id = api.exec_create(container.id, command, stdout=True, stderr=True, environment=environment, workdir=workdir)["Id"]
    stdout, stderr = bytearray(), bytearray()
    sunk = 0
    with closing(api.exec_start(exec_id, stream=True, demux=True)) as stream:
        for stdout_chunk, stderr_chunk in stream:
            if stderr_chunk:
                stderr.extend(stderr_chunk)
            if stdout_chunk and workspace_sink is None:
                stdout.extend(stdout_chunk)
            elif stdout_chunk:
                sunk += len(stdout_chunk)
                if sunk > MAX_WORKSPACE_ARCHIVE_BYTES:
                    raise ArchiveError(f"captured workspace exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} byte limit")
                workspace_sink(stdout_chunk)
            if len(stdout) + len(stderr) > MAX_RETAINED_OUTPUT_BYTES:
                raise ContainerRuntimeError(f"command output exceeds {MAX_RETAINED_OUTPUT_BYTES} byte limit")
    exit_code = api.exec_inspect(exec_id)["ExitCode"]
    if exit_code is None:
        raise ContainerRuntimeError("Docker did not report an exec exit code")
    return ExecResult(exit_code, stdout.decode(errors="replace"), stderr.decode(errors="replace"))


def _remove_container(container: Container, label: str, error: type[AgentSandboxError], failure: BaseException | None) -> None:
    """Force-remove a container, keeping any operation failure visible beside a cleanup failure."""
    try:
        container.remove(force=True)
    except Exception as cleanup_error:
        if failure is None:
            raise error(f"failed to force-remove {label} container: {cleanup_error}") from cleanup_error
        raise error(
            f"{label} execution failed ({failure}). Additionally failed to force-remove container: {cleanup_error}"
        ) from cleanup_error


class SandboxRuntime:
    """Run isolated containers for one profile."""

    def __init__(
        self,
        profile: Profile,
        resource_names: RuntimeResourceNames | None = None,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.profile = profile
        self._resource_names = resource_names or runtime_resource_names()
        if client is None:
            try:
                client = docker.DockerClient(base_url=DOCKER_HOST)
            except docker.errors.DockerException as exc:
                raise SecurityEnvironmentError(f"Failed to connect to Docker at {DOCKER_HOST}: {exc}") from exc
        self.client = client

    def verify_environment(self) -> str:
        """Verify rootless Docker, gVisor, and the profile image; return the resolved image ID."""
        try:
            info = self.client.info()
        except docker.errors.DockerException as exc:
            raise SecurityEnvironmentError(f"Docker daemon unreachable: {exc}") from exc
        if not any("rootless" in option.lower() for option in info.get("SecurityOptions", [])):
            raise SecurityEnvironmentError("Docker daemon is not rootless; rerun scripts/setup_agent_sandbox.sh")
        if "runsc" not in info.get("Runtimes", {}):
            raise SecurityEnvironmentError("gVisor 'runsc' is not registered with Docker; rerun scripts/setup_agent_sandbox.sh")
        try:
            return self.client.images.get(self.profile.image).id
        except docker.errors.ImageNotFound as exc:
            raise SecurityEnvironmentError(f"image '{self.profile.image}' not found; build docker/Dockerfile") from exc
        except docker.errors.DockerException as exc:
            raise SecurityEnvironmentError(f"could not inspect image '{self.profile.image}': {exc}") from exc

    @contextmanager
    def _fresh_container(
        self,
        image: str,
        *,
        label: str,
        error: type[AgentSandboxError],
        network: bool,
        mounts: list[DockerMount],
        runtime_tmpfs: bool = False,
        environment: dict[str, str] | None = None,
        name: str | None = None,
    ) -> Iterator[Container]:
        """Create and start one hardened idle container, and always force-remove it."""
        tmpfs = {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"}
        if runtime_tmpfs:
            # OMP extracts and dlopens its native addon under $HOME, which lives here.
            tmpfs["/runtime"] = f"rw,exec,nosuid,nodev,size={RUNTIME_TMPFS_SIZE}"
        container = self.client.containers.create(
            image,
            command=["sleep", "infinity"],
            name=name,
            read_only=True,
            tmpfs=tmpfs,
            mounts=mounts,
            environment=environment or {},
            runtime="runsc",
            user="0:0",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            **({"network_mode": "bridge"} if network else {"network_disabled": True}),
        )
        try:
            container.start()
            yield container
        except BaseException as failure:
            _remove_container(container, label, error, failure)
            raise
        _remove_container(container, label, error, None)

    def run_probe(self, argv: list[str]) -> str:
        """Run one diagnostic command in a fresh bridge-networked container and return its stdout."""
        mounts = [_bind(mount.source, mount.target, read_only=mount.read_only) for mount in self.profile.mounts]
        with self._fresh_container(
            self.profile.image, label="probe", error=ContainerRuntimeError, network=True, mounts=mounts
        ) as container:
            result = _stream_container_exec(container, argv)
        if result.exit_code != 0:
            raise ContainerRuntimeError(f"probe exited {result.exit_code}: {(result.stdout + result.stderr).strip()}")
        return result.stdout

    def execute(self, task: OmpTask, restore: WorkspaceRestore | None = None) -> ManagedExecutionResult:
        """Run one OMP task in a fresh managed container under the namespace lease."""
        with (
            _runtime_lease(self._resource_names.lock_path),
            tempfile.TemporaryDirectory(prefix="agent-sandbox-inputs-") as staging_dir,
        ):
            image_id = self.verify_environment()
            self._remove_stale_container()
            fingerprint = self._effective_runtime_fingerprint(image_id)
            mounts = [*self._managed_mounts(), *self._stage_task_inputs(task, staging_dir)]
            if restore is not None:
                self._verify_restore(restore, fingerprint)
                mounts.append(_bind(restore.archive.path, _RESTORE_ARCHIVE_PATH))
            argv = build_omp_argv(
                self.profile.omp_binary,
                task,
                session_dir=_SESSION_DIR,
                resuming=restore is not None and restore.resuming,
                images_dir=_PROMPT_IMAGES_DIR,
                append_system_path=_APPEND_SYSTEM_PATH,
            )
            capture: WorkspaceArchive | None = None
            try:
                with self._fresh_container(
                    image_id,
                    label="managed",
                    error=ContainerRuntimeError,
                    network=True,
                    mounts=mounts,
                    runtime_tmpfs=True,
                    environment=self._environment(),
                    name=self._resource_names.container,
                ) as container:
                    self._prepare_container(container, restoring=restore is not None)
                    exec_result = _stream_container_exec(
                        container,
                        ["timeout", "--signal=KILL", f"{MANAGED_OMP_TIMEOUT_SECONDS}s", *argv],
                        workdir=INTERNAL_WORKDIR,
                        environment=self._omp_environment(),
                    )
                    if exec_result.exit_code in _TIMEOUT_EXIT_CODES:
                        raise ContainerRuntimeError(f"OMP execution exceeded {MANAGED_OMP_TIMEOUT_SECONDS}s limit")
                    capture = self._capture_workspace(container)
            except BaseException:
                if capture is not None:  # Only container removal can fail after capture.
                    with suppress(FileNotFoundError):
                        os.unlink(capture.path)
                raise
            return ManagedExecutionResult(exec_result, capture, fingerprint)

    def execute_python(self, interpreter: str, script: str, timeout: float) -> str:
        """Run a script in a fresh network-disabled container and return bounded stdout."""
        interpreter_path = self.profile.python_interpreters.get(interpreter)
        if interpreter_path is None:
            raise PythonScriptError(f"Unknown profile Python interpreter: {interpreter!r}.")
        if not isinstance(script, str):
            raise PythonScriptError("Python script must be text.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise PythonScriptError("Python timeout must be a positive finite number.")
        try:
            image_id = self.verify_environment()
            with self._fresh_container(
                image_id, label="Python", error=PythonScriptError, network=False, mounts=self._python_mounts()
            ) as container:
                result = _stream_container_exec(
                    container,
                    ["timeout", "--signal=KILL", f"{timeout:g}s", interpreter_path, "-c", script],
                    workdir="/",
                    environment={},
                )
        except PythonScriptError:
            raise
        except Exception as exc:
            raise PythonScriptError(f"Sandbox Python execution failed: {exc}") from exc
        if result.exit_code in _TIMEOUT_EXIT_CODES:
            raise PythonScriptError(f"Python script did not finish within {timeout:g}s.")
        if result.exit_code != 0:
            raise PythonScriptError(result.stderr.strip()[-4000:] or f"exited with status {result.exit_code}")
        return result.stdout

    def _effective_runtime_fingerprint(self, image_id: str) -> str:
        """Bind canonical profile policy to the image bytes Docker resolved for this run."""
        material = f"{self.profile.runtime_fingerprint}\x00{image_id}".encode("utf-8")
        return f"sha256:{hashlib.sha256(material).hexdigest()}"

    def _remove_stale_container(self) -> None:
        """Remove a stopped or leaked predecessor before reusing this namespace's container name."""
        try:
            stale = self.client.containers.get(self._resource_names.container)
        except docker.errors.NotFound:
            return
        stale.remove(force=True)

    @staticmethod
    def _verify_restore(restore: WorkspaceRestore, runtime_fingerprint: str) -> None:
        """Refuse a restore whose runtime or archive bytes differ from what the caller pinned."""
        if restore.resuming and restore.prior_runtime_fingerprint != runtime_fingerprint:
            raise ContainerRuntimeError("Refusing restore: runtime fingerprint mismatch")
        digest = verify_archive(restore.archive.path)
        if digest != restore.archive.sha256:
            raise ArchiveError(f"Refusing restore: workspace hash mismatch (expected {restore.archive.sha256}, got {digest})")

    def _rewrite_target(self, target: str) -> str:
        """Move workspace-relative bind mount targets into the ephemeral workspace."""
        workdir = self.profile.workdir.rstrip("/")
        if target == workdir:
            return INTERNAL_WORKDIR
        if target.startswith(workdir + "/"):
            return INTERNAL_WORKDIR + target[len(workdir) :]
        return target

    def _managed_mounts(self) -> list[DockerMount]:
        return [_bind(mount.source, self._rewrite_target(mount.target), read_only=mount.read_only) for mount in self.profile.mounts]

    def _python_mounts(self) -> list[DockerMount]:
        """Mount only profile locations that provide registered Python interpreters."""
        interpreter_paths = tuple(self.profile.python_interpreters.values())

        def provides_interpreter(target: str) -> bool:
            root = target.rstrip("/")
            return target == "/" or any(path == root or path.startswith(root + "/") for path in interpreter_paths)

        return [_bind(mount.source, mount.target) for mount in self.profile.mounts if provides_interpreter(mount.target)]

    @staticmethod
    def _stage_task_inputs(task: OmpTask, staging_dir: str) -> list[DockerMount]:
        """Write per-run inputs to host files and expose them as read-only bind mounts.

        Under gVisor the /runtime tmpfs lives in the sandbox kernel, invisible to Docker's
        host-side archive API, so inputs must enter the way the restore archive does.
        """
        mounts = []
        if task.append_system is not None:
            append_dir = os.path.join(staging_dir, "append-system")
            os.mkdir(append_dir)
            with open(os.path.join(append_dir, "APPEND_SYSTEM.md"), "w", encoding="utf-8") as file:
                file.write(task.append_system)
            mounts.append(_bind(append_dir, _APPEND_SYSTEM_DIR))
        if task.prompt_images:
            images_dir = os.path.join(staging_dir, "prompt-images")
            os.mkdir(images_dir)
            for image in task.prompt_images:
                with open(os.path.join(images_dir, image.filename), "xb") as file:
                    file.write(image.content)
            mounts.append(_bind(images_dir, _PROMPT_IMAGES_DIR))
        return mounts

    def _environment(self) -> dict[str, str]:
        """Forward only allowlisted variables and declared bridged host services."""
        forwarded = {name: os.environ[name] for name in self.profile.env_passthrough if name in os.environ}
        services = {name: f"http://{HOST_SERVICE_ADDRESS}:{port}" for name, port in self.profile.host_services.items()}
        return {**forwarded, **services}

    def _omp_environment(self) -> dict[str, str]:
        bin_dirs = [str(PurePosixPath(path).parent) for path in self.profile.python_interpreters.values()]
        path = ":".join([*dict.fromkeys(bin_dirs), "/usr/local/bin", "/usr/bin", "/bin"])
        return {**self._environment(), "HOME": _OMP_HOME, "PATH": path}

    @staticmethod
    def _prepare_container(container: Container, *, restoring: bool) -> None:
        """Create the workspace, extract any mounted restore archive, and seed a writable OMP home."""
        steps = [f"mkdir -p {INTERNAL_WORKDIR}"]
        if restoring:
            steps.append(
                f"tar --extract --file={_RESTORE_ARCHIVE_PATH} --directory={INTERNAL_WORKDIR} --no-same-owner --no-same-permissions"
            )
        steps.append(f"rm -rf {_OMP_HOME} && mkdir -p {_OMP_HOME}/.omp")
        steps.append(f"{{ [ ! -d /root/.omp ] || cp -a /root/.omp/. {_OMP_HOME}/.omp/; }}")
        command = " && ".join(steps)
        result = _stream_container_exec(container, ["sh", "-c", command])
        if result.exit_code != 0:
            raise ContainerRuntimeError(
                f"container setup failed (exit {result.exit_code}): {command}\n{result.stdout or result.stderr}"
            )

    def _captured_exclusions(self) -> list[str]:
        """Return workspace-relative bind-mount paths to omit from a snapshot."""
        prefix = INTERNAL_WORKDIR + "/"
        targets = (self._rewrite_target(mount.target) for mount in self.profile.mounts)
        return sorted({target[len(prefix) :] for target in targets if target.startswith(prefix)})

    def _capture_workspace(self, container: Container) -> WorkspaceArchive:
        """Stream a bounded tar snapshot to a private file, then validate and pin it."""
        exclusions = [f"--exclude=./{path}" for path in self._captured_exclusions()]
        command = ["tar", "-C", INTERNAL_WORKDIR, "--anchored", "--no-wildcards", *exclusions, "-cf", "-", "."]
        descriptor, archive_path = tempfile.mkstemp(suffix=".tar", prefix="agent-sandbox-snapshot-")
        try:
            with os.fdopen(descriptor, "wb") as archive:
                result = _stream_container_exec(container, command, workspace_sink=archive.write)
            if result.exit_code != 0:
                raise ContainerRuntimeError(f"capture failed: tar exited {result.exit_code}: {result.stderr}")
            return WorkspaceArchive(archive_path, verify_archive(archive_path))
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(archive_path)
            raise
