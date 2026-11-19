"""Run profiled Pi work in short-lived rootless Docker/gVisor containers.

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
import shlex
import tempfile
import threading
from typing import Any

from docker.models.containers import Container
from docker.types import Mount as DockerMount

from agent_sandbox.agent_dir import (
    GATEWAY_HOST_SERVICE,
    GATEWAY_PROVIDER,
    AgentSeed,
    build_agent_seed,
    gateway_environment,
    read_gateway_token,
)
from agent_sandbox.archive import WorkspaceArchive, verify_archive
from agent_sandbox.config import (
    DOCKER_HOST,
    HOST_SERVICE_ADDRESS,
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
from agent_sandbox.pi import PiEventStream, PiOutcome, PiTask, build_pi_argv
from agent_sandbox.profile import Profile
import docker

INTERNAL_WORKDIR = "/runtime/active"
_SESSION_DIR = f"{INTERNAL_WORKDIR}/.pi-session"
_PROMPT_IMAGES_DIR = "/runtime/prompt-images"
_APPEND_SYSTEM_DIR = "/runtime/append-system"
_APPEND_SYSTEM_PATH = f"{_APPEND_SYSTEM_DIR}/APPEND_SYSTEM.md"
_HOME = "/runtime/home"
_AGENT_SEED_DIR = "/runtime/pi-agent-seed"
# Pi writes to its agent dir, so the read-only seed is copied here; the host package tree binds read-only inside it.
_AGENT_DIR = "/runtime/pi-agent"
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
    """Restore a pinned workspace archive, optionally continuing the Pi session inside it."""

    archive: WorkspaceArchive
    prior_runtime_fingerprint: str | None = None  # Set when resuming; must equal this run's effective fingerprint.

    @property
    def resuming(self) -> bool:
        return self.prior_runtime_fingerprint is not None


@dataclass(frozen=True)
class ManagedExecutionResult:
    """Return Pi's outcome and the transient workspace capture from one transaction."""

    exit_code: int
    outcome: PiOutcome
    stderr: str
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


def hardened_container_options(*, network: bool, runtime_tmpfs: bool = False) -> dict[str, Any]:
    """Return the Docker SDK create options every sandbox container shares."""
    tmpfs = {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"}
    if runtime_tmpfs:
        # Pi's native assets and its agent dir live here, so it must allow exec.
        tmpfs["/runtime"] = f"rw,exec,nosuid,nodev,size={RUNTIME_TMPFS_SIZE}"
    return {
        "read_only": True,
        "tmpfs": tmpfs,
        "runtime": "runsc",
        "user": "0:0",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        **({"network_mode": "bridge"} if network else {"network_disabled": True}),
    }


def hardened_docker_run_flags(*, network: bool, runtime_tmpfs: bool = False) -> list[str]:
    """Render hardened_container_options as `docker run` flags for callers that exec the Docker CLI."""
    options = hardened_container_options(network=network, runtime_tmpfs=runtime_tmpfs)
    return [
        "--read-only",
        *(f"--tmpfs={path}:{mount_options}" for path, mount_options in options["tmpfs"].items()),
        f"--runtime={options['runtime']}",
        f"--user={options['user']}",
        *(f"--cap-drop={capability}" for capability in options["cap_drop"]),
        *(f"--security-opt={option}" for option in options["security_opt"]),
        f"--network={options.get('network_mode', 'none')}",
    ]


def _stream_container_exec(
    container: Container,
    command: list[str],
    *,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    stdout_sink: Callable[[bytes], None] | None = None,
) -> ExecResult:
    """Run one exec with bounded output.

    Retained stdout and stderr share one MAX_RETAINED_OUTPUT_BYTES budget. With a stdout sink,
    stdout streams into it unretained and the sink bounds what it keeps.
    """
    api = container.client.api
    exec_id = api.exec_create(container.id, command, stdout=True, stderr=True, environment=environment, workdir=workdir)["Id"]
    stdout, stderr = bytearray(), bytearray()
    with closing(api.exec_start(exec_id, stream=True, demux=True)) as stream:
        for stdout_chunk, stderr_chunk in stream:
            if stderr_chunk:
                stderr.extend(stderr_chunk)
            if stdout_chunk and stdout_sink is None:
                stdout.extend(stdout_chunk)
            elif stdout_chunk:
                stdout_sink(stdout_chunk)
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
        container = self.client.containers.create(
            image,
            command=["sleep", "infinity"],
            name=name,
            mounts=mounts,
            environment=environment or {},
            **hardened_container_options(network=network, runtime_tmpfs=runtime_tmpfs),
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

    def execute(self, task: PiTask, restore: WorkspaceRestore | None = None) -> ManagedExecutionResult:
        """Run one Pi task in a fresh managed container under the namespace lease."""
        gateway_port = self.profile.host_services.get(GATEWAY_HOST_SERVICE)
        if gateway_port is None:
            raise ContainerRuntimeError(f"profile '{self.profile.name}' declares no {GATEWAY_HOST_SERVICE} host service")
        with (
            _runtime_lease(self._resource_names.lock_path),
            tempfile.TemporaryDirectory(prefix="agent-sandbox-inputs-") as staging_dir,
        ):
            image_id = self.verify_environment()
            self._remove_stale_container()
            fingerprint = self._effective_runtime_fingerprint(image_id)
            if restore is not None:
                self._verify_restore(restore, fingerprint)
            token = read_gateway_token()
            with build_agent_seed(gateway_port, token=token) as seed:
                return self._execute_in_container(task, restore, staging_dir, seed, token, image_id, fingerprint)

    def _execute_in_container(
        self,
        task: PiTask,
        restore: WorkspaceRestore | None,
        staging_dir: str,
        seed: AgentSeed,
        token: str,
        image_id: str,
        fingerprint: str,
    ) -> ManagedExecutionResult:
        """Run Pi in one fresh named container and capture its workspace."""
        mounts = [*self._managed_mounts(), *self._stage_task_inputs(task, staging_dir), _bind(str(seed.directory), _AGENT_SEED_DIR)]
        if seed.npm_dir is not None:
            mounts.append(_bind(str(seed.npm_dir), f"{_AGENT_DIR}/npm"))
        if restore is not None:
            mounts.append(_bind(restore.archive.path, _RESTORE_ARCHIVE_PATH))
        argv = build_pi_argv(
            self.profile.pi_binary,
            task,
            session_dir=_SESSION_DIR,
            resuming=restore is not None and restore.resuming,
            images_dir=_PROMPT_IMAGES_DIR,
            append_system_path=_APPEND_SYSTEM_PATH,
            provider=GATEWAY_PROVIDER,
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
                events = PiEventStream()
                exec_result = _stream_container_exec(
                    container,
                    ["timeout", "--signal=KILL", f"{task.timeout_s}s", *argv],
                    workdir=INTERNAL_WORKDIR,
                    environment=self._pi_environment(token),
                    stdout_sink=events.feed,
                )
                if exec_result.exit_code in _TIMEOUT_EXIT_CODES:
                    raise ContainerRuntimeError(f"Pi execution exceeded {task.timeout_s}s limit")
                capture = self._capture_workspace(container)
        except BaseException:
            if capture is not None:  # Only container removal can fail after capture.
                with suppress(FileNotFoundError):
                    os.unlink(capture.path)
            raise
        return ManagedExecutionResult(exec_result.exit_code, events.finish(), exec_result.stderr, capture, fingerprint)

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
    def _stage_task_inputs(task: PiTask, staging_dir: str) -> list[DockerMount]:
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

    def _pi_environment(self, token: str) -> dict[str, str]:
        """Give the Pi exec a writable home, the profile interpreters on PATH, and its gateway agent dir."""
        bin_dirs = [str(PurePosixPath(path).parent) for path in self.profile.python_interpreters.values()]
        path = ":".join([*dict.fromkeys(bin_dirs), "/usr/local/bin", "/usr/bin", "/bin"])
        return {**self._environment(), "HOME": _HOME, "PATH": path, **gateway_environment(_AGENT_DIR, token)}

    def _prepare_container(self, container: Container, *, restoring: bool) -> None:
        """Create the workspace, extract any mounted restore archive, and copy the agent-dir seed into a writable home."""
        steps = [f"mkdir -p {INTERNAL_WORKDIR}"]
        if restoring:
            skipped = [
                f"--exclude={shlex.quote(prefix + path)}" for path in self.profile.workdir_mount_paths() for prefix in ("./", "")
            ]
            extract = [
                "tar",
                "--extract",
                f"--file={_RESTORE_ARCHIVE_PATH}",
                f"--directory={INTERNAL_WORKDIR}",
                "--no-same-owner",
                "--no-same-permissions",
                "--anchored",
                "--no-wildcards",
                *skipped,
            ]
            steps.append(" ".join(extract))
        steps.append(f"mkdir -p {_HOME} {_AGENT_DIR}")
        steps.append(f"cp -R {_AGENT_SEED_DIR}/. {_AGENT_DIR}/")
        command = " && ".join(steps)
        result = _stream_container_exec(container, ["sh", "-c", command])
        if result.exit_code != 0:
            raise ContainerRuntimeError(
                f"container setup failed (exit {result.exit_code}): {command}\n{result.stdout or result.stderr}"
            )

    def _capture_workspace(self, container: Container) -> WorkspaceArchive:
        """Stream a bounded tar snapshot to a private file, then validate and pin it."""
        exclusions = [f"--exclude=./{path}" for path in self.profile.workdir_mount_paths()]
        command = ["tar", "-C", INTERNAL_WORKDIR, "--anchored", "--no-wildcards", *exclusions, "-cf", "-", "."]
        descriptor, archive_path = tempfile.mkstemp(suffix=".tar", prefix="agent-sandbox-snapshot-")
        try:
            with os.fdopen(descriptor, "wb") as archive:
                written = 0

                def write_bounded(chunk: bytes) -> None:
                    nonlocal written
                    written += len(chunk)
                    if written > MAX_WORKSPACE_ARCHIVE_BYTES:
                        raise ArchiveError(f"captured workspace exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} byte limit")
                    archive.write(chunk)

                result = _stream_container_exec(container, command, stdout_sink=write_bounded)
            if result.exit_code != 0:
                raise ContainerRuntimeError(f"capture failed: tar exited {result.exit_code}: {result.stderr}")
            return WorkspaceArchive(archive_path, verify_archive(archive_path))
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(archive_path)
            raise
