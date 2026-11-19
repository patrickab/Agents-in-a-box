"""Run `pi --mode rpc` in a hardened container with the stdio contract of a local Pi child.

pi-host's `sandboxed` profile spawns `agent-sandbox pi-rpc` instead of Pi. The container bind-mounts the
working directory and the session directory at their host paths, so session files and edits land on the
host. Pi runs from a private writable copy of the agent-dir seed that only reaches the model gateway.
Docker runs as a child rather than through exec so the seed can be removed once it exits.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import shutil
import signal
import subprocess
from types import FrameType
import uuid

from agent_sandbox.agent_dir import GATEWAY_HOST_SERVICE, GATEWAY_TOKEN_ENV, AgentSeed, build_agent_seed, gateway_environment
from agent_sandbox.config import DOCKER_HOST, HOST_SERVICE_ADDRESS
from agent_sandbox.errors import ContainerRuntimeError
from agent_sandbox.profile import Profile, load_profile
from agent_sandbox.runtime import hardened_docker_run_flags

CONTAINER_PREFIX = "agent-sandbox-pi-"
_AGENT_DIR = "/runtime/pi-agent"
_HOME = "/runtime"
_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@dataclass(frozen=True)
class PiRpcRequest:
    """Describe one sandboxed Pi RPC child as pi-host requests it."""

    profile: Profile
    cwd: str
    session_dir: str
    gateway_port: int
    pi_args: tuple[str, ...]


def container_name() -> str:
    """Return a fresh container name, so concurrent children never collide."""
    return f"{CONTAINER_PREFIX}{uuid.uuid4().hex}"


def _bind_flag(source: str, target: str, *, read_only: bool) -> str:
    return f"--mount=type=bind,source={source},target={target}{',readonly' if read_only else ''}"


def _profile_binds(profile: Profile) -> list[str]:
    """Bind the profile mounts except those inside its workdir; the host cwd replaces the workdir."""
    prefix = profile.workdir.rstrip("/") + "/"
    return [
        _bind_flag(mount.source, mount.target, read_only=mount.read_only)
        for mount in profile.mounts
        if mount.target != profile.workdir and not mount.target.startswith(prefix)
    ]


def _container_environment(request: PiRpcRequest) -> dict[str, str]:
    """Return the container variables; the gateway token is named only so its value stays out of argv."""
    profile = request.profile
    services = {name: f"http://{HOST_SERVICE_ADDRESS}:{port}" for name, port in profile.host_services.items()}
    services[GATEWAY_HOST_SERVICE] = f"http://{HOST_SERVICE_ADDRESS}:{request.gateway_port}"
    bin_dirs = [str(PurePosixPath(path).parent) for path in profile.python_interpreters.values()]
    path = ":".join([*dict.fromkeys(bin_dirs), "/usr/local/bin", "/usr/bin", "/bin"])
    pi_environment = {key: value for key, value in gateway_environment(_AGENT_DIR, "").items() if key != GATEWAY_TOKEN_ENV}
    return {**services, "HOME": _HOME, "PATH": path, **pi_environment}


def build_docker_argv(request: PiRpcRequest, seed: AgentSeed, name: str) -> list[str]:
    """Build the `docker run` argv for one sandboxed Pi RPC child."""
    profile = request.profile
    argv = ["docker", "run", "-i", "--rm", f"--name={name}", *hardened_docker_run_flags(network=True, runtime_tmpfs=True)]
    argv.extend(_profile_binds(profile))
    argv.append(_bind_flag(request.cwd, request.cwd, read_only=False))
    if request.session_dir != request.cwd:
        argv.append(_bind_flag(request.session_dir, request.session_dir, read_only=False))
    # Pi writes auth, model and session state into its agent dir, so the private seed copy binds writable.
    argv.append(_bind_flag(str(seed.directory), _AGENT_DIR, read_only=False))
    if seed.npm_dir is not None:
        argv.append(_bind_flag(str(seed.npm_dir), f"{_AGENT_DIR}/npm", read_only=True))
    passthrough = [name for name in profile.env_passthrough if name in os.environ]
    for variable in (*passthrough, GATEWAY_TOKEN_ENV):
        argv.append(f"--env={variable}")
    argv.extend(f"--env={key}={value}" for key, value in _container_environment(request).items())
    argv.extend([f"--workdir={request.cwd}", profile.image, profile.pi_binary, *request.pi_args])
    return argv


def _require_directory(label: str, path: str) -> str:
    """Return `path` unchanged; the container binds it at the same path pi-host passes in Pi's arguments."""
    if not os.path.isabs(path):
        raise ValueError(f"{label} must be an absolute path: {path}")
    if not os.path.isdir(path):
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _require_docker() -> None:
    """Fail early with one line when the Docker CLI or the rootless daemon socket is missing."""
    if shutil.which("docker") is None:
        raise ContainerRuntimeError("docker CLI not found on PATH")
    socket_path = DOCKER_HOST.removeprefix("unix://")
    if not os.path.exists(socket_path):
        raise ContainerRuntimeError(f"rootless Docker socket {socket_path} is missing; is the user docker service running?")


def prepare_request(profile_name: str, cwd: str, session_dir: str, gateway_port: int, pi_args: list[str]) -> PiRpcRequest:
    """Validate the CLI inputs and resolve the profile."""
    if not 0 < gateway_port < 65536:
        raise ValueError(f"gateway port out of range: {gateway_port}")
    return PiRpcRequest(
        profile=load_profile(profile_name),
        cwd=_require_directory("--cwd", cwd),
        session_dir=_require_directory("--session-dir", session_dir),
        gateway_port=gateway_port,
        pi_args=tuple(pi_args),
    )


def supervise(argv: list[str], name: str, environment: Mapping[str, str], seed: AgentSeed) -> int:
    """Run docker with inherited stdio, forward SIGTERM and SIGINT, and always remove the seed.

    A nonzero docker exit may leave the container behind (a killed CLI does not stop it), so it is force-removed.
    """
    child: subprocess.Popen[bytes] | None = None
    pending: list[int] = []

    def forward(signum: int, _frame: FrameType | None) -> None:
        if child is None:
            pending.append(signum)
        else:
            child.send_signal(signum)

    previous = {signum: signal.signal(signum, forward) for signum in _FORWARDED_SIGNALS}
    try:
        child = subprocess.Popen(argv, env=dict(environment))
        for signum in pending:
            child.send_signal(signum)
        code = child.wait()
        if code != 0:
            subprocess.run(
                ["docker", "rm", "-f", name], env=dict(environment), stdin=subprocess.DEVNULL, capture_output=True, check=False
            )
        return 128 - code if code < 0 else code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        seed.remove()


def run_pi_rpc(profile_name: str, cwd: str, session_dir: str, gateway_port: int, pi_args: list[str]) -> int:
    """Start one sandboxed `pi --mode rpc` child and return its exit code."""
    token = os.environ.get(GATEWAY_TOKEN_ENV, "").strip()
    if not token:
        raise ContainerRuntimeError(f"{GATEWAY_TOKEN_ENV} is not set")
    request = prepare_request(profile_name, cwd, session_dir, gateway_port, pi_args)
    _require_docker()
    seed = build_agent_seed(gateway_port, token=token)
    name = container_name()
    environment = {**os.environ, "DOCKER_HOST": DOCKER_HOST, GATEWAY_TOKEN_ENV: token}
    return supervise(build_docker_argv(request, seed, name), name, environment, seed)
