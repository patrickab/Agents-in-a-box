"""Verify a profile and its Docker runtime prerequisites.

Return actionable results instead of raising from individual checks.
"""

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path

from agent_sandbox.config import HOST_SERVICE_ADDRESS, ROOTLESS_DOCKER_DAEMON_CONFIG
from agent_sandbox.errors import AgentSandboxError
from agent_sandbox.profile import Profile, load_profile
from agent_sandbox.runtime import SandboxRuntime
import docker

_PROBE_ERRORS = (AgentSandboxError, docker.errors.DockerException)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check_environment(runtime: SandboxRuntime) -> CheckResult:
    try:
        image_id = runtime.verify_environment()
    except AgentSandboxError as e:
        return CheckResult("docker-environment", False, str(e))
    return CheckResult("docker-environment", True, f"rootless Docker with gVisor runsc; image '{runtime.profile.image}' is {image_id}")


def _runsc_config_failure(detail: str) -> CheckResult:
    return CheckResult("runsc-cgroup-handling", False, f"{detail}. rerun scripts/setup_agent_sandbox.sh")


def _check_runsc_cgroup_handling() -> CheckResult:
    """Ensure runsc skips unsupported rootless cgroup setup."""
    config_path = Path(ROOTLESS_DOCKER_DAEMON_CONFIG)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _runsc_config_failure(f"rootless Docker config '{config_path}' is missing")
    except json.JSONDecodeError as e:
        return _runsc_config_failure(f"rootless Docker config '{config_path}' is malformed: {e.msg}")
    except (OSError, UnicodeError) as e:
        return _runsc_config_failure(f"could not read rootless Docker config '{config_path}': {e}")

    if not isinstance(config, dict):
        return _runsc_config_failure(f"rootless Docker config '{config_path}' must be a JSON object")
    runtimes = config.get("runtimes")
    if not isinstance(runtimes, dict):
        return _runsc_config_failure(f"rootless Docker config '{config_path}' field 'runtimes' must be an object")
    runsc = runtimes.get("runsc")
    if not isinstance(runsc, dict):
        return _runsc_config_failure(f"rootless Docker config '{config_path}' runtime 'runsc' must be an object")
    runtime_args = runsc.get("runtimeArgs")
    if not isinstance(runtime_args, list) or not all(isinstance(argument, str) for argument in runtime_args):
        return _runsc_config_failure(f"rootless Docker config '{config_path}' runsc runtimeArgs must be a list of strings")
    if "--ignore-cgroups" not in runtime_args:
        return _runsc_config_failure(
            "runsc runtimeArgs must contain '--ignore-cgroups' because rootless gVisor cannot create host cgroups"
        )
    return CheckResult("runsc-cgroup-handling", True, "runsc CPU, memory, and PID cgroup limits are disabled")


def _check_mounts_exist(profile: Profile) -> list[CheckResult]:
    results = []
    for m in profile.mounts:
        if not os.path.exists(m.source):
            results.append(
                CheckResult(f"mount-exists:{m.target}", False, f"mount source '{m.source}' for target '{m.target}' does not exist")
            )
        elif not os.access(m.source, os.R_OK):
            results.append(
                CheckResult(f"mount-readable:{m.target}", False, f"mount source '{m.source}' for target '{m.target}' is not readable")
            )
        else:
            results.append(CheckResult(f"mount-ok:{m.target}", True, f"'{m.source}' -> '{m.target}' ok"))
    return results


def _check_runsc_startup(runtime: SandboxRuntime) -> CheckResult:
    """Verify runsc can create a rootless container."""
    try:
        runtime.run_probe(["true"])
    except _PROBE_ERRORS as e:
        if "interactive authentication" in str(e).lower():
            return CheckResult(
                "runsc-rootless-cgroups",
                False,
                "runsc cannot create a rootless systemd cgroup due to upstream gVisor limitation "
                "https://github.com/google/gvisor/issues/11543. rerun scripts/setup_agent_sandbox.sh",
            )
        return CheckResult("runsc-rootless-cgroups", False, f"rootless runsc container failed to start: {e}")
    return CheckResult("runsc-rootless-cgroups", True, "rootless runsc container started")


def _check_omp(runtime: SandboxRuntime, profile: Profile) -> CheckResult:
    try:
        output = runtime.run_probe([profile.omp_binary, "--version"])
    except _PROBE_ERRORS as e:
        return CheckResult("omp-binary", False, f"profile field 'omp_binary' ({profile.omp_binary}) failed to start: {e}")
    return CheckResult("omp-binary", True, f"omp reports: {output.strip()}")


def _check_python(runtime: SandboxRuntime, profile: Profile) -> list[CheckResult]:
    results = []
    for label, python_path in profile.python_interpreters.items():
        try:
            output = runtime.run_probe([python_path, "-c", "import encodings, json, ssl; print('ok')"])
            results.append(CheckResult(label, True, f"{python_path} imports ok: {output.strip()}"))
        except _PROBE_ERRORS as e:
            results.append(
                CheckResult(label, False, f"profile field 'python_interpreters.{label}' ({python_path}) failed to start: {e}")
            )
    return results


def _check_agents_md(runtime: SandboxRuntime, profile: Profile) -> CheckResult:
    mount = next((m for m in profile.mounts if m.target.endswith("AGENTS.md")), None)
    if mount is None:
        return CheckResult("agents-md-visible", False, "profile declares no AGENTS.md mount")
    try:
        runtime.run_probe(["test", "-f", mount.target])
    except _PROBE_ERRORS as e:
        return CheckResult("agents-md-visible", False, f"'{mount.target}' not visible: {e}")
    return CheckResult("agents-md-visible", True, f"'{mount.target}' visible from {profile.workdir}")


def _check_host_services(runtime: SandboxRuntime, profile: Profile) -> list[CheckResult]:
    """Confirm each bridged host daemon answers from inside a sandbox container."""
    host = HOST_SERVICE_ADDRESS
    interpreter = next(iter(profile.python_interpreters.values()), None)
    if interpreter is None:
        return [CheckResult(name, False, "profile declares no python_interpreters to probe with") for name in profile.host_services]
    results = []
    for name, port in profile.host_services.items():
        url = f"http://{host}:{port}"
        script = f"import socket; socket.create_connection(('{host}', {port}), timeout=5).close(); print('ok')"
        try:
            runtime.run_probe([interpreter, "-c", script])
            results.append(CheckResult(name, True, f"host service reachable at {url}"))
        except _PROBE_ERRORS as e:
            detail = (
                f"host service at {url} unreachable; rerun scripts/setup_host_services.sh to add the {host} "
                f"loopback alias and proxy port {port} to 127.0.0.1: {e}"
            )
            results.append(CheckResult(name, False, detail))
    return results


def run_doctor(profile_name: str) -> list[CheckResult]:
    """Run profile checks in dependency stages, stopping after the first stage with a failure."""
    try:
        profile = load_profile(profile_name)
    except AgentSandboxError as e:
        return [CheckResult("profile-load", False, str(e))]
    try:
        runtime = SandboxRuntime(profile)
    except AgentSandboxError as e:
        return [CheckResult("docker-connection", False, str(e))]

    stages: tuple[Callable[[], list[CheckResult]], ...] = (
        lambda: [_check_environment(runtime), _check_runsc_cgroup_handling()],
        lambda: _check_mounts_exist(profile),
        lambda: [_check_runsc_startup(runtime)],
        lambda: [
            _check_omp(runtime, profile),
            *_check_python(runtime, profile),
            _check_agents_md(runtime, profile),
            *_check_host_services(runtime, profile),
        ],
    )
    results: list[CheckResult] = []
    for stage in stages:
        results.extend(stage())
        if not all(result.passed for result in results):
            break
    return results
