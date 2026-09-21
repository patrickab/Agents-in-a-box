"""Verify a profile and its Docker runtime prerequisites.

Return actionable results instead of raising from individual checks.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Optional

from agent_sandbox.config import ROOTLESS_DOCKER_DAEMON_CONFIG
from agent_sandbox.profile import Mount, Profile, ProfileError, load_profile
from agent_sandbox.runtime import ContainerRuntimeError, SandboxRuntime, SecurityEnvironmentError
import docker


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _runsc_config_failure(detail: str) -> CheckResult:
    return CheckResult(
        "runsc-cgroup-handling",
        False,
        f"{detail}. rerun scripts/setup_agent_sandbox.sh",
    )


def _check_runsc_cgroup_handling() -> CheckResult:
    """Ensure runsc skips unsupported rootless cgroup setup."""
    config_path = Path(ROOTLESS_DOCKER_DAEMON_CONFIG)
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _runsc_config_failure(f"rootless Docker config '{config_path}' is missing")
    except (OSError, UnicodeError) as e:
        return _runsc_config_failure(f"could not read rootless Docker config '{config_path}': {e}")

    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as e:
        return _runsc_config_failure(f"rootless Docker config '{config_path}' is malformed: {e.msg}")

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
        return _runsc_config_failure(
            f"rootless Docker config '{config_path}' runsc runtimeArgs must be a list of strings"
        )
    if "--ignore-cgroups" not in runtime_args:
        return _runsc_config_failure(
            "runsc runtimeArgs must contain '--ignore-cgroups' because rootless gVisor cannot create host cgroups"
        )
    return CheckResult("runsc-cgroup-handling", True, "runsc CPU, memory, and PID cgroup limits are disabled")



def _check_rootless_and_runsc(runtime: SandboxRuntime) -> list:
    try:
        info = runtime.client.info()
    except docker.errors.DockerException as e:
        return [CheckResult("docker-connection", False, f"Cannot reach Docker daemon: {e}")]

    results = []
    security_opts = info.get("SecurityOptions", [])
    rootless = any("rootless" in opt.lower() for opt in security_opts)
    results.append(CheckResult(
        "rootless-docker", rootless,
        "rootless mode enabled" if rootless else "Docker daemon is not rootless; rerun scripts/setup_agent_sandbox.sh",
    ))
    runsc = "runsc" in info.get("Runtimes", {})
    results.append(CheckResult(
        "gvisor-runsc", runsc,
        "runsc runtime registered" if runsc else "gVisor 'runsc' missing from Docker daemon.json; rerun setup script",
    ))
    return results


def _check_image(runtime: SandboxRuntime, profile: Profile) -> CheckResult:
    try:
        runtime.client.images.get(profile.image)
        return CheckResult("image-available", True, f"image '{profile.image}' present")
    except docker.errors.ImageNotFound:
        return CheckResult("image-available", False, f"profile field 'image': '{profile.image}' not found; build docker/Dockerfile")
    except docker.errors.DockerException as e:
        return CheckResult("image-available", False, f"could not inspect image '{profile.image}': {e}")


def _check_mounts_exist(profile: Profile) -> list:
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


def _find_mount(profile: Profile, predicate: Callable[[Mount], bool]) -> Optional[Mount]:
    return next((m for m in profile.mounts if predicate(m)), None)


def _run_ephemeral(runtime: SandboxRuntime, argv: list[str]) -> bytes:
    """Run a check in a disposable container through the runtime seam."""
    return runtime.run_probe(argv)

def _check_runsc_startup(runtime: SandboxRuntime) -> CheckResult:
    """Verify runsc can create a rootless container."""
    try:
        _run_ephemeral(runtime, ["true"])
    except docker.errors.DockerException as e:
        detail = str(e)
        if "interactive authentication" in detail.lower():
            return CheckResult(
                "runsc-rootless-cgroups",
                False,
                "runsc cannot create a rootless systemd cgroup due to upstream gVisor limitation "
                "https://github.com/google/gvisor/issues/11543. rerun scripts/setup_agent_sandbox.sh",
            )
        return CheckResult(
            "runsc-rootless-cgroups",
            False,
            f"rootless runsc container failed to start: {e}",
        )
    return CheckResult(
        "runsc-rootless-cgroups",
        True,
        "rootless runsc container started",
    )


def _check_omp(runtime: SandboxRuntime, profile: Profile) -> CheckResult:
    try:
        output = _run_ephemeral(runtime, [profile.omp_binary, "--version"])
        return CheckResult("omp-binary", True, f"omp reports: {output.decode(errors='replace').strip()}")
    except docker.errors.DockerException as e:
        return CheckResult("omp-binary", False, f"profile field 'omp_binary' ({profile.omp_binary}) failed to start: {e}")


def _check_python(runtime: SandboxRuntime, profile: Profile) -> list:
    results = []
    for label, python_path in profile.python_interpreters:
        try:
            output = _run_ephemeral(runtime, [python_path, "-c", "import encodings, json, ssl; print('ok')"])
            results.append(CheckResult(label, True, f"{python_path} imports ok: {output.decode(errors='replace').strip()}"))
        except docker.errors.DockerException as e:
            results.append(
                CheckResult(label, False, f"profile field 'python_interpreters.{label}' ({python_path}) failed to start: {e}")
            )
    return results


def _check_agents_md(runtime: SandboxRuntime, profile: Profile) -> CheckResult:
    mount = _find_mount(profile, lambda m: m.target.endswith("AGENTS.md"))
    if mount is None:
        return CheckResult("agents-md-visible", False, "profile declares no AGENTS.md mount")
    try:
        _run_ephemeral(runtime, ["test", "-f", mount.target])
        return CheckResult("agents-md-visible", True, f"'{mount.target}' visible from {profile.workdir}")
    except docker.errors.DockerException as e:
        return CheckResult("agents-md-visible", False, f"'{mount.target}' not visible: {e}")


def _check_host_services(runtime: SandboxRuntime, profile: Profile) -> list:
    """Confirm each bridged host daemon answers from inside a sandbox container."""
    if not profile.host_services:
        return []
    try:
        host = runtime.host_address()
    except ContainerRuntimeError as e:
        return [CheckResult("host-services", False, str(e))]
    interpreter = next((path for _label, path in profile.python_interpreters), None)
    results = []
    for name, port in profile.host_services:
        url = f"http://{host}:{port}"
        if interpreter is None:
            results.append(CheckResult(name, False, "profile declares no python_interpreters to probe with"))
            continue
        script = f"import socket; socket.create_connection(('{host}', {port}), timeout=5).close(); print('ok')"
        try:
            _run_ephemeral(runtime, [interpreter, "-c", script])
            results.append(CheckResult(name, True, f"host service reachable at {url}"))
        except docker.errors.DockerException as e:
            detail = f"host service at {url} unreachable; bind the daemon to {host} or drop it from host_services: {e}"
            results.append(CheckResult(name, False, detail))
    return results


def run_doctor(profile_name: str) -> list:
    """Run every profile check and return its results."""
    try:
        profile = load_profile(profile_name)
    except ProfileError as e:
        return [CheckResult("profile-load", False, str(e))]

    try:
        runtime = SandboxRuntime(profile)
    except SecurityEnvironmentError as e:
        return [CheckResult("docker-connection", False, str(e))]

    prerequisites = list(_check_rootless_and_runsc(runtime))
    if not all(result.passed for result in prerequisites):
        return prerequisites
    prerequisites.append(_check_runsc_cgroup_handling())
    if not all(result.passed for result in prerequisites):
        return prerequisites

    results = prerequisites
    image = _check_image(runtime, profile)
    mounts = _check_mounts_exist(profile)
    results.append(image)
    results.extend(mounts)
    if not image.passed or not all(result.passed for result in mounts):
        return results

    runsc_startup = _check_runsc_startup(runtime)
    results.append(runsc_startup)
    if not runsc_startup.passed:
        return results

    results.append(_check_omp(runtime, profile))
    results.extend(_check_python(runtime, profile))
    results.append(_check_agents_md(runtime, profile))
    results.extend(_check_host_services(runtime, profile))
    return results
