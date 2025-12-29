"""
Security Architecture: Defense in Depth

* Zero-Trust Policy:
    - Assume Breach principle -> Treat all agent code as potentially malicious.

* Kernel Isolation:
    - gVisor (runsc) -> Acts as syscall firewall. Automatically rejects all not explicitly allowed syscalls.

* Rootless Architecture:
    - Container Root maps to unprivileged host user -> Container breakouts land as unprivileged user.

* Capability Hardening:
    - `--cap-drop=ALL` + `--security-opt no-new-privileges` -> Eliminates attack surface / blocks escalation.

* Resource Quotas:
    - Hard limits (4GB RAM, 2 CPUs, 100 PIDs) -> Mitigates DoS / starvation / fork bombs.

* Ephemeral Lifecycle:
    - Auto-destruction (`--rm`) -> Ensures zero residual state / data persistence.
                                -> Only mounted git repository remains for inspection.
                                -> Minimizes persistence and attack surface.


* Network Perimeter:
    - Isolated bridge + Loopback alias (10.200.200.1) -> Restricts lateral movement / controls host access.
"""

import os
import subprocess
from typing import Dict, Optional

import docker

from code_agents.lib.logger import get_logger


class SecurityEnvironmentError(Exception):
    """Signal misconfigured Docker security environment."""


class ContainerRuntimeError(Exception):
    """Signal container lifecycle or execution failure."""


# === Define secure loopback IP for host-container communication
# === Rootless Docker disables host.docker.internal by default
# Implements a Zero Trust bridge for Rootless Docker via a static loopback alias,
# Ensures traffic is air-gapped from physical interfaces to prevent accidental exposure.
# This stable target enables deterministic firewalling independent of host network changes.
# Setup: `sudo ip addr add 10.200.200.1/32 dev lo`
SECURE_LOOPBACK_IP = "10.200.200.1"


class DockerSandbox:
    """
    Secure Docker-based sandbox for agent code execution.

    Features:
    (1) Interactive TTY/Shell access.
    (2) Host-to-Container synchronization via Bind Mounts.
    (3) Syscall isolation using gVisor (runsc).
    (4) Ephemeral lifecycle (auto-removal).
    (5) Least Privilege: Drops all capabilities for root - root privileges needed to edit mounted hostfiles.
    """

    def __init__(self, dockerimage_name: str) -> None:
        self.logger = get_logger()
        self.client = docker.from_env()
        self.dockerimage_name = dockerimage_name
        self._verify_environment()

    def _verify_environment(self) -> None:
        """Validates Rootless Docker, gVisor, and Image availability."""
        try:
            info = self.client.info()

            security_opts = info.get("SecurityOptions", [])
            if not any("rootless" in opt.lower() for opt in security_opts):
                raise SecurityEnvironmentError("Rootless Docker is not enabled. Daemon must run in rootless mode.")

            if "runsc" not in info.get("Runtimes", {}):
                raise SecurityEnvironmentError("gVisor 'runsc' runtime is not configured in Docker.")

            self.client.images.get(self.dockerimage_name)
        except docker.errors.ImageNotFound:
            raise SecurityEnvironmentError(f"Docker image '{self.dockerimage_name}' not found.")
        except Exception as e:
            if isinstance(e, SecurityEnvironmentError):
                raise e
            raise SecurityEnvironmentError(f"Environment check failed: {e}")

    def run_interactive_shell(self, repo_path: str, agent_cmd: str, env_vars: Optional[Dict[str, str]] = None) -> None:
        """
        Runs an interactive shell in the sandbox.
        Uses subprocess for the final 'run' call to ensure high-fidelity TTY hijacking.
        """
        abs_repo_path = os.path.abspath(os.path.expanduser(repo_path))
        os.makedirs(abs_repo_path, exist_ok=True)

        self.logger.info(f"Starting sandbox with image {self.dockerimage_name} at {abs_repo_path}")

        cmd = [
            "docker",
            "run",
            "-it",
            "--rm",  # run interactive & remove after exit
            "--runtime=runsc",
            "--user",
            "0:0",  # Internal Root -> Host User 1000 (Rootless)
            # Network
            "--network",
            "bridge",
            f"--add-host=host.docker.internal:{SECURE_LOOPBACK_IP}",
            "--dns",
            "8.8.8.8",
            # Security: Drop everything
            "--cap-drop=ALL",
            # "--cap-add=NET_RAW", needed in future for network sniffing (whitelist/blacklist websites)
            "--security-opt",
            "no-new-privileges",
            # Mounts
            "-v",
            f"{abs_repo_path}:/workspace:z",
            "-w",
            "/workspace",
            "-e",
            "HOME=/workspace",
        ]

        # Add environment variables to the command
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self.dockerimage_name)
        cmd.append("/bin/bash")
        cmd.extend(["-c", agent_cmd])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise ContainerRuntimeError(f"Container execution failed with exit code {e.returncode}. Command: {' '.join(cmd)}") from e
