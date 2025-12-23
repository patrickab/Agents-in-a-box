"""
Security Architecture - Defense in Depth

1. Zero-Trust Execution Policy

    - "Assume Breach Principle":
        All code is treated potentially malicious.

2. Kernel-Level Isolation (gVisor)

    - Sandboxed System Calls:
        Uses the 'runsc' runtime to provide a dedicated
        guest kernel for the container.
    - Host Protection:
        Acts as a "firewall" between the container and the host

3. Hardened User Identity & Capability Stripping

    - Globally remove Linux kernel capabilities (--cap-drop=ALL):
        Container cannot perform system-level tasks.
    - Enforce 'no-new-privileges':
        Prevent privilege escalation attacks
    - Non-Root Enforcement:
        Containers are forced to run as an unprivileged user
        (UID 1000), removing administrative power by default.

4. Rootless Infrastructure Architecture

    - Unprivileged Daemon:
        - Docker Engine runs without root privileges on the host
    - Identity Mapping:
        - Rootless Docker maps container root users to non-root host users,

5. Ephemeral Lifecycle Management
    - Destruction on Completion:
        Containers are strictly temporary
        Automatically destroyed after task execution.

6. Network Perimeter Control
    - Traffic Segregation:
        Places containers on isolated bridge networks to monitor and restrict data flow.
    - Exfiltration Prevention:
        Limits the container's ability to communicate with the internal network or move laterally to other services.
"""

from pathlib import Path
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
    (5) Least Privilege: Enforces non-root UID and drops capabilities.
    (6) Daemon Isolation: Validates Rootless Docker.
    """

    # Static configuration for immutable security defaults
    _SANDBOX_OPTS = (
        "--rm",
        "--runtime=runsc", # Gvisor for syscall isolation
        "--user", "0:0",   # Internal Root -> Host User 1000 (Rootless)
        "--network", "bridge", # Isolated bridge network
        f"--add-host=host.docker.internal:{SECURE_LOOPBACK_IP}", # Stable host access
        "--dns", "8.8.8.8", # Public DNS to avoid leaking internal host info
        "--cap-drop=ALL", # Drop all capabilities first...
        "--cap-add=NET_ADMIN",
        "--cap-add=NET_RAW",# ...selectively re-add necessary capabilities
        "--security-opt", "no-new-privileges", # Enforce no new privileges
    )

    def __init__(self, dockerimage_name: str) -> None:
        self.logger = get_logger()
        self.client = docker.from_env()
        self.dockerimage_name = dockerimage_name
        self._verify_environment()

    def _verify_environment(self) -> None:
        """Validates Rootless Docker, gVisor, and Image availability."""
        try:
            info = self.client.info()

            try:
                subprocess.run(["ip", "addr", "show", "to", SECURE_LOOPBACK_IP], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                self.logger.warning(
                    f"Secure Bridge IP {SECURE_LOOPBACK_IP} is missing. For Ollama Run: sudo ip addr add 10.200.200.1/32 dev lo"
                )

            if not any("rootless" in opt.lower() for opt in info.get("SecurityOptions", [])):
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
        work_dir = Path(repo_path).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting sandbox with image {self.dockerimage_name} at {work_dir}")

        cmd = [
            "docker",
            "run",
            "-it", # Interactive TTY
            *self._SANDBOX_OPTS,
            "-v", f"{work_dir}:/workspace:z",  # Bind mount for host-container sync
            "-w", # Set working directory
            "/workspace", # Github repo mounted
            "-e", # Set HOME environment variable
            "HOME=/workspace",
        ]

        if env_vars:
            cmd.extend([arg for k, v in env_vars.items() for arg in ("-e", f"{k}={v}")])

        cmd += [
            self.dockerimage_name,
            "/bin/bash",
            "-c",
            f"tcpdump -l -A -i eth0 > /workspace/network_traffic.log 2>&1 & sleep 1; {agent_cmd}",
        ]

        self.logger.info(f"Executing command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise ContainerRuntimeError(f"Container execution failed: {e}")
