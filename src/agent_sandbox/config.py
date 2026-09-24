"""Define static sandbox runtime settings."""

from dataclasses import dataclass
import os
import re

IMAGE_PREFIX = "agent-sandbox"
DEFAULT_IMAGE_TAG = "agent-sandbox:trixie"
RUNTIME_CONTAINER_NAME = "agent-sandbox-runtime"
DOCKER_HOST = f"unix:///run/user/{os.getuid()}/docker.sock"
ROOTLESS_DOCKER_DAEMON_CONFIG = os.path.join(os.path.expanduser("~"), ".config", "docker", "daemon.json")
# Rootless Docker blocks host loopback, so containers reach declared host services through this
# lo alias. scripts/setup_agent_sandbox.sh adds it and proxies each service port to 127.0.0.1.
HOST_SERVICE_ADDRESS = "10.200.200.1"

_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class RuntimeResourceNames:
    """Name the Docker container and lock used by one managed runtime."""

    container: str
    lock_path: str


def runtime_resource_names(namespace: str | None = None) -> RuntimeResourceNames:
    """Return validated deterministic transient-container names for an optional namespace."""
    if namespace is None:
        return RuntimeResourceNames(RUNTIME_CONTAINER_NAME, "/tmp/agent-sandbox-runtime.lock")
    if not isinstance(namespace, str) or not _NAMESPACE_PATTERN.fullmatch(namespace):
        raise ValueError("namespace must be a lowercase alphanumeric slug with optional hyphens (max 64 characters)")
    prefix = f"agent-sandbox-{namespace}-runtime"
    return RuntimeResourceNames(prefix, f"/tmp/{prefix}.lock")


MANAGED_OMP_TIMEOUT_SECONDS = 300
MAX_PROMPT_IMAGE_BYTES = 32 * 1024 * 1024
MAX_RETAINED_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_RESTORE_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_CAPTURED_WORKSPACE_BYTES = 512 * 1024 * 1024
RUNTIME_TMPFS_SIZE = "1g"
TMP_TMPFS_SIZE = "64m"

# Bump this marker whenever static runtime behavior changes in a way that affects reuse.
RUNTIME_POLICY_DESCRIPTOR = "agent-sandbox-runtime-policy-v4"

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

_BUNDLED_PROFILES_DIR = PROFILES_DIR
PROFILE_SEARCH_PATH = [
    p for p in os.environ.get("AGENT_SANDBOX_PROFILE_PATH", "").split(os.pathsep) if p
] + [_BUNDLED_PROFILES_DIR]
