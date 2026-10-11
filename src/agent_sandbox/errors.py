"""Define the caller-facing sandbox error hierarchy."""


class AgentSandboxError(RuntimeError):
    """Base error for every caller-facing sandbox failure."""


class ProfileError(AgentSandboxError):
    """Report an invalid or unavailable profile."""


class SecurityEnvironmentError(AgentSandboxError):
    """Report missing rootless Docker, runsc, or image prerequisites."""


class ContainerRuntimeError(AgentSandboxError):
    """Report a container lifecycle or command failure."""


class ArchiveError(AgentSandboxError):
    """Report an unsafe, oversized, mismatched, or unreadable workspace archive."""


class PythonScriptError(AgentSandboxError):
    """Report a failed or timed-out disposable Python execution."""
