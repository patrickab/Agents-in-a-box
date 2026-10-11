"""Test the caller-facing error contract."""

import unittest

from agent_sandbox import (
    AgentSandboxError,
    ArchiveError,
    ContainerRuntimeError,
    ProfileError,
    PythonScriptError,
    SecurityEnvironmentError,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_every_sandbox_error_shares_the_public_base_class(self) -> None:
        self.assertTrue(issubclass(AgentSandboxError, RuntimeError))
        for error in (ProfileError, SecurityEnvironmentError, ContainerRuntimeError, ArchiveError, PythonScriptError):
            with self.subTest(error=error):
                self.assertTrue(issubclass(error, AgentSandboxError))


if __name__ == "__main__":
    unittest.main()
