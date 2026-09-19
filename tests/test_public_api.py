"""Test the supported top-level package API."""

import unittest
from unittest.mock import patch

import agent_sandbox
from agent_sandbox.invocation import PromptImage, Sandbox, SandboxInvocation, SandboxRun, run, run_python_script
from agent_sandbox.manifest import SandboxManifest, manifest_from_dict, manifest_to_dict
from agent_sandbox.profile import Profile, load_profile
from agent_sandbox.runtime import AgentSandboxError, ContainerRuntimeError, PythonScriptError, SecurityEnvironmentError


class TestPublicAPI(unittest.TestCase):
    def test_supported_exports_are_available_at_package_top_level(self) -> None:
        self.assertEqual(
            set(agent_sandbox.__all__),
            {
                "AgentSandboxError",
                "Profile",
                "PromptImage",
                "Sandbox",
                "PythonScriptError",
                "SandboxInvocation",
                "SandboxManifest",
                "SandboxRun",
                "load_profile",
                "manifest_from_dict",
                "manifest_to_dict",
                "run",
                "run_python_script",
            },
        )
        self.assertIs(agent_sandbox.Sandbox, Sandbox)
        self.assertIs(agent_sandbox.AgentSandboxError, AgentSandboxError)
        self.assertIs(agent_sandbox.Profile, Profile)
        self.assertIs(agent_sandbox.PromptImage, PromptImage)
        self.assertIs(agent_sandbox.PythonScriptError, PythonScriptError)
        self.assertIs(agent_sandbox.SandboxInvocation, SandboxInvocation)
        self.assertIs(agent_sandbox.SandboxManifest, SandboxManifest)
        self.assertIs(agent_sandbox.SandboxRun, SandboxRun)
        self.assertIs(agent_sandbox.manifest_from_dict, manifest_from_dict)
        self.assertIs(agent_sandbox.manifest_to_dict, manifest_to_dict)
        self.assertIs(agent_sandbox.load_profile, load_profile)
        self.assertIs(agent_sandbox.run, run)
        self.assertIs(agent_sandbox.run_python_script, run_python_script)

    def test_runtime_errors_share_the_public_base_class(self) -> None:
        self.assertTrue(issubclass(AgentSandboxError, RuntimeError))
        for error in (SecurityEnvironmentError, ContainerRuntimeError, PythonScriptError):
            with self.subTest(error=error):
                self.assertTrue(issubclass(error, AgentSandboxError))

    @patch("agent_sandbox.invocation._default_sandbox")
    def test_top_level_helpers_delegate_to_default_sandbox(self, mock_default_sandbox) -> None:
        invocation = SandboxInvocation("slot", "profile", "prompt", "run")
        expected_run = object()
        mock_default_sandbox.run.return_value = expected_run
        mock_default_sandbox.run_python_script.return_value = "output"

        self.assertIs(run(invocation), expected_run)
        self.assertEqual(run_python_script("python", "print(1)", 2, profile_name="test"), "output")
        mock_default_sandbox.run.assert_called_once_with(invocation)
        mock_default_sandbox.run_python_script.assert_called_once_with("python", "print(1)", 2, profile_name="test")


if __name__ == "__main__":
    unittest.main()
