"""Test doctor result aggregation without a Docker daemon."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox.doctor import (
    CheckResult,
    _check_environment,
    _check_host_services,
    _check_runsc_cgroup_handling,
    _check_runsc_startup,
    run_doctor,
)
from agent_sandbox.errors import ContainerRuntimeError, ProfileError, SecurityEnvironmentError
from agent_sandbox.profile import Profile

_STAGED_CHECKS = (
    ("_check_environment", CheckResult("docker-environment", True, "ready")),
    ("_check_runsc_cgroup_handling", CheckResult("runsc-cgroup-handling", True, "disabled")),
    ("_check_mounts_exist", [CheckResult("mount-ok:/workspace", True, "ok")]),
    ("_check_runsc_startup", CheckResult("runsc-rootless-cgroups", True, "ready")),
    ("_check_omp", CheckResult("omp-binary", True, "ready")),
    ("_check_python", [CheckResult("venv", True, "ready")]),
    ("_check_agents_md", CheckResult("agents-md-visible", True, "visible")),
    ("_check_host_services", [CheckResult("OLLAMA_HOST", True, "reachable")]),
)


def _flatten(value: CheckResult | list[CheckResult]) -> list[CheckResult]:
    return value if isinstance(value, list) else [value]


class TestRunDoctor(unittest.TestCase):
    """Exercise the doctor's staged result contract."""

    def _run(self, failing: str | None) -> tuple[list[CheckResult], dict[str, MagicMock]]:
        checks: dict[str, MagicMock] = {}
        with patch("agent_sandbox.doctor.load_profile"), patch("agent_sandbox.doctor.SandboxRuntime"):
            patches = []
            for name, result in _STAGED_CHECKS:
                if name == failing:
                    result = (
                        [replace(item, passed=False) for item in result] if isinstance(result, list) else replace(result, passed=False)
                    )
                patcher = patch(f"agent_sandbox.doctor.{name}", return_value=result)
                checks[name] = patcher.start()
                patches.append(patcher)
            try:
                return run_doctor("gigachad"), checks
            finally:
                for patcher in patches:
                    patcher.stop()

    def test_runs_every_stage_when_all_checks_pass(self) -> None:
        results, _checks = self._run(failing=None)

        self.assertEqual(results, [item for _name, result in _STAGED_CHECKS for item in _flatten(result)])

    def test_stops_after_the_stage_with_a_failure(self) -> None:
        # Stage boundaries: environment+cgroup config, mounts, runsc startup, then in-container probes.
        for failing, stage_end in (
            ("_check_environment", 2),
            ("_check_runsc_cgroup_handling", 2),
            ("_check_mounts_exist", 3),
            ("_check_runsc_startup", 4),
            ("_check_omp", 8),
        ):
            with self.subTest(failing=failing):
                results, checks = self._run(failing)

                ran = [name for name, _result in _STAGED_CHECKS[:stage_end]]
                skipped = [name for name, _result in _STAGED_CHECKS[stage_end:]]
                self.assertTrue(all(checks[name].called for name in ran))
                self.assertFalse(any(checks[name].called for name in skipped))
                self.assertIn(False, [result.passed for result in results])

    @patch("agent_sandbox.doctor.load_profile")
    def test_returns_profile_load_failure_as_result(self, load_profile: MagicMock) -> None:

        load_profile.side_effect = ProfileError("profile missing")

        self.assertEqual(run_doctor("missing"), [CheckResult("profile-load", False, "profile missing")])

    @patch("agent_sandbox.doctor.SandboxRuntime")
    @patch("agent_sandbox.doctor.load_profile")
    def test_returns_runtime_connection_failure_as_result(self, load_profile: MagicMock, runtime_class: MagicMock) -> None:

        load_profile.return_value = MagicMock()
        runtime_class.side_effect = SecurityEnvironmentError("Docker unavailable")

        self.assertEqual(run_doctor("gigachad"), [CheckResult("docker-connection", False, "Docker unavailable")])

    def test_runsc_startup_explains_upstream_rootless_cgroup_failure(self) -> None:
        import docker

        runtime = MagicMock()
        runtime.run_probe.side_effect = docker.errors.DockerException(
            "systemd error: Access denied as the requested operation requires interactive authentication."
        )

        result = _check_runsc_startup(runtime)

        runtime.run_probe.assert_called_once_with(["true"])
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "runsc-rootless-cgroups")
        self.assertIn("https://github.com/google/gvisor/issues/11543", result.detail)
        self.assertIn("rerun scripts/setup_agent_sandbox.sh", result.detail)

    def test_cgroup_check_accepts_ignore_cgroups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "daemon.json"
            config_path.write_text(
                json.dumps({"runtimes": {"runsc": {"runtimeArgs": ["--ignore-cgroups"]}}}),
                encoding="utf-8",
            )
            with patch("agent_sandbox.doctor.ROOTLESS_DOCKER_DAEMON_CONFIG", str(config_path)):
                result = _check_runsc_cgroup_handling()

        self.assertTrue(result.passed)
        self.assertEqual(result.name, "runsc-cgroup-handling")
        self.assertIn("limits are disabled", result.detail)

    def test_cgroup_check_rejects_missing_ignore_cgroups(self) -> None:
        cases = {
            "other arguments": {"runtimes": {"runsc": {"runtimeArgs": ["--debug"]}}},
            "missing arguments": {"runtimes": {"runsc": {"path": "/home/user/.local/bin/runsc"}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "daemon.json"
            with patch("agent_sandbox.doctor.ROOTLESS_DOCKER_DAEMON_CONFIG", str(config_path)):
                for name, config in cases.items():
                    with self.subTest(name=name):
                        config_path.write_text(json.dumps(config), encoding="utf-8")
                        result = _check_runsc_cgroup_handling()
                        self.assertFalse(result.passed)
                        self.assertIn("rerun scripts/setup_agent_sandbox.sh", result.detail)

    def test_cgroup_check_rejects_malformed_config(self) -> None:
        cases = {
            "invalid JSON": ('{"runtimes":', "malformed"),
            "invalid runtime arguments": (
                json.dumps({"runtimes": {"runsc": {"runtimeArgs": None}}}),
                "runtimeArgs must be a list of strings",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "daemon.json"
            with patch("agent_sandbox.doctor.ROOTLESS_DOCKER_DAEMON_CONFIG", str(config_path)):
                for name, (contents, expected_detail) in cases.items():
                    with self.subTest(name=name):
                        config_path.write_text(contents, encoding="utf-8")
                        result = _check_runsc_cgroup_handling()
                        self.assertFalse(result.passed)
                        self.assertIn(expected_detail, result.detail)
                        self.assertIn("rerun scripts/setup_agent_sandbox.sh", result.detail)

    @patch("agent_sandbox.doctor.Path.read_text", side_effect=PermissionError("permission denied"))
    def test_cgroup_check_rejects_unreadable_config(self, read_text: MagicMock) -> None:
        result = _check_runsc_cgroup_handling()

        read_text.assert_called_once_with(encoding="utf-8")
        self.assertFalse(result.passed)
        self.assertIn("could not read rootless Docker config", result.detail)
        self.assertIn("rerun scripts/setup_agent_sandbox.sh", result.detail)

    def test_environment_failure_is_reported_not_raised(self) -> None:
        runtime = MagicMock()
        runtime.verify_environment.side_effect = SecurityEnvironmentError("Docker daemon is not rootless")

        self.assertEqual(_check_environment(runtime), CheckResult("docker-environment", False, "Docker daemon is not rootless"))

    def test_host_services_probe_each_port_and_explain_failures(self) -> None:
        profile = Profile("p", "img", "/workspace", (), "/omp", host_services={"A": 1, "B": 2})
        runtime = MagicMock()
        runtime.run_probe.side_effect = [None, ContainerRuntimeError("probe exited 1")]

        results = _check_host_services(runtime, profile)

        self.assertEqual([(result.name, result.passed) for result in results], [("A", True), ("B", False)])
        self.assertIn("setup_host_services.sh", results[1].detail)
