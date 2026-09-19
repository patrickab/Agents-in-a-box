"""Test doctor result aggregation without a Docker daemon."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox.doctor import (
    CheckResult,
    _check_runsc_cgroup_handling,
    _check_runsc_startup,
    run_doctor,
)


class TestRunDoctor(unittest.TestCase):
    """Exercise the doctor's public aggregated result contract."""

    @patch("agent_sandbox.doctor._check_host_services")
    @patch("agent_sandbox.doctor._check_agents_md")
    @patch("agent_sandbox.doctor._check_python")
    @patch("agent_sandbox.doctor._check_omp")
    @patch("agent_sandbox.doctor._check_runsc_startup")
    @patch("agent_sandbox.doctor._check_mounts_exist")
    @patch("agent_sandbox.doctor._check_image")
    @patch("agent_sandbox.doctor._check_runsc_cgroup_handling")
    @patch("agent_sandbox.doctor._check_rootless_and_runsc")
    @patch("agent_sandbox.doctor.SandboxRuntime")
    @patch("agent_sandbox.doctor.load_profile")
    def test_aggregates_every_check_result(
        self,
        load_profile: MagicMock,
        runtime_class: MagicMock,
        rootless: MagicMock,
        cgroup_handling: MagicMock,
        image: MagicMock,
        mounts: MagicMock,
        runsc_startup: MagicMock,
        omp: MagicMock,
        python: MagicMock,
        agents_md: MagicMock,
        host_services: MagicMock,
    ) -> None:
        profile = MagicMock()
        runtime = MagicMock()
        load_profile.return_value = profile
        runtime_class.return_value = runtime
        rootless.return_value = [CheckResult("rootless-docker", True, "enabled")]
        cgroup_handling.return_value = CheckResult("runsc-cgroup-handling", True, "limits enforced")
        image.return_value = CheckResult("image-available", True, "present")
        runsc_startup.return_value = CheckResult("runsc-rootless-cgroups", True, "ready")
        mounts.return_value = [CheckResult("mount-ok:/workspace", True, "ok")]
        omp.return_value = CheckResult("omp-binary", True, "ready")
        python.return_value = [CheckResult("venv", True, "ready")]
        agents_md.return_value = CheckResult("agents-md-visible", True, "visible")
        host_services.return_value = [CheckResult("OLLAMA_HOST", False, "unreachable")]

        results = run_doctor("gigachad")

        runtime_class.assert_called_once_with(profile)
        rootless.assert_called_once_with(runtime)
        cgroup_handling.assert_called_once_with()
        image.assert_called_once_with(runtime, profile)
        mounts.assert_called_once_with(profile)
        runsc_startup.assert_called_once_with(runtime)
        omp.assert_called_once_with(runtime, profile)
        python.assert_called_once_with(runtime, profile)
        agents_md.assert_called_once_with(runtime, profile)
        host_services.assert_called_once_with(runtime, profile)
        self.assertEqual(
            results,
            [
                CheckResult("rootless-docker", True, "enabled"),
                CheckResult("runsc-cgroup-handling", True, "limits enforced"),
                CheckResult("image-available", True, "present"),
                CheckResult("mount-ok:/workspace", True, "ok"),
                CheckResult("runsc-rootless-cgroups", True, "ready"),
                CheckResult("omp-binary", True, "ready"),
                CheckResult("venv", True, "ready"),
                CheckResult("agents-md-visible", True, "visible"),
                CheckResult("OLLAMA_HOST", False, "unreachable"),
            ],
        )

    @patch("agent_sandbox.doctor._check_host_services")
    @patch("agent_sandbox.doctor._check_agents_md")
    @patch("agent_sandbox.doctor._check_python")
    @patch("agent_sandbox.doctor._check_omp")
    @patch("agent_sandbox.doctor._check_runsc_startup")
    @patch("agent_sandbox.doctor._check_mounts_exist")
    @patch("agent_sandbox.doctor._check_image")
    @patch("agent_sandbox.doctor._check_runsc_cgroup_handling")
    @patch("agent_sandbox.doctor._check_rootless_and_runsc")
    @patch("agent_sandbox.doctor.SandboxRuntime")
    @patch("agent_sandbox.doctor.load_profile")
    def test_stops_before_runtime_probes_when_prerequisite_fails(
        self,
        load_profile: MagicMock,
        runtime_class: MagicMock,
        rootless: MagicMock,
        cgroup_handling: MagicMock,
        image: MagicMock,
        mounts: MagicMock,
        runsc_startup: MagicMock,
        omp: MagicMock,
        python: MagicMock,
        agents_md: MagicMock,
        host_services: MagicMock,
    ) -> None:
        profile = MagicMock()
        runtime = MagicMock()
        load_profile.return_value = profile
        runtime_class.return_value = runtime
        cgroup_handling.return_value = CheckResult("runsc-cgroup-handling", True, "limits enforced")

        for prerequisites in (
            [CheckResult("docker-connection", False, "Docker unavailable")],
            [
                CheckResult("rootless-docker", False, "disabled"),
                CheckResult("gvisor-runsc", True, "registered"),
            ],
            [
                CheckResult("rootless-docker", True, "enabled"),
                CheckResult("gvisor-runsc", False, "missing"),
            ],
        ):
            with self.subTest(prerequisites=prerequisites):
                rootless.return_value = prerequisites

                self.assertEqual(run_doctor("gigachad"), prerequisites)

                image.assert_not_called()
                mounts.assert_not_called()
                omp.assert_not_called()
                runsc_startup.assert_not_called()
                python.assert_not_called()
                agents_md.assert_not_called()
                host_services.assert_not_called()
                runtime.run_probe.assert_not_called()
                cgroup_handling.assert_not_called()

        rootless.return_value = [
            CheckResult("rootless-docker", True, "enabled"),
            CheckResult("gvisor-runsc", True, "registered"),
        ]
        cgroup_handling.return_value = CheckResult("runsc-cgroup-handling", False, "stale configuration")

        expected = rootless.return_value + [cgroup_handling.return_value]
        self.assertEqual(run_doctor("gigachad"), expected)
        image.assert_not_called()
        mounts.assert_not_called()
        omp.assert_not_called()
        python.assert_not_called()
        agents_md.assert_not_called()
        host_services.assert_not_called()
        runtime.run_probe.assert_not_called()

        cgroup_handling.return_value = CheckResult("runsc-cgroup-handling", True, "limits enabled")
        image.return_value = CheckResult("image-available", True, "present")
        mounts.return_value = [CheckResult("mount-ok:/workspace", True, "ok")]
        runsc_startup.return_value = CheckResult(
            "runsc-rootless-cgroups",
            False,
            "upstream limitation",
        )
        expected = rootless.return_value + [
            cgroup_handling.return_value,
            image.return_value,
            *mounts.return_value,
            runsc_startup.return_value,
        ]
        self.assertEqual(run_doctor("gigachad"), expected)
        omp.assert_not_called()
        python.assert_not_called()
        agents_md.assert_not_called()
        host_services.assert_not_called()

    @patch("agent_sandbox.doctor.load_profile")
    def test_returns_profile_load_failure_as_result(self, load_profile: MagicMock) -> None:
        from agent_sandbox.profile import ProfileError

        load_profile.side_effect = ProfileError("profile missing")

        self.assertEqual(run_doctor("missing"), [CheckResult("profile-load", False, "profile missing")])

    @patch("agent_sandbox.doctor.SandboxRuntime")
    @patch("agent_sandbox.doctor.load_profile")
    def test_returns_runtime_connection_failure_as_result(
        self, load_profile: MagicMock, runtime_class: MagicMock
    ) -> None:
        from agent_sandbox.runtime import SecurityEnvironmentError

        load_profile.return_value = MagicMock()
        runtime_class.side_effect = SecurityEnvironmentError("Docker unavailable")

        self.assertEqual(run_doctor("gigachad"), [CheckResult("docker-connection", False, "Docker unavailable")])

    def test_runsc_startup_explains_upstream_rootless_cgroup_failure(self) -> None:
        import docker

        runtime = MagicMock()
        runtime.run_probe.side_effect = docker.errors.DockerException(
            "OCI runtime create failed: systemd error: Interactive authentication required."
        )

        result = _check_runsc_startup(runtime)

        runtime.run_probe.assert_called_once_with(["true"])
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "runsc-rootless-cgroups")
        self.assertIn("https://github.com/google/gvisor/issues/11543", result.detail)
        self.assertIn("'--ignore-cgroups' workaround is intentionally rejected", result.detail)

    def test_cgroup_check_rejects_stale_ignore_cgroups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "daemon.json"
            config_path.write_text(
                json.dumps({"runtimes": {"runsc": {"runtimeArgs": ["--ignore-cgroups"]}}}),
                encoding="utf-8",
            )
            with patch("agent_sandbox.doctor.ROOTLESS_DOCKER_DAEMON_CONFIG", str(config_path)):
                result = _check_runsc_cgroup_handling()

        self.assertFalse(result.passed)
        self.assertEqual(result.name, "runsc-cgroup-handling")
        self.assertIn("--ignore-cgroups", result.detail)
        self.assertIn("rerun scripts/setup_agent_sandbox.sh", result.detail)

    def test_cgroup_check_allows_clean_or_missing_runtime_args(self) -> None:
        cases = {
            "clean": {"runtimes": {"runsc": {"runtimeArgs": ["--debug"]}}},
            "missing": {"runtimes": {"runsc": {"path": "/home/user/.local/bin/runsc"}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "daemon.json"
            with patch("agent_sandbox.doctor.ROOTLESS_DOCKER_DAEMON_CONFIG", str(config_path)):
                for name, config in cases.items():
                    with self.subTest(name=name):
                        config_path.write_text(json.dumps(config), encoding="utf-8")
                        self.assertTrue(_check_runsc_cgroup_handling().passed)

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
