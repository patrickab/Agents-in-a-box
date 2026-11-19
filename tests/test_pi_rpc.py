"""Test the pi-rpc docker argv, its startup checks, and its stdio, signal, and cleanup contract."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_sandbox.agent_dir import AgentSeed
from agent_sandbox.errors import ContainerRuntimeError
from agent_sandbox.pi_rpc import PiRpcRequest, build_docker_argv, container_name, prepare_request, run_pi_rpc
from agent_sandbox.profile import Mount, Profile
from agent_sandbox.runtime import hardened_docker_run_flags

_FAKE_DOCKER = """\
#!{python}
# Stand in for the Docker CLI: `run` echoes stdin lines until EOF; SIGTERM exits 143; `rm` is logged.
import os, signal, sys
log = os.environ["FAKE_DOCKER_LOG"]
if sys.argv[1] == "rm":
    open(log, "a").write(" ".join(sys.argv[1:]) + "\\n")
    sys.exit(0)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
print("ready", flush=True)
for line in sys.stdin:
    print("echo " + line.strip(), flush=True)
sys.exit(int(os.environ.get("FAKE_DOCKER_EXIT", "0")))
"""

_SUPERVISOR = """\
import os
from pathlib import Path
import sys
from agent_sandbox.agent_dir import AgentSeed
from agent_sandbox.pi_rpc import supervise
sys.exit(supervise(["docker", "run"], "agent-sandbox-pi-test", os.environ, AgentSeed(Path(sys.argv[1]), None)))
"""


def _profile() -> Profile:
    return Profile(
        name="gigachad",
        image="agent-sandbox:trixie",
        workdir="/workspace",
        mounts=(
            Mount("/host/pi", "/opt/pi", "ro"),
            Mount("/host/venv", "/host/venv", "rw"),
            Mount("/host/AGENTS.md", "/workspace/AGENTS.md", "ro"),
        ),
        pi_binary="/opt/pi/pi",
        python_interpreters={"venv": "/host/venv/bin/python3"},
        env_passthrough=("PASSED", "UNSET_PASSED"),
        host_services={"PI_GATEWAY_URL": 8300, "OTHER": 9000},
    )


class TestDockerArgv(unittest.TestCase):
    def test_builds_a_hardened_interactive_run_of_pi_with_host_path_binds(self) -> None:
        request = PiRpcRequest(_profile(), "/work", "/sessions", 8301, ("--mode", "rpc", "--session-dir", "/sessions"))
        seed = AgentSeed(Path("/tmp/seed"), Path("/home/u/.pi/agent/npm"))

        with patch.dict(os.environ, {"PASSED": "1"}):
            argv = build_docker_argv(request, seed, "agent-sandbox-pi-x")

        self.assertEqual(
            argv,
            [
                "docker",
                "run",
                "-i",
                "--rm",
                "--name=agent-sandbox-pi-x",
                *hardened_docker_run_flags(network=True, runtime_tmpfs=True),
                "--mount=type=bind,source=/host/pi,target=/opt/pi,readonly",
                "--mount=type=bind,source=/host/venv,target=/host/venv",
                "--mount=type=bind,source=/work,target=/work",
                "--mount=type=bind,source=/sessions,target=/sessions",
                "--mount=type=bind,source=/tmp/seed,target=/runtime/pi-agent",
                "--mount=type=bind,source=/home/u/.pi/agent/npm,target=/runtime/pi-agent/npm,readonly",
                "--env=PASSED",
                "--env=PI_GATEWAY_TOKEN",
                "--env=OTHER=http://10.200.200.1:9000",
                "--env=PI_GATEWAY_URL=http://10.200.200.1:8301",
                "--env=HOME=/runtime",
                "--env=PATH=/host/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "--env=PI_CODING_AGENT_DIR=/runtime/pi-agent",
                "--env=PI_OFFLINE=1",
                "--workdir=/work",
                "agent-sandbox:trixie",
                "/opt/pi/pi",
                "--mode",
                "rpc",
                "--session-dir",
                "/sessions",
            ],
        )
        self.assertIn("--tmpfs=/runtime:rw,exec,nosuid,nodev,size=1g", argv)

    def test_binds_a_session_dir_inside_cwd_once_and_skips_absent_npm(self) -> None:
        argv = build_docker_argv(PiRpcRequest(_profile(), "/work", "/work", 8300, ()), AgentSeed(Path("/tmp/seed"), None), "n")

        self.assertEqual([flag for flag in argv if "target=/work" in flag], ["--mount=type=bind,source=/work,target=/work"])
        self.assertFalse(any("/npm" in flag for flag in argv))

    def test_container_names_are_unique(self) -> None:
        self.assertTrue(container_name().startswith("agent-sandbox-pi-"))
        self.assertNotEqual(container_name(), container_name())


@patch("agent_sandbox.pi_rpc.load_profile", return_value=_profile())
class TestStartupChecks(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name

    def test_rejects_missing_or_relative_directories_and_bad_ports(self, _load: object) -> None:
        for args, message in (
            ((f"{self.root}/missing", self.root, 8300), "--cwd does not exist"),
            ((self.root, "relative", 8300), "--session-dir must be an absolute path"),
            ((self.root, self.root, 0), "gateway port out of range"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepare_request("gigachad", *args, [])

    def test_requires_the_gateway_token_before_anything_else(self, _load: object) -> None:
        with patch.dict(os.environ, {"PI_GATEWAY_TOKEN": ""}), self.assertRaisesRegex(ContainerRuntimeError, "PI_GATEWAY_TOKEN"):
            run_pi_rpc("gigachad", self.root, self.root, 8300, [])

    @patch("agent_sandbox.pi_rpc.build_agent_seed")
    def test_reports_a_missing_docker_cli_without_building_a_seed(self, build_seed: object, _load: object) -> None:
        with (
            patch.dict(os.environ, {"PI_GATEWAY_TOKEN": "t"}),
            patch("agent_sandbox.pi_rpc.shutil.which", return_value=None),
            self.assertRaisesRegex(ContainerRuntimeError, "docker CLI not found"),
        ):
            run_pi_rpc("gigachad", self.root, self.root, 8300, [])
        build_seed.assert_not_called()


class TestSupervise(unittest.TestCase):
    """Run the supervisor in a child process against a fake docker, as pi-host would."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(_FAKE_DOCKER.format(python=sys.executable))
        docker.chmod(0o755)
        self.seed = root / "seed"
        self.seed.mkdir()
        (self.seed / "models.json").write_text("{}")
        self.log = root / "docker.log"
        self.environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_DOCKER_LOG": str(self.log)}

    def _start(self, **environment: str) -> subprocess.Popen[str]:
        child = subprocess.Popen(
            [sys.executable, "-c", _SUPERVISOR, str(self.seed)],
            env={**self.environment, **environment},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(child.kill)
        self.assertEqual(child.stdout.readline(), "ready\n")
        return child

    def test_stdin_eof_ends_the_child_and_removes_the_seed(self) -> None:
        child = self._start()
        child.stdin.write("hello\n")
        child.stdin.flush()
        self.assertEqual(child.stdout.readline(), "echo hello\n")

        child.stdin.close()

        self.assertEqual(child.wait(timeout=10), 0)
        self.assertFalse(self.seed.exists())
        self.assertFalse(self.log.exists())

    def test_sigterm_is_forwarded_and_the_container_force_removed(self) -> None:
        child = self._start()

        child.send_signal(signal.SIGTERM)

        self.assertEqual(child.wait(timeout=10), 143)
        self.assertFalse(self.seed.exists())
        self.assertEqual(self.log.read_text(), "rm -f agent-sandbox-pi-test\n")

    def test_abnormal_exit_keeps_docker_code_and_removes_the_container(self) -> None:
        child = self._start(FAKE_DOCKER_EXIT="125")

        child.stdin.close()

        self.assertEqual(child.wait(timeout=10), 125)
        self.assertFalse(self.seed.exists())
        self.assertEqual(self.log.read_text(), "rm -f agent-sandbox-pi-test\n")


if __name__ == "__main__":
    unittest.main()
