"""Test CLI presentation, invocation construction, and durable OMP output."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox.cli import main
from agent_sandbox.doctor import CheckResult
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef, manifest_to_dict
from agent_sandbox.outputs import ErrorEvent, StreamEvent


def _manifest(run_id: str = "next-run", sha256: str = "a" * 64) -> SandboxManifest:
    return SandboxManifest(
        schema_version=1,
        manifest_id="next-manifest",
        profile="gigachad",
        runtime_fingerprint="sha256:" + "b" * 64,
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=None)},
        execution=Execution(run_id=run_id, status="completed", exit_code=0, created_at="2026-01-01T00:00:00+00:00"),
    )


def _run(capture_path: str, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        summary=f"[{status}] concise summary",
        next_manifest=_manifest(),
        outputs=(StreamEvent(name="stdout", text="result\n"),),
        capture_path=capture_path,
        release=MagicMock(),
    )


class TestDoctorCommand(unittest.TestCase):
    """Exercise the doctor subcommand without Docker."""

    @patch("agent_sandbox.cli.run_doctor")
    def test_exits_zero_and_reports_successful_checks(self, run_doctor) -> None:
        run_doctor.return_value = [CheckResult("rootless-docker", True, "rootless mode enabled")]

        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["doctor", "--profile", "gigachad"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "[OK  ] rootless-docker: rootless mode enabled\n\nAll checks passed.\n")

    @patch("agent_sandbox.cli.run_doctor")
    def test_exits_one_and_reports_failed_checks(self, run_doctor) -> None:
        run_doctor.return_value = [
            CheckResult("rootless-docker", True, "rootless mode enabled"),
            CheckResult("gvisor-runsc", False, "runsc missing"),
        ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout), redirect_stderr(stderr):
            main(["doctor", "--profile", "gigachad"])

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(
            stdout.getvalue(),
            "[OK  ] rootless-docker: rootless mode enabled\n[FAIL] gvisor-runsc: runsc missing\n",
        )
        self.assertEqual(stderr.getvalue(), "\n1 check(s) failed.\n")


class TestOmpCommand(unittest.TestCase):
    """Exercise OMP orchestration without starting Docker."""

    @patch("agent_sandbox.cli.Sandbox")
    def test_fresh_run_hashes_workspace_and_persists_outputs(self, sandbox: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace_in = root / "initial.tar"
            workspace_in.write_bytes(b"initial workspace")
            capture = root / "capture.tar"
            capture.write_bytes(b"captured workspace")
            workspace_out = root / "out.tar"
            manifest_out = root / "manifest.json"
            result = _run(str(capture))
            sandbox.return_value.run.return_value = result

            def release_after_persist() -> None:
                self.assertEqual(workspace_out.read_bytes(), b"captured workspace")
                self.assertEqual(json.loads(manifest_out.read_text()), manifest_to_dict(result.next_manifest))

            result.release.side_effect = release_after_persist

            stdout = io.StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                main(
                    [
                        "omp",
                        "do work",
                        "--profile",
                        "gigachad",
                        "--workspace-in",
                        str(workspace_in),
                        "--workspace-out",
                        str(workspace_out),
                        "--manifest-out",
                        str(manifest_out),
                        "--run-id",
                        "fresh-run",
                        "--model",
                        "model-x",
                        "--thinking",
                        "high",
                        "--append-system",
                        "be concise",
                        "--lean",
                    ]
                )

            self.assertEqual(raised.exception.code, 0)
            invocation = sandbox.return_value.run.call_args.args[0]
            self.assertEqual(invocation.slot_key, "cli")
            self.assertEqual(invocation.workspace_archive_path, str(workspace_in))
            self.assertEqual(invocation.workspace_archive_sha256, hashlib.sha256(b"initial workspace").hexdigest())
            self.assertIsNone(invocation.active_manifest)
            self.assertEqual(
                (invocation.run_id, invocation.model, invocation.thinking, invocation.append_system, invocation.lean),
                ("fresh-run", "model-x", "high", "be concise", True),
            )
            self.assertEqual(workspace_out.read_bytes(), b"captured workspace")
            self.assertEqual(json.loads(manifest_out.read_text()), manifest_to_dict(result.next_manifest))
            self.assertEqual(
                stdout.getvalue(),
                'status: completed\nsummary: [completed] concise summary\n'
                '[{"name":"stdout","text":"result\\n","type":"StreamEvent"}]\n',
            )
            result.release.assert_called_once_with()

    @patch("agent_sandbox.cli.Sandbox")
    def test_resumed_run_passes_strict_manifest_without_rehashing(self, sandbox: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace_in = root / "prior.tar"
            workspace_in.write_bytes(b"prior workspace")
            manifest_in = root / "prior.json"
            prior = _manifest(run_id="prior-run", sha256="c" * 64)
            manifest_in.write_text(json.dumps(manifest_to_dict(prior)))
            capture = root / "capture.tar"
            capture.write_bytes(b"new workspace")
            result = _run(str(capture))
            sandbox.return_value.run.return_value = result

            with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
                main(
                    [
                        "omp",
                        "continue work",
                        "--profile",
                        "gigachad",
                        "--workspace-in",
                        str(workspace_in),
                        "--manifest-in",
                        str(manifest_in),
                        "--workspace-out",
                        str(root / "out.tar"),
                        "--manifest-out",
                        str(root / "next.json"),
                    ]
                )

            self.assertEqual(raised.exception.code, 0)
            invocation = sandbox.return_value.run.call_args.args[0]
            self.assertEqual(invocation.active_manifest, prior)
            self.assertEqual(invocation.workspace_archive_path, str(workspace_in))
            self.assertIsNone(invocation.workspace_archive_sha256)
            result.release.assert_called_once_with()

    @patch("agent_sandbox.cli.Sandbox")
    def test_failed_run_exits_one_after_persisting_outputs(self, sandbox: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.tar"
            capture.write_bytes(b"failed workspace")
            result = _run(str(capture), status="failed")
            result.outputs = (ErrorEvent(ename="OMPExecutionError", evalue="omp exited with code 1"),)
            sandbox.return_value.run.return_value = result
            workspace_out = root / "out.tar"
            manifest_out = root / "next.json"

            with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
                main(
                    [
                        "omp",
                        "do work",
                        "--profile",
                        "gigachad",
                        "--workspace-out",
                        str(workspace_out),
                        "--manifest-out",
                        str(manifest_out),
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(workspace_out.read_bytes(), b"failed workspace")
            self.assertEqual(json.loads(manifest_out.read_text()), manifest_to_dict(result.next_manifest))
            result.release.assert_called_once_with()

    @patch("agent_sandbox.cli.Sandbox")
    def test_manifest_input_requires_workspace_input(self, sandbox: MagicMock) -> None:
        with self.assertRaises(SystemExit) as raised, redirect_stderr(io.StringIO()):
            main(
                [
                    "omp",
                    "do work",
                    "--profile",
                    "gigachad",
                    "--manifest-in",
                    "prior.json",
                    "--workspace-out",
                    "out.tar",
                    "--manifest-out",
                    "next.json",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        sandbox.assert_not_called()

    @patch("agent_sandbox.cli.shutil.copyfile", side_effect=OSError("write failed"))
    @patch("agent_sandbox.cli.Sandbox")
    def test_releases_run_when_workspace_write_fails(self, sandbox: MagicMock, copyfile: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _run(str(root / "capture.tar"))
            sandbox.return_value.run.return_value = result

            with self.assertRaisesRegex(OSError, "write failed"):
                main(
                    [
                        "omp",
                        "do work",
                        "--profile",
                        "gigachad",
                        "--workspace-out",
                        str(root / "out.tar"),
                        "--manifest-out",
                        str(root / "next.json"),
                    ]
                )

            copyfile.assert_called_once_with(str(root / "capture.tar"), str(root / "out.tar"))
            result.release.assert_called_once_with()
