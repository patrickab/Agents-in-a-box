"""Test CLI presentation, invocation construction, and durable OMP output."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox import OmpTask, SandboxRun, WorkspaceArchive
from agent_sandbox.cli import main
from agent_sandbox.doctor import CheckResult
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef, manifest_to_dict
from agent_sandbox.omp import ErrorEvent, StreamEvent


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


def _run(capture: Path, on_release: MagicMock, status: str = "completed", outputs: tuple = ()) -> SandboxRun:
    capture.write_bytes(b"captured workspace")
    return SandboxRun(
        status=status,
        summary=f"[{status}] concise summary",
        next_manifest=_manifest(),
        outputs=outputs or (StreamEvent(name="stdout", text="result\n"),),
        workspace_changed=True,
        workspace=WorkspaceArchive(str(capture), "a" * 64),
        _on_release=on_release,
    )


def _write_tar(path: Path) -> str:
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("file.txt")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"ok"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestDoctorCommand(unittest.TestCase):
    @patch("agent_sandbox.cli.run_doctor")
    def test_exits_zero_and_reports_successful_checks(self, run_doctor: MagicMock) -> None:
        run_doctor.return_value = [CheckResult("docker-environment", True, "ready")]

        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["doctor", "--profile", "gigachad"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "[OK  ] docker-environment: ready\n\nAll checks passed.\n")

    @patch("agent_sandbox.cli.run_doctor")
    def test_exits_one_and_reports_failed_checks(self, run_doctor: MagicMock) -> None:
        run_doctor.return_value = [CheckResult("docker-environment", True, "ready"), CheckResult("omp-binary", False, "missing")]

        stdout, stderr = io.StringIO(), io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout), redirect_stderr(stderr):
            main(["doctor", "--profile", "gigachad"])

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "[OK  ] docker-environment: ready\n[FAIL] omp-binary: missing\n")
        self.assertEqual(stderr.getvalue(), "\n1 check(s) failed.\n")


@patch("agent_sandbox.cli.Sandbox")
class TestOmpCommand(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace_out = self.root / "out.tar"
        self.manifest_out = self.root / "next.json"
        self.release = MagicMock()

    def _main(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = [
            "omp",
            *args,
            "--profile",
            "gigachad",
            "--workspace-out",
            str(self.workspace_out),
            "--manifest-out",
            str(self.manifest_out),
        ]
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout), redirect_stderr(stderr):
            main(argv)
        return raised.exception.code, stdout.getvalue(), stderr.getvalue()

    def test_fresh_run_pins_workspace_persists_outputs_then_releases(self, sandbox: MagicMock) -> None:
        workspace_in = self.root / "initial.tar"
        workspace_sha = _write_tar(workspace_in)
        result = _run(self.root / "capture.tar", self.release)
        sandbox.return_value.run.return_value = result

        def release_after_persist() -> None:
            self.assertEqual(self.workspace_out.read_bytes(), b"captured workspace")
            self.assertEqual(json.loads(self.manifest_out.read_text()), manifest_to_dict(result.next_manifest))

        self.release.side_effect = release_after_persist

        code, stdout, _stderr = self._main(
            "do work",
            "--workspace-in",
            str(workspace_in),
            "--run-id",
            "fresh-run",
            "--model",
            "model-x",
            "--thinking",
            "high",
            "--append-system",
            "be concise",
            "--lean",
        )

        self.assertEqual(code, 0)
        invocation = sandbox.return_value.run.call_args.args[0]
        self.assertEqual((invocation.slot_key, invocation.run_id), ("cli", "fresh-run"))
        self.assertEqual(invocation.workspace, WorkspaceArchive(str(workspace_in), workspace_sha))
        self.assertIsNone(invocation.active_manifest)
        self.assertEqual(invocation.task, OmpTask("do work", model="model-x", thinking="high", append_system="be concise", lean=True))
        self.assertEqual(
            stdout,
            'status: completed\nsummary: [completed] concise summary\n[{"name":"stdout","text":"result\\n","type":"StreamEvent"}]\n',
        )
        self.release.assert_called_once_with()
        self.assertFalse(Path(result.workspace.path).exists())

    def test_resumed_run_passes_the_manifest_with_its_matching_workspace(self, sandbox: MagicMock) -> None:
        workspace_in = self.root / "prior.tar"
        prior = _manifest(run_id="prior-run", sha256=_write_tar(workspace_in))
        manifest_in = self.root / "prior.json"
        manifest_in.write_text(json.dumps(manifest_to_dict(prior)))
        sandbox.return_value.run.return_value = _run(self.root / "capture.tar", self.release)

        code, _stdout, _stderr = self._main("continue work", "--workspace-in", str(workspace_in), "--manifest-in", str(manifest_in))

        self.assertEqual(code, 0)
        invocation = sandbox.return_value.run.call_args.args[0]
        self.assertEqual(invocation.active_manifest, prior)
        self.assertEqual(invocation.workspace.path, str(workspace_in))

    def test_failed_run_exits_one_after_persisting_outputs(self, sandbox: MagicMock) -> None:
        outputs = (ErrorEvent(ename="OMPExecutionError", evalue="omp exited with code 1"),)
        sandbox.return_value.run.return_value = _run(self.root / "capture.tar", self.release, status="failed", outputs=outputs)

        code, _stdout, _stderr = self._main("do work")

        self.assertEqual(code, 1)
        self.assertEqual(self.workspace_out.read_bytes(), b"captured workspace")
        self.release.assert_called_once_with()

    def test_invalid_inputs_exit_two_without_running(self, sandbox: MagicMock) -> None:
        manifest_in = self.root / "prior.json"
        manifest_in.write_text(json.dumps(manifest_to_dict(_manifest())))
        unsafe_tar = self.root / "unsafe.tar"
        with tarfile.open(unsafe_tar, "w") as archive:
            archive.addfile(tarfile.TarInfo("../escape"))

        for args, message in (
            (("--manifest-in", str(manifest_in)), "active_manifest requires workspace"),
            (("--workspace-in", str(unsafe_tar)), "unsafe member path"),
            (("--workspace-in", str(self.root / "missing.tar")), "Refusing workspace archive"),
        ):
            with self.subTest(message=message):
                code, _stdout, stderr = self._main("do work", *args)
                self.assertEqual(code, 2)
                self.assertIn(message, stderr)
        sandbox.return_value.run.assert_not_called()

    def test_releases_run_when_workspace_write_fails(self, sandbox: MagicMock) -> None:
        sandbox.return_value.run.return_value = _run(self.root / "capture.tar", self.release)

        with patch("agent_sandbox.cli.shutil.copyfile", side_effect=OSError("write failed")):
            code, _stdout, stderr = self._main("do work")

        self.assertEqual(code, 2)
        self.assertIn("write failed", stderr)
        self.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
