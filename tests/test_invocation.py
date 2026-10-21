"""Test invocation validation, idempotency, release, and result construction."""

import io
import os
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox import ArchiveError, OmpTask, PromptImage, Sandbox, SandboxInvocation, SandboxRun, WorkspaceArchive
from agent_sandbox.invocation import _bounded_summary
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef
from agent_sandbox.omp import ErrorEvent, StreamEvent
from agent_sandbox.profile import Profile
from agent_sandbox.runtime import ExecResult, ManagedExecutionResult, WorkspaceRestore

_TASK = OmpTask("task")


def _execution(
    exit_code: int = 0,
    stdout: str = "ok",
    stderr: str = "",
    archive_path: str = "/tmp/archive.tar",
    sha256: str = "a" * 64,
    runtime_fingerprint: str = "sha256:effective",
) -> ManagedExecutionResult:
    return ManagedExecutionResult(
        ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr), WorkspaceArchive(archive_path, sha256), runtime_fingerprint
    )


def _manifest(sha256: str = "b" * 64) -> SandboxManifest:
    return SandboxManifest(
        schema_version=1,
        manifest_id="prior-manifest",
        profile="test-profile",
        runtime_fingerprint="sha256:" + "a" * 64,
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=None)},
        execution=Execution(run_id="prior-run", status="completed", exit_code=0, created_at=""),
    )


def _invocation(run_id: str = "run", slot_key: str = "slot", **overrides: object) -> SandboxInvocation:
    fields: dict[str, object] = {"slot_key": slot_key, "profile_name": "test-profile", "run_id": run_id, "task": _TASK}
    return SandboxInvocation(**(fields | overrides))


def _temp_capture() -> str:
    descriptor, path = tempfile.mkstemp(suffix=".tar")
    os.close(descriptor)
    return path


@patch("agent_sandbox.invocation.load_profile", return_value=Profile("test-profile", "test-image", "/workspace", (), "/usr/bin/omp"))
@patch("agent_sandbox.invocation.SandboxRuntime")
class TestSandboxRun(unittest.TestCase):
    def _runtime(self, runtime_class: MagicMock, **execution: object) -> MagicMock:
        runtime = runtime_class.return_value
        runtime.execute.return_value = _execution(**execution)
        return runtime

    def test_namespace_isolation_uses_distinct_runtime_resources(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        self._runtime(runtime_class)

        Sandbox().run(_invocation("default-run"))
        Sandbox(namespace="tenant-a").run(_invocation("isolated-run"))

        default_resources = runtime_class.call_args_list[0].args[1]
        isolated_resources = runtime_class.call_args_list[1].args[1]
        self.assertEqual(default_resources.container, "agent-sandbox-runtime")
        self.assertEqual(isolated_resources.container, "agent-sandbox-tenant-a-runtime")

    def test_restore_carries_the_archive_and_resumes_only_with_a_manifest(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        runtime = self._runtime(runtime_class)
        prior = _manifest()
        archive = WorkspaceArchive("/tmp/prior.tar", prior.workspace.sha256)

        Sandbox().run(_invocation("resume", active_manifest=prior, workspace=archive))
        Sandbox().run(_invocation("initial", workspace=WorkspaceArchive("/tmp/initial.tar", "c" * 64)))
        Sandbox().run(_invocation("fresh"))

        restores = [call.args[1] for call in runtime.execute.call_args_list]
        self.assertEqual(restores[0], WorkspaceRestore(archive, prior.runtime_fingerprint))
        self.assertTrue(restores[0].resuming)
        self.assertEqual(restores[1], WorkspaceRestore(WorkspaceArchive("/tmp/initial.tar", "c" * 64), None))
        self.assertFalse(restores[1].resuming)
        self.assertIsNone(restores[2])

    def test_next_manifest_records_effective_fingerprint_status_and_capture_digest(
        self, runtime_class: MagicMock, _load: MagicMock
    ) -> None:
        self._runtime(runtime_class, exit_code=1, sha256="deadbeef" * 8, runtime_fingerprint="sha256:resolved-image")

        result = Sandbox().run(_invocation("run-id-1"))

        manifest = result.next_manifest
        self.assertEqual(result.status, "failed")
        self.assertEqual(manifest.manifest_id, "run-id-1-deadbeefdeadbeef")
        self.assertEqual(manifest.runtime_fingerprint, "sha256:resolved-image")
        self.assertEqual(manifest.workspace, WorkspaceRef(snapshot_asset_id=None, sha256="deadbeef" * 8))
        self.assertEqual((manifest.execution.status, manifest.execution.exit_code), ("failed", 1))
        self.assertEqual(dict(manifest.omp_sessions), {"main": OmpSessionState(session_id="main", state_asset_id=None)})

    def test_workspace_changed_compares_against_the_restored_archive(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        self._runtime(runtime_class, sha256="c" * 64)
        same = WorkspaceArchive("/tmp/in.tar", "c" * 64)

        self.assertTrue(Sandbox().run(_invocation("fresh")).workspace_changed)
        self.assertFalse(Sandbox().run(_invocation("same", workspace=same)).workspace_changed)

    def test_output_events_preserve_stdout_stderr_then_error_order(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        self._runtime(runtime_class, exit_code=7, stdout="normal output", stderr="warning output")

        result = Sandbox().run(_invocation("ordered-events"))

        self.assertEqual(
            result.outputs,
            (
                StreamEvent(name="stdout", text="normal output"),
                StreamEvent(name="stderr", text="warning output"),
                ErrorEvent(ename="OMPExecutionError", evalue="omp exited with code 7"),
            ),
        )

    def test_unreleased_same_run_is_cached_per_slot(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        runtime = self._runtime(runtime_class)
        sandbox = Sandbox()

        first = sandbox.run(_invocation("call-0", slot_key="chat-a"))
        self.assertIs(first, sandbox.run(_invocation("call-0", slot_key="chat-a")))
        self.assertIsNot(first, sandbox.run(_invocation("call-0", slot_key="chat-b")))
        self.assertEqual(runtime.execute.call_count, 2)

    def test_release_deletes_capture_and_evicts_only_within_the_owning_sandbox(
        self, runtime_class: MagicMock, _load: MagicMock
    ) -> None:
        capture_path = _temp_capture()
        runtime = self._runtime(runtime_class, archive_path=capture_path)
        first_sandbox, second_sandbox = Sandbox(), Sandbox()

        with first_sandbox.run(_invocation("same-run")) as first:
            second = second_sandbox.run(_invocation("same-run"))
            self.assertTrue(os.path.exists(capture_path))
        self.assertFalse(os.path.exists(capture_path))

        self.assertIsNot(first, first_sandbox.run(_invocation("same-run")))
        self.assertIs(second, second_sandbox.run(_invocation("same-run")))
        self.assertEqual(runtime.execute.call_count, 3)

    def test_repeated_release_preserves_a_later_retry(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        runtime = self._runtime(runtime_class)
        sandbox = Sandbox()

        released = sandbox.run(_invocation("release-twice"))
        released.release()
        retry = sandbox.run(_invocation("release-twice"))
        released.release()

        self.assertIs(retry, sandbox.run(_invocation("release-twice")))
        self.assertEqual(runtime.execute.call_count, 2)

    def test_release_evicts_even_when_unlink_fails(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        runtime = self._runtime(runtime_class)
        sandbox = Sandbox()
        run = sandbox.run(_invocation("unlink-failure"))

        with (
            patch("agent_sandbox.invocation.os.unlink", side_effect=OSError("unlink failed")),
            self.assertRaisesRegex(OSError, "unlink failed"),
        ):
            run.release()

        self.assertIsNot(run, sandbox.run(_invocation("unlink-failure")))
        self.assertEqual(runtime.execute.call_count, 2)

    def test_result_construction_failure_removes_the_capture(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        capture_path = _temp_capture()
        self._runtime(runtime_class, archive_path=capture_path)
        try:
            with (
                patch("agent_sandbox.invocation._build_manifest", side_effect=RuntimeError("manifest failure")),
                self.assertRaisesRegex(RuntimeError, "manifest failure"),
            ):
                Sandbox().run(_invocation("capture-cleanup"))
            self.assertFalse(os.path.exists(capture_path))
        finally:
            if os.path.exists(capture_path):
                os.unlink(capture_path)

    def test_distinct_slots_do_not_share_the_logical_slot_lock(self, runtime_class: MagicMock, _load: MagicMock) -> None:
        first_entered, second_entered, release_first = threading.Event(), threading.Event(), threading.Event()

        def execute_transaction(task: OmpTask, _restore: object) -> ManagedExecutionResult:
            if task.prompt == "first":
                first_entered.set()
                if not release_first.wait(timeout=1):
                    raise AssertionError("first invocation was not released")
            else:
                second_entered.set()
            return _execution()

        runtime_class.return_value.execute.side_effect = execute_transaction
        sandbox = Sandbox()
        errors: list[BaseException] = []

        def execute(invocation: SandboxInvocation) -> None:
            try:
                sandbox.run(invocation)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=execute, args=(_invocation("first-run", "slot-one", task=OmpTask("first")),))
        second = threading.Thread(target=execute, args=(_invocation("second-run", "slot-two", task=OmpTask("second")),))
        first.start()
        try:
            self.assertTrue(first_entered.wait(timeout=1))
            second.start()
            self.assertTrue(second_entered.wait(timeout=1))
        finally:
            release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertEqual(errors, [])


class TestInvocationValidation(unittest.TestCase):
    def test_invalid_invocation_inputs_are_rejected(self) -> None:
        for overrides in (
            {"slot_key": ""},
            {"run_id": ""},
            {"profile_name": "../profile"},
            {"task": "task"},
            {"workspace": "/tmp/archive.tar"},
            {"active_manifest": _manifest()},
            {"active_manifest": _manifest(), "workspace": WorkspaceArchive("/tmp/prior.tar", "c" * 64)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                _invocation(**overrides)

    def test_task_validates_prompt_options_and_inputs(self) -> None:
        for kwargs in (
            {"prompt": ""},
            {"prompt": "task", "model": object()},
            {"prompt": "task", "thinking": object()},
            {"prompt": "task", "append_system": object()},
            {"prompt": "task", "append_system": "x" * (64 * 1024 + 1)},
            {"prompt": "task", "prompt_images": [PromptImage("image.png", b"image")]},
            {"prompt": "task", "prompt_images": (("image.png", b"image"),)},
            {"prompt": "task", "prompt_images": (PromptImage("a.png", b"1"), PromptImage("a.png", b"2"))},
            {"prompt": "task", "timeout_s": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                OmpTask(**kwargs)

    def test_prompt_images_validate_safety_bytes_and_combined_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe basename"):
            PromptImage("../photo.png", b"image-bytes")
        with self.assertRaisesRegex(TypeError, "content must be bytes"):
            PromptImage("photo.png", "image-bytes")  # type: ignore[arg-type]
        with patch("agent_sandbox.omp.MAX_PROMPT_IMAGE_BYTES", 3), self.assertRaisesRegex(ValueError, "must not exceed"):
            OmpTask("inspect this", prompt_images=(PromptImage("one.png", b"ab"), PromptImage("two.png", b"cd")))

    def test_summary_is_bounded_single_line_and_prefers_stdout(self) -> None:
        summary = _bounded_summary(ExecResult(exit_code=0, stdout="one\ntwo " + "x" * 500, stderr="ignored"), "completed", limit=20)

        self.assertEqual(summary, "[completed] one two " + "x" * 11 + "…")
        self.assertEqual(_bounded_summary(ExecResult(exit_code=1, stdout="", stderr="\nstderr\n"), "failed"), "[failed] stderr")


class TestSandboxRunWorkspaceRead(unittest.TestCase):
    def test_reads_from_the_capture_until_release(self) -> None:
        path = _temp_capture()
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo("./result.txt")
            member.size = 4
            archive.addfile(member, io.BytesIO(b"root"))
        run = SandboxRun("completed", "", _manifest(), (), False, WorkspaceArchive(path, "a" * 64))

        self.assertEqual(run.read_workspace_file("result.txt"), b"root")
        run.release()
        with self.assertRaisesRegex(ArchiveError, "Refusing workspace archive"):
            run.read_workspace_file("result.txt")


if __name__ == "__main__":
    unittest.main()
