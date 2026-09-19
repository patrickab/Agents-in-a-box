"""Test invocation idempotency and failure handling."""

import io
import os
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox.invocation import PromptImage, Sandbox, SandboxInvocation, SandboxRun, _bounded_summary, _InProcessRunStore, run
from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef
from agent_sandbox.outputs import ErrorEvent, StreamEvent
from agent_sandbox.runtime import AgentSandboxError, ExecResult, ManagedExecutionResult


def _execution(
    exit_code: int = 0,
    stdout: str = "ok",
    stderr: str = "",
    archive_path: str = "/tmp/archive.tar",
    sha256: str = "a" * 64,
    runtime_fingerprint: str = "",
) -> ManagedExecutionResult:
    return ManagedExecutionResult(
        ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr), archive_path, sha256, runtime_fingerprint
    )


def _profile() -> MagicMock:
    profile = MagicMock()
    profile.name = "test-profile"
    profile.image = "test-image"
    profile.runtime_fingerprint = "test-image"
    profile.omp_binary = "/usr/local/bin/omp"
    return profile


def _manifest(profile: str = "test-profile", sha256: str = "b" * 64) -> SandboxManifest:
    return SandboxManifest(
        schema_version=1,
        manifest_id="prior-manifest",
        profile=profile,
        runtime_fingerprint="sha256:" + "a" * 64,
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=None)},
        execution=Execution(run_id="prior-run", status="completed", exit_code=0, created_at=""),
    )


def _write_workspace_capture(*entries: tuple[tarfile.TarInfo, bytes]) -> str:
    descriptor, path = tempfile.mkstemp(suffix=".tar")
    os.close(descriptor)
    with tarfile.open(path, "w") as archive:
        for member, content in entries:
            if member.isfile():
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                archive.addfile(member)
    return path


def _run_with_capture(path: str) -> SandboxRun:
    return SandboxRun("completed", "", _manifest(), (), False, path)


class TestInvocationIdempotency(unittest.TestCase):
    """Exercise invocation idempotency."""

    def setUp(self) -> None:
        # Isolate each test from cached default-sandbox results.
        import agent_sandbox.invocation as inv

        inv._default_sandbox = Sandbox()

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_namespace_isolation_uses_distinct_runtime_resources(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_load_profile.return_value = _profile()
        mock_runtime_class.return_value.execute.return_value = _execution()

        default = Sandbox()
        isolated = Sandbox(namespace="tenant-a")
        default.run(SandboxInvocation("default", "test-profile", "task", "default-run"))
        isolated.run(SandboxInvocation("isolated", "test-profile", "task", "isolated-run"))

        default_resources = mock_runtime_class.call_args_list[0].kwargs["resource_names"]
        isolated_resources = mock_runtime_class.call_args_list[1].kwargs["resource_names"]
        self.assertEqual(
            (default_resources.container, default_resources.lock_path),
            ("agent-sandbox-runtime", "/tmp/agent-sandbox-runtime.lock"),
        )
        self.assertNotEqual(default_resources, isolated_resources)
        self.assertEqual(isolated_resources.container, "agent-sandbox-tenant-a-runtime")

    def test_namespace_rejects_unsafe_resource_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "namespace"):
            Sandbox(namespace="Tenant_A")

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_prior_runtime_fingerprint_reaches_managed_request(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_load_profile.return_value = _profile()
        mock_runtime_class.return_value.execute.return_value = _execution()
        prior = _manifest()
        Sandbox().run(
            SandboxInvocation("slot", "test-profile", "task", "run", active_manifest=prior, workspace_archive_path="/tmp/prior.tar")
        )

        request = mock_runtime_class.return_value.execute.call_args.args[0]
        self.assertEqual(request.prior_runtime_fingerprint, prior.runtime_fingerprint)
        self.assertEqual(request.expected_workspace_sha256, prior.workspace.sha256)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_initial_archive_digest_is_forwarded_to_the_runtime_transaction(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_load_profile.return_value = _profile()
        mock_runtime_class.return_value.execute.return_value = _execution()

        run(
            SandboxInvocation(
                "slot",
                "test-profile",
                "task",
                "initial-archive",
                workspace_archive_path="/tmp/workspace.tar",
                workspace_archive_sha256="c" * 64,
            ),
            _InProcessRunStore(),
        )

        request = mock_runtime_class.return_value.execute.call_args.args[0]
        self.assertEqual(request.workspace_archive_path, "/tmp/workspace.tar")
        self.assertEqual(request.expected_workspace_sha256, "c" * 64)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_effective_runtime_fingerprint_is_written_to_the_next_manifest(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_load_profile.return_value = _profile()
        mock_runtime_class.return_value.execute.return_value = _execution(runtime_fingerprint="sha256:resolved-image")

        result = run(SandboxInvocation("slot", "test-profile", "task", "effective-fingerprint"), _InProcessRunStore())

        self.assertEqual(result.next_manifest.runtime_fingerprint, "sha256:resolved-image")

    def test_prior_manifest_requires_a_caller_supplied_archive(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires workspace_archive_path"):
            SandboxInvocation("slot", "test-profile", "task", "prior-without-archive", active_manifest=_manifest())

    def test_restore_metadata_rejects_conflicting_digest(self) -> None:
        prior = _manifest()

        with self.assertRaisesRegex(ValueError, "conflicts"):
            SandboxInvocation(
                "slot",
                "test-profile",
                "task",
                "conflicting-digest",
                active_manifest=prior,
                workspace_archive_path="/tmp/prior.tar",
                workspace_archive_sha256="c" * 64,
            )

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_invalid_invocation_inputs_are_rejected_before_runtime_construction(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        invalid_kwargs = (
            {"slot_key": ""},
            {"run_id": ""},
            {"profile_name": "../profile"},
            {"prompt": ""},
            {"model": object()},
            {"thinking": object()},
            {"prompt_images": (("../image.png", b"image"),)},
            {"prompt_images": (("image.png", "image"),)},
            {"prompt_images": [("image.png", b"image")]},
            {"workspace_archive_path": "/tmp/archive.tar"},
        )
        valid = {"slot_key": "slot", "profile_name": "test-profile", "prompt": "task", "run_id": "invalid-input"}

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                SandboxInvocation(**(valid | kwargs))

        mock_load_profile.assert_not_called()
        mock_runtime_class.assert_not_called()

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_result_construction_failure_removes_the_capture(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_load_profile.return_value = _profile()
        descriptor, capture_path = tempfile.mkstemp(suffix=".tar")
        os.close(descriptor)
        mock_runtime_class.return_value.execute.return_value = _execution(archive_path=capture_path)

        try:
            with (
                patch("agent_sandbox.invocation._build_manifest", side_effect=RuntimeError("manifest failure")),
                self.assertRaisesRegex(RuntimeError, "manifest failure"),
            ):
                run(SandboxInvocation("slot", "test-profile", "task", "capture-cleanup"), _InProcessRunStore())
            self.assertFalse(os.path.exists(capture_path))
        finally:
            if os.path.exists(capture_path):
                os.unlink(capture_path)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_unreleased_same_run_reuses_active_capture(self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock) -> None:
        """Reuse the result while its local capture remains active."""
        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_profile.omp_binary = "/usr/local/bin/omp"
        mock_load_profile.return_value = mock_profile

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        mock_runtime.execute.return_value = _execution(stdout="success output")

        invocation = SandboxInvocation(
            slot_key="user:chat",
            profile_name="test-profile",
            prompt="test prompt",
            run_id="same-run-id",
            active_manifest=None,
            workspace_archive_path=None,
        )
        sandbox = Sandbox()

        result1 = sandbox.run(invocation)
        result2 = sandbox.run(invocation)

        self.assertIs(result1, result2)
        self.assertEqual(result1.next_manifest.runtime_fingerprint, "test-image")
        mock_runtime.execute.assert_called_once()
        self.assertEqual(mock_runtime.execute.call_args.args[0].prompt_images, ())

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_release_is_scoped_to_the_owning_sandbox(self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = mock_runtime_class.return_value
        mock_runtime.execute.return_value = _execution()
        invocation = SandboxInvocation("user:chat", "test-profile", "task", "same-run")
        first_sandbox = Sandbox()
        second_sandbox = Sandbox()

        first = first_sandbox.run(invocation)
        second = second_sandbox.run(invocation)
        first.release()

        self.assertIs(second, second_sandbox.run(invocation))
        self.assertIsNot(first, first_sandbox.run(invocation))
        self.assertEqual(mock_runtime.execute.call_count, 3)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_failed_exit_code_produces_failed_status_and_error_event(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        """Emit a failed result and error event for a failed process."""
        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_profile.omp_binary = "/usr/local/bin/omp"
        mock_load_profile.return_value = mock_profile

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        mock_runtime.execute.return_value = _execution(exit_code=1, stderr="something went wrong")

        invocation = SandboxInvocation(
            slot_key="user:chat",
            profile_name="test-profile",
            prompt="test prompt",
            run_id="fresh-run-id",
            active_manifest=None,
            workspace_archive_path=None,
        )

        result = run(invocation)

        self.assertEqual(result.status, "failed")
        self.assertIsInstance(result.outputs, tuple)
        self.assertGreater(len(result.outputs), 0)
        error_events = [event for event in result.outputs if isinstance(event, ErrorEvent)]
        self.assertGreater(len(error_events), 0, "Expected at least one ErrorEvent in outputs")
        error_event = error_events[0]
        self.assertEqual(error_event.ename, "OMPExecutionError")

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_different_run_ids_produce_different_manifest_ids_with_sha256(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        """Include distinct run IDs in distinct manifest IDs."""
        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_profile.omp_binary = "/usr/local/bin/omp"
        mock_load_profile.return_value = mock_profile

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        mock_runtime.execute.return_value = _execution(stdout="output", sha256="deadbeef" * 8)

        invocation1 = SandboxInvocation(
            slot_key="user:chat",
            profile_name="test-profile",
            prompt="prompt 1",
            run_id="run-id-1",
            active_manifest=None,
            workspace_archive_path=None,
        )
        invocation2 = SandboxInvocation(
            slot_key="user:chat",
            profile_name="test-profile",
            prompt="prompt 2",
            run_id="run-id-2",
            active_manifest=None,
            workspace_archive_path=None,
        )
        result1 = run(invocation1)
        result2 = run(invocation2)

        self.assertNotEqual(result1.next_manifest.manifest_id, result2.next_manifest.manifest_id)
        self.assertIn("deadbeefdeadbeef", result1.next_manifest.manifest_id)
        self.assertIn("deadbeefdeadbeef", result2.next_manifest.manifest_id)
        self.assertTrue(result1.next_manifest.manifest_id.startswith("run-id-1-deadbeefdeadbeef"))
        self.assertTrue(result2.next_manifest.manifest_id.startswith("run-id-2-deadbeefdeadbeef"))

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_same_run_id_in_different_slots_does_not_reuse_a_result(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_profile.omp_binary = "/usr/local/bin/omp"
        mock_load_profile.return_value = mock_profile
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime
        mock_runtime.execute.return_value = _execution()

        store = _InProcessRunStore()
        first = run(SandboxInvocation("user:chat-a", "test-profile", "first", "call-0"), store)
        second = run(SandboxInvocation("user:chat-b", "test-profile", "second", "call-0"), store)

        self.assertIsNot(first, second)
        self.assertEqual(mock_runtime.execute.call_count, 2)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_release_removes_the_transient_capture_and_evicts_it(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        import os
        import tempfile

        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_profile.omp_binary = "/usr/local/bin/omp"
        mock_load_profile.return_value = mock_profile
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime
        descriptor, capture_path = tempfile.mkstemp(suffix=".tar")
        os.close(descriptor)
        mock_runtime.execute.return_value = _execution(archive_path=capture_path)

        sandbox = Sandbox()
        invocation = SandboxInvocation("user:chat", "test-profile", "task", "call-release")
        with sandbox.run(invocation) as result:
            self.assertIsNone(result.next_manifest.workspace.snapshot_asset_id)
            self.assertTrue(os.path.exists(capture_path))
        self.assertFalse(os.path.exists(capture_path))

        retry = sandbox.run(invocation)

        self.assertIsNot(result, retry)
        self.assertEqual(mock_runtime.execute.call_count, 2)

    def test_release_evicts_cached_run_when_unlink_fails(self) -> None:
        store = _InProcessRunStore()
        run = SandboxRun("completed", "", _manifest(), (), False, "/tmp/capture.tar", store, "user:chat", "unlink-failure")
        store.put("user:chat", "unlink-failure", run)

        with (
            patch("agent_sandbox.invocation.os.unlink", side_effect=OSError("unlink failed")),
            self.assertRaisesRegex(OSError, "unlink failed"),
        ):
            run.release()

        self.assertIsNone(store.get("user:chat", "unlink-failure"))

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_repeated_release_preserves_a_later_retry(self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = mock_runtime_class.return_value
        mock_runtime.execute.return_value = _execution()
        sandbox = Sandbox()
        invocation = SandboxInvocation("user:chat", "test-profile", "task", "release-twice")

        released = sandbox.run(invocation)
        released.release()
        retry = sandbox.run(invocation)
        released.release()

        self.assertIs(retry, sandbox.run(invocation))
        self.assertEqual(mock_runtime.execute.call_count, 2)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_omp_sessions_populated_in_manifest(self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock) -> None:
        """Record the main OMP session in successful manifests."""
        mock_profile = MagicMock()
        mock_profile.name = "test-profile"
        mock_profile.image = "test-image"
        mock_profile.runtime_fingerprint = "test-image"
        mock_load_profile.return_value = mock_profile

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        mock_runtime.execute.return_value = _execution(stdout="success")

        invocation = SandboxInvocation(
            slot_key="user:chat",
            profile_name="test-profile",
            prompt="test prompt",
            run_id="omp-session-test",
            active_manifest=None,
            workspace_archive_path=None,
        )

        result = run(invocation)

        # Keep transient host paths out of durable manifest state.
        main_session = result.next_manifest.omp_sessions["main"]
        self.assertEqual(main_session.session_id, "main")
        self.assertIsNone(main_session.state_asset_id)
        self.assertEqual(result.capture_path, "/tmp/archive.tar")

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_prompt_images_are_forwarded_to_the_runtime_transaction(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = MagicMock()
        mock_runtime.execute.return_value = _execution()
        mock_runtime_class.return_value = mock_runtime

        legacy_prompt_images = (("photo.png", b"image-bytes"),)
        invocation = SandboxInvocation("user:chat", "test-profile", "inspect this", "images", prompt_images=legacy_prompt_images)
        run(invocation, _InProcessRunStore())

        request = mock_runtime.execute.call_args.args[0]
        self.assertEqual(request.prompt_images, (PromptImage("photo.png", b"image-bytes"),))
        self.assertEqual(request.prompt, "inspect this")

    def test_typed_prompt_images_are_preserved_as_immutable_values(self) -> None:
        image = PromptImage("photo.png", b"image-bytes")
        invocation = SandboxInvocation("user:chat", "test-profile", "inspect this", "typed-images", prompt_images=(image,))

        self.assertEqual(invocation.prompt_images, (image,))
        with self.assertRaisesRegex(AttributeError, "cannot assign to field"):
            image.filename = "replacement.png"  # type: ignore[misc]

    def test_prompt_images_validate_safety_bytes_and_combined_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe basename"):
            PromptImage("../photo.png", b"image-bytes")
        with self.assertRaisesRegex(TypeError, "content must be bytes"):
            PromptImage("photo.png", "image-bytes")  # type: ignore[arg-type]
        with patch("agent_sandbox.runtime.MAX_PROMPT_IMAGE_BYTES", 3), self.assertRaisesRegex(ValueError, "must not exceed"):
            SandboxInvocation(
                "user:chat",
                "test-profile",
                "inspect this",
                "oversized-images",
                prompt_images=(PromptImage("one.png", b"ab"), PromptImage("two.png", b"cd")),
            )

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_append_system_is_forwarded_to_the_runtime_transaction(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = MagicMock()
        mock_runtime.execute.return_value = _execution()
        mock_runtime_class.return_value = mock_runtime

        run(
            SandboxInvocation("user:chat", "test-profile", "inspect this", "append-system", append_system="obey policy"),
            _InProcessRunStore(),
        )

        request = mock_runtime.execute.call_args.args[0]
        self.assertEqual(request.append_system, "obey policy")

    def test_summary_is_bounded_single_line_and_prefers_stdout(self) -> None:
        summary = _bounded_summary(ExecResult(exit_code=0, stdout="one\ntwo " + "x" * 500, stderr="ignored"), "completed", limit=20)

        self.assertEqual(summary, "[completed] one two " + "x" * 11 + "…")
        self.assertEqual(_bounded_summary(ExecResult(exit_code=1, stdout="", stderr="\nstderr\n"), "failed"), "[failed] stderr")

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_output_events_preserve_stdout_stderr_then_error_order(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = mock_runtime_class.return_value
        mock_runtime.execute.return_value = _execution(exit_code=7, stdout="normal output", stderr="warning output")

        result = run(SandboxInvocation("user:chat", "test-profile", "task", "ordered-events"), _InProcessRunStore())

        self.assertEqual(
            result.outputs[:2],
            (StreamEvent(name="stdout", text="normal output"), StreamEvent(name="stderr", text="warning output")),
        )
        self.assertEqual(result.outputs[2], ErrorEvent(ename="OMPExecutionError", evalue="omp exited with code 7"))

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_model_thinking_and_lean_options_reach_omp(self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = mock_runtime_class.return_value
        mock_runtime.execute.return_value = _execution()

        run(
            SandboxInvocation(
                "user:chat", "test-profile", "task", "forward-options", model="provider/model", thinking="high", lean=True
            ),
            _InProcessRunStore(),
        )

        request = mock_runtime.execute.call_args.args[0]
        self.assertEqual(request.model, "provider/model")
        self.assertEqual(request.thinking, "high")
        self.assertTrue(request.lean)

    @patch("agent_sandbox.invocation.load_profile")
    @patch("agent_sandbox.invocation.SandboxRuntime")
    def test_distinct_slots_do_not_share_the_logical_slot_lock(
        self, mock_runtime_class: MagicMock, mock_load_profile: MagicMock
    ) -> None:
        mock_profile = _profile()
        mock_load_profile.return_value = mock_profile
        mock_runtime = mock_runtime_class.return_value

        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        calls_lock = threading.Lock()
        calls = 0

        def execute_transaction(*args, **kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_entered.set()
                if not release_first.wait(timeout=1):
                    raise AssertionError("first invocation was not released")
            else:
                second_entered.set()
            return _execution()

        mock_runtime.execute.side_effect = execute_transaction
        store = _InProcessRunStore()
        errors: list[BaseException] = []

        def execute(invocation: SandboxInvocation) -> None:
            try:
                run(invocation, store)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=execute, args=(SandboxInvocation("slot-one", "test-profile", "first", "first-run"),))
        second = threading.Thread(target=execute, args=(SandboxInvocation("slot-two", "test-profile", "second", "second-run"),))
        first.start()
        try:
            self.assertTrue(first_entered.wait(timeout=1))
            second.start()
            self.assertTrue(second_entered.wait(timeout=1))
        finally:
            release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(mock_runtime.execute.call_count, 2)

    def test_append_system_requires_text_within_64_kib(self) -> None:
        with self.assertRaises(TypeError):
            SandboxInvocation("user:chat", "test-profile", "task", "invalid-append", append_system=object())
        with self.assertRaises(ValueError):
            SandboxInvocation("user:chat", "test-profile", "task", "oversized-append", append_system="x" * (64 * 1024 + 1))


class TestSandboxRunWorkspaceRead(unittest.TestCase):
    """Exercise bounded reads from retained workspace captures."""

    def test_reads_regular_and_nested_members_with_conventional_prefix(self) -> None:
        capture = _write_workspace_capture(
            (tarfile.TarInfo("./result.txt"), b"root"),
            (tarfile.TarInfo("./nested/result.txt"), b"nested"),
        )
        run = _run_with_capture(capture)
        try:
            self.assertEqual(run.read_workspace_file("result.txt"), b"root")
            self.assertEqual(run.read_workspace_file("nested/result.txt"), b"nested")
        finally:
            run.release()

    def test_rejects_traversal_missing_and_released_captures(self) -> None:
        capture = _write_workspace_capture((tarfile.TarInfo("result.txt"), b"result"))
        run = _run_with_capture(capture)
        try:
            with self.assertRaises(ValueError):
                run.read_workspace_file("../result.txt")
            with self.assertRaisesRegex(AgentSandboxError, "does not contain"):
                run.read_workspace_file("missing.txt")
            run.release()
            with self.assertRaisesRegex(AgentSandboxError, "Refusing workspace capture"):
                run.read_workspace_file("result.txt")
        finally:
            run.release()

    def test_rejects_duplicate_members(self) -> None:
        capture = _write_workspace_capture(
            (tarfile.TarInfo("result.txt"), b"first"),
            (tarfile.TarInfo("result.txt"), b"second"),
        )
        run = _run_with_capture(capture)
        try:
            with self.assertRaisesRegex(AgentSandboxError, "duplicate"):
                run.read_workspace_file("result.txt")
        finally:
            run.release()

    def test_rejects_unsafe_and_nonregular_members(self) -> None:
        directory = tarfile.TarInfo("result.txt")
        directory.type = tarfile.DIRTYPE
        link = tarfile.TarInfo("result.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "target.txt"
        fifo = tarfile.TarInfo("result.txt")
        fifo.type = tarfile.FIFOTYPE
        unsafe = tarfile.TarInfo("../result.txt")

        for member, message in (
            (directory, "regular file"),
            (link, "regular file"),
            (fifo, "unsupported member type"),
            (unsafe, "unsafe member path"),
        ):
            with self.subTest(member=member.type):
                capture = _write_workspace_capture((member, b""))
                run = _run_with_capture(capture)
                try:
                    with self.assertRaisesRegex(AgentSandboxError, message):
                        run.read_workspace_file("result.txt")
                finally:
                    run.release()

    def test_enforces_declared_and_requested_size_limits(self) -> None:
        capture = _write_workspace_capture((tarfile.TarInfo("result.txt"), b"toolong"))
        run = _run_with_capture(capture)
        try:
            with self.assertRaisesRegex(AgentSandboxError, "exceeds 3 byte limit"):
                run.read_workspace_file("result.txt", max_bytes=3)
            for limit, error in ((0, ValueError), (-1, ValueError), (1.0, TypeError), (True, TypeError)):
                with self.subTest(limit=limit), self.assertRaises(error):
                    run.read_workspace_file("result.txt", max_bytes=limit)
        finally:
            run.release()

    def test_rejects_truncated_capture_member(self) -> None:
        capture = _write_workspace_capture((tarfile.TarInfo("result.txt"), b"result"))
        os.truncate(capture, 513)
        run = _run_with_capture(capture)
        try:
            with self.assertRaises(AgentSandboxError):
                run.read_workspace_file("result.txt")
        finally:
            run.release()


if __name__ == "__main__":
    unittest.main()
