"""Exercise the public managed-runtime transaction with mocked Docker APIs."""

from contextlib import nullcontext
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from agent_sandbox.config import RUNTIME_TMPFS_SIZE, TMP_TMPFS_SIZE, runtime_resource_names
from agent_sandbox.profile import Mount, Profile, WorkspaceConfig
from agent_sandbox.runtime import (
    INTERNAL_WORKDIR,
    ContainerRuntimeError,
    ManagedExecutionRequest,
    PythonScriptError,
    SandboxRuntime,
)
import docker


class _ExecApi:
    """Small Docker exec API fake that exposes chunks only while iterated."""

    def __init__(self, plans: list[tuple[list[tuple[bytes | None, bytes | None]], int]]) -> None:
        self._plans = plans
        self._by_id: dict[str, tuple[list[tuple[bytes | None, bytes | None]], int]] = {}
        self.commands: list[list[str]] = []
        self.streamed_chunks = 0

    def exec_create(self, _container_id: str, command: list[str], **_kwargs: object) -> dict[str, str]:
        exec_id = f"exec-{len(self.commands)}"
        self.commands.append(command)
        self._by_id[exec_id] = self._plans[len(self.commands) - 1]
        return {"Id": exec_id}

    def exec_start(self, exec_id: str, *, stream: bool, demux: bool):
        self.assert_streaming(stream, demux)
        for chunk in self._by_id[exec_id][0]:
            self.streamed_chunks += 1
            yield chunk

    def exec_inspect(self, exec_id: str) -> dict[str, int]:
        return {"ExitCode": self._by_id[exec_id][1]}

    @staticmethod
    def assert_streaming(stream: bool, demux: bool) -> None:
        if not stream or not demux:
            raise AssertionError("runtime must request a demultiplexed output stream")


class _Container:
    def __init__(self, plans: list[tuple[list[tuple[bytes | None, bytes | None]], int]]) -> None:
        self.id = "new-container"
        self.api = _ExecApi(plans)
        self.client = SimpleNamespace(api=self.api)
        self.put_archive = MagicMock(return_value=True)
        self.start = MagicMock()
        self.remove = MagicMock()


class _WorkspaceCaptureApi(_ExecApi):
    """Run the tar command received by a transaction against a local fixture workspace."""

    def __init__(self, plans: list[tuple[list[tuple[bytes | None, bytes | None]], int]], workspace: str) -> None:
        super().__init__(plans)
        self._workspace = workspace

    def exec_start(self, exec_id: str, *, stream: bool, demux: bool):
        command = self.commands[int(exec_id.removeprefix("exec-"))]
        if command[:1] == ["tar"]:
            local_command = [self._workspace if value == INTERNAL_WORKDIR else value for value in command]
            completed = subprocess.run(local_command, capture_output=True, check=False)
            self._by_id[exec_id] = ([(completed.stdout, completed.stderr)], completed.returncode)
        yield from super().exec_start(exec_id, stream=stream, demux=demux)


def _profile(**overrides: object) -> Profile:
    fields: dict[str, object] = {
        "name": "test",
        "image": "test:latest",
        "workdir": "/workspace",
        "mounts": (),
        "workspace": WorkspaceConfig(mode="managed"),
        "omp_binary": "/usr/local/bin/omp",
    }
    fields.update(overrides)
    return Profile(**fields)

def _request(**overrides: object) -> ManagedExecutionRequest:
    fields: dict[str, object] = {
        "run_id": "run-1",
        "manifest_id": None,
        "workspace_archive_path": None,
        "expected_workspace_sha256": None,
        "prompt": "perform work",
        "resuming": False,
    }
    fields.update(overrides)
    return ManagedExecutionRequest(**fields)

def _tar_bytes(member: tarfile.TarInfo, content: bytes = b"") -> bytes:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        if member.isfile():
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        else:
            archive.addfile(member)
    return archive_bytes.getvalue()


_regular_capture_member = tarfile.TarInfo("result.txt")
_CAPTURE_ARCHIVE = _tar_bytes(_regular_capture_member, b"workspace archive")


def _plans(
    omp_chunks: list[tuple[bytes | None, bytes | None]] | None = None,
    capture_chunks: list[tuple[bytes | None, bytes | None]] | None = None,
    *,
    restore: bool = False,
) -> list[tuple[list[tuple[bytes | None, bytes | None]], int]]:
    plans = [([], 0)]  # mkdir workspace
    if restore:
        plans.append(([], 0))  # extract mounted restore archive
    return [
        *plans,
        ([], 0),  # materialize a new OMP home
        (omp_chunks or [(b"omp output", b"")], 0),
        (capture_chunks or [(_CAPTURE_ARCHIVE, None)], 0),
    ]


def _runtime(
    container: _Container,
    *,
    image_id: str = "sha256:resolved-image",
    profile: Profile | None = None,
    stale: object | None = None,
) -> tuple[SandboxRuntime, MagicMock]:
    client = MagicMock()
    client.info.return_value = {"SecurityOptions": ["name=rootless"], "Runtimes": {"runsc": {}}}
    client.images.get.return_value = SimpleNamespace(id=image_id)
    if stale is None:
        client.containers.get.side_effect = docker.errors.NotFound("missing")
    else:
        client.containers.get.return_value = stale
    client.containers.create.return_value = container

    runtime = SandboxRuntime.__new__(SandboxRuntime)
    runtime.profile = profile or _profile()
    runtime.client = client
    runtime._resource_names = runtime_resource_names("transaction-test")
    return runtime, client


class TestManagedTransaction(unittest.TestCase):
    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_fresh_container_has_read_only_root_bounded_tmpfs_and_is_removed_on_success(
        self, _lease: MagicMock, _argv: MagicMock
    ) -> None:
        container = _Container(_plans())
        runtime, client = _runtime(container)

        result = runtime.execute(_request())
        try:
            kwargs = client.containers.create.call_args.kwargs
            self.assertEqual(client.containers.create.call_args.args[0], "sha256:resolved-image")
            self.assertEqual(kwargs["name"], "agent-sandbox-transaction-test-runtime")
            self.assertTrue(kwargs["read_only"])
            self.assertEqual(kwargs["network_mode"], "bridge")
            self.assertEqual(
                kwargs["tmpfs"],
                {
                    "/runtime": f"rw,exec,nosuid,nodev,size={RUNTIME_TMPFS_SIZE}",
                    "/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}",
                },
            )
            self.assertEqual(kwargs["mounts"], [])
            self.assertEqual(result.exec_result.stdout, "omp output")
            self.assertEqual(result.workspace_sha256, hashlib.sha256(_CAPTURE_ARCHIVE).hexdigest())
            with tarfile.open(result.archive_path) as archive:
                self.assertEqual(archive.extractfile("result.txt").read(), b"workspace archive")
            self.assertTrue(os.path.exists(result.archive_path))
            self.assertEqual(oct(os.stat(result.archive_path).st_mode & 0o777), "0o600")
            self.assertEqual(container.api.commands[1][0:2], ["sh", "-c"])
            self.assertIn("rm -rf /runtime/omp-home", container.api.commands[1][2])
            self.assertNotIn("ACTIVE_STATE_ID", " ".join(" ".join(command) for command in container.api.commands))
            client.volumes.assert_not_called()
        finally:
            Path(result.archive_path).unlink(missing_ok=True)
        container.remove.assert_called_once_with(force=True)
        container.start.assert_called_once_with()

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_removes_a_stale_stopped_named_container_before_creating_fresh_one(
        self, _lease: MagicMock, _argv: MagicMock
    ) -> None:
        events: list[str] = []
        stale = MagicMock(status="exited")
        stale.remove.side_effect = lambda **_kwargs: events.append("stale removed")
        container = _Container(_plans())
        runtime, client = _runtime(container, stale=stale)
        client.containers.create.side_effect = lambda *_args, **_kwargs: events.append("fresh created") or container

        result = runtime.execute(_request())
        Path(result.archive_path).unlink(missing_ok=True)

        self.assertEqual(events[:2], ["stale removed", "fresh created"])
        stale.remove.assert_called_once_with(force=True)
        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_managed_start_failure_removes_created_container(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        container.start.side_effect = RuntimeError("start denied")
        runtime, _client = _runtime(container)

        with self.assertRaisesRegex(RuntimeError, "start denied"):
            runtime.execute(_request())

        container.start.assert_called_once_with()
        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_removes_fresh_container_after_streaming_error(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans(omp_chunks=[(b"ab", None), (b"cd", None)]))
        runtime, _client = _runtime(container)

        with patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 3), self.assertRaisesRegex(
            ContainerRuntimeError, "command output exceeds 3 byte limit"
        ):
            runtime.execute(_request())

        self.assertEqual(container.api.streamed_chunks, 2)
        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_managed_remove_failure_unlinks_capture_and_is_typed(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        container.remove.side_effect = RuntimeError("remove denied")
        runtime, _client = _runtime(container)
        with tempfile.TemporaryDirectory() as directory:
            capture_path = os.path.join(directory, "capture.tar")
            descriptor = os.open(capture_path, os.O_RDWR | os.O_CREAT, 0o666)
            with patch("agent_sandbox.runtime.tempfile.mkstemp", return_value=(descriptor, capture_path)), self.assertRaisesRegex(
                ContainerRuntimeError, "failed to force-remove managed container: remove denied"
            ):
                runtime.execute(_request())
            self.assertFalse(os.path.exists(capture_path))

        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_managed_cleanup_failure_keeps_both_errors_visible(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans(omp_chunks=[(b"ab", None), (b"cd", None)]))
        container.remove.side_effect = RuntimeError("remove denied")
        runtime, _client = _runtime(container)

        pattern = (
            "command output exceeds 3 byte limit.*"
            "[Aa]dditionally failed to force-remove container: remove denied"
        )
        with patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 3), self.assertRaisesRegex(
            ContainerRuntimeError, pattern
        ):
            runtime.execute(_request())

        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_capture_is_incremental_hashed_to_private_tempfile_and_hard_capped(
        self, _lease: MagicMock, _argv: MagicMock
    ) -> None:
        container = _Container(_plans(capture_chunks=[(b"ab", None), (b"cd", None)]))
        runtime, _client = _runtime(container)
        with tempfile.TemporaryDirectory() as directory:
            capture_path = os.path.join(directory, "capture.tar")
            descriptor = os.open(capture_path, os.O_RDWR | os.O_CREAT, 0o666)
            with patch("agent_sandbox.runtime.tempfile.mkstemp", return_value=(descriptor, capture_path)), patch(
                "agent_sandbox.runtime.MAX_CAPTURED_WORKSPACE_BYTES", 3
            ), self.assertRaisesRegex(ContainerRuntimeError, "captured workspace exceeds 3 byte limit"):
                runtime.execute(_request())
            self.assertFalse(os.path.exists(capture_path))

        self.assertEqual(container.api.streamed_chunks, 3)
        container.remove.assert_called_once_with(force=True)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_capture_rejects_unrestorable_members_and_unlinks_tempfile(
        self, _lease: MagicMock, _argv: MagicMock
    ) -> None:
        fifo = tarfile.TarInfo("pipe")
        fifo.type = tarfile.FIFOTYPE
        absolute_link = tarfile.TarInfo("absolute-link")
        absolute_link.type = tarfile.SYMTYPE
        absolute_link.linkname = "/outside"
        escaping_link = tarfile.TarInfo("nested/escape-link")
        escaping_link.type = tarfile.SYMTYPE
        escaping_link.linkname = "../../outside"

        for member, message in (
            (fifo, "unsupported member type"),
            (absolute_link, "unsafe link target"),
            (escaping_link, "unsafe link target"),
        ):
            with self.subTest(member=member.name), tempfile.TemporaryDirectory() as directory:
                capture_path = os.path.join(directory, "capture.tar")
                descriptor = os.open(capture_path, os.O_RDWR | os.O_CREAT, 0o666)
                container = _Container(_plans(capture_chunks=[(_tar_bytes(member), None)]))
                runtime, _client = _runtime(container)

                with patch("agent_sandbox.runtime.tempfile.mkstemp", return_value=(descriptor, capture_path)), self.assertRaisesRegex(
                    ContainerRuntimeError, message
                ):
                    runtime.execute(_request())

                self.assertFalse(os.path.exists(capture_path))
                container.remove.assert_called_once_with(force=True)
    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_effective_fingerprint_uses_resolved_image_id(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        runtime, _client = _runtime(container, image_id="sha256:image-bytes")

        result = runtime.execute(_request())
        Path(result.archive_path).unlink(missing_ok=True)

        expected = hashlib.sha256(f"{runtime.profile.runtime_fingerprint}\x00sha256:image-bytes".encode()).hexdigest()
        self.assertEqual(result.runtime_fingerprint, f"sha256:{expected}")
        self.assertNotEqual(result.runtime_fingerprint, runtime.profile.runtime_fingerprint)

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_initial_verified_archive_is_read_only_mounted_and_extracted(self, _lease: MagicMock, _argv: MagicMock) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as source:
            archive_path = source.name
        try:
            with tarfile.open(archive_path, "w") as archive:
                entry = tarfile.TarInfo("file.txt")
                entry.size = 2
                archive.addfile(entry, fileobj=io.BytesIO(b"ok"))
            archive_bytes = Path(archive_path).read_bytes()
            container = _Container(_plans(restore=True))
            runtime, client = _runtime(container)
            result = runtime.execute(
                _request(workspace_archive_path=archive_path, expected_workspace_sha256=hashlib.sha256(archive_bytes).hexdigest())
            )
            Path(result.archive_path).unlink(missing_ok=True)

            restore_mount = next(
                mount
                for mount in client.containers.create.call_args.kwargs["mounts"]
                if mount["Target"] == "/runtime/restore-archive.tar"
            )
            self.assertEqual(restore_mount["Source"], archive_path)
            self.assertTrue(restore_mount["ReadOnly"])
            self.assertIn("tar --extract --file=/runtime/restore-archive.tar", container.api.commands[1][2])
            self.assertIn("--no-same-owner --no-same-permissions", container.api.commands[1][2])
            container.put_archive.assert_not_called()
        finally:
            Path(archive_path).unlink(missing_ok=True)

    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_restore_hash_mismatch_never_creates_a_container(self, _lease: MagicMock) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as source:
            archive_path = source.name
        try:
            with tarfile.open(archive_path, "w") as archive:
                entry = tarfile.TarInfo("file.txt")
                entry.size = 2
                archive.addfile(entry, fileobj=io.BytesIO(b"ok"))
            container = _Container(_plans())
            runtime, client = _runtime(container)

            with self.assertRaisesRegex(ContainerRuntimeError, "workspace hash mismatch"):
                runtime.execute(_request(workspace_archive_path=archive_path, expected_workspace_sha256="0" * 64))

            client.containers.create.assert_not_called()
        finally:
            Path(archive_path).unlink(missing_ok=True)

    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_removes_stale_container_before_rejecting_prior_manifest_without_archive(self, _lease: MagicMock) -> None:
        stale = MagicMock(status="exited")
        container = _Container(_plans())
        runtime, client = _runtime(container, stale=stale)
        image_id = "sha256:resolved-image"
        material = f"{runtime.profile.runtime_fingerprint}\x00{image_id}".encode()
        effective = f"sha256:{hashlib.sha256(material).hexdigest()}"

        with self.assertRaisesRegex(ContainerRuntimeError, "workspace archive is required"):
            runtime.execute(
                _request(
                    manifest_id="prior-manifest",
                    prior_runtime_fingerprint=effective,
                    expected_workspace_sha256="a" * 64,
                )
            )

        stale.remove.assert_called_once_with(force=True)
        client.containers.create.assert_not_called()

    @patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
    @patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
    def test_anchored_non_wildcard_exclusion_keeps_same_basename_nested_paths(
        self, _lease: MagicMock, _argv: MagicMock
    ) -> None:
        profile = _profile(
            mounts=(Mount(source="/host/AGENTS.md", target="/workspace/AGENTS.md", mode="ro"),),
        )
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "AGENTS.md").write_text("mounted instructions")
            nested = Path(workspace, "nested")
            nested.mkdir()
            Path(nested, "AGENTS.md").write_text("workspace file")
            container = _Container(_plans())
            container.api = _WorkspaceCaptureApi(_plans(), workspace)
            container.client = SimpleNamespace(api=container.api)
            runtime, _client = _runtime(container, profile=profile)

            result = runtime.execute(_request())
            try:
                with tarfile.open(result.archive_path) as archive:
                    names = archive.getnames()
                self.assertNotIn("./AGENTS.md", names)
                self.assertIn("./nested/AGENTS.md", names)
            finally:
                Path(result.archive_path).unlink(missing_ok=True)

        capture_command = container.api.commands[-1]
        self.assertIn("--anchored", capture_command)
        self.assertIn("--no-wildcards", capture_command)
        self.assertIn("--exclude=./AGENTS.md", capture_command)
        self.assertNotIn("--exclude=AGENTS.md", capture_command)


class TestDoctorProbePolicy(unittest.TestCase):
    def test_probe_uses_a_read_only_root_bounded_tmpfs_and_cleanup(self) -> None:
        runtime = SandboxRuntime.__new__(SandboxRuntime)
        runtime.profile = _profile(mounts=(Mount(source="/host/project", target="/workspace", mode="rw"),))
        runtime.client = MagicMock()
        container = runtime.client.containers.create.return_value
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = b"ok"

        self.assertEqual(runtime.run_probe(["true"]), b"ok")

        kwargs = runtime.client.containers.create.call_args.kwargs
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["tmpfs"], {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"})
        self.assertEqual(kwargs["network_mode"], "bridge")
        self.assertNotIn("remove", kwargs)
        container.start.assert_called_once_with()
        container.remove.assert_called_once_with(force=True)

    def test_probe_start_failure_removes_created_container(self) -> None:
        runtime = SandboxRuntime.__new__(SandboxRuntime)
        runtime.profile = _profile()
        runtime.client = MagicMock()
        container = runtime.client.containers.create.return_value
        container.start.side_effect = docker.errors.DockerException("start denied")

        with self.assertRaisesRegex(docker.errors.DockerException, "start denied"):
            runtime.run_probe(["true"])

        container.remove.assert_called_once_with(force=True)


class TestDisposablePythonStreaming(unittest.TestCase):
    def test_python_uses_the_same_bounded_stream_and_removes_its_container(self) -> None:
        profile = _profile(
            mounts=(Mount(source="/host/venv", target="/opt/venv", mode="ro"),),
            python_interpreters=(("venv", "/opt/venv/bin/python3"),),
        )
        container = _Container([([(b"ab", None), (b"cd", None)], 0)])
        runtime, client = _runtime(container, profile=profile)

        with patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 3), self.assertRaisesRegex(
            PythonScriptError, "command output exceeds 3 byte limit"
        ):
            runtime.execute_python("venv", "print('too much')", 1)

        kwargs = client.containers.create.call_args.kwargs
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["tmpfs"], {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"})
        self.assertTrue(kwargs["network_disabled"])
        self.assertEqual(container.api.streamed_chunks, 2)
        container.remove.assert_called_once_with(force=True)

    def test_python_start_failure_removes_created_container(self) -> None:
        profile = _profile(
            mounts=(Mount(source="/host/venv", target="/opt/venv", mode="ro"),),
            python_interpreters=(("venv", "/opt/venv/bin/python3"),),
        )
        container = _Container([])
        container.start.side_effect = RuntimeError("start denied")
        runtime, _client = _runtime(container, profile=profile)

        with self.assertRaisesRegex(PythonScriptError, "Sandbox Python execution failed: start denied"):
            runtime.execute_python("venv", "print('ok')", 1)

        container.start.assert_called_once_with()
        container.remove.assert_called_once_with(force=True)

    def test_python_remove_failure_is_typed(self) -> None:
        profile = _profile(
            mounts=(Mount(source="/host/venv", target="/opt/venv", mode="ro"),),
            python_interpreters=(("venv", "/opt/venv/bin/python3"),),
        )
        container = _Container([([(b"ok", None)], 0)])
        container.remove.side_effect = RuntimeError("remove denied")
        runtime, _client = _runtime(container, profile=profile)

        with self.assertRaisesRegex(PythonScriptError, "failed to force-remove Python container: remove denied"):
            runtime.execute_python("venv", "print('ok')", 1)

        container.remove.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
