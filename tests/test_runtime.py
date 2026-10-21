"""Exercise managed-runtime transactions with mocked Docker APIs."""

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

from agent_sandbox.archive import WorkspaceArchive
from agent_sandbox.config import RUNTIME_TMPFS_SIZE, TMP_TMPFS_SIZE, runtime_resource_names
from agent_sandbox.errors import ArchiveError, ContainerRuntimeError, PythonScriptError
from agent_sandbox.omp import OmpTask
from agent_sandbox.profile import Mount, Profile
from agent_sandbox.runtime import INTERNAL_WORKDIR, SandboxRuntime, WorkspaceRestore
import docker

_Plan = tuple[list[tuple[bytes | None, bytes | None]], int]
_TASK = OmpTask("perform work")


class _ExecApi:
    """Small Docker exec API fake that exposes chunks only while iterated."""

    def __init__(self, plans: list[_Plan]) -> None:
        self._plans = plans
        self._by_id: dict[str, _Plan] = {}
        self.commands: list[list[str]] = []
        self.streamed_chunks = 0

    def exec_create(self, _container_id: str, command: list[str], **_kwargs: object) -> dict[str, str]:
        exec_id = f"exec-{len(self.commands)}"
        self.commands.append(command)
        self._by_id[exec_id] = self._plans[len(self.commands) - 1]
        return {"Id": exec_id}

    def exec_start(self, exec_id: str, *, stream: bool, demux: bool):
        if not stream or not demux:
            raise AssertionError("runtime must request a demultiplexed output stream")
        for chunk in self._by_id[exec_id][0]:
            self.streamed_chunks += 1
            yield chunk

    def exec_inspect(self, exec_id: str) -> dict[str, int]:
        return {"ExitCode": self._by_id[exec_id][1]}


class _Container:
    def __init__(self, plans: list[_Plan]) -> None:
        self.id = "new-container"
        self.api = _ExecApi(plans)
        self.client = SimpleNamespace(api=self.api)
        self.start = MagicMock()
        self.remove = MagicMock()


class _WorkspaceCaptureApi(_ExecApi):
    """Run the tar command received by a transaction against a local fixture workspace."""

    def __init__(self, plans: list[_Plan], workspace: str) -> None:
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
        "omp_binary": "/usr/local/bin/omp",
    }
    return Profile(**(fields | overrides))


def _tar_bytes(member: tarfile.TarInfo, content: bytes = b"") -> bytes:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        if member.isfile():
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        else:
            archive.addfile(member)
    return archive_bytes.getvalue()


_CAPTURE_ARCHIVE = _tar_bytes(tarfile.TarInfo("result.txt"), b"workspace archive")


def _plans(
    omp_chunks: list[tuple[bytes | None, bytes | None]] | None = None,
    capture_chunks: list[tuple[bytes | None, bytes | None]] | None = None,
) -> list[_Plan]:
    return [
        ([], 0),  # prepare workspace, optional restore, and OMP home
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
    return SandboxRuntime(profile or _profile(), runtime_resource_names("transaction-test"), client=client), client


def _effective_fingerprint(runtime: SandboxRuntime, image_id: str = "sha256:resolved-image") -> str:
    material = f"{runtime.profile.runtime_fingerprint}\x00{image_id}".encode()
    return f"sha256:{hashlib.sha256(material).hexdigest()}"


def _restore_archive(test: unittest.TestCase) -> WorkspaceArchive:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as source:
        source.write(_tar_bytes(tarfile.TarInfo("file.txt"), b"ok"))
    test.addCleanup(Path(source.name).unlink, missing_ok=True)
    return WorkspaceArchive.from_file(source.name)


def _patched_capture_path(test: unittest.TestCase) -> tuple[str, object]:
    """Return a capture path and a patch that makes the runtime write its snapshot there."""
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    capture_path = os.path.join(directory.name, "capture.tar")
    descriptor = os.open(capture_path, os.O_RDWR | os.O_CREAT, 0o600)
    return capture_path, patch("agent_sandbox.runtime.tempfile.mkstemp", return_value=(descriptor, capture_path))


@patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "perform work"])
@patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
class TestManagedTransaction(unittest.TestCase):
    def test_fresh_container_is_hardened_captured_and_removed(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        runtime, client = _runtime(container)

        result = runtime.execute(_TASK)
        self.addCleanup(Path(result.workspace.path).unlink, missing_ok=True)

        kwargs = client.containers.create.call_args.kwargs
        self.assertEqual(client.containers.create.call_args.args[0], "sha256:resolved-image")
        self.assertEqual(kwargs["name"], "agent-sandbox-transaction-test-runtime")
        self.assertTrue(kwargs["read_only"])
        self.assertEqual((kwargs["runtime"], kwargs["cap_drop"], kwargs["network_mode"]), ("runsc", ["ALL"], "bridge"))
        self.assertEqual(
            kwargs["tmpfs"],
            {"/runtime": f"rw,exec,nosuid,nodev,size={RUNTIME_TMPFS_SIZE}", "/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"},
        )
        self.assertEqual(kwargs["mounts"], [])
        self.assertEqual(result.exec_result.stdout, "omp output")
        self.assertEqual(result.workspace.sha256, hashlib.sha256(_CAPTURE_ARCHIVE).hexdigest())
        self.assertEqual(result.runtime_fingerprint, _effective_fingerprint(runtime))
        self.assertEqual(Path(result.workspace.path).read_bytes(), _CAPTURE_ARCHIVE)
        self.assertEqual(oct(os.stat(result.workspace.path).st_mode & 0o777), "0o600")
        setup = container.api.commands[0]
        self.assertEqual(setup[:2], ["sh", "-c"])
        self.assertIn("rm -rf /runtime/omp-home", setup[2])
        self.assertNotIn("tar --extract", setup[2])
        container.start.assert_called_once_with()
        container.remove.assert_called_once_with(force=True)

    def test_removes_a_stale_named_container_before_creating_fresh_one(self, _lease: MagicMock, _argv: MagicMock) -> None:
        events: list[str] = []
        stale = MagicMock()
        stale.remove.side_effect = lambda **_kwargs: events.append("stale removed")
        container = _Container(_plans())
        runtime, client = _runtime(container, stale=stale)
        client.containers.create.side_effect = lambda *_args, **_kwargs: events.append("fresh created") or container

        Path(runtime.execute(_TASK).workspace.path).unlink()

        self.assertEqual(events, ["stale removed", "fresh created"])

    def test_start_failure_removes_created_container(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        container.start.side_effect = RuntimeError("start denied")
        runtime, _client = _runtime(container)

        with self.assertRaisesRegex(RuntimeError, "start denied"):
            runtime.execute(_TASK)

        container.remove.assert_called_once_with(force=True)

    def test_output_limit_stops_streaming_and_removes_container(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans(omp_chunks=[(b"ab", None), (b"cd", None)]))
        runtime, _client = _runtime(container)

        with (
            patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 3),
            self.assertRaisesRegex(ContainerRuntimeError, "command output exceeds 3 byte limit"),
        ):
            runtime.execute(_TASK)

        self.assertEqual(container.api.streamed_chunks, 2)
        container.remove.assert_called_once_with(force=True)

    def test_remove_failure_unlinks_capture_and_is_typed(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans())
        container.remove.side_effect = RuntimeError("remove denied")
        runtime, _client = _runtime(container)
        capture_path, capture_patch = _patched_capture_path(self)

        with capture_patch, self.assertRaisesRegex(ContainerRuntimeError, "failed to force-remove managed container: remove denied"):
            runtime.execute(_TASK)

        self.assertFalse(os.path.exists(capture_path))

    def test_cleanup_failure_keeps_both_errors_visible(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans(omp_chunks=[(b"ab", None), (b"cd", None)]))
        container.remove.side_effect = RuntimeError("remove denied")
        runtime, _client = _runtime(container)

        pattern = "command output exceeds 3 byte limit.*Additionally failed to force-remove container: remove denied"
        with patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 3), self.assertRaisesRegex(ContainerRuntimeError, pattern):
            runtime.execute(_TASK)

    def test_capture_is_hard_capped_and_unlinked(self, _lease: MagicMock, _argv: MagicMock) -> None:
        container = _Container(_plans(capture_chunks=[(b"ab", None), (b"cd", None)]))
        runtime, _client = _runtime(container)
        capture_path, capture_patch = _patched_capture_path(self)

        with (
            capture_patch,
            patch("agent_sandbox.runtime.MAX_WORKSPACE_ARCHIVE_BYTES", 3),
            self.assertRaisesRegex(ArchiveError, "captured workspace exceeds 3 byte limit"),
        ):
            runtime.execute(_TASK)

        self.assertFalse(os.path.exists(capture_path))
        self.assertEqual(container.api.streamed_chunks, 3)
        container.remove.assert_called_once_with(force=True)

    def test_capture_rejects_unrestorable_members_and_unlinks_it(self, _lease: MagicMock, _argv: MagicMock) -> None:
        fifo = tarfile.TarInfo("pipe")
        fifo.type = tarfile.FIFOTYPE
        container = _Container(_plans(capture_chunks=[(_tar_bytes(fifo), None)]))
        runtime, _client = _runtime(container)
        capture_path, capture_patch = _patched_capture_path(self)

        with capture_patch, self.assertRaisesRegex(ArchiveError, "unsupported member type"):
            runtime.execute(_TASK)

        self.assertFalse(os.path.exists(capture_path))
        container.remove.assert_called_once_with(force=True)

    def test_verified_restore_is_read_only_mounted_and_extracted(self, _lease: MagicMock, _argv: MagicMock) -> None:
        archive = _restore_archive(self)
        container = _Container(_plans())
        runtime, client = _runtime(container)

        Path(runtime.execute(_TASK, WorkspaceRestore(archive)).workspace.path).unlink()

        mounts = client.containers.create.call_args.kwargs["mounts"]
        restore_mount = next(mount for mount in mounts if mount["Target"] == "/runtime/restore-archive.tar")
        self.assertEqual((restore_mount["Source"], restore_mount["ReadOnly"]), (archive.path, True))
        self.assertIn("tar --extract --file=/runtime/restore-archive.tar", container.api.commands[0][2])
        self.assertIn("--no-same-owner --no-same-permissions", container.api.commands[0][2])

    def test_resume_requires_the_matching_effective_fingerprint(self, _lease: MagicMock, argv: MagicMock) -> None:
        archive = _restore_archive(self)
        container = _Container(_plans())
        runtime, client = _runtime(container)

        with self.assertRaisesRegex(ContainerRuntimeError, "runtime fingerprint mismatch"):
            runtime.execute(_TASK, WorkspaceRestore(archive, "sha256:other"))
        client.containers.create.assert_not_called()

        Path(runtime.execute(_TASK, WorkspaceRestore(archive, _effective_fingerprint(runtime))).workspace.path).unlink()
        self.assertTrue(argv.call_args.kwargs["resuming"])

    def test_restore_hash_mismatch_never_creates_a_container(self, _lease: MagicMock, _argv: MagicMock) -> None:
        archive = _restore_archive(self)
        runtime, client = _runtime(_Container(_plans()))

        with self.assertRaisesRegex(ArchiveError, "workspace hash mismatch"):
            runtime.execute(_TASK, WorkspaceRestore(WorkspaceArchive(archive.path, "0" * 64)))

        client.containers.create.assert_not_called()

    def test_capture_excludes_only_the_exact_mounted_workspace_path(self, _lease: MagicMock, _argv: MagicMock) -> None:
        profile = _profile(mounts=(Mount(source="/host/AGENTS.md", target="/workspace/AGENTS.md", mode="ro"),))
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "AGENTS.md").write_text("mounted instructions")
            Path(workspace, "nested").mkdir()
            Path(workspace, "nested", "AGENTS.md").write_text("workspace file")
            container = _Container(_plans())
            container.api = _WorkspaceCaptureApi(_plans(), workspace)
            container.client = SimpleNamespace(api=container.api)
            runtime, client = _runtime(container, profile=profile)

            result = runtime.execute(_TASK)
            self.addCleanup(Path(result.workspace.path).unlink, missing_ok=True)
            with tarfile.open(result.workspace.path) as archive:
                names = archive.getnames()

        self.assertNotIn("./AGENTS.md", names)
        self.assertIn("./nested/AGENTS.md", names)
        self.assertEqual(client.containers.create.call_args.kwargs["mounts"][0]["Target"], f"{INTERNAL_WORKDIR}/AGENTS.md")


@patch("agent_sandbox.runtime._runtime_lease", return_value=nullcontext())
class TestOmpTimeout(unittest.TestCase):
    def test_omp_execution_uses_the_task_timeout(self, _lease: MagicMock) -> None:
        container = _Container(_plans())
        runtime, _client = _runtime(container)

        with patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "go"]):
            result = runtime.execute(OmpTask("go", timeout_s=7))
        self.addCleanup(Path(result.workspace.path).unlink, missing_ok=True)
        self.assertEqual(container.api.commands[1][:3], ["timeout", "--signal=KILL", "7s"])

        timed_out_runtime, _client = _runtime(_Container([([], 0), ([], 124)]))
        with (
            patch("agent_sandbox.runtime.build_omp_argv", return_value=["omp", "go"]),
            self.assertRaisesRegex(ContainerRuntimeError, "OMP execution exceeded 9s limit"),
        ):
            timed_out_runtime.execute(OmpTask("go", timeout_s=9))


class TestProbe(unittest.TestCase):
    def test_probe_runs_in_a_hardened_container_with_profile_mounts(self) -> None:
        container = _Container([([(b"ok\n", None)], 0)])
        runtime, client = _runtime(container, profile=_profile(mounts=(Mount("/host/project", "/workspace/project", "rw"),)))

        self.assertEqual(runtime.run_probe(["true"]), "ok\n")

        kwargs = client.containers.create.call_args.kwargs
        self.assertEqual(client.containers.create.call_args.args[0], "test:latest")
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["tmpfs"], {"/tmp": f"rw,nosuid,nodev,size={TMP_TMPFS_SIZE}"})
        self.assertEqual(kwargs["network_mode"], "bridge")
        self.assertEqual((kwargs["mounts"][0]["Target"], kwargs["mounts"][0]["ReadOnly"]), ("/workspace/project", False))
        self.assertEqual(container.api.commands, [["true"]])
        container.remove.assert_called_once_with(force=True)

    def test_probe_failures_are_raised_after_cleanup(self) -> None:
        container = _Container([([(None, b"boom")], 3)])
        runtime, _client = _runtime(container)
        with self.assertRaisesRegex(ContainerRuntimeError, "probe exited 3: boom"):
            runtime.run_probe(["false"])
        container.remove.assert_called_once_with(force=True)

        container = _Container([])
        container.start.side_effect = docker.errors.DockerException("start denied")
        runtime, _client = _runtime(container)
        with self.assertRaisesRegex(docker.errors.DockerException, "start denied"):
            runtime.run_probe(["true"])
        container.remove.assert_called_once_with(force=True)


class TestDisposablePython(unittest.TestCase):
    def _runtime(self, container: _Container) -> tuple[SandboxRuntime, MagicMock]:
        profile = _profile(
            mounts=(Mount("/host/venv", "/opt/venv", "ro"), Mount("/host/other", "/opt/other", "ro")),
            python_interpreters={"venv": "/opt/venv/bin/python3"},
        )
        return _runtime(container, profile=profile)

    def test_python_runs_network_disabled_with_only_interpreter_mounts(self) -> None:
        container = _Container([([(b"42\n", None)], 0)])
        runtime, client = self._runtime(container)

        self.assertEqual(runtime.execute_python("venv", "print(42)", 1), "42\n")

        kwargs = client.containers.create.call_args.kwargs
        self.assertTrue(kwargs["network_disabled"])
        self.assertNotIn("network_mode", kwargs)
        self.assertEqual([mount["Target"] for mount in kwargs["mounts"]], ["/opt/venv"])
        self.assertEqual(container.api.commands[0][-3:], ["/opt/venv/bin/python3", "-c", "print(42)"])
        container.remove.assert_called_once_with(force=True)

    def test_python_rejects_unknown_interpreter_and_invalid_timeout(self) -> None:
        runtime, client = self._runtime(_Container([]))
        for interpreter, timeout, message in (("missing", 1, "Unknown"), ("venv", 0, "timeout"), ("venv", True, "timeout")):
            with self.subTest(interpreter=interpreter, timeout=timeout), self.assertRaisesRegex(PythonScriptError, message):
                runtime.execute_python(interpreter, "print(1)", timeout)
        client.containers.create.assert_not_called()

    def test_python_failures_are_typed_and_remove_the_container(self) -> None:
        for plans, start_error, remove_error, message in (
            ([([(b"abcde", None), (b"fghij", None)], 0)], None, None, "command output exceeds 8 byte limit"),
            ([], RuntimeError("start denied"), None, "Sandbox Python execution failed: start denied"),
            ([([(b"ok", None)], 0)], None, RuntimeError("remove denied"), "failed to force-remove Python container: remove denied"),
            ([([(None, b"boom")], 1)], None, None, "boom"),
            ([([], 124)], None, None, "did not finish within 1s"),
        ):
            with self.subTest(message=message):
                container = _Container(plans)
                container.start.side_effect = start_error
                container.remove.side_effect = remove_error
                runtime, _client = self._runtime(container)
                with patch("agent_sandbox.runtime.MAX_RETAINED_OUTPUT_BYTES", 8), self.assertRaisesRegex(PythonScriptError, message):
                    runtime.execute_python("venv", "print('ok')", 1)
                container.remove.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
