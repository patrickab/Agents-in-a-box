"""Test envelope validation, the correction turn, and the local and sandbox runners."""

import dataclasses
import json
from pathlib import Path
import stat
import tarfile
import textwrap

import pytest

from agent_sandbox import WorkspaceArchive
from agent_sandbox.factory import runners
from agent_sandbox.factory.models import EnvelopeError, parse_envelope
from agent_sandbox.factory.runners import AgentRunError, AgentTurn, CommandResult, LocalRunner

VALID_CODE = {"status": "ok", "summary": "done", "changed_files": ["a.py"], "question": None}


@pytest.mark.parametrize(
    ("phase", "raw", "message"),
    [
        ("code", None, "was not written"),
        ("code", b"not json", "invalid"),
        ("code", json.dumps({"status": "maybe", "summary": "x"}), "status"),
        ("code", json.dumps({"status": "ok"}), "summary"),
        ("review", json.dumps({"status": "ok", "findings": []}), "approved"),
        ("plan", json.dumps({"status": "ok", "summary": "s", "steps": "not a list"}), "steps"),
    ],
)
def test_invalid_envelopes_raise_with_a_correction_ready_message(phase: str, raw: bytes | str | None, message: str) -> None:
    with pytest.raises(EnvelopeError, match=message):
        parse_envelope(phase, raw)


TURN = AgentTurn(phase="code", prompt="do it", system_prompt="sys", output_path=".factory/out/code.json")


def write_fake_omp(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_local_runner_deletes_a_stale_envelope_before_the_turn(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    (worktree / ".factory/out").mkdir(parents=True)
    (worktree / ".factory/out/code.json").write_text(json.dumps(VALID_CODE))
    omp = write_fake_omp(tmp_path / "omp", "print('does nothing')\n")
    result = LocalRunner("task00000001", worktree, tmp_path / "task", omp_binary=str(omp)).agent_turn(TURN)
    assert result.envelope is None


def test_local_runner_reports_failure_and_timeout(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    failing = write_fake_omp(tmp_path / "omp-fail", "import sys\nprint('boom')\nsys.exit(3)\n")
    with pytest.raises(AgentRunError, match=r"(?s)exited 3.*boom"):
        LocalRunner("task00000001", worktree, tmp_path / "task", omp_binary=str(failing)).agent_turn(TURN)

    slow = write_fake_omp(tmp_path / "omp-slow", "import time\ntime.sleep(30)\n")
    with pytest.raises(AgentRunError, match="exceeded 1s"):
        LocalRunner("task00000001", worktree, tmp_path / "task", omp_binary=str(slow)).agent_turn(
            dataclasses.replace(TURN, timeout_s=1)
        )


def test_local_runner_commands_run_in_the_worktree_without_a_shell(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    runner = LocalRunner("task00000001", worktree, tmp_path / "task")

    ok = runner.command(["python3", "-c", "import os; print(os.getcwd())"], 30)
    assert ok.exit_code == 0 and str(worktree) in ok.output

    literal = runner.command(["python3", "-c", "import sys; print(sys.argv[1])", "$(echo injected); true"], 30)
    assert "$(echo injected); true" in literal.output

    assert runner.command(["python3", "-c", "raise SystemExit(5)"], 30).exit_code == 5
    timed_out = runner.command(["python3", "-c", "import time; time.sleep(30)"], 1)
    assert timed_out.timed_out


def test_kill_task_processes_terminates_a_running_command(tmp_path: Path) -> None:
    import threading
    import time

    worktree = tmp_path / "wt"
    worktree.mkdir()
    runner = LocalRunner("task00000009", worktree, tmp_path / "task")
    results: list[CommandResult] = []
    thread = threading.Thread(target=lambda: results.append(runner.command(["python3", "-c", "import time; time.sleep(60)"], 120)))
    thread.start()
    time.sleep(1)
    runners.kill_task_processes("task00000009")
    thread.join(10)

    assert not thread.is_alive() and results[0].exit_code != 0 and not results[0].timed_out


def test_orphaned_process_of_a_crashed_engine_is_reaped_before_the_next_run(tmp_path: Path) -> None:
    import subprocess
    import sys
    import time

    logs = tmp_path / "task/logs"
    logs.mkdir(parents=True)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    (logs / runners.ACTIVE_PROCESS_FILE).write_text(json.dumps({"pid": orphan.pid, "start": runners._start_time(orphan.pid)}))

    worktree = tmp_path / "wt"
    worktree.mkdir()
    LocalRunner("task00000001", worktree, tmp_path / "task").command(["true"], 30)

    deadline = time.monotonic() + 5
    while orphan.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert orphan.poll() is not None
    assert not (logs / runners.ACTIVE_PROCESS_FILE).exists()


class FakeSandboxRun:
    def __init__(self, workspace: WorkspaceArchive, outputs: tuple, status: str, manifest: object) -> None:
        self.workspace, self.outputs, self.status, self.next_manifest = workspace, outputs, status, manifest
        self.released = False

    def read_workspace_file(self, relative_path: str, max_bytes: int = 0) -> bytes:
        with tarfile.open(self.workspace.path) as archive:
            return archive.extractfile("./" + relative_path).read()

    def release(self) -> None:
        self.released = True

    def __enter__(self) -> "FakeSandboxRun":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@pytest.fixture(autouse=True)
def demo_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the tests' `demo` sandbox profile a workdir mount of AGENTS.md like the real gigachad profile."""
    from agent_sandbox.profile import Mount, Profile

    profile = Profile(
        name="demo",
        image="img",
        workdir="/workspace",
        mounts=(Mount("/host/AGENTS.md", "/workspace/AGENTS.md", "ro"), Mount("/host/omp", "/usr/local/bin/omp", "ro")),
        omp_binary="/usr/local/bin/omp",
    )
    monkeypatch.setattr(runners, "load_profile", lambda name: profile)


def test_sandbox_runner_round_trips_worktree_session_and_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive SandboxRunner against a fake Sandbox: the worktree and session survive across turns."""
    from agent_sandbox import StreamEvent
    from agent_sandbox.manifest import Execution, OmpSessionState, Outputs, SandboxManifest, WorkspaceRef

    worktree = tmp_path / "wt"
    (worktree / "src").mkdir(parents=True)
    (worktree / "src/a.py").write_text("old\n")
    (worktree / ".git").write_text("gitdir: /elsewhere\n")
    (worktree / "AGENTS.md").write_text("host-mounted in the container\n")
    kept_on_host = [
        ".venv/bin/python",
        "node_modules/pkg/index.js",
        "src/frontend/node_modules/pkg/index.js",
        "src/frontend/.next/cache/blob",
        "src/__pycache__/a.pyc",
        ".factory/out/stale.json",
    ]
    for relative in kept_on_host:
        (worktree / relative).parent.mkdir(parents=True, exist_ok=True)
        (worktree / relative).write_text("host only\n")
    invocations = []
    restored_names = []

    class FakeSandbox:
        def run(self, invocation):
            invocations.append(invocation)
            with tarfile.open(invocation.workspace.path) as archive:
                restored_names.append({name.removeprefix("./") for name in archive.getnames()})
            # Emulate the container: restore the archive, mutate files, add a session, write the envelope, capture.
            scratch = tmp_path / f"scratch{len(invocations)}"
            scratch.mkdir()
            with tarfile.open(invocation.workspace.path) as archive:
                archive.extractall(scratch, filter="data")
            assert not (scratch / ".git").exists()
            (scratch / "src/a.py").write_text(f"turn {len(invocations)}\n")
            session = scratch / ".omp-session"
            session.mkdir(exist_ok=True)
            (session / "log.jsonl").write_text(f"session after turn {len(invocations)}\n")
            (scratch / ".factory/out").mkdir(parents=True, exist_ok=True)
            (scratch / ".factory/out/code.json").write_text(json.dumps(VALID_CODE))
            (scratch / ".factory/out/scenario.py").write_text(f"print('scenario {len(invocations)}')\n")
            capture = str(tmp_path / f"capture{len(invocations)}.tar")
            workspace = WorkspaceArchive.from_directory(str(scratch), capture, exclude=())
            manifest = SandboxManifest(
                schema_version=1,
                manifest_id=f"m{len(invocations)}",
                profile="demo",
                runtime_fingerprint="sha256:" + "0" * 64,
                workspace=WorkspaceRef(None, workspace.sha256),
                outputs=Outputs(None, ()),
                omp_sessions={"main": OmpSessionState("main", None)},
                execution=Execution("run", "completed", 0, "2026-01-01T00:00:00+00:00"),
            )
            return FakeSandboxRun(workspace, (StreamEvent("stdout", "sandbox says hi"),), "completed", manifest)

    fake = FakeSandbox()
    monkeypatch.setattr(runners, "_sandbox_for", lambda task_id: fake)
    runner = runners.SandboxRunner("task00000001", worktree, tmp_path / "task", "demo")

    first = runner.agent_turn(TURN)
    second = runner.agent_turn(TURN)

    assert json.loads(first.envelope)["summary"] == "done" and "sandbox says hi" in first.output
    assert invocations[0].active_manifest is None
    assert invocations[1].active_manifest is not None
    assert invocations[1].active_manifest.workspace.sha256 == invocations[1].workspace.sha256  # re-pinned to the rebuilt archive
    assert ".omp-session" not in restored_names[0]
    assert ".omp-session/log.jsonl" in restored_names[1] and "src/a.py" in restored_names[1] and ".git" not in restored_names[1]
    assert (worktree / "src/a.py").read_text() == "turn 2\n"
    assert "AGENTS.md" not in restored_names[0]  # profile mounts inside the workdir are neither archived...
    assert (worktree / "AGENTS.md").read_text() == "host-mounted in the container\n"  # ...nor deleted by extraction
    assert not (worktree / ".omp-session").exists() and (worktree / ".git").is_file()
    assert not any(set(name.split("/")) & {".venv", "node_modules", ".next", "__pycache__"} for name in restored_names[1])
    assert ".factory/out/stale.json" not in restored_names[1] and ".factory/out/scenario.py" in restored_names[1]
    assert all((worktree / relative).read_text() == "host only\n" for relative in kept_on_host)
    assert not (worktree / ".factory/out/code.json").exists()  # envelopes are read from the capture, never extracted
    assert (worktree / ".factory/out/scenario.py").read_text() == "print('scenario 2')\n"
    assert second.envelope is not None
