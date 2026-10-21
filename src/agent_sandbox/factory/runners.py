"""Run agent turns on the host (LocalRunner) or inside an AiaB sandbox (SandboxRunner); hooks and gates always run on the host."""

from collections.abc import Sequence
import contextlib
import dataclasses
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import tempfile
import threading
from typing import Protocol
import uuid

from agent_sandbox import (
    AgentSandboxError,
    OmpTask,
    PromptImage,
    Sandbox,
    SandboxInvocation,
    StreamEvent,
    WorkspaceArchive,
    extract_archive,
    load_profile,
    manifest_from_dict,
    manifest_to_dict,
)
from agent_sandbox.archive import archived_member_name
from agent_sandbox.factory.config import OUTPUT_DIR, SESSION_DIR, TaskConfig
from agent_sandbox.factory.models import Envelope, EnvelopeError, parse_envelope
from agent_sandbox.factory.processes import terminate_process_groups
from agent_sandbox.factory.prompts import correction_prompt
from agent_sandbox.manifest import WorkspaceRef
from agent_sandbox.omp import build_omp_argv

OUTPUT_TAIL_CHARS = 8000
ACTIVE_PROCESS_FILE = "active-process.json"
_HEAVY_DIRECTORIES = (".venv", "node_modules", ".next", "__pycache__")
_SANDBOX_EXCLUDE = (
    ".git",
    SESSION_DIR,
    f"{OUTPUT_DIR}/*.json",  # Envelopes are read from the capture; scenario files are kept and travel both ways.
    *_HEAVY_DIRECTORIES,
    *(f"*/{name}" for name in _HEAVY_DIRECTORIES),
)

_running: dict[str, set[subprocess.Popen]] = {}
_running_lock = threading.Lock()
_sandboxes: dict[str, Sandbox] = {}
_sandboxes_lock = threading.Lock()


class AgentRunError(RuntimeError):
    """Report an agent turn that crashed, timed out, or exited nonzero."""


@dataclass(frozen=True)
class AgentTurn:
    """Describe one agent invocation."""

    phase: str
    prompt: str
    system_prompt: str
    output_path: str  # Repo-relative envelope path.
    model: str | None = None
    thinking: str | None = None
    images: tuple[PromptImage, ...] = ()
    timeout_s: int = 3600

    def omp_task(self) -> OmpTask:
        """Build the omp task this turn runs."""
        return OmpTask(
            self.prompt,
            prompt_images=self.images,
            model=self.model,
            thinking=self.thinking,
            append_system=self.system_prompt,
            timeout_s=self.timeout_s,
        )


@dataclass(frozen=True)
class TurnResult:
    """Hold the outcome of one agent turn."""

    output: str
    envelope: bytes | None


@dataclass(frozen=True)
class CommandResult:
    """Hold the outcome of one command."""

    exit_code: int
    output: str
    timed_out: bool = False


class Runner(Protocol):
    """Execute agent turns and commands against a task worktree."""

    def agent_turn(self, turn: AgentTurn) -> TurnResult:
        """Run one agent turn (continuing the phase's session when one exists) and return its output and envelope bytes."""

    def command(self, argv: Sequence[str], timeout_s: int) -> CommandResult:
        """Run one command in the worktree without a shell; changes persist in the worktree."""


# --- process tracking ---


def kill_task_processes(task_id: str) -> None:
    """Terminate every host process group the task currently runs (used on cancel)."""
    with _running_lock:
        processes = list(_running.get(task_id, ()))
    terminate_process_groups(processes)


def release_sandbox(task_id: str) -> None:
    """Forget the cached Sandbox object of a finished task."""
    with _sandboxes_lock:
        _sandboxes.pop(task_id, None)


def _sandbox_for(task_id: str) -> Sandbox:
    with _sandboxes_lock:
        if task_id not in _sandboxes:
            _sandboxes[task_id] = Sandbox(namespace=f"task-{task_id}")
        return _sandboxes[task_id]


def _tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    return text if len(text) <= limit else "..." + text[-limit:]


def _start_time(pid: int) -> str | None:
    """Return the kernel start tick of a process (distinguishes it from a later process reusing the pid)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rpartition(")")[2].split()[19]


def _reap_orphan(active_file: Path) -> None:
    """Kill the process group a crashed engine left running for this task, then forget it."""
    try:
        record = json.loads(active_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if _start_time(record["pid"]) == record["start"]:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(record["pid"], signal.SIGKILL)
    active_file.unlink(missing_ok=True)


def _run_logged(task_id: str, argv: Sequence[str], cwd: Path, log_path: Path, timeout_s: int) -> tuple[int, str, bool]:
    """Run argv in its own process group, logging combined output to a file; return (exit code, output tail, timed out)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    active_file = log_path.parent / ACTIVE_PROCESS_FILE
    _reap_orphan(active_file)
    timed_out = False
    with open(log_path, "ab") as log:
        log.write(f"\n$ {' '.join(argv)}\n".encode())
        log.flush()
        offset = log.tell()
        process = subprocess.Popen(
            list(argv), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
        )
        with _running_lock:
            _running.setdefault(task_id, set()).add(process)
        active_file.write_text(json.dumps({"pid": process.pid, "start": _start_time(process.pid)}), encoding="utf-8")
        try:
            try:
                exit_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process_groups([process])
                exit_code = process.returncode
        finally:
            active_file.unlink(missing_ok=True)
            with _running_lock:
                _running.get(task_id, set()).discard(process)
    with open(log_path, "rb") as log:
        log.seek(offset)
        output = log.read().decode(errors="replace")
    return exit_code, _tail(output), timed_out


# --- host command execution ---


class _HostCommands:
    """Run hooks and gates directly on the host in the task worktree, where the dependencies live."""

    def __init__(self, task_id: str, worktree: Path, task_dir: Path) -> None:
        self.task_id = task_id
        self.worktree = worktree
        self.task_dir = task_dir

    def command(self, argv: Sequence[str], timeout_s: int) -> CommandResult:
        exit_code, output, timed_out = _run_logged(
            self.task_id, argv, self.worktree, self.task_dir / "logs" / "commands.log", timeout_s
        )
        return CommandResult(exit_code, output, timed_out)


# --- local runner ---


class LocalRunner(_HostCommands):
    """Run omp directly on the host inside the task worktree."""

    def __init__(self, task_id: str, worktree: Path, task_dir: Path, omp_binary: str = "omp") -> None:
        super().__init__(task_id, worktree, task_dir)
        self.omp_binary = omp_binary

    def agent_turn(self, turn: AgentTurn) -> TurnResult:
        envelope_file = self.worktree / turn.output_path
        envelope_file.unlink(missing_ok=True)
        session_dir = self.task_dir / "session" / turn.phase
        resuming = session_dir.is_dir() and any(session_dir.iterdir())
        session_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir = self.task_dir / "prompts"
        images_dir = self.task_dir / "images" / uuid.uuid4().hex
        prompts_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True)
        system_file = prompts_dir / f"{turn.phase}.md"
        system_file.write_text(turn.system_prompt, encoding="utf-8")
        for image in turn.images:
            (images_dir / image.filename).write_bytes(image.content)
        task = turn.omp_task()
        argv = build_omp_argv(
            self.omp_binary,
            task,
            session_dir=str(session_dir),
            resuming=resuming,
            images_dir=str(images_dir),
            append_system_path=str(system_file),
        )
        exit_code, output, timed_out = _run_logged(
            self.task_id, argv, self.worktree, self.task_dir / "logs" / f"{turn.phase}.log", turn.timeout_s
        )
        if timed_out:
            raise AgentRunError(f"omp {turn.phase} turn exceeded {turn.timeout_s}s")
        if exit_code != 0:
            raise AgentRunError(f"omp {turn.phase} turn exited {exit_code}:\n{output}")
        envelope = envelope_file.read_bytes() if envelope_file.is_file() else None
        return TurnResult(output, envelope)


# --- sandbox runner ---


def _is_session_member(member: tarfile.TarInfo) -> bool:
    name = archived_member_name(member.name)
    return name == SESSION_DIR or name.startswith(SESSION_DIR + "/")


def _save_session_members(capture_path: str, session_tar: Path) -> None:
    """Persist only the OMP session tree of a captured workspace."""
    session_tar.parent.mkdir(parents=True, exist_ok=True)
    temporary = session_tar.with_suffix(".tmp")
    with tarfile.open(capture_path, "r:") as source, tarfile.open(temporary, "w:") as target:
        for member in source:
            if not _is_session_member(member):
                continue
            member.name = archived_member_name(member.name)
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
    os.replace(temporary, session_tar)


def _append_session_members(archive_path: str, session_tar: Path) -> None:
    """Add a saved OMP session tree to a workspace tar."""
    with tarfile.open(session_tar, "r:") as source, tarfile.open(archive_path, "a:") as target:
        for member in source:
            target.addfile(member, source.extractfile(member) if member.isfile() else None)


class SandboxRunner(_HostCommands):
    """Run omp turns inside an AiaB sandbox; the worktree stays authoritative between runs."""

    def __init__(self, task_id: str, worktree: Path, task_dir: Path, profile_name: str) -> None:
        super().__init__(task_id, worktree, task_dir)
        self.sandbox_dir = task_dir / "sandbox"
        self.profile_name = profile_name
        self.mount_exclusions = load_profile(profile_name).workdir_mount_paths()

    def _session_tar(self, phase: str) -> Path:
        return self.sandbox_dir / phase / "session.tar"

    def _manifest_path(self, phase: str) -> Path:
        return self.sandbox_dir / phase / "manifest.json"

    def _restore_archive(self, phase: str | None) -> tuple[WorkspaceArchive, bool]:
        """Tar the worktree (plus the phase's saved session) and report whether the session can be resumed."""
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        descriptor, path = tempfile.mkstemp(suffix=".tar", prefix="factory-workspace-", dir=self.sandbox_dir)
        os.close(descriptor)
        try:
            WorkspaceArchive.from_directory(str(self.worktree), path, exclude=(*_SANDBOX_EXCLUDE, *self.mount_exclusions))
            resuming = phase is not None and self._session_tar(phase).is_file() and self._manifest_path(phase).is_file()
            if resuming:
                _append_session_members(path, self._session_tar(phase))
            return WorkspaceArchive.from_file(path), resuming
        except BaseException:
            Path(path).unlink(missing_ok=True)
            raise

    def _extract(self, capture_path: str) -> None:
        extract_archive(capture_path, str(self.worktree), exclude=(*_SANDBOX_EXCLUDE, *self.mount_exclusions))

    def agent_turn(self, turn: AgentTurn) -> TurnResult:
        (self.worktree / turn.output_path).unlink(missing_ok=True)
        archive, resuming = self._restore_archive(turn.phase)
        try:
            manifest = None
            if resuming:
                prior = manifest_from_dict(json.loads(self._manifest_path(turn.phase).read_text(encoding="utf-8")))
                manifest = dataclasses.replace(prior, workspace=WorkspaceRef(snapshot_asset_id=None, sha256=archive.sha256))
            invocation = SandboxInvocation(
                slot_key=f"task-{self.task_id}",
                profile_name=self.profile_name,
                run_id=f"{turn.phase}-{uuid.uuid4().hex}",
                task=turn.omp_task(),
                active_manifest=manifest,
                workspace=archive,
            )
            try:
                run = _sandbox_for(self.task_id).run(invocation)
            except AgentSandboxError as error:
                raise AgentRunError(f"sandbox {turn.phase} turn failed: {error}") from error
        finally:
            Path(archive.path).unlink(missing_ok=True)
        with run:
            self._extract(run.workspace.path)
            _save_session_members(run.workspace.path, self._session_tar(turn.phase))
            self._manifest_path(turn.phase).write_text(json.dumps(manifest_to_dict(run.next_manifest)), encoding="utf-8")
            output = "".join(event.text for event in run.outputs if isinstance(event, StreamEvent))
            if run.status != "completed":
                raise AgentRunError(f"sandbox {turn.phase} turn failed:\n{_tail(output)}")
            try:
                envelope = run.read_workspace_file(turn.output_path)
            except AgentSandboxError:
                envelope = None
        return TurnResult(_tail(output), envelope)


def make_runner(config: TaskConfig, task_id: str, worktree: Path, task_dir: Path, omp_binary: str) -> Runner:
    """Build the runner a task's configuration asks for."""
    if config.sandbox:
        return SandboxRunner(task_id, worktree, task_dir, config.profile)
    return LocalRunner(task_id, worktree, task_dir, omp_binary)


# --- envelope protocol ---


def run_agent_turn(runner: Runner, turn: AgentTurn) -> Envelope:
    """Run an agent turn and validate its envelope, with one `--continue` correction turn on failure.

    Raises EnvelopeError when the correction also fails.
    """
    result = runner.agent_turn(turn)
    try:
        return parse_envelope(turn.phase, result.envelope)
    except EnvelopeError as error:
        correction = dataclasses.replace(turn, prompt=correction_prompt(turn.phase, str(error), turn.output_path), images=())
    return parse_envelope(turn.phase, runner.agent_turn(correction).envelope)
