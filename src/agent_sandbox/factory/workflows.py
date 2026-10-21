"""Define the durable DBOS workflows of the task pipeline: work (agent rounds), merge, and the per-task lifecycle."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict

from dbos import DBOS, DBOSConfig, Queue, SetWorkflowID

from agent_sandbox.factory import git_ops, prompts
from agent_sandbox.factory.config import TaskConfig, agent_output_path, agent_writes
from agent_sandbox.factory.engine import Engine
from agent_sandbox.factory.models import Envelope, EnvelopeError, GateResult, ReviewEnvelope, envelope_summary
from agent_sandbox.factory.prompts import CodeMode
from agent_sandbox.factory.runners import AgentRunError, AgentTurn, Runner, run_agent_turn
from agent_sandbox.omp import PromptImage

LOGGER = logging.getLogger("agent_sandbox.factory")
APP_NAME = "agent-sandbox-factory"
APP_VERSION = "2"  # Bump when the workflow code changes incompatibly with pending workflows.
TASKS_QUEUE = "tasks"
MERGE_QUEUE = "merge"
COMMAND_TOPIC = "command"
COMMAND_WAIT_SECONDS = 24 * 3600
MERGE_ATTEMPTS = 3
ACTIVE_FLOW_STATUSES = frozenset({"PENDING", "ENQUEUED", "DELAYED"})
GATE_OUTPUT_TAIL = 4000
MERGE_GATES_ROUND = 0  # Phase runs of merge-time gates use round 0.
DEFAULT_SYSTEM_PROMPT = "You are the {phase} agent of an automated software factory. Follow the task instructions exactly."
ENTRY_CODE_MODES: dict[str, CodeMode] = {"fresh": "initial", "retry": "retry", "rework": "rework"}

_engine: Engine | None = None


def bind_engine(engine: Engine | None) -> None:
    """Install the engine the workflow steps run against."""
    global _engine
    _engine = engine


def current_engine() -> Engine:
    """Return the bound engine, or raise when none is installed."""
    if _engine is None:
        raise RuntimeError("factory engine is not bound")
    return _engine


# --- step results ---


class StepResult(TypedDict):
    """Report whether a step succeeded, with the failure text when it did not."""

    ok: bool
    error: str


class MergePrepareResult(StepResult):
    """Report the commit and rebase outcome; `noop` means the branch has nothing to merge."""

    noop: bool


class FastForwardResult(StepResult):
    """Report the fast-forward outcome; `retry` means main moved and the merge can start over."""

    retry: bool


class CodeResult(TypedDict):
    """Report a coder turn: `fail` blocks the task, `breach` lists reverted paths for the next round."""

    outcome: Literal["ok", "fail", "breach"]
    text: str


class ReviewResult(TypedDict):
    """Report a reviewer turn: `fail` blocks the task, `changes` carries the findings for the next round."""

    outcome: Literal["approved", "changes", "fail"]
    text: str


class _RoundProblem(NamedTuple):
    """Describe why a round did not pass and what the coder is told next."""

    mode: CodeMode
    text: str
    phase: str


# --- step context ---


@dataclass
class _PhaseRecord:
    """Collect the outcome of one phase run while its scope is open."""

    run_id: int
    status: str = "ok"
    summary: str = ""
    envelope: dict[str, Any] | None = None
    gates: list[GateResult] = field(default_factory=list)


@dataclass(frozen=True)
class _AgentTurnResult:
    """Hold a finished agent turn: its envelope (None when the turn failed) and the worktree tree from before it ran."""

    envelope: Envelope | None
    before_tree: str


@dataclass
class _Ctx:
    engine: Engine
    task_id: str
    config: TaskConfig
    worktree: Path

    @property
    def store(self):
        return self.engine.store

    @cached_property
    def runner(self) -> Runner:
        """Build the runner on first use, because git-only steps never need one."""
        return self.engine.runner_for(self.task_id, self.config, self.worktree)


def _open(task_id: str, config: dict[str, Any]) -> _Ctx:
    task_config = TaskConfig.model_validate(config)
    return _Ctx(current_engine(), task_id, task_config, Path(task_config.worktree_root) / task_id)


@contextmanager
def _phase_run(ctx: _Ctx, phase: str, round_number: int) -> Iterator[_PhaseRecord]:
    """Open a phase run row, close it with the recorded outcome, and mark it failed on exceptions."""
    ctx.store.update_task(ctx.task_id, phase=phase, round=round_number)
    record = _PhaseRecord(ctx.store.begin_phase(ctx.task_id, phase, round_number))
    try:
        yield record
    except BaseException as error:
        ctx.store.finish_phase(record.run_id, "failed", f"{type(error).__name__}: {error}")
        raise
    ctx.store.finish_phase(record.run_id, record.status, record.summary, record.envelope, record.gates)


def _agent_turn(ctx: _Ctx, phase: str, prompt: str, images: tuple[PromptImage, ...] = ()) -> AgentTurn:
    agent = ctx.config.agents[phase]
    return AgentTurn(
        phase=phase,
        prompt=prompt,
        system_prompt=agent.prompt or DEFAULT_SYSTEM_PROMPT.format(phase=phase),
        output_path=agent_output_path(agent, phase),
        model=agent.model,
        thinking=agent.thinking,
        images=images,
        timeout_s=ctx.config.agent_timeout_s,
    )


def _agent_phase(
    ctx: _Ctx, record: _PhaseRecord, phase: str, prompt: str, images: tuple[PromptImage, ...] = (), *, read_only: bool
) -> _AgentTurnResult:
    """Run one agent turn inside an open phase run, marking the record failed on a crash and discarding changes when read-only."""
    before_tree = git_ops.snapshot_tree(ctx.worktree)
    try:
        envelope = run_agent_turn(ctx.runner, _agent_turn(ctx, phase, prompt, images))
    except (AgentRunError, EnvelopeError) as error:
        record.status, record.summary = "failed", str(error)
        envelope = None
    finally:
        if read_only:
            _discard_changes(ctx, before_tree)
    return _AgentTurnResult(envelope, before_tree)


def _record_envelope(ctx: _Ctx, record: _PhaseRecord, envelope: Envelope) -> str:
    summary = envelope_summary(envelope)
    record.envelope = envelope.model_dump(mode="json")
    record.summary = summary
    text = summary
    if isinstance(envelope, ReviewEnvelope) and envelope.findings:
        text = f"{summary}\n" + "\n".join(f"- {finding}" for finding in envelope.findings)
    ctx.store.add_message(ctx.task_id, "agent", text, record.run_id)
    return summary


def _discard_changes(ctx: _Ctx, before_tree: str) -> None:
    """Throw away every worktree change made since `before_tree`."""
    changed = git_ops.tree_changes(ctx.worktree, before_tree, git_ops.snapshot_tree(ctx.worktree))
    git_ops.restore_paths(ctx.worktree, before_tree, changed)


def _run_commands(ctx: _Ctx, phase: str, round_number: int, commands: list[list[str]]) -> StepResult:
    """Run commands in order until one fails, recording each as a gate result."""
    if not commands:
        return {"ok": True, "error": ""}
    with _phase_run(ctx, phase, round_number) as record:
        failure = ""
        for argv in commands:
            result = ctx.runner.command(argv, ctx.config.command_timeout_s)
            record.gates.append(GateResult(argv=list(argv), exit_code=result.exit_code, output_tail=result.output[-GATE_OUTPUT_TAIL:]))
            if result.exit_code != 0:
                failure = prompts.failure_text(argv, result.exit_code, result.output, result.timed_out)
                break
        record.status = "failed" if failure else "ok"
        record.summary = failure.splitlines()[0] if failure else f"{len(commands)} command(s) passed"
    return {"ok": not failure, "error": failure}


# --- steps ---


@DBOS.step()
def state_step(task_id: str, state: str, phase: str | None, error: str | None) -> None:
    """Record a task state transition."""
    current_engine().store.update_task(task_id, state=state, phase=phase, error=error)


@DBOS.step()
def block_step(task_id: str, reason: str, phase: str | None) -> None:
    """Mark the task blocked and tell the user why."""
    store = current_engine().store
    if store.update_task(task_id, state="blocked", phase=phase, error=reason) is not None:
        store.add_message(task_id, "agent", f"Blocked: {reason}")


@DBOS.step()
def completed_phases_step(task_id: str) -> list[str]:
    """List the scout/plan phases that already produced an envelope (for retries)."""
    store = current_engine().store
    return [phase for phase in ("scout", "plan") if store.completed_phase_envelope(task_id, phase) is not None]


@DBOS.step()
def prepare_step(task_id: str, config: dict[str, Any]) -> StepResult:
    """Create the worktree and run after_create hooks once."""
    ctx = _open(task_id, config)
    marker = ctx.engine.task_dir(task_id) / "after_create.done"
    if marker.exists() and ctx.worktree.is_dir():
        return {"ok": True, "error": ""}
    with _phase_run(ctx, "prepare", 0) as record:
        info = git_ops.add_worktree(Path(ctx.config.repo_path), Path(ctx.config.worktree_root), task_id, ctx.config.main_branch)
        ctx.store.update_task(task_id, worktree=str(info.path), branch=info.branch, base_sha=info.base_sha, emit=False)
        record.summary = f"worktree {info.path} on {info.branch} at {info.base_sha[:10]}"
        outcome = _run_commands(ctx, "hooks", 0, ctx.config.hooks.after_create)
        if not outcome["ok"]:
            record.status = "failed"
            record.summary = f"after_create hook failed: {outcome['error'].splitlines()[0]}"
            return {"ok": False, "error": f"after_create hook failed:\n{outcome['error']}"}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    return {"ok": True, "error": ""}


@DBOS.step()
def readonly_step(task_id: str, config: dict[str, Any], phase: str, image_names: list[str]) -> StepResult:
    """Run the scout or plan agent and discard any workspace changes it made."""
    ctx = _open(task_id, config)
    agent = ctx.config.agents[phase]
    output_path = agent_output_path(agent, phase)
    task = ctx.engine.get_task(task_id)
    if phase == "scout":
        prompt = prompts.scout_prompt(task.title, task.request, output_path)
    else:
        prompt = prompts.plan_prompt(task.title, task.request, ctx.store.completed_phase_envelope(task_id, "scout"), output_path)
    images = ctx.engine.load_images(task_id, image_names)
    with _phase_run(ctx, phase, 0) as record:
        turn = _agent_phase(ctx, record, phase, prompt, images, read_only=True)
        if turn.envelope is None:
            return {"ok": False, "error": f"{phase} failed: {record.summary}"}
        summary = _record_envelope(ctx, record, turn.envelope)
        if turn.envelope.status != "ok":
            record.status = "failed"
            return {"ok": False, "error": f"{phase} reported failure: {summary}"}
    return {"ok": True, "error": ""}


@DBOS.step()
def code_step(
    task_id: str, config: dict[str, Any], round_number: int, mode: CodeMode, text: str, image_names: list[str]
) -> CodeResult:
    """Run one coder turn, revert writes outside the allowlist, and report the outcome."""
    ctx = _open(task_id, config)
    agent = ctx.config.agents["code"]
    writes = agent_writes(agent, "code")
    output_path = agent_output_path(agent, "code")
    task = ctx.engine.get_task(task_id)
    prompt = prompts.code_prompt(
        mode,
        title=task.title,
        request=task.request,
        scout=ctx.store.completed_phase_envelope(task_id, "scout"),
        plan=ctx.store.completed_phase_envelope(task_id, "plan"),
        text=text,
        writes=writes,
        protected=ctx.config.protected,
        output_path=output_path,
    )
    images = ctx.engine.load_images(task_id, image_names)
    with _phase_run(ctx, "code", round_number) as record:
        turn = _agent_phase(ctx, record, "code", prompt, images, read_only=False)
        if turn.envelope is None:
            return {"outcome": "fail", "text": f"code failed: {record.summary}"}
        changed = git_ops.tree_changes(ctx.worktree, turn.before_tree, git_ops.snapshot_tree(ctx.worktree))
        breaches = git_ops.find_breaches(changed, writes, ctx.config.protected)
        if breaches:
            git_ops.restore_paths(ctx.worktree, turn.before_tree, breaches)
        summary = _record_envelope(ctx, record, turn.envelope)
        if turn.envelope.status != "ok":
            record.status = "failed"
            return {"outcome": "fail", "text": turn.envelope.question or summary}
        if breaches:
            record.status = "failed"
            record.summary = f"reverted writes outside the allowlist: {', '.join(breaches)}"
            return {"outcome": "breach", "text": "\n".join(f"- {path}" for path in breaches)}
    return {"outcome": "ok", "text": summary}


@DBOS.step()
def hooks_step(task_id: str, config: dict[str, Any], round_number: int) -> StepResult:
    """Run the after_code hooks."""
    ctx = _open(task_id, config)
    return _run_commands(ctx, "hooks", round_number, ctx.config.hooks.after_code)


@DBOS.step()
def gates_step(task_id: str, config: dict[str, Any], round_number: int) -> StepResult:
    """Run the quality gates."""
    ctx = _open(task_id, config)
    return _run_commands(ctx, "gates", round_number, ctx.config.gates)


@DBOS.step()
def review_step(task_id: str, config: dict[str, Any], round_number: int) -> ReviewResult:
    """Run the reviewer agent over the current diff and discard any workspace changes it made."""
    ctx = _open(task_id, config)
    agent = ctx.config.agents["review"]
    task = ctx.engine.get_task(task_id)
    with _phase_run(ctx, "review", round_number) as record:
        prompt = prompts.review_prompt(
            task.title,
            task.request,
            ctx.store.completed_phase_envelope(task_id, "plan"),
            git_ops.changed_files(ctx.worktree),
            git_ops.diff_text(ctx.worktree),
            agent_output_path(agent, "review"),
        )
        turn = _agent_phase(ctx, record, "review", prompt, read_only=True)
        if turn.envelope is None:
            return {"outcome": "fail", "text": f"review failed: {record.summary}"}
        _record_envelope(ctx, record, turn.envelope)
        findings = "\n".join(f"- {finding}" for finding in turn.envelope.findings)
        if turn.envelope.status != "ok":
            record.status = "failed"
            return {"outcome": "fail", "text": findings or "review reported failure"}
        if not turn.envelope.approved:
            record.status = "failed"
            return {"outcome": "changes", "text": findings or "the reviewer requested changes without details"}
    return {"outcome": "approved", "text": ""}


@DBOS.step()
def preview_step(task_id: str, config: dict[str, Any]) -> None:
    """(Re)start the task preview; a failing preview does not block the task."""
    ctx = _open(task_id, config)
    if ctx.config.preview is None:
        return
    with _phase_run(ctx, "preview", 0) as record:
        try:
            ctx.engine.previews.stop(task_id)
            preview = ctx.engine.previews.start(task_id, ctx.worktree, ctx.config.preview)
        except Exception as error:
            LOGGER.exception("preview for %s failed", task_id)
            record.status, record.summary = "failed", f"preview failed: {error}"
            return
        record.status = "ok" if preview.status == "ready" else "failed"
        record.summary = preview.open_url if preview.status == "ready" else (preview.error or preview.status)
    ctx.store.add_event(task_id, "preview", {"status": preview.status})


@DBOS.step()
def merge_prepare_step(task_id: str, config: dict[str, Any]) -> MergePrepareResult:
    """Commit the changed files and rebase the branch onto the main branch."""
    ctx = _open(task_id, config)
    task = ctx.engine.get_task(task_id)
    with _phase_run(ctx, "merge", 0) as record:
        committed = git_ops.commit_changes(ctx.worktree, task.title, ctx.config.protected)
        repo = Path(ctx.config.repo_path)
        branch = task.branch or git_ops.branch_name(task_id)
        if not committed and not git_ops.is_ahead_of(repo, branch, ctx.config.main_branch):
            record.summary = "no changes to merge"
            return {"ok": True, "noop": True, "error": ""}
        outcome = git_ops.rebase_onto(ctx.worktree, ctx.config.main_branch)
        if not outcome.ok:
            _uncommit(ctx)
            record.status, record.summary = "failed", outcome.error.splitlines()[0]
            return {"ok": False, "noop": False, "error": outcome.error}
        record.summary = f"committed and rebased onto {ctx.config.main_branch}"
    return {"ok": True, "noop": False, "error": ""}


def _uncommit(ctx: _Ctx) -> None:
    """Return the branch to uncommitted state on top of its merge base with the main branch."""
    base = git_ops.git_text(ctx.worktree, "merge-base", ctx.config.main_branch, "HEAD")
    git_ops.uncommit(ctx.worktree, base)
    ctx.store.update_task(ctx.task_id, base_sha=base, emit=False)


@DBOS.step()
def merge_abort_step(task_id: str, config: dict[str, Any]) -> None:
    """Undo the merge commit so the worktree diff shows the changes again."""
    _uncommit(_open(task_id, config))


@DBOS.step()
def fast_forward_step(task_id: str, config: dict[str, Any]) -> FastForwardResult:
    """Fast-forward the main checkout to the task branch while main is still its ancestor."""
    ctx = _open(task_id, config)
    repo = Path(ctx.config.repo_path)
    branch = ctx.engine.get_task(task_id).branch or git_ops.branch_name(task_id)
    if not git_ops.can_fast_forward(repo, branch, ctx.config.main_branch):
        return {"ok": False, "retry": True, "error": f"{ctx.config.main_branch} moved during the merge"}
    outcome = git_ops.fast_forward_main(repo, branch, ctx.config.main_branch)
    return {"ok": outcome.ok, "retry": False, "error": outcome.error}


@DBOS.step()
def finalize_step(task_id: str) -> None:
    """Tear down the preview and worktree, then mark the task done."""
    engine = current_engine()
    try:
        engine.teardown_task(task_id)
    except Exception:
        LOGGER.exception("teardown of merged task %s failed", task_id)
        engine.store.add_event(task_id, "teardown.failed", {})
    engine.store.update_task(task_id, state="done", phase=None, error=None)


# --- workflows ---


def _block(task_id: str, reason: str, phase: str | None) -> str:
    block_step(task_id, reason, phase)
    return "blocked"


@DBOS.workflow()
def work(task_id: str, config: dict[str, Any], entry: str, feedback: dict[str, Any] | None) -> str:
    """Run prepare, scout/plan, and the code rounds, then preview; return human_review, merge, or blocked."""
    try:
        return _work(task_id, config, entry, feedback)
    except Exception as error:
        LOGGER.exception("work workflow of %s crashed", task_id)
        return _block(task_id, f"unexpected error: {type(error).__name__}: {error}", None)


def _work(task_id: str, config: dict[str, Any], entry: str, feedback: dict[str, Any] | None) -> str:
    settings = TaskConfig.model_validate(config)
    if entry != "rework":
        state_step(task_id, "in_progress", "prepare", None)
    prepared = prepare_step(task_id, config)
    if not prepared["ok"]:
        return _block(task_id, prepared["error"], "prepare")
    request_images = current_engine().request_image_names(task_id)
    if entry != "rework":
        finished = completed_phases_step(task_id)
        for phase in ("scout", "plan"):
            if phase in settings.phases and phase not in finished:
                result = readonly_step(task_id, config, phase, request_images)
                if not result["ok"]:
                    return _block(task_id, result["error"], phase)
    mode, text, images = ENTRY_CODE_MODES[entry], "", request_images
    if entry == "retry":
        text = "retry requested after a block"
    if entry == "rework":
        assert feedback is not None
        text, images = feedback["text"], feedback["images"]
    review = "review" in settings.phases and entry != "rework"
    for round_number in range(1, settings.max_rounds + 1):
        coded = code_step(task_id, config, round_number, mode, text, images)
        if coded["outcome"] == "fail":
            return _block(task_id, coded["text"], "code")
        images = []
        problem = None
        if coded["outcome"] == "breach":
            problem = _RoundProblem("breach", coded["text"], "code")
        if problem is None:
            hooked = hooks_step(task_id, config, round_number)
            if not hooked["ok"]:
                problem = _RoundProblem("hooks_fail", hooked["error"], "hooks")
        if problem is None:
            gated = gates_step(task_id, config, round_number)
            if not gated["ok"]:
                problem = _RoundProblem("gate_fail", gated["error"], "gates")
        if problem is None and review:
            reviewed = review_step(task_id, config, round_number)
            if reviewed["outcome"] == "fail":
                return _block(task_id, reviewed["text"], "review")
            if reviewed["outcome"] == "changes":
                problem = _RoundProblem("review_findings", reviewed["text"], "review")
        if problem is None:
            break
        mode, text = problem.mode, problem.text
        if round_number == settings.max_rounds:
            return _block(task_id, f"no passing round after {settings.max_rounds} round(s); last problem:\n{text}", problem.phase)
    if settings.preview_enabled:
        preview_step(task_id, config)
    if settings.merge == "auto" and entry != "rework":
        return "merge"
    state_step(task_id, "human_review", None, None)
    return "human_review"


@DBOS.workflow()
def merge_flow(task_id: str, config: dict[str, Any]) -> str:
    """Commit, rebase, gate, and fast-forward the task branch into main; return done or blocked."""
    try:
        return _merge(task_id, config)
    except Exception as error:
        LOGGER.exception("merge workflow of %s crashed", task_id)
        return _block(task_id, f"unexpected error: {type(error).__name__}: {error}", "merge")


def _merge(task_id: str, config: dict[str, Any]) -> str:
    state_step(task_id, "merging", "merge", None)
    for attempt in range(1, MERGE_ATTEMPTS + 1):
        prepared = merge_prepare_step(task_id, config)
        if not prepared["ok"]:
            return _block(task_id, prepared["error"], "merge")
        if prepared["noop"]:
            break
        gated = gates_step(task_id, config, MERGE_GATES_ROUND)
        if not gated["ok"]:
            merge_abort_step(task_id, config)
            return _block(task_id, f"gates failed after rebasing onto main:\n{gated['error']}", "merge")
        forwarded = fast_forward_step(task_id, config)
        if forwarded["ok"]:
            break
        merge_abort_step(task_id, config)
        if not forwarded["retry"] or attempt == MERGE_ATTEMPTS:
            return _block(task_id, forwarded["error"], "merge")
    finalize_step(task_id)
    return "done"


def _queue(name: str) -> Queue:
    queue = DBOS.retrieve_queue(name)
    if queue is None:
        raise RuntimeError(f"DBOS queue {name!r} is not registered")
    return queue


def _run_child(queue_name: str, task_id: str, blocked_phase: str | None, child: Any, *args: Any) -> str:
    """Run a child workflow on a queue and return its outcome, blocking the task if the child itself fails."""
    try:
        return _queue(queue_name).enqueue(child, task_id, *args).get_result()
    except Exception as error:
        LOGGER.exception("child workflow %s of %s failed", child.__name__, task_id)
        return _block(task_id, f"workflow failure: {type(error).__name__}: {error}", blocked_phase)


def _wait_command() -> dict[str, Any]:
    while True:
        command = DBOS.recv(COMMAND_TOPIC, timeout_seconds=COMMAND_WAIT_SECONDS)
        if command is not None:
            return command


@DBOS.workflow()
def task_flow(task_id: str, config: dict[str, Any], entry: str = "fresh") -> str:
    """Own one task's lifecycle: queue the work, wait for the human, queue the merge.

    `entry` is `fresh` for a new task; a workflow restarted after the previous one ended resumes with `retry` or `merge`.
    """
    feedback = None
    while True:
        merge_blocked = False
        outcome = "merge" if entry == "merge" else _run_child(TASKS_QUEUE, task_id, None, work, config, entry, feedback)
        while True:
            if outcome == "merge":
                outcome = _run_child(MERGE_QUEUE, task_id, "merge", merge_flow, config)
                if outcome == "done":
                    return "done"
                merge_blocked = True
            command = _wait_command()
            kind = command.get("kind")
            if kind == "message":
                entry, feedback, merge_blocked = "rework", command, False
                break
            if kind == "retry" and merge_blocked:
                outcome = "merge"
            elif kind == "retry":
                entry, feedback = "retry", None
                break
            elif kind == "approve":
                outcome = "merge"
            else:
                LOGGER.warning("task %s ignored command %r", task_id, command)


# --- DBOS lifecycle ---


class DbosDispatcher:
    """Drive task workflows through DBOS; a task's first workflow id is its task id, restarts append `.r<n>`."""

    @staticmethod
    def _flows(task_id: str) -> list[Any]:
        """List the task's top-level workflows, oldest first."""
        statuses = DBOS.list_workflows(workflow_id_prefix=task_id, has_parent=False, load_input=False, load_output=False)
        flows = [status for status in statuses if status.workflow_id == task_id or status.workflow_id.startswith(f"{task_id}.r")]
        return sorted(flows, key=lambda status: status.created_at or 0)

    def _current_flow_id(self, task_id: str) -> str:
        flows = self._flows(task_id)
        return flows[-1].workflow_id if flows else task_id

    def start(self, task_id: str, config: dict[str, Any]) -> None:
        with SetWorkflowID(task_id):
            DBOS.start_workflow(task_flow, task_id, config)

    def send(self, task_id: str, payload: dict[str, Any]) -> None:
        DBOS.send(self._current_flow_id(task_id), payload, COMMAND_TOPIC)

    def retry(self, task_id: str, config: dict[str, Any], entry: str) -> None:
        flows = self._flows(task_id)
        if flows and flows[-1].status in ACTIVE_FLOW_STATUSES:
            DBOS.send(flows[-1].workflow_id, {"kind": "retry"}, COMMAND_TOPIC)
            return
        with SetWorkflowID(f"{task_id}.r{len(flows) + 1}"):
            DBOS.start_workflow(task_flow, task_id, config, entry)

    def cancel(self, task_id: str) -> None:
        DBOS.cancel_workflow(self._current_flow_id(task_id), cancel_children=True)


def launch_dbos(database_url: str, in_progress_limit: int, otel_endpoint: str | None = None) -> None:
    """Launch DBOS against the engine database, register the queues, and recover interrupted workflows."""
    config: DBOSConfig = {"name": APP_NAME, "system_database_url": database_url, "application_version": APP_VERSION}
    if otel_endpoint:
        config.update({"enable_otlp": True, "otlp_traces_endpoints": [otel_endpoint]})
    DBOS(config=config)
    DBOS.launch()
    DBOS.register_queue(TASKS_QUEUE, global_concurrency=in_progress_limit)
    DBOS.register_queue(MERGE_QUEUE, global_concurrency=1)


def shutdown_dbos() -> None:
    """Stop DBOS and its workflow threads."""
    DBOS.destroy()
