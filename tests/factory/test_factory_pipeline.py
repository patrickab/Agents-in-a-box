"""Test the work and merge pipelines with a scripted fake omp, real git, and the real store (DBOS for the human-loop flows)."""

from collections.abc import Iterator
import json
from pathlib import Path
import secrets
import stat
import sys
import textwrap
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dbos import DBOS
from factory_helpers import FakePreviews, RecordingDispatcher, git, write
import psycopg
import pytest

from agent_sandbox.factory import git_ops, workflows
from agent_sandbox.factory.config import EngineConfig, TaskConfig, build_task_config
from agent_sandbox.factory.engine import Engine
from agent_sandbox.factory.models import MessageCreate, Overrides, RepoCreate, TaskCreate
from agent_sandbox.factory.store import Store

FAKE_OMP = """\
#!{python}
import json, os, pathlib, sys

args = sys.argv[1:]
script = pathlib.Path(os.environ["FAKE_OMP_SCRIPT"])
steps = json.loads(script.read_text())
step = steps.pop(0)
script.write_text(json.dumps(steps))
session = pathlib.Path(args[args.index("--session-dir") + 1])
(session / "session.jsonl").touch()
phase = session.name
with open(os.environ["FAKE_OMP_LOG"], "a") as log:
    log.write(json.dumps({{"phase": phase, "prompt": args[-1], "continue": "--continue" in args}}) + "\\n")
for relative, content in step.get("edit", {{}}).items():
    path = pathlib.Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
envelope = step.get("envelope", {{"status": "ok", "summary": f"{{phase}} done", "changed_files": [], "question": None}})
if envelope is not None:
    out = pathlib.Path(".factory/out"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{{phase}}.json").write_text(json.dumps(envelope))
"""

TASK_ID = "task00000001"


class Harness:
    """Bundle the engine, repo, and scripting helpers for one test."""

    def __init__(self, engine: Engine, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_name: str = "demo") -> None:
        self.engine, self.repo, self.tmp_path, self.repo_name = engine, repo, tmp_path, repo_name
        self.script = tmp_path / "omp-script.json"
        self.log = tmp_path / "omp-log.jsonl"
        self.monkeypatch = monkeypatch

    def script_omp(self, *steps: dict[str, Any]) -> None:
        self.script.write_text(json.dumps(list(steps)))
        self.log.write_text("")

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def create(self, workflow: str = "code", **overrides: Any) -> tuple[dict[str, Any], TaskConfig]:
        task_config = build_task_config(
            self.engine.config, self.repo_name, self.repo, workflow, Overrides(**overrides), git_ops.checked_out_branch(self.repo)
        )
        dumped = task_config.model_dump(mode="json")
        self.engine.store.create_task(
            task_id=TASK_ID,
            repo=self.repo_name,
            workflow=workflow,
            title="Add feature",
            request="make it so",
            state="todo",
            config=dumped,
        )
        return dumped, task_config

    def task(self) -> Any:
        return self.engine.store.get_task(TASK_ID)

    def worktree(self) -> Path:
        return Path(self.task().worktree)

    def run_work(self, config: dict[str, Any], entry: str = "fresh", feedback: dict[str, Any] | None = None) -> str:
        return workflows._work(TASK_ID, config, entry, feedback)


@pytest.fixture
def harness(repo: Path, tmp_path: Path, database_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    omp = tmp_path / "bin/omp"
    omp.parent.mkdir()
    omp.write_text(FAKE_OMP.format(python=sys.executable))
    omp.chmod(omp.stat().st_mode | stat.S_IEXEC)
    write(repo / ".factory/workflows/review.yaml", "name: Code and review\nphases: [code, review]\nmax_rounds: 3\npreview: false\n")
    write(repo / ".factory/workflows/full.yaml", "name: Full\nphases: [scout, plan, code]\npreview: false\n")
    write(repo / ".factory/workflows/auto.yaml", "name: Auto\nphases: [code]\nmerge: auto\npreview: false\n")
    for name, phase in (("scout", "scout"), ("planner", "plan"), ("reviewer", "review")):
        write(repo / f".omp/agents/{name}.md", f"---\nname: {name}\ndescription: {phase}\n---\nYou are the {phase} agent.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "more templates")
    config = EngineConfig(
        database_url=database_url,
        state_dir=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        defaults_dir=tmp_path / "defaults",
        omp_binary=str(omp),
    )
    store = Store(database_url)
    store.register_repo(str(repo), "demo")
    engine = Engine(config, store, FakePreviews(), RecordingDispatcher())
    workflows.bind_engine(engine)
    monkeypatch.setenv("FAKE_OMP_SCRIPT", str(tmp_path / "omp-script.json"))
    monkeypatch.setenv("FAKE_OMP_LOG", str(tmp_path / "omp-log.jsonl"))
    yield Harness(engine, repo, tmp_path, monkeypatch)
    workflows.bind_engine(None)


def set_gates(repo: Path, *gates: list[str]) -> None:
    text = (repo / "factory.yaml").read_text().split("gates:")[0]
    write(repo / "factory.yaml", text + "gates:\n" + "".join(f"  - {json.dumps(gate)}\n" for gate in gates))


def test_happy_path_ends_in_human_review_with_uncommitted_changes(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp({"edit": {"feature.py": "ok = True\n"}})

    assert harness.run_work(config) == "human_review"

    task = harness.task()
    assert (task.state, task.error) == ("human_review", None)
    assert [(run.phase, run.round, run.status) for run in harness.engine.store.list_phases(TASK_ID)] == [
        ("prepare", 0, "ok"),
        ("code", 1, "ok"),
        ("gates", 1, "ok"),
    ]
    assert (harness.worktree() / "feature.py").read_text() == "ok = True\n"
    assert git(harness.worktree(), "status", "--porcelain").split() == ["??", "feature.py"]
    assert git(harness.worktree(), "rev-parse", "HEAD") == git(harness.repo, "rev-parse", "main")  # nothing committed yet
    assert [m.text for m in harness.engine.store.list_messages(TASK_ID)] == ["code done"]
    assert harness.calls()[0]["continue"] is False and "make it so" in harness.calls()[0]["prompt"]


def test_zero_config_repo_runs_scout_plan_code_review_and_merges_with_packaged_defaults(harness: Harness, bare_repo: Path) -> None:
    name = harness.engine.register_repo(RepoCreate(path=str(bare_repo))).name
    zero_config = Harness(harness.engine, bare_repo, harness.tmp_path, harness.monkeypatch, repo_name=name)
    config, task_config = zero_config.create("scout-plan-code-review", sandbox=False)
    zero_config.script_omp(
        {"envelope": {"status": "ok", "summary": "hello.txt holds the greeting", "relevant_files": ["hello.txt"], "notes": ""}},
        {"envelope": {"status": "ok", "summary": "one step", "steps": ["edit hello.txt"]}},
        {"edit": {"hello.txt": "hello world\n"}},
        {"envelope": {"status": "ok", "approved": True, "findings": []}},
    )

    assert (task_config.main_branch, task_config.profile, task_config.merge, task_config.max_rounds) == (
        "main",
        "factory-default",
        "review",
        3,
    )
    assert task_config.worktree_root == str(harness.engine.config.worktree_root / name) and task_config.gates == []
    assert zero_config.run_work(config) == "human_review"

    assert [call["phase"] for call in zero_config.calls()] == ["scout", "plan", "code", "review"]
    assert zero_config.worktree() == harness.engine.config.worktree_root / name / TASK_ID
    assert workflows._merge(TASK_ID, config) == "done"
    assert (bare_repo / "hello.txt").read_text() == "hello world\n"
    assert zero_config.task().state == "done"


def test_protected_write_is_reverted_and_the_agent_is_told(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp({"edit": {"factory.yaml": "gates: []\n", "feature.py": "v = 1\n"}}, {"edit": {"feature.py": "v = 2\n"}})

    assert harness.run_work(config) == "human_review"

    assert "gates: []" not in (harness.worktree() / "factory.yaml").read_text()
    assert (harness.worktree() / "feature.py").read_text() == "v = 2\n"
    second = harness.calls()[1]
    assert second["continue"] is True and "factory.yaml" in second["prompt"] and "reverted" in second["prompt"]
    assert [(run.phase, run.round, run.status) for run in harness.engine.store.list_phases(TASK_ID) if run.phase == "code"] == [
        ("code", 1, "failed"),
        ("code", 2, "ok"),
    ]


def test_writes_under_factory_out_never_count_as_breaches(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp(
        {"edit": {"feature.py": "v = 1\n", ".factory/out/scenario.py": "print('{}')\n", ".factory/out/notes.txt": "x\n"}}
    )

    assert harness.run_work(config) == "human_review"

    assert len(harness.calls()) == 1
    assert [(run.phase, run.round, run.status) for run in harness.engine.store.list_phases(TASK_ID) if run.phase == "code"] == [
        ("code", 1, "ok")
    ]
    assert (harness.worktree() / ".factory/out/scenario.py").is_file() and (harness.worktree() / ".factory/out/notes.txt").is_file()


def test_gate_failure_feeds_output_back_and_counts_as_a_round(harness: Harness) -> None:
    marker = harness.tmp_path / "gate-marker"
    gate = [
        sys.executable,
        "-c",
        f"import pathlib,sys; p=pathlib.Path({str(marker)!r}); ok=p.exists(); p.touch(); "
        "print('GATE-OUTPUT'); sys.exit(0 if ok else 1)",
    ]
    set_gates(harness.repo, gate)
    git(harness.repo, "commit", "-qam", "gate")
    config, _ = harness.create()
    harness.script_omp({"edit": {"a.py": "1\n"}}, {"edit": {"a.py": "2\n"}})

    assert harness.run_work(config) == "human_review"

    assert "GATE-OUTPUT" in harness.calls()[1]["prompt"] and "gate failed" in harness.calls()[1]["prompt"]
    gates = [run for run in harness.engine.store.list_phases(TASK_ID) if run.phase == "gates"]
    assert [(run.round, run.status, run.gate_results[0].exit_code) for run in gates] == [(1, "failed", 1), (2, "ok", 0)]
    assert harness.task().round == 2


def test_exhausted_rounds_block_the_task(harness: Harness) -> None:
    set_gates(harness.repo, [sys.executable, "-c", "print('always red'); raise SystemExit(1)"])
    git(harness.repo, "commit", "-qam", "red gate")
    config, _ = harness.create(max_rounds=2)
    harness.script_omp({}, {})

    assert harness.run_work(config) == "blocked"

    task = harness.task()
    assert (
        task.state == "blocked" and task.phase == "gates" and "no passing round after 2" in task.error and "always red" in task.error
    )
    assert len(harness.calls()) == 2


def test_agent_failure_with_a_question_blocks_with_the_question(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp(
        {"envelope": {"status": "fail", "summary": "stuck", "changed_files": [], "question": "Which database should I use?"}}
    )

    assert harness.run_work(config) == "blocked"
    assert harness.task().error == "Which database should I use?" and harness.task().phase == "code"


def test_malformed_envelope_is_corrected_once_then_blocks(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp({"envelope": None}, {"envelope": None})

    assert harness.run_work(config) == "blocked"
    assert "envelope" in harness.task().error
    assert [call["continue"] for call in harness.calls()] == [False, True]


def test_malformed_envelope_recovers_after_the_correction_turn(harness: Harness) -> None:
    config, _ = harness.create()
    harness.script_omp({"envelope": None}, {})

    assert harness.run_work(config) == "human_review"
    assert "could not be used" in harness.calls()[1]["prompt"]


def test_review_findings_go_back_to_the_coder_until_approved(harness: Harness) -> None:
    config, _ = harness.create("review")
    approved = {"status": "ok", "approved": True, "findings": []}
    rejected = {"status": "ok", "approved": False, "findings": ["missing a test"]}
    harness.script_omp(
        {"edit": {"a.py": "1\n"}},
        {"envelope": rejected, "edit": {"sneaky.py": "reviewers must not write\n"}},
        {"edit": {"a.py": "2\n"}},
        {"envelope": approved},
    )

    assert harness.run_work(config) == "human_review"

    calls = harness.calls()
    assert [call["phase"] for call in calls] == ["code", "review", "code", "review"]
    assert "missing a test" in calls[2]["prompt"] and "diff" in calls[1]["prompt"] and "a.py" in calls[1]["prompt"]
    assert not (harness.worktree() / "sneaky.py").exists()  # read-only phases cannot write
    assert (harness.worktree() / "a.py").read_text() == "2\n"


def test_scout_and_plan_feed_the_coder_and_cannot_write(harness: Harness) -> None:
    config, _ = harness.create("full")
    harness.script_omp(
        {
            "envelope": {"status": "ok", "summary": "auth lives in auth.py", "relevant_files": ["auth.py"], "notes": ""},
            "edit": {"leak.txt": "x"},
        },
        {"envelope": {"status": "ok", "summary": "two steps", "steps": ["edit auth.py", "add test"]}},
        {"edit": {"auth.py": "new\n"}},
    )

    assert harness.run_work(config) == "human_review"

    scout, plan, code = harness.calls()
    assert (scout["phase"], plan["phase"], code["phase"]) == ("scout", "plan", "code")
    assert "auth lives in auth.py" in plan["prompt"]
    assert "edit auth.py" in code["prompt"] and "auth lives in auth.py" in code["prompt"]
    assert not (harness.worktree() / "leak.txt").exists()


def test_retry_skips_finished_scout_and_plan_and_continues_the_coder_session(harness: Harness) -> None:
    config, _ = harness.create("full")
    scout = {"status": "ok", "summary": "s", "relevant_files": [], "notes": ""}
    plan = {"status": "ok", "summary": "p", "steps": ["one"]}
    harness.script_omp(
        {"envelope": scout},
        {"envelope": plan},
        {"envelope": {"status": "fail", "summary": "no", "changed_files": [], "question": None}},
    )
    assert harness.run_work(config) == "blocked"

    harness.script_omp({"edit": {"done.py": "1\n"}})
    assert harness.run_work(config, entry="retry") == "human_review"

    (call,) = harness.calls()
    assert call["phase"] == "code" and call["continue"] is True and "retry requested" in call["prompt"]


def test_rework_runs_the_coder_with_feedback_and_skips_review(harness: Harness) -> None:
    config, _ = harness.create("review")
    harness.script_omp({"edit": {"a.py": "1\n"}}, {"envelope": {"status": "ok", "approved": True, "findings": []}})
    harness.run_work(config)
    harness.engine.store.update_task(TASK_ID, state="rework")

    harness.script_omp({"edit": {"a.py": "2\n"}})
    assert harness.run_work(config, entry="rework", feedback={"text": "rename it please", "images": []}) == "human_review"

    (call,) = harness.calls()
    assert call["phase"] == "code" and "rename it please" in call["prompt"]
    assert harness.task().state == "human_review"


def test_auto_merge_workflows_skip_the_review_stop(harness: Harness) -> None:
    config, _ = harness.create("auto")
    harness.script_omp({"edit": {"a.py": "1\n"}})
    assert harness.run_work(config) == "merge"
    assert harness.task().state == "in_progress"


def approved_work(harness: Harness, files: dict[str, str]) -> dict[str, Any]:
    config, _ = harness.create()
    harness.script_omp({"edit": files})
    assert harness.run_work(config) == "human_review"
    return config


def test_merge_commits_rebases_and_fast_forwards_then_removes_the_worktree(harness: Harness) -> None:
    config = approved_work(harness, {"feature.py": "ok = True\n"})
    worktree = harness.worktree()
    write(worktree / ".factory/workflows/rogue.yaml", "name: rogue\nphases: [code]\n")
    write(harness.repo / "other.txt", "main moved meanwhile\n")
    git(harness.repo, "add", "-A")
    git(harness.repo, "commit", "-qm", "main moves")

    assert workflows._merge(TASK_ID, config) == "done"

    assert harness.task().state == "done"
    assert (harness.repo / "feature.py").read_text() == "ok = True\n" and (harness.repo / "other.txt").is_file()
    assert not (harness.repo / ".factory/workflows/rogue.yaml").exists()
    assert git(harness.repo, "log", "-1", "--format=%s").strip() == "factory: Add feature"
    assert not worktree.exists() and "factory/" not in git(harness.repo, "branch", "--list")
    assert harness.engine.previews.stopped == [TASK_ID]


def test_merge_with_unrelated_local_edits_in_the_main_checkout_succeeds_and_keeps_them(harness: Harness) -> None:
    config = approved_work(harness, {"feature.py": "ok = True\n"})
    write(harness.repo / "hello.txt", "my local edit\n")

    assert workflows._merge(TASK_ID, config) == "done"

    assert (harness.repo / "feature.py").is_file() and (harness.repo / "hello.txt").read_text() == "my local edit\n"


def test_merge_blocks_with_gits_error_when_local_edits_overlap_and_retries_after_they_are_gone(harness: Harness) -> None:
    config = approved_work(harness, {"hello.txt": "task version\n"})
    write(harness.repo / "hello.txt", "my local edit\n")
    head_before = git(harness.repo, "rev-parse", "HEAD")

    assert workflows._merge(TASK_ID, config) == "blocked"

    task = harness.task()
    assert task.state == "blocked" and task.phase == "merge" and "would be overwritten" in task.error
    assert git(harness.repo, "rev-parse", "HEAD") == head_before
    assert harness.worktree().is_dir() and git(harness.worktree(), "status", "--porcelain").split() == ["M", "hello.txt"]

    git(harness.repo, "checkout", "-q", "--", "hello.txt")
    assert workflows._merge(TASK_ID, config) == "done"
    assert (harness.repo / "hello.txt").read_text() == "task version\n"


def test_merge_gate_failure_after_rebase_blocks_without_merging(harness: Harness) -> None:
    config = approved_work(harness, {"feature.py": "ok = True\n"})
    # A gate that passes in the worktree before rebasing (feature.py only) but fails once main gains `poison.txt`.
    gate = [sys.executable, "-c", "import pathlib,sys; sys.exit(1 if pathlib.Path('poison.txt').exists() else 0)"]
    blocked_config = json.loads(json.dumps(config))
    blocked_config["gates"] = [gate]
    write(harness.repo / "poison.txt", "x")
    git(harness.repo, "add", "-A")
    git(harness.repo, "commit", "-qm", "poison lands on main")

    assert workflows._merge(TASK_ID, blocked_config) == "blocked"

    assert "gates failed after rebasing" in harness.task().error
    assert not (harness.repo / "feature.py").exists()
    assert git(harness.worktree(), "status", "--porcelain").strip() == "?? feature.py"


def test_merge_without_changes_is_a_noop_that_finishes_the_task(harness: Harness) -> None:
    config = approved_work(harness, {})
    head_before = git(harness.repo, "rev-parse", "HEAD")

    assert workflows._merge(TASK_ID, config) == "done"
    assert git(harness.repo, "rev-parse", "HEAD") == head_before and harness.task().state == "done"


def test_rebase_conflict_blocks_and_restores_the_uncommitted_diff(harness: Harness) -> None:
    config = approved_work(harness, {"hello.txt": "task version\n"})
    write(harness.repo / "hello.txt", "main version\n")
    git(harness.repo, "commit", "-qam", "main edits hello")

    assert workflows._merge(TASK_ID, config) == "blocked"
    assert "rebase onto main failed" in harness.task().error
    assert (harness.worktree() / "hello.txt").read_text() == "task version\n"
    assert git(harness.worktree(), "status", "--porcelain").strip() == "M hello.txt"
    assert textwrap.dedent(git(harness.repo, "show", "HEAD:hello.txt")) == "main version\n"


@pytest.fixture
def dbos_runtime(database_url: str) -> Iterator[None]:
    """Launch DBOS against a throwaway system database next to the test database."""
    name = f"factory_dbos_{secrets.token_hex(4)}"
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    workflows.launch_dbos(urlunsplit(urlsplit(database_url)._replace(path=f"/{name}")), 4)
    yield
    for flow in DBOS.list_workflows(status=["PENDING", "ENQUEUED"], load_input=False, load_output=False):
        DBOS.cancel_workflow(flow.workflow_id)  # A flow parked on `recv` would otherwise keep its thread retrying a dropped database.
    DBOS.destroy(workflow_completion_timeout_sec=20)  # Let the cancelled flows exit before their database disappears.
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def wait_for_state(store: Store, task_id: str, state: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while store.get_task(task_id).state != state:
        assert time.monotonic() < deadline, f"task stuck in {store.get_task(task_id).state}, wanted {state}"
        time.sleep(0.2)


def test_retry_restarts_a_task_workflow_that_ended_and_resumes_the_coder_session(harness: Harness, dbos_runtime: None) -> None:
    harness.engine.dispatcher = workflows.DbosDispatcher()
    harness.script_omp({"edit": {"feature.py": "v = 1\n"}}, {"edit": {"feature.py": "v = 2\n"}})
    store = harness.engine.store

    task_id = harness.engine.create_task(TaskCreate(repo="demo", workflow="code", title="Add feature", request="make it so")).id
    wait_for_state(store, task_id, "human_review")
    DBOS.cancel_workflow(task_id, cancel_children=True)  # The task's workflow ends without the task knowing.
    assert store.transition(task_id, ("human_review",), "blocked", error="workflow ended") is not None

    assert harness.engine.retry(task_id).state == "todo"
    wait_for_state(store, task_id, "human_review")

    flows = harness.engine.dispatcher._flows(task_id)
    assert [(flow.workflow_id, flow.status) for flow in flows] == [(task_id, "CANCELLED"), (f"{task_id}.r2", "PENDING")]
    second = harness.calls()[1]
    assert second["phase"] == "code" and second["continue"] is True and "retry requested" in second["prompt"]
    assert (Path(store.get_task(task_id).worktree) / "feature.py").read_text() == "v = 2\n"

    assert harness.engine.dispatcher._current_flow_id(task_id) == f"{task_id}.r2"  # Later commands reach the restarted workflow.


def run_to_human_review(harness: Harness, *steps: dict[str, Any]) -> str:
    """Create a code task on the DBOS dispatcher and wait until it needs the human."""
    harness.engine.dispatcher = workflows.DbosDispatcher()
    harness.script_omp(*steps)
    task_id = harness.engine.create_task(TaskCreate(repo="demo", workflow="code", title="Add feature", request="make it so")).id
    wait_for_state(harness.engine.store, task_id, "human_review")
    return task_id


def block_merge_on_local_edit(harness: Harness, task_id: str) -> None:
    """Approve a task whose change overlaps an uncommitted edit in the main checkout, and wait for the merge block."""
    write(harness.repo / "hello.txt", "my local edit\n")
    harness.engine.approve(task_id)
    wait_for_state(harness.engine.store, task_id, "blocked")
    task = harness.engine.store.get_task(task_id)
    assert task.phase == "merge" and "would be overwritten" in task.error


def test_message_in_human_review_reworks_the_code_and_returns_to_human_review(harness: Harness, dbos_runtime: None) -> None:
    task_id = run_to_human_review(harness, {"edit": {"feature.py": "v = 1\n"}}, {"edit": {"feature.py": "v = 2\n"}})
    store = harness.engine.store

    harness.engine.post_message(task_id, MessageCreate(text="rename it please"))
    assert store.get_task(task_id).state == "rework"
    wait_for_state(store, task_id, "human_review")

    second = harness.calls()[1]
    assert second["phase"] == "code" and second["continue"] is True and "rename it please" in second["prompt"]
    assert (Path(store.get_task(task_id).worktree) / "feature.py").read_text() == "v = 2\n"
    assert [flow.workflow_id for flow in harness.engine.dispatcher._flows(task_id)] == [task_id]


def test_approve_merges_the_branch_and_removes_the_worktree(harness: Harness, dbos_runtime: None) -> None:
    task_id = run_to_human_review(harness, {"edit": {"feature.py": "v = 1\n"}})
    store = harness.engine.store
    worktree = Path(store.get_task(task_id).worktree)

    assert harness.engine.approve(task_id).state == "merging"
    wait_for_state(store, task_id, "done")

    assert (harness.repo / "feature.py").read_text() == "v = 1\n"
    assert git(harness.repo, "log", "-1", "--format=%s").strip() == "factory: Add feature"
    assert not worktree.exists()


def test_retry_after_a_merge_block_reruns_the_merge_in_the_same_workflow(harness: Harness, dbos_runtime: None) -> None:
    task_id = run_to_human_review(harness, {"edit": {"hello.txt": "task version\n"}})
    store = harness.engine.store
    block_merge_on_local_edit(harness, task_id)

    git(harness.repo, "checkout", "-q", "--", "hello.txt")
    assert harness.engine.retry(task_id).state == "merging"
    wait_for_state(store, task_id, "done")

    assert (harness.repo / "hello.txt").read_text() == "task version\n"
    assert [flow.workflow_id for flow in harness.engine.dispatcher._flows(task_id)] == [task_id]
    assert len(harness.calls()) == 1  # The retry re-ran the merge, not the coder.


def test_message_after_a_merge_block_reworks_and_the_task_then_merges(harness: Harness, dbos_runtime: None) -> None:
    task_id = run_to_human_review(harness, {"edit": {"hello.txt": "task version\n"}}, {"edit": {"hello.txt": "reworked version\n"}})
    store = harness.engine.store
    block_merge_on_local_edit(harness, task_id)

    git(harness.repo, "checkout", "-q", "--", "hello.txt")
    harness.engine.post_message(task_id, MessageCreate(text="use another wording"))
    assert store.get_task(task_id).state == "rework"
    wait_for_state(store, task_id, "human_review")
    assert "use another wording" in harness.calls()[1]["prompt"]

    harness.engine.approve(task_id)
    wait_for_state(store, task_id, "done")
    assert (harness.repo / "hello.txt").read_text() == "reworked version\n"
