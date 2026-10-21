"""Test the engine tables: guarded transitions, phase runs, messages, and events (needs the factory test database)."""

import threading

import psycopg
from psycopg.rows import dict_row
import pytest

from agent_sandbox.factory.models import GateResult
from agent_sandbox.factory.store import Store

CONFIG = {"repo": "demo", "agents": {"code": {"name": "coder", "prompt": "secret prompt"}}}


@pytest.fixture
def store(database_url: str) -> Store:
    return Store(database_url)


def new_task(store: Store, state: str = "todo") -> str:
    return store.create_task(
        task_id="task00000001", repo="demo", workflow="code", title="T", request="R", state=state, config=CONFIG
    ).id


def test_task_config_hides_prompts_publicly_but_keeps_them_for_the_engine(store: Store) -> None:
    task_id = new_task(store)
    assert "prompt" not in store.get_task(task_id).config["agents"]["code"]
    assert store.get_config(task_id)["agents"]["code"]["prompt"] == "secret prompt"


def test_update_task_never_touches_finished_tasks(store: Store) -> None:
    task_id = new_task(store)
    assert store.update_task(task_id, phase="code", round=2).round == 2
    store.transition(task_id, ("todo",), "cancelled")

    assert store.update_task(task_id, state="in_progress") is None
    assert store.update_task(task_id, error="late failure") is None
    assert store.get_task(task_id).state == "cancelled" and store.get_task(task_id).error is None


def test_phase_runs_record_outcomes_and_close_interrupted_runs(store: Store) -> None:
    task_id = new_task(store)
    interrupted = store.begin_phase(task_id, "code", 1)
    retried = store.begin_phase(task_id, "code", 1)
    store.finish_phase(
        retried, "ok", "done", {"status": "ok", "summary": "done"}, [GateResult(argv=["true"], exit_code=0, output_tail="")]
    )

    runs = {run.id: run for run in store.list_phases(task_id)}
    assert (runs[interrupted].status, runs[interrupted].summary) == ("failed", "interrupted")
    assert runs[retried].status == "ok" and runs[retried].ended_at is not None
    assert runs[retried].gate_results[0].argv == ["true"]
    assert store.completed_phase_envelope(task_id, "code") == {"status": "ok", "summary": "done"}
    assert store.completed_phase_envelope(task_id, "scout") is None


def test_event_inserts_wait_for_in_flight_emitters_so_seq_order_is_commit_order(store: Store, database_url: str) -> None:
    task_id = new_task(store)
    writer = threading.Thread(target=store.add_event, args=(task_id, "second", {}))
    with psycopg.connect(database_url, row_factory=dict_row) as connection, connection.cursor() as cursor:
        Store._emit(cursor, task_id, "first", {})  # Holds the event lock until this transaction commits.
        writer.start()
        writer.join(timeout=0.5)
        assert writer.is_alive()
        connection.commit()
    writer.join(timeout=10)

    assert not writer.is_alive()
    assert [event.kind for event in store.list_events()] == ["task.created", "first", "second"]
