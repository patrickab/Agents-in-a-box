"""Persist repos, tasks, phase runs, messages, and events in the factory Postgres database."""

from collections.abc import Collection, Iterator
from contextlib import contextmanager
import itertools
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_sandbox.factory.models import TERMINAL_STATES, Event, GateResult, Message, PhaseRun, RepoInfo, Task

SCHEMA_LOCK_KEY = 7_424_001
EVENT_LOCK_KEY = 7_424_002
REPO_LOCK_KEY = 7_424_003
EVENTS_PAGE_LIMIT = 500

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id text PRIMARY KEY,
        repo text NOT NULL,
        workflow text NOT NULL,
        title text NOT NULL,
        request text NOT NULL,
        state text NOT NULL,
        phase text,
        round integer NOT NULL DEFAULT 0,
        config jsonb NOT NULL,
        worktree text,
        branch text,
        base_sha text,
        error text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS tasks_repo_state_idx ON tasks (repo, state)",
    """
    CREATE TABLE IF NOT EXISTS phase_runs (
        id bigserial PRIMARY KEY,
        task_id text NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
        phase text NOT NULL,
        round integer NOT NULL DEFAULT 0,
        status text NOT NULL DEFAULT 'running',
        started_at timestamptz NOT NULL DEFAULT now(),
        ended_at timestamptz,
        summary text NOT NULL DEFAULT '',
        envelope jsonb,
        gate_results jsonb NOT NULL DEFAULT '[]'
    )
    """,
    "CREATE INDEX IF NOT EXISTS phase_runs_task_idx ON phase_runs (task_id, id)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        id bigserial PRIMARY KEY,
        task_id text NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
        role text NOT NULL,
        text text NOT NULL,
        phase_run_id bigint,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS messages_task_idx ON messages (task_id, id)",
    """
    CREATE TABLE IF NOT EXISTS events (
        seq bigserial PRIMARY KEY,
        task_id text NOT NULL,
        repo text NOT NULL,
        kind text NOT NULL,
        data jsonb NOT NULL DEFAULT '{}',
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_repo_seq_idx ON events (repo, seq)",
    """
    CREATE TABLE IF NOT EXISTS repos (
        name text PRIMARY KEY,
        path text NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
)

UPDATABLE_TASK_COLUMNS = frozenset({"phase", "round", "worktree", "branch", "base_sha", "error"})


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    agents = {
        phase: {key: value for key, value in agent.items() if key != "prompt"} for phase, agent in config.get("agents", {}).items()
    }
    return {**config, "agents": agents}


def _name_candidates(base_name: str) -> Iterator[str]:
    yield base_name
    yield from (f"{base_name}-{number}" for number in itertools.count(2))


def _task_from_row(row: dict[str, Any]) -> Task:
    return Task(**{**row, "config": _public_config(row["config"])})


def _phase_from_row(row: dict[str, Any]) -> PhaseRun:
    return PhaseRun(**{**row, "gate_results": [GateResult(**gate) for gate in row["gate_results"]]})


class Store:
    """Read and write engine tables through short-lived psycopg connections."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextmanager
    def _cursor(self) -> Iterator[psycopg.Cursor[dict[str, Any]]]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection, connection.cursor() as cursor:
            yield cursor

    def ensure_schema(self) -> None:
        """Create the engine tables if they do not exist; safe to call concurrently and repeatedly."""
        with self._cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)

    @staticmethod
    def _emit(cursor: psycopg.Cursor[dict[str, Any]], task_id: str, kind: str, data: dict[str, Any]) -> None:
        """Insert an event under a transaction lock so seq order matches commit order and pollers never skip an event."""
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (EVENT_LOCK_KEY,))
        cursor.execute(
            "INSERT INTO events (task_id, repo, kind, data) SELECT id, repo, %s, %s FROM tasks WHERE id = %s",
            (kind, Jsonb(data), task_id),
        )

    # --- repos ---

    def list_repos(self) -> list[RepoInfo]:
        """List the registered repos by name."""
        with self._cursor() as cursor:
            cursor.execute("SELECT name, path FROM repos ORDER BY name")
            return [RepoInfo(**row) for row in cursor.fetchall()]

    def get_repo(self, name: str) -> RepoInfo | None:
        """Return one registered repo, or None when the name is unknown."""
        with self._cursor() as cursor:
            cursor.execute("SELECT name, path FROM repos WHERE name = %s", (name,))
            row = cursor.fetchone()
        return None if row is None else RepoInfo(**row)

    def register_repo(self, path: str, base_name: str) -> RepoInfo:
        """Register a checkout path once, naming it `base_name` or the first free `base_name-N` (N >= 2)."""
        with self._cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (REPO_LOCK_KEY,))
            cursor.execute("SELECT name, path FROM repos WHERE path = %s", (path,))
            existing = cursor.fetchone()
            if existing is not None:
                return RepoInfo(**existing)
            cursor.execute("SELECT name FROM repos")
            taken = {row["name"] for row in cursor.fetchall()}
            name = next(candidate for candidate in _name_candidates(base_name) if candidate not in taken)
            cursor.execute("INSERT INTO repos (name, path) VALUES (%s, %s)", (name, path))
        return RepoInfo(name=name, path=path)

    # --- tasks ---

    def create_task(
        self, *, task_id: str, repo: str, workflow: str, title: str, request: str, state: str, config: dict[str, Any]
    ) -> Task:
        """Insert a new task and its creation event."""
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO tasks (id, repo, workflow, title, request, state, config) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (task_id, repo, workflow, title, request, state, Jsonb(config)),
            )
            row = cursor.fetchone()
            self._emit(cursor, task_id, "task.created", {"state": state})
        return _task_from_row(row)

    def get_task(self, task_id: str) -> Task | None:
        """Return one task, or None when it does not exist."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
            row = cursor.fetchone()
        return None if row is None else _task_from_row(row)

    def get_config(self, task_id: str) -> dict[str, Any] | None:
        """Return the full frozen task config including agent prompts."""
        with self._cursor() as cursor:
            cursor.execute("SELECT config FROM tasks WHERE id = %s", (task_id,))
            row = cursor.fetchone()
        return None if row is None else row["config"]

    def list_tasks(self, repo: str | None = None, state: str | None = None) -> list[Task]:
        """List tasks newest first, optionally filtered by repo and state."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM tasks WHERE (%s::text IS NULL OR repo = %s) AND (%s::text IS NULL OR state = %s) "
                "ORDER BY created_at DESC, id",
                (repo, repo, state, state),
            )
            return [_task_from_row(row) for row in cursor.fetchall()]

    def transition(self, task_id: str, allowed: Collection[str], to_state: str, **fields: Any) -> Task | None:
        """Atomically move a task to `to_state` if it is currently in `allowed`; return None otherwise."""
        assignments, values = self._assignments(fields)
        with self._cursor() as cursor:
            cursor.execute(
                f"UPDATE tasks SET state = %s, updated_at = now(){assignments} WHERE id = %s AND state = ANY(%s) RETURNING *",
                (to_state, *values, task_id, list(allowed)),
            )
            row = cursor.fetchone()
            if row is not None:
                self._emit(cursor, task_id, "task.state", {"state": to_state, "phase": row["phase"]})
        return None if row is None else _task_from_row(row)

    def update_task(self, task_id: str, *, state: str | None = None, emit: bool = True, **fields: Any) -> Task | None:
        """Update a task unless it is done or cancelled; return None when nothing changed."""
        assignments, values = self._assignments(fields)
        if state is not None:
            assignments += ", state = %s"
            values.append(state)
        if not assignments:
            return self.get_task(task_id)
        with self._cursor() as cursor:
            cursor.execute(
                f"UPDATE tasks SET updated_at = now(){assignments} WHERE id = %s AND state <> ALL(%s) RETURNING *",
                (*values, task_id, list(TERMINAL_STATES)),
            )
            row = cursor.fetchone()
            if row is not None and emit:
                self._emit(cursor, task_id, "task.state", {"state": row["state"], "phase": row["phase"], "round": row["round"]})
        return None if row is None else _task_from_row(row)

    @staticmethod
    def _assignments(fields: dict[str, Any]) -> tuple[str, list[Any]]:
        unknown = set(fields) - UPDATABLE_TASK_COLUMNS
        if unknown:
            raise ValueError(f"unknown task columns {sorted(unknown)}")
        return "".join(f", {column} = %s" for column in fields), list(fields.values())

    # --- phase runs ---

    def begin_phase(self, task_id: str, phase: str, round_number: int) -> int:
        """Open a phase run, closing any interrupted run of the same phase and round left by a crash."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE phase_runs SET status = 'failed', ended_at = now(), summary = 'interrupted' "
                "WHERE task_id = %s AND phase = %s AND round = %s AND status = 'running'",
                (task_id, phase, round_number),
            )
            cursor.execute(
                "INSERT INTO phase_runs (task_id, phase, round) VALUES (%s, %s, %s) RETURNING id", (task_id, phase, round_number)
            )
            run_id = cursor.fetchone()["id"]
            self._emit(cursor, task_id, "phase.started", {"phase": phase, "round": round_number, "phase_run_id": run_id})
        return run_id

    def finish_phase(
        self,
        run_id: int,
        status: str,
        summary: str,
        envelope: dict[str, Any] | None = None,
        gate_results: list[GateResult] | None = None,
    ) -> None:
        """Close a phase run with its outcome."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE phase_runs SET status = %s, summary = %s, envelope = %s, gate_results = %s, ended_at = now() "
                "WHERE id = %s RETURNING task_id, phase, round",
                (
                    status,
                    summary,
                    None if envelope is None else Jsonb(envelope),
                    Jsonb([gate.model_dump() for gate in gate_results or []]),
                    run_id,
                ),
            )
            row = cursor.fetchone()
            self._emit(
                cursor,
                row["task_id"],
                "phase.finished",
                {"phase": row["phase"], "round": row["round"], "status": status, "phase_run_id": run_id},
            )

    def list_phases(self, task_id: str) -> list[PhaseRun]:
        """List a task's phase runs, oldest first."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM phase_runs WHERE task_id = %s ORDER BY id", (task_id,))
            return [_phase_from_row(row) for row in cursor.fetchall()]

    def completed_phase_envelope(self, task_id: str, phase: str) -> dict[str, Any] | None:
        """Return the envelope of the latest successful run of a phase, if any."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT envelope FROM phase_runs WHERE task_id = %s AND phase = %s AND status = 'ok' ORDER BY id DESC LIMIT 1",
                (task_id, phase),
            )
            row = cursor.fetchone()
        return None if row is None else row["envelope"]

    # --- messages and events ---

    def add_message(self, task_id: str, role: str, text: str, phase_run_id: int | None = None) -> Message:
        """Store a message and emit its event."""
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO messages (task_id, role, text, phase_run_id) VALUES (%s, %s, %s, %s) RETURNING *",
                (task_id, role, text, phase_run_id),
            )
            row = cursor.fetchone()
            self._emit(cursor, task_id, "message", {"message_id": row["id"], "role": role})
        return Message(**row)

    def list_messages(self, task_id: str) -> list[Message]:
        """List a task's messages, oldest first."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM messages WHERE task_id = %s ORDER BY id", (task_id,))
            return [Message(**row) for row in cursor.fetchall()]

    def add_event(self, task_id: str, kind: str, data: dict[str, Any]) -> None:
        """Record a custom event for a task."""
        with self._cursor() as cursor:
            self._emit(cursor, task_id, kind, data)

    def list_events(self, after: int = 0, repo: str | None = None) -> list[Event]:
        """List up to EVENTS_PAGE_LIMIT events after `after`, oldest first, optionally for one repo."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM events WHERE seq > %s AND (%s::text IS NULL OR repo = %s) ORDER BY seq LIMIT %s",
                (after, repo, repo, EVENTS_PAGE_LIMIT),
            )
            return [Event(**row) for row in cursor.fetchall()]

    def latest_event_seq(self) -> int:
        """Return the newest event seq, or 0 when there are no events."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COALESCE(MAX(seq), 0) AS seq FROM events")
            return cursor.fetchone()["seq"]
