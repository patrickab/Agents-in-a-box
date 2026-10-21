"""Per-task preview environments: port allocation, optional database, processes, readiness and scenario."""

from dataclasses import dataclass, field
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from urllib.parse import urlsplit, urlunsplit
import urllib.request

import psycopg
from psycopg import sql

from agent_sandbox.factory.config import OUTPUT_DIR, PreviewConfig, task_dir
from agent_sandbox.factory.models import Preview
from agent_sandbox.factory.processes import terminate_process_groups

LOOPBACK = "127.0.0.1"
INHERITED_ENV_KEYS = ("PATH", "HOME", "USER", "LANG")
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
DEFAULT_LANG = "C.UTF-8"
SCENARIO_RELATIVE_PATH = f"{OUTPUT_DIR}/scenario.py"
SCENARIO_TIMEOUT_S = 120
READY_POLL_INTERVAL_S = 0.25
READY_REQUEST_TIMEOUT_S = 2
LOG_TAIL_BYTES = 1500
PLACEHOLDER = re.compile(r"\{(port\.[^{}\s]+|database_url|data_dir|worktree|task_id)\}")
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class PreviewError(Exception):
    """Report a preview that could not be brought up or torn down."""


@dataclass
class _RunningProcess:
    name: str
    popen: subprocess.Popen


@dataclass
class _Entry:
    task_id: str
    preview: Preview = field(default_factory=lambda: Preview(status="starting", url="", open_url="", error=None))
    ports: dict[str, int] = field(default_factory=dict)
    processes: list[_RunningProcess] = field(default_factory=list)
    database_name: str | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)


class PreviewManager:
    """Start, track, and stop one preview (processes plus optional database) per task."""

    def __init__(self, ports: tuple[int, int], state_dir: Path, admin_database_url: str):
        self._first_port, self._last_port = ports
        self._state_dir = Path(state_dir)
        self._admin_database_url = admin_database_url
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def start(self, task_id: str, worktree: Path, config: PreviewConfig) -> Preview:
        """Start the preview for a task and block until it is ready or failed."""
        self.stop(task_id)
        entry = _Entry(task_id=task_id)
        with self._lock:
            self._entries[task_id] = entry
        try:
            self._launch(entry, Path(worktree), config)
        except (PreviewError, psycopg.Error, OSError) as error:
            cleanup_error = self._teardown_quietly(entry)
            message = str(error) if not entry.cancelled.is_set() else "preview stopped"
            if cleanup_error:
                message = f"{message} (cleanup failed: {cleanup_error})"
            status = "stopped" if entry.cancelled.is_set() else "failed"
            entry.preview = Preview(status=status, url=entry.preview.url, open_url=entry.preview.url, error=message)
        return entry.preview

    def stop(self, task_id: str) -> None:
        """Kill the preview processes and drop its database; safe to call repeatedly."""
        with self._lock:
            entry = self._entries.get(task_id)
        if entry is None:
            return
        entry.cancelled.set()
        self._teardown(entry)
        entry.preview = Preview(status="stopped", url=entry.preview.url, open_url=entry.preview.url, error=None)
        with self._lock:
            self._entries.pop(task_id, None)

    def status(self, task_id: str) -> Preview | None:
        """Return the preview of a task, flipping a ready preview to failed when one of its processes died."""
        with self._lock:
            entry = self._entries.get(task_id)
        if entry is None:
            return None
        with entry.lock:
            if entry.preview.status == "ready":
                dead = next((process for process in entry.processes if process.popen.poll() is not None), None)
                if dead is not None:
                    message = self._exit_message(entry, dead)
                    cleanup_error = self._teardown_quietly(entry)
                    if cleanup_error:
                        message = f"{message} (cleanup failed: {cleanup_error})"
                    entry.preview = Preview(status="failed", url=entry.preview.url, open_url=entry.preview.url, error=message)
            return entry.preview

    def stop_all(self) -> None:
        """Stop every known preview, for engine shutdown."""
        with self._lock:
            task_ids = list(self._entries)
        for task_id in task_ids:
            self.stop(task_id)

    def _launch(self, entry: _Entry, worktree: Path, config: PreviewConfig) -> None:
        directory = task_dir(self._state_dir, entry.task_id)
        data_dir = directory / "preview-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        values = {"worktree": str(worktree), "task_id": entry.task_id, "data_dir": str(data_dir)}

        entry.ports = self._allocate_ports(entry, [process.name for process in config.processes])
        values.update({f"port.{name}": str(port) for name, port in entry.ports.items()})
        if config.database:
            entry.database_name = f"preview_{entry.task_id}"
            self._create_database(entry.database_name)
            values["database_url"] = self._database_url(entry.database_name)

        url = _render(config.url, values)
        entry.preview = Preview(status="starting", url=url, open_url=url, error=None)
        env = {key: os.environ[key] for key in INHERITED_ENV_KEYS if key in os.environ}
        env.setdefault("PATH", DEFAULT_PATH)
        env.setdefault("LANG", DEFAULT_LANG)
        env.update({key: _render(value, values) for key, value in config.env.items()})

        ready_urls: dict[str, str] = {}
        for process in config.processes:
            self._spawn(entry, process.name, [_render(arg, values) for arg in process.argv], worktree, env, directory)
            if process.ready_url:
                ready_urls[process.name] = _render(process.ready_url, values)
        self._await_ready(entry, ready_urls, config.ready_timeout_s)

        api_url = _render(config.api_url, values) if config.api_url else url
        open_url, scenario_error = self._run_scenario(worktree, env, directory, api_url, url)
        if entry.cancelled.is_set():
            raise PreviewError("preview stopped")
        entry.preview = Preview(status="ready", url=url, open_url=open_url, error=scenario_error)

    def _allocate_ports(self, entry: _Entry, names: list[str]) -> dict[str, int]:
        with self._lock:
            taken = {port for other in self._entries.values() for port in other.ports.values()}
            allocated: dict[str, int] = {}
            for name in dict.fromkeys(names):
                port = next((p for p in range(self._first_port, self._last_port + 1) if p not in taken and _is_bindable(p)), None)
                if port is None:
                    raise PreviewError(f"no free preview port for {name} in {self._first_port}-{self._last_port}")
                taken.add(port)
                allocated[name] = port
            return allocated

    def _database_url(self, database_name: str) -> str:
        parts = urlsplit(self._admin_database_url)
        if not parts.scheme.startswith("postgres"):
            raise PreviewError("admin database url must be a postgres URL")
        return urlunsplit(parts._replace(path=f"/{database_name}"))

    def _create_database(self, database_name: str) -> None:
        self._drop_database(database_name)
        with psycopg.connect(self._admin_database_url, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))

    def _drop_database(self, database_name: str) -> None:
        with psycopg.connect(self._admin_database_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database_name)))

    def _spawn(self, entry: _Entry, name: str, argv: list[str], worktree: Path, env: dict[str, str], directory: Path) -> None:
        with entry.lock:
            if entry.cancelled.is_set():
                raise PreviewError("preview stopped")
            with (directory / f"preview-{name}.log").open("wb") as log:
                popen = subprocess.Popen(
                    argv, cwd=worktree, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            entry.processes.append(_RunningProcess(name=name, popen=popen))

    def _await_ready(self, entry: _Entry, ready_urls: dict[str, str], timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        pending = dict(ready_urls)
        while True:
            if entry.cancelled.is_set():
                raise PreviewError("preview stopped")
            for process in entry.processes:
                if process.popen.poll() is not None:
                    raise PreviewError(self._exit_message(entry, process))
            pending = {name: url for name, url in pending.items() if not _responds(url)}
            if not pending:
                return
            if time.monotonic() >= deadline:
                waiting = ", ".join(f"{name} ({url})" for name, url in pending.items())
                raise PreviewError(f"not ready after {timeout_s}s: {waiting}")
            time.sleep(READY_POLL_INTERVAL_S)

    def _run_scenario(
        self, worktree: Path, env: dict[str, str], directory: Path, api_url: str, preview_url: str
    ) -> tuple[str, str | None]:
        """Run the optional agent-written scenario; returns (open_url, error), degrading to the plain preview url on failure."""
        if not (worktree / SCENARIO_RELATIVE_PATH).is_file():
            return preview_url, None
        log_path = directory / "preview-scenario.log"
        argv = [sys.executable, SCENARIO_RELATIVE_PATH, "--api", api_url, "--url", preview_url]
        with log_path.open("wb") as log:
            popen = subprocess.Popen(
                argv, cwd=worktree, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, start_new_session=True
            )
            try:
                stdout, _ = popen.communicate(timeout=SCENARIO_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                terminate_process_groups([popen])
                popen.communicate()
                return preview_url, f"scenario timed out after {SCENARIO_TIMEOUT_S}s"
        if popen.returncode != 0:
            return preview_url, f"scenario exited with code {popen.returncode}{_tail_suffix(log_path)}"
        opened = _parse_scenario_open(stdout.decode(errors="replace"))
        if opened is None:
            return preview_url, f"scenario printed no JSON line with an 'open' path{_tail_suffix(log_path)}"
        return _join_open(preview_url, opened), None

    def _teardown(self, entry: _Entry) -> None:
        with entry.lock:
            terminate_process_groups([process.popen for process in entry.processes])
            entry.processes = []
            entry.ports = {}
            if entry.database_name is not None:
                self._drop_database(entry.database_name)
                entry.database_name = None

    def _teardown_quietly(self, entry: _Entry) -> str | None:
        try:
            self._teardown(entry)
        except (psycopg.Error, OSError) as error:
            return str(error)
        return None

    def _exit_message(self, entry: _Entry, process: _RunningProcess) -> str:
        log_path = task_dir(self._state_dir, entry.task_id) / f"preview-{process.name}.log"
        return f"process {process.name} exited with code {process.popen.returncode}{_tail_suffix(log_path)}"


def _render(template: str, values: dict[str, str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise PreviewError(f"unknown placeholder {{{key}}} (is preview.database enabled and the process name defined?)")
        return values[key]

    return PLACEHOLDER.sub(substitute, template)


def _is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((LOOPBACK, port))
        except OSError:
            return False
    return True


def _responds(url: str) -> bool:
    try:
        with DIRECT_OPENER.open(url, timeout=READY_REQUEST_TIMEOUT_S) as response:
            return response.status < 500
    except urllib.error.HTTPError as error:
        return error.code < 500
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return False


def _tail_suffix(log_path: Path) -> str:
    try:
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - LOG_TAIL_BYTES))
            tail = log.read().decode(errors="replace").strip()
    except OSError:
        return ""
    return f": {tail}" if tail else ""


def _parse_scenario_open(stdout: str) -> str | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    opened = payload.get("open") if isinstance(payload, dict) else None
    return opened if isinstance(opened, str) else None


def _join_open(preview_url: str, opened: str) -> str:
    base = preview_url.rstrip("/")
    return base + opened if opened.startswith(("/", "?", "#")) else f"{base}/{opened}"
