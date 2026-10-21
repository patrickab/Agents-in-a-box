import json
import os
from pathlib import Path
import socket
import sys
import textwrap
import time
from urllib.parse import urlsplit
import urllib.request

import psycopg
import pytest

from agent_sandbox.factory.config import PreviewConfig
from agent_sandbox.factory.preview import PreviewManager

TASK_ID = "abcd1234"
PYTHON = sys.executable
ADMIN_URL_ENV = "FACTORY_TEST_DATABASE_URL"

SERVER_SCRIPT = textwrap.dedent(
    """
    import http.server, json, os, sys

    with open(os.environ["DUMP_FILE"], "w") as dump:
        json.dump(dict(os.environ) | {"ARGV": sys.argv[1:], "CWD": os.getcwd()}, dump)
    http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), http.server.SimpleHTTPRequestHandler).serve_forever()
    """
)

SPAWNING_SCRIPT = textwrap.dedent(
    """
    import http.server, os, subprocess, sys, time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(sys.argv[1], "w") as pids:
        pids.write(f"{os.getpid()} {child.pid}")
    if len(sys.argv) > 2:
        http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), http.server.SimpleHTTPRequestHandler).serve_forever()
    time.sleep(60)
    """
)


def free_port_base() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except OSError:
        return False


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def preview_config(**overrides) -> PreviewConfig:
    base = {
        "processes": [
            {
                "name": "web",
                "argv": [PYTHON, "-m", "http.server", "{port.web}", "--bind", "127.0.0.1", "--directory", "{worktree}"],
                "ready_url": "http://127.0.0.1:{port.web}/",
            }
        ],
        "url": "http://127.0.0.1:{port.web}",
        "ready_timeout_s": 20,
    }
    return PreviewConfig.model_validate(base | overrides)


def dumping_server_config(**overrides) -> PreviewConfig:
    base = {
        "processes": [
            {
                "name": "web",
                "argv": [PYTHON, "{worktree}/server.py", "{port.web}", "{task_id}", "{data_dir}"],
                "ready_url": "http://127.0.0.1:{port.web}/",
            }
        ],
        "env": {"DUMP_FILE": "{data_dir}/env.json"},
    }
    return preview_config(**(base | overrides))


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    path = tmp_path / "worktree"
    path.mkdir()
    (path / "server.py").write_text(SERVER_SCRIPT)
    return path


@pytest.fixture
def manager(tmp_path: Path):
    base = free_port_base()
    instance = PreviewManager((base, base + 40), tmp_path / "state", os.environ.get(ADMIN_URL_ENV, "postgresql://unused/postgres"))
    yield instance
    instance.stop_all()


@pytest.fixture
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        pytest.skip(f"{ADMIN_URL_ENV} not set")
    return url


def database_exists(admin_url: str, name: str) -> bool:
    with psycopg.connect(admin_url, autocommit=True) as connection:
        return connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is not None


def read_dump(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "state" / "tasks" / TASK_ID / "preview-data" / "env.json").read_text())


def test_start_reports_ready_and_stop_is_idempotent(manager: PreviewManager, worktree: Path, tmp_path: Path):
    preview = manager.start(TASK_ID, worktree, preview_config())

    assert preview.status == "ready"
    assert preview.error is None
    assert preview.open_url == preview.url
    assert preview.url.startswith("http://127.0.0.1:")
    assert http_ok(preview.url + "/")
    assert manager.status(TASK_ID) == preview
    assert (tmp_path / "state" / "tasks" / TASK_ID / "preview-web.log").is_file()

    manager.stop(TASK_ID)
    manager.stop(TASK_ID)

    assert manager.status(TASK_ID) is None
    assert wait_until(lambda: not http_ok(preview.url + "/"))


def test_env_is_built_from_scratch_with_rendered_placeholders(
    manager: PreviewManager, worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    config = dumping_server_config(env={"DUMP_FILE": "{data_dir}/env.json", "WHERE": "{worktree}|{task_id}|{port.web}"})

    preview = manager.start(TASK_ID, worktree, config)

    assert preview.status == "ready", preview.error
    dump = read_dump(tmp_path)
    port = urlsplit(preview.url).port
    assert "OPENAI_API_KEY" not in dump
    assert {"PATH", "HOME", "USER", "LANG"} <= dump.keys()
    assert dump["WHERE"] == f"{worktree}|{TASK_ID}|{port}"
    assert dump["ARGV"] == [str(port), TASK_ID, str(tmp_path / "state" / "tasks" / TASK_ID / "preview-data")]
    assert dump["CWD"] == str(worktree)


def test_failing_readiness_kills_process_group_and_reports_failure(manager: PreviewManager, worktree: Path, tmp_path: Path):
    pid_file = tmp_path / "pids"
    (worktree / "spawn.py").write_text(SPAWNING_SCRIPT)
    config = preview_config(
        processes=[
            {"name": "stuck", "argv": [PYTHON, "{worktree}/spawn.py", str(pid_file)], "ready_url": "http://127.0.0.1:{port.stuck}/"}
        ],
        url="http://127.0.0.1:{port.stuck}",
        ready_timeout_s=1,
    )

    preview = manager.start(TASK_ID, worktree, config)

    assert preview.status == "failed"
    assert "not ready after 1s" in (preview.error or "")
    parent_pid, child_pid = map(int, pid_file.read_text().split())
    assert wait_until(lambda: not pid_alive(parent_pid) and not pid_alive(child_pid))
    assert manager.status(TASK_ID) == preview
    manager.stop(TASK_ID)
    assert manager.status(TASK_ID) is None


def test_ports_are_distinct_per_process_and_task_and_skip_unbindable(tmp_path: Path, worktree: Path):
    base = free_port_base()
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", base))
        blocker.listen()
        manager = PreviewManager((base, base + 40), tmp_path / "state", "postgresql://unused/postgres")
        two_processes = preview_config(
            processes=[
                {
                    "name": name,
                    "argv": [PYTHON, "-m", "http.server", f"{{port.{name}}}", "--bind", "127.0.0.1"],
                    "ready_url": f"http://127.0.0.1:{{port.{name}}}/",
                }
                for name in ("api", "ui")
            ],
            url="http://127.0.0.1:{port.ui}",
        )
        try:
            first = manager.start("task0001", worktree, two_processes)
            second = manager.start("task0002", worktree, two_processes)
        finally:
            manager.stop_all()

    assert first.status == second.status == "ready"
    assert first.url != second.url
    assert urlsplit(first.url).port != base
    assert urlsplit(second.url).port != base


def test_status_flips_to_failed_when_a_process_dies(manager: PreviewManager, worktree: Path, tmp_path: Path):
    pid_file = tmp_path / "pids"
    (worktree / "spawn.py").write_text(SPAWNING_SCRIPT)
    config = preview_config(
        processes=[
            {
                "name": "flaky",
                "argv": [PYTHON, "{worktree}/spawn.py", str(pid_file), "{port.flaky}"],
                "ready_url": "http://127.0.0.1:{port.flaky}/",
            }
        ],
        url="http://127.0.0.1:{port.flaky}",
    )
    assert manager.start(TASK_ID, worktree, config).status == "ready"
    assert wait_until(lambda: pid_file.exists() and pid_file.read_text().count(" ") == 1)
    parent_pid, child_pid = map(int, pid_file.read_text().split())

    os.kill(parent_pid, 9)

    assert wait_until(lambda: manager.status(TASK_ID).status == "failed")
    assert "process flaky exited" in (manager.status(TASK_ID).error or "")
    assert wait_until(lambda: not pid_alive(child_pid))


def test_database_is_created_for_the_preview_and_dropped_on_stop(
    manager: PreviewManager, worktree: Path, tmp_path: Path, admin_url: str
):
    config = dumping_server_config(database=True, env={"DUMP_FILE": "{data_dir}/env.json", "DB": "{database_url}"})
    database_name = f"preview_{TASK_ID}"

    preview = manager.start(TASK_ID, worktree, config)

    assert preview.status == "ready", preview.error
    assert database_exists(admin_url, database_name)
    database_url = urlsplit(read_dump(tmp_path)["DB"])
    admin = urlsplit(admin_url)
    assert database_url.path == f"/{database_name}"
    assert (database_url.scheme, database_url.netloc) == (admin.scheme, admin.netloc)

    manager.stop(TASK_ID)
    manager.stop(TASK_ID)

    assert not database_exists(admin_url, database_name)


def test_database_is_dropped_when_start_fails(manager: PreviewManager, worktree: Path, admin_url: str):
    config = preview_config(
        database=True,
        processes=[{"name": "boom", "argv": [PYTHON, "-c", "import sys; sys.exit(1)"], "ready_url": "http://127.0.0.1:{port.boom}/"}],
        url="http://127.0.0.1:{port.boom}",
    )

    preview = manager.start(TASK_ID, worktree, config)

    assert preview.status == "failed"
    assert not database_exists(admin_url, f"preview_{TASK_ID}")


def write_scenario(worktree: Path, body: str) -> None:
    out_dir = worktree / ".factory" / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "scenario.py").write_text(textwrap.dedent(body))


def test_scenario_open_path_is_appended_to_preview_url(manager: PreviewManager, worktree: Path):
    write_scenario(
        worktree,
        """
        import argparse, json, os

        parser = argparse.ArgumentParser()
        parser.add_argument("--api")
        parser.add_argument("--url")
        args = parser.parse_args()
        with open(os.path.join(os.path.dirname(__file__), "args.json"), "w") as out:
            json.dump({"api": args.api, "url": args.url}, out)
        print("some log noise")
        print(json.dumps({"open": "/?open=abc"}))
        """,
    )
    config = preview_config(api_url="http://127.0.0.1:{port.web}/api")

    preview = manager.start(TASK_ID, worktree, config)

    assert preview.status == "ready", preview.error
    assert preview.error is None
    assert preview.open_url == f"{preview.url}/?open=abc"
    recorded = json.loads((worktree / ".factory" / "out" / "args.json").read_text())
    assert recorded == {"api": f"{preview.url}/api", "url": preview.url}


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        ("import sys; print('traceback here', file=sys.stderr); sys.exit(2)", "scenario exited with code 2"),
        ("import json; print(json.dumps({'other': 1}))", "no JSON line"),
    ],
)
def test_failing_scenario_keeps_preview_ready_with_plain_open_url(
    manager: PreviewManager, worktree: Path, body: str, expected_error: str
):
    write_scenario(worktree, body)

    preview = manager.start(TASK_ID, worktree, preview_config())

    assert preview.status == "ready"
    assert preview.open_url == preview.url
    assert expected_error in (preview.error or "")
