"""Share fixtures for the factory engine tests."""

from collections.abc import Iterator
import os
from pathlib import Path

from factory_helpers import git, write
import pytest

from agent_sandbox.factory.store import Store

TEST_DATABASE_ENV = "FACTORY_TEST_DATABASE_URL"
ENGINE_TABLES = "events, messages, phase_runs, repos, tasks"


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    """A zero-config git repo on `main`: no factory.yaml, no workflows, no agents."""
    root = tmp_path / "bare"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    write(root / "hello.txt", "hello\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo on `main` with a factory.yaml, one workflow, and a coder agent."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    write(root / "hello.txt", "hello\n")
    write(
        root / "factory.yaml",
        """
        main_branch: main
        sandbox: false
        gates:
          - [echo, ok]
        """,
    )
    write(root / ".factory/.gitignore", "out/\n")
    write(root / ".factory/workflows/code.yaml", "name: Just code\nphases: [code]\n")
    write(
        root / ".omp/agents/coder.md",
        """
        ---
        name: coder
        description: writes code
        writes: ["**"]
        output: .factory/out/code.json
        ---
        You are the coder.
        """,
    )
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    return root


@pytest.fixture
def database_url() -> Iterator[str]:
    """A clean engine database, or skip when no test database is configured."""
    url = os.environ.get(TEST_DATABASE_ENV)
    if not url:
        pytest.skip(f"set {TEST_DATABASE_ENV} to run database-backed factory tests")
    import psycopg

    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(f"DROP TABLE IF EXISTS {ENGINE_TABLES} CASCADE")
    Store(url).ensure_schema()
    yield url
