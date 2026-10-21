"""Provide small git and file helpers for the factory engine tests."""

from pathlib import Path
import subprocess
import textwrap
from typing import Any

from agent_sandbox.factory.config import PreviewConfig
from agent_sandbox.factory.models import Preview


def git(cwd: Path, *args: str) -> str:
    """Run git in cwd and return stdout."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def write(path: Path, text: str) -> None:
    """Write dedented text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    def start(self, task_id: str, config: dict[str, Any]) -> None:
        self.calls.append(("start", task_id, config["repo"]))

    def send(self, task_id: str, payload: dict[str, Any]) -> None:
        self.calls.append(("send", task_id, payload))

    def retry(self, task_id: str, config: dict[str, Any], entry: str) -> None:
        self.calls.append(("retry", task_id, entry))

    def cancel(self, task_id: str) -> None:
        self.calls.append(("cancel", task_id, None))


class FakePreviews:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start(self, task_id: str, worktree: Path, config: PreviewConfig) -> Preview:
        self.started.append(task_id)
        return Preview(status="ready", url="http://x", open_url="http://x/?open=1", error=None)

    def stop(self, task_id: str) -> None:
        self.stopped.append(task_id)

    def status(self, task_id: str) -> Preview | None:
        return None

    def stop_all(self) -> None:
        pass
