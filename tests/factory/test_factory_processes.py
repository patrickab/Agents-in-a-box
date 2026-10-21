"""Test process-group termination shared by command runs, previews, and scenarios."""

import subprocess
import sys

import pytest

from agent_sandbox.factory import processes

IGNORE_SIGTERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"


def spawn(code: str) -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True)
    assert process.stdout.readline().strip() == b"ready"
    return process


def test_process_ignoring_sigterm_is_killed_after_the_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processes, "KILL_GRACE_SECONDS", 0.2)
    process = spawn(IGNORE_SIGTERM)

    processes.terminate_process_groups([process])

    assert process.returncode == -9
