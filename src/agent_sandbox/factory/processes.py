"""Terminate the host process groups the engine starts for commands, previews, and scenarios."""

from collections.abc import Sequence
import contextlib
import os
import signal
import subprocess
import time

KILL_GRACE_SECONDS = 5


def terminate_process_groups(processes: Sequence[subprocess.Popen]) -> None:
    """SIGTERM the process group of each started process, wait out the grace period, then SIGKILL and reap each one."""
    for popen in processes:
        _signal_group(popen.pid, signal.SIGTERM)
    deadline = time.monotonic() + KILL_GRACE_SECONDS
    for popen in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            popen.wait(timeout=max(0.0, deadline - time.monotonic()))
    for popen in processes:
        _signal_group(popen.pid, signal.SIGKILL)
        popen.wait()


def _signal_group(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)
