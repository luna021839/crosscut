"""Readiness and scheduler checks for native subprocess tests, not product budgets."""
import os
from pathlib import Path
import signal
import subprocess
from threading import Event
from time import monotonic


def wait_for_pid_file(process, path: Path, *, startup_seconds=10.):
    deadline = monotonic() + startup_seconds
    while monotonic() < deadline:
        try:
            text = path.read_text().strip()
            if text.isdecimal():
                return int(text)
        except FileNotFoundError:
            pass
        if process.poll() is not None:
            raise AssertionError(f"fixture exited before readiness: {process.returncode}")
        Event().wait(.005)
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    raise AssertionError("fixture startup never reached readiness")


def assert_group_member_stopped(test, pid, group, *, reap_seconds=1.):
    """KILL delivery and ps observation need not occur in the same timeslice."""
    until = monotonic() + reap_seconds
    while True:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "pgid=,stat="],
                                capture_output=True, text=True, timeout=2.)
        values = result.stdout.split()
        if not values or values[0] != str(group) or values[1].startswith("Z"):
            return
        if monotonic() >= until:
            test.fail(f"owned descendant still active after cleanup: {result.stdout.strip()}")
        Event().wait(.01)
