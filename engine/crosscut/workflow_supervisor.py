"""A process boundary for the complete acquisition-to-QA wall-clock budget.

Python alarms cannot interrupt an arbitrary native model call.  The app entry
point therefore runs analysis in a separate process group; the supervisor does
not execute model code, decode media, or call UI progress callbacks.  Every
descendant stays in that group, including FFmpeg, and is reaped/terminated on
timeout, cancellation, failure, and normal completion.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
from threading import Event
from time import monotonic
from typing import Sequence

SUPERVISED_WORKFLOW_ENV = "IVE_CROSSCUT_SUPERVISED_WORKFLOW"
WORKFLOW_CLEANUP_GRACE_SECONDS = 0.5


def terminate_owned_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float = WORKFLOW_CLEANUP_GRACE_SECONDS,
) -> None:
    """Clean descendants even when their group leader has already exited."""

    if os.name != "posix":
        if process.poll() is None:
            process.kill()
        process.wait(timeout=max(0.1, grace_seconds))
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    # Waiting for the leader alone is insufficient: a descendant can ignore
    # TERM and keep media files / the app's stdout pipe open after it exits.
    until = monotonic() + grace_seconds
    while monotonic() < until:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=min(0.02, max(0.001, until - monotonic())))
        except subprocess.TimeoutExpired:
            continue
        # A finished leader is not a sleep primitive; use a bounded event wait.
        Event().wait(min(0.02, max(0.0, until - monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(0.1, grace_seconds))


def run_supervised_workflow(
    command: Sequence[str],
    *,
    absolute_deadline_monotonic_seconds: float,
) -> int:
    """Stream through inherited pipes while an independent process enforces time."""

    deadline = absolute_deadline_monotonic_seconds
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("workflow deadline must be finite")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(list(command), 0)
    if os.name != "posix":
        raise RuntimeError("bounded workflow supervision requires POSIX process groups")
    environment = dict(os.environ)
    environment[SUPERVISED_WORKFLOW_ENV] = "1"
    process = subprocess.Popen(list(command), env=environment, start_new_session=True)
    try:
        return process.wait(timeout=max(0.001, deadline - monotonic()))
    finally:
        terminate_owned_process_group(process)
