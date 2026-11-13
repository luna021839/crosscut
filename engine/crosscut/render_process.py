"""Observe FFmpeg progress while enforcing one killable process deadline."""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import subprocess
import threading
import math
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

from .workflow_supervisor import (
    SUPERVISED_WORKFLOW_ENV,
    terminate_owned_process_group,
)


_TERMINATE_GRACE_SECONDS = 2.0
_POPEN_TYPE = subprocess.Popen


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - monotonic())


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    process_group_owned: bool,
) -> None:
    """Terminate and reap FFmpeg plus descendants from its private session."""

    if process_group_owned:
        terminate_owned_process_group(process)
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait()


def _callback_before_deadline(
    callback: Callable[[dict[str, Any]], None],
    payload: dict[str, Any],
    *,
    deadline: float | None,
    process: subprocess.Popen[str],
    process_group_owned: bool,
    command: Sequence[str],
    timeout_seconds: float | None,
) -> None:
    """Prevent a stalled UI callback from defeating the render deadline."""

    if deadline is None:
        callback(payload)
        return
    completed = threading.Event()
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            callback(payload)
        except BaseException as exc:  # Re-raised on the observing thread.
            failures.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(
        target=invoke,
        name="ive-render-progress-callback",
        daemon=True,
    )
    worker.start()
    remaining = _remaining(deadline)
    if remaining is None or not completed.wait(remaining):
        _terminate_process_tree(
            process,
            process_group_owned=process_group_owned,
        )
        raise subprocess.TimeoutExpired(list(command), timeout_seconds)
    if failures:
        raise failures[0]


def _progress_payload(
    record: Mapping[str, str],
    *,
    pid: int,
    phase: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"phase": phase, "pid": pid}
    for field in ("frame", "out_time_us", "total_size"):
        try:
            payload[field] = max(0, int(record[field]))
        except (KeyError, TypeError, ValueError):
            pass
    if "speed" in record:
        payload["speed"] = record["speed"]
    return payload


def run_observed_render(
    command: Sequence[str],
    *,
    stderr_path: Path,
    progress_callback: Callable[[dict[str, Any]], None],
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("render timeout_seconds must be positive")
    launched_command = list(command)
    launched_command[1:1] = ["-progress", "pipe:1", "-nostats"]
    deadline = (
        monotonic() + float(timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    with Path(stderr_path).open("w+b") as errors:
        popen_arguments: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": errors,
            "text": True,
            "bufsize": 1,
        }
        isolated_group = os.name == "posix" and os.environ.get(SUPERVISED_WORKFLOW_ENV) != "1"
        if isolated_group:
            popen_arguments["start_new_session"] = True
        process = subprocess.Popen(launched_command, **popen_arguments)
        process_group_owned = bool(
            isolated_group and isinstance(process, _POPEN_TYPE)
        )
        try:
            assert process.stdout is not None
            record: dict[str, str] = {}

            def consume(line: str) -> None:
                key, separator, value = line.strip().partition("=")
                if not separator:
                    return
                record[key] = value
                if key != "progress":
                    return
                payload = _progress_payload(
                    record,
                    pid=process.pid,
                    phase=value,
                )
                record.clear()
                _callback_before_deadline(
                    progress_callback,
                    payload,
                    deadline=deadline,
                    process=process,
                    process_group_owned=process_group_owned,
                    command=launched_command,
                    timeout_seconds=timeout_seconds,
                )

            try:
                stdout_fd = process.stdout.fileno()
            except (AttributeError, OSError):
                stdout_fd = None
            if stdout_fd is None:
                # Test doubles and non-file streams cannot be selected. The
                # production subprocess path always uses the bounded selector.
                for line in process.stdout:
                    if deadline is not None and _remaining(deadline) == 0.0:
                        raise subprocess.TimeoutExpired(
                            launched_command, timeout_seconds
                        )
                    consume(line)
            else:
                selector = selectors.DefaultSelector()
                try:
                    os.set_blocking(stdout_fd, False)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    pending = b""
                    while True:
                        remaining = _remaining(deadline)
                        if remaining is not None and remaining <= 0.0:
                            raise subprocess.TimeoutExpired(
                                launched_command, timeout_seconds
                            )
                        events = selector.select(
                            0.1 if remaining is None else min(0.1, remaining)
                        )
                        if events:
                            chunk = os.read(stdout_fd, 65536)
                            if chunk:
                                pending += chunk
                                if len(pending) > 1024 * 1024:
                                    raise RuntimeError("render progress line exceeds bounded buffer")
                                while b"\n" in pending:
                                    line, pending = pending.split(b"\n", 1)
                                    consume(line.decode("utf-8", errors="replace"))
                                continue
                            if process.poll() is not None:
                                if pending:
                                    consume(pending.decode("utf-8", errors="replace"))
                                break
                        elif process.poll() is not None:
                            # Do not wait for an unrelated grandchild that kept
                            # a copy of the progress pipe open.
                            break
                finally:
                    selector.close()
            remaining = _remaining(deadline)
            wait_timeout = remaining if remaining is not None else None
            try:
                code = process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                raise subprocess.TimeoutExpired(
                    launched_command, timeout_seconds
                ) from None
            if process_group_owned:
                terminate_owned_process_group(process)
        except BaseException:
            _terminate_process_tree(
                process,
                process_group_owned=process_group_owned,
            )
            raise
        finally:
            process.stdout.close()
        errors.flush()
        errors.seek(0, 2)
        size = errors.tell()
        errors.seek(max(0, size - 65536))
        detail = errors.read().decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(
        launched_command,
        code,
        stdout="",
        stderr=detail,
    )


def render_log_callback(reporter, *, total_frames):
    """Log actual progress at 5-percent milestones; never claim QA completion."""

    previous = {}

    def report(event):
        frame = event.get("frame", 0)
        percent = min(100, int(100 * frame / max(1, total_frames)))
        bucket = percent // 5
        pid = event["pid"]
        if bucket > previous.get(pid, -1) or event["phase"] == "end":
            previous[pid] = bucket
            reporter.log(
                f"인코딩 실측 · {frame}/{total_frames} 프레임 · {percent}% · "
                f"{event.get('speed', '속도 측정 중')} · 품질 QA 별도"
            )

    return report
