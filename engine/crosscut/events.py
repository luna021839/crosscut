"""Machine-readable JSONL progress reporting."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO


class EventReporter:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        self._stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def status(self, message: str) -> None:
        self.emit("status", message=message)

    def progress(self, value: float, message: str) -> None:
        self.emit("progress", value=max(0.0, min(1.0, value)), message=message)

    def log(self, message: str) -> None:
        self.emit("log", message=message)

    def result(self, **payload: Any) -> None:
        self.emit("result", **payload)

    def error(self, message: str) -> None:
        self.emit("error", message=message)


class ProgressRangeReporter:
    """Delegate events while mapping progress into one monotonic subrange."""

    def __init__(
        self,
        reporter: EventReporter,
        *,
        start: float,
        end: float,
    ) -> None:
        if not 0.0 <= start < end <= 1.0:
            raise ValueError("progress range must satisfy 0 <= start < end <= 1")
        self._reporter = reporter
        self._start = start
        self._span = end - start

    def emit(self, event_type: str, **payload: Any) -> None:
        self._reporter.emit(event_type, **payload)

    def status(self, message: str) -> None:
        self._reporter.status(message)

    def progress(self, value: float, message: str) -> None:
        clamped = max(0.0, min(1.0, value))
        self._reporter.progress(self._start + self._span * clamped, message)

    def log(self, message: str) -> None:
        self._reporter.log(message)

    def result(self, **payload: Any) -> None:
        self._reporter.result(**payload)

    def error(self, message: str) -> None:
        self._reporter.error(message)
