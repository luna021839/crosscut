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
