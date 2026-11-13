"""Select the newest working yt-dlp frontend available to this Python runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable


RunProcess = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class YTDLPFrontend:
    command: tuple[str, ...]
    version: str
    kind: str

    @property
    def display(self) -> str:
        return " ".join(self.command)

    def to_dict(self) -> dict[str, object]:
        return {
            "available": True,
            "command": list(self.command),
            "display": self.display,
            "version": self.version,
            "kind": self.kind,
            "command_succeeded": True,
        }


def _version_key(value: str) -> tuple[int, ...]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    return numbers or (0,)


def discover_ytdlp_frontend(
    *,
    runner: RunProcess = subprocess.run,
    python_executable: str | None = None,
    path_executable: str | None = None,
) -> YTDLPFrontend | None:
    """Probe module and PATH installations, preferring the newest one.

    Homebrew can lag behind a user-site ``yt_dlp`` module.  That distinction is
    material: recent YouTube player changes can make an older executable list
    formats successfully but return HTTP 403 for the media URL.
    """

    python = python_executable or sys.executable
    candidates: list[tuple[tuple[str, ...], str]] = []
    if importlib.util.find_spec("yt_dlp") is not None:
        candidates.append(((python, "-m", "yt_dlp"), "python_module"))
    resolved_path = path_executable if path_executable is not None else shutil.which("yt-dlp")
    if resolved_path:
        candidates.append(((str(Path(resolved_path).resolve()),), "path_executable"))

    working: list[YTDLPFrontend] = []
    seen: set[tuple[str, ...]] = set()
    for command, kind in candidates:
        if command in seen:
            continue
        seen.add(command)
        try:
            completed = runner(
                (*command, "--version"),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode != 0:
            continue
        lines = (completed.stdout or completed.stderr).strip().splitlines()
        if not lines:
            continue
        working.append(
            YTDLPFrontend(command=command, version=lines[0].strip(), kind=kind)
        )
    if not working:
        return None
    return max(
        working,
        key=lambda item: (_version_key(item.version), item.kind == "python_module"),
    )


__all__ = ["YTDLPFrontend", "discover_ytdlp_frontend"]
