"""Small, dependency-free wrappers around ffprobe.

The rest of the engine should consume :class:`MediaInfo` instead of depending
on ffprobe's loosely typed JSON response.  ``ffprobe_json`` remains public for
callers that need fields not represented by the dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot inspect a media file."""


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _frame_rate(value: Any) -> float | None:
    if not isinstance(value, str) or value in ("", "0/0", "N/A"):
        return None
    numerator, separator, denominator = value.partition("/")
    try:
        if not separator:
            return float(numerator)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    index: int
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    duration_seconds: float | None
    bit_rate: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "codec_name": self.codec_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "duration_seconds": self.duration_seconds,
            "bit_rate": self.bit_rate,
        }


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    index: int
    codec_name: str | None
    width: int | None
    height: int | None
    frame_rate: float | None
    duration_seconds: float | None
    bit_rate: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "codec_name": self.codec_name,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "duration_seconds": self.duration_seconds,
            "bit_rate": self.bit_rate,
        }


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: str
    format_name: str | None
    duration_seconds: float | None
    start_time_seconds: float | None
    size_bytes: int | None
    bit_rate: int | None
    audio_streams: tuple[AudioStreamInfo, ...]
    video_streams: tuple[VideoStreamInfo, ...]
    tags: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a structure accepted directly by ``json.dumps``."""

        return {
            "path": self.path,
            "format_name": self.format_name,
            "duration_seconds": self.duration_seconds,
            "start_time_seconds": self.start_time_seconds,
            "size_bytes": self.size_bytes,
            "bit_rate": self.bit_rate,
            "audio_streams": [stream.to_dict() for stream in self.audio_streams],
            "video_streams": [stream.to_dict() for stream in self.video_streams],
            "tags": dict(self.tags),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def ffprobe_json(
    path: str | os.PathLike[str],
    *,
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    """Return ffprobe's stream/format JSON for a local media file."""

    media_path = Path(path).expanduser()
    if not media_path.is_file():
        raise FileNotFoundError(media_path)

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        os.fspath(media_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise MediaProbeError(f"could not execute {ffprobe_bin!r}: {exc}") from exc

    if completed.returncode:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise MediaProbeError(f"ffprobe failed for {media_path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON for {media_path}") from exc
    if not isinstance(payload, dict):
        raise MediaProbeError(f"ffprobe returned an unexpected response for {media_path}")
    return payload


def probe_media(
    path: str | os.PathLike[str],
    *,
    ffprobe_bin: str = "ffprobe",
) -> MediaInfo:
    """Inspect a media file and return stable, typed metadata."""

    media_path = Path(path).expanduser().resolve()
    payload = ffprobe_json(media_path, ffprobe_bin=ffprobe_bin)
    raw_format = payload.get("format")
    format_info = raw_format if isinstance(raw_format, dict) else {}
    raw_streams = payload.get("streams")
    streams = raw_streams if isinstance(raw_streams, list) else []

    audio_streams: list[AudioStreamInfo] = []
    video_streams: list[VideoStreamInfo] = []
    for fallback_index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            continue
        index = _as_int(stream.get("index"))
        stream_index = fallback_index if index is None else index
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            audio_streams.append(
                AudioStreamInfo(
                    index=stream_index,
                    codec_name=stream.get("codec_name"),
                    sample_rate=_as_int(stream.get("sample_rate")),
                    channels=_as_int(stream.get("channels")),
                    channel_layout=stream.get("channel_layout"),
                    duration_seconds=_as_float(stream.get("duration")),
                    bit_rate=_as_int(stream.get("bit_rate")),
                )
            )
        elif codec_type == "video":
            video_streams.append(
                VideoStreamInfo(
                    index=stream_index,
                    codec_name=stream.get("codec_name"),
                    width=_as_int(stream.get("width")),
                    height=_as_int(stream.get("height")),
                    frame_rate=_frame_rate(
                        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                    ),
                    duration_seconds=_as_float(stream.get("duration")),
                    bit_rate=_as_int(stream.get("bit_rate")),
                )
            )

    raw_tags = format_info.get("tags")
    tags = (
        {str(key): str(value) for key, value in raw_tags.items()}
        if isinstance(raw_tags, dict)
        else {}
    )
    return MediaInfo(
        path=os.fspath(media_path),
        format_name=format_info.get("format_name"),
        duration_seconds=_as_float(format_info.get("duration")),
        start_time_seconds=_as_float(format_info.get("start_time")),
        size_bytes=_as_int(format_info.get("size")),
        bit_rate=_as_int(format_info.get("bit_rate")),
        audio_streams=tuple(audio_streams),
        video_streams=tuple(video_streams),
        tags=tags,
    )


# A readable alias for callers that think in terms of metadata rather than probes.
probe_metadata = probe_media
