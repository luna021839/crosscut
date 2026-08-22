"""Validated request objects shared by the CLI and macOS app."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .ive import AspectRatio, EditMode, IVEMember


@dataclass(frozen=True, slots=True)
class MemberAnchorRequest:
    source_index: int
    time_seconds: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("anchor source_index must be a non-negative integer")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("anchor time_seconds must be non-negative and finite")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("anchor bbox_xyxy must contain four numbers")
        left, top, right, bottom = self.bbox_xyxy
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in self.bbox_xyxy
        ):
            raise ValueError("anchor bbox_xyxy must contain four numbers")
        if not all(math.isfinite(float(value)) for value in self.bbox_xyxy):
            raise ValueError("anchor bbox_xyxy values must be finite")
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("anchor bbox_xyxy must fit inside normalized image bounds")

    @classmethod
    def from_dict(cls, payload: Any) -> "MemberAnchorRequest":
        if not isinstance(payload, dict):
            raise ValueError("each member anchor must be an object")
        source_index = payload.get("source_index")
        time_seconds = payload.get("time_seconds")
        raw_bbox = payload.get("bbox_xyxy")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError("anchor source_index must be an integer")
        if isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float)):
            raise ValueError("anchor time_seconds must be numeric")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("anchor bbox_xyxy must contain four numbers")
        return cls(
            source_index=source_index,
            time_seconds=float(time_seconds),
            bbox_xyxy=tuple(raw_bbox),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "time_seconds": self.time_seconds,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    song: str
    edit_mode: EditMode
    aspect_ratio: AspectRatio
    sources: tuple[Path, ...]
    output_directory: Path
    target_member: IVEMember | None = None
    reference_source: int = 0
    member_anchors: tuple[MemberAnchorRequest, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalysisRequest":
        raw_mode = payload.get("edit_mode", EditMode.GROUP.value)
        raw_aspect = payload.get("aspect_ratio", AspectRatio.LANDSCAPE.value)
        if not isinstance(raw_mode, str) or not isinstance(raw_aspect, str):
            raise ValueError("edit_mode and aspect_ratio must be strings")
        try:
            mode = EditMode(raw_mode)
            aspect = AspectRatio(raw_aspect)
        except ValueError as exc:
            raise ValueError(f"Invalid edit request option: {exc}") from exc
        if mode is EditMode.GROUP and aspect is not AspectRatio.LANDSCAPE:
            raise ValueError("group cross-edit mode uses the original 16:9 composition")

        raw_song = payload.get("song")
        if not isinstance(raw_song, str):
            raise ValueError("song must be a string")
        song = raw_song.strip()
        if not song:
            raise ValueError("song is required")
        if len(song) > 120:
            raise ValueError("song must be 120 characters or fewer")

        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list) or len(raw_sources) < 2:
            raise ValueError("at least two source videos are required")
        if any(not isinstance(value, str) or not value.strip() for value in raw_sources):
            raise ValueError("each source path must be a non-empty string")
        sources = tuple(Path(value).expanduser().resolve() for value in raw_sources)
        if len(set(sources)) != len(sources):
            raise ValueError("source videos must be unique")
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise ValueError(f"source videos do not exist: {', '.join(missing)}")

        raw_reference = payload.get("reference_source", 0)
        if isinstance(raw_reference, bool) or not isinstance(raw_reference, int):
            raise ValueError("reference_source must be an integer")
        reference_source = raw_reference
        if reference_source < 0 or reference_source >= len(sources):
            raise ValueError("reference_source is outside the source list")

        member_value = payload.get("target_member")
        if member_value is not None and not isinstance(member_value, str):
            raise ValueError("target_member must be a string")
        target_member = IVEMember.parse(member_value) if member_value else None
        if mode is EditMode.MEMBER and target_member is None:
            raise ValueError("target_member is required for member mode")

        raw_anchors = payload.get("member_anchors", [])
        if not isinstance(raw_anchors, list):
            raise ValueError("member_anchors must be an array")
        member_anchors = tuple(
            MemberAnchorRequest.from_dict(item) for item in raw_anchors
        )
        anchor_indices = [anchor.source_index for anchor in member_anchors]
        if len(set(anchor_indices)) != len(anchor_indices):
            raise ValueError("member anchors must use unique source_index values")
        if any(index >= len(sources) for index in anchor_indices):
            raise ValueError("member anchor source_index is outside the source list")
        if mode is EditMode.MEMBER and set(anchor_indices) != set(range(len(sources))):
            raise ValueError("member mode needs one target-member anchor for every source")
        if mode is EditMode.GROUP and member_anchors:
            raise ValueError("group mode does not accept member anchors")

        raw_output = payload.get("output_directory") or "outputs"
        if not isinstance(raw_output, str):
            raise ValueError("output_directory must be a string")
        output_directory = Path(raw_output).expanduser().resolve()
        if output_directory.exists() and not output_directory.is_dir():
            raise ValueError("output_directory points to an existing file")

        return cls(
            song=song,
            edit_mode=mode,
            aspect_ratio=aspect,
            sources=sources,
            output_directory=output_directory,
            target_member=target_member,
            reference_source=reference_source,
            member_anchors=member_anchors,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "AnalysisRequest":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("request JSON must contain an object")
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": "IVE",
            "song": self.song,
            "edit_mode": self.edit_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "sources": [str(path) for path in self.sources],
            "output_directory": str(self.output_directory),
            "target_member": self.target_member.value if self.target_member else None,
            "reference_source": self.reference_source,
            "member_anchors": [anchor.as_dict() for anchor in self.member_anchors],
        }
