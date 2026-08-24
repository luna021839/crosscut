"""Validated request objects shared by the CLI and macOS app."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from .ive import AspectRatio, EditMode, IVEMember, SourceMode


DEFAULT_AUTOMATIC_SOURCE_COUNT = 10
MINIMUM_AUTOMATIC_SOURCE_COUNT = 8
MAXIMUM_AUTOMATIC_SOURCE_COUNT = 24
AUTOMATIC_MEMBER_SOURCE_COUNT = 10
MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT = 4
MAXIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT = AUTOMATIC_MEMBER_SOURCE_COUNT


def _safe_path_component(value: str) -> str:
    """Return a readable, traversal-free directory component."""

    cleaned = "".join(
        character
        if character.isalnum() or character in {" ", "-", "_", "."}
        else "-"
        for character in value.strip()
    ).strip(" .-_")
    return cleaned[:120] or "untitled"


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
    source_mode: SourceMode = SourceMode.LOCAL
    desired_source_count: int = DEFAULT_AUTOMATIC_SOURCE_COUNT
    source_cache_directory: Path | None = None
    acquire_only: bool = False
    professional_reference_profile: Path | None = None
    professional_reference_cadence_profile: Path | None = None
    stage_edit_checkpoint_directory: Path | None = None
    stage_edit_production_training_report: Path | None = None
    stage_edit_generation_checkpoint_directory: Path | None = None
    stage_edit_generation_feature_snapshot: Path | None = None
    stage_edit_generation_source_ids: tuple[str, ...] = ()
    stage_edit_learned_reviewer_checkpoint_directory: Path | None = None
    stage_edit_learned_reviewer_production_training_report: Path | None = None

    @property
    def automatic_source_count_floor(self) -> int:
        """Return the request-scoped floor for materialized automatic media.

        The default group request is a ten-source target with an eight-source
        quality floor. Explicit group counts, including eight and every count
        above ten, remain exact user requirements. Automatic member mode keeps
        a ten-source acquisition target while permitting a four-source
        materialized floor for short promotion runs.
        """

        if self.edit_mode is EditMode.MEMBER:
            return MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
        if self.desired_source_count == DEFAULT_AUTOMATIC_SOURCE_COUNT:
            return MINIMUM_AUTOMATIC_SOURCE_COUNT
        return self.desired_source_count

    def __post_init__(self) -> None:
        if self.professional_reference_profile is not None:
            profile_path = Path(self.professional_reference_profile).expanduser()
            if not profile_path.is_file():
                raise ValueError(
                    "professional_reference_profile must be an existing file"
                )
            object.__setattr__(
                self,
                "professional_reference_profile",
                profile_path.resolve(),
            )
        if self.professional_reference_cadence_profile is not None:
            cadence_profile_path = Path(
                self.professional_reference_cadence_profile
            ).expanduser()
            if not cadence_profile_path.is_file():
                raise ValueError(
                    "professional_reference_cadence_profile must be an existing file"
                )
            object.__setattr__(
                self,
                "professional_reference_cadence_profile",
                cadence_profile_path.resolve(),
            )
        checkpoint_directory = self.stage_edit_checkpoint_directory
        training_report = self.stage_edit_production_training_report
        if (checkpoint_directory is None) != (training_report is None):
            raise ValueError(
                "stage_edit_checkpoint_directory and "
                "stage_edit_production_training_report must be supplied together"
            )
        if checkpoint_directory is not None and training_report is not None:
            checkpoint_path = Path(checkpoint_directory).expanduser()
            report_path = Path(training_report).expanduser()
            if not checkpoint_path.is_dir():
                raise ValueError(
                    "stage_edit_checkpoint_directory must be an existing directory"
                )
            if not report_path.is_file():
                raise ValueError(
                    "stage_edit_production_training_report must be an existing file"
                )
            object.__setattr__(
                self,
                "stage_edit_checkpoint_directory",
                checkpoint_path.resolve(),
            )
            object.__setattr__(
                self,
                "stage_edit_production_training_report",
                report_path.resolve(),
            )
            if (
                self.edit_mode is not EditMode.GROUP
                or self.source_mode is not SourceMode.AUTOMATIC
            ):
                raise ValueError(
                    "StageEdit candidate reranking is available only for "
                    "automatic group cross-edit requests"
                )
            if self.acquire_only:
                raise ValueError(
                    "StageEdit candidate reranking cannot be used with acquire_only"
                )
        generation_checkpoint = self.stage_edit_generation_checkpoint_directory
        generation_snapshot = self.stage_edit_generation_feature_snapshot
        generation_source_ids = tuple(
            str(value).strip() for value in self.stage_edit_generation_source_ids
        )
        generation_fields_present = (
            generation_checkpoint is not None,
            generation_snapshot is not None,
            bool(generation_source_ids),
        )
        if any(generation_fields_present) and not all(generation_fields_present):
            raise ValueError(
                "stage_edit_generation_checkpoint_directory, "
                "stage_edit_generation_feature_snapshot, and "
                "stage_edit_generation_source_ids must be supplied together"
            )
        if all(generation_fields_present):
            assert generation_checkpoint is not None
            assert generation_snapshot is not None
            if any(not value for value in generation_source_ids) or len(
                generation_source_ids
            ) != len(set(generation_source_ids)):
                raise ValueError(
                    "stage_edit_generation_source_ids must be non-empty and unique"
                )
            generation_checkpoint_path = Path(
                generation_checkpoint
            ).expanduser()
            generation_snapshot_path = Path(generation_snapshot).expanduser()
            if not generation_checkpoint_path.is_dir():
                raise ValueError(
                    "stage_edit_generation_checkpoint_directory must be an "
                    "existing directory"
                )
            if not generation_snapshot_path.is_file():
                raise ValueError(
                    "stage_edit_generation_feature_snapshot must be an existing file"
                )
            if (
                self.edit_mode is not EditMode.GROUP
                or self.source_mode is not SourceMode.AUTOMATIC
            ):
                raise ValueError(
                    "StageEdit candidate generation is available only for "
                    "automatic group cross-edit requests"
                )
            if self.acquire_only:
                raise ValueError(
                    "StageEdit candidate generation cannot be used with acquire_only"
                )
            expected_source_count = (
                len(self.sources) if self.sources else self.desired_source_count
            )
            if len(generation_source_ids) != expected_source_count:
                raise ValueError(
                    "stage_edit_generation_source_ids must align one-to-one with "
                    "the materialized or requested automatic sources"
                )
            object.__setattr__(
                self,
                "stage_edit_generation_checkpoint_directory",
                generation_checkpoint_path.resolve(),
            )
            object.__setattr__(
                self,
                "stage_edit_generation_feature_snapshot",
                generation_snapshot_path.resolve(),
            )
            object.__setattr__(
                self,
                "stage_edit_generation_source_ids",
                generation_source_ids,
            )
            if checkpoint_directory is not None:
                raise ValueError(
                    "StageEdit generated-candidate promotion and ordinary visual "
                    "candidate reranking cannot be enabled together"
                )
        reviewer_checkpoint = (
            self.stage_edit_learned_reviewer_checkpoint_directory
        )
        reviewer_report = (
            self.stage_edit_learned_reviewer_production_training_report
        )
        if (reviewer_checkpoint is None) != (reviewer_report is None):
            raise ValueError(
                "stage_edit_learned_reviewer_checkpoint_directory and "
                "stage_edit_learned_reviewer_production_training_report "
                "must be supplied together"
            )
        if reviewer_checkpoint is not None and reviewer_report is not None:
            checkpoint_path = Path(reviewer_checkpoint).expanduser()
            report_path = Path(reviewer_report).expanduser()
            if not checkpoint_path.is_dir():
                raise ValueError(
                    "stage_edit_learned_reviewer_checkpoint_directory must "
                    "be an existing directory"
                )
            if not report_path.is_file():
                raise ValueError(
                    "stage_edit_learned_reviewer_production_training_report "
                    "must be an existing file"
                )
            object.__setattr__(
                self,
                "stage_edit_learned_reviewer_checkpoint_directory",
                checkpoint_path.resolve(),
            )
            object.__setattr__(
                self,
                "stage_edit_learned_reviewer_production_training_report",
                report_path.resolve(),
            )
            if (
                self.edit_mode is not EditMode.GROUP
                or self.source_mode is not SourceMode.AUTOMATIC
            ):
                raise ValueError(
                    "StageEdit learned reviewer is available only for "
                    "automatic group cross-edit requests"
                )
            if self.acquire_only:
                raise ValueError(
                    "StageEdit learned reviewer cannot be used with acquire_only"
                )
        if self.edit_mode is EditMode.GROUP and self.target_member is not None:
            object.__setattr__(self, "target_member", None)
        if self.edit_mode is EditMode.GROUP and self.aspect_ratio is not AspectRatio.LANDSCAPE:
            raise ValueError("group cross-edit mode uses the original 16:9 composition")
        if (
            self.edit_mode is EditMode.MEMBER
            and self.source_mode is SourceMode.AUTOMATIC
            and self.aspect_ratio is not AspectRatio.PORTRAIT
        ):
            raise ValueError(
                "automatic member cross-edit mode requires original 9:16 fancams"
            )
        if self.source_mode is SourceMode.AUTOMATIC and self.member_anchors:
            raise ValueError(
                "automatic source mode does not accept member anchors; "
                "use local member mode for confirmed-anchor tracking"
            )
        if (
            self.source_mode is SourceMode.AUTOMATIC
            and self.edit_mode is EditMode.MEMBER
            and self.desired_source_count != AUTOMATIC_MEMBER_SOURCE_COUNT
        ):
            raise ValueError(
                "automatic member cross-edit mode keeps an exact target of "
                f"{AUTOMATIC_MEMBER_SOURCE_COUNT} sources"
            )
        if (
            self.source_mode is SourceMode.AUTOMATIC
            and self.sources
            and not (
                self.automatic_source_count_floor
                <= len(self.sources)
                <= self.desired_source_count
            )
        ):
            raise ValueError(
                "automatic acquisition must materialize between "
                f"{self.automatic_source_count_floor} and "
                f"{self.desired_source_count} sources"
            )

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

        raw_source_mode = payload.get("source_mode", SourceMode.LOCAL.value)
        if not isinstance(raw_source_mode, str):
            raise ValueError("source_mode must be a string")
        try:
            source_mode = SourceMode(raw_source_mode)
        except ValueError as exc:
            raise ValueError("source_mode must be 'local' or 'automatic'") from exc
        if (
            mode is EditMode.MEMBER
            and source_mode is SourceMode.AUTOMATIC
            and aspect is not AspectRatio.PORTRAIT
        ):
            raise ValueError(
                "automatic member cross-edit mode requires original 9:16 fancams"
            )

        raw_desired_count = payload.get(
            "desired_source_count", DEFAULT_AUTOMATIC_SOURCE_COUNT
        )
        if isinstance(raw_desired_count, bool) or not isinstance(raw_desired_count, int):
            raise ValueError("desired_source_count must be an integer")
        desired_source_count = raw_desired_count
        if source_mode is SourceMode.AUTOMATIC:
            minimum_count = MINIMUM_AUTOMATIC_SOURCE_COUNT
            maximum_count = MAXIMUM_AUTOMATIC_SOURCE_COUNT
            if (
                mode is EditMode.MEMBER
                and desired_source_count != AUTOMATIC_MEMBER_SOURCE_COUNT
            ):
                raise ValueError(
                    "automatic member desired_source_count must remain "
                    f"{AUTOMATIC_MEMBER_SOURCE_COUNT}"
                )
            if (
                mode is EditMode.GROUP
                and not minimum_count <= desired_source_count <= maximum_count
            ):
                raise ValueError(
                    "automatic desired_source_count must be between "
                    f"{minimum_count} and {maximum_count}"
                )

        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError("sources must be an array")
        if source_mode is SourceMode.LOCAL and len(raw_sources) < 2:
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
        if reference_source < 0 or (
            sources and reference_source >= len(sources)
        ):
            raise ValueError("reference_source is outside the source list")
        if not sources and reference_source != 0:
            raise ValueError("reference_source must be 0 before automatic acquisition")

        member_value = payload.get("target_member")
        if member_value is not None and not isinstance(member_value, str):
            raise ValueError("target_member must be a string")
        target_member = (
            IVEMember.parse(member_value)
            if member_value and mode is EditMode.MEMBER
            else None
        )
        if mode is EditMode.MEMBER and target_member is None:
            raise ValueError("target_member is required for member mode")

        raw_anchors = payload.get("member_anchors", [])
        if not isinstance(raw_anchors, list):
            raise ValueError("member_anchors must be an array")
        member_anchors = tuple(
            MemberAnchorRequest.from_dict(item) for item in raw_anchors
        )
        if source_mode is SourceMode.AUTOMATIC and member_anchors:
            raise ValueError(
                "automatic source mode does not accept member anchors; "
                "use local member mode for confirmed-anchor tracking"
            )
        anchor_indices = [anchor.source_index for anchor in member_anchors]
        if len(set(anchor_indices)) != len(anchor_indices):
            raise ValueError("member anchors must use unique source_index values")
        if any(index >= len(sources) for index in anchor_indices):
            raise ValueError("member anchor source_index is outside the source list")
        if (
            mode is EditMode.MEMBER
            and source_mode is SourceMode.LOCAL
            and set(anchor_indices) != set(range(len(sources)))
        ):
            raise ValueError("member mode needs one target-member anchor for every source")
        if mode is EditMode.GROUP and member_anchors:
            raise ValueError("group mode does not accept member anchors")

        raw_output = payload.get("output_directory") or "outputs"
        if not isinstance(raw_output, str):
            raise ValueError("output_directory must be a string")
        output_directory = Path(raw_output).expanduser().resolve()
        if output_directory.exists() and not output_directory.is_dir():
            raise ValueError("output_directory points to an existing file")

        raw_cache = payload.get("source_cache_directory")
        if raw_cache is not None and (
            not isinstance(raw_cache, str) or not raw_cache.strip()
        ):
            raise ValueError("source_cache_directory must be a non-empty string")
        if raw_cache is None:
            mode_component = (
                "group"
                if mode is EditMode.GROUP
                else f"member-{target_member.value if target_member else 'unknown'}"
            )
            source_cache_directory = (
                output_directory
                / "CrossCut Sources"
                / "IVE"
                / _safe_path_component(song)
                / _safe_path_component(mode_component)
            ).resolve()
        else:
            source_cache_directory = Path(raw_cache).expanduser().resolve()
        if (
            source_cache_directory.exists()
            and not source_cache_directory.is_dir()
        ):
            raise ValueError("source_cache_directory points to an existing file")

        raw_acquire_only = payload.get("acquire_only", False)
        if not isinstance(raw_acquire_only, bool):
            raise ValueError("acquire_only must be a boolean")
        if raw_acquire_only and source_mode is not SourceMode.AUTOMATIC:
            raise ValueError("acquire_only is available only in automatic source mode")

        raw_professional_profile = payload.get("professional_reference_profile")
        if raw_professional_profile is not None and (
            not isinstance(raw_professional_profile, str)
            or not raw_professional_profile.strip()
        ):
            raise ValueError(
                "professional_reference_profile must be a non-empty string"
            )
        professional_reference_profile = (
            Path(raw_professional_profile).expanduser().resolve()
            if raw_professional_profile is not None
            else None
        )
        if (
            professional_reference_profile is not None
            and not professional_reference_profile.is_file()
        ):
            raise ValueError(
                "professional_reference_profile must be an existing file"
            )

        raw_professional_cadence_profile = payload.get(
            "professional_reference_cadence_profile"
        )
        if raw_professional_cadence_profile is not None and (
            not isinstance(raw_professional_cadence_profile, str)
            or not raw_professional_cadence_profile.strip()
        ):
            raise ValueError(
                "professional_reference_cadence_profile must be a non-empty string"
            )
        professional_reference_cadence_profile = (
            Path(raw_professional_cadence_profile).expanduser().resolve()
            if raw_professional_cadence_profile is not None
            else None
        )
        if (
            professional_reference_cadence_profile is not None
            and not professional_reference_cadence_profile.is_file()
        ):
            raise ValueError(
                "professional_reference_cadence_profile must be an existing file"
            )

        raw_stage_edit_checkpoint = payload.get(
            "stage_edit_checkpoint_directory"
        )
        raw_stage_edit_report = payload.get(
            "stage_edit_production_training_report"
        )
        for key, raw_value in (
            ("stage_edit_checkpoint_directory", raw_stage_edit_checkpoint),
            ("stage_edit_production_training_report", raw_stage_edit_report),
        ):
            if raw_value is not None and (
                not isinstance(raw_value, str) or not raw_value.strip()
            ):
                raise ValueError(f"{key} must be a non-empty string")
        if (raw_stage_edit_checkpoint is None) != (raw_stage_edit_report is None):
            raise ValueError(
                "stage_edit_checkpoint_directory and "
                "stage_edit_production_training_report must be supplied together"
            )
        stage_edit_checkpoint_directory = (
            Path(raw_stage_edit_checkpoint).expanduser().resolve()
            if raw_stage_edit_checkpoint is not None
            else None
        )
        stage_edit_production_training_report = (
            Path(raw_stage_edit_report).expanduser().resolve()
            if raw_stage_edit_report is not None
            else None
        )
        if (
            stage_edit_checkpoint_directory is not None
            and not stage_edit_checkpoint_directory.is_dir()
        ):
            raise ValueError(
                "stage_edit_checkpoint_directory must be an existing directory"
            )
        if (
            stage_edit_production_training_report is not None
            and not stage_edit_production_training_report.is_file()
        ):
            raise ValueError(
                "stage_edit_production_training_report must be an existing file"
            )

        raw_generation_checkpoint = payload.get(
            "stage_edit_generation_checkpoint_directory"
        )
        raw_generation_snapshot = payload.get(
            "stage_edit_generation_feature_snapshot"
        )
        raw_generation_source_ids = payload.get(
            "stage_edit_generation_source_ids", []
        )
        for key, raw_value in (
            (
                "stage_edit_generation_checkpoint_directory",
                raw_generation_checkpoint,
            ),
            ("stage_edit_generation_feature_snapshot", raw_generation_snapshot),
        ):
            if raw_value is not None and (
                not isinstance(raw_value, str) or not raw_value.strip()
            ):
                raise ValueError(f"{key} must be a non-empty string")
        if not isinstance(raw_generation_source_ids, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_generation_source_ids
        ):
            raise ValueError(
                "stage_edit_generation_source_ids must be an array of "
                "non-empty strings"
            )
        generation_presence = (
            raw_generation_checkpoint is not None,
            raw_generation_snapshot is not None,
            bool(raw_generation_source_ids),
        )
        if any(generation_presence) and not all(generation_presence):
            raise ValueError(
                "stage_edit_generation_checkpoint_directory, "
                "stage_edit_generation_feature_snapshot, and "
                "stage_edit_generation_source_ids must be supplied together"
            )
        generation_checkpoint_directory = (
            Path(raw_generation_checkpoint).expanduser().resolve()
            if raw_generation_checkpoint is not None
            else None
        )
        generation_feature_snapshot = (
            Path(raw_generation_snapshot).expanduser().resolve()
            if raw_generation_snapshot is not None
            else None
        )

        raw_learned_reviewer_checkpoint = payload.get(
            "stage_edit_learned_reviewer_checkpoint_directory"
        )
        raw_learned_reviewer_report = payload.get(
            "stage_edit_learned_reviewer_production_training_report"
        )
        for key, raw_value in (
            (
                "stage_edit_learned_reviewer_checkpoint_directory",
                raw_learned_reviewer_checkpoint,
            ),
            (
                "stage_edit_learned_reviewer_production_training_report",
                raw_learned_reviewer_report,
            ),
        ):
            if raw_value is not None and (
                not isinstance(raw_value, str) or not raw_value.strip()
            ):
                raise ValueError(f"{key} must be a non-empty string")
        if (raw_learned_reviewer_checkpoint is None) != (
            raw_learned_reviewer_report is None
        ):
            raise ValueError(
                "stage_edit_learned_reviewer_checkpoint_directory and "
                "stage_edit_learned_reviewer_production_training_report "
                "must be supplied together"
            )
        learned_reviewer_checkpoint_directory = (
            Path(raw_learned_reviewer_checkpoint).expanduser().resolve()
            if raw_learned_reviewer_checkpoint is not None
            else None
        )
        learned_reviewer_production_training_report = (
            Path(raw_learned_reviewer_report).expanduser().resolve()
            if raw_learned_reviewer_report is not None
            else None
        )
        if (
            learned_reviewer_checkpoint_directory is not None
            and not learned_reviewer_checkpoint_directory.is_dir()
        ):
            raise ValueError(
                "stage_edit_learned_reviewer_checkpoint_directory must be "
                "an existing directory"
            )
        if (
            learned_reviewer_production_training_report is not None
            and not learned_reviewer_production_training_report.is_file()
        ):
            raise ValueError(
                "stage_edit_learned_reviewer_production_training_report must "
                "be an existing file"
            )

        return cls(
            song=song,
            edit_mode=mode,
            aspect_ratio=aspect,
            sources=sources,
            output_directory=output_directory,
            target_member=target_member,
            reference_source=reference_source,
            member_anchors=member_anchors,
            source_mode=source_mode,
            desired_source_count=desired_source_count,
            source_cache_directory=source_cache_directory,
            acquire_only=raw_acquire_only,
            professional_reference_profile=professional_reference_profile,
            professional_reference_cadence_profile=(
                professional_reference_cadence_profile
            ),
            stage_edit_checkpoint_directory=stage_edit_checkpoint_directory,
            stage_edit_production_training_report=(
                stage_edit_production_training_report
            ),
            stage_edit_generation_checkpoint_directory=(
                generation_checkpoint_directory
            ),
            stage_edit_generation_feature_snapshot=(
                generation_feature_snapshot
            ),
            stage_edit_generation_source_ids=tuple(raw_generation_source_ids),
            stage_edit_learned_reviewer_checkpoint_directory=(
                learned_reviewer_checkpoint_directory
            ),
            stage_edit_learned_reviewer_production_training_report=(
                learned_reviewer_production_training_report
            ),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "AnalysisRequest":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("request JSON must contain an object")
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group": "IVE",
            "song": self.song,
            "edit_mode": self.edit_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "sources": [str(path) for path in self.sources],
            "output_directory": str(self.output_directory),
            "target_member": self.target_member.value if self.target_member else None,
            "reference_source": self.reference_source,
            "member_anchors": [anchor.as_dict() for anchor in self.member_anchors],
            "source_mode": self.source_mode.value,
            "desired_source_count": self.desired_source_count,
            "source_cache_directory": (
                str(self.source_cache_directory)
                if self.source_cache_directory is not None
                else None
            ),
            "acquire_only": self.acquire_only,
            "professional_reference_profile": (
                str(self.professional_reference_profile)
                if self.professional_reference_profile is not None
                else None
            ),
            "professional_reference_cadence_profile": (
                str(self.professional_reference_cadence_profile)
                if self.professional_reference_cadence_profile is not None
                else None
            ),
        }
        # Preserve the exact legacy request schema unless learned inference is
        # explicitly activated.  This also makes accidental activation visible
        # in every persisted request manifest.
        if self.stage_edit_checkpoint_directory is not None:
            payload["stage_edit_checkpoint_directory"] = str(
                self.stage_edit_checkpoint_directory
            )
            payload["stage_edit_production_training_report"] = str(
                self.stage_edit_production_training_report
            )
        if self.stage_edit_generation_checkpoint_directory is not None:
            payload["stage_edit_generation_checkpoint_directory"] = str(
                self.stage_edit_generation_checkpoint_directory
            )
            payload["stage_edit_generation_feature_snapshot"] = str(
                self.stage_edit_generation_feature_snapshot
            )
            payload["stage_edit_generation_source_ids"] = list(
                self.stage_edit_generation_source_ids
            )
        if self.stage_edit_learned_reviewer_checkpoint_directory is not None:
            payload["stage_edit_learned_reviewer_checkpoint_directory"] = str(
                self.stage_edit_learned_reviewer_checkpoint_directory
            )
            payload[
                "stage_edit_learned_reviewer_production_training_report"
            ] = str(
                self.stage_edit_learned_reviewer_production_training_report
            )
        return payload

    def with_materialized_sources(
        self,
        sources: tuple[Path, ...] | list[Path],
        *,
        member_anchors: tuple[MemberAnchorRequest, ...] | list[MemberAnchorRequest] | None = None,
    ) -> "AnalysisRequest":
        """Return this automatic request with validated downloaded paths."""

        resolved = tuple(Path(path).expanduser().resolve() for path in sources)
        if (
            self.source_mode is SourceMode.AUTOMATIC
            and not (
                self.automatic_source_count_floor
                <= len(resolved)
                <= self.desired_source_count
            )
        ):
            raise ValueError(
                "automatic acquisition must materialize between "
                f"{self.automatic_source_count_floor} and "
                f"{self.desired_source_count} sources"
            )
        minimum_materialized_count = (
            self.automatic_source_count_floor
            if self.source_mode is SourceMode.AUTOMATIC
            else MINIMUM_AUTOMATIC_SOURCE_COUNT
        )
        if len(resolved) < minimum_materialized_count:
            raise ValueError(
                "automatic acquisition must materialize at least "
                f"{minimum_materialized_count} sources"
            )
        if len(set(resolved)) != len(resolved):
            raise ValueError("materialized source videos must be unique")
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise ValueError(
                f"materialized source videos do not exist: {', '.join(missing)}"
            )
        resolved_anchors = (
            self.member_anchors if member_anchors is None else tuple(member_anchors)
        )
        if self.source_mode is SourceMode.AUTOMATIC and resolved_anchors:
            raise ValueError(
                "automatic source mode does not accept member anchors; "
                "use local member mode for confirmed-anchor tracking"
            )
        if (
            self.edit_mode is EditMode.MEMBER
            and resolved_anchors
            and (
                len(resolved_anchors) != len(resolved)
                or set(anchor.source_index for anchor in resolved_anchors)
                != set(range(len(resolved)))
            )
        ):
            raise ValueError(
                "materialized automatic member sources need one safe anchor per source"
            )
        return replace(
            self,
            sources=resolved,
            reference_source=0,
            member_anchors=resolved_anchors,
            acquire_only=False,
        )
