"""Patchable intermediate representation for a reviewed stage edit.

``RenderPlan`` is deliberately a final, renderer-facing contract.  StageEdit
keeps the section assignment, candidate/reviewer history and repair lineage
that are needed to review and partially re-edit a timeline before committing
to a full-resolution encode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .render import RenderPlan, RenderSegment, RenderTransition
from .request import AspectRatio
from .section_director import SectionPlan


SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class StageShot:
    shot_index: int
    section_index: int
    segment: RenderSegment
    decision_id: str
    decision_confidence: float | None = None
    color_grade: Mapping[str, float] | None = None
    # Exact serialized inference-token music structure for this selected shot.
    # Repair code must preserve or replace these values with the corresponding
    # newly decoded decision; downstream reviewers never synthesize them from
    # wall-clock time.
    section_role: str | None = None
    section_progress: float | None = None
    phrase_progress: float | None = None

    def __post_init__(self) -> None:
        if self.shot_index < 0 or self.section_index < 0:
            raise ValueError("shot and section indices must be non-negative")
        if not self.decision_id.strip():
            raise ValueError("decision_id cannot be blank")
        if self.decision_confidence is not None and (
            not math.isfinite(self.decision_confidence)
            or not 0.0 <= self.decision_confidence <= 1.0
        ):
            raise ValueError("decision confidence must lie in 0...1")
        if self.color_grade is not None and any(
            not isinstance(name, str)
            or not name.strip()
            or not math.isfinite(float(value))
            for name, value in self.color_grade.items()
        ):
            raise ValueError("color grade must contain finite named scalars")
        if self.section_role is not None:
            role = str(self.section_role).strip()
            if role not in {
                "intro",
                "verse_like",
                "build",
                "chorus_like",
                "bridge_like",
                "outro",
            }:
                raise ValueError("section_role is outside the editing-role schema")
            object.__setattr__(self, "section_role", role)
        for name in ("section_progress", "phrase_progress"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in 0...1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "section_index": self.section_index,
            "decision_id": self.decision_id,
            "decision_confidence": self.decision_confidence,
            "source_index": self.segment.source_index,
            "source_start": self.segment.source_start,
            "source_duration": self.segment.effective_source_duration,
            "master_start": self.segment.master_start,
            "duration": self.segment.duration,
            "has_crop": self.segment.crop is not None,
            "has_pan_crop": self.segment.pan_crop is not None,
            "has_editorial_transform": self.segment.editorial_transform is not None,
            "color_grade": dict(self.color_grade) if self.color_grade else None,
            "section_role": self.section_role,
            "section_progress": self.section_progress,
            "phrase_progress": self.phrase_progress,
        }


@dataclass(frozen=True, slots=True)
class StageBoundary:
    boundary_index: int
    transition: RenderTransition
    reviewer_status: str = "unreviewed"
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.boundary_index < 0:
            raise ValueError("boundary_index must be non-negative")
        if self.reviewer_status not in {"unreviewed", "passed", "warning", "failed"}:
            raise ValueError("invalid reviewer status")
        normalized = tuple(str(code).strip() for code in self.issue_codes)
        if any(not code for code in normalized):
            raise ValueError("issue codes cannot be blank")
        object.__setattr__(self, "issue_codes", normalized)

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "reviewer_status": self.reviewer_status,
            "issue_codes": list(self.issue_codes),
            "transition": self.transition.to_dict(fps),
        }


@dataclass(frozen=True, slots=True)
class StageEditLineage:
    round_index: int
    reason: str
    replaced_shot_start: int
    replaced_shot_end_exclusive: int
    score_before: float | None
    score_after: float | None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("repair round must start at one")
        if not self.reason.strip():
            raise ValueError("repair reason cannot be blank")
        if (
            self.replaced_shot_start < 0
            or self.replaced_shot_end_exclusive <= self.replaced_shot_start
        ):
            raise ValueError("repair shot window must be positive")
        for value in (self.score_before, self.score_after):
            if value is not None and not math.isfinite(value):
                raise ValueError("repair scores must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "reason": self.reason,
            "replaced_shot_start": self.replaced_shot_start,
            "replaced_shot_end_exclusive": self.replaced_shot_end_exclusive,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class StageEdit:
    sources: tuple[Path, ...]
    reference_source: int
    aspect_ratio: AspectRatio
    width: int
    height: int
    fps: float
    master_audio_start: float
    section_plan: SectionPlan
    shots: tuple[StageShot, ...]
    boundaries: tuple[StageBoundary, ...]
    output: Path
    candidate_history: tuple[Mapping[str, Any], ...] = ()
    reviewer_history: tuple[Mapping[str, Any], ...] = ()
    lineage: tuple[StageEditLineage, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported StageEdit schema version")
        if not self.sources or not self.shots:
            raise ValueError("StageEdit needs sources and shots")
        if not 0 <= self.reference_source < len(self.sources):
            raise ValueError("reference source is outside the source list")
        if len(self.boundaries) != len(self.shots) - 1:
            raise ValueError("StageEdit needs one boundary per shot edge")
        if tuple(shot.shot_index for shot in self.shots) != tuple(range(len(self.shots))):
            raise ValueError("StageEdit shot indices must be contiguous")
        if tuple(item.boundary_index for item in self.boundaries) != tuple(
            range(len(self.boundaries))
        ):
            raise ValueError("StageEdit boundary indices must be contiguous")
        expected = 0.0
        for shot in self.shots:
            if abs(shot.segment.master_start - expected) > 1e-6:
                raise ValueError("StageEdit shots must be contiguous on master time")
            if shot.section_index >= len(self.section_plan.sections):
                raise ValueError("shot references an unknown section")
            absolute_master = self.master_audio_start + shot.segment.master_start
            section = self.section_plan.sections[shot.section_index]
            midpoint = absolute_master + 0.5 * shot.segment.duration
            if not section.start_seconds - 1e-6 <= midpoint <= section.end_seconds + 1e-6:
                raise ValueError("shot section does not contain its midpoint")
            expected += shot.segment.duration
        # Reuse the renderer's exact timing, transition and transform guards.
        self.to_render_plan()

    @property
    def duration_seconds(self) -> float:
        return sum(shot.segment.duration for shot in self.shots)

    def to_render_plan(self, *, output: Path | None = None) -> RenderPlan:
        return RenderPlan(
            sources=self.sources,
            segments=tuple(shot.segment for shot in self.shots),
            reference_source=self.reference_source,
            aspect_ratio=self.aspect_ratio,
            output=output or self.output,
            master_audio_start=self.master_audio_start,
            width=self.width,
            height=self.height,
            fps=self.fps,
            transitions=tuple(item.transition for item in self.boundaries),
        )

    def with_reviewer_result(self, result: Mapping[str, Any]) -> "StageEdit":
        if not result:
            raise ValueError("reviewer result cannot be empty")
        return replace(self, reviewer_history=self.reviewer_history + (dict(result),))

    def replace_window(
        self,
        *,
        start_shot: int,
        end_shot_exclusive: int,
        replacement_shots: Sequence[StageShot],
        replacement_boundaries: Sequence[StageBoundary],
        reason: str,
        score_before: float | None,
        score_after: float | None,
    ) -> "StageEdit":
        """Replace only one reviewed window while preserving outside decisions.

        ``replacement_boundaries`` covers every edge touched by the patch:
        the left exterior edge when it exists, all internal replacement edges,
        and the right exterior edge when it exists.  This explicit contract
        prevents a partial repair from silently retaining an unreviewed edge.
        """

        if not 0 <= start_shot < end_shot_exclusive <= len(self.shots):
            raise ValueError("replacement window is outside the StageEdit")
        replacement = tuple(replacement_shots)
        if not replacement:
            raise ValueError("replacement window cannot be empty")
        old_start = self.shots[start_shot].segment.master_start
        old_end = (
            self.shots[end_shot_exclusive - 1].segment.master_start
            + self.shots[end_shot_exclusive - 1].segment.duration
        )
        new_start = replacement[0].segment.master_start
        new_end = replacement[-1].segment.master_start + replacement[-1].segment.duration
        if not math.isclose(old_start, new_start, abs_tol=1e-6) or not math.isclose(
            old_end, new_end, abs_tol=1e-6
        ):
            raise ValueError("partial repair must preserve the window time span")
        if score_after is not None and score_before is not None and score_after <= score_before:
            raise ValueError("partial repair must improve the hybrid review score")

        boundary_start = max(0, start_shot - 1)
        boundary_end = min(len(self.boundaries), end_shot_exclusive)
        expected_boundary_count = (
            len(replacement) - 1
            + (1 if start_shot > 0 else 0)
            + (1 if end_shot_exclusive < len(self.shots) else 0)
        )
        boundary_patch = tuple(replacement_boundaries)
        if len(boundary_patch) != expected_boundary_count:
            raise ValueError("replacement boundary patch has the wrong edge count")

        combined_shots = (
            self.shots[:start_shot] + replacement + self.shots[end_shot_exclusive:]
        )
        combined_boundaries = (
            self.boundaries[:boundary_start]
            + boundary_patch
            + self.boundaries[boundary_end:]
        )
        reindexed_shots = tuple(
            shot if shot.shot_index == index else replace(shot, shot_index=index)
            for index, shot in enumerate(combined_shots)
        )
        reindexed_boundaries = tuple(
            (
                boundary
                if boundary.boundary_index == index
                else replace(boundary, boundary_index=index)
            )
            for index, boundary in enumerate(combined_boundaries)
        )
        lineage = StageEditLineage(
            round_index=len(self.lineage) + 1,
            reason=reason,
            replaced_shot_start=start_shot,
            replaced_shot_end_exclusive=end_shot_exclusive,
            score_before=score_before,
            score_after=score_after,
        )
        return replace(
            self,
            shots=reindexed_shots,
            boundaries=reindexed_boundaries,
            lineage=self.lineage + (lineage,),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [str(path) for path in self.sources],
            "reference_source": self.reference_source,
            "aspect_ratio": self.aspect_ratio.value,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "master_audio_start": self.master_audio_start,
            "duration_seconds": self.duration_seconds,
            "output": str(self.output),
            "section_plan": self.section_plan.to_dict(),
            "shots": [shot.to_dict() for shot in self.shots],
            "boundaries": [item.to_dict(self.fps) for item in self.boundaries],
            "candidate_history": [dict(item) for item in self.candidate_history],
            "reviewer_history": [dict(item) for item in self.reviewer_history],
            "lineage": [item.to_dict() for item in self.lineage],
        }


__all__ = [
    "SCHEMA_VERSION",
    "StageBoundary",
    "StageEdit",
    "StageEditLineage",
    "StageShot",
]
