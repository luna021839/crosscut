"""Safe RenderPlan materialization for reviewed visual timeline alternatives.

Candidate generation deliberately stays independent of ``pipeline.py``.  This
module is the narrow bridge that lets proxy-review or a later StageEdit stage
consume more than the one timeline selected by ``build_visual_timeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

from .ive import AspectRatio
from .render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    aligned_render_segment,
    output_size,
)
from .visual_score import (
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelinePlan,
)


@dataclass(frozen=True, slots=True)
class MaterializedVisualTimelineCandidate:
    """One ranked candidate mapped onto renderer-safe source-local clocks."""

    candidate_set: VisualTimelineCandidateSet = field(repr=False)
    candidate_record: VisualTimelineCandidateRecord
    visual_plan: VisualTimelinePlan = field(repr=False)
    segments: tuple[RenderSegment, ...]
    common_master_start: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.common_master_start) or self.common_master_start < 0:
            raise ValueError("common_master_start must be non-negative and finite")
        index = self.candidate_record.candidate_index
        if (
            not 0 <= index < len(self.candidate_set.candidates)
            or self.candidate_set.candidates[index] is not self.candidate_record
        ):
            raise ValueError("candidate_record does not belong to candidate_set")
        if not self.segments:
            raise ValueError("materialized candidate needs at least one segment")
        if len(self.segments) != len(
            self.candidate_record.timeline_result.candidates
        ):
            raise ValueError("materialized segment count does not match candidate")
        if self.visual_plan.selected_candidate_index != index:
            raise ValueError("visual plan does not select candidate_record")
        if len(self.visual_plan.choices) != len(self.segments):
            raise ValueError("visual plan choice count does not match segments")
        expected_master_start = 0.0
        for segment in self.segments:
            if not math.isclose(
                segment.master_start,
                expected_master_start,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("materialized segments must be contiguous")
            expected_master_start += segment.duration

    @property
    def candidate_index(self) -> int:
        return self.candidate_record.candidate_index

    @property
    def review_rank(self) -> int:
        return self.candidate_record.review_rank

    @property
    def cadence_profile(self) -> str:
        return self.candidate_record.cadence_profile

    @property
    def candidate_history_entry(self) -> dict[str, Any]:
        """Privacy-safe metadata suitable for ``StageEdit.candidate_history``."""

        return self.candidate_record.to_dict()

    @property
    def reviewer_history_entry(self) -> dict[str, Any]:
        """Full structural review suitable for ``StageEdit.reviewer_history``."""

        return self.candidate_record.review.to_dict()

    def to_render_plan(
        self,
        *,
        output: Path,
        reference_source: int = 0,
        aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
        width: int | None = None,
        height: int | None = None,
        fps: float = 29.97,
        transitions: Sequence[RenderTransition] = (),
    ) -> RenderPlan:
        """Create an exact hard-cut proxy plan unless transitions are supplied."""

        ratio = (
            aspect_ratio
            if isinstance(aspect_ratio, AspectRatio)
            else AspectRatio(aspect_ratio)
        )
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together")
        if width is None or height is None:
            width, height = output_size(ratio)
        return RenderPlan(
            sources=self.candidate_set.sources,
            segments=self.segments,
            reference_source=reference_source,
            aspect_ratio=ratio,
            output=Path(output),
            master_audio_start=self.common_master_start,
            width=width,
            height=height,
            fps=fps,
            transitions=tuple(transitions),
        )


def _materialize_candidate_record(
    candidate_set: VisualTimelineCandidateSet,
    record: VisualTimelineCandidateRecord,
) -> MaterializedVisualTimelineCandidate:
    visual_plan = candidate_set.materialize(record)
    path = visual_plan.choices
    common_master_start = path[0].canonical_master_start
    expected_master_start = common_master_start
    segments: list[RenderSegment] = []
    for expected_segment_index, choice in enumerate(path):
        if choice.segment_index != expected_segment_index:
            raise ValueError("candidate segment indices must start at zero and be contiguous")
        if not math.isclose(
            choice.canonical_master_start,
            expected_master_start,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("candidate canonical spans must be contiguous")
        source_index = choice.source_index
        duration = choice.duration_seconds
        segments.append(
            aligned_render_segment(
                source_index=source_index,
                reference_start=choice.canonical_master_start,
                common_master_start=common_master_start,
                source_offset=candidate_set.audio_offsets[source_index],
                duration=duration,
                timewarp=candidate_set.source_timewarps[source_index],
            )
        )
        expected_master_start = choice.canonical_master_end
    return MaterializedVisualTimelineCandidate(
        candidate_set=candidate_set,
        candidate_record=record,
        visual_plan=visual_plan,
        segments=tuple(segments),
        common_master_start=common_master_start,
    )


def materialize_visual_timeline_candidate_record(
    candidate_set: VisualTimelineCandidateSet,
    record: VisualTimelineCandidateRecord,
) -> MaterializedVisualTimelineCandidate:
    """Materialize one retained record without evaluating unrelated runners-up."""

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    if not isinstance(record, VisualTimelineCandidateRecord):
        raise TypeError("record must be a VisualTimelineCandidateRecord")
    return _materialize_candidate_record(candidate_set, record)


def materialize_visual_timeline_candidates(
    candidate_set: VisualTimelineCandidateSet,
    *,
    maximum_candidates: int = 8,
    passed_only: bool = False,
) -> tuple[MaterializedVisualTimelineCandidate, ...]:
    """Materialize the deterministic reviewer top-N without losing runners-up."""

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 32
    ):
        raise ValueError("maximum_candidates must be between 1 and 32")
    if not isinstance(passed_only, bool):
        raise TypeError("passed_only must be a bool")
    materialized: list[MaterializedVisualTimelineCandidate] = []
    signatures: set[tuple[str, ...]] = set()
    for record in candidate_set.ranked_candidates:
        if passed_only and not record.review.passed:
            continue
        if record.path_signature in signatures:
            continue
        signatures.add(record.path_signature)
        materialized.append(
            materialize_visual_timeline_candidate_record(candidate_set, record)
        )
        if len(materialized) >= maximum_candidates:
            break
    return tuple(materialized)


def materialize_visual_timeline_render_plans(
    candidate_set: VisualTimelineCandidateSet,
    *,
    output_directory: Path,
    filename_prefix: str = "visual-candidate",
    maximum_candidates: int = 8,
    passed_only: bool = False,
    reference_source: int = 0,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    width: int | None = None,
    height: int | None = None,
    fps: float = 29.97,
) -> tuple[RenderPlan, ...]:
    """Build deterministic proxy RenderPlans; no files or directories are written."""

    prefix = filename_prefix.strip()
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("filename_prefix must be one safe path component")
    materialized = materialize_visual_timeline_candidates(
        candidate_set,
        maximum_candidates=maximum_candidates,
        passed_only=passed_only,
    )
    root = Path(output_directory)
    return tuple(
        item.to_render_plan(
            output=root
            / (
                f"{prefix}-rank-{item.review_rank + 1:02d}-"
                f"candidate-{item.candidate_index:02d}.mp4"
            ),
            reference_source=reference_source,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            fps=fps,
        )
        for item in materialized
    )


__all__ = [
    "MaterializedVisualTimelineCandidate",
    "materialize_visual_timeline_candidate_record",
    "materialize_visual_timeline_candidates",
    "materialize_visual_timeline_render_plans",
]
