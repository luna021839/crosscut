"""Fail-closed visual/proxy preparation for generated StageEdit candidates.

The generation bridge deliberately stops before selection.  This adapter is
the next explicit production boundary: it sends each retained semi-Markov path
through the existing frame-descriptor, transition-hard-gate, structural-review
and visual-candidate materialization stack.  It never substitutes a source or
borrows a candidate from the ordinary visual optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    TransitionKind,
)
from .stage_edit_pipeline_bridge import (
    RetainedStageEditCandidate,
    StageEditPipelineBridgeResult,
)
from .timeline_review import TimelineReview
from .timewarp import TimeWarpMap
from .visual_candidate_materialization import (
    MaterializedVisualTimelineCandidate,
    materialize_visual_timeline_candidates,
)
from .visual_score import (
    VisualTimelineCandidateRecord,
    VisualTimelinePlan,
    build_prescribed_visual_timeline_candidate_set,
)


STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT = (
    "exact_eight_stage_edit_visual_hard_gate_materialization_v1"
)
STAGE_EDIT_VISUAL_CANDIDATE_COUNT = 8


class StageEditVisualCandidateError(RuntimeError):
    """A generated candidate batch cannot enter rendered-proxy review."""


@dataclass(frozen=True, slots=True)
class StageEditVisualCandidateRecord:
    """Globally indexed record backed by one exact prescribed visual path."""

    candidate_index: int
    review_rank: int
    cadence_profile: str
    base_record: VisualTimelineCandidateRecord
    retained_candidate: RetainedStageEditCandidate

    @property
    def optimizer_score(self) -> float:
        return self.base_record.optimizer_score

    @property
    def review(self) -> TimelineReview:
        return self.base_record.review

    @property
    def path_signature(self) -> tuple[str, ...]:
        return self.base_record.path_signature

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_record.to_dict()
        payload.update(
            {
                "candidate_index": self.candidate_index,
                "review_rank": self.review_rank,
                "cadence_profile": self.cadence_profile,
                "stage_edit_candidate_rank": self.retained_candidate.rank,
                "stage_edit_semimarkov_score": self.retained_candidate.score,
                "candidate_origin": "explicit_stage_edit_generation",
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class StageEditMaterializedVisualCandidate:
    """Duck-compatible visual candidate with immutable StageEdit provenance."""

    candidate_record: StageEditVisualCandidateRecord
    base_materialized: MaterializedVisualTimelineCandidate
    visual_plan: VisualTimelinePlan
    retained_candidate: RetainedStageEditCandidate

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
    def segments(self) -> tuple[Any, ...]:
        return self.base_materialized.segments

    @property
    def candidate_history_entry(self) -> dict[str, Any]:
        return self.candidate_record.to_dict()

    @property
    def reviewer_history_entry(self) -> dict[str, Any]:
        return self.candidate_record.review.to_dict()

    @property
    def lineage(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(item) for item in self.retained_candidate.edit.candidate_history
        ) + (
            {
                "adapter": STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
                "candidate_index": self.candidate_index,
                "stage_edit_candidate_rank": self.retained_candidate.rank,
                "prescribed_source_path": [
                    shot.segment.source_index
                    for shot in self.retained_candidate.edit.shots
                ],
                "visual_hard_gates_passed": self.candidate_record.review.passed,
                "visual_review": self.candidate_record.review.to_dict(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "review_rank": self.review_rank,
            "cadence_profile": self.cadence_profile,
            "stage_edit_candidate_rank": self.retained_candidate.rank,
            "stage_edit_semimarkov_score": self.retained_candidate.score,
            "visual_candidate": self.candidate_record.to_dict(),
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True, slots=True)
class StageEditVisualCandidateBatch:
    candidates: tuple[StageEditMaterializedVisualCandidate, ...]
    contract: str = STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT

    def __post_init__(self) -> None:
        if len(self.candidates) != STAGE_EDIT_VISUAL_CANDIDATE_COUNT:
            raise ValueError("StageEdit visual batch must contain exactly 8 candidates")
        if tuple(item.candidate_index for item in self.candidates) != tuple(range(8)):
            raise ValueError("StageEdit visual candidate indices must be contiguous")
        if any(not item.candidate_record.review.passed for item in self.candidates):
            raise ValueError("StageEdit visual batch cannot retain failed candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "activation": "explicit_stage_edit_generation_request_only",
            "candidate_pool": "stage_edit_generation_only",
            "ordinary_visual_baseline_candidate_pool_eligible": False,
            "silent_baseline_fallback_allowed": False,
            "required_candidate_count": STAGE_EDIT_VISUAL_CANDIDATE_COUNT,
            "candidate_count": len(self.candidates),
            "all_prescribed_paths_passed_visual_hard_gates": all(
                item.candidate_record.review.passed for item in self.candidates
            ),
            "promotion_state": "awaiting_decoded_proxy_review",
            "candidates": [item.to_dict() for item in self.candidates],
        }


def _absolute_stage_edit_spans(
    retained: RetainedStageEditCandidate,
) -> tuple[tuple[float, float], ...]:
    edit = retained.edit
    spans = tuple(
        (
            edit.master_audio_start + shot.segment.master_start,
            edit.master_audio_start
            + shot.segment.master_start
            + shot.segment.duration,
        )
        for shot in edit.shots
    )
    for index, (start, end) in enumerate(spans):
        if end <= start:
            raise StageEditVisualCandidateError(
                f"StageEdit candidate {retained.rank} has a non-positive span"
            )
        if index and not math.isclose(
            spans[index - 1][1], start, rel_tol=0.0, abs_tol=1e-6
        ):
            raise StageEditVisualCandidateError(
                f"StageEdit candidate {retained.rank} spans are not contiguous"
            )
    return spans


def materialize_stage_edit_visual_candidates(
    bridge_result: StageEditPipelineBridgeResult,
    *,
    sources: Sequence[Path],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    feature_cache_directory: Path | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> StageEditVisualCandidateBatch:
    """Hard-gate and materialize exactly eight learned source/cut paths.

    A failed path aborts the entire explicitly requested learned batch.  The
    ordinary visual optimizer is intentionally never consulted here.
    """

    if not isinstance(bridge_result, StageEditPipelineBridgeResult):
        raise TypeError("bridge_result must be a StageEditPipelineBridgeResult")
    retained = bridge_result.retained_candidates
    if len(retained) != STAGE_EDIT_VISUAL_CANDIDATE_COUNT:
        raise StageEditVisualCandidateError(
            "StageEdit rendered-proxy promotion requires exactly 8 retained "
            f"candidates; got {len(retained)}"
        )
    normalized_sources = tuple(Path(path).expanduser().resolve() for path in sources)
    if len(normalized_sources) != len(bridge_result.preflight.source_ids):
        raise StageEditVisualCandidateError(
            "StageEdit visual source list does not match explicit source identities"
        )
    materialized: list[StageEditMaterializedVisualCandidate] = []
    for position, candidate in enumerate(retained):
        materialized.append(
            materialize_stage_edit_visual_candidate(
                candidate,
                candidate_index=position,
                review_rank=position,
                sources=normalized_sources,
                audio_offsets=audio_offsets,
                sync_scores=sync_scores,
                scene_cut_times=scene_cut_times,
                source_timewarps=source_timewarps,
                feature_cache_directory=feature_cache_directory,
                progress_callback=(
                    (
                        lambda completed, total, position=position: progress_callback(
                            position + 1,
                            len(retained),
                            completed,
                            total,
                        )
                    )
                    if progress_callback is not None
                    else None
                ),
            )
        )
    return StageEditVisualCandidateBatch(tuple(materialized))


def materialize_stage_edit_visual_candidate(
    retained_candidate: RetainedStageEditCandidate,
    *,
    candidate_index: int,
    review_rank: int,
    sources: Sequence[Path],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    feature_cache_directory: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> StageEditMaterializedVisualCandidate:
    """Hard-gate one exact StageEdit path for bounded beam-repair review."""

    if not isinstance(retained_candidate, RetainedStageEditCandidate):
        raise TypeError("retained_candidate must be a RetainedStageEditCandidate")
    for name, value in (
        ("candidate_index", candidate_index),
        ("review_rank", review_rank),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    normalized_sources = tuple(Path(path).expanduser().resolve() for path in sources)
    candidate = retained_candidate
    if tuple(candidate.edit.sources) != normalized_sources:
        raise StageEditVisualCandidateError(
            f"StageEdit candidate {candidate.rank} media identity changed"
        )
    spans = _absolute_stage_edit_spans(candidate)
    source_path = tuple(
        shot.segment.source_index for shot in candidate.edit.shots
    )
    try:
        candidate_set = build_prescribed_visual_timeline_candidate_set(
            sources=normalized_sources,
            canonical_spans=spans,
            prescribed_source_indices=source_path,
            audio_offsets=audio_offsets,
            sync_scores=sync_scores,
            scene_cut_times=scene_cut_times,
            source_timewarps=source_timewarps,
            feature_cache_directory=feature_cache_directory,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise StageEditVisualCandidateError(
            "StageEdit candidate failed prescribed visual hard gates: "
            f"rank={candidate.rank}: {type(exc).__name__}: {exc}"
        ) from exc
    if len(candidate_set.candidates) != 1:
        raise StageEditVisualCandidateError(
            "prescribed visual gate returned an ambiguous candidate set"
        )
    base_record = candidate_set.candidates[0]
    if not base_record.review.passed:
        codes = [issue.code for issue in base_record.review.hard_failures]
        raise StageEditVisualCandidateError(
            "StageEdit candidate failed deterministic visual timeline review: "
            f"rank={candidate.rank}, hard_failures={codes}"
        )
    base_candidates = materialize_visual_timeline_candidates(
        candidate_set,
        maximum_candidates=1,
        passed_only=True,
    )
    if len(base_candidates) != 1:
        raise StageEditVisualCandidateError(
            "prescribed StageEdit candidate did not materialize exactly once"
        )
    global_record = StageEditVisualCandidateRecord(
        candidate_index=candidate_index,
        review_rank=review_rank,
        cadence_profile="stage_edit_dynamic_crop_v2_prescribed_visual_gate",
        base_record=base_record,
        retained_candidate=candidate,
    )
    visual_plan = replace(
        base_candidates[0].visual_plan,
        analysis_basis=STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
        selected_candidate_index=candidate_index,
        candidate_generation_basis=STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
        candidate_review_summaries=(global_record.to_dict(),),
    )
    return StageEditMaterializedVisualCandidate(
        candidate_record=global_record,
        base_materialized=base_candidates[0],
        visual_plan=visual_plan,
        retained_candidate=candidate,
    )


def _reparameterized_dynamic_crop_track(
    track: EditorialTransformTrack | None,
    *,
    original_duration: float,
    target_duration: float,
) -> EditorialTransformTrack | None:
    if track is None:
        return None
    if original_duration <= 0.0 or target_duration <= 0.0:
        raise StageEditVisualCandidateError(
            "Dynamic Crop v2 track cannot be applied to a non-positive segment"
        )
    scale = target_duration / original_duration
    return EditorialTransformTrack(
        keyframes=tuple(
            EditorialTransformKeyframe(
                time_seconds=keyframe.time_seconds * scale,
                zoom=keyframe.zoom,
                center_x=keyframe.center_x,
                center_y=keyframe.center_y,
                roll_degrees=keyframe.roll_degrees,
            )
            for keyframe in track.keyframes
        ),
        reason=(
            f"{track.reason}_stage_edit_preserved"
            if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
            else f"{track.reason}_stage_edit_targeted_repair_reparameterized"
        ),
    )


def apply_stage_edit_dynamic_crop_to_render_plan(
    materialized: StageEditMaterializedVisualCandidate,
    render_plan: RenderPlan,
) -> RenderPlan:
    """Overlay this candidate's learned track without invalidating face matches.

    Timing repairs may move an adjacent cut while preserving the full edit
    duration.  In that case the original segment-relative model trajectory is
    time-reparameterized; its zoom/pan/roll values are never regenerated by the
    baseline editorial-motion heuristic.  The exception is a segment touching
    a verified face dissolve: that transition was hard-gated using its existing
    face-alignment crop, so that reviewed geometry has precedence over an
    independently inferred Dynamic Crop v2 trajectory on both sides.
    """

    if not isinstance(materialized, StageEditMaterializedVisualCandidate):
        raise TypeError("materialized must be a StageEdit visual candidate")
    original_segments = tuple(
        shot.segment for shot in materialized.retained_candidate.edit.shots
    )
    if len(original_segments) != len(render_plan.segments):
        raise StageEditVisualCandidateError(
            "Dynamic Crop v2 overlay segment count changed"
        )
    face_dissolve_segments: set[int] = set()
    for boundary_index, transition in enumerate(render_plan.transitions):
        if transition.kind is TransitionKind.FACE_DISSOLVE:
            face_dissolve_segments.update((boundary_index, boundary_index + 1))
    updated = []
    for index, (original, current) in enumerate(
        zip(original_segments, render_plan.segments, strict=True)
    ):
        if original.source_index != current.source_index:
            raise StageEditVisualCandidateError(
                "Dynamic Crop v2 overlay source path changed at segment "
                f"{index}"
            )
        if index in face_dissolve_segments:
            updated.append(replace(current, editorial_transform=None))
            continue
        updated.append(
            replace(
                current,
                crop=None,
                pan_crop=None,
                editorial_transform=_reparameterized_dynamic_crop_track(
                    original.editorial_transform,
                    original_duration=original.duration,
                    target_duration=current.duration,
                ),
            )
        )
    return replace(render_plan, segments=tuple(updated))


__all__ = [
    "STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT",
    "STAGE_EDIT_VISUAL_CANDIDATE_COUNT",
    "StageEditMaterializedVisualCandidate",
    "StageEditVisualCandidateBatch",
    "StageEditVisualCandidateError",
    "StageEditVisualCandidateRecord",
    "apply_stage_edit_dynamic_crop_to_render_plan",
    "materialize_stage_edit_visual_candidate",
    "materialize_stage_edit_visual_candidates",
]
