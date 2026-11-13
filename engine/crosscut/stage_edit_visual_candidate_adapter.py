"""Fail-closed visual/proxy preparation for generated StageEdit candidates.

The generation bridge deliberately stops before selection.  This adapter is
the next explicit production boundary: it sends each retained semi-Markov path
through the existing frame-descriptor, transition-hard-gate, structural-review
and visual-candidate materialization stack.  It never substitutes a source or
borrows a candidate from the ordinary visual optimizer.
"""

from __future__ import annotations
from .optional_native_worker import isolated_optional

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderTransition,
    TransitionKind,
    _editorial_transform_track_is_identity,
)
from .prescribed_render_geometry import PRESCRIBED_RENDER_GEOMETRY
from .stage_edit import StageBoundary, StageEdit, StageShot
from .stage_edit_release_policy import (
    STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT,
    stage_edit_release_policy_payload,
)
from .stage_edit_pipeline_bridge import (
    STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
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
    PersistentVisualFeatureCacheInitializationError,
    PersistentVisualFeatureCacheRuntimeError,
    VisualTimelineCandidateRecord,
    VisualTimelinePlan,
    build_prescribed_visual_timeline_candidate_set,
)


STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT = (
    "exact_eight_stage_edit_visual_hard_gate_dynamic_crop_v2_v3_"
    "typed_hybrid_floor_materialization_v3"
)
STAGE_EDIT_VISUAL_CANDIDATE_COUNT = 8
STAGE_EDIT_LEARNED_ONLY_CANDIDATE_POOL = "stage_edit_generation_only"
STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL = (
    "stage_edit_generation_plus_typed_visual_floor_v1"
)
HYBRID_VISUAL_FLOOR_CONTRACT = (
    "reviewed_visual_timeline_exact_pts_dynamic_crop_v3_seed_v1"
)
HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN = (
    "hybrid_visual_floor_dynamic_crop_v3"
)


class StageEditVisualCandidateError(RuntimeError):
    """A generated candidate batch cannot enter rendered-proxy review."""


def _retained_dynamic_crop_version(
    retained: RetainedStageEditCandidate,
) -> int:
    versions: set[int] = set()
    for item in retained.edit.candidate_history:
        if not isinstance(item, Mapping) or "dynamic_crop_version" not in item:
            continue
        value = item["dynamic_crop_version"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (2, 3)
        ):
            raise StageEditVisualCandidateError(
                "StageEdit candidate history has an invalid Dynamic Crop version"
            )
        versions.add(value)
    if len(versions) > 1:
        raise StageEditVisualCandidateError(
            "StageEdit candidate history mixes Dynamic Crop versions"
        )
    # Pre-v3 retained candidates did not serialize the version.  Their only
    # valid interpretation is the legacy v2 path.
    return next(iter(versions), 2)


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

    @property
    def dynamic_crop_version(self) -> int:
        return _retained_dynamic_crop_version(self.retained_candidate)

    @property
    def candidate_origin(self) -> str:
        return getattr(
            self.retained_candidate,
            "candidate_origin",
            "stage_edit_semimarkov",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_record.to_dict()
        payload.update(
            {
                "candidate_index": self.candidate_index,
                "review_rank": self.review_rank,
                "cadence_profile": self.cadence_profile,
                "stage_edit_candidate_rank": self.retained_candidate.rank,
                "stage_edit_candidate_score": self.retained_candidate.score,
                "stage_edit_semimarkov_score": (
                    self.retained_candidate.score
                    if self.candidate_origin == "stage_edit_semimarkov"
                    else None
                ),
                "hybrid_visual_floor_optimizer_score": (
                    self.retained_candidate.score
                    if self.candidate_origin
                    == HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN
                    else None
                ),
                "candidate_origin": self.candidate_origin,
                "dynamic_crop_version": self.dynamic_crop_version,
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
    quality_gated_single_delivery: bool = False

    @property
    def adapter_contract(self) -> str:
        return (STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT
                if self.quality_gated_single_delivery else STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT)

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
    def dynamic_crop_version(self) -> int:
        return _retained_dynamic_crop_version(self.retained_candidate)

    @property
    def candidate_origin(self) -> str:
        return getattr(
            self.retained_candidate,
            "candidate_origin",
            "stage_edit_semimarkov",
        )

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
                "adapter": self.adapter_contract,
                "candidate_index": self.candidate_index,
                "stage_edit_candidate_rank": self.retained_candidate.rank,
                "candidate_origin": self.candidate_origin,
                "prescribed_source_path": [
                    shot.segment.source_index
                    for shot in self.retained_candidate.edit.shots
                ],
                "visual_hard_gates_passed": self.candidate_record.review.passed,
                "visual_review": self.candidate_record.review.to_dict(),
                "dynamic_crop_version": self.dynamic_crop_version,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "review_rank": self.review_rank,
            "cadence_profile": self.cadence_profile,
            "stage_edit_candidate_rank": self.retained_candidate.rank,
            "stage_edit_candidate_score": self.retained_candidate.score,
            "stage_edit_semimarkov_score": (
                self.retained_candidate.score
                if self.candidate_origin == "stage_edit_semimarkov"
                else None
            ),
            "candidate_origin": self.candidate_origin,
            "dynamic_crop_version": self.dynamic_crop_version,
            "visual_candidate": self.candidate_record.to_dict(),
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True, slots=True)
class StageEditVisualCandidateBatch:
    candidates: tuple[StageEditMaterializedVisualCandidate, ...]
    learned_candidate_rejections: tuple[Mapping[str, Any], ...] = ()
    contract: str = STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT
    quality_gated_single_delivery: bool = False

    def __post_init__(self) -> None:
        if type(self.quality_gated_single_delivery) is not bool:
            raise TypeError("StageEdit delivery policy must be explicit boolean")
        if self.quality_gated_single_delivery:
            if not 1 <= len(self.candidates) <= STAGE_EDIT_VISUAL_CANDIDATE_COUNT:
                raise ValueError("StageEdit delivery batch requires 1...8 candidates")
            object.__setattr__(self, "contract", STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT)
        elif len(self.candidates) != STAGE_EDIT_VISUAL_CANDIDATE_COUNT:
            raise ValueError("StageEdit visual batch must contain exactly 8 candidates")
        if any(
            not isinstance(item, StageEditMaterializedVisualCandidate)
            for item in self.candidates
        ):
            raise TypeError(
                "StageEdit visual batch requires typed materialized candidates"
            )
        if any(item.quality_gated_single_delivery is not self.quality_gated_single_delivery
               for item in self.candidates):
            raise ValueError("StageEdit candidate lineage delivery policy differs from batch")
        if any(
            not isinstance(item.candidate_record, StageEditVisualCandidateRecord)
            for item in self.candidates
        ):
            raise TypeError(
                "StageEdit visual batch requires typed candidate records"
            )
        if any(
            not isinstance(item.retained_candidate, RetainedStageEditCandidate)
            or not isinstance(
                item.candidate_record.retained_candidate,
                RetainedStageEditCandidate,
            )
            or item.candidate_record.retained_candidate
            is not item.retained_candidate
            for item in self.candidates
        ):
            raise TypeError(
                "StageEdit visual batch candidate lineage is not typed and bound"
            )
        if tuple(item.candidate_index for item in self.candidates) != tuple(range(len(self.candidates))):
            raise ValueError("StageEdit visual candidate indices must be contiguous")
        if any(not item.candidate_record.review.passed for item in self.candidates):
            raise ValueError("StageEdit visual batch cannot retain failed candidates")
        versions = {item.dynamic_crop_version for item in self.candidates}
        if len(versions) != 1:
            raise ValueError("StageEdit visual batch mixes Dynamic Crop versions")
        hybrid_count = sum(
            item.candidate_origin == HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN
            for item in self.candidates
        )
        if hybrid_count > 1:
            raise ValueError(
                "StageEdit visual batch can retain only one hybrid visual floor"
            )
        if self.quality_gated_single_delivery and hybrid_count == len(self.candidates):
            raise ValueError("StageEdit delivery cannot substitute a floor for all learned paths")
        if hybrid_count and versions != {3}:
            raise ValueError(
                "hybrid visual floor requires Dynamic Crop v3"
            )
        if any(
            not isinstance(item, Mapping)
            for item in self.learned_candidate_rejections
        ):
            raise TypeError(
                "learned candidate rejections must be scalar audit mappings"
            )

    @property
    def minimum_passing_candidate_count(self) -> int:
        return 1 if self.quality_gated_single_delivery else STAGE_EDIT_VISUAL_CANDIDATE_COUNT

    @property
    def hybrid_visual_floor_candidate_count(self) -> int:
        return sum(
            item.candidate_origin == HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN
            for item in self.candidates
        )

    @property
    def candidate_pool(self) -> str:
        if self.hybrid_visual_floor_candidate_count:
            return STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL
        return STAGE_EDIT_LEARNED_ONLY_CANDIDATE_POOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "activation": "explicit_stage_edit_generation_request_only",
            "candidate_pool": self.candidate_pool,
            "ordinary_visual_baseline_candidate_pool_eligible": False,
            "silent_baseline_fallback_allowed": False,
            "raw_baseline_candidate_direct_selection_allowed": False,
            "hybrid_visual_floor_contract": (
                HYBRID_VISUAL_FLOOR_CONTRACT
                if self.hybrid_visual_floor_candidate_count
                else None
            ),
            "hybrid_visual_floor_candidate_count": (
                self.hybrid_visual_floor_candidate_count
            ),
            "learned_candidate_count": (
                len(self.candidates)
                - self.hybrid_visual_floor_candidate_count
            ),
            "learned_candidate_rejections": [
                dict(item) for item in self.learned_candidate_rejections
            ],
            "required_candidate_count": self.minimum_passing_candidate_count,
            "release_policy": (stage_edit_release_policy_payload()
                               if self.quality_gated_single_delivery else None),
            "candidate_count": len(self.candidates),
            "dynamic_crop_version": self.candidates[0].dynamic_crop_version,
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


def _canonical_path_sha256(
    visual_plan: VisualTimelinePlan,
) -> str:
    payload = [
        {
            "segment_index": item.segment_index,
            "source_index": item.source_index,
            "canonical_master_start": round(
                float(item.canonical_master_start), 9
            ),
            "canonical_master_end": round(
                float(item.canonical_master_end), 9
            ),
        }
        for item in visual_plan.choices
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _exact_head_frame(
    head_frames: Sequence[Any],
    master_time_seconds: float,
) -> Any:
    matches = tuple(
        frame
        for frame in head_frames
        if math.isclose(
            float(frame.master_time_seconds),
            float(master_time_seconds),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    if len(matches) != 1:
        raise StageEditVisualCandidateError(
            "hybrid visual floor boundary is not one exact Transformer "
            f"decision time: {master_time_seconds:.6f}"
        )
    return matches[0]


def _build_hybrid_visual_floor_retained_candidate_from_parts(
    *,
    visual_plan: VisualTimelinePlan,
    segments: Sequence[Any],
    common_master_start: float,
    head_frames: Sequence[Any],
    canonical_source_ids: Sequence[str],
    template_edit: StageEdit,
    candidate_rank: int,
    optimizer_score: float,
    cadence_profile: str,
    path_signature: Sequence[str],
    continuity_authority: str,
) -> RetainedStageEditCandidate:
    """Bind a reviewed visual path to exact v3 source-local crop decodes.

    The visual optimizer remains the explicit source/cut authority for this
    one quality-floor candidate.  The Transformer supplies only the exact
    source-slot Dynamic Crop v3 prediction at each already selected boundary.
    Every source gate is rechecked across the entire shot, so the conversion
    cannot smuggle an unavailable source into a typed StageEdit.
    """

    if not isinstance(visual_plan, VisualTimelinePlan):
        raise TypeError("hybrid visual floor requires a VisualTimelinePlan")
    if not isinstance(template_edit, StageEdit):
        raise TypeError("hybrid visual floor requires a StageEdit template")
    choices = tuple(visual_plan.choices)
    renderer_segments = tuple(segments)
    frames = tuple(head_frames)
    source_ids = tuple(str(value).strip() for value in canonical_source_ids)
    if not choices or len(choices) != len(renderer_segments):
        raise StageEditVisualCandidateError(
            "hybrid visual floor choices and exact renderer segments differ"
        )
    if not frames or any(not value for value in source_ids):
        raise StageEditVisualCandidateError(
            "hybrid visual floor lacks Transformer frames or source identities"
        )
    if tuple(template_edit.sources) == () or len(template_edit.sources) != len(
        source_ids
    ):
        raise StageEditVisualCandidateError(
            "hybrid visual floor source identities do not align"
        )
    if not math.isfinite(float(optimizer_score)):
        raise StageEditVisualCandidateError(
            "hybrid visual floor optimizer score is not finite"
        )
    section_plan = template_edit.section_plan
    if not math.isclose(
        float(common_master_start),
        float(section_plan.start_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        float(template_edit.master_audio_start),
        float(common_master_start),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise StageEditVisualCandidateError(
            "hybrid visual floor master clock differs from StageEdit"
        )
    if not math.isclose(
        float(choices[0].canonical_master_start),
        float(section_plan.start_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        float(choices[-1].canonical_master_end),
        float(section_plan.end_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise StageEditVisualCandidateError(
            "hybrid visual floor does not cover the exact StageEdit range"
        )
    frame_times = tuple(float(frame.master_time_seconds) for frame in frames)
    if frame_times != tuple(sorted(frame_times)) or len(set(frame_times)) != len(
        frame_times
    ):
        raise StageEditVisualCandidateError(
            "hybrid visual floor Transformer decision times are not ordered"
        )

    shots: list[StageShot] = []
    previous_end = float(common_master_start)
    for index, (choice, renderer_segment) in enumerate(
        zip(choices, renderer_segments, strict=True)
    ):
        start = float(choice.canonical_master_start)
        end = float(choice.canonical_master_end)
        duration = end - start
        source_index = int(choice.source_index)
        if choice.segment_index != index or not math.isclose(
            start,
            previous_end,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise StageEditVisualCandidateError(
                "hybrid visual floor spans are not contiguous"
            )
        if not 0 <= source_index < len(source_ids):
            raise StageEditVisualCandidateError(
                "hybrid visual floor source index is outside the snapshot"
            )
        if (
            renderer_segment.source_index != source_index
            or not math.isclose(
                float(renderer_segment.master_start),
                start - float(common_master_start),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(renderer_segment.duration),
                duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise StageEditVisualCandidateError(
                "hybrid visual floor renderer clock or source changed"
            )
        start_frame = _exact_head_frame(frames, start)
        if end < section_plan.end_seconds - 1e-6:
            _exact_head_frame(frames, end)
        span_frames = tuple(
            frame
            for frame in frames
            if start - 1e-6
            <= float(frame.master_time_seconds)
            < end - 1e-6
        )
        if not span_frames or any(
            source_index >= len(frame.effective_source_mask)
            or not frame.effective_source_mask[source_index]
            for frame in span_frames
        ):
            raise StageEditVisualCandidateError(
                "hybrid visual floor selected a source that fails a "
                f"Transformer hard gate inside shot {index}"
            )
        predictions = tuple(start_frame.dynamic_crop_v3_by_source)
        if source_index >= len(predictions):
            raise StageEditVisualCandidateError(
                "hybrid visual floor Dynamic Crop source slot is missing"
            )
        prediction = predictions[source_index]
        if (
            prediction is None
            or prediction.source_index != source_index
            or prediction.source_id != source_ids[source_index]
        ):
            raise StageEditVisualCandidateError(
                "hybrid visual floor Dynamic Crop source identity differs"
            )
        if (
            start_frame.section_role is None
            or start_frame.section_progress is None
            or start_frame.phrase_progress is None
        ):
            raise StageEditVisualCandidateError(
                "hybrid visual floor lacks exact token music structure"
            )
        decode = prediction.decode(duration)
        directive = section_plan.directive_at(start + 0.5 * duration)
        segment = replace(
            renderer_segment,
            crop=None,
            pan_crop=None,
            editorial_transform=decode.track,
        )
        shots.append(
            StageShot(
                shot_index=index,
                section_index=directive.section_index,
                segment=segment,
                decision_id=(
                    f"hybrid-visual-floor-v3-r{candidate_rank:02d}-"
                    f"s{index:04d}"
                ),
                decision_confidence=None,
                section_role=start_frame.section_role,
                section_progress=start_frame.section_progress,
                phrase_progress=start_frame.phrase_progress,
                dynamic_crop_decode_v3=decode,
            )
        )
        previous_end = end

    boundaries = tuple(
        StageBoundary(
            boundary_index=index,
            transition=RenderTransition(
                reason="hybrid_visual_floor_reviewed_hard_cut",
                tier="C",
            ),
        )
        for index in range(len(shots) - 1)
    )
    path_sha256 = _canonical_path_sha256(visual_plan)
    output = template_edit.output.with_name(
        f".{template_edit.output.stem}-hybrid-visual-floor-v3.mp4"
    )
    edit = StageEdit(
        sources=template_edit.sources,
        reference_source=template_edit.reference_source,
        aspect_ratio=template_edit.aspect_ratio,
        width=template_edit.width,
        height=template_edit.height,
        fps=template_edit.fps,
        master_audio_start=float(common_master_start),
        section_plan=section_plan,
        shots=tuple(shots),
        boundaries=boundaries,
        output=output,
        candidate_history=(
            {
                "generator": HYBRID_VISUAL_FLOOR_CONTRACT,
                "candidate_origin": HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
                "source_cut_authority": (
                    "reviewed_ordinary_visual_optimizer"
                ),
                "dynamic_crop_authority": (
                    "source_exact_transformer_v3_prediction_at_selected_boundary"
                ),
                "raw_baseline_direct_selection_allowed": False,
                "dynamic_crop_version": 3,
                "visual_optimizer_score": float(optimizer_score),
                "cadence_profile": str(cadence_profile),
                "visual_path_signature": list(path_signature),
                "visual_path_sha256": path_sha256,
                "exact_source_gate_checked_across_every_shot": True,
            },
            {
                "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
                "visual_baseline_replaced": False,
                "selected_for_render": False,
                "exact_source_clock_reprojection": True,
                "timewarped_segment_count": sum(
                    segment.source_duration is not None
                    for segment in renderer_segments
                ),
                "dynamic_crop_version": 3,
                "continuity_authority": str(continuity_authority),
                "hybrid_visual_floor_contract": (
                    HYBRID_VISUAL_FLOOR_CONTRACT
                ),
            },
        ),
    )
    return RetainedStageEditCandidate(
        rank=candidate_rank,
        score=float(optimizer_score),
        edit=edit,
        exact_clock_reprojected=True,
        timewarped_segment_count=sum(
            segment.source_duration is not None
            for segment in renderer_segments
        ),
        candidate_origin=HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
    )


def _build_hybrid_visual_floor_retained_candidate(
    bridge_result: StageEditPipelineBridgeResult,
    materialized: MaterializedVisualTimelineCandidate,
    *,
    candidate_rank: int,
) -> RetainedStageEditCandidate:
    if not isinstance(bridge_result, StageEditPipelineBridgeResult):
        raise TypeError("bridge_result must be a StageEditPipelineBridgeResult")
    if not isinstance(materialized, MaterializedVisualTimelineCandidate):
        raise TypeError(
            "hybrid visual floor must be a materialized visual candidate"
        )
    if bridge_result.preflight.dynamic_crop_version != 3:
        raise StageEditVisualCandidateError(
            "hybrid visual floor is supported only for Dynamic Crop v3"
        )
    if not materialized.candidate_record.review.passed:
        raise StageEditVisualCandidateError(
            "hybrid visual floor must pass the ordinary visual hard gates"
        )
    template = bridge_result.retained_candidates[0].edit
    normalized_sources = tuple(
        Path(value).expanduser().resolve()
        for value in materialized.candidate_set.sources
    )
    if tuple(template.sources) != normalized_sources:
        raise StageEditVisualCandidateError(
            "hybrid visual floor media identity differs from StageEdit"
        )
    return _build_hybrid_visual_floor_retained_candidate_from_parts(
        visual_plan=materialized.visual_plan,
        segments=materialized.segments,
        common_master_start=materialized.common_master_start,
        head_frames=bridge_result.head_materialization.frames,
        canonical_source_ids=bridge_result.preflight.source_ids,
        template_edit=template,
        candidate_rank=candidate_rank,
        optimizer_score=materialized.candidate_record.optimizer_score,
        cadence_profile=materialized.candidate_record.cadence_profile,
        path_signature=materialized.candidate_record.path_signature,
        continuity_authority=bridge_result.preflight.continuity_authority,
    )


def _reindex_stage_edit_materialized_candidate(
    materialized: StageEditMaterializedVisualCandidate,
    *,
    candidate_index: int,
) -> StageEditMaterializedVisualCandidate:
    record = replace(
        materialized.candidate_record,
        candidate_index=candidate_index,
        review_rank=candidate_index,
    )
    return replace(
        materialized,
        candidate_record=record,
        visual_plan=replace(
            materialized.visual_plan,
            selected_candidate_index=candidate_index,
            candidate_review_summaries=(record.to_dict(),),
        ),
    )


def _wrap_hybrid_visual_floor_candidate(
    materialized: MaterializedVisualTimelineCandidate,
    retained: RetainedStageEditCandidate,
    *,
    candidate_index: int,
) -> StageEditMaterializedVisualCandidate:
    record = StageEditVisualCandidateRecord(
        candidate_index=candidate_index,
        review_rank=candidate_index,
        cadence_profile=(
            "hybrid_visual_floor_dynamic_crop_v3_prescribed_visual_gate"
        ),
        base_record=materialized.candidate_record,
        retained_candidate=retained,
    )
    visual_plan = replace(
        materialized.visual_plan,
        analysis_basis=HYBRID_VISUAL_FLOOR_CONTRACT,
        selected_candidate_index=candidate_index,
        candidate_generation_basis=HYBRID_VISUAL_FLOOR_CONTRACT,
        candidate_review_summaries=(record.to_dict(),),
    )
    return StageEditMaterializedVisualCandidate(
        candidate_record=record,
        base_materialized=materialized,
        visual_plan=visual_plan,
        retained_candidate=retained,
    )


def materialize_stage_edit_visual_candidates(
    bridge_result: StageEditPipelineBridgeResult,
    *,
    sources: Sequence[Path],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    hybrid_visual_floor_candidate: (
        MaterializedVisualTimelineCandidate | None
    ) = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
    quality_gated_single_delivery: bool = False,
) -> StageEditVisualCandidateBatch:
    """Hard-gate retained StageEdit paths before any rendered-proxy selection.

    The legacy path retains eight learned semi-Markov candidates.  Production
    Dynamic Crop v3 may instead predeclare one already reviewed visual quality
    floor: all eight learned paths are still checked, the best seven passing
    paths are retained, and the floor is converted to an exact-PTS StageEdit
    with source-local v3 crop decodes.  It is never a post-failure fallback or
    a raw baseline candidate selected outside StageEdit proxy/montage QA.

    With explicit single-delivery policy, retain up to eight passing paths;
    at least one learned path must pass even when a hybrid floor was declared.
    Per-path failures remain audited and runtime/storage errors still propagate.
    """

    if type(quality_gated_single_delivery) is not bool:
        raise TypeError("StageEdit delivery policy must be explicit boolean")
    if not isinstance(bridge_result, StageEditPipelineBridgeResult):
        raise TypeError("bridge_result must be a StageEditPipelineBridgeResult")
    expected_dynamic_crop_version = getattr(
        bridge_result.preflight, "dynamic_crop_version", 2
    )
    if expected_dynamic_crop_version not in (2, 3):
        raise StageEditVisualCandidateError(
            "StageEdit bridge has an invalid Dynamic Crop version"
        )
    if getattr(
        bridge_result.head_materialization,
        "dynamic_crop_version",
        expected_dynamic_crop_version,
    ) != expected_dynamic_crop_version:
        raise StageEditVisualCandidateError(
            "StageEdit visual materialization head version differs from preflight"
        )
    director = bridge_result.learned_section_director
    if (
        director is not None
        and getattr(director, "dynamic_crop_version", None)
        != expected_dynamic_crop_version
    ):
        raise StageEditVisualCandidateError(
            "StageEdit visual materialization Section Director version differs"
        )
    retained = bridge_result.retained_candidates
    if not (1 <= len(retained) <= STAGE_EDIT_VISUAL_CANDIDATE_COUNT
            if quality_gated_single_delivery
            else len(retained) == STAGE_EDIT_VISUAL_CANDIDATE_COUNT):
        raise StageEditVisualCandidateError(
            "StageEdit rendered-proxy promotion requires exactly 8 retained "
            f"candidates; got {len(retained)}"
        )
    if any(
        _retained_dynamic_crop_version(candidate)
        != expected_dynamic_crop_version
        for candidate in retained
    ):
        raise StageEditVisualCandidateError(
            "StageEdit retained candidate version differs from bridge preflight"
        )
    normalized_sources = tuple(Path(path).expanduser().resolve() for path in sources)
    if len(normalized_sources) != len(bridge_result.preflight.source_ids):
        raise StageEditVisualCandidateError(
            "StageEdit visual source list does not match explicit source identities"
        )
    if hybrid_visual_floor_candidate is not None and not isinstance(
        hybrid_visual_floor_candidate,
        MaterializedVisualTimelineCandidate,
    ):
        raise TypeError(
            "hybrid_visual_floor_candidate must be a materialized visual candidate"
        )
    if (
        hybrid_visual_floor_candidate is not None
        and expected_dynamic_crop_version != 3
    ):
        raise StageEditVisualCandidateError(
            "hybrid visual floor requires Dynamic Crop v3"
        )

    if hybrid_visual_floor_candidate is not None:
        learned_materialized: list[StageEditMaterializedVisualCandidate] = []
        learned_rejections: list[dict[str, Any]] = []
        analysis_total = len(retained) + 1
        for position, candidate in enumerate(retained):
            try:
                value = materialize_stage_edit_visual_candidate(
                    candidate,
                    candidate_index=position,
                    review_rank=position,
                    sources=normalized_sources,
                    audio_offsets=audio_offsets,
                    sync_scores=sync_scores,
                    scene_cut_times=scene_cut_times,
                    source_timewarps=source_timewarps,
                    feature_cache_directory=feature_cache_directory,
                    feature_derived_cache_directory=(
                        feature_derived_cache_directory
                    ),
                    require_persistent_feature_cache=(
                        require_persistent_feature_cache
                    ),
                    progress_callback=(
                        (
                            lambda completed, total, position=position: (
                                progress_callback(
                                    position + 1,
                                    analysis_total,
                                    completed,
                                    total,
                                )
                            )
                        )
                        if progress_callback is not None
                        else None
                    ),
                )
            except StageEditVisualCandidateError as exc:
                learned_rejections.append(
                    {
                        "stage_edit_candidate_rank": candidate.rank,
                        "candidate_origin": candidate.candidate_origin,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            learned_materialized.append(replace(value,
                quality_gated_single_delivery=quality_gated_single_delivery))
        required_learned_count = (
            1 if quality_gated_single_delivery else STAGE_EDIT_VISUAL_CANDIDATE_COUNT - 1
        )
        if len(learned_materialized) < required_learned_count:
            raise StageEditVisualCandidateError(
                f"hybrid StageEdit batch requires {required_learned_count} learned paths to pass "
                "the visual hard gates; "
                f"passed={len(learned_materialized)}, "
                f"rejected={len(learned_rejections)}"
            )
        selected_learned = tuple(
            _reindex_stage_edit_materialized_candidate(
                value,
                candidate_index=index,
            )
            for index, value in enumerate(
                learned_materialized[:STAGE_EDIT_VISUAL_CANDIDATE_COUNT - 1]
            )
        )
        hybrid_rank = max(candidate.rank for candidate in retained) + 1
        hybrid_retained = _build_hybrid_visual_floor_retained_candidate(
            bridge_result,
            hybrid_visual_floor_candidate,
            candidate_rank=hybrid_rank,
        )
        hybrid_materialized = _wrap_hybrid_visual_floor_candidate(
            hybrid_visual_floor_candidate,
            hybrid_retained,
            candidate_index=len(selected_learned),
        )
        hybrid_materialized = replace(hybrid_materialized,
            quality_gated_single_delivery=quality_gated_single_delivery)
        if progress_callback is not None:
            progress_callback(analysis_total, analysis_total, 1, 1)
        return StageEditVisualCandidateBatch(
            selected_learned + (hybrid_materialized,),
            learned_candidate_rejections=tuple(learned_rejections),
            quality_gated_single_delivery=quality_gated_single_delivery,
        )

    materialized: list[StageEditMaterializedVisualCandidate] = []
    rejected: list[dict[str, Any]] = []
    for position, candidate in enumerate(retained):
        try:
            value = materialize_stage_edit_visual_candidate(
                candidate,
                candidate_index=position,
                review_rank=position,
                sources=normalized_sources,
                audio_offsets=audio_offsets,
                sync_scores=sync_scores,
                scene_cut_times=scene_cut_times,
                source_timewarps=source_timewarps,
                feature_cache_directory=feature_cache_directory,
                feature_derived_cache_directory=(
                    feature_derived_cache_directory
                ),
                require_persistent_feature_cache=(
                    require_persistent_feature_cache
                ),
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
        except StageEditVisualCandidateError as exc:
            if not quality_gated_single_delivery:
                raise
            rejected.append({"stage_edit_candidate_rank": candidate.rank,
                             "candidate_origin": candidate.candidate_origin,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        materialized.append(_reindex_stage_edit_materialized_candidate(
            replace(value, quality_gated_single_delivery=quality_gated_single_delivery),
            candidate_index=len(materialized)))
    if not materialized:
        raise StageEditVisualCandidateError("No learned path passed visual hard gates")
    return StageEditVisualCandidateBatch(tuple(materialized),
        learned_candidate_rejections=tuple(rejected),
        quality_gated_single_delivery=quality_gated_single_delivery)


@isolated_optional(progress_callbacks=("progress_callback",))
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
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
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
            # Match the learned overlay's actual policy: its editorial track
            # replaces base crop/pan, rather than being composed with them.
            prescribed_render_segments=tuple(
                replace(shot.segment, crop=None, pan_crop=None)
                for shot in candidate.edit.shots
            ),
            audio_offsets=audio_offsets,
            sync_scores=sync_scores,
            scene_cut_times=scene_cut_times,
            source_timewarps=source_timewarps,
            feature_cache_directory=feature_cache_directory,
            feature_derived_cache_directory=feature_derived_cache_directory,
            require_persistent_feature_cache=(
                require_persistent_feature_cache
            ),
            progress_callback=progress_callback,
        )
    except (
        PersistentVisualFeatureCacheInitializationError,
        PersistentVisualFeatureCacheRuntimeError,
    ):
        # A user-selected direct derived-cache root is a production
        # prerequisite, not a per-candidate visual rejection.  Propagate
        # initialization and mid-analysis storage failures before the hybrid
        # loop can retry the same broken source set.
        raise
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
        cadence_profile=(
            "stage_edit_dynamic_crop_"
            f"v{_retained_dynamic_crop_version(candidate)}_prescribed_visual_gate"
        ),
        base_record=base_record,
        retained_candidate=candidate,
    )
    visual_plan = replace(
        base_candidates[0].visual_plan,
        analysis_basis=STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
        selected_candidate_index=candidate_index,
        candidate_generation_basis=STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
        candidate_review_summaries=(global_record.to_dict(),),
        geometry_basis=PRESCRIBED_RENDER_GEOMETRY,
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
            "Dynamic Crop track cannot be applied to a non-positive segment"
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


def dynamic_crop_v3_deterministic_geometry_override_indices(
    render_plan: RenderPlan,
) -> tuple[int, ...]:
    """Return v3 segments whose verified visual geometry must keep authority.

    The ordinary visual derivation has already run face matching, exact-frame
    boundary subject alignment, and renderer-safety validation before the
    StageEdit overlay is applied.  Dynamic Crop v3 must not erase those
    source-specific results merely because its learned trajectory occupies the
    same renderer field.  Face-dissolve sides remain governed by their older,
    explicit suppression contract and are therefore excluded here.
    """

    if not isinstance(render_plan, RenderPlan):
        raise TypeError("render_plan must be a RenderPlan")
    face_dissolve_segments = {
        segment_index
        for boundary_index, transition in enumerate(render_plan.transitions)
        if transition.kind is TransitionKind.FACE_DISSOLVE
        for segment_index in (boundary_index, boundary_index + 1)
    }
    def has_nonidentity_crop(crop):
        return crop is not None and any(not math.isclose(value, expected, abs_tol=1e-9)
            for value, expected in zip((crop.x, crop.y, crop.width, crop.height), (0., 0., 1., 1.)))

    return tuple(
        index
        for index, segment in enumerate(render_plan.segments)
        if index not in face_dissolve_segments
        and (
            has_nonidentity_crop(segment.crop)
            or (segment.pan_crop is not None and any(has_nonidentity_crop(point.crop) for point in segment.pan_crop.keyframes))
            or not _editorial_transform_track_is_identity(segment.editorial_transform)
        )
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
    independently inferred Dynamic Crop trajectory on both sides.
    """

    if not isinstance(materialized, StageEditMaterializedVisualCandidate):
        raise TypeError("materialized must be a StageEdit visual candidate")
    original_segments = tuple(
        shot.segment for shot in materialized.retained_candidate.edit.shots
    )
    if len(original_segments) != len(render_plan.segments):
        raise StageEditVisualCandidateError(
            "Dynamic Crop overlay segment count changed"
        )
    face_dissolve_segments: set[int] = set()
    for boundary_index, transition in enumerate(render_plan.transitions):
        if transition.kind is TransitionKind.FACE_DISSOLVE:
            face_dissolve_segments.update((boundary_index, boundary_index + 1))
    deterministic_geometry_segments = (
        set(dynamic_crop_v3_deterministic_geometry_override_indices(render_plan))
        if materialized.dynamic_crop_version == 3
        else set()
    )
    updated = []
    for index, (original, current) in enumerate(
        zip(original_segments, render_plan.segments, strict=True)
    ):
        if (materialized.retained_candidate.edit.sources[original.source_index].resolve()
                != render_plan.sources[current.source_index].resolve()):
            raise StageEditVisualCandidateError(
                "Dynamic Crop overlay source path changed at segment "
                f"{index}"
            )
        if index in face_dissolve_segments:
            updated.append(replace(current, editorial_transform=None))
            continue
        if index in deterministic_geometry_segments:
            # V3 is a hybrid policy: learned source/cut/framing intent remains
            # authoritative where no stronger source-local geometry exists,
            # while already validated face tracking and boundary alignment
            # retain the exact crop/track that was reviewed on real frames.
            updated.append(current)
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
    "HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN",
    "HYBRID_VISUAL_FLOOR_CONTRACT",
    "STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL",
    "STAGE_EDIT_LEARNED_ONLY_CANDIDATE_POOL",
    "STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT",
    "STAGE_EDIT_VISUAL_CANDIDATE_COUNT",
    "StageEditMaterializedVisualCandidate",
    "StageEditVisualCandidateBatch",
    "StageEditVisualCandidateError",
    "StageEditVisualCandidateRecord",
    "apply_stage_edit_dynamic_crop_to_render_plan",
    "dynamic_crop_v3_deterministic_geometry_override_indices",
    "materialize_stage_edit_visual_candidate",
    "materialize_stage_edit_visual_candidates",
]
