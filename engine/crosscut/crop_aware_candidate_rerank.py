"""Cheap post-crop projection for decoded-proxy candidate admission.

The visual optimizer classifies source frames before face matching.  That is
useful for generation, but it can underrank a candidate whose already-proven
``PanCropTrack`` turns a medium or wide source frame into a non-wide output.
This module projects only deterministic, containment-checked base crops.  It
orders which candidates deserve the expensive decoded proxy review; the proxy
and rendered montage grammar remain the only output authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Sequence

from .face_match import FaceMatchPlan, FaceMatchSegment, build_face_match_plan
from .rendered_montage_grammar_review import RenderedMontageGrammarConfig
from .section_director import SectionPlan, SectionRole
from .timeline import ShotType
from .visual_candidate_materialization import (
    MaterializedVisualTimelineCandidate,
)
from .visual_score import (
    ProfessionalMontageSelectionPolicy,
    VisualTimelineChoice,
    VisualTimelinePlan,
)


CROP_AWARE_CANDIDATE_RERANK_CONTRACT = (
    "deterministic_safe_base_crop_projected_candidate_rerank_v1"
)


def _base_shot_class(shot_type: ShotType) -> str:
    if shot_type in {ShotType.GROUP_WIDE, ShotType.WIDE}:
        return "wide"
    if shot_type is ShotType.CLOSE_UP:
        return "close"
    return "medium"


def _valid_box(value: Any) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        return None
    box = tuple(float(item) for item in value)
    if not (
        0.0 <= box[0] <= 1.0
        and 0.0 <= box[1] <= 1.0
        and 0.0 < box[2] <= 1.0
        and 0.0 < box[3] <= 1.0
        and box[0] + box[2] <= 1.000001
        and box[1] + box[3] <= 1.000001
    ):
        return None
    return box


def _projected_sample_class(
    choice: VisualTimelineChoice,
    match: FaceMatchSegment,
    sample_index: int,
    config: RenderedMontageGrammarConfig,
) -> tuple[str, str, float]:
    shot_types = (
        choice.entry_shot_type or choice.shot_type,
        choice.shot_type,
        choice.exit_shot_type or choice.shot_type,
    )
    base = _base_shot_class(shot_types[sample_index])
    if not match.applied or match.pan_crop is None:
        return base, base, 1.0
    _width, crop_height = match.pan_crop.fixed_crop_size
    zoom = 1.0 / crop_height
    crowd_proven = max(
        int(choice.detected_person_counts[sample_index]),
        int(choice.formation_person_counts[sample_index]),
    ) >= 3
    if match.alignment_subject == "face" and not crowd_proven:
        face = _valid_box(choice.face_samples_xywh[sample_index])
        if face is not None:
            output_height = face[3] * zoom
            if (
                output_height + 1e-9
                >= config.minimum_closeup_proxy_face_height_fraction
            ):
                return base, "close", zoom
            if (
                output_height + 1e-9
                >= config.minimum_nonwide_proxy_face_height_fraction
            ):
                return base, "medium", zoom
    if match.alignment_subject == "silhouette" and not crowd_proven:
        silhouette = _valid_box(choice.silhouette_samples_xywh[sample_index])
        if silhouette is not None:
            output_width = silhouette[2] * zoom
            output_height = silhouette[3] * zoom
            if output_height >= 0.54 - 1e-9 and output_width <= 0.54 + 1e-9:
                return base, "medium", zoom
    return base, base, zoom


def _role_thresholds(
    policy: ProfessionalMontageSelectionPolicy,
    role: SectionRole,
) -> tuple[float, float, float | None] | None:
    if role is SectionRole.CHORUS_LIKE:
        return (
            policy.highlight_minimum_closeup_fraction,
            policy.highlight_minimum_nonwide_fraction,
            policy.highlight_minimum_source_switches_per_minute,
        )
    if role is SectionRole.BUILD:
        return (
            policy.build_minimum_closeup_fraction,
            policy.build_minimum_nonwide_fraction,
            None,
        )
    if role is SectionRole.BRIDGE_LIKE:
        return (
            policy.bridge_minimum_closeup_fraction,
            policy.bridge_minimum_nonwide_fraction,
            None,
        )
    return None


@dataclass(frozen=True, slots=True)
class CropAwareCandidateEvaluation:
    candidate_index: int
    cadence_profile: str
    input_rank: int
    face_match_plan: FaceMatchPlan = field(repr=False)
    metadata_role_attainment_score: float
    minimum_highlight_metadata_attainment_score: float
    all_highlight_metadata_floors_passed: bool
    safe_crop_applied_segment_count: int
    safe_crop_applied_duration_seconds: float
    highlight_safe_crop_applied_duration_seconds: float
    crop_upgraded_sample_count: int
    highlight_crop_upgraded_duration_seconds: float
    sections: tuple[dict[str, Any], ...]
    projected_rank: int = -1

    def __post_init__(self) -> None:
        if self.candidate_index < 0 or self.input_rank < 0:
            raise ValueError("candidate indices and ranks must be non-negative")
        if self.projected_rank < -1:
            raise ValueError("projected_rank cannot be below -1")
        for value in (
            self.metadata_role_attainment_score,
            self.minimum_highlight_metadata_attainment_score,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("candidate attainment must be normalized")

    @property
    def priority(self) -> tuple[Any, ...]:
        return (
            self.all_highlight_metadata_floors_passed,
            self.minimum_highlight_metadata_attainment_score,
            self.metadata_role_attainment_score,
            self.highlight_crop_upgraded_duration_seconds,
            self.highlight_safe_crop_applied_duration_seconds,
            -self.input_rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "cadence_profile": self.cadence_profile,
            "input_rank": self.input_rank,
            "projected_rank": self.projected_rank,
            "metadata_role_attainment_score": (
                self.metadata_role_attainment_score
            ),
            "minimum_highlight_metadata_attainment_score": (
                self.minimum_highlight_metadata_attainment_score
            ),
            "all_highlight_metadata_floors_passed": (
                self.all_highlight_metadata_floors_passed
            ),
            "safe_crop_applied_segment_count": (
                self.safe_crop_applied_segment_count
            ),
            "continuous_run_stitched_segment_count": (
                self.face_match_plan.continuous_run_stitched_segment_count
            ),
            "nonwide_rescue_applied_count": (
                self.face_match_plan.nonwide_rescue_applied_count
            ),
            "safe_crop_applied_duration_seconds": (
                self.safe_crop_applied_duration_seconds
            ),
            "highlight_safe_crop_applied_duration_seconds": (
                self.highlight_safe_crop_applied_duration_seconds
            ),
            "crop_upgraded_sample_count": self.crop_upgraded_sample_count,
            "highlight_crop_upgraded_duration_seconds": (
                self.highlight_crop_upgraded_duration_seconds
            ),
            "sections": [dict(item) for item in self.sections],
            "decoded_output_authority": False,
        }


def evaluate_crop_aware_visual_plan(
    visual_plan: VisualTimelinePlan,
    *,
    candidate_index: int,
    cadence_profile: str,
    input_rank: int,
    source_aspect_ratios: Sequence[float],
    section_plan: SectionPlan,
    policy: ProfessionalMontageSelectionPolicy,
    rendered_config: RenderedMontageGrammarConfig,
) -> CropAwareCandidateEvaluation:
    """Project safe base crops, without claiming decoded-output evidence."""

    face_match_plan = build_face_match_plan(
        visual_timeline=visual_plan,
        source_aspect_ratios=source_aspect_ratios,
        section_plan=section_plan,
    )
    matches = {item.segment_index: item for item in face_match_plan.segments}
    if len(matches) != len(visual_plan.choices):
        raise ValueError("face-match projection did not cover the visual plan")

    choices_by_section: dict[int, list[VisualTimelineChoice]] = {}
    for choice in visual_plan.choices:
        directive = section_plan.directive_at(choice.canonical_master_start)
        choices_by_section.setdefault(directive.section_index, []).append(choice)
    normalized_scores: list[float] = []
    highlight_scores: list[float] = []
    highlight_passes: list[bool] = []
    sections: list[dict[str, Any]] = []
    safe_crop_seconds = 0.0
    highlight_safe_crop_seconds = 0.0
    upgraded_sample_count = 0
    highlight_upgraded_seconds = 0.0
    for directive in section_plan.sections:
        thresholds = _role_thresholds(policy, directive.role)
        choices = choices_by_section.get(directive.section_index, [])
        if thresholds is None or not choices:
            continue
        minimum_close, minimum_nonwide, minimum_switches = thresholds
        duration = sum(choice.duration_seconds for choice in choices)
        close_seconds = 0.0
        nonwide_seconds = 0.0
        base_close_seconds = 0.0
        base_nonwide_seconds = 0.0
        section_safe_crop_seconds = 0.0
        section_upgraded_seconds = 0.0
        section_upgraded_count = 0
        for choice in choices:
            match = matches[choice.segment_index]
            span = choice.duration_seconds
            if match.applied:
                safe_crop_seconds += span
                section_safe_crop_seconds += span
            for sample_index in range(3):
                base, projected, _zoom = _projected_sample_class(
                    choice,
                    match,
                    sample_index,
                    rendered_config,
                )
                weight = span / 3.0
                base_close_seconds += weight * (base == "close")
                base_nonwide_seconds += weight * (base in {"close", "medium"})
                close_seconds += weight * (projected == "close")
                nonwide_seconds += weight * (
                    projected in {"close", "medium"}
                )
                upgraded = (
                    base == "wide" and projected in {"medium", "close"}
                ) or (base == "medium" and projected == "close")
                if upgraded:
                    upgraded_sample_count += 1
                    section_upgraded_count += 1
                    section_upgraded_seconds += weight
        close_fraction = close_seconds / duration
        nonwide_fraction = nonwide_seconds / duration
        switches = sum(
            left.source_index != right.source_index
            for left, right in zip(choices, choices[1:])
        )
        switches_per_minute = switches * 60.0 / duration
        requirements = [
            min(1.0, close_fraction / max(minimum_close, 1e-9)),
            min(1.0, nonwide_fraction / max(minimum_nonwide, 1e-9)),
        ]
        passed = bool(
            close_fraction + 1e-9 >= minimum_close
            and nonwide_fraction + 1e-9 >= minimum_nonwide
            and (
                minimum_switches is None
                or switches_per_minute + 1e-9 >= minimum_switches
            )
        )
        if minimum_switches is not None:
            requirements.append(
                min(1.0, switches_per_minute / max(minimum_switches, 1e-9))
            )
            highlight_scores.append(sum(requirements) / len(requirements))
            highlight_passes.append(passed)
            highlight_safe_crop_seconds += section_safe_crop_seconds
            highlight_upgraded_seconds += section_upgraded_seconds
        attainment = sum(requirements) / len(requirements)
        normalized_scores.append(attainment)
        sections.append(
            {
                "section_index": directive.section_index,
                "role": directive.role.value,
                "duration_seconds": duration,
                "pre_crop_closeup_duration_fraction": (
                    base_close_seconds / duration
                ),
                "pre_crop_nonwide_duration_fraction": (
                    base_nonwide_seconds / duration
                ),
                "projected_closeup_duration_fraction": close_fraction,
                "projected_nonwide_duration_fraction": nonwide_fraction,
                "source_switch_count": switches,
                "source_switches_per_minute": switches_per_minute,
                "minimum_closeup_duration_fraction": minimum_close,
                "minimum_nonwide_duration_fraction": minimum_nonwide,
                "minimum_source_switches_per_minute": minimum_switches,
                "safe_crop_applied_duration_seconds": section_safe_crop_seconds,
                "crop_upgraded_sample_count": section_upgraded_count,
                "crop_upgraded_duration_seconds": section_upgraded_seconds,
                "projected_metadata_floor_passed": passed,
            }
        )
    return CropAwareCandidateEvaluation(
        candidate_index=candidate_index,
        cadence_profile=cadence_profile,
        input_rank=input_rank,
        face_match_plan=face_match_plan,
        metadata_role_attainment_score=(
            sum(normalized_scores) / len(normalized_scores)
            if normalized_scores
            else 0.0
        ),
        minimum_highlight_metadata_attainment_score=(
            min(highlight_scores) if highlight_scores else 0.0
        ),
        all_highlight_metadata_floors_passed=bool(
            highlight_passes and all(highlight_passes)
        ),
        safe_crop_applied_segment_count=face_match_plan.applied_count,
        safe_crop_applied_duration_seconds=safe_crop_seconds,
        highlight_safe_crop_applied_duration_seconds=highlight_safe_crop_seconds,
        crop_upgraded_sample_count=upgraded_sample_count,
        highlight_crop_upgraded_duration_seconds=highlight_upgraded_seconds,
        sections=tuple(sections),
    )


@dataclass(frozen=True, slots=True)
class CropAwareCandidateRerankResult:
    ordered_candidates: tuple[MaterializedVisualTimelineCandidate, ...] = field(
        repr=False
    )
    evaluations: tuple[CropAwareCandidateEvaluation, ...]
    selected_candidate_indices: tuple[int, ...]
    maximum_candidates: int

    def __post_init__(self) -> None:
        ordered_indices = tuple(item.candidate_index for item in self.ordered_candidates)
        if len(set(ordered_indices)) != len(ordered_indices):
            raise ValueError("crop-aware rerank received duplicate candidates")
        if set(ordered_indices) != {
            item.candidate_index for item in self.evaluations
        }:
            raise ValueError("crop-aware evaluations do not bind every candidate")
        if self.selected_candidate_indices != ordered_indices[: self.maximum_candidates]:
            raise ValueError("selected crop-aware candidates must be the ranked prefix")

    @property
    def selected_candidates(
        self,
    ) -> tuple[MaterializedVisualTimelineCandidate, ...]:
        return self.ordered_candidates[: self.maximum_candidates]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": CROP_AWARE_CANDIDATE_RERANK_CONTRACT,
            "analysis_basis": (
                "pre_render_safe_pan_crop_projected_face_and_silhouette_"
                "shot_scale_v1"
            ),
            "candidate_count": len(self.ordered_candidates),
            "maximum_selected_candidate_count": self.maximum_candidates,
            "selected_candidate_indices": list(self.selected_candidate_indices),
            "ordered_candidate_indices": [
                item.candidate_index for item in self.ordered_candidates
            ],
            "evaluations": [item.to_dict() for item in self.evaluations],
            "decoded_output_authority": False,
            "decoded_proxy_review_required": True,
            "rendered_montage_grammar_required": True,
            "unrendered_candidate_can_be_promoted": False,
        }


def rerank_materialized_visual_candidates_for_safe_crops(
    candidates: Sequence[MaterializedVisualTimelineCandidate],
    *,
    source_aspect_ratios: Sequence[float],
    section_plan: SectionPlan,
    policy: ProfessionalMontageSelectionPolicy,
    rendered_config: RenderedMontageGrammarConfig,
    maximum_candidates: int = 8,
) -> CropAwareCandidateRerankResult:
    """Prioritize crop-capable candidates before expensive proxy decoding."""

    values = tuple(candidates)
    if not values:
        raise ValueError("crop-aware rerank needs at least one candidate")
    if not 1 <= maximum_candidates <= len(values):
        raise ValueError("maximum_candidates is outside the candidate pool")
    if len({item.candidate_index for item in values}) != len(values):
        raise ValueError("crop-aware candidate indices must be unique")
    evaluations = tuple(
        evaluate_crop_aware_visual_plan(
            item.visual_plan,
            candidate_index=item.candidate_index,
            cadence_profile=item.cadence_profile,
            input_rank=input_rank,
            source_aspect_ratios=source_aspect_ratios,
            section_plan=section_plan,
            policy=policy,
            rendered_config=rendered_config,
        )
        for input_rank, item in enumerate(values)
    )
    by_index = {item.candidate_index: item for item in evaluations}

    qualified = sorted(
        (
            item
            for item in values
            if by_index[item.candidate_index].all_highlight_metadata_floors_passed
        ),
        key=lambda item: by_index[item.candidate_index].priority,
        reverse=True,
    )
    ordered = list(qualified)
    used = {item.candidate_index for item in ordered}
    remaining_by_profile: dict[
        str, list[MaterializedVisualTimelineCandidate]
    ] = {}
    for item in values:
        if item.candidate_index in used:
            continue
        remaining_by_profile.setdefault(item.cadence_profile, []).append(item)
    for profile_values in remaining_by_profile.values():
        profile_values.sort(
            key=lambda item: by_index[item.candidate_index].priority,
            reverse=True,
        )
    profile_order = tuple(
        profile
        for profile in ("balanced", "reference", "dense_safe")
        if profile in remaining_by_profile
    ) + tuple(
        sorted(
            set(remaining_by_profile)
            - {"balanced", "reference", "dense_safe"}
        )
    )
    rank = 0
    while len(ordered) < len(values):
        appended = False
        for profile in profile_order:
            profile_values = remaining_by_profile[profile]
            if rank < len(profile_values):
                ordered.append(profile_values[rank])
                appended = True
        if not appended:
            break
        rank += 1
    if len(ordered) != len(values):
        raise RuntimeError("crop-aware rerank lost candidates")
    projected_rank = {
        item.candidate_index: position for position, item in enumerate(ordered)
    }
    ranked_evaluations = tuple(
        replace(item, projected_rank=projected_rank[item.candidate_index])
        for item in sorted(
            evaluations,
            key=lambda item: projected_rank[item.candidate_index],
        )
    )
    return CropAwareCandidateRerankResult(
        ordered_candidates=tuple(ordered),
        evaluations=ranked_evaluations,
        selected_candidate_indices=tuple(
            item.candidate_index for item in ordered[:maximum_candidates]
        ),
        maximum_candidates=maximum_candidates,
    )


__all__ = [
    "CROP_AWARE_CANDIDATE_RERANK_CONTRACT",
    "CropAwareCandidateEvaluation",
    "CropAwareCandidateRerankResult",
    "evaluate_crop_aware_visual_plan",
    "rerank_materialized_visual_candidates_for_safe_crops",
]
