"""Frame scoring for the group cross-edit draft.

The engine does not identify IVE members in broadcast footage. It combines
technical image evidence with optional on-device human pose and aggregate
silhouette models. Entry and exit descriptors remain separate so a cut compares
the actual outgoing and incoming frames.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .face_identity import SFACE_BACKEND, extract_sface_embedding
from .pose import (
    FormationPersonDescriptor,
    SubjectGeometryDescriptor,
    describe_subject_geometry,
    model_asset_status as pose_model_asset_status,
)
from .timeline import (
    BOUNDARY_SIDE_MAX_POSE_MISMATCH,
    BOUNDARY_SIDE_MAX_SCALE_DELTA,
    BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH,
    BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH,
    BoundarySideStability,
    NoCompleteTimelinePath,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineConfig,
    TimelineResult,
    comparable_group_formation_mismatch,
    compile_timeline_no_good_constraint_index,
    cross_source_layout_mismatch,
    face_geometry_minimax_mismatch,
    select_timeline_candidates,
    timeline_transition_hard_gates_admit,
)
# Selector-only policy constants stay in a separate import node so changing
# editorial policy cannot invalidate persisted frame descriptors.
from .timeline import (
    INTENTIONAL_PHRASE_ANCHOR_MAX_FOCUS_MISMATCH,
    INTENTIONAL_PHRASE_ANCHOR_MAX_POSE_MISMATCH,
    INTENTIONAL_PHRASE_ANCHOR_MAX_SCALE_MISMATCH,
    INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS,
    PRODUCTION_MAX_LAYOUT_MISMATCH,
    PRODUCTION_MAX_SCALE_MISMATCH,
    PRODUCTION_MAX_SILHOUETTE_MISMATCH,
)
from .timeline_review import TimelineReview, review_timeline
from .timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
)
from .timewarp import TimeWarpMap
from .section_director import SectionPlan, SectionRole
from .visual_feature_cache import (
    AnalyzerFingerprint,
    CanonicalFrameIdentity,
    DescriptorRequestIdentity,
    DescriptorRequestLocator,
    PersistentVisualFeatureCache,
    PersistentVisualFeatureCacheIOError,
    build_analyzer_fingerprint,
    python_symbol_dependency_sha256,
)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _source_clock_maps(
    source_timewarps: Sequence[TimeWarpMap | None] | None,
    *,
    audio_offsets: Sequence[float],
    source_count: int,
) -> tuple[TimeWarpMap | None, ...]:
    """Validate optional per-source maps while retaining the scalar contract."""

    if source_timewarps is None:
        return (None,) * source_count
    if len(source_timewarps) != source_count:
        raise ValueError("source_timewarps must match the source count")
    normalized = tuple(source_timewarps)
    for index, timewarp in enumerate(normalized):
        if timewarp is None:
            continue
        if not isinstance(timewarp, TimeWarpMap):
            raise TypeError("source_timewarps entries must be TimeWarpMap or None")
        if not math.isclose(
            timewarp.global_offset_seconds,
            float(audio_offsets[index]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"source_timewarps[{index}] global offset does not match audio_offsets"
            )
    return normalized


def _source_clock_time(
    master_seconds: float,
    *,
    source_index: int,
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
) -> float:
    timewarp = source_timewarps[source_index]
    if timewarp is None:
        return master_seconds + float(audio_offsets[source_index])
    return timewarp.master_to_source(master_seconds)


def _master_clock_time(
    source_seconds: float,
    *,
    source_index: int,
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
) -> float:
    """Map detector timestamps back to the canonical song clock exactly."""

    timewarp = source_timewarps[source_index]
    if timewarp is None:
        return source_seconds - float(audio_offsets[source_index])
    return timewarp.source_to_master(source_seconds)


@dataclass(frozen=True, slots=True)
class VisualDiversityConfig:
    """Safe preference strength for the metadata source-coverage order."""

    # Coverage is a whole-song preference, never permission to take an
    # attention-jumping cut.  The original stronger cycle bonus produced a
    # technically diverse but visibly random source rotation.
    preferred_source_bonus: float = 1.30
    maximum_preferred_score_sacrifice: float = 0.16

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.preferred_source_bonus)
            or self.preferred_source_bonus < 0.0
        ):
            raise ValueError("preferred source bonus must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_preferred_score_sacrifice)
            or self.maximum_preferred_score_sacrifice < 0.0
        ):
            raise ValueError(
                "maximum preferred score sacrifice must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class ProfessionalMontageSelectionPolicy:
    """Human-reference shot-scale/cadence targets for candidate generation.

    These values are soft optimizer controls only.  They cannot admit an
    unsafe visual edge, and every resulting candidate still has to pass the
    decoded-output montage grammar independently.
    """

    highlight_minimum_closeup_fraction: float
    highlight_preferred_closeup_fraction: float
    highlight_minimum_nonwide_fraction: float
    highlight_preferred_nonwide_fraction: float
    build_minimum_closeup_fraction: float
    build_preferred_closeup_fraction: float
    build_minimum_nonwide_fraction: float
    build_preferred_nonwide_fraction: float
    bridge_minimum_closeup_fraction: float
    bridge_preferred_closeup_fraction: float
    bridge_minimum_nonwide_fraction: float
    bridge_preferred_nonwide_fraction: float
    highlight_minimum_source_switches_per_minute: float
    highlight_preferred_source_switches_per_minute: float
    minimum_early_wide_anchor_fraction: float
    minimum_phrase_entry_wide_anchor_fraction: float

    def __post_init__(self) -> None:
        fraction_pairs = (
            (
                "highlight_closeup",
                self.highlight_minimum_closeup_fraction,
                self.highlight_preferred_closeup_fraction,
            ),
            (
                "highlight_nonwide",
                self.highlight_minimum_nonwide_fraction,
                self.highlight_preferred_nonwide_fraction,
            ),
            (
                "build_closeup",
                self.build_minimum_closeup_fraction,
                self.build_preferred_closeup_fraction,
            ),
            (
                "build_nonwide",
                self.build_minimum_nonwide_fraction,
                self.build_preferred_nonwide_fraction,
            ),
            (
                "bridge_closeup",
                self.bridge_minimum_closeup_fraction,
                self.bridge_preferred_closeup_fraction,
            ),
            (
                "bridge_nonwide",
                self.bridge_minimum_nonwide_fraction,
                self.bridge_preferred_nonwide_fraction,
            ),
        )
        for name, minimum, preferred in fraction_pairs:
            if not (
                math.isfinite(minimum)
                and math.isfinite(preferred)
                and 0.0 <= minimum <= preferred <= 1.0
            ):
                raise ValueError(f"{name} fractions are invalid")
        if (
            self.highlight_preferred_closeup_fraction
            > self.highlight_preferred_nonwide_fraction + 1e-12
            or self.build_preferred_closeup_fraction
            > self.build_preferred_nonwide_fraction + 1e-12
            or self.bridge_preferred_closeup_fraction
            > self.bridge_preferred_nonwide_fraction + 1e-12
        ):
            raise ValueError("closeup targets cannot exceed nonwide targets")
        if not (
            math.isfinite(self.highlight_minimum_source_switches_per_minute)
            and math.isfinite(self.highlight_preferred_source_switches_per_minute)
            and 0.0
            < self.highlight_minimum_source_switches_per_minute
            <= self.highlight_preferred_source_switches_per_minute
            <= 60.0
        ):
            raise ValueError("highlight source-switch targets are invalid")
        for name in (
            "minimum_early_wide_anchor_fraction",
            "minimum_phrase_entry_wide_anchor_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in 0...1")

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "contract": (
                "human_reference_role_conditioned_visual_candidate_soft_"
                "controls_v1"
            ),
            **{
                name: float(getattr(self, name))
                for name in self.__dataclass_fields__
            },
            "hard_visual_gate_relaxed": False,
            "decoded_output_montage_gate_required": True,
        }


@dataclass(frozen=True, slots=True)
class _CadenceProfile:
    name: str
    config: TimelineConfig


# The selected-shot fifth percentile across the existing IVE BLACKHOLE, BANG
# BANG, and XOXZ production analyses is 0.854--0.880.  A 0.84 shoulder targets
# only the visibly weakest tail while leaving at least 95% of those real edits
# untouched; this remains a soft score penalty, never a source rejection.
PROFESSIONAL_QUALITY_SOFT_FLOOR = 0.84
PROFESSIONAL_QUALITY_SOFT_FLOOR_PENALTY = 0.65
# Keep selector cadence inside the independent production review envelope.
# At the current decision-span quantization, 5.2 seconds yields a 4.90-second
# observed run ceiling and 12.8--14.1 switches/minute on the XOXZ holdout,
# versus the reviewer's 12.5 hard minimum and 13.5--17.5 preferred band.
PROFESSIONAL_MAX_SOURCE_HOLD_SECONDS = 5.2


def _quality_soft_floor_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "scope": "per_shot_entry_mid_exit_mean_quality",
        "floor": PROFESSIONAL_QUALITY_SOFT_FLOOR,
        "maximum_penalty": PROFESSIONAL_QUALITY_SOFT_FLOOR_PENALTY,
        "curve": "normalized_quadratic_deficit_below_floor",
        "calibration": (
            "IVE_BLACKHOLE_BANG_BANG_XOXZ_selected_shot_p05_0.854_to_0.880"
        ),
        "above_floor_emission_unchanged": True,
        "hard_rejection": False,
    }


def _safe_cadence_profiles(
    diversity: VisualDiversityConfig,
) -> tuple[_CadenceProfile, ...]:
    """Build soft cadence variants around one shared set of visual hard gates."""

    balanced = TimelineConfig(
        quality_weight=0.82,
        quality_soft_floor=PROFESSIONAL_QUALITY_SOFT_FLOOR,
        quality_soft_floor_penalty=(
            PROFESSIONAL_QUALITY_SOFT_FLOOR_PENALTY
        ),
        sync_weight=0.55,
        composition_weight=0.30,
        boundary_weight=0.62,
        continuity_bonus=0.30,
        cut_penalty=0.22,
        early_cut_penalty=0.48,
        long_hold_penalty=0.22,
        pose_transition_weight=1.35,
        silhouette_transition_weight=1.65,
        motion_transition_weight=0.42,
        scale_transition_weight=1.75,
        color_transition_weight=0.10,
        face_geometry_transition_weight=0.72,
        focus_position_transition_weight=1.80,
        subject_geometry_transition_weight=1.65,
        person_count_transition_weight=0.60,
        person_layout_transition_weight=2.10,
        formation_transition_weight=2.45,
        shot_type_transition_weight=1.20,
        maximum_cross_source_scale_delta=PRODUCTION_MAX_SCALE_MISMATCH,
        # The decoded +/-3-frame boundary repair explicitly corrects focal
        # subject jumps. Keep the strong transition penalty, then require the
        # unchanged post-repair 0.24 review gate instead of deleting those
        # potentially useful close-shot edges before repair can inspect them.
        maximum_cross_source_focus_mismatch=None,
        maximum_cross_source_layout_mismatch=PRODUCTION_MAX_LAYOUT_MISMATCH,
        # Screen-space formation is strongly perspective-dependent across
        # broadcasters. Keep its 2.45 transition penalty, then require the
        # decoded +/-3-frame repair and final reviewer to satisfy the unchanged
        # 0.30 gate instead of disconnecting the selector graph prematurely.
        maximum_cross_source_formation_mismatch=None,
        maximum_cross_source_silhouette_mismatch=(
            PRODUCTION_MAX_SILHOUETTE_MISMATCH
        ),
        minimum_cross_source_face_similarity=0.36,
        require_cross_source_visual_evidence=True,
        allow_intentional_phrase_anchor_transitions=True,
        maximum_boundary_side_scale_delta=BOUNDARY_SIDE_MAX_SCALE_DELTA,
        maximum_boundary_side_structure_mismatch=(
            BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH
        ),
        maximum_boundary_side_pose_mismatch=BOUNDARY_SIDE_MAX_POSE_MISMATCH,
        maximum_boundary_side_shot_type_mismatch=(
            BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH
        ),
        require_cross_source_boundary_side_stability=True,
        minimum_hold_segments=2,
        preferred_max_hold_segments=6,
        preferred_max_source_hold_segments=5,
        maximum_same_source_hold_segments=6,
        preferred_max_source_hold_seconds=4.2,
        maximum_same_source_hold_seconds=(
            PROFESSIONAL_MAX_SOURCE_HOLD_SECONDS
        ),
        source_hold_penalty=0.22,
        source_switch_bonus=0.20,
        ping_pong_penalty=0.42,
        maximum_switch_score_sacrifice=0.65,
        preferred_source_bonus=diversity.preferred_source_bonus,
        preferred_source_max_score_sacrifice=(
            diversity.maximum_preferred_score_sacrifice
        ),
    )
    return (
        _CadenceProfile("balanced", balanced),
        _CadenceProfile(
            "reference",
            replace(
                balanced,
                continuity_bonus=0.18,
                cut_penalty=0.18,
                early_cut_penalty=0.38,
                preferred_max_source_hold_seconds=3.8,
                source_hold_penalty=0.32,
                source_switch_bonus=0.48,
                ping_pong_penalty=0.52,
                maximum_switch_score_sacrifice=0.75,
            ),
        ),
        _CadenceProfile(
            "dense_safe",
            replace(
                balanced,
                continuity_bonus=0.12,
                cut_penalty=0.14,
                early_cut_penalty=0.30,
                preferred_max_source_hold_seconds=3.2,
                source_hold_penalty=0.44,
                source_switch_bonus=0.85,
                ping_pong_penalty=0.68,
                maximum_switch_score_sacrifice=0.90,
            ),
        ),
    )


def _timeline_path_signature(result: TimelineResult) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        or f"{candidate.segment_index}:{candidate.source_id}"
        for candidate in result.candidates
    )


def _review_ranked_candidate_indices(
    candidate_reviews: Sequence[TimelineReview],
) -> tuple[int, ...]:
    """Return the reviewer order used for both selection and materialization.

    The final index term is intentionally part of the sort key.  Equal
    candidates therefore keep their deterministic generation order instead of
    depending on object identity or an unordered container.
    """

    if not candidate_reviews:
        raise ValueError("candidate_reviews cannot be empty")
    return tuple(
        sorted(
            range(len(candidate_reviews)),
            key=lambda index: (
                candidate_reviews[index].passed,
                -len(candidate_reviews[index].hard_failures),
                candidate_reviews[index].weighted_score,
                -len(candidate_reviews[index].warnings),
                -index,
            ),
            reverse=True,
        )
    )


def _role_fraction_targets(
    policy: ProfessionalMontageSelectionPolicy,
    *,
    role: SectionRole,
    cadence_profile: str,
) -> tuple[float, float] | None:
    if role is SectionRole.CHORUS_LIKE:
        minimum_close = policy.highlight_minimum_closeup_fraction
        preferred_close = policy.highlight_preferred_closeup_fraction
        minimum_nonwide = policy.highlight_minimum_nonwide_fraction
        preferred_nonwide = policy.highlight_preferred_nonwide_fraction
    elif role is SectionRole.BUILD:
        minimum_close = policy.build_minimum_closeup_fraction
        preferred_close = policy.build_preferred_closeup_fraction
        minimum_nonwide = policy.build_minimum_nonwide_fraction
        preferred_nonwide = policy.build_preferred_nonwide_fraction
    elif role is SectionRole.BRIDGE_LIKE:
        minimum_close = policy.bridge_minimum_closeup_fraction
        preferred_close = policy.bridge_preferred_closeup_fraction
        minimum_nonwide = policy.bridge_minimum_nonwide_fraction
        preferred_nonwide = policy.bridge_preferred_nonwide_fraction
    else:
        return None
    if cadence_profile == "balanced":
        close_lift, nonwide_lift = 0.08, 0.00
    elif cadence_profile == "reference":
        close_lift, nonwide_lift = 0.16, 0.08
    elif cadence_profile == "dense_safe":
        close_lift, nonwide_lift = 0.24, 0.16
    else:
        raise ValueError("unknown safe cadence profile")
    close = min(0.82, max(minimum_close, preferred_close + close_lift))
    nonwide = min(
        0.90,
        max(minimum_nonwide, preferred_nonwide + nonwide_lift, close),
    )
    return close, nonwide


def _evenly_spaced_positions(
    values: Sequence[int],
    count: int,
) -> set[int]:
    ordered = tuple(values)
    if count <= 0 or not ordered:
        return set()
    if count >= len(ordered):
        return set(ordered)
    if count == 1:
        return {ordered[0]}
    return {
        ordered[
            min(
                len(ordered) - 1,
                int(round(index * (len(ordered) - 1) / (count - 1))),
            )
        ]
        for index in range(count)
    }


def _spread_distance(position: int, selected: set[int]) -> int:
    if not selected:
        return 1 << 30
    return min(abs(position - value) for value in selected)


def _wide_positions_for_duration_target(
    values: Sequence[int],
    *,
    duration_by_position: Mapping[int, float],
    maximum_wide_seconds: float,
    mandatory_positions: set[int],
    preference_by_position: Mapping[int, float] | None = None,
) -> set[int]:
    """Spread wide anchors without exceeding the profile duration budget."""

    ordered = tuple(values)
    preferences = dict(preference_by_position or {})
    selected = {value for value in mandatory_positions if value in ordered}
    selected_seconds = sum(duration_by_position[value] for value in selected)
    while True:
        eligible = tuple(
            value
            for value in ordered
            if value not in selected
            and selected_seconds + duration_by_position[value]
            <= maximum_wide_seconds + 1e-9
        )
        if not eligible:
            break
        choice = max(
            eligible,
            key=lambda value: (
                float(preferences.get(value, 0.0)),
                selected_seconds + duration_by_position[value],
                _spread_distance(value, selected),
                -value,
            ),
        )
        selected.add(choice)
        selected_seconds += duration_by_position[choice]
    return selected


def _close_positions_for_duration_target(
    values: Sequence[int],
    *,
    duration_by_position: Mapping[int, float],
    minimum_close_seconds: float,
    seeded_positions: set[int],
    preference_by_position: Mapping[int, float] | None = None,
) -> set[int]:
    """Spread close targets until their duration meets the profile floor."""

    ordered = tuple(values)
    preferences = dict(preference_by_position or {})
    selected = {value for value in seeded_positions if value in ordered}
    selected_seconds = sum(duration_by_position[value] for value in selected)
    while selected_seconds + 1e-9 < minimum_close_seconds:
        available = tuple(value for value in ordered if value not in selected)
        if not available:
            break
        choice = min(
            available,
            key=lambda value: (
                -float(preferences.get(value, 0.0)),
                abs(
                    selected_seconds
                    + duration_by_position[value]
                    - minimum_close_seconds
                ),
                -_spread_distance(value, selected),
                value,
            ),
        )
        selected.add(choice)
        selected_seconds += duration_by_position[choice]
    return selected


def _preferred_spread_positions(
    values: Sequence[int],
    count: int,
    *,
    preference_by_position: Mapping[int, float],
    seeded_positions: set[int] | None = None,
) -> set[int]:
    """Choose evidence-backed positions while retaining temporal spread."""

    ordered = tuple(values)
    selected = {
        value for value in (seeded_positions or set()) if value in ordered
    }
    target_count = min(len(ordered), max(0, int(count)))
    while len(selected) < target_count:
        available = tuple(value for value in ordered if value not in selected)
        if not available:
            break
        choice = max(
            available,
            key=lambda value: (
                float(preference_by_position.get(value, 0.0)),
                _spread_distance(value, selected),
                -value,
            ),
        )
        selected.add(choice)
    return selected


def _complete_path_reachable_candidate_keys(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    timeline_config: TimelineConfig,
    forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
) -> set[tuple[int, str]]:
    """Find candidates lying on at least one full hard-gate-admitted path."""

    segments = tuple(tuple(segment) for segment in candidates_by_segment)
    if not segments or any(not segment for segment in segments):
        raise ValueError("reachability requires non-empty candidate segments")
    forward: list[set[int]] = [set(range(len(segments[0])))]
    for position in range(1, len(segments)):
        reachable: set[int] = set()
        for current_index, current in enumerate(segments[position]):
            if any(
                timeline_transition_hard_gates_admit(
                    segments[position - 1][previous_index],
                    current,
                    config=timeline_config,
                )
                and (
                    not forbidden_boundary_edges
                    or build_timeline_no_good_edge_key(
                        position - 1,
                        segments[position - 1][previous_index],
                        current,
                    )
                    not in forbidden_boundary_edges
                )
                for previous_index in forward[-1]
            ):
                reachable.add(current_index)
        if not reachable:
            raise NoCompleteTimelinePath(
                "visual hard-gate graph has no complete forward path",
                reason_code="visual_hard_gate_forward_path_exhausted",
                segment_index=segments[position][0].segment_index,
            )
        forward.append(reachable)
    backward: list[set[int]] = [set() for _ in segments]
    backward[-1] = set(range(len(segments[-1])))
    for position in range(len(segments) - 2, -1, -1):
        reachable = set()
        for previous_index, previous in enumerate(segments[position]):
            if any(
                timeline_transition_hard_gates_admit(
                    previous,
                    segments[position + 1][current_index],
                    config=timeline_config,
                )
                and (
                    not forbidden_boundary_edges
                    or build_timeline_no_good_edge_key(
                        position,
                        previous,
                        segments[position + 1][current_index],
                    )
                    not in forbidden_boundary_edges
                )
                for current_index in backward[position + 1]
            ):
                reachable.add(previous_index)
        if not reachable:
            raise NoCompleteTimelinePath(
                "visual hard-gate graph has no complete backward path",
                reason_code="visual_hard_gate_backward_path_exhausted",
                segment_index=segments[position][0].segment_index,
            )
        backward[position] = reachable
    return {
        (candidate.segment_index, candidate.source_id)
        for position, segment in enumerate(segments)
        for candidate_index, candidate in enumerate(segment)
        if candidate_index in forward[position]
        and candidate_index in backward[position]
    }


def _role_target_categories(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    section_plan: SectionPlan,
    phrase_times: Sequence[float],
    policy: ProfessionalMontageSelectionPolicy,
    cadence_profile: str,
    subject_metadata_by_candidate: Mapping[
        str, "_CandidateSubjectMetadata"
    ] | None = None,
    reachable_candidate_keys: set[tuple[int, str]] | None = None,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Build an auditable wide/close/nonwide motif from profile fractions."""

    if not candidates_by_segment:
        raise ValueError("role-conditioned selection needs timeline segments")
    phrases = tuple(sorted({float(value) for value in phrase_times}))
    if any(not math.isfinite(value) for value in phrases):
        raise ValueError("role-conditioned phrase times must be finite")
    by_section: dict[int, list[int]] = {}
    segment_by_index: dict[int, TimelineCandidate] = {}
    candidates_by_index: dict[int, Sequence[TimelineCandidate]] = {}
    for segment in candidates_by_segment:
        if not segment:
            raise ValueError("role-conditioned segment cannot be empty")
        reference = segment[0]
        if any(
            item.segment_index != reference.segment_index
            or not math.isclose(
                item.start_seconds,
                reference.start_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                item.end_seconds,
                reference.end_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for item in segment[1:]
        ):
            raise ValueError(
                "role-conditioned candidates disagree on canonical spans"
            )
        directive = section_plan.directive_at(reference.start_seconds)
        by_section.setdefault(directive.section_index, []).append(
            reference.segment_index
        )
        segment_by_index[reference.segment_index] = reference
        candidates_by_index[reference.segment_index] = segment

    targets: dict[int, str] = {}
    subject_metadata = dict(subject_metadata_by_candidate or {})
    section_audit: list[dict[str, Any]] = []
    for section_index, segment_indices in sorted(by_section.items()):
        directive = section_plan.sections[section_index]
        fractions = _role_fraction_targets(
            policy,
            role=directive.role,
            cadence_profile=cadence_profile,
        )
        if fractions is None:
            continue
        close_fraction, nonwide_fraction = fractions
        count = len(segment_indices)
        duration_by_position = {
            position: (
                segment_by_index[segment_index].end_seconds
                - segment_by_index[segment_index].start_seconds
            )
            for position, segment_index in enumerate(segment_indices)
        }
        total_duration = sum(duration_by_position.values())
        close_availability_by_position: dict[int, float] = {}
        nonwide_availability_by_position: dict[int, float] = {}
        wide_availability_by_position: dict[int, float] = {}
        for position, segment_index in enumerate(segment_indices):
            segment = candidates_by_index[segment_index]
            close_availability = 0.0
            nonwide_availability = 0.0
            wide_availability = 0.0
            for candidate in segment:
                if (
                    reachable_candidate_keys is not None
                    and (candidate.segment_index, candidate.source_id)
                    not in reachable_candidate_keys
                ):
                    continue
                samples, formation_proven = _candidate_role_shot_samples(
                    candidate,
                    subject_metadata,
                )
                close = samples.count("close") / 3.0
                wide = samples.count("wide") / 3.0
                nonwide = (
                    samples.count("close") + samples.count("medium")
                ) / 3.0
                close_availability = max(close_availability, close)
                nonwide_availability = max(nonwide_availability, nonwide)
                wide_availability = max(
                    wide_availability,
                    wide * (1.0 if formation_proven else 0.72),
                )
            close_availability_by_position[position] = close_availability
            nonwide_availability_by_position[position] = nonwide_availability
            wide_availability_by_position[position] = wide_availability
        mandatory_wide_positions: set[int] = set()
        phrase_entry_positions: list[int] = []
        if directive.role is SectionRole.CHORUS_LIKE:
            mandatory_wide_positions.add(0)
            for local_position, segment_index in enumerate(segment_indices):
                start = segment_by_index[segment_index].start_seconds
                if local_position == 0 or any(
                    math.isclose(start, value, rel_tol=0.0, abs_tol=1e-6)
                    for value in phrases
                ):
                    phrase_entry_positions.append(local_position)
            required_phrase_wide = int(
                math.ceil(
                    policy.minimum_phrase_entry_wide_anchor_fraction
                    * len(phrase_entry_positions)
                    - 1e-12
                )
            )
            mandatory_wide_positions = _preferred_spread_positions(
                phrase_entry_positions,
                required_phrase_wide,
                preference_by_position=wide_availability_by_position,
                seeded_positions=mandatory_wide_positions,
            )
        wide_budget_positions = _wide_positions_for_duration_target(
            tuple(range(count)),
            duration_by_position=duration_by_position,
            maximum_wide_seconds=(1.0 - nonwide_fraction) * total_duration,
            mandatory_positions=mandatory_wide_positions,
            preference_by_position={
                position: (
                    wide_availability_by_position[position]
                    - 0.30 * close_availability_by_position[position]
                    - 0.15 * nonwide_availability_by_position[position]
                )
                for position in range(count)
            },
        )

        # ``1 - preferred_nonwide_fraction`` is a ceiling on how much wide
        # coverage the human-reference grammar can tolerate, not a quota that
        # the selector must fill.  Pre-assigning every position up to that
        # ceiling made the later Viterbi path add unavoidable connector wides
        # on top, leaving real XOXZ chorus paths below the non-wide floor even
        # though safe non-wide candidates were reachable.  Protect only the
        # evidence-backed formation anchors here; all remaining positions may
        # choose their best graph-compatible close/non-wide candidate.  The
        # decoded montage reviewer still requires real formation-wide anchors
        # and resets, so this does not erase the wide-shot grammar.
        wide_positions = set(mandatory_wide_positions)

        nonwide_positions = [
            position for position in range(count) if position not in wide_positions
        ]
        close_positions: set[int] = set()
        if directive.role is SectionRole.CHORUS_LIKE:
            # Seed one close frame in each qualifying nonwide pair so the
            # decoded reviewer can observe close/nonwide bursts followed by a
            # real wide reset.  This remains a soft source preference.
            run: list[int] = []
            for position in range(count + 1):
                if position < count and position not in wide_positions:
                    run.append(position)
                    continue
                if len(run) >= 2:
                    close_positions.add(
                        max(
                            run,
                            key=lambda value: (
                                close_availability_by_position[value],
                                duration_by_position[value],
                                -value,
                            ),
                        )
                    )
                run = []
        close_positions = _close_positions_for_duration_target(
            nonwide_positions,
            duration_by_position=duration_by_position,
            minimum_close_seconds=close_fraction * total_duration,
            seeded_positions=close_positions,
            preference_by_position=close_availability_by_position,
        )
        for local_position, segment_index in enumerate(segment_indices):
            targets[segment_index] = (
                "wide"
                if local_position in wide_positions
                else "close"
                if local_position in close_positions
                else "nonwide"
            )
        section_audit.append(
            {
                "section_index": section_index,
                "role": directive.role.value,
                "segment_count": count,
                "target_closeup_fraction": close_fraction,
                "target_nonwide_fraction": nonwide_fraction,
                "assigned_closeup_duration_fraction": (
                    sum(
                        duration_by_position[position]
                        for position in close_positions
                    )
                    / total_duration
                ),
                "assigned_nonwide_duration_fraction": (
                    sum(
                        duration_by_position[position]
                        for position in nonwide_positions
                    )
                    / total_duration
                ),
                "target_wide_position_count": len(wide_positions),
                "wide_budget_position_count": len(wide_budget_positions),
                "optional_wide_budget_position_count": len(
                    wide_budget_positions - mandatory_wide_positions
                ),
                "optional_wide_budget_forced": False,
                "target_close_position_count": len(close_positions),
                "target_nonwide_position_count": count - len(wide_positions),
                "mandatory_wide_positions": sorted(
                    mandatory_wide_positions
                ),
                "phrase_entry_positions": phrase_entry_positions,
                "availability_aware_target_assignment": True,
            }
        )
    return targets, {
        "cadence_profile": cadence_profile,
        "section_targets": section_audit,
        "target_segment_count": len(targets),
    }


def _candidate_role_shot_samples(
    candidate: TimelineCandidate,
    subject_metadata_by_candidate: Mapping[str, "_CandidateSubjectMetadata"],
) -> tuple[tuple[str, str, str], bool]:
    shot_types = (
        candidate.entry_shot_type or candidate.shot_type,
        candidate.shot_type,
        candidate.exit_shot_type or candidate.shot_type,
    )
    metadata = subject_metadata_by_candidate.get(candidate.candidate_id)
    bases = (
        metadata.shot_scale_bases
        if metadata is not None
        else ("legacy", "legacy", "legacy")
    )
    samples: list[str] = []
    for shot_type, basis in zip(shot_types, bases, strict=True):
        if shot_type in {ShotType.GROUP_WIDE, ShotType.WIDE}:
            samples.append("wide")
        elif shot_type is ShotType.CLOSE_UP:
            samples.append("close")
        elif basis == "unknown_no_subject_geometry":
            samples.append("unknown")
        else:
            samples.append("medium")
    formation_proven = bool(
        metadata is not None
        and (
            max(metadata.formation_person_counts, default=0) >= 4
            or max(metadata.formation_upper_body_counts, default=0) >= 1
        )
    )
    return (tuple(samples), formation_proven)


def _role_conditioned_visual_controls(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    section_plan: SectionPlan,
    phrase_times: Sequence[float],
    policy: ProfessionalMontageSelectionPolicy,
    cadence_profile: str,
    subject_metadata_by_candidate: Mapping[str, "_CandidateSubjectMetadata"],
    timeline_config: TimelineConfig | None = None,
    forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
) -> tuple[dict[tuple[int, str], float], dict[int, float], dict[str, Any]]:
    reachable_candidate_keys = _complete_path_reachable_candidate_keys(
        candidates_by_segment,
        timeline_config=timeline_config or TimelineConfig(),
        forbidden_boundary_edges=forbidden_boundary_edges,
    )
    targets, audit = _role_target_categories(
        candidates_by_segment,
        section_plan=section_plan,
        phrase_times=phrase_times,
        policy=policy,
        cadence_profile=cadence_profile,
        subject_metadata_by_candidate=subject_metadata_by_candidate,
        reachable_candidate_keys=reachable_candidate_keys,
    )
    # A complete source-local emission spans roughly 0...2.3, while the sum of
    # all admitted transition terms can still exceed 16.  Wide separation is
    # intentional: balanced preserves aesthetics-first exploration, reference
    # makes the human motif decisive, and dense_safe is a lexicographic grammar
    # exploration inside the unchanged visual hard-gate graph.
    strength = {
        "balanced": 8.00,
        "reference": 32.00,
        "dense_safe": 128.00,
    }[cadence_profile]
    adjustments: dict[tuple[int, str], float] = {}
    penalized = 0
    availability: dict[int, dict[str, float | int]] = {}
    for segment in candidates_by_segment:
        reference = segment[0]
        target = targets.get(reference.segment_index)
        if target is None:
            continue
        directive = section_plan.directive_at(reference.start_seconds)
        duration = reference.end_seconds - reference.start_seconds
        best_close = 0.0
        best_nonwide = 0.0
        best_wide = 0.0
        exact_target_available = False
        for candidate in segment:
            if (
                candidate.segment_index,
                candidate.source_id,
            ) not in reachable_candidate_keys:
                continue
            samples, formation_proven = _candidate_role_shot_samples(
                candidate,
                subject_metadata_by_candidate,
            )
            close_fraction = samples.count("close") / 3.0
            wide_fraction = samples.count("wide") / 3.0
            unknown_fraction = samples.count("unknown") / 3.0
            nonwide_fraction = (
                samples.count("close") + samples.count("medium")
            ) / 3.0
            best_close = max(best_close, close_fraction)
            best_nonwide = max(best_nonwide, nonwide_fraction)
            best_wide = max(best_wide, wide_fraction)
            if target == "wide":
                penalty = strength * (1.0 - wide_fraction)
                if not formation_proven:
                    penalty += 0.30 * strength
                exact_target_available = exact_target_available or (
                    math.isclose(
                        wide_fraction,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and formation_proven
                )
            elif target in {"close", "nonwide"}:
                role_targets = _role_fraction_targets(
                    policy,
                    role=directive.role,
                    cadence_profile=cadence_profile,
                )
                if role_targets is None:  # pragma: no cover - target invariant
                    raise RuntimeError(
                        "role density target is missing for targeted section"
                    )
                close_target, nonwide_target = role_targets
                close_pressure = min(
                    0.90,
                    max(0.20, close_target / max(nonwide_target, 1e-9)),
                )
                # Do not pin close-ups to one preselected beat.  All non-anchor
                # positions compete for the section-wide duration quota, so a
                # visually admitted close shot can move to the neighboring
                # beat that actually connects to the full path.
                penalty = strength * (
                    wide_fraction
                    + unknown_fraction
                    + 0.45 * close_pressure * (1.0 - close_fraction)
                )
                exact_target_available = exact_target_available or (
                    nonwide_fraction >= 1.0 - 1e-12
                    and (
                        target != "close"
                        or close_fraction >= 1.0 - 1e-12
                    )
                )
            else:  # pragma: no cover - helper invariant
                raise RuntimeError("role-conditioned shot target is unknown")
            if penalty > 1e-12:
                adjustments[(candidate.segment_index, candidate.source_id)] = (
                    -float(penalty)
                )
                penalized += 1

        values = availability.setdefault(
            directive.section_index,
            {
                "duration_seconds": 0.0,
                "attainable_closeup_seconds": 0.0,
                "attainable_nonwide_seconds": 0.0,
                "attainable_wide_seconds": 0.0,
                "target_segment_count": 0,
                "exact_target_available_segment_count": 0,
            },
        )
        values["duration_seconds"] = float(
            values["duration_seconds"]
        ) + duration
        values["attainable_closeup_seconds"] = float(
            values["attainable_closeup_seconds"]
        ) + duration * best_close
        values["attainable_nonwide_seconds"] = float(
            values["attainable_nonwide_seconds"]
        ) + duration * best_nonwide
        values["attainable_wide_seconds"] = float(
            values["attainable_wide_seconds"]
        ) + duration * best_wide
        values["target_segment_count"] = int(
            values["target_segment_count"]
        ) + 1
        values["exact_target_available_segment_count"] = int(
            values["exact_target_available_segment_count"]
        ) + int(exact_target_available)

    section_availability: list[dict[str, float | int]] = []
    for section_index, values in sorted(availability.items()):
        duration = float(values["duration_seconds"])
        section_availability.append(
            {
                "section_index": section_index,
                "duration_seconds": duration,
                "attainable_closeup_duration_fraction_upper_bound": (
                    float(values["attainable_closeup_seconds"]) / duration
                ),
                "attainable_nonwide_duration_fraction_upper_bound": (
                    float(values["attainable_nonwide_seconds"]) / duration
                ),
                "attainable_wide_duration_fraction_upper_bound": (
                    float(values["attainable_wide_seconds"]) / duration
                ),
                "target_segment_count": int(values["target_segment_count"]),
                "exact_target_available_segment_count": int(
                    values["exact_target_available_segment_count"]
                ),
            }
        )

    switch_bonus: dict[int, float] = {}
    if cadence_profile == "balanced":
        highlight_rate = (
            policy.highlight_minimum_source_switches_per_minute
        )
        highlight_bonus = (
            0.65 * highlight_rate
        )
    elif cadence_profile == "reference":
        highlight_rate = (
            policy.highlight_preferred_source_switches_per_minute
        )
        highlight_bonus = (
            1.20 * highlight_rate
        )
    else:
        highlight_rate = min(
            30.0,
            1.15 * policy.highlight_preferred_source_switches_per_minute,
        )
        highlight_bonus = min(
            72.0,
            2.40 * highlight_rate,
        )
    highlight_segments_by_section: dict[int, list[TimelineCandidate]] = {}
    for segment in candidates_by_segment:
        reference = segment[0]
        directive = section_plan.directive_at(reference.start_seconds)
        if directive.role is SectionRole.CHORUS_LIKE:
            highlight_segments_by_section.setdefault(
                directive.section_index,
                [],
            ).append(reference)
    switch_target_audit: list[dict[str, Any]] = []
    for section_index, references in sorted(
        highlight_segments_by_section.items()
    ):
        if len(references) < 2:
            continue
        duration = references[-1].end_seconds - references[0].start_seconds
        target_count = min(
            len(references) - 1,
            max(
                1,
                int(math.ceil(highlight_rate * duration / 60.0 - 1e-12)),
            ),
        )
        local_positions = _evenly_spaced_positions(
            tuple(range(1, len(references))),
            target_count,
        )
        target_segment_indices = tuple(
            references[position].segment_index
            for position in sorted(local_positions)
        )
        for segment_index in target_segment_indices:
            switch_bonus[segment_index] = highlight_bonus
        switch_target_audit.append(
            {
                "section_index": section_index,
                "duration_seconds": duration,
                "target_source_switches_per_minute": highlight_rate,
                "target_source_switch_count": target_count,
                "bonus_segment_indices": list(target_segment_indices),
            }
        )
    return adjustments, switch_bonus, {
        **audit,
        "penalty_strength": strength,
        "penalized_candidate_count": penalized,
        "emission_adjustment_count": len(adjustments),
        "highlight_source_switch_bonus": highlight_bonus,
        "highlight_source_switch_target_rate": highlight_rate,
        "highlight_source_switch_targets": switch_target_audit,
        "source_switch_bonus_override_segment_count": len(switch_bonus),
        "section_candidate_availability": section_availability,
        "flexible_role_density_allocation": True,
        "fixed_role_positions": ["formation_wide_anchors"],
        "complete_path_reachable_candidate_count": len(
            reachable_candidate_keys
        ),
        "hard_visual_gate_relaxed": False,
    }


def _role_conditioned_candidate_outcomes(
    records: Sequence["VisualTimelineCandidateRecord"],
    *,
    section_plan: SectionPlan,
    policy: ProfessionalMontageSelectionPolicy,
    subject_metadata_by_candidate: Mapping[
        str, "_CandidateSubjectMetadata"
    ],
) -> tuple[dict[str, Any], ...]:
    """Serialize scalar pre-render role attainment without claiming QA."""

    outcomes: list[dict[str, Any]] = []
    for record in records:
        choices_by_section: dict[int, list[TimelineChoice]] = {}
        for choice in record.timeline_result.choices:
            directive = section_plan.directive_at(
                choice.candidate.start_seconds
            )
            choices_by_section.setdefault(directive.section_index, []).append(
                choice
            )
        sections: list[dict[str, Any]] = []
        normalized_scores: list[float] = []
        highlight_passes: list[bool] = []
        highlight_scores: list[float] = []
        for directive in section_plan.sections:
            choices = choices_by_section.get(directive.section_index, [])
            if not choices or directive.role not in {
                SectionRole.BUILD,
                SectionRole.CHORUS_LIKE,
                SectionRole.BRIDGE_LIKE,
            }:
                continue
            duration = sum(
                choice.candidate.end_seconds
                - choice.candidate.start_seconds
                for choice in choices
            )
            close_seconds = 0.0
            nonwide_seconds = 0.0
            for choice in choices:
                candidate = choice.candidate
                span = candidate.end_seconds - candidate.start_seconds
                samples, _formation_proven = _candidate_role_shot_samples(
                    candidate,
                    subject_metadata_by_candidate,
                )
                close_seconds += span * samples.count("close") / 3.0
                nonwide_seconds += span * (
                    samples.count("close") + samples.count("medium")
                ) / 3.0
            close_fraction = close_seconds / duration
            nonwide_fraction = nonwide_seconds / duration
            switches = sum(
                outgoing.candidate.source_id
                != incoming.candidate.source_id
                for outgoing, incoming in zip(choices, choices[1:])
            )
            switches_per_minute = switches * 60.0 / duration
            if directive.role is SectionRole.CHORUS_LIKE:
                minimum_close = policy.highlight_minimum_closeup_fraction
                minimum_nonwide = policy.highlight_minimum_nonwide_fraction
                minimum_switches = (
                    policy.highlight_minimum_source_switches_per_minute
                )
            elif directive.role is SectionRole.BUILD:
                minimum_close = policy.build_minimum_closeup_fraction
                minimum_nonwide = policy.build_minimum_nonwide_fraction
                minimum_switches = None
            else:
                minimum_close = policy.bridge_minimum_closeup_fraction
                minimum_nonwide = policy.bridge_minimum_nonwide_fraction
                minimum_switches = None
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
                    min(
                        1.0,
                        switches_per_minute / max(minimum_switches, 1e-9),
                    )
                )
                highlight_passes.append(passed)
            attainment_score = sum(requirements) / len(requirements)
            normalized_scores.append(attainment_score)
            if minimum_switches is not None:
                highlight_scores.append(attainment_score)
            sections.append(
                {
                    "section_index": directive.section_index,
                    "role": directive.role.value,
                    "duration_seconds": duration,
                    "closeup_duration_fraction": close_fraction,
                    "nonwide_duration_fraction": nonwide_fraction,
                    "source_switch_count": switches,
                    "source_switches_per_minute": switches_per_minute,
                    "minimum_closeup_duration_fraction": minimum_close,
                    "minimum_nonwide_duration_fraction": minimum_nonwide,
                    "minimum_source_switches_per_minute": minimum_switches,
                    "metadata_floor_passed": passed,
                }
            )
        outcomes.append(
            {
                "candidate_index": record.candidate_index,
                "cadence_profile": record.cadence_profile,
                "metadata_role_attainment_score": (
                    sum(normalized_scores) / len(normalized_scores)
                    if normalized_scores
                    else 0.0
                ),
                "minimum_highlight_metadata_attainment_score": (
                    min(highlight_scores) if highlight_scores else 0.0
                ),
                "highlight_metadata_floor_pass_count": sum(highlight_passes),
                "highlight_metadata_floor_count": len(highlight_passes),
                "all_highlight_metadata_floors_passed": bool(
                    highlight_passes and all(highlight_passes)
                ),
                "timeline_review_passed": record.review.passed,
                "timeline_review_hard_failure_codes": [
                    issue.code for issue in record.review.hard_failures
                ],
                "sections": sections,
                "decoded_output_authority": False,
            }
        )
    return tuple(outcomes)


def _select_safe_cadence_candidates(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    preferred_source_ids: Sequence[str | None] | None,
    diversity: VisualDiversityConfig,
    maximum_results: int = 32,
    section_plan: SectionPlan | None = None,
    phrase_times: Sequence[float] = (),
    professional_montage_policy: (
        ProfessionalMontageSelectionPolicy | None
    ) = None,
    subject_metadata_by_candidate: Mapping[
        str, "_CandidateSubjectMetadata"
    ] | None = None,
    role_conditioning_audit: list[dict[str, Any]] | None = None,
    forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
) -> tuple[tuple[TimelineResult, ...], tuple[str, ...]]:
    """Generate profile-local alternatives, then interleave unique paths."""

    profiles = _safe_cadence_profiles(diversity)
    if (section_plan is None) != (professional_montage_policy is None):
        raise ValueError(
            "section_plan and professional_montage_policy must be supplied together"
        )
    metadata = dict(subject_metadata_by_candidate or {})
    compiled_no_good_constraints = compile_timeline_no_good_constraint_index(
        candidates_by_segment,
        forbidden_boundary_edges,
    )
    pools: list[tuple[TimelineResult, ...]] = []
    profile_failures: list[tuple[str, NoCompleteTimelinePath]] = []
    for profile in profiles:
        selection_kwargs: dict[str, Any] = {}
        role_audit: dict[str, Any] | None = None
        if professional_montage_policy is not None:
            assert section_plan is not None
            adjustments, switch_bonus, audit = (
                _role_conditioned_visual_controls(
                    candidates_by_segment,
                    section_plan=section_plan,
                    phrase_times=phrase_times,
                    policy=professional_montage_policy,
                    cadence_profile=profile.name,
                    subject_metadata_by_candidate=metadata,
                    timeline_config=profile.config,
                    forbidden_boundary_edges=forbidden_boundary_edges,
                )
            )
            selection_kwargs.update(
                {
                    "emission_adjustments": adjustments,
                    "source_switch_bonus_by_segment": switch_bonus,
                }
            )
            role_audit = audit
            if role_conditioning_audit is not None:
                role_conditioning_audit.append(audit)
        try:
            pool = select_timeline_candidates(
                candidates_by_segment,
                config=profile.config,
                preferred_source_ids=preferred_source_ids,
                maximum_results=16,
                diversity_penalty=0.12,
                minimum_hamming_fraction=0.02,
                forbidden_boundary_edges=forbidden_boundary_edges,
                compiled_no_good_constraints=compiled_no_good_constraints,
                **selection_kwargs,
            )
        except NoCompleteTimelinePath as exc:
            # The three cadence profiles intentionally use different soft
            # sacrifices.  One constrained graph can therefore be infeasible
            # for balanced while remaining feasible for reference/dense-safe.
            # Keep profile exhaustion local instead of discarding survivors.
            profile_failures.append((profile.name, exc))
            if role_audit is not None:
                role_audit["candidate_generation_exhaustion"] = exc.to_dict()
            pool = ()
        pools.append(pool)
    pooled = tuple(pools)
    candidates: list[TimelineResult] = []
    cadence_profiles: list[str] = []
    signatures: set[tuple[str, ...]] = set()
    for rank in range(max(len(pool) for pool in pooled)):
        for profile, pool in zip(profiles, pooled, strict=True):
            if rank >= len(pool):
                continue
            result = pool[rank]
            signature = _timeline_path_signature(result)
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates.append(result)
            cadence_profiles.append(profile.name)
            if len(candidates) >= maximum_results:
                return (tuple(candidates), tuple(cadence_profiles))
    if not candidates:
        if profile_failures:
            failure_summary = ",".join(
                f"{profile}:{failure.reason_code}"
                for profile, failure in profile_failures
            )
            rejection_counts = {
                f"{profile}:{failure.reason_code}": 1
                for profile, failure in profile_failures
            }
            raise NoCompleteTimelinePath(
                "all safe cadence profiles exhausted; "
                f"profile_failures={failure_summary}",
                reason_code="all_safe_cadence_profiles_exhausted",
                rejection_counts=rejection_counts,
                profile_failures=tuple(profile_failures),
            )
        raise NoCompleteTimelinePath(
            "safe cadence candidate generation produced no result",
            reason_code="safe_cadence_candidate_generation_empty",
        )
    return (tuple(candidates), tuple(cadence_profiles))


@dataclass(frozen=True, slots=True)
class FrameVisualDescriptor:
    quality_score: float
    composition_score: float
    shot_scale: float
    shot_type: ShotType
    face_count: int
    structure_embedding: tuple[float, ...]
    motion: tuple[float, float] | None = None
    # Normalized x/y/width/height for the only safely dominant face. Multiple
    # similarly-sized detections deliberately remain ``None``: basic
    # face-match cross-edit must fall back to a raw cut rather than crop an
    # arbitrary group member.
    primary_face_bbox: tuple[float, float, float, float] | None = None
    color_embedding: tuple[float, ...] = ()
    face_detector_backend: str = "none"
    primary_face_landmarks_xy: tuple[tuple[float, float], ...] | None = None
    primary_face_head_roll_radians: float | None = None
    primary_face_eye_scale: float | None = None
    primary_face_geometry: tuple[float, ...] | None = None
    pose_embedding: tuple[float, ...] = ()
    pose_confidence: float = 0.0
    detected_person_count: int = 0
    person_layout_embedding: tuple[float, ...] = ()
    formation_people: tuple[FormationPersonDescriptor, ...] = field(
        default=(), repr=False
    )
    pose_backend: str = "structure_hog_fallback"
    silhouette_embedding: tuple[float, ...] = ()
    silhouette_bbox_xywh: tuple[float, float, float, float] | None = None
    silhouette_centroid_xy: tuple[float, float] | None = None
    silhouette_area_ratio: float = 0.0
    silhouette_confidence: float = 0.0
    silhouette_backend: str = "none"
    # Non-identifying raw-frame geometry: a high value means a dark, thick
    # edge-connected foreground object crosses the performance area.  This is
    # deliberately not an appearance embedding and is used to reject camera
    # booms/rigs before direct-member rendering.
    foreground_obstruction_score: float = 0.0
    # Non-identifying raw-frame scalar used only to distinguish a persistent
    # blown-out stage-light entry from a legitimate one-frame flash.  No
    # pixels, appearance vectors, or identity evidence are retained.
    highlight_clipping_ratio: float = 0.0
    # Explains which decoded geometry produced ``shot_scale``.  In
    # particular, a detector miss is no longer silently serialized as a wide
    # shot: callers can keep the neutral MEDIUM enum for compatibility while
    # excluding ``unknown_no_subject_geometry`` from observed-medium counts.
    shot_scale_basis: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "composition_score": self.composition_score,
            "shot_scale": self.shot_scale,
            "shot_type": self.shot_type.value,
            "shot_scale_basis": self.shot_scale_basis,
            "face_count": self.face_count,
            "primary_face_bbox_xywh": (
                list(self.primary_face_bbox)
                if self.primary_face_bbox is not None
                else None
            ),
            "color_embedding": list(self.color_embedding),
            "face_detector_backend": self.face_detector_backend,
            "primary_face_landmarks_xy": (
                [list(point) for point in self.primary_face_landmarks_xy]
                if self.primary_face_landmarks_xy is not None
                else None
            ),
            "primary_face_head_roll_radians": self.primary_face_head_roll_radians,
            "primary_face_eye_scale": self.primary_face_eye_scale,
            "primary_face_geometry": (
                list(self.primary_face_geometry)
                if self.primary_face_geometry is not None
                else None
            ),
            "primary_face_head_pose_proxies": (
                {
                    "yaw_nose_horizontal": self.primary_face_geometry[6],
                    "pitch_nose_vertical": self.primary_face_geometry[7],
                    "mouth_vertical": self.primary_face_geometry[8],
                    "mouth_asymmetry": self.primary_face_geometry[9],
                }
                if self.primary_face_geometry is not None
                else None
            ),
            "pose_backend": self.pose_backend,
            "pose_confidence": self.pose_confidence,
            "pose_available": self.pose_backend != "structure_hog_fallback",
            "detected_person_count": self.detected_person_count,
            "person_layout_available": bool(self.person_layout_embedding),
            "formation_person_count": len(self.formation_people),
            "formation_upper_body_count": sum(
                bool(person.upper_body_keypoints_xyc)
                for person in self.formation_people
            ),
            "silhouette_backend": self.silhouette_backend,
            "silhouette_confidence": self.silhouette_confidence,
            "silhouette_bbox_xywh": (
                list(self.silhouette_bbox_xywh)
                if self.silhouette_bbox_xywh is not None
                else None
            ),
            "silhouette_centroid_xy": (
                list(self.silhouette_centroid_xy)
                if self.silhouette_centroid_xy is not None
                else None
            ),
            "silhouette_area_ratio": self.silhouette_area_ratio,
            "foreground_obstruction_score": self.foreground_obstruction_score,
            "highlight_clipping_ratio": self.highlight_clipping_ratio,
            "motion": list(self.motion) if self.motion is not None else None,
        }


@dataclass(frozen=True, slots=True)
class VisualTimelineChoice:
    segment_index: int
    source_index: int
    canonical_master_start: float
    canonical_master_end: float
    source_local_start: float
    source_local_end: float
    quality_score: float
    composition_score: float
    boundary_score: float
    shot_scale: float
    shot_type: ShotType
    shot_id: str
    entry_shot_scale: float | None = None
    exit_shot_scale: float | None = None
    entry_shot_type: ShotType | None = None
    exit_shot_type: ShotType | None = None
    entry_scene_cut_near: bool = False
    exit_scene_cut_near: bool = False
    intentional_transition_anchor_basis: str | None = None
    internal_scene_cut_master_seconds: tuple[float, ...] = ()
    entry_boundary_side_stability: BoundarySideStability | None = None
    exit_boundary_side_stability: BoundarySideStability | None = None
    face_samples_xywh: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ] = (None, None, None)
    face_match_candidate_failure: str | None = None
    face_detector_backends: tuple[str, str, str] = ("none", "none", "none")
    face_landmark_samples_xy: tuple[
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
    ] = (None, None, None)
    face_head_roll_samples_radians: tuple[
        float | None, float | None, float | None
    ] = (None, None, None)
    face_eye_scale_samples: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    # Dissolves need different evidence from hard cuts: both sources are shown
    # *after* the nominal boundary.  These windows sample cut, handle midpoint,
    # and handle end and must never be substituted with the pre-cut minimax
    # windows above.
    dissolve_entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    dissolve_exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    dissolve_entry_face_box_window: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    dissolve_exit_face_box_window: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    # Raw appearance/body vectors are transient safety evidence. ``to_dict``
    # intentionally records only backend/count/confidence metadata; biometric
    # embeddings are never persisted to the analysis manifest.
    dissolve_entry_face_identity_window: tuple[tuple[float, ...], ...] = field(
        default=(), repr=False
    )
    dissolve_exit_face_identity_window: tuple[tuple[float, ...], ...] = field(
        default=(), repr=False
    )
    dissolve_entry_pose_embedding_window: tuple[tuple[float, ...], ...] = ()
    dissolve_exit_pose_embedding_window: tuple[tuple[float, ...], ...] = ()
    dissolve_entry_pose_backend_window: tuple[str, ...] = ()
    dissolve_exit_pose_backend_window: tuple[str, ...] = ()
    dissolve_entry_pose_confidence_window: tuple[float, ...] = ()
    dissolve_exit_pose_confidence_window: tuple[float, ...] = ()
    dissolve_entry_silhouette_embedding_window: tuple[
        tuple[float, ...], ...
    ] = ()
    dissolve_exit_silhouette_embedding_window: tuple[
        tuple[float, ...], ...
    ] = ()
    dissolve_entry_silhouette_backend_window: tuple[str, ...] = ()
    dissolve_exit_silhouette_backend_window: tuple[str, ...] = ()
    dissolve_entry_silhouette_confidence_window: tuple[float, ...] = ()
    dissolve_exit_silhouette_confidence_window: tuple[float, ...] = ()
    face_transition_minimax_mismatch_from_previous: float | None = None
    pose_backends: tuple[str, str, str] = (
        "structure_hog_fallback",
        "structure_hog_fallback",
        "structure_hog_fallback",
    )
    pose_confidences: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detected_person_counts: tuple[int, int, int] = (0, 0, 0)
    formation_person_counts: tuple[int, int, int] = (0, 0, 0)
    formation_upper_body_counts: tuple[int, int, int] = (0, 0, 0)
    shot_scale_bases: tuple[str, str, str] = ("legacy", "legacy", "legacy")
    silhouette_samples_xywh: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ] = (None, None, None)
    silhouette_centroid_samples_xy: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None, None)
    silhouette_backends: tuple[str, str, str] = ("none", "none", "none")
    silhouette_confidences: tuple[float, float, float] = (0.0, 0.0, 0.0)
    layout_transition_mismatch_from_previous: float | None = None
    pose_transition_basis_from_previous: str | None = None
    pose_transition_mismatch_from_previous: float | None = None
    silhouette_transition_mismatch_from_previous: float | None = None
    formation_transition_mismatch_from_previous: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.canonical_master_end - self.canonical_master_start

    @property
    def source_duration_seconds(self) -> float:
        return self.source_local_end - self.source_local_start

    @property
    def source_playback_rate(self) -> float:
        return self.source_duration_seconds / self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "canonical_master_start": self.canonical_master_start,
            "canonical_master_end": self.canonical_master_end,
            "source_local_start": self.source_local_start,
            "source_local_end": self.source_local_end,
            "duration_seconds": self.duration_seconds,
            "source_duration_seconds": self.source_duration_seconds,
            "source_playback_rate": self.source_playback_rate,
            "quality_score": self.quality_score,
            "composition_score": self.composition_score,
            "boundary_score": self.boundary_score,
            "shot_scale": self.shot_scale,
            "shot_type": self.shot_type.value,
            "shot_id": self.shot_id,
            "entry_shot_scale": self.entry_shot_scale,
            "exit_shot_scale": self.exit_shot_scale,
            "entry_shot_type": (
                self.entry_shot_type.value
                if self.entry_shot_type is not None
                else None
            ),
            "exit_shot_type": (
                self.exit_shot_type.value
                if self.exit_shot_type is not None
                else None
            ),
            "entry_scene_cut_near": self.entry_scene_cut_near,
            "exit_scene_cut_near": self.exit_scene_cut_near,
            "intentional_transition_anchor_basis": (
                self.intentional_transition_anchor_basis
            ),
            "internal_scene_cut_master_seconds": list(
                self.internal_scene_cut_master_seconds
            ),
            "entry_boundary_side_stability": (
                self.entry_boundary_side_stability.to_dict()
                if self.entry_boundary_side_stability is not None
                else None
            ),
            "exit_boundary_side_stability": (
                self.exit_boundary_side_stability.to_dict()
                if self.exit_boundary_side_stability is not None
                else None
            ),
            "face_samples_xywh": [
                list(sample) if sample is not None else None
                for sample in self.face_samples_xywh
            ],
            "face_match_candidate_failure": self.face_match_candidate_failure,
            "face_detector_backends": list(self.face_detector_backends),
            "face_landmark_samples_xy": [
                [list(point) for point in sample] if sample is not None else None
                for sample in self.face_landmark_samples_xy
            ],
            "face_head_roll_samples_radians": list(
                self.face_head_roll_samples_radians
            ),
            "face_eye_scale_samples": list(self.face_eye_scale_samples),
            "entry_face_geometry_window": [
                list(geometry) for geometry in self.entry_face_geometry_window
            ],
            "exit_face_geometry_window": [
                list(geometry) for geometry in self.exit_face_geometry_window
            ],
            "dissolve_entry_face_geometry_window": [
                list(geometry)
                for geometry in self.dissolve_entry_face_geometry_window
            ],
            "dissolve_exit_face_geometry_window": [
                list(geometry)
                for geometry in self.dissolve_exit_face_geometry_window
            ],
            "dissolve_entry_face_box_window": [
                list(box) for box in self.dissolve_entry_face_box_window
            ],
            "dissolve_exit_face_box_window": [
                list(box) for box in self.dissolve_exit_face_box_window
            ],
            "dissolve_entry_evidence": {
                "appearance_compatibility_backend": SFACE_BACKEND,
                "appearance_embedding_count": sum(
                    bool(embedding)
                    for embedding in self.dissolve_entry_face_identity_window
                ),
                "pose_backends": list(self.dissolve_entry_pose_backend_window),
                "pose_confidences": list(
                    self.dissolve_entry_pose_confidence_window
                ),
                "silhouette_backends": list(
                    self.dissolve_entry_silhouette_backend_window
                ),
                "silhouette_confidences": list(
                    self.dissolve_entry_silhouette_confidence_window
                ),
                "raw_embeddings_persisted": False,
            },
            "dissolve_exit_evidence": {
                "appearance_compatibility_backend": SFACE_BACKEND,
                "appearance_embedding_count": sum(
                    bool(embedding)
                    for embedding in self.dissolve_exit_face_identity_window
                ),
                "pose_backends": list(self.dissolve_exit_pose_backend_window),
                "pose_confidences": list(
                    self.dissolve_exit_pose_confidence_window
                ),
                "silhouette_backends": list(
                    self.dissolve_exit_silhouette_backend_window
                ),
                "silhouette_confidences": list(
                    self.dissolve_exit_silhouette_confidence_window
                ),
                "raw_embeddings_persisted": False,
            },
            "face_transition_minimax_mismatch_from_previous": (
                self.face_transition_minimax_mismatch_from_previous
            ),
            "pose_backends": list(self.pose_backends),
            "pose_confidences": list(self.pose_confidences),
            "detected_person_counts": list(self.detected_person_counts),
            "formation_person_counts": list(self.formation_person_counts),
            "formation_upper_body_counts": list(
                self.formation_upper_body_counts
            ),
            "shot_scale_bases": list(self.shot_scale_bases),
            "silhouette_samples_xywh": [
                list(sample) if sample is not None else None
                for sample in self.silhouette_samples_xywh
            ],
            "silhouette_centroid_samples_xy": [
                list(sample) if sample is not None else None
                for sample in self.silhouette_centroid_samples_xy
            ],
            "silhouette_backends": list(self.silhouette_backends),
            "silhouette_confidences": list(self.silhouette_confidences),
            "layout_transition_mismatch_from_previous": (
                self.layout_transition_mismatch_from_previous
            ),
            "pose_transition_basis_from_previous": (
                self.pose_transition_basis_from_previous
            ),
            "pose_transition_mismatch_from_previous": (
                self.pose_transition_mismatch_from_previous
            ),
            "silhouette_transition_mismatch_from_previous": (
                self.silhouette_transition_mismatch_from_previous
            ),
            "formation_transition_mismatch_from_previous": (
                self.formation_transition_mismatch_from_previous
            ),
        }


def _candidate_review_summary(
    *,
    candidate_index: int,
    cadence_profile: str,
    selected: bool,
    candidate: TimelineResult,
    review: TimelineReview,
) -> dict[str, Any]:
    """Return the stable compact summary already exposed by timeline plans."""

    return {
        "candidate_index": candidate_index,
        "cadence_profile": cadence_profile,
        "selected": selected,
        "optimizer_score": candidate.total_score,
        "passed": review.passed,
        "weighted_score": review.weighted_score,
        "hard_failure_codes": [item.code for item in review.hard_failures],
        "warning_codes": [item.code for item in review.warnings],
        "source_switch_count": review.source_switch_count,
        "source_switch_rate_on_visual_boundaries": review.source_switch_rate,
        "source_switches_per_minute": review.source_switches_per_minute,
        "comparable_visual_boundary_count": (
            review.comparable_visual_boundary_count
        ),
        "unique_source_count": review.unique_source_count,
        "longest_source_run_seconds": review.longest_source_run_seconds,
        "source_run_median_seconds": review.source_run_median_seconds,
        "visual_shot_median_seconds": review.visual_shot_median_seconds,
        "ping_pong_count": len(review.ping_pongs),
    }


@dataclass(frozen=True, slots=True)
class VisualTimelineCandidateRecord:
    """One fully retained optimizer alternative and its reviewer result.

    ``TimelineResult`` intentionally remains available to trusted in-process
    consumers so a rejected or runner-up path can be materialized later.  It
    is excluded from ``repr`` and :meth:`to_dict` because candidates can carry
    transient face-appearance embeddings used by visual gates.
    """

    candidate_index: int
    cadence_profile: str
    timeline_result: TimelineResult = field(repr=False)
    review: TimelineReview = field(repr=False)
    review_rank: int
    selected: bool = False
    path_signature: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if (
            isinstance(self.review_rank, bool)
            or not isinstance(self.review_rank, int)
            or self.review_rank < 0
        ):
            raise ValueError("review_rank must be a non-negative integer")
        if not self.cadence_profile.strip():
            raise ValueError("cadence_profile cannot be blank")
        if not isinstance(self.timeline_result, TimelineResult):
            raise TypeError("timeline_result must be a TimelineResult")
        if not isinstance(self.review, TimelineReview):
            raise TypeError("review must be a TimelineReview")
        if not self.timeline_result.candidates:
            raise ValueError("timeline_result cannot be empty")
        if not math.isfinite(self.timeline_result.total_score):
            raise ValueError("timeline optimizer score must be finite")
        signature = tuple(self.path_signature) or _timeline_path_signature(
            self.timeline_result
        )
        if signature != _timeline_path_signature(self.timeline_result):
            raise ValueError("path_signature does not match timeline_result")
        object.__setattr__(self, "path_signature", signature)

    @property
    def optimizer_score(self) -> float:
        return self.timeline_result.total_score

    def to_summary_dict(self) -> dict[str, Any]:
        return _candidate_review_summary(
            candidate_index=self.candidate_index,
            cadence_profile=self.cadence_profile,
            selected=self.selected,
            candidate=self.timeline_result,
            review=self.review,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_summary_dict()
        payload.update(
            {
                "review_rank": self.review_rank,
                "path_signature": list(self.path_signature),
                "source_sequence": [
                    candidate.source_id
                    for candidate in self.timeline_result.candidates
                ],
                "review": self.review.to_dict(),
                "raw_candidate_embeddings_persisted": False,
            }
        )
        return payload


_VISUAL_OPTIMIZER_INPUT_IDENTITY_CONTRACT = (
    "visual_optimizer_workspace_input_identity_v1"
)


def _optimizer_identity_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _optimizer_identity_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("optimizer identity mapping keys must be strings")
        return {
            key: _optimizer_identity_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_optimizer_identity_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer identity values must be finite")
        return value
    raise TypeError(
        f"optimizer identity cannot encode {type(value).__name__}"
    )


def _visual_optimizer_input_identity(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    sources: Sequence[Path | str],
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_clock_basis: str,
    scene_cut_times: Sequence[Sequence[float]],
    preferred_source_ids: Sequence[str | None] | None = None,
    diversity: VisualDiversityConfig | None = None,
    section_plan: SectionPlan | None = None,
    phrase_times: Sequence[float] = (),
    professional_montage_policy: ProfessionalMontageSelectionPolicy | None = None,
    subject_metadata_by_candidate: Mapping[str, object] | None = None,
    prescribed_source_ids: Sequence[str] | None = None,
    maximum_results: int = 32,
) -> str:
    segments = tuple(tuple(segment) for segment in candidates_by_segment)
    if not segments or any(not segment for segment in segments):
        raise ValueError("optimizer input identity needs a non-empty lattice")
    payload = {
        "contract": _VISUAL_OPTIMIZER_INPUT_IDENTITY_CONTRACT,
        "sources": [str(Path(path)) for path in sources],
        "audio_offsets": [float(value) for value in audio_offsets],
        "source_timewarps": [
            None if timewarp is None else timewarp.to_dict()
            for timewarp in source_timewarps
        ],
        "source_clock_basis": str(source_clock_basis),
        "scene_cut_times": [
            [float(value) for value in source_cuts]
            for source_cuts in scene_cut_times
        ],
        "candidate_lattice": [
            [_optimizer_identity_value(candidate) for candidate in segment]
            for segment in segments
        ],
        "selection_context": {
            "preferred_source_ids": _optimizer_identity_value(
                None
                if preferred_source_ids is None
                else tuple(preferred_source_ids)
            ),
            "diversity": _optimizer_identity_value(
                diversity or VisualDiversityConfig()
            ),
            "section_plan": _optimizer_identity_value(section_plan),
            "phrase_times": _optimizer_identity_value(tuple(phrase_times)),
            "professional_montage_policy": _optimizer_identity_value(
                professional_montage_policy
            ),
            "subject_metadata_by_candidate": _optimizer_identity_value(
                dict(subject_metadata_by_candidate or {})
            ),
            "prescribed_source_ids": _optimizer_identity_value(
                None
                if prescribed_source_ids is None
                else tuple(prescribed_source_ids)
            ),
            "maximum_results": maximum_results,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256_identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class VisualTimelineOptimizerWorkspace:
    """Transient descriptor lattice and every input needed to rerun the DP.

    The workspace never enters :meth:`VisualTimelineCandidateSet.to_dict`.
    Keeping it in process lets fresh splice failures feed exact no-good edges
    back into the optimizer without decoding or describing any frame again.
    """

    candidates_by_segment: tuple[tuple[TimelineCandidate, ...], ...] = field(
        repr=False
    )
    input_identity: str
    preferred_source_ids: tuple[str | None, ...] | None = None
    diversity: VisualDiversityConfig = field(default_factory=VisualDiversityConfig)
    section_plan: SectionPlan | None = field(default=None, repr=False)
    phrase_times: tuple[float, ...] = ()
    professional_montage_policy: ProfessionalMontageSelectionPolicy | None = field(
        default=None,
        repr=False,
    )
    subject_metadata_by_candidate: Mapping[
        str, "_CandidateSubjectMetadata"
    ] = field(default_factory=dict, repr=False, compare=False)
    prescribed_source_ids: tuple[str, ...] | None = None
    maximum_results: int = 32

    def __post_init__(self) -> None:
        segments = tuple(
            tuple(candidate for candidate in segment)
            for segment in self.candidates_by_segment
        )
        if not segments or any(not segment for segment in segments):
            raise ValueError("optimizer workspace needs non-empty candidate segments")
        expected_segment_indices = tuple(
            range(
                segments[0][0].segment_index,
                segments[0][0].segment_index + len(segments),
            )
        )
        actual_segment_indices = tuple(segment[0].segment_index for segment in segments)
        if actual_segment_indices != expected_segment_indices:
            raise ValueError("optimizer workspace segment indices must be contiguous")
        for segment in segments:
            reference = segment[0]
            if any(
                not isinstance(candidate, TimelineCandidate)
                or candidate.segment_index != reference.segment_index
                or not math.isclose(
                    candidate.start_seconds,
                    reference.start_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    candidate.end_seconds,
                    reference.end_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for candidate in segment
            ):
                raise ValueError(
                    "optimizer workspace candidates must share canonical spans"
                )
        preferred = (
            None
            if self.preferred_source_ids is None
            else tuple(self.preferred_source_ids)
        )
        prescribed = (
            None
            if self.prescribed_source_ids is None
            else tuple(self.prescribed_source_ids)
        )
        if preferred is not None and len(preferred) != len(segments):
            raise ValueError("workspace preferences must match candidate segments")
        if prescribed is not None and len(prescribed) != len(segments):
            raise ValueError("workspace prescribed path must match candidate segments")
        if preferred is not None and prescribed is not None:
            raise ValueError(
                "workspace cannot combine preferences with a prescribed path"
            )
        if (self.section_plan is None) != (
            self.professional_montage_policy is None
        ):
            raise ValueError(
                "workspace section plan and montage policy must be supplied together"
            )
        phrases = tuple(float(value) for value in self.phrase_times)
        if any(not math.isfinite(value) for value in phrases):
            raise ValueError("workspace phrase times must be finite")
        if (
            isinstance(self.maximum_results, bool)
            or not isinstance(self.maximum_results, int)
            or not 8 <= self.maximum_results <= 32
        ):
            raise ValueError("workspace maximum_results must be between 8 and 32")
        object.__setattr__(self, "candidates_by_segment", segments)
        object.__setattr__(
            self,
            "input_identity",
            _validated_sha256_identity(
                self.input_identity,
                "optimizer workspace input_identity",
            ),
        )
        object.__setattr__(self, "preferred_source_ids", preferred)
        object.__setattr__(self, "prescribed_source_ids", prescribed)
        object.__setattr__(self, "phrase_times", phrases)
        object.__setattr__(
            self,
            "subject_metadata_by_candidate",
            MappingProxyType(dict(self.subject_metadata_by_candidate)),
        )


@dataclass(frozen=True, slots=True)
class VisualTimelineCandidateSet:
    """Deterministic top-N alternatives before selected-only post-processing."""

    candidates: tuple[VisualTimelineCandidateRecord, ...]
    selected_candidate_index: int
    sources: tuple[Path, ...]
    audio_offsets: tuple[float, ...]
    source_timewarps: tuple[TimeWarpMap | None, ...]
    source_clock_basis: str
    requested_maximum_results: int
    scene_cut_times: tuple[tuple[float, ...], ...] = ()
    face_metadata_by_candidate: Mapping[str, _CandidateFaceMetadata] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    subject_metadata_by_candidate: Mapping[str, _CandidateSubjectMetadata] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    candidate_generation_basis: str = (
        "three_safe_cadence_profiles_round_robin_dedup_then_reference_review_v5"
    )
    minimum_materialization_target: int = 8
    role_conditioned_selection: Mapping[str, Any] | None = None
    optimizer_workspace: VisualTimelineOptimizerWorkspace | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    optimizer_input_identity: str | None = None
    feature_cache_directory: Path | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    feature_derived_cache_directory: Path | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    require_persistent_feature_cache: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        records = tuple(self.candidates)
        sources = tuple(Path(path) for path in self.sources)
        offsets = tuple(float(value) for value in self.audio_offsets)
        timewarps = tuple(self.source_timewarps)
        scene_cuts = tuple(
            tuple(float(value) for value in source_cuts)
            for source_cuts in self.scene_cut_times
        )
        face_metadata = MappingProxyType(dict(self.face_metadata_by_candidate))
        subject_metadata = MappingProxyType(dict(self.subject_metadata_by_candidate))
        role_conditioning = (
            None
            if self.role_conditioned_selection is None
            else MappingProxyType(
                deepcopy(dict(self.role_conditioned_selection))
            )
        )
        object.__setattr__(self, "candidates", records)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "audio_offsets", offsets)
        object.__setattr__(self, "source_timewarps", timewarps)
        object.__setattr__(self, "scene_cut_times", scene_cuts)
        object.__setattr__(self, "face_metadata_by_candidate", face_metadata)
        object.__setattr__(self, "subject_metadata_by_candidate", subject_metadata)
        object.__setattr__(self, "role_conditioned_selection", role_conditioning)
        object.__setattr__(
            self,
            "feature_cache_directory",
            (
                None
                if self.feature_cache_directory is None
                else Path(self.feature_cache_directory)
            ),
        )
        object.__setattr__(
            self,
            "feature_derived_cache_directory",
            (
                None
                if self.feature_derived_cache_directory is None
                else Path(self.feature_derived_cache_directory)
            ),
        )
        if not isinstance(self.require_persistent_feature_cache, bool):
            raise TypeError("require_persistent_feature_cache must be boolean")
        optimizer_input_identity = (
            None
            if self.optimizer_input_identity is None
            else _validated_sha256_identity(
                self.optimizer_input_identity,
                "candidate-set optimizer_input_identity",
            )
        )
        object.__setattr__(
            self,
            "optimizer_input_identity",
            optimizer_input_identity,
        )
        if self.optimizer_workspace is not None and not isinstance(
            self.optimizer_workspace,
            VisualTimelineOptimizerWorkspace,
        ):
            raise TypeError(
                "optimizer_workspace must be a VisualTimelineOptimizerWorkspace"
            )
        if (
            isinstance(self.requested_maximum_results, bool)
            or not isinstance(self.requested_maximum_results, int)
            or not 8 <= self.requested_maximum_results <= 32
        ):
            raise ValueError("requested_maximum_results must be between 8 and 32")
        if not records:
            raise ValueError("visual timeline candidate set cannot be empty")
        if len(records) > self.requested_maximum_results:
            raise ValueError("candidate set exceeds requested_maximum_results")
        if len(sources) < 2:
            raise ValueError("visual timeline candidate set needs at least two sources")
        if len(offsets) != len(sources) or len(timewarps) != len(sources):
            raise ValueError("candidate set source clock inputs must match sources")
        if any(not math.isfinite(value) for value in offsets):
            raise ValueError("candidate set audio offsets must be finite")
        _source_clock_maps(
            timewarps,
            audio_offsets=offsets,
            source_count=len(sources),
        )
        if scene_cuts and len(scene_cuts) != len(sources):
            raise ValueError("candidate set scene cuts must match sources")
        if any(
            not math.isfinite(value)
            or any(right <= left for left, right in zip(cuts, cuts[1:]))
            for cuts in scene_cuts
            for value in cuts
        ):
            raise ValueError("candidate set scene cuts must be finite and ordered")
        if not self.source_clock_basis.strip():
            raise ValueError("source_clock_basis cannot be blank")
        if (
            isinstance(self.minimum_materialization_target, bool)
            or not isinstance(self.minimum_materialization_target, int)
            or self.minimum_materialization_target < 1
        ):
            raise ValueError("minimum_materialization_target must be positive")
        expected_indices = tuple(range(len(records)))
        if tuple(record.candidate_index for record in records) != expected_indices:
            raise ValueError("candidate indices must be contiguous generation order")
        signatures = tuple(record.path_signature for record in records)
        if len(set(signatures)) != len(signatures):
            raise ValueError("candidate set cannot contain duplicate timeline paths")
        if not 0 <= self.selected_candidate_index < len(records):
            raise ValueError("selected_candidate_index is outside the candidate set")
        selected_indices = tuple(
            record.candidate_index for record in records if record.selected
        )
        if selected_indices != (self.selected_candidate_index,):
            raise ValueError("candidate set must mark exactly the selected candidate")
        ranked_indices = _review_ranked_candidate_indices(
            tuple(record.review for record in records)
        )
        rank_by_index = {
            candidate_index: rank
            for rank, candidate_index in enumerate(ranked_indices)
        }
        if any(
            record.review_rank != rank_by_index[record.candidate_index]
            for record in records
        ):
            raise ValueError("candidate review ranks do not match reviewer ordering")
        reference_spans = tuple(
            (
                item.segment_index,
                item.start_seconds,
                item.end_seconds,
            )
            for item in records[0].timeline_result.candidates
        )
        for record in records:
            spans = tuple(
                (item.segment_index, item.start_seconds, item.end_seconds)
                for item in record.timeline_result.candidates
            )
            if spans != reference_spans:
                raise ValueError("candidate alternatives must share canonical spans")
            for item in record.timeline_result.candidates:
                try:
                    source_index = int(item.source_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "visual candidate source_id must be an integer index"
                    ) from exc
                if str(source_index) != item.source_id or not 0 <= source_index < len(
                    sources
                ):
                    raise ValueError(
                        "visual candidate source_id is outside the source list"
                    )
        if self.optimizer_workspace is not None:
            if optimizer_input_identity is None:
                raise ValueError(
                    "candidate set with optimizer workspace needs input identity"
                )
            expected_workspace_identity = _visual_optimizer_input_identity(
                self.optimizer_workspace.candidates_by_segment,
                sources=sources,
                audio_offsets=offsets,
                source_timewarps=timewarps,
                source_clock_basis=self.source_clock_basis,
                scene_cut_times=scene_cuts,
                preferred_source_ids=(
                    self.optimizer_workspace.preferred_source_ids
                ),
                diversity=self.optimizer_workspace.diversity,
                section_plan=self.optimizer_workspace.section_plan,
                phrase_times=self.optimizer_workspace.phrase_times,
                professional_montage_policy=(
                    self.optimizer_workspace.professional_montage_policy
                ),
                subject_metadata_by_candidate=(
                    self.optimizer_workspace.subject_metadata_by_candidate
                ),
                prescribed_source_ids=(
                    self.optimizer_workspace.prescribed_source_ids
                ),
                maximum_results=self.optimizer_workspace.maximum_results,
            )
            if (
                self.optimizer_workspace.input_identity
                != expected_workspace_identity
                or optimizer_input_identity != expected_workspace_identity
            ):
                raise ValueError(
                    "optimizer workspace identity does not match candidate-set inputs"
                )

    @property
    def selected(self) -> VisualTimelineCandidateRecord:
        return self.candidates[self.selected_candidate_index]

    @property
    def ranked_candidates(self) -> tuple[VisualTimelineCandidateRecord, ...]:
        return tuple(sorted(self.candidates, key=lambda item: item.review_rank))

    @property
    def role_conditioned_ranked_candidates(
        self,
    ) -> tuple[VisualTimelineCandidateRecord, ...]:
        """Preserve human-grammar candidates before generic review tie-breaks.

        The scalar metadata estimate only orders candidates for decoded proxy
        review; it never replaces the decoded-output montage authority.
        """

        payload = self.role_conditioned_selection
        if payload is None:
            return self.ranked_candidates
        raw_outcomes = payload.get("candidate_outcomes")
        if not isinstance(raw_outcomes, list):
            raise ValueError(
                "role-conditioned candidate outcomes are missing"
            )
        outcomes: dict[int, Mapping[str, Any]] = {}
        for raw in raw_outcomes:
            if not isinstance(raw, Mapping):
                raise ValueError("role-conditioned candidate outcome is invalid")
            candidate_index = raw.get("candidate_index")
            score = raw.get("metadata_role_attainment_score")
            highlight_score = raw.get(
                "minimum_highlight_metadata_attainment_score"
            )
            passed = raw.get("all_highlight_metadata_floors_passed")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index in outcomes
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                or isinstance(highlight_score, bool)
                or not isinstance(highlight_score, (int, float))
                or not math.isfinite(float(highlight_score))
                or not 0.0 <= float(highlight_score) <= 1.0
                or not isinstance(passed, bool)
            ):
                raise ValueError(
                    "role-conditioned candidate outcome values are invalid"
                )
            outcomes[candidate_index] = raw
        if set(outcomes) != {
            record.candidate_index for record in self.candidates
        }:
            raise ValueError(
                "role-conditioned outcomes do not bind every candidate"
            )

        def priority(record: VisualTimelineCandidateRecord) -> tuple[Any, ...]:
            outcome = outcomes[record.candidate_index]
            return (
                bool(outcome["all_highlight_metadata_floors_passed"]),
                float(
                    outcome[
                        "minimum_highlight_metadata_attainment_score"
                    ]
                ),
                float(outcome["metadata_role_attainment_score"]),
                record.review.passed,
                -len(record.review.hard_failures),
                record.review.weighted_score,
                -record.review_rank,
            )

        qualified = sorted(
            (
                record
                for record in self.candidates
                if bool(
                    outcomes[record.candidate_index][
                        "all_highlight_metadata_floors_passed"
                    ]
                )
            ),
            key=priority,
            reverse=True,
        )
        ordered = list(qualified)
        used = {record.candidate_index for record in ordered}
        remaining_by_profile: dict[
            str, list[VisualTimelineCandidateRecord]
        ] = {}
        for record in self.candidates:
            if record.candidate_index in used:
                continue
            remaining_by_profile.setdefault(record.cadence_profile, []).append(
                record
            )
        for records in remaining_by_profile.values():
            records.sort(key=priority, reverse=True)
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
        while len(ordered) < len(self.candidates):
            appended = False
            for profile in profile_order:
                records = remaining_by_profile[profile]
                if rank < len(records):
                    ordered.append(records[rank])
                    appended = True
            if not appended:
                break
            rank += 1
        if len(ordered) != len(self.candidates):
            raise RuntimeError(
                "role-conditioned candidate ordering lost candidates"
            )
        return tuple(ordered)

    @property
    def minimum_materialization_target_satisfied(self) -> bool:
        return len(self.candidates) >= self.minimum_materialization_target

    @property
    def candidate_review_summaries(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_summary_dict() for record in self.candidates)

    def materialize(
        self,
        candidate: int | VisualTimelineCandidateRecord | None = None,
    ) -> VisualTimelinePlan:
        """Materialize the selected path or any retained runner-up completely."""

        return materialize_visual_timeline_candidate(self, candidate=candidate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_generation_basis": self.candidate_generation_basis,
            "candidate_count": len(self.candidates),
            "requested_maximum_results": self.requested_maximum_results,
            "minimum_materialization_target": self.minimum_materialization_target,
            "minimum_materialization_target_satisfied": (
                self.minimum_materialization_target_satisfied
            ),
            "selected_candidate_index": self.selected_candidate_index,
            "source_count": len(self.sources),
            "source_clock_basis": self.source_clock_basis,
            "optimizer_input_identity": self.optimizer_input_identity,
            "per_shot_quality_soft_floor": _quality_soft_floor_policy(),
            "role_conditioned_selection": (
                deepcopy(dict(self.role_conditioned_selection))
                if self.role_conditioned_selection is not None
                else None
            ),
            "transient_materialization_metadata": {
                "scene_cut_source_count": len(self.scene_cut_times),
                "face_candidate_count": len(self.face_metadata_by_candidate),
                "subject_candidate_count": len(self.subject_metadata_by_candidate),
                "raw_metadata_persisted": False,
            },
            "candidates": [record.to_dict() for record in self.candidates],
            "raw_candidate_embeddings_persisted": False,
        }


def _normalise_timeline_no_goods(
    value: TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey],
) -> TimelineNoGoodSet:
    if isinstance(value, TimelineNoGoodSet):
        return value
    if isinstance(value, (str, bytes)):
        raise TypeError("forbidden_boundary_edges must contain exact edge keys")
    return TimelineNoGoodSet(tuple(value))


def _select_optimizer_workspace_candidates(
    workspace: VisualTimelineOptimizerWorkspace,
    *,
    forbidden_boundary_edges: TimelineNoGoodSet,
) -> tuple[tuple[TimelineResult, ...], tuple[str, ...], list[dict[str, Any]]]:
    role_conditioning_audit: list[dict[str, Any]] = []
    if workspace.prescribed_source_ids is None:
        selection_lattice = [
            list(segment) for segment in workspace.candidates_by_segment
        ]
        timeline_candidates, cadence_profiles = _select_safe_cadence_candidates(
            selection_lattice,
            preferred_source_ids=workspace.preferred_source_ids,
            diversity=workspace.diversity,
            maximum_results=workspace.maximum_results,
            section_plan=workspace.section_plan,
            phrase_times=workspace.phrase_times,
            professional_montage_policy=(
                workspace.professional_montage_policy
            ),
            subject_metadata_by_candidate=(
                workspace.subject_metadata_by_candidate
            ),
            role_conditioning_audit=role_conditioning_audit,
            forbidden_boundary_edges=forbidden_boundary_edges,
        )
        return timeline_candidates, cadence_profiles, role_conditioning_audit

    restricted_candidates: list[tuple[TimelineCandidate, ...]] = []
    for segment, source_id in zip(
        workspace.candidates_by_segment,
        workspace.prescribed_source_ids,
        strict=True,
    ):
        exact = tuple(
            candidate for candidate in segment if candidate.source_id == source_id
        )
        if len(exact) != 1:
            raise ValueError(
                "prescribed source is not visually readable for segment "
                f"{segment[0].segment_index}"
            )
        restricted_candidates.append(exact)
    timeline_candidates = select_timeline_candidates(
        restricted_candidates,
        config=_safe_cadence_profiles(workspace.diversity)[0].config,
        maximum_results=1,
        diversity_penalty=0.12,
        minimum_hamming_fraction=0.02,
        forbidden_boundary_edges=forbidden_boundary_edges,
    )
    return (
        timeline_candidates,
        ("prescribed_balanced_hard_gates",),
        role_conditioning_audit,
    )


def _review_visual_timeline_candidates(
    timeline_candidates: Sequence[TimelineResult],
    cadence_profiles: Sequence[str],
    *,
    available_source_count: int,
) -> tuple[tuple[VisualTimelineCandidateRecord, ...], int]:
    candidates = tuple(timeline_candidates)
    profiles = tuple(cadence_profiles)
    if not candidates or len(candidates) != len(profiles):
        raise ValueError("timeline candidates and cadence profiles must align")
    reviews = tuple(
        review_timeline(
            candidate,
            available_source_count=available_source_count,
        )
        for candidate in candidates
    )
    ranked_indices = _review_ranked_candidate_indices(reviews)
    selected_candidate_index = ranked_indices[0]
    rank_by_index = {
        candidate_index: review_rank
        for review_rank, candidate_index in enumerate(ranked_indices)
    }
    records = tuple(
        VisualTimelineCandidateRecord(
            candidate_index=index,
            cadence_profile=profiles[index],
            timeline_result=candidate,
            review=review,
            review_rank=rank_by_index[index],
            selected=index == selected_candidate_index,
        )
        for index, (candidate, review) in enumerate(
            zip(candidates, reviews, strict=True)
        )
    )
    return records, selected_candidate_index


def _role_conditioned_selection_payload(
    records: Sequence[VisualTimelineCandidateRecord],
    *,
    workspace: VisualTimelineOptimizerWorkspace,
    role_conditioning_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    policy = workspace.professional_montage_policy
    if policy is None:
        return None
    section_plan = workspace.section_plan
    if section_plan is None:  # pragma: no cover - workspace invariant
        raise RuntimeError("montage policy lost its section plan")
    return {
        "contract": (
            "human_reference_role_conditioned_visual_candidate_soft_controls_v1"
        ),
        "policy": policy.to_dict(),
        "section_plan_analysis_basis": section_plan.analysis_basis,
        "phrase_boundary_count": len(
            tuple(sorted({float(value) for value in workspace.phrase_times}))
        ),
        "cadence_profiles": [deepcopy(dict(item)) for item in role_conditioning_audit],
        "intentional_phrase_anchor_transition_authority": {
            "analysis_basis": INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS,
            "scope": ["build", "chorus_like", "bridge_like"],
            "timing": "exact_acoustic_phrase_entry_only",
            "authorized_mismatch_metrics": ["scale", "focus"],
            "maximum_scale_mismatch": (
                INTENTIONAL_PHRASE_ANCHOR_MAX_SCALE_MISMATCH
            ),
            "maximum_focus_mismatch": (
                INTENTIONAL_PHRASE_ANCHOR_MAX_FOCUS_MISMATCH
            ),
            "maximum_pose_mismatch": (
                INTENTIONAL_PHRASE_ANCHOR_MAX_POSE_MISMATCH
            ),
            "layout_silhouette_formation_gates_unchanged": True,
            "scene_cut_and_boundary_side_gates_unchanged": True,
            "ordinary_boundaries_relaxed": False,
        },
        "candidate_outcomes": list(
            _role_conditioned_candidate_outcomes(
                records,
                section_plan=section_plan,
                policy=policy,
                subject_metadata_by_candidate=(
                    workspace.subject_metadata_by_candidate
                ),
            )
        ),
        "raw_visual_descriptors_persisted": False,
        "decoded_output_montage_gate_required": True,
    }


def regenerate_visual_timeline_candidate_set(
    candidate_set: VisualTimelineCandidateSet,
    *,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ),
) -> VisualTimelineCandidateSet:
    """Rerun candidate selection against exact failed edges, with no decode.

    This function intentionally consumes only the transient descriptor lattice
    retained by the original candidate set.  If a set came from a serialized
    checkpoint and no longer has that workspace, regeneration fails closed
    instead of silently rebuilding descriptors or ignoring the no-goods.
    """

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    workspace = candidate_set.optimizer_workspace
    if workspace is None:
        raise ValueError(
            "candidate set has no transient optimizer workspace; rebuild it once"
        )
    expected_workspace_identity = _visual_optimizer_input_identity(
        workspace.candidates_by_segment,
        sources=candidate_set.sources,
        audio_offsets=candidate_set.audio_offsets,
        source_timewarps=candidate_set.source_timewarps,
        source_clock_basis=candidate_set.source_clock_basis,
        scene_cut_times=candidate_set.scene_cut_times,
        preferred_source_ids=workspace.preferred_source_ids,
        diversity=workspace.diversity,
        section_plan=workspace.section_plan,
        phrase_times=workspace.phrase_times,
        professional_montage_policy=workspace.professional_montage_policy,
        subject_metadata_by_candidate=workspace.subject_metadata_by_candidate,
        prescribed_source_ids=workspace.prescribed_source_ids,
        maximum_results=workspace.maximum_results,
    )
    if (
        workspace.input_identity != expected_workspace_identity
        or candidate_set.optimizer_input_identity != expected_workspace_identity
    ):
        raise ValueError(
            "optimizer workspace identity does not match candidate-set inputs"
        )
    no_goods = _normalise_timeline_no_goods(forbidden_boundary_edges)
    timeline_candidates, cadence_profiles, role_audit = (
        _select_optimizer_workspace_candidates(
            workspace,
            forbidden_boundary_edges=no_goods,
        )
    )
    records, selected_candidate_index = _review_visual_timeline_candidates(
        timeline_candidates,
        cadence_profiles,
        available_source_count=len(candidate_set.sources),
    )
    role_conditioning = _role_conditioned_selection_payload(
        records,
        workspace=workspace,
        role_conditioning_audit=role_audit,
    )
    return replace(
        candidate_set,
        candidates=records,
        selected_candidate_index=selected_candidate_index,
        role_conditioned_selection=role_conditioning,
        optimizer_workspace=workspace,
    )


@dataclass(frozen=True, slots=True)
class VisualTimelinePlan:
    choices: tuple[VisualTimelineChoice, ...]
    optimizer_score: float
    analysis_basis: str = (
        "reviewed_thirty_two_candidate_shot_hierarchy_phrase_anchor_contrast_v14"
    )
    selected_candidate_index: int = 0
    candidate_generation_basis: str = (
        "three_safe_cadence_profiles_round_robin_dedup_then_reference_review_v5"
    )
    candidate_review_summaries: tuple[dict[str, Any], ...] = ()
    source_clock_basis: str = "global_scalar_offset"

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "optimizer_score": self.optimizer_score,
            "selected_candidate_index": self.selected_candidate_index,
            "candidate_generation_basis": self.candidate_generation_basis,
            "candidate_count": len(self.candidate_review_summaries) or 1,
            "candidate_review_summaries": list(self.candidate_review_summaries),
            "source_clock_basis": self.source_clock_basis,
            "per_shot_quality_soft_floor": _quality_soft_floor_policy(),
            "face_transition_policy": {
                "detector": "opencv_zoo_yunet_2023mar",
                "appearance_model": SFACE_BACKEND,
                "window": "three_consecutive_synchronized_frame_pairs",
                "objective": "worst_pair_mismatch_penalty",
                "closeup_appearance_gate": "compatible_face_evidence_required",
                "missing_or_ambiguous_face": (
                    "closeup_cross_source_rejected_otherwise_body_layout_fallback"
                ),
                "selection_effect": "screen_geometry_penalty_plus_closeup_appearance_gate",
                "raw_appearance_embeddings_persisted": False,
                "provenance": "independent_modern_adaptation_of_cross_cutting_eye_minimax",
            },
            "dissolve_safety_window": {
                "detector": "opencv_zoo_yunet_2023mar",
                "appearance_model": SFACE_BACKEND,
                "sample_count": 3,
                "handle_frames": _DISSOLVE_HANDLE_FRAMES,
                "timing": "cut_handle_midpoint_handle_end_on_both_sources",
                "scene_cut_or_ambiguous_face": "empty_never_blend",
                "future_body_requirement": (
                    "reliable_pose_or_aggregate_silhouette_at_all_three_samples"
                ),
                "missing_future_body": "tier_b_subject_aligned_hard_cut",
                "purpose": "eligibility_gate_not_timeline_reward",
                "inference_scope": (
                    "post_dp_selected_source_change_boundaries_with_safe_face_candidates"
                ),
                "interpretation": (
                    "appearance_compatibility_gate_not_named_identity_or_authentication"
                ),
                "raw_appearance_embeddings_persisted": False,
            },
            "body_transition_policy": {
                "pose": "opencv_zoo_mediapipe_33_keypoints_when_one_subject_is_dominant",
                "silhouette": "opencv_zoo_pphumanseg_aggregate_human_layout",
                "person_layout": "all_detected_people_screen_space_occupancy_and_boxes",
                "formation_assignment": (
                    "up_to_six_left_to_right_boxes_centroids_and_optional_upper_body"
                ),
                "formation_identity": "not_established_geometry_only",
                "pose_missing": "deterministic_structure_hog_fallback",
                "silhouette_missing": "require_other_screen_space_visual_evidence",
                "selection_effect": (
                    "scale_focus_formation_and_silhouette_hard_gates_plus_dp_penalties"
                ),
                "motion_window": "actual_outgoing_exit_to_incoming_entry",
                "dissolve_effect": (
                    "future_model_evidence_required_for_tier_a_only"
                ),
            },
            "native_scene_cut_policy": {
                "detector_clock": "source_local_seconds",
                "serialized_clock": "canonical_master_seconds",
                "clock_conversion": (
                    "exact_inverse_local_timewarp_or_scalar_offset"
                ),
                "same_source_selection_effect": (
                    "preserve_continuity_bonus_reset_shot_run_skip_crosscut_penalties"
                ),
                "review_effect": (
                    "split_actual_visual_shots_without_midpoint_shot_id_double_count"
                ),
            },
            "choices": [choice.to_dict() for choice in self.choices],
        }


_FACE_CASCADE: cv2.CascadeClassifier | None = None
_YUNET_DETECTORS: dict[tuple[int, int], Any] = {}
_YUNET_DISABLED = False
_YUNET_INTEGRITY_VALID: bool | None = None
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_EXPECTED_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)
_YUNET_MODEL = Path(__file__).resolve().parent / "models" / YUNET_MODEL_NAME


def yunet_model_asset_status(
    *,
    model_path: Path | None = None,
    verify_load: bool = False,
) -> dict[str, Any]:
    """Report exact YuNet integrity and optional FaceDetectorYN loadability."""

    path = model_path or _YUNET_MODEL
    actual_digest: str | None = None
    if path.is_file():
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual_digest = digest.hexdigest()
        except OSError:
            actual_digest = None
    digest_valid = actual_digest == YUNET_EXPECTED_SHA256
    loadable: bool | None = None
    if verify_load and digest_valid:
        try:
            cv2.FaceDetectorYN.create(
                str(path),
                "",
                (320, 180),
                0.76,
                0.3,
                5000,
            )
            loadable = True
        except (cv2.error, AttributeError):
            loadable = False
    return {
        "name": YUNET_MODEL_NAME,
        "path": str(path),
        "present": path.is_file(),
        "digest_valid": digest_valid,
        "loadable": loadable,
        "expected_sha256": YUNET_EXPECTED_SHA256,
        "actual_sha256": actual_digest,
    }


@dataclass(frozen=True, slots=True)
class _DetectedFace:
    bbox_xywh: tuple[float, float, float, float]
    landmarks_xy: tuple[tuple[float, float], ...] | None = None
    confidence: float | None = None


def _face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(str(path))
        _FACE_CASCADE = classifier
    return None if _FACE_CASCADE.empty() else _FACE_CASCADE


def _yunet_detector(width: int, height: int) -> Any | None:
    """Load the bundled MIT-licensed OpenCV Zoo YuNet detector when possible."""

    global _YUNET_DISABLED, _YUNET_INTEGRITY_VALID
    if _YUNET_DISABLED or not _YUNET_MODEL.is_file() or not hasattr(
        cv2, "FaceDetectorYN"
    ):
        return None
    if _YUNET_INTEGRITY_VALID is None:
        _YUNET_INTEGRITY_VALID = bool(
            yunet_model_asset_status()["digest_valid"]
        )
    if not _YUNET_INTEGRITY_VALID:
        _YUNET_DISABLED = True
        return None
    size = (width, height)
    if size not in _YUNET_DETECTORS:
        try:
            _YUNET_DETECTORS[size] = cv2.FaceDetectorYN.create(
                str(_YUNET_MODEL),
                "",
                size,
                0.76,
                0.3,
                5000,
            )
        except (cv2.error, AttributeError):
            _YUNET_DISABLED = True
            return None
    return _YUNET_DETECTORS[size]


def _detect_faces(
    working: np.ndarray,
    gray: np.ndarray,
) -> tuple[tuple[_DetectedFace, ...], str]:
    height, width = gray.shape
    detector = _yunet_detector(width, height)
    if detector is not None:
        try:
            _result, detections = detector.detect(working)
            if detections is None:
                return ((), "yunet")
            faces = tuple(
                _DetectedFace(
                    bbox_xywh=tuple(float(value) for value in detection[:4]),
                    landmarks_xy=tuple(
                        (float(detection[index]), float(detection[index + 1]))
                        for index in range(4, 14, 2)
                    ),
                    confidence=float(detection[-1]),
                )
                for detection in detections
                if float(detection[-1]) >= 0.76
            )
            return (faces, "yunet")
        except cv2.error:
            # A malformed frame/model mismatch is not a reason to abort the
            # complete edit. Haar remains a local, dependency-free fallback.
            pass
    cascade = _face_cascade()
    if cascade is None or min(gray.shape) < 48:
        return ((), "none")
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.16,
        minNeighbors=4,
        minSize=(18, 18),
    )
    return (
        tuple(
            _DetectedFace(tuple(float(value) for value in box))
            for box in boxes
        ),
        "haar",
    )


def _working_frame(frame: np.ndarray, maximum_width: int = 480) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("frame must be a non-empty grayscale or BGR image")
    height, width = frame.shape[:2]
    if width > maximum_width:
        scale = maximum_width / width
        frame = cv2.resize(
            frame,
            (maximum_width, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError("frame must use 1, 3, or 4 channels")
    return frame


def _dark_foreground_obstruction_score(
    gray: np.ndarray,
    *,
    attention_bbox_xywh: tuple[float, float, float, float] | None,
    secondary_attention_bbox_xywh: (
        tuple[float, float, float, float] | None
    ) = None,
) -> float:
    """Score unmistakable dark foreground camera equipment.

    Two deliberately narrow shapes are recognized.  A boom is a thick,
    shallow-angle connected component spanning both horizontal edges.  A
    camera rig is a large top-edge component that overlaps the detected
    subject/face attention box.  Requiring edge contact, component thickness,
    normalized elongation, boundary contrast, and performance-area overlap
    avoids treating black clothing or a dark stage background as equipment.

    Only the final scalar is retained by :class:`FrameVisualDescriptor`.
    """

    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.size == 0:
        raise ValueError("obstruction scoring needs a non-empty grayscale frame")
    height, width = gray.shape
    if min(height, width) < 32:
        return 0.0

    # True foreground equipment in the target fancams is nearly black.  An
    # absolute threshold is intentionally conservative: adapting to each
    # frame's percentile would turn ordinary dark stages into foreground.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.0)
    dark = (blurred <= 20).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        dark,
        connectivity=8,
    )
    frame_area = float(height * width)
    best_score = 0.0

    def _maximum_attention_overlap(component: np.ndarray) -> float:
        overlap = 0.0
        for attention_bbox in (
            attention_bbox_xywh,
            secondary_attention_bbox_xywh,
        ):
            if attention_bbox is None:
                continue
            attention_x, attention_y, attention_width, attention_height = (
                float(value) for value in attention_bbox
            )
            x0 = max(0, min(width, int(attention_x * width)))
            y0 = max(0, min(height, int(attention_y * height)))
            x1 = max(
                x0,
                min(width, int((attention_x + attention_width) * width)),
            )
            y1 = max(
                y0,
                min(height, int((attention_y + attention_height) * height)),
            )
            attention_component = component[y0:y1, x0:x1]
            if attention_component.size:
                overlap = max(overlap, float(attention_component.mean()))
        return overlap

    for component_index in range(1, component_count):
        x, y, component_width, component_height, component_area = (
            int(value) for value in stats[component_index]
        )
        area_ratio = component_area / frame_area
        if not 0.05 <= area_ratio <= 0.36:
            continue

        component = labels == component_index
        moments = cv2.moments(component.astype(np.uint8), binaryImage=True)
        if moments["m00"] <= 1.0:
            continue
        covariance_xx = moments["mu20"] / moments["m00"] / (width * width)
        covariance_yy = moments["mu02"] / moments["m00"] / (height * height)
        covariance_xy = moments["mu11"] / moments["m00"] / (width * height)
        trace = covariance_xx + covariance_yy
        discriminant = math.sqrt(
            max(
                0.0,
                (covariance_xx - covariance_yy) ** 2
                + 4.0 * covariance_xy * covariance_xy,
            )
        )
        major_variance = max(0.0, (trace + discriminant) / 2.0)
        minor_variance = max(1e-9, (trace - discriminant) / 2.0)
        elongation = math.sqrt(major_variance / minor_variance)
        angle_degrees = abs(
            math.degrees(
                0.5
                * math.atan2(
                    2.0 * covariance_xy,
                    covariance_xx - covariance_yy,
                )
            )
        )
        angle_from_horizontal = min(angle_degrees, 180.0 - angle_degrees)

        width_ratio = component_width / width
        height_ratio = component_height / height
        touches_left = x <= 2
        touches_right = x + component_width >= width - 2
        touches_top = y <= 2
        touches_top_region = y <= int(round(0.04 * height))

        center_component = component[
            int(0.08 * height) : int(0.60 * height),
            int(0.20 * width) : int(0.80 * width),
        ]
        center_coverage = (
            float(center_component.mean()) if center_component.size else 0.0
        )

        attention_overlap = _maximum_attention_overlap(component)

        expanded = cv2.dilate(
            component.astype(np.uint8),
            np.ones((7, 7), dtype=np.uint8),
        ).astype(bool)
        outside_ring = expanded & ~component
        boundary_contrast = (
            float(gray[outside_ring].mean() - gray[component].mean())
            if outside_ring.any()
            else 0.0
        )

        boom = bool(
            width_ratio >= 0.94
            and touches_left
            and touches_right
            and area_ratio >= 0.13
            and elongation >= 2.9
            and 4.0 <= angle_from_horizontal <= 35.0
            and center_coverage >= 0.22
            and boundary_contrast >= 25.0
        )
        rig = bool(
            touches_top
            and 0.62 <= width_ratio <= 0.92
            and 0.11 <= area_ratio <= 0.28
            and 0.34 <= height_ratio <= 0.72
            and angle_from_horizontal <= 30.0
            and center_coverage >= 0.23
            and attention_overlap >= 0.10
            and boundary_contrast >= 28.0
        )
        # Some broadcast rigs fill almost the complete top width, so their
        # irregular housing is not elongated enough to look like a boom.  The
        # stronger central/subject overlap and bounded top-region geometry
        # distinguish that case from a normal black backdrop.
        large_rig = bool(
            touches_top_region
            and 0.13 <= area_ratio <= 0.25
            and width_ratio >= 0.80
            and 0.45 <= height_ratio <= 0.70
            and center_coverage >= 0.31
            and attention_overlap >= 0.12
            and boundary_contrast >= 30.0
        )

        # A boom can also enter from below.  Once it merges with a dark stage
        # band, component PCA looks almost horizontal even though the visible
        # foreground edge still sweeps diagonally across the frame.  Measure
        # robust per-column component envelopes to retain that directional
        # evidence while rejecting a genuinely horizontal black backdrop.
        edge_envelope_slope = 0.0
        lower_crossing_geometry = bool(
            touches_left
            and touches_right
            and width_ratio >= 0.94
            and 0.16 <= area_ratio <= 0.33
            and 0.30 <= height_ratio <= 0.72
            and 0.08 <= y / height <= 0.50
            and (y + component_height) / height <= 0.87
            and 0.40
            <= moments["m01"] / moments["m00"] / height
            <= 0.66
            and elongation >= 1.9
            and angle_from_horizontal <= 12.0
            and center_coverage >= 0.19
            and attention_overlap >= 0.18
            and boundary_contrast >= 35.0
        )
        if lower_crossing_geometry:
            bin_centers: list[float] = []
            top_envelope: list[float] = []
            bottom_envelope: list[float] = []
            for bin_index in range(24):
                bin_start = bin_index * width // 24
                bin_end = (bin_index + 1) * width // 24
                component_rows = np.nonzero(component[:, bin_start:bin_end])[0]
                if component_rows.size == 0:
                    continue
                bin_centers.append((bin_start + bin_end) / (2.0 * width))
                top_envelope.append(
                    float(np.quantile(component_rows, 0.03)) / height
                )
                bottom_envelope.append(
                    float(np.quantile(component_rows, 0.97)) / height
                )

            def _envelope_slope(values: Sequence[float]) -> float:
                if len(values) != len(bin_centers) or len(values) < 12:
                    return 0.0
                center_mean = sum(bin_centers) / len(bin_centers)
                value_mean = sum(values) / len(values)
                denominator = sum(
                    (center - center_mean) ** 2 for center in bin_centers
                )
                if denominator <= 1e-9:
                    return 0.0
                return sum(
                    (center - center_mean) * (value - value_mean)
                    for center, value in zip(bin_centers, values, strict=True)
                ) / denominator

            edge_envelope_slope = max(
                abs(_envelope_slope(top_envelope)),
                abs(_envelope_slope(bottom_envelope)),
            )
        lower_crossing_boom = bool(
            lower_crossing_geometry and edge_envelope_slope >= 0.165
        )

        # Near-lens broadcast cameras can be tall rather than elongated and
        # need not touch the top or both side edges.  Their unusually large
        # central coverage, subject overlap, boundary contrast and bounded
        # area distinguish them from black clothing and distant stage pieces.
        near_lens_rig = bool(
            touches_top_region
            and 0.20 <= area_ratio <= 0.34
            and 0.65 <= width_ratio <= 0.92
            and 0.70 <= height_ratio <= 0.96
            and 0.28
            <= moments["m01"] / moments["m00"] / height
            <= 0.56
            and center_coverage >= 0.35
            and attention_overlap >= 0.20
            and boundary_contrast >= 40.0
        )

        edge_cable_shape = bool(
            (touches_left or touches_top_region)
            and 0.06 <= area_ratio <= 0.13
            and 0.40 <= width_ratio <= 0.68
            and 0.42 <= height_ratio <= 0.62
            and center_coverage >= 0.10
            and attention_overlap >= 0.06
            and boundary_contrast >= 40.0
        )
        large_loop_shape = bool(
            touches_left
            and touches_right
            and 0.13 <= area_ratio <= 0.22
            and width_ratio >= 0.95
            and 0.70 <= height_ratio <= 0.90
            and 0.08 <= y / height <= 0.18
            and center_coverage >= 0.25
            and attention_overlap >= 0.15
            and boundary_contrast >= 55.0
        )
        edge_cable_rig = False
        large_loop_rig = False
        if edge_cable_shape or large_loop_shape:
            component_u8 = component.astype(np.uint8)
            contours, _hierarchy = cv2.findContours(
                component_u8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                solidity = (
                    float(cv2.contourArea(contour)) / hull_area
                    if hull_area > 1.0
                    else 1.0
                )
                rectangle = cv2.minAreaRect(contour)
                rectangle_area = float(rectangle[1][0] * rectangle[1][1])
                rectangle_fill = (
                    component_area / rectangle_area
                    if rectangle_area > 1.0
                    else 1.0
                )
                edge_cable_rig = bool(
                    edge_cable_shape
                    and 0.35 <= solidity <= 0.60
                    and 0.25 <= rectangle_fill <= 0.45
                )
                large_loop_rig = bool(
                    large_loop_shape
                    and solidity <= 0.45
                    and rectangle_fill <= 0.30
                )

        if boom or lower_crossing_boom:
            best_score = max(best_score, 1.0)
        elif edge_cable_rig or large_loop_rig:
            best_score = max(best_score, 0.93)
        elif rig or large_rig or near_lens_rig:
            # Preserve a little scalar confidence variation for diagnostics;
            # the strict geometric gates above carry the classification.
            best_score = max(
                best_score,
                _unit(0.88 + 0.12 * (center_coverage - 0.23) / 0.25),
            )
    if best_score < 0.85:
        # Dark textured housings can contain enough reflections that the
        # strict near-black mask fragments them.  A second, still-conservative
        # threshold is used only for a wide, irregular top-edge assembly that
        # overlaps the subject.  The low solidity/fill requirements keep LED
        # panels, vertical lighting trusses, and black costumes out.
        soft_dark = (blurred <= 32).astype(np.uint8)
        soft_dark = cv2.morphologyEx(
            soft_dark,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        (
            soft_component_count,
            soft_labels,
            soft_stats,
            _soft_centroids,
        ) = cv2.connectedComponentsWithStats(soft_dark, connectivity=8)
        for component_index in range(1, soft_component_count):
            x, y, component_width, component_height, component_area = (
                int(value) for value in soft_stats[component_index]
            )
            area_ratio = component_area / frame_area
            width_ratio = component_width / width
            height_ratio = component_height / height
            touches_left = x <= 2
            touches_right = x + component_width >= width - 2
            touches_top = y <= 2
            if not (
                touches_top
                and (touches_left or touches_right)
                and 0.025 <= area_ratio <= 0.065
                and 0.72 <= width_ratio <= 0.90
                and 0.18 <= height_ratio <= 0.40
            ):
                continue
            component = soft_labels == component_index
            top_center = component[
                : int(0.45 * height),
                int(0.10 * width) : int(0.90 * width),
            ]
            top_center_coverage = (
                float(top_center.mean()) if top_center.size else 0.0
            )
            attention_overlap = _maximum_attention_overlap(component)
            if top_center_coverage < 0.08 or attention_overlap < 0.035:
                continue
            moments = cv2.moments(component.astype(np.uint8), binaryImage=True)
            if moments["m00"] <= 1.0:
                continue
            covariance_xx = moments["mu20"] / moments["m00"] / (width * width)
            covariance_yy = moments["mu02"] / moments["m00"] / (height * height)
            covariance_xy = moments["mu11"] / moments["m00"] / (width * height)
            trace = covariance_xx + covariance_yy
            discriminant = math.sqrt(
                max(
                    0.0,
                    (covariance_xx - covariance_yy) ** 2
                    + 4.0 * covariance_xy * covariance_xy,
                )
            )
            major_variance = max(0.0, (trace + discriminant) / 2.0)
            minor_variance = max(1e-9, (trace - discriminant) / 2.0)
            elongation = math.sqrt(major_variance / minor_variance)
            angle_degrees = abs(
                math.degrees(
                    0.5
                    * math.atan2(
                        2.0 * covariance_xy,
                        covariance_xx - covariance_yy,
                    )
                )
            )
            angle_from_horizontal = min(angle_degrees, 180.0 - angle_degrees)
            if not (
                2.2 <= elongation <= 5.0
                and 5.0 <= angle_from_horizontal <= 35.0
            ):
                continue
            expanded = cv2.dilate(
                component.astype(np.uint8),
                np.ones((7, 7), dtype=np.uint8),
            ).astype(bool)
            outside_ring = expanded & ~component
            boundary_contrast = (
                float(gray[outside_ring].mean() - gray[component].mean())
                if outside_ring.any()
                else 0.0
            )
            contours, _hierarchy = cv2.findContours(
                component.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = (
                float(cv2.contourArea(contour)) / hull_area
                if hull_area > 1.0
                else 1.0
            )
            rectangle = cv2.minAreaRect(contour)
            rectangle_area = float(rectangle[1][0] * rectangle[1][1])
            rectangle_fill = (
                component_area / rectangle_area if rectangle_area > 1.0 else 1.0
            )
            if (
                boundary_contrast >= 40.0
                and 0.20 <= solidity <= 0.45
                and 0.12 <= rectangle_fill <= 0.32
            ):
                best_score = max(best_score, 0.94)
    return best_score


def _structure_embedding(gray: np.ndarray) -> tuple[float, ...]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    rows, columns, bins = 4, 6, 4
    embedding: list[float] = []
    height, width = gray.shape
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x0, x1 = column * width // columns, (column + 1) * width // columns
            cell_magnitude = magnitude[y0:y1, x0:x1]
            cell_angle = angle[y0:y1, x0:x1]
            histogram = np.zeros(bins, dtype=np.float64)
            indices = np.floor(cell_angle * bins / (2 * math.pi)).astype(np.int32) % bins
            for bin_index in range(bins):
                histogram[bin_index] = float(cell_magnitude[indices == bin_index].sum())
            total = float(histogram.sum())
            if total > 1e-9:
                histogram /= total
            embedding.extend(float(value) for value in histogram)
    return tuple(embedding)


def _color_embedding(frame: np.ndarray, gray: np.ndarray) -> tuple[float, ...]:
    """Compact global color/exposure signature used only at cut boundaries."""

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    means = lab.reshape(-1, 3).mean(axis=0) / 255.0
    deviations = lab.reshape(-1, 3).std(axis=0) / 128.0
    return tuple(
        float(min(1.0, max(0.0, value)))
        for value in (
            *means,
            *deviations,
            float(gray.mean()) / 255.0,
            float(gray.std()) / 128.0,
        )
    )


def _motion(gray: np.ndarray, following: np.ndarray | None) -> tuple[float, float] | None:
    if following is None:
        return None
    following_gray = cv2.cvtColor(_working_frame(following), cv2.COLOR_BGR2GRAY)
    if following_gray.shape != gray.shape:
        following_gray = cv2.resize(following_gray, (gray.shape[1], gray.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(
        gray,
        following_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    )
    height, width = gray.shape
    center = flow[height // 6 : height * 5 // 6, width // 6 : width * 5 // 6]
    if center.size == 0:
        return None
    x = float(np.median(center[:, :, 0])) / max(1.0, width * 0.03)
    y = float(np.median(center[:, :, 1])) / max(1.0, height * 0.03)
    return (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))


def _motion_between_frames(
    frame: np.ndarray,
    following_frame: np.ndarray | None,
) -> tuple[float, float] | None:
    """Measure forward-time motion without re-running face/body models."""

    gray = cv2.cvtColor(_working_frame(frame), cv2.COLOR_BGR2GRAY)
    return _motion(gray, following_frame)


def _dominant_face(faces: Sequence[_DetectedFace]) -> _DetectedFace | None:
    """Return one unambiguous dominant face without guessing in group shots."""

    if not faces:
        return None
    ordered = sorted(
        faces,
        key=lambda face: face.bbox_xywh[2] * face.bbox_xywh[3],
        reverse=True,
    )
    largest = ordered[0]
    if len(ordered) > 1:
        largest_area = float(largest.bbox_xywh[2] * largest.bbox_xywh[3])
        runner_up_area = float(
            ordered[1].bbox_xywh[2] * ordered[1].bbox_xywh[3]
        )
        # A group shot often contains several valid faces. Only a clearly
        # dominant detection is safe enough to drive a crop.
        if largest_area < runner_up_area * 1.8:
            return None
    return largest


def _normalized_face_metadata(
    face: _DetectedFace | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[tuple[float, float], ...] | None,
    float | None,
    float | None,
    tuple[float, ...] | None,
]:
    """Normalize YuNet's five points and derive stable 2D head geometry.

    The compact geometry vector is intentionally detector-agnostic at the DP
    boundary: image-left/right eye endpoints, eye scale, roll, and nose/mouth
    offsets in the eye-aligned coordinate system. Haar detections retain the
    crop bbox but return no geometry, making the transition term neutral.
    """

    if face is None:
        return (None, None, None, None, None)
    x, y, width, height = face.bbox_xywh
    left_px = max(0.0, min(float(frame_width), x))
    top_px = max(0.0, min(float(frame_height), y))
    right_px = max(0.0, min(float(frame_width), x + width))
    bottom_px = max(0.0, min(float(frame_height), y + height))
    if right_px <= left_px or bottom_px <= top_px:
        return (None, None, None, None, None)
    bbox = (
        left_px / frame_width,
        top_px / frame_height,
        (right_px - left_px) / frame_width,
        (bottom_px - top_px) / frame_height,
    )
    if face.landmarks_xy is None or len(face.landmarks_xy) != 5:
        return (bbox, None, None, None, None)
    normalized_landmarks = tuple(
        (_unit(point_x / frame_width), _unit(point_y / frame_height))
        for point_x, point_y in face.landmarks_xy
    )
    # OpenCV YuNet returns two eyes, nose, and two mouth corners. Canonicalize
    # eye direction by image x so roll sign is stable across detector versions.
    eye_left_px, eye_right_px = sorted(face.landmarks_xy[:2], key=lambda point: point[0])
    delta_x = eye_right_px[0] - eye_left_px[0]
    delta_y = eye_right_px[1] - eye_left_px[1]
    eye_distance_px = math.hypot(delta_x, delta_y)
    if eye_distance_px < 2.0:
        return (bbox, normalized_landmarks, None, None, None)
    roll = math.atan2(delta_y, delta_x)
    eye_scale = eye_distance_px / frame_height
    cosine = delta_x / eye_distance_px
    sine = delta_y / eye_distance_px
    eye_mid_x = (eye_left_px[0] + eye_right_px[0]) / 2.0
    eye_mid_y = (eye_left_px[1] + eye_right_px[1]) / 2.0

    def local_offset(point: tuple[float, float]) -> tuple[float, float]:
        point_x = point[0] - eye_mid_x
        point_y = point[1] - eye_mid_y
        return (
            (point_x * cosine + point_y * sine) / eye_distance_px,
            (-point_x * sine + point_y * cosine) / eye_distance_px,
        )

    nose = local_offset(face.landmarks_xy[2])
    mouth_corners = sorted(
        (local_offset(face.landmarks_xy[3]), local_offset(face.landmarks_xy[4])),
        key=lambda point: point[0],
    )
    mouth_vertical = (mouth_corners[0][1] + mouth_corners[1][1]) / 2.0
    mouth_asymmetry = mouth_corners[1][1] - mouth_corners[0][1]
    normalized_eye_left, normalized_eye_right = sorted(
        normalized_landmarks[:2], key=lambda point: point[0]
    )
    geometry = (
        normalized_eye_left[0],
        normalized_eye_left[1],
        normalized_eye_right[0],
        normalized_eye_right[1],
        eye_scale,
        roll,
        # Five 2D points cannot provide a full 3D head pose, but these stable
        # eye-aligned quantities act as lightweight yaw/pitch/asymmetry proxies.
        nose[0],
        nose[1],
        mouth_vertical,
        mouth_asymmetry,
    )
    return (
        bbox,
        normalized_landmarks,
        roll,
        eye_scale,
        geometry,
    )


def describe_frame(
    frame: np.ndarray,
    *,
    following_frame: np.ndarray | None = None,
    include_subject_geometry: bool = True,
    include_formation_pose: bool = False,
) -> FrameVisualDescriptor:
    working = _working_frame(frame)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    structure_embedding = _structure_embedding(gray)
    subject = (
        describe_subject_geometry(
            working,
            include_formation_pose=include_formation_pose,
        )
        if include_subject_geometry
        else SubjectGeometryDescriptor()
    )
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = 1.0 - math.exp(-laplacian_variance / 180.0)
    mean = float(gray.mean())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    highlight_clipping_ratio = float(np.mean(gray > 247))
    exposure = math.exp(-((mean - 118.0) / 105.0) ** 2) * (1.0 - min(0.8, clipped))
    contrast_score = min(1.0, contrast / 58.0)
    quality = _unit(0.45 * sharpness + 0.32 * exposure + 0.23 * contrast_score)

    edges = cv2.Canny(gray, 50, 140).astype(np.float32) / 255.0
    moments = cv2.moments(edges)
    if moments["m00"] > 1e-6:
        center_x = moments["m10"] / moments["m00"] / gray.shape[1]
        center_y = moments["m01"] / moments["m00"] / gray.shape[0]
        center_distance = math.hypot(center_x - 0.5, center_y - 0.48) / 0.72
        composition = 1.0 - _unit(center_distance)
    else:
        composition = 0.35

    if subject.silhouette_centroid_xy is not None:
        subject_x, subject_y = subject.silhouette_centroid_xy
        subject_center_score = 1.0 - _unit(
            math.hypot(subject_x - 0.5, subject_y - 0.50) / 0.72
        )
        composition = _unit(0.72 * composition + 0.28 * subject_center_score)

    faces, face_detector_backend = _detect_faces(working, gray)
    face_count = len(faces)
    dominant_face = _dominant_face(faces)
    (
        primary_face_bbox,
        primary_face_landmarks,
        primary_face_roll,
        primary_face_eye_scale,
        primary_face_geometry,
    ) = _normalized_face_metadata(
        dominant_face,
        frame_width=gray.shape[1],
        frame_height=gray.shape[0],
    )
    foreground_obstruction_score = _dark_foreground_obstruction_score(
        gray,
        attention_bbox_xywh=primary_face_bbox,
        secondary_attention_bbox_xywh=subject.silhouette_bbox_xywh,
    )
    if face_count:
        largest = max(
            faces,
            key=lambda face: face.bbox_xywh[2] * face.bbox_xywh[3],
        )
        x, y, width, height = largest.bbox_xywh
        face_scale = height / gray.shape[0]
        face_center_x = (x + width / 2) / gray.shape[1]
        face_center_y = (y + height / 2) / gray.shape[0]
        face_center_score = 1.0 - _unit(
            math.hypot(face_center_x - 0.5, face_center_y - 0.42) / 0.68
        )
        composition = _unit(0.55 * composition + 0.45 * face_center_score)
        shot_scale = _unit(face_scale / 0.42)
        shot_scale_basis = "dominant_face_height"
    else:
        # Face detection is a poor wide-shot oracle: profile faces, motion
        # blur and stage lighting routinely hide a face in a tight shot.  Use
        # the decoded full-person/aggregate geometry before falling back to an
        # explicitly unknown neutral label.
        people = tuple(subject.formation_people)
        if people:
            heights = sorted(float(person.bbox_xywh[3]) for person in people)
            median_height = heights[len(heights) // 2]
            union_left = min(person.bbox_xywh[0] for person in people)
            union_top = min(person.bbox_xywh[1] for person in people)
            union_right = max(
                person.bbox_xywh[0] + person.bbox_xywh[2] for person in people
            )
            union_bottom = max(
                person.bbox_xywh[1] + person.bbox_xywh[3] for person in people
            )
            union_width = union_right - union_left
            boundary_clipped = union_top <= 0.035 or union_bottom >= 0.965
            if len(people) >= 3:
                shot_scale = min(0.21, max(0.08, median_height * 0.30))
                shot_type = ShotType.GROUP_WIDE
                shot_scale_basis = "aggregate_formation_people"
            elif (
                len(people) == 1
                and boundary_clipped
                and median_height >= 0.68
                and union_width <= 0.58
            ):
                shot_scale = max(0.53, min(0.78, median_height * 0.78))
                shot_type = ShotType.CLOSE_UP
                shot_scale_basis = "clipped_single_performer_bbox"
            elif median_height >= 0.43 and union_width <= 0.72:
                shot_scale = max(0.28, min(0.50, median_height * 0.56))
                shot_type = ShotType.MEDIUM
                shot_scale_basis = "single_or_pair_performer_bbox"
            else:
                shot_scale = min(0.21, max(0.10, median_height * 0.34))
                shot_type = ShotType.GROUP_WIDE
                shot_scale_basis = "distant_single_or_pair_performer_bbox"
        elif (
            subject.silhouette_bbox_xywh is not None
            and subject.silhouette_confidence >= 0.72
        ):
            left, top, width, height = subject.silhouette_bbox_xywh
            boundary_clipped = top <= 0.035 or top + height >= 0.965
            if boundary_clipped and height >= 0.70 and width <= 0.46:
                shot_scale = max(0.53, min(0.72, height * 0.72))
                shot_type = ShotType.CLOSE_UP
                shot_scale_basis = "clipped_single_subject_silhouette"
            elif height >= 0.54 and width <= 0.54:
                shot_scale = max(0.28, min(0.50, height * 0.58))
                shot_type = ShotType.MEDIUM
                shot_scale_basis = "single_subject_silhouette"
            else:
                shot_scale = min(0.21, max(0.10, height * 0.30))
                shot_type = ShotType.GROUP_WIDE
                shot_scale_basis = "aggregate_or_distant_silhouette"
        else:
            # ShotType has no UNKNOWN case and is persisted in older plans.
            # MEDIUM is the least destructive optimizer label; the basis is
            # the authoritative signal that review/statistics must treat this
            # observation as unknown rather than an observed medium shot.
            shot_scale = 0.34
            shot_type = ShotType.MEDIUM
            shot_scale_basis = "unknown_no_subject_geometry"

    if face_count:
        if face_count < 4 and shot_scale >= 0.52:
            shot_type = ShotType.CLOSE_UP
        elif len(subject.formation_people) >= 3:
            heights = sorted(
                float(person.bbox_xywh[3])
                for person in subject.formation_people
            )
            shot_scale = min(
                0.21,
                max(0.08, heights[len(heights) // 2] * 0.30),
            )
            shot_type = ShotType.GROUP_WIDE
            shot_scale_basis = "aggregate_formation_people"
        elif face_count >= 4 or shot_scale < 0.22:
            shot_type = ShotType.GROUP_WIDE
        elif shot_scale < 0.52:
            shot_type = ShotType.MEDIUM
    return FrameVisualDescriptor(
        quality_score=quality,
        composition_score=composition,
        shot_scale=shot_scale,
        shot_type=shot_type,
        face_count=face_count,
        structure_embedding=structure_embedding,
        motion=_motion(gray, following_frame),
        primary_face_bbox=primary_face_bbox,
        color_embedding=_color_embedding(working, gray),
        face_detector_backend=face_detector_backend,
        primary_face_landmarks_xy=primary_face_landmarks,
        primary_face_head_roll_radians=primary_face_roll,
        primary_face_eye_scale=primary_face_eye_scale,
        primary_face_geometry=primary_face_geometry,
        pose_embedding=(
            subject.pose_embedding
            if subject.pose_embedding
            else structure_embedding
        ),
        pose_confidence=subject.pose_confidence,
        detected_person_count=subject.detected_person_count,
        person_layout_embedding=subject.person_layout_embedding,
        formation_people=subject.formation_people,
        pose_backend=(
            subject.pose_backend
            if subject.pose_embedding
            else "structure_hog_fallback"
        ),
        silhouette_embedding=subject.silhouette_embedding,
        silhouette_bbox_xywh=subject.silhouette_bbox_xywh,
        silhouette_centroid_xy=subject.silhouette_centroid_xy,
        silhouette_area_ratio=subject.silhouette_area_ratio,
        silhouette_confidence=subject.silhouette_confidence,
        silhouette_backend=subject.silhouette_backend,
        foreground_obstruction_score=foreground_obstruction_score,
        highlight_clipping_ratio=highlight_clipping_ratio,
        shot_scale_basis=shot_scale_basis,
    )


def _face_match_candidate_failure(
    descriptors: tuple[
        FrameVisualDescriptor,
        FrameVisualDescriptor,
        FrameVisualDescriptor,
    ],
    *,
    has_internal_camera_cut: bool,
) -> str | None:
    """Gate face-driven crops without claiming identity recognition."""

    if has_internal_camera_cut:
        return "internal_camera_cut"
    boxes = tuple(item.primary_face_bbox for item in descriptors)
    if any(box is None for box in boxes):
        if any(item.face_count == 0 for item in descriptors):
            return "face_not_detected"
        return "ambiguous_multiple_faces"
    safe_boxes = tuple(box for box in boxes if box is not None)
    if any(box[3] < 0.035 for box in safe_boxes):
        return "face_too_small"
    centers = tuple(
        (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        for box in safe_boxes
    )
    maximum_center_span = max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )
    if maximum_center_span > 0.28:
        return "unstable_face_position"
    heights = tuple(box[3] for box in safe_boxes)
    if max(heights) / min(heights) > 1.8:
        return "unstable_face_scale"
    return None


@dataclass(frozen=True, slots=True)
class _CandidateFaceMetadata:
    boxes: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ]
    failure: str | None
    backends: tuple[str, str, str]
    landmarks: tuple[
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
    ]
    rolls: tuple[float | None, float | None, float | None]
    eye_scales: tuple[float | None, float | None, float | None]


@dataclass(frozen=True, slots=True)
class _DissolveEvidence:
    """Transient cut/mid/end evidence for one selected source handle.

    These vectors never enter timeline scoring and the appearance/body
    embeddings are intentionally omitted from the serialized analysis.  An
    empty field is a safety failure, never permission to blend.
    """

    face_geometry: tuple[tuple[float, ...], ...] = ()
    face_boxes: tuple[tuple[float, float, float, float], ...] = ()
    face_identity_embeddings: tuple[tuple[float, ...], ...] = ()
    pose_embeddings: tuple[tuple[float, ...], ...] = ()
    pose_backends: tuple[str, ...] = ()
    pose_confidences: tuple[float, ...] = ()
    silhouette_embeddings: tuple[tuple[float, ...], ...] = ()
    silhouette_backends: tuple[str, ...] = ()
    silhouette_confidences: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _CandidateSubjectMetadata:
    pose_backends: tuple[str, str, str]
    pose_confidences: tuple[float, float, float]
    person_counts: tuple[int, int, int]
    formation_person_counts: tuple[int, int, int]
    formation_upper_body_counts: tuple[int, int, int]
    shot_scale_bases: tuple[str, str, str]
    silhouette_boxes: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ]
    silhouette_centroids: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]
    silhouette_backends: tuple[str, str, str]
    silhouette_confidences: tuple[float, float, float]


_BOUNDARY_WINDOW_FRAMES = 3
_BOUNDARY_FRAME_STEP_SECONDS = 1.0 / 29.97
_DISSOLVE_HANDLE_FRAMES = 11
_DISSOLVE_HANDLE_SECONDS = _DISSOLVE_HANDLE_FRAMES / 29.97
_VISUAL_ANALYZER_FINGERPRINT: AnalyzerFingerprint | None = None


def _visual_analyzer_fingerprint() -> AnalyzerFingerprint:
    """Bind persisted descriptors to code, models, schema, and runtime."""

    global _VISUAL_ANALYZER_FINGERPRINT
    if _VISUAL_ANALYZER_FINGERPRINT is not None:
        return _VISUAL_ANALYZER_FINGERPRINT
    yunet = yunet_model_asset_status(verify_load=True)
    pose_assets = pose_model_asset_status(verify_load=True)
    model_assets = {
        f"pose:{status.name}": (
            f"{status.actual_sha256 or 'missing'}:loadable={status.loadable}"
        )
        for status in pose_assets
    }
    model_assets[f"face:{YUNET_MODEL_NAME}"] = str(
        f"{yunet.get('actual_sha256') or 'missing'}:"
        f"loadable={yunet.get('loadable')}"
    )
    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if haar_path.is_file():
        haar_digest = hashlib.sha256(haar_path.read_bytes()).hexdigest()
    else:
        haar_digest = "missing"
    model_assets["face:haarcascade_frontalface_default.xml"] = haar_digest
    opencv_build_digest = hashlib.sha256(
        cv2.getBuildInformation().encode("utf-8")
    ).hexdigest()
    descriptor_implementation_digest = python_symbol_dependency_sha256(
        Path(__file__),
        root_symbols=(
            "FrameVisualDescriptor",
            "_VideoSampler",
            "_descriptor_cache_payload",
            "_descriptor_from_cache_payload",
            "describe_frame",
        ),
    )
    _VISUAL_ANALYZER_FINGERPRINT = build_analyzer_fingerprint(
        descriptor_schema=(
            "frame_visual_descriptor_v1_numeric_geometry_pose_silhouette_"
            "nanodet_formation_motion_no_sface_no_raw_frame"
        ),
        code_paths=(Path(__file__).with_name("pose.py"),),
        code_digests={
            "visual_score_descriptor_ast_v1": (
                descriptor_implementation_digest
            ),
        },
        model_assets=model_assets,
        runtime_components={
            "opencv": str(cv2.__version__),
            "opencv_build_sha256": opencv_build_digest,
            "numpy": str(np.__version__),
        },
    )
    return _VISUAL_ANALYZER_FINGERPRINT


def _descriptor_cache_payload(
    descriptor: FrameVisualDescriptor,
) -> dict[str, Any]:
    """Serialize only derived numeric/geometry features, never frame pixels."""

    return {
        "quality_score": descriptor.quality_score,
        "composition_score": descriptor.composition_score,
        "shot_scale": descriptor.shot_scale,
        "shot_type": descriptor.shot_type.value,
        "face_count": descriptor.face_count,
        "structure_embedding": list(descriptor.structure_embedding),
        "motion": list(descriptor.motion) if descriptor.motion is not None else None,
        "primary_face_bbox": (
            list(descriptor.primary_face_bbox)
            if descriptor.primary_face_bbox is not None
            else None
        ),
        "color_embedding": list(descriptor.color_embedding),
        "face_detector_backend": descriptor.face_detector_backend,
        "primary_face_landmarks_xy": (
            [list(point) for point in descriptor.primary_face_landmarks_xy]
            if descriptor.primary_face_landmarks_xy is not None
            else None
        ),
        "primary_face_head_roll_radians": (
            descriptor.primary_face_head_roll_radians
        ),
        "primary_face_eye_scale": descriptor.primary_face_eye_scale,
        "primary_face_geometry": (
            list(descriptor.primary_face_geometry)
            if descriptor.primary_face_geometry is not None
            else None
        ),
        "pose_embedding": list(descriptor.pose_embedding),
        "pose_confidence": descriptor.pose_confidence,
        "detected_person_count": descriptor.detected_person_count,
        "person_layout_embedding": list(descriptor.person_layout_embedding),
        "formation_people": [
            {
                "bbox_xywh": list(person.bbox_xywh),
                "centroid_xy": list(person.centroid_xy),
                "detection_confidence": person.detection_confidence,
                "upper_body_keypoints_xyc": [
                    list(point) for point in person.upper_body_keypoints_xyc
                ],
            }
            for person in descriptor.formation_people
        ],
        "pose_backend": descriptor.pose_backend,
        "silhouette_embedding": list(descriptor.silhouette_embedding),
        "silhouette_bbox_xywh": (
            list(descriptor.silhouette_bbox_xywh)
            if descriptor.silhouette_bbox_xywh is not None
            else None
        ),
        "silhouette_centroid_xy": (
            list(descriptor.silhouette_centroid_xy)
            if descriptor.silhouette_centroid_xy is not None
            else None
        ),
        "silhouette_area_ratio": descriptor.silhouette_area_ratio,
        "silhouette_confidence": descriptor.silhouette_confidence,
        "silhouette_backend": descriptor.silhouette_backend,
        "foreground_obstruction_score": descriptor.foreground_obstruction_score,
        "highlight_clipping_ratio": descriptor.highlight_clipping_ratio,
    }


def frame_visual_numeric_features(
    descriptor: FrameVisualDescriptor,
) -> dict[str, Any]:
    """Return the persisted, inference-safe numeric descriptor payload.

    StageEdit feature snapshots must consume the exact same derived fields as
    the production visual-feature cache.  Keeping this narrow public adapter
    here prevents a second serializer from drifting or accidentally retaining
    frames/identity embeddings.
    """

    if not isinstance(descriptor, FrameVisualDescriptor):
        raise TypeError("descriptor must use FrameVisualDescriptor")
    return _descriptor_cache_payload(descriptor)


def visual_analyzer_fingerprint_sha256() -> str:
    """Return the content-bound fingerprint of the active visual analyzer."""

    return _visual_analyzer_fingerprint().digest


_DESCRIPTOR_CACHE_PAYLOAD_KEYS = frozenset(
    _descriptor_cache_payload(
        FrameVisualDescriptor(
            quality_score=0.0,
            composition_score=0.0,
            shot_scale=0.0,
            shot_type=ShotType.GROUP_WIDE,
            face_count=0,
            structure_embedding=(),
        )
    )
)


def _cache_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"cached {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"cached {label} must be finite")
    return result


def _cache_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"cached {label} must be a non-negative integer")
    return value


def _cache_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f"cached {label} must be a non-empty bounded string")
    return value


def _cache_vector(
    value: Any,
    label: str,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"cached {label} must be a numeric list")
    result = tuple(_cache_float(item, f"{label}[]") for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"cached {label} must contain {length} values")
    return result


def _cache_optional_vector(
    value: Any,
    label: str,
    *,
    length: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _cache_vector(value, label, length=length)


def _descriptor_from_cache_payload(
    payload: Mapping[str, Any],
) -> FrameVisualDescriptor:
    """Reconstruct one exact descriptor or reject the cache record entirely."""

    if set(payload) != _DESCRIPTOR_CACHE_PAYLOAD_KEYS:
        raise ValueError("cached visual descriptor schema does not match")
    try:
        shot_type = ShotType(_cache_string(payload["shot_type"], "shot_type"))
    except ValueError as exc:
        raise ValueError("cached shot_type is invalid") from exc
    raw_landmarks = payload["primary_face_landmarks_xy"]
    landmarks = (
        None
        if raw_landmarks is None
        else tuple(
            _cache_vector(point, "primary_face_landmarks_xy[]", length=2)
            for point in raw_landmarks
        )
        if isinstance(raw_landmarks, list)
        else None
    )
    if raw_landmarks is not None and (
        landmarks is None or len(landmarks) != 5
    ):
        raise ValueError("cached face landmarks must contain five points")
    raw_people = payload["formation_people"]
    if not isinstance(raw_people, list):
        raise ValueError("cached formation_people must be a list")
    formation_people: list[FormationPersonDescriptor] = []
    for raw_person in raw_people:
        if not isinstance(raw_person, dict) or set(raw_person) != {
            "bbox_xywh",
            "centroid_xy",
            "detection_confidence",
            "upper_body_keypoints_xyc",
        }:
            raise ValueError("cached formation person schema does not match")
        raw_keypoints = raw_person["upper_body_keypoints_xyc"]
        if not isinstance(raw_keypoints, list):
            raise ValueError("cached formation keypoints must be a list")
        formation_people.append(
            FormationPersonDescriptor(
                bbox_xywh=_cache_vector(
                    raw_person["bbox_xywh"], "formation bbox", length=4
                ),
                centroid_xy=_cache_vector(
                    raw_person["centroid_xy"], "formation centroid", length=2
                ),
                detection_confidence=_cache_float(
                    raw_person["detection_confidence"],
                    "formation detection confidence",
                ),
                upper_body_keypoints_xyc=tuple(
                    _cache_vector(point, "formation keypoint", length=3)
                    for point in raw_keypoints
                ),
            )
        )
    roll = payload["primary_face_head_roll_radians"]
    eye_scale = payload["primary_face_eye_scale"]
    motion = _cache_optional_vector(payload["motion"], "motion", length=2)
    descriptor = FrameVisualDescriptor(
        quality_score=_cache_float(payload["quality_score"], "quality_score"),
        composition_score=_cache_float(
            payload["composition_score"], "composition_score"
        ),
        shot_scale=_cache_float(payload["shot_scale"], "shot_scale"),
        shot_type=shot_type,
        face_count=_cache_integer(payload["face_count"], "face_count"),
        structure_embedding=_cache_vector(
            payload["structure_embedding"], "structure_embedding"
        ),
        motion=(motion[0], motion[1]) if motion is not None else None,
        primary_face_bbox=_cache_optional_vector(
            payload["primary_face_bbox"], "primary_face_bbox", length=4
        ),
        color_embedding=_cache_vector(
            payload["color_embedding"], "color_embedding"
        ),
        face_detector_backend=_cache_string(
            payload["face_detector_backend"], "face_detector_backend"
        ),
        primary_face_landmarks_xy=landmarks,
        primary_face_head_roll_radians=(
            _cache_float(roll, "primary_face_head_roll_radians")
            if roll is not None
            else None
        ),
        primary_face_eye_scale=(
            _cache_float(eye_scale, "primary_face_eye_scale")
            if eye_scale is not None
            else None
        ),
        primary_face_geometry=_cache_optional_vector(
            payload["primary_face_geometry"],
            "primary_face_geometry",
            length=10,
        ),
        pose_embedding=_cache_vector(payload["pose_embedding"], "pose_embedding"),
        pose_confidence=_cache_float(
            payload["pose_confidence"], "pose_confidence"
        ),
        detected_person_count=_cache_integer(
            payload["detected_person_count"], "detected_person_count"
        ),
        person_layout_embedding=_cache_vector(
            payload["person_layout_embedding"], "person_layout_embedding"
        ),
        formation_people=tuple(formation_people),
        pose_backend=_cache_string(payload["pose_backend"], "pose_backend"),
        silhouette_embedding=_cache_vector(
            payload["silhouette_embedding"], "silhouette_embedding"
        ),
        silhouette_bbox_xywh=_cache_optional_vector(
            payload["silhouette_bbox_xywh"], "silhouette_bbox_xywh", length=4
        ),
        silhouette_centroid_xy=_cache_optional_vector(
            payload["silhouette_centroid_xy"],
            "silhouette_centroid_xy",
            length=2,
        ),
        silhouette_area_ratio=_cache_float(
            payload["silhouette_area_ratio"], "silhouette_area_ratio"
        ),
        silhouette_confidence=_cache_float(
            payload["silhouette_confidence"], "silhouette_confidence"
        ),
        silhouette_backend=_cache_string(
            payload["silhouette_backend"], "silhouette_backend"
        ),
        foreground_obstruction_score=_cache_float(
            payload["foreground_obstruction_score"],
            "foreground_obstruction_score",
        ),
        highlight_clipping_ratio=_cache_float(
            payload["highlight_clipping_ratio"], "highlight_clipping_ratio"
        ),
    )
    if len(descriptor.structure_embedding) != 96:
        raise ValueError("cached structure embedding must contain 96 values")
    if len(descriptor.color_embedding) != 8:
        raise ValueError("cached color embedding must contain 8 values")
    if len(descriptor.pose_embedding) not in {60, 96}:
        raise ValueError("cached pose embedding must contain 60 or 96 values")
    if len(descriptor.person_layout_embedding) not in {0, 134}:
        raise ValueError("cached person layout must be empty or contain 134 values")
    if len(descriptor.silhouette_embedding) not in {0, 121}:
        raise ValueError("cached silhouette must be empty or contain 121 values")
    if len(descriptor.formation_people) > 6:
        raise ValueError("cached formation cannot contain more than six people")
    if (
        descriptor.pose_backend == "structure_hog_fallback"
        and len(descriptor.pose_embedding) != 96
    ):
        raise ValueError("cached HOG fallback must carry the structure embedding")
    if (
        descriptor.pose_backend != "structure_hog_fallback"
        and len(descriptor.pose_embedding) != 60
    ):
        raise ValueError("cached model pose must contain 60 values")
    if descriptor.silhouette_embedding and descriptor.silhouette_backend == "none":
        raise ValueError("cached silhouette vector has no model backend")
    normalized_boxes = (
        descriptor.primary_face_bbox,
        descriptor.silhouette_bbox_xywh,
        *(person.bbox_xywh for person in descriptor.formation_people),
    )
    for box in normalized_boxes:
        if box is None:
            continue
        left, top, width, height = box
        if (
            left < 0.0
            or top < 0.0
            or width <= 0.0
            or height <= 0.0
            or left + width > 1.000001
            or top + height > 1.000001
        ):
            raise ValueError("cached normalized bbox is outside the frame")
    if descriptor.primary_face_landmarks_xy is not None and any(
        not 0.0 <= coordinate <= 1.0
        for point in descriptor.primary_face_landmarks_xy
        for coordinate in point
    ):
        raise ValueError("cached face landmarks are outside the frame")
    unit_vectors = (
        descriptor.color_embedding,
        descriptor.person_layout_embedding,
        descriptor.silhouette_embedding,
    )
    if any(
        not 0.0 <= value <= 1.0
        for vector in unit_vectors
        for value in vector
    ):
        raise ValueError("cached unit feature vector is outside [0, 1]")
    if descriptor.motion is not None and any(
        not -1.0 <= value <= 1.0 for value in descriptor.motion
    ):
        raise ValueError("cached motion vector is outside [-1, 1]")
    bounded_units = (
        descriptor.quality_score,
        descriptor.composition_score,
        descriptor.shot_scale,
        descriptor.pose_confidence,
        descriptor.silhouette_area_ratio,
        descriptor.silhouette_confidence,
        descriptor.foreground_obstruction_score,
        descriptor.highlight_clipping_ratio,
    )
    if any(not 0.0 <= value <= 1.0 for value in bounded_units):
        raise ValueError("cached normalized descriptor value is outside [0, 1]")
    return descriptor


def _working_frame_sha256(frame: np.ndarray) -> str:
    """Identify decoded working-frame content without persisting any pixels."""

    contiguous = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _descriptor_matches_analyzer_runtime(
    descriptor: FrameVisualDescriptor,
    *,
    analyzer: AnalyzerFingerprint,
    include_subject_geometry: bool,
) -> bool:
    """Do not persist a transient model failure under a healthy fingerprint."""

    raw_models = analyzer.components.get("model_asset_sha256_or_state")
    if not isinstance(raw_models, Mapping):
        return True
    models = {str(key): str(value) for key, value in raw_models.items()}
    yunet_healthy = any(
        key == f"face:{YUNET_MODEL_NAME}" and "loadable=True" in value
        for key, value in models.items()
    )
    if yunet_healthy and descriptor.face_detector_backend != "yunet":
        return False
    segmenter_healthy = any(
        "human_segmentation_pphumanseg" in key and "loadable=True" in value
        for key, value in models.items()
    )
    if (
        include_subject_geometry
        and segmenter_healthy
        and descriptor.silhouette_backend == "none"
    ):
        return False
    return True


class PersistentVisualFeatureCacheInitializationError(RuntimeError):
    """A required descriptor cache could not be initialized safely."""


class PersistentVisualFeatureCacheRuntimeError(RuntimeError):
    """A required descriptor cache failed after successful initialization."""


class _VideoSampler:
    def __init__(
        self,
        path: Path,
        *,
        feature_cache_directory: Path | None = None,
        feature_derived_cache_directory: Path | None = None,
        require_persistent_feature_cache: bool = False,
    ) -> None:
        if not isinstance(require_persistent_feature_cache, bool):
            raise TypeError("require_persistent_feature_cache must be boolean")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"cannot open video for visual scoring: {path}")
        reported_frame_rate = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_rate = (
            reported_frame_rate
            if math.isfinite(reported_frame_rate) and reported_frame_rate > 0.0
            else 29.97
        )
        # A full song with nine sources can touch thousands of frames. Keep a
        # small per-source LRU: adjacent segments reuse boundary frames without
        # retaining the complete decoded performance in RAM.
        self.cache: OrderedDict[
            CanonicalFrameIdentity, np.ndarray | None
        ] = OrderedDict()
        self.frame_digest_cache: OrderedDict[
            CanonicalFrameIdentity, str
        ] = OrderedDict()
        self.descriptor_cache: OrderedDict[
            DescriptorRequestIdentity, FrameVisualDescriptor | None
        ] = OrderedDict()
        self.descriptor_locator_cache: OrderedDict[
            DescriptorRequestLocator, FrameVisualDescriptor | None
        ] = OrderedDict()
        self.sface_cache: OrderedDict[
            CanonicalFrameIdentity, tuple[float, ...]
        ] = OrderedDict()
        self.frame_decode_count = 0
        self.persistent_feature_cache: PersistentVisualFeatureCache | None = None
        self._persistent_feature_cache_configured = bool(
            feature_cache_directory is not None
            or feature_derived_cache_directory is not None
        )
        self._persistent_feature_cache_required = (
            require_persistent_feature_cache
        )
        self._persistent_feature_cache_direct_root_configured = (
            feature_derived_cache_directory is not None
        )
        self._persistent_feature_cache_status = "not_configured"
        self._persistent_feature_cache_initialization_error_type: str | None = None
        self._persistent_feature_cache_runtime_error_type: str | None = None
        self._persistent_feature_cache_runtime_failure_count = 0
        if (
            require_persistent_feature_cache
            and not self._persistent_feature_cache_configured
        ):
            self.capture.release()
            raise PersistentVisualFeatureCacheInitializationError(
                "persistent visual feature cache is required but was not configured"
            )
        if self._persistent_feature_cache_configured:
            source_cache_directory = (
                feature_cache_directory
                if feature_cache_directory is not None
                else feature_derived_cache_directory
            )
            assert source_cache_directory is not None
            try:
                self.persistent_feature_cache = PersistentVisualFeatureCache(
                    source_cache_directory=source_cache_directory,
                    source_path=path,
                    analyzer=_visual_analyzer_fingerprint(),
                    derived_cache_directory=feature_derived_cache_directory,
                )
                self._persistent_feature_cache_status = "enabled"
            except (OSError, TypeError, ValueError) as exc:
                self._persistent_feature_cache_initialization_error_type = (
                    type(exc).__name__
                )
                self._persistent_feature_cache_status = (
                    "initialization_failed_optional"
                )
                if require_persistent_feature_cache:
                    self.capture.release()
                    raise PersistentVisualFeatureCacheInitializationError(
                        "required persistent visual feature cache initialization "
                        "failed"
                    ) from None
                # Cache setup is strictly optional.  Any unsafe path, changing
                # source, or unreadable fingerprint falls back to fresh
                # in-memory analysis rather than weakening visual gates.
                self.persistent_feature_cache = None

    def _handle_persistent_feature_cache_io_error(
        self,
        exc: PersistentVisualFeatureCacheIOError,
    ) -> None:
        """Fail required runs, or disable an optional cache without path leaks."""

        self._persistent_feature_cache_runtime_error_type = type(exc).__name__
        self._persistent_feature_cache_runtime_failure_count += 1
        if self._persistent_feature_cache_required:
            self._persistent_feature_cache_status = "runtime_io_failed_required"
            raise PersistentVisualFeatureCacheRuntimeError(
                "required persistent visual feature cache failed during analysis"
            ) from None
        self._persistent_feature_cache_status = "runtime_io_failed_optional"
        self.persistent_feature_cache = None

    def assert_required_persistent_cache_available(self) -> None:
        """Attest strict cache storage before any process-local fast return."""

        if not self._persistent_feature_cache_required:
            return
        persistent = self.persistent_feature_cache
        if persistent is None:  # pragma: no cover - constructor invariant
            raise PersistentVisualFeatureCacheRuntimeError(
                "required persistent visual feature cache failed during analysis"
            )
        try:
            persistent.assert_storage_root_available()
        except PersistentVisualFeatureCacheIOError as exc:
            self._handle_persistent_feature_cache_io_error(exc)

    def frame_identity(self, seconds: float) -> CanonicalFrameIdentity:
        return CanonicalFrameIdentity.from_seconds(
            seconds,
            frame_rate=self.frame_rate,
        )

    def frame_at(self, seconds: float) -> np.ndarray | None:
        key = self.frame_identity(seconds)
        if key not in self.cache:
            # Preserve source-time seeking (important for VFR media) while the
            # LRU key retains the exact rational request rather than a rounded
            # millisecond.  The persistent key additionally verifies decoded
            # working-frame SHA-256 before reusing numeric features.
            self.capture.set(
                cv2.CAP_PROP_POS_MSEC,
                key.canonical_seconds * 1000.0,
            )
            success, frame = self.capture.read()
            self.frame_decode_count += 1
            self.cache[key] = _working_frame(frame) if success else None
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def frame_digest_at(self, seconds: float, frame: np.ndarray) -> str:
        key = self.frame_identity(seconds)
        if key not in self.frame_digest_cache:
            self.frame_digest_cache[key] = _working_frame_sha256(frame)
            while len(self.frame_digest_cache) > 96:
                self.frame_digest_cache.popitem(last=False)
        else:
            self.frame_digest_cache.move_to_end(key)
        return self.frame_digest_cache[key]

    def feature_descriptor_at(
        self,
        seconds: float,
        *,
        following_seconds: float | None = None,
        include_subject_geometry: bool = True,
        include_formation_pose: bool = False,
    ) -> FrameVisualDescriptor | None:
        """Return one exact descriptor with optional persistent reuse."""

        self.assert_required_persistent_cache_available()
        persistent = self.persistent_feature_cache

        primary = self.frame_identity(seconds)
        following = (
            self.frame_identity(following_seconds)
            if following_seconds is not None
            else None
        )
        locator = DescriptorRequestLocator(
            primary=primary,
            following=following,
            include_subject_geometry=include_subject_geometry,
            include_formation_pose=include_formation_pose,
        )
        if locator in self.descriptor_locator_cache:
            self.descriptor_locator_cache.move_to_end(locator)
            return self.descriptor_locator_cache[locator]
        if persistent is not None:
            try:
                cached = persistent.get_by_locator(locator)
            except PersistentVisualFeatureCacheIOError as exc:
                self._handle_persistent_feature_cache_io_error(exc)
                persistent = self.persistent_feature_cache
                cached = None
            if cached is not None:
                try:
                    descriptor = _descriptor_from_cache_payload(cached)
                except (TypeError, ValueError):
                    descriptor = None
                if descriptor is not None and _descriptor_matches_analyzer_runtime(
                    descriptor,
                    analyzer=persistent.analyzer,
                    include_subject_geometry=include_subject_geometry,
                ):
                    self.descriptor_locator_cache[locator] = descriptor
                    while len(self.descriptor_locator_cache) > 64:
                        self.descriptor_locator_cache.popitem(last=False)
                    return descriptor

        frame = self.frame_at(seconds)
        if frame is None:
            self.descriptor_locator_cache[locator] = None
            return None
        following_frame = (
            self.frame_at(following_seconds)
            if following_seconds is not None
            else None
        )
        if following_seconds is not None and following_frame is None:
            self.descriptor_locator_cache[locator] = None
            return None
        request = DescriptorRequestIdentity(
            primary=primary,
            following=following,
            primary_working_frame_sha256=self.frame_digest_at(seconds, frame),
            following_working_frame_sha256=(
                self.frame_digest_at(following_seconds, following_frame)
                if following_frame is not None
                and following_seconds is not None
                else None
            ),
            include_subject_geometry=include_subject_geometry,
            include_formation_pose=include_formation_pose,
        )
        if request in self.descriptor_cache:
            self.descriptor_cache.move_to_end(request)
            return self.descriptor_cache[request]
        descriptor: FrameVisualDescriptor | None = None
        if persistent is not None:
            try:
                cached = persistent.get(request)
            except PersistentVisualFeatureCacheIOError as exc:
                self._handle_persistent_feature_cache_io_error(exc)
                persistent = self.persistent_feature_cache
                cached = None
            if cached is not None:
                try:
                    descriptor = _descriptor_from_cache_payload(cached)
                except (TypeError, ValueError):
                    descriptor = None
                if descriptor is not None and not _descriptor_matches_analyzer_runtime(
                    descriptor,
                    analyzer=persistent.analyzer,
                    include_subject_geometry=include_subject_geometry,
                ):
                    descriptor = None
        if descriptor is None:
            describe_arguments: dict[str, Any] = {
                "following_frame": following_frame,
            }
            if not include_subject_geometry:
                describe_arguments["include_subject_geometry"] = False
            if include_formation_pose:
                describe_arguments["include_formation_pose"] = True
            descriptor = describe_frame(frame, **describe_arguments)
            if persistent is not None and _descriptor_matches_analyzer_runtime(
                descriptor,
                analyzer=persistent.analyzer,
                include_subject_geometry=include_subject_geometry,
            ):
                try:
                    persistent.put(
                        request,
                        _descriptor_cache_payload(descriptor),
                    )
                except PersistentVisualFeatureCacheIOError as exc:
                    self._handle_persistent_feature_cache_io_error(exc)
                except (OSError, TypeError, ValueError):
                    # A cache write can never make a freshly computed
                    # descriptor unusable for the current analysis.
                    pass
        self.descriptor_cache[request] = descriptor
        while len(self.descriptor_cache) > 64:
            self.descriptor_cache.popitem(last=False)
        self.descriptor_locator_cache[locator] = descriptor
        while len(self.descriptor_locator_cache) > 64:
            self.descriptor_locator_cache.popitem(last=False)
        return descriptor

    def descriptor_at(self, seconds: float) -> FrameVisualDescriptor | None:
        return self.feature_descriptor_at(
            seconds,
            include_subject_geometry=False,
        )

    def transition_descriptor_at(
        self, seconds: float
    ) -> FrameVisualDescriptor | None:
        """Describe one selected handle frame with full body evidence."""

        return self.feature_descriptor_at(
            seconds,
            include_subject_geometry=True,
        )

    def sface_embedding_at(
        self,
        seconds: float,
        descriptor: FrameVisualDescriptor,
    ) -> tuple[float, ...]:
        """Embed an already-unambiguous YuNet face in one selected frame."""

        key = self.frame_identity(seconds)
        if key not in self.sface_cache:
            bbox = descriptor.primary_face_bbox
            landmarks = descriptor.primary_face_landmarks_xy
            embedding: tuple[float, ...] = ()
            if (
                descriptor.face_detector_backend == "yunet"
                and bbox is not None
                and landmarks is not None
                and len(landmarks) == 5
            ):
                # Descriptor-only eligibility is authoritative and avoids a
                # random media seek when no SFace measurement can possibly be
                # produced.  Eligible frames retain the exact previous model
                # path and process-local privacy boundary.
                frame = self.frame_at(seconds)
                if frame is not None:
                    embedding = extract_sface_embedding(
                        frame,
                        bbox_xywh=bbox,
                        landmarks_xy=landmarks,
                    )
            self.sface_cache[key] = embedding
            while len(self.sface_cache) > 16:
                self.sface_cache.popitem(last=False)
        else:
            self.sface_cache.move_to_end(key)
        return self.sface_cache[key]

    def close(self) -> None:
        self.capture.release()
        self.cache.clear()
        self.frame_digest_cache.clear()
        self.descriptor_cache.clear()
        self.descriptor_locator_cache.clear()
        # SFace appearance vectors are RAM-only and explicitly dropped as soon
        # as the sampler closes; they never enter PersistentVisualFeatureCache.
        self.sface_cache.clear()

    def persistent_feature_cache_diagnostics(self) -> dict[str, Any]:
        """Report cache state and counters without exposing filesystem paths."""

        persistent = self.persistent_feature_cache
        diagnostics: dict[str, Any] = {
            "enabled": persistent is not None,
            "status": self._persistent_feature_cache_status,
            "configured": self._persistent_feature_cache_configured,
            "required": self._persistent_feature_cache_required,
            "direct_derived_cache_configured": (
                self._persistent_feature_cache_direct_root_configured
            ),
            "initialization_error_type": (
                self._persistent_feature_cache_initialization_error_type
            ),
            "runtime_error_type": self._persistent_feature_cache_runtime_error_type,
            "runtime_failure_count": (
                self._persistent_feature_cache_runtime_failure_count
            ),
            "source_frame_decode_count": self.frame_decode_count,
            "decode_free_hit_count": 0,
            "decode_free_legacy_hit_count": 0,
            "local_paths_exposed": False,
        }
        if persistent is None:
            return diagnostics
        raw = persistent.diagnostics()
        safe_keys = (
            "cache_contract",
            "source_sha256",
            "analyzer_fingerprint_sha256",
            "hit_count",
            "miss_count",
            "write_count",
            "invalid_record_count",
            "decode_free_hit_count",
            "decode_free_legacy_hit_count",
            "locator_index_build_count",
            "locator_index_status",
            "locator_index_record_budget",
            "locator_index_preflight_record_count",
            "locator_index_budget_fallback_count",
            "locator_index_process_memo_hit_count",
            "ambiguous_locator_count",
            "disabled_by_source_change",
            "storage_io_failure_count",
            "atomic_write_failure_count",
            "raw_frames_persisted",
            "face_identity_embeddings_persisted",
        )
        diagnostics.update(
            {key: raw[key] for key in safe_keys if key in raw}
        )
        diagnostics["locator_index_over_budget"] = bool(
            raw.get("locator_index_over_budget_root")
        )
        return diagnostics

    def feature_cache_diagnostics(self) -> dict[str, Any] | None:
        """Retain the legacy Optional contract for existing aggregators."""

        if self.persistent_feature_cache is None:
            return None
        return self.persistent_feature_cache_diagnostics()


def _create_video_sampler(
    path: Path,
    *,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> _VideoSampler:
    """Create a sampler while keeping default test/provider call shapes stable."""

    arguments: dict[str, Any] = {}
    if feature_cache_directory is not None:
        arguments["feature_cache_directory"] = feature_cache_directory
    if feature_derived_cache_directory is not None:
        arguments["feature_derived_cache_directory"] = (
            feature_derived_cache_directory
        )
    if require_persistent_feature_cache:
        arguments["require_persistent_feature_cache"] = True
    return _VideoSampler(Path(path), **arguments)


def _create_video_samplers(
    sources: Sequence[Path],
    *,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> list[_VideoSampler]:
    """Open every source before analysis and release partial construction."""

    samplers: list[_VideoSampler] = []
    try:
        for source in sources:
            samplers.append(
                _create_video_sampler(
                    Path(source),
                    feature_cache_directory=feature_cache_directory,
                    feature_derived_cache_directory=(
                        feature_derived_cache_directory
                    ),
                    require_persistent_feature_cache=(
                        require_persistent_feature_cache
                    ),
                )
            )
    except BaseException:
        for sampler in samplers:
            sampler.close()
        raise
    return samplers


def _face_geometry_boundary_window(
    sampler: _VideoSampler,
    *,
    boundary_seconds: float,
    incoming: bool,
    scene_cut_times: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Sample three synchronized frames around one possible hard cut.

    Outgoing samples end one frame before the cut; incoming samples end at the
    cut. Paired entries therefore follow the original consecutive-frame eye
    comparison idea (outgoing ``i`` versus incoming ``i+1``).
    """

    step = _BOUNDARY_FRAME_STEP_SECONDS
    last_offset = 0 if incoming else -1
    offsets = tuple(
        last_offset - (_BOUNDARY_WINDOW_FRAMES - 1) + index
        for index in range(_BOUNDARY_WINDOW_FRAMES)
    )
    times = tuple(boundary_seconds + offset * step for offset in offsets)
    if times[0] < 0.0:
        return ()
    if any(times[0] - 1e-6 <= float(cut) <= times[-1] + 1e-6 for cut in scene_cut_times):
        return ()
    descriptors = tuple(sampler.descriptor_at(time_value) for time_value in times)
    if any(
        descriptor is None or descriptor.primary_face_geometry is None
        for descriptor in descriptors
    ):
        return ()
    geometries = tuple(
        descriptor.primary_face_geometry
        for descriptor in descriptors
        if descriptor is not None and descriptor.primary_face_geometry is not None
    )
    # Guard against a detector switching between people inside the tiny window.
    eye_centers = tuple(
        (
            (geometry[0] + geometry[2]) / 2.0,
            (geometry[1] + geometry[3]) / 2.0,
        )
        for geometry in geometries
    )
    if max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in eye_centers
        for right in eye_centers
    ) > 0.16:
        return ()
    scales = tuple(geometry[4] for geometry in geometries)
    if max(scales) / min(scales) > 1.5:
        return ()
    rolls = tuple(geometry[5] for geometry in geometries)
    if max(rolls) - min(rolls) > math.radians(28.0):
        return ()
    return geometries


def _dissolve_evidence_window(
    sampler: _VideoSampler,
    *,
    scene_cut_times: Sequence[float],
    boundary_seconds: float | None = None,
    sample_times_seconds: Sequence[float] | None = None,
) -> _DissolveEvidence:
    """Sample actual cut/mid/end frames from one selected dissolve handle."""

    if (boundary_seconds is None) == (sample_times_seconds is None):
        raise ValueError("provide exactly one dissolve boundary or sample-time set")
    if sample_times_seconds is None:
        assert boundary_seconds is not None
        times = (
            boundary_seconds,
            boundary_seconds + _DISSOLVE_HANDLE_SECONDS / 2.0,
            boundary_seconds + _DISSOLVE_HANDLE_SECONDS,
        )
    else:
        times = tuple(float(value) for value in sample_times_seconds)
        if len(times) != 3 or any(not math.isfinite(value) for value in times):
            raise ValueError("dissolve sample times must contain three finite values")
        if not times[0] < times[1] < times[2]:
            raise ValueError("dissolve sample times must be strictly increasing")
    half_frame = _BOUNDARY_FRAME_STEP_SECONDS / 2.0
    if any(
        times[0] - half_frame <= float(cut) <= times[-1] + half_frame
        for cut in scene_cut_times
    ):
        return _DissolveEvidence()
    descriptors = tuple(
        sampler.transition_descriptor_at(time_value) for time_value in times
    )
    if any(
        descriptor is None
        or descriptor.face_detector_backend != "yunet"
        or descriptor.primary_face_geometry is None
        or descriptor.primary_face_bbox is None
        or descriptor.primary_face_landmarks_xy is None
        or len(descriptor.primary_face_landmarks_xy) != 5
        for descriptor in descriptors
    ):
        return _DissolveEvidence()
    safe_descriptors = tuple(
        descriptor for descriptor in descriptors if descriptor is not None
    )
    geometries = tuple(
        descriptor.primary_face_geometry
        for descriptor in safe_descriptors
        if descriptor.primary_face_geometry is not None
    )
    boxes = tuple(
        descriptor.primary_face_bbox
        for descriptor in safe_descriptors
        if descriptor.primary_face_bbox is not None
    )
    centers = tuple(
        (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0) for box in boxes
    )
    if max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    ) > 0.16:
        return _DissolveEvidence()
    scales = tuple(geometry[4] for geometry in geometries)
    if min(scales) <= 0.0 or max(scales) / min(scales) > 1.5:
        return _DissolveEvidence()
    rolls = tuple(geometry[5] for geometry in geometries)
    if max(
        abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)
        for left in rolls
        for right in rolls
    ) > math.radians(28.0):
        return _DissolveEvidence()

    identity_embeddings = tuple(
        sampler.sface_embedding_at(time_value, descriptor)
        for time_value, descriptor in zip(times, safe_descriptors, strict=True)
    )
    return _DissolveEvidence(
        face_geometry=geometries,
        face_boxes=boxes,
        face_identity_embeddings=identity_embeddings,
        pose_embeddings=tuple(
            tuple(descriptor.pose_embedding) for descriptor in safe_descriptors
        ),
        pose_backends=tuple(
            descriptor.pose_backend for descriptor in safe_descriptors
        ),
        pose_confidences=tuple(
            float(descriptor.pose_confidence) for descriptor in safe_descriptors
        ),
        silhouette_embeddings=tuple(
            tuple(descriptor.silhouette_embedding)
            for descriptor in safe_descriptors
        ),
        silhouette_backends=tuple(
            descriptor.silhouette_backend for descriptor in safe_descriptors
        ),
        silhouette_confidences=tuple(
            float(descriptor.silhouette_confidence)
            for descriptor in safe_descriptors
        ),
    )


def _embedding_rms_distance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    squared = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    return min(1.0, math.sqrt(squared / len(left)))


def describe_boundary_side_stability(
    earlier: FrameVisualDescriptor,
    later: FrameVisualDescriptor,
    *,
    sample_span_seconds: float,
) -> BoundarySideStability:
    """Summarize a same-source three-frame neighborhood without raw vectors."""

    shot_order = {
        ShotType.GROUP_WIDE: 0,
        ShotType.WIDE: 1,
        ShotType.MEDIUM: 2,
        ShotType.CLOSE_UP: 3,
    }
    actual_pose_pair = (
        earlier.pose_backend != "structure_hog_fallback"
        and later.pose_backend != "structure_hog_fallback"
        and bool(earlier.pose_embedding)
        and len(earlier.pose_embedding) == len(later.pose_embedding)
    )
    return BoundarySideStability(
        sample_span_seconds=sample_span_seconds,
        scale_delta=abs(earlier.shot_scale - later.shot_scale),
        structure_mismatch=_embedding_rms_distance(
            earlier.structure_embedding,
            later.structure_embedding,
        ),
        pose_mismatch=(
            _embedding_rms_distance(
                earlier.pose_embedding,
                later.pose_embedding,
            )
            if actual_pose_pair
            else None
        ),
        shot_type_mismatch=(
            abs(shot_order[earlier.shot_type] - shot_order[later.shot_type]) / 3.0
        ),
    )


def _candidate_can_enter_dissolve_postpass(
    metadata: _CandidateFaceMetadata | None,
) -> bool:
    """Cheap prefilter before opening sources for selected-boundary evidence."""

    return bool(
        metadata is not None
        and metadata.failure is None
        and all(backend == "yunet" for backend in metadata.backends)
        and all(
            landmarks is not None and len(landmarks) == 5
            for landmarks in metadata.landmarks
        )
    )


def _selected_dissolve_evidence(
    *,
    sources: Sequence[Path],
    optimized: TimelineResult,
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
    scene_cut_times: Sequence[Sequence[float]],
    face_metadata_by_candidate: dict[str, _CandidateFaceMetadata],
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> tuple[dict[str, _DissolveEvidence], dict[str, _DissolveEvidence]]:
    """Run future-overlap safety models only on DP-selected switch boundaries.

    The first result is entry evidence keyed by the incoming candidate id; the
    second is exit evidence keyed by the outgoing candidate id.  Candidate
    Candidate scoring already has one-frame SFace evidence for close-up source
    switches. This post-pass alone stays independent of *future-handle* SFace
    and body inference so rejected candidates do not pay that extra cost.
    """

    entry: dict[str, _DissolveEvidence] = {}
    exit_: dict[str, _DissolveEvidence] = {}
    boundaries = tuple(zip(optimized.candidates, optimized.candidates[1:]))
    eligible_boundaries = tuple(
        (outgoing, incoming)
        for outgoing, incoming in boundaries
        if outgoing.source_id != incoming.source_id
        and incoming.duration_seconds + 1e-9 >= _DISSOLVE_HANDLE_SECONDS
        and _candidate_can_enter_dissolve_postpass(
            face_metadata_by_candidate.get(outgoing.candidate_id)
        )
        and _candidate_can_enter_dissolve_postpass(
            face_metadata_by_candidate.get(incoming.candidate_id)
        )
    )
    if not eligible_boundaries:
        return (entry, exit_)

    samplers = _create_video_samplers(
        sources,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
    )
    try:
        for outgoing, incoming in eligible_boundaries:
            outgoing_source = int(outgoing.source_id)
            incoming_source = int(incoming.source_id)
            handle_master_times = (
                outgoing.end_seconds,
                outgoing.end_seconds + _DISSOLVE_HANDLE_SECONDS / 2.0,
                outgoing.end_seconds + _DISSOLVE_HANDLE_SECONDS,
            )
            exit_[outgoing.candidate_id] = _dissolve_evidence_window(
                samplers[outgoing_source],
                sample_times_seconds=tuple(
                    _source_clock_time(
                        master_time,
                        source_index=outgoing_source,
                        audio_offsets=audio_offsets,
                        source_timewarps=source_timewarps,
                    )
                    for master_time in handle_master_times
                ),
                scene_cut_times=scene_cut_times[outgoing_source],
            )
            handle_master_times = (
                incoming.start_seconds,
                incoming.start_seconds + _DISSOLVE_HANDLE_SECONDS / 2.0,
                incoming.start_seconds + _DISSOLVE_HANDLE_SECONDS,
            )
            entry[incoming.candidate_id] = _dissolve_evidence_window(
                samplers[incoming_source],
                sample_times_seconds=tuple(
                    _source_clock_time(
                        master_time,
                        source_index=incoming_source,
                        audio_offsets=audio_offsets,
                        source_timewarps=source_timewarps,
                    )
                    for master_time in handle_master_times
                ),
                scene_cut_times=scene_cut_times[incoming_source],
            )
    finally:
        for sampler in samplers:
            sampler.close()
    return (entry, exit_)


_DISSOLVE_EVIDENCE_FIELD_NAMES = (
    "dissolve_entry_face_geometry_window",
    "dissolve_exit_face_geometry_window",
    "dissolve_entry_face_box_window",
    "dissolve_exit_face_box_window",
    "dissolve_entry_face_identity_window",
    "dissolve_exit_face_identity_window",
    "dissolve_entry_pose_embedding_window",
    "dissolve_exit_pose_embedding_window",
    "dissolve_entry_pose_backend_window",
    "dissolve_exit_pose_backend_window",
    "dissolve_entry_pose_confidence_window",
    "dissolve_exit_pose_confidence_window",
    "dissolve_entry_silhouette_embedding_window",
    "dissolve_exit_silhouette_embedding_window",
    "dissolve_entry_silhouette_backend_window",
    "dissolve_exit_silhouette_backend_window",
    "dissolve_entry_silhouette_confidence_window",
    "dissolve_exit_silhouette_confidence_window",
)
_DISSOLVE_REHYDRATION_ANALYSIS_BASIS = (
    "post_repair_ram_dissolve_rehydration_v1"
)


def _choice_can_enter_dissolve_postpass(
    choice: VisualTimelineChoice,
) -> bool:
    """Apply the retained face-metadata prefilter to a materialized choice."""

    return bool(
        choice.face_match_candidate_failure is None
        and all(backend == "yunet" for backend in choice.face_detector_backends)
        and all(
            landmarks is not None and len(landmarks) == 5
            for landmarks in choice.face_landmark_samples_xy
        )
    )


def rehydrate_visual_timeline_dissolve_evidence(
    visual_timeline: VisualTimelinePlan,
    *,
    sources: Sequence[Path],
    audio_offsets: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> VisualTimelinePlan:
    """Re-measure transient dissolve evidence on the final repaired clock.

    Checkpoints deliberately omit raw SFace, pose, and silhouette vectors, and
    local boundary repair invalidates evidence sampled at the former cut time.
    This authoritative post-repair pass therefore clears *all* retained
    dissolve fields before sampling only eligible selected source switches.
    SFace vectors remain process-local inside the returned immutable plan;
    :meth:`VisualTimelineChoice.to_dict` continues to serialize counts and
    backend/confidence metadata only.
    """

    if not isinstance(visual_timeline, VisualTimelinePlan):
        raise TypeError("visual_timeline must be a VisualTimelinePlan")
    if not visual_timeline.choices:
        raise ValueError("visual timeline cannot be empty")
    source_paths = tuple(Path(path) for path in sources)
    if not source_paths:
        raise ValueError("dissolve evidence rehydration needs sources")
    if not (
        len(source_paths) == len(audio_offsets) == len(scene_cut_times)
    ):
        raise ValueError(
            "dissolve evidence source arrays must have matching lengths"
        )
    clock_maps = _source_clock_maps(
        source_timewarps,
        audio_offsets=audio_offsets,
        source_count=len(source_paths),
    )
    if any(
        choice.source_index < 0 or choice.source_index >= len(source_paths)
        for choice in visual_timeline.choices
    ):
        raise ValueError("visual timeline references an unknown source")
    for outgoing, incoming in zip(
        visual_timeline.choices,
        visual_timeline.choices[1:],
    ):
        if not math.isclose(
            outgoing.canonical_master_end,
            incoming.canonical_master_start,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "visual timeline contains a non-contiguous dissolve boundary"
            )

    empty_fields = {
        name: () for name in _DISSOLVE_EVIDENCE_FIELD_NAMES
    }
    cleared_choices = tuple(
        replace(choice, **empty_fields)
        for choice in visual_timeline.choices
    )
    eligible_boundaries = tuple(
        (boundary_index, outgoing, incoming)
        for boundary_index, (outgoing, incoming) in enumerate(
            zip(cleared_choices, cleared_choices[1:])
        )
        if outgoing.source_index != incoming.source_index
        and incoming.duration_seconds + 1e-9 >= _DISSOLVE_HANDLE_SECONDS
        and _choice_can_enter_dissolve_postpass(outgoing)
        and _choice_can_enter_dissolve_postpass(incoming)
    )

    entry_by_choice_index: dict[int, _DissolveEvidence] = {}
    exit_by_choice_index: dict[int, _DissolveEvidence] = {}
    samplers: dict[int, _VideoSampler] = {}

    def sampler_for(source_index: int) -> _VideoSampler:
        sampler = samplers.get(source_index)
        if sampler is None:
            sampler = _create_video_sampler(
                source_paths[source_index],
                feature_cache_directory=feature_cache_directory,
                feature_derived_cache_directory=(
                    feature_derived_cache_directory
                ),
                require_persistent_feature_cache=(
                    require_persistent_feature_cache
                ),
            )
            samplers[source_index] = sampler
        return sampler

    try:
        for boundary_index, outgoing, incoming in eligible_boundaries:
            boundary_master_seconds = outgoing.canonical_master_end
            handle_master_times = (
                boundary_master_seconds,
                boundary_master_seconds + _DISSOLVE_HANDLE_SECONDS / 2.0,
                boundary_master_seconds + _DISSOLVE_HANDLE_SECONDS,
            )
            outgoing_source = outgoing.source_index
            incoming_source = incoming.source_index
            exit_by_choice_index[boundary_index] = _dissolve_evidence_window(
                sampler_for(outgoing_source),
                sample_times_seconds=tuple(
                    _source_clock_time(
                        master_time,
                        source_index=outgoing_source,
                        audio_offsets=audio_offsets,
                        source_timewarps=clock_maps,
                    )
                    for master_time in handle_master_times
                ),
                scene_cut_times=scene_cut_times[outgoing_source],
            )
            entry_by_choice_index[boundary_index + 1] = (
                _dissolve_evidence_window(
                    sampler_for(incoming_source),
                    sample_times_seconds=tuple(
                        _source_clock_time(
                            master_time,
                            source_index=incoming_source,
                            audio_offsets=audio_offsets,
                            source_timewarps=clock_maps,
                        )
                        for master_time in handle_master_times
                    ),
                    scene_cut_times=scene_cut_times[incoming_source],
                )
            )
    finally:
        for sampler in samplers.values():
            sampler.close()

    rehydrated_choices: list[VisualTimelineChoice] = []
    for choice_index, choice in enumerate(cleared_choices):
        entry_evidence = entry_by_choice_index.get(
            choice_index,
            _DissolveEvidence(),
        )
        exit_evidence = exit_by_choice_index.get(
            choice_index,
            _DissolveEvidence(),
        )
        rehydrated_choices.append(
            replace(
                choice,
                dissolve_entry_face_geometry_window=(
                    entry_evidence.face_geometry
                ),
                dissolve_exit_face_geometry_window=exit_evidence.face_geometry,
                dissolve_entry_face_box_window=entry_evidence.face_boxes,
                dissolve_exit_face_box_window=exit_evidence.face_boxes,
                dissolve_entry_face_identity_window=(
                    entry_evidence.face_identity_embeddings
                ),
                dissolve_exit_face_identity_window=(
                    exit_evidence.face_identity_embeddings
                ),
                dissolve_entry_pose_embedding_window=(
                    entry_evidence.pose_embeddings
                ),
                dissolve_exit_pose_embedding_window=(
                    exit_evidence.pose_embeddings
                ),
                dissolve_entry_pose_backend_window=entry_evidence.pose_backends,
                dissolve_exit_pose_backend_window=exit_evidence.pose_backends,
                dissolve_entry_pose_confidence_window=(
                    entry_evidence.pose_confidences
                ),
                dissolve_exit_pose_confidence_window=(
                    exit_evidence.pose_confidences
                ),
                dissolve_entry_silhouette_embedding_window=(
                    entry_evidence.silhouette_embeddings
                ),
                dissolve_exit_silhouette_embedding_window=(
                    exit_evidence.silhouette_embeddings
                ),
                dissolve_entry_silhouette_backend_window=(
                    entry_evidence.silhouette_backends
                ),
                dissolve_exit_silhouette_backend_window=(
                    exit_evidence.silhouette_backends
                ),
                dissolve_entry_silhouette_confidence_window=(
                    entry_evidence.silhouette_confidences
                ),
                dissolve_exit_silhouette_confidence_window=(
                    exit_evidence.silhouette_confidences
                ),
            )
        )

    basis_tokens = visual_timeline.analysis_basis.split("+")
    analysis_basis = visual_timeline.analysis_basis
    if _DISSOLVE_REHYDRATION_ANALYSIS_BASIS not in basis_tokens:
        analysis_basis = (
            f"{analysis_basis}+{_DISSOLVE_REHYDRATION_ANALYSIS_BASIS}"
        )
    return replace(
        visual_timeline,
        choices=tuple(rehydrated_choices),
        analysis_basis=analysis_basis,
    )


def _resolve_visual_timeline_candidate_record(
    candidate_set: VisualTimelineCandidateSet,
    candidate: int | VisualTimelineCandidateRecord | None,
) -> VisualTimelineCandidateRecord:
    if candidate is None:
        return candidate_set.selected
    if isinstance(candidate, bool):
        raise TypeError("candidate must be an index or candidate record")
    if isinstance(candidate, int):
        if not 0 <= candidate < len(candidate_set.candidates):
            raise IndexError("visual timeline candidate index is out of range")
        return candidate_set.candidates[candidate]
    if not isinstance(candidate, VisualTimelineCandidateRecord):
        raise TypeError("candidate must be an index or candidate record")
    index = candidate.candidate_index
    if (
        not 0 <= index < len(candidate_set.candidates)
        or candidate_set.candidates[index] is not candidate
    ):
        raise ValueError("candidate record does not belong to candidate set")
    return candidate


def _missing_candidate_face_metadata() -> _CandidateFaceMetadata:
    return _CandidateFaceMetadata(
        boxes=(None, None, None),
        failure="face_metadata_unavailable",
        backends=("none", "none", "none"),
        landmarks=(None, None, None),
        rolls=(None, None, None),
        eye_scales=(None, None, None),
    )


def _missing_candidate_subject_metadata() -> _CandidateSubjectMetadata:
    return _CandidateSubjectMetadata(
        pose_backends=(
            "structure_hog_fallback",
            "structure_hog_fallback",
            "structure_hog_fallback",
        ),
        pose_confidences=(0.0, 0.0, 0.0),
        person_counts=(0, 0, 0),
        formation_person_counts=(0, 0, 0),
        formation_upper_body_counts=(0, 0, 0),
        shot_scale_bases=(
            "unknown_no_subject_geometry",
            "unknown_no_subject_geometry",
            "unknown_no_subject_geometry",
        ),
        silhouette_boxes=(None, None, None),
        silhouette_centroids=(None, None, None),
        silhouette_backends=("none", "none", "none"),
        silhouette_confidences=(0.0, 0.0, 0.0),
    )


def materialize_visual_timeline_candidate(
    candidate_set: VisualTimelineCandidateSet,
    *,
    candidate: int | VisualTimelineCandidateRecord | None = None,
) -> VisualTimelinePlan:
    """Materialize any retained alternative into a complete visual plan.

    The candidate set shares frame descriptors across all paths.  Only the
    requested path pays for the transient post-DP dissolve-handle inference,
    so runner-up review remains exact without multiplying the retained frame
    metadata by the candidate count.
    """

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    record = _resolve_visual_timeline_candidate_record(candidate_set, candidate)
    optimized = record.timeline_result
    scene_cut_times: tuple[tuple[float, ...], ...] = (
        candidate_set.scene_cut_times
        if candidate_set.scene_cut_times
        else ((),) * len(candidate_set.sources)
    )
    dissolve_entry_evidence, dissolve_exit_evidence = (
        _selected_dissolve_evidence(
            sources=candidate_set.sources,
            optimized=optimized,
            audio_offsets=candidate_set.audio_offsets,
            source_timewarps=candidate_set.source_timewarps,
            scene_cut_times=scene_cut_times,
            face_metadata_by_candidate=candidate_set.face_metadata_by_candidate,
            feature_cache_directory=candidate_set.feature_cache_directory,
            feature_derived_cache_directory=(
                candidate_set.feature_derived_cache_directory
            ),
            require_persistent_feature_cache=(
                candidate_set.require_persistent_feature_cache
            ),
        )
    )
    choices: list[VisualTimelineChoice] = []
    for optimized_index, choice in enumerate(optimized.candidates):
        source_index = int(choice.source_id)
        source_local_start = _source_clock_time(
            choice.start_seconds,
            source_index=source_index,
            audio_offsets=candidate_set.audio_offsets,
            source_timewarps=candidate_set.source_timewarps,
        )
        source_local_end = _source_clock_time(
            choice.end_seconds,
            source_index=source_index,
            audio_offsets=candidate_set.audio_offsets,
            source_timewarps=candidate_set.source_timewarps,
        )
        face_metadata = candidate_set.face_metadata_by_candidate.get(
            choice.candidate_id,
            _missing_candidate_face_metadata(),
        )
        subject_metadata = candidate_set.subject_metadata_by_candidate.get(
            choice.candidate_id,
            _missing_candidate_subject_metadata(),
        )
        entry_evidence = dissolve_entry_evidence.get(
            choice.candidate_id,
            _DissolveEvidence(),
        )
        exit_evidence = dissolve_exit_evidence.get(
            choice.candidate_id,
            _DissolveEvidence(),
        )
        if optimized_index == 0:
            face_transition_mismatch = None
            layout_transition_mismatch = None
            pose_transition_basis = None
            pose_transition_mismatch = None
            silhouette_transition_mismatch = None
            formation_transition_mismatch = None
        else:
            previous = optimized.candidates[optimized_index - 1]
            previous_subject_metadata = (
                candidate_set.subject_metadata_by_candidate.get(
                    previous.candidate_id
                )
            )
            mismatch = face_geometry_minimax_mismatch(
                previous.exit_face_geometry_window,
                choice.entry_face_geometry_window,
            )
            face_transition_mismatch = (
                mismatch
                if previous.exit_face_geometry_window
                and choice.entry_face_geometry_window
                else None
            )
            previous_pose = previous.exit_pose_embedding or previous.pose_embedding
            current_pose = choice.entry_pose_embedding or choice.pose_embedding
            actual_pose_pair = (
                previous_subject_metadata is not None
                and previous_subject_metadata.pose_backends[2]
                != "structure_hog_fallback"
                and subject_metadata.pose_backends[0]
                != "structure_hog_fallback"
                and bool(previous_pose)
                and bool(current_pose)
                and len(previous_pose) == len(current_pose)
            )
            if actual_pose_pair:
                pose_transition_basis = "opencv_zoo_mediapipe_pose"
            else:
                pose_transition_basis = "structure_hog_fallback"
                previous_pose = (
                    previous.exit_structure_embedding
                    or previous.structure_embedding
                )
                current_pose = (
                    choice.entry_structure_embedding
                    or choice.structure_embedding
                )
            pose_transition_mismatch = _embedding_rms_distance(
                previous_pose,
                current_pose,
            )
            silhouette_transition_mismatch = _embedding_rms_distance(
                previous.exit_silhouette_embedding
                or previous.silhouette_embedding,
                choice.entry_silhouette_embedding
                or choice.silhouette_embedding,
            )
            layout_transition_mismatch = cross_source_layout_mismatch(
                previous,
                choice,
            )
            formation_transition_mismatch = comparable_group_formation_mismatch(
                previous.exit_formation_people,
                choice.entry_formation_people,
                left_shot_type=(
                    previous.exit_shot_type or previous.shot_type
                ),
                right_shot_type=(
                    choice.entry_shot_type or choice.shot_type
                ),
            )
        choices.append(
            VisualTimelineChoice(
                segment_index=choice.segment_index,
                source_index=source_index,
                canonical_master_start=choice.start_seconds,
                canonical_master_end=choice.end_seconds,
                source_local_start=source_local_start,
                source_local_end=source_local_end,
                quality_score=choice.quality_score,
                composition_score=choice.composition_score,
                boundary_score=choice.boundary_score,
                shot_scale=choice.shot_scale,
                shot_type=choice.shot_type,
                shot_id=choice.shot_id,
                entry_shot_scale=choice.entry_shot_scale,
                exit_shot_scale=choice.exit_shot_scale,
                entry_shot_type=choice.entry_shot_type,
                exit_shot_type=choice.exit_shot_type,
                entry_scene_cut_near=choice.entry_scene_cut_near,
                exit_scene_cut_near=choice.exit_scene_cut_near,
                intentional_transition_anchor_basis=(
                    choice.intentional_transition_anchor_basis
                ),
                internal_scene_cut_master_seconds=(
                    choice.internal_scene_cut_master_seconds
                ),
                entry_boundary_side_stability=(
                    choice.entry_boundary_side_stability
                ),
                exit_boundary_side_stability=choice.exit_boundary_side_stability,
                face_samples_xywh=face_metadata.boxes,
                face_match_candidate_failure=face_metadata.failure,
                face_detector_backends=face_metadata.backends,
                face_landmark_samples_xy=face_metadata.landmarks,
                face_head_roll_samples_radians=face_metadata.rolls,
                face_eye_scale_samples=face_metadata.eye_scales,
                entry_face_geometry_window=choice.entry_face_geometry_window,
                exit_face_geometry_window=choice.exit_face_geometry_window,
                dissolve_entry_face_geometry_window=entry_evidence.face_geometry,
                dissolve_exit_face_geometry_window=exit_evidence.face_geometry,
                dissolve_entry_face_box_window=entry_evidence.face_boxes,
                dissolve_exit_face_box_window=exit_evidence.face_boxes,
                dissolve_entry_face_identity_window=(
                    entry_evidence.face_identity_embeddings
                ),
                dissolve_exit_face_identity_window=(
                    exit_evidence.face_identity_embeddings
                ),
                dissolve_entry_pose_embedding_window=(
                    entry_evidence.pose_embeddings
                ),
                dissolve_exit_pose_embedding_window=exit_evidence.pose_embeddings,
                dissolve_entry_pose_backend_window=entry_evidence.pose_backends,
                dissolve_exit_pose_backend_window=exit_evidence.pose_backends,
                dissolve_entry_pose_confidence_window=(
                    entry_evidence.pose_confidences
                ),
                dissolve_exit_pose_confidence_window=(
                    exit_evidence.pose_confidences
                ),
                dissolve_entry_silhouette_embedding_window=(
                    entry_evidence.silhouette_embeddings
                ),
                dissolve_exit_silhouette_embedding_window=(
                    exit_evidence.silhouette_embeddings
                ),
                dissolve_entry_silhouette_backend_window=(
                    entry_evidence.silhouette_backends
                ),
                dissolve_exit_silhouette_backend_window=(
                    exit_evidence.silhouette_backends
                ),
                dissolve_entry_silhouette_confidence_window=(
                    entry_evidence.silhouette_confidences
                ),
                dissolve_exit_silhouette_confidence_window=(
                    exit_evidence.silhouette_confidences
                ),
                face_transition_minimax_mismatch_from_previous=(
                    face_transition_mismatch
                ),
                pose_backends=subject_metadata.pose_backends,
                pose_confidences=subject_metadata.pose_confidences,
                detected_person_counts=subject_metadata.person_counts,
                formation_person_counts=subject_metadata.formation_person_counts,
                formation_upper_body_counts=(
                    subject_metadata.formation_upper_body_counts
                ),
                shot_scale_bases=subject_metadata.shot_scale_bases,
                silhouette_samples_xywh=subject_metadata.silhouette_boxes,
                silhouette_centroid_samples_xy=(
                    subject_metadata.silhouette_centroids
                ),
                silhouette_backends=subject_metadata.silhouette_backends,
                silhouette_confidences=subject_metadata.silhouette_confidences,
                layout_transition_mismatch_from_previous=(
                    layout_transition_mismatch
                ),
                pose_transition_basis_from_previous=pose_transition_basis,
                pose_transition_mismatch_from_previous=pose_transition_mismatch,
                silhouette_transition_mismatch_from_previous=(
                    silhouette_transition_mismatch
                ),
                formation_transition_mismatch_from_previous=(
                    formation_transition_mismatch
                ),
            )
        )
    candidate_review_summaries = tuple(
        _candidate_review_summary(
            candidate_index=item.candidate_index,
            cadence_profile=item.cadence_profile,
            selected=item.candidate_index == record.candidate_index,
            candidate=item.timeline_result,
            review=item.review,
        )
        for item in candidate_set.candidates
    )
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=optimized.total_score,
        selected_candidate_index=record.candidate_index,
        candidate_generation_basis=candidate_set.candidate_generation_basis,
        candidate_review_summaries=candidate_review_summaries,
        source_clock_basis=candidate_set.source_clock_basis,
    )


def _describe_sampler_frame(
    sampler: Any,
    seconds: float,
    *,
    following_seconds: float | None = None,
    include_subject_geometry: bool = True,
    include_formation_pose: bool = False,
) -> FrameVisualDescriptor | None:
    """Use persistent-aware sampling while retaining lightweight test providers."""

    feature_method = getattr(sampler, "feature_descriptor_at", None)
    if callable(feature_method):
        return feature_method(
            seconds,
            following_seconds=following_seconds,
            include_subject_geometry=include_subject_geometry,
            include_formation_pose=include_formation_pose,
        )
    frame = sampler.frame_at(seconds)
    if frame is None:
        return None
    following = (
        sampler.frame_at(following_seconds)
        if following_seconds is not None
        else None
    )
    if following_seconds is not None and following is None:
        return None
    describe_arguments: dict[str, Any] = {"following_frame": following}
    if not include_subject_geometry:
        describe_arguments["include_subject_geometry"] = False
    if include_formation_pose:
        describe_arguments["include_formation_pose"] = True
    return describe_frame(frame, **describe_arguments)


def _professional_intentional_transition_anchor_basis(
    master_start: float,
    *,
    section_plan: SectionPlan | None,
    phrase_times: Sequence[float],
    professional_policy: ProfessionalMontageSelectionPolicy | None,
) -> str | None:
    """Mark only authored highlight phrase entries for bounded contrast cuts."""

    if section_plan is None or professional_policy is None:
        return None
    directive = section_plan.directive_at(master_start)
    if directive.role not in {
        SectionRole.BUILD,
        SectionRole.CHORUS_LIKE,
        SectionRole.BRIDGE_LIKE,
    }:
        return None
    if not any(
        math.isclose(
            master_start,
            float(phrase_time),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        for phrase_time in phrase_times
    ):
        return None
    return INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS


def _build_visual_timeline(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    preferred_source_indices: Sequence[int] | None = None,
    diversity_config: VisualDiversityConfig | None = None,
    section_plan: SectionPlan | None = None,
    phrase_times: Sequence[float] = (),
    professional_montage_selection_policy: (
        ProfessionalMontageSelectionPolicy | None
    ) = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
    prescribed_source_indices: Sequence[int] | None = None,
    maximum_candidate_results: int,
    candidate_set_only: bool,
) -> VisualTimelinePlan | VisualTimelineCandidateSet:
    """Score synchronized source options and globally select the edit.

    ``source_timewarps`` is optional and positional to the source arrays.  A
    ``None`` entry retains ``master + audio_offset``; a supplied map controls
    every decoded sample and the persisted source-local span.
    """

    if (
        isinstance(maximum_candidate_results, bool)
        or not isinstance(maximum_candidate_results, int)
        or not 8 <= maximum_candidate_results <= 32
    ):
        raise ValueError("maximum_candidate_results must be between 8 and 32")
    source_count = len(sources)
    if source_count < 2:
        raise ValueError("visual timeline needs at least two sources")
    if not (
        len(audio_offsets)
        == len(sync_scores)
        == len(scene_cut_times)
        == source_count
    ):
        raise ValueError("per-source visual inputs must have matching lengths")
    if not canonical_spans:
        raise ValueError("canonical_spans cannot be empty")
    clock_maps = _source_clock_maps(
        source_timewarps,
        audio_offsets=audio_offsets,
        source_count=source_count,
    )
    preferred_indices = (
        tuple(preferred_source_indices)
        if preferred_source_indices is not None
        else None
    )
    preferred_source_ids: tuple[str | None, ...] | None = None
    if preferred_indices is not None:
        if len(preferred_indices) != len(canonical_spans):
            raise ValueError("preferred source assignments must match canonical spans")
        if any(
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= source_count
            for source_index in preferred_indices
        ):
            raise ValueError("preferred source assignment contains an invalid index")
        # Coverage is meaningful only when there are enough edit spans to show
        # every source.  The metadata draft provides a full first-use cycle.
        if len(canonical_spans) >= source_count:
            if set(preferred_indices) != set(range(source_count)):
                raise ValueError(
                    "preferred source assignments must cover every source before reuse"
                )
            preferred_source_ids = tuple(str(index) for index in preferred_indices)
    prescribed_indices = (
        tuple(prescribed_source_indices)
        if prescribed_source_indices is not None
        else None
    )
    if prescribed_indices is not None:
        if len(prescribed_indices) != len(canonical_spans):
            raise ValueError("prescribed source path must match canonical spans")
        if any(
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= source_count
            for source_index in prescribed_indices
        ):
            raise ValueError("prescribed source path contains an invalid index")
        if preferred_indices is not None:
            raise ValueError(
                "prescribed source path cannot be combined with soft preferences"
            )
    diversity = diversity_config or VisualDiversityConfig()
    samplers = _create_video_samplers(
        sources,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
    )
    candidates_by_segment: list[list[TimelineCandidate]] = []
    face_metadata_by_candidate: dict[str, _CandidateFaceMetadata] = {}
    subject_metadata_by_candidate: dict[str, _CandidateSubjectMetadata] = {}
    try:
        for segment_index, (master_start, master_end) in enumerate(canonical_spans):
            if master_end <= master_start:
                raise ValueError("canonical spans must have positive duration")
            duration = master_end - master_start
            intentional_anchor_basis = (
                _professional_intentional_transition_anchor_basis(
                    master_start,
                    section_plan=section_plan,
                    phrase_times=phrase_times,
                    professional_policy=(
                        professional_montage_selection_policy
                    ),
                )
            )
            candidates: list[TimelineCandidate] = []
            for source_index, sampler in enumerate(samplers):
                source_start = _source_clock_time(
                    master_start,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                source_end = _source_clock_time(
                    master_end,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                inset = min(0.05, duration * 0.12)
                entry_master = master_start + inset
                exit_master = master_end - inset
                middle_master = (master_start + master_end) / 2.0
                entry_time = _source_clock_time(
                    entry_master,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                exit_time = _source_clock_time(
                    exit_master,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                middle_time = _source_clock_time(
                    middle_master,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                entry_following_master = min(master_end, entry_master + 0.08)
                entry_following_time = _source_clock_time(
                    entry_following_master,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                middle_following_time = _source_clock_time(
                    min(master_end, middle_master + 0.08),
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                # Outgoing motion must end before the rendered cut. Sampling
                # exit -> exit+80ms looked into footage the viewer never sees
                # and could reverse direction on a beat. Use a preceding frame
                # to measure forward-time motion into the actual exit sample.
                exit_preceding_master = max(master_start, exit_master - 0.08)
                exit_preceding_time = _source_clock_time(
                    exit_preceding_master,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                entry = _describe_sampler_frame(
                    sampler,
                    entry_time,
                    following_seconds=entry_following_time,
                    include_formation_pose=True,
                )
                middle = _describe_sampler_frame(
                    sampler,
                    middle_time,
                    following_seconds=middle_following_time,
                )
                exit_descriptor = _describe_sampler_frame(
                    sampler,
                    exit_time,
                    include_formation_pose=True,
                )
                entry_following_descriptor = _describe_sampler_frame(
                    sampler,
                    entry_following_time,
                )
                exit_preceding_descriptor = _describe_sampler_frame(
                    sampler,
                    exit_preceding_time,
                    following_seconds=exit_time,
                )
                if (
                    entry is None
                    or middle is None
                    or exit_descriptor is None
                ):
                    continue
                entry_boundary_stability = (
                    describe_boundary_side_stability(
                        entry,
                        entry_following_descriptor,
                        sample_span_seconds=(
                            entry_following_master - entry_master
                        ),
                    )
                    if entry_following_descriptor is not None
                    and entry_following_master > entry_master
                    else None
                )
                exit_boundary_stability = (
                    describe_boundary_side_stability(
                        exit_preceding_descriptor,
                        exit_descriptor,
                        sample_span_seconds=(
                            exit_master - exit_preceding_master
                        ),
                    )
                    if exit_preceding_descriptor is not None
                    and exit_master > exit_preceding_master
                    else None
                )
                exit_motion = (
                    exit_preceding_descriptor.motion
                    if exit_preceding_descriptor is not None
                    else None
                )
                cuts = tuple(
                    sorted({float(value) for value in scene_cut_times[source_index]})
                )
                internal_scene_cut_master_seconds = tuple(
                    master_cut
                    for source_cut in cuts
                    if source_start + 1e-6 < source_cut < source_end - 1e-6
                    for master_cut in (
                        _master_clock_time(
                            source_cut,
                            source_index=source_index,
                            audio_offsets=audio_offsets,
                            source_timewarps=clock_maps,
                        ),
                    )
                    if master_start + 1e-6 < master_cut < master_end - 1e-6
                )
                tolerance = min(0.14, max(0.06, duration * 0.12))
                boundary_risk_window = max(0.16, tolerance)
                entry_risk_end = _source_clock_time(
                    master_start + boundary_risk_window,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                exit_risk_start = _source_clock_time(
                    master_end - boundary_risk_window,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                entry_safe_start = _source_clock_time(
                    master_start - tolerance,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                exit_safe_end = _source_clock_time(
                    master_end + tolerance,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                # Camera cuts have a direction at an edit boundary. A source
                # cut just *before* an incoming segment or just *after* an
                # outgoing segment is hidden by the cross-edit and is often a
                # useful professional edit point. Only a cut that lands inside
                # the rendered handle creates the objectionable two-step jump.
                # Treat one decoded frame around the nominal boundary as the
                # same quantized cut rather than rejecting it for sub-frame
                # timestamp noise.
                boundary_frame_seconds = 1.0 / max(
                    float(getattr(sampler, "frame_rate", 29.97)),
                    1.0,
                )
                entry_scene_cut_near = any(
                    source_start + boundary_frame_seconds
                    < cut
                    <= entry_risk_end
                    for cut in cuts
                )
                exit_scene_cut_near = any(
                    exit_risk_start
                    <= cut
                    < source_end - boundary_frame_seconds
                    for cut in cuts
                )
                safely_aligned_boundary = any(
                    entry_safe_start
                    <= cut
                    <= source_start + boundary_frame_seconds
                    or source_end - boundary_frame_seconds
                    <= cut
                    <= exit_safe_end
                    for cut in cuts
                )
                boundary_score = 0.92 if safely_aligned_boundary else 0.52
                shot_number = bisect_right(cuts, middle_time)
                internal_start = _source_clock_time(
                    master_start + tolerance,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                internal_end = _source_clock_time(
                    master_end - tolerance,
                    source_index=source_index,
                    audio_offsets=audio_offsets,
                    source_timewarps=clock_maps,
                )
                has_internal_camera_cut = any(
                    internal_start < cut < internal_end
                    for cut in cuts
                )
                candidate_id = f"{source_index}:{segment_index}"
                face_samples = (
                    entry.primary_face_bbox,
                    middle.primary_face_bbox,
                    exit_descriptor.primary_face_bbox,
                )
                face_metadata_by_candidate[candidate_id] = _CandidateFaceMetadata(
                    boxes=face_samples,
                    failure=_face_match_candidate_failure(
                        (entry, middle, exit_descriptor),
                        has_internal_camera_cut=has_internal_camera_cut,
                    ),
                    backends=(
                        entry.face_detector_backend,
                        middle.face_detector_backend,
                        exit_descriptor.face_detector_backend,
                    ),
                    landmarks=(
                        entry.primary_face_landmarks_xy,
                        middle.primary_face_landmarks_xy,
                        exit_descriptor.primary_face_landmarks_xy,
                    ),
                    rolls=(
                        entry.primary_face_head_roll_radians,
                        middle.primary_face_head_roll_radians,
                        exit_descriptor.primary_face_head_roll_radians,
                    ),
                    eye_scales=(
                        entry.primary_face_eye_scale,
                        middle.primary_face_eye_scale,
                        exit_descriptor.primary_face_eye_scale,
                    ),
                )
                subject_metadata_by_candidate[candidate_id] = _CandidateSubjectMetadata(
                    pose_backends=(
                        entry.pose_backend,
                        middle.pose_backend,
                        exit_descriptor.pose_backend,
                    ),
                    pose_confidences=(
                        entry.pose_confidence,
                        middle.pose_confidence,
                        exit_descriptor.pose_confidence,
                    ),
                    person_counts=(
                        entry.detected_person_count,
                        middle.detected_person_count,
                        exit_descriptor.detected_person_count,
                    ),
                    formation_person_counts=(
                        len(entry.formation_people),
                        len(middle.formation_people),
                        len(exit_descriptor.formation_people),
                    ),
                    formation_upper_body_counts=(
                        sum(
                            bool(person.upper_body_keypoints_xyc)
                            for person in entry.formation_people
                        ),
                        sum(
                            bool(person.upper_body_keypoints_xyc)
                            for person in middle.formation_people
                        ),
                        sum(
                            bool(person.upper_body_keypoints_xyc)
                            for person in exit_descriptor.formation_people
                        ),
                    ),
                    shot_scale_bases=(
                        entry.shot_scale_basis,
                        middle.shot_scale_basis,
                        exit_descriptor.shot_scale_basis,
                    ),
                    silhouette_boxes=(
                        entry.silhouette_bbox_xywh,
                        middle.silhouette_bbox_xywh,
                        exit_descriptor.silhouette_bbox_xywh,
                    ),
                    silhouette_centroids=(
                        entry.silhouette_centroid_xy,
                        middle.silhouette_centroid_xy,
                        exit_descriptor.silhouette_centroid_xy,
                    ),
                    silhouette_backends=(
                        entry.silhouette_backend,
                        middle.silhouette_backend,
                        exit_descriptor.silhouette_backend,
                    ),
                    silhouette_confidences=(
                        entry.silhouette_confidence,
                        middle.silhouette_confidence,
                        exit_descriptor.silhouette_confidence,
                    ),
                )
                if duration >= 4.0 * _BOUNDARY_FRAME_STEP_SECONDS:
                    entry_face_window = _face_geometry_boundary_window(
                        sampler,
                        boundary_seconds=source_start,
                        incoming=True,
                        scene_cut_times=cuts,
                    )
                    exit_face_window = _face_geometry_boundary_window(
                        sampler,
                        boundary_seconds=source_end,
                        incoming=False,
                        scene_cut_times=cuts,
                    )
                else:
                    entry_face_window = ()
                    exit_face_window = ()
                entry_identity = sampler.sface_embedding_at(entry_time, entry)
                # This builder always supplies entry/exit shot-scale evidence,
                # so every transition takes the boundary-specific identity
                # branch in ``timeline``.  A midpoint SFace vector is never
                # consulted by emission, joint selection, review, or render;
                # keep it empty rather than paying one random decode per cell.
                middle_identity: tuple[float, ...] = ()
                exit_identity = sampler.sface_embedding_at(
                    exit_time,
                    exit_descriptor,
                )
                quality = (entry.quality_score + middle.quality_score + exit_descriptor.quality_score) / 3.0
                composition = (
                    entry.composition_score
                    + middle.composition_score
                    + exit_descriptor.composition_score
                ) / 3.0
                candidates.append(
                    TimelineCandidate(
                        segment_index=segment_index,
                        source_id=str(source_index),
                        start_seconds=master_start,
                        end_seconds=master_end,
                        quality_score=quality,
                        sync_score=_unit(sync_scores[source_index]),
                        composition_score=composition,
                        boundary_score=boundary_score,
                        shot_scale=middle.shot_scale,
                        entry_shot_scale=entry.shot_scale,
                        exit_shot_scale=exit_descriptor.shot_scale,
                        shot_type=middle.shot_type,
                        entry_shot_type=entry.shot_type,
                        exit_shot_type=exit_descriptor.shot_type,
                        entry_scene_cut_near=entry_scene_cut_near,
                        exit_scene_cut_near=exit_scene_cut_near,
                        intentional_transition_anchor_basis=(
                            intentional_anchor_basis
                        ),
                        internal_scene_cut_master_seconds=(
                            internal_scene_cut_master_seconds
                        ),
                        entry_boundary_side_stability=(
                            entry_boundary_stability
                        ),
                        exit_boundary_side_stability=(
                            exit_boundary_stability
                        ),
                        is_group_wide=middle.shot_type is ShotType.GROUP_WIDE,
                        pose_embedding=(
                            middle.pose_embedding
                            if middle.pose_backend != "structure_hog_fallback"
                            else ()
                        ),
                        entry_pose_embedding=(
                            entry.pose_embedding
                            if entry.pose_backend != "structure_hog_fallback"
                            else ()
                        ),
                        exit_pose_embedding=(
                            exit_descriptor.pose_embedding
                            if exit_descriptor.pose_backend
                            != "structure_hog_fallback"
                            else ()
                        ),
                        structure_embedding=middle.structure_embedding,
                        entry_structure_embedding=entry.structure_embedding,
                        exit_structure_embedding=exit_descriptor.structure_embedding,
                        silhouette_embedding=middle.silhouette_embedding,
                        entry_silhouette_embedding=entry.silhouette_embedding,
                        exit_silhouette_embedding=exit_descriptor.silhouette_embedding,
                        person_layout_embedding=middle.person_layout_embedding,
                        entry_person_layout_embedding=entry.person_layout_embedding,
                        exit_person_layout_embedding=exit_descriptor.person_layout_embedding,
                        formation_people=middle.formation_people,
                        entry_formation_people=entry.formation_people,
                        exit_formation_people=exit_descriptor.formation_people,
                        face_identity_embedding=middle_identity,
                        entry_face_identity_embedding=entry_identity,
                        exit_face_identity_embedding=exit_identity,
                        color_embedding=middle.color_embedding,
                        entry_color_embedding=entry.color_embedding,
                        exit_color_embedding=exit_descriptor.color_embedding,
                        entry_face_geometry_window=entry_face_window,
                        exit_face_geometry_window=exit_face_window,
                        focus_bbox_xywh=middle.primary_face_bbox,
                        entry_focus_bbox_xywh=entry.primary_face_bbox,
                        exit_focus_bbox_xywh=exit_descriptor.primary_face_bbox,
                        subject_bbox_xywh=middle.silhouette_bbox_xywh,
                        entry_subject_bbox_xywh=entry.silhouette_bbox_xywh,
                        exit_subject_bbox_xywh=exit_descriptor.silhouette_bbox_xywh,
                        subject_centroid_xy=middle.silhouette_centroid_xy,
                        entry_subject_centroid_xy=entry.silhouette_centroid_xy,
                        exit_subject_centroid_xy=exit_descriptor.silhouette_centroid_xy,
                        detected_person_count=middle.detected_person_count,
                        entry_detected_person_count=entry.detected_person_count,
                        exit_detected_person_count=exit_descriptor.detected_person_count,
                        motion=middle.motion,
                        entry_motion=entry.motion,
                        exit_motion=exit_motion,
                        candidate_id=candidate_id,
                        shot_id=f"{source_index}:shot-{shot_number}",
                    )
                )
            if not candidates:
                raise ValueError(f"no source frame is readable for segment {segment_index}")
            candidates_by_segment.append(candidates)
            if progress_callback is not None:
                progress_callback(segment_index + 1, len(canonical_spans))
    finally:
        for sampler in samplers:
            sampler.close()

    candidate_generation_basis = (
        "three_safe_cadence_profiles_round_robin_dedup_then_reference_review_v5"
    )
    minimum_materialization_target = 1 if prescribed_indices is not None else 8
    if prescribed_indices is not None:
        candidate_generation_basis = (
            "prescribed_source_path_visual_hard_gates_then_reference_review_v1"
        )
    elif professional_montage_selection_policy is not None:
        candidate_generation_basis = (
            "human_reference_role_conditioned_three_cadence_profiles_"
            "round_robin_phrase_anchor_contrast_decoded_montage_gate_v7"
        )
    source_clock_basis = (
        "local_monotonic_timewarp_with_per_source_scalar_fallback"
        if any(
            timewarp is not None and not timewarp.scalar_fallback
            for timewarp in clock_maps
        )
        else "global_scalar_offset"
    )
    workspace_candidates = tuple(
        tuple(segment) for segment in candidates_by_segment
    )
    workspace_scene_cut_times = tuple(
        tuple(float(value) for value in source_cuts)
        for source_cuts in scene_cut_times
    )
    workspace_phrase_times = tuple(float(value) for value in phrase_times)
    workspace_prescribed_source_ids = (
        tuple(str(value) for value in prescribed_indices)
        if prescribed_indices is not None
        else None
    )
    optimizer_input_identity = _visual_optimizer_input_identity(
        workspace_candidates,
        sources=sources,
        audio_offsets=audio_offsets,
        source_timewarps=clock_maps,
        source_clock_basis=source_clock_basis,
        scene_cut_times=workspace_scene_cut_times,
        preferred_source_ids=preferred_source_ids,
        diversity=diversity,
        section_plan=section_plan,
        phrase_times=workspace_phrase_times,
        professional_montage_policy=professional_montage_selection_policy,
        subject_metadata_by_candidate=subject_metadata_by_candidate,
        prescribed_source_ids=workspace_prescribed_source_ids,
        maximum_results=maximum_candidate_results,
    )
    optimizer_workspace = VisualTimelineOptimizerWorkspace(
        candidates_by_segment=workspace_candidates,
        input_identity=optimizer_input_identity,
        preferred_source_ids=preferred_source_ids,
        diversity=diversity,
        section_plan=section_plan,
        phrase_times=workspace_phrase_times,
        professional_montage_policy=professional_montage_selection_policy,
        subject_metadata_by_candidate=subject_metadata_by_candidate,
        prescribed_source_ids=workspace_prescribed_source_ids,
        maximum_results=maximum_candidate_results,
    )
    timeline_candidates, candidate_cadence_profiles, role_conditioning_audit = (
        _select_optimizer_workspace_candidates(
            optimizer_workspace,
            forbidden_boundary_edges=TimelineNoGoodSet(),
        )
    )
    candidate_records, selected_candidate_index = (
        _review_visual_timeline_candidates(
            timeline_candidates,
            candidate_cadence_profiles,
            available_source_count=source_count,
        )
    )
    role_conditioned_selection_payload = _role_conditioned_selection_payload(
        candidate_records,
        workspace=optimizer_workspace,
        role_conditioning_audit=role_conditioning_audit,
    )
    candidate_set = VisualTimelineCandidateSet(
        candidates=candidate_records,
        selected_candidate_index=selected_candidate_index,
        sources=tuple(Path(path) for path in sources),
        audio_offsets=tuple(float(value) for value in audio_offsets),
        source_timewarps=clock_maps,
        source_clock_basis=source_clock_basis,
        requested_maximum_results=maximum_candidate_results,
        scene_cut_times=workspace_scene_cut_times,
        face_metadata_by_candidate=face_metadata_by_candidate,
        subject_metadata_by_candidate=subject_metadata_by_candidate,
        candidate_generation_basis=candidate_generation_basis,
        minimum_materialization_target=minimum_materialization_target,
        role_conditioned_selection=role_conditioned_selection_payload,
        optimizer_workspace=optimizer_workspace,
        optimizer_input_identity=optimizer_input_identity,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
    )
    if candidate_set_only:
        return candidate_set
    return materialize_visual_timeline_candidate(candidate_set)



def build_visual_timeline(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    preferred_source_indices: Sequence[int] | None = None,
    diversity_config: VisualDiversityConfig | None = None,
    section_plan: SectionPlan | None = None,
    phrase_times: Sequence[float] = (),
    professional_montage_selection_policy: (
        ProfessionalMontageSelectionPolicy | None
    ) = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> VisualTimelinePlan:
    """Build the selected visual plan while preserving the existing API."""

    result = _build_visual_timeline(
        sources=sources,
        canonical_spans=canonical_spans,
        audio_offsets=audio_offsets,
        sync_scores=sync_scores,
        scene_cut_times=scene_cut_times,
        preferred_source_indices=preferred_source_indices,
        diversity_config=diversity_config,
        section_plan=section_plan,
        phrase_times=phrase_times,
        professional_montage_selection_policy=(
            professional_montage_selection_policy
        ),
        source_timewarps=source_timewarps,
        progress_callback=progress_callback,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
        maximum_candidate_results=32,
        candidate_set_only=False,
    )
    if not isinstance(result, VisualTimelinePlan):  # pragma: no cover - invariant
        raise TypeError("selected visual timeline builder returned a candidate set")
    return result


def build_visual_timeline_candidate_set(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    preferred_source_indices: Sequence[int] | None = None,
    diversity_config: VisualDiversityConfig | None = None,
    section_plan: SectionPlan | None = None,
    phrase_times: Sequence[float] = (),
    professional_montage_selection_policy: (
        ProfessionalMontageSelectionPolicy | None
    ) = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
    maximum_results: int = 32,
) -> VisualTimelineCandidateSet:
    """Retain reviewed top-N alternatives without selected-only postpasses.

    At least eight alternatives are requested so callers can perform a useful
    proxy-review batch; the optimizer can still return fewer when the visual
    search space contains fewer than eight unique safe paths.
    """

    result = _build_visual_timeline(
        sources=sources,
        canonical_spans=canonical_spans,
        audio_offsets=audio_offsets,
        sync_scores=sync_scores,
        scene_cut_times=scene_cut_times,
        preferred_source_indices=preferred_source_indices,
        diversity_config=diversity_config,
        section_plan=section_plan,
        phrase_times=phrase_times,
        professional_montage_selection_policy=(
            professional_montage_selection_policy
        ),
        source_timewarps=source_timewarps,
        progress_callback=progress_callback,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
        maximum_candidate_results=maximum_results,
        candidate_set_only=True,
    )
    if not isinstance(
        result, VisualTimelineCandidateSet
    ):  # pragma: no cover - invariant
        raise TypeError("candidate-set builder returned a selected visual plan")
    return result


def build_prescribed_visual_timeline_candidate_set(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    prescribed_source_indices: Sequence[int],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    diversity_config: VisualDiversityConfig | None = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_cache_directory: Path | None = None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
) -> VisualTimelineCandidateSet:
    """Materialize one externally proposed path through existing visual gates.

    Every frame descriptor and transition constraint is identical to the normal
    visual optimizer.  Restricting each span to the prescribed source means an
    unsafe learned edge raises instead of being replaced by a baseline source.
    """

    result = _build_visual_timeline(
        sources=sources,
        canonical_spans=canonical_spans,
        audio_offsets=audio_offsets,
        sync_scores=sync_scores,
        scene_cut_times=scene_cut_times,
        preferred_source_indices=None,
        diversity_config=diversity_config,
        source_timewarps=source_timewarps,
        progress_callback=progress_callback,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
        prescribed_source_indices=prescribed_source_indices,
        maximum_candidate_results=8,
        candidate_set_only=True,
    )
    if not isinstance(
        result, VisualTimelineCandidateSet
    ):  # pragma: no cover - invariant
        raise TypeError("prescribed candidate builder returned a visual plan")
    return result


__all__ = [
    "FrameVisualDescriptor",
    "PersistentVisualFeatureCacheInitializationError",
    "PersistentVisualFeatureCacheRuntimeError",
    "ProfessionalMontageSelectionPolicy",
    "VisualDiversityConfig",
    "VisualTimelineCandidateRecord",
    "VisualTimelineCandidateSet",
    "VisualTimelineOptimizerWorkspace",
    "VisualTimelineChoice",
    "VisualTimelinePlan",
    "YUNET_EXPECTED_SHA256",
    "YUNET_MODEL_NAME",
    "build_visual_timeline",
    "build_visual_timeline_candidate_set",
    "build_prescribed_visual_timeline_candidate_set",
    "describe_boundary_side_stability",
    "describe_frame",
    "frame_visual_numeric_features",
    "materialize_visual_timeline_candidate",
    "regenerate_visual_timeline_candidate_set",
    "rehydrate_visual_timeline_dissolve_evidence",
    "visual_analyzer_fingerprint_sha256",
    "yunet_model_asset_status",
]
