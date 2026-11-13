"""Temporal source attribution for recovered reference stage mixes.

This module is deliberately independent from the render timeline.  It accepts
already synchronized per-sample source-match evidence, keeps ``UNKNOWN`` as a
real Viterbi state, and refines a proposed known-source switch from frame-local
old/new evidence.  Media decoding and ORB feature extraction live in
``reference_match`` so the policy remains deterministic and cheap to test.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


UNKNOWN_SOURCE_ID = "__UNKNOWN__"
REFERENCE_ATTRIBUTION_BASIS = "conservative_visual_match_temporal_dp_v1"
REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION = 1
REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION = 1
REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION = 1
REFERENCE_EVIDENCE_ANALYSIS_BASIS = (
    "reference_hard_cut_reverse_match_evidence_v1"
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ReferenceAttributionConfig:
    """Conservative match gates and temporal source-state policy."""

    sample_rate_hz: float = 4.0
    minimum_score: float = 0.40
    high_confidence_score: float = 0.68
    minimum_homography_inliers: int = 10
    minimum_homography_inlier_ratio: float = 0.25
    minimum_top_two_margin: float = 0.055
    strong_margin: float = 0.25
    same_source_bonus: float = 0.04
    source_switch_penalty: float = 0.06
    unknown_transition_penalty: float = 0.02
    unknown_when_known_penalty: float = 0.04
    ping_pong_penalty: float = 0.10
    ping_pong_window_seconds: float = 0.50
    refinement_radius_seconds: float = 0.50
    boundary_identity_margin: float = 0.08
    boundary_timestamp_step_tolerance_ratio: float = 0.50
    minimum_boundary_identity_frames: int = 2
    maximum_uncertain_boundary_frames: int = 1

    def __post_init__(self) -> None:
        for name in (
            "sample_rate_hz",
            "minimum_score",
            "high_confidence_score",
            "minimum_homography_inlier_ratio",
            "minimum_top_two_margin",
            "strong_margin",
            "same_source_bonus",
            "source_switch_penalty",
            "unknown_transition_penalty",
            "unknown_when_known_penalty",
            "ping_pong_penalty",
            "ping_pong_window_seconds",
            "refinement_radius_seconds",
            "boundary_identity_margin",
            "boundary_timestamp_step_tolerance_ratio",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        for name in (
            "minimum_score",
            "high_confidence_score",
            "minimum_homography_inlier_ratio",
            "minimum_top_two_margin",
            "strong_margin",
            "boundary_identity_margin",
            "boundary_timestamp_step_tolerance_ratio",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")
        if self.high_confidence_score < self.minimum_score:
            raise ValueError("high_confidence_score cannot be below minimum_score")
        for name in (
            "minimum_homography_inliers",
            "minimum_boundary_identity_frames",
            "maximum_uncertain_boundary_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.minimum_boundary_identity_frames < 1:
            raise ValueError("minimum_boundary_identity_frames must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": REFERENCE_ATTRIBUTION_BASIS,
            "sample_rate_hz": self.sample_rate_hz,
            "acceptance": {
                "minimum_score": self.minimum_score,
                "high_confidence_score": self.high_confidence_score,
                "minimum_homography_inliers": (
                    self.minimum_homography_inliers
                ),
                "minimum_homography_inlier_ratio": (
                    self.minimum_homography_inlier_ratio
                ),
                "minimum_top_two_margin": self.minimum_top_two_margin,
                "strong_margin": self.strong_margin,
                "unknown_on_failed_gate": True,
            },
            "temporal_path": {
                "same_source_bonus": self.same_source_bonus,
                "source_switch_penalty": self.source_switch_penalty,
                "unknown_transition_penalty": (
                    self.unknown_transition_penalty
                ),
                "unknown_when_known_penalty": (
                    self.unknown_when_known_penalty
                ),
                "ping_pong_penalty": self.ping_pong_penalty,
                "ping_pong_window_seconds": self.ping_pong_window_seconds,
            },
            "boundary_refinement": {
                "radius_seconds": self.refinement_radius_seconds,
                "identity_margin": self.boundary_identity_margin,
                "timestamp_step_tolerance_ratio": (
                    self.boundary_timestamp_step_tolerance_ratio
                ),
                "minimum_identity_frames_per_side": (
                    self.minimum_boundary_identity_frames
                ),
                "maximum_uncertain_middle_frames": (
                    self.maximum_uncertain_boundary_frames
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ReferenceCandidateScore:
    source_id: str
    score: float
    homography_inliers: int
    homography_inlier_ratio: float
    good_matches: int = 0
    structure_ncc: float | None = None
    color_similarity: float | None = None
    source_frame_index: int | None = None
    source_time_seconds: float | None = None
    time_adjustment_seconds: float | None = None
    homography_reference_to_source: (
        tuple[tuple[float, float, float], ...] | None
    ) = None

    def __post_init__(self) -> None:
        source = self.source_id.strip()
        if not source:
            raise ValueError("source_id cannot be empty")
        if source == UNKNOWN_SOURCE_ID:
            raise ValueError("source_id collides with the reserved UNKNOWN state")
        score = _finite(self.score, "candidate score")
        ratio = _finite(self.homography_inlier_ratio, "homography inlier ratio")
        if not 0.0 <= score <= 1.0 or not 0.0 <= ratio <= 1.0:
            raise ValueError("candidate score and inlier ratio must be normalized")
        for name in ("homography_inliers", "good_matches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.source_frame_index is not None:
            if (
                isinstance(self.source_frame_index, bool)
                or not isinstance(self.source_frame_index, int)
                or self.source_frame_index < 0
            ):
                raise ValueError(
                    "source_frame_index must be a non-negative integer"
                )
        if self.source_time_seconds is not None:
            source_time = _finite(self.source_time_seconds, "source time")
            if source_time < 0.0:
                raise ValueError("source_time_seconds cannot be negative")
            object.__setattr__(self, "source_time_seconds", source_time)
        if self.time_adjustment_seconds is not None:
            object.__setattr__(
                self,
                "time_adjustment_seconds",
                _finite(self.time_adjustment_seconds, "time adjustment"),
            )
        if self.homography_reference_to_source is not None:
            matrix = self.homography_reference_to_source
            if not isinstance(matrix, Sequence) or len(matrix) != 3:
                raise ValueError("reference homography must be a 3x3 matrix")
            normalized_rows: list[tuple[float, float, float]] = []
            for row in matrix:
                if not isinstance(row, Sequence) or len(row) != 3:
                    raise ValueError("reference homography must be a 3x3 matrix")
                normalized_rows.append(
                    tuple(
                        _finite(value, "reference homography value")
                        for value in row
                    )
                )
            object.__setattr__(
                self,
                "homography_reference_to_source",
                tuple(normalized_rows),
            )
        for name in ("structure_ncc", "color_similarity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "homography_inlier_ratio", ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.source_id,
            "score": self.score,
            "orb_good_matches": self.good_matches,
            "orb_homography_inliers": self.homography_inliers,
            "orb_inlier_ratio": self.homography_inlier_ratio,
            "structure_ncc": self.structure_ncc,
            "color_similarity": self.color_similarity,
            "source_frame_index": self.source_frame_index,
            "source_time_seconds": self.source_time_seconds,
            "time_adjustment_seconds": self.time_adjustment_seconds,
            "homography_reference_to_source": (
                [list(row) for row in self.homography_reference_to_source]
                if self.homography_reference_to_source is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    master_seconds: float
    candidates: tuple[ReferenceCandidateScore, ...]

    def __post_init__(self) -> None:
        time_value = _finite(self.master_seconds, "sample master time")
        if time_value < 0.0:
            raise ValueError("sample master time cannot be negative")
        if any(
            not isinstance(candidate, ReferenceCandidateScore)
            for candidate in self.candidates
        ):
            raise TypeError("sample candidates must be ReferenceCandidateScore values")
        source_ids = tuple(candidate.source_id for candidate in self.candidates)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("sample candidates cannot repeat a source")
        object.__setattr__(self, "master_seconds", time_value)
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda item: item.source_id)),
        )


@dataclass(frozen=True, slots=True)
class ReferencePathState:
    sample_index: int
    master_seconds: float
    source_id: str | None
    candidate: ReferenceCandidateScore | None
    emission_score: float
    transition_score: float
    cumulative_score: float
    decision_reason: str

    @property
    def is_unknown(self) -> bool:
        return self.source_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "master_seconds": self.master_seconds,
            "state": self.source_id or UNKNOWN_SOURCE_ID,
            "attributed_video_id": self.source_id,
            "accepted": self.source_id is not None,
            "decision_reason": self.decision_reason,
            "emission_score": self.emission_score,
            "transition_score": self.transition_score,
            "cumulative_score": self.cumulative_score,
            "best": self.candidate.to_dict() if self.candidate is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ReferenceSourceRun:
    source_id: str | None
    start_master_seconds: float
    end_master_seconds: float
    first_sample_index: int
    last_sample_index: int
    peak_candidate_score: float | None

    @property
    def duration_seconds(self) -> float:
        return self.end_master_seconds - self.start_master_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.source_id or UNKNOWN_SOURCE_ID,
            "attributed_video_id": self.source_id,
            "start_master_seconds": self.start_master_seconds,
            "end_master_seconds": self.end_master_seconds,
            "duration_seconds": self.duration_seconds,
            "first_sample_index": self.first_sample_index,
            "last_sample_index": self.last_sample_index,
            "peak_candidate_score": self.peak_candidate_score,
        }


@dataclass(frozen=True, slots=True)
class BoundaryFrameEvidence:
    frame_index: int
    master_seconds: float
    old_score: float
    new_score: float
    old_confident: bool
    new_confident: bool

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        master = _finite(self.master_seconds, "boundary frame master time")
        old_score = _finite(self.old_score, "old source score")
        new_score = _finite(self.new_score, "new source score")
        if master < 0.0:
            raise ValueError("boundary frame master time cannot be negative")
        if not 0.0 <= old_score <= 1.0 or not 0.0 <= new_score <= 1.0:
            raise ValueError("boundary source scores must be normalized")
        if not isinstance(self.old_confident, bool) or not isinstance(
            self.new_confident,
            bool,
        ):
            raise TypeError("boundary confidence flags must be booleans")
        object.__setattr__(self, "master_seconds", master)
        object.__setattr__(self, "old_score", old_score)
        object.__setattr__(self, "new_score", new_score)


@dataclass(frozen=True, slots=True)
class ReferenceBoundaryRefinement:
    old_source_id: str
    new_source_id: str
    nominal_boundary_seconds: float
    nominal_frame_index: int
    refinement_radius_seconds: float
    status: str
    selected_boundary_seconds: float | None
    selected_frame_index: int | None
    frame_offset: int | None
    objective_score: float | None
    old_identity_frames: int
    new_identity_frames: int
    uncertain_middle_frames: int
    frame_identities: tuple[str, ...]
    evidence_frame_indices: tuple[int, ...]
    evidence_master_seconds: tuple[float, ...]

    @property
    def resolved_hard_cut(self) -> bool:
        return self.status == "resolved_hard_cut"

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_source_id": self.old_source_id,
            "new_source_id": self.new_source_id,
            "nominal_boundary_seconds": self.nominal_boundary_seconds,
            "nominal_frame_index": self.nominal_frame_index,
            "refinement_radius_seconds": self.refinement_radius_seconds,
            "status": self.status,
            "selected_boundary_seconds": self.selected_boundary_seconds,
            "selected_frame_index": self.selected_frame_index,
            "frame_offset": self.frame_offset,
            "objective_score": self.objective_score,
            "old_identity_frames": self.old_identity_frames,
            "new_identity_frames": self.new_identity_frames,
            "uncertain_middle_frames": self.uncertain_middle_frames,
            "frame_identities": list(self.frame_identities),
            "evidence_frame_indices": list(self.evidence_frame_indices),
            "evidence_master_seconds": list(self.evidence_master_seconds),
        }


@dataclass(frozen=True, slots=True)
class _DPKey:
    source_id: str | None
    source_before_run: str | None
    run_start_index: int
    run_has_strong_evidence: bool


@dataclass(frozen=True, slots=True)
class _DPNode:
    cumulative_score: float
    previous_key: _DPKey | None
    candidate: ReferenceCandidateScore | None
    emission_score: float
    transition_score: float
    decision_reason: str


def _ranked_candidates(
    sample: ReferenceSample,
) -> tuple[ReferenceCandidateScore, ...]:
    return tuple(
        sorted(sample.candidates, key=lambda item: (-item.score, item.source_id))
    )


def _eligible_candidates(
    sample: ReferenceSample,
    config: ReferenceAttributionConfig,
) -> tuple[tuple[ReferenceCandidateScore, float, bool], ...]:
    ranked = _ranked_candidates(sample)
    output: list[tuple[ReferenceCandidateScore, float, bool]] = []
    for candidate in ranked:
        runner_score = max(
            (
                other.score
                for other in ranked
                if other.source_id != candidate.source_id
            ),
            default=0.0,
        )
        margin = candidate.score - runner_score
        if candidate.score < config.minimum_score:
            continue
        if candidate.homography_inliers < config.minimum_homography_inliers:
            continue
        if (
            candidate.homography_inlier_ratio
            < config.minimum_homography_inlier_ratio
        ):
            continue
        if (
            candidate.score < config.high_confidence_score
            and margin < config.minimum_top_two_margin
        ):
            continue
        emission = candidate.score - config.minimum_score
        strong = (
            candidate.score >= config.high_confidence_score
            and margin >= config.strong_margin
        )
        output.append((candidate, emission, strong))
    return tuple(output)


def _state_sort_key(source_id: str | None) -> tuple[int, str]:
    return (1, "") if source_id is None else (0, source_id)


def _transition(
    previous: _DPKey,
    current_source: str | None,
    *,
    current_index: int,
    samples: Sequence[ReferenceSample],
    current_is_strong: bool,
    config: ReferenceAttributionConfig,
) -> tuple[float, _DPKey]:
    if previous.source_id == current_source:
        return (
            config.same_source_bonus if current_source is not None else 0.0,
            _DPKey(
                source_id=current_source,
                source_before_run=previous.source_before_run,
                run_start_index=previous.run_start_index,
                run_has_strong_evidence=(
                    previous.run_has_strong_evidence or current_is_strong
                ),
            ),
        )

    if previous.source_id is None or current_source is None:
        transition_score = -config.unknown_transition_penalty
    else:
        transition_score = -config.source_switch_penalty

    if (
        previous.source_id is not None
        and current_source is not None
        and previous.source_before_run == current_source
    ):
        run_duration = (
            samples[current_index].master_seconds
            - samples[previous.run_start_index].master_seconds
        )
        if (
            run_duration <= config.ping_pong_window_seconds + 1e-9
            and not previous.run_has_strong_evidence
        ):
            transition_score -= config.ping_pong_penalty

    source_before_run = previous.source_id if previous.source_id is not None else None
    return (
        transition_score,
        _DPKey(
            source_id=current_source,
            source_before_run=source_before_run,
            run_start_index=current_index,
            run_has_strong_evidence=current_is_strong,
        ),
    )


def solve_reference_source_path(
    samples: Sequence[ReferenceSample],
    *,
    config: ReferenceAttributionConfig | None = None,
) -> tuple[ReferencePathState, ...]:
    """Select a source or explicit UNKNOWN for every synchronized sample."""

    settings = config or ReferenceAttributionConfig()
    ordered = tuple(samples)
    if not ordered:
        raise ValueError("reference attribution needs at least one sample")
    if any(not isinstance(sample, ReferenceSample) for sample in ordered):
        raise TypeError("samples must contain ReferenceSample values")
    if any(
        right.master_seconds <= left.master_seconds
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("reference samples must have strictly increasing times")

    options: list[
        tuple[tuple[str | None, ReferenceCandidateScore | None, float, bool, str], ...]
    ] = []
    for sample in ordered:
        eligible = _eligible_candidates(sample, settings)
        row: list[tuple[str | None, ReferenceCandidateScore | None, float, bool, str]] = [
            (
                candidate.source_id,
                candidate,
                emission,
                strong,
                "eligible_source_match",
            )
            for candidate, emission, strong in eligible
        ]
        row.append(
            (
                None,
                None,
                -settings.unknown_when_known_penalty if eligible else 0.0,
                False,
                "explicit_unknown_with_eligible_source"
                if eligible
                else "no_source_passed_conservative_gates",
            )
        )
        row.sort(key=lambda item: _state_sort_key(item[0]))
        options.append(tuple(row))

    layers: list[dict[_DPKey, _DPNode]] = []
    first: dict[_DPKey, _DPNode] = {}
    for source_id, candidate, emission, strong, reason in options[0]:
        key = _DPKey(source_id, None, 0, strong)
        first[key] = _DPNode(
            cumulative_score=emission,
            previous_key=None,
            candidate=candidate,
            emission_score=emission,
            transition_score=0.0,
            decision_reason=reason,
        )
    layers.append(first)

    for index in range(1, len(ordered)):
        current: dict[_DPKey, _DPNode] = {}
        for previous_key, previous_node in sorted(
            layers[-1].items(),
            key=lambda item: (
                _state_sort_key(item[0].source_id),
                _state_sort_key(item[0].source_before_run),
                item[0].run_start_index,
                item[0].run_has_strong_evidence,
            ),
        ):
            for source_id, candidate, emission, strong, reason in options[index]:
                transition_score, key = _transition(
                    previous_key,
                    source_id,
                    current_index=index,
                    samples=ordered,
                    current_is_strong=strong,
                    config=settings,
                )
                cumulative = (
                    previous_node.cumulative_score + emission + transition_score
                )
                existing = current.get(key)
                if existing is None or cumulative > existing.cumulative_score + 1e-12:
                    current[key] = _DPNode(
                        cumulative_score=cumulative,
                        previous_key=previous_key,
                        candidate=candidate,
                        emission_score=emission,
                        transition_score=transition_score,
                        decision_reason=reason,
                    )
        if not current:
            raise ValueError(f"sample {index} has no attribution state")
        layers.append(current)

    final_key, _final_node = max(
        layers[-1].items(),
        key=lambda item: (
            item[1].cumulative_score,
            -_state_sort_key(item[0].source_id)[0],
            tuple(-ord(character) for character in item[0].source_id or ""),
            -item[0].run_start_index,
        ),
    )
    keys = [final_key]
    for index in range(len(layers) - 1, 0, -1):
        previous = layers[index][keys[-1]].previous_key
        assert previous is not None
        keys.append(previous)
    keys.reverse()

    output: list[ReferencePathState] = []
    for index, key in enumerate(keys):
        node = layers[index][key]
        output.append(
            ReferencePathState(
                sample_index=index,
                master_seconds=ordered[index].master_seconds,
                source_id=key.source_id,
                candidate=node.candidate,
                emission_score=node.emission_score,
                transition_score=node.transition_score,
                cumulative_score=node.cumulative_score,
                decision_reason=node.decision_reason,
            )
        )
    return tuple(output)


def collapse_reference_source_runs(
    path: Sequence[ReferencePathState],
    *,
    master_start_seconds: float | None = None,
    master_end_seconds: float | None = None,
) -> tuple[ReferenceSourceRun, ...]:
    """Collapse a dense source path into midpoint-bounded source-state runs."""

    states = tuple(path)
    if not states:
        return ()
    if any(
        right.master_seconds <= left.master_seconds
        for left, right in zip(states, states[1:], strict=False)
    ):
        raise ValueError("path states must have strictly increasing times")
    if len(states) >= 2:
        leading_half_step = (
            states[1].master_seconds - states[0].master_seconds
        ) / 2.0
        trailing_half_step = (
            states[-1].master_seconds - states[-2].master_seconds
        ) / 2.0
    else:
        # A single sample still represents one sample interval rather than a
        # zero-duration run.  The caller can provide exact media bounds when
        # they are known.
        leading_half_step = 0.5
        trailing_half_step = 0.5
    start = (
        _finite(master_start_seconds, "master start")
        if master_start_seconds is not None
        else max(0.0, states[0].master_seconds - leading_half_step)
    )
    end = (
        _finite(master_end_seconds, "master end")
        if master_end_seconds is not None
        else states[-1].master_seconds + trailing_half_step
    )
    if start < 0.0 or end < states[-1].master_seconds or start > states[0].master_seconds:
        raise ValueError("master bounds must contain every path sample")
    if end <= start:
        raise ValueError("master end must follow master start")
    boundaries = [start]
    boundaries.extend(
        (left.master_seconds + right.master_seconds) / 2.0
        for left, right in zip(states, states[1:], strict=False)
    )
    boundaries.append(end)

    runs: list[ReferenceSourceRun] = []
    run_start = 0
    for index in range(1, len(states) + 1):
        if index < len(states) and states[index].source_id == states[run_start].source_id:
            continue
        run_states = states[run_start:index]
        scores = [
            state.candidate.score
            for state in run_states
            if state.candidate is not None
        ]
        runs.append(
            ReferenceSourceRun(
                source_id=states[run_start].source_id,
                start_master_seconds=boundaries[run_start],
                end_master_seconds=boundaries[index],
                first_sample_index=run_states[0].sample_index,
                last_sample_index=run_states[-1].sample_index,
                peak_candidate_score=max(scores) if scores else None,
            )
        )
        run_start = index
    return tuple(runs)


def _frame_identity(
    evidence: BoundaryFrameEvidence,
    *,
    margin: float,
) -> str:
    if evidence.old_confident and evidence.new_confident:
        difference = evidence.old_score - evidence.new_score
        if abs(difference) < margin:
            return "ambiguous"
        return "old" if difference > 0.0 else "new"
    if evidence.old_confident:
        return "old"
    if evidence.new_confident:
        return "new"
    return "neither"


def _validate_boundary_frame_sequence(
    frames: Sequence[BoundaryFrameEvidence],
    *,
    nominal_boundary_seconds: float,
    nominal_frame_index: int,
    timestamp_step_tolerance_ratio: float,
) -> None:
    if any(
        right.frame_index != left.frame_index + 1
        for left, right in zip(frames, frames[1:], strict=False)
    ):
        raise ValueError("boundary evidence frame indices must be consecutive")
    time_steps = tuple(
        right.master_seconds - left.master_seconds
        for left, right in zip(frames, frames[1:], strict=False)
    )
    if any(step <= 0.0 for step in time_steps):
        raise ValueError("boundary evidence timestamps must increase")
    expected_step = median(time_steps)
    minimum_step = expected_step * (1.0 - timestamp_step_tolerance_ratio)
    maximum_step = expected_step * (1.0 + timestamp_step_tolerance_ratio)
    if any(
        step < minimum_step - 1e-9 or step > maximum_step + 1e-9
        for step in time_steps
    ):
        raise ValueError("boundary evidence timestamp cadence is inconsistent")
    nominal_frame = next(
        (item for item in frames if item.frame_index == nominal_frame_index),
        None,
    )
    if nominal_frame is None:
        raise ValueError("boundary evidence must include the nominal frame")
    if (
        abs(nominal_frame.master_seconds - nominal_boundary_seconds)
        > maximum_step + 1e-9
    ):
        raise ValueError("nominal frame and boundary timestamp are inconsistent")


def refine_reference_source_boundary(
    *,
    old_source_id: str,
    new_source_id: str,
    nominal_boundary_seconds: float,
    nominal_frame_index: int,
    frame_evidence: Sequence[BoundaryFrameEvidence],
    config: ReferenceAttributionConfig | None = None,
) -> ReferenceBoundaryRefinement:
    """Resolve one known A->B source change with a single change-point search."""

    settings = config or ReferenceAttributionConfig()
    old_source = old_source_id.strip()
    new_source = new_source_id.strip()
    if not old_source or not new_source or old_source == new_source:
        raise ValueError("boundary refinement needs two distinct known sources")
    nominal_seconds = _finite(nominal_boundary_seconds, "nominal boundary")
    if nominal_seconds < 0.0:
        raise ValueError("nominal boundary cannot be negative")
    if (
        isinstance(nominal_frame_index, bool)
        or not isinstance(nominal_frame_index, int)
        or nominal_frame_index < 0
    ):
        raise ValueError("nominal_frame_index must be a non-negative integer")
    supplied_frames = tuple(frame_evidence)
    if any(not isinstance(item, BoundaryFrameEvidence) for item in supplied_frames):
        raise TypeError("frame_evidence must contain BoundaryFrameEvidence values")
    all_frames = tuple(sorted(supplied_frames, key=lambda item: item.frame_index))
    if any(
        right.frame_index <= left.frame_index
        for left, right in zip(all_frames, all_frames[1:], strict=False)
    ):
        raise ValueError("boundary frame indices must be unique and increasing")
    frames = tuple(
        item
        for item in all_frames
        if abs(item.master_seconds - nominal_seconds)
        <= settings.refinement_radius_seconds + 1e-9
    )
    if len(frames) < 2 * settings.minimum_boundary_identity_frames:
        return ReferenceBoundaryRefinement(
            old_source_id=old_source,
            new_source_id=new_source,
            nominal_boundary_seconds=nominal_seconds,
            nominal_frame_index=nominal_frame_index,
            refinement_radius_seconds=settings.refinement_radius_seconds,
            status="unresolved_insufficient_evidence",
            selected_boundary_seconds=None,
            selected_frame_index=None,
            frame_offset=None,
            objective_score=None,
            old_identity_frames=0,
            new_identity_frames=0,
            uncertain_middle_frames=len(frames),
            frame_identities=tuple(
                _frame_identity(item, margin=settings.boundary_identity_margin)
                for item in frames
            ),
            evidence_frame_indices=tuple(item.frame_index for item in frames),
            evidence_master_seconds=tuple(
                item.master_seconds for item in frames
            ),
        )
    _validate_boundary_frame_sequence(
        frames,
        nominal_boundary_seconds=nominal_seconds,
        nominal_frame_index=nominal_frame_index,
        timestamp_step_tolerance_ratio=(
            settings.boundary_timestamp_step_tolerance_ratio
        ),
    )
    identities = tuple(
        _frame_identity(item, margin=settings.boundary_identity_margin)
        for item in frames
    )

    prefix_old = [0.0]
    for item in frames:
        prefix_old.append(prefix_old[-1] + item.old_score)
    suffix_new = [0.0] * (len(frames) + 1)
    for index in range(len(frames) - 1, -1, -1):
        suffix_new[index] = suffix_new[index + 1] + frames[index].new_score
    candidates = [
        (prefix_old[split] + suffix_new[split], split)
        for split in range(1, len(frames))
    ]
    objective, split = max(
        candidates,
        key=lambda item: (
            item[0],
            -abs(frames[item[1]].frame_index - nominal_frame_index),
            -frames[item[1]].frame_index,
        ),
    )

    old_count = 0
    for identity in reversed(identities[:split]):
        if identity != "old":
            break
        old_count += 1
    new_count = 0
    for identity in identities[split:]:
        if identity != "new":
            break
        new_count += 1
    old_positions = [index for index, identity in enumerate(identities) if identity == "old"]
    new_positions = [index for index, identity in enumerate(identities) if identity == "new"]
    if old_positions and new_positions:
        last_old = max((index for index in old_positions if index < split), default=-1)
        first_new = min((index for index in new_positions if index >= split), default=len(frames))
        uncertain = sum(
            identities[index] in {"ambiguous", "neither"}
            for index in range(last_old + 1, first_new)
        )
    else:
        uncertain = len(frames)

    contradictory_identity = any(
        identity == "new" for identity in identities[:split]
    ) or any(identity == "old" for identity in identities[split:])
    resolved = (
        old_count >= settings.minimum_boundary_identity_frames
        and new_count >= settings.minimum_boundary_identity_frames
        and uncertain <= settings.maximum_uncertain_boundary_frames
        and not contradictory_identity
    )
    if resolved:
        selected = frames[split]
        status = "resolved_hard_cut"
        selected_seconds: float | None = selected.master_seconds
        selected_frame: int | None = selected.frame_index
        offset: int | None = selected.frame_index - nominal_frame_index
    else:
        if contradictory_identity:
            status = "unresolved_non_monotonic_identity"
        elif uncertain > settings.maximum_uncertain_boundary_frames:
            status = "unresolved_blend_or_unknown"
        else:
            status = "unresolved_insufficient_identity_run"
        selected_seconds = None
        selected_frame = None
        offset = None
    return ReferenceBoundaryRefinement(
        old_source_id=old_source,
        new_source_id=new_source,
        nominal_boundary_seconds=nominal_seconds,
        nominal_frame_index=nominal_frame_index,
        refinement_radius_seconds=settings.refinement_radius_seconds,
        status=status,
        selected_boundary_seconds=selected_seconds,
        selected_frame_index=selected_frame,
        frame_offset=offset,
        objective_score=objective,
        old_identity_frames=old_count,
        new_identity_frames=new_count,
        uncertain_middle_frames=uncertain,
        frame_identities=identities,
        evidence_frame_indices=tuple(item.frame_index for item in frames),
        evidence_master_seconds=tuple(item.master_seconds for item in frames),
    )


def _candidate_from_mapping(value: Mapping[str, Any]) -> ReferenceCandidateScore:
    source_id = str(value.get("video_id") or value.get("source_id") or "")
    return ReferenceCandidateScore(
        source_id=source_id,
        score=value.get("score"),
        homography_inliers=value.get(
            "orb_homography_inliers",
            value.get("homography_inliers"),
        ),
        homography_inlier_ratio=value.get(
            "orb_inlier_ratio",
            value.get("homography_inlier_ratio"),
        ),
        good_matches=value.get(
            "orb_good_matches",
            value.get("good_matches", 0),
        ),
        structure_ncc=value.get("structure_ncc"),
        color_similarity=value.get("color_similarity"),
        source_frame_index=value.get("source_frame_index"),
        source_time_seconds=value.get("source_time_seconds"),
        time_adjustment_seconds=value.get("time_adjustment_seconds"),
        homography_reference_to_source=value.get(
            "homography_reference_to_source"
        ),
    )


def _sample_from_mapping(value: Mapping[str, Any]) -> ReferenceSample:
    raw_candidates = value.get("ranked_candidates", value.get("candidates"))
    if not isinstance(raw_candidates, list):
        raise ValueError("each reference sample needs a ranked_candidates array")
    candidates: list[ReferenceCandidateScore] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(
                f"reference sample candidate {candidate_index} must be an object"
            )
        candidates.append(_candidate_from_mapping(raw_candidate))
    return ReferenceSample(
        master_seconds=value.get("master_seconds"),
        candidates=tuple(candidates),
    )


def _boundary_evidence_from_mapping(
    value: Mapping[str, Any],
) -> BoundaryFrameEvidence:
    return BoundaryFrameEvidence(
        frame_index=value.get("frame_index"),
        master_seconds=value.get("master_seconds"),
        old_score=value.get("old_score"),
        new_score=value.get("new_score"),
        old_confident=value.get("old_confident"),
        new_confident=value.get("new_confident"),
    )


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sample_reason_set(evidence_sample: Mapping[str, Any]) -> frozenset[str]:
    raw = evidence_sample.get("sample_reasons", ())
    if not isinstance(raw, list):
        raise ValueError("sample_reasons must be an array")
    reasons = tuple(str(value) for value in raw)
    if len(reasons) != len(set(reasons)):
        raise ValueError("sample_reasons cannot contain duplicates")
    return frozenset(reasons)


def _canonical_representative_shot(
    *,
    shot_index: int,
    start_seconds: float,
    end_seconds: float,
    sample_indices: Sequence[int],
    path: Sequence[ReferencePathState],
    samples: Sequence[ReferenceSample],
    raw_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate one detector-owned shot without merging its cut boundaries.

    A shot midpoint is the detector's attribution anchor when present.  Other
    in-shot samples may corroborate it or remain UNKNOWN, but a conflicting
    known source masks the entire shot.  Synthetic/legacy evidence without a
    midpoint remains usable only when every assigned path state agrees.
    """

    indices = tuple(sample_indices)
    states = tuple(path[index] for index in indices)
    known_sources = {
        state.source_id for state in states if state.source_id is not None
    }
    midpoint_reason = f"shot_midpoint:{shot_index}"
    midpoint_indices = tuple(
        index
        for index in indices
        if midpoint_reason in _sample_reason_set(raw_samples[index])
    )
    if len(midpoint_indices) > 1:
        raise ValueError("a canonical shot cannot have two midpoint samples")

    selected_source: str | None = None
    aggregation_reason: str
    if midpoint_indices:
        midpoint_state = path[midpoint_indices[0]]
        if midpoint_state.source_id is None:
            aggregation_reason = "shot_midpoint_failed_conservative_gate"
        elif known_sources != {midpoint_state.source_id}:
            aggregation_reason = "conflicting_known_source_inside_detected_shot"
        else:
            selected_source = midpoint_state.source_id
            aggregation_reason = (
                "detected_shot_midpoint_with_no_conflicting_known_source"
            )
    elif not states:
        aggregation_reason = "detected_shot_has_no_attribution_sample"
    elif None in {state.source_id for state in states}:
        aggregation_reason = "legacy_partition_contains_unknown_path_state"
    elif len(known_sources) != 1:
        aggregation_reason = "legacy_partition_has_no_unique_known_source"
    else:
        selected_source = next(iter(known_sources))
        aggregation_reason = "all_detector_partition_samples_agree"

    selected_states = tuple(
        state
        for state in states
        if selected_source is not None
        and state.source_id == selected_source
        and state.candidate is not None
    )
    if selected_states:
        representative = max(
            selected_states,
            key=lambda state: (
                state.candidate.score if state.candidate is not None else -1.0,
                -state.sample_index,
            ),
        )
    elif midpoint_indices:
        representative = path[midpoint_indices[0]]
    elif states:
        representative = states[len(states) // 2]
    else:
        representative = None

    ranked = (
        _ranked_candidates(samples[representative.sample_index])
        if representative is not None
        else ()
    )
    best = (
        representative.candidate
        if representative is not None and representative.candidate is not None
        else (ranked[0] if ranked else None)
    )
    if selected_source is not None and (
        best is None or best.source_id != selected_source
    ):
        best = next(
            (
                candidate
                for candidate in ranked
                if candidate.source_id == selected_source
            ),
            None,
        )
    runner_up = next(
        (
            candidate
            for candidate in ranked
            if best is None or candidate.source_id != best.source_id
        ),
        None,
    )
    margin = (
        best.score - (runner_up.score if runner_up is not None else 0.0)
        if best is not None
        else None
    )

    contiguous = bool(indices) and all(
        right == left + 1
        for left, right in zip(indices, indices[1:], strict=False)
    )
    dense_ready = bool(indices) and contiguous and all(
        state.source_id == selected_source for state in states
    )
    first_index = indices[0] if indices else None
    last_index = indices[-1] if indices else None
    return {
        "shot_index": shot_index,
        "reference_start_seconds": start_seconds,
        "reference_end_seconds": end_seconds,
        "accepted": selected_source is not None,
        "attributed_video_id": selected_source,
        "decision_reason": aggregation_reason,
        "top_two_margin": margin,
        "best": best.to_dict() if best is not None else None,
        "runner_up": runner_up.to_dict() if runner_up is not None else None,
        "ranked_candidates": [candidate.to_dict() for candidate in ranked],
        "first_sample_index": first_index,
        "last_sample_index": last_index,
        "peak_candidate_score": (
            max(
                state.candidate.score
                for state in selected_states
                if state.candidate is not None
            )
            if selected_states
            else None
        ),
        "canonical_detected_shot": {
            "schema_version": REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
            "detector_shot_index": shot_index,
            "sample_indices": list(indices),
            "midpoint_sample_indices": list(midpoint_indices),
            "known_source_sample_counts": dict(
                sorted(
                    Counter(
                        state.source_id
                        for state in states
                        if state.source_id is not None
                    ).items()
                )
            ),
            "unknown_sample_count": sum(
                state.source_id is None for state in states
            ),
            "aggregation_reason": aggregation_reason,
            "source_run_merge_does_not_define_shot_partition": True,
        },
        "dense_attribution_span": {
            "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
            "first_sample_index": first_index,
            "last_sample_index": last_index,
            "sample_count": len(indices),
            "selected_source_id": selected_source,
            "state": "known" if selected_source is not None else "unknown",
            "source_path_samples_are_contiguous": contiguous,
            "ready": dense_ready,
            "reason": (
                "validated_synchronized_sample_path"
                if dense_ready
                else aggregation_reason
            ),
        },
    }


def _canonical_detected_shot_partition(
    evidence: Mapping[str, Any],
    *,
    path: Sequence[ReferencePathState],
    samples: Sequence[ReferenceSample],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Return detector-owned shots, or ``None`` for legacy evidence JSON."""

    raw_cuts = evidence.get("detected_reference_cuts")
    if raw_cuts is None:
        return None
    if evidence.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS:
        raise ValueError(
            "detected_reference_cuts require trusted reference-evidence provenance"
        )
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported reference-evidence schema version")
    if not isinstance(raw_cuts, list):
        raise ValueError("detected_reference_cuts must be an array")
    raw_samples_value = evidence.get("samples")
    if not isinstance(raw_samples_value, list) or len(raw_samples_value) != len(path):
        raise ValueError("canonical shot partition needs one raw sample per path state")
    raw_samples = tuple(
        value
        for value in raw_samples_value
        if isinstance(value, Mapping)
    )
    if len(raw_samples) != len(raw_samples_value):
        raise ValueError("canonical shot samples must be objects")

    master_start = _finite(
        evidence.get("master_start_seconds"), "master start seconds"
    )
    master_end = _finite(evidence.get("master_end_seconds"), "master end seconds")
    if master_start < 0.0 or master_end <= master_start:
        raise ValueError("canonical master bounds are invalid")

    cuts: list[dict[str, Any]] = []
    previous_frame = -1
    previous_seconds = master_start
    for cut_index, raw_cut in enumerate(raw_cuts):
        if not isinstance(raw_cut, Mapping):
            raise ValueError(f"detected reference cut {cut_index} must be an object")
        frame_index = _nonnegative_int(
            raw_cut.get("frame_index"), "detected cut frame index"
        )
        seconds = _finite(raw_cut.get("master_seconds"), "detected cut time")
        if frame_index <= previous_frame or seconds <= previous_seconds:
            raise ValueError("detected reference cuts must be strictly chronological")
        if not master_start < seconds < master_end:
            raise ValueError("detected reference cut is outside canonical master bounds")
        normalized = deepcopy(dict(raw_cut))
        normalized["frame_index"] = frame_index
        normalized["master_seconds"] = seconds
        cuts.append(normalized)
        previous_frame = frame_index
        previous_seconds = seconds

    cut_times = tuple(float(cut["master_seconds"]) for cut in cuts)
    sample_groups: list[list[int]] = [[] for _ in range(len(cuts) + 1)]
    for sample_index, state in enumerate(path):
        if state.master_seconds < master_start - 1e-9 or state.master_seconds >= master_end + 1e-9:
            raise ValueError("source-path sample is outside canonical master bounds")
        shot_index = bisect_right(cut_times, state.master_seconds)
        if shot_index >= len(sample_groups):
            shot_index = len(sample_groups) - 1
        sample_groups[shot_index].append(sample_index)

    boundaries = (master_start, *cut_times, master_end)
    shots = [
        _canonical_representative_shot(
            shot_index=shot_index,
            start_seconds=boundaries[shot_index],
            end_seconds=boundaries[shot_index + 1],
            sample_indices=sample_groups[shot_index],
            path=path,
            samples=samples,
            raw_samples=raw_samples,
        )
        for shot_index in range(len(sample_groups))
    ]

    detected_boundaries: list[dict[str, Any]] = []
    for boundary_index, cut in enumerate(cuts):
        cut_frame = int(cut["frame_index"])
        sample_indices_by_offset: dict[int, int] = {}
        for sample_index, raw_sample in enumerate(raw_samples):
            reasons = _sample_reason_set(raw_sample)
            for offset in (-2, -1, 0, 1, 2):
                expected = f"hard_cut_boundary:{cut_frame}:{offset:+d}"
                if expected not in reasons:
                    continue
                if offset in sample_indices_by_offset:
                    raise ValueError(
                        "detected boundary has duplicate offset evidence"
                    )
                sample_indices_by_offset[offset] = sample_index
        complete = set(sample_indices_by_offset) == {-2, -1, 0, 1, 2}
        ordered_indices = tuple(
            sample_indices_by_offset[offset]
            for offset in (-2, -1, 0, 1, 2)
            if offset in sample_indices_by_offset
        )
        state_ids = tuple(path[index].source_id for index in ordered_indices)
        initial_classification = "loss_masked"
        initial_reason = "missing_boundary_window_samples"
        old_source: str | None = None
        new_source: str | None = None
        if complete and any(source is None for source in state_ids):
            initial_reason = "unknown_or_failed_candidate_gate"
        elif complete and len(set(state_ids)) == 1:
            old_source = new_source = state_ids[0]
            initial_classification = "native_source_camera_cut_hard_negative"
            initial_reason = "same_known_source_across_complete_boundary_window"
        elif complete:
            changes = tuple(
                index
                for index in range(1, len(state_ids))
                if state_ids[index] != state_ids[index - 1]
            )
            if (
                len(changes) == 1
                and changes[0] >= 2
                and len(state_ids) - changes[0] >= 2
            ):
                split = changes[0]
                old_source = state_ids[0]
                new_source = state_ids[-1]
                initial_reason = "stable_cross_source_window_requires_refinement"
            else:
                initial_reason = "ambiguous_or_unstable_source_change"
        detected_boundaries.append(
            {
                "schema_version": REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION,
                "boundary_index": boundary_index,
                "incoming_shot_index": boundary_index + 1,
                "detector_cut": deepcopy(cut),
                "boundary_sample_indices_by_offset": {
                    str(offset): sample_indices_by_offset[offset]
                    for offset in sorted(sample_indices_by_offset)
                },
                "boundary_path_source_ids": list(state_ids),
                "old_source_id": old_source,
                "new_source_id": new_source,
                "supervision_classification": initial_classification,
                "classification_reason": initial_reason,
                "cut_target": (
                    0.0
                    if initial_classification
                    == "native_source_camera_cut_hard_negative"
                    else None
                ),
                "loss_masked": initial_classification == "loss_masked",
                "dense_attribution_ready": (
                    initial_classification
                    == "native_source_camera_cut_hard_negative"
                ),
            }
        )

    partition_identity = {
        "master_start_seconds": master_start,
        "master_end_seconds": master_end,
        "detected_reference_cuts": cuts,
        "sample_assignment": sample_groups,
    }
    schema = {
        "schema_version": REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
        "partition_basis": "reference_hard_cut_detector",
        "reference_evidence_analysis": REFERENCE_EVIDENCE_ANALYSIS_BASIS,
        "detected_cut_count": len(cuts),
        "canonical_shot_count": len(shots),
        "source_runs_are_attribution_only_not_shot_partitions": True,
        "same_source_adjacent_detector_shots_are_never_merged": True,
        "unknown_or_failed_boundary_gate_is_loss_masked": True,
        "partition_content_sha256": _canonical_json_sha256(partition_identity),
        "hash_basis": (
            "sha256_canonical_json_of_master_bounds_detected_cuts_and_sample_assignment"
        ),
    }
    return shots, detected_boundaries, schema


def _canonical_attribution_hash_identity(
    *,
    partition_content_sha256: str,
    path_source_ids: Sequence[str | None],
    shots: Sequence[Mapping[str, Any]],
    detected_boundaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "partition_content_sha256": partition_content_sha256,
        "source_path_source_ids": list(path_source_ids),
        "shot_aggregates": [
            {
                "shot_index": shot.get("shot_index"),
                "accepted": shot.get("accepted"),
                "attributed_video_id": shot.get("attributed_video_id"),
                "canonical_detected_shot": deepcopy(
                    shot.get("canonical_detected_shot")
                ),
                "dense_attribution_span": deepcopy(
                    shot.get("dense_attribution_span")
                ),
                "boundary_refinement": deepcopy(
                    shot.get("boundary_refinement")
                ),
            }
            for shot in shots
        ],
        "detected_boundary_labels": [
            {
                "boundary_index": boundary.get("boundary_index"),
                "incoming_shot_index": boundary.get("incoming_shot_index"),
                "old_source_id": boundary.get("old_source_id"),
                "new_source_id": boundary.get("new_source_id"),
                "supervision_classification": boundary.get(
                    "supervision_classification"
                ),
                "classification_reason": boundary.get(
                    "classification_reason"
                ),
                "cut_target": boundary.get("cut_target"),
                "loss_masked": boundary.get("loss_masked"),
                "dense_attribution_ready": boundary.get(
                    "dense_attribution_ready"
                ),
                "boundary_refinement": deepcopy(
                    boundary.get("boundary_refinement")
                ),
            }
            for boundary in detected_boundaries
        ],
    }


def _representative_shot(
    run_index: int,
    run: ReferenceSourceRun,
    path: Sequence[ReferencePathState],
    samples: Sequence[ReferenceSample],
) -> dict[str, Any]:
    run_states = tuple(
        path[index]
        for index in range(run.first_sample_index, run.last_sample_index + 1)
    )
    if run.source_id is not None:
        representative = max(
            (state for state in run_states if state.candidate is not None),
            key=lambda state: (
                state.candidate.score if state.candidate is not None else -1.0,
                -state.sample_index,
            ),
        )
    else:
        representative = run_states[len(run_states) // 2]
    ranked = _ranked_candidates(samples[representative.sample_index])
    if representative.candidate is not None:
        best = representative.candidate
    else:
        best = ranked[0] if ranked else None
    runner_up = next(
        (
            candidate
            for candidate in ranked
            if best is None or candidate.source_id != best.source_id
        ),
        None,
    )
    margin = (
        best.score - (runner_up.score if runner_up is not None else 0.0)
        if best is not None
        else None
    )
    return {
        "shot_index": run_index,
        "reference_start_seconds": run.start_master_seconds,
        "reference_end_seconds": run.end_master_seconds,
        "accepted": run.source_id is not None,
        "attributed_video_id": run.source_id,
        "decision_reason": representative.decision_reason,
        "top_two_margin": margin,
        "best": best.to_dict() if best is not None else None,
        "runner_up": runner_up.to_dict() if runner_up is not None else None,
        "ranked_candidates": [candidate.to_dict() for candidate in ranked],
        "first_sample_index": run.first_sample_index,
        "last_sample_index": run.last_sample_index,
        "peak_candidate_score": run.peak_candidate_score,
        "dense_attribution_span": {
            "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
            "first_sample_index": run.first_sample_index,
            "last_sample_index": run.last_sample_index,
            "sample_count": run.last_sample_index - run.first_sample_index + 1,
            "selected_source_id": run.source_id,
            "state": "known" if run.source_id is not None else "unknown",
            "source_path_samples_are_contiguous": True,
        },
    }


def resolve_reference_attribution_payload(
    evidence: Mapping[str, Any],
    *,
    config: ReferenceAttributionConfig | None = None,
) -> dict[str, Any]:
    """Resolve dense synchronized match evidence into dataset-ready shots.

    ``boundary_requests`` are optional.  Each request can include a
    ``to_run_index`` to attach its result to that incoming source-run boundary;
    unresolved results remain diagnostic and never become hard-cut labels.
    """

    settings = config or ReferenceAttributionConfig()
    raw_samples = evidence.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("reference evidence needs a non-empty samples array")
    samples: list[ReferenceSample] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"reference sample {index} must be an object")
        samples.append(_sample_from_mapping(raw_sample))

    path = solve_reference_source_path(samples, config=settings)
    master_start = evidence.get("master_start_seconds")
    master_end = evidence.get("master_end_seconds")
    run_master_end = master_end
    clock_policy = evidence.get("clock_policy", {})
    if not isinstance(clock_policy, Mapping):
        raise ValueError("reference clock_policy must be an object")
    sample_end = clock_policy.get("attribution_sample_master_end_seconds")
    if sample_end is not None:
        sample_end = _finite(sample_end, "attribution sample master end")
        run_master_end = (
            min(sample_end, _finite(master_end, "master end"))
            if master_end is not None else sample_end
        )
    # Detector ownership can include an unsampled outro. It must not make the
    # final observed source run extend beyond the caller's observation window.
    # Keep the detector partition unchanged; these are attribution-only runs.
    runs = collapse_reference_source_runs(
        path,
        master_start_seconds=master_start,
        master_end_seconds=run_master_end,
    )
    canonical_partition = _canonical_detected_shot_partition(
        evidence,
        path=path,
        samples=samples,
    )
    if canonical_partition is None:
        shots = [
            _representative_shot(index, run, path, samples)
            for index, run in enumerate(runs)
        ]
        detected_boundaries: list[dict[str, Any]] = []
        canonical_shot_schema: dict[str, Any] | None = None
    else:
        shots, detected_boundaries, canonical_shot_schema = canonical_partition

    raw_requests = evidence.get("boundary_requests", [])
    if not isinstance(raw_requests, list):
        raise ValueError("boundary_requests must be an array")
    refinements: list[dict[str, Any]] = []
    attached_runs: set[int] = set()
    for request_index, raw_request in enumerate(raw_requests):
        if not isinstance(raw_request, Mapping):
            raise ValueError(f"boundary request {request_index} must be an object")
        raw_frames = raw_request.get("frame_evidence")
        if not isinstance(raw_frames, list):
            raise ValueError(
                f"boundary request {request_index} needs frame_evidence"
            )
        frames: list[BoundaryFrameEvidence] = []
        for frame_index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, Mapping):
                raise ValueError(
                    "boundary frame evidence "
                    f"{request_index}:{frame_index} must be an object"
                )
            frames.append(_boundary_evidence_from_mapping(raw_frame))
        refinement = refine_reference_source_boundary(
            old_source_id=str(raw_request.get("old_source_id") or ""),
            new_source_id=str(raw_request.get("new_source_id") or ""),
            nominal_boundary_seconds=raw_request.get(
                "nominal_boundary_seconds"
            ),
            nominal_frame_index=raw_request.get("nominal_frame_index"),
            frame_evidence=frames,
            config=settings,
        )
        serialized = refinement.to_dict()
        to_run_index = raw_request.get("to_run_index")
        if to_run_index is not None:
            if (
                isinstance(to_run_index, bool)
                or not isinstance(to_run_index, int)
                or not 1 <= to_run_index < len(runs)
            ):
                raise ValueError("to_run_index must identify an incoming run")
            if to_run_index in attached_runs:
                raise ValueError("a run boundary cannot have two refinements")
            previous_run = runs[to_run_index - 1]
            incoming_run = runs[to_run_index]
            if (
                previous_run.source_id != refinement.old_source_id
                or incoming_run.source_id != refinement.new_source_id
            ):
                raise ValueError(
                    "boundary request sources do not match the selected runs"
                )
            run_boundary = incoming_run.start_master_seconds
            if (
                abs(refinement.nominal_boundary_seconds - run_boundary)
                > settings.refinement_radius_seconds + 1e-9
            ):
                raise ValueError(
                    "boundary request nominal time does not match the incoming run"
                )
            if (
                refinement.selected_boundary_seconds is not None
                and abs(
                    refinement.selected_boundary_seconds - run_boundary
                )
                > settings.refinement_radius_seconds + 1e-9
            ):
                raise ValueError(
                    "refined boundary time does not match the incoming run"
                )
            serialized["to_run_index"] = to_run_index
            if canonical_shot_schema is None:
                shots[to_run_index]["boundary_refinement"] = dict(serialized)
            else:
                matching_boundaries = [
                    boundary
                    for boundary in detected_boundaries
                    if (
                        boundary["detector_cut"]["frame_index"]
                        == refinement.nominal_frame_index
                        and abs(
                            float(boundary["detector_cut"]["master_seconds"])
                            - refinement.nominal_boundary_seconds
                        )
                        <= 1e-9
                    )
                ]
                if len(matching_boundaries) != 1:
                    raise ValueError(
                        "boundary request does not identify one detected reference cut"
                    )
                detected_boundary = matching_boundaries[0]
                to_shot_index = int(detected_boundary["incoming_shot_index"])
                declared_shot_index = raw_request.get("to_shot_index")
                if declared_shot_index is not None and (
                    isinstance(declared_shot_index, bool)
                    or declared_shot_index != to_shot_index
                ):
                    raise ValueError(
                        "boundary request to_shot_index does not match detector partition"
                    )
                serialized["to_shot_index"] = to_shot_index
                detected_boundary["boundary_refinement"] = dict(serialized)
                outgoing_source = shots[to_shot_index - 1].get(
                    "attributed_video_id"
                )
                incoming_source = shots[to_shot_index].get(
                    "attributed_video_id"
                )
                window_old = detected_boundary.get("old_source_id")
                window_new = detected_boundary.get("new_source_id")
                if (
                    refinement.status == "resolved_hard_cut"
                    and outgoing_source == refinement.old_source_id
                    and incoming_source == refinement.new_source_id
                    and window_old == refinement.old_source_id
                    and window_new == refinement.new_source_id
                ):
                    shots[to_shot_index]["boundary_refinement"] = dict(
                        serialized
                    )
                    detected_boundary["supervision_classification"] = (
                        "proven_cross_source_editorial_switch_positive"
                    )
                    detected_boundary["classification_reason"] = (
                        "stable_cross_source_window_and_resolved_frame_boundary"
                    )
                    detected_boundary["cut_target"] = 1.0
                    detected_boundary["loss_masked"] = False
                    detected_boundary["dense_attribution_ready"] = True
                else:
                    detected_boundary["supervision_classification"] = (
                        "loss_masked"
                    )
                    detected_boundary["classification_reason"] = (
                        "resolved_run_boundary_disagrees_with_canonical_shot_aggregate"
                    )
                    detected_boundary["cut_target"] = None
                    detected_boundary["loss_masked"] = True
                    detected_boundary["dense_attribution_ready"] = False
            attached_runs.add(to_run_index)
        refinements.append(serialized)

    if canonical_shot_schema is not None:
        for detected_boundary in detected_boundaries:
            if (
                detected_boundary.get("supervision_classification")
                != "native_source_camera_cut_hard_negative"
            ):
                continue
            incoming_index = int(detected_boundary["incoming_shot_index"])
            old_source = detected_boundary.get("old_source_id")
            if (
                old_source is None
                or detected_boundary.get("new_source_id") != old_source
                or shots[incoming_index - 1].get("attributed_video_id")
                != old_source
                or shots[incoming_index].get("attributed_video_id")
                != old_source
            ):
                detected_boundary["supervision_classification"] = "loss_masked"
                detected_boundary["classification_reason"] = (
                    "boundary_window_disagrees_with_canonical_shot_aggregate"
                )
                detected_boundary["cut_target"] = None
                detected_boundary["loss_masked"] = True
                detected_boundary["dense_attribution_ready"] = False

        attribution_hash_identity = _canonical_attribution_hash_identity(
            partition_content_sha256=canonical_shot_schema[
                "partition_content_sha256"
            ],
            path_source_ids=[state.source_id for state in path],
            shots=shots,
            detected_boundaries=detected_boundaries,
        )
        canonical_shot_schema["attribution_content_sha256"] = (
            _canonical_json_sha256(attribution_hash_identity)
        )
        canonical_shot_schema["attribution_hash_basis"] = (
            "sha256_canonical_json_of_partition_path_shot_aggregates_and_boundary_labels"
        )

    output = {
        "schema_version": 1,
        "analysis": REFERENCE_ATTRIBUTION_BASIS,
        "dense_attribution_schema": {
            "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
            "temporal_granularity": "synchronized_sample_path",
            "sample_rate_hz": settings.sample_rate_hz,
            "sample_level_source_path_available": True,
            "shot_spans_reference_contiguous_source_path_samples": True,
            "frame_dense_boundary_proof_required_for_hard_cut": True,
            "native_source_camera_cut_is_not_editorial_source_switch": True,
        },
        "reference": evidence.get("reference"),
        "sources": evidence.get("sources", []),
        "decision_policy": settings.to_dict(),
        "source_path": [
            {
                **state.to_dict(),
                "ranked_candidates": [
                    candidate.to_dict()
                    for candidate in _ranked_candidates(samples[state.sample_index])
                ],
            }
            for state in path
        ],
        "source_runs": [run.to_dict() for run in runs],
        "source_run_coverage": {
            "contract": "caller_bounded_source_attribution_runs_v1",
            "requested_sample_master_end_seconds": sample_end,
            "effective_run_master_end_seconds": runs[-1].end_master_seconds,
            "canonical_detector_partition_unchanged": True,
            "unobserved_outro_attribution_authorized": False,
        },
        "boundary_refinements": refinements,
        "shots": shots,
    }
    if canonical_shot_schema is not None:
        output.update(
            {
                "canonical_shot_schema": canonical_shot_schema,
                "detected_reference_cuts": deepcopy(
                    evidence.get("detected_reference_cuts", [])
                ),
                "detected_boundaries": detected_boundaries,
                "reference_evidence_provenance": {
                    "schema_version": evidence.get("schema_version"),
                    "analysis": evidence.get("analysis"),
                    "provenance": deepcopy(evidence.get("provenance")),
                    "reference_video": deepcopy(evidence.get("reference_video")),
                    "partition_content_sha256": canonical_shot_schema[
                        "partition_content_sha256"
                    ],
                    "attribution_content_sha256": canonical_shot_schema[
                        "attribution_content_sha256"
                    ],
                },
            }
        )
    return output


def resolve_reference_attribution_file(
    evidence_path: Path,
    output_path: Path,
    *,
    config: ReferenceAttributionConfig | None = None,
) -> Path:
    raw = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("reference evidence JSON root must be an object")
    payload = resolve_reference_attribution_payload(raw, config=config)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination.resolve()


__all__ = [
    "REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION",
    "REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION",
    "REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION",
    "REFERENCE_EVIDENCE_ANALYSIS_BASIS",
    "REFERENCE_ATTRIBUTION_BASIS",
    "UNKNOWN_SOURCE_ID",
    "BoundaryFrameEvidence",
    "ReferenceAttributionConfig",
    "ReferenceBoundaryRefinement",
    "ReferenceCandidateScore",
    "ReferencePathState",
    "ReferenceSample",
    "ReferenceSourceRun",
    "collapse_reference_source_runs",
    "refine_reference_source_boundary",
    "resolve_reference_attribution_file",
    "resolve_reference_attribution_payload",
    "solve_reference_source_path",
]
