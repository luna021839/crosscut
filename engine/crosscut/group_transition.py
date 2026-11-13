"""Safety-gated group-transition planning.

The design is an independent implementation informed by the *PopStage*
(SIGGRAPH Asia 2022, DOI 10.1145/3550454.3555467) and *AutoStageMix*
(Applied Sciences 2025, DOI 10.3390/app15137613) papers.  No paper or
third-party source code is copied here.  The important editing invariant is
that a dissolve may only overlap audio-synchronized pictures at the same
canonical song time; otherwise the default remains a hard cut.

Tier policy:

* A: three safe future samples on both sources, aligned face crops, compatible
  head geometry, conservative SFace appearance compatibility, actual future
  body continuity, and a full outgoing source handle -> short face dissolve.
* B: both face or stable aggregate-silhouette crops are safe but the stricter
  face-only dissolve proof is incomplete -> subject-aligned hard cut.
* C: the selected rhythm/motion/pose timeline remains a normal hard cut.
* D: the optimizer already held the same source, so there is no visible switch.

Missing evidence can never enable a blend.  This intentionally prefers a clean
hard cut to dissolve ghosting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .face_identity import SFACE_EMBEDDING_SIZE, cosine_similarity
from .face_match import FaceMatchPlan, FaceMatchSegment
from .render import PanCropTrack, RenderTransition, TransitionKind
from .stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from .timeline import face_geometry_minimax_mismatch
from .visual_score import VisualTimelineChoice, VisualTimelinePlan


_FUTURE_SAMPLE_COUNT = 3


@dataclass(frozen=True, slots=True)
class GroupTransitionConfig:
    dissolve_frame_count: int = 11
    maximum_projected_face_mismatch: float = 0.22
    maximum_zoom_delta: float = 0.13
    face_box_crop_margin: float = 0.018
    maximum_reliable_pose_mismatch: float = 0.34
    maximum_reliable_silhouette_mismatch: float = 0.38
    minimum_pose_confidence: float = 0.55
    minimum_silhouette_confidence: float = 0.55
    # OpenCV Zoo's general SFace cosine example threshold is 0.363. A short
    # dissolve is less tolerant than ordinary verification, so require a
    # conservative margin at all three synchronized pairs plus stable tracks.
    minimum_sface_cross_similarity: float = 0.48
    minimum_sface_track_stability: float = 0.55

    def __post_init__(self) -> None:
        if (
            isinstance(self.dissolve_frame_count, bool)
            or not isinstance(self.dissolve_frame_count, int)
            or self.dissolve_frame_count < 3
        ):
            raise ValueError("dissolve_frame_count must be an integer of at least 3")
        if not 0.0 < self.maximum_projected_face_mismatch < 1.0:
            raise ValueError("maximum projected face mismatch must be normalized")
        if not 0.0 <= self.maximum_zoom_delta <= 0.5:
            raise ValueError("maximum zoom delta must be conservative")
        if not 0.0 <= self.face_box_crop_margin <= 0.1:
            raise ValueError("face box crop margin must be normalized")
        for value in (
            self.maximum_reliable_pose_mismatch,
            self.maximum_reliable_silhouette_mismatch,
            self.minimum_pose_confidence,
            self.minimum_silhouette_confidence,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("body transition thresholds must be normalized")
        for value in (
            self.minimum_sface_cross_similarity,
            self.minimum_sface_track_stability,
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("SFace similarity thresholds must be normalized")


@dataclass(frozen=True, slots=True)
class GroupTransitionPlan:
    transitions: tuple[RenderTransition, ...]
    analysis_basis: str = (
        "tiered_future_face_head_sface_appearance_body_continuity_v4"
    )

    @property
    def dissolve_count(self) -> int:
        return sum(
            transition.kind is TransitionKind.FACE_DISSOLVE
            for transition in self.transitions
        )

    @property
    def introduced_hard_cut_count(self) -> int:
        """Count visible hard-cut boundaries introduced by the edit."""

        return sum(
            transition.kind is TransitionKind.HARD_CUT
            and transition.tier != "D"
            for transition in self.transitions
        )

    @property
    def same_source_hold_count(self) -> int:
        """Count Tier-D boundaries that continue the same visible source shot."""

        return sum(transition.tier == "D" for transition in self.transitions)

    @property
    def hard_cut_boundary_count(self) -> int:
        """Count all hard-coded segment boundaries, including invisible holds."""

        return self.introduced_hard_cut_count + self.same_source_hold_count

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "transition_authority": StageEditTransitionAuthority(
                mode=RENDERER_OWNED_TRANSITION_AUTHORITY
            ).to_dict(),
            "count_basis": "transition_boundaries",
            "policy": {
                "A": "safe_future_face_head_sface_appearance_body_micro_dissolve",
                "B": "subject_aligned_hard_cut",
                "C": "rhythm_motion_pose_matched_hard_cut",
                "D": "same_source_hold_no_visible_switch",
                "missing_evidence": "never_blend",
                "canonical_overlap": "outgoing_handle_at_incoming_song_time",
            },
            "dissolve_count": self.dissolve_count,
            "introduced_hard_cut_count": self.introduced_hard_cut_count,
            "same_source_hold_count": self.same_source_hold_count,
            "hard_cut_boundary_count": self.hard_cut_boundary_count,
            "transitions": [item.to_dict(fps) for item in self.transitions],
        }


def _hard(
    tier: str,
    reason: str,
    mismatch: float | None = None,
    *,
    appearance_similarity: float | None = None,
    future_pose_mismatch: float | None = None,
    future_silhouette_mismatch: float | None = None,
) -> RenderTransition:
    return RenderTransition(
        kind=TransitionKind.HARD_CUT,
        frame_count=0,
        tier=tier,
        reason=reason,
        face_mismatch=mismatch,
        appearance_min_cosine_similarity=appearance_similarity,
        future_pose_max_mismatch=future_pose_mismatch,
        future_silhouette_max_mismatch=future_silhouette_mismatch,
    )


def _fixed_crop_at(
    track: PanCropTrack,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    """Evaluate the renderer's fixed-size, piecewise-linear crop path."""

    width, height = track.fixed_crop_size
    positions = track.fixed_crop_positions()
    if time_seconds <= positions[0][0]:
        return (positions[0][1], positions[0][2], width, height)
    for left, right in zip(positions, positions[1:]):
        if time_seconds <= right[0]:
            alpha = (time_seconds - left[0]) / (right[0] - left[0])
            x = left[1] + (right[1] - left[1]) * alpha
            y = left[2] + (right[2] - left[2]) * alpha
            return (x, y, width, height)
    # The FFmpeg expression holds its final crop position during an outgoing
    # dissolve handle; mirror that exact behavior in the safety analysis.
    return (positions[-1][1], positions[-1][2], width, height)


def _project_geometry(
    geometry: tuple[float, ...],
    crop: tuple[float, float, float, float],
) -> tuple[float, ...]:
    x, y, width, height = crop
    return (
        (geometry[0] - x) / width,
        (geometry[1] - y) / height,
        (geometry[2] - x) / width,
        (geometry[3] - y) / height,
        geometry[4] / height,
        *geometry[5:],
    )


def _box_is_contained(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    margin: float,
) -> bool:
    crop_x, crop_y, crop_width, crop_height = crop
    box_x, box_y, box_width, box_height = box
    inset_x = crop_width * margin
    inset_y = crop_height * margin
    return (
        box_x >= crop_x + inset_x
        and box_y >= crop_y + inset_y
        and box_x + box_width <= crop_x + crop_width - inset_x
        and box_y + box_height <= crop_y + crop_height - inset_y
    )


def _window(choice: VisualTimelineChoice, name: str) -> tuple[tuple[float, ...], ...]:
    value = getattr(choice, name, ())
    return tuple(value) if value else ()


def _box_window(
    choice: VisualTimelineChoice,
    name: str,
) -> tuple[tuple[float, float, float, float], ...]:
    value = getattr(choice, name, ())
    return tuple(value) if value else ()


def _valid_future_evidence(
    geometries: tuple[tuple[float, ...], ...],
    boxes: tuple[tuple[float, float, float, float], ...],
) -> bool:
    if len(geometries) != _FUTURE_SAMPLE_COUNT or len(boxes) != _FUTURE_SAMPLE_COUNT:
        return False
    if any(
        len(geometry) != 10
        or geometry[4] <= 0.0
        or any(not math.isfinite(float(value)) for value in geometry)
        for geometry in geometries
    ):
        return False
    return not any(
        len(box) != 4
        or any(not math.isfinite(float(value)) for value in box)
        or box[0] < 0.0
        or box[1] < 0.0
        or box[2] <= 0.0
        or box[3] <= 0.0
        or box[0] + box[2] > 1.000001
        or box[1] + box[3] > 1.000001
        for box in boxes
    )


def _valid_embedding_window(
    window: tuple[tuple[float, ...], ...],
    *,
    expected_size: int | None = None,
) -> bool:
    if len(window) != _FUTURE_SAMPLE_COUNT:
        return False
    sizes = {len(embedding) for embedding in window}
    if len(sizes) != 1 or 0 in sizes:
        return False
    if expected_size is not None and sizes != {expected_size}:
        return False
    return all(
        all(math.isfinite(float(value)) for value in embedding)
        for embedding in window
    )


def _minimum_pair_cosine(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> float | None:
    values = tuple(
        cosine_similarity(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )
    if any(value is None for value in values):
        return None
    return min(float(value) for value in values if value is not None)


def _minimum_track_cosine(
    window: tuple[tuple[float, ...], ...],
) -> float | None:
    values = tuple(
        cosine_similarity(left, right)
        for left, right in zip(window, window[1:])
    )
    if any(value is None for value in values):
        return None
    return min(float(value) for value in values if value is not None)


def _maximum_pair_rms_mismatch(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> float | None:
    if not _valid_embedding_window(left) or not _valid_embedding_window(right):
        return None
    if len(left[0]) != len(right[0]):
        return None
    mismatches = tuple(
        min(
            1.0,
            math.sqrt(
                sum((float(a) - float(b)) ** 2 for a, b in zip(x, y, strict=True))
                / len(x)
            ),
        )
        for x, y in zip(left, right, strict=True)
    )
    return max(mismatches)


def _reliable_body_window(
    choice: VisualTimelineChoice,
    *,
    direction: str,
    kind: str,
    minimum_confidence: float,
) -> tuple[tuple[float, ...], ...] | None:
    embeddings = _window(
        choice, f"dissolve_{direction}_{kind}_embedding_window"
    )
    backends = tuple(
        getattr(choice, f"dissolve_{direction}_{kind}_backend_window", ())
    )
    confidences = tuple(
        getattr(choice, f"dissolve_{direction}_{kind}_confidence_window", ())
    )
    if (
        not _valid_embedding_window(embeddings)
        or len(backends) != _FUTURE_SAMPLE_COUNT
        or len(confidences) != _FUTURE_SAMPLE_COUNT
        or any(
            not math.isfinite(float(confidence))
            or float(confidence) < minimum_confidence
            for confidence in confidences
        )
    ):
        return None
    if kind == "pose":
        reliable_backend = all(
            backend.startswith("opencv_zoo_mediapipe_pose")
            for backend in backends
        )
    else:
        reliable_backend = all(
            backend.startswith("opencv_zoo_pphumanseg")
            for backend in backends
        )
    return embeddings if reliable_backend else None


def _frame_counts(
    choices: Sequence[VisualTimelineChoice],
    fps: float,
) -> tuple[int, ...]:
    counts: list[int] = []
    cumulative = 0.0
    previous = 0
    for choice in choices:
        cumulative += choice.duration_seconds
        boundary = int(round(cumulative * fps))
        counts.append(boundary - previous)
        previous = boundary
    return tuple(counts)


def _face_matches_by_segment(
    plan: FaceMatchPlan | None,
) -> dict[int, FaceMatchSegment]:
    if plan is None:
        return {}
    result = {segment.segment_index: segment for segment in plan.segments}
    if len(result) != len(plan.segments):
        raise ValueError("face-match plan contains duplicate segment indices")
    return result


def build_group_transition_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    face_match_plan: FaceMatchPlan | None,
    source_durations_seconds: Sequence[float],
    fps: float = 29.97,
    config: GroupTransitionConfig | None = None,
) -> GroupTransitionPlan:
    """Choose a transition tier for every selected group-timeline boundary."""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    choices = visual_timeline.choices
    if not choices:
        raise ValueError("visual timeline cannot be empty")
    if not source_durations_seconds or any(
        not math.isfinite(value) or value <= 0.0
        for value in source_durations_seconds
    ):
        raise ValueError("source durations must be positive and finite")
    if any(
        choice.source_index < 0
        or choice.source_index >= len(source_durations_seconds)
        for choice in choices
    ):
        raise ValueError("visual timeline references an unknown source duration")

    settings = config or GroupTransitionConfig()
    handle_seconds = settings.dissolve_frame_count / fps
    counts = _frame_counts(choices, fps)
    matches = _face_matches_by_segment(face_match_plan)
    transitions: list[RenderTransition] = []

    for index, (outgoing, incoming) in enumerate(zip(choices, choices[1:])):
        if outgoing.source_index == incoming.source_index:
            if outgoing.shot_id == incoming.shot_id:
                transitions.append(
                    _hard("D", "same_source_hold_no_visible_switch")
                )
            else:
                transitions.append(
                    _hard("C", "same_source_internal_camera_cut")
                )
            continue

        outgoing_match = matches.get(outgoing.segment_index)
        incoming_match = matches.get(incoming.segment_index)
        both_aligned = bool(
            outgoing_match is not None
            and incoming_match is not None
            and outgoing_match.applied
            and incoming_match.applied
            and outgoing_match.pan_crop is not None
            and incoming_match.pan_crop is not None
        )
        hard_tier = "B" if both_aligned else "C"

        if outgoing.face_match_candidate_failure == "internal_camera_cut" or (
            incoming.face_match_candidate_failure == "internal_camera_cut"
        ):
            transitions.append(_hard(hard_tier, "internal_camera_cut"))
            continue
        if not both_aligned:
            transitions.append(
                _hard(
                    "C",
                    "rhythm_motion_pose_hard_cut_face_alignment_unavailable",
                )
            )
            continue
        assert outgoing_match is not None and outgoing_match.pan_crop is not None
        assert incoming_match is not None and incoming_match.pan_crop is not None

        # Silhouette framing is valuable for clean formation-matched hard cuts,
        # but it cannot prove that two visible faces will dissolve without
        # ghosting. Tier A is deliberately face-aligned on both sides.
        if (
            outgoing_match.alignment_subject != "face"
            or incoming_match.alignment_subject != "face"
        ):
            transitions.append(_hard("B", "silhouette_aligned_hard_cut_only"))
            continue

        if counts[index + 1] < settings.dissolve_frame_count:
            transitions.append(_hard("B", "incoming_segment_shorter_than_handle"))
            continue
        if (
            outgoing.source_local_end
            + handle_seconds * outgoing.source_playback_rate
            > source_durations_seconds[outgoing.source_index] + 1e-6
        ):
            transitions.append(_hard("B", "outgoing_source_handle_unavailable"))
            continue
        if outgoing_match.applied_zoom is None or incoming_match.applied_zoom is None:
            transitions.append(_hard("B", "aligned_crop_zoom_unavailable"))
            continue
        if (
            abs(outgoing_match.applied_zoom - incoming_match.applied_zoom)
            > settings.maximum_zoom_delta
        ):
            transitions.append(_hard("B", "aligned_crop_zoom_mismatch"))
            continue

        outgoing_geometry = _window(
            outgoing, "dissolve_exit_face_geometry_window"
        )
        incoming_geometry = _window(
            incoming, "dissolve_entry_face_geometry_window"
        )
        outgoing_boxes = _box_window(outgoing, "dissolve_exit_face_box_window")
        incoming_boxes = _box_window(incoming, "dissolve_entry_face_box_window")
        if not _valid_future_evidence(outgoing_geometry, outgoing_boxes) or not (
            _valid_future_evidence(incoming_geometry, incoming_boxes)
        ):
            transitions.append(_hard("B", "future_face_evidence_missing_or_ambiguous"))
            continue

        outgoing_identity = _window(
            outgoing, "dissolve_exit_face_identity_window"
        )
        incoming_identity = _window(
            incoming, "dissolve_entry_face_identity_window"
        )
        if not _valid_embedding_window(
            outgoing_identity,
            expected_size=SFACE_EMBEDDING_SIZE,
        ) or not _valid_embedding_window(
            incoming_identity,
            expected_size=SFACE_EMBEDDING_SIZE,
        ):
            transitions.append(
                _hard("B", "future_face_appearance_evidence_unavailable")
            )
            continue
        outgoing_track_similarity = _minimum_track_cosine(outgoing_identity)
        incoming_track_similarity = _minimum_track_cosine(incoming_identity)
        available_track_similarities = tuple(
            value
            for value in (outgoing_track_similarity, incoming_track_similarity)
            if value is not None
        )
        minimum_track_similarity = (
            min(available_track_similarities)
            if available_track_similarities
            else None
        )
        if (
            outgoing_track_similarity is None
            or incoming_track_similarity is None
            or minimum_track_similarity is None
            or minimum_track_similarity < settings.minimum_sface_track_stability
        ):
            transitions.append(
                _hard(
                    "B",
                    "future_face_appearance_track_unstable",
                    appearance_similarity=minimum_track_similarity,
                )
            )
            continue
        appearance_similarity = _minimum_pair_cosine(
            outgoing_identity,
            incoming_identity,
        )
        if (
            appearance_similarity is None
            or appearance_similarity < settings.minimum_sface_cross_similarity
        ):
            transitions.append(
                _hard(
                    "B",
                    "future_face_appearance_mismatch",
                    appearance_similarity=appearance_similarity,
                )
            )
            continue

        # Body evidence must refer to the same future cut/mid/end frames as the
        # pictures that will overlap. A reliable pose or aggregate silhouette
        # window is sufficient; absent evidence safely stays a Tier-B hard cut.
        outgoing_pose = _reliable_body_window(
            outgoing,
            direction="exit",
            kind="pose",
            minimum_confidence=settings.minimum_pose_confidence,
        )
        incoming_pose = _reliable_body_window(
            incoming,
            direction="entry",
            kind="pose",
            minimum_confidence=settings.minimum_pose_confidence,
        )
        future_pose_mismatch = (
            _maximum_pair_rms_mismatch(outgoing_pose, incoming_pose)
            if outgoing_pose is not None and incoming_pose is not None
            else None
        )
        outgoing_silhouette = _reliable_body_window(
            outgoing,
            direction="exit",
            kind="silhouette",
            minimum_confidence=settings.minimum_silhouette_confidence,
        )
        incoming_silhouette = _reliable_body_window(
            incoming,
            direction="entry",
            kind="silhouette",
            minimum_confidence=settings.minimum_silhouette_confidence,
        )
        future_silhouette_mismatch = (
            _maximum_pair_rms_mismatch(outgoing_silhouette, incoming_silhouette)
            if outgoing_silhouette is not None
            and incoming_silhouette is not None
            else None
        )
        if future_pose_mismatch is None and future_silhouette_mismatch is None:
            transitions.append(
                _hard(
                    "B",
                    "future_body_evidence_unavailable",
                    appearance_similarity=appearance_similarity,
                )
            )
            continue
        if (
            future_pose_mismatch is not None
            and future_pose_mismatch > settings.maximum_reliable_pose_mismatch
        ):
            transitions.append(
                _hard(
                    "B",
                    "future_body_pose_mismatch",
                    appearance_similarity=appearance_similarity,
                    future_pose_mismatch=future_pose_mismatch,
                    future_silhouette_mismatch=future_silhouette_mismatch,
                )
            )
            continue
        if (
            future_silhouette_mismatch is not None
            and future_silhouette_mismatch
            > settings.maximum_reliable_silhouette_mismatch
        ):
            transitions.append(
                _hard(
                    "B",
                    "future_body_silhouette_mismatch",
                    appearance_similarity=appearance_similarity,
                    future_pose_mismatch=future_pose_mismatch,
                    future_silhouette_mismatch=future_silhouette_mismatch,
                )
            )
            continue

        sample_times = (0.0, handle_seconds / 2.0, handle_seconds)
        outgoing_crops = tuple(
            _fixed_crop_at(outgoing_match.pan_crop, outgoing.duration_seconds + time)
            for time in sample_times
        )
        incoming_crops = tuple(
            _fixed_crop_at(incoming_match.pan_crop, time) for time in sample_times
        )
        if any(
            not _box_is_contained(box, crop, settings.face_box_crop_margin)
            for box, crop in zip(outgoing_boxes, outgoing_crops, strict=True)
        ) or any(
            not _box_is_contained(box, crop, settings.face_box_crop_margin)
            for box, crop in zip(incoming_boxes, incoming_crops, strict=True)
        ):
            transitions.append(_hard("B", "future_face_near_crop_edge"))
            continue

        projected_outgoing = tuple(
            _project_geometry(geometry, crop)
            for geometry, crop in zip(
                outgoing_geometry, outgoing_crops, strict=True
            )
        )
        projected_incoming = tuple(
            _project_geometry(geometry, crop)
            for geometry, crop in zip(
                incoming_geometry, incoming_crops, strict=True
            )
        )
        mismatch = face_geometry_minimax_mismatch(
            projected_outgoing,
            projected_incoming,
        )
        if mismatch > settings.maximum_projected_face_mismatch:
            transitions.append(_hard("B", "future_face_head_pose_mismatch", mismatch))
            continue

        transitions.append(
            RenderTransition(
                kind=TransitionKind.FACE_DISSOLVE,
                frame_count=settings.dissolve_frame_count,
                tier="A",
                reason=(
                    "safe_aligned_future_face_head_sface_appearance_body_evidence"
                ),
                face_mismatch=mismatch,
                appearance_min_cosine_similarity=appearance_similarity,
                future_pose_max_mismatch=future_pose_mismatch,
                future_silhouette_max_mismatch=(
                    future_silhouette_mismatch
                ),
            )
        )

    return GroupTransitionPlan(tuple(transitions))


__all__ = [
    "GroupTransitionConfig",
    "GroupTransitionPlan",
    "build_group_transition_plan",
]
