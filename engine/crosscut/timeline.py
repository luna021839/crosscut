"""Global source selection for natural cross-cut timelines.

The optimiser works on short, already synchronized decision segments (usually
beats, phrases, or safe camera boundaries).  It uses dynamic programming rather
than greedily picking the prettiest frame at every segment, which lets a slightly
weaker current shot win when it avoids an awkward cut on the next beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import ceil, isfinite, log, pi, sqrt
from typing import Mapping, Sequence

from .face_identity import cosine_similarity
from .ive import EditMode, IVEMember
from .pose import FormationPersonDescriptor
from .selection_arm_motion import SelectionArmWindow, selection_arm_cost
from .timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
)


def _score(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return numeric


class TimelineMode(str, Enum):
    GENERAL = "general"
    MEMBER = "member"


class NoCompleteTimelinePath(ValueError):
    """Typed, structured exhaustion of the constrained timeline graph."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        segment_index: int | None = None,
        rejection_counts: Mapping[str, int] | None = None,
        rejection_details: Mapping[str, int] | None = None,
        profile_failures: Sequence[
            tuple[str, "NoCompleteTimelinePath"]
        ] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.segment_index = segment_index
        self.rejection_counts = dict(sorted((rejection_counts or {}).items()))
        # Diagnostic causes are separate from edge counts and must never alter
        # exhaustion classification, repair budgets, scores or admission.
        self.rejection_details = dict(sorted((rejection_details or {}).items()))
        failures = tuple(profile_failures)
        if any(
            not isinstance(profile, str)
            or not profile
            or profile != profile.strip()
            or not isinstance(failure, NoCompleteTimelinePath)
            for profile, failure in failures
        ):
            raise TypeError("profile no-path diagnostics are invalid")
        if len({profile for profile, _failure in failures}) != len(failures):
            raise ValueError("profile no-path diagnostics cannot repeat")
        self.profile_failures = failures

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason_code": self.reason_code,
            "segment_index": self.segment_index,
            "rejection_counts": dict(self.rejection_counts),
            "message": str(self),
        }
        if self.profile_failures:
            payload["profile_failures"] = [
                {
                    "cadence_profile": profile,
                    "no_complete_path": failure.to_dict(),
                }
                for profile, failure in self.profile_failures
            ]
        if self.rejection_details:
            payload["rejection_details"] = dict(self.rejection_details)
        return payload


@dataclass(frozen=True, slots=True)
class TimelineNoGoodConstraintIndex:
    """One-time validation of no-goods against an immutable candidate lattice."""

    lattice_identity: str
    no_goods: TimelineNoGoodSet

    def assert_matches(
        self,
        candidates_by_segment: Sequence[Sequence["TimelineCandidate"]],
    ) -> None:
        if self.lattice_identity != _timeline_lattice_identity(
            candidates_by_segment
        ):
            raise ValueError(
                "compiled no-good constraints belong to a different timeline lattice"
            )


class ShotType(str, Enum):
    UNKNOWN = "unknown"
    GROUP_WIDE = "group_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"


BOUNDARY_SIDE_MAX_SCALE_DELTA = 0.20
BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH = 0.34
BOUNDARY_SIDE_MAX_POSE_MISMATCH = 0.42
BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH = 0.34

# Ordinary cuts still use the strict production continuity gates below.  Human
# stage edits also make deliberate wide/close or reframed-face contrasts at
# phrase anchors, so one explicitly authored anchor may exceed only scale and
# focus when pose, screen layout, and silhouette all prove continuity.  The
# alternate scale ceiling stays just below the observed human-reference
# silhouette-scale p75 (~0.64); this is not a global gate relaxation.
PRODUCTION_MAX_SCALE_MISMATCH = 0.20
PRODUCTION_MAX_FOCUS_MISMATCH = 0.24
PRODUCTION_MAX_LAYOUT_MISMATCH = 0.32
PRODUCTION_MAX_SILHOUETTE_MISMATCH = 0.36
PRODUCTION_MAX_FORMATION_MISMATCH = 0.30
PRODUCTION_MAX_POSE_MISMATCH = 0.42
INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS = (
    "human_reference_highlight_phrase_anchor_pose_layout_v1"
)
INTENTIONAL_PHRASE_ANCHOR_MAX_SCALE_MISMATCH = 0.62
INTENTIONAL_PHRASE_ANCHOR_MAX_FOCUS_MISMATCH = 0.40
INTENTIONAL_PHRASE_ANCHOR_MAX_POSE_MISMATCH = 0.18


def intentional_phrase_anchor_authorized_mismatches(
    *,
    anchor_basis: str | None,
    scale_mismatch: float | None,
    focus_mismatch: float | None,
    layout_mismatch: float | None,
    silhouette_mismatch: float | None,
    formation_mismatch: float | None,
    pose_mismatch: float | None,
) -> tuple[str, ...]:
    """Return the only ordinary-gate mismatches one authored anchor may waive.

    Missing continuity evidence fails closed.  Layout, silhouette, formation,
    pose, scene-cut proximity, close-up identity, and same-side stability are
    never waived; only a bounded scale and/or focus contrast can be authored.
    """

    if anchor_basis != INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS:
        return ()
    values = (
        scale_mismatch,
        focus_mismatch,
        layout_mismatch,
        silhouette_mismatch,
        formation_mismatch,
        pose_mismatch,
    )
    if any(value is not None and not isfinite(float(value)) for value in values):
        return ()
    if (
        layout_mismatch is None
        or silhouette_mismatch is None
        or pose_mismatch is None
        or layout_mismatch > PRODUCTION_MAX_LAYOUT_MISMATCH
        or silhouette_mismatch > PRODUCTION_MAX_SILHOUETTE_MISMATCH
        or pose_mismatch > INTENTIONAL_PHRASE_ANCHOR_MAX_POSE_MISMATCH
        or (
            formation_mismatch is not None
            and formation_mismatch > PRODUCTION_MAX_FORMATION_MISMATCH
        )
        or (
            scale_mismatch is not None
            and scale_mismatch > INTENTIONAL_PHRASE_ANCHOR_MAX_SCALE_MISMATCH
        )
        or (
            focus_mismatch is not None
            and focus_mismatch > INTENTIONAL_PHRASE_ANCHOR_MAX_FOCUS_MISMATCH
        )
    ):
        return ()
    authorized: list[str] = []
    if (
        scale_mismatch is not None
        and scale_mismatch > PRODUCTION_MAX_SCALE_MISMATCH
    ):
        authorized.append("scale")
    if (
        focus_mismatch is not None
        and focus_mismatch > PRODUCTION_MAX_FOCUS_MISMATCH
    ):
        authorized.append("focus")
    return tuple(authorized)


@dataclass(frozen=True, slots=True)
class BoundarySideStability:
    """Multi-frame stability immediately inside one side of an edit boundary.

    The values compare frames from the same source over a roughly three-frame
    window.  They are deliberately scalar diagnostics: raw pose/appearance
    embeddings never need to leave the transient candidate.
    """

    sample_span_seconds: float
    scale_delta: float
    structure_mismatch: float | None = None
    pose_mismatch: float | None = None
    shot_type_mismatch: float = 0.0

    def __post_init__(self) -> None:
        try:
            sample_span = float(self.sample_span_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "boundary-side sample span must be positive and finite"
            ) from exc
        if not isfinite(sample_span) or sample_span <= 0.0:
            raise ValueError("boundary-side sample span must be positive and finite")
        object.__setattr__(self, "sample_span_seconds", sample_span)
        for name in (
            "scale_delta",
            "structure_mismatch",
            "pose_mismatch",
            "shot_type_mismatch",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _score(value, name))

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "scale",
                "structure",
                "pose",
                "shot_type",
            )
            if getattr(self, f"{name}_mismatch", None) is not None
            or (name == "scale" and self.scale_delta is not None)
        )

    def to_dict(self) -> dict[str, float | list[str] | None]:
        return {
            "sample_span_seconds": self.sample_span_seconds,
            "available_metrics": list(self.available_metrics),
            "scale_delta": self.scale_delta,
            "structure_mismatch": self.structure_mismatch,
            "pose_mismatch": self.pose_mismatch,
            "shot_type_mismatch": self.shot_type_mismatch,
        }


def boundary_side_stability_exceeded(
    evidence: BoundarySideStability,
    *,
    maximum_scale_delta: float | None,
    maximum_structure_mismatch: float | None,
    maximum_pose_mismatch: float | None,
    maximum_shot_type_mismatch: float | None,
) -> tuple[str, ...]:
    """Return same-source boundary-window metrics above configured hard gates."""

    checks = (
        ("scale", evidence.scale_delta, maximum_scale_delta),
        ("structure", evidence.structure_mismatch, maximum_structure_mismatch),
        ("pose", evidence.pose_mismatch, maximum_pose_mismatch),
        ("shot_type", evidence.shot_type_mismatch, maximum_shot_type_mismatch),
    )
    return tuple(
        name
        for name, value, maximum in checks
        if value is not None and maximum is not None and value > maximum + 1e-9
    )


@dataclass(frozen=True, slots=True)
class TimelineCandidate:
    """One source option for one synchronized timeline segment.

    ``start_seconds`` / ``end_seconds`` are canonical reference-song time, not
    source-local decode time.  ``pose_embedding`` can be any fixed-length,
    normalized landmark vector. ``structure_embedding`` is its deterministic
    model-free fallback, while ``silhouette_embedding`` represents aggregate
    on-screen human layout.
    ``motion`` is an image-plane direction vector.  They are optional so the
    engine remains usable before pose estimation is wired in.

    For member mode, a close/medium candidate is eligible only when
    ``target_member`` matches the requested IVE member, confidence passes the
    configured gate, and ``target_crop_available`` is true.  A candidate marked
    ``is_group_wide`` remains eligible as a safety fallback.
    """

    segment_index: int
    source_id: str
    start_seconds: float
    end_seconds: float
    quality_score: float = 1.0
    sync_score: float = 1.0
    composition_score: float = 0.5
    boundary_score: float = 0.5
    shot_scale: float = 0.5
    entry_shot_scale: float | None = None
    exit_shot_scale: float | None = None
    shot_type: ShotType | str = ShotType.MEDIUM
    entry_shot_type: ShotType | str | None = None
    exit_shot_type: ShotType | str | None = None
    entry_scene_cut_near: bool = False
    exit_scene_cut_near: bool = False
    intentional_transition_anchor_basis: str | None = None
    internal_scene_cut_master_seconds: tuple[float, ...] = ()
    entry_boundary_side_stability: BoundarySideStability | None = None
    exit_boundary_side_stability: BoundarySideStability | None = None
    target_member: IVEMember | str | None = None
    target_confidence: float = 0.0
    target_crop_available: bool = False
    is_group_wide: bool = False
    crop_stability_score: float = 0.5
    pose_embedding: tuple[float, ...] = ()
    entry_pose_embedding: tuple[float, ...] = ()
    exit_pose_embedding: tuple[float, ...] = ()
    structure_embedding: tuple[float, ...] = ()
    entry_structure_embedding: tuple[float, ...] = ()
    exit_structure_embedding: tuple[float, ...] = ()
    silhouette_embedding: tuple[float, ...] = ()
    entry_silhouette_embedding: tuple[float, ...] = ()
    exit_silhouette_embedding: tuple[float, ...] = ()
    person_layout_embedding: tuple[float, ...] = ()
    entry_person_layout_embedding: tuple[float, ...] = ()
    exit_person_layout_embedding: tuple[float, ...] = ()
    formation_people: tuple[FormationPersonDescriptor, ...] = field(
        default=(), repr=False
    )
    entry_formation_people: tuple[FormationPersonDescriptor, ...] = field(
        default=(), repr=False
    )
    exit_formation_people: tuple[FormationPersonDescriptor, ...] = field(
        default=(), repr=False
    )
    face_identity_embedding: tuple[float, ...] = field(default=(), repr=False)
    entry_face_identity_embedding: tuple[float, ...] = field(default=(), repr=False)
    exit_face_identity_embedding: tuple[float, ...] = field(default=(), repr=False)
    color_embedding: tuple[float, ...] = ()
    entry_color_embedding: tuple[float, ...] = ()
    exit_color_embedding: tuple[float, ...] = ()
    entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    focus_bbox_xywh: tuple[float, float, float, float] | None = None
    entry_focus_bbox_xywh: tuple[float, float, float, float] | None = None
    exit_focus_bbox_xywh: tuple[float, float, float, float] | None = None
    subject_bbox_xywh: tuple[float, float, float, float] | None = None
    entry_subject_bbox_xywh: tuple[float, float, float, float] | None = None
    exit_subject_bbox_xywh: tuple[float, float, float, float] | None = None
    subject_centroid_xy: tuple[float, float] | None = None
    entry_subject_centroid_xy: tuple[float, float] | None = None
    exit_subject_centroid_xy: tuple[float, float] | None = None
    detected_person_count: int = 0
    entry_detected_person_count: int = 0
    exit_detected_person_count: int = 0
    motion: tuple[float, float] | None = None
    entry_motion: tuple[float, float] | None = None
    exit_motion: tuple[float, float] | None = None
    candidate_id: str = ""
    shot_id: str = ""
    # A measured endpoint is not a multi-frame window. Keep the contracts
    # separate so exact-offset comparison does not fabricate duplicate frames.
    entry_face_geometry_exact: tuple[float, ...] = field(default=(), repr=False)
    exit_face_geometry_exact: tuple[float, ...] = field(default=(), repr=False)
    entry_arm_window: SelectionArmWindow | None = field(default=None, repr=False)
    exit_arm_window: SelectionArmWindow | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        for window in (self.entry_arm_window, self.exit_arm_window):
            if window is not None and (not isinstance(window, SelectionArmWindow)
                    or window.source_id != self.source_id
                    or window.master_times[0] < self.start_seconds - 1e-7
                    or window.master_times[1] > self.end_seconds + 1e-7):
                raise ValueError('selection arm evidence must belong to this source and segment clock')
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("candidate time range must have positive duration")
        for field_name in (
            "quality_score",
            "sync_score",
            "composition_score",
            "boundary_score",
            "shot_scale",
            "target_confidence",
            "crop_stability_score",
        ):
            object.__setattr__(self, field_name, _score(getattr(self, field_name), field_name))
        for field_name in ("entry_shot_scale", "exit_shot_scale"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _score(value, field_name))
        if not isinstance(self.shot_type, ShotType):
            object.__setattr__(self, "shot_type", ShotType(self.shot_type))
        for field_name in ("entry_shot_type", "exit_shot_type"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ShotType):
                object.__setattr__(self, field_name, ShotType(value))
        if not isinstance(self.entry_scene_cut_near, bool) or not isinstance(
            self.exit_scene_cut_near, bool
        ):
            raise ValueError("scene-cut proximity flags must be booleans")
        if self.intentional_transition_anchor_basis not in {
            None,
            INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS,
        }:
            raise ValueError("intentional transition anchor basis is invalid")
        internal_cut_times = tuple(
            float(value) for value in self.internal_scene_cut_master_seconds
        )
        if any(not isfinite(value) for value in internal_cut_times):
            raise ValueError("internal scene-cut times must be finite")
        if any(
            not self.start_seconds < value < self.end_seconds
            for value in internal_cut_times
        ):
            raise ValueError("internal scene-cut times must lie inside the candidate")
        if any(
            right <= left
            for left, right in zip(internal_cut_times, internal_cut_times[1:])
        ):
            raise ValueError("internal scene-cut times must be strictly increasing")
        object.__setattr__(
            self,
            "internal_scene_cut_master_seconds",
            internal_cut_times,
        )
        for stability in (
            self.entry_boundary_side_stability,
            self.exit_boundary_side_stability,
        ):
            if stability is not None and not isinstance(
                stability, BoundarySideStability
            ):
                raise ValueError(
                    "boundary-side stability evidence must use BoundarySideStability"
                )
        if self.target_member is not None:
            object.__setattr__(self, "target_member", IVEMember.parse(self.target_member))
        if self.is_group_wide and self.shot_type not in (
            ShotType.GROUP_WIDE,
            ShotType.WIDE,
        ):
            raise ValueError("is_group_wide candidates must use a wide shot type")
        for motion in (self.motion, self.entry_motion, self.exit_motion):
            if motion is not None and len(motion) != 2:
                raise ValueError("motion must contain x/y components")
        for embedding in (
            self.pose_embedding,
            self.entry_pose_embedding,
            self.exit_pose_embedding,
            self.structure_embedding,
            self.entry_structure_embedding,
            self.exit_structure_embedding,
            self.silhouette_embedding,
            self.entry_silhouette_embedding,
            self.exit_silhouette_embedding,
            self.person_layout_embedding,
            self.entry_person_layout_embedding,
            self.exit_person_layout_embedding,
            self.face_identity_embedding,
            self.entry_face_identity_embedding,
            self.exit_face_identity_embedding,
            self.color_embedding,
            self.entry_color_embedding,
            self.exit_color_embedding,
        ):
            if any(not isinstance(value, (int, float)) for value in embedding):
                raise ValueError("pose embeddings must be numeric")
        for geometry in (self.entry_face_geometry_exact, self.exit_face_geometry_exact):
            if geometry and (len(geometry) != 10
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not isfinite(float(value)) for value in geometry)
                    or geometry[4] <= 0):
                raise ValueError("exact face geometry needs ten finite values and positive eye scale")
        for window in (
            self.entry_face_geometry_window,
            self.exit_face_geometry_window,
        ):
            if window and len(window) < 2:
                raise ValueError("face geometry windows need at least two frames")
            for geometry in window:
                if len(geometry) != 10:
                    raise ValueError("face geometry vectors must contain ten values")
                if any(
                    not isinstance(value, (int, float))
                    or not isfinite(float(value))
                    for value in geometry
                ):
                    raise ValueError("face geometry values must be finite numbers")
                if geometry[4] <= 0.0:
                    raise ValueError("face eye scale must be positive")
        for bbox in (
            self.focus_bbox_xywh,
            self.entry_focus_bbox_xywh,
            self.exit_focus_bbox_xywh,
            self.subject_bbox_xywh,
            self.entry_subject_bbox_xywh,
            self.exit_subject_bbox_xywh,
        ):
            if bbox is None:
                continue
            if len(bbox) != 4 or any(
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in bbox
            ):
                raise ValueError("normalized subject boxes must contain four unit values")
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                raise ValueError("normalized subject boxes must have positive size")
            if bbox[0] + bbox[2] > 1.000001 or bbox[1] + bbox[3] > 1.000001:
                raise ValueError("normalized subject box exceeds frame bounds")
        for centroid in (
            self.subject_centroid_xy,
            self.entry_subject_centroid_xy,
            self.exit_subject_centroid_xy,
        ):
            if centroid is None:
                continue
            if len(centroid) != 2 or any(
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in centroid
            ):
                raise ValueError("subject centroids must contain two unit values")
        for count in (
            self.detected_person_count,
            self.entry_detected_person_count,
            self.exit_detected_person_count,
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("detected person counts must be non-negative integers")
        for formation in (
            self.formation_people,
            self.entry_formation_people,
            self.exit_formation_people,
        ):
            if len(formation) > 6:
                raise ValueError("formation descriptors support at most six people")
            if any(
                not isinstance(person, FormationPersonDescriptor)
                for person in formation
            ):
                raise ValueError("formation entries must be person descriptors")
            if any(
                right.centroid_xy[0] < left.centroid_xy[0]
                for left, right in zip(formation, formation[1:])
            ):
                raise ValueError("formation descriptors must be ordered left-to-right")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def is_safe_for_member(
        self, target: IVEMember, minimum_confidence: float
    ) -> bool:
        return (
            self.target_member is target
            and self.target_confidence >= minimum_confidence
            and self.target_crop_available
        )


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    """Weights and editing constraints for Viterbi source selection."""

    quality_weight: float = 1.25
    quality_soft_floor: float | None = None
    quality_soft_floor_penalty: float = 0.0
    sync_weight: float = 1.7
    composition_weight: float = 0.65
    boundary_weight: float = 0.35
    group_wide_frequency_preference_weight: float = 0.0
    group_wide_distance_preference_weight: float = 0.0
    target_weight: float = 2.4
    crop_stability_weight: float = 0.7
    continuity_bonus: float = 0.28
    cut_penalty: float = 0.20
    early_cut_penalty: float = 0.65
    long_hold_penalty: float = 0.14
    source_hold_penalty: float = 0.0
    source_switch_bonus: float = 0.0
    # Different camera versions of one broadcast still count as separate
    # usable sources. Discount only their soft stage-change reward.
    source_broadcast_slots: tuple[tuple[str, str], ...] = ()
    same_broadcast_switch_bonus_multiplier: float = 0.25
    ping_pong_penalty: float = 0.0
    preferred_source_bonus: float = 0.0
    preferred_source_max_score_sacrifice: float = 0.0
    pose_transition_weight: float = 0.85
    silhouette_transition_weight: float = 0.62
    motion_transition_weight: float = 0.48
    scale_transition_weight: float = 0.40
    color_transition_weight: float = 0.24
    face_geometry_transition_weight: float = 0.28
    focus_position_transition_weight: float = 1.35
    subject_geometry_transition_weight: float = 1.10
    person_count_transition_weight: float = 0.42
    person_layout_transition_weight: float = 1.55
    formation_transition_weight: float = 1.90
    shot_type_transition_weight: float = 0.72
    face_identity_transition_weight: float = 1.60
    maximum_cross_source_scale_delta: float | None = None
    maximum_cross_source_focus_mismatch: float | None = None
    maximum_cross_source_layout_mismatch: float | None = None
    maximum_cross_source_formation_mismatch: float | None = None
    maximum_cross_source_silhouette_mismatch: float | None = None
    minimum_cross_source_face_similarity: float | None = None
    require_cross_source_visual_evidence: bool = False
    allow_intentional_phrase_anchor_transitions: bool = False
    maximum_boundary_side_scale_delta: float | None = None
    maximum_boundary_side_structure_mismatch: float | None = None
    maximum_boundary_side_pose_mismatch: float | None = None
    maximum_boundary_side_shot_type_mismatch: float | None = None
    require_cross_source_boundary_side_stability: bool = False
    group_wide_fallback_penalty: float = 0.55
    minimum_member_confidence: float = 0.72
    minimum_hold_segments: int = 2
    preferred_max_hold_segments: int = 6
    preferred_max_source_hold_segments: int | None = None
    maximum_same_source_hold_segments: int | None = None
    preferred_max_source_hold_seconds: float | None = None
    maximum_same_source_hold_seconds: float | None = None
    maximum_switch_score_sacrifice: float = 0.0

    def __post_init__(self) -> None:
        for name in ('group_wide_frequency_preference_weight','group_wide_distance_preference_weight'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not isfinite(value) or value<0:
                raise ValueError('GROUP framing preference weights must be finite and non-negative')
        weight_names = (
            "quality_weight",
            "sync_weight",
            "composition_weight",
            "boundary_weight",
            "target_weight",
            "crop_stability_weight",
            "continuity_bonus",
            "cut_penalty",
            "early_cut_penalty",
            "long_hold_penalty",
            "source_hold_penalty",
            "ping_pong_penalty",
            "preferred_source_bonus",
            "preferred_source_max_score_sacrifice",
            "pose_transition_weight",
            "silhouette_transition_weight",
            "motion_transition_weight",
            "scale_transition_weight",
            "color_transition_weight",
            "face_geometry_transition_weight",
            "focus_position_transition_weight",
            "subject_geometry_transition_weight",
            "person_count_transition_weight",
            "person_layout_transition_weight",
            "formation_transition_weight",
            "shot_type_transition_weight",
            "face_identity_transition_weight",
            "group_wide_fallback_penalty",
            "maximum_switch_score_sacrifice",
        )
        if any(getattr(self, name) < 0 for name in weight_names):
            raise ValueError("timeline weights and penalties cannot be negative")
        if self.quality_soft_floor is not None and (
            not isfinite(self.quality_soft_floor)
            or not 0.0 <= self.quality_soft_floor <= 1.0
        ):
            raise ValueError("quality_soft_floor must be normalized or None")
        if (
            not isfinite(self.quality_soft_floor_penalty)
            or self.quality_soft_floor_penalty < 0.0
        ):
            raise ValueError(
                "quality_soft_floor_penalty must be finite and non-negative"
            )
        if (
            self.quality_soft_floor is None
            and self.quality_soft_floor_penalty > 0.0
        ):
            raise ValueError(
                "quality_soft_floor_penalty requires quality_soft_floor"
            )
        if not isfinite(self.source_switch_bonus) or self.source_switch_bonus < 0.0:
            raise ValueError("source_switch_bonus must be finite and non-negative")
        if (
            not isfinite(self.same_broadcast_switch_bonus_multiplier)
            or not 0.0 <= self.same_broadcast_switch_bonus_multiplier <= 1.0
        ):
            raise ValueError("same broadcast switch multiplier must be normalized")
        if not isinstance(self.source_broadcast_slots, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in self.source_broadcast_slots
        ):
            raise ValueError("broadcast slots must be immutable source/slot pairs")
        if len(dict(self.source_broadcast_slots)) != len(self.source_broadcast_slots):
            raise ValueError("broadcast slot source IDs must be unique")
        for name in (
            "maximum_cross_source_scale_delta",
            "maximum_cross_source_focus_mismatch",
            "maximum_cross_source_layout_mismatch",
            "maximum_cross_source_formation_mismatch",
            "maximum_cross_source_silhouette_mismatch",
            "maximum_boundary_side_scale_delta",
            "maximum_boundary_side_structure_mismatch",
            "maximum_boundary_side_pose_mismatch",
            "maximum_boundary_side_shot_type_mismatch",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a normalized value or None")
        if self.minimum_cross_source_face_similarity is not None and (
            not isfinite(self.minimum_cross_source_face_similarity)
            or not -1.0 <= self.minimum_cross_source_face_similarity <= 1.0
        ):
            raise ValueError(
                "minimum_cross_source_face_similarity must be cosine similarity or None"
            )
        if (
            not isinstance(self.require_cross_source_visual_evidence, bool)
            or not isinstance(
                self.require_cross_source_boundary_side_stability, bool
            )
            or not isinstance(
                self.allow_intentional_phrase_anchor_transitions, bool
            )
        ):
            raise ValueError("cross-source evidence requirements must be booleans")
        _score(self.minimum_member_confidence, "minimum_member_confidence")
        if self.minimum_hold_segments < 1:
            raise ValueError("minimum_hold_segments must be at least 1")
        if self.preferred_max_hold_segments < self.minimum_hold_segments:
            raise ValueError(
                "preferred_max_hold_segments cannot be below minimum_hold_segments"
            )
        if (
            self.preferred_max_source_hold_segments is not None
            and self.preferred_max_source_hold_segments < 1
        ):
            raise ValueError("preferred source hold must be at least 1")
        if (
            self.maximum_same_source_hold_segments is not None
            and self.maximum_same_source_hold_segments < 1
        ):
            raise ValueError("maximum same-source hold must be at least 1")
        if (
            self.preferred_max_source_hold_segments is not None
            and self.maximum_same_source_hold_segments is not None
            and self.maximum_same_source_hold_segments
            < self.preferred_max_source_hold_segments
        ):
            raise ValueError("maximum same-source hold cannot be below preferred source hold")
        for name in (
            "preferred_max_source_hold_seconds",
            "maximum_same_source_hold_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite or None")
        if (
            self.preferred_max_source_hold_seconds is not None
            and self.maximum_same_source_hold_seconds is not None
            and self.maximum_same_source_hold_seconds
            < self.preferred_max_source_hold_seconds
        ):
            raise ValueError(
                "maximum same-source hold seconds cannot be below preferred seconds"
            )


@dataclass(frozen=True, slots=True)
class TimelineChoice:
    candidate: TimelineCandidate
    emission_score: float
    transition_score: float
    cumulative_score: float
    used_group_wide_fallback: bool


@dataclass(frozen=True, slots=True)
class TimelineResult:
    mode: TimelineMode
    target_member: IVEMember | None
    choices: tuple[TimelineChoice, ...]
    total_score: float

    @property
    def candidates(self) -> tuple[TimelineCandidate, ...]:
        return tuple(choice.candidate for choice in self.choices)

    @property
    def fallback_segment_indices(self) -> tuple[int, ...]:
        return tuple(
            choice.candidate.segment_index
            for choice in self.choices
            if choice.used_group_wide_fallback
        )


@dataclass(frozen=True, slots=True)
class _Node:
    cumulative_score: float
    previous_state: tuple[int, int, int, int] | None
    emission_score: float
    transition_score: float
    used_fallback: bool


def _normalised_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    mismatch = _optional_normalised_distance(left, right)
    return mismatch if mismatch is not None else 0.0


def _optional_normalised_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Return embedding RMS while preserving unavailable evidence as ``None``."""

    if not left or not right or len(left) != len(right):
        return None
    squared = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    # RMS is less sensitive to embedding dimension. Clamp because landmark
    # vectors from third-party models are not guaranteed to be normalized.
    return min(1.0, sqrt(squared / len(left)))


def _motion_mismatch(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> float:
    if left is None or right is None:
        return 0.0
    left_norm = sqrt(left[0] ** 2 + left[1] ** 2)
    right_norm = sqrt(right[0] ** 2 + right[1] ** 2)
    if left_norm < 1e-8 or right_norm < 1e-8:
        return 0.0
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    return min(1.0, max(0.0, (1.0 - cosine) / 2.0))


def _bbox_geometry_mismatch(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
    *,
    left_centroid: tuple[float, float] | None = None,
    right_centroid: tuple[float, float] | None = None,
) -> float | None:
    """Screen-space attention jump between two subject layouts."""

    if left is None or right is None:
        return None
    left_center = left_centroid or (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
    right_center = right_centroid or (
        right[0] + right[2] / 2.0,
        right[1] + right[3] / 2.0,
    )
    center = min(1.0, sqrt(
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) / 0.42)
    left_area = max(left[2] * left[3], 1e-6)
    right_area = max(right[2] * right[3], 1e-6)
    scale = min(1.0, abs(log(left_area / right_area)) / log(3.2))
    width = min(1.0, abs(left[2] - right[2]) / 0.55)
    height = min(1.0, abs(left[3] - right[3]) / 0.55)
    return min(1.0, 0.52 * center + 0.28 * scale + 0.10 * width + 0.10 * height)


def screen_layout_mismatch(
    left_embedding: Sequence[float],
    right_embedding: Sequence[float],
    *,
    left_bbox: tuple[float, float, float, float] | None = None,
    right_bbox: tuple[float, float, float, float] | None = None,
    left_centroid: tuple[float, float] | None = None,
    right_centroid: tuple[float, float] | None = None,
) -> float | None:
    """Return the reviewable person-layout mismatch for one source switch.

    The richer layout embedding remains authoritative when both boundary
    vectors are comparable.  If inference missed on either side, use the
    exact subject-box/centroid scalar that survives serialization.  When
    neither representation is complete the signal remains unavailable; a
    missing detector result is never converted into evidence of compatibility.
    """

    embedding_mismatch = _optional_normalised_distance(
        left_embedding,
        right_embedding,
    )
    if embedding_mismatch is not None:
        return embedding_mismatch
    return _bbox_geometry_mismatch(
        left_bbox,
        right_bbox,
        left_centroid=left_centroid,
        right_centroid=right_centroid,
    )


def cross_source_layout_mismatch(
    previous: TimelineCandidate,
    current: TimelineCandidate,
) -> float | None:
    """Mirror ``timeline_review`` layout evidence for a candidate boundary."""

    # Empty embedding tuples are explicit boundary detector misses and must
    # not fall back to a segment midpoint.  Optional bbox/centroid fields use
    # the reviewer's normal boundary-value fallback when the boundary sample
    # itself is absent.
    previous_embedding = previous.exit_person_layout_embedding
    current_embedding = current.entry_person_layout_embedding
    previous_bbox = (
        previous.exit_subject_bbox_xywh
        if previous.exit_subject_bbox_xywh is not None
        else previous.subject_bbox_xywh
    )
    current_bbox = (
        current.entry_subject_bbox_xywh
        if current.entry_subject_bbox_xywh is not None
        else current.subject_bbox_xywh
    )
    previous_centroid = (
        previous.exit_subject_centroid_xy
        if previous.exit_subject_centroid_xy is not None
        else previous.subject_centroid_xy
    )
    current_centroid = (
        current.entry_subject_centroid_xy
        if current.entry_subject_centroid_xy is not None
        else current.subject_centroid_xy
    )
    return screen_layout_mismatch(
        previous_embedding,
        current_embedding,
        left_bbox=previous_bbox,
        right_bbox=current_bbox,
        left_centroid=previous_centroid,
        right_centroid=current_centroid,
    )


def _person_count_mismatch(left: int, right: int) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return min(1.0, abs(log((left + 1.0) / (right + 1.0))) / log(4.0))


def _upper_body_shape_mismatch(
    left: FormationPersonDescriptor,
    right: FormationPersonDescriptor,
) -> float | None:
    left_points = left.upper_body_keypoints_xyc
    right_points = right.upper_body_keypoints_xyc
    if not left_points or not right_points or len(left_points) != len(right_points):
        return None
    weighted_squared = 0.0
    total_weight = 0.0
    visibility_delta = 0.0
    comparable = 0
    for left_point, right_point in zip(left_points, right_points, strict=True):
        left_confidence = float(left_point[2])
        right_confidence = float(right_point[2])
        visibility_delta += abs(left_confidence - right_confidence)
        weight = min(left_confidence, right_confidence)
        if weight < 0.20:
            continue
        weighted_squared += weight * (
            (float(left_point[0]) - float(right_point[0])) ** 2
            + (float(left_point[1]) - float(right_point[1])) ** 2
        )
        total_weight += weight
        comparable += 1
    if comparable < 4 or total_weight <= 1e-9:
        return None
    # Coordinates are torso-normalized to [-1, 1]. A 0.72 RMS displacement is
    # already a visibly different arm/shoulder pose at a hard cut.
    shape = min(1.0, sqrt(weighted_squared / total_weight) / 0.72)
    visibility = min(1.0, visibility_delta / len(left_points))
    return min(1.0, 0.88 * shape + 0.12 * visibility)


def _formation_pair_cost(
    left: FormationPersonDescriptor,
    right: FormationPersonDescriptor,
    *,
    left_rank: int,
    right_rank: int,
    left_count: int,
    right_count: int,
) -> float:
    center = min(
        1.0,
        sqrt(
            (left.centroid_xy[0] - right.centroid_xy[0]) ** 2
            + (left.centroid_xy[1] - right.centroid_xy[1]) ** 2
        )
        / 0.52,
    )
    left_width = max(1e-5, left.bbox_xywh[2])
    left_height = max(1e-5, left.bbox_xywh[3])
    right_width = max(1e-5, right.bbox_xywh[2])
    right_height = max(1e-5, right.bbox_xywh[3])
    scale = min(
        1.0,
        (
            abs(log(left_width / right_width))
            + abs(log(left_height / right_height))
        )
        / (2.0 * log(2.4)),
    )
    shape = _upper_body_shape_mismatch(left, right)
    if shape is None:
        base = 0.72 * center + 0.28 * scale
    else:
        # When two dancers exchange left/right while keeping different arm
        # poses, a shape-only assignment wants to cross the pairs and a
        # position-only assignment wants to ignore the choreography. Keeping
        # both terms strong makes either explanation pay for the visible jump.
        base = 0.36 * center + 0.08 * scale + 0.56 * shape
    left_fraction = left_rank / max(1, left_count - 1)
    right_fraction = right_rank / max(1, right_count - 1)
    rank = abs(left_fraction - right_fraction)
    return min(1.0, 0.92 * base + 0.08 * rank)


def formation_assignment_mismatch(
    left: Sequence[FormationPersonDescriptor],
    right: Sequence[FormationPersonDescriptor],
) -> float | None:
    """Minimum-cost one-to-one formation mismatch for at most six people.

    A small bitmask dynamic program is equivalent to Hungarian assignment at
    this fixed IVE-sized problem.  Unmatched detections retain an explicit
    penalty, while each pair jointly compares position, scale, left-to-right
    rank, and optional non-facial upper-body shape.
    """

    if not left or not right:
        return None
    if len(left) > 6 or len(right) > 6:
        raise ValueError("formation matching supports at most six people")

    # Assign every item from the smaller side to a unique item on the larger
    # side.  Keep original left/right ranks even when the sides are swapped.
    if len(left) <= len(right):
        smaller = tuple(left)
        larger = tuple(right)
        swapped = False
    else:
        smaller = tuple(right)
        larger = tuple(left)
        swapped = True
    states: dict[int, float] = {0: 0.0}
    for smaller_index in range(len(smaller)):
        next_states: dict[int, float] = {}
        for mask, accumulated in states.items():
            for larger_index in range(len(larger)):
                bit = 1 << larger_index
                if mask & bit:
                    continue
                if swapped:
                    pair = _formation_pair_cost(
                        larger[larger_index],
                        smaller[smaller_index],
                        left_rank=larger_index,
                        right_rank=smaller_index,
                        left_count=len(left),
                        right_count=len(right),
                    )
                else:
                    pair = _formation_pair_cost(
                        smaller[smaller_index],
                        larger[larger_index],
                        left_rank=smaller_index,
                        right_rank=larger_index,
                        left_count=len(left),
                        right_count=len(right),
                    )
                new_mask = mask | bit
                score = accumulated + pair
                previous = next_states.get(new_mask)
                if previous is None or score < previous:
                    next_states[new_mask] = score
        states = next_states
    matched_cost = min(states.values())
    unmatched_count = abs(len(left) - len(right))
    unmatched_cost = 0.82 * unmatched_count
    return min(1.0, (matched_cost + unmatched_cost) / max(len(left), len(right)))


def _relative_group_formation(
    people: Sequence[FormationPersonDescriptor],
) -> tuple[FormationPersonDescriptor, ...]:
    """Remove one camera's global pan/zoom while retaining group geometry.

    Broadcast cameras observe the same choreography with different framing.
    The renderer can compensate one global translation and uniform scale, so
    the production formation gate must compare the relative member layout and
    upper-body pose rather than treating that correctable camera transform as
    a choreography discontinuity.
    """

    if not people:
        return ()
    group_left = min(person.bbox_xywh[0] for person in people)
    group_top = min(person.bbox_xywh[1] for person in people)
    group_right = max(
        person.bbox_xywh[0] + person.bbox_xywh[2] for person in people
    )
    group_bottom = max(
        person.bbox_xywh[1] + person.bbox_xywh[3] for person in people
    )
    group_center_x = (group_left + group_right) / 2.0
    group_center_y = (group_top + group_bottom) / 2.0
    uniform_extent = max(
        group_right - group_left,
        group_bottom - group_top,
        1e-6,
    )

    normalized: list[FormationPersonDescriptor] = []
    for person in people:
        left, top, width, height = person.bbox_xywh
        relative_width = min(1.0, max(1e-6, width / uniform_extent))
        relative_height = min(1.0, max(1e-6, height / uniform_extent))
        relative_left = 0.5 + (left - group_center_x) / uniform_extent
        relative_top = 0.5 + (top - group_center_y) / uniform_extent
        relative_left = min(
            max(0.0, relative_left),
            1.0 - relative_width,
        )
        relative_top = min(
            max(0.0, relative_top),
            1.0 - relative_height,
        )
        relative_centroid = (
            min(
                1.0,
                max(
                    0.0,
                    0.5
                    + (person.centroid_xy[0] - group_center_x)
                    / uniform_extent,
                ),
            ),
            min(
                1.0,
                max(
                    0.0,
                    0.5
                    + (person.centroid_xy[1] - group_center_y)
                    / uniform_extent,
                ),
            ),
        )
        normalized.append(
            FormationPersonDescriptor(
                bbox_xywh=(
                    relative_left,
                    relative_top,
                    relative_width,
                    relative_height,
                ),
                centroid_xy=relative_centroid,
                detection_confidence=person.detection_confidence,
                upper_body_keypoints_xyc=person.upper_body_keypoints_xyc,
            )
        )
    return tuple(normalized)


def comparable_group_formation_mismatch(
    left: Sequence[FormationPersonDescriptor],
    right: Sequence[FormationPersonDescriptor],
    *,
    left_shot_type: ShotType | str,
    right_shot_type: ShotType | str,
) -> float | None:
    """Return a hard-gate mismatch only for reliable group-wide evidence.

    Close-up and medium-shot edits intentionally change visible person count
    and framing. Their rendered continuity is reviewed through face, pose,
    silhouette, and decoded-output geometry. Group formation is a separate
    hard signal only when both boundary frames are classified group-wide and
    expose the same multi-person cardinality.
    """

    try:
        left_type = (
            left_shot_type
            if isinstance(left_shot_type, ShotType)
            else ShotType(left_shot_type)
        )
        right_type = (
            right_shot_type
            if isinstance(right_shot_type, ShotType)
            else ShotType(right_shot_type)
        )
    except (TypeError, ValueError):
        return None
    if (
        left_type is not ShotType.GROUP_WIDE
        or right_type is not ShotType.GROUP_WIDE
    ):
        return None
    if len(left) < 2 or len(left) != len(right):
        return None
    return formation_assignment_mismatch(
        _relative_group_formation(left),
        _relative_group_formation(right),
    )


def _shot_type_mismatch(left: ShotType, right: ShotType) -> float:
    rank = {
        ShotType.GROUP_WIDE: 0,
        ShotType.WIDE: 1,
        ShotType.MEDIUM: 2,
        ShotType.CLOSE_UP: 3,
    }
    return abs(rank[left] - rank[right]) / 3.0


def _angle_distance(left: float, right: float) -> float:
    return abs((left - right + pi) % (2.0 * pi) - pi)


def _face_geometry_mismatch(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Compare normalized eye endpoints, scale, roll, and coarse head pose."""

    left_eye = sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
    right_eye = sqrt((left[2] - right[2]) ** 2 + (left[3] - right[3]) ** 2)
    # The original Cross-Cutting idea compares both eye endpoints. Normalize
    # that screen-space quantity, then supplement it with modern YuNet pose
    # geometry rather than copying the obsolete dlib/MoviePy implementation.
    endpoint_mismatch = min(1.0, (left_eye + right_eye) / (2.0 * 0.30))
    scale_mismatch = min(
        1.0,
        abs(log(float(left[4]) / float(right[4]))) / log(1.7),
    )
    roll_mismatch = min(1.0, _angle_distance(left[5], right[5]) / (pi / 4.0))
    pose_squared = sum((left[index] - right[index]) ** 2 for index in range(6, 10))
    pose_mismatch = min(1.0, sqrt(pose_squared / 4.0) / 0.75)
    return min(
        1.0,
        0.50 * endpoint_mismatch
        + 0.20 * scale_mismatch
        + 0.12 * roll_mismatch
        + 0.18 * pose_mismatch,
    )


def face_geometry_minimax_mismatch(
    outgoing_window: tuple[tuple[float, ...], ...],
    incoming_window: tuple[tuple[float, ...], ...],
) -> float:
    """Worst consecutive-frame face mismatch around a proposed hard cut.

    Audio-synchronized windows are paired as outgoing frame ``i`` and incoming
    frame ``i+1``. The maximum pair mismatch is the local minimax objective.
    Missing/ambiguous detections are represented by an empty window and remain
    neutral; this term can reject no candidate and awards no face bonus.
    """

    if (
        not outgoing_window
        or not incoming_window
        or len(outgoing_window) != len(incoming_window)
    ):
        return 0.0
    if any(
        len(geometry) != 10
        or geometry[4] <= 0.0
        or any(not isfinite(float(value)) for value in geometry)
        for window in (outgoing_window, incoming_window)
        for geometry in window
    ):
        return 0.0
    return max(
        _face_geometry_mismatch(outgoing, incoming)
        for outgoing, incoming in zip(
            outgoing_window,
            incoming_window,
            strict=True,
        )
    )


def _candidate_emission(
    candidate: TimelineCandidate,
    *,
    mode: TimelineMode,
    target: IVEMember | None,
    config: TimelineConfig,
) -> tuple[float, bool] | None:
    quality_floor_penalty = 0.0
    if (
        config.quality_soft_floor is not None
        and candidate.quality_score < config.quality_soft_floor
    ):
        # Normalize by the configured floor so the maximum penalty remains the
        # reviewable config value.  A quadratic shoulder keeps near-floor
        # measurements from causing source churn while strongly discouraging
        # genuinely soft/compressed shots.
        deficit = (
            config.quality_soft_floor - candidate.quality_score
        ) / max(config.quality_soft_floor, 1e-9)
        quality_floor_penalty = config.quality_soft_floor_penalty * deficit**2
    base = (
        config.quality_weight * candidate.quality_score
        + config.sync_weight * candidate.sync_score
        + config.composition_weight * candidate.composition_score
        + config.boundary_weight * candidate.boundary_score
        - quality_floor_penalty
    )
    if mode is TimelineMode.GENERAL:
        if config.group_wide_frequency_preference_weight or config.group_wide_distance_preference_weight:
            from .shot_proximity import observed_shot_proximity
            proximity=observed_shot_proximity(candidate)
            penalty=(config.group_wide_frequency_preference_weight*proximity.wide_fraction
                +config.group_wide_distance_preference_weight*(proximity.distant_wide_cost
                    +proximity.wide_height_uncertainty_fraction))
            if config.quality_soft_floor is not None:
                # Editorial proximity must not spend away the quality margin
                # above the existing soft floor. Otherwise a blurry close-up
                # can win solely because an excellent wide pays both taste
                # costs. Scale the whole preference, retaining nearer-vs-far
                # ordering; do not clamp all wide shots to the same score or
                # introduce a new hard exclusion around the quality floor.
                maximum_penalty=(config.group_wide_frequency_preference_weight
                    +config.group_wide_distance_preference_weight)
                quality_margin=config.quality_weight*max(0.,candidate.quality_score-config.quality_soft_floor)
                penalty*=min(1.,quality_margin/maximum_penalty)
            base-=penalty
        return (base, False)

    assert target is not None
    if candidate.is_safe_for_member(target, config.minimum_member_confidence):
        return (
            base
            + config.target_weight * candidate.target_confidence
            + config.crop_stability_weight * candidate.crop_stability_score,
            False,
        )
    if candidate.is_group_wide:
        # A verified group-wide shot is the only low-confidence fallback.  An
        # unrelated member close-up is intentionally ineligible even if its
        # generic visual score is excellent.
        return (base - config.group_wide_fallback_penalty, True)
    return None


def _has_exit_boundary_descriptor(candidate: TimelineCandidate) -> bool:
    """Whether the caller supplied boundary-specific outgoing evidence."""

    return bool(
        candidate.exit_shot_scale is not None
        or candidate.exit_shot_type is not None
        or candidate.exit_scene_cut_near
        or candidate.exit_focus_bbox_xywh is not None
        or candidate.exit_subject_bbox_xywh is not None
        or candidate.exit_subject_centroid_xy is not None
        or candidate.exit_pose_embedding
        or candidate.exit_structure_embedding
        or candidate.exit_silhouette_embedding
        or candidate.exit_person_layout_embedding
        or candidate.exit_formation_people
        or candidate.exit_face_identity_embedding
        or candidate.exit_color_embedding
        or candidate.exit_face_geometry_window
        or candidate.exit_face_geometry_exact
        or candidate.exit_motion is not None
    )


def _has_entry_boundary_descriptor(candidate: TimelineCandidate) -> bool:
    """Whether the caller supplied boundary-specific incoming evidence."""

    return bool(
        candidate.entry_shot_scale is not None
        or candidate.entry_shot_type is not None
        or candidate.entry_scene_cut_near
        or candidate.entry_focus_bbox_xywh is not None
        or candidate.entry_subject_bbox_xywh is not None
        or candidate.entry_subject_centroid_xy is not None
        or candidate.entry_pose_embedding
        or candidate.entry_structure_embedding
        or candidate.entry_silhouette_embedding
        or candidate.entry_person_layout_embedding
        or candidate.entry_formation_people
        or candidate.entry_face_identity_embedding
        or candidate.entry_color_embedding
        or candidate.entry_face_geometry_window
        or candidate.entry_face_geometry_exact
        or candidate.entry_motion is not None
    )


def _boundary_side_stability_allowed(
    evidence: BoundarySideStability | None,
    config: TimelineConfig,
) -> bool:
    if evidence is None:
        return not config.require_cross_source_boundary_side_stability
    return not boundary_side_stability_exceeded(
        evidence,
        maximum_scale_delta=config.maximum_boundary_side_scale_delta,
        maximum_structure_mismatch=(
            config.maximum_boundary_side_structure_mismatch
        ),
        maximum_pose_mismatch=config.maximum_boundary_side_pose_mismatch,
        maximum_shot_type_mismatch=(
            config.maximum_boundary_side_shot_type_mismatch
        ),
    )


def _cross_source_transition_rejection_reasons(
    previous: TimelineCandidate,
    current: TimelineCandidate,
    config: TimelineConfig,
) -> tuple[str, ...]:
    """Return every hard-gate reason that rejects one cross-source edge."""

    reasons: list[str] = []
    if not _boundary_side_stability_allowed(
        previous.exit_boundary_side_stability,
        config,
    ):
        reasons.append("outgoing_boundary_side_instability")
    if not _boundary_side_stability_allowed(
        current.entry_boundary_side_stability,
        config,
    ):
        reasons.append("incoming_boundary_side_instability")
    evidence_available = False
    if previous.exit_scene_cut_near or current.entry_scene_cut_near:
        reasons.append("source_local_camera_cut_near_boundary")
    previous_scale = (
        previous.exit_shot_scale
        if previous.exit_shot_scale is not None
        else previous.shot_scale
    )
    current_scale = (
        current.entry_shot_scale
        if current.entry_shot_scale is not None
        else current.shot_scale
    )
    scale_delta = abs(previous_scale - current_scale)
    if (
        config.maximum_cross_source_scale_delta is not None
        and scale_delta > config.maximum_cross_source_scale_delta
    ):
        reasons.append("scale_delta")

    # ``entry/exit_shot_scale`` doubles as a compatibility marker for callers
    # that provide full boundary descriptors.  Once that marker is present, a
    # missed boundary detector must stay missing: falling back to the segment
    # midpoint can be seconds away and would falsely certify a visually unknown
    # cut as safe. Legacy callers that only populate midpoint fields retain the
    # old fallback behavior.
    previous_has_boundary_descriptor = _has_exit_boundary_descriptor(previous)
    current_has_boundary_descriptor = _has_entry_boundary_descriptor(current)
    previous_focus = (
        previous.exit_focus_bbox_xywh
        if previous_has_boundary_descriptor
        else previous.focus_bbox_xywh
    )
    current_focus = (
        current.entry_focus_bbox_xywh
        if current_has_boundary_descriptor
        else current.focus_bbox_xywh
    )
    focus_mismatch = _bbox_geometry_mismatch(previous_focus, current_focus)
    if focus_mismatch is not None:
        evidence_available = True
        if (
            config.maximum_cross_source_focus_mismatch is not None
            and focus_mismatch > config.maximum_cross_source_focus_mismatch
        ):
            reasons.append("focus_geometry")

    previous_identity = (
        previous.exit_face_identity_embedding
        if previous_has_boundary_descriptor
        else previous.face_identity_embedding
    )
    current_identity = (
        current.entry_face_identity_embedding
        if current_has_boundary_descriptor
        else current.face_identity_embedding
    )
    face_similarity = cosine_similarity(previous_identity, current_identity)
    if face_similarity is not None:
        evidence_available = True
    previous_shot_type = previous.exit_shot_type or previous.shot_type
    current_shot_type = current.entry_shot_type or current.shot_type
    both_close = (
        previous_shot_type is ShotType.CLOSE_UP
        and current_shot_type is ShotType.CLOSE_UP
    )
    if config.minimum_cross_source_face_similarity is not None and both_close:
        # A close-up switch without same-face evidence is the most noticeable
        # member jump.  Hold the current broadcaster instead of guessing.
        if (
            face_similarity is None
            or face_similarity < config.minimum_cross_source_face_similarity
        ):
            reasons.append("close_up_face_identity")

    layout_mismatch = cross_source_layout_mismatch(previous, current)
    if layout_mismatch is not None:
        evidence_available = True
        if (
            config.maximum_cross_source_layout_mismatch is not None
            and layout_mismatch > config.maximum_cross_source_layout_mismatch
        ):
            reasons.append("person_layout")

    previous_formation = (
        previous.exit_formation_people
        if previous_has_boundary_descriptor
        else previous.formation_people
    )
    current_formation = (
        current.entry_formation_people
        if current_has_boundary_descriptor
        else current.formation_people
    )
    formation_mismatch = comparable_group_formation_mismatch(
        previous_formation,
        current_formation,
        left_shot_type=previous_shot_type,
        right_shot_type=current_shot_type,
    )
    if formation_mismatch is not None:
        evidence_available = True
        if (
            config.maximum_cross_source_formation_mismatch is not None
            and formation_mismatch
            > config.maximum_cross_source_formation_mismatch
        ):
            reasons.append("formation")

    previous_silhouette = (
        previous.exit_silhouette_embedding
        if previous_has_boundary_descriptor
        else previous.silhouette_embedding
    )
    current_silhouette = (
        current.entry_silhouette_embedding
        if current_has_boundary_descriptor
        else current.silhouette_embedding
    )
    silhouette_mismatch = (
        _normalised_distance(previous_silhouette, current_silhouette)
        if previous_silhouette and current_silhouette
        else None
    )
    if silhouette_mismatch is not None:
        evidence_available = True
        if (
            config.maximum_cross_source_silhouette_mismatch is not None
            and silhouette_mismatch
            > config.maximum_cross_source_silhouette_mismatch
        ):
            reasons.append("silhouette")
    previous_pose = (
        previous.exit_pose_embedding
        if previous_has_boundary_descriptor
        else previous.pose_embedding
    )
    current_pose = (
        current.entry_pose_embedding
        if current_has_boundary_descriptor
        else current.pose_embedding
    )
    pose_mismatch = _optional_normalised_distance(previous_pose, current_pose)
    if pose_mismatch is None:
        previous_structure = (
            previous.exit_structure_embedding
            if previous_has_boundary_descriptor
            else previous.structure_embedding
        )
        current_structure = (
            current.entry_structure_embedding
            if current_has_boundary_descriptor
            else current.structure_embedding
        )
        pose_mismatch = _optional_normalised_distance(
            previous_structure,
            current_structure,
        )
    if config.allow_intentional_phrase_anchor_transitions:
        authorized = intentional_phrase_anchor_authorized_mismatches(
            anchor_basis=current.intentional_transition_anchor_basis,
            scale_mismatch=scale_delta,
            focus_mismatch=focus_mismatch,
            layout_mismatch=layout_mismatch,
            silhouette_mismatch=silhouette_mismatch,
            formation_mismatch=formation_mismatch,
            pose_mismatch=pose_mismatch,
        )
        if "scale" in authorized:
            reasons = [reason for reason in reasons if reason != "scale_delta"]
        if "focus" in authorized:
            reasons = [
                reason for reason in reasons if reason != "focus_geometry"
            ]
    if not evidence_available and config.require_cross_source_visual_evidence:
        reasons.append("missing_visual_evidence")
    return tuple(reasons)


def _cross_source_transition_allowed(
    previous: TimelineCandidate,
    current: TimelineCandidate,
    config: TimelineConfig,
) -> bool:
    """Reject attention-jumping cross-source edges before Viterbi scoring."""

    return not _cross_source_transition_rejection_reasons(
        previous,
        current,
        config,
    )


def _transition_score(
    previous: TimelineCandidate,
    current: TimelineCandidate,
    previous_run_length: int,
    config: TimelineConfig,
) -> tuple[float, int] | None:
    same_source = previous.source_id == current.source_id
    same_continuous_shot = same_source and (
        (not previous.shot_id and not current.shot_id)
        or (previous.shot_id and previous.shot_id == current.shot_id)
    )
    same_source_native_cut = bool(
        same_source
        and previous.shot_id
        and current.shot_id
        and previous.shot_id != current.shot_id
    )
    if same_continuous_shot:
        run_length = previous_run_length + 1
        score = config.continuity_bonus
        if run_length > config.preferred_max_hold_segments:
            score -= config.long_hold_penalty * (
                run_length - config.preferred_max_hold_segments
            )
    elif same_source_native_cut:
        # The broadcaster already made this edit inside one source. Preserve
        # the source-continuity reward, but apply neither a cut/early penalty
        # nor visual mismatch penalties as if the cross-cut engine introduced
        # it. The shot run still resets, while the separate same-source run
        # remains uninterrupted and therefore cannot evade source-rotation caps.
        run_length = 1
        score = config.continuity_bonus
    else:
        if not same_source and not _cross_source_transition_allowed(
            previous,
            current,
            config,
        ):
            return None
        run_length = 1
        score = -config.cut_penalty
        if previous_run_length < config.minimum_hold_segments:
            score -= config.early_cut_penalty * (
                config.minimum_hold_segments - previous_run_length
            )
        previous_has_boundary_descriptor = _has_exit_boundary_descriptor(previous)
        current_has_boundary_descriptor = _has_entry_boundary_descriptor(current)
        # An observed boundary with a detector miss must remain missing. The
        # segment midpoint describes a different moment and can otherwise
        # overturn the exact-frame score used by the joint solver.
        previous_pose = (
            previous.exit_pose_embedding
            if previous_has_boundary_descriptor
            else previous.pose_embedding
        )
        current_pose = (
            current.entry_pose_embedding
            if current_has_boundary_descriptor
            else current.pose_embedding
        )
        # Never compare an actual landmark vector against the unrelated
        # gradient-HOG space. If either pose model missed, compare the same
        # deterministic structure descriptor used by the previous engine.
        if not previous_pose or not current_pose or len(previous_pose) != len(current_pose):
            previous_structure = (
                previous.exit_structure_embedding
                if previous_has_boundary_descriptor
                else previous.structure_embedding
            )
            current_structure = (
                current.entry_structure_embedding
                if current_has_boundary_descriptor
                else current.structure_embedding
            )
            previous_pose = previous_structure
            current_pose = current_structure
        score -= config.pose_transition_weight * _normalised_distance(
            previous_pose, current_pose
        )
        previous_silhouette = (
            previous.exit_silhouette_embedding
            if previous_has_boundary_descriptor
            else previous.silhouette_embedding
        )
        current_silhouette = (
            current.entry_silhouette_embedding
            if current_has_boundary_descriptor
            else current.silhouette_embedding
        )
        score -= config.silhouette_transition_weight * _normalised_distance(
            previous_silhouette,
            current_silhouette,
        )
        arm_motion = selection_arm_cost(previous.exit_arm_window, current.entry_arm_window)
        score -= config.motion_transition_weight * (arm_motion.cost if arm_motion is not None else _motion_mismatch(
            previous.exit_motion
            if previous_has_boundary_descriptor
            else previous.motion,
            current.entry_motion
            if current_has_boundary_descriptor
            else current.motion,
        ))
        previous_scale = (
            previous.exit_shot_scale
            if previous.exit_shot_scale is not None
            else previous.shot_scale
        )
        current_scale = (
            current.entry_shot_scale
            if current.entry_shot_scale is not None
            else current.shot_scale
        )
        score -= config.scale_transition_weight * abs(
            previous_scale - current_scale
        )
        score -= config.shot_type_transition_weight * _shot_type_mismatch(
            previous.exit_shot_type or previous.shot_type,
            current.entry_shot_type or current.shot_type,
        )
        previous_focus = (
            previous.exit_focus_bbox_xywh
            if previous_has_boundary_descriptor
            else previous.focus_bbox_xywh
        )
        current_focus = (
            current.entry_focus_bbox_xywh
            if current_has_boundary_descriptor
            else current.focus_bbox_xywh
        )
        focus_mismatch = _bbox_geometry_mismatch(previous_focus, current_focus)
        if focus_mismatch is not None:
            score -= config.focus_position_transition_weight * focus_mismatch
        elif (previous_focus is None) != (current_focus is None) and (
            previous.shot_type is ShotType.CLOSE_UP
            or current.shot_type is ShotType.CLOSE_UP
        ):
            # A close-up with no comparable incoming/outgoing focal subject is
            # exactly the viewer-attention jump that makes a synced cut feel
            # like a random compilation.
            score -= config.focus_position_transition_weight * 0.72
        previous_subject = (
            previous.exit_subject_bbox_xywh
            if previous_has_boundary_descriptor
            else previous.subject_bbox_xywh
        )
        current_subject = (
            current.entry_subject_bbox_xywh
            if current_has_boundary_descriptor
            else current.subject_bbox_xywh
        )
        subject_mismatch = _bbox_geometry_mismatch(
            previous_subject,
            current_subject,
            left_centroid=(
                previous.exit_subject_centroid_xy
                if previous_has_boundary_descriptor
                else previous.subject_centroid_xy
            ),
            right_centroid=(
                current.entry_subject_centroid_xy
                if current_has_boundary_descriptor
                else current.subject_centroid_xy
            ),
        )
        if subject_mismatch is not None:
            score -= config.subject_geometry_transition_weight * subject_mismatch
        score -= config.person_count_transition_weight * _person_count_mismatch(
            (
                previous.exit_detected_person_count
                if previous_has_boundary_descriptor
                else previous.detected_person_count
            ),
            (
                current.entry_detected_person_count
                if current_has_boundary_descriptor
                else current.detected_person_count
            ),
        )
        previous_layout = (
            previous.exit_person_layout_embedding
            if previous_has_boundary_descriptor
            else previous.person_layout_embedding
        )
        current_layout = (
            current.entry_person_layout_embedding
            if current_has_boundary_descriptor
            else current.person_layout_embedding
        )
        score -= config.person_layout_transition_weight * _normalised_distance(
            previous_layout,
            current_layout,
        )
        previous_formation = (
            previous.exit_formation_people
            if previous_has_boundary_descriptor
            else previous.formation_people
        )
        current_formation = (
            current.entry_formation_people
            if current_has_boundary_descriptor
            else current.formation_people
        )
        formation_mismatch = comparable_group_formation_mismatch(
            previous_formation,
            current_formation,
            left_shot_type=(
                previous.exit_shot_type or previous.shot_type
            ),
            right_shot_type=(
                current.entry_shot_type or current.shot_type
            ),
        )
        if formation_mismatch is not None:
            score -= config.formation_transition_weight * formation_mismatch
        previous_identity = (
            previous.exit_face_identity_embedding
            if previous_has_boundary_descriptor
            else previous.face_identity_embedding
        )
        current_identity = (
            current.entry_face_identity_embedding
            if current_has_boundary_descriptor
            else current.face_identity_embedding
        )
        face_similarity = cosine_similarity(previous_identity, current_identity)
        if face_similarity is not None:
            identity_mismatch = max(0.0, 0.55 - face_similarity) / 0.55
            score -= config.face_identity_transition_weight * identity_mismatch
        previous_color = (
            previous.exit_color_embedding
            if previous_has_boundary_descriptor
            else previous.color_embedding
        )
        current_color = (
            current.entry_color_embedding
            if current_has_boundary_descriptor
            else current.color_embedding
        )
        score -= config.color_transition_weight * _normalised_distance(
            previous_color,
            current_color,
        )
        if previous.exit_face_geometry_exact and current.entry_face_geometry_exact:
            face_mismatch = _face_geometry_mismatch(previous.exit_face_geometry_exact,
                                                   current.entry_face_geometry_exact)
        else:
            face_mismatch = face_geometry_minimax_mismatch(previous.exit_face_geometry_window,
                                                          current.entry_face_geometry_window)
        score -= config.face_geometry_transition_weight * face_mismatch
    return (score, run_length)


def timeline_transition_hard_gates_admit(
    previous: TimelineCandidate,
    current: TimelineCandidate,
    *,
    config: TimelineConfig | None = None,
) -> bool:
    """Return whether one edge survives the selector's visual hard gates.

    Hold length only changes a soft early-cut/continuity score, so a fixed
    admitted run length is sufficient for graph reachability diagnostics.
    """

    settings = config or TimelineConfig()
    return _transition_score(
        previous,
        current,
        max(1, settings.minimum_hold_segments),
        settings,
    ) is not None


def _normalise_segments(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
) -> tuple[tuple[TimelineCandidate, ...], ...]:
    if isinstance(candidates_by_segment, Mapping):
        ordered_keys = sorted(candidates_by_segment)
        if ordered_keys and ordered_keys != list(
            range(ordered_keys[0], ordered_keys[0] + len(ordered_keys))
        ):
            raise ValueError("timeline segment indices must be contiguous")
        groups = tuple(tuple(candidates_by_segment[key]) for key in ordered_keys)
    else:
        groups = tuple(tuple(group) for group in candidates_by_segment)
    if not groups:
        raise ValueError("at least one timeline segment is required")
    for group in groups:
        if not group:
            raise ValueError("each timeline segment needs at least one candidate")
        expected = group[0].segment_index
        if any(candidate.segment_index != expected for candidate in group):
            raise ValueError("all candidates in a group must share segment_index")
        canonical_span = (group[0].start_seconds, group[0].end_seconds)
        if any(
            abs(candidate.start_seconds - canonical_span[0]) > 1e-7
            or abs(candidate.end_seconds - canonical_span[1]) > 1e-7
            for candidate in group
        ):
            raise ValueError("all candidates in a group must share canonical span")
    indices = [group[0].segment_index for group in groups]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("candidate groups must be ordered by contiguous segment_index")
    return groups


def _timeline_lattice_identity(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
) -> str:
    segments = _normalise_segments(candidates_by_segment)
    payload = {
        "contract": "timeline_no_good_candidate_lattice_v1",
        "segments": [
            {
                "segment_index": segment[0].segment_index,
                "start_seconds": segment[0].start_seconds,
                "end_seconds": segment[0].end_seconds,
                "candidate_edges": [
                    {
                        "source_id": candidate.source_id,
                        "shot_id": candidate.shot_id,
                    }
                    for candidate in segment
                ],
            }
            for segment in segments
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_timeline_no_good_constraint_index(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
) -> TimelineNoGoodConstraintIndex:
    """Validate exact no-goods against a lattice once per optimizer round."""

    segments = _normalise_segments(candidates_by_segment)
    no_goods = (
        forbidden_boundary_edges
        if isinstance(forbidden_boundary_edges, TimelineNoGoodSet)
        else TimelineNoGoodSet(tuple(forbidden_boundary_edges))
    )
    if no_goods:
        graph_edges = {
            build_timeline_no_good_edge_key(
                boundary_index,
                outgoing,
                incoming,
            )
            for boundary_index, (outgoing_group, incoming_group) in enumerate(
                zip(segments, segments[1:])
            )
            for outgoing in outgoing_group
            for incoming in incoming_group
        }
        unknown_no_goods = tuple(edge for edge in no_goods if edge not in graph_edges)
        if unknown_no_goods:
            raise ValueError(
                "forbidden boundary edge does not belong to the current "
                f"timeline graph: {unknown_no_goods[0].to_dict()}"
            )
    return TimelineNoGoodConstraintIndex(
        lattice_identity=_timeline_lattice_identity(segments),
        no_goods=no_goods,
    )


def _resolved_preferred_sources(
    segments: tuple[tuple[TimelineCandidate, ...], ...],
    emissions: Sequence[Mapping[int, tuple[float, bool]]],
    requested: tuple[str | None, ...],
    config: TimelineConfig,
) -> tuple[str | None, ...]:
    """Rematch safe metadata preferences when their original slot is unavailable.

    The metadata draft orders sources in full first-use cycles.  Frame decoding
    or local visual quality can make one preferred source unsuitable at its
    original segment, so each complete cycle is treated as a deterministic
    bipartite matching problem.  Only source/segment edges within the configured
    score sacrifice are considered.  A genuinely weak source therefore has no
    edge and is omitted instead of being forced into the edit.
    """

    source_order = tuple(
        dict.fromkeys(source for source in requested if source is not None)
    )
    if not source_order:
        return requested
    resolved: list[str | None] = [None] * len(segments)

    def source_score(position: int, source_id: str) -> float | None:
        scores = [
            emissions[position][candidate_index][0]
            for candidate_index in emissions[position]
            if segments[position][candidate_index].source_id == source_id
        ]
        return max(scores) if scores else None

    def is_comparable(position: int, source_id: str) -> bool:
        score = source_score(position, source_id)
        if score is None:
            return False
        best = max(value for value, _fallback in emissions[position].values())
        return score >= best - config.preferred_source_max_score_sacrifice

    # Coverage is a whole-song objective.  Repeating a forced ten-source cycle
    # every few seconds overwhelmed body-layout continuity in real edits.  Give
    # each source one well-spaced anchor across the complete song instead; the
    # Viterbi pass may extend that source around the anchor to enter/leave on a
    # visually safer boundary.
    target_position: dict[str, int] = {}
    for rank, source_id in enumerate(source_order):
        if len(source_order) == 1:
            position = len(segments) // 2
        else:
            position = round(rank * (len(segments) - 1) / (len(source_order) - 1))
        target_position[source_id] = position

    options: dict[str, tuple[int, ...]] = {}
    for source_id in source_order:
        ranked: list[tuple[float, int, int]] = []
        target = target_position[source_id]
        for position in range(len(segments)):
            score = source_score(position, source_id)
            if score is None or not is_comparable(position, source_id):
                continue
            best = max(value for value, _fallback in emissions[position].values())
            ranked.append(
                (
                    best - score,
                    abs(position - target),
                    position,
                )
            )
        options[source_id] = tuple(item[-1] for item in sorted(ranked))

    owner_by_position: dict[int, str] = {}

    def assign(source_id: str, visited: set[int]) -> bool:
        for position in options[source_id]:
            if position in visited:
                continue
            visited.add(position)
            owner = owner_by_position.get(position)
            if owner is None or assign(owner, visited):
                owner_by_position[position] = source_id
                return True
        return False

    source_rank = {source_id: rank for rank, source_id in enumerate(source_order)}
    for source_id in sorted(
        source_order,
        key=lambda value: (len(options[value]), source_rank[value]),
    ):
        if options[source_id]:
            assign(source_id, set())
    for position, source_id in owner_by_position.items():
        resolved[position] = source_id

    return tuple(resolved)


def select_timeline(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
    emission_adjustments: Mapping[tuple[int, str], float] | None = None,
    source_switch_bonus_by_segment: Mapping[int, float] | None = None,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
    compiled_no_good_constraints: TimelineNoGoodConstraintIndex | None = None,
) -> TimelineResult:
    """Select one candidate per segment with global DP/Viterbi optimisation.

    Raises ``ValueError`` when a member segment contains neither a confident
    target crop nor an explicit group-wide fallback.  This is preferable to
    emitting a plausible-looking crop of the wrong IVE member.
    """

    if isinstance(mode, TimelineMode):
        timeline_mode = mode
    elif isinstance(mode, EditMode):
        timeline_mode = (
            TimelineMode.MEMBER if mode is EditMode.MEMBER else TimelineMode.GENERAL
        )
    else:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == EditMode.GROUP.value:
            normalized_mode = TimelineMode.GENERAL.value
        timeline_mode = TimelineMode(normalized_mode)
    target = IVEMember.parse(target_member) if target_member is not None else None
    if timeline_mode is TimelineMode.MEMBER and target is None:
        raise ValueError("target_member is required in member mode")
    settings = config or TimelineConfig()
    broadcast_slot_by_source = (
        dict(settings.source_broadcast_slots)
        if timeline_mode is TimelineMode.GENERAL else {}
    )
    segments = _normalise_segments(candidates_by_segment)
    supplied_no_goods = (
        forbidden_boundary_edges
        if isinstance(forbidden_boundary_edges, TimelineNoGoodSet)
        else TimelineNoGoodSet(tuple(forbidden_boundary_edges))
    )
    if compiled_no_good_constraints is None:
        compiled_constraints = compile_timeline_no_good_constraint_index(
            segments,
            supplied_no_goods,
        )
    else:
        if not isinstance(
            compiled_no_good_constraints,
            TimelineNoGoodConstraintIndex,
        ):
            raise TypeError(
                "compiled_no_good_constraints must be a "
                "TimelineNoGoodConstraintIndex"
            )
        compiled_no_good_constraints.assert_matches(segments)
        if supplied_no_goods and (
            supplied_no_goods != compiled_no_good_constraints.no_goods
        ):
            raise ValueError(
                "compiled and supplied timeline no-good constraints disagree"
            )
        compiled_constraints = compiled_no_good_constraints
    no_goods = compiled_constraints.no_goods
    if settings.maximum_same_source_hold_seconds is not None:
        oversized_segment = next(
            (
                segment[0]
                for segment in segments
                if segment[0].end_seconds - segment[0].start_seconds
                > settings.maximum_same_source_hold_seconds
            ),
            None,
        )
        if oversized_segment is not None:
            raise ValueError(
                f"segment {oversized_segment.segment_index} is longer than the "
                "maximum same-source hold seconds"
            )
    adjustments = dict(emission_adjustments or {})
    for key, value in adjustments.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or not isinstance(key[1], str)
        ):
            raise ValueError(
                "emission adjustment keys must be (segment_index, source_id)"
            )
        if not isfinite(float(value)) or float(value) > 0.0:
            raise ValueError("emission adjustments must be finite non-positive values")
    switch_bonus_overrides = dict(source_switch_bonus_by_segment or {})
    valid_segment_indices = {
        segment[0].segment_index for segment in segments
    }
    for segment_index, value in switch_bonus_overrides.items():
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index not in valid_segment_indices
        ):
            raise ValueError(
                "source-switch bonus override keys must be timeline segment indices"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "source-switch bonus overrides must be finite and non-negative"
            )
    ordered_source_ids = tuple(
        sorted({candidate.source_id for segment in segments for candidate in segment})
    )
    source_rank_by_id = {
        source_id: rank for rank, source_id in enumerate(ordered_source_ids)
    }
    preferred_sources = (
        tuple(preferred_source_ids) if preferred_source_ids is not None else None
    )
    if preferred_sources is not None:
        if timeline_mode is not TimelineMode.GENERAL:
            raise ValueError("preferred source assignments are only supported in general mode")
        if len(preferred_sources) != len(segments):
            raise ValueError("preferred source assignments must match timeline segments")

    emissions: list[dict[int, tuple[float, bool]]] = []
    for segment in segments:
        eligible: dict[int, tuple[float, bool]] = {}
        for candidate_index, candidate in enumerate(segment):
            emission = _candidate_emission(
                candidate,
                mode=timeline_mode,
                target=target,
                config=settings,
            )
            if emission is not None:
                score, used_fallback = emission
                eligible[candidate_index] = (
                    score
                    + float(
                        adjustments.get(
                            (candidate.segment_index, candidate.source_id),
                            0.0,
                        )
                    ),
                    used_fallback,
                )
        if not eligible:
            raise ValueError(
                f"segment {segment[0].segment_index} has no safe candidate for "
                f"{target.value if target else timeline_mode.value}"
            )
        emissions.append(eligible)

    if preferred_sources is not None:
        resolved_preferences = _resolved_preferred_sources(
            segments,
            emissions,
            preferred_sources,
            settings,
        )
        for position, preferred_source in enumerate(resolved_preferences):
            if preferred_source is None:
                continue
            for candidate_index, (score, used_fallback) in tuple(
                emissions[position].items()
            ):
                if segments[position][candidate_index].source_id == preferred_source:
                    emissions[position][candidate_index] = (
                        score + settings.preferred_source_bonus,
                        used_fallback,
                    )

    # State carries the continuous-shot run, the same-source run and the source
    # used immediately before that run.  The last value lets the global pass
    # discourage A -> B -> A ping-pong without retaining the whole history.
    # A camera cut inside one source resets the former while preserving the
    # same-source continuity reward because it was authored by the broadcaster,
    # not this edit. It does not reset the latter, so it cannot bypass an
    # opted-in source-rotation policy.
    layers: list[dict[tuple[int, int, int, int], _Node]] = []
    first_layer: dict[tuple[int, int, int, int], _Node] = {}
    for candidate_index, (emission_score, used_fallback) in emissions[0].items():
        first_layer[(candidate_index, 1, 1, -1)] = _Node(
            cumulative_score=emission_score,
            previous_state=None,
            emission_score=emission_score,
            transition_score=0.0,
            used_fallback=used_fallback,
        )
    layers.append(first_layer)

    for segment_position in range(1, len(segments)):
        previous_segment = segments[segment_position - 1]
        current_segment = segments[segment_position]
        previous_layer = layers[-1]
        current_layer: dict[tuple[int, int, int, int], _Node] = {}
        rejection_counts: dict[str, int] = {}
        # Source-run/history states can share this identical visual edge.
        # The score depends only on the fixed pair, shot-run and config.
        transition_cache: dict[tuple[int, int, int], tuple[float, int] | None] = {}

        def record_rejection(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        for previous_state, previous_node in previous_layer.items():
            (
                previous_index,
                previous_shot_run,
                previous_source_run,
                source_before_previous_run,
            ) = previous_state
            previous_candidate = previous_segment[previous_index]
            local_options: list[
                tuple[int, float, bool, float, int, int, int]
            ] = []
            for current_index, (emission_score, used_fallback) in emissions[
                segment_position
            ].items():
                current_candidate = current_segment[current_index]
                if no_goods and build_timeline_no_good_edge_key(
                    segment_position - 1,
                    previous_candidate,
                    current_candidate,
                ) in no_goods:
                    record_rejection("exact_boundary_no_good")
                    continue
                transition_key = (previous_index, current_index, previous_shot_run)
                if transition_key not in transition_cache:
                    transition_cache[transition_key] = _transition_score(
                        previous_candidate,
                        current_candidate,
                        previous_shot_run,
                        settings,
                    )
                transition_result = transition_cache[transition_key]
                if transition_result is None:
                    for reason in _cross_source_transition_rejection_reasons(
                        previous_candidate,
                        current_candidate,
                        settings,
                    ):
                        record_rejection(reason)
                    continue
                transition_score, shot_run = transition_result
                source_run = (
                    previous_source_run + 1
                    if previous_candidate.source_id == current_candidate.source_id
                    else 1
                )
                source_before_run = (
                    source_before_previous_run
                    if previous_candidate.source_id == current_candidate.source_id
                    else source_rank_by_id[previous_candidate.source_id]
                )
                local_options.append(
                    (
                        current_index,
                        emission_score,
                        used_fallback,
                        transition_score,
                        shot_run,
                        source_run,
                        source_before_run,
                    )
                )

            for (
                current_index,
                emission_score,
                used_fallback,
                transition_score,
                shot_run,
                source_run,
                source_before_run,
            ) in local_options:
                current_candidate = current_segment[current_index]
                same_source = previous_candidate.source_id == current_candidate.source_id
                # Comparable means the alternate picture itself is not a
                # material quality sacrifice.  Including the fixed hold bonus
                # and cut penalty here made the test self-referential: after a
                # long hold, an otherwise identical safe camera could never be
                # considered comparable precisely because switching is a cut.
                # Cross-source hard gates have already rejected unsafe edges;
                # use emission quality for the adaptive cadence escape hatch.
                comparable_alternate_exists = any(
                    previous_candidate.source_id
                    != current_segment[alternate_index].source_id
                    and alternate_emission
                    >= emission_score - settings.maximum_switch_score_sacrifice
                    for (
                        alternate_index,
                        alternate_emission,
                        _alternate_fallback,
                        alternate_transition,
                        _alternate_shot_run,
                        _alternate_source_run,
                        _alternate_source_before_run,
                    ) in local_options
                )
                source_run_start_position = segment_position - source_run + 1
                source_run_seconds = (
                    current_candidate.end_seconds
                    - segments[source_run_start_position][0].start_seconds
                )
                if (
                    same_source
                    and settings.maximum_same_source_hold_segments is not None
                    and source_run > settings.maximum_same_source_hold_segments
                    and comparable_alternate_exists
                ):
                    record_rejection("same_source_segment_hold_cap")
                    continue
                if (
                    same_source
                    and settings.maximum_same_source_hold_seconds is not None
                    and source_run_seconds
                    > settings.maximum_same_source_hold_seconds
                ):
                    record_rejection("same_source_duration_hold_cap")
                    continue

                # Only edges already admitted by _transition_score reach this
                # point, so the switch incentive can never bypass a visual
                # hard gate.  Same-source shot changes are internal cuts and
                # deliberately receive no source-switch cadence bonus.
                cadence_score = (
                    float(
                        switch_bonus_overrides.get(
                            current_candidate.segment_index,
                            settings.source_switch_bonus,
                        )
                    )
                    if not same_source
                    else 0.0
                )
                previous_slot = broadcast_slot_by_source.get(previous_candidate.source_id)
                if (
                    not same_source and previous_slot is not None
                    and previous_slot == broadcast_slot_by_source.get(current_candidate.source_id)
                ):
                    cadence_score *= settings.same_broadcast_switch_bonus_multiplier
                if (
                    same_source
                    and settings.preferred_max_source_hold_segments is not None
                    and source_run > settings.preferred_max_source_hold_segments
                    and comparable_alternate_exists
                ):
                    cadence_score = -settings.source_hold_penalty * (
                        source_run - settings.preferred_max_source_hold_segments
                    )
                if (
                    same_source
                    and settings.preferred_max_source_hold_seconds is not None
                    and source_run_seconds
                    > settings.preferred_max_source_hold_seconds
                    and comparable_alternate_exists
                ):
                    cadence_score -= settings.source_hold_penalty * (
                        source_run_seconds
                        - settings.preferred_max_source_hold_seconds
                    )
                if (
                    not same_source
                    and source_before_previous_run >= 0
                    and current_candidate.source_id
                    == ordered_source_ids[source_before_previous_run]
                ):
                    cadence_score -= settings.ping_pong_penalty
                transition_with_cadence = transition_score + cadence_score
                state = (
                    current_index,
                    shot_run,
                    source_run,
                    source_before_run,
                )
                cumulative = (
                    previous_node.cumulative_score
                    + transition_with_cadence
                    + emission_score
                )
                existing = current_layer.get(state)
                # Deterministic tie-break: prefer the lexicographically earlier
                # predecessor, making repeated synthetic runs byte-for-byte stable.
                if existing is None or cumulative > existing.cumulative_score + 1e-12:
                    current_layer[state] = _Node(
                        cumulative_score=cumulative,
                        previous_state=previous_state,
                        emission_score=emission_score,
                        transition_score=transition_with_cadence,
                        used_fallback=used_fallback,
                    )
        if not current_layer:
            rejection_summary = ",".join(
                f"{reason}:{count}"
                for reason, count in sorted(rejection_counts.items())
            ) or "unknown:0"
            raise NoCompleteTimelinePath(
                f"segment {current_segment[0].segment_index} has no "
                "cadence-safe candidate; rejection_counts="
                f"{rejection_summary}",
                reason_code="cadence_state_graph_exhausted",
                segment_index=current_segment[0].segment_index,
                rejection_counts=rejection_counts,
            )
        layers.append(current_layer)

    final_state, final_node = max(
        layers[-1].items(),
        key=lambda item: (
            item[1].cumulative_score,
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
        ),
    )
    states: list[tuple[int, int, int, int]] = [final_state]
    for layer_position in range(len(layers) - 1, 0, -1):
        previous_state = layers[layer_position][states[-1]].previous_state
        assert previous_state is not None
        states.append(previous_state)
    states.reverse()

    choices: list[TimelineChoice] = []
    for segment_position, state in enumerate(states):
        node = layers[segment_position][state]
        choices.append(
            TimelineChoice(
                candidate=segments[segment_position][state[0]],
                emission_score=node.emission_score,
                transition_score=node.transition_score,
                cumulative_score=node.cumulative_score,
                used_group_wide_fallback=node.used_fallback,
            )
        )
    return TimelineResult(
        mode=timeline_mode,
        target_member=target,
        choices=tuple(choices),
        total_score=final_node.cumulative_score,
    )


def select_timeline_candidates(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
    maximum_results: int = 8,
    diversity_penalty: float = 0.10,
    minimum_hamming_fraction: float = 0.035,
    emission_adjustments: Mapping[tuple[int, str], float] | None = None,
    source_switch_bonus_by_segment: Mapping[int, float] | None = None,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
    compiled_no_good_constraints: TimelineNoGoodConstraintIndex | None = None,
) -> tuple[TimelineResult, ...]:
    """Generate several globally safe, meaningfully different edit paths.

    Each pass is a full constrained Viterbi optimisation.  Previously selected
    source/segment pairs receive a small deterministic negative adjustment on
    the next pass, which exposes near-optimal alternatives without weakening
    any visual/member hard gate.  A later reviewer can therefore compare whole
    timelines instead of accepting the first argmax unconditionally.
    """

    if (
        isinstance(maximum_results, bool)
        or not isinstance(maximum_results, int)
        or not 1 <= maximum_results <= 32
    ):
        raise ValueError("maximum_results must be an integer from 1 to 32")
    if not isfinite(diversity_penalty) or diversity_penalty <= 0.0:
        raise ValueError("diversity_penalty must be positive and finite")
    if (
        not isfinite(minimum_hamming_fraction)
        or not 0.0 <= minimum_hamming_fraction <= 1.0
    ):
        raise ValueError("minimum_hamming_fraction must be between zero and one")

    segments = _normalise_segments(candidates_by_segment)
    supplied_no_goods = (
        forbidden_boundary_edges
        if isinstance(forbidden_boundary_edges, TimelineNoGoodSet)
        else TimelineNoGoodSet(tuple(forbidden_boundary_edges))
    )
    if compiled_no_good_constraints is None:
        compiled_constraints = compile_timeline_no_good_constraint_index(
            segments,
            supplied_no_goods,
        )
    else:
        if not isinstance(
            compiled_no_good_constraints,
            TimelineNoGoodConstraintIndex,
        ):
            raise TypeError(
                "compiled_no_good_constraints must be a "
                "TimelineNoGoodConstraintIndex"
            )
        compiled_no_good_constraints.assert_matches(segments)
        if supplied_no_goods and (
            supplied_no_goods != compiled_no_good_constraints.no_goods
        ):
            raise ValueError(
                "compiled and supplied timeline no-good constraints disagree"
            )
        compiled_constraints = compiled_no_good_constraints

    # Preserve the caller's role-conditioned soft penalties on every search
    # pass, then accumulate only the deterministic diversity penalty locally.
    # ``select_timeline`` validates that all values are non-positive and that
    # switch-bonus overrides are finite, non-negative, and segment-bound.
    adjustments: dict[tuple[int, str], float] = dict(
        emission_adjustments or {}
    )
    results: list[TimelineResult] = []
    signatures: list[tuple[str, ...]] = []
    attempts = 0
    maximum_attempts = maximum_results * 4
    while len(results) < maximum_results and attempts < maximum_attempts:
        attempts += 1
        try:
            result = select_timeline(
                segments,
                mode=mode,
                target_member=target_member,
                config=config,
                preferred_source_ids=preferred_source_ids,
                emission_adjustments=adjustments,
                source_switch_bonus_by_segment=(
                    source_switch_bonus_by_segment
                ),
                compiled_no_good_constraints=compiled_constraints,
            )
        except NoCompleteTimelinePath:
            # Diversity penalties deliberately perturb emissions after each
            # accepted path.  They can make a later search infeasible under
            # hard cadence constraints even though earlier results remain
            # valid for the unchanged candidate lattice.  Exhaustion therefore
            # terminates only the alternate search; it must not erase already
            # proven paths.  The first-pass exception still propagates.
            if results:
                break
            raise
        signature = tuple(candidate.source_id for candidate in result.candidates)
        minimum_distance = max(1, ceil(len(signature) * minimum_hamming_fraction))
        sufficiently_different = all(
            sum(left != right for left, right in zip(signature, existing, strict=True))
            >= minimum_distance
            for existing in signatures
        )
        if sufficiently_different:
            results.append(result)
            signatures.append(signature)

        # Penalize even a rejected duplicate. Repeated passes then push the DP
        # far enough away to expose a genuinely different near-optimal path.
        for candidate in result.candidates:
            key = (candidate.segment_index, candidate.source_id)
            # A small segment-dependent phase prevents perfectly symmetric test
            # footage from yielding only one all-A, all-B and all-C path.  It is
            # deterministic and too small to overwhelm actual quality scores.
            phase = (candidate.segment_index * 5 + attempts * 3) % 11
            phased_penalty = diversity_penalty * (0.85 + 0.30 * phase / 10.0)
            adjustments[key] = adjustments.get(key, 0.0) - phased_penalty

    if not results:
        raise NoCompleteTimelinePath(
            "timeline candidate generation produced no result",
            reason_code="timeline_candidate_generation_empty",
        )
    return tuple(results)


# A name that reads naturally at call sites which think in terms of Viterbi.
optimize_timeline = select_timeline


__all__ = [
    "BOUNDARY_SIDE_MAX_POSE_MISMATCH",
    "BOUNDARY_SIDE_MAX_SCALE_DELTA",
    "BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH",
    "BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH",
    "INTENTIONAL_PHRASE_ANCHOR_TRANSITION_BASIS",
    "PRODUCTION_MAX_FOCUS_MISMATCH",
    "PRODUCTION_MAX_FORMATION_MISMATCH",
    "PRODUCTION_MAX_LAYOUT_MISMATCH",
    "PRODUCTION_MAX_POSE_MISMATCH",
    "PRODUCTION_MAX_SCALE_MISMATCH",
    "PRODUCTION_MAX_SILHOUETTE_MISMATCH",
    "BoundarySideStability",
    "NoCompleteTimelinePath",
    "ShotType",
    "TimelineCandidate",
    "TimelineChoice",
    "TimelineConfig",
    "TimelineNoGoodConstraintIndex",
    "TimelineMode",
    "TimelineResult",
    "boundary_side_stability_exceeded",
    "comparable_group_formation_mismatch",
    "compile_timeline_no_good_constraint_index",
    "cross_source_layout_mismatch",
    "face_geometry_minimax_mismatch",
    "formation_assignment_mismatch",
    "intentional_phrase_anchor_authorized_mismatches",
    "optimize_timeline",
    "screen_layout_mismatch",
    "select_timeline",
    "select_timeline_candidates",
]
