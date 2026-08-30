"""Production bridge from the visual lattice to the joint boundary DP.

The visual scorer deliberately retains a transient optimizer workspace.  This
module consumes that workspace without rebuilding descriptors, runs the three
existing professional cadence profiles through the joint source/offset/crop
selector, and keeps the exact decoded proof selected by the DP.

Only scalar identities and reviewer data cross the audit boundary.  Decoded
descriptors, appearance embeddings, and transform-path samples remain attached
to process-local provider materializations and are never serialized here.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .ive import AspectRatio
from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .joint_boundary_provider import (
    BoundaryMaterializationProof,
    DecodedBoundaryOptionProvider,
    JointSelectionMaterializationProof,
    SegmentTransformPathMaterializationProof,
)
from .joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOptionEvidenceScope,
    BoundaryOptionSet,
    BoundarySelectionNoGood,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
    JointTimelineBoundaryProofIdentity,
    JointTimelineCandidateClock,
    JointTimelineProgress,
    JointTimelineProgressCallback,
    JointTimelinePathSignature,
    JointTimelineResult,
    JointTimelineSharedBridgeEvidenceCache,
    select_joint_timeline_candidates,
)
from .timeline import (
    NoCompleteTimelinePath,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineResult,
)
from .timeline_no_good import TimelineNoGoodSet
from .timeline_review import (
    JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY,
    JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT,
    TimelineReview,
    review_timeline,
)
from .render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    aligned_render_segment,
    output_size,
)
from .visual_score import (
    _CandidateSubjectMetadata,
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelineChoice,
    VisualTimelineOptimizerCarrier,
    VisualTimelineOptimizerSeed,
    VisualTimelinePlan,
    _candidate_review_summary,
    _master_clock_time,
    _missing_candidate_face_metadata,
    _missing_candidate_subject_metadata,
    _review_ranked_candidate_indices,
    _role_conditioned_visual_controls,
    _safe_cadence_profiles,
    _source_clock_time,
)
from .visual_boundary_repair import BoundaryVisualEvidence


JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS = (
    "joint_visual_candidate_adapter_v1+"
    "three_existing_safe_cadence_profiles_v1+"
    "exact_source_shot_proof_variant_coalescing_v2+"
    "mathematical_unique_path_floor_minimum_eight_render_variants_v2+"
    "shifted_scene_cut_clock_rematerialization_v1+"
    "workspace_only_optimizer_seed_v1+"
    "selection_scalar_exact_rehydration_v1+"
    "finite_exact_boundary_nogood_retry_v1+"
    "post_transform_result_evidence_projection_v1+"
    "selection_materialization_review_parity_v1+"
    "same_source_exact_renderer_clock_continuity_projection_v2+"
    "authoritative_passing_cohort_retry_v2+"
    "bounded_exact_proof_materialization_v1+"
    "exact_decoded_montage_soft_role_feedback_v1"
)
JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS = (
    "decoded_joint_boundary_exact_renderer_track_editorial_motion_authority_v1"
)
MINIMUM_PRODUCTION_JOINT_CANDIDATES = 8
MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH = 2
MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS = math.ceil(
    MINIMUM_PRODUCTION_JOINT_CANDIDATES
    / MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH
)
MAXIMUM_PROFILE_JOINT_CANDIDATES = 8
_SCENE_CUT_INTERIOR_EPSILON_SECONDS = 1e-6
MAXIMUM_JOINT_VISUAL_RETRY_ROUNDS = 2
MAXIMUM_JOINT_VISUAL_PROOF_CALLS = 32
MINIMUM_DECODED_MONTAGE_FEEDBACK_RENDER_VARIANTS = 2
_DECODED_MONTAGE_SHOT_CLASSES = frozenset(
    {"wide", "medium", "close", "unknown"}
)
_DECODED_MONTAGE_PROMOTABLE_CLASSES = frozenset({"wide", "unknown"})


def _validated_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class JointDecodedMontageSegmentObservation:
    """One exact rendered-segment observation from a bound proxy review.

    The pipeline owns RenderPlan/review binding before constructing this value.
    This adapter additionally binds the observation to one exact optimizer
    candidate clock and requires a distinct rendered-plan identity.  The three
    shot classes correspond to the rendered entry/interior/exit samples.
    """

    candidate_clock: JointTimelineCandidateClock
    render_variant_signature: str
    sampled_proxy_shot_classes: tuple[str, str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_clock, JointTimelineCandidateClock):
            raise TypeError(
                "decoded montage observation requires an exact candidate clock"
            )
        object.__setattr__(
            self,
            "render_variant_signature",
            _validated_sha256(
                self.render_variant_signature,
                "decoded montage render variant signature",
            ),
        )
        samples = tuple(self.sampled_proxy_shot_classes)
        if len(samples) != 3 or any(
            not isinstance(value, str)
            or value not in _DECODED_MONTAGE_SHOT_CLASSES
            for value in samples
        ):
            raise ValueError(
                "decoded montage observation needs three canonical shot classes"
            )
        object.__setattr__(self, "sampled_proxy_shot_classes", samples)

    @classmethod
    def from_candidate(
        cls,
        candidate: TimelineCandidate,
        *,
        render_variant_signature: str,
        sampled_proxy_shot_classes: Sequence[str],
    ) -> "JointDecodedMontageSegmentObservation":
        """Build the public feedback value after caller-side review binding."""

        if not isinstance(candidate, TimelineCandidate):
            raise TypeError(
                "decoded montage observation requires a TimelineCandidate"
            )
        return cls(
            candidate_clock=JointTimelineCandidateClock.from_candidate(candidate),
            render_variant_signature=render_variant_signature,
            sampled_proxy_shot_classes=tuple(sampled_proxy_shot_classes),
        )

    @classmethod
    def from_choice(
        cls,
        choice: TimelineChoice,
        *,
        render_variant_signature: str,
        sampled_proxy_shot_classes: Sequence[str],
    ) -> "JointDecodedMontageSegmentObservation":
        """Bind feedback to an unshifted choice from ``joint_result.timeline``."""

        if not isinstance(choice, TimelineChoice):
            raise TypeError("decoded montage observation requires a TimelineChoice")
        return cls.from_candidate(
            choice.candidate,
            render_variant_signature=render_variant_signature,
            sampled_proxy_shot_classes=sampled_proxy_shot_classes,
        )


@dataclass(frozen=True, slots=True)
class JointDecodedMontageFeedback:
    """Bound same-run decoded evidence used only by soft role controls.

    A single rendered observation can retire its exact complete path upstream,
    but it cannot alter candidate availability here.  Only the same wide or
    unknown sample observed in at least two distinct render variants is stable
    enough to replace stale metadata for the next selector batch.
    """

    observations: tuple[JointDecodedMontageSegmentObservation, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if any(
            not isinstance(item, JointDecodedMontageSegmentObservation)
            for item in observations
        ):
            raise TypeError(
                "decoded montage feedback contains an invalid observation"
            )
        identities = tuple(
            (item.candidate_clock, item.render_variant_signature)
            for item in observations
        )
        if len(set(identities)) != len(identities):
            raise ValueError(
                "decoded montage feedback repeats one clock/render variant"
            )
        object.__setattr__(self, "observations", observations)

    def to_dict(self) -> dict[str, object]:
        clocks = {item.candidate_clock for item in self.observations}
        variants = {item.render_variant_signature for item in self.observations}
        return {
            "contract": "joint_decoded_montage_soft_role_feedback_v1",
            "observation_count": len(self.observations),
            "candidate_clock_count": len(clocks),
            "render_variant_count": len(variants),
            "minimum_distinct_render_variant_count": (
                MINIMUM_DECODED_MONTAGE_FEEDBACK_RENDER_VARIANTS
            ),
            "hard_no_good_authoring_allowed": False,
            "thresholds_lowered": False,
            "raw_frames_persisted": False,
        }


def _empty_subject_metadata(
    *,
    shot_scale_bases: tuple[str, str, str],
) -> _CandidateSubjectMetadata:
    return _CandidateSubjectMetadata(
        pose_backends=("none", "none", "none"),
        pose_confidences=(0.0, 0.0, 0.0),
        person_counts=(0, 0, 0),
        formation_person_counts=(0, 0, 0),
        formation_upper_body_counts=(0, 0, 0),
        shot_scale_bases=shot_scale_bases,
        silhouette_boxes=(None, None, None),
        silhouette_centroids=(None, None, None),
        silhouette_backends=("none", "none", "none"),
        silhouette_confidences=(0.0, 0.0, 0.0),
    )


def _candidate_role_sample_types(
    candidate: TimelineCandidate,
) -> tuple[ShotType, ShotType, ShotType]:
    return (
        candidate.entry_shot_type or candidate.shot_type,
        candidate.shot_type,
        candidate.exit_shot_type or candidate.shot_type,
    )


def _candidate_role_face_boxes(
    candidate: TimelineCandidate,
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    return (
        candidate.entry_focus_bbox_xywh or candidate.focus_bbox_xywh,
        candidate.focus_bbox_xywh,
        candidate.exit_focus_bbox_xywh or candidate.focus_bbox_xywh,
    )


def _canonical_decoded_montage_feedback(
    lattice: Sequence[Sequence[TimelineCandidate]],
    feedback: JointDecodedMontageFeedback | None,
) -> tuple[
    dict[JointTimelineCandidateClock, tuple[str | None, str | None, str | None]],
    dict[str, object],
]:
    """Validate feedback against the exact lattice and promote stable samples."""

    if feedback is not None and not isinstance(
        feedback,
        JointDecodedMontageFeedback,
    ):
        raise TypeError(
            "decoded_montage_feedback must be JointDecodedMontageFeedback or None"
        )
    observations = () if feedback is None else feedback.observations
    lattice_by_clock: dict[JointTimelineCandidateClock, TimelineCandidate] = {}
    source_slot_by_clock: dict[
        JointTimelineCandidateClock,
        tuple[int, str],
    ] = {}
    source_slot_counts: dict[tuple[int, str], int] = {}
    for segment in lattice:
        for candidate in segment:
            clock = JointTimelineCandidateClock.from_candidate(candidate)
            if clock in lattice_by_clock:
                raise ValueError("candidate lattice repeats an exact clock")
            lattice_by_clock[clock] = candidate
            source_slot = (candidate.segment_index, candidate.source_id)
            source_slot_by_clock[clock] = source_slot
            source_slot_counts[source_slot] = (
                source_slot_counts.get(source_slot, 0) + 1
            )

    observed: dict[
        JointTimelineCandidateClock,
        list[dict[str, set[str]]],
    ] = {}
    for item in observations:
        if item.candidate_clock not in lattice_by_clock:
            raise ValueError(
                "decoded montage feedback clock does not belong to the lattice"
            )
        if source_slot_counts[source_slot_by_clock[item.candidate_clock]] != 1:
            raise ValueError(
                "decoded montage feedback cannot target an ambiguous source slot"
            )
        slots = observed.setdefault(
            item.candidate_clock,
            [dict(), dict(), dict()],
        )
        for position, shot_class in enumerate(
            item.sampled_proxy_shot_classes
        ):
            slots[position].setdefault(shot_class, set()).add(
                item.render_variant_signature
            )

    promoted: dict[
        JointTimelineCandidateClock,
        tuple[str | None, str | None, str | None],
    ] = {}
    promoted_counts = {"wide": 0, "unknown": 0}
    insufficient_sample_count = 0
    conflicting_sample_count = 0
    redundant_wide_sample_count = 0
    for clock, sample_slots in observed.items():
        original_samples = _candidate_role_sample_types(lattice_by_clock[clock])
        promoted_samples: list[str | None] = [None, None, None]
        for position, variants_by_class in enumerate(sample_slots):
            supported = tuple(
                sorted(
                    shot_class
                    for shot_class, variants in variants_by_class.items()
                    if shot_class in _DECODED_MONTAGE_PROMOTABLE_CLASSES
                    and len(variants)
                    >= MINIMUM_DECODED_MONTAGE_FEEDBACK_RENDER_VARIANTS
                )
            )
            if len(supported) > 1:
                conflicting_sample_count += 1
                continue
            if not supported:
                if any(
                    shot_class in _DECODED_MONTAGE_PROMOTABLE_CLASSES
                    for shot_class in variants_by_class
                ):
                    insufficient_sample_count += 1
                continue
            shot_class = supported[0]
            if shot_class == "wide" and original_samples[position] in {
                ShotType.WIDE,
                ShotType.GROUP_WIDE,
            }:
                redundant_wide_sample_count += 1
                continue
            promoted_samples[position] = shot_class
            promoted_counts[shot_class] += 1
        if any(value is not None for value in promoted_samples):
            promoted[clock] = tuple(promoted_samples)  # type: ignore[arg-type]

    return promoted, {
        "contract": "joint_decoded_montage_soft_role_feedback_application_v1",
        "feedback_supplied": feedback is not None,
        "observation_count": len(observations),
        "observed_candidate_clock_count": len(observed),
        "promoted_candidate_clock_count": len(promoted),
        "promoted_wide_sample_count": promoted_counts["wide"],
        "promoted_unknown_sample_count": promoted_counts["unknown"],
        "insufficient_distinct_variant_sample_count": insufficient_sample_count,
        "conflicting_repeated_sample_count": conflicting_sample_count,
        "redundant_wide_sample_count": redundant_wide_sample_count,
        "minimum_distinct_render_variant_count": (
            MINIMUM_DECODED_MONTAGE_FEEDBACK_RENDER_VARIANTS
        ),
        "exact_candidate_clock_binding_valid": True,
        "hard_no_good_count": 0,
        "hard_no_good_authoring_allowed": False,
        "optimizer_effect": "existing_non_positive_emission_adjustments_only",
        "thresholds_lowered": False,
        "raw_frames_persisted": False,
    }


def _decoded_montage_role_control_inputs(
    lattice: Sequence[Sequence[TimelineCandidate]],
    subject_metadata_by_candidate: Mapping[str, _CandidateSubjectMetadata],
    feedback: JointDecodedMontageFeedback | None,
) -> tuple[
    tuple[tuple[TimelineCandidate, ...], ...],
    Mapping[str, _CandidateSubjectMetadata],
    dict[str, object],
]:
    """Build a scoring-only lattice without changing provider proof identity."""

    promoted, audit = _canonical_decoded_montage_feedback(lattice, feedback)
    metadata = dict(subject_metadata_by_candidate)
    shadow_segments: list[tuple[TimelineCandidate, ...]] = []
    cold_start_close_demotion_count = 0
    changed_candidate_count = 0
    for segment in lattice:
        shadow_segment: list[TimelineCandidate] = []
        for candidate in segment:
            clock = JointTimelineCandidateClock.from_candidate(candidate)
            sample_types = list(_candidate_role_sample_types(candidate))
            face_boxes = _candidate_role_face_boxes(candidate)
            bases = list(
                metadata[candidate.candidate_id].shot_scale_bases
                if candidate.candidate_id in metadata
                else ("legacy", "legacy", "legacy")
            )
            changed = False
            promoted_samples = promoted.get(clock, (None, None, None))
            for position, decoded_class in enumerate(promoted_samples):
                if decoded_class == "wide":
                    sample_types[position] = ShotType.WIDE
                    changed = True
                elif decoded_class == "unknown":
                    sample_types[position] = ShotType.MEDIUM
                    bases[position] = "unknown_no_subject_geometry"
                    changed = True
            for position, (shot_type, face_box) in enumerate(
                zip(sample_types, face_boxes, strict=True)
            ):
                if shot_type is ShotType.CLOSE_UP and face_box is None:
                    sample_types[position] = ShotType.MEDIUM
                    cold_start_close_demotion_count += 1
                    changed = True
            shadow = (
                replace(
                    candidate,
                    entry_shot_type=sample_types[0],
                    shot_type=sample_types[1],
                    exit_shot_type=sample_types[2],
                )
                if changed
                else candidate
            )
            if bases != list(
                metadata[candidate.candidate_id].shot_scale_bases
                if candidate.candidate_id in metadata
                else ("legacy", "legacy", "legacy")
            ):
                current = metadata.get(candidate.candidate_id)
                metadata[candidate.candidate_id] = (
                    replace(current, shot_scale_bases=tuple(bases))
                    if current is not None
                    else _empty_subject_metadata(
                        shot_scale_bases=tuple(bases)  # type: ignore[arg-type]
                    )
                )
            changed_candidate_count += int(changed)
            shadow_segment.append(shadow)
        shadow_segments.append(tuple(shadow_segment))
    audit.update(
        {
            "cold_start_metadata_close_without_bbox_demoted_sample_count": (
                cold_start_close_demotion_count
            ),
            "role_control_shadow_candidate_count": changed_candidate_count,
            "original_candidate_objects_mutated": False,
            "availability_override_applied": bool(changed_candidate_count),
        }
    )
    return tuple(shadow_segments), MappingProxyType(metadata), audit


def _canonical_external_boundary_no_goods(
    lattice: Sequence[Sequence[TimelineCandidate]],
    values: Collection[BoundarySelectionNoGood],
) -> frozenset[BoundarySelectionNoGood]:
    """Validate process-local exact boundary feedback before any search."""

    if (
        isinstance(values, (str, bytes, Mapping))
        or not isinstance(values, Collection)
    ):
        raise TypeError("forbidden_boundary_selections must be a collection")
    canonical: set[BoundarySelectionNoGood] = set()
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(
                "forbidden_boundary_selections must contain exact "
                "(BoundaryMeasurementKey, frame_offset) tuples"
            )
        key, frame_offset = value
        if not isinstance(key, BoundaryMeasurementKey):
            raise TypeError(
                "forbidden boundary selection key must be a "
                "BoundaryMeasurementKey"
            )
        if isinstance(frame_offset, bool) or not isinstance(frame_offset, int):
            raise TypeError(
                "forbidden boundary selection frame_offset must be an integer"
            )
        if frame_offset not in JOINT_BOUNDARY_FRAME_OFFSETS:
            raise ValueError(
                "forbidden boundary selection frame_offset must be in -8..+8"
            )
        if (
            key.boundary_index >= len(lattice) - 1
            or key.outgoing not in lattice[key.boundary_index]
            or key.incoming not in lattice[key.boundary_index + 1]
        ):
            raise ValueError(
                "forbidden boundary selection does not belong to the "
                "candidate lattice"
            )
        canonical.add((key, frame_offset))
    return frozenset(canonical)


def _canonical_external_complete_path_no_goods(
    lattice: Sequence[Sequence[TimelineCandidate]],
    values: Collection[JointTimelinePathSignature],
) -> frozenset[JointTimelinePathSignature]:
    """Validate stable complete-path feedback before any profile search."""

    if (
        isinstance(values, (str, bytes, Mapping))
        or not isinstance(values, Collection)
    ):
        raise TypeError("forbidden_complete_paths must be a collection")
    canonical: set[JointTimelinePathSignature] = set()
    for value in values:
        if type(value) is not JointTimelinePathSignature:
            raise TypeError(
                "forbidden_complete_paths must contain exact "
                "JointTimelinePathSignature values"
            )
        canonical.add(
            JointTimelinePathSignature(
                candidate_clocks=tuple(
                    JointTimelineCandidateClock(
                        segment_index=item.segment_index,
                        candidate_id=item.candidate_id,
                        source_id=item.source_id,
                        shot_id=item.shot_id,
                        start_seconds=item.start_seconds,
                        end_seconds=item.end_seconds,
                    )
                    for item in value.candidate_clocks
                ),
                boundary_proofs=tuple(
                    JointTimelineBoundaryProofIdentity(
                        boundary_index=item.boundary_index,
                        proof_identity=item.proof_identity,
                        frame_offset=item.frame_offset,
                    )
                    for item in value.boundary_proofs
                ),
            )
        )
    if not canonical:
        return frozenset()

    lattice_clocks = tuple(
        frozenset(
            JointTimelineCandidateClock.from_candidate(candidate)
            for candidate in segment
        )
        for segment in lattice
    )
    for path in canonical:
        if len(path.candidate_clocks) != len(lattice):
            raise ValueError(
                "forbidden complete path length does not match the "
                "candidate lattice"
            )
        if any(
            candidate_clock not in lattice_clocks[position]
            for position, candidate_clock in enumerate(path.candidate_clocks)
        ):
            raise ValueError(
                "forbidden complete path does not belong to the candidate lattice"
            )
    return frozenset(canonical)


def _neutral_transition_annotations() -> dict[str, object]:
    return {
        "authority": None,
        "contract": None,
        "scale": None,
        "focus": None,
        "layout": None,
        "pose_basis": None,
        "pose": None,
        "silhouette": None,
        "formation": None,
        "entry_side_stability": None,
        "exit_side_stability": None,
    }


class JointVisualEvidenceCoverageError(ValueError):
    """Privacy-safe cross-source result-evidence contract violation."""

    def __init__(
        self,
        *,
        candidate_index: int,
        cadence_profile: str,
        boundary: BoundaryMaterializationProof,
    ) -> None:
        key = boundary.measurement_key
        coverage = boundary.result_evidence.evidence_coverage

        def candidate_payload(candidate: TimelineCandidate) -> dict[str, object]:
            return {
                "segment_index": candidate.segment_index,
                "source_id": candidate.source_id,
                "candidate_id": candidate.candidate_id,
                "shot_id": candidate.shot_id,
            }

        measurement_key_payload = {
            "boundary_index": key.boundary_index,
            "outgoing": candidate_payload(key.outgoing),
            "incoming": candidate_payload(key.incoming),
        }
        self._payload = {
            "contract": "joint-visual-result-evidence-coverage-failure-v1",
            "reason_code": (
                "joint_boundary_result_evidence_coverage_contract_violation"
            ),
            "joint_candidate_index": candidate_index,
            "cadence_profile": cadence_profile,
            "boundary_index": key.boundary_index,
            "measurement_key": measurement_key_payload,
            "measurement_key_sha256": _sha256_json(measurement_key_payload),
            "frame_offset": boundary.frame_offset,
            "coverage": {
                "professional_evidence_basis": (
                    coverage.professional_evidence_basis
                ),
                "professional_gate_passed": coverage.professional_gate_passed,
                "failed_coverage_fields": list(coverage.missing_mask),
            },
            "raw_frames_persisted": False,
            "raw_visual_descriptors_persisted": False,
            "raw_candidate_embeddings_persisted": False,
            "biometric_model_vectors_persisted": False,
        }
        self._boundary_no_good = (key, boundary.frame_offset)
        super().__init__(
            "joint cross-source result evidence coverage contract violated "
            f"for candidate {candidate_index} ({cadence_profile}), boundary "
            f"{key.boundary_index}, offset {boundary.frame_offset}; missing="
            + ",".join(coverage.missing_mask)
        )

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self._payload, allow_nan=False))

    @property
    def boundary_no_good(self) -> tuple[BoundaryMeasurementKey, int]:
        """Return the process-local exact option that violated coverage."""

        return self._boundary_no_good


def _freeze_json_audit(value: object) -> object:
    """Recursively freeze a JSON-safe audit without retaining caller aliases."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_audit(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_audit(item) for item in value)
    return value


def _thaw_json_audit(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_audit(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_audit(item) for item in value]
    return value


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_shot_path_signature(
    result: JointTimelineResult,
) -> tuple[tuple[int, str, str], ...]:
    """Return the exact editorial path, intentionally excluding proof choice."""

    return tuple(
        (candidate.segment_index, candidate.source_id, candidate.shot_id)
        for candidate in result.candidates
    )


def _proof_variant_signature(result: JointTimelineResult) -> tuple[str, ...]:
    return tuple(option.proof_identity for option in result.boundary_options)


def _render_variant_signature(
    result: JointTimelineResult,
) -> tuple[tuple[object, ...], ...]:
    """Identify the exact boundary clocks and renderer-authority transforms."""

    return tuple(
        (
            option.frame_offset,
            option.outgoing_transform.transform_identity,
            option.outgoing_transform.crop_xywh,
            option.outgoing_transform.rotation_degrees,
            option.outgoing_transform.track_id,
            option.outgoing_transform.subject_basis,
            option.incoming_transform.transform_identity,
            option.incoming_transform.crop_xywh,
            option.incoming_transform.rotation_degrees,
            option.incoming_transform.track_id,
            option.incoming_transform.subject_basis,
        )
        for option in result.boundary_options
    )


@dataclass(frozen=True, slots=True)
class JointVisualSelectionRecord:
    """One coalesced joint result before render-track materialization."""

    cadence_profile: str
    joint_result: JointTimelineResult = field(repr=False)
    shifted_timeline_result: TimelineResult = field(repr=False)
    source_shot_path: tuple[tuple[int, str, str], ...]
    proof_variant_signature: tuple[str, ...]
    role_control_audit: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cadence_profile, str)
            or not self.cadence_profile
            or self.cadence_profile != self.cadence_profile.strip()
        ):
            raise ValueError("cadence_profile must be an exact identifier")
        if not isinstance(self.joint_result, JointTimelineResult):
            raise TypeError("joint_result must be a JointTimelineResult")
        if not isinstance(self.shifted_timeline_result, TimelineResult):
            raise TypeError("shifted_timeline_result must be a TimelineResult")
        expected_path = _source_shot_path_signature(self.joint_result)
        if tuple(self.source_shot_path) != expected_path:
            raise ValueError("source_shot_path does not match joint_result")
        expected_proofs = _proof_variant_signature(self.joint_result)
        if tuple(self.proof_variant_signature) != expected_proofs:
            raise ValueError("proof_variant_signature does not match joint_result")
        if tuple(
            (item.segment_index, item.source_id, item.shot_id)
            for item in self.shifted_timeline_result.candidates
        ) != expected_path:
            raise ValueError("shifted timeline changed the exact source/shot path")
        audit = self.role_control_audit
        object.__setattr__(
            self,
            "role_control_audit",
            None if audit is None else MappingProxyType(dict(audit)),
        )

    @property
    def optimizer_score(self) -> float:
        return self.shifted_timeline_result.total_score

    @property
    def privacy_safe_signature(self) -> str:
        return "joint-visual-path:" + _sha256_json(
            {
                "contract": "joint-visual-source-shot-proof-signature-v1",
                "source_shot_path": [list(item) for item in self.source_shot_path],
                "boundary_proof_identities": list(
                    self.proof_variant_signature
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class JointVisualSelectionSet:
    """Round-robin profile output with exact decoded proof variants retained."""

    candidate_set: VisualTimelineOptimizerCarrier = field(repr=False)
    selections: tuple[JointVisualSelectionRecord, ...]
    candidate_records: tuple[VisualTimelineCandidateRecord, ...]
    selected_candidate_index: int
    boundary_proof_bundles: tuple["JointVisualBoundaryProofBundle", ...] = field(
        repr=False
    )
    profile_failures: tuple[tuple[str, NoCompleteTimelinePath], ...] = field(
        default=(),
        repr=False,
    )
    role_control_audits: tuple[Mapping[str, Any], ...] = ()
    shared_option_evidence_cache_audit: Mapping[str, Any] = field(
        default_factory=dict
    )
    shared_bridge_evidence_cache_audit: Mapping[str, Any] = field(
        default_factory=dict
    )
    selection_retry_audit: Mapping[str, Any] = field(default_factory=dict)
    analysis_basis: str = JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS

    def __post_init__(self) -> None:
        if not isinstance(
            self.candidate_set,
            (VisualTimelineCandidateSet, VisualTimelineOptimizerSeed),
        ):
            raise TypeError(
                "candidate_set must carry a visual optimizer workspace"
            )
        selections = tuple(self.selections)
        records = tuple(self.candidate_records)
        proof_bundles = tuple(self.boundary_proof_bundles)
        if len(selections) < MINIMUM_PRODUCTION_JOINT_CANDIDATES:
            raise ValueError("joint selection set requires at least eight variants")
        if len(selections) != len(records) or len(selections) != len(
            proof_bundles
        ):
            raise ValueError(
                "joint selections, records, and proof bundles must align"
            )
        if any(
            not isinstance(bundle, JointVisualBoundaryProofBundle)
            for bundle in proof_bundles
        ):
            raise TypeError("joint selection contains an invalid proof bundle")
        for selection, bundle in zip(
            selections,
            proof_bundles,
            strict=True,
        ):
            options = selection.joint_result.boundary_options
            boundaries = bundle.boundary_proofs
            paths = bundle.segment_path_proofs
            if (
                bundle.frame_offsets
                != tuple(option.frame_offset for option in options)
                or tuple(boundary.proof_identity for boundary in boundaries)
                != selection.proof_variant_signature
                or len(boundaries) != len(options)
            ):
                raise ValueError(
                    "joint selection is bound to another boundary proof variant"
                )
            for option, boundary in zip(options, boundaries, strict=True):
                if (
                    boundary.measurement_key != option.measurement_key
                    or boundary.frame_offset != option.frame_offset
                    or boundary.outgoing_transform != option.outgoing_transform
                    or boundary.incoming_transform != option.incoming_transform
                    or boundary.scoring_outgoing != option.scoring_outgoing
                    or boundary.scoring_incoming != option.scoring_incoming
                ):
                    raise ValueError(
                        "joint boundary proof does not match its selected option"
                    )
            expected_bridges = tuple(
                proof.proof_identity
                for proof in selection.joint_result.segment_bridge_proofs
            )
            if tuple(path.proof_identity for path in paths[1:-1]) != (
                expected_bridges
            ):
                raise ValueError(
                    "joint segment proof does not match selected bridge identity"
                )
            shifted_candidates = selection.shifted_timeline_result.candidates
            for candidate, path in zip(
                shifted_candidates,
                paths,
                strict=True,
            ):
                if (
                    path.candidate.segment_index != candidate.segment_index
                    or path.candidate.source_id != candidate.source_id
                    or path.candidate.shot_id != candidate.shot_id
                    or path.candidate.candidate_id != candidate.candidate_id
                    or path.source_index != int(candidate.source_id)
                    or not math.isclose(
                        path.start_master_seconds,
                        candidate.start_seconds,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        path.end_master_seconds,
                        candidate.end_seconds,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        "joint segment proof path does not match shifted selection"
                    )
        if tuple(record.candidate_index for record in records) != tuple(
            range(len(records))
        ):
            raise ValueError("joint candidate indices must be contiguous")
        if any(
            record.timeline_result is not selection.shifted_timeline_result
            for record, selection in zip(records, selections, strict=True)
        ):
            raise ValueError("candidate records lost their shifted joint timeline")
        paths = tuple(selection.source_shot_path for selection in selections)
        unique_path_count = len(set(paths))
        if unique_path_count < MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS:
            raise ValueError(
                "joint selection set does not retain the mathematical minimum "
                "of unique source/shot paths"
            )
        supplement_count = len(paths) - unique_path_count
        required_supplement_count = max(
            0,
            MINIMUM_PRODUCTION_JOINT_CANDIDATES - unique_path_count,
        )
        if supplement_count != required_supplement_count:
            raise ValueError(
                "joint selection set retained an invalid proof supplement count"
            )
        path_multiplicity = {
            path: paths.count(path)
            for path in set(paths)
        }
        if any(
            count > MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH
            for count in path_multiplicity.values()
        ):
            raise ValueError(
                "joint selection set retained too many variants for one path"
            )
        proof_variants = tuple(
            (selection.source_shot_path, selection.proof_variant_signature)
            for selection in selections
        )
        if len(set(proof_variants)) != len(proof_variants):
            raise ValueError("joint selection set contains a duplicate proof variant")
        render_variants = tuple(
            (
                selection.source_shot_path,
                _render_variant_signature(selection.joint_result),
            )
            for selection in selections
        )
        if len(set(render_variants)) != len(render_variants):
            raise ValueError("joint selection set contains a duplicate render variant")
        if not 0 <= self.selected_candidate_index < len(records):
            raise ValueError("selected joint candidate index is outside the set")
        selected = tuple(
            record.candidate_index for record in records if record.selected
        )
        if selected != (self.selected_candidate_index,):
            raise ValueError("joint selection set must mark one selected record")
        if any(not record.review.passed for record in records):
            raise ValueError(
                "production joint selection set contains a failed review"
            )
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "candidate_records", records)
        object.__setattr__(self, "boundary_proof_bundles", proof_bundles)
        object.__setattr__(
            self,
            "role_control_audits",
            tuple(MappingProxyType(dict(item)) for item in self.role_control_audits),
        )
        object.__setattr__(
            self,
            "shared_option_evidence_cache_audit",
            MappingProxyType(dict(self.shared_option_evidence_cache_audit)),
        )
        object.__setattr__(
            self,
            "shared_bridge_evidence_cache_audit",
            MappingProxyType(dict(self.shared_bridge_evidence_cache_audit)),
        )
        object.__setattr__(
            self,
            "selection_retry_audit",
            _freeze_json_audit(
                json.loads(
                    json.dumps(
                        dict(self.selection_retry_audit),
                        allow_nan=False,
                    )
                )
            ),
        )

    @property
    def ranked_records(self) -> tuple[VisualTimelineCandidateRecord, ...]:
        return tuple(sorted(self.candidate_records, key=lambda item: item.review_rank))

    @property
    def unique_source_shot_path_count(self) -> int:
        return len({selection.source_shot_path for selection in self.selections})

    @property
    def proof_variant_supplement_count(self) -> int:
        return len(self.selections) - self.unique_source_shot_path_count

    def selection_for_candidate(
        self,
        candidate_index: int,
    ) -> JointVisualSelectionRecord:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(self.selections)
        ):
            raise IndexError("joint candidate index is outside the selection set")
        return self.selections[candidate_index]

    def proof_bundle_for_candidate(
        self,
        candidate_index: int,
    ) -> "JointVisualBoundaryProofBundle":
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(self.boundary_proof_bundles)
        ):
            raise IndexError("joint candidate index is outside the proof set")
        return self.boundary_proof_bundles[candidate_index]

    def to_dict(self) -> dict[str, object]:
        """Serialize only scalar review/audit data, never transient descriptors."""

        return {
            "analysis_basis": self.analysis_basis,
            "candidate_count": len(self.candidate_records),
            "unique_source_shot_path_count": self.unique_source_shot_path_count,
            "proof_variant_supplement_count": (
                self.proof_variant_supplement_count
            ),
            "selected_candidate_index": self.selected_candidate_index,
            "candidates": [
                {
                    **record.to_dict(),
                    "joint_path_signature": selection.privacy_safe_signature,
                    "boundary_frame_offsets": [
                        option.frame_offset
                        for option in selection.joint_result.boundary_options
                    ],
                    "boundary_proof_count": len(
                        selection.joint_result.boundary_options
                    ),
                }
                for record, selection in zip(
                    self.candidate_records,
                    self.selections,
                    strict=True,
                )
            ],
            "profile_failures": [
                {
                    "cadence_profile": profile,
                    "no_complete_path": failure.to_dict(),
                }
                for profile, failure in self.profile_failures
            ],
            "role_control_audits": [dict(item) for item in self.role_control_audits],
            "shared_option_evidence_cache": dict(
                self.shared_option_evidence_cache_audit
            ),
            "shared_bridge_evidence_cache": dict(
                self.shared_bridge_evidence_cache_audit
            ),
            "selection_retry_audit": _thaw_json_audit(
                self.selection_retry_audit
            ),
            "raw_visual_descriptors_persisted": False,
            "raw_candidate_embeddings_persisted": False,
            "biometric_model_vectors_persisted": False,
        }


def _copy_endpoint_fields(
    base: TimelineCandidate,
    measured: TimelineCandidate,
    *,
    prefix: str,
) -> TimelineCandidate:
    replacements = {
        item.name: getattr(measured, item.name)
        for item in fields(TimelineCandidate)
        if item.name.startswith(prefix)
    }
    return replace(base, **replacements)


def _shift_joint_timeline_result(
    result: JointTimelineResult,
    *,
    fps: float,
    candidate_set: VisualTimelineOptimizerCarrier,
) -> TimelineResult:
    """Apply each selected boundary offset to both adjacent segment clocks."""

    if not isinstance(result, JointTimelineResult):
        raise TypeError("result must be a JointTimelineResult")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    choices = result.timeline.choices
    options = result.boundary_options
    if len(options) != max(0, len(choices) - 1):
        raise ValueError("joint result boundary option count is inconsistent")
    shifted_choices: list[TimelineChoice] = []
    for index, choice in enumerate(choices):
        candidate = choice.candidate
        if index > 0:
            incoming = options[index - 1].scoring_incoming
            if incoming is None:  # pragma: no cover - option invariant
                raise RuntimeError("joint incoming scoring candidate is unavailable")
            candidate = _copy_endpoint_fields(candidate, incoming, prefix="entry_")
        if index < len(options):
            outgoing = options[index].scoring_outgoing
            if outgoing is None:  # pragma: no cover - option invariant
                raise RuntimeError("joint outgoing scoring candidate is unavailable")
            candidate = _copy_endpoint_fields(candidate, outgoing, prefix="exit_")
        start = candidate.start_seconds + (
            options[index - 1].frame_offset / fps if index > 0 else 0.0
        )
        end = candidate.end_seconds + (
            options[index].frame_offset / fps if index < len(options) else 0.0
        )
        if end - start < 2.0 / fps - 1e-9:
            raise ValueError("joint offset produced a segment shorter than two frames")
        source_index = int(candidate.source_id)
        if not 0 <= source_index < len(candidate_set.sources):
            raise ValueError("joint candidate source_id is outside the candidate set")
        if candidate_set.scene_cut_times:
            source_start = _source_clock_time(
                start,
                source_index=source_index,
                audio_offsets=candidate_set.audio_offsets,
                source_timewarps=candidate_set.source_timewarps,
            )
            source_end = _source_clock_time(
                end,
                source_index=source_index,
                audio_offsets=candidate_set.audio_offsets,
                source_timewarps=candidate_set.source_timewarps,
            )
            internal_scene_cut_master_seconds = tuple(
                master_cut
                for source_cut in candidate_set.scene_cut_times[source_index]
                if source_start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < source_cut
                < source_end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                for master_cut in (
                    _master_clock_time(
                        source_cut,
                        source_index=source_index,
                        audio_offsets=candidate_set.audio_offsets,
                        source_timewarps=candidate_set.source_timewarps,
                    ),
                )
                if start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < master_cut
                < end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
            )
        else:
            internal_scene_cut_master_seconds = tuple(
                value
                for value in candidate.internal_scene_cut_master_seconds
                if start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < value
                < end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
            )
        candidate = replace(
            candidate,
            start_seconds=start,
            end_seconds=end,
            internal_scene_cut_master_seconds=(
                internal_scene_cut_master_seconds
            ),
        )
        shifted_choices.append(replace(choice, candidate=candidate))
    for left, right in zip(shifted_choices, shifted_choices[1:]):
        if not math.isclose(
            left.candidate.end_seconds,
            right.candidate.start_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("joint offset materialization is not contiguous")
    return TimelineResult(
        mode=result.timeline.mode,
        target_member=result.timeline.target_member,
        choices=tuple(shifted_choices),
        total_score=result.total_score,
    )


def _coalesce_profile_results(
    profile_results: Sequence[tuple[str, Sequence[JointTimelineResult]]],
) -> tuple[
    dict[str, tuple[JointTimelineResult, ...]],
    dict[str, tuple[JointTimelineResult, ...]],
]:
    """Retain unique path primaries plus exact alternate render variants."""

    profile_order = {
        profile: index for index, (profile, _results) in enumerate(profile_results)
    }
    variant_winners: dict[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
        ],
        tuple[str, JointTimelineResult],
    ] = {}
    for profile, results in profile_results:
        for result in results:
            signature = (
                _source_shot_path_signature(result),
                _proof_variant_signature(result),
            )
            previous = variant_winners.get(signature)
            if previous is None:
                variant_winners[signature] = (profile, result)
                continue
            previous_profile, previous_result = previous
            current_key = (result.total_score, -profile_order[profile])
            previous_key = (
                previous_result.total_score,
                -profile_order[previous_profile],
            )
            if current_key > previous_key:
                variant_winners[signature] = (profile, result)

    primary_winners: dict[
        tuple[tuple[int, str, str], ...],
        tuple[str, JointTimelineResult],
    ] = {}
    for profile, result in variant_winners.values():
        path = _source_shot_path_signature(result)
        previous = primary_winners.get(path)
        if previous is None:
            primary_winners[path] = (profile, result)
            continue
        previous_profile, previous_result = previous
        current_key = (
            result.total_score,
            -profile_order[profile],
            tuple(reversed(_proof_variant_signature(result))),
        )
        previous_key = (
            previous_result.total_score,
            -profile_order[previous_profile],
            tuple(reversed(_proof_variant_signature(previous_result))),
        )
        if current_key > previous_key:
            primary_winners[path] = (profile, result)

    primary_signatures = {
        (
            _source_shot_path_signature(result),
            _proof_variant_signature(result),
        )
        for _profile, result in primary_winners.values()
    }
    primary_pools: dict[str, list[JointTimelineResult]] = {
        profile: [] for profile, _results in profile_results
    }
    alternate_pools: dict[str, list[JointTimelineResult]] = {
        profile: [] for profile, _results in profile_results
    }
    for profile, result in primary_winners.values():
        primary_pools[profile].append(result)
    for signature, (profile, result) in variant_winners.items():
        if signature not in primary_signatures:
            alternate_pools[profile].append(result)

    def sorted_pools(
        pools: Mapping[str, Sequence[JointTimelineResult]],
    ) -> dict[str, tuple[JointTimelineResult, ...]]:
        return {
            profile: tuple(
                sorted(
                    results,
                    key=lambda item: (
                        -item.total_score,
                        _source_shot_path_signature(item),
                        _proof_variant_signature(item),
                    ),
                )
            )
            for profile, results in pools.items()
        }

    return sorted_pools(primary_pools), sorted_pools(alternate_pools)


def _ordered_profile_candidate_pool(
    profile_results: Sequence[tuple[str, Sequence[JointTimelineResult]]],
) -> tuple[tuple[str, JointTimelineResult], ...]:
    """Return all distinct primary paths, then at most one supplement each."""

    coalesced, alternates = _coalesce_profile_results(profile_results)
    maximum_pool_size = 3 * MAXIMUM_PROFILE_JOINT_CANDIDATES
    ordered: list[tuple[str, JointTimelineResult]] = []
    maximum_primary_rank = max(
        (len(pool) for pool in coalesced.values()),
        default=0,
    )
    for rank in range(maximum_primary_rank):
        for profile, _unused in profile_results:
            pool = coalesced[profile]
            if rank < len(pool):
                ordered.append((profile, pool[rank]))
                if len(ordered) >= maximum_pool_size:
                    return tuple(ordered)

    selected_proof_variants = {
        (
            _source_shot_path_signature(result),
            _proof_variant_signature(result),
        )
        for _profile, result in ordered
    }
    selected_render_variants = {
        (
            _source_shot_path_signature(result),
            _render_variant_signature(result),
        )
        for _profile, result in ordered
    }
    supplemented_paths: set[tuple[tuple[int, str, str], ...]] = set()
    maximum_alternate_rank = max(
        (len(pool) for pool in alternates.values()),
        default=0,
    )
    for rank in range(maximum_alternate_rank):
        for profile, _unused in profile_results:
            pool = alternates[profile]
            if rank >= len(pool):
                continue
            result = pool[rank]
            path = _source_shot_path_signature(result)
            proof_signature = (path, _proof_variant_signature(result))
            render_signature = (path, _render_variant_signature(result))
            if (
                path in supplemented_paths
                or proof_signature in selected_proof_variants
                or render_signature in selected_render_variants
            ):
                continue
            ordered.append((profile, result))
            supplemented_paths.add(path)
            selected_proof_variants.add(proof_signature)
            selected_render_variants.add(render_signature)
            if len(ordered) >= maximum_pool_size:
                return tuple(ordered)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class _ReviewedJointCandidate:
    selection: JointVisualSelectionRecord
    proof_bundle: "JointVisualBoundaryProofBundle" = field(repr=False)
    review: TimelineReview
    global_candidate_attempt_index: int


def _boundary_review_no_goods(
    selection: JointVisualSelectionRecord,
    review: TimelineReview,
) -> frozenset[tuple[BoundaryMeasurementKey, int]]:
    """Project only exact boundary-local hard review failures to no-goods."""

    no_goods: set[tuple[BoundaryMeasurementKey, int]] = set()
    options = selection.joint_result.boundary_options
    for diagnostic in review.boundaries:
        if not diagnostic.source_switch:
            continue
        local_failure = bool(
            diagnostic.exceeded_metrics
            or diagnostic.scene_cut_risk
            or diagnostic.outgoing_side_stability_exceeded_metrics
            or diagnostic.incoming_side_stability_exceeded_metrics
        )
        if not local_failure:
            continue
        if not 0 <= diagnostic.boundary_index < len(options):
            raise RuntimeError("review boundary index is outside the joint proof")
        option = options[diagnostic.boundary_index]
        if option.measurement_key.boundary_index != diagnostic.boundary_index:
            raise RuntimeError("review boundary index drifted from the joint option")
        no_goods.add((option.measurement_key, option.frame_offset))
    return frozenset(no_goods)


def _production_passing_cohort(
    reviewed: Sequence[_ReviewedJointCandidate],
    *,
    maximum_candidates: int,
    minimum_unique_paths: int,
) -> tuple[_ReviewedJointCandidate, ...]:
    """Choose passed candidates while preserving path and proof diversity."""

    passing = tuple(item for item in reviewed if item.review.passed)
    if not passing:
        return ()
    ranked = _review_ranked_candidate_indices(
        tuple(item.review for item in passing)
    )
    ordered = tuple(passing[index] for index in ranked)
    primaries: list[_ReviewedJointCandidate] = []
    primary_paths: set[tuple[tuple[int, str, str], ...]] = set()
    for item in ordered:
        path = item.selection.source_shot_path
        if path in primary_paths:
            continue
        primaries.append(item)
        primary_paths.add(path)
        if len(primaries) >= maximum_candidates:
            break
    if len(primary_paths) < minimum_unique_paths:
        return ()
    cohort = list(primaries)
    if len(cohort) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
        # ``maximum_candidates`` is an upper bound, not a target.  Once at
        # least eight unique passing paths exist, retaining alternate proofs
        # would violate the exact supplement invariant and waste materialized
        # proof memory.
        return tuple(cohort[:maximum_candidates])
    required_supplement_count = (
        MINIMUM_PRODUCTION_JOINT_CANDIDATES - len(cohort)
    )
    selected_variants = {
        (
            item.selection.source_shot_path,
            item.selection.proof_variant_signature,
            _render_variant_signature(item.selection.joint_result),
        )
        for item in cohort
    }
    selected_candidate_ids = {id(item) for item in cohort}
    supplemented_paths: set[tuple[tuple[int, str, str], ...]] = set()
    for item in ordered:
        path = item.selection.source_shot_path
        signature = (
            path,
            item.selection.proof_variant_signature,
            _render_variant_signature(item.selection.joint_result),
        )
        if (
            path not in primary_paths
            or path in supplemented_paths
            or signature in selected_variants
            or id(item) in selected_candidate_ids
        ):
            continue
        cohort.append(item)
        supplemented_paths.add(path)
        selected_variants.add(signature)
        selected_candidate_ids.add(id(item))
        if len(supplemented_paths) >= required_supplement_count:
            break
    if len(cohort) < MINIMUM_PRODUCTION_JOINT_CANDIDATES:
        return ()
    return tuple(cohort[:maximum_candidates])


def _assert_provider_binding(
    candidate_set: VisualTimelineOptimizerCarrier,
    provider: DecodedBoundaryOptionProvider,
    *,
    fps: float,
) -> None:
    """Reject a decoded provider configured for any other candidate lattice."""

    if not math.isclose(
        float(provider.output_fps),
        fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("joint provider output_fps does not match adapter fps")
    workspace = candidate_set.optimizer_workspace
    if workspace is None:  # pragma: no cover - caller checks first
        raise ValueError("joint provider binding needs an optimizer workspace")
    provider.validate_segment_count(len(workspace.candidates_by_segment))
    if tuple(Path(item) for item in provider.sources) != candidate_set.sources:
        raise ValueError("joint provider sources do not match candidate set")
    if len(provider.audio_offsets) != len(candidate_set.audio_offsets) or any(
        not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(
            provider.audio_offsets,
            candidate_set.audio_offsets,
            strict=True,
        )
    ):
        raise ValueError("joint provider audio offsets do not match candidate set")
    if tuple(provider.source_timewarps) != candidate_set.source_timewarps:
        raise ValueError("joint provider timewarps do not match candidate set")
    if tuple(tuple(item) for item in provider.scene_cut_times) != (
        candidate_set.scene_cut_times
    ):
        raise ValueError("joint provider scene cuts do not match candidate set")
    if tuple(provider.source_ids) != tuple(
        str(index) for index in range(len(candidate_set.sources))
    ):
        raise ValueError("joint provider source identifiers are not renderer indices")


def select_joint_visual_candidates(
    candidate_set: VisualTimelineOptimizerCarrier,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    hard_forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
    minimum_unique_paths: int = MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS,
    maximum_candidates: int | None = None,
    forbidden_boundary_selections: Collection[
        BoundarySelectionNoGood
    ] = (),
    forbidden_complete_paths: Collection[
        JointTimelinePathSignature
    ] = (),
    maximum_proof_materialization_calls: int = (
        MAXIMUM_JOINT_VISUAL_PROOF_CALLS
    ),
    decoded_montage_feedback: JointDecodedMontageFeedback | None = None,
    progress_callback: JointTimelineProgressCallback | None = None,
) -> JointVisualSelectionSet:
    """Run the existing professional controls through the decoded joint DP.

    ``hard_forbidden_boundary_edges`` is deliberately typed as an already
    classified :class:`TimelineNoGoodSet`.  Missing or indeterminate boundary
    evidence must not be projected into this argument.  Exact boundary and
    complete-path feedback is validated against this candidate lattice before
    being forwarded unchanged to every cadence profile.  Decoded montage
    feedback is separately restricted to exact candidate clocks and the
    existing role-control soft scoring hooks; it never authors hard no-goods.
    The proof budget is per call, allowing a caller to pass its remaining
    bounded-session budget.
    """

    if not isinstance(
        candidate_set,
        (VisualTimelineCandidateSet, VisualTimelineOptimizerSeed),
    ):
        raise TypeError("candidate_set must carry a visual optimizer workspace")
    workspace = candidate_set.optimizer_workspace
    if workspace is None:
        raise ValueError(
            "joint visual selection requires the transient optimizer workspace"
        )
    if not isinstance(provider, DecodedBoundaryOptionProvider):
        raise TypeError("provider must be a DecodedBoundaryOptionProvider")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if not isinstance(hard_forbidden_boundary_edges, TimelineNoGoodSet):
        raise TypeError("hard forbidden edges must be a TimelineNoGoodSet")
    if decoded_montage_feedback is not None and not isinstance(
        decoded_montage_feedback,
        JointDecodedMontageFeedback,
    ):
        raise TypeError(
            "decoded_montage_feedback must be JointDecodedMontageFeedback or None"
        )
    if (
        isinstance(minimum_unique_paths, bool)
        or not isinstance(minimum_unique_paths, int)
        or minimum_unique_paths < MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS
    ):
        raise ValueError(
            "minimum_unique_paths is below the mathematical render cohort floor"
        )
    requested_maximum = (
        min(workspace.maximum_results, 3 * MAXIMUM_PROFILE_JOINT_CANDIDATES)
        if maximum_candidates is None
        else maximum_candidates
    )
    if (
        isinstance(requested_maximum, bool)
        or not isinstance(requested_maximum, int)
        or requested_maximum < max(
            minimum_unique_paths,
            MINIMUM_PRODUCTION_JOINT_CANDIDATES,
        )
        or requested_maximum > 3 * MAXIMUM_PROFILE_JOINT_CANDIDATES
    ):
        raise ValueError(
            "maximum_candidates must retain eight variants and be at most 24"
        )
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    if (
        isinstance(maximum_proof_materialization_calls, bool)
        or not isinstance(maximum_proof_materialization_calls, int)
        or not MINIMUM_PRODUCTION_JOINT_CANDIDATES
        <= maximum_proof_materialization_calls
        <= MAXIMUM_JOINT_VISUAL_PROOF_CALLS
    ):
        raise ValueError(
            "proof materialization budget must be between 8 and 32"
        )
    _assert_provider_binding(candidate_set, provider, fps=fps)

    lattice: tuple[tuple[TimelineCandidate, ...], ...]
    preferred_source_ids = workspace.preferred_source_ids
    if workspace.prescribed_source_ids is None:
        lattice = workspace.candidates_by_segment
    else:
        restricted: list[tuple[TimelineCandidate, ...]] = []
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
            restricted.append(exact)
        lattice = tuple(restricted)
        preferred_source_ids = None

    if (
        decoded_montage_feedback is not None
        and workspace.professional_montage_policy is None
    ):
        raise ValueError(
            "decoded montage feedback requires a professional montage policy"
        )
    role_control_lattice: tuple[tuple[TimelineCandidate, ...], ...] = lattice
    role_control_subject_metadata: Mapping[
        str, _CandidateSubjectMetadata
    ] = workspace.subject_metadata_by_candidate
    if workspace.professional_montage_policy is not None:
        if workspace.section_plan is None:  # pragma: no cover
            raise RuntimeError("montage policy lost its section plan")
        (
            role_control_lattice,
            role_control_subject_metadata,
            decoded_montage_feedback_audit,
        ) = _decoded_montage_role_control_inputs(
            lattice,
            workspace.subject_metadata_by_candidate,
            decoded_montage_feedback,
        )
        decoded_montage_feedback_audit.update(
            {
                "professional_montage_policy_available": True,
                "existing_role_control_hooks_reused": [
                    "emission_adjustments",
                    "source_switch_bonus_by_segment",
                ],
            }
        )
    else:
        _, decoded_montage_feedback_audit = (
            _canonical_decoded_montage_feedback(lattice, None)
        )
        decoded_montage_feedback_audit.update(
            {
                "cold_start_metadata_close_without_bbox_demoted_sample_count": 0,
                "role_control_shadow_candidate_count": 0,
                "original_candidate_objects_mutated": False,
                "availability_override_applied": False,
                "professional_montage_policy_available": False,
                "existing_role_control_hooks_reused": [],
                "optimizer_effect": "none_without_professional_montage_policy",
            }
        )

    external_boundary_no_goods = _canonical_external_boundary_no_goods(
        lattice,
        forbidden_boundary_selections,
    )
    external_complete_path_no_goods = (
        _canonical_external_complete_path_no_goods(
            lattice,
            forbidden_complete_paths,
        )
    )
    profiles = tuple(_safe_cadence_profiles(workspace.diversity))
    if len(profiles) != 3:
        raise RuntimeError(
            "joint visual selection requires exactly three cadence profiles"
        )
    total_boundary_layers = max(0, len(lattice) - 1)
    shared_option_cache: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
    shared_bridge_evidence_cache = JointTimelineSharedBridgeEvidenceCache(
        maximum_entries=min(
            JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
            sum(
                len(lattice[index]) * len(JOINT_BOUNDARY_FRAME_OFFSETS) ** 2
                for index in range(1, len(lattice) - 1)
            ),
        )
    )
    shared_option_cache_hit_count = 0
    shared_option_cache_miss_count = 0

    def shared_option_provider(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
        nonlocal shared_option_cache_hit_count
        nonlocal shared_option_cache_miss_count
        if not isinstance(key, BoundaryMeasurementKey):
            raise TypeError("shared option cache key is invalid")
        cached = shared_option_cache.get(key)
        if cached is not None:
            shared_option_cache_hit_count += 1
            return cached
        shared_option_cache_miss_count += 1
        measured = provider.selection_options(key)
        if not isinstance(measured, BoundaryOptionSet):
            raise TypeError("decoded provider returned an invalid option set")
        if measured.measurement_key != key:
            raise RuntimeError("decoded provider returned evidence for another key")
        if (
            measured.evidence_scope
            is not BoundaryOptionEvidenceScope.SELECTION_SCALAR
        ):
            raise RuntimeError(
                "decoded selection provider returned non-scalar evidence"
            )
        shared_option_cache[key] = measured
        return measured

    def emit_progress(
        *,
        phase: str,
        profile_name: str,
        profile_index: int,
        core: JointTimelineProgress | None = None,
        profile_status: str,
    ) -> None:
        if progress_callback is None:
            return
        completed = core.completed_boundary_layers if core is not None else 0
        option_keys = core.option_key_count if core is not None else 0
        bridge_calls = core.bridge_provider_call_count if core is not None else 0
        batch_calls = (
            core.bridge_disposition_batch_call_count
            if core is not None
            else 0
        )
        batch_requests = (
            core.bridge_disposition_batch_request_count
            if core is not None
            else 0
        )
        progress_callback(
            JointTimelineProgress(
                phase=phase,
                cadence_profile=profile_name,
                profile_index=profile_index,
                total_profiles=max(
                    len(profiles),
                    (retry_round + 1) * len(profiles),
                ),
                completed_boundary_layers=completed,
                total_boundary_layers=total_boundary_layers,
                shared_option_cache_hit_count=(
                    shared_option_cache_hit_count
                ),
                shared_option_cache_miss_count=(
                    shared_option_cache_miss_count
                ),
                shared_option_cache_entry_count=len(shared_option_cache),
                option_key_count=option_keys,
                bridge_provider_call_count=bridge_calls,
                bridge_disposition_batch_call_count=batch_calls,
                bridge_disposition_batch_request_count=batch_requests,
                **shared_bridge_evidence_cache._progress_counts(),
                profile_status=profile_status,
            )
        )

    role_audits: list[Mapping[str, Any]] = []
    bridge_disposition_batch_call_count = 0
    bridge_disposition_batch_request_count = 0
    global_profile_attempt_index = 0
    global_candidate_attempt_index = 0
    proof_materialization_call_count = 0
    review_call_count = 0
    review_failure_count = 0
    review_cache_hit_count = 0
    total_unique_candidate_attempt_count = 0
    discarded_candidate_count = 0
    coverage_contract_failure_count = 0
    total_round_count = 0
    retry_round = 0
    exact_boundary_no_goods: frozenset[
        tuple[BoundaryMeasurementKey, int]
    ] = external_boundary_no_goods
    initial_exact_boundary_no_good_count = len(exact_boundary_no_goods)
    reviewed_variants: dict[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
            tuple[tuple[object, ...], ...],
        ],
        _ReviewedJointCandidate,
    ] = {}
    attempted_variant_keys: set[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
            tuple[tuple[object, ...], ...],
        ]
    ] = set()
    rejected_variant_no_goods: dict[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
            tuple[tuple[object, ...], ...],
        ],
        frozenset[tuple[BoundaryMeasurementKey, int]],
    ] = {}
    discarded_variant_keys: set[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
            tuple[tuple[object, ...], ...],
        ]
    ] = set()
    retry_events: list[dict[str, object]] = []
    coverage_failure_events: list[dict[str, object]] = []
    maximum_exact_no_goods = len(JOINT_BOUNDARY_FRAME_OFFSETS) * sum(
        len(lattice[index]) * len(lattice[index + 1])
        for index in range(len(lattice) - 1)
    )
    profile_failures: tuple[tuple[str, NoCompleteTimelinePath], ...] = ()
    profile_failure_by_name: dict[str, NoCompleteTimelinePath] = {}
    profile_failure_events: list[dict[str, object]] = []

    def run_profile_round() -> tuple[
        tuple[tuple[str, tuple[JointTimelineResult, ...]], ...],
        Mapping[str, Mapping[str, Any]],
    ]:
        nonlocal bridge_disposition_batch_call_count
        nonlocal bridge_disposition_batch_request_count
        nonlocal global_profile_attempt_index
        round_results: list[tuple[str, tuple[JointTimelineResult, ...]]] = []
        round_failures: list[tuple[str, NoCompleteTimelinePath]] = []
        round_role_audit_by_profile: dict[str, Mapping[str, Any]] = {}
        for profile in profiles:
            global_profile_attempt_index += 1
            profile_index = global_profile_attempt_index
            latest_core_progress: JointTimelineProgress | None = None
            callback_no_path_failure: NoCompleteTimelinePath | None = None
            emit_progress(
                phase="profile_start",
                profile_name=profile.name,
                profile_index=profile_index,
                profile_status="in_progress",
            )

            def report_core_progress(progress: JointTimelineProgress) -> None:
                nonlocal latest_core_progress
                nonlocal callback_no_path_failure
                if not isinstance(progress, JointTimelineProgress):
                    raise TypeError("joint timeline progress payload is invalid")
                latest_core_progress = progress
                if progress.phase != "boundary_layer_complete":
                    return
                try:
                    emit_progress(
                        phase="boundary_layer_complete",
                        profile_name=profile.name,
                        profile_index=profile_index,
                        core=progress,
                        profile_status="in_progress",
                    )
                except NoCompleteTimelinePath as exc:
                    callback_no_path_failure = exc
                    raise

            selection_kwargs: dict[str, Any] = {}
            if workspace.professional_montage_policy is not None:
                if workspace.section_plan is None:  # pragma: no cover
                    raise RuntimeError("montage policy lost its section plan")
                adjustments, switch_bonus, raw_role_audit = (
                    _role_conditioned_visual_controls(
                        role_control_lattice,
                        section_plan=workspace.section_plan,
                        phrase_times=workspace.phrase_times,
                        policy=workspace.professional_montage_policy,
                        cadence_profile=profile.name,
                        subject_metadata_by_candidate=(
                            role_control_subject_metadata
                        ),
                        timeline_config=profile.config,
                        forbidden_boundary_edges=hard_forbidden_boundary_edges,
                    )
                )
                if any(
                    not math.isfinite(float(value)) or float(value) > 0.0
                    for value in adjustments.values()
                ):
                    raise RuntimeError(
                        "role-control emission adjustments must be finite and "
                        "non-positive"
                    )
                role_audit = {
                    **dict(raw_role_audit),
                    "decoded_montage_feedback": dict(
                        decoded_montage_feedback_audit
                    ),
                    "emission_adjustments_non_positive": True,
                    "retry_round": retry_round,
                    "global_profile_attempt_index": (
                        global_profile_attempt_index
                    ),
                    "exact_boundary_no_good_count": len(
                        exact_boundary_no_goods
                    ),
                }
                selection_kwargs = {
                    "emission_adjustments": adjustments,
                    "source_switch_bonus_by_segment": switch_bonus,
                }
                round_role_audit_by_profile[profile.name] = role_audit
                role_audits.append(role_audit)
            try:
                results = select_joint_timeline_candidates(
                    lattice,
                    option_provider=shared_option_provider,
                    option_rehydrator=provider.rehydrate_selection_option,
                    bridge_provider=provider.bridge_provider,
                    bridge_disposition_batch_provider=(
                        provider.bridge_disposition_batch
                    ),
                    fps=fps,
                    config=profile.config,
                    preferred_source_ids=preferred_source_ids,
                    maximum_results=MAXIMUM_PROFILE_JOINT_CANDIDATES,
                    forbidden_boundary_edges=hard_forbidden_boundary_edges,
                    forbidden_boundary_selections=exact_boundary_no_goods,
                    forbidden_complete_paths=external_complete_path_no_goods,
                    progress_callback=report_core_progress,
                    shared_bridge_evidence_cache=shared_bridge_evidence_cache,
                    **selection_kwargs,
                )
            except NoCompleteTimelinePath as exc:
                if exc is callback_no_path_failure:
                    raise
                round_failures.append((profile.name, exc))
                audit = round_role_audit_by_profile.get(profile.name)
                if isinstance(audit, dict):
                    audit["candidate_generation_exhaustion"] = exc.to_dict()
                results = ()
                emit_progress(
                    phase="profile_complete",
                    profile_name=profile.name,
                    profile_index=profile_index,
                    core=latest_core_progress,
                    profile_status="no_complete_path",
                )
            else:
                if progress_callback is not None and (
                    latest_core_progress is None
                    or latest_core_progress.phase != "selection_complete"
                ):
                    raise RuntimeError(
                        "joint timeline emitted no terminal progress"
                    )
                emit_progress(
                    phase="profile_complete",
                    profile_name=profile.name,
                    profile_index=profile_index,
                    core=latest_core_progress,
                    profile_status="completed",
                )
            if latest_core_progress is not None:
                bridge_disposition_batch_call_count += (
                    latest_core_progress.bridge_disposition_batch_call_count
                )
                bridge_disposition_batch_request_count += (
                    latest_core_progress.bridge_disposition_batch_request_count
                )
            round_results.append((profile.name, tuple(results)))
        nonlocal profile_failures
        for profile_name, failure in round_failures:
            profile_failure_by_name[profile_name] = failure
            profile_failure_events.append(
                {
                    "retry_round": retry_round,
                    "cadence_profile": profile_name,
                    "no_complete_path": failure.to_dict(),
                }
            )
        profile_failures = tuple(profile_failure_by_name.items())
        return tuple(round_results), round_role_audit_by_profile

    def proof_budget_exhaustion() -> NoCompleteTimelinePath:
        return NoCompleteTimelinePath(
            "joint visual proof materialization budget was exhausted before "
            "eight authoritative candidates passed",
            reason_code="joint_visual_proof_budget_exhausted",
            rejection_counts={
                "proof_materialization_call_count": (
                    proof_materialization_call_count
                ),
                "maximum_proof_materialization_calls": (
                    maximum_proof_materialization_calls
                ),
                "review_call_count": review_call_count,
                "review_failure_count": review_failure_count,
                "review_cache_hit_count": review_cache_hit_count,
                "total_unique_candidate_attempt_count": (
                    total_unique_candidate_attempt_count
                ),
                "discarded_candidate_count": discarded_candidate_count,
                "passing_candidate_count": len(reviewed_variants),
                "total_round_count": total_round_count,
            },
            profile_failures=profile_failures,
        )

    final_reviewed: tuple[_ReviewedJointCandidate, ...] = ()
    while True:
        total_round_count += 1
        profile_results, role_audit_by_profile = run_profile_round()
        ordered_pool = _ordered_profile_candidate_pool(profile_results)
        if (
            not ordered_pool
            and len(profile_failures) == len(profiles)
            and all(
                failure.reason_code
                == "joint_timeline_complete_path_no_good_exhausted"
                for _profile_name, failure in profile_failures
            )
        ):
            raise NoCompleteTimelinePath(
                "every cadence profile exhausted its exact complete-path "
                "search space under external feedback",
                reason_code="joint_visual_complete_path_no_good_exhausted",
                rejection_counts={
                    "exhausted_profile_count": len(profile_failures),
                    "forbidden_complete_path_count": len(
                        external_complete_path_no_goods
                    ),
                    "proof_materialization_call_count": (
                        proof_materialization_call_count
                    ),
                },
                profile_failures=profile_failures,
            )
        round_new_no_goods: set[tuple[BoundaryMeasurementKey, int]] = set()
        for pool_index, (profile, result) in enumerate(ordered_pool):
            variant_key = (
                _source_shot_path_signature(result),
                _proof_variant_signature(result),
                _render_variant_signature(result),
            )
            cached_reviewed = reviewed_variants.get(variant_key)
            if cached_reviewed is not None:
                review_cache_hit_count += 1
                final_reviewed = _production_passing_cohort(
                    tuple(reviewed_variants.values()),
                    maximum_candidates=requested_maximum,
                    minimum_unique_paths=minimum_unique_paths,
                )
                if len(final_reviewed) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
                    break
                continue
            rejected_no_good = rejected_variant_no_goods.get(variant_key)
            if rejected_no_good is not None:
                round_new_no_goods.update(rejected_no_good)
                continue
            if proof_materialization_call_count >= (
                maximum_proof_materialization_calls
            ):
                raise proof_budget_exhaustion()
            if variant_key not in attempted_variant_keys:
                attempted_variant_keys.add(variant_key)
                total_unique_candidate_attempt_count += 1
            global_candidate_attempt_index += 1
            selection = JointVisualSelectionRecord(
                cadence_profile=profile,
                joint_result=result,
                shifted_timeline_result=_shift_joint_timeline_result(
                    result,
                    fps=fps,
                    candidate_set=candidate_set,
                ),
                source_shot_path=_source_shot_path_signature(result),
                proof_variant_signature=_proof_variant_signature(result),
                role_control_audit=role_audit_by_profile.get(profile),
            )
            proof_materialization_call_count += 1
            proof_bundle = _proof_bundle(selection, provider)
            try:
                projected_plan = _project_visual_plan_from_joint_proof(
                    candidate_set=candidate_set,
                    timeline_result=selection.shifted_timeline_result,
                    proof_bundle=proof_bundle,
                    candidate_index=pool_index,
                    cadence_profile=selection.cadence_profile,
                )
            except JointVisualEvidenceCoverageError as exc:
                coverage_contract_failure_count += 1
                round_new_no_goods.add(exc.boundary_no_good)
                rejected_variant_no_goods[variant_key] = frozenset(
                    (exc.boundary_no_good,)
                )
                coverage_failure_events.append(exc.to_dict())
                if variant_key not in discarded_variant_keys:
                    discarded_variant_keys.add(variant_key)
                    discarded_candidate_count += 1
                if proof_materialization_call_count >= (
                    maximum_proof_materialization_calls
                ):
                    raise proof_budget_exhaustion()
                del proof_bundle
                continue
            review = review_timeline(
                projected_plan,
                available_source_count=len(candidate_set.sources),
            )
            review_call_count += 1
            if review.passed:
                reviewed_variants[variant_key] = _ReviewedJointCandidate(
                    selection=selection,
                    proof_bundle=proof_bundle,
                    review=review,
                    global_candidate_attempt_index=(
                        global_candidate_attempt_index
                    ),
                )
            else:
                review_failure_count += 1
                review_no_goods = _boundary_review_no_goods(
                    selection,
                    review,
                )
                round_new_no_goods.update(review_no_goods)
                # Cache every deterministic review rejection by its exact
                # path/proof key, including global-only failures with no local
                # no-good.  The empty frozenset is a lightweight negative cache
                # entry and deliberately retains no decoded proof bundle.
                rejected_variant_no_goods[variant_key] = review_no_goods
                if variant_key not in discarded_variant_keys:
                    discarded_variant_keys.add(variant_key)
                    discarded_candidate_count += 1
                del proof_bundle, projected_plan, review
            final_reviewed = _production_passing_cohort(
                tuple(reviewed_variants.values()),
                maximum_candidates=requested_maximum,
                minimum_unique_paths=minimum_unique_paths,
            )
            if len(final_reviewed) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
                break
            if proof_materialization_call_count >= (
                maximum_proof_materialization_calls
            ):
                raise proof_budget_exhaustion()

        if len(final_reviewed) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
            break

        new_no_goods = frozenset(
            round_new_no_goods - set(exact_boundary_no_goods)
        )
        passing_candidate_count = len(reviewed_variants)
        passing_unique_path_count = len(
            {
                item.selection.source_shot_path
                for item in reviewed_variants.values()
            }
        )
        rejection_counts = {
            "reviewed_unique_candidate_count": review_call_count,
            "retained_reviewed_candidate_count": len(reviewed_variants),
            "passing_candidate_count": passing_candidate_count,
            "unique_source_shot_path_count": passing_unique_path_count,
            "required_unique_source_shot_path_count": minimum_unique_paths,
            "distinct_render_candidate_count": passing_candidate_count,
            "required_render_candidate_count": (
                MINIMUM_PRODUCTION_JOINT_CANDIDATES
            ),
            "proof_variant_supplement_count": max(
                0,
                passing_candidate_count - passing_unique_path_count,
            ),
            "exact_boundary_no_good_count": len(exact_boundary_no_goods),
            "new_exact_boundary_no_good_count": len(new_no_goods),
            "retry_round": retry_round,
            "proof_materialization_call_count": (
                proof_materialization_call_count
            ),
            "maximum_proof_materialization_calls": (
                maximum_proof_materialization_calls
            ),
        }
        if not new_no_goods:
            if passing_unique_path_count < minimum_unique_paths:
                reason_code = "joint_visual_unique_path_floor_not_met"
                message = (
                    "authoritative review produced fewer than the required "
                    "unique passing source/shot paths and no exact boundary "
                    "retry made progress"
                )
            elif review_failure_count == 0:
                reason_code = "joint_visual_candidate_cohort_floor_not_met"
                message = (
                    "joint visual selection could not retain eight distinct "
                    "passing render variants and no exact boundary retry made "
                    "progress"
                )
            else:
                reason_code = "joint_visual_authoritative_review_fixed_point"
                message = (
                    "authoritative review produced fewer than eight passing "
                    "candidates and no boundary-local retry made progress"
                )
            raise NoCompleteTimelinePath(
                message,
                reason_code=reason_code,
                rejection_counts=rejection_counts,
                profile_failures=profile_failures,
            )
        if retry_round >= MAXIMUM_JOINT_VISUAL_RETRY_ROUNDS:
            raise NoCompleteTimelinePath(
                "authoritative review retry round limit was reached before "
                "eight passing candidates existed",
                reason_code="joint_visual_authoritative_review_retry_exhausted",
                rejection_counts=rejection_counts,
                profile_failures=profile_failures,
            )
        updated_no_goods = frozenset(
            set(exact_boundary_no_goods) | set(new_no_goods)
        )
        if len(updated_no_goods) > maximum_exact_no_goods:
            raise RuntimeError("exact boundary no-good bound was exceeded")
        retry_events.append(
            {
                "retry_round": retry_round,
                "global_profile_attempt_index_after_round": (
                    global_profile_attempt_index
                ),
                "global_candidate_attempt_index_after_round": (
                    global_candidate_attempt_index
                ),
                "reviewed_unique_candidate_count": review_call_count,
                "passing_candidate_count": len(reviewed_variants),
                "new_exact_boundary_no_good_count": len(new_no_goods),
                "total_exact_boundary_no_good_count": len(updated_no_goods),
                "proof_materialization_call_count": (
                    proof_materialization_call_count
                ),
            }
        )
        exact_boundary_no_goods = updated_no_goods
        pruned_variant_keys = {
            key
            for key, item in reviewed_variants.items()
            if any(
                (option.measurement_key, option.frame_offset)
                in exact_boundary_no_goods
                for option in item.selection.joint_result.boundary_options
            )
        }
        for key in pruned_variant_keys - discarded_variant_keys:
            discarded_variant_keys.add(key)
            discarded_candidate_count += 1
        for key in pruned_variant_keys:
            item = reviewed_variants[key]
            rejected_variant_no_goods[key] = frozenset(
                (option.measurement_key, option.frame_offset)
                for option in item.selection.joint_result.boundary_options
                if (
                    option.measurement_key,
                    option.frame_offset,
                )
                in exact_boundary_no_goods
            )
        reviewed_variants = {
            key: item
            for key, item in reviewed_variants.items()
            if key not in pruned_variant_keys
        }
        retry_round += 1

    selections = tuple(item.selection for item in final_reviewed)
    proof_bundles = tuple(item.proof_bundle for item in final_reviewed)
    reviews = tuple(item.review for item in final_reviewed)
    ranked_indices = _review_ranked_candidate_indices(reviews)
    selected_candidate_index = ranked_indices[0]
    rank_by_index = {
        candidate_index: review_rank
        for review_rank, candidate_index in enumerate(ranked_indices)
    }
    records = tuple(
        VisualTimelineCandidateRecord(
            candidate_index=index,
            cadence_profile=selection.cadence_profile,
            timeline_result=selection.shifted_timeline_result,
            review=reviews[index],
            review_rank=rank_by_index[index],
            selected=index == selected_candidate_index,
        )
        for index, selection in enumerate(selections)
    )
    return JointVisualSelectionSet(
        candidate_set=candidate_set,
        selections=selections,
        candidate_records=records,
        selected_candidate_index=selected_candidate_index,
        boundary_proof_bundles=proof_bundles,
        profile_failures=tuple(profile_failures),
        role_control_audits=tuple(role_audits),
        shared_option_evidence_cache_audit={
            "contract": "joint_visual_shared_boundary_option_cache_v1",
            "entry_count": len(shared_option_cache),
            "hit_count": shared_option_cache_hit_count,
            "miss_count": shared_option_cache_miss_count,
            "evidence_scope": (
                BoundaryOptionEvidenceScope.SELECTION_SCALAR.value
            ),
            "immutable_boundary_option_sets": True,
            "raw_visual_evidence_persisted": False,
        },
        shared_bridge_evidence_cache_audit=(
            {
                **shared_bridge_evidence_cache.to_dict(),
                "bridge_disposition_batch_call_count": (
                    bridge_disposition_batch_call_count
                ),
                "bridge_disposition_batch_request_count": (
                    bridge_disposition_batch_request_count
                ),
            }
        ),
        selection_retry_audit={
            "contract": "joint_visual_authoritative_passing_retry_v2",
            "production_review_floor_required": True,
            "total_round_count": total_round_count,
            "retry_round_count": retry_round,
            "global_profile_attempt_count": global_profile_attempt_index,
            "global_candidate_attempt_count": global_candidate_attempt_index,
            "total_unique_candidate_attempt_count": (
                total_unique_candidate_attempt_count
            ),
            "proof_materialization_call_count": (
                proof_materialization_call_count
            ),
            "maximum_proof_materialization_calls": (
                maximum_proof_materialization_calls
            ),
            "review_call_count": review_call_count,
            "review_failure_count": review_failure_count,
            "review_cache_hit_count": review_cache_hit_count,
            "reviewed_unique_candidate_count": review_call_count,
            "retained_reviewed_candidate_count": len(reviewed_variants),
            "discarded_candidate_count": discarded_candidate_count,
            "coverage_contract_failure_count": (
                coverage_contract_failure_count
            ),
            "passing_candidate_count": len(reviewed_variants),
            "initial_exact_boundary_no_good_count": (
                initial_exact_boundary_no_good_count
            ),
            "exact_boundary_no_good_count": len(exact_boundary_no_goods),
            "external_complete_path_no_good_count": len(
                external_complete_path_no_goods
            ),
            "maximum_retry_rounds": MAXIMUM_JOINT_VISUAL_RETRY_ROUNDS,
            "events": retry_events,
            "profile_failure_events": profile_failure_events,
            "coverage_failure_events": coverage_failure_events,
            "decoded_montage_feedback": dict(
                decoded_montage_feedback_audit
            ),
            "raw_frames_persisted": False,
            "raw_visual_descriptors_persisted": False,
        },
    )


def candidate_review_summaries(
    selection_set: JointVisualSelectionSet,
    *,
    selected_candidate_index: int,
) -> tuple[dict[str, Any], ...]:
    """Build the existing compact plan summaries for a selected joint path."""

    if not isinstance(selection_set, JointVisualSelectionSet):
        raise TypeError("selection_set must be a JointVisualSelectionSet")
    if not 0 <= selected_candidate_index < len(selection_set.candidate_records):
        raise IndexError("selected candidate index is outside the joint set")
    return tuple(
        _candidate_review_summary(
            candidate_index=record.candidate_index,
            cadence_profile=record.cadence_profile,
            selected=record.candidate_index == selected_candidate_index,
            candidate=record.timeline_result,
            review=record.review,
        )
        for record in selection_set.candidate_records
    )


def _editorial_track_payload(
    path: SegmentTransformPathMaterializationProof,
) -> dict[str, object]:
    track = path.editorial_transform_track
    if track is None:
        raise ValueError("segment transform proof has no renderer track")
    return {
        "reason": track.reason,
        "keyframes": [
            {
                "time_seconds": keyframe.time_seconds,
                "zoom": keyframe.zoom,
                "center_x": keyframe.center_x,
                "center_y": keyframe.center_y,
                "roll_degrees": keyframe.roll_degrees,
            }
            for keyframe in track.keyframes
        ],
    }


@dataclass(frozen=True, slots=True)
class JointVisualBoundaryProofBundle:
    """Atomic process-local proof consumed by render without another decode.

    ``selection_proof`` owns transient descriptors and face-appearance anchors,
    so it is hidden from ``repr`` and never traversed by :meth:`to_dict`.
    The serialized surface contains only exact identities, scalar completeness
    evidence, and the renderer transform tracks already authorized by the
    provider.
    """

    selection_proof: JointSelectionMaterializationProof = field(repr=False)
    frame_offsets: tuple[int, ...]
    bridge_audits: tuple[Mapping[str, object], ...] = ()
    analysis_basis: str = (
        "joint_selection_process_local_boundary_and_segment_proof_bundle_v1"
    )

    def __post_init__(self) -> None:
        proof = self.selection_proof
        if not isinstance(proof, JointSelectionMaterializationProof):
            raise TypeError(
                "selection_proof must be a JointSelectionMaterializationProof"
            )
        offsets = tuple(self.frame_offsets)
        if len(offsets) != len(proof.boundaries) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in offsets
        ):
            raise ValueError("proof-bundle frame offsets are invalid")
        if offsets != tuple(item.frame_offset for item in proof.boundaries):
            raise ValueError("proof-bundle offsets disagree with boundary proofs")
        if proof.boundary_proof_identities != tuple(
            item.proof_identity for item in proof.boundaries
        ):
            raise ValueError("boundary proof identities are inconsistent")
        if proof.segment_path_proof_identities != tuple(
            item.proof_identity for item in proof.segment_paths
        ):
            raise ValueError("segment path proof identities are inconsistent")
        if len(proof.segment_paths) != len(proof.boundaries) + 1:
            raise ValueError("proof bundle needs one path for every segment")
        expected_indices = tuple(range(len(proof.segment_paths)))
        if tuple(
            item.candidate.segment_index for item in proof.segment_paths
        ) != expected_indices:
            raise ValueError("segment path proofs are not exact and contiguous")
        for path in proof.segment_paths:
            if (
                not path.evidence_complete
                or not path.crop_compatible
                or not path.track_compatible
                or not path.endpoints_included
                or not path.full_frame_centers_included
                or path.failure_codes
                or path.editorial_transform_track is None
                or not path.samples
                or len(path.samples) != path.expected_sample_count
            ):
                raise ValueError(
                    "proof bundle contains an unauthorized segment transform path"
                )
        for boundary_index, boundary in enumerate(proof.boundaries):
            outgoing_path = proof.segment_paths[boundary_index]
            incoming_path = proof.segment_paths[boundary_index + 1]
            if (
                outgoing_path.exit_transform != boundary.outgoing_transform
                or incoming_path.entry_transform != boundary.incoming_transform
                or outgoing_path.samples[-1].transform
                != outgoing_path.exit_transform
                or incoming_path.samples[0].transform
                != incoming_path.entry_transform
            ):
                raise ValueError(
                    "boundary proof is not bound to adjacent renderer tracks"
                )
        audits = tuple(MappingProxyType(dict(item)) for item in self.bridge_audits)
        if len(audits) != max(0, len(proof.segment_paths) - 2):
            raise ValueError("proof bundle middle-bridge audit count is invalid")
        expected_audits = tuple(
            dict(item) for item in proof.selected_bridge_audits
        )
        if tuple(dict(item) for item in audits) != expected_audits:
            raise ValueError(
                "proof bundle bridge audits differ from the atomic selection proof"
            )
        if any(
            audit.get("proof_identity")
            != proof.segment_paths[index + 1].proof_identity
            for index, audit in enumerate(audits)
        ):
            raise ValueError("bridge audit identity lost its middle-segment path")
        object.__setattr__(self, "frame_offsets", offsets)
        object.__setattr__(self, "bridge_audits", audits)

    @property
    def selection_identity(self) -> str:
        return self.selection_proof.selection_identity

    @property
    def boundary_proofs(self) -> tuple[BoundaryMaterializationProof, ...]:
        return self.selection_proof.boundaries

    @property
    def segment_path_proofs(
        self,
    ) -> tuple[SegmentTransformPathMaterializationProof, ...]:
        return self.selection_proof.segment_paths

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_basis": self.analysis_basis,
            "selection_identity": self.selection_identity,
            "boundary_proof_identities": list(
                self.selection_proof.boundary_proof_identities
            ),
            "boundary_frame_offsets": list(self.frame_offsets),
            "segment_path_proof_identities": list(
                self.selection_proof.segment_path_proof_identities
            ),
            "segment_paths": [
                {
                    "proof_identity": path.proof_identity,
                    "segment_index": path.candidate.segment_index,
                    "source_index": path.source_index,
                    "start_master_seconds": path.start_master_seconds,
                    "end_master_seconds": path.end_master_seconds,
                    "duration_frames": path.duration_frames,
                    "expected_sample_count": path.expected_sample_count,
                    "observed_sample_count": len(path.samples),
                    "endpoints_included": path.endpoints_included,
                    "full_frame_centers_included": (
                        path.full_frame_centers_included
                    ),
                    "evidence_complete": path.evidence_complete,
                    "crop_compatible": path.crop_compatible,
                    "track_compatible": path.track_compatible,
                    "failure_codes": list(path.failure_codes),
                    "editorial_transform_track": _editorial_track_payload(path),
                }
                for path in self.selection_proof.segment_paths
            ],
            "middle_bridge_audits": [dict(item) for item in self.bridge_audits],
            "raw_frames_persisted": False,
            "raw_visual_descriptors_persisted": False,
            "raw_candidate_embeddings_persisted": False,
            "biometric_model_vectors_persisted": False,
            "process_local_materialization_retained": True,
        }


def _maximum_available_metric(*values: float | None) -> float | None:
    available = tuple(float(value) for value in values if value is not None)
    return max(available) if available else None


def _authoritative_transition_annotations(
    proof_bundle: JointVisualBoundaryProofBundle,
    index: int,
    *,
    candidate_index: int,
    cadence_profile: str,
) -> dict[str, object]:
    """Project the exact post-transform provider result into review scalars."""

    if index == 0:
        return _neutral_transition_annotations()
    boundary = proof_bundle.boundary_proofs[index - 1]
    if boundary.measurement_key.boundary_index != index - 1:
        raise ValueError("joint boundary proof index drifted during projection")
    evidence = boundary.result_evidence
    if not isinstance(evidence, BoundaryVisualEvidence):
        raise TypeError("joint boundary proof contains invalid result evidence")
    if (
        boundary.measurement_key.outgoing.source_id
        == boundary.measurement_key.incoming.source_id
    ):
        if boundary.frame_offset != 0:
            raise ValueError(
                "same-source joint boundary proof changed the native clock"
            )
        key = boundary.measurement_key
        if not (
            math.isclose(
                key.outgoing.end_seconds,
                key.incoming.start_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                boundary.boundary_master_seconds,
                key.outgoing.end_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and boundary.outgoing_source_index
            == boundary.incoming_source_index
            and math.isclose(
                boundary.outgoing_boundary_source_seconds,
                boundary.incoming_boundary_source_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "same-source joint boundary lost exact renderer clock "
                "continuity"
            )
        for transform in (
            boundary.outgoing_transform,
            boundary.incoming_transform,
        ):
            if (
                transform.crop_xywh is not None
                or not math.isclose(
                    transform.rotation_degrees,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or transform.subject_basis != "full_frame_identity"
            ):
                raise ValueError(
                    "same-source renderer continuity used a non-identity "
                    "boundary transform"
                )
        # This is one continuous decode clock with identity transforms.  It may
        # contain a broadcaster-authored native camera cut, but the cross-cut
        # engine introduces no new visual discontinuity at this boundary.
        # Cross-source pose/layout coverage is therefore not the authority for
        # this path; the exact renderer clock proof is.
        return _neutral_transition_annotations()
    if not evidence.evidence_coverage.professional_gate_passed:
        raise JointVisualEvidenceCoverageError(
            candidate_index=candidate_index,
            cadence_profile=cadence_profile,
            boundary=boundary,
        )
    return {
        "authority": JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY,
        "contract": JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT,
        "scale": evidence.shot_scale_delta,
        "focus": evidence.focus_mismatch,
        # The provider gates both transient and persisted layout estimates.
        # Carry the stricter available scalar into the single generic review
        # lane so neither gate can disappear at the representation boundary.
        "layout": _maximum_available_metric(
            evidence.layout_mismatch,
            evidence.persisted_layout_mismatch,
        ),
        "pose_basis": evidence.pose_basis,
        "pose": evidence.pose_mismatch,
        "silhouette": evidence.silhouette_mismatch,
        "formation": evidence.formation_mismatch,
        "entry_side_stability": evidence.incoming_side_stability,
        "exit_side_stability": evidence.outgoing_side_stability,
    }


def _replace_endpoint_sample(
    samples: Sequence[object],
    *,
    entry: object,
    exit_: object,
) -> tuple[object, object, object]:
    if len(samples) != 3:
        raise ValueError("joint visual metadata must contain three samples")
    return (entry, samples[1], exit_)


def _project_visual_plan_from_joint_proof(
    *,
    candidate_set: VisualTimelineOptimizerCarrier,
    timeline_result: TimelineResult,
    proof_bundle: JointVisualBoundaryProofBundle,
    candidate_index: int,
    cadence_profile: str,
    candidate_review_summaries_payload: tuple[dict[str, Any], ...] = (),
) -> VisualTimelinePlan:
    candidates = timeline_result.candidates
    if len(proof_bundle.boundary_proofs) != max(0, len(candidates) - 1):
        raise ValueError("joint proof count does not match projected timeline")
    choices: list[VisualTimelineChoice] = []
    for index, candidate in enumerate(candidates):
        source_index = int(candidate.source_id)
        face_metadata = candidate_set.face_metadata_by_candidate.get(
            candidate.candidate_id,
            _missing_candidate_face_metadata(),
        )
        subject_metadata = candidate_set.subject_metadata_by_candidate.get(
            candidate.candidate_id,
            _missing_candidate_subject_metadata(),
        )
        transition = _authoritative_transition_annotations(
            proof_bundle,
            index,
            candidate_index=candidate_index,
            cadence_profile=cadence_profile,
        )
        face_samples = _replace_endpoint_sample(
            face_metadata.boxes,
            entry=(
                candidate.entry_focus_bbox_xywh
                if index > 0
                else face_metadata.boxes[0]
            ),
            exit_=(
                candidate.exit_focus_bbox_xywh
                if index < len(candidates) - 1
                else face_metadata.boxes[2]
            ),
        )
        silhouette_samples = _replace_endpoint_sample(
            subject_metadata.silhouette_boxes,
            entry=(
                candidate.entry_subject_bbox_xywh
                if index > 0
                else subject_metadata.silhouette_boxes[0]
            ),
            exit_=(
                candidate.exit_subject_bbox_xywh
                if index < len(candidates) - 1
                else subject_metadata.silhouette_boxes[2]
            ),
        )
        silhouette_centroids = _replace_endpoint_sample(
            subject_metadata.silhouette_centroids,
            entry=(
                candidate.entry_subject_centroid_xy
                if index > 0
                else subject_metadata.silhouette_centroids[0]
            ),
            exit_=(
                candidate.exit_subject_centroid_xy
                if index < len(candidates) - 1
                else subject_metadata.silhouette_centroids[2]
            ),
        )
        entry_side_stability = (
            transition["entry_side_stability"]
            if index > 0
            else candidate.entry_boundary_side_stability
        )
        if index < len(candidates) - 1:
            exit_side_stability = proof_bundle.boundary_proofs[
                index
            ].result_evidence.outgoing_side_stability
        else:
            exit_side_stability = candidate.exit_boundary_side_stability
        choices.append(
            VisualTimelineChoice(
                segment_index=candidate.segment_index,
                source_index=source_index,
                canonical_master_start=candidate.start_seconds,
                canonical_master_end=candidate.end_seconds,
                source_local_start=_source_clock_time(
                    candidate.start_seconds,
                    source_index=source_index,
                    audio_offsets=candidate_set.audio_offsets,
                    source_timewarps=candidate_set.source_timewarps,
                ),
                source_local_end=_source_clock_time(
                    candidate.end_seconds,
                    source_index=source_index,
                    audio_offsets=candidate_set.audio_offsets,
                    source_timewarps=candidate_set.source_timewarps,
                ),
                quality_score=candidate.quality_score,
                composition_score=candidate.composition_score,
                boundary_score=candidate.boundary_score,
                shot_scale=candidate.shot_scale,
                shot_type=candidate.shot_type,
                shot_id=candidate.shot_id,
                entry_shot_scale=candidate.entry_shot_scale,
                exit_shot_scale=candidate.exit_shot_scale,
                entry_shot_type=candidate.entry_shot_type,
                exit_shot_type=candidate.exit_shot_type,
                entry_scene_cut_near=candidate.entry_scene_cut_near,
                exit_scene_cut_near=candidate.exit_scene_cut_near,
                intentional_transition_anchor_basis=(
                    candidate.intentional_transition_anchor_basis
                ),
                internal_scene_cut_master_seconds=(
                    candidate.internal_scene_cut_master_seconds
                ),
                entry_boundary_side_stability=entry_side_stability,
                exit_boundary_side_stability=exit_side_stability,
                face_samples_xywh=face_samples,
                face_match_candidate_failure=face_metadata.failure,
                face_detector_backends=face_metadata.backends,
                face_landmark_samples_xy=face_metadata.landmarks,
                face_head_roll_samples_radians=face_metadata.rolls,
                face_eye_scale_samples=face_metadata.eye_scales,
                entry_face_geometry_window=candidate.entry_face_geometry_window,
                exit_face_geometry_window=candidate.exit_face_geometry_window,
                pose_backends=subject_metadata.pose_backends,
                pose_confidences=subject_metadata.pose_confidences,
                detected_person_counts=subject_metadata.person_counts,
                formation_person_counts=subject_metadata.formation_person_counts,
                formation_upper_body_counts=(
                    subject_metadata.formation_upper_body_counts
                ),
                shot_scale_bases=subject_metadata.shot_scale_bases,
                silhouette_samples_xywh=silhouette_samples,
                silhouette_centroid_samples_xy=silhouette_centroids,
                silhouette_backends=subject_metadata.silhouette_backends,
                silhouette_confidences=subject_metadata.silhouette_confidences,
                transition_evidence_authority_from_previous=(
                    transition["authority"]
                ),
                transition_evidence_contract_from_previous=(
                    transition["contract"]
                ),
                scale_transition_mismatch_from_previous=transition["scale"],
                focus_transition_mismatch_from_previous=transition["focus"],
                layout_transition_mismatch_from_previous=transition["layout"],
                pose_transition_basis_from_previous=transition["pose_basis"],
                pose_transition_mismatch_from_previous=transition["pose"],
                silhouette_transition_mismatch_from_previous=(
                    transition["silhouette"]
                ),
                formation_transition_mismatch_from_previous=(
                    transition["formation"]
                ),
            )
        )
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=timeline_result.total_score,
        analysis_basis=(
            "joint_decoded_boundary_offset_transform_and_result_evidence_"
            "materialization_v2"
        ),
        selected_candidate_index=candidate_index,
        candidate_generation_basis=JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS,
        candidate_review_summaries=candidate_review_summaries_payload,
        source_clock_basis=candidate_set.source_clock_basis,
    )


def _materialize_visual_plan_without_decode(
    selection_set: JointVisualSelectionSet,
    record: VisualTimelineCandidateRecord,
    proof_bundle: JointVisualBoundaryProofBundle,
) -> VisualTimelinePlan:
    return _project_visual_plan_from_joint_proof(
        candidate_set=selection_set.candidate_set,
        timeline_result=record.timeline_result,
        proof_bundle=proof_bundle,
        candidate_index=record.candidate_index,
        cadence_profile=record.cadence_profile,
        candidate_review_summaries_payload=candidate_review_summaries(
            selection_set,
            selected_candidate_index=record.candidate_index,
        ),
    )


def _transform_effects(
    transform: object,
) -> tuple[bool, bool, bool]:
    crop = getattr(transform, "crop_xywh", None)
    rotation = float(getattr(transform, "rotation_degrees", 0.0))
    if crop is None:
        zoom = False
        pan = False
    else:
        x, y, width, height = (float(value) for value in crop)
        zoom = (
            not math.isclose(width, 1.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(height, 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
        pan = (
            not math.isclose(
                x + width / 2.0,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                y + height / 2.0,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return (zoom, pan, not math.isclose(rotation, 0.0, abs_tol=1e-9))


def _joint_editorial_motion_plan(
    proof_bundle: JointVisualBoundaryProofBundle,
    segments: Sequence[RenderSegment],
) -> EditorialMotionPlan:
    """Expose provider tracks as the sole downstream motion authority."""

    paths = proof_bundle.segment_path_proofs
    render_segments = tuple(segments)
    if len(paths) != len(render_segments):
        raise ValueError("editorial motion paths and render segments must align")
    accepted: list[int] = []
    natural: list[int] = []
    zoom_count = 0
    pan_count = 0
    roll_count = 0
    for boundary_index, boundary in enumerate(proof_bundle.boundary_proofs):
        outgoing_effects = _transform_effects(boundary.outgoing_transform)
        incoming_effects = _transform_effects(boundary.incoming_transform)
        effects = tuple(
            left or right
            for left, right in zip(
                outgoing_effects,
                incoming_effects,
                strict=True,
            )
        )
        if any(effects):
            accepted.append(boundary_index)
            zoom_count += int(effects[0])
            pan_count += int(effects[1])
            roll_count += int(effects[2])
        else:
            natural.append(boundary_index)
    accepted_set = set(accepted)
    motion_segments: list[EditorialMotionSegment] = []
    for path, render_segment in zip(paths, render_segments, strict=True):
        track = path.editorial_transform_track
        if track is None:  # pragma: no cover - proof-bundle invariant
            raise RuntimeError("joint segment path lost its renderer track")
        if render_segment.editorial_transform is not track:
            raise ValueError("editorial motion plan lost provider track identity")
        adjacent_boundaries = tuple(
            index
            for index in (
                path.candidate.segment_index - 1,
                path.candidate.segment_index,
            )
            if index in accepted_set
        )
        reasons = ["decoded_joint_renderer_track_authority"]
        if adjacent_boundaries:
            reasons.extend(
                f"joint_boundary_{index}_transform_authorized"
                for index in adjacent_boundaries
            )
        else:
            reasons.append("joint_identity_or_natural_match_track_authority")
        motion_segments.append(
            EditorialMotionSegment(
                segment_index=path.candidate.segment_index,
                source_index=path.source_index,
                track=track,
                reasons=tuple(reasons),
            )
        )
    return EditorialMotionPlan(
        segments=tuple(motion_segments),
        boundary_match_count=len(accepted),
        zoom_event_count=zoom_count,
        pan_event_count=pan_count,
        roll_event_count=roll_count,
        blocked_dissolve_segment_count=0,
        # The provider proves an exact renderer crop path.  It does not run
        # the separate pixel-feature affine estimator, so reporting these
        # crop matches as affine matches would overstate the evidence.
        affine_boundary_match_count=0,
        interior_zoom_pulse_count=0,
        skipped_boundary_reasons=(
            (("already_naturally_matched", len(natural)),)
            if natural
            else ()
        ),
        section_directed=False,
        analysis_basis=JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
        accepted_boundary_indices=tuple(accepted),
        natural_match_boundary_indices=tuple(natural),
    )


@dataclass(frozen=True, slots=True)
class JointMaterializedVisualCandidate:
    """Renderer-ready joint alternative and its atomic decoded proof bundle."""

    selection_set: JointVisualSelectionSet = field(repr=False)
    candidate_record: VisualTimelineCandidateRecord
    visual_plan: VisualTimelinePlan = field(repr=False)
    segments: tuple[RenderSegment, ...]
    common_master_start: float
    boundary_proof_bundle: JointVisualBoundaryProofBundle = field(repr=False)
    editorial_motion_plan: EditorialMotionPlan = field(repr=False)
    output_fps: float = 29.97

    def __post_init__(self) -> None:
        index = self.candidate_record.candidate_index
        if (
            not 0 <= index < len(self.selection_set.candidate_records)
            or self.selection_set.candidate_records[index]
            is not self.candidate_record
        ):
            raise ValueError("candidate record does not belong to joint selection")
        if self.visual_plan.selected_candidate_index != index:
            raise ValueError("visual plan selects another joint candidate")
        segments = tuple(self.segments)
        paths = self.boundary_proof_bundle.segment_path_proofs
        if not segments or len(segments) != len(paths):
            raise ValueError("render segments lost an authorized transform path")
        if len(self.visual_plan.choices) != len(segments):
            raise ValueError("visual plan and render segments do not align")
        motion_by_segment = {
            item.segment_index: item
            for item in self.editorial_motion_plan.segments
        }
        if len(motion_by_segment) != len(segments):
            raise ValueError("editorial motion authority does not cover every segment")
        expected_master_start = 0.0
        for segment, choice, path in zip(
            segments,
            self.visual_plan.choices,
            paths,
            strict=True,
        ):
            if not math.isclose(
                segment.master_start,
                expected_master_start,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("joint render segments must be contiguous")
            if segment.editorial_transform is not path.editorial_transform_track:
                raise ValueError("render segment is not bound to its decoded track")
            motion_segment = motion_by_segment.get(path.candidate.segment_index)
            if (
                motion_segment is None
                or motion_segment.source_index != path.source_index
                or motion_segment.track is not path.editorial_transform_track
            ):
                raise ValueError(
                    "editorial motion plan is not bound to provider track identity"
                )
            if (
                segment.source_index != path.source_index
                or choice.segment_index != path.candidate.segment_index
                or choice.source_index != path.source_index
                or choice.shot_id != path.candidate.shot_id
            ):
                raise ValueError("render segment path identity drifted")
            expected_master_start += segment.duration
        object.__setattr__(self, "segments", segments)
        if (
            not math.isfinite(self.common_master_start)
            or self.common_master_start < 0.0
        ):
            raise ValueError("common_master_start must be non-negative and finite")
        if not math.isfinite(self.output_fps) or self.output_fps <= 0.0:
            raise ValueError("output_fps must be positive and finite")
        if (
            self.editorial_motion_plan.analysis_basis
            != JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS
        ):
            raise ValueError("editorial motion plan is not joint-track authority")

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
    def candidate_set(self) -> VisualTimelineOptimizerCarrier:
        return self.selection_set.candidate_set

    @property
    def joint_result(self) -> JointTimelineResult:
        return self.selection_set.selection_for_candidate(
            self.candidate_index
        ).joint_result

    @property
    def privacy_safe_signature(self) -> str:
        selection = self.selection_set.selection_for_candidate(
            self.candidate_index
        )
        return selection.privacy_safe_signature

    @property
    def path_signature(self) -> str:
        """Privacy-safe exact source/shot/proof identity for history joins."""

        return self.privacy_safe_signature

    @property
    def candidate_history_entry(self) -> dict[str, Any]:
        return {
            **self.candidate_record.to_dict(),
            "joint_path_signature": self.privacy_safe_signature,
            "boundary_proof_bundle": self.boundary_proof_bundle.to_dict(),
            "editorial_motion_plan": self.editorial_motion_plan.to_dict(),
        }

    @property
    def reviewer_history_entry(self) -> dict[str, Any]:
        return self.candidate_record.review.to_dict()

    def to_render_plan(
        self,
        *,
        output: Path,
        reference_source: int = 0,
        aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        transitions: Sequence[RenderTransition] = (),
    ) -> RenderPlan:
        ratio = (
            aspect_ratio
            if isinstance(aspect_ratio, AspectRatio)
            else AspectRatio(aspect_ratio)
        )
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together")
        if width is None or height is None:
            width, height = output_size(ratio)
        resolved_fps = self.output_fps if fps is None else float(fps)
        if not math.isclose(
            resolved_fps,
            self.output_fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("render fps cannot differ from decoded joint evidence")
        return RenderPlan(
            sources=self.selection_set.candidate_set.sources,
            segments=self.segments,
            reference_source=reference_source,
            aspect_ratio=ratio,
            output=Path(output),
            master_audio_start=self.common_master_start,
            width=width,
            height=height,
            fps=resolved_fps,
            transitions=tuple(transitions),
        )


def _proof_bundle(
    selection: JointVisualSelectionRecord,
    provider: DecodedBoundaryOptionProvider,
) -> JointVisualBoundaryProofBundle:
    result = selection.joint_result
    proof = provider.selection_materialization_proof(
        result.boundary_options,
        segment_bridge_proofs=result.segment_bridge_proofs,
    )
    return JointVisualBoundaryProofBundle(
        selection_proof=proof,
        frame_offsets=tuple(
            item.frame_offset for item in result.boundary_options
        ),
        bridge_audits=proof.selected_bridge_audits,
    )


def _materialize_joint_candidate(
    selection_set: JointVisualSelectionSet,
    record: VisualTimelineCandidateRecord,
    *,
    fps: float,
) -> JointMaterializedVisualCandidate:
    proof_bundle = selection_set.proof_bundle_for_candidate(
        record.candidate_index
    )
    visual_plan = _materialize_visual_plan_without_decode(
        selection_set,
        record,
        proof_bundle,
    )
    materialized_review = review_timeline(
        visual_plan,
        available_source_count=len(selection_set.candidate_set.sources),
    )
    if materialized_review.to_dict() != record.review.to_dict():
        raise RuntimeError(
            "joint authoritative review changed during materialization"
        )
    common_master_start = visual_plan.choices[0].canonical_master_start
    segments: list[RenderSegment] = []
    for choice, path in zip(
        visual_plan.choices,
        proof_bundle.segment_path_proofs,
        strict=True,
    ):
        if (
            path.candidate.source_id != str(choice.source_index)
            or path.candidate.shot_id != choice.shot_id
            or path.source_index != choice.source_index
            or not math.isclose(
                path.start_master_seconds,
                choice.canonical_master_start,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                path.end_master_seconds,
                choice.canonical_master_end,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                path.duration_frames,
                choice.duration_seconds * fps,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("decoded segment path does not bind the shifted plan")
        track = path.editorial_transform_track
        if track is None:  # pragma: no cover - proof-bundle invariant
            raise RuntimeError("decoded segment path lost its renderer track")
        segment = aligned_render_segment(
            source_index=choice.source_index,
            reference_start=choice.canonical_master_start,
            common_master_start=common_master_start,
            source_offset=selection_set.candidate_set.audio_offsets[
                choice.source_index
            ],
            duration=choice.duration_seconds,
            editorial_transform=track,
            timewarp=selection_set.candidate_set.source_timewarps[
                choice.source_index
            ],
        )
        if (
            not math.isclose(
                segment.source_start,
                path.samples[0].source_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                segment.source_end,
                path.samples[-1].source_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("renderer source clock disagrees with decoded path")
        segments.append(segment)
    return JointMaterializedVisualCandidate(
        selection_set=selection_set,
        candidate_record=record,
        visual_plan=visual_plan,
        segments=tuple(segments),
        common_master_start=common_master_start,
        boundary_proof_bundle=proof_bundle,
        editorial_motion_plan=_joint_editorial_motion_plan(
            proof_bundle,
            segments,
        ),
        output_fps=fps,
    )


def materialize_joint_visual_candidates(
    selection_set: JointVisualSelectionSet,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    maximum_candidates: int = MINIMUM_PRODUCTION_JOINT_CANDIDATES,
) -> tuple[JointMaterializedVisualCandidate, ...]:
    """Materialize reviewer-ranked alternatives from provider RAM only."""

    if not isinstance(selection_set, JointVisualSelectionSet):
        raise TypeError("selection_set must be a JointVisualSelectionSet")
    if not isinstance(provider, DecodedBoundaryOptionProvider):
        raise TypeError("provider must be a DecodedBoundaryOptionProvider")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    _assert_provider_binding(selection_set.candidate_set, provider, fps=fps)
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not MINIMUM_PRODUCTION_JOINT_CANDIDATES
        <= maximum_candidates
        <= len(selection_set.candidate_records)
    ):
        raise ValueError("maximum_candidates must retain at least exact eight")
    ranked = selection_set.ranked_records[:maximum_candidates]
    # Build in a local tuple.  If any proof is missing or stale, no partial
    # renderer candidate escapes this atomic adapter call.
    return tuple(
        _materialize_joint_candidate(
            selection_set,
            record,
            fps=fps,
        )
        for record in ranked
    )


def build_joint_visual_candidates(
    candidate_set: VisualTimelineOptimizerCarrier,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    hard_forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
) -> tuple[JointMaterializedVisualCandidate, ...]:
    """Select and atomically materialize the exact-eight production cohort."""

    selected = select_joint_visual_candidates(
        candidate_set,
        provider=provider,
        fps=fps,
        hard_forbidden_boundary_edges=hard_forbidden_boundary_edges,
        minimum_unique_paths=MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS,
        maximum_candidates=MINIMUM_PRODUCTION_JOINT_CANDIDATES,
    )
    return materialize_joint_visual_candidates(
        selected,
        provider=provider,
        fps=fps,
        maximum_candidates=MINIMUM_PRODUCTION_JOINT_CANDIDATES,
    )


__all__ = [
    "JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS",
    "JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS",
    "JointDecodedMontageFeedback",
    "JointDecodedMontageSegmentObservation",
    "JointMaterializedVisualCandidate",
    "JointVisualBoundaryProofBundle",
    "JointVisualSelectionRecord",
    "JointVisualSelectionSet",
    "MINIMUM_PRODUCTION_JOINT_CANDIDATES",
    "MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS",
    "build_joint_visual_candidates",
    "candidate_review_summaries",
    "materialize_joint_visual_candidates",
    "select_joint_visual_candidates",
]
