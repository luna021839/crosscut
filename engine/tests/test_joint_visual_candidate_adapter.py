from __future__ import annotations

from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.group_boundary_grammar_adapter import _validated_transforms
from crosscut.joint_boundary_provider import (
    BoundaryMaterializationProof,
    JointSelectionMaterializationProof,
    ProgressDecodedBoundaryOptionProvider,
    SegmentTransformPathMaterializationProof,
    SegmentTransformPathSample,
    _SelectionProofProgressObserver,
)
from crosscut.joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOffsetAudit,
    BoundaryOffsetDisposition,
    BoundaryOptionEvidenceScope,
    BoundaryOptionSet,
    BoundarySelectionNoGood,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    JointTimelineProgress,
    JointTimelinePathSignature,
    JointTimelineResult,
    SegmentBridgeDisposition,
    SegmentBridgeDispositionRequest,
    SegmentBridgeProof,
)
from crosscut.joint_visual_candidate_adapter import (
    JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
    MAXIMUM_JOINT_VISUAL_PROOF_CALLS,
    JointDecodedMontageFeedback,
    JointDecodedMontageSegmentObservation,
    JointVisualEvidenceCoverageError,
    JointVisualBoundaryProofBundle,
    JointVisualSelectionRecord,
    _decoded_montage_role_control_inputs,
    _proof_bundle,
    _project_visual_plan_from_joint_proof,
    _shift_joint_timeline_result,
    materialize_joint_visual_candidates as _production_materialize_joint_visual_candidates,
    select_joint_visual_candidates as _production_select_joint_visual_candidates,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
)
from crosscut.section_director import (
    SectionDirective,
    SectionPlan,
    SectionRole,
)
from crosscut.timeline import (
    BoundarySideStability,
    NoCompleteTimelinePath,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.visual_boundary_repair import (
    BoundaryEvidenceCoverage,
    BoundaryVisualEvidence,
    TemporalPoseMotionEvidence,
)
from crosscut.timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
)
from crosscut.timeline_review import (
    JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT,
    review_timeline,
)
from crosscut.visual_score import (
    FrameVisualDescriptor,
    ProfessionalMontageSelectionPolicy,
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelineOptimizerSeed,
    VisualTimelineOptimizerWorkspace,
    _CandidateFaceMetadata,
    _review_ranked_candidate_indices,
    _visual_optimizer_input_identity,
)


RAW_SENTINEL = 0.987654321234567


def _synthetic_review(
    plan: object,
    *,
    available_source_count: int | None = None,
):  # type: ignore[no-untyped-def]
    """Keep real boundary diagnostics while relaxing tiny-fixture global floors."""

    return replace(
        review_timeline(
            plan,
            available_source_count=available_source_count,
        ),
        hard_failures=(),
    )


def select_joint_visual_candidates(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    """Exercise production selection with only synthetic global review floors relaxed."""

    with patch(
        "crosscut.joint_visual_candidate_adapter.review_timeline",
        side_effect=_synthetic_review,
    ):
        return _production_select_joint_visual_candidates(*args, **kwargs)


def materialize_joint_visual_candidates(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    """Use the same synthetic review contract for materialization parity."""

    with patch(
        "crosscut.joint_visual_candidate_adapter.review_timeline",
        side_effect=_synthetic_review,
    ):
        return _production_materialize_joint_visual_candidates(*args, **kwargs)


class StubDecodedProvider(DecodedBoundaryOptionProvider):
    """Type-correct provider whose calls are replaced by the selector stub."""

    def __init__(self) -> None:
        self.output_fps = 10.0
        self.total_segment_count = 3
        self.sources = (Path("source-0.mp4"), Path("source-1.mp4"))
        self.audio_offsets = (0.25, 0.50)
        self.source_timewarps = (None, None)
        self.scene_cut_times = ((), ())
        self.source_ids = ("0", "1")
        self.validated_segment_counts: list[int] = []

    def validate_segment_count(self, segment_count: int) -> None:
        self.validated_segment_counts.append(segment_count)
        if segment_count != self.total_segment_count:
            raise ValueError("synthetic provider segment-count mismatch")

    def bridge_provider(self, *args: object) -> SegmentBridgeProof:
        raise AssertionError("selector stub must not invoke bridge_provider")

    def bridge_disposition_batch(
        self,
        requests: tuple[SegmentBridgeDispositionRequest, ...],
    ) -> tuple[SegmentBridgeDisposition, ...]:
        raise AssertionError(
            "selector stub must not invoke bridge_disposition_batch"
        )

    def selection_materialization_proof(
        self,
        boundary_options: tuple[AuthorizedBoundaryOption, ...],
        segment_bridge_proofs: tuple[SegmentBridgeProof, ...] = (),
    ) -> JointSelectionMaterializationProof:
        return StubMaterializingProvider(
            fps=self.output_fps
        ).selection_materialization_proof(
            boundary_options,
            segment_bridge_proofs,
        )


class CountingStubDecodedProvider(StubDecodedProvider):
    def __init__(self) -> None:
        super().__init__()
        self.option_calls: dict[BoundaryMeasurementKey, int] = {}
        self.selection_option_sets: dict[
            BoundaryMeasurementKey, BoundaryOptionSet
        ] = {}
        self.bridge_calls = 0
        self.bridge_batch_calls: list[
            tuple[SegmentBridgeDispositionRequest, ...]
        ] = []

    def __call__(self, key: BoundaryMeasurementKey) -> BoundaryOptionSet:
        self.option_calls[key] = self.option_calls.get(key, 0) + 1
        measured, _selected = _option_set(
            key,
            selected_offset=0,
            variant="shared-profile",
        )
        return measured

    def selection_options(
        self,
        key: BoundaryMeasurementKey,
    ) -> BoundaryOptionSet:
        measured = replace(
            self(key),
            evidence_scope=(
                BoundaryOptionEvidenceScope.SELECTION_SCALAR
            ),
        )
        self.selection_option_sets[key] = measured
        return measured

    def rehydrate_selection_option(
        self,
        option: AuthorizedBoundaryOption,
    ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet] | None:
        option_set = replace(
            self.selection_option_sets[option.measurement_key],
            evidence_scope=BoundaryOptionEvidenceScope.EXACT,
        )
        exact = option_set.audit_for(option.frame_offset).authorized_option
        if exact is None:
            return None
        return exact, option_set

    def bridge_provider(
        self,
        candidate: TimelineCandidate,
        entry: AuthorizedBoundaryOption,
        exit: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> SegmentBridgeProof:
        self.bridge_calls += 1
        return SegmentBridgeProof(
            candidate,
            entry,
            exit,
            duration_frames,
            f"bridge:{candidate.candidate_id}:{entry.proof_identity}:{exit.proof_identity}",
            True,
            True,
            True,
        )

    def bridge_disposition_batch(
        self,
        requests: tuple[SegmentBridgeDispositionRequest, ...],
    ) -> tuple[SegmentBridgeDisposition, ...]:
        self.bridge_batch_calls.append(requests)
        return tuple(
            SegmentBridgeDisposition(
                request=request,
                evidence_complete=True,
                crop_compatible=True,
                track_compatible=True,
                failure_codes=(),
            )
            for request in requests
        )


class StubMaterializingProvider(StubDecodedProvider):
    def __init__(self, *, fps: float = 10.0) -> None:
        super().__init__()
        self.fps = fps
        self.output_fps = fps
        self._bridge_audits: dict[str, dict[str, object]] = {}
        self.selection_materialization_calls = 0

    @staticmethod
    def _descriptor() -> FrameVisualDescriptor:
        return FrameVisualDescriptor(
            quality_score=0.9,
            composition_score=0.8,
            shot_scale=0.6,
            shot_type=ShotType.MEDIUM,
            face_count=1,
            structure_embedding=(0.1, 0.2),
        )

    @staticmethod
    def _evidence() -> BoundaryVisualEvidence:
        stability = BoundarySideStability(
            sample_span_seconds=0.3,
            scale_delta=0.02,
            structure_mismatch=0.03,
            pose_mismatch=0.04,
            shot_type_mismatch=0.0,
        )
        return BoundaryVisualEvidence(
            total_cost=0.05,
            base_total_cost=0.05,
            formation_mismatch=0.05,
            pose_mismatch=0.05,
            pose_basis="synthetic_actual_pose",
            exact_pose_mismatch=0.05,
            nearest_temporal_pose_mismatch=0.05,
            silhouette_mismatch=0.05,
            layout_mismatch=0.05,
            persisted_layout_mismatch=0.05,
            focus_mismatch=0.05,
            structure_mismatch=0.05,
            shot_scale_delta=0.05,
            shot_type_mismatch=0.0,
            temporal_pose_motion=TemporalPoseMotionEvidence(
                calibration_status="synthetic",
                analysis_executed=True,
                input_valid=True,
                qualifies=True,
                classification="synthetic_match",
                failure_reasons=(),
                feature_dimension=2,
                outgoing_sample_count=2,
                incoming_sample_count=2,
                outgoing_boundary_motion_magnitude=0.1,
                incoming_boundary_motion_magnitude=0.1,
                endpoint_pose_cosine=1.0,
                velocity_direction_cosine=1.0,
                phase_compatibility=1.0,
                confidence=1.0,
                static_pose_match=True,
                dynamic_pose_phase_match=True,
                direction_evidence_available=True,
                dynamic_direction_match_proven=True,
                soft_preference_applied=0.0,
                threshold_values={},
                threshold_identity_sha256="0" * 64,
                preference_weight=0.0,
                configuration_fingerprint_sha256="1" * 64,
            ),
            evidence_coverage=BoundaryEvidenceCoverage(
                exact_pose_available=(True, True),
                temporal_pose_available=(True, True),
                temporal_pose_offsets_frames=(0, 0),
                formation_person_counts=(1, 1),
                formation_upper_body_counts=(1, 1),
                person_layout_available=(True, True),
                silhouette_available=(True, True),
                professional_evidence_basis="synthetic_complete",
                professional_gate_passed=True,
                missing_mask=(),
            ),
            outgoing_side_stability=stability,
            incoming_side_stability=stability,
        )

    @staticmethod
    def _identity(candidate: TimelineCandidate, suffix: str) -> BoundaryTransformState:
        return BoundaryTransformState(
            transform_identity=f"identity:{candidate.candidate_id}:{suffix}",
            track_id=f"track:{candidate.candidate_id}",
            subject_basis="dominant_face",
        )

    @staticmethod
    def _keyframe(
        transform: BoundaryTransformState,
        time_seconds: float,
    ) -> EditorialTransformKeyframe:
        crop = transform.crop_xywh
        if crop is None:
            return EditorialTransformKeyframe(time_seconds=time_seconds, zoom=1.0)
        x, y, width, height = crop
        if not math.isclose(width, height, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("synthetic transform must be isotropic")
        return EditorialTransformKeyframe(
            time_seconds=time_seconds,
            zoom=1.0 / width,
            center_x=x + width / 2.0,
            center_y=y + height / 2.0,
            roll_degrees=transform.rotation_degrees,
        )

    @staticmethod
    def _sample_transform(
        keyframe: EditorialTransformKeyframe,
        *,
        track_id: str,
        identity: str,
    ) -> BoundaryTransformState:
        if (
            math.isclose(keyframe.zoom, 1.0, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                keyframe.center_x,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                keyframe.center_y,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                keyframe.roll_degrees,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            crop = None
        else:
            size = 1.0 / keyframe.zoom
            crop = (
                keyframe.center_x - size / 2.0,
                keyframe.center_y - size / 2.0,
                size,
                size,
            )
        return BoundaryTransformState(
            transform_identity=identity,
            crop_xywh=crop,
            rotation_degrees=keyframe.roll_degrees,
            track_id=track_id,
            subject_basis="dominant_face",
        )

    def selection_materialization_proof(
        self,
        boundary_options: tuple[AuthorizedBoundaryOption, ...],
        segment_bridge_proofs: tuple[SegmentBridgeProof, ...] = (),
    ) -> JointSelectionMaterializationProof:
        self.selection_materialization_calls += 1
        options = tuple(boundary_options)
        candidates = (options[0].measurement_key.outgoing,) + tuple(
            option.measurement_key.incoming for option in options
        )
        offsets = tuple(option.frame_offset for option in options)
        descriptor = self._descriptor()
        evidence = self._evidence()
        boundaries = tuple(
            BoundaryMaterializationProof(
                proof_identity=option.proof_identity,
                raw_proof_identity=option.proof_identity + ":raw",
                result_proof_identity=option.proof_identity + ":result",
                measurement_key=option.measurement_key,
                frame_offset=option.frame_offset,
                side_sample_radius_frames=3,
                boundary_master_seconds=option.measurement_key.outgoing.end_seconds,
                outgoing_source_index=int(option.measurement_key.outgoing.source_id),
                incoming_source_index=int(option.measurement_key.incoming.source_id),
                outgoing_boundary_source_seconds=(
                    option.measurement_key.outgoing.end_seconds
                ),
                incoming_boundary_source_seconds=(
                    option.measurement_key.incoming.start_seconds
                ),
                outgoing_exact_sample_seconds=(
                    option.measurement_key.outgoing.end_seconds
                ),
                incoming_exact_sample_seconds=(
                    option.measurement_key.incoming.start_seconds
                ),
                outgoing_preceding_sample_seconds=(
                    option.measurement_key.outgoing.end_seconds - 0.1
                ),
                incoming_following_sample_seconds=(
                    option.measurement_key.incoming.start_seconds + 0.1
                ),
                outgoing_transform=option.outgoing_transform,
                incoming_transform=option.incoming_transform,
                raw_outgoing_descriptor=descriptor,
                raw_outgoing_preceding_descriptor=descriptor,
                raw_incoming_descriptor=descriptor,
                raw_incoming_following_descriptor=descriptor,
                result_outgoing_descriptor=descriptor,
                result_outgoing_preceding_descriptor=descriptor,
                result_incoming_descriptor=descriptor,
                result_incoming_following_descriptor=descriptor,
                raw_evidence=evidence,
                result_evidence=evidence,
                outgoing_face_identity_embedding=(RAW_SENTINEL,),
                incoming_face_identity_embedding=(RAW_SENTINEL,),
                scoring_outgoing=option.scoring_outgoing,
                scoring_incoming=option.scoring_incoming,
            )
            for option in options
        )
        paths: list[SegmentTransformPathMaterializationProof] = []
        for index, candidate in enumerate(candidates):
            start = candidate.start_seconds + (
                offsets[index - 1] / self.fps if index > 0 else 0.0
            )
            end = candidate.end_seconds + (
                offsets[index] / self.fps if index < len(offsets) else 0.0
            )
            entry = (
                options[index - 1].incoming_transform
                if index > 0
                else self._identity(candidate, "entry")
            )
            exit_ = (
                options[index].outgoing_transform
                if index < len(options)
                else self._identity(candidate, "exit")
            )
            duration = end - start
            track = EditorialTransformTrack(
                (
                    self._keyframe(entry, 0.0),
                    self._keyframe(exit_, duration),
                ),
                reason="synthetic_decoded_joint_track",
            )
            proof_identity = (
                segment_bridge_proofs[index - 1].proof_identity
                if 0 < index < len(candidates) - 1
                else f"outer-segment-path:{index}:{candidate.candidate_id}"
            )
            source_offset = 0.25 if candidate.source_id == "0" else 0.50
            duration_frames = duration * self.fps
            center_count = int(
                math.ceil(max(0.0, duration_frames - 0.5 - 1e-9))
            )
            frame_centers = tuple(
                start + (frame_index + 0.5) / self.fps
                for frame_index in range(center_count)
                if start + (frame_index + 0.5) / self.fps < end - 1e-9
            )
            sample_times = (start, *frame_centers, end)
            samples_list: list[SegmentTransformPathSample] = []
            for sample_index, master in enumerate(sample_times):
                sample_transform = (
                    entry
                    if sample_index == 0
                    else exit_
                    if sample_index == len(sample_times) - 1
                    else self._sample_transform(
                        track.evaluate(master - start),
                        track_id=entry.track_id or "missing-track",
                        identity=(
                            "synthetic-sample:"
                            f"{candidate.candidate_id}:{sample_index}"
                        ),
                    )
                )
                samples_list.append(
                    SegmentTransformPathSample(
                    master_seconds=master,
                    source_seconds=master + source_offset,
                    transform=sample_transform,
                    subject_basis=sample_transform.subject_basis,
                    subject_bbox_xywh=(0.3, 0.2, 0.2, 0.4),
                    subject_bbox_coverage=1.0,
                    face_similarity_to_anchor=1.0,
                    face_similarity_to_previous=1.0,
                    descriptor=None,
                    face_identity_embedding=(RAW_SENTINEL,),
                    )
                )
            samples = tuple(samples_list)
            paths.append(
                SegmentTransformPathMaterializationProof(
                    proof_identity=proof_identity,
                    candidate=candidate,
                    source_index=int(candidate.source_id),
                    entry_transform=entry,
                    exit_transform=exit_,
                    editorial_transform_track=track,
                    start_master_seconds=start,
                    end_master_seconds=end,
                    duration_frames=duration_frames,
                    expected_sample_count=len(sample_times),
                    endpoints_included=True,
                    full_frame_centers_included=True,
                    evidence_complete=True,
                    crop_compatible=True,
                    track_compatible=True,
                    failure_codes=(),
                    samples=samples,
                )
            )
        for bridge in segment_bridge_proofs:
            self._bridge_audits[bridge.proof_identity] = {
                "proof_identity": bridge.proof_identity,
                "evidence_complete": True,
                "crop_compatible": True,
                "track_compatible": True,
                "raw_frames_persisted": False,
                "biometric_model_vectors_persisted": False,
            }
        return JointSelectionMaterializationProof(
            selection_identity="synthetic-selection:" + ":".join(
                option.proof_identity for option in options
            ),
            boundary_proof_identities=tuple(
                option.proof_identity for option in options
            ),
            segment_path_proof_identities=tuple(
                path.proof_identity for path in paths
            ),
            boundaries=boundaries,
            segment_paths=tuple(paths),
            selected_bridge_audits=tuple(
                self._bridge_audits[bridge.proof_identity]
                for bridge in segment_bridge_proofs
            ),
        )

    def bridge_audit_for(self, proof_identity: str) -> dict[str, object]:
        raise AssertionError(
            "adapter must consume atomic selected audits, not transient LRU"
        )


def _candidate(segment: int, source: str) -> TimelineCandidate:
    return TimelineCandidate(
        segment_index=segment,
        source_id=source,
        start_seconds=5.0 + segment,
        end_seconds=6.0 + segment,
        quality_score=0.90,
        sync_score=0.90,
        composition_score=0.70,
        boundary_score=0.70,
        candidate_id=f"candidate:{segment}:{source}",
        shot_id=f"shot:{segment}:{source}",
        face_identity_embedding=(RAW_SENTINEL,),
    )


def _timeline(
    lattice: tuple[tuple[TimelineCandidate, ...], ...],
    sources: tuple[str, ...],
    *,
    score: float,
) -> TimelineResult:
    candidates = tuple(
        next(candidate for candidate in lattice[index] if candidate.source_id == source)
        for index, source in enumerate(sources)
    )
    choices = tuple(
        TimelineChoice(
            candidate=candidate,
            emission_score=1.0,
            transition_score=0.0,
            cumulative_score=score - len(candidates) + index + 1,
            used_group_wide_fallback=False,
        )
        for index, candidate in enumerate(candidates)
    )
    return TimelineResult(
        mode=TimelineMode.GENERAL,
        target_member=None,
        choices=choices,
        total_score=score,
    )


def _transform(
    identity: str,
    candidate: TimelineCandidate,
    *,
    boundary_index: int,
) -> BoundaryTransformState:
    if boundary_index == 0:
        crop = (0.08, 0.10, 0.80, 0.80)
        rotation = 0.0
    else:
        crop = (0.10, 0.10, 0.80, 0.80)
        rotation = 0.5
    return BoundaryTransformState(
        transform_identity=identity,
        crop_xywh=crop,
        rotation_degrees=rotation,
        track_id=f"track:{candidate.candidate_id}",
        subject_basis="dominant_face",
    )


def _option_set(
    key: BoundaryMeasurementKey,
    *,
    selected_offset: int,
    variant: str,
) -> tuple[BoundaryOptionSet, AuthorizedBoundaryOption]:
    audits: list[BoundaryOffsetAudit] = []
    selected: AuthorizedBoundaryOption | None = None
    for offset in JOINT_BOUNDARY_FRAME_OFFSETS:
        identity = (
            f"proof:{variant}:{key.boundary_index}:"
            f"{key.outgoing.source_id}:{key.incoming.source_id}:{offset}"
        )
        if offset == selected_offset:
            if key.outgoing.source_id == key.incoming.source_id:
                outgoing_transform = BoundaryTransformState(
                    transform_identity=identity + ":out",
                    crop_xywh=None,
                    rotation_degrees=0.0,
                    track_id=f"track:{key.outgoing.candidate_id}",
                    subject_basis="full_frame_identity",
                )
                incoming_transform = BoundaryTransformState(
                    transform_identity=identity + ":in",
                    crop_xywh=None,
                    rotation_degrees=0.0,
                    track_id=f"track:{key.incoming.candidate_id}",
                    subject_basis="full_frame_identity",
                )
            else:
                outgoing_transform = _transform(
                    identity + ":out",
                    key.outgoing,
                    boundary_index=key.boundary_index,
                )
                incoming_transform = _transform(
                    identity + ":in",
                    key.incoming,
                    boundary_index=key.boundary_index,
                )
            selected = AuthorizedBoundaryOption(
                measurement_key=key,
                frame_offset=offset,
                outgoing_transform=outgoing_transform,
                incoming_transform=incoming_transform,
                proof_identity=identity,
            )
            audits.append(
                BoundaryOffsetAudit(
                    frame_offset=offset,
                    disposition=BoundaryOffsetDisposition.AUTHORIZED,
                    evidence_identity=identity,
                    authorized_option=selected,
                )
            )
        else:
            audits.append(
                BoundaryOffsetAudit(
                    frame_offset=offset,
                    disposition=BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
                    evidence_identity=identity,
                    failure_codes=("synthetic_hard_incompatible",),
                )
            )
    assert selected is not None
    return BoundaryOptionSet(key, tuple(audits)), selected


def _joint(
    timeline: TimelineResult,
    *,
    offsets: tuple[int, ...],
    score: float,
    variant: str,
    allow_same_source_offsets: bool = False,
) -> JointTimelineResult:
    option_sets: list[BoundaryOptionSet] = []
    options: list[AuthorizedBoundaryOption] = []
    for index, (outgoing, incoming) in enumerate(
        zip(timeline.candidates, timeline.candidates[1:])
    ):
        selected_offset = (
            0
            if (
                outgoing.source_id == incoming.source_id
                and not allow_same_source_offsets
            )
            else offsets[index]
        )
        option_set, option = _option_set(
            BoundaryMeasurementKey(index, outgoing, incoming),
            selected_offset=selected_offset,
            variant=variant,
        )
        option_sets.append(option_set)
        options.append(option)
    bridges = tuple(
        SegmentBridgeProof(
            candidate=timeline.candidates[index],
            entry_option=options[index - 1],
            exit_option=options[index],
            duration_frames=(
                timeline.candidates[index].duration_seconds * 10.0
                + options[index].frame_offset
                - options[index - 1].frame_offset
            ),
            proof_identity=f"bridge:{variant}:{index}",
            evidence_complete=True,
            crop_compatible=True,
            track_compatible=True,
        )
        for index in range(1, len(timeline.candidates) - 1)
    )
    scored_timeline = replace(timeline, total_score=score)
    return JointTimelineResult(
        timeline=scored_timeline,
        boundary_options=tuple(options),
        selected_boundary_option_sets=tuple(option_sets),
        segment_bridge_proofs=bridges,
        measured_boundary_option_sets=tuple(option_sets),
        rejection_counts=(),
        option_provider_call_count=len(option_sets),
        bridge_provider_call_count=len(bridges),
    )


def _policy() -> ProfessionalMontageSelectionPolicy:
    return ProfessionalMontageSelectionPolicy(
        highlight_minimum_closeup_fraction=0.25,
        highlight_preferred_closeup_fraction=0.34,
        highlight_minimum_nonwide_fraction=0.49,
        highlight_preferred_nonwide_fraction=0.66,
        build_minimum_closeup_fraction=0.12,
        build_preferred_closeup_fraction=0.25,
        build_minimum_nonwide_fraction=0.15,
        build_preferred_nonwide_fraction=0.31,
        bridge_minimum_closeup_fraction=0.27,
        bridge_preferred_closeup_fraction=0.43,
        bridge_minimum_nonwide_fraction=0.42,
        bridge_preferred_nonwide_fraction=0.58,
        highlight_minimum_source_switches_per_minute=18.6,
        highlight_preferred_source_switches_per_minute=23.5,
        minimum_early_wide_anchor_fraction=1.0 / 3.0,
        minimum_phrase_entry_wide_anchor_fraction=0.28,
    )


def _section_plan() -> SectionPlan:
    return SectionPlan(
        sections=(
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=5.0,
                end_seconds=8.0,
                energy_score=0.9,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=18.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.3,
                establish_with_wide_shot=True,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _candidate_set(
    *,
    role_conditioned: bool = False,
    scene_cut_times: tuple[tuple[float, ...], ...] = ((), ()),
) -> VisualTimelineCandidateSet:
    lattice = tuple(
        (_candidate(segment, "0"), _candidate(segment, "1"))
        for segment in range(3)
    )
    paths = tuple(product(("0", "1"), repeat=3))
    timelines = tuple(
        _timeline(lattice, path, score=20.0 - index)
        for index, path in enumerate(paths)
    )
    reviews = tuple(
        review_timeline(timeline, available_source_count=2)
        for timeline in timelines
    )
    ranking = _review_ranked_candidate_indices(reviews)
    rank_by_index = {value: rank for rank, value in enumerate(ranking)}
    records = tuple(
        VisualTimelineCandidateRecord(
            candidate_index=index,
            cadence_profile=("balanced", "reference", "dense_safe")[index % 3],
            timeline_result=timeline,
            review=reviews[index],
            review_rank=rank_by_index[index],
            selected=index == ranking[0],
        )
        for index, timeline in enumerate(timelines)
    )
    section_plan = _section_plan() if role_conditioned else None
    policy = _policy() if role_conditioned else None
    workspace_identity = _visual_optimizer_input_identity(
        lattice,
        sources=(Path("source-0.mp4"), Path("source-1.mp4")),
        audio_offsets=(0.25, 0.50),
        source_timewarps=(None, None),
        source_clock_basis="global_scalar_offset",
        scene_cut_times=scene_cut_times,
        preferred_source_ids=("0", "1", "0"),
        section_plan=section_plan,
        phrase_times=(6.0,),
        professional_montage_policy=policy,
        maximum_results=8,
    )
    workspace = VisualTimelineOptimizerWorkspace(
        candidates_by_segment=lattice,
        input_identity=workspace_identity,
        preferred_source_ids=("0", "1", "0"),
        section_plan=section_plan,
        phrase_times=(6.0,),
        professional_montage_policy=policy,
        maximum_results=8,
    )
    return VisualTimelineCandidateSet(
        candidates=records,
        selected_candidate_index=ranking[0],
        sources=(Path("source-0.mp4"), Path("source-1.mp4")),
        audio_offsets=(0.25, 0.50),
        source_timewarps=(None, None),
        source_clock_basis="global_scalar_offset",
        requested_maximum_results=8,
        scene_cut_times=scene_cut_times,
        optimizer_workspace=workspace,
        optimizer_input_identity=workspace_identity,
    )


def _optimizer_seed(
    candidate_set: VisualTimelineCandidateSet,
) -> VisualTimelineOptimizerSeed:
    workspace = candidate_set.optimizer_workspace
    assert workspace is not None
    optimizer_input_identity = candidate_set.optimizer_input_identity
    assert optimizer_input_identity is not None
    return VisualTimelineOptimizerSeed(
        optimizer_workspace=workspace,
        optimizer_input_identity=optimizer_input_identity,
        sources=candidate_set.sources,
        audio_offsets=candidate_set.audio_offsets,
        source_timewarps=candidate_set.source_timewarps,
        source_clock_basis=candidate_set.source_clock_basis,
        requested_maximum_results=candidate_set.requested_maximum_results,
        scene_cut_times=candidate_set.scene_cut_times,
        face_metadata_by_candidate=candidate_set.face_metadata_by_candidate,
        subject_metadata_by_candidate=(
            candidate_set.subject_metadata_by_candidate
        ),
        minimum_materialization_target=(
            candidate_set.minimum_materialization_target
        ),
    )


def _profile_pools(
    candidate_set: VisualTimelineCandidateSet,
) -> tuple[tuple[JointTimelineResult, ...], ...]:
    assert candidate_set.optimizer_workspace is not None
    lattice = candidate_set.optimizer_workspace.candidates_by_segment
    paths = tuple(product(("0", "1"), repeat=3))

    def result(
        path_index: int,
        *,
        score: float,
        variant: str,
        offsets: tuple[int, int] = (0, 0),
    ) -> JointTimelineResult:
        return _joint(
            _timeline(lattice, paths[path_index], score=score),
            offsets=offsets,
            score=score,
            variant=variant,
        )

    return (
        (
            result(0, score=10.0, variant="low", offsets=(1, -1)),
            result(1, score=19.0, variant="balanced-1"),
            result(4, score=18.0, variant="balanced-4"),
        ),
        (
            result(0, score=30.0, variant="high", offsets=(2, -2)),
            result(2, score=20.0, variant="reference-2"),
            result(5, score=17.0, variant="reference-5"),
        ),
        (
            result(3, score=21.0, variant="dense-3"),
            result(6, score=16.0, variant="dense-6"),
            result(7, score=15.0, variant="dense-7"),
        ),
    )


def _limited_profile_pools(
    candidate_set: VisualTimelineCandidateSet,
    *,
    unique_path_count: int,
    alternate_paths: tuple[int, ...],
) -> tuple[tuple[JointTimelineResult, ...], ...]:
    """Build deterministic primary paths plus render-distinct proof variants."""

    assert candidate_set.optimizer_workspace is not None
    lattice = candidate_set.optimizer_workspace.candidates_by_segment
    paths = tuple(product(("0", "1"), repeat=3))
    primary_path_indices = (1, 2, 3, 4, 5, 6, 0, 7)
    pools: list[list[JointTimelineResult]] = [[], [], []]
    for path_index in range(unique_path_count):
        actual_path_index = primary_path_indices[path_index]
        pools[path_index % 3].append(
            _joint(
                _timeline(
                    lattice,
                    paths[actual_path_index],
                    score=40.0 - path_index,
                ),
                offsets=(0, 0),
                score=40.0 - path_index,
                variant=f"primary-{path_index}",
            )
        )
    alternate_offsets = ((1, -1), (-1, 1))
    for alternate_index, path_index in enumerate(alternate_paths):
        actual_path_index = primary_path_indices[path_index]
        pools[alternate_index % 3].append(
            _joint(
                _timeline(
                    lattice,
                    paths[actual_path_index],
                    score=20.0 - alternate_index,
                ),
                offsets=alternate_offsets[alternate_index % 2],
                score=20.0 - alternate_index,
                variant=f"alternate-{alternate_index}-{path_index}",
            )
        )
    return tuple(tuple(pool) for pool in pools)


class JointVisualCandidateAdapterTests(unittest.TestCase):
    def test_one_off_decoded_wide_observation_stays_audited_only(self) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        lattice = workspace.candidates_by_segment
        target = lattice[0][0]
        feedback = JointDecodedMontageFeedback(
            observations=(
                JointDecodedMontageSegmentObservation.from_choice(
                    candidate_set.candidates[0].timeline_result.choices[0],
                    render_variant_signature="a" * 64,
                    sampled_proxy_shot_classes=(
                        "wide",
                        "medium",
                        "medium",
                    ),
                ),
            )
        )

        shadow, metadata, audit = _decoded_montage_role_control_inputs(
            lattice,
            workspace.subject_metadata_by_candidate,
            feedback,
        )

        self.assertIs(shadow[0][0], target)
        self.assertEqual(metadata, workspace.subject_metadata_by_candidate)
        self.assertEqual(audit["promoted_wide_sample_count"], 0)
        self.assertEqual(
            audit["insufficient_distinct_variant_sample_count"],
            1,
        )
        self.assertEqual(audit["hard_no_good_count"], 0)
        self.assertFalse(audit["availability_override_applied"])

    def test_repeated_distinct_decoded_wide_and_unknown_override_shadow_only(
        self,
    ) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        lattice = workspace.candidates_by_segment
        target = lattice[0][0]
        feedback = JointDecodedMontageFeedback(
            observations=tuple(
                JointDecodedMontageSegmentObservation.from_candidate(
                    target,
                    render_variant_signature=signature * 64,
                    sampled_proxy_shot_classes=(
                        "wide",
                        "unknown",
                        "medium",
                    ),
                )
                for signature in ("a", "b")
            )
        )

        shadow, metadata, audit = _decoded_montage_role_control_inputs(
            lattice,
            workspace.subject_metadata_by_candidate,
            feedback,
        )

        self.assertIs(target.shot_type, ShotType.MEDIUM)
        self.assertIsNone(target.entry_shot_type)
        self.assertIs(shadow[0][0].entry_shot_type, ShotType.WIDE)
        self.assertIs(shadow[0][0].shot_type, ShotType.MEDIUM)
        self.assertEqual(
            metadata[target.candidate_id].shot_scale_bases,
            ("legacy", "unknown_no_subject_geometry", "legacy"),
        )
        self.assertEqual(audit["promoted_wide_sample_count"], 1)
        self.assertEqual(audit["promoted_unknown_sample_count"], 1)
        self.assertEqual(audit["role_control_shadow_candidate_count"], 1)
        self.assertEqual(audit["hard_no_good_count"], 0)
        self.assertTrue(audit["availability_override_applied"])

    def test_decoded_feedback_rejects_a_foreign_exact_candidate_clock(self) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        feedback = JointDecodedMontageFeedback(
            observations=(
                JointDecodedMontageSegmentObservation.from_candidate(
                    _candidate(9, "0"),
                    render_variant_signature="a" * 64,
                    sampled_proxy_shot_classes=(
                        "wide",
                        "medium",
                        "medium",
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "does not belong"):
            _decoded_montage_role_control_inputs(
                workspace.candidates_by_segment,
                workspace.subject_metadata_by_candidate,
                feedback,
            )

    def test_metadata_only_close_is_not_counted_as_cold_start_availability(
        self,
    ) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        lattice = workspace.candidates_by_segment
        close_without_bbox = replace(
            lattice[0][0],
            shot_type=ShotType.CLOSE_UP,
            entry_shot_type=ShotType.CLOSE_UP,
            exit_shot_type=ShotType.CLOSE_UP,
        )
        close_lattice = (
            (close_without_bbox, lattice[0][1]),
            *lattice[1:],
        )

        shadow, _metadata, audit = _decoded_montage_role_control_inputs(
            close_lattice,
            workspace.subject_metadata_by_candidate,
            None,
        )

        self.assertIs(close_without_bbox.shot_type, ShotType.CLOSE_UP)
        self.assertIs(shadow[0][0].entry_shot_type, ShotType.MEDIUM)
        self.assertIs(shadow[0][0].shot_type, ShotType.MEDIUM)
        self.assertIs(shadow[0][0].exit_shot_type, ShotType.MEDIUM)
        self.assertEqual(
            audit[
                "cold_start_metadata_close_without_bbox_demoted_sample_count"
            ],
            3,
        )

        close_with_bbox = replace(
            close_without_bbox,
            focus_bbox_xywh=(0.25, 0.20, 0.30, 0.50),
        )
        bbox_lattice = (
            (close_with_bbox, lattice[0][1]),
            *lattice[1:],
        )
        bbox_shadow, _bbox_metadata, bbox_audit = (
            _decoded_montage_role_control_inputs(
                bbox_lattice,
                workspace.subject_metadata_by_candidate,
                None,
            )
        )
        self.assertIs(bbox_shadow[0][0], close_with_bbox)
        self.assertEqual(
            bbox_audit[
                "cold_start_metadata_close_without_bbox_demoted_sample_count"
            ],
            0,
        )

    def test_selection_set_rejects_failed_review_with_two_source_fixture(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        pools = _profile_pools(candidate_set)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
            )

        failed_record = replace(
            selected.candidate_records[0],
            review=replace(
                selected.candidate_records[0].review,
                hard_failures=("synthetic_global_review_failure",),
            ),
        )
        with self.assertRaisesRegex(ValueError, "failed review"):
            replace(
                selected,
                candidate_records=(
                    failed_record,
                    *selected.candidate_records[1:],
                ),
            )

    def test_two_source_selector_cannot_bypass_production_review_floor(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )

        def failed_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            return replace(
                _synthetic_review(
                    plan,
                    available_source_count=available_source_count,
                ),
                hard_failures=("synthetic_global_review_failure",),
            )

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=pools,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=failed_review,
            ),
        ):
            with self.assertRaises(NoCompleteTimelinePath):
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                )

        self.assertEqual(provider.selection_materialization_calls, 8)

    def test_same_source_native_continuity_does_not_require_crosscut_coverage(
        self,
    ) -> None:
        class IncompleteCoverageProvider(StubMaterializingProvider):
            @staticmethod
            def _evidence() -> BoundaryVisualEvidence:
                base = StubMaterializingProvider._evidence()
                return replace(
                    base,
                    evidence_coverage=replace(
                        base.evidence_coverage,
                        professional_evidence_basis=(
                            "insufficient_combined_geometry_evidence"
                        ),
                        professional_gate_passed=False,
                        missing_mask=("outgoing_exact_actual_pose",),
                    ),
                )

        candidate_set = _candidate_set()
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        result = _joint(
            _timeline(
                workspace.candidates_by_segment,
                ("0", "0", "0"),
                score=10.0,
            ),
            offsets=(0, 0),
            score=10.0,
            variant="same-source-incomplete",
        )
        selection = JointVisualSelectionRecord(
            cadence_profile="balanced",
            joint_result=result,
            shifted_timeline_result=_shift_joint_timeline_result(
                result,
                fps=10.0,
                candidate_set=candidate_set,
            ),
            source_shot_path=tuple(
                (item.segment_index, item.source_id, item.shot_id)
                for item in result.candidates
            ),
            proof_variant_signature=tuple(
                item.proof_identity for item in result.boundary_options
            ),
        )
        provider = IncompleteCoverageProvider(fps=10.0)
        plan = _project_visual_plan_from_joint_proof(
            candidate_set=candidate_set,
            timeline_result=selection.shifted_timeline_result,
            proof_bundle=_proof_bundle(selection, provider),
            candidate_index=0,
            cadence_profile="balanced",
        )
        reviewed = review_timeline(plan, available_source_count=2)
        self.assertTrue(all(not item.source_switch for item in reviewed.boundaries))

    def test_cross_source_coverage_violation_is_exact_and_privacy_safe(
        self,
    ) -> None:
        class IncompleteCoverageProvider(StubMaterializingProvider):
            @staticmethod
            def _evidence() -> BoundaryVisualEvidence:
                base = StubMaterializingProvider._evidence()
                return replace(
                    base,
                    evidence_coverage=replace(
                        base.evidence_coverage,
                        professional_evidence_basis=(
                            "insufficient_combined_geometry_evidence"
                        ),
                        professional_gate_passed=False,
                        missing_mask=(
                            "incoming_temporal_actual_pose",
                            "incoming_silhouette",
                        ),
                    ),
                )

        candidate_set = _candidate_set()
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        result = _joint(
            _timeline(
                workspace.candidates_by_segment,
                ("0", "1", "0"),
                score=10.0,
            ),
            offsets=(1, -1),
            score=10.0,
            variant="cross-source-incomplete",
        )
        selection = JointVisualSelectionRecord(
            cadence_profile="dense_safe",
            joint_result=result,
            shifted_timeline_result=_shift_joint_timeline_result(
                result,
                fps=10.0,
                candidate_set=candidate_set,
            ),
            source_shot_path=tuple(
                (item.segment_index, item.source_id, item.shot_id)
                for item in result.candidates
            ),
            proof_variant_signature=tuple(
                item.proof_identity for item in result.boundary_options
            ),
        )
        provider = IncompleteCoverageProvider(fps=10.0)
        with self.assertRaises(JointVisualEvidenceCoverageError) as caught:
            _project_visual_plan_from_joint_proof(
                candidate_set=candidate_set,
                timeline_result=selection.shifted_timeline_result,
                proof_bundle=_proof_bundle(selection, provider),
                candidate_index=7,
                cadence_profile="dense_safe",
            )
        payload = caught.exception.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["joint_candidate_index"], 7)
        self.assertEqual(payload["cadence_profile"], "dense_safe")
        self.assertEqual(payload["boundary_index"], 0)
        self.assertEqual(payload["frame_offset"], 1)
        self.assertEqual(
            payload["coverage"]["failed_coverage_fields"],
            ["incoming_temporal_actual_pose", "incoming_silhouette"],
        )
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn(str(RAW_SENTINEL), serialized)
        self.assertNotIn(str(RAW_SENTINEL), str(caught.exception))
        self.assertFalse(payload["raw_frames_persisted"])
        self.assertFalse(payload["raw_visual_descriptors_persisted"])

    def test_selector_turns_cross_source_coverage_failure_into_exact_retry(
        self,
    ) -> None:
        class IncompleteCoverageProvider(StubMaterializingProvider):
            @staticmethod
            def _evidence() -> BoundaryVisualEvidence:
                base = StubMaterializingProvider._evidence()
                return replace(
                    base,
                    evidence_coverage=replace(
                        base.evidence_coverage,
                        professional_evidence_basis=(
                            "insufficient_combined_geometry_evidence"
                        ),
                        professional_gate_passed=False,
                        missing_mask=("incoming_temporal_actual_pose",),
                    ),
                )

        candidate_set = _candidate_set()
        provider = IncompleteCoverageProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )
        selector_calls: list[dict[str, object]] = []

        def selector(*_args: object, **kwargs: object):
            selector_calls.append(dict(kwargs))
            return pools[(len(selector_calls) - 1) % 3]

        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=selector,
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                )

        self.assertEqual(len(selector_calls), 6)
        self.assertEqual(provider.selection_materialization_calls, 8)
        first_retry_no_goods = set(
            selector_calls[3]["forbidden_boundary_selections"]
        )
        self.assertTrue(first_retry_no_goods)
        self.assertTrue(
            all(
                set(call["forbidden_boundary_selections"])
                == first_retry_no_goods
                for call in selector_calls[3:]
            )
        )
        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_unique_path_floor_not_met",
        )

    def test_authoritative_result_evidence_blocks_midpoint_and_hog_fallback(
        self,
    ) -> None:
        class ExactUnavailableProvider(StubMaterializingProvider):
            @staticmethod
            def _evidence() -> BoundaryVisualEvidence:
                base = StubMaterializingProvider._evidence()
                return replace(
                    base,
                    silhouette_mismatch=None,
                    pose_mismatch=0.05,
                    evidence_coverage=replace(
                        base.evidence_coverage,
                        silhouette_available=(False, False),
                    ),
                )

        candidate_set = _candidate_set()
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        face_metadata = {
            candidate.candidate_id: _CandidateFaceMetadata(
                boxes=(
                    (
                        (0.05, 0.05, 0.10, 0.10)
                        if candidate.source_id == "0"
                        else (0.80, 0.80, 0.10, 0.10)
                    ),
                )
                * 3,
                failure=None,
                backends=("synthetic",) * 3,
                landmarks=(None, None, None),
                rolls=(None, None, None),
                eye_scales=(None, None, None),
            )
            for segment in workspace.candidates_by_segment
            for candidate in segment
        }
        candidate_set = replace(
            candidate_set,
            face_metadata_by_candidate=face_metadata,
        )

        def contaminated(result: JointTimelineResult) -> JointTimelineResult:
            choices = tuple(
                replace(
                    choice,
                    candidate=replace(
                        choice.candidate,
                        structure_embedding=(
                            (0.0, 0.0)
                            if choice.candidate.source_id == "0"
                            else (1.0, 1.0)
                        ),
                        silhouette_embedding=(
                            (0.0, 0.0)
                            if choice.candidate.source_id == "0"
                            else (1.0, 1.0)
                        ),
                    ),
                )
                for choice in result.timeline.choices
            )
            return replace(
                result,
                timeline=replace(result.timeline, choices=choices),
            )

        pools = tuple(
            tuple(contaminated(result) for result in pool)
            for pool in _profile_pools(candidate_set)
        )
        provider = ExactUnavailableProvider(fps=10.0)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )

        switched_boundary_count = 0
        for record in selected.candidate_records:
            for boundary in record.review.boundaries:
                if not boundary.source_switch:
                    continue
                switched_boundary_count += 1
                self.assertEqual(boundary.focus_mismatch, 0.05)
                self.assertIsNone(boundary.silhouette_mismatch)
                self.assertEqual(boundary.pose_mismatch, 0.05)
                self.assertTrue(
                    {"focus", "silhouette", "pose"}.isdisjoint(
                        boundary.exceeded_metrics
                    )
                )
        self.assertGreater(switched_boundary_count, 0)

        materialized = materialize_joint_visual_candidates(
            selected,
            provider=provider,
            fps=10.0,
            maximum_candidates=8,
        )
        self.assertEqual(len(materialized), 8)
        for item in materialized:
            self.assertEqual(
                _synthetic_review(
                    item.visual_plan,
                    available_source_count=2,
                ).to_dict(),
                item.candidate_record.review.to_dict(),
            )

        spoof_source = next(
            item
            for item in materialized
            if any(boundary.source_switch for boundary in item.candidate_record.review.boundaries)
        )
        spoof_boundary = next(
            boundary
            for boundary in spoof_source.candidate_record.review.boundaries
            if boundary.source_switch
        )
        serialized = spoof_source.visual_plan.to_dict()
        outgoing = serialized["choices"][spoof_boundary.boundary_index]
        incoming = serialized["choices"][spoof_boundary.boundary_index + 1]
        self.assertNotIn(
            "transition_evidence_authority_from_previous",
            incoming,
        )
        outgoing["face_samples_xywh"][-1] = [0.02, 0.02, 0.10, 0.10]
        incoming["face_samples_xywh"][0] = [0.88, 0.88, 0.10, 0.10]
        incoming["transition_evidence_contract_from_previous"] = (
            JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT
        )
        incoming["focus_transition_mismatch_from_previous"] = None
        spoofed_review = review_timeline(
            serialized,
            available_source_count=2,
        )
        self.assertGreater(
            spoofed_review.boundaries[
                spoof_boundary.boundary_index
            ].focus_mismatch
            or 0.0,
            0.24,
        )

    def test_optimizer_seed_and_full_candidate_set_produce_same_joint_result(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        seed = _optimizer_seed(candidate_set)
        pools = _profile_pools(candidate_set)

        def select(carrier, provider):  # type: ignore[no-untyped-def]
            with patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=pools,
            ):
                return select_joint_visual_candidates(
                    carrier,
                    provider=provider,
                    fps=10.0,
                )

        full_provider = StubMaterializingProvider(fps=10.0)
        seed_provider = StubMaterializingProvider(fps=10.0)
        selected_full = select(candidate_set, full_provider)
        selected_seed = select(seed, seed_provider)

        self.assertEqual(selected_seed.to_dict(), selected_full.to_dict())
        self.assertEqual(selected_seed.selections, selected_full.selections)
        self.assertEqual(
            selected_seed.candidate_records,
            selected_full.candidate_records,
        )
        materialized_full = materialize_joint_visual_candidates(
            selected_full,
            provider=full_provider,
            fps=10.0,
            maximum_candidates=8,
        )
        materialized_seed = materialize_joint_visual_candidates(
            selected_seed,
            provider=seed_provider,
            fps=10.0,
            maximum_candidates=8,
        )
        self.assertEqual(
            [item.candidate_history_entry for item in materialized_seed],
            [item.candidate_history_entry for item in materialized_full],
        )
        self.assertEqual(
            [item.visual_plan.to_dict() for item in materialized_seed],
            [item.visual_plan.to_dict() for item in materialized_full],
        )
        self.assertEqual(
            [item.segments for item in materialized_seed],
            [item.segments for item in materialized_full],
        )

    def test_real_three_profile_search_reuses_scalar_bridges_and_keeps_exact_calls(
        self,
    ) -> None:
        provider = CountingStubDecodedProvider()
        progress: list[JointTimelineProgress] = []

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_visual_candidates(
                _candidate_set(),
                provider=provider,
                fps=10.0,
                progress_callback=progress.append,
            )

        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_unique_path_floor_not_met",
        )
        self.assertEqual(len(provider.option_calls), 8)
        self.assertEqual(set(provider.option_calls.values()), {1})
        completed = [item for item in progress if item.phase == "profile_complete"]
        self.assertEqual(
            [item.cadence_profile for item in completed],
            ["balanced", "reference", "dense_safe"],
        )
        self.assertEqual(
            [item.bridge_provider_call_count for item in completed],
            [2, 2, 2],
        )
        self.assertEqual(
            [item.bridge_disposition_batch_call_count for item in completed],
            [1, 0, 0],
        )
        self.assertEqual(
            [item.bridge_disposition_batch_request_count for item in completed],
            [2, 0, 0],
        )
        self.assertEqual(
            [item.shared_bridge_cache_hit_count for item in completed],
            [0, 2, 4],
        )
        self.assertEqual(provider.bridge_calls, 6)
        self.assertEqual(len(provider.bridge_batch_calls), 1)
        self.assertEqual(len(provider.bridge_batch_calls[0]), 2)

    def test_authoritative_floor_retries_only_exact_failed_offsets(self) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        first_round = _profile_pools(candidate_set)
        retry_round = tuple(
            tuple(
                _joint(
                    result.timeline,
                    offsets=(1, -1),
                    score=result.total_score + 0.25,
                    variant=f"retry-{profile_index}-{result_index}",
                )
                for result_index, result in enumerate(pool)
            )
            for profile_index, pool in enumerate(first_round)
        )
        selector_calls: list[dict[str, object]] = []
        selector_pools = (*first_round, *retry_round)

        def selector(*_args: object, **kwargs: object):
            selector_calls.append(dict(kwargs))
            return selector_pools[len(selector_calls) - 1]

        review_calls = 0

        def controlled_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            nonlocal review_calls
            reviewed = review_timeline(
                plan,
                available_source_count=available_source_count,
            )
            review_calls += 1
            if review_calls > 9:
                return replace(reviewed, hard_failures=())
            boundaries = list(reviewed.boundaries)
            switched = next(
                (item for item in boundaries if item.source_switch),
                None,
            )
            if switched is not None:
                boundaries[switched.boundary_index] = replace(
                    switched,
                    exceeded_metrics=("pose",),
                )
            return replace(reviewed, boundaries=tuple(boundaries))

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=controlled_review,
            ),
        ):
            selected = _production_select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertTrue(all(item.review.passed for item in selected.candidate_records))
        self.assertEqual(len(selector_calls), 6)
        self.assertTrue(
            all(not call["forbidden_boundary_selections"] for call in selector_calls[:3])
        )
        retry_no_goods = tuple(
            selector_calls[3]["forbidden_boundary_selections"]
        )
        self.assertTrue(retry_no_goods)
        self.assertTrue(
            all(
                set(call["forbidden_boundary_selections"])
                == set(retry_no_goods)
                for call in selector_calls[3:]
            )
        )
        self.assertTrue(
            all(
                isinstance(key, BoundaryMeasurementKey)
                and offset == 0
                for key, offset in retry_no_goods
            )
        )
        self.assertEqual(
            selected.selection_retry_audit["retry_round_count"],
            1,
        )
        self.assertEqual(
            selected.selection_retry_audit["global_profile_attempt_count"],
            6,
        )

    def test_first_passing_eight_stop_before_ninth_proof(self) -> None:
        self.assertEqual(MAXIMUM_JOINT_VISUAL_PROOF_CALLS, 32)
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=_profile_pools(candidate_set),
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(provider.selection_materialization_calls, 8)
        audit = selected.selection_retry_audit
        self.assertEqual(audit["proof_materialization_call_count"], 8)
        self.assertEqual(
            audit["maximum_proof_materialization_calls"],
            MAXIMUM_JOINT_VISUAL_PROOF_CALLS,
        )
        self.assertEqual(audit["total_unique_candidate_attempt_count"], 8)
        self.assertEqual(audit["review_cache_hit_count"], 0)
        self.assertEqual(audit["total_round_count"], 1)
        self.assertEqual(audit["retry_round_count"], 0)

    def test_external_feedback_is_validated_and_forwarded_to_every_profile(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        pools = _profile_pools(candidate_set)
        result = pools[0][0]
        option = result.boundary_options[0]
        boundary_no_good: BoundarySelectionNoGood = (
            option.measurement_key,
            option.frame_offset,
        )
        complete_path_no_good = result.complete_path_signature
        selector_calls: list[dict[str, object]] = []

        def selector(*_args: object, **kwargs: object):
            selector_calls.append(dict(kwargs))
            return pools[len(selector_calls) - 1]

        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=selector,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubMaterializingProvider(fps=10.0),
                fps=10.0,
                forbidden_boundary_selections=(
                    boundary_no_good,
                    boundary_no_good,
                ),
                forbidden_complete_paths=(
                    complete_path_no_good,
                    complete_path_no_good,
                ),
                maximum_proof_materialization_calls=8,
            )

        self.assertEqual(len(selector_calls), 3)
        for call in selector_calls:
            self.assertEqual(
                call["forbidden_boundary_selections"],
                frozenset((boundary_no_good,)),
            )
            self.assertEqual(
                call["forbidden_complete_paths"],
                frozenset((complete_path_no_good,)),
            )
        audit = selected.selection_retry_audit
        self.assertEqual(audit["initial_exact_boundary_no_good_count"], 1)
        self.assertEqual(audit["exact_boundary_no_good_count"], 1)
        self.assertEqual(audit["external_complete_path_no_good_count"], 1)
        self.assertEqual(audit["maximum_proof_materialization_calls"], 8)

    def test_external_feedback_malformed_values_fail_before_proof_calls(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        pools = _profile_pools(candidate_set)
        result = pools[0][0]
        path = result.complete_path_signature
        option = result.boundary_options[0]
        foreign_path = replace(
            path,
            candidate_clocks=(
                replace(path.candidate_clocks[0], source_id="foreign"),
                *path.candidate_clocks[1:],
            ),
        )
        short_path = JointTimelinePathSignature(
            candidate_clocks=(path.candidate_clocks[0],),
            boundary_proofs=(),
        )
        foreign_key = BoundaryMeasurementKey.from_candidates(
            option.measurement_key.boundary_index,
            replace(
                option.measurement_key.outgoing,
                candidate_id="foreign-candidate",
            ),
            option.measurement_key.incoming,
        )
        invalid_cases = (
            (
                "boundary_raw",
                {"forbidden_boundary_selections": (("bad", 0),)},
                TypeError,
                "BoundaryMeasurementKey",
            ),
            (
                "boundary_bool",
                {
                    "forbidden_boundary_selections": (
                        ((option.measurement_key, True)),
                    )
                },
                TypeError,
                "must be an integer",
            ),
            (
                "boundary_foreign",
                {"forbidden_boundary_selections": ((foreign_key, 0),)},
                ValueError,
                "does not belong to the candidate lattice",
            ),
            (
                "complete_raw",
                {"forbidden_complete_paths": (("bad",),)},
                TypeError,
                "JointTimelinePathSignature",
            ),
            (
                "complete_mapping",
                {"forbidden_complete_paths": {"bad": path}},
                TypeError,
                "must be a collection",
            ),
            (
                "complete_foreign",
                {"forbidden_complete_paths": (foreign_path,)},
                ValueError,
                "does not belong to the candidate lattice",
            ),
            (
                "complete_short",
                {"forbidden_complete_paths": (short_path,)},
                ValueError,
                "length does not match the candidate lattice",
            ),
        )
        for name, kwargs, exception_type, message in invalid_cases:
            provider = StubMaterializingProvider(fps=10.0)
            with self.subTest(name=name):
                with self.assertRaisesRegex(exception_type, message):
                    _production_select_joint_visual_candidates(
                        candidate_set,
                        provider=provider,
                        fps=10.0,
                        **kwargs,
                    )
            self.assertEqual(provider.selection_materialization_calls, 0)

    def test_proof_materialization_budget_accepts_only_eight_through_32(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        for invalid in (True, 7, 33, 8.0):
            with self.subTest(invalid=invalid):
                provider = StubMaterializingProvider(fps=10.0)
                with self.assertRaisesRegex(
                    ValueError,
                    "between 8 and 32",
                ):
                    _production_select_joint_visual_candidates(
                        candidate_set,
                        provider=provider,
                        fps=10.0,
                        maximum_proof_materialization_calls=invalid,
                    )
                self.assertEqual(provider.validated_segment_counts, [])

        for valid in (8, 32):
            with self.subTest(valid=valid):
                provider = StubMaterializingProvider(fps=10.0)
                with patch(
                    "crosscut.joint_visual_candidate_adapter."
                    "select_joint_timeline_candidates",
                    side_effect=_profile_pools(candidate_set),
                ):
                    selected = select_joint_visual_candidates(
                        candidate_set,
                        provider=provider,
                        fps=10.0,
                        maximum_proof_materialization_calls=valid,
                    )
                self.assertEqual(
                    selected.selection_retry_audit[
                        "maximum_proof_materialization_calls"
                    ],
                    valid,
                )
                self.assertEqual(provider.selection_materialization_calls, 8)

    def test_complete_path_exhaustion_is_typed_across_all_profiles(self) -> None:
        candidate_set = _candidate_set()
        path_no_good = _profile_pools(candidate_set)[0][
            0
        ].complete_path_signature
        selector_calls: list[dict[str, object]] = []

        def exhausted(*_args: object, **kwargs: object):
            selector_calls.append(dict(kwargs))
            raise NoCompleteTimelinePath(
                "synthetic exact path exhaustion",
                reason_code=(
                    "joint_timeline_complete_path_no_good_exhausted"
                ),
                rejection_counts={
                    "complete_path_external_no_good_rejected": 1
                },
            )

        provider = StubMaterializingProvider(fps=10.0)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=exhausted,
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                    forbidden_complete_paths=(path_no_good,),
                )

        self.assertEqual(len(selector_calls), 3)
        self.assertTrue(
            all(
                call["forbidden_complete_paths"]
                == frozenset((path_no_good,))
                for call in selector_calls
            )
        )
        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_complete_path_no_good_exhausted",
        )
        self.assertEqual(
            tuple(name for name, _failure in caught.exception.profile_failures),
            ("balanced", "reference", "dense_safe"),
        )
        self.assertEqual(
            caught.exception.rejection_counts,
            {
                "exhausted_profile_count": 3,
                "forbidden_complete_path_count": 1,
                "proof_materialization_call_count": 0,
            },
        )
        self.assertEqual(provider.selection_materialization_calls, 0)

    def test_repeated_review_failures_stop_at_proof_budget(self) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )

        def rejected_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            reviewed = _synthetic_review(
                plan,
                available_source_count=available_source_count,
            )
            boundaries = tuple(
                replace(boundary, exceeded_metrics=("pose",))
                if boundary.source_switch
                else boundary
                for boundary in reviewed.boundaries
            )
            return replace(
                reviewed,
                boundaries=boundaries,
                hard_failures=("synthetic_repeated_review_failure",),
            )

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=pools,
            ) as selector_mock,
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=rejected_review,
            ),
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                    maximum_proof_materialization_calls=8,
                )

        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_proof_budget_exhausted",
        )
        self.assertEqual(provider.selection_materialization_calls, 8)
        self.assertEqual(selector_mock.call_count, 3)
        self.assertEqual(
            caught.exception.rejection_counts[
                "proof_materialization_call_count"
            ],
            8,
        )
        self.assertEqual(
            caught.exception.rejection_counts[
                "maximum_proof_materialization_calls"
            ],
            8,
        )

    def test_repeated_profile_failure_names_are_coalesced_across_retry(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )
        selector_call_count = 0

        def selector(*_args: object, **_kwargs: object):
            nonlocal selector_call_count
            profile_index = selector_call_count % 3
            selector_call_count += 1
            if profile_index == 0:
                raise NoCompleteTimelinePath(
                    "synthetic repeated cadence failure",
                    reason_code="synthetic_repeated_cadence_failure",
                )
            return pools[profile_index]

        def rejected_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            reviewed = _synthetic_review(
                plan,
                available_source_count=available_source_count,
            )
            boundaries = tuple(
                replace(boundary, exceeded_metrics=("pose",))
                if boundary.source_switch
                else boundary
                for boundary in reviewed.boundaries
            )
            return replace(
                reviewed,
                boundaries=boundaries,
                hard_failures=("synthetic_boundary_review_failure",),
            )

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=rejected_review,
            ),
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                )

        self.assertEqual(selector_call_count, 6)
        self.assertEqual(
            tuple(name for name, _failure in caught.exception.profile_failures),
            ("balanced",),
        )

    def test_global_only_review_rejection_is_lightweight_cached_on_retry(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )
        selector_call_count = 0
        review_call_count = 0

        def selector(*_args: object, **_kwargs: object):
            nonlocal selector_call_count
            result = pools[selector_call_count % 3]
            selector_call_count += 1
            return result

        def controlled_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            nonlocal review_call_count
            review_call_count += 1
            reviewed = _synthetic_review(
                plan,
                available_source_count=available_source_count,
            )
            if review_call_count == 1:
                return replace(
                    reviewed,
                    hard_failures=("synthetic_global_only_failure",),
                )
            if review_call_count == 2:
                boundaries = list(reviewed.boundaries)
                switched = next(
                    item for item in boundaries if item.source_switch
                )
                boundaries[switched.boundary_index] = replace(
                    switched,
                    exceeded_metrics=("pose",),
                )
                return replace(
                    reviewed,
                    boundaries=tuple(boundaries),
                    hard_failures=("synthetic_local_boundary_failure",),
                )
            return reviewed

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=controlled_review,
            ),
        ):
            with self.assertRaises(NoCompleteTimelinePath):
                _production_select_joint_visual_candidates(
                    candidate_set,
                    provider=provider,
                    fps=10.0,
                )

        self.assertEqual(selector_call_count, 6)
        self.assertEqual(review_call_count, 8)
        self.assertEqual(provider.selection_materialization_calls, 8)

    def test_retry_reuses_passing_proofs_without_spending_budget_again(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        first_round = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )
        replacement_index = 0
        second_round_lists: list[list[JointTimelineResult]] = [[], [], []]
        replacement_offsets = ((2, -2), (-2, 2))
        for profile_index, pool in enumerate(first_round):
            for result in pool:
                if result.total_score >= 30.0:
                    second_round_lists[profile_index].append(result)
                    continue
                offsets = replacement_offsets[replacement_index % 2]
                second_round_lists[profile_index].append(
                    _joint(
                        result.timeline,
                        offsets=offsets,
                        score=result.total_score,
                        variant=f"retry-cache-replacement-{replacement_index}",
                    )
                )
                replacement_index += 1
        second_round = tuple(tuple(pool) for pool in second_round_lists)
        selector_pools = (*first_round, *second_round)
        selector_call_count = 0
        review_call_count = 0

        def selector(*_args: object, **_kwargs: object):
            nonlocal selector_call_count
            result = selector_pools[selector_call_count]
            selector_call_count += 1
            return result

        def controlled_review(plan, *, available_source_count=None):  # type: ignore[no-untyped-def]
            nonlocal review_call_count
            reviewed = _synthetic_review(
                plan,
                available_source_count=available_source_count,
            )
            review_call_count += 1
            if review_call_count <= 4 or review_call_count > 8:
                return reviewed
            boundaries = tuple(
                replace(boundary, exceeded_metrics=("pose",))
                if boundary.source_switch
                else boundary
                for boundary in reviewed.boundaries
            )
            return replace(
                reviewed,
                boundaries=boundaries,
                hard_failures=("synthetic_retryable_review_failure",),
            )

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter.review_timeline",
                side_effect=controlled_review,
            ),
        ):
            selected = _production_select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertTrue(all(item.review.passed for item in selected.candidate_records))
        self.assertEqual(selector_call_count, 6)
        self.assertEqual(provider.selection_materialization_calls, 12)
        self.assertEqual(review_call_count, 12)
        audit = selected.selection_retry_audit
        self.assertEqual(audit["proof_materialization_call_count"], 12)
        self.assertEqual(audit["review_cache_hit_count"], 4)
        self.assertEqual(audit["total_unique_candidate_attempt_count"], 12)
        self.assertEqual(audit["discarded_candidate_count"], 4)
        self.assertEqual(audit["total_round_count"], 2)
        self.assertEqual(audit["retry_round_count"], 1)

    def test_boundary_option_evidence_is_shared_across_all_profiles_with_progress(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        provider = CountingStubDecodedProvider()
        progress: list[JointTimelineProgress] = []
        pools = _profile_pools(candidate_set)
        selector_call_count = 0
        shared_bridge_cache_ids: list[int] = []
        option_set_ids: list[int] = []

        def selector(
            lattice: tuple[tuple[TimelineCandidate, ...], ...],
            **kwargs: object,
        ) -> tuple[JointTimelineResult, ...]:
            nonlocal selector_call_count
            option_provider = kwargs["option_provider"]
            core_progress = kwargs["progress_callback"]
            assert callable(option_provider)
            assert callable(core_progress)
            shared_bridge_cache_ids.append(
                id(kwargs["shared_bridge_evidence_cache"])
            )
            core_progress(
                JointTimelineProgress(
                    phase="selection_start",
                    completed_boundary_layers=0,
                    total_boundary_layers=2,
                    option_key_count=0,
                    bridge_provider_call_count=0,
                )
            )
            key = BoundaryMeasurementKey(0, lattice[0][0], lattice[1][0])
            measured = option_provider(key)
            self.assertIsInstance(measured, BoundaryOptionSet)
            option_set_ids.append(id(measured))
            for boundary_index in range(2):
                core_progress(
                    JointTimelineProgress(
                        phase="boundary_layer_complete",
                        completed_boundary_layers=boundary_index + 1,
                        total_boundary_layers=2,
                        option_key_count=1,
                        bridge_provider_call_count=boundary_index * 4,
                    )
                )
            core_progress(
                JointTimelineProgress(
                    phase="selection_complete",
                    completed_boundary_layers=2,
                    total_boundary_layers=2,
                    option_key_count=1,
                    bridge_provider_call_count=5,
                )
            )
            result = pools[selector_call_count]
            selector_call_count += 1
            return result

        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=selector,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
                progress_callback=progress.append,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(len(set(shared_bridge_cache_ids)), 1)
        self.assertEqual(len(set(option_set_ids)), 1)
        self.assertEqual(list(provider.option_calls.values()), [1])
        self.assertEqual(
            [item.cadence_profile for item in progress if item.phase == "profile_start"],
            ["balanced", "reference", "dense_safe"],
        )
        self.assertEqual(
            [item.phase for item in progress if not item.phase.startswith("proof_")],
            [
                "profile_start",
                "boundary_layer_complete",
                "boundary_layer_complete",
                "profile_complete",
            ]
            * 3,
        )
        self.assertEqual(
            [item.completed_boundary_layers for item in progress
             if not item.phase.startswith("proof_")],
            [0, 1, 2, 2] * 3,
        )
        final_progress = [item for item in progress if item.phase == "profile_complete"][-1]
        self.assertEqual(final_progress.total_profiles, 3)
        self.assertEqual(final_progress.shared_option_cache_miss_count, 1)
        self.assertEqual(final_progress.shared_option_cache_hit_count, 2)
        self.assertEqual(final_progress.shared_option_cache_entry_count, 1)
        self.assertEqual(final_progress.option_key_count, 1)
        proof_progress = [item for item in progress if item.phase.startswith("proof_")]
        self.assertEqual(
            [item.phase for item in proof_progress],
            ["proof_start", "proof_complete", "proof_review_complete"] * 8,
        )
        self.assertEqual(
            [item.proof_materialization_call_count for item in proof_progress
             if item.phase == "proof_start"], list(range(1, 9)),
        )
        self.assertEqual(proof_progress[-1].retained_candidate_count, 8)
        self.assertEqual(proof_progress[-1].required_candidate_count, 8)
        self.assertEqual(proof_progress[-1].completed_proof_boundary_count, 2)
        self.assertEqual(proof_progress[-1].completed_proof_bridge_count, 1)
        self.assertEqual(
            (
                selected.shared_option_evidence_cache_audit["entry_count"],
                selected.shared_option_evidence_cache_audit["hit_count"],
                selected.shared_option_evidence_cache_audit["miss_count"],
            ),
            (1, 2, 1),
        )
        self.assertFalse(
            selected.shared_option_evidence_cache_audit[
                "raw_visual_evidence_persisted"
            ]
        )
        self.assertEqual(
            selected.shared_option_evidence_cache_audit["evidence_scope"],
            BoundaryOptionEvidenceScope.SELECTION_SCALAR.value,
        )
        self.assertEqual(
            selected.shared_bridge_evidence_cache_audit[
                "maximum_entry_count"
            ],
            2 * len(JOINT_BOUNDARY_FRAME_OFFSETS) ** 2,
        )
        self.assertEqual(
            selected.shared_bridge_evidence_cache_audit[
                "bridge_disposition_batch_call_count"
            ],
            0,
        )
        self.assertEqual(
            selected.shared_bridge_evidence_cache_audit[
                "bridge_disposition_batch_request_count"
            ],
            0,
        )

    def test_progress_callback_failure_aborts_before_partial_selection(self) -> None:
        provider = CountingStubDecodedProvider()

        def fail_closed(_progress: JointTimelineProgress) -> None:
            raise RuntimeError("synthetic adapter progress sink failure")

        with self.assertRaisesRegex(
            RuntimeError,
            "synthetic adapter progress sink failure",
        ):
            select_joint_visual_candidates(
                _candidate_set(),
                provider=provider,
                fps=10.0,
                progress_callback=fail_closed,
            )
        self.assertEqual(provider.option_calls, {})

        callback_failure = NoCompleteTimelinePath(
            "synthetic no-path progress sink failure",
            reason_code="synthetic_progress_sink_failure",
        )

        def fail_during_search(progress: JointTimelineProgress) -> None:
            if progress.phase == "boundary_layer_complete":
                raise callback_failure

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_visual_candidates(
                _candidate_set(),
                provider=CountingStubDecodedProvider(),
                fps=10.0,
                progress_callback=fail_during_search,
            )
        self.assertIs(caught.exception, callback_failure)

    def test_proof_observer_is_throttled_and_reports_exact_stage_ends(self) -> None:
        progress: list[JointTimelineProgress] = []
        observer = _SelectionProofProgressObserver(progress.append, 119, 118)
        with patch("crosscut.joint_boundary_provider.monotonic", return_value=0.0):
            for index in range(119):
                observer.completed(boundary_index=index)
            for index in range(1, 119):
                observer.completed(bridge_index=index)
            observer.completed(bridge_index=118)
        self.assertEqual(len(progress), 4)
        self.assertEqual(
            [(item.completed_proof_boundary_count, item.completed_proof_bridge_count)
             for item in progress], [(1, 0), (119, 0), (119, 1), (119, 118)],
        )
        json.dumps([item.to_dict() for item in progress], allow_nan=False)

    def test_proof_observation_preserves_real_proof_and_provider_counters(self) -> None:
        from . import test_joint_boundary_provider as fixtures

        descriptor = fixtures._descriptor(0.5)
        def descriptor_provider(_source: int, _seconds: float) -> FrameVisualDescriptor:
            return descriptor
        providers = [fixtures._provider(
            descriptor_provider, total_segment_count=3,
        )]
        with patch.object(fixtures, "DecodedBoundaryOptionProvider",
                          ProgressDecodedBoundaryOptionProvider):
            providers.append(fixtures._provider(
                descriptor_provider, total_segment_count=3,
            ))
        results = []
        progress: list[JointTimelineProgress] = []
        for index, provider in enumerate(providers):
            middle, entries, exits = fixtures._bridge_fixture(provider)
            entry = next(item for item in entries if item.frame_offset == 0)
            exit_ = next(item for item in exits if item.frame_offset == 0)
            bridge = provider.bridge_provider(middle, entry, exit_, fixtures.FPS)
            kwargs = {"progress_callback": progress.append} if index else {}
            results.append(provider.selection_materialization_proof(
                (entry, exit_), (bridge,), **kwargs,
            ))
        self.assertEqual(results[0], results[1])
        self.assertEqual(providers[0].to_dict(), providers[1].to_dict())
        self.assertEqual(
            [item.phase for item in progress],
            ["proof_boundary_complete", "proof_boundary_complete", "proof_bridge_complete"],
        )
        self.assertIsNone(providers[1]._selection_progress_observer)
        for provider in providers:
            provider.close()

    def test_proof_observer_failure_rolls_back_and_next_attempt_is_clean(self) -> None:
        from . import test_joint_boundary_provider as fixtures

        with patch.object(fixtures, "DecodedBoundaryOptionProvider",
                          ProgressDecodedBoundaryOptionProvider):
            provider = fixtures._provider(
                lambda _source, _seconds: fixtures._descriptor(0.5),
            )
        option = provider(fixtures._key()).audit_for(0).authorized_option
        self.assertIsNotNone(option)
        baseline = provider.selection_materialization_proof((option,))
        before = provider.to_dict()
        failure = RuntimeError("synthetic proof observer failure")

        def reject(_progress: JointTimelineProgress) -> None:
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            provider.selection_materialization_proof((option,), progress_callback=reject)
        self.assertIs(caught.exception, failure)
        self.assertIsNone(provider._selection_progress_observer)
        self.assertIsNone(provider._active_transaction_journal)
        after = provider.to_dict()
        for field in ("descriptor_decode_count", "bridge_measurement_count",
                      "selection_materialization_count", "last_selection"):
            if field in before:
                self.assertEqual(before[field], after[field])
        self.assertEqual(after["selection_failure_count"], before["selection_failure_count"] + 1)
        progress: list[JointTimelineProgress] = []
        recovered = provider.selection_materialization_proof(
            (option,), progress_callback=progress.append,
        )
        self.assertEqual(recovered, baseline)
        self.assertEqual(len(progress), 1)
        provider.close()

    def test_proof_failure_progress_preserves_original_error_without_private_text(self) -> None:
        candidate_set = _candidate_set()
        pools = iter(_profile_pools(candidate_set))
        progress: list[JointTimelineProgress] = []
        provider = StubMaterializingProvider()

        def selector(_lattice: object, **kwargs: object):  # type: ignore[no-untyped-def]
            callback = kwargs["progress_callback"]
            assert callable(callback)
            callback(JointTimelineProgress(
                phase="selection_complete", completed_boundary_layers=2,
                total_boundary_layers=2, option_key_count=0,
                bridge_provider_call_count=0,
            ))
            return next(pools)

        failure = ValueError("private-media-path/source.mp4 failed")
        with patch("crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates",
                   side_effect=selector), patch.object(provider, "selection_materialization_proof",
                                                       side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                select_joint_visual_candidates(candidate_set, provider=provider,
                                               fps=10.0, progress_callback=progress.append)
        self.assertIs(caught.exception, failure)
        self.assertEqual(progress[-1].phase, "proof_failed")
        self.assertEqual(progress[-1].proof_failure_code, "ValueError")
        self.assertEqual(progress[-1].proof_candidate_ordinal, 1)
        self.assertEqual(progress[-1].proof_materialization_call_count, 1)
        self.assertNotIn("private-media-path", json.dumps([item.to_dict() for item in progress]))

    def test_proof_failure_observer_error_does_not_mask_provider_error(self) -> None:
        candidate_set = _candidate_set()
        pools = iter(_profile_pools(candidate_set))
        provider = StubMaterializingProvider()

        def selector(_lattice: object, **kwargs: object):  # type: ignore[no-untyped-def]
            callback = kwargs["progress_callback"]
            assert callable(callback)
            callback(JointTimelineProgress(
                phase="selection_complete", completed_boundary_layers=2,
                total_boundary_layers=2, option_key_count=0,
                bridge_provider_call_count=0,
            ))
            return next(pools)

        def observer(progress: JointTimelineProgress) -> None:
            if progress.phase == "proof_failed":
                raise BrokenPipeError("private-observer-destination")

        failure = ValueError("original provider proof failure")
        with patch(
            "crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates",
            side_effect=selector,
        ), patch.object(provider, "selection_materialization_proof", side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                select_joint_visual_candidates(
                    candidate_set, provider=provider, fps=10.0,
                    progress_callback=observer,
                )
        self.assertIs(caught.exception, failure)
        self.assertEqual(
            failure.__notes__,
            ["proof failure progress callback failed: BrokenPipeError"],
        )

    def test_offsets_shift_clocks_and_highest_proof_variant_wins(self) -> None:
        candidate_set = _candidate_set()
        pools = _profile_pools(candidate_set)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(
            selected.to_dict()["proof_variant_supplement_count"],
            0,
        )
        profile_names = tuple(
            item.cadence_profile for item in selected.selections
        )
        self.assertEqual(
            {
                profile: profile_names.count(profile)
                for profile in set(profile_names)
            },
            {"balanced": 2, "reference": 3, "dense_safe": 3},
        )
        path_zero = next(
            item
            for item in selected.selections
            if tuple(source for _index, source, _shot in item.source_shot_path)
            == ("0", "0", "0")
        )
        self.assertEqual(path_zero.cadence_profile, "reference")
        self.assertIn("proof:high", path_zero.proof_variant_signature[0])
        spans = tuple(
            (candidate.start_seconds, candidate.end_seconds)
            for candidate in path_zero.shifted_timeline_result.candidates
        )
        self.assertEqual(spans, ((5.0, 6.0), (6.0, 7.0), (7.0, 8.0)))
        self.assertNotIn(str(RAW_SENTINEL), repr(selected.to_dict()))

    def test_unique_path_floor_fails_closed_after_proof_dedup(self) -> None:
        candidate_set = _candidate_set()
        duplicate = _profile_pools(candidate_set)[0][0]
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=((duplicate,), (duplicate,), (duplicate,)),
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                select_joint_visual_candidates(
                    candidate_set,
                    provider=StubDecodedProvider(),
                    fps=10.0,
                )
        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_unique_path_floor_not_met",
        )
        self.assertEqual(
            caught.exception.rejection_counts["unique_source_shot_path_count"],
            1,
        )

    def test_six_unique_paths_retain_two_distinct_render_variants(self) -> None:
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=6,
            alternate_paths=(0, 1),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(
            len({item.source_shot_path for item in selected.selections}),
            6,
        )
        self.assertEqual(
            selected.to_dict()["proof_variant_supplement_count"],
            2,
        )
        self.assertEqual(
            len({item.privacy_safe_signature for item in selected.selections}),
            8,
        )
        self.assertEqual(
            len(
                {
                    item.source_shot_path
                    for item in selected.selections[:6]
                }
            ),
            6,
        )
        supplement_paths = tuple(
            item.source_shot_path for item in selected.selections[6:]
        )
        self.assertEqual(len(set(supplement_paths)), 2)
        for supplement in selected.selections[6:]:
            primary = next(
                item
                for item in selected.selections[:6]
                if item.source_shot_path == supplement.source_shot_path
            )
            self.assertNotEqual(
                tuple(
                    option.frame_offset
                    for option in primary.joint_result.boundary_options
                ),
                tuple(
                    option.frame_offset
                    for option in supplement.joint_result.boundary_options
                ),
            )
            self.assertNotEqual(
                tuple(
                    (candidate.start_seconds, candidate.end_seconds)
                    for candidate in primary.shifted_timeline_result.candidates
                ),
                tuple(
                    (candidate.start_seconds, candidate.end_seconds)
                    for candidate in supplement.shifted_timeline_result.candidates
                ),
            )
            primary_record = selected.candidate_records[
                selected.selections.index(primary)
            ]
            supplement_record = selected.candidate_records[
                selected.selections.index(supplement)
            ]
            self.assertEqual(
                primary_record.path_signature,
                supplement_record.path_signature,
            )
            self.assertNotEqual(
                primary.privacy_safe_signature,
                supplement.privacy_safe_signature,
            )

        materialized = materialize_joint_visual_candidates(
            selected,
            provider=provider,
            fps=10.0,
            maximum_candidates=8,
        )
        self.assertEqual(len(materialized), 8)
        self.assertEqual(len({item.path_signature for item in materialized}), 8)
        by_raw_path: dict[str, list[object]] = {}
        for item in materialized:
            by_raw_path.setdefault(
                item.candidate_record.path_signature,
                [],
            ).append(item)
        repeated = tuple(items for items in by_raw_path.values() if len(items) == 2)
        self.assertEqual(len(repeated), 2)
        for primary_and_supplement in repeated:
            self.assertNotEqual(
                primary_and_supplement[0].boundary_proof_bundle.frame_offsets,
                primary_and_supplement[1].boundary_proof_bundle.frame_offsets,
            )
            self.assertNotEqual(
                primary_and_supplement[0].candidate_history_entry[
                    "joint_path_signature"
                ],
                primary_and_supplement[1].candidate_history_entry[
                    "joint_path_signature"
                ],
            )

    def test_seven_unique_paths_retain_one_distinct_render_variant(self) -> None:
        candidate_set = _candidate_set()
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=7,
            alternate_paths=(0,),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(
            len({item.source_shot_path for item in selected.selections}),
            7,
        )
        self.assertEqual(
            selected.to_dict()["proof_variant_supplement_count"],
            1,
        )

    def test_five_unique_paths_retain_three_distinct_render_variants(self) -> None:
        candidate_set = _candidate_set()
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=5,
            alternate_paths=(0, 1, 2),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(
            selected.unique_source_shot_path_count,
            5,
        )
        self.assertEqual(
            selected.proof_variant_supplement_count,
            3,
        )

    def test_four_unique_paths_are_the_mathematical_render_cohort_floor(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=4,
            alternate_paths=(0, 1, 2, 3),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
            )

        self.assertEqual(len(selected.selections), 8)
        self.assertEqual(selected.unique_source_shot_path_count, 4)
        self.assertEqual(selected.proof_variant_supplement_count, 4)

    def test_three_unique_paths_fail_even_with_render_variants(self) -> None:
        candidate_set = _candidate_set()
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=3,
            alternate_paths=(0, 1, 2),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                select_joint_visual_candidates(
                    candidate_set,
                    provider=StubDecodedProvider(),
                    fps=10.0,
                )

        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_unique_path_floor_not_met",
        )
        self.assertEqual(
            caught.exception.rejection_counts["unique_source_shot_path_count"],
            3,
        )

    def test_six_unique_paths_fail_without_two_eligible_supplements(self) -> None:
        candidate_set = _candidate_set()
        pools = _limited_profile_pools(
            candidate_set,
            unique_path_count=6,
            alternate_paths=(0, 0),
        )
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            with self.assertRaises(NoCompleteTimelinePath) as caught:
                select_joint_visual_candidates(
                    candidate_set,
                    provider=StubDecodedProvider(),
                    fps=10.0,
                )

        self.assertEqual(
            caught.exception.reason_code,
            "joint_visual_candidate_cohort_floor_not_met",
        )
        self.assertEqual(
            caught.exception.rejection_counts["distinct_render_candidate_count"],
            7,
        )
        self.assertEqual(
            caught.exception.rejection_counts["proof_variant_supplement_count"],
            1,
        )

    def test_existing_profile_controls_and_hard_no_good_are_forwarded_exactly(
        self,
    ) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        pools = _profile_pools(candidate_set)
        hard_no_goods = TimelineNoGoodSet(
            (
                TimelineNoGoodEdgeKey(
                    boundary_index=0,
                    outgoing_source_id="1",
                    outgoing_shot_id="shot:0:1",
                    incoming_source_id="1",
                    incoming_shot_id="shot:1:1",
                ),
            )
        )
        control_calls: list[dict[str, object]] = []
        selector_calls: list[dict[str, object]] = []

        def controls(*_args: object, **kwargs: object):
            profile = str(kwargs["cadence_profile"])
            control_calls.append(dict(kwargs))
            return (
                {(0, "1"): -float(len(control_calls))},
                {1: float(len(control_calls))},
                {"cadence_profile": profile},
            )

        def selector(*_args: object, **kwargs: object):
            selector_calls.append(dict(kwargs))
            return pools[len(selector_calls) - 1]

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "_role_conditioned_visual_controls",
                side_effect=controls,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
        ):
            select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
                hard_forbidden_boundary_edges=hard_no_goods,
            )

        self.assertEqual(
            [call["cadence_profile"] for call in control_calls],
            ["balanced", "reference", "dense_safe"],
        )
        self.assertEqual(len(selector_calls), 3)
        for index, call in enumerate(selector_calls, start=1):
            self.assertIs(call["forbidden_boundary_edges"], hard_no_goods)
            self.assertEqual(call["preferred_source_ids"], ("0", "1", "0"))
            self.assertEqual(call["emission_adjustments"], {(0, "1"): -index})
            self.assertEqual(call["source_switch_bonus_by_segment"], {1: index})
            self.assertEqual(call["maximum_results"], 8)

    def test_decoded_feedback_uses_shadow_role_controls_and_original_joint_lattice(
        self,
    ) -> None:
        candidate_set = _candidate_set(role_conditioned=True)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        lattice = workspace.candidates_by_segment
        target = lattice[0][0]
        feedback = JointDecodedMontageFeedback(
            observations=tuple(
                JointDecodedMontageSegmentObservation.from_candidate(
                    target,
                    render_variant_signature=signature * 64,
                    sampled_proxy_shot_classes=(
                        "wide",
                        "unknown",
                        "medium",
                    ),
                )
                for signature in ("a", "b")
            )
        )
        pools = _profile_pools(candidate_set)
        control_lattices: list[
            tuple[tuple[TimelineCandidate, ...], ...]
        ] = []
        control_metadata: list[object] = []
        selector_lattices: list[object] = []
        selector_calls: list[dict[str, object]] = []

        def controls(
            candidates: tuple[tuple[TimelineCandidate, ...], ...],
            **kwargs: object,
        ):
            control_lattices.append(candidates)
            control_metadata.append(kwargs["subject_metadata_by_candidate"])
            return (
                {(0, "0"): -4.0},
                {1: 3.0},
                {"cadence_profile": kwargs["cadence_profile"]},
            )

        def selector(*args: object, **kwargs: object):
            selector_lattices.append(args[0])
            selector_calls.append(dict(kwargs))
            return pools[len(selector_calls) - 1]

        with (
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "_role_conditioned_visual_controls",
                side_effect=controls,
            ),
            patch(
                "crosscut.joint_visual_candidate_adapter."
                "select_joint_timeline_candidates",
                side_effect=selector,
            ),
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
                decoded_montage_feedback=feedback,
            )

        self.assertEqual(len(control_lattices), 3)
        self.assertEqual(len(selector_lattices), 3)
        for role_lattice, metadata in zip(
            control_lattices,
            control_metadata,
            strict=True,
        ):
            self.assertIs(role_lattice[0][0].entry_shot_type, ShotType.WIDE)
            self.assertIs(role_lattice[0][0].shot_type, ShotType.MEDIUM)
            self.assertEqual(
                metadata[target.candidate_id].shot_scale_bases,
                ("legacy", "unknown_no_subject_geometry", "legacy"),
            )
        for joint_lattice, call in zip(
            selector_lattices,
            selector_calls,
            strict=True,
        ):
            self.assertIs(joint_lattice, lattice)
            self.assertIs(joint_lattice[0][0], target)
            self.assertIsNone(joint_lattice[0][0].entry_shot_type)
            self.assertEqual(call["emission_adjustments"], {(0, "0"): -4.0})
            self.assertEqual(call["source_switch_bonus_by_segment"], {1: 3.0})
        for role_audit in selected.role_control_audits:
            decoded_audit = role_audit["decoded_montage_feedback"]
            self.assertEqual(decoded_audit["promoted_wide_sample_count"], 1)
            self.assertEqual(decoded_audit["promoted_unknown_sample_count"], 1)
            self.assertEqual(decoded_audit["hard_no_good_count"], 0)
            self.assertFalse(decoded_audit["thresholds_lowered"])
            self.assertTrue(role_audit["emission_adjustments_non_positive"])
        retry_audit = selected.selection_retry_audit[
            "decoded_montage_feedback"
        ]
        self.assertEqual(retry_audit["promoted_candidate_clock_count"], 1)
        self.assertEqual(retry_audit["hard_no_good_count"], 0)

    def test_decoded_feedback_requires_the_existing_montage_policy(self) -> None:
        candidate_set = _candidate_set(role_conditioned=False)
        workspace = candidate_set.optimizer_workspace
        assert workspace is not None
        target = workspace.candidates_by_segment[0][0]
        feedback = JointDecodedMontageFeedback(
            observations=(
                JointDecodedMontageSegmentObservation.from_candidate(
                    target,
                    render_variant_signature="a" * 64,
                    sampled_proxy_shot_classes=(
                        "wide",
                        "medium",
                        "medium",
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "professional montage policy"):
            select_joint_visual_candidates(
                candidate_set,
                provider=StubDecodedProvider(),
                fps=10.0,
                decoded_montage_feedback=feedback,
            )

    def test_materialization_binds_all_outer_and_middle_tracks_without_decode(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        pools = _profile_pools(candidate_set)
        provider = StubMaterializingProvider(fps=10.0)
        with patch(
            "crosscut.joint_visual_candidate_adapter."
            "select_joint_timeline_candidates",
            side_effect=pools,
        ):
            selected = select_joint_visual_candidates(
                candidate_set,
                provider=provider,
                fps=10.0,
            )
        self.assertEqual(provider.selection_materialization_calls, 8)
        with self.assertRaisesRegex(
            ValueError,
            "another boundary proof variant",
        ):
            replace(
                selected,
                boundary_proof_bundles=tuple(
                    reversed(selected.boundary_proof_bundles)
                ),
            )
        materialized = materialize_joint_visual_candidates(
            selected,
            provider=provider,
            fps=10.0,
            maximum_candidates=8,
        )
        self.assertEqual(provider.selection_materialization_calls, 8)

        self.assertEqual(len(materialized), 8)
        self.assertEqual(
            [item.review_rank for item in materialized],
            list(range(8)),
        )
        self.assertEqual(provider.validated_segment_counts, [3, 3])
        for item in materialized:
            self.assertEqual(len(item.segments), 3)
            self.assertIs(
                item.candidate_record,
                selected.candidate_records[item.candidate_index],
            )
            self.assertEqual(
                item.cadence_profile,
                item.candidate_record.cadence_profile,
            )
            self.assertEqual(
                item.visual_plan.selected_candidate_index,
                item.candidate_index,
            )
            self.assertEqual(
                [
                    path.candidate.segment_index
                    for path in item.boundary_proof_bundle.segment_path_proofs
                ],
                [0, 1, 2],
            )
            motion = item.editorial_motion_plan
            self.assertEqual(
                motion.analysis_basis,
                JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
            )
            source_path = tuple(
                choice.candidate.source_id
                for choice in item.candidate_record.timeline_result.choices
            )
            switched_boundaries = tuple(
                index
                for index, (outgoing, incoming) in enumerate(
                    zip(source_path, source_path[1:])
                )
                if outgoing != incoming
            )
            native_boundaries = tuple(
                index
                for index in range(len(source_path) - 1)
                if index not in switched_boundaries
            )
            self.assertEqual(motion.applied_count, 3)
            self.assertEqual(
                motion.boundary_match_count,
                len(switched_boundaries),
            )
            self.assertEqual(motion.affine_boundary_match_count, 0)
            transform_evidence = _validated_transforms(
                plan=item.to_render_plan(output=Path("fixture.mp4")),
                editorial_motion=motion,
                eligible_to_request=tuple(range(len(item.candidate_set.sources))),
                dynamic_crop_version=None,
                dynamic_crop_motion_v3=None,
            )
            self.assertEqual(
                motion.zoom_event_count,
                len(transform_evidence.zoom_event_times_seconds),
            )
            self.assertEqual(
                motion.pan_event_count,
                len(transform_evidence.pan_event_times_seconds),
            )
            self.assertEqual(
                motion.roll_event_count,
                len(transform_evidence.roll_event_times_seconds),
            )
            self.assertEqual(
                motion.accepted_boundary_indices,
                switched_boundaries,
            )
            self.assertEqual(
                motion.natural_match_boundary_indices,
                native_boundaries,
            )
            self.assertEqual(motion.blocked_dissolve_segment_count, 0)
            for segment, path, motion_segment in zip(
                item.segments,
                item.boundary_proof_bundle.segment_path_proofs,
                motion.segments,
                strict=True,
            ):
                self.assertIs(
                    segment.editorial_transform,
                    path.editorial_transform_track,
                )
                self.assertIs(
                    motion_segment.track,
                    path.editorial_transform_track,
                )
                self.assertEqual(
                    motion_segment.segment_index,
                    path.candidate.segment_index,
                )
                self.assertEqual(motion_segment.source_index, path.source_index)
                self.assertAlmostEqual(
                    segment.source_start,
                    path.samples[0].source_seconds,
                )
                self.assertAlmostEqual(
                    segment.source_end,
                    path.samples[-1].source_seconds,
                )
                expected_center_count = int(
                    math.ceil(max(0.0, path.duration_frames - 0.5 - 1e-9))
                )
                expected_centers = tuple(
                    path.start_master_seconds
                    + (frame_index + 0.5) / provider.fps
                    for frame_index in range(expected_center_count)
                    if path.start_master_seconds
                    + (frame_index + 0.5) / provider.fps
                    < path.end_master_seconds - 1e-9
                )
                expected_sample_clocks = (
                    path.start_master_seconds,
                    *expected_centers,
                    path.end_master_seconds,
                )
                self.assertEqual(
                    len(path.samples),
                    path.expected_sample_count,
                )
                self.assertEqual(
                    len(path.samples),
                    len(expected_sample_clocks),
                )
                source_offset = provider.audio_offsets[path.source_index]
                for sample, expected_master in zip(
                    path.samples,
                    expected_sample_clocks,
                    strict=True,
                ):
                    self.assertAlmostEqual(sample.master_seconds, expected_master)
                    self.assertAlmostEqual(
                        sample.source_seconds,
                        expected_master + source_offset,
                    )
            self.assertEqual(
                [segment.master_start for segment in item.segments],
                [
                    0.0,
                    item.segments[0].duration,
                    item.segments[0].duration + item.segments[1].duration,
                ],
            )
            history = item.candidate_history_entry
            self.assertFalse(
                history["boundary_proof_bundle"][
                    "raw_visual_descriptors_persisted"
                ]
            )
            self.assertEqual(
                history["editorial_motion_plan"]["analysis_basis"],
                JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
            )
            self.assertEqual(
                history["editorial_motion_plan"]["applied_segment_count"],
                3,
            )
            self.assertNotIn(str(RAW_SENTINEL), repr(history))
            self.assertTrue(
                item.privacy_safe_signature.startswith("joint-visual-path:")
            )

    def test_full_120_segment_selected_audits_bypass_transient_lru(self) -> None:
        provider = StubMaterializingProvider(fps=10.0)
        provider.total_segment_count = 120
        candidates = tuple(
            _candidate(index, str(index % 2)) for index in range(120)
        )
        options: list[AuthorizedBoundaryOption] = []
        for boundary_index, (outgoing, incoming) in enumerate(
            zip(candidates, candidates[1:])
        ):
            _option_set_value, selected = _option_set(
                BoundaryMeasurementKey(
                    boundary_index,
                    outgoing,
                    incoming,
                ),
                selected_offset=0,
                variant="full-capacity",
            )
            options.append(selected)
        bridges = tuple(
            SegmentBridgeProof(
                candidate=candidates[index],
                entry_option=options[index - 1],
                exit_option=options[index],
                duration_frames=10.0,
                proof_identity=f"full-capacity-bridge:{index}",
                evidence_complete=True,
                crop_compatible=True,
                track_compatible=True,
            )
            for index in range(1, 119)
        )

        proof = provider.selection_materialization_proof(
            tuple(options),
            bridges,
        )
        bundle = JointVisualBoundaryProofBundle(
            selection_proof=proof,
            frame_offsets=tuple(option.frame_offset for option in options),
            bridge_audits=proof.selected_bridge_audits,
        )

        self.assertEqual(len(proof.segment_paths), 120)
        self.assertEqual(len(proof.selected_bridge_audits), 118)
        self.assertEqual(len(bundle.bridge_audits), 118)
        self.assertEqual(
            tuple(item["proof_identity"] for item in bundle.bridge_audits),
            tuple(bridge.proof_identity for bridge in bridges),
        )

    def test_proof_bundle_rejects_tampered_selected_bridge_audit(self) -> None:
        provider = StubMaterializingProvider(fps=10.0)
        candidates = tuple(
            _candidate(index, str(index % 2)) for index in range(3)
        )
        options: list[AuthorizedBoundaryOption] = []
        for boundary_index, (outgoing, incoming) in enumerate(
            zip(candidates, candidates[1:])
        ):
            _option_set_value, selected = _option_set(
                BoundaryMeasurementKey(boundary_index, outgoing, incoming),
                selected_offset=0,
                variant="tamper",
            )
            options.append(selected)
        bridge = SegmentBridgeProof(
            candidate=candidates[1],
            entry_option=options[0],
            exit_option=options[1],
            duration_frames=10.0,
            proof_identity="tamper-bridge",
            evidence_complete=True,
            crop_compatible=True,
            track_compatible=True,
        )
        proof = provider.selection_materialization_proof(
            tuple(options),
            (bridge,),
        )
        tampered = dict(proof.selected_bridge_audits[0])
        tampered["evidence_complete"] = False

        with self.assertRaisesRegex(ValueError, "atomic selection proof"):
            JointVisualBoundaryProofBundle(
                selection_proof=proof,
                frame_offsets=tuple(
                    option.frame_offset for option in options
                ),
                bridge_audits=(tampered,),
            )

    def test_shift_rematerializes_scene_cuts_across_moved_boundary(self) -> None:
        candidate_set = _candidate_set(scene_cut_times=((6.35,), ()))
        assert candidate_set.optimizer_workspace is not None
        lattice = candidate_set.optimizer_workspace.candidates_by_segment
        timeline = _timeline(lattice, ("0", "0", "0"), score=10.0)
        choices = list(timeline.choices)
        choices[1] = replace(
            choices[1],
            candidate=replace(
                choices[1].candidate,
                internal_scene_cut_master_seconds=(6.1,),
            ),
        )
        result = _joint(
            replace(timeline, choices=tuple(choices)),
            offsets=(2, 0),
            score=10.0,
            variant="scene-cut-shift",
            allow_same_source_offsets=True,
        )

        shifted = _shift_joint_timeline_result(
            result,
            fps=10.0,
            candidate_set=candidate_set,
        )

        self.assertEqual(
            shifted.candidates[0].internal_scene_cut_master_seconds,
            (6.1,),
        )
        self.assertEqual(
            shifted.candidates[1].internal_scene_cut_master_seconds,
            (),
        )
        self.assertAlmostEqual(shifted.candidates[0].end_seconds, 6.2)
        self.assertAlmostEqual(shifted.candidates[1].start_seconds, 6.2)

    def test_shift_rejects_sub_two_frame_segment(self) -> None:
        candidate_set = _candidate_set()
        assert candidate_set.optimizer_workspace is not None
        lattice = candidate_set.optimizer_workspace.candidates_by_segment
        result = _joint(
            _timeline(lattice, ("0", "0", "0"), score=10.0),
            offsets=(-8, 0),
            score=10.0,
            variant="short",
            allow_same_source_offsets=True,
        )
        with self.assertRaisesRegex(ValueError, "shorter than two frames"):
            _shift_joint_timeline_result(
                result,
                fps=9.0,
                candidate_set=candidate_set,
            )


if __name__ == "__main__":
    unittest.main()
