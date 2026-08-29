from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.joint_boundary_provider import (
    BoundaryMaterializationProof,
    JointSelectionMaterializationProof,
    SegmentTransformPathMaterializationProof,
    SegmentTransformPathSample,
)
from crosscut.joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOffsetAudit,
    BoundaryOffsetDisposition,
    BoundaryOptionSet,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    JointTimelineResult,
    SegmentBridgeProof,
)
from crosscut.joint_visual_candidate_adapter import (
    JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
    JointVisualBoundaryProofBundle,
    _shift_joint_timeline_result,
    materialize_joint_visual_candidates,
    select_joint_visual_candidates,
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
    NoCompleteTimelinePath,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
)
from crosscut.timeline_review import review_timeline
from crosscut.visual_score import (
    FrameVisualDescriptor,
    ProfessionalMontageSelectionPolicy,
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelineOptimizerWorkspace,
    _review_ranked_candidate_indices,
    _visual_optimizer_input_identity,
)


RAW_SENTINEL = 0.987654321234567


class StubDecodedProvider(DecodedBoundaryOptionProvider):
    """Type-correct provider whose calls are replaced by the selector stub."""

    def __init__(self) -> None:
        self.output_fps = 10.0
        self.total_segment_count = 3
        self.sources = (Path("source-0.mp4"), Path("source-1.mp4"))
        self.audio_offsets = (0.25, 0.50)
        self.source_timewarps = (None, None)
        self.source_ids = ("0", "1")
        self.validated_segment_counts: list[int] = []

    def validate_segment_count(self, segment_count: int) -> None:
        self.validated_segment_counts.append(segment_count)
        if segment_count != self.total_segment_count:
            raise ValueError("synthetic provider segment-count mismatch")

    def bridge_provider(self, *args: object) -> SegmentBridgeProof:
        raise AssertionError("selector stub must not invoke bridge_provider")


class StubMaterializingProvider(StubDecodedProvider):
    def __init__(self, *, fps: float = 10.0) -> None:
        super().__init__()
        self.fps = fps
        self.output_fps = fps
        self._bridge_audits: dict[str, dict[str, object]] = {}

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
        options = tuple(boundary_options)
        candidates = (options[0].measurement_key.outgoing,) + tuple(
            option.measurement_key.incoming for option in options
        )
        offsets = tuple(option.frame_offset for option in options)
        descriptor = self._descriptor()
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
                raw_evidence=object(),
                result_evidence=object(),
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
            samples = tuple(
                SegmentTransformPathSample(
                    master_seconds=master,
                    source_seconds=master + source_offset,
                    transform=(
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
                    ),
                    subject_basis="dominant_face",
                    subject_bbox_xywh=(0.3, 0.2, 0.2, 0.4),
                    subject_bbox_coverage=1.0,
                    face_similarity_to_anchor=1.0,
                    face_similarity_to_previous=1.0,
                    descriptor=None,
                    face_identity_embedding=(RAW_SENTINEL,),
                )
                for sample_index, master in enumerate(sample_times)
            )
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
            selected = AuthorizedBoundaryOption(
                measurement_key=key,
                frame_offset=offset,
                outgoing_transform=_transform(
                    identity + ":out",
                    key.outgoing,
                    boundary_index=key.boundary_index,
                ),
                incoming_transform=_transform(
                    identity + ":in",
                    key.incoming,
                    boundary_index=key.boundary_index,
                ),
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
) -> JointTimelineResult:
    option_sets: list[BoundaryOptionSet] = []
    options: list[AuthorizedBoundaryOption] = []
    for index, (outgoing, incoming) in enumerate(
        zip(timeline.candidates, timeline.candidates[1:])
    ):
        option_set, option = _option_set(
            BoundaryMeasurementKey(index, outgoing, incoming),
            selected_offset=offsets[index],
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


def _candidate_set(*, role_conditioned: bool = False) -> VisualTimelineCandidateSet:
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
        scene_cut_times=((), ()),
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
        scene_cut_times=((), ()),
        optimizer_workspace=workspace,
        optimizer_input_identity=workspace_identity,
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


class JointVisualCandidateAdapterTests(unittest.TestCase):
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
            [item.cadence_profile for item in selected.selections],
            [
                "balanced",
                "reference",
                "dense_safe",
                "balanced",
                "reference",
                "dense_safe",
                "reference",
                "dense_safe",
            ],
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
        self.assertEqual(spans, ((5.0, 6.2), (6.2, 6.8), (6.8, 8.0)))
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
        materialized = materialize_joint_visual_candidates(
            selected,
            provider=provider,
            fps=10.0,
            maximum_candidates=8,
        )

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
            self.assertEqual(motion.applied_count, 3)
            self.assertEqual(motion.boundary_match_count, 2)
            self.assertEqual(motion.affine_boundary_match_count, 0)
            self.assertEqual(motion.zoom_event_count, 2)
            self.assertEqual(motion.pan_event_count, 1)
            self.assertEqual(motion.roll_event_count, 1)
            self.assertEqual(motion.accepted_boundary_indices, (0, 1))
            self.assertEqual(motion.natural_match_boundary_indices, ())
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

    def test_shift_rejects_sub_two_frame_segment(self) -> None:
        candidate_set = _candidate_set()
        assert candidate_set.optimizer_workspace is not None
        lattice = candidate_set.optimizer_workspace.candidates_by_segment
        result = _joint(
            _timeline(lattice, ("0", "0", "0"), score=10.0),
            offsets=(-8, 0),
            score=10.0,
            variant="short",
        )
        with self.assertRaisesRegex(ValueError, "shorter than two frames"):
            _shift_joint_timeline_result(result, fps=9.0)


if __name__ == "__main__":
    unittest.main()
