from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from crosscut.joint_boundary_provider import (
    DecodedBoundaryOptionProvider,
    SegmentTransformPathSample,
)
from crosscut.joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOffsetDisposition,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    SegmentBridgeProof,
    SegmentBridgeDisposition,
    SegmentBridgeDispositionRequest,
    select_joint_timeline,
)
from crosscut.pose import FormationPersonDescriptor, POSE_BACKEND
from crosscut.timeline import ShotType, TimelineCandidate
from crosscut.timewarp import (
    TimeWarpAnchor,
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.visual_boundary_repair import BoundaryDescriptorCache
from crosscut.visual_score import FrameVisualDescriptor


FPS = 30.0
NOMINAL = 10.0


def _candidate(segment: int, source: str) -> TimelineCandidate:
    return TimelineCandidate(
        segment_index=segment,
        source_id=source,
        start_seconds=9.0 if segment == 0 else 10.0,
        end_seconds=10.0 if segment == 0 else 11.0,
        quality_score=0.8,
        sync_score=0.9,
        composition_score=0.8,
        boundary_score=0.7,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        candidate_id=f"candidate:{segment}:{source}",
        shot_id=f"shot:{segment}:{source}",
    )


def _key() -> BoundaryMeasurementKey:
    return BoundaryMeasurementKey(
        0,
        _candidate(0, "0"),
        _candidate(1, "1"),
    )


def _descriptor(
    center: float,
    *,
    shot_scale: float = 0.4,
    face_bbox: tuple[float, float, float, float] | None = None,
    pose_value: float | None = None,
) -> FrameVisualDescriptor:
    people = tuple(
        FormationPersonDescriptor(
            bbox_xywh=(x - 0.07, 0.2, 0.14, 0.62),
            centroid_xy=(x, 0.51),
            detection_confidence=0.95,
        )
        for x in (center - 0.12, center + 0.12)
    )
    pose_embedding = (
        (pose_value,) * 60 if pose_value is not None else ()
    )
    return FrameVisualDescriptor(
        quality_score=0.85,
        composition_score=0.82,
        shot_scale=shot_scale,
        shot_type=ShotType.MEDIUM,
        face_count=1 if face_bbox is not None else 0,
        structure_embedding=(center, 1.0 - center),
        primary_face_bbox=face_bbox,
        color_embedding=(0.2, 0.4),
        pose_embedding=pose_embedding,
        pose_backend=POSE_BACKEND if pose_embedding else "structure_hog_fallback",
        pose_confidence=0.95 if pose_embedding else 0.0,
        detected_person_count=2,
        person_layout_embedding=(center - 0.12, center + 0.12, 0.5),
        formation_people=people,
        silhouette_embedding=(center, 1.0 - center, 0.25),
        silhouette_bbox_xywh=(center - 0.22, 0.16, 0.44, 0.70),
        silhouette_centroid_xy=(center, 0.51),
        silhouette_confidence=0.95,
        silhouette_backend="opencv_hog_person_union_v1",
    )


def _provider(
    descriptor_provider,
    *,
    scene_cut_times=((), ()),
    audio_offsets=(0.0, 0.0),
    source_timewarps=None,
    source_ids=None,
    transformed_descriptor_provider=None,
    descriptor_cache=None,
    sources=(Path("source-a.mp4"), Path("source-b.mp4")),
    total_segment_count=2,
    output_fps=FPS,
    beat_period_seconds=None,
    beat_intensity=None,
    face_identity_provider=None,
    feature_cache_directory=None,
    feature_derived_cache_directory=None,
    require_persistent_feature_cache=False,
) -> DecodedBoundaryOptionProvider:
    return DecodedBoundaryOptionProvider(
        sources=sources,
        audio_offsets=audio_offsets,
        scene_cut_times=scene_cut_times,
        source_timewarps=source_timewarps,
        descriptor_cache=(
            descriptor_cache
            if descriptor_cache is not None
            else BoundaryDescriptorCache(maximum_entries=4096)
        ),
        output_fps=output_fps,
        total_segment_count=total_segment_count,
        beat_period_seconds=beat_period_seconds,
        beat_intensity=beat_intensity,
        descriptor_provider=descriptor_provider,
        face_identity_provider=face_identity_provider,
        transformed_descriptor_provider=transformed_descriptor_provider,
        source_ids=source_ids,
        feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache,
    )


def _timewarp(rate: float) -> TimeWarpMap:
    config = TimeWarpConfig(maximum_drift_rate=0.01)
    masters = (5.0, 15.0)
    anchors = tuple(
        TimeWarpAnchor(
            window_index=index,
            window_start_seconds=max(0.0, master - 5.0),
            window_end_seconds=master + 5.0,
            master_seconds=master,
            source_seconds=rate * master,
            offset_seconds=(rate - 1.0) * master,
            score=0.9,
            second_best_score=0.2,
            score_margin=0.7,
            accepted=True,
            trend_residual_seconds=0.0,
        )
        for index, master in enumerate(masters)
    )
    return TimeWarpMap(
        global_offset_seconds=0.0,
        master_duration_seconds=20.0,
        source_duration_seconds=rate * 20.0 + 1.0,
        control_points=tuple(
            TimeWarpControlPoint(anchor.master_seconds, anchor.source_seconds)
            for anchor in anchors
        ),
        anchors=anchors,
        residuals=TimeWarpResidualDiagnostics(0.0, 0.0, 0.0, 2),
        config=config,
        scalar_fallback=False,
        fallback_reason=None,
    )


def _bridge_fixture(
    provider: DecodedBoundaryOptionProvider,
) -> tuple[
    TimelineCandidate,
    tuple[AuthorizedBoundaryOption, ...],
    tuple[AuthorizedBoundaryOption, ...],
]:
    first = _candidate(0, "0")
    middle = _candidate(1, "1")
    last = TimelineCandidate(
        segment_index=2,
        source_id="0",
        start_seconds=11.0,
        end_seconds=12.0,
        quality_score=0.8,
        candidate_id="candidate:2:0",
        shot_id="shot:2:0",
    )
    entries = provider(
        BoundaryMeasurementKey(0, first, middle)
    ).authorized_options
    exits = provider(
        BoundaryMeasurementKey(1, middle, last)
    ).authorized_options
    return middle, entries, exits


def _bridge_request(
    candidate: TimelineCandidate,
    entry: AuthorizedBoundaryOption,
    exit_: AuthorizedBoundaryOption,
    *,
    fps: float = FPS,
) -> SegmentBridgeDispositionRequest:
    return SegmentBridgeDispositionRequest(
        candidate=candidate,
        entry_option=entry,
        exit_option=exit_,
        duration_frames=(
            candidate.duration_seconds * fps
            + exit_.frame_offset
            - entry.frame_offset
        ),
    )


class _CountingBoundaryDescriptorCache(BoundaryDescriptorCache):
    def __init__(self, *, maximum_entries: int = 4096) -> None:
        super().__init__(maximum_entries=maximum_entries)
        self.begin_count = 0
        self.commit_count = 0
        self.rollback_count = 0

    def begin_transaction(self):
        self.begin_count += 1
        return super().begin_transaction()

    def commit_transaction(self, transaction) -> None:
        self.commit_count += 1
        super().commit_transaction(transaction)

    def rollback_transaction(self, transaction) -> None:
        self.rollback_count += 1
        super().rollback_transaction(transaction)

    def reset_transaction_counts(self) -> None:
        self.begin_count = 0
        self.commit_count = 0
        self.rollback_count = 0


class DecodedBoundaryOptionProviderTests(unittest.TestCase):
    def test_required_direct_feature_cache_preflights_every_source_before_dp(
        self,
    ) -> None:
        created: list[tuple[Path, dict[str, object]]] = []
        closed: list[Path] = []

        class _PersistentSampler:
            def __init__(self, path: Path, **kwargs: object) -> None:
                self.path = path
                created.append((path, dict(kwargs)))

            def persistent_feature_cache_diagnostics(self) -> dict[str, object]:
                return {
                    "enabled": True,
                    "status": "enabled",
                    "source_frame_decode_count": 0,
                    "decode_free_hit_count": 0,
                }

            def close(self) -> None:
                closed.append(self.path)

        sources = tuple(Path(f"/private/source-{index}.mp4") for index in range(3))
        direct_cache = Path("/private/direct-derived-cache")
        with mock.patch(
            "crosscut.joint_boundary_provider._VideoSampler",
            _PersistentSampler,
        ):
            provider = DecodedBoundaryOptionProvider(
                sources=sources,
                audio_offsets=(0.0,) * 3,
                scene_cut_times=((),) * 3,
                source_timewarps=(None,) * 3,
                descriptor_cache=BoundaryDescriptorCache(maximum_entries=128),
                output_fps=FPS,
                feature_cache_directory=Path("/private/legacy-source-cache"),
                feature_derived_cache_directory=direct_cache,
                require_persistent_feature_cache=True,
            )

            self.assertEqual(tuple(item[0] for item in created), sources)
            for _path, kwargs in created:
                self.assertEqual(
                    kwargs["feature_cache_directory"],
                    Path("/private/legacy-source-cache"),
                )
                self.assertEqual(
                    kwargs["feature_derived_cache_directory"], direct_cache
                )
                self.assertIs(kwargs["require_persistent_feature_cache"], True)
            audit = provider.to_dict()["persistent_feature_cache"]
            self.assertEqual(audit["required_source_count"], 3)
            self.assertEqual(audit["initialized_sampler_count"], 3)
            self.assertEqual(audit["enabled_sampler_count"], 3)
            self.assertTrue(audit["all_required_sources_preflighted"])
            self.assertEqual(audit["source_frame_decode_count"], 0)
            self.assertEqual(audit["decode_free_hit_count"], 0)
            serialized = json.dumps(audit, sort_keys=True)
            self.assertNotIn("/private/", serialized)
            self.assertEqual(provider.request_count, 0)
            provider.close()

        self.assertEqual(closed, list(sources))

    def test_required_direct_feature_cache_failure_closes_partial_preflight(
        self,
    ) -> None:
        created: list[Path] = []
        closed: list[Path] = []

        class _FailingPersistentSampler:
            def __init__(self, path: Path, **_kwargs: object) -> None:
                created.append(path)
                if path.name == "source-1.mp4":
                    raise ValueError("persistent feature cache unavailable")
                self.path = path

            def close(self) -> None:
                closed.append(self.path)

        sources = tuple(Path(f"/private/source-{index}.mp4") for index in range(3))
        with (
            mock.patch(
                "crosscut.joint_boundary_provider._VideoSampler",
                _FailingPersistentSampler,
            ),
            self.assertRaisesRegex(ValueError, "cache unavailable"),
        ):
            DecodedBoundaryOptionProvider(
                sources=sources,
                audio_offsets=(0.0,) * 3,
                scene_cut_times=((),) * 3,
                source_timewarps=(None,) * 3,
                descriptor_cache=BoundaryDescriptorCache(maximum_entries=128),
                output_fps=FPS,
                feature_derived_cache_directory=Path("/private/direct-cache"),
                require_persistent_feature_cache=True,
            )

        self.assertEqual(created, list(sources[:2]))
        self.assertEqual(closed, [sources[0]])

    def test_required_persistent_feature_cache_needs_direct_directory(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "feature_derived_cache_directory",
        ):
            DecodedBoundaryOptionProvider(
                sources=(Path("source-a.mp4"), Path("source-b.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
                descriptor_cache=BoundaryDescriptorCache(maximum_entries=128),
                output_fps=FPS,
                require_persistent_feature_cache=True,
            )

    def test_required_cache_guard_preempts_joint_ram_and_memo_fast_paths(
        self,
    ) -> None:
        class _GuardedSampler:
            instances: list["_GuardedSampler"] = []

            def __init__(self, path: Path, **_kwargs: object) -> None:
                self.path = path
                self.guard_unavailable = False
                self.guard_call_count = 0
                self.instances.append(self)

            def assert_required_persistent_cache_available(self) -> None:
                self.guard_call_count += 1
                if self.guard_unavailable:
                    raise RuntimeError("required direct cache unavailable")

            def close(self) -> None:
                return None

        with mock.patch(
            "crosscut.joint_boundary_provider._VideoSampler",
            _GuardedSampler,
        ):
            provider = DecodedBoundaryOptionProvider(
                sources=(Path("source-a.mp4"), Path("source-b.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
                descriptor_cache=BoundaryDescriptorCache(maximum_entries=128),
                output_fps=FPS,
                total_segment_count=3,
                feature_derived_cache_directory=Path("direct-cache"),
                require_persistent_feature_cache=True,
            )
            key = _key()
            provider._memo[provider._memo_key(key)] = object()  # type: ignore[assignment]

            first = _candidate(0, "0")
            middle = _candidate(1, "1")
            last = TimelineCandidate(
                segment_index=2,
                source_id="0",
                start_seconds=11.0,
                end_seconds=12.0,
                candidate_id="guard-last",
                shot_id="guard-last-shot",
            )
            entry = AuthorizedBoundaryOption(
                measurement_key=BoundaryMeasurementKey(0, first, middle),
                frame_offset=0,
                outgoing_transform=BoundaryTransformState("guard-entry-out"),
                incoming_transform=BoundaryTransformState("guard-entry-in"),
                proof_identity="guard-entry-proof",
            )
            exit_ = AuthorizedBoundaryOption(
                measurement_key=BoundaryMeasurementKey(1, middle, last),
                frame_offset=0,
                outgoing_transform=BoundaryTransformState("guard-exit-out"),
                incoming_transform=BoundaryTransformState("guard-exit-in"),
                proof_identity="guard-exit-proof",
            )
            duration = middle.duration_seconds * FPS
            cached_bridge = SegmentBridgeProof(
                candidate=middle,
                entry_option=entry,
                exit_option=exit_,
                duration_frames=duration,
                proof_identity="guard-bridge-proof",
                evidence_complete=True,
                crop_compatible=True,
                track_compatible=True,
            )
            provider._bridge_memo[
                (
                    provider.analysis_basis,
                    provider._configuration_identity,
                    middle,
                    entry,
                    exit_,
                    duration,
                )
            ] = cached_bridge
            provider._bridge_audit[cached_bridge.proof_identity] = object()  # type: ignore[assignment]
            provider.descriptor_cache.store(
                provider._cache_source_indices[1],
                middle.start_seconds,
                _descriptor(0.5),
            )

            _GuardedSampler.instances[1].guard_unavailable = True
            guarded_requests = (
                lambda: provider(key),
                lambda: provider.bridge_provider(
                    middle,
                    entry,
                    exit_,
                    duration,
                ),
                lambda: provider.bridge_disposition_batch(
                    (_bridge_request(middle, entry, exit_),)
                ),
            )
            for request in guarded_requests:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "required direct cache unavailable",
                ):
                    request()

            self.assertEqual(provider.request_count, 0)
            self.assertEqual(provider.bridge_request_count, 0)
            self.assertEqual(provider.raw_descriptor_cache_hit_count, 0)
            provider.close()

        self.assertEqual(
            tuple(
                sampler.guard_call_count
                for sampler in _GuardedSampler.instances
            ),
            (1, 3),
        )

    def test_segment_sample_requires_exact_immutable_subject_basis(self) -> None:
        transform = BoundaryTransformState(
            transform_identity="sample-transform",
            track_id="sample-track",
            subject_basis="dominant_face",
        )
        values = {
            "master_seconds": 1.0,
            "source_seconds": 1.0,
            "transform": transform,
            "subject_bbox_xywh": None,
            "subject_bbox_coverage": None,
            "face_similarity_to_anchor": None,
            "face_similarity_to_previous": None,
            "descriptor": None,
            "face_identity_embedding": (),
        }

        with self.assertRaisesRegex(ValueError, "unsupported"):
            SegmentTransformPathSample(
                **values,
                subject_basis="untrusted_basis",
            )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            SegmentTransformPathSample(
                **values,
                subject_basis="full_frame_identity",
            )
        accepted = SegmentTransformPathSample(
            **values,
            subject_basis="dominant_face",
        )
        self.assertEqual(accepted.subject_basis, transform.subject_basis)

    def test_complete_measurement_preserves_all_seventeen_offsets(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
        )

        options = provider(_key())

        self.assertEqual(
            tuple(item.frame_offset for item in options.offset_audits),
            JOINT_BOUNDARY_FRAME_OFFSETS,
        )
        self.assertEqual(len(options.offset_audits), 17)
        self.assertEqual(len(options.authorized_options), 17)
        self.assertTrue(
            all(
                "raw=" in option.proof_identity
                and ":result=" in option.proof_identity
                for option in options.authorized_options
            )
        )
        selected = options.audit_for(0).authorized_option
        assert selected is not None
        materialization = provider.materialization_proof(
            selected.proof_identity
        )
        self.assertEqual(materialization.frame_offset, 0)
        self.assertEqual(
            materialization.outgoing_track_id,
            selected.outgoing_transform.track_id,
        )
        self.assertEqual(len(materialization.outer_track_proof_identities), 2)
        self.assertTrue(
            all(
                option.scoring_outgoing.exit_shot_scale == 0.4
                and option.scoring_incoming.entry_shot_scale == 0.4
                for option in options.authorized_options
            )
        )

    def test_plus_five_can_be_viable_while_zero_is_nonwaivably_hard(self) -> None:
        incoming_exact_zero = NOMINAL + 0.5 / FPS

        def descriptor(source: int, seconds: float) -> FrameVisualDescriptor:
            if source == 0:
                return _descriptor(0.25)
            relative_frame = round((seconds - incoming_exact_zero) * FPS)
            return _descriptor(0.25 if relative_frame >= 4 else 0.75)

        options = _provider(descriptor)(_key())

        zero = options.audit_for(0)
        plus_five = options.audit_for(5)
        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertTrue(
            set(zero.failure_codes)
            & {
                "person_layout_mismatch",
                "persisted_layout_mismatch",
                "silhouette_mismatch",
                "formation_mismatch",
            }
        )
        self.assertIs(
            plus_five.disposition,
            BoundaryOffsetDisposition.AUTHORIZED,
        )
        assert plus_five.authorized_option is not None
        self.assertEqual(plus_five.authorized_option.frame_offset, 5)
        self.assertEqual(
            plus_five.authorized_option.scoring_incoming.entry_person_layout_embedding,
            _descriptor(0.25).person_layout_embedding,
        )

    def test_missing_decode_evidence_is_incomplete_not_hard(self) -> None:
        options = _provider(
            lambda source, _seconds: None if source == 1 else _descriptor(0.5),
        )(_key())

        self.assertTrue(options.evidence_incomplete)
        self.assertFalse(options.hard_incompatible)
        self.assertTrue(
            all(
                item.disposition
                is BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE
                for item in options.offset_audits
            )
        )
        self.assertTrue(
            all(
                "incoming_exact_descriptor_unavailable" in item.failure_codes
                for item in options.offset_audits
            )
        )

    def test_nonwaivable_layout_pose_and_silhouette_failure_is_hard(self) -> None:
        options = _provider(
            lambda source, _seconds: _descriptor(
                0.22 if source == 0 else 0.78,
                pose_value=0.0 if source == 0 else 1.0,
            ),
        )(_key())

        zero = options.audit_for(0)
        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn("silhouette_mismatch", zero.failure_codes)
        self.assertIn("person_layout_mismatch", zero.failure_codes)
        self.assertIn("pose_mismatch", zero.failure_codes)
        self.assertIsNone(zero.authorized_option)

    def test_focus_and_scale_only_can_produce_crop_resolved_option(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )
        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )

        zero = provider(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.AUTHORIZED,
        )
        option = zero.authorized_option
        assert option is not None
        self.assertIsNotNone(option.outgoing_transform.crop_xywh)
        self.assertIsNotNone(option.incoming_transform.crop_xywh)
        self.assertEqual(option.scoring_outgoing.exit_shot_scale, 0.72)
        self.assertEqual(option.scoring_incoming.entry_shot_scale, 0.72)
        self.assertNotEqual(
            option.scoring_outgoing.exit_focus_bbox_xywh,
            outgoing.primary_face_bbox,
        )
        selection = provider.selection_materialization_proof((option,))
        self.assertEqual(len(selection.segment_paths), 2)
        for path in selection.segment_paths:
            track = path.editorial_transform_track
            assert track is not None
            self.assertTrue(
                all(keyframe.zoom <= 2.0 for keyframe in track.keyframes)
            )
            self.assertTrue(
                all(
                    math.hypot(
                        keyframe.center_x - 0.5,
                        keyframe.center_y - 0.5,
                    )
                    <= 0.25 + 1e-9
                    for keyframe in track.keyframes
                )
            )

    def test_crop_subject_basis_is_immutable_across_large_lru_churn(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )
        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )
        base = _key()
        option = provider(base).audit_for(0).authorized_option
        assert option is not None
        before = provider._bridge_transform_semantic_state(
            option.incoming_transform
        )

        for index in range(140):
            provider(
                BoundaryMeasurementKey(
                    0,
                    replace(
                        base.outgoing,
                        candidate_id=f"basis-outgoing:{index}",
                        shot_id=f"basis-outgoing-shot:{index}",
                    ),
                    replace(
                        base.incoming,
                        candidate_id=f"basis-incoming:{index}",
                        shot_id=f"basis-incoming-shot:{index}",
                    ),
                )
            )

        self.assertEqual(option.incoming_transform.subject_basis, "dominant_face")
        self.assertEqual(
            provider._bridge_transform_semantic_state(
                option.incoming_transform
            ),
            before,
        )
        payload = provider.to_dict()
        self.assertEqual(payload["transform_subject_basis_entry_count"], 0)
        self.assertEqual(
            payload["transient_cache_limits"]["transform_subject_basis"],
            0,
        )

    def test_selection_rechecks_side_evidence_with_exact_renderer_track(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )
        phase = {"name": "measure", "selection_calls": 0}
        transformed_calls: list[
            tuple[int, float, tuple[float, float, float, float] | None]
        ] = []

        def transformed(
            source: int,
            seconds: float,
            _raw: FrameVisualDescriptor,
            transform: BoundaryTransformState,
        ) -> FrameVisualDescriptor:
            if phase["name"].startswith("selection"):
                phase["selection_calls"] += 1
                transformed_calls.append(
                    (source, seconds, transform.crop_xywh)
                )
                if (
                    phase["name"] == "selection_fail"
                    and phase["selection_calls"] == 2
                ):
                    raise RuntimeError("injected transformed decode failure")
            return _descriptor(
                0.5,
                shot_scale=0.72,
                face_bbox=(0.35, 0.28, 0.30, 0.40),
            )

        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=transformed,
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )
        option = provider(_key()).audit_for(0).authorized_option
        assert option is not None

        provider._transformed_descriptor_cache.clear()
        count_before = provider.transformed_descriptor_decode_count
        phase["name"] = "selection_fail"
        with self.assertRaisesRegex(RuntimeError, "injected transformed"):
            provider.selection_materialization_proof((option,))
        self.assertEqual(
            provider.to_dict()["transformed_descriptor_cache_entry_count"],
            0,
        )
        self.assertEqual(
            provider.transformed_descriptor_decode_count,
            count_before,
        )

        phase["name"] = "selection_ok"
        phase["selection_calls"] = 0
        transformed_calls.clear()
        selection = provider.selection_materialization_proof((option,))
        boundary = selection.boundaries[0]
        self.assertEqual(boundary.side_sample_radius_frames, 3)
        expected = {
            (0, NOMINAL - 0.5 / FPS): option.outgoing_transform.crop_xywh,
            (0, NOMINAL - 3.5 / FPS): option.outgoing_transform.crop_xywh,
            (1, NOMINAL + 0.5 / FPS): option.incoming_transform.crop_xywh,
            (1, NOMINAL + 3.5 / FPS): option.incoming_transform.crop_xywh,
        }
        for (source, seconds), crop in expected.items():
            matches = [
                observed_crop
                for observed_source, observed_seconds, observed_crop
                in transformed_calls
                if observed_source == source
                and math.isclose(observed_seconds, seconds, abs_tol=1e-9)
            ]
            self.assertEqual(len(matches), 1)
            assert crop is not None
            assert matches[0] is not None
            self.assertTrue(
                all(
                    math.isclose(left, right, abs_tol=1e-9)
                    for left, right in zip(matches[0], crop, strict=True)
                )
            )

    def test_crop_without_post_transform_decode_stays_incomplete(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )

        zero = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
        )(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
        )
        self.assertIn(
            "post_transform_descriptor_evidence_incomplete",
            zero.failure_codes,
        )

    def test_post_transform_nonwaivable_failure_is_never_authorized(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )

        def transformed(
            source: int,
            _seconds: float,
            _raw: FrameVisualDescriptor,
            _transform: BoundaryTransformState,
        ) -> FrameVisualDescriptor:
            center = 0.22 if source == 0 else 0.78
            return _descriptor(
                center,
                shot_scale=0.72,
                face_bbox=(center - 0.10, 0.30, 0.20, 0.28),
                pose_value=0.0 if source == 0 else 1.0,
            )

        zero = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=transformed,
        )(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertTrue(
            any(code.startswith("post_transform_") for code in zero.failure_codes)
        )
        self.assertIsNone(zero.authorized_option)

    def test_shared_cache_is_namespaced_by_exact_provider_source_context(self) -> None:
        cache = BoundaryDescriptorCache(maximum_entries=8192)
        first = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            descriptor_cache=cache,
            sources=(Path("first-a.mp4"), Path("first-b.mp4")),
        )
        self.assertEqual(len(first(_key()).authorized_options), 17)
        second_calls = 0

        def second_descriptor(
            source: int,
            _seconds: float,
        ) -> FrameVisualDescriptor:
            nonlocal second_calls
            second_calls += 1
            return _descriptor(0.22 if source == 0 else 0.78)

        second = _provider(
            second_descriptor,
            descriptor_cache=cache,
            sources=(Path("second-a.mp4"), Path("second-b.mp4")),
        )
        second_options = second(_key())

        self.assertGreater(second_calls, 0)
        self.assertIs(
            second_options.audit_for(0).disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )

    def test_existing_beat_deviation_penalty_is_bound_to_option_score(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            beat_period_seconds=0.5,
            beat_intensity=1.0,
        )
        options = provider(_key())
        zero = options.audit_for(0).authorized_option
        plus_five = options.audit_for(5).authorized_option
        assert zero is not None and plus_five is not None

        expected_penalty = 0.04 * (5.0 / FPS) / 0.5
        self.assertAlmostEqual(
            plus_five.score_adjustment,
            zero.score_adjustment - expected_penalty,
            places=12,
        )

    def test_middle_segment_bridge_is_full_range_memoized_and_audited(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            quality_score=0.8,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=3,
        )
        entry_set = provider(BoundaryMeasurementKey(0, first, middle))
        exit_set = provider(BoundaryMeasurementKey(1, middle, last))
        entry = entry_set.audit_for(0).authorized_option
        exit_ = exit_set.audit_for(0).authorized_option
        assert entry is not None and exit_ is not None
        before_decode = provider.descriptor_decode_count

        first_proof = provider.bridge_provider(
            middle,
            entry,
            exit_,
            middle.duration_seconds * FPS,
        )
        after_first_decode = provider.descriptor_decode_count
        second_proof = provider.bridge_provider(
            middle,
            entry,
            exit_,
            middle.duration_seconds * FPS,
        )

        self.assertTrue(first_proof.authorized)
        self.assertIs(second_proof, first_proof)
        self.assertGreaterEqual(after_first_decode, before_decode)
        self.assertEqual(provider.descriptor_decode_count, after_first_decode)
        bridge_audit = provider.bridge_audit_for(
            first_proof.proof_identity
        )
        self.assertTrue(bridge_audit["endpoints_included"])
        self.assertEqual(bridge_audit["sampled_frame_count"], 32)
        self.assertEqual(bridge_audit["expected_sample_count"], 32)
        self.assertTrue(bridge_audit["full_frame_centers_included"])
        self.assertEqual(provider.bridge_request_count, 2)
        self.assertEqual(provider.bridge_measurement_count, 1)
        self.assertEqual(
            len(
                provider.materialization_proof(
                    entry.proof_identity
                ).outer_track_proof_identities
            ),
            1,
        )

    def test_scalar_batch_matches_exact_oracle_for_all_seventeen_by_seventeen(
        self,
    ) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1),
            total_segment_count=3,
        )
        middle, entries, exits = _bridge_fixture(provider)
        self.assertEqual(len(entries), 17)
        self.assertEqual(len(exits), 17)
        requests = tuple(
            _bridge_request(middle, entry, exit_)
            for entry in entries
            for exit_ in exits
        )
        bridge_state_before = (
            len(provider._bridge_memo),
            len(provider._semantic_bridge_path_memo),
            len(provider._bridge_audit),
            len(provider._segment_materialization_by_proof),
            provider.bridge_request_count,
            provider.bridge_measurement_count,
            provider._bridge_audit_registration_count,
            provider._bridge_audit_digest,
        )

        scalar = provider.bridge_disposition_batch(requests)

        self.assertEqual(len(scalar), 17 * 17)
        self.assertTrue(
            all(
                isinstance(item, SegmentBridgeDisposition)
                and item.request is request
                for item, request in zip(scalar, requests, strict=True)
            )
        )
        self.assertEqual(
            (
                len(provider._bridge_memo),
                len(provider._semantic_bridge_path_memo),
                len(provider._bridge_audit),
                len(provider._segment_materialization_by_proof),
                provider.bridge_request_count,
                provider.bridge_measurement_count,
                provider._bridge_audit_registration_count,
                provider._bridge_audit_digest,
            ),
            bridge_state_before,
        )

        exact = tuple(
            provider.bridge_provider(
                request.candidate,
                request.entry_option,
                request.exit_option,
                request.duration_frames,
            )
            for request in requests
        )
        for scalar_item, exact_item in zip(scalar, exact, strict=True):
            self.assertEqual(
                (
                    scalar_item.evidence_complete,
                    scalar_item.crop_compatible,
                    scalar_item.track_compatible,
                    scalar_item.failure_codes,
                    scalar_item.score_adjustment,
                ),
                (
                    exact_item.evidence_complete,
                    exact_item.crop_compatible,
                    exact_item.track_compatible,
                    exact_item.failure_codes,
                    exact_item.score_adjustment,
                ),
            )

    def test_scalar_batch_uses_one_transaction_lazy_snapshots_and_no_proofs(
        self,
    ) -> None:
        cache = _CountingBoundaryDescriptorCache()
        provider = _provider(
            lambda _source, _seconds: _descriptor(
                0.5,
                pose_value=0.1,
                face_bbox=(0.42, 0.30, 0.16, 0.22),
            ),
            descriptor_cache=cache,
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
            total_segment_count=3,
        )
        middle, entries, exits = _bridge_fixture(provider)
        crop = (0.1, 0.1, 0.8, 0.8)
        entry = replace(
            entries[8],
            incoming_transform=BoundaryTransformState(
                transform_identity="batch-entry-transform",
                crop_xywh=crop,
                track_id="batch-track",
                subject_basis="dominant_face",
            ),
        )
        exit_ = replace(
            exits[8],
            outgoing_transform=BoundaryTransformState(
                transform_identity="batch-exit-transform",
                crop_xywh=crop,
                track_id="batch-track",
                subject_basis="dominant_face",
            ),
        )
        request = _bridge_request(middle, entry, exit_)
        proof_state_before = (
            tuple(provider._bridge_memo.items()),
            tuple(provider._semantic_bridge_path_memo.items()),
            tuple(provider._bridge_audit.items()),
            tuple(provider._segment_materialization_by_proof.items()),
            provider._bridge_audit_registration_count,
            provider._bridge_audit_digest,
        )
        cache.reset_transaction_counts()

        with mock.patch(
            "crosscut.joint_boundary_provider._sha256_json",
            side_effect=AssertionError("scalar batch attempted SHA proof work"),
        ), mock.patch(
            "crosscut.joint_boundary_provider.SegmentTransformPathSample",
            side_effect=AssertionError("scalar batch attempted sample materialization"),
        ):
            result = provider.bridge_disposition_batch((request, request))

        self.assertEqual(len(result), 2)
        self.assertTrue(all(item.authorized for item in result))
        self.assertEqual(cache.begin_count, 1)
        self.assertEqual(cache.commit_count, 1)
        self.assertEqual(cache.rollback_count, 0)
        self.assertEqual(
            (
                tuple(provider._bridge_memo.items()),
                tuple(provider._semantic_bridge_path_memo.items()),
                tuple(provider._bridge_audit.items()),
                tuple(provider._segment_materialization_by_proof.items()),
                provider._bridge_audit_registration_count,
                provider._bridge_audit_digest,
            ),
            proof_state_before,
        )
        audit = provider.to_dict()["scalar_bridge_batch"]
        self.assertEqual(audit["batch_call_count"], 1)
        self.assertEqual(audit["request_count"], 2)
        self.assertGreater(audit["descriptor_snapshot_hit_count"], 0)
        self.assertGreater(audit["face_identity_snapshot_hit_count"], 0)
        for key in (
            "proof_identities_created",
            "exact_aliases_created",
            "audit_records_created",
            "sha_chains_created",
            "transform_path_samples_created",
            "materializations_created",
        ):
            self.assertFalse(audit[key], key)

        exact = provider.bridge_provider(
            middle,
            entry,
            exit_,
            request.duration_frames,
        )
        self.assertEqual(
            (
                result[0].evidence_complete,
                result[0].crop_compatible,
                result[0].track_compatible,
                result[0].failure_codes,
                result[0].score_adjustment,
            ),
            (
                exact.evidence_complete,
                exact.crop_compatible,
                exact.track_compatible,
                exact.failure_codes,
                exact.score_adjustment,
            ),
        )

    def test_scalar_batch_parity_across_timewarp_fps_and_offset_cases(self) -> None:
        randomizer = random.Random(20260829)
        for fps, rate in ((24.0, 1.003), (29.97, 0.997), (60.0, 1.005)):
            with self.subTest(fps=fps, rate=rate):
                provider = _provider(
                    lambda _source, seconds: _descriptor(
                        0.45 + 0.02 * math.sin(seconds),
                        pose_value=0.1,
                    ),
                    source_timewarps=(_timewarp(rate), _timewarp(rate)),
                    output_fps=fps,
                    total_segment_count=3,
                )
                middle, entries, exits = _bridge_fixture(provider)
                pairs = tuple(
                    (
                        randomizer.randrange(len(entries)),
                        randomizer.randrange(len(exits)),
                    )
                    for _ in range(24)
                )
                requests = tuple(
                    _bridge_request(
                        middle,
                        entries[entry_index],
                        exits[exit_index],
                        fps=fps,
                    )
                    for entry_index, exit_index in pairs
                )

                scalar = provider.bridge_disposition_batch(requests)
                exact = tuple(
                    provider.bridge_provider(
                        request.candidate,
                        request.entry_option,
                        request.exit_option,
                        request.duration_frames,
                    )
                    for request in requests
                )

                self.assertEqual(
                    tuple(
                        (
                            item.evidence_complete,
                            item.crop_compatible,
                            item.track_compatible,
                            item.failure_codes,
                            item.score_adjustment,
                        )
                        for item in scalar
                    ),
                    tuple(
                        (
                            item.evidence_complete,
                            item.crop_compatible,
                            item.track_compatible,
                            item.failure_codes,
                            item.score_adjustment,
                        )
                        for item in exact
                    ),
                )

    def test_scalar_batch_matches_exact_oracle_at_every_segment_gate(self) -> None:
        descriptor_with_face = _descriptor(
            0.5,
            pose_value=0.1,
            face_bbox=(0.42, 0.30, 0.16, 0.22),
        )
        donor = _provider(
            lambda _source, _seconds: descriptor_with_face,
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
            total_segment_count=3,
        )
        middle, entries, exits = _bridge_fixture(donor)
        base_entry = entries[8]
        base_exit = exits[8]

        def transform(
            identity: str,
            *,
            crop=(0.1, 0.1, 0.8, 0.8),
            rotation=0.0,
            track="segment-track",
            basis="dominant_face",
        ) -> BoundaryTransformState:
            return BoundaryTransformState(
                transform_identity=identity,
                crop_xywh=crop,
                rotation_degrees=rotation,
                track_id=track,
                subject_basis=basis,
            )

        def request_with_transforms(
            entry_transform: BoundaryTransformState,
            exit_transform: BoundaryTransformState,
        ) -> SegmentBridgeDispositionRequest:
            return _bridge_request(
                middle,
                replace(base_entry, incoming_transform=entry_transform),
                replace(base_exit, outgoing_transform=exit_transform),
            )

        def assert_parity(
            provider: DecodedBoundaryOptionProvider,
            request: SegmentBridgeDispositionRequest,
            expected_code: str | None,
        ) -> None:
            scalar = provider.bridge_disposition_batch((request,))[0]
            exact = provider.bridge_provider(
                request.candidate,
                request.entry_option,
                request.exit_option,
                request.duration_frames,
            )
            self.assertEqual(
                (
                    scalar.evidence_complete,
                    scalar.crop_compatible,
                    scalar.track_compatible,
                    scalar.failure_codes,
                    scalar.score_adjustment,
                ),
                (
                    exact.evidence_complete,
                    exact.crop_compatible,
                    exact.track_compatible,
                    exact.failure_codes,
                    exact.score_adjustment,
                ),
            )
            if expected_code is None:
                self.assertTrue(scalar.authorized)
            else:
                self.assertIn(expected_code, scalar.failure_codes)

        ordinary_provider = _provider(
            lambda _source, _seconds: descriptor_with_face,
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
            total_segment_count=3,
        )
        transform_cases = (
            (
                transform("authorized-entry"),
                transform("authorized-exit"),
                None,
            ),
            (
                transform("missing-track-entry", track=None),
                transform("missing-track-exit"),
                "segment_track_identity_evidence_incomplete",
            ),
            (
                transform("mismatch-entry", track="track-a"),
                transform("mismatch-exit", track="track-b"),
                "segment_track_identity_mismatch",
            ),
            (
                transform("rotation-entry", rotation=1.0),
                transform("rotation-exit"),
                "segment_rotation_path_evidence_incomplete",
            ),
            (
                transform("envelope-entry", crop=(0.1, 0.1, 0.8, 0.7)),
                transform("envelope-exit", crop=(0.1, 0.1, 0.8, 0.7)),
                "segment_renderer_transform_envelope_violation",
            ),
            (
                transform("out-entry", crop=(0.5, 0.1, 0.8, 0.8)),
                transform("out-exit", crop=(0.5, 0.1, 0.8, 0.8)),
                "segment_renderer_transform_envelope_violation",
            ),
            (
                transform("coverage-entry", crop=(0.0, 0.25, 0.5, 0.5)),
                transform("coverage-exit", crop=(0.0, 0.25, 0.5, 0.5)),
                "segment_subject_exits_crop_track",
            ),
        )
        for entry_transform, exit_transform, expected_code in transform_cases:
            with self.subTest(segment_gate=expected_code or "authorized"):
                assert_parity(
                    ordinary_provider,
                    request_with_transforms(entry_transform, exit_transform),
                    expected_code,
                )

        unsafe_frame = SimpleNamespace(
            time_seconds=0.0,
            zoom=1.25,
            center_x=0.9,
            center_y=0.5,
            roll_degrees=0.0,
        )
        unsafe_track = SimpleNamespace(
            reason="injected-out-of-frame-track",
            keyframes=(unsafe_frame,),
            evaluate=lambda _seconds: unsafe_frame,
        )
        unsafe_request = request_with_transforms(
            transform(
                "unsafe-entry",
                crop=(0.12, 0.12, 0.76, 0.76),
                track="unsafe-track",
            ),
            transform(
                "unsafe-exit",
                crop=(0.12, 0.12, 0.76, 0.76),
                track="unsafe-track",
            ),
        )
        with mock.patch.object(
            ordinary_provider,
            "_renderer_track",
            return_value=unsafe_track,
        ):
            assert_parity(
                ordinary_provider,
                unsafe_request,
                "segment_crop_path_out_of_frame",
            )

        specialized_cases: list[
            tuple[
                str,
                DecodedBoundaryOptionProvider,
                SegmentBridgeDispositionRequest,
                str,
            ]
        ] = []
        cropped_request = request_with_transforms(
            transform("special-entry"),
            transform("special-exit"),
        )
        specialized_cases.append(
            (
                "scene-cut",
                _provider(
                    lambda _source, _seconds: descriptor_with_face,
                    scene_cut_times=((), (10.5,)),
                    face_identity_provider=(
                        lambda _source, _seconds, _descriptor: (1.0, 0.0)
                    ),
                    total_segment_count=3,
                ),
                cropped_request,
                "segment_source_scene_cut_breaks_track",
            )
        )
        specialized_cases.append(
            (
                "descriptor-none",
                _provider(
                    lambda source, seconds: (
                        None
                        if source == 1 and 10.45 < seconds < 10.55
                        else descriptor_with_face
                    ),
                    face_identity_provider=(
                        lambda _source, _seconds, _descriptor: (1.0, 0.0)
                    ),
                    total_segment_count=3,
                ),
                cropped_request,
                "segment_descriptor_evidence_incomplete",
            )
        )
        descriptor_without_face = replace(
            descriptor_with_face,
            primary_face_bbox=None,
            face_count=0,
        )
        specialized_cases.append(
            (
                "missing-geometry",
                _provider(
                    lambda _source, _seconds: descriptor_without_face,
                    total_segment_count=3,
                ),
                cropped_request,
                "segment_subject_track_evidence_incomplete",
            )
        )
        specialized_cases.append(
            (
                "face-empty",
                _provider(
                    lambda _source, _seconds: descriptor_with_face,
                    face_identity_provider=(
                        lambda _source, _seconds, _descriptor: ()
                    ),
                    total_segment_count=3,
                ),
                cropped_request,
                "segment_dominant_face_identity_evidence_incomplete",
            )
        )
        specialized_cases.append(
            (
                "negative-endpoint",
                _provider(
                    lambda _source, _seconds: descriptor_with_face,
                    audio_offsets=(0.0, -20.0),
                    total_segment_count=3,
                ),
                cropped_request,
                "segment_negative_source_local_time",
            )
        )
        for name, provider, request, expected_code in specialized_cases:
            with self.subTest(segment_gate=name):
                assert_parity(provider, request, expected_code)

        for similarity, expected_code in (
            (0.549, "segment_dominant_face_track_identity_mismatch"),
            (0.551, None),
        ):
            identity_after_anchor = (
                similarity,
                math.sqrt(1.0 - similarity * similarity),
            )
            identity_provider = _provider(
                lambda _source, _seconds: descriptor_with_face,
                face_identity_provider=(
                    lambda _source, seconds, _descriptor, later=(
                        identity_after_anchor
                    ): (1.0, 0.0) if seconds < 10.1 else later
                ),
                total_segment_count=3,
            )
            with self.subTest(face_similarity=similarity):
                assert_parity(identity_provider, cropped_request, expected_code)

        def manual_request(
            *,
            duration_frames: float,
            entry_transform: BoundaryTransformState,
            exit_transform: BoundaryTransformState,
        ) -> SegmentBridgeDispositionRequest:
            duration_seconds = duration_frames / FPS
            short_middle = TimelineCandidate(
                segment_index=1,
                source_id="1",
                start_seconds=10.0,
                end_seconds=10.0 + duration_seconds,
                candidate_id=f"short-middle:{duration_frames}",
                shot_id=f"short-middle-shot:{duration_frames}",
            )
            first = TimelineCandidate(
                segment_index=0,
                source_id="0",
                start_seconds=9.0,
                end_seconds=10.0,
                candidate_id=f"short-first:{duration_frames}",
                shot_id=f"short-first-shot:{duration_frames}",
            )
            last = TimelineCandidate(
                segment_index=2,
                source_id="0",
                start_seconds=short_middle.end_seconds,
                end_seconds=short_middle.end_seconds + 1.0,
                candidate_id=f"short-last:{duration_frames}",
                shot_id=f"short-last-shot:{duration_frames}",
            )
            entry = AuthorizedBoundaryOption(
                measurement_key=BoundaryMeasurementKey(0, first, short_middle),
                frame_offset=0,
                outgoing_transform=transform(
                    f"short-outer-entry:{duration_frames}",
                    crop=None,
                    track="outer-track",
                    basis="full_frame_identity",
                ),
                incoming_transform=entry_transform,
                proof_identity=f"short-entry-proof:{duration_frames}",
            )
            exit_ = AuthorizedBoundaryOption(
                measurement_key=BoundaryMeasurementKey(1, short_middle, last),
                frame_offset=0,
                outgoing_transform=exit_transform,
                incoming_transform=transform(
                    f"short-outer-exit:{duration_frames}",
                    crop=None,
                    track="outer-track",
                    basis="full_frame_identity",
                ),
                proof_identity=f"short-exit-proof:{duration_frames}",
            )
            return SegmentBridgeDispositionRequest(
                candidate=short_middle,
                entry_option=entry,
                exit_option=exit_,
                duration_frames=duration_frames,
            )

        short_provider = _provider(
            lambda _source, _seconds: descriptor_with_face,
            total_segment_count=3,
        )
        assert_parity(
            short_provider,
            manual_request(
                duration_frames=1.0,
                entry_transform=transform("one-frame-entry"),
                exit_transform=transform("one-frame-exit"),
            ),
            "segment_duration_below_two_frames",
        )
        assert_parity(
            short_provider,
            manual_request(
                duration_frames=6.0,
                entry_transform=transform(
                    "short-hold-entry",
                    crop=(0.1, 0.1, 0.8, 0.8),
                ),
                exit_transform=transform(
                    "short-hold-exit",
                    crop=(0.15, 0.15, 0.7, 0.7),
                ),
            ),
            "segment_too_short_for_exact_boundary_transform_holds",
        )

        interior_negative_provider = _provider(
            lambda _source, _seconds: descriptor_with_face,
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
            total_segment_count=3,
        )

        def interior_negative_time(_source_index: int, master_seconds: float) -> float:
            if 10.0 + _EPSILON_FOR_TEST < master_seconds < 11.0 - _EPSILON_FOR_TEST:
                return -1.0
            return master_seconds

        _EPSILON_FOR_TEST = 1e-9
        with mock.patch.object(
            interior_negative_provider,
            "_source_time",
            side_effect=interior_negative_time,
        ):
            assert_parity(
                interior_negative_provider,
                cropped_request,
                "segment_negative_source_local_time",
            )

    def test_scalar_batch_rounded_timestamp_snapshots_keep_first_value_and_none(
        self,
    ) -> None:
        state = {"none": False}
        descriptor_calls: list[float] = []
        face_calls: list[float] = []
        descriptor_value = _descriptor(
            0.5,
            pose_value=0.1,
            face_bbox=(0.42, 0.30, 0.16, 0.22),
        )

        def descriptor(_source: int, seconds: float):
            descriptor_calls.append(seconds)
            return None if state["none"] else descriptor_value

        def face_identity(
            _source: int,
            seconds: float,
            _descriptor_value: FrameVisualDescriptor,
        ) -> tuple[float, ...]:
            face_calls.append(seconds)
            return (1.0, seconds)

        provider = _provider(
            descriptor,
            face_identity_provider=face_identity,
            total_segment_count=3,
        )
        middle, entries, exits = _bridge_fixture(provider)
        crop = (0.1, 0.1, 0.8, 0.8)
        entry = replace(
            entries[8],
            incoming_transform=BoundaryTransformState(
                transform_identity="collision-entry",
                crop_xywh=crop,
                track_id="collision-track",
                subject_basis="dominant_face",
            ),
        )
        exit_ = replace(
            exits[8],
            outgoing_transform=BoundaryTransformState(
                transform_identity="collision-exit",
                crop_xywh=crop,
                track_id="collision-track",
                subject_basis="dominant_face",
            ),
        )
        request = _bridge_request(middle, entry, exit_)
        descriptor_calls.clear()
        face_calls.clear()

        def colliding_time(
            _source_index: int,
            master_seconds: float,
        ) -> float:
            return 1.00000001 + (master_seconds - 10.0) * 1e-8

        before = provider.to_dict()["scalar_bridge_batch"]
        with mock.patch.object(
            provider,
            "_source_time",
            side_effect=colliding_time,
        ):
            dispositions = provider.bridge_disposition_batch((request, request))

        self.assertTrue(all(item.authorized for item in dispositions))
        self.assertEqual(len(descriptor_calls), 1)
        self.assertEqual(len(face_calls), 1)
        self.assertAlmostEqual(descriptor_calls[0], 1.00000001, places=12)
        self.assertAlmostEqual(face_calls[0], 1.00000001, places=12)
        after = provider.to_dict()["scalar_bridge_batch"]
        self.assertEqual(
            after["descriptor_snapshot_miss_count"]
            - before["descriptor_snapshot_miss_count"],
            1,
        )
        self.assertGreaterEqual(
            after["descriptor_snapshot_hit_count"]
            - before["descriptor_snapshot_hit_count"],
            63,
        )
        self.assertEqual(
            after["face_identity_snapshot_miss_count"]
            - before["face_identity_snapshot_miss_count"],
            1,
        )
        self.assertGreaterEqual(
            after["face_identity_snapshot_hit_count"]
            - before["face_identity_snapshot_hit_count"],
            63,
        )

        state["none"] = True
        descriptor_calls.clear()
        face_calls.clear()

        def colliding_none_time(
            _source_index: int,
            master_seconds: float,
        ) -> float:
            return 2.00000001 + (master_seconds - 10.0) * 1e-8

        with mock.patch.object(
            provider,
            "_source_time",
            side_effect=colliding_none_time,
        ):
            missing = provider.bridge_disposition_batch((request, request))

        self.assertTrue(all(not item.evidence_complete for item in missing))
        self.assertTrue(
            all(
                item.failure_codes
                == ("segment_descriptor_evidence_incomplete",)
                for item in missing
            )
        )
        self.assertEqual(len(descriptor_calls), 1)
        self.assertEqual(face_calls, [])
        self.assertAlmostEqual(descriptor_calls[0], 2.00000001, places=12)

    def test_scalar_batch_failure_and_malformed_inputs_restore_all_state(
        self,
    ) -> None:
        state = {"fail": False}

        def descriptor(
            source: int,
            seconds: float,
        ) -> FrameVisualDescriptor:
            if state["fail"] and source == 1 and 12.45 < seconds < 12.55:
                raise RuntimeError("injected scalar batch descriptor failure")
            return _descriptor(0.5, pose_value=0.1)

        cache = _CountingBoundaryDescriptorCache()
        provider = _provider(
            descriptor,
            descriptor_cache=cache,
            total_segment_count=3,
        )
        middle, entries, exits = _bridge_fixture(provider)
        request = _bridge_request(middle, entries[8], exits[8])
        later_first = TimelineCandidate(
            segment_index=0,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="later-first",
            shot_id="later-first-shot",
        )
        later_middle = TimelineCandidate(
            segment_index=1,
            source_id="1",
            start_seconds=12.0,
            end_seconds=13.0,
            candidate_id="later-middle",
            shot_id="later-middle-shot",
        )
        later_last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=13.0,
            end_seconds=14.0,
            candidate_id="later-last",
            shot_id="later-last-shot",
        )
        later_entry = provider(
            BoundaryMeasurementKey(0, later_first, later_middle)
        ).audit_for(0).authorized_option
        later_exit = provider(
            BoundaryMeasurementKey(1, later_middle, later_last)
        ).audit_for(0).authorized_option
        assert later_entry is not None and later_exit is not None
        later_request = _bridge_request(
            later_middle,
            later_entry,
            later_exit,
        )
        before = provider.to_dict()
        cache.reset_transaction_counts()
        state["fail"] = True

        with self.assertRaisesRegex(RuntimeError, "injected scalar batch"):
            provider.bridge_disposition_batch((request, later_request))

        self.assertEqual(provider.to_dict(), before)
        self.assertEqual(cache.begin_count, 1)
        self.assertEqual(cache.commit_count, 0)
        self.assertEqual(cache.rollback_count, 1)
        state["fail"] = False

        malformed = replace(
            request,
            duration_frames=request.duration_frames + 1.0,
        )
        cache.reset_transaction_counts()
        before_malformed = provider.to_dict()
        with self.assertRaisesRegex(ValueError, "duration identity"):
            provider.bridge_disposition_batch((request, malformed))
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            provider.bridge_disposition_batch(())
        self.assertEqual(provider.to_dict(), before_malformed)
        self.assertEqual(cache.begin_count, 0)
        self.assertEqual(cache.commit_count, 0)
        self.assertEqual(cache.rollback_count, 0)

        exact = provider.bridge_provider(
            middle,
            request.entry_option,
            request.exit_option,
            request.duration_frames,
        )
        self.assertTrue(exact.authorized)

    def test_semantic_bridge_aliases_share_scalar_measurement_and_selected_rehydrate(
        self,
    ) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1),
            total_segment_count=3,
        )
        entry_set = provider(BoundaryMeasurementKey(0, first, middle))
        exit_set = provider(BoundaryMeasurementKey(1, middle, last))
        entry = entry_set.audit_for(0).authorized_option
        exit_ = exit_set.audit_for(0).authorized_option
        assert entry is not None and exit_ is not None
        first_proof = provider.bridge_provider(middle, entry, exit_, 30.0)
        decoded_after_first = provider.descriptor_decode_count

        alias_entry = replace(
            entry,
            proof_identity=entry.proof_identity + ":exact-alias",
        )
        alias_exit = replace(
            exit_,
            proof_identity=exit_.proof_identity + ":exact-alias",
        )
        alias_proof = provider.bridge_provider(
            middle,
            alias_entry,
            alias_exit,
            30.0,
        )

        self.assertNotEqual(alias_proof.proof_identity, first_proof.proof_identity)
        self.assertEqual(alias_proof.entry_option, alias_entry)
        self.assertEqual(alias_proof.exit_option, alias_exit)
        self.assertEqual(provider.bridge_measurement_count, 1)
        self.assertEqual(provider.descriptor_decode_count, decoded_after_first)
        self.assertGreaterEqual(provider.bridge_semantic_memo_hit_count, 1)
        alias_audit = provider.bridge_audit_for(alias_proof.proof_identity)
        self.assertEqual(
            alias_audit["entry_proof_identity"],
            alias_entry.proof_identity,
        )
        self.assertEqual(
            alias_audit["exit_proof_identity"],
            alias_exit.proof_identity,
        )
        self.assertTrue(
            all(
                not hasattr(item, "materialization")
                for item in provider._semantic_bridge_path_memo.values()
            )
        )

        measured = 0
        for candidate_entry in entry_set.authorized_options:
            for candidate_exit in exit_set.authorized_options:
                if (
                    candidate_entry.frame_offset == 0
                    and candidate_exit.frame_offset == 0
                ):
                    continue
                duration = (
                    30.0
                    + candidate_exit.frame_offset
                    - candidate_entry.frame_offset
                )
                provider.bridge_provider(
                    middle,
                    candidate_entry,
                    candidate_exit,
                    duration,
                )
                measured += 1
                if measured >= 34:
                    break
            if measured >= 34:
                break
        with self.assertRaises(KeyError):
            provider.segment_materialization_proof(first_proof.proof_identity)

        materialization = provider.selection_materialization_proof(
            (entry, exit_),
            (first_proof,),
        )

        self.assertEqual(len(materialization.segment_paths), 3)
        rehydrated = provider.segment_materialization_proof(
            first_proof.proof_identity
        )
        self.assertEqual(rehydrated.entry_transform, entry.incoming_transform)
        self.assertEqual(rehydrated.exit_transform, exit_.outgoing_transform)
        self.assertGreaterEqual(
            provider.boundary_materialization_rehydration_count,
            1,
        )
        self.assertGreaterEqual(
            provider.bridge_materialization_rehydration_count,
            1,
        )
        payload = provider.to_dict()
        self.assertLessEqual(payload["semantic_bridge_path_entry_count"], 4096)
        self.assertLessEqual(payload["segment_materialization_proof_count"], 32)
        self.assertEqual(
            payload["last_selection"]["segment_records"][1]["proof_identity"],
            first_proof.proof_identity,
        )
        selected_payload = json.dumps(
            payload["last_selection"],
            sort_keys=True,
        )
        self.assertNotIn("descriptor", selected_payload)
        self.assertNotIn("face_identity_embedding", selected_payload)
        self.assertEqual(
            len(
                provider.materialization_proof(
                    exit_.proof_identity
                ).outer_track_proof_identities
            ),
            1,
        )

    def test_joint_core_runs_three_segments_with_provider_bridge_api(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=3,
        )

        result = select_joint_timeline(
            ((first,), (middle,), (last,)),
            option_provider=provider,
            bridge_provider=provider.bridge_provider,
            fps=FPS,
        )

        self.assertEqual(len(result.choices), 3)
        self.assertEqual(len(result.boundary_options), 2)
        self.assertEqual(len(result.segment_bridge_proofs), 1)
        self.assertTrue(result.segment_bridge_proofs[0].authorized)

    def test_bridge_track_mismatch_is_hard(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=3,
        )
        entry = provider(
            BoundaryMeasurementKey(0, first, middle)
        ).audit_for(0).authorized_option
        exit_ = provider(
            BoundaryMeasurementKey(1, middle, last)
        ).audit_for(0).authorized_option
        assert entry is not None and exit_ is not None
        mismatched_exit = replace(
            exit_,
            outgoing_transform=BoundaryTransformState(
                transform_identity="mismatched-track-transform",
                track_id="another-track",
            ),
        )

        proof = provider.bridge_provider(
            middle,
            entry,
            mismatched_exit,
            middle.duration_seconds * FPS,
        )

        self.assertTrue(proof.hard_incompatible)
        self.assertIn("segment_track_identity_mismatch", proof.failure_codes)

    def test_bridge_missing_interior_descriptor_is_incomplete_not_hard(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )

        def descriptor(
            source: int,
            seconds: float,
        ) -> FrameVisualDescriptor | None:
            if source == 1 and math.isclose(
                seconds,
                10.5 + 0.5 / FPS,
                abs_tol=1e-9,
            ):
                return None
            return _descriptor(0.5)

        provider = _provider(descriptor, total_segment_count=3)
        entry = provider(
            BoundaryMeasurementKey(0, first, middle)
        ).audit_for(0).authorized_option
        exit_ = provider(
            BoundaryMeasurementKey(1, middle, last)
        ).audit_for(0).authorized_option
        assert entry is not None and exit_ is not None

        proof = provider.bridge_provider(
            middle,
            entry,
            exit_,
            middle.duration_seconds * FPS,
        )

        self.assertTrue(proof.evidence_incomplete)
        self.assertFalse(proof.hard_incompatible)
        self.assertIn(
            "segment_descriptor_evidence_incomplete",
            proof.failure_codes,
        )

    def test_crop_track_rejects_an_interior_source_scene_cut(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )
        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            scene_cut_times=((), (10.5,)),
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )

        zero = provider(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn(
            "outer_track_segment_source_scene_cut_breaks_track",
            zero.failure_codes,
        )

    def test_crop_track_samples_every_visible_half_frame_center(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )

        def descriptor(source: int, seconds: float) -> FrameVisualDescriptor:
            if source == 0:
                return outgoing
            if math.isclose(seconds, 10.05, rel_tol=0.0, abs_tol=1e-9):
                return replace(
                    incoming,
                    primary_face_bbox=(0.0, 0.22, 0.12, 0.20),
                )
            return incoming

        provider = _provider(
            descriptor,
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )

        zero = provider(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn(
            "outer_track_segment_subject_exits_crop_track",
            zero.failure_codes,
        )
        failed_paths = tuple(
            item
            for item in provider.to_dict()["outer_track_audits"]
            if "segment_subject_exits_crop_track" in item["failure_codes"]
        )
        self.assertTrue(failed_paths)
        self.assertTrue(
            any(
                not item["full_frame_centers_included"]
                for item in failed_paths
            )
        )

    def test_dominant_face_track_rejects_member_identity_drift(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )

        def identity(
            source: int,
            seconds: float,
            _descriptor_value: FrameVisualDescriptor,
        ) -> tuple[float, ...]:
            if source == 1 and seconds >= 10.5:
                return (0.0, 1.0)
            return (1.0, 0.0)

        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=identity,
        )

        zero = provider(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn(
            "outer_track_segment_dominant_face_track_identity_mismatch",
            zero.failure_codes,
        )

    def test_identity_path_does_not_require_subject_tracking_geometry(self) -> None:
        descriptor = replace(
            _descriptor(0.5),
            face_count=0,
            primary_face_bbox=None,
            silhouette_bbox_xywh=None,
            silhouette_centroid_xy=None,
            silhouette_confidence=0.0,
        )
        key = BoundaryMeasurementKey(
            0,
            _candidate(0, "0"),
            _candidate(1, "0"),
        )
        provider = _provider(lambda _source, _seconds: descriptor)

        zero = provider(key).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.AUTHORIZED,
        )

    def test_selection_materialization_binds_all_real_segment_tracks(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=3,
        )
        result = select_joint_timeline(
            ((first,), (middle,), (last,)),
            option_provider=provider,
            bridge_provider=provider.bridge_provider,
            fps=FPS,
        )

        materialization = provider.selection_materialization_proof(
            result.boundary_options,
            result.segment_bridge_proofs,
        )

        self.assertEqual(len(materialization.boundaries), 2)
        self.assertEqual(len(materialization.segment_paths), 3)
        self.assertEqual(len(materialization.selected_bridge_audits), 1)
        self.assertEqual(
            materialization.selected_bridge_audits[0]["proof_identity"],
            result.segment_bridge_proofs[0].proof_identity,
        )
        self.assertEqual(
            tuple(
                path.candidate.segment_index
                for path in materialization.segment_paths
            ),
            (0, 1, 2),
        )
        for path in materialization.segment_paths:
            self.assertEqual(len(path.samples), path.expected_sample_count)
            self.assertEqual(
                path.samples[0].master_seconds,
                path.start_master_seconds,
            )
            self.assertEqual(
                path.samples[-1].master_seconds,
                path.end_master_seconds,
            )
            self.assertIsNotNone(path.editorial_transform_track)
            self.assertIsNotNone(path.samples[0].descriptor)
        self.assertFalse(hasattr(materialization, "to_dict"))
        first_boundary = materialization.boundaries[0]
        self.assertIn(first_boundary.raw_proof_identity, first_boundary.proof_identity)
        self.assertIn(
            first_boundary.result_proof_identity,
            first_boundary.proof_identity,
        )
        self.assertIsNotNone(
            first_boundary.raw_outgoing_preceding_descriptor
        )

    def test_selection_rejects_bridge_from_different_boundary_options(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1),
            total_segment_count=3,
        )
        first_options = provider(BoundaryMeasurementKey(0, first, middle))
        second_options = provider(BoundaryMeasurementKey(1, middle, last))
        selected_entry = first_options.audit_for(0).authorized_option
        selected_exit = second_options.audit_for(0).authorized_option
        mismatched_entry = first_options.audit_for(-1).authorized_option
        assert selected_entry is not None
        assert selected_exit is not None
        assert mismatched_entry is not None
        mismatched_bridge = provider.bridge_provider(
            middle,
            mismatched_entry,
            selected_exit,
            31.0,
        )
        self.assertTrue(mismatched_bridge.authorized)

        with self.assertRaisesRegex(ValueError, "endpoints do not match"):
            provider.selection_materialization_proof(
                (selected_entry, selected_exit),
                (mismatched_bridge,),
            )

    def test_failed_selection_restores_previous_successful_pin_set(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1)
        )
        option = provider(_key()).audit_for(0).authorized_option
        assert option is not None
        successful = provider.selection_materialization_proof((option,))
        before = provider.to_dict()

        with self.assertRaisesRegex(ValueError, "one boundary option"):
            provider.selection_materialization_proof(())

        after = provider.to_dict()
        self.assertEqual(after["last_selection"], before["last_selection"])
        self.assertEqual(
            after["selected_boundary_materialization_pin_count"],
            before["selected_boundary_materialization_pin_count"],
        )
        self.assertEqual(
            after["selected_segment_materialization_pin_count"],
            before["selected_segment_materialization_pin_count"],
        )
        self.assertIs(
            provider.materialization_proof(option.proof_identity),
            successful.boundaries[0],
        )

    def test_bridge_callback_exception_rolls_back_partial_state(self) -> None:
        first = _candidate(0, "0")
        middle = _candidate(1, "1")
        last = TimelineCandidate(
            segment_index=2,
            source_id="0",
            start_seconds=11.0,
            end_seconds=12.0,
            candidate_id="candidate:2:0",
            shot_id="shot:2:0",
        )
        state = {"fail": False}

        def descriptor(source: int, seconds: float) -> FrameVisualDescriptor:
            if state["fail"] and source == 1 and 10.49 < seconds < 10.54:
                raise RuntimeError("injected bridge decode failure")
            return _descriptor(0.5, pose_value=0.1)

        provider = _provider(
            descriptor,
            total_segment_count=3,
        )
        entry = provider(
            BoundaryMeasurementKey(0, first, middle)
        ).audit_for(0).authorized_option
        exit_ = provider(
            BoundaryMeasurementKey(1, middle, last)
        ).audit_for(0).authorized_option
        assert entry is not None
        assert exit_ is not None
        before = provider.to_dict()
        state["fail"] = True
        with self.assertRaisesRegex(RuntimeError, "injected bridge"):
            provider.bridge_provider(middle, entry, exit_, 30.0)
        failed = provider.to_dict()
        self.assertEqual(failed["bridge_measurement_count"], 0)
        self.assertEqual(failed["bridge_memo_entry_count"], 0)
        self.assertEqual(failed["bridge_audits"], [])
        self.assertEqual(
            failed["segment_materialization_proof_count"],
            before["segment_materialization_proof_count"],
        )
        self.assertEqual(
            failed["descriptor_cache"]["entry_count"],
            before["descriptor_cache"]["entry_count"],
        )

        state["fail"] = False
        proof = provider.bridge_provider(middle, entry, exit_, 30.0)
        self.assertTrue(proof.authorized)
        self.assertEqual(provider.bridge_measurement_count, 1)

    def test_short_span_clamps_all_gate_samples_to_rendered_centers(self) -> None:
        calls: list[tuple[int, float]] = []

        def descriptor(source: int, seconds: float) -> FrameVisualDescriptor:
            calls.append((source, seconds))
            return _descriptor(0.5, pose_value=0.1)

        outgoing = TimelineCandidate(
            segment_index=0,
            source_id="0",
            start_seconds=9.9,
            end_seconds=10.0,
            candidate_id="short:0",
            shot_id="short-shot:0",
        )
        incoming = TimelineCandidate(
            segment_index=1,
            source_id="1",
            start_seconds=10.0,
            end_seconds=10.1,
            candidate_id="short:1",
            shot_id="short-shot:1",
        )
        key = BoundaryMeasurementKey(0, outgoing, incoming)
        provider = _provider(descriptor)

        options = provider(key)
        zero = options.audit_for(0)
        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.AUTHORIZED,
        )
        option = zero.authorized_option
        assert option is not None
        materialization = provider.materialization_proof(option.proof_identity)
        self.assertEqual(materialization.side_sample_radius_frames, 2)
        provider.selection_materialization_proof((option,))
        self.assertTrue(
            all(9.9 - 1e-9 <= seconds <= 10.1 + 1e-9 for _, seconds in calls)
        )
        self.assertFalse(
            any(
                math.isclose(seconds, 9.883333333333333, abs_tol=1e-9)
                or math.isclose(seconds, 10.116666666666667, abs_tol=1e-9)
                for _, seconds in calls
            )
        )
        self.assertIs(
            options.audit_for(-2).disposition,
            BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
        )

    def test_same_source_outer_failure_retains_exact_raw_identity(self) -> None:
        def descriptor(_source: int, seconds: float):
            if math.isclose(seconds, 9.0, abs_tol=1e-9):
                return None
            return _descriptor(0.5, pose_value=0.1)

        key = BoundaryMeasurementKey(
            0,
            _candidate(0, "0"),
            _candidate(1, "0"),
        )
        provider = _provider(descriptor)

        zero = provider(key).audit_for(0)
        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
        )
        audit_zero = provider.audit_for(key)["offset_audits"][8]
        self.assertEqual(audit_zero["frame_offset"], 0)
        self.assertIsInstance(audit_zero["raw_evidence_identity"], str)
        self.assertTrue(audit_zero["raw_evidence_identity"])

    def test_same_source_negative_clock_is_hard(self) -> None:
        key = BoundaryMeasurementKey(
            0,
            _candidate(0, "0"),
            _candidate(1, "0"),
        )
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1),
            audio_offsets=(-10.0, 0.0),
        )

        zero = provider(key).audit_for(0)
        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn("negative_source_local_time", zero.failure_codes)

    def test_provider_callback_exception_rolls_back_all_owned_state(self) -> None:
        state = {"calls": 0, "fail": True}

        def descriptor(_source: int, _seconds: float) -> FrameVisualDescriptor:
            state["calls"] += 1
            if state["fail"] and state["calls"] == 70:
                raise RuntimeError("injected provider decode failure")
            return _descriptor(0.5, pose_value=0.1)

        provider = _provider(descriptor)
        with self.assertRaisesRegex(RuntimeError, "injected provider"):
            provider(_key())
        payload = provider.to_dict()
        self.assertEqual(payload["measurement_count"], 0)
        self.assertEqual(payload["measurement_failure_count"], 1)
        self.assertEqual(payload["memo_hit_count"], 0)
        self.assertEqual(payload["boundary_memo_entry_count"], 0)
        self.assertEqual(payload["measurement_audits"], [])
        self.assertEqual(payload["outer_track_audits"], [])
        self.assertEqual(payload["outer_track_audit_registration_count"], 0)
        self.assertIsNone(payload["outer_track_audit_chain_sha256"])
        self.assertEqual(payload["materialization_proof_count"], 0)
        self.assertEqual(payload["segment_materialization_proof_count"], 0)
        self.assertEqual(payload["face_identity_cache_entry_count"], 0)
        self.assertEqual(payload["transformed_descriptor_cache_entry_count"], 0)
        self.assertEqual(payload["transform_subject_basis_entry_count"], 0)
        self.assertEqual(payload["descriptor_cache"]["entry_count"], 0)

        state["fail"] = False
        options = provider(_key())
        self.assertEqual(len(options.offset_audits), 17)
        self.assertEqual(provider.measurement_count, 1)

    def test_failed_measurement_restores_shared_descriptor_lru_and_counters(
        self,
    ) -> None:
        cache = BoundaryDescriptorCache(maximum_entries=1)
        sentinel = _descriptor(0.42, pose_value=0.1)
        cache.store(999, 1.0, sentinel)
        before = cache.diagnostics()
        state = {"calls": 0}

        def descriptor(_source: int, _seconds: float) -> FrameVisualDescriptor:
            state["calls"] += 1
            if state["calls"] == 2:
                raise RuntimeError("injected full-cache decode failure")
            return _descriptor(0.5, pose_value=0.1)

        provider = _provider(descriptor, descriptor_cache=cache)
        with self.assertRaisesRegex(RuntimeError, "full-cache"):
            provider(_key())

        self.assertEqual(cache.diagnostics(), before)
        hit, restored = cache.lookup(999, 1.0)
        self.assertTrue(hit)
        self.assertIs(restored, sentinel)

    def test_close_releases_all_process_local_evidence(self) -> None:
        outgoing = _descriptor(
            0.5,
            shot_scale=0.22,
            face_bbox=(0.18, 0.30, 0.16, 0.22),
        )
        incoming = _descriptor(
            0.5,
            shot_scale=0.72,
            face_bbox=(0.60, 0.22, 0.30, 0.40),
        )
        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=(
                lambda _source, _seconds, _raw, _transform: _descriptor(
                    0.5,
                    shot_scale=0.72,
                    face_bbox=(0.35, 0.28, 0.30, 0.40),
                )
            ),
            face_identity_provider=(
                lambda _source, _seconds, _descriptor: (1.0, 0.0)
            ),
        )
        key = _key()
        option = provider(key).audit_for(0).authorized_option
        assert option is not None
        provider.selection_materialization_proof((option,))
        self.assertGreater(provider.to_dict()["materialization_proof_count"], 0)

        provider.close()
        payload = provider.to_dict()
        self.assertTrue(payload["closed"])
        self.assertEqual(payload["measurement_audits"], [])
        self.assertEqual(payload["bridge_audits"], [])
        self.assertEqual(payload["outer_track_audits"], [])
        for name in (
            "materialization_proof_count",
            "segment_materialization_proof_count",
            "boundary_memo_entry_count",
            "bridge_memo_entry_count",
            "semantic_bridge_path_entry_count",
            "selected_boundary_materialization_pin_count",
            "selected_segment_materialization_pin_count",
            "face_identity_cache_entry_count",
            "transformed_descriptor_cache_entry_count",
            "sampler_entry_count",
            "transform_subject_basis_entry_count",
            "measurement_failure_count",
            "memo_hit_count",
            "bridge_measurement_failure_count",
            "bridge_memo_hit_count",
            "selection_failure_count",
            "raw_descriptor_cache_hit_count",
            "raw_descriptor_cache_miss_count",
            "transformed_descriptor_cache_hit_count",
            "face_identity_cache_hit_count",
        ):
            self.assertEqual(payload[name], 0, name)
        self.assertEqual(payload["descriptor_cache"]["entry_count"], 0)
        with self.assertRaises(RuntimeError):
            provider(key)
        with self.assertRaises(RuntimeError):
            provider.audit_for(key)
        with self.assertRaises(RuntimeError):
            provider.materialization_proof(option.proof_identity)

    def test_segment_count_preflight_rejects_mismatched_selector_shape(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=4,
        )
        provider.validate_segment_count(4)
        with self.assertRaisesRegex(ValueError, "analysis basis"):
            provider.validate_segment_count(3)

    def test_capacity_audit_preserves_full_120_by_8_offset_domain(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            total_segment_count=120,
        )

        audit = provider.capacity_audit(8)

        self.assertEqual(audit["boundary_count"], 119)
        self.assertEqual(audit["measurement_key_upper_bound"], 7_616)
        self.assertEqual(audit["offsets_per_measurement"], 17)
        self.assertEqual(audit["offset_audit_upper_bound"], 129_472)
        self.assertEqual(
            audit["distinct_source_bridge_request_upper_bound"],
            13_593_600,
        )
        self.assertEqual(
            audit["all_cross_bridge_request_upper_bound"],
            17_460_224,
        )
        self.assertEqual(
            audit["exact_option_state_upper_bound_per_layer"],
            960,
        )
        self.assertEqual(
            audit["identity_semantic_bridge_measurement_upper_bound"],
            272_816,
        )
        self.assertTrue(audit["full_offset_domain_required"])
        self.assertFalse(audit["offset_subsampling_authorized"])

    def test_manifest_audits_are_bounded_summaries_with_exact_totals(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5, pose_value=0.1),
        )
        base = _key()
        for index in range(12):
            provider(
                BoundaryMeasurementKey(
                    0,
                    replace(
                        base.outgoing,
                        candidate_id=f"bounded-outgoing:{index}",
                        shot_id=f"bounded-outgoing-shot:{index}",
                    ),
                    replace(
                        base.incoming,
                        candidate_id=f"bounded-incoming:{index}",
                        shot_id=f"bounded-incoming-shot:{index}",
                    ),
                )
            )

        payload = provider.to_dict()
        encoded = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["measurement_audit_total_count"], 12)
        self.assertEqual(payload["measurement_audit_retained_count"], 8)
        self.assertTrue(payload["measurement_audits_truncated"])
        self.assertEqual(payload["offset_audit_count"], 12 * 17)
        self.assertTrue(payload["exact_offset_domain_complete"])
        self.assertEqual(payload["outer_track_audit_retained_count"], 32)
        self.assertGreater(
            payload["outer_track_audit_registration_count"],
            payload["outer_track_audit_retained_count"],
        )
        self.assertTrue(payload["outer_track_audits_truncated"])
        self.assertIsNotNone(payload["measurement_audit_chain_sha256"])
        self.assertIsNotNone(payload["outer_track_audit_chain_sha256"])
        self.assertTrue(payload["audit_payload_bounded"])
        self.assertLess(len(encoded), 2_000_000)
        self.assertNotIn("face_identity_embedding", encoded)
        self.assertNotIn("raw_outgoing_descriptor", encoded)

    def test_common_raw_descriptors_hit_cache_across_distinct_edges(self) -> None:
        calls = 0

        def descriptor(_source: int, _seconds: float) -> FrameVisualDescriptor:
            nonlocal calls
            calls += 1
            return _descriptor(0.5, pose_value=0.1)

        provider = _provider(descriptor)
        first_key = _key()
        provider(first_key)
        first_calls = calls
        second_key = BoundaryMeasurementKey(
            0,
            replace(
                first_key.outgoing,
                candidate_id="alternate-outgoing",
                shot_id="alternate-outgoing-shot",
            ),
            replace(
                first_key.incoming,
                candidate_id="alternate-incoming",
                shot_id="alternate-incoming-shot",
            ),
        )

        provider(second_key)
        payload = provider.to_dict()

        self.assertEqual(calls, first_calls)
        self.assertEqual(payload["measurement_count"], 2)
        self.assertEqual(payload["cross_source_measurement_count"], 2)
        self.assertEqual(payload["offset_audit_count"], 34)
        self.assertEqual(payload["expected_offset_audit_count"], 34)
        self.assertTrue(payload["exact_offset_domain_complete"])
        self.assertGreater(payload["raw_descriptor_cache_hit_count"], 0)

    def test_same_source_edges_reuse_common_raw_descriptors(self) -> None:
        calls = 0

        def descriptor(_source: int, _seconds: float) -> FrameVisualDescriptor:
            nonlocal calls
            calls += 1
            return _descriptor(0.5, pose_value=0.1)

        provider = _provider(descriptor)
        outgoing = _candidate(0, "0")
        incoming = _candidate(1, "0")
        provider(BoundaryMeasurementKey(0, outgoing, incoming))
        first_calls = calls

        provider(
            BoundaryMeasurementKey(
                0,
                replace(
                    outgoing,
                    candidate_id="same-source-alternate-outgoing",
                ),
                replace(
                    incoming,
                    candidate_id="same-source-alternate-incoming",
                ),
            )
        )
        payload = provider.to_dict()

        self.assertEqual(calls, first_calls)
        self.assertEqual(payload["same_source_measurement_count"], 2)
        self.assertEqual(payload["offset_audit_count"], 34)
        self.assertTrue(payload["exact_offset_domain_complete"])
        self.assertGreater(payload["raw_descriptor_cache_hit_count"], 0)

    def test_raw_proof_identity_binds_each_offset_decoded_evidence(self) -> None:
        state = {"center": 0.5}

        def descriptor(
            _source: int,
            _seconds: float,
        ) -> FrameVisualDescriptor:
            return _descriptor(state["center"])

        first_provider = _provider(descriptor)
        first = first_provider(_key())
        first_raw = tuple(
            first_provider.materialization_proof(option.proof_identity)
            .raw_proof_identity
            for option in first.authorized_options
        )
        self.assertEqual(len(set(first_raw)), 17)
        state["center"] = 0.51
        second_provider = _provider(descriptor)
        second = second_provider(_key())
        second_zero = second.audit_for(0).authorized_option
        assert second_zero is not None
        second_raw = second_provider.materialization_proof(
            second_zero.proof_identity
        ).raw_proof_identity
        self.assertNotEqual(first_raw[8], second_raw)

    def test_scene_cut_failure_is_hard_and_never_crop_waived(self) -> None:
        provider = _provider(
            lambda _source, _seconds: _descriptor(0.5),
            scene_cut_times=((NOMINAL - 1.0 / FPS,), ()),
        )

        zero = provider(_key()).audit_for(0)

        self.assertIs(
            zero.disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertIn(
            "outgoing_source_local_camera_cut_near_boundary",
            zero.failure_codes,
        )

    def test_nonlinear_clock_maps_every_exact_and_side_sample_from_master(self) -> None:
        rate = 1.005
        calls: list[tuple[int, float]] = []

        def descriptor(source: int, seconds: float) -> FrameVisualDescriptor:
            calls.append((source, seconds))
            return _descriptor(0.5)

        provider = DecodedBoundaryOptionProvider(
            sources=(Path("source-a.mp4"), Path("source-b.mp4")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            source_timewarps=(_timewarp(rate), _timewarp(rate)),
            descriptor_cache=BoundaryDescriptorCache(maximum_entries=4096),
            output_fps=FPS,
            descriptor_provider=descriptor,
        )

        provider(_key())

        outgoing_exact = rate * (NOMINAL - 0.5 / FPS)
        outgoing_preceding = rate * (NOMINAL - 3.5 / FPS)
        incoming_exact = rate * (NOMINAL + 0.5 / FPS)
        incoming_following = rate * (NOMINAL + 3.5 / FPS)
        for expected in (
            (0, outgoing_exact),
            (0, outgoing_preceding),
            (1, incoming_exact),
            (1, incoming_following),
        ):
            self.assertTrue(
                any(
                    source == expected[0]
                    and math.isclose(seconds, expected[1], abs_tol=1e-9)
                    for source, seconds in calls
                ),
                expected,
            )

    def test_exact_memo_repeats_with_zero_additional_decoding(self) -> None:
        calls = 0

        def descriptor(_source: int, _seconds: float) -> FrameVisualDescriptor:
            nonlocal calls
            calls += 1
            return _descriptor(0.5)

        provider = _provider(descriptor)
        key = _key()
        first = provider(key)
        first_calls = calls
        first_cache_diagnostics = provider.descriptor_cache.diagnostics()

        second = provider(key)

        self.assertIs(second, first)
        self.assertEqual(calls, first_calls)
        self.assertEqual(
            provider.descriptor_cache.diagnostics(),
            first_cache_diagnostics,
        )
        self.assertEqual(provider.request_count, 2)
        self.assertEqual(provider.measurement_count, 1)
        self.assertEqual(provider.to_dict()["memo_hit_count"], 1)

    def test_audit_payload_never_persists_raw_or_biometric_vectors(self) -> None:
        vector = tuple(math.sin(index * 0.37) for index in range(60))
        descriptor = replace(
            _descriptor(0.5),
            pose_embedding=vector,
            pose_backend=POSE_BACKEND,
            pose_confidence=0.99,
            structure_embedding=tuple(math.cos(index * 0.19) for index in range(64)),
            silhouette_embedding=tuple(
                math.sin(index * 0.13) for index in range(48)
            ),
        )
        provider = _provider(lambda _source, _seconds: descriptor)
        provider(_key())

        payload = provider.to_dict()
        serialized = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "pose_embedding",
            "structure_embedding",
            "silhouette_embedding",
            "formation_people",
            "face_identity_embedding",
            "decoded_pixels",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(payload["raw_frames_persisted"])
        self.assertFalse(payload["biometric_model_vectors_persisted"])

        def assert_no_model_vector(value: object) -> None:
            if isinstance(value, dict):
                for nested in value.values():
                    assert_no_model_vector(nested)
            elif isinstance(value, list):
                if value and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    for item in value
                ):
                    self.assertNotIn(len(value), {48, 60, 64})
                for nested in value:
                    assert_no_model_vector(nested)

        assert_no_model_vector(payload)


if __name__ == "__main__":
    unittest.main()
