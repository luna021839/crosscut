from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.pose import FormationPersonDescriptor, POSE_BACKEND
from crosscut.timeline import ShotType
from crosscut.timeline_review import review_timeline
from crosscut.boundary_repair import BoundaryRepairConfig
from crosscut.stage_edit_temporal_pose_motion import (
    TemporalPoseMotionThresholds,
)
from crosscut.visual_boundary_repair import (
    BoundaryDescriptorCache,
    _passes_production_transition_gates,
    _visual_evidence,
    repair_visual_timeline_boundaries,
)
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualTimelineChoice,
    VisualTimelinePlan,
)


def _descriptor(center: float) -> FrameVisualDescriptor:
    person = FormationPersonDescriptor(
        bbox_xywh=(center - 0.1, 0.2, 0.2, 0.65),
        centroid_xy=(center, 0.5),
        detection_confidence=0.9,
    )
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(center, 1.0 - center),
        silhouette_embedding=(center, 1.0 - center),
        person_layout_embedding=(center, 0.5, 0.2),
        formation_people=(person,),
        detected_person_count=1,
        silhouette_bbox_xywh=(center - 0.1, 0.2, 0.2, 0.65),
        silhouette_centroid_xy=(center, 0.5),
    )


def _pose_descriptor(center: float, angle: float) -> FrameVisualDescriptor:
    embedding = (
        math.cos(angle),
        math.sin(angle),
        *((0.0,) * 58),
    )
    return replace(
        _descriptor(center),
        pose_embedding=embedding,
        pose_backend=POSE_BACKEND,
        pose_confidence=0.95,
    )


def _temporal_thresholds() -> TemporalPoseMotionThresholds:
    return TemporalPoseMotionThresholds(
        minimum_boundary_motion_magnitude=0.5,
        minimum_endpoint_pose_cosine=0.95,
        minimum_velocity_direction_cosine=0.8,
        minimum_phase_compatibility=0.9,
        pose_norm_epsilon=1e-8,
    )


def _choice(index: int, source: int) -> VisualTimelineChoice:
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source,
        canonical_master_start=float(index),
        canonical_master_end=float(index + 1),
        source_local_start=float(index),
        source_local_end=float(index + 1),
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.5,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source}:shot",
        dissolve_exit_face_geometry_window=((0.0,) * 10,) * 3,
        dissolve_entry_face_geometry_window=((0.0,) * 10,) * 3,
    )


class VisualBoundaryRepairTests(unittest.TestCase):
    def test_process_local_descriptor_cache_is_bounded_and_reuses_none(self) -> None:
        cache = BoundaryDescriptorCache(maximum_entries=2)
        first = _descriptor(0.4)
        cache.store(0, 1.0, first)
        cache.store(1, 2.0, None)

        self.assertEqual(cache.lookup(0, 1.0), (True, first))
        self.assertEqual(cache.lookup(1, 2.0), (True, None))
        self.assertEqual(cache.lookup(2, 3.0), (False, None))
        cache.store(2, 3.0, _descriptor(0.6))

        diagnostics = cache.diagnostics()
        self.assertEqual(diagnostics["entry_count"], 2)
        self.assertEqual(diagnostics["hit_count"], 2)
        self.assertEqual(diagnostics["miss_count"], 1)
        self.assertEqual(diagnostics["eviction_count"], 1)
        self.assertFalse(diagnostics["raw_frames_persisted"])
        self.assertFalse(diagnostics["persistent"])

    def test_full_ram_descriptor_cache_hit_opens_no_video_sampler(self) -> None:
        descriptor = _descriptor(0.5)

        class AlwaysHitBoundaryDescriptorCache(BoundaryDescriptorCache):
            def lookup(
                self,
                source_index: int,
                seconds: float,
            ) -> tuple[bool, FrameVisualDescriptor | None]:
                self._key(source_index, seconds)
                self.hit_count += 1
                return True, descriptor

        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        cache = AlwaysHitBoundaryDescriptorCache()
        with patch(
            "crosscut.visual_boundary_repair._VideoSampler"
        ) as sampler:
            result = repair_visual_timeline_boundaries(
                visual_timeline=plan,
                sources=(Path("source-a.mp4"), Path("source-b.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                descriptor_cache=cache,
            )

        sampler.assert_not_called()
        self.assertGreater(cache.hit_count, 0)
        self.assertEqual(result.reviewed_cross_source_count, 1)

    def test_one_source_descriptor_miss_opens_only_that_sampler(self) -> None:
        descriptor = _descriptor(0.5)

        class SourceOneHitBoundaryDescriptorCache(BoundaryDescriptorCache):
            def lookup(
                self,
                source_index: int,
                seconds: float,
            ) -> tuple[bool, FrameVisualDescriptor | None]:
                if source_index == 1:
                    self._key(source_index, seconds)
                    self.hit_count += 1
                    return True, descriptor
                return super().lookup(source_index, seconds)

        opened: list[Path] = []
        closed: list[Path] = []

        class FakeSampler:
            def __init__(
                self,
                path: Path,
                *,
                feature_cache_directory: Path | None = None,
            ) -> None:
                del feature_cache_directory
                self.path = Path(path)
                opened.append(self.path)

            def feature_descriptor_at(
                self,
                seconds: float,
                *,
                include_subject_geometry: bool,
                include_formation_pose: bool,
            ) -> FrameVisualDescriptor:
                del seconds, include_subject_geometry, include_formation_pose
                return descriptor

            def close(self) -> None:
                closed.append(self.path)

        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        cache = SourceOneHitBoundaryDescriptorCache()
        with patch(
            "crosscut.visual_boundary_repair._VideoSampler",
            side_effect=FakeSampler,
        ):
            result = repair_visual_timeline_boundaries(
                visual_timeline=plan,
                sources=(Path("source-a.mp4"), Path("source-b.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                descriptor_cache=cache,
            )

        self.assertEqual(opened, [Path("source-a.mp4")])
        self.assertEqual(closed, opened)
        self.assertEqual(result.reviewed_cross_source_count, 1)

    def test_descriptor_cache_cannot_wrap_test_provider(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        with self.assertRaisesRegex(
            ValueError,
            "cannot wrap an explicit descriptor_provider",
        ):
            repair_visual_timeline_boundaries(
                visual_timeline=plan,
                sources=(Path("a"), Path("b")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                descriptor_provider=lambda _source, _seconds: _descriptor(0.5),
                descriptor_cache=BoundaryDescriptorCache(),
            )

    def test_unbound_temporal_pose_motion_preserves_exact_prior_cost(self) -> None:
        kwargs = {
            "outgoing_preceding": _pose_descriptor(0.5, -0.20),
            "incoming_following": _pose_descriptor(0.5, 0.22),
            "side_sample_span_seconds": 0.10,
        }
        prior = _visual_evidence(
            _pose_descriptor(0.5, 0.0),
            _pose_descriptor(0.5, 0.02),
            **kwargs,
        )
        unbound = _visual_evidence(
            _pose_descriptor(0.5, 0.0),
            _pose_descriptor(0.5, 0.02),
            temporal_pose_motion_thresholds=None,
            temporal_pose_motion_preference_weight=100.0,
            **kwargs,
        )

        self.assertEqual(unbound.total_cost, prior.total_cost)
        self.assertEqual(unbound.base_total_cost, prior.base_total_cost)
        self.assertEqual(
            unbound.temporal_pose_motion.classification,
            "calibration_unbound",
        )
        self.assertFalse(
            unbound.temporal_pose_motion.direction_evidence_available
        )
        self.assertFalse(
            unbound.temporal_pose_motion.dynamic_direction_match_proven
        )
        self.assertEqual(
            unbound.temporal_pose_motion.soft_preference_applied,
            0.0,
        )

    def test_arbitrary_threshold_activation_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "immutable evidence loader/registry",
        ):
            _visual_evidence(
                _pose_descriptor(0.5, 0.0),
                _pose_descriptor(0.5, 0.02),
                outgoing_preceding=_pose_descriptor(0.5, -0.20),
                incoming_following=_pose_descriptor(0.5, 0.22),
                side_sample_span_seconds=0.10,
                temporal_pose_motion_thresholds=_temporal_thresholds(),
                temporal_pose_motion_preference_weight=1.0,
            )

    def test_repair_rejects_arbitrary_threshold_activation(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        with self.assertRaisesRegex(
            ValueError,
            "immutable evidence loader/registry",
        ):
            repair_visual_timeline_boundaries(
                visual_timeline=plan,
                sources=(Path("a"), Path("b")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                config=BoundaryRepairConfig(
                    temporal_pose_motion_thresholds=_temporal_thresholds(),
                    temporal_pose_motion_preference_weight=1.0,
                ),
            )

    def test_missing_real_60d_pose_stays_neutral_while_calibration_unbound(self) -> None:
        missing = _visual_evidence(
            _pose_descriptor(0.5, 0.0),
            _pose_descriptor(0.5, 0.02),
            outgoing_preceding=_descriptor(0.5),
            incoming_following=_pose_descriptor(0.5, 0.22),
            side_sample_span_seconds=0.10,
            temporal_pose_motion_thresholds=None,
            temporal_pose_motion_preference_weight=10.0,
        )

        motion = missing.temporal_pose_motion
        self.assertFalse(motion.analysis_executed)
        self.assertFalse(motion.input_valid)
        self.assertFalse(motion.qualifies)
        self.assertEqual(motion.classification, "calibration_unbound")
        self.assertIn(
            "temporal_pose_motion_thresholds_unbound",
            motion.failure_reasons,
        )
        self.assertFalse(motion.direction_evidence_available)
        self.assertFalse(motion.dynamic_direction_match_proven)
        self.assertEqual(motion.soft_preference_applied, 0.0)
        self.assertEqual(missing.total_cost, missing.base_total_cost)
        # Missing direction evidence is neutral to the pre-existing geometry
        # and professional transition gates.
        self.assertTrue(_passes_production_transition_gates(missing))

    def test_temporal_pose_motion_serialization_is_scalar_only_and_covered(
        self,
    ) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        def provider(_source: int, seconds: float) -> FrameVisualDescriptor:
            return _pose_descriptor(0.5, 0.8 * seconds)

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=provider,
            config=BoundaryRepairConfig(
                temporal_pose_motion_thresholds=None,
                temporal_pose_motion_preference_weight=1.0,
            ),
        )

        payload = repaired.to_dict()
        configuration = payload["temporal_pose_motion_configuration"]
        self.assertEqual(
            configuration["calibration_status"],
            "calibration_unbound",
        )
        self.assertEqual(configuration["preference_weight"], 1.0)
        self.assertIsNone(configuration["threshold_identity_sha256"])
        motion_coverage = payload["evidence_coverage"][
            "temporal_pose_motion"
        ]
        self.assertEqual(
            motion_coverage["dynamic_pose_phase_match_count"],
            0,
        )
        motion = payload["repairs"][0]["selected_visual_evidence"][
            "temporal_pose_motion"
        ]
        self.assertIsNone(motion["feature_dimension"])
        self.assertEqual(motion["classification"], "calibration_unbound")
        self.assertEqual(motion["soft_preference_applied"], 0.0)
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("normalized_velocity", serialized)
        self.assertNotIn("pose_embedding", serialized)
        self.assertNotIn("embeddings", serialized)

        def assert_no_raw_numeric_vectors(value: object) -> None:
            if isinstance(value, dict):
                for nested in value.values():
                    assert_no_raw_numeric_vectors(nested)
            elif isinstance(value, list):
                if value and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    for item in value
                ):
                    self.assertNotIn(len(value), {60, 120})
                for nested in value:
                    assert_no_raw_numeric_vectors(nested)

        assert_no_raw_numeric_vectors(payload)

    def test_hog_structure_is_never_serialized_or_gated_as_pose(self) -> None:
        evidence = _visual_evidence(
            _descriptor(0.48),
            _descriptor(0.52),
            outgoing_preceding=_descriptor(0.48),
            incoming_following=_descriptor(0.52),
            side_sample_span_seconds=3.0 / 29.97,
        )

        self.assertEqual(evidence.pose_basis, "actual_pose_pair_unavailable")
        self.assertIsNone(evidence.pose_mismatch)
        self.assertIsNone(evidence.exact_pose_mismatch)
        self.assertIsNotNone(evidence.structure_mismatch)
        self.assertTrue(evidence.evidence_coverage.professional_gate_passed)
        self.assertTrue(_passes_production_transition_gates(evidence))

        for forged_backend in (
            "unknown_pose_backend",
            f"{POSE_BACKEND}_forged_suffix",
        ):
            with self.subTest(pose_backend=forged_backend):
                forged = replace(
                    _descriptor(0.50),
                    pose_embedding=(0.1,) * 60,
                    pose_backend=forged_backend,
                )
                evidence = _visual_evidence(forged, forged)
                self.assertEqual(
                    evidence.pose_basis,
                    "actual_pose_pair_unavailable",
                )
                self.assertIsNone(evidence.pose_mismatch)

    def test_nearest_interior_actual_pose_is_separate_from_exact_frame(self) -> None:
        outgoing = _descriptor(0.50)
        incoming = replace(
            _descriptor(0.51),
            pose_embedding=(0.20,) * 60,
            pose_backend="opencv_zoo_mediapipe_pose_2023mar",
            pose_confidence=0.9,
        )
        outgoing_previous = replace(
            outgoing,
            pose_embedding=(0.10,) * 60,
            pose_backend="opencv_zoo_mediapipe_pose_2023mar",
            pose_confidence=0.9,
        )

        evidence = _visual_evidence(
            outgoing,
            incoming,
            outgoing_preceding=outgoing,
            incoming_following=incoming,
            outgoing_temporal_pose_candidates=((-1, outgoing_previous),),
            output_fps=29.97,
            side_sample_span_seconds=3.0 / 29.97,
        )

        self.assertIsNone(evidence.exact_pose_mismatch)
        self.assertAlmostEqual(
            evidence.nearest_temporal_pose_mismatch or 0.0,
            0.10,
        )
        self.assertEqual(
            evidence.pose_basis,
            "nearest_interior_temporal_opencv_zoo_mediapipe_pose",
        )
        self.assertEqual(
            evidence.evidence_coverage.exact_pose_available,
            (False, True),
        )
        self.assertEqual(
            evidence.evidence_coverage.temporal_pose_offsets_frames,
            (-1, 0),
        )
        self.assertTrue(_passes_production_transition_gates(evidence))

    def test_combined_professional_gate_fails_closed_on_structure_alone(self) -> None:
        descriptor = replace(
            _descriptor(0.50),
            person_layout_embedding=(),
            formation_people=(),
            silhouette_embedding=(),
            silhouette_bbox_xywh=None,
            silhouette_centroid_xy=None,
        )
        evidence = _visual_evidence(
            descriptor,
            descriptor,
            outgoing_preceding=descriptor,
            incoming_following=descriptor,
            side_sample_span_seconds=3.0 / 29.97,
        )

        self.assertEqual(
            evidence.evidence_coverage.professional_evidence_basis,
            "insufficient_combined_geometry_evidence",
        )
        self.assertFalse(evidence.evidence_coverage.professional_gate_passed)
        self.assertFalse(_passes_production_transition_gates(evidence))

    def test_plan_serializes_pose_coverage_and_geometry_provider_health(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=lambda _source, _seconds: _descriptor(0.5),
        )

        payload = repaired.to_dict()
        coverage = payload["evidence_coverage"]
        self.assertEqual(payload["geometry_provider_failure_count"], 0)
        self.assertGreater(payload["geometry_provider_attempt_count"], 0)
        self.assertEqual(coverage["exact_actual_pose_pair_count"], 0)
        self.assertEqual(coverage["hog_used_as_pose_count"], 0)
        self.assertEqual(coverage["professional_gate_failure_count"], 0)
        self.assertEqual(
            payload["repairs"][0]["selected_visual_evidence"]["pose_basis"],
            "actual_pose_pair_unavailable",
        )

    def test_geometry_provider_decode_failure_is_audited(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=lambda _source, _seconds: None,
        )

        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertGreater(repaired.geometry_provider_failure_count, 0)
        self.assertEqual(len(repaired.failures), 1)

    def test_selects_best_frame_and_updates_both_clocks(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 4.0)

        def provider(source: int, seconds: float) -> FrameVisualDescriptor:
            frame_offset = round((seconds - (1.0 + (-0.5 if source == 0 else 0.5) / 29.97)) * 29.97)
            if source == 0:
                return _descriptor(0.4)
            # The incoming-side far sample is three frames after its near
            # sample. Only the +2 candidate therefore has two compatible
            # boundary-side frames; the -1 candidate sees 0.8 -> 0.4.
            return _descriptor(0.4 if frame_offset in (2, 5) else 0.8)

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.changed_count, 1)
        self.assertEqual(repaired.repairs[0].result.selected_frame_offset, 2)
        boundary = 1.0 + 2.0 / 29.97
        self.assertAlmostEqual(repaired.timeline.choices[0].canonical_master_end, boundary)
        self.assertAlmostEqual(repaired.timeline.choices[1].canonical_master_start, boundary)
        self.assertIsNotNone(
            repaired.timeline.choices[0].exit_boundary_side_stability
        )
        self.assertIsNotNone(
            repaired.timeline.choices[1].entry_boundary_side_stability
        )
        serialized = repaired.timeline.to_dict()["choices"]
        self.assertIsNotNone(serialized[0]["exit_boundary_side_stability"])
        self.assertIsNotNone(serialized[1]["entry_boundary_side_stability"])
        self.assertFalse(
            repaired.timeline.choices[0].dissolve_exit_face_geometry_window
        )

    def test_same_source_boundary_is_not_retimed(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 0)), 2.0)
        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"),),
            audio_offsets=(0.0,),
            scene_cut_times=((),),
            descriptor_provider=lambda _source, _seconds: _descriptor(0.5),
        )
        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(repaired.skipped_same_source_boundaries, 1)
        self.assertEqual(repaired.timeline.choices, plan.choices)

    def test_proxy_partial_reedit_only_touches_requested_failed_boundary(self) -> None:
        plan = VisualTimelinePlan(
            (_choice(0, 0), _choice(1, 1), _choice(2, 0)),
            3.0,
        )
        fps = 29.97

        def provider(source: int, seconds: float) -> FrameVisualDescriptor:
            # Both sides are compatible everywhere; a scene cut immediately
            # before the second nominal boundary makes +8 frames the safest
            # requested move.  Boundary zero must remain byte-for-byte stable.
            return _descriptor(0.5 if source == 0 else 0.51)

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), (2.0 - 1.0 / fps,)),
            output_fps=fps,
            descriptor_provider=provider,
            boundary_indices=(1,),
            config=BoundaryRepairConfig(
                search_radius_frames=8,
                interior_scene_cut_guard_frames=9,
            ),
        )

        self.assertEqual(repaired.requested_boundary_indices, (1,))
        self.assertEqual(
            repaired.timeline.choices[0],
            plan.choices[0],
        )
        self.assertEqual(repaired.reviewed_cross_source_count, 1)
        self.assertEqual(
            repaired.to_dict()["requested_boundary_indices"],
            [1],
        )

    def test_proxy_partial_reedit_rejects_duplicate_or_invalid_boundaries(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        kwargs = dict(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=lambda _source, _seconds: _descriptor(0.5),
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            repair_visual_timeline_boundaries(
                **kwargs,
                boundary_indices=(0, 0),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            repair_visual_timeline_boundaries(
                **kwargs,
                boundary_indices=(1,),
            )

    def test_visible_source_cut_takes_priority_over_lower_visual_cost(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        def provider(source: int, seconds: float) -> FrameVisualDescriptor:
            return _descriptor(0.5 if source == 0 else 0.52)

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((1.0 - 1.0 / 29.97,), ()),
            descriptor_provider=provider,
        )
        self.assertLessEqual(
            repaired.repairs[0].result.selected.interior_scene_cut_count,
            repaired.repairs[0].result.nominal_candidate.interior_scene_cut_count,
        )

    def test_local_repair_cannot_promote_a_hard_gate_layout_mismatch(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        def provider(source: int, _seconds: float) -> FrameVisualDescriptor:
            descriptor = _descriptor(0.5)
            if source == 1:
                return replace(
                    descriptor,
                    person_layout_embedding=(1.0, 1.0, 1.0),
                )
            return descriptor

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(len(repaired.failures), 1)
        self.assertEqual(repaired.timeline.choices, plan.choices)

    def test_local_repair_uses_the_reviewers_persisted_layout_metric(self) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        def provider(source: int, _seconds: float) -> FrameVisualDescriptor:
            descriptor = _descriptor(0.5)
            if source == 1:
                # Keep the transient layout embedding identical so this only
                # exercises the silhouette box/centroid metric persisted in
                # VisualTimelineChoice and recomputed by timeline_review.
                return replace(
                    descriptor,
                    silhouette_bbox_xywh=(0.72, 0.2, 0.2, 0.65),
                    silhouette_centroid_xy=(0.82, 0.5),
                )
            return descriptor

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(len(repaired.failures), 1)
        self.assertEqual(repaired.timeline.choices, plan.choices)

    def test_local_repair_persists_bbox_fallback_when_embeddings_are_missing(
        self,
    ) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)

        def provider(source: int, _seconds: float) -> FrameVisualDescriptor:
            return replace(
                _descriptor(0.46 if source == 0 else 0.52),
                person_layout_embedding=(),
            )

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.reviewed_cross_source_count, 1)
        self.assertFalse(repaired.failures)
        persisted = (
            repaired.timeline.choices[1]
            .layout_transition_mismatch_from_previous
        )
        reviewed = review_timeline(repaired.timeline).boundaries[0].layout_mismatch
        self.assertIsNotNone(persisted)
        self.assertAlmostEqual(persisted or 0.0, reviewed or 0.0)
        self.assertLess(persisted or 1.0, 0.32)

    def test_scene_safe_original_is_kept_when_only_repair_candidate_adds_risk(
        self,
    ) -> None:
        plan = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 2.0)
        fps = 29.97
        frame_seconds = 1.0 / fps

        def provider(source: int, seconds: float) -> FrameVisualDescriptor | None:
            boundary = (
                seconds + frame_seconds / 2.0
                if source == 0
                else seconds - frame_seconds / 2.0
            )
            frame_offset = round((boundary - 1.0) * fps)
            # Near samples map to +3. Their same-side samples map to 0 on the
            # outgoing source and +6 on the incoming source.
            return _descriptor(0.5) if frame_offset in (0, 3, 6) else None

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            # The only decodable repair candidate at +3 frames exposes this
            # outgoing cut. The selected timeline's original flags are safe.
            scene_cut_times=((1.0,), ()),
            output_fps=fps,
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.timeline.choices, plan.choices)
        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(len(repaired.failures), 1)
        self.assertEqual(
            repaired.failures[0]["reason"],
            "selected_scene_cut_risk_exceeds_original",
        )
        self.assertFalse(review_timeline(repaired.timeline).boundaries[0].scene_cut_risk)

    def test_boundary_update_is_rejected_when_it_adds_a_timeline_hard_failure(
        self,
    ) -> None:
        outgoing = replace(
            _choice(0, 0),
            canonical_master_end=7.15,
            source_local_end=7.15,
        )
        incoming = replace(
            _choice(1, 1),
            canonical_master_start=7.15,
            canonical_master_end=10.0,
            source_local_start=7.15,
            source_local_end=10.0,
        )
        plan = VisualTimelinePlan((outgoing, incoming), 2.0)
        fps = 29.97
        frame_seconds = 1.0 / fps

        def provider(source: int, seconds: float) -> FrameVisualDescriptor | None:
            boundary = (
                seconds + frame_seconds / 2.0
                if source == 0
                else seconds - frame_seconds / 2.0
            )
            frame_offset = round((boundary - 7.15) * fps)
            return _descriptor(0.5) if frame_offset in (0, 3, 6) else None

        before = review_timeline(plan)
        self.assertNotIn(
            "source_run_hard_max_exceeded",
            {failure.code for failure in before.hard_failures},
        )
        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.0, 0.0),
            scene_cut_times=((), ()),
            output_fps=fps,
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.timeline.choices, plan.choices)
        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(
            repaired.failures[0]["reason"],
            "repair_would_introduce_hard_timeline_failure",
        )
        introduced = repaired.failures[0]["introduced_hard_failures"]
        self.assertIn(
            "source_run_hard_max_exceeded",
            {failure["code"] for failure in introduced},
        )

    def test_one_frame_outgoing_camera_flash_is_not_a_repair_candidate(
        self,
    ) -> None:
        outgoing = replace(
            _choice(0, 9),
            entry_shot_scale=0.1631326356,
            exit_shot_scale=0.6768598338,
            entry_shot_type=ShotType.GROUP_WIDE,
            exit_shot_type=ShotType.CLOSE_UP,
        )
        incoming = replace(
            _choice(1, 3),
            entry_shot_scale=0.584663,
            entry_shot_type=ShotType.CLOSE_UP,
        )
        plan = VisualTimelinePlan((outgoing, incoming), 2.0)
        source_calls = {9: 0, 3: 0}

        def provider(source: int, _seconds: float) -> FrameVisualDescriptor:
            source_calls[source] += 1
            if source == 9:
                # Repair samples outgoing near then three frames earlier. The
                # near frame is the real 32.041s close-up flash while the
                # preceding group-wide frame remains at scale .163.
                near = source_calls[source] % 2 == 1
                return replace(
                    _descriptor(0.5),
                    shot_scale=0.6768598338 if near else 0.1631326356,
                    shot_type=ShotType.CLOSE_UP if near else ShotType.GROUP_WIDE,
                    structure_embedding=((1.0, 1.0) if near else (0.0, 0.0)),
                )
            return replace(
                _descriptor(0.5),
                shot_scale=0.584663,
                shot_type=ShotType.CLOSE_UP,
                structure_embedding=(0.5, 0.5),
            )

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("0"),) * 10,
            audio_offsets=(0.0,) * 10,
            scene_cut_times=((),) * 10,
            descriptor_provider=provider,
        )

        self.assertEqual(repaired.reviewed_cross_source_count, 0)
        self.assertEqual(repaired.timeline.choices, plan.choices)
        self.assertEqual(len(repaired.failures), 1)
        self.assertIn(
            "no boundary candidate preserves",
            repaired.failures[0]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
