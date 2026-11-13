from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.direct_member_editorial_motion import (
    DirectMemberDenseFaceProvider,
    DirectMemberEditorialMotionError,
    DirectMemberEditorialMotionConfig,
    DirectMemberFaceSample,
    build_direct_member_editorial_motion_plan,
)
from crosscut.direct_member_transform import (
    DirectMemberBoundaryTransformRepair,
    direct_member_source_frame_geometry,
)
from crosscut.direct_member_target_face import (
    FOCAL_FACE_BASIS,
    FocalFaceCandidate,
)
from crosscut.pose import SILHOUETTE_BACKEND
from crosscut.render import RenderSegment, RenderTransition, TransitionKind
from crosscut.visual_feature_cache import CanonicalFrameIdentity


_FACE_BASIS = "unambiguous_dominant_face_bbox_geometry_only"


class DirectMemberEditorialMotionTests(unittest.TestCase):
    @staticmethod
    def _audited_sample(
        box: tuple[float, float, float, float],
        *,
        basis: str = _FACE_BASIS,
        face_count: int = 1,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            attention=SimpleNamespace(bbox_xywh=box, basis=basis),
            descriptor=SimpleNamespace(face_count=face_count),
        )

    @classmethod
    def _selection(
        cls,
        source_indices: tuple[int, ...],
        boxes: tuple[tuple[float, float, float, float], ...],
        *,
        boundary_overrides: dict[int, dict[str, object]] | None = None,
    ) -> SimpleNamespace:
        overrides = boundary_overrides or {}
        boundaries = []
        for index, (left_source, right_source) in enumerate(
            zip(source_indices, source_indices[1:])
        ):
            values: dict[str, object] = {
                "boundary_index": index,
                "outgoing_eligible_source_index": left_source,
                "incoming_eligible_source_index": right_source,
                "outgoing_sample": cls._audited_sample(boxes[index]),
                "incoming_sample": cls._audited_sample(boxes[index + 1]),
                "mismatch": SimpleNamespace(transform_repair=None),
            }
            values.update(overrides.get(index, {}))
            boundaries.append(SimpleNamespace(**values))
        return SimpleNamespace(
            source_indices=source_indices,
            boundaries=tuple(boundaries),
        )

    @staticmethod
    def _renders(
        source_indices: tuple[int, ...],
        durations: tuple[float, ...],
    ) -> tuple[RenderSegment, ...]:
        cursor = 0.0
        output = []
        for source_index, duration in zip(
            source_indices, durations, strict=True
        ):
            output.append(
                RenderSegment(
                    source_index=source_index,
                    source_start=cursor,
                    duration=duration,
                    master_start=cursor,
                )
            )
            cursor += duration
        return tuple(output)

    def test_registered_reference_temporally_bridges_exact_face(self) -> None:
        outgoing_reference = FocalFaceCandidate(
            (0.15, 0.10, 0.10, 0.10),
            (1.0, 0.0),
        )
        incoming_reference = FocalFaceCandidate(
            (0.60, 0.10, 0.10, 0.10),
            (1.0, 0.0),
        )
        temporal_bridge = FocalFaceCandidate(
            (0.48, 0.10, 0.10, 0.10),
            (0.573576, 0.819152),
        )
        exact_target = FocalFaceCandidate(
            (0.42, 0.10, 0.10, 0.10),
            (-0.342020, 0.939693),
        )
        exact_distractor = FocalFaceCandidate(
            (0.82, 0.10, 0.10, 0.10),
            (-1.0, 0.0),
        )
        provider = object.__new__(DirectMemberDenseFaceProvider)
        provider._sources = (Path("left.mp4"), Path("right.mp4"))
        provider._focal_references = [
            [outgoing_reference],
            [incoming_reference],
        ]

        def candidates(source_index: int, seconds: float):
            if source_index == 0:
                return (outgoing_reference,)
            if abs(seconds - 10.0) < 1e-6:
                return (exact_target, exact_distractor)
            if abs(seconds - 10.1) < 1e-6:
                return (temporal_bridge,)
            return ()

        provider._focal_candidates = candidates
        recovered = (
            provider.recover_exact_focal_pair_from_registered_references(
                0,
                10.0,
                1,
                10.0,
            )
        )

        self.assertEqual(
            recovered,
            (outgoing_reference, exact_target),
        )
        self.assertIn(exact_target, provider._focal_references[1])

    def test_temporal_bridge_never_fabricates_missing_exact_face(self) -> None:
        reference = FocalFaceCandidate(
            (0.20, 0.10, 0.10, 0.10),
            (1.0, 0.0),
        )
        provider = object.__new__(DirectMemberDenseFaceProvider)
        provider._sources = (Path("left.mp4"), Path("right.mp4"))
        provider._focal_references = [[reference], [reference]]
        provider._focal_candidates = lambda source, seconds: (
            (reference,)
            if source == 0 or seconds > 10.0 + 1e-6
            else ()
        )

        self.assertIsNone(
            provider.recover_exact_focal_pair_from_registered_references(
                0,
                10.0,
                1,
                10.0,
            )
        )

    @staticmethod
    def _sample(
        box: tuple[float, float, float, float],
        *,
        subject_box: tuple[float, float, float, float] | None = None,
    ) -> DirectMemberFaceSample:
        return DirectMemberFaceSample(
            box,
            1,
            "yunet",
            subject_box,
            0.95 if subject_box is not None else 0.0,
            (
                "pphumanseg_full_subject_silhouette"
                if subject_box is not None
                else "none"
            ),
        )

    @classmethod
    def _basic_inputs(cls, *, duration: float = 8.0):
        source_indices = (0, 1)
        boxes = (
            (0.35, 0.225, 0.20, 0.55),
            (0.40, 0.200, 0.20, 0.60),
        )
        return (
            cls._selection(source_indices, boxes),
            cls._renders(source_indices, (duration, duration)),
            boxes,
        )

    def test_safe_dense_single_face_match_is_accepted(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertEqual(plan.zoom_event_count, 1)
        self.assertEqual(plan.roll_event_count, 0)
        self.assertIsNotNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertGreater(plan.track_for(0).final_keyframe.zoom, 1.05)
        self.assertEqual(
            plan.analysis_basis,
            "official_target_fancam_face_or_verified_full_subject_dense_"
            "output_frame_boundary_anchored_subject_relative_face_track_"
            "adaptive_review_v10",
        )

    def test_required_equal_scale_offset_uses_bilateral_camera_match(self) -> None:
        boxes = (
            (0.3750, 0.25, 0.15, 0.40),
            (0.4217, 0.25, 0.15, 0.40),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "mismatch": SimpleNamespace(
                        transform_repair=None,
                        dense_attention_validation_required=True,
                    )
                }
            },
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (8.0, 8.0)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source],
                subject_box=(0.15, 0.02, 0.70, 0.90),
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertIsNotNone(plan.track_for(0))
        self.assertIsNotNone(plan.track_for(1))
        self.assertEqual(plan.zoom_event_count, 2)
        boundary_index, position, scale = (
            plan.boundary_residual_attention_mismatches[0]
        )
        self.assertEqual(boundary_index, 0)
        self.assertAlmostEqual(position, 0.0, places=7)
        self.assertAlmostEqual(scale, 0.0, places=7)

    def test_required_boundary_already_matched_is_natural_success(self) -> None:
        boxes = (
            (0.390, 0.220, 0.20, 0.56),
            (0.398, 0.218, 0.20, 0.56),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "mismatch": SimpleNamespace(
                        transform_repair=None,
                        dense_attention_validation_required=True,
                    )
                }
            },
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (8.0, 8.0)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source]
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertEqual(plan.natural_match_boundary_indices, (0,))
        self.assertEqual(plan.boundary_match_count, 0)
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))

    def test_stored_geometry_keeps_omitted_trailing_source_slots(self) -> None:
        selection, renders, boxes = self._basic_inputs()
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_counts = (1, 1, 0)
        selection.source_frame_geometries = (
            geometry,
            geometry,
            geometry,
        )

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)

    def test_paired_focal_face_provider_overrides_coarse_person_geometry(self) -> None:
        coarse_boxes = (
            (0.10, 0.10, 0.70, 0.80),
            (0.20, 0.05, 0.65, 0.90),
        )
        focal_boxes = (
            (0.35, 0.35, 0.12, 0.12),
            (0.42, 0.30, 0.16, 0.16),
        )
        selection = self._selection((0, 1), coarse_boxes)
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_frame_geometries = (geometry, geometry)

        class PairedFocalProvider:
            def __init__(self) -> None:
                self.paired_call_count = 0
                self.scalar_call_count = 0

            @staticmethod
            def sample(source: int) -> DirectMemberFaceSample:
                return DirectMemberFaceSample(
                    focal_boxes[source],
                    1,
                    "yunet",
                    (0.20, 0.05, 0.60, 0.90),
                    0.95,
                    "pphumanseg_full_subject_silhouette",
                    FOCAL_FACE_BASIS,
                )

            def paired_samples(
                self,
                left_source: int,
                _left_time: float,
                right_source: int,
                _right_time: float,
            ) -> tuple[DirectMemberFaceSample, DirectMemberFaceSample]:
                self.paired_call_count += 1
                return (self.sample(left_source), self.sample(right_source))

            def __call__(
                self,
                source: int,
                _time: float,
            ) -> DirectMemberFaceSample:
                self.scalar_call_count += 1
                return self.sample(source)

        provider = PairedFocalProvider()
        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (8.0, 8.0)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=provider,
            fps=30.0,
        )

        self.assertEqual(provider.paired_call_count, 1)
        self.assertGreater(provider.scalar_call_count, 0)
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertEqual(plan.boundary_residual_attention_mismatches[0][0], 0)
        self.assertLessEqual(
            plan.boundary_residual_attention_mismatches[0][1],
            0.18,
        )
        self.assertLessEqual(
            plan.boundary_residual_attention_mismatches[0][2],
            0.14,
        )

    def test_verified_full_subject_match_is_accepted_without_face(self) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )

        def audited(box):
            return SimpleNamespace(
                attention=SimpleNamespace(
                    bbox_xywh=box,
                    basis="aggregate_human_silhouette",
                ),
                descriptor=SimpleNamespace(
                    face_count=0,
                    silhouette_backend=SILHOUETTE_BACKEND,
                    silhouette_confidence=0.95,
                ),
            )

        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "outgoing_sample": audited(boxes[0]),
                    "incoming_sample": audited(boxes[1]),
                }
            },
        )
        renders = self._renders((0, 1), (8.0, 8.0))

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: DirectMemberFaceSample(
                boxes[source],
                0,
                "none",
                boxes[source],
                0.95,
                "pphumanseg_full_subject_silhouette",
            ),
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertIsNotNone(plan.track_for(0))

    def test_provisional_edge_can_establish_common_basis_from_dense_endpoints(
        self,
    ) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )
        ambiguous = SimpleNamespace(attention=None, descriptor=SimpleNamespace())
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "outgoing_sample": ambiguous,
                    "incoming_sample": ambiguous,
                    "mismatch": SimpleNamespace(
                        transform_repair=None,
                        dense_attention_validation_required=True,
                    ),
                }
            },
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (8.0, 8.0)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: DirectMemberFaceSample(
                boxes[source],
                1 if source == 0 else 0,
                "yunet" if source == 0 else "none",
                boxes[source],
                0.95,
                "pphumanseg_full_subject_silhouette",
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertEqual(plan.boundary_match_count, 1)

    def test_direct_defaults_expand_only_to_renderer_safe_reference_bounds(self) -> None:
        config = DirectMemberEditorialMotionConfig()

        self.assertEqual(config.maximum_boundary_zoom, 1.50)
        self.assertEqual(config.maximum_boundary_upscale_ratio, 1.15)
        self.assertEqual(config.maximum_center_shift, 0.16)
        self.assertEqual(config.maximum_face_residual_position_mismatch, 0.085)
        self.assertEqual(config.maximum_face_residual_scale_mismatch, 0.065)
        self.assertEqual(config.minimum_dense_face_visible_fraction, 0.98)
        self.assertEqual(config.minimum_widened_subject_visible_fraction, 0.93)
        self.assertEqual(
            config.minimum_widened_subject_vertical_visible_fraction,
            0.98,
        )
        self.assertEqual(config.maximum_dense_center_step, 0.15)
        self.assertEqual(config.maximum_dense_subject_center_step, 0.20)
        self.assertEqual(config.maximum_dense_follow_zoom_correction, 0.08)
        self.assertEqual(config.maximum_dense_face_follow_zoom_correction, 0.03)
        self.assertEqual(config.maximum_dense_follow_center_correction, 0.12)
        self.assertEqual(config.maximum_dense_follow_zoom_step, 0.04)
        self.assertEqual(config.maximum_dense_follow_center_step, 0.05)
        self.assertEqual(
            config.minimum_face_associated_subject_visible_fraction,
            0.72,
        )
        self.assertEqual(config.legacy_face_only_maximum_zoom, 1.12)
        self.assertEqual(config.legacy_face_only_maximum_center_shift, 0.025)
        self.assertEqual(config.target_events_per_minute, 12.0)
        self.assertEqual(config.minimum_event_spacing_seconds, 2.0)
        self.assertFalse(config.enable_interior_zoom_pulses)
        self.assertFalse(config.allow_partial_required_boundary_matches)
        self.assertEqual(config.target_rich_motion_events_per_minute, 4.0)
        self.assertEqual(config.interior_pulse_zoom, 1.05)

    def test_boundary_upscale_budget_is_strictly_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "upscale ratio"):
            DirectMemberEditorialMotionConfig(
                maximum_boundary_upscale_ratio=1.151,
            )
        with self.assertRaisesRegex(ValueError, "upscale ratio"):
            DirectMemberEditorialMotionConfig(
                maximum_boundary_upscale_ratio=0.999,
            )

    def test_1080p_required_match_uses_bounded_upscale_only_as_fallback(
        self,
    ) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.22, 0.18, 0.56),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "mismatch": SimpleNamespace(
                        transform_repair=None,
                        dense_attention_validation_required=True,
                    )
                }
            },
        )
        geometry = direct_member_source_frame_geometry((1080, 1920))
        self.assertEqual(geometry.lossless_zoom_capacity, 1.0)
        selection.source_frame_geometries = (geometry, geometry)
        renders = self._renders((0, 1), (8.0, 8.0))

        with self.assertRaises(DirectMemberEditorialMotionError):
            build_direct_member_editorial_motion_plan(
                visual_selection=selection,
                render_segments=renders,
                transitions=(RenderTransition(),),
                common_master_start=0.0,
                geometry_provider=lambda source, _time: self._sample(
                    boxes[source]
                ),
                fps=30.0,
                config=DirectMemberEditorialMotionConfig(
                    maximum_boundary_upscale_ratio=1.0,
                ),
            )

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source]
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertIn(
            "official_fancam_dense_subject_bounded_perceptual_upscale_match",
            plan.segments[0].reasons + plan.segments[1].reasons,
        )
        applied_zooms = tuple(
            keyframe.zoom
            for segment in plan.segments
            if segment.track is not None
            for keyframe in segment.track.keyframes
        )
        self.assertTrue(applied_zooms)
        self.assertLessEqual(max(applied_zooms), 1.15 + 1e-9)

    def test_direct_residual_gates_must_preserve_face_before_subject_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "position residual"):
            DirectMemberEditorialMotionConfig(
                maximum_face_residual_position_mismatch=0.13,
                maximum_subject_residual_position_mismatch=0.12,
            )
        with self.assertRaisesRegex(ValueError, "scale residual"):
            DirectMemberEditorialMotionConfig(
                maximum_face_residual_scale_mismatch=0.11,
                maximum_subject_residual_scale_mismatch=0.10,
            )

    def test_enabled_interior_pulse_fills_sparse_motion_with_lossless_headroom(
        self,
    ) -> None:
        source_indices = (0, 1, 2)
        boxes = (
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.20, 0.20, 0.60),
        )
        selection = self._selection(source_indices, boxes)
        geometry = direct_member_source_frame_geometry((2160, 3840))
        selection.source_frame_geometries = (geometry, geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders(source_indices, (5.0, 5.0, 5.0)),
            transitions=(RenderTransition(), RenderTransition()),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                enable_interior_zoom_pulses=True,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 0)
        self.assertEqual(plan.interior_zoom_pulse_count, 1)
        self.assertEqual(plan.applied_count, 1)
        track = plan.track_for(0)
        self.assertIsNotNone(track)
        assert track is not None
        self.assertEqual(len(track.keyframes), 3)
        self.assertEqual(track.keyframes[0].zoom, 1.0)
        self.assertAlmostEqual(track.keyframes[1].zoom, 1.05)
        self.assertEqual(track.keyframes[-1].zoom, 1.0)

    def test_interior_pulse_never_upscales_native_resolution_source(self) -> None:
        source_indices = (0, 1, 2)
        boxes = (
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.20, 0.20, 0.60),
        )
        selection = self._selection(source_indices, boxes)
        geometry = direct_member_source_frame_geometry((1080, 1920))
        selection.source_frame_geometries = (geometry, geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders(source_indices, (5.0, 5.0, 5.0)),
            transitions=(RenderTransition(), RenderTransition()),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                enable_interior_zoom_pulses=True,
            ),
        )

        self.assertEqual(plan.interior_zoom_pulse_count, 0)
        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("interior_pulse_no_lossless_zoom_headroom", 3),
            plan.skipped_boundary_reasons,
        )

    def test_required_lossless_repair_bypasses_decorative_density_cap(self) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="outgoing",
            transformed_source_index=0,
            zoom=1.20,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.12,
            raw_attention_scale_mismatch=0.40,
            residual_attention_position_mismatch=0.0,
            residual_attention_scale_mismatch=0.0,
            baseline_face_geometry_mismatch=0.20,
            transformed_face_geometry_mismatch=0.0,
            improvement_fraction=1.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=("attention_scale",),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {
                    "mismatch": SimpleNamespace(transform_repair=repair),
                }
            },
        )
        selection.source_frame_geometries = (geometry, geometry)
        renders = self._renders((0, 1), (1.5, 1.5))

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source],
                subject_box=(0.30, 0.10, 0.40, 0.75),
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertEqual(plan.boundary_match_count, 1)
        self.assertGreater(plan.track_for(0).final_keyframe.zoom, 1.18)

    def test_required_repair_records_dense_natural_match_without_forced_crop(
        self,
    ) -> None:
        boxes = (
            (0.00, 0.000, 1.00, 0.550),
            (0.03, 0.030, 0.94, 0.561),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="outgoing",
            transformed_source_index=0,
            zoom=1.02,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.0355,
            raw_attention_scale_mismatch=0.02,
            residual_attention_position_mismatch=0.0355,
            residual_attention_scale_mismatch=0.02,
            baseline_face_geometry_mismatch=0.05,
            transformed_face_geometry_mismatch=0.05,
            improvement_fraction=0.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=(),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {"mismatch": SimpleNamespace(transform_repair=repair)}
            },
        )
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (2.0, 2.0)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source],
                subject_box=(0.00, 0.00, 1.00, 1.00),
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertEqual(plan.natural_match_boundary_indices, (0,))
        self.assertEqual(plan.applied_count, 0)
        self.assertEqual(
            tuple(item[0] for item in plan.boundary_residual_attention_mismatches),
            (0,),
        )
        _boundary_index, position, scale = (
            plan.boundary_residual_attention_mismatches[0]
        )
        self.assertLessEqual(
            position,
            DirectMemberEditorialMotionConfig()
            .maximum_face_residual_position_mismatch,
        )
        self.assertLessEqual(
            scale,
            DirectMemberEditorialMotionConfig()
            .maximum_face_residual_scale_mismatch,
        )
        self.assertIn(
            ("required_dense_natural_match", 1),
            plan.skipped_boundary_reasons,
        )

    def test_required_dense_rejection_reports_the_measured_cause(self) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="outgoing",
            transformed_source_index=0,
            zoom=1.20,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.12,
            raw_attention_scale_mismatch=0.40,
            residual_attention_position_mismatch=0.0,
            residual_attention_scale_mismatch=0.0,
            baseline_face_geometry_mismatch=0.20,
            transformed_face_geometry_mismatch=0.0,
            improvement_fraction=1.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=("attention_scale",),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {"mismatch": SimpleNamespace(transform_repair=repair)}
            },
        )
        selection.source_frame_geometries = (geometry, geometry)

        with self.assertRaises(DirectMemberEditorialMotionError) as raised:
            build_direct_member_editorial_motion_plan(
                visual_selection=selection,
                render_segments=self._renders((0, 1), (1.5, 1.5)),
                transitions=(RenderTransition(),),
                common_master_start=0.0,
                geometry_provider=lambda source, _time: self._sample(
                    boxes[source]
                ),
                fps=30.0,
                config=DirectMemberEditorialMotionConfig(
                    minimum_mismatch_improvement_fraction=0.80,
                ),
            )

        self.assertEqual(raised.exception.missing_boundary_indices, (0,))
        self.assertIn(
            "dense_widened_subject_evidence_missing",
            raised.exception.rejection_reasons[0],
        )
        self.assertIn(
            "dense_widened_subject_evidence_missing=1",
            str(raised.exception),
        )

    def test_partial_required_match_keeps_safe_plan_for_rendered_proxy_review(
        self,
    ) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="outgoing",
            transformed_source_index=0,
            zoom=1.20,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.12,
            raw_attention_scale_mismatch=0.40,
            residual_attention_position_mismatch=0.0,
            residual_attention_scale_mismatch=0.0,
            baseline_face_geometry_mismatch=0.20,
            transformed_face_geometry_mismatch=0.0,
            improvement_fraction=1.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=("attention_scale",),
        )
        selection = self._selection(
            (0, 1),
            boxes,
            boundary_overrides={
                0: {"mismatch": SimpleNamespace(transform_repair=repair)}
            },
        )
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1), (1.5, 1.5)),
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                minimum_mismatch_improvement_fraction=0.80,
                allow_partial_required_boundary_matches=True,
            ),
        )

        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertEqual(plan.natural_match_boundary_indices, ())
        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("required_dense_partial_unresolved", 1),
            plan.skipped_boundary_reasons,
        )

    def test_tier_d_neighbor_does_not_block_a_continuous_required_ramp(
        self,
    ) -> None:
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.20, 0.18, 0.60),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="outgoing",
            transformed_source_index=0,
            zoom=1.20,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.12,
            raw_attention_scale_mismatch=0.40,
            residual_attention_position_mismatch=0.0,
            residual_attention_scale_mismatch=0.0,
            baseline_face_geometry_mismatch=0.20,
            transformed_face_geometry_mismatch=0.0,
            improvement_fraction=1.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=("attention_scale",),
        )
        selection = self._selection(
            (0, 0, 1),
            boxes,
            boundary_overrides={
                1: {"mismatch": SimpleNamespace(transform_repair=repair)}
            },
        )
        selection.source_frame_geometries = (geometry, geometry)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 0, 1), (2.0, 2.0, 2.0)),
            transitions=(
                RenderTransition(tier="D"),
                RenderTransition(),
            ),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[1 if source == 0 else 2],
                subject_box=(0.30, 0.10, 0.40, 0.75),
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (1,))
        self.assertIsNotNone(plan.track_for(1))

    def test_isolated_face_detector_jump_is_stabilized_by_subject_track(
        self,
    ) -> None:
        selection, renders, boxes = self._basic_inputs()
        subject_box = (0.20, 0.08, 0.60, 0.84)

        def geometry(source: int, time_seconds: float):
            box = boxes[source]
            if source == 0 and 7.49 <= time_seconds <= 7.51:
                box = (0.56, 0.225, 0.20, 0.55)
            return self._sample(box, subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertIsNotNone(plan.track_for(0))

    def test_multi_frame_face_switch_returns_to_boundary_anchored_track(
        self,
    ) -> None:
        selection, renders, boxes = self._basic_inputs()
        subject_box = (0.20, 0.08, 0.60, 0.84)

        def geometry(source: int, time_seconds: float):
            box = boxes[source]
            if source == 0 and 7.40 <= time_seconds <= 7.60:
                box = (0.61, 0.225, 0.18, 0.55)
            return self._sample(box, subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertIsNotNone(plan.track_for(0))

    def test_short_face_detection_gap_is_reconstructed_from_subject_track(
        self,
    ) -> None:
        selection, renders, boxes = self._basic_inputs()
        subject_box = (0.20, 0.08, 0.60, 0.84)

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.49 <= time_seconds <= 7.51:
                return DirectMemberFaceSample(
                    subject_box,
                    0,
                    "none",
                    subject_box,
                    0.95,
                    "pphumanseg_full_subject_silhouette",
                )
            return self._sample(boxes[source], subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)

    def test_long_face_detection_gap_is_not_invented(self) -> None:
        selection, renders, boxes = self._basic_inputs()
        subject_box = (0.20, 0.08, 0.60, 0.84)

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.40 <= time_seconds <= 7.80:
                return DirectMemberFaceSample(
                    subject_box,
                    0,
                    "none",
                    subject_box,
                    0.95,
                    "pphumanseg_full_subject_silhouette",
                )
            return self._sample(boxes[source], subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                maximum_dense_face_interpolation_gap_frames=8,
                maximum_dense_face_edge_extrapolation_frames=2,
            ),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            plan.skipped_boundary_reasons,
        )

    def test_required_repair_uses_non_overlapping_edge_beside_effect(self) -> None:
        boxes = (
            (0.40, 0.20, 0.18, 0.60),
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.25, 0.18, 0.50),
        )
        geometry = direct_member_source_frame_geometry((2160, 3840))
        repair = DirectMemberBoundaryTransformRepair(
            transformed_side="incoming",
            transformed_source_index=1,
            zoom=1.20,
            center_x=0.5,
            center_y=0.5,
            source_geometry=geometry,
            raw_attention_position_mismatch=0.12,
            raw_attention_scale_mismatch=0.40,
            residual_attention_position_mismatch=0.0,
            residual_attention_scale_mismatch=0.0,
            baseline_face_geometry_mismatch=0.20,
            transformed_face_geometry_mismatch=0.0,
            improvement_fraction=1.0,
            face_visible_fraction=1.0,
            subject_geometry_basis="pphumanseg_full_subject_silhouette",
            subject_visible_fraction=1.0,
            repaired_severe_flags=("attention_scale",),
        )
        selection = self._selection(
            (0, 1, 2),
            boxes,
            boundary_overrides={
                0: {"mismatch": SimpleNamespace(transform_repair=repair)}
            },
        )
        selection.source_frame_geometries = (
            direct_member_source_frame_geometry((1080, 1920)),
            geometry,
            geometry,
        )

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders((0, 1, 2), (2.0, 2.0, 2.0)),
            transitions=(
                RenderTransition(),
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=3,
                ),
            ),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source],
                subject_box=(0.30, 0.10, 0.40, 0.75),
            ),
            fps=30.0,
        )

        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertIsNone(plan.track_for(0))
        self.assertIsNotNone(plan.track_for(1))

    def test_dense_crop_falls_back_to_a_smaller_still_improving_zoom(
        self,
    ) -> None:
        selection, renders, boxes = self._basic_inputs()
        subject_box = (0.05, 0.05, 0.90, 0.90)

        def geometry(source: int, time_seconds: float):
            box = boxes[source]
            if source == 0 and 7.72 <= time_seconds <= 7.92:
                box = (0.02, 0.225, 0.94, 0.55)
            return self._sample(box, subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        track = plan.track_for(0)
        self.assertIsNotNone(track)
        assert track is not None
        self.assertLess(track.final_keyframe.zoom, 1.08)
        self.assertGreaterEqual(track.final_keyframe.zoom, 1.02)

    def test_dense_provider_uses_canonical_frame_sampler_and_numeric_cache(self) -> None:
        descriptor = SimpleNamespace(
            silhouette_bbox_xywh=(0.25, 0.10, 0.50, 0.80),
            silhouette_confidence=0.90,
            silhouette_backend=SILHOUETTE_BACKEND,
            primary_face_bbox=(0.40, 0.20, 0.20, 0.20),
            face_count=1,
            face_detector_backend="yunet",
            formation_people=(),
        )

        class FakeSampler:
            instances: list["FakeSampler"] = []

            def __init__(self, _path, *, feature_cache_directory=None):
                self.feature_cache_directory = feature_cache_directory
                self.frame_decode_count = 0
                self.feature_calls = 0
                self.closed = False
                self.instances.append(self)

            def frame_identity(self, seconds):
                return CanonicalFrameIdentity.from_seconds(
                    seconds,
                    frame_rate=30.0,
                )

            def feature_descriptor_at(self, _seconds, **_kwargs):
                self.feature_calls += 1
                self.frame_decode_count += 1
                return descriptor

            def feature_cache_diagnostics(self):
                return {
                    "hit_count": 2,
                    "miss_count": 1,
                    "write_count": 1,
                }

            def close(self):
                self.closed = True

        with patch(
            "crosscut.direct_member_editorial_motion._VideoSampler",
            FakeSampler,
        ):
            provider = DirectMemberDenseFaceProvider(
                (Path("source.mp4"),),
                feature_cache_directory=Path("cache"),
            )
            first = provider.sample(0, 1.0)
            second = provider.sample(0, 1.001)
            diagnostics = provider.diagnostics()
            provider.close()

        self.assertIs(first, second)
        self.assertEqual(FakeSampler.instances[0].feature_calls, 1)
        self.assertEqual(diagnostics["decode_count"], 1)
        self.assertEqual(diagnostics["persistent_feature_cache_hit_count"], 2)
        self.assertTrue(FakeSampler.instances[0].closed)

    def test_dense_provider_forwards_direct_cache_and_closes_partial_open(
        self,
    ) -> None:
        direct_root = Path("direct-derived-cache")
        calls: list[dict[str, object]] = []
        instances: list[object] = []

        class FailingSecondSampler:
            def __init__(self, _path: Path, **kwargs: object) -> None:
                self.closed = False
                calls.append(dict(kwargs))
                instances.append(self)
                if len(instances) == 2:
                    raise RuntimeError("required cache failed")

            def close(self) -> None:
                self.closed = True

        with patch(
            "crosscut.direct_member_editorial_motion._VideoSampler",
            FailingSecondSampler,
        ), patch(
            "crosscut.direct_member_editorial_motion."
            "_enable_dense_sequential_frame_reads",
        ):
            with self.assertRaisesRegex(RuntimeError, "required cache failed"):
                DirectMemberDenseFaceProvider(
                    (Path("first.mp4"), Path("second.mp4")),
                    feature_cache_directory=Path("legacy-cache"),
                    feature_derived_cache_directory=direct_root,
                    require_persistent_feature_cache=True,
                )

        self.assertEqual(len(calls), 2)
        for kwargs in calls:
            self.assertEqual(
                kwargs["feature_cache_directory"],
                Path("legacy-cache"),
            )
            self.assertEqual(
                kwargs["feature_derived_cache_directory"],
                direct_root,
            )
            self.assertIs(kwargs["require_persistent_feature_cache"], True)
        self.assertTrue(instances[0].closed)

    def test_required_cache_guard_preempts_focal_and_dense_ram_fast_paths(
        self,
    ) -> None:
        descriptor = SimpleNamespace(
            silhouette_bbox_xywh=(0.25, 0.10, 0.50, 0.80),
            silhouette_confidence=0.90,
            silhouette_backend=SILHOUETTE_BACKEND,
            primary_face_bbox=(0.40, 0.20, 0.20, 0.20),
            face_count=1,
            face_detector_backend="yunet",
            formation_people=(),
        )
        candidate = FocalFaceCandidate(
            (0.40, 0.20, 0.20, 0.20),
            (1.0, 0.0),
        )

        class GuardedSampler:
            instances: list["GuardedSampler"] = []

            def __init__(self, _path: Path, **_kwargs: object) -> None:
                self.frame_decode_count = 0
                self.guard_unavailable = False
                self.guard_call_count = 0
                self.instances.append(self)

            def assert_required_persistent_cache_available(self) -> None:
                self.guard_call_count += 1
                if self.guard_unavailable:
                    raise RuntimeError("required direct cache unavailable")

            def frame_identity(self, seconds: float) -> CanonicalFrameIdentity:
                return CanonicalFrameIdentity.from_seconds(
                    seconds,
                    frame_rate=30.0,
                )

            def frame_at(self, _seconds: float) -> np.ndarray:
                self.frame_decode_count += 1
                return np.zeros((8, 8, 3), dtype="uint8")

            def feature_descriptor_at(
                self,
                _seconds: float,
                **_kwargs: object,
            ) -> object:
                self.frame_decode_count += 1
                return descriptor

            def close(self) -> None:
                return None

        with patch(
            "crosscut.direct_member_editorial_motion._VideoSampler",
            GuardedSampler,
        ), patch(
            "crosscut.direct_member_editorial_motion."
            "_enable_dense_sequential_frame_reads",
        ), patch(
            "crosscut.direct_member_editorial_motion."
            "detect_focal_face_candidates",
            return_value=(candidate,),
        ):
            provider = DirectMemberDenseFaceProvider(
                (Path("left.mp4"), Path("right.mp4")),
                feature_derived_cache_directory=Path("direct-cache"),
                require_persistent_feature_cache=True,
            )
            paired = provider.paired_samples(0, 1.0, 1, 1.0)
            self.assertTrue(all(sample is not None for sample in paired))
            self.assertTrue(provider._focal_candidate_cache)
            self.assertTrue(provider._cache)
            decode_counts = tuple(
                sampler.frame_decode_count
                for sampler in GuardedSampler.instances
            )
            for sampler in GuardedSampler.instances:
                sampler.guard_unavailable = True

            guarded_requests = (
                lambda: provider.paired_samples(0, 1.0, 1, 1.0),
                lambda: provider.sample(0, 1.0),
                lambda: provider.recover_exact_focal_pair_from_registered_references(
                    0,
                    1.0,
                    1,
                    1.0,
                ),
            )
            for request in guarded_requests:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "required direct cache unavailable",
                ):
                    request()
            provider.close()

        self.assertEqual(
            tuple(
                sampler.frame_decode_count
                for sampler in GuardedSampler.instances
            ),
            decode_counts,
        )
        self.assertEqual(GuardedSampler.instances[0].guard_call_count, 4)
        self.assertEqual(GuardedSampler.instances[1].guard_call_count, 1)

    def test_dense_provider_decodes_proven_adjacent_frames_sequentially(
        self,
    ) -> None:
        descriptor = SimpleNamespace(
            silhouette_bbox_xywh=(0.25, 0.10, 0.50, 0.80),
            silhouette_confidence=0.90,
            silhouette_backend=SILHOUETTE_BACKEND,
            primary_face_bbox=(0.40, 0.20, 0.20, 0.20),
            face_count=1,
            face_detector_backend="yunet",
            formation_people=(),
        )

        class FakeCapture:
            def __init__(self):
                self.position = 0.0
                self.sequential_reads = 0

            def get(self, property_id):
                if property_id == cv2.CAP_PROP_POS_FRAMES:
                    return self.position
                return 0.0

            def read(self):
                self.sequential_reads += 1
                self.position += 1.0
                return True, np.zeros(
                    (8, 8, 3),
                    dtype="uint8",
                )

        class FakeSampler:
            instances: list["FakeSampler"] = []

            def __init__(self, _path, *, feature_cache_directory=None):
                self.feature_cache_directory = feature_cache_directory
                self.frame_rate = 30.0
                self.capture = FakeCapture()
                self.cache = OrderedDict()
                self.frame_decode_count = 0
                self.random_seeks = 0
                self.instances.append(self)

            def frame_identity(self, seconds):
                return CanonicalFrameIdentity.from_seconds(
                    seconds,
                    frame_rate=self.frame_rate,
                )

            def frame_at(self, seconds):
                key = self.frame_identity(seconds)
                if key not in self.cache:
                    self.random_seeks += 1
                    self.capture.position = float(key.frame_index + 1)
                    self.frame_decode_count += 1
                    self.cache[key] = np.zeros(
                        (8, 8, 3),
                        dtype="uint8",
                    )
                return self.cache[key]

            def feature_descriptor_at(self, seconds, **_kwargs):
                self.frame_at(seconds)
                return descriptor

            def feature_cache_diagnostics(self):
                return None

            def close(self):
                pass

        with patch(
            "crosscut.direct_member_editorial_motion._VideoSampler",
            FakeSampler,
        ):
            provider = DirectMemberDenseFaceProvider((Path("source.mp4"),))
            for frame_index in range(3):
                self.assertIsNotNone(provider.sample(0, frame_index / 30.0))
            diagnostics = provider.diagnostics()

        sampler = FakeSampler.instances[0]
        self.assertEqual(sampler.random_seeks, 1)
        self.assertEqual(sampler.capture.sequential_reads, 2)
        self.assertEqual(diagnostics["random_seek_frame_decode_count"], 1)
        self.assertEqual(diagnostics["sequential_frame_decode_count"], 2)

    def test_widened_zoom_requires_dense_full_subject_and_enlarges_smaller_side(self) -> None:
        source_indices = (0, 1)
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.205, 0.18, 0.585),
        )
        selection = self._selection(source_indices, boxes)
        renders = self._renders(source_indices, (8.0, 8.0))
        subject_box = (0.31, 0.08, 0.38, 0.80)

        without_subject = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
        )
        with_subject = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source], subject_box=subject_box
            ),
            fps=30.0,
        )

        self.assertEqual(without_subject.applied_count, 1)
        self.assertLessEqual(
            without_subject.track_for(0).final_keyframe.zoom,
            1.12,
        )
        self.assertEqual(with_subject.applied_count, 1)
        self.assertIsNotNone(with_subject.track_for(0))
        self.assertIsNone(with_subject.track_for(1))
        self.assertGreater(with_subject.track_for(0).final_keyframe.zoom, 1.12)
        self.assertLessEqual(with_subject.track_for(0).final_keyframe.zoom, 1.18)

    def test_face_preserving_closeup_allows_intentional_partial_body_crop(self) -> None:
        source_indices = (0, 1)
        boxes = (
            (0.40, 0.25, 0.18, 0.50),
            (0.40, 0.205, 0.18, 0.585),
        )
        selection = self._selection(source_indices, boxes)
        renders = self._renders(source_indices, (8.0, 8.0))

        def geometry_provider(
            source: int,
            canonical_time: float,
        ) -> DirectMemberFaceSample:
            subject_box = (
                (0.29, 0.0, 0.42, 1.0)
                if source == 0 and round(canonical_time * 30.0) == 225
                else (0.31, 0.08, 0.38, 0.80)
            )
            return self._sample(boxes[source], subject_box=subject_box)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry_provider,
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                minimum_mismatch_improvement_fraction=0.80,
            ),
        )

        self.assertEqual(plan.applied_count, 1)
        self.assertGreater(plan.track_for(0).final_keyframe.zoom, 1.12)

    def test_widened_pan_uses_subject_proof_beyond_legacy_fallback(self) -> None:
        source_indices = (0, 1)
        boxes = (
            (0.31, 0.25, 0.18, 0.50),
            (0.41, 0.1875, 0.18, 0.625),
        )
        selection = self._selection(source_indices, boxes)
        renders = self._renders(source_indices, (8.0, 8.0))

        without_subject = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
        )
        with_subject = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                boxes[source],
                subject_box=(0.20, 0.10, 0.60, 0.75),
            ),
            fps=30.0,
        )

        self.assertEqual(without_subject.applied_count, 0)
        self.assertIsNone(without_subject.track_for(0))
        self.assertEqual(with_subject.applied_count, 1)
        track = with_subject.track_for(0)
        self.assertIsNotNone(track)
        assert track is not None
        self.assertGreater(track.final_keyframe.zoom, 1.12)
        self.assertLessEqual(track.final_keyframe.zoom, 1.50)
        self.assertGreater(
            abs(track.final_keyframe.center_x - 0.5),
            0.025,
        )
        self.assertLessEqual(
            abs(track.final_keyframe.center_x - 0.5),
            0.16,
        )

    def test_non_face_or_multiple_face_audit_is_rejected(self) -> None:
        selection, renders, boxes = self._basic_inputs()
        cases = (
            self._audited_sample(
                boxes[0], basis="aggregate_human_silhouette"
            ),
            self._audited_sample(boxes[0], face_count=2),
        )
        for outgoing_sample in cases:
            with self.subTest(outgoing_sample=outgoing_sample):
                selection = self._selection(
                    (0, 1),
                    boxes,
                    boundary_overrides={
                        0: {"outgoing_sample": outgoing_sample}
                    },
                )
                plan = build_direct_member_editorial_motion_plan(
                    visual_selection=selection,
                    render_segments=renders,
                    transitions=(RenderTransition(),),
                    common_master_start=0.0,
                    geometry_provider=lambda source, _time: self._sample(
                        boxes[source]
                    ),
                    fps=30.0,
                )

                self.assertEqual(plan.applied_count, 0)
                self.assertIn(
                    ("no_unambiguous_face_pair", 1),
                    plan.skipped_boundary_reasons,
                )

    def test_missing_dense_face_sample_is_rejected(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: (
                None if source == 0 else self._sample(boxes[source])
            ),
            fps=30.0,
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_boundary_face_missing", 1),
            plan.skipped_boundary_reasons,
        )

    def test_dense_sample_contract_rejects_ambiguous_multiple_faces(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one face"):
            DirectMemberFaceSample(
                (0.35, 0.225, 0.20, 0.55),
                2,
                "yunet",
            )

    def test_dense_sample_contract_rejects_unverified_face_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified YuNet"):
            DirectMemberFaceSample(
                (0.35, 0.225, 0.20, 0.55),
                1,
                "haar_fallback",
            )

    def test_dense_endpoint_must_agree_with_audited_face(self) -> None:
        selection, renders, boxes = self._basic_inputs()
        disagreeing = (0.46, 0.225, 0.20, 0.55)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(
                disagreeing if source == 0 else boxes[source]
            ),
            fps=30.0,
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_boundary_face_disagrees_with_audit", 1),
            plan.skipped_boundary_reasons,
        )

    def test_dense_per_frame_face_continuity_failure_rejects_motion(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.49 <= time_seconds <= 7.51:
                return self._sample((0.56, 0.225, 0.20, 0.55))
            return self._sample(boxes[source])

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            plan.skipped_boundary_reasons,
        )

    def test_face_discontinuity_at_ramp_edge_is_rejected(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.216 < time_seconds < 7.96:
                # The face jumps on the first transformed frame, then moves
                # slowly enough that checking only transformed-frame pairs
                # would miss the discontinuity at the ramp edge.
                alpha = (time_seconds - 7.233333333333333) / (
                    7.933333333333334 - 7.233333333333333
                )
                center_x = 0.85 + (0.45 - 0.85) * max(
                    0.0, min(1.0, alpha)
                )
                return self._sample(
                    (center_x - 0.01, 0.225, 0.02, 0.55)
                )
            return self._sample(boxes[source])

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            plan.skipped_boundary_reasons,
        )

    def test_dense_per_frame_face_containment_failure_rejects_motion(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.82 <= time_seconds <= 7.84:
                # Same center and height as the tracked face, but far too wide
                # to survive the proposed zoom without cropping the face.
                return self._sample((0.0, 0.225, 1.00, 0.55))
            return self._sample(boxes[source])

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
            config=DirectMemberEditorialMotionConfig(
                minimum_dense_face_visible_fraction=0.995,
            ),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            plan.skipped_boundary_reasons,
        )

    def test_full_run_review_accepts_wider_pan_only_when_dense_safe(self) -> None:
        selection, renders, boxes = self._basic_inputs()

        def geometry(source: int, time_seconds: float):
            if source == 0 and 7.53 <= time_seconds <= 7.97:
                # A continuous rightward face move remains fully visible under
                # centered scale, but the panned candidate clips it.  Its
                # frame-to-frame center step stays below the dense continuity
                # gate so containment is the differentiator.
                if time_seconds < 7.72:
                    center_x = 0.45 + 0.51 * (time_seconds - 7.53) / 0.19
                elif time_seconds <= 7.78:
                    center_x = 0.96
                else:
                    center_x = 0.96 - 0.51 * (time_seconds - 7.78) / 0.19
                center_x = max(0.45, min(0.96, center_x))
                return self._sample(
                    (center_x - 0.0075, 0.225, 0.015, 0.55)
                )
            return self._sample(boxes[source])

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=geometry,
            fps=30.0,
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertEqual(plan.pan_event_count, 1)
        track = plan.track_for(0)
        self.assertIsNotNone(track)
        self.assertTrue(
            any(
                keyframe.center_x != 0.5 or keyframe.center_y != 0.5
                for keyframe in track.keyframes
            )
        )

    def test_short_duration_uses_strict_event_density_floor(self) -> None:
        selection, renders, boxes = self._basic_inputs(duration=1.5)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=renders,
            transitions=(RenderTransition(),),
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=30.0,
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(("event_density_cap", 1), plan.skipped_boundary_reasons)

    def test_spacing_uses_quantized_output_boundary_clock(self) -> None:
        fps = 29.97
        durations = (2.6 / fps, 1.75, 1.75, 1.50)
        sources = (0, 1, 2, 3)
        boxes = (
            (0.40, 0.17, 0.20, 0.66),
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.20, 0.20, 0.60),
            (0.40, 0.17, 0.20, 0.66),
        )
        selection = self._selection(sources, boxes)

        plan = build_direct_member_editorial_motion_plan(
            visual_selection=selection,
            render_segments=self._renders(sources, durations),
            transitions=(RenderTransition(),) * 3,
            common_master_start=0.0,
            geometry_provider=lambda source, _time: self._sample(boxes[source]),
            fps=fps,
            config=DirectMemberEditorialMotionConfig(
                target_events_per_minute=30.0,
                minimum_event_spacing_seconds=3.5,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertIn(("event_spacing_cap", 1), plan.skipped_boundary_reasons)

    def test_misaligned_selection_evidence_fails_closed(self) -> None:
        selection, renders, boxes = self._basic_inputs()
        cases = (
            self._selection(
                (0, 1),
                boxes,
                boundary_overrides={0: {"boundary_index": 1}},
            ),
            self._selection(
                (0, 1),
                boxes,
                boundary_overrides={
                    0: {"outgoing_eligible_source_index": 7}
                },
            ),
        )
        for invalid_selection in cases:
            with self.subTest(selection=invalid_selection):
                with self.assertRaises(ValueError):
                    build_direct_member_editorial_motion_plan(
                        visual_selection=invalid_selection,
                        render_segments=renders,
                        transitions=(RenderTransition(),),
                        common_master_start=0.0,
                        geometry_provider=lambda source, _time: self._sample(
                            boxes[source]
                        ),
                        fps=30.0,
                    )


if __name__ == "__main__":
    unittest.main()
