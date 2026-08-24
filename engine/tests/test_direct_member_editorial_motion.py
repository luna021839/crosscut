from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.direct_member_editorial_motion import (
    DirectMemberEditorialMotionConfig,
    DirectMemberFaceSample,
    build_direct_member_editorial_motion_plan,
)
from crosscut.render import RenderSegment, RenderTransition


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
            "official_target_fancam_unambiguous_face_dense_output_frame_"
            "plus_widened_full_subject_containment_review_v2",
        )

    def test_direct_defaults_expand_only_to_renderer_safe_reference_bounds(self) -> None:
        config = DirectMemberEditorialMotionConfig()

        self.assertEqual(config.maximum_boundary_zoom, 1.18)
        self.assertEqual(config.maximum_center_shift, 0.04)
        self.assertEqual(config.legacy_face_only_maximum_zoom, 1.12)
        self.assertEqual(config.legacy_face_only_maximum_center_shift, 0.025)

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

        self.assertEqual(without_subject.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            without_subject.skipped_boundary_reasons,
        )
        self.assertEqual(with_subject.applied_count, 1)
        self.assertIsNotNone(with_subject.track_for(0))
        self.assertIsNone(with_subject.track_for(1))
        self.assertGreater(with_subject.track_for(0).final_keyframe.zoom, 1.12)
        self.assertLessEqual(with_subject.track_for(0).final_keyframe.zoom, 1.18)

    def test_widened_zoom_rejects_full_subject_edge_clipping_on_any_ramp_frame(self) -> None:
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
                (0.0, 0.02, 0.42, 0.96)
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
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("dense_face_containment_or_continuity_failed", 1),
            plan.skipped_boundary_reasons,
        )

    def test_widened_pan_alone_also_requires_full_subject_proof(self) -> None:
        source_indices = (0, 1)
        boxes = (
            (0.31, 0.25, 0.18, 0.50),
            (0.41, 0.235, 0.18, 0.53),
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
        self.assertEqual(with_subject.applied_count, 1)
        track = with_subject.track_for(0)
        self.assertIsNotNone(track)
        assert track is not None
        self.assertLessEqual(track.final_keyframe.zoom, 1.12)
        self.assertGreater(
            abs(track.final_keyframe.center_x - 0.5),
            0.025,
        )
        self.assertLessEqual(
            abs(track.final_keyframe.center_x - 0.5),
            0.04,
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
            if source == 0 and 7.49 <= time_seconds <= 7.51:
                # Same center and height as the tracked face, but far too wide
                # to survive the proposed zoom without cropping the face.
                return self._sample((0.0, 0.225, 0.90, 0.55))
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

    def test_full_run_review_falls_back_from_pan_to_centered_scale(self) -> None:
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
        self.assertEqual(plan.pan_event_count, 0)
        track = plan.track_for(0)
        self.assertIsNotNone(track)
        self.assertTrue(
            all(
                keyframe.center_x == 0.5 and keyframe.center_y == 0.5
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
