from __future__ import annotations

from dataclasses import replace
import unittest

from crosscut.member_draft import (
    MemberDraftSegment,
    MemberSegmentKind,
    SegmentCropSample,
    SegmentSubjectSample,
)
from crosscut.member_editorial_motion import (
    MemberEditorialMotionConfig,
    build_member_editorial_motion_plan,
)
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderSegment,
    RenderTransition,
)


class MemberEditorialMotionTests(unittest.TestCase):
    @staticmethod
    def _member_segment(
        *,
        index: int,
        source_index: int,
        start: float,
        duration: float,
        subject_height: float,
        subject_center_x: float,
        subject_center_y: float = 0.50,
        crop_start_x: float = 0.32,
        crop_end_x: float = 0.36,
        include_subjects: bool = True,
    ) -> MemberDraftSegment:
        crop_width = 0.31640625
        times = tuple(
            min(duration, step * 0.10)
            for step in range(int(duration / 0.10) + 2)
            if step * 0.10 <= duration + 1e-8
        )
        if times[-1] < duration - 1e-8:
            times += (duration,)
        crop_samples = []
        subject_samples = []
        for time_value in times:
            alpha = time_value / duration
            crop_x = crop_start_x + (crop_end_x - crop_start_x) * alpha
            crop_samples.append(
                SegmentCropSample(
                    time_value,
                    crop_x,
                    0.0,
                    crop_width,
                    1.0,
                )
            )
            subject_width = 0.25
            left = crop_x + (subject_center_x - subject_width / 2.0) * crop_width
            right = crop_x + (subject_center_x + subject_width / 2.0) * crop_width
            top = subject_center_y - subject_height / 2.0
            bottom = subject_center_y + subject_height / 2.0
            subject_samples.append(
                SegmentSubjectSample(
                    time_value,
                    (left, top, right, bottom),
                    0.96,
                )
            )
        return MemberDraftSegment(
            segment_index=index,
            canonical_master_start=start,
            canonical_master_end=start + duration,
            output_start_seconds=start,
            kind=MemberSegmentKind.TARGET_CROP,
            source_index=source_index,
            source_local_start=start,
            source_local_end=start + duration,
            audio_offset_seconds=0.0,
            crop_samples=tuple(crop_samples),
            interval_coverage=1.0,
            mean_confidence=0.96,
            stability=0.94,
            selection_score=0.91,
            subject_samples=(tuple(subject_samples) if include_subjects else ()),
        )

    @staticmethod
    def _render_segment(member: MemberDraftSegment, *, timewarp: bool = False) -> RenderSegment:
        pan_crop = PanCropTrack(
            tuple(
                PanCropKeyframe(
                    sample.time_seconds,
                    NormalizedCrop(
                        sample.x,
                        sample.y,
                        sample.width,
                        sample.height,
                    ),
                )
                for sample in member.crop_samples
            )
        )
        return RenderSegment(
            source_index=int(member.source_index),
            source_start=float(member.source_local_start),
            duration=member.duration_seconds,
            master_start=member.output_start_seconds,
            pan_crop=pan_crop,
            source_duration=(
                member.duration_seconds * 1.01 if timewarp else None
            ),
        )

    def _pair(self, *, include_left_subjects: bool = True):
        duration = 1.50
        left = self._member_segment(
            index=0,
            source_index=0,
            start=0.0,
            duration=duration,
            subject_height=0.60,
            subject_center_x=0.44,
            include_subjects=include_left_subjects,
        )
        right = self._member_segment(
            index=1,
            source_index=1,
            start=duration,
            duration=duration,
            subject_height=0.66,
            subject_center_x=0.52,
            crop_start_x=0.29,
            crop_end_x=0.34,
        )
        return (left, right)

    def test_matches_confirmed_member_scale_and_position_across_cut(self) -> None:
        members = self._pair()
        renders = tuple(self._render_segment(item) for item in members)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=renders,
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
            config=MemberEditorialMotionConfig(target_events_per_minute=30.0),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertEqual(plan.zoom_event_count, 1)
        self.assertEqual(plan.roll_event_count, 0)
        self.assertIsNotNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertGreater(plan.track_for(0).final_keyframe.zoom, 1.05)
        self.assertIn("every_output_frame_99.5_percent", plan.to_dict()["policy"]["subject_safety"])

    def test_missing_confirmed_subject_geometry_disables_motion(self) -> None:
        members = self._pair(include_left_subjects=False)
        renders = tuple(self._render_segment(item) for item in members)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=renders,
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("no_confirmed_subject_geometry", 1),
            plan.skipped_boundary_reasons,
        )

    def test_internal_camera_cut_disables_boundary_motion(self) -> None:
        members = self._pair()
        renders = tuple(self._render_segment(item) for item in members)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=renders,
            transitions=(RenderTransition(),),
            source_scene_cut_times=((0.75,), ()),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(("internal_camera_cut", 1), plan.skipped_boundary_reasons)

    def test_dynamic_crop_and_local_timewarp_pass_every_frame_review(self) -> None:
        members = self._pair()
        renders = tuple(
            self._render_segment(item, timewarp=True) for item in members
        )
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=renders,
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
            fps=29.97,
            config=MemberEditorialMotionConfig(target_events_per_minute=30.0),
        )

        track = plan.track_for(0)
        self.assertIsNotNone(track)
        self.assertEqual(track.reason, "confirmed_member_subject_scale_position_match")
        self.assertEqual(track.final_keyframe.roll_degrees, 0.0)

    def test_scale_gap_beyond_bounded_zoom_is_not_forced(self) -> None:
        duration = 1.50
        left = self._member_segment(
            index=0,
            source_index=0,
            start=0.0,
            duration=duration,
            subject_height=0.50,
            subject_center_x=0.50,
        )
        right = self._member_segment(
            index=1,
            source_index=1,
            start=duration,
            duration=duration,
            subject_height=0.70,
            subject_center_x=0.50,
        )
        members = (left, right)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=tuple(self._render_segment(item) for item in members),
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("boundary_scale_exceeds_transform_bound", 1),
            plan.skipped_boundary_reasons,
        )

    def test_small_improvement_does_not_claim_a_still_distant_position_match(self) -> None:
        duration = 1.50
        left = self._member_segment(
            index=0,
            source_index=0,
            start=0.0,
            duration=duration,
            subject_height=0.60,
            subject_center_x=0.32,
        )
        right = self._member_segment(
            index=1,
            source_index=1,
            start=duration,
            duration=duration,
            subject_height=0.55,
            subject_center_x=0.50,
        )
        members = (left, right)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=tuple(self._render_segment(item) for item in members),
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(
            ("no_improving_contained_transform", 1),
            plan.skipped_boundary_reasons,
        )

    def test_short_clip_does_not_exceed_default_event_density(self) -> None:
        members = self._pair()
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=tuple(self._render_segment(item) for item in members),
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
        )

        self.assertEqual(plan.applied_count, 0)
        self.assertIn(("event_density_cap", 1), plan.skipped_boundary_reasons)

    def test_spacing_uses_quantized_output_boundaries(self) -> None:
        fps = 29.97
        durations = (2.6 / fps, 1.75, 1.75, 1.50)
        sources = (0, 1, 1, 2)
        heights = (0.66, 0.60, 0.60, 0.66)
        members = []
        cursor = 0.0
        for index, (duration, source, height) in enumerate(
            zip(durations, sources, heights, strict=True)
        ):
            members.append(
                self._member_segment(
                    index=index,
                    source_index=source,
                    start=cursor,
                    duration=duration,
                    subject_height=height,
                    subject_center_x=0.50,
                )
            )
            cursor += duration
        plan = build_member_editorial_motion_plan(
            member_segments=tuple(members),
            render_segments=tuple(
                self._render_segment(item) for item in members
            ),
            transitions=(RenderTransition(),) * 3,
            source_scene_cut_times=((), (), ()),
            fps=fps,
            config=MemberEditorialMotionConfig(
                target_events_per_minute=30.0,
                minimum_event_spacing_seconds=3.5,
            ),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        self.assertIn(("event_spacing_cap", 1), plan.skipped_boundary_reasons)

    def test_diagonal_pan_uses_renderer_radial_cap_and_keeps_centered_fallback(self) -> None:
        duration = 8.0
        left = self._member_segment(
            index=0,
            source_index=0,
            start=0.0,
            duration=duration,
            subject_height=0.60,
            subject_center_x=0.50,
            subject_center_y=0.50,
        )
        right = self._member_segment(
            index=1,
            source_index=1,
            start=duration,
            duration=duration,
            subject_height=0.55,
            subject_center_x=0.56,
            subject_center_y=0.56,
        )
        members = (left, right)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=tuple(self._render_segment(item) for item in members),
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
            config=MemberEditorialMotionConfig(maximum_center_shift=0.04),
        )

        self.assertEqual(plan.boundary_match_count, 1)
        track = next(item.track for item in plan.segments if item.track is not None)
        self.assertTrue(
            all(
                ((keyframe.center_x - 0.5) ** 2 + (keyframe.center_y - 0.5) ** 2)
                ** 0.5
                <= 0.04 + 1e-9
                for keyframe in track.keyframes
            )
        )

    def test_full_run_containment_falls_back_from_pan_to_safe_scale_only(self) -> None:
        duration = 8.0
        left = self._member_segment(
            index=0,
            source_index=0,
            start=0.0,
            duration=duration,
            subject_height=0.55,
            subject_center_x=0.45,
        )
        crop_width = left.crop_samples[0].width
        moving_subjects = []
        for crop_sample in left.crop_samples:
            time_value = crop_sample.time_seconds
            output_center_x = 0.935 if 7.0 <= time_value <= 7.8 else 0.45
            output_width = 0.05
            source_left = crop_sample.x + (
                output_center_x - output_width / 2.0
            ) * crop_width
            source_right = crop_sample.x + (
                output_center_x + output_width / 2.0
            ) * crop_width
            moving_subjects.append(
                SegmentSubjectSample(
                    time_value,
                    (source_left, 0.225, source_right, 0.775),
                    0.96,
                )
            )
        left = replace(left, subject_samples=tuple(moving_subjects))
        right = self._member_segment(
            index=1,
            source_index=1,
            start=duration,
            duration=duration,
            subject_height=0.60,
            subject_center_x=0.50,
        )
        members = (left, right)
        plan = build_member_editorial_motion_plan(
            member_segments=members,
            render_segments=tuple(self._render_segment(item) for item in members),
            transitions=(RenderTransition(),),
            source_scene_cut_times=((), ()),
            config=MemberEditorialMotionConfig(target_events_per_minute=30.0),
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


if __name__ == "__main__":
    unittest.main()
