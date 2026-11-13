from __future__ import annotations

import unittest

from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from crosscut.joint_editorial_motion_overlay import (
    merge_joint_interior_motion_overlay,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
)


_FPS = 10.0
_PULSE_REASON = "reference_calibrated_same_source_interior_zoom_pulse"


def _keyframe(
    time_seconds: float,
    zoom: float,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> EditorialTransformKeyframe:
    return EditorialTransformKeyframe(
        time_seconds,
        zoom,
        center_x,
        center_y,
    )


def _track(
    *keyframes: EditorialTransformKeyframe,
) -> EditorialTransformTrack:
    return EditorialTransformTrack(keyframes, reason="fixture")


def _safe_single_segment_pulse(
    peak_zoom: float = 1.065,
    *,
    peak_center_x: float = 0.5,
) -> EditorialTransformTrack:
    # At 10 fps the default 0.75 s ramp is rounded up to 0.8 s.  A 3.0 s
    # segment renders frames at local times 0.0 .. 2.9 inclusive.
    return _track(
        _keyframe(0.0, 1.0),
        _keyframe(0.6, 1.0),
        _keyframe(1.4, peak_zoom, center_x=peak_center_x),
        _keyframe(2.2, 1.0),
        _keyframe(2.9, 1.0),
    )


def _safe_split_pulse() -> tuple[
    EditorialTransformTrack,
    EditorialTransformTrack,
]:
    # This is one global 0.6 -> 1.4 -> 2.2 triangular pulse sliced across
    # two 1.5 s segments.  The neighboring output frames intentionally have
    # different zooms: 1.065 at 1.4 s, then 1.056875 at 1.5 s.
    return (
        _track(
            _keyframe(0.0, 1.0),
            _keyframe(0.6, 1.0),
            _keyframe(1.4, 1.065),
        ),
        _track(
            _keyframe(0.0, 1.056875),
            _keyframe(0.7, 1.0),
            _keyframe(1.4, 1.0),
        ),
    )


def _plan(
    tracks: tuple[EditorialTransformTrack | None, ...],
    sources: tuple[int, ...],
    *,
    pulse_indices: frozenset[int] = frozenset(),
    reasons_by_index: dict[int, tuple[str, ...]] | None = None,
    zoom_event_count: int = 0,
    pan_event_count: int = 0,
    roll_event_count: int = 0,
    interior_zoom_pulse_count: int = 0,
) -> EditorialMotionPlan:
    reason_values = reasons_by_index or {}
    return EditorialMotionPlan(
        segments=tuple(
            EditorialMotionSegment(
                segment_index=index,
                source_index=sources[index],
                track=track,
                reasons=tuple(
                    sorted(
                        set(reason_values.get(index, ()))
                        | ({_PULSE_REASON} if index in pulse_indices else set())
                    )
                ),
            )
            for index, track in enumerate(tracks)
        ),
        boundary_match_count=0,
        zoom_event_count=zoom_event_count,
        pan_event_count=pan_event_count,
        roll_event_count=roll_event_count,
        blocked_dissolve_segment_count=0,
        interior_zoom_pulse_count=interior_zoom_pulse_count,
    )


def _segments(
    tracks: tuple[EditorialTransformTrack | None, ...],
    sources: tuple[int, ...],
    durations: tuple[float, ...],
) -> tuple[RenderSegment, ...]:
    elapsed = 0.0
    output: list[RenderSegment] = []
    for track, source, duration in zip(tracks, sources, durations, strict=True):
        output.append(
            RenderSegment(
                source_index=source,
                source_start=elapsed,
                duration=duration,
                master_start=elapsed,
                editorial_transform=track,
            )
        )
        elapsed += duration
    return tuple(output)


def _hard_cut() -> RenderTransition:
    return RenderTransition(tier="C", reason="hard_cut")


def _tier_d_hold() -> RenderTransition:
    return RenderTransition(
        tier="D",
        reason="joint_decoded_same_source_same_shot_hold",
    )


class JointEditorialMotionOverlayTests(unittest.TestCase):
    def test_safe_tier_d_pulse_preserves_unrelated_provider_motion(self) -> None:
        sources = (0, 1, 1)
        durations = (1.0, 1.5, 1.5)
        provider_track = _track(
            _keyframe(0.0, 1.048),
            _keyframe(0.9, 1.048),
        )
        provider_tracks = (provider_track, None, None)
        left_pulse, right_pulse = _safe_split_pulse()
        proposal_tracks = (None, left_pulse, right_pulse)
        provider_plan = _plan(
            provider_tracks,
            sources,
            reasons_by_index={1: ("provider_neutral_boundary_proof",)},
            zoom_event_count=4,
            pan_event_count=2,
            roll_event_count=1,
            interior_zoom_pulse_count=3,
        )
        provider_segments = _segments(provider_tracks, sources, durations)

        result = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(_hard_cut(), _tier_d_hold()),
            proposal_plan=_plan(
                proposal_tracks,
                sources,
                pulse_indices=frozenset({1, 2}),
            ),
            output_fps=_FPS,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.applied_pulse_count, 1)
        self.assertEqual(result.applied_segment_indices, (1, 2))
        self.assertIs(result.segments[0], provider_segments[0])
        self.assertIs(result.segments[0].editorial_transform, provider_track)
        self.assertIs(result.segments[1].editorial_transform, left_pulse)
        self.assertIs(result.segments[2].editorial_transform, right_pulse)
        self.assertIn(
            "provider_neutral_boundary_proof",
            result.plan.segments[1].reasons,
        )
        # The provider crop is constant. Its stale event counters must not
        # become motion evidence; only the one actual pulse is visible.
        self.assertEqual(result.plan.zoom_event_count, 1)
        self.assertEqual(result.plan.pan_event_count, 0)
        self.assertEqual(result.plan.roll_event_count, 0)
        self.assertEqual(result.plan.interior_zoom_pulse_count, 4)

    def test_non_tier_d_internal_boundary_rejects_with_exact_noop(self) -> None:
        sources = (0, 0)
        durations = (1.5, 1.5)
        provider_tracks = (None, None)
        left_pulse, right_pulse = _safe_split_pulse()
        provider_plan = _plan(provider_tracks, sources)
        provider_segments = _segments(provider_tracks, sources, durations)

        result = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(_hard_cut(),),
            proposal_plan=_plan(
                (left_pulse, right_pulse),
                sources,
                pulse_indices=frozenset({0, 1}),
            ),
            output_fps=_FPS,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.rejected_group_count, 1)
        self.assertIs(result.plan, provider_plan)
        self.assertIs(result.segments, provider_segments)

    def test_track_must_end_on_neutral_actual_last_rendered_frame(self) -> None:
        sources = (0,)
        durations = (3.0,)
        provider_tracks = (None,)
        provider_plan = _plan(provider_tracks, sources)
        provider_segments = _segments(provider_tracks, sources, durations)
        duration_endpoint_track = _track(
            _keyframe(0.0, 1.0),
            _keyframe(0.6, 1.0),
            _keyframe(1.4, 1.065),
            _keyframe(2.2, 1.0),
            _keyframe(3.0, 1.0),
        )

        rejected = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(),
            proposal_plan=_plan(
                (duration_endpoint_track,),
                sources,
                pulse_indices=frozenset({0}),
            ),
            output_fps=_FPS,
        )
        accepted = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(),
            proposal_plan=_plan(
                (_safe_single_segment_pulse(),),
                sources,
                pulse_indices=frozenset({0}),
            ),
            output_fps=_FPS,
        )

        self.assertFalse(rejected.applied)
        self.assertIs(rejected.plan, provider_plan)
        self.assertIs(rejected.segments, provider_segments)
        self.assertTrue(accepted.applied)
        accepted_track = accepted.segments[0].editorial_transform
        self.assertIsNotNone(accepted_track)
        assert accepted_track is not None
        self.assertEqual(accepted_track.keyframes[-1].time_seconds, 2.9)
        self.assertEqual(accepted_track.keyframes[-1].zoom, 1.0)

    def test_only_planner_zoom_levels_and_centered_pulses_are_allowed(self) -> None:
        sources = (0,)
        durations = (3.0,)
        provider_tracks = (None,)
        provider_plan = _plan(provider_tracks, sources)
        provider_segments = _segments(provider_tracks, sources, durations)

        for peak_zoom in (1.048, 1.058, 1.065):
            with self.subTest(allowed_peak_zoom=peak_zoom):
                result = merge_joint_interior_motion_overlay(
                    provider_plan=provider_plan,
                    provider_segments=provider_segments,
                    transitions=(),
                    proposal_plan=_plan(
                        (_safe_single_segment_pulse(peak_zoom),),
                        sources,
                        pulse_indices=frozenset({0}),
                    ),
                    output_fps=_FPS,
                )
                self.assertTrue(result.applied)

        rejected_tracks = (
            _safe_single_segment_pulse(1.060),
            _safe_single_segment_pulse(1.065, peak_center_x=0.501),
        )
        for track in rejected_tracks:
            with self.subTest(rejected_track=track):
                result = merge_joint_interior_motion_overlay(
                    provider_plan=provider_plan,
                    provider_segments=provider_segments,
                    transitions=(),
                    proposal_plan=_plan(
                        (track,),
                        sources,
                        pulse_indices=frozenset({0}),
                    ),
                    output_fps=_FPS,
                )
                self.assertFalse(result.applied)
                self.assertIs(result.plan, provider_plan)
                self.assertIs(result.segments, provider_segments)

    def test_off_frame_clock_pulse_is_rejected(self) -> None:
        sources = (0,)
        durations = (3.0,)
        provider_tracks = (None,)
        provider_plan = _plan(provider_tracks, sources)
        provider_segments = _segments(provider_tracks, sources, durations)
        off_clock = _track(
            _keyframe(0.0, 1.0),
            _keyframe(0.61, 1.0),
            _keyframe(1.41, 1.065),
            _keyframe(2.21, 1.0),
            _keyframe(2.9, 1.0),
        )

        result = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(),
            proposal_plan=_plan(
                (off_clock,),
                sources,
                pulse_indices=frozenset({0}),
            ),
            output_fps=_FPS,
        )

        self.assertFalse(result.applied)
        self.assertIs(result.plan, provider_plan)
        self.assertIs(result.segments, provider_segments)

    def test_adjacent_neutral_pulses_count_as_two_zoom_events_not_pans(self) -> None:
        sources = (0, 1)
        durations = (3.0, 3.0)
        provider_tracks = (None, None)
        provider_plan = _plan(
            provider_tracks,
            sources,
            zoom_event_count=7,
            pan_event_count=4,
            roll_event_count=3,
            interior_zoom_pulse_count=2,
        )
        provider_segments = _segments(provider_tracks, sources, durations)

        result = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(_hard_cut(),),
            proposal_plan=_plan(
                (_safe_single_segment_pulse(), _safe_single_segment_pulse()),
                sources,
                pulse_indices=frozenset({0, 1}),
                zoom_event_count=99,
                pan_event_count=99,
                roll_event_count=99,
                interior_zoom_pulse_count=99,
            ),
            output_fps=_FPS,
        )

        self.assertEqual(result.applied_pulse_count, 2)
        self.assertEqual(result.applied_segment_indices, (0, 1))
        self.assertEqual(result.plan.zoom_event_count, 2)
        self.assertEqual(result.plan.pan_event_count, 0)
        self.assertEqual(result.plan.roll_event_count, 0)
        self.assertEqual(result.plan.interior_zoom_pulse_count, 4)
        self.assertEqual(result.to_dict()["applied_zoom_event_count"], 2)
        self.assertEqual(result.to_dict()["applied_pan_event_count"], 0)

    def test_no_candidates_returns_exact_provider_objects(self) -> None:
        sources = (0,)
        provider_tracks = (None,)
        provider_plan = _plan(provider_tracks, sources)
        provider_segments = _segments(provider_tracks, sources, (3.0,))

        result = merge_joint_interior_motion_overlay(
            provider_plan=provider_plan,
            provider_segments=provider_segments,
            transitions=(),
            proposal_plan=_plan(provider_tracks, sources),
            output_fps=_FPS,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.rejected_group_count, 0)
        self.assertIs(result.plan, provider_plan)
        self.assertIs(result.segments, provider_segments)


if __name__ == "__main__":
    unittest.main()
