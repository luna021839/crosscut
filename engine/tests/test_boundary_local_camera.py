import math
import unittest

from crosscut.boundary_local_camera import build_boundary_local_camera, is_identity
from crosscut.render import EditorialTransformKeyframe


class BoundaryLocalCameraTests(unittest.TestCase):
    def build(self, entry=1.4, exit=1.3, duration=2., **kwargs):
        return build_boundary_local_camera(
            entry=EditorialTransformKeyframe(0., entry),
            exit=EditorialTransformKeyframe(duration, exit),
            duration_seconds=duration, output_fps=30., **kwargs)

    def test_separate_handles_have_original_middle(self):
        plan = self.build()
        self.assertFalse(plan.handles_overlap)
        self.assertIsNone(plan.active_side_at(1.))
        self.assertEqual(plan.active_side_at(0.), 'entry')
        self.assertEqual(plan.active_side_at(2.), 'exit')

    def test_same_crop_does_not_force_full_shot_identity(self):
        plan = self.build(entry=1.4, exit=1.4)
        self.assertTrue(is_identity(plan.track.evaluate(1.)))

    def test_short_same_crop_is_constant_without_zoom_pump(self):
        plan = self.build(entry=1.4, exit=1.4, duration=.4)
        self.assertTrue(plan.handles_overlap)
        self.assertEqual(plan.track.evaluate(.2).zoom, 1.4)

    def test_exact_side_samples_stay_on_measured_states(self):
        plan = self.build()
        for delta in (.5/30, 3.5/30):
            self.assertEqual(plan.track.evaluate(delta).zoom, 1.4)
            self.assertEqual(plan.track.evaluate(2.-delta).zoom, 1.3)

    def test_identity_endpoint_does_not_get_face_tracking_scope(self):
        for entry, exit in ((1., 1.4), (1.4, 1.)):
            plan = self.build(entry=entry, exit=exit)
            self.assertIsNone(plan.active_side_at(0. if entry == 1. else 2.))
            self.assertIsNone(plan.active_side_at(1.))

    def test_identity_whole_segment_has_no_tracking_scope(self):
        plan = self.build(entry=1., exit=1.)
        self.assertTrue(all(plan.active_side_at(i/30) is None for i in range(61)))

    def test_short_different_states_interpolate_without_neutral_detour(self):
        plan = self.build(duration=.6)
        self.assertTrue(plan.handles_overlap)
        self.assertAlmostEqual(plan.track.evaluate(.3).zoom, 1.35)
        self.assertEqual(plan.active_side_at(.3), 'bridge')

    def test_incompatible_exact_holds_cannot_be_squeezed(self):
        with self.assertRaisesRegex(ValueError, 'too short'):
            self.build(duration=.2)

    def test_every_sample_is_renderer_valid_and_scoped(self):
        entry = EditorialTransformKeyframe(0., 1.8, .65, .42, 1.2)
        exit = EditorialTransformKeyframe(2., 1.4, .40, .52, -.6)
        plan = build_boundary_local_camera(entry=entry, exit=exit,
                    duration_seconds=2., output_fps=29.97)
        for i in range(600):
            state = plan.track.evaluate(i/300)
            self.assertEqual(plan.active_side_at(i/300) is None, is_identity(state))

    def test_keyframe_times_are_strict_and_inside_segment(self):
        for duration in (.24, .7, 1.04, 1.1, 2.):
            plan = self.build(duration=duration)
            times = [k.time_seconds for k in plan.track.keyframes]
            self.assertEqual(times, sorted(set(times)))
            self.assertEqual(times[0], 0.)
            self.assertEqual(times[-1], duration)

    def test_timing_domain_rejected(self):
        for kwargs in ({'hold_frames':-1.}, {'settle_frames':0.}, {'settle_frames':math.nan}):
            with self.assertRaises(ValueError):
                self.build(**kwargs)
        for seconds in (-.1, 2.1, math.nan):
            with self.assertRaises(ValueError):
                self.build().active_side_at(seconds)

    def test_endpoint_input_timestamps_do_not_change_relative_track(self):
        plan = build_boundary_local_camera(entry=EditorialTransformKeyframe(5.,1.4),
                    exit=EditorialTransformKeyframe(7.,1.3), duration_seconds=2., output_fps=30.)
        self.assertEqual(plan.track.keyframes[0].time_seconds, 0.)
        self.assertEqual(plan.track.keyframes[-1].time_seconds, 2.)


if __name__ == '__main__':
    unittest.main()
