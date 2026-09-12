from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.direct_member_rendered_continuity import evaluate_direct_member_rendered_continuity
from crosscut.rendered_body_reference import RenderedBodyMeasurement
from crosscut.rendered_body_window import RenderedBodyWindowProvider
from crosscut.direct_member_target_face import FocalFaceCandidate
from engine.tests import test_rendered_body_reference as body_fixture
from engine.tests.test_direct_member_rendered_continuity import _plan, _effect_plan, _descriptor, _provider, _FocalProvider


class RenderedBodyIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture = body_fixture.RenderedBodyReferenceTests()
        fixture.setUp()
        self.observed = RenderedBodyMeasurement('observed',
            fixture.evidence.as_formation_person(.9), fixture.evidence, .91, .2, 2, .02)

    def evaluate(self, plan, body):
        descriptors = {i: _descriptor(position=.3, pose=.2, include_face=False)
                       for i in range(plan['output_frame_count'])}
        return evaluate_direct_member_rendered_continuity(Path('unused.mp4'), plan,
            descriptor_provider=_provider(descriptors), body_window_provider=body)

    def test_hidden_faces_use_body_on_all_four_frames_without_inventing_face_coverage(self):
        body = Mock(return_value=(self.observed,) * 4)
        result = self.evaluate(_plan(), body)
        body.assert_called_once_with((2, 3, 4, 5))
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.metrics['body_reference_guided_boundary_count'], 1)
        self.assertEqual(result.metrics['focal_face_reference_guided_fraction'], 0.)
        self.assertEqual(result.metrics['target_reference_guided_fraction'], 1.)
        self.assertEqual(result.metrics['actual_pose_coverage_fraction'], 1.)
        self.assertNotIn('observed_output_pose_xyc', result.boundaries[0]['body_reference_measurements'][0])

    def test_complete_source_guided_face_window_does_not_run_body_inference(self):
        face = FocalFaceCandidate((.3, .28, .1, .1), (1., 0., 0.))
        descriptors = {i: _descriptor(position=.3, pose=.2) for i in range(8)}
        body = Mock()
        result = evaluate_direct_member_rendered_continuity(Path('unused.mp4'), _plan(),
            descriptor_provider=_FocalProvider(descriptors, {i: (face,) for i in range(8)}),
            focal_reference_candidates_by_boundary_index={0: (face, face)},
            focal_geometry_provider=lambda *_: self.observed_geometry(), body_window_provider=body)
        body.assert_not_called()
        self.assertEqual(result.metrics['focal_face_reference_guided_fraction'], 1.)
        self.assertEqual(result.metrics['body_reference_guided_boundary_count'], 0)

    def observed_geometry(self):
        from crosscut.rendered_body_window import RenderedBodyGeometry
        return RenderedBodyGeometry(self.observed, 0.)

    def test_one_unresolved_body_cannot_authorize_a_four_frame_window(self):
        body = Mock(return_value=(self.observed, self.observed,
            RenderedBodyMeasurement('output_target_body_ambiguous'), self.observed))
        result = self.evaluate(_plan(), body)
        self.assertFalse(result.passed)
        self.assertIn('rendered_boundary_target_reference_unresolved', result.failure_reasons)
        self.assertEqual(result.metrics['target_reference_guided_fraction'], 0.)
        self.assertEqual(result.metrics['actual_pose_coverage_fraction'], 0.)

    def test_effect_checks_only_pure_body_endpoints(self):
        body = Mock(return_value=(self.observed,) * 4)
        result = self.evaluate(_effect_plan(), body)
        body.assert_called_once_with((2, 3, 7, 7))
        self.assertEqual(result.metrics['effect_body_reference_guided_boundary_count'], 1)
        self.assertNotIn('rendered_boundary_target_reference_unresolved', result.failure_reasons)

    def test_body_match_does_not_disable_pose_discontinuity_gate(self):
        pose = replace(self.observed.pose,
            upper_body_xyc=tuple((1-x, y, c) for x, y, c in self.observed.pose.upper_body_xyc))
        other = replace(self.observed, pose=pose, person=pose.as_formation_person(.9))
        result = self.evaluate(_plan(), Mock(return_value=(self.observed, self.observed, other, other)))
        self.assertIn('rendered_boundary_pose_discontinuity', result.failure_reasons)

    def test_typed_body_measurements_are_required(self):
        with self.assertRaisesRegex(ValueError, 'typed measurements'):
            self.evaluate(_plan(), Mock(return_value=(None,) * 4))


class BodyWindowClockTests(unittest.TestCase):
    def plan(self):
        plan = _plan()
        plan.update(width=1080, height=1920, sources=['first.mp4', 'second.mp4'])
        plan['segments'] = [dict(s, source_local_start=10.001, source_playback_rate=1.)
                            for s in plan['segments']]
        return plan

    def test_trim_uses_ceil_start_plus_relative_frames_and_validates_source_on_cache_hit(self):
        source = Mock()
        source.faces._samplers = [SimpleNamespace(frame_rate=30.)] * 2
        provider = RenderedBodyWindowProvider(self.plan(), Mock(), body_model_path=None,
            pose_model_path=None, source_provider=source)
        measured = RenderedBodyMeasurement('output_body_unobserved')
        provider._cache[3] = measured
        self.assertIs(provider.measurement(3), measured)
        source.selector_geometry.assert_called_once_with(0, 310/30.)
        source.selector_geometry.side_effect = OSError('SSD disconnected')
        with self.assertRaisesRegex(OSError, 'SSD disconnected'):
            provider.measurement(3)

    def test_blended_frame_is_rejected_before_opening_source_or_models(self):
        plan = self.plan()
        plan['transitions'] = [{'kind': 'member_zoom_bridge', 'frame_count': 3}]
        provider = RenderedBodyWindowProvider(plan, Mock(), body_model_path=None,
            pose_model_path=None)
        with patch.object(provider, '_open') as opened:
            result = provider.measurement(4)
        self.assertEqual(result.reason, 'blended_output_has_no_single_source_owner')
        opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
