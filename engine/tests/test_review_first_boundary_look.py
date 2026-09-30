from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crosscut.boundary_look_match import (
    BoundaryLookDecision, BoundaryLookMatchConfig, BoundaryLookRenderAdmissionConfig,
    BoundaryLookMeasurementCache, _boundary_pair_render_plan,
    analyze_boundary_look_match, verify_boundary_look_match_render_domain,
)
from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment
from crosscut.review_first_boundary_look import refine_review_first_boundary_look
from tests.test_boundary_look_match import _MetricCache, _metric


def _review(plan, _cut_frame):
    adjusted = any(segment.boundary_look_track is not None for segment in plan.segments)
    return dict(discontinuities=dict(sharpness_discontinuity=.10,
        exposure_discontinuity=.05 if adjusted else .08,
        color_shock=.24 if adjusted else .30, structure_shock=.30,
        motion_direction_discontinuity=.20),
        frame_metrics=[dict(offset=i, mean_luma=.40 if i < 0 else .44,
                            clipped_pixel_ratio=.002) for i in (-2, -1, 0, 1)],
        technical_integrity_issue_reasons=[], issue_reasons=[])


class ReviewFirstBoundaryLookTests(unittest.TestCase):
    def test_pair_admission_cannot_approve_a_different_global_frame_clock(self):
        plan = RenderPlan(sources=tuple(Path(f'/synthetic/{i}.mp4') for i in range(3)),
            segments=tuple(RenderSegment(i, 0., .24, i*.24) for i in range(3)),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path('/synthetic/out.mp4'), fps=10.)
        with self.assertRaisesRegex(ValueError, 'global frame clock'):
            _boundary_pair_render_plan(plan, BoundaryLookDecision(1, 1, 2, True, True, 'fixture'),
                output=plan.output, adjusted=False, config=BoundaryLookMatchConfig(),
                admission=BoundaryLookRenderAdmissionConfig())

    def plan_and_proposal(self, output):
        plan = RenderPlan(sources=tuple(Path(f'/synthetic/{i}.mp4') for i in range(4)),
            segments=tuple(RenderSegment(i, 2., 1., float(i)) for i in range(4)),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE, output=output)
        cache = _MetricCache({f'{i}.mp4': (_metric(luma, .33, .38),)*7
                              for i, luma in enumerate((.38, .42, .51, .46))})
        proposed = analyze_boundary_look_match(plan, group_16x9_safe=True,
                                              measurement_cache=cache)
        self.assertEqual(proposed.applied_count, 3)
        return plan, proposed

    def test_work_limit_ranks_gain_and_cannot_apply_unverified_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, proposed = self.plan_and_proposal(Path(directory)/'out.mp4')
            calls = []
            def reviewer(pair, cut):
                calls.append(pair)
                return _review(pair, cut)
            result = verify_boundary_look_match_render_domain(plan, proposed,
                maximum_verified_boundaries=1, review_function=reviewer)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.applied_count, 1)
        self.assertEqual(result.render_verified_count, 1)
        expected = max(proposed.decisions, key=lambda row:
                       row.predicted_distance_before-row.predicted_distance_after)
        admitted = next(row for row in result.decisions if row.applied)
        self.assertEqual(admitted.boundary_index, expected.boundary_index)
        self.assertEqual(tuple(row.boundary_index for row in result.decisions), (0, 1, 2))
        skipped = [row for row in result.decisions if not row.applied]
        self.assertTrue(all(row.reason == 'render_verification_work_limit' for row in skipped))
        self.assertTrue(all(row.outgoing_adjustment.is_identity
                            and row.incoming_adjustment.is_identity for row in skipped))
        self.assertEqual(result.to_dict()['render_domain_admission']['maximum_verified_boundaries'], 1)

    def test_invalid_work_limits_do_not_start_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, proposed = self.plan_and_proposal(Path(directory)/'out.mp4')
            for limit in (True, 0, -1, 1.5, float('nan')):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    verify_boundary_look_match_render_domain(plan, proposed,
                        maximum_verified_boundaries=limit, review_function=_review)

    def test_clipping_backoff_is_single_bounded_attempt_and_keeps_both_receipts(self):
        for half_passes in (True, False):
            with self.subTest(half_passes=half_passes), tempfile.TemporaryDirectory() as directory:
                plan, proposed = self.plan_and_proposal(Path(directory)/'out.mp4')
                calls = []
                def reviewer(pair, cut):
                    calls.append(pair)
                    payload = _review(pair, cut)
                    if len(calls) == 2 or (len(calls) == 3 and not half_passes):
                        for frame in payload['frame_metrics']:
                            frame['clipped_pixel_ratio'] = .03
                    return payload
                result = verify_boundary_look_match_render_domain(plan, proposed,
                    maximum_verified_boundaries=1, clipping_backoff_enabled=True,
                    review_function=reviewer)
            self.assertEqual(len(calls), 3)
            decision = next(row for row in result.decisions if row.render_verification is not None)
            self.assertEqual([strength for strength, _ in decision.render_verification_attempts], [1., .5])
            self.assertEqual(decision.applied, half_passes)
            self.assertEqual(decision.render_admitted_strength, .5 if half_passes else None)
            self.assertFalse(decision.render_verification_attempts[0][1].passed)
            if half_passes:
                original = proposed.decisions[decision.boundary_index]
                self.assertAlmostEqual(decision.outgoing_adjustment.brightness,
                                       original.outgoing_adjustment.brightness*.5)
                self.assertGreater(decision.predicted_distance_after, original.predicted_distance_after)

    def test_backoff_does_not_retry_color_or_technical_regressions(self):
        for hazard in ('color', 'technical'):
            with self.subTest(hazard=hazard), tempfile.TemporaryDirectory() as directory:
                plan, proposed = self.plan_and_proposal(Path(directory)/'out.mp4')
                calls = []
                def reviewer(pair, cut):
                    calls.append(pair)
                    payload = _review(pair, cut)
                    if len(calls) == 2:
                        if hazard == 'color':
                            payload['discontinuities']['color_shock'] = .35
                        else:
                            payload['technical_integrity_issue_reasons'] = ['decode_failed']
                    return payload
                result = verify_boundary_look_match_render_domain(plan, proposed,
                    maximum_verified_boundaries=1, clipping_backoff_enabled=True,
                    review_function=reviewer)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result.applied_count, 0)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'needs ffmpeg')
    def test_verified_sequential_measurements_exactly_match_timestamp_seeks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=s=320x180:r=30000/1001', '-t', '2', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', str(source)], check=True, capture_output=True)
            times = (.01, .033367, .0667, .2002, .2003, .4004, 1.001, 1.020)
            ordinary, optimized = BoundaryLookMeasurementCache(), BoundaryLookMeasurementCache()
            with patch('crosscut.boundary_look_match.probe_constant_frame_clock', return_value=None):
                expected = ordinary.measure_many(source, times)
            actual = optimized.measure_many(source, times)
            self.assertEqual(actual, expected)
            self.assertTrue(all(value is not None for value in actual.values()))
            self.assertEqual(ordinary.decoded_frame_count, optimized.decoded_frame_count)

    def test_helper_preserves_all_nonlook_authority_and_requires_pixel_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, proposed = self.plan_and_proposal(Path(directory)/'out.mp4')
            admitted = verify_boundary_look_match_render_domain(plan, proposed,
                maximum_verified_boundaries=2, review_function=_review)
            with patch('crosscut.review_first_boundary_look.analyze_boundary_look_match',
                       return_value=proposed), patch(
                    'crosscut.review_first_boundary_look.verify_boundary_look_match_render_domain',
                    return_value=admitted) as verify:
                adjusted, audit = refine_review_first_boundary_look(plan, group_16x9_safe=True)
                verify.assert_called_once_with(plan, proposed, maximum_verified_boundaries=6,
                                               ffmpeg='ffmpeg', clipping_backoff_enabled=True)
            self.assertEqual(replace(adjusted, segments=tuple(
                replace(row, boundary_look_track=None) for row in adjusted.segments)), plan)
            self.assertEqual(audit.applied_count, 2)
            with patch('crosscut.review_first_boundary_look.analyze_boundary_look_match',
                       return_value=proposed), patch(
                    'crosscut.review_first_boundary_look.verify_boundary_look_match_render_domain',
                    return_value=proposed), self.assertRaisesRegex(ValueError, 'decoded proof'):
                refine_review_first_boundary_look(plan, group_16x9_safe=True)
            with self.assertRaisesRegex(ValueError, 'ungraded'):
                refine_review_first_boundary_look(adjusted, group_16x9_safe=True)


if __name__ == '__main__':
    unittest.main()
