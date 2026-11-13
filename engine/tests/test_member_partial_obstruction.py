from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    DirectMemberVisualSelection,
    _BoundarySample,
    _review_transient_partial_obstruction,
)
from engine.tests.test_direct_member_visual import _descriptor


class PartialObstructionTests(unittest.TestCase):
    def review(self, *, duration=1., hits=(2,), flank=None, center=None):
        clean = _descriptor(.2)
        obstructed = center or replace(clean, foreground_obstruction_score=.94)
        samples = tuple(_BoundarySample(t * duration, t * duration,
            obstructed if i in hits else clean, None)
            for i, t in enumerate((.05, .25, .5, .7, .85, .95)))
        calls = []

        def probe(seconds):
            calls.append(seconds)
            descriptor = flank(seconds) if flank is not None else clean
            return _BoundarySample(seconds, seconds, descriptor, None)

        result = _review_transient_partial_obstruction(samples,
            span_start=0., span_end=duration, sample_at=probe,
            config=DirectMemberVisualConfig())
        return result, calls

    def test_bracket_limit_is_not_reported_as_measured_duration(self):
        for duration in (.2, 1., 4., 10.):
            with self.subTest(duration=duration):
                report, calls = self.review(duration=duration)
                self.assertIsNotNone(report)
                self.assertEqual(len(calls), 2)
                self.assertLessEqual(report['sampled_window_seconds'], .400000001)
                self.assertLessEqual(report['sampled_window_span_fraction'], .200000001)
                self.assertFalse(report['continuous_duration_measured'])
                self.assertFalse(report['target_visibility_verified'])
                self.assertFalse(report['target_identity_or_cut_quality_authorized'])

    def test_no_extra_decode_for_clear_multiple_or_endpoint_obstruction(self):
        for hits in ((), (0,), (5,), (1, 2)):
            with self.subTest(hits=hits):
                report, calls = self.review(hits=hits)
                self.assertIsNone(report)
                self.assertEqual(calls, [])

    def test_missing_low_quality_or_still_obstructed_flank_cannot_pass(self):
        for descriptor in (None, _descriptor(.2, quality_score=.1),
                           _descriptor(.2, foreground_obstruction_score=.94)):
            with self.subTest(descriptor=descriptor):
                report, calls = self.review(flank=lambda seconds: descriptor)
                self.assertIsNone(report)
                self.assertEqual(len(calls), 2)

    def test_visible_body_without_face_is_not_a_whole_source_rejection(self):
        body = replace(_descriptor(.2), face_count=0, primary_face_bbox=None)
        report, _ = self.review(center=replace(body, foreground_obstruction_score=.94),
                                flank=lambda seconds: body)
        self.assertIsNotNone(report)
        self.assertFalse(report['target_identity_or_cut_quality_authorized'])

    def test_summary_counts_only_selected_spans_and_keeps_duration_unknown(self):
        review, _ = self.review()
        marked = SimpleNamespace(transient_obstruction_review=review)
        clear = SimpleNamespace(transient_obstruction_review=None)
        spans = tuple(SimpleNamespace(segment_index=i, canonical_master_start=float(i),
            canonical_master_end=float(i+1), sources=(marked, clear)) for i in range(5))
        selection = SimpleNamespace(span_validity=spans, source_indices=(0, 0, 1, 0, 1))
        report = DirectMemberVisualSelection.obstruction_diagnostics(selection)
        self.assertEqual(report['selected_tolerated_span_count'], 3)
        self.assertEqual(report['selected_tolerated_span_fraction'], .6)
        self.assertEqual(report['maximum_consecutive_tolerated_spans'], 2)
        self.assertAlmostEqual(report['selected_selection_penalty'], .3)
        self.assertAlmostEqual(report['sampled_bracket_window_sum_seconds'], .6)
        self.assertIsNone(report['continuous_obstruction_duration_seconds'])
        self.assertFalse(report['rendered_output_evaluated'])
        self.assertEqual([item['segment_index'] for item in report['selected_observations']], [0, 1, 3])


if __name__ == '__main__':
    unittest.main()
