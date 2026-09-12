from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, DirectMemberVisualSelectionError, _boundary_endpoint_content_eligibility,
    select_balanced_direct_member_sources,
)
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_source_first_focal_geometry import geometry


class BoundaryObservationPruningTests(unittest.TestCase):
    def test_overlap_handles_survive_an_invalid_next_full_span(self):
        self.assertEqual(_boundary_endpoint_content_eligibility(True, False),
                         (True, True, False, True, True))
        self.assertEqual(_boundary_endpoint_content_eligibility(False, True),
                         (False, False, True, True, True))
        self.assertEqual(_boundary_endpoint_content_eligibility(False, False), (False,) * 5)
        self.assertEqual(_boundary_endpoint_content_eligibility(True, True), (True,) * 5)

    def run_selection(self, *, exhaustive=False, partial=False):
        raw_calls, focal_calls = [], []
        def provider(source, path, seconds):
            raw_calls.append((source, seconds))
            blocked = source == 0 and (0.48 < seconds < .52 if partial else 0 < seconds < 2.9)
            return _descriptor(.2 + .02*source, foreground_obstruction_score=.94 if blocked else 0.)
        def focal(source, seconds, descriptor):
            focal_calls.append((source, seconds))
            return replace(geometry(source), source_seconds=seconds)
        provider.focal_geometry_at = focal
        args = dict(sources=tuple(Path(f'{i}.mp4') for i in range(3)),
            canonical_spans=tuple((float(i), float(i+1)) for i in range(6)),
            audio_offsets=(0.,)*3, descriptor_provider=provider,
            config=DirectMemberVisualConfig(beam_width=1000))
        if exhaustive:
            with patch('crosscut.direct_member_visual._boundary_endpoint_content_eligibility',
                       return_value=(True,)*5):
                result = select_balanced_direct_member_sources(**args)
        else:
            result = select_balanced_direct_member_sources(**args)
        return result, raw_calls, focal_calls

    def test_actual_selector_preserves_legal_path_and_effect_handles_with_fewer_observations(self):
        result, raw, focal = self.run_selection()
        exhaustive, all_raw, all_focal = self.run_selection(exhaustive=True)
        self.assertEqual(result.source_indices, exhaustive.source_indices)
        self.assertEqual(result.selected_boundary_master_seconds, exhaustive.selected_boundary_master_seconds)
        self.assertEqual(result.optimizer_cost, exhaustive.optimizer_cost)
        self.assertEqual(result.span_validity, exhaustive.span_validity)
        for left, right in zip(result.boundaries, exhaustive.boundaries, strict=True):
            self.assertEqual(left.outgoing_sample, right.outgoing_sample)
            self.assertEqual(left.incoming_sample, right.incoming_sample)
            self.assertEqual(left.outgoing_handle_samples, right.outgoing_handle_samples)
            self.assertEqual(left.incoming_handle_samples, right.incoming_handle_samples)
        skipped = result.content_invalid_boundary_samples_skipped
        self.assertGreater(skipped, 0)
        self.assertEqual(len(all_raw) - len(raw), skipped)
        self.assertEqual(len(all_focal) - len(focal), skipped)
        self.assertEqual(result.failed_sample_count, exhaustive.failed_sample_count)

    def test_tolerated_brief_occlusion_keeps_all_exact_observations(self):
        result, raw, focal = self.run_selection(partial=True)
        exhaustive, all_raw, all_focal = self.run_selection(partial=True, exhaustive=True)
        self.assertTrue(result.span_validity[0].sources[0].valid)
        self.assertIsNotNone(result.span_validity[0].sources[0].transient_obstruction_review)
        self.assertEqual(result.content_invalid_boundary_samples_skipped, 0)
        self.assertEqual(raw, all_raw)
        self.assertEqual(focal, all_focal)
        self.assertEqual(result.source_indices, exhaustive.source_indices)

    def test_proven_empty_content_span_fails_before_any_boundary_model_work(self):
        calls = []
        def provider(source, path, seconds):
            return _descriptor(.2, quality_score=0. if 0 < seconds < 1. else 1.)
        def focal(*args):
            calls.append(args)
            raise AssertionError('unreachable boundary model work')
        provider.focal_geometry_at = focal
        with self.assertRaises(DirectMemberVisualSelectionError) as caught:
            select_balanced_direct_member_sources(
                sources=(Path('a'), Path('b')), canonical_spans=((0., 1.), (1., 2.)),
                audio_offsets=(0., 0.), descriptor_provider=provider)
        self.assertEqual(caught.exception.diagnostics['empty_segment_indices'], [0])
        self.assertFalse(caught.exception.diagnostics['observation_work']['boundary_observations_evaluated'])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
