from dataclasses import replace
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import random
import tempfile
import unittest

from crosscut.reference_crop_phase_lattice import CropPhaseCandidate as C, CropPhaseObservation as O
from crosscut.reference_crop_source_clock import (
    CropClockBounds, CropClockSearchLimitExceeded, resolve_bounded_rate_source_clock,
    phase_lattice_from_crop_payload,
)


def c(index, score=.9, geometry=(1., .5, .5, 0.)):
    return C(index, score, geometry)


def path_rank(rows, path):
    phases = [candidate.source_frame_index-row.canonical_frame_position for row, candidate in zip(rows, path)]
    return (-sum(candidate.warped_structure_ncc for candidate in path),
            sum(abs(b-a) for a, b in zip(phases, phases[1:])), sum(abs(x) for x in phases),
            tuple(candidate.source_frame_index for candidate in path))


def brute_feasible(rows, path, bounds):
    # Independent interval arithmetic for each complete enumerated path.
    lower = max(0., path[0].source_frame_index-.5+1e-10)
    upper = path[0].source_frame_index+.5-1e-10
    for a, b, candidate in zip(rows, rows[1:], path[1:]):
        dt = (b.time_seconds-a.time_seconds)*bounds.source_frame_rate
        lower = max(candidate.source_frame_index-.5+1e-10, lower+dt*bounds.minimum_playback_rate)
        upper = min(candidate.source_frame_index+.5-1e-10, upper+dt*bounds.maximum_playback_rate)
        if lower > upper:
            return False
    return True


class CropSourceClockTests(unittest.TestCase):
    def test_repeated_frame_can_have_a_moving_subframe_clock(self):
        rows = (O(0., 10., (c(10),)), O(1/60, 10.5, (c(10),)))
        result = resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(30., 1., 1.))
        self.assertIsNotNone(result)
        self.assertEqual(tuple(x.source_frame_index for x in result.candidates), (10, 10))
        self.assertAlmostEqual(result.playback_rates[0], 1.)
        self.assertGreater(result.source_frame_positions[1], result.source_frame_positions[0])
        self.assertFalse(result.to_dict()['retiming_authorized'])

    def test_long_freeze_is_not_made_valid_by_nondecreasing_ids(self):
        rows = (O(0., 10., (c(10),)), O(1., 40., (c(10),)))
        self.assertIsNone(resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(30., .9, 1.1)))

    def test_excessive_source_jump_keeps_measured_slower_alternative(self):
        alternative = c(11, .8)
        rows = (O(0., 10., (c(10),)), O(.04, 11.2, (c(16, 1.), alternative)))
        result = resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(30., .9, 1.1))
        self.assertEqual(result.candidates[-1], alternative)

    def test_half_frame_ties_are_avoided_in_both_samplers(self):
        rows = (O(0., 10.5, (c(10),)), O(1/30, 11.5, (c(11),)))
        result = resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(30., 1., 1.))
        for position, candidate in zip(result.source_frame_positions, result.candidates):
            self.assertEqual(round(position), candidate.source_frame_index)
            self.assertEqual(math.floor(position+.5), candidate.source_frame_index)

    def test_empty_candidate_layer_is_not_dropped(self):
        self.assertIsNone(resolve_bounded_rate_source_clock(
            (O(0., 0., (c(0),)), O(1., 30., ())), bounds=CropClockBounds(30., .9, 1.1)))

    def test_renderer_unsafe_edge_keeps_existing_safe_candidate(self):
        left = c(0, .9, (1.0325846172744377, .49955211673394156, .5002950952466916, .5404180985226001))
        unsafe = c(30, 1., (1.0872770599014896, .5099961525870863, .49354861234563674, 1.568702272916933))
        alternative = c(31, .8, left.geometry)
        result = resolve_bounded_rate_source_clock(
            (O(0., 0., (left,)), O(1., 30., (unsafe, alternative))), bounds=CropClockBounds(30., .9, 1.1))
        self.assertEqual(result.candidates, (left, alternative))

    def test_observation_order_and_real_zero_are_required(self):
        for rows in ((O(.1, 0., (c(0),)),), (O(0., 0., (c(0),)), O(0., 1., (c(1),)))):
            with self.assertRaises(ValueError):
                resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(30., .9, 1.1))

    def test_resource_exhaustion_is_not_infeasibility(self):
        rows = (O(0., 0., (c(0), c(1))), O(1., 2., (c(1), c(2))))
        with self.assertRaises(CropClockSearchLimitExceeded):
            resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(2., .5, 1.5, maximum_extension_count=1))
        with self.assertRaises(CropClockSearchLimitExceeded):
            resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(2., .5, 1.5, maximum_state_count=1))

    def test_continuous_witness_tampering_is_rejected(self):
        result = resolve_bounded_rate_source_clock((O(0., 10., (c(10),)), O(1., 40., (c(40),))),
                                                  bounds=CropClockBounds(30., .99, 1.01))
        with self.assertRaises(ValueError):
            replace(result, source_frame_positions=(9., 40.))
        with self.assertRaises(ValueError):
            replace(result, source_frame_positions=(9.51, 40.49))
        for change in ({'source_frame_positions': (True, 40.)},
                       {'maximum_frontier_size': -1}, {'explored_extensions': True}):
            with self.assertRaises(ValueError):
                replace(result, **change)

    def test_invalid_rate_and_resource_bounds_are_rejected(self):
        for changes in ({'source_frame_rate': 0}, {'source_frame_rate': True},
                        {'minimum_playback_rate': float('nan')},
                        {'minimum_playback_rate': 1.2},
                        {'maximum_playback_rate': float('inf')},
                        {'maximum_state_count': 1.2}, {'maximum_extension_count': False}):
            with self.assertRaises(ValueError):
                CropClockBounds(**(dict(source_frame_rate=30., minimum_playback_rate=.9,
                                       maximum_playback_rate=1.1) | changes))

    def test_deterministic_small_lattices_match_exhaustive_optimum(self):
        rng = random.Random(303091)
        bounds = CropClockBounds(3., .6, 1.4)
        for case in range(240):
            rows = tuple(O(index*.41, index*1.23, tuple(c(n, rng.choice((.5, .75, 1.)))
                for n in sorted(rng.sample(range(7), rng.randint(1, 4))))) for index in range(4))
            paths = [path for path in product(*(row.candidates for row in rows)) if brute_feasible(rows, path, bounds)]
            actual = resolve_bounded_rate_source_clock(rows, bounds=bounds)
            if not paths:
                self.assertIsNone(actual, case)
            else:
                self.assertEqual(actual.candidates, min(paths, key=lambda path: path_rank(rows, path)), case)
                self.assertEqual(len(actual.observations), len(rows))


class CropSourceClockArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from crosscut.reference_dynamic_crop_supervision import build_reference_dynamic_crop_supervision
        from tests.test_reference_dynamic_crop_supervision import (
            _write_dynamic_pair, _write_canonical_inputs, _request, _config,
        )
        with tempfile.TemporaryDirectory(prefix='crop-source-clock-') as directory:
            root = Path(directory)
            identity = np.asarray(((1., 0., 0.), (0., 1., 0.)))
            reference, source = _write_dynamic_pair(root, [identity]*18)
            attribution, evidence, shot = _write_canonical_inputs(root, reference, source)
            cls.payload = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source), config=_config()).payload

    def test_real_builder_preserves_measured_candidates_and_every_teacher_observation(self):
        original = deepcopy(self.payload)
        rows = phase_lattice_from_crop_payload(self.payload)
        frames = self.payload['supervision']['renderer_keyframes']
        self.assertEqual(len(rows), len(frames))
        self.assertEqual(tuple(row.time_seconds for row in rows),
                         tuple(frame['time_within_shot_seconds'] for frame in frames))
        for row, frame in zip(rows, frames):
            selected = next(c for c in row.candidates if c.source_frame_index == frame['source_frame_index'])
            self.assertEqual(selected.geometry, (1., .5, .5, 0.))
        result = resolve_bounded_rate_source_clock(rows, bounds=CropClockBounds(
            self.payload['provenance']['source_media']['frame_rate'], .9, 1.1))
        self.assertIsNotNone(result)
        self.assertEqual(self.payload, original)
        self.assertFalse(result.to_dict()['labels_modified'])

    def test_changed_content_hash_and_removed_measurements_fail_closed(self):
        from crosscut.reference_dynamic_crop_supervision import _artifact_content_sha256
        changed = deepcopy(self.payload)
        changed['supervision']['renderer_keyframes'][0]['zoom'] = 1.01
        with self.assertRaises(ValueError):
            phase_lattice_from_crop_payload(changed)
        changed = deepcopy(self.payload)
        for sample in changed['samples']:
            if sample['accepted']:
                sample['audit']['local_visual_search']['candidates'] = []
                break
        changed['provenance']['artifact_content_sha256'] = _artifact_content_sha256(changed)
        with self.assertRaises(ValueError):
            phase_lattice_from_crop_payload(changed)


if __name__ == '__main__':
    unittest.main()
