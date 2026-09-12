import itertools
import math
import random
import unittest

from crosscut.reference_crop_phase_lattice import (
    CropPhaseCandidate as Candidate, CropPhaseObservation as Observation,
    resolve_constant_source_phase,
)


def candidate(index, score):
    return Candidate(index, score, (1.02, .5, .5, 0.))


class CropPhaseLatticeTests(unittest.TestCase):
    def test_backward_independent_winners_are_replaced_jointly_not_clamped(self):
        rows = (Observation(0., 10., (candidate(10, .8), candidate(11, .95))),
                Observation(.02, 10.6, (candidate(10, .99), candidate(11, .9), candidate(12, .4))))
        result = resolve_constant_source_phase(rows)
        self.assertEqual(tuple(c.source_frame_index for c in result.candidates), (11, 11))
        self.assertEqual(tuple(c.source_frame_index for c in rows[1].candidates), (10, 11, 12))
        self.assertAlmostEqual(result.mean_warped_structure_ncc, .925)

    def test_empty_or_incompatible_observation_cannot_be_dropped(self):
        self.assertIsNone(resolve_constant_source_phase((Observation(0., 10., (candidate(11,.9),)),
                                                        Observation(1., 40., ()))))
        self.assertIsNone(resolve_constant_source_phase((Observation(0., 10., (candidate(11,.9),)),
                                                        Observation(1., 40., (candidate(37,.9),)))))

    def test_geometry_is_taken_from_same_measured_frame(self):
        chosen = Candidate(12, .99, (1.1,.52,.5,.2))
        result = resolve_constant_source_phase((Observation(0., 10., (candidate(10,.8),chosen)),))
        self.assertIs(result.candidates[0], chosen)
        self.assertEqual(result.candidates[0].geometry, chosen.geometry)

    def test_forward_phase_may_repeat_frames_but_never_reverse(self):
        rows = tuple(Observation(i/60, 10+i/2, tuple(candidate(n,.8) for n in range(10,15))) for i in range(5))
        result = resolve_constant_source_phase(rows)
        indices = tuple(c.source_frame_index for c in result.candidates)
        self.assertEqual(indices, tuple(sorted(indices)))
        self.assertLess(len(set(indices)), len(indices))

    def test_matches_exhaustive_candidate_path_interval_intersection(self):
        rng = random.Random(13917)
        for _ in range(100):
            positions = (10.13, 10.79, 12.08, 15.51)
            rows = tuple(Observation(float(i), x, tuple(candidate(n, rng.uniform(.4,.99))
                         for n in range(math.floor(x)-1, math.floor(x)+3) if rng.random()>.2))
                         for i,x in enumerate(positions))
            feasible=[]
            for path in itertools.product(*(row.candidates for row in rows)):
                lower=max(c.source_frame_index-.5-row.canonical_frame_position for row,c in zip(rows,path))
                upper=min(c.source_frame_index+.5-row.canonical_frame_position for row,c in zip(rows,path))
                if lower < upper:
                    feasible.append(sum(c.warped_structure_ncc for c in path)/len(rows))
            result=resolve_constant_source_phase(rows)
            if feasible:
                self.assertAlmostEqual(result.mean_warped_structure_ncc,max(feasible))
            else:
                self.assertIsNone(result)

    def test_rejects_bad_clock_or_duplicate_frames(self):
        with self.assertRaises(ValueError):
            resolve_constant_source_phase((Observation(1.,10.,(candidate(10,.9),)),Observation(1.,11.,(candidate(11,.9),))))
        with self.assertRaises(ValueError):
            Observation(0.,10.,(candidate(10,.9),candidate(10,.8)))


if __name__ == '__main__':
    unittest.main()
