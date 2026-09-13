from itertools import product
import random
import unittest
from crosscut.reference_crop_phase_lattice import (
    CropPhaseCandidate as C, CropPhaseObservation as O,
    resolve_monotone_source_phase,
)


class MonotonePhaseTests(unittest.TestCase):
    def test_best_backward_argmax_is_replaced_with_measured_forward_candidate(self):
        a, b, c = C(3, .9, (1., .5, .5, 0.)), C(2, 1., (1.2, .4, .5, 0.)), C(4, .8, (1.1, .6, .5, 0.))
        p = resolve_monotone_source_phase((O(0., 3., (a,)), O(1., 4., (b,c))))
        self.assertEqual(p.candidates, (a,c))
        self.assertEqual(p.candidates[-1].geometry, c.geometry)
        self.assertAlmostEqual(p.mean_warped_structure_ncc, .85)

    def test_empty_or_backward_only_layer_remains_unresolved(self):
        a, b = C(3, .9, (1., .5, .5, 0.)), C(2, 1., (1.2, .4, .5, 0.))
        for candidates in ((), (b,)):
            self.assertIsNone(resolve_monotone_source_phase((O(0.,3.,(a,)), O(1.,4.,candidates))))

    def test_repeat_is_reported_not_misrepresented_as_valid_retiming(self):
        a = C(3,.9,(1.,.5,.5,0.))
        p = resolve_monotone_source_phase((O(0.,3.,(a,)), O(1.,4.,(a,))))
        self.assertEqual(p.repeated_adjacent_frame_count, 1)
        self.assertEqual(p.residual_phase_frames, (0.,-1.))
        self.assertEqual(p.phase_total_variation_frames, 1.)

    def test_small_lattices_match_exhaustive_optimum_and_tie_breaks(self):
        rng = random.Random(1719)
        for case in range(100):
            rows = tuple(O(float(i),float(i),tuple(C(n,rng.choice((.5,.75,1.)),(1.,.5,.5,0.))
                         for n in sorted(rng.sample(range(6),rng.randint(1,4))))) for i in range(4))
            possible = [path for path in product(*(r.candidates for r in rows))
                        if all(a.source_frame_index <= b.source_frame_index for a,b in zip(path,path[1:]))]
            actual = resolve_monotone_source_phase(rows)
            if not possible:
                self.assertIsNone(actual)
                continue
            def rank(path):
                phases=[c.source_frame_index-r.canonical_frame_position for c,r in zip(path,rows)]
                return (-sum(c.warped_structure_ncc for c in path),
                        sum(abs(a-b) for a,b in zip(phases,phases[1:])),
                        sum(abs(p) for p in phases), tuple(c.source_frame_index for c in path))
            self.assertEqual(actual.candidates,min(possible,key=rank),case)

    def test_clock_order_is_not_silently_sorted(self):
        c = C(0,.5,(1.,.5,.5,0.))
        with self.assertRaises(ValueError):
            resolve_monotone_source_phase((O(1.,1.,(c,)),O(0.,0.,(c,))))


if __name__ == '__main__':
    unittest.main()
