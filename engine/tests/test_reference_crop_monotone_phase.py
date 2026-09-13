from itertools import product
import random
import unittest
from crosscut.reference_crop_phase_lattice import (
    CropPhaseCandidate as C, CropPhaseObservation as O,
    resolve_monotone_source_phase, resolve_renderer_monotone_source_phase,
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

    def test_edge_predicate_keeps_alternative_prefix_needed_by_future_edge(self):
        rows = tuple(O(float(i),float(i),tuple(C(n,score,(1.,.5,.5,0.)) for n,score in layer))
                     for i,layer in enumerate((((0,.9),(1,.8)),((2,.9),(3,.8)),((4,.9),))))
        allowed = {(0,2),(1,3),(3,4)}
        result = resolve_monotone_source_phase(rows, edge_allowed=lambda a,ca,b,cb:
                                              (ca.source_frame_index,cb.source_frame_index) in allowed)
        self.assertEqual(tuple(c.source_frame_index for c in result.candidates),(1,3,4))

    def test_edge_constrained_lattices_match_exhaustive_search(self):
        rng = random.Random(2241)
        for _ in range(80):
            rows = tuple(O(float(i),float(i),tuple(C(n,rng.choice((.5,.75,1.)),(1.,.5,.5,0.))
                         for n in sorted(rng.sample(range(6),3)))) for i in range(4))
            edges = {(i,a.source_frame_index,b.source_frame_index)
                     for i,(left,right) in enumerate(zip(rows,rows[1:]))
                     for a in left.candidates for b in right.candidates if rng.random()>.4}
            allowed = lambda a,ca,b,cb: (int(a.time_seconds),ca.source_frame_index,cb.source_frame_index) in edges
            possible = [path for path in product(*(r.candidates for r in rows))
                        if all(a.source_frame_index<=b.source_frame_index and allowed(rows[i],a,rows[i+1],b)
                               for i,(a,b) in enumerate(zip(path,path[1:])))]
            actual = resolve_monotone_source_phase(rows,edge_allowed=allowed)
            if not possible:
                self.assertIsNone(actual)
                continue
            def rank(path):
                phases=[c.source_frame_index-r.canonical_frame_position for c,r in zip(path,rows)]
                return (-sum(c.warped_structure_ncc for c in path),
                        sum(abs(a-b) for a,b in zip(phases,phases[1:])),
                        sum(abs(p) for p in phases),tuple(c.source_frame_index for c in path))
            self.assertEqual(actual.candidates,min(possible,key=rank))

    def test_renderer_selects_measured_alternative_instead_of_rejecting_winner(self):
        left = C(0,.9,(1.0325846172744377,.49955211673394156,.5002950952466916,.5404180985226001))
        unsafe = C(30,1.,(1.0872770599014896,.5099961525870863,.49354861234563674,1.568702272916933))
        alternative = C(31,.8,left.geometry)
        rows=(O(0.,0.,(left,)),O(1.,30.,(unsafe,alternative)))
        self.assertEqual(resolve_monotone_source_phase(rows).candidates,(left,unsafe))
        proposal=resolve_renderer_monotone_source_phase(rows)
        self.assertEqual(proposal.candidates,(left,alternative))
        self.assertIn('renderer_interpolable',proposal.contract)
        self.assertIsNone(resolve_renderer_monotone_source_phase((rows[0],O(1.,30.,(unsafe,)))))

    def test_renderer_never_clamps_bad_evidence_or_synthesizes_time_zero(self):
        c=C(0,.9,(1.,.5,.5,0.))
        with self.assertRaises(ValueError):
            resolve_renderer_monotone_source_phase((O(.1,0.,(c,)),))
        with self.assertRaises(ValueError):
            resolve_renderer_monotone_source_phase((O(0.,0.,(C(0,.9,(.9,.5,.5,0.)),)),))
        self.assertEqual(resolve_renderer_monotone_source_phase((O(0.,0.,(c,)),)).candidates,(c,))


if __name__ == '__main__':
    unittest.main()
