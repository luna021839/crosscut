from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _DirectBoundarySearchCandidate,
    _boundary_sample_with_focal_geometry, _boundary_mismatch,
    _observe_member_boundary_pair, _scored_boundary_search_mismatch,
)
from crosscut.member_boundary_observations import MemberBoundaryObservationStore
from crosscut.member_observation_search import MemberObservationSearch, TransitionObservation
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_source_first_focal_geometry import geometry
from engine.tests.test_member_boundary_observations import plan


class MemberPairObservationTests(unittest.TestCase):
    def store(self):
        plans = tuple(replace(plan(offset), eligibility_by_source=((True,)*5,)*4)
                      for offset in (0, -2, 2))
        def observe(p, source):
            provider = SimpleNamespace(focal_geometry_at=lambda s, t, raw:
                replace(geometry(s), source_seconds=t))
            return tuple(_boundary_sample_with_focal_geometry(
                master_seconds=master, source_local_seconds=local, source_index=source,
                descriptor=_descriptor(.2+.02*source), provider=provider)
                for master, local in zip(p.master_times, p.local_times_by_source[source]))
        observer, prepare = Mock(side_effect=observe), Mock()
        return MemberBoundaryObservationStore(plans, observe=observer, prepare=prepare), observer, prepare

    def pair(self, store, left=0, right=3, adjust=None):
        return _observe_member_boundary_pair(store,
            outgoing_source_index=left, incoming_source_index=right,
            source_frame_geometries=(None,)*4,
            continuous_official_portrait_fancam_contract=True,
            priority_boundary=True, config=DirectMemberVisualConfig(), adjust_mismatch=adjust)

    def test_exact_pair_scores_and_handles_equal_full_matrix_without_other_source_observations(self):
        sparse, observer, prepare = self.store()
        actual = self.pair(sparse)
        self.assertEqual(observer.call_count, 6)
        self.assertEqual({source for source, seconds in prepare.call_args.args[0]}, {0, 3})
        reference, _, _ = self.store()
        config = DirectMemberVisualConfig()
        for result in actual:
            p = result.plan
            samples = tuple(reference.samples_for(p.offset_frames, source) for source in range(4))
            matrix = tuple(tuple(_boundary_mismatch(samples[left][1], samples[right][2], config,
                outgoing_previous=samples[left][0], incoming_next=samples[right][3],
                outgoing_source_index=left, incoming_source_index=right,
                outgoing_source_geometry=None, incoming_source_geometry=None,
                continuous_official_portrait_fancam_contract=True)
                for right in range(4)) for left in range(4))
            candidate = _DirectBoundarySearchCandidate(p.offset_frames, p.cut_master_seconds,
                tuple(s[0] for s in samples), tuple(s[1] for s in samples), tuple(s[2] for s in samples),
                tuple(s[3] for s in samples), matrix, tuple(s[4] for s in samples))
            expected = _scored_boundary_search_mismatch(candidate, outgoing_source_index=0,
                incoming_source_index=3, candidate_count=3, priority_boundary=True, config=config)
            self.assertEqual(result.mismatch, expected)
            self.assertEqual(result.outgoing_samples, samples[0])
            self.assertEqual(result.incoming_samples, samples[3])
        # Shared sources are observed only once when a different pair is tried.
        self.pair(sparse, left=0, right=2)
        self.assertEqual(observer.call_count, 9)
        self.assertEqual({source for source, seconds in prepare.call_args.args[0]}, {2})
        self.assertNotIn((0, 1), sparse.observed_demands)

    def test_feedback_is_applied_to_the_exact_pair_and_clock_before_scoring(self):
        store, _, _ = self.store()
        seen = []
        def adjustment(cut, left, right, mismatch):
            seen.append((cut, left, right))
            return replace(mismatch, cost=mismatch.cost+2.,
                severe_flags=(*mismatch.severe_flags, 'actual_camera_failure'))
        results = self.pair(store, adjust=adjustment)
        self.assertEqual(seen, [(p.cut_master_seconds, 0, 3) for p in store.plans])
        self.assertTrue(all('actual_camera_failure' in item.mismatch.severe_flags for item in results))

    def test_hold_demand_observes_canonical_source_only_without_cross_source_claim(self):
        store, observer, prepare = self.store()
        result = self.pair(store, left=2, right=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].plan.offset_frames, 0)
        self.assertEqual(observer.call_count, 1)
        self.assertEqual(len(prepare.call_args.args[0]), 5)
        self.assertEqual(result[0].mismatch.boundary_search_candidate_count, 3)

    def test_coordinator_observes_complete_proposed_path_with_actual_pair_scorer(self):
        stores = [self.store() for _ in range(3)]
        search = MemberObservationSearch(((True,)*4,)*4,
            maximum_observations=6, maximum_solver_calls=1)
        def observe(demand):
            candidates = self.pair(stores[demand.boundary_index][0],
                left=demand.outgoing_source_index, right=demand.incoming_source_index)
            best = min(candidates, key=lambda item: (bool(item.mismatch.severe_flags), item.mismatch.cost))
            return TransitionObservation(demand, not best.mismatch.severe_flags,
                best.mismatch.cost, candidates)
        result = search.run(propose=lambda _: (0, 3, 2, 1), observe=observe)
        self.assertEqual(result.observation_count, 3)
        self.assertEqual(result.unobserved_transition_count, 33)
        self.assertTrue(all(len(item.evidence) == 3 for item in result.transitions))
        self.assertTrue(all(observer.call_count == 6 for _, observer, _ in stores))


if __name__ == '__main__': unittest.main()
