from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

    def pair(self, store, left=0, right=3, adjust=None, observed_only=False, **kwargs):
        return _observe_member_boundary_pair(store,
            outgoing_source_index=left, incoming_source_index=right,
            source_frame_geometries=(None,)*4,
            continuous_official_portrait_fancam_contract=True,
            priority_boundary=True, config=DirectMemberVisualConfig(), adjust_mismatch=adjust,
            observed_only=observed_only, **kwargs)

    def test_progressive_offset_then_full_refinement_preserves_every_original_measurement(self):
        store, observer, prepare = self.store()
        first = self.pair(store, stop_after_first_eligible=True)
        self.assertEqual(len(first),1)
        self.assertFalse(first[0].mismatch.severe_flags)
        self.assertEqual(first[0].mismatch.boundary_search_candidate_count,1)
        self.assertEqual(observer.call_count,2)
        self.assertEqual(store.observed_demands,frozenset(((0,0),(0,3))))
        for offset in (-2,2):
            with self.assertRaises(KeyError): store.observed_samples(offset,0)
        observer.reset_mock()
        prepare.reset_mock()
        full=self.pair(store)
        self.assertEqual(len(full),3)
        self.assertEqual(observer.call_count,4)
        self.assertEqual(len(prepare.call_args.args[0]),20)
        self.assertEqual(full[0].outgoing_samples,first[0].outgoing_samples)
        self.assertEqual(full[0].incoming_samples,first[0].incoming_samples)
        self.assertEqual(full[0].raw_mismatch,first[0].raw_mismatch)
        baseline,_,_=self.store()
        self.assertEqual(full,self.pair(baseline))
        observer.reset_mock()
        self.assertEqual(full,self.pair(store,stop_after_first_eligible=True))
        observer.assert_not_called()

    def test_progressive_rejection_still_observes_all_offsets(self):
        store, observer, prepare=self.store()
        result=self.pair(store,stop_after_first_eligible=True,
            adjust=lambda clock,left,right,value: replace(value,
                severe_flags=('actual_failure',),cinematic_bridge=None))
        self.assertEqual(len(result),3)
        self.assertEqual(observer.call_count,6)
        self.assertEqual(prepare.call_count,2)
        self.assertEqual([len(call.args[0]) for call in prepare.call_args_list],[10,20])
        self.assertTrue(all(item.mismatch.severe_flags for item in result))

    def test_disabled_effect_does_not_hide_a_later_valid_hard_offset(self):
        import crosscut.direct_member_visual as visual
        store,_,_=self.store()
        actual=visual._scored_boundary_pair_mismatch
        def score(*args, **kwargs):
            result=actual(*args,**kwargs)
            if kwargs['offset_frames']==0:
                bridge=visual.DirectMemberCinematicBridge('pose_dissolve',False,False,
                    ('attention_scale',),.02,.05,.02)
                return replace(result,severe_flags=('attention_scale',),cinematic_bridge=bridge)
            return result
        with patch.object(visual,'_scored_boundary_pair_mismatch',side_effect=score):
            result=self.pair(store,stop_after_first_eligible=True,cinematic_bridge_enabled=False)
        self.assertEqual([item.plan.offset_frames for item in result],[0,-2,2])
        self.assertFalse(result[-1].mismatch.severe_flags)
        with self.assertRaises(ValueError):
            self.pair(store,stop_after_first_eligible=True,observed_only=True)

    def test_read_only_pair_reuses_real_samples_without_preparation_or_inference(self):
        store, observer, prepare = self.store()
        with self.assertRaises(KeyError): self.pair(store, observed_only=True)
        observer.assert_not_called()
        prepare.assert_not_called()
        expected = self.pair(store)
        observer.reset_mock()
        prepare.reset_mock()
        self.assertEqual(self.pair(store, observed_only=True), expected)
        self.pair(store, left=3, right=0, observed_only=True)
        with self.assertRaises(KeyError): self.pair(store, left=0, right=2, observed_only=True)
        observer.assert_not_called()
        prepare.assert_not_called()

    def test_partial_cached_offset_is_not_a_complete_cross_source_observation(self):
        store, observer, prepare = self.store()
        self.pair(store, left=0, right=0)
        self.pair(store, left=3, right=3)
        observer.reset_mock()
        prepare.reset_mock()
        with self.assertRaises(KeyError): self.pair(store, observed_only=True)
        observer.assert_not_called()
        prepare.assert_not_called()

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
            maximum_observations=6, maximum_solver_calls=2)
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
