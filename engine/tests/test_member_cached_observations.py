"""Cached evidence can save work, never manufacture a target or a cut."""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.member_cached_observations import (
    CachedMemberObservation, attach_cached_member_observations, seed_cached_boundary_samples,
)
from crosscut.member_boundary_observations import MemberBoundaryObservationStore
from crosscut.member_observation_search import (
    MemberObservationSearch, MemberObservationSearchIncomplete, TransitionDemand,
)
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from engine.tests.test_source_first_focal_geometry import geometry
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_member_boundary_observations import plan
from engine.tests.test_member_observation_search import observed


class CachedMemberProviderTests(unittest.TestCase):
    def fixture(self, *, memory=False, value=None, descriptor=None):
        value = geometry() if value is None else value
        raw = _descriptor(.2) if descriptor is None else descriptor
        identity = Mock(return_value=(1,2,3,4))
        persistent = SimpleNamespace(get=Mock(return_value=value))
        focal = SimpleNamespace(sources=(Path('source'),), _identity=identity,
            _identities=((1,2,3,4),), _cache=OrderedDict({(0,30):value} if memory else {}),
            _persistent=persistent, _persistent_key=lambda source,key:str(key),
            faces=SimpleNamespace(_samplers=(object(),), _identity_key=lambda source,sampler,t:(source,int(t*30))),
            selector_geometry=Mock(side_effect=AssertionError('cache read must not infer')))
        sampler = SimpleNamespace(frame_identity=lambda t:int(t*30),
            feature_descriptor_at=Mock(return_value=raw),
            assert_required_persistent_cache_available=Mock())
        visual = SimpleNamespace(_samplers=(sampler,), _require_persistent_feature_cache=True,
            _descriptor_cache={})
        attach_cached_member_observations(focal, visual)
        return focal, visual, sampler

    def test_exact_positive_cache_read_does_not_run_the_target_resolver(self):
        for memory in (False,True):
            focal, visual, sampler = self.fixture(memory=memory)
            result = visual.cached_focal_observation_at(0,1.01)
            self.assertIsInstance(result,CachedMemberObservation)
            self.assertEqual(result.geometry.source_seconds,1.01)
            self.assertEqual(result.geometry.face_bbox_xywh,geometry().face_bbox_xywh)
            focal.selector_geometry.assert_not_called()
            self.assertTrue(all(call.kwargs['cache_only'] for call in sampler.feature_descriptor_at.call_args_list))
            sampler.assert_required_persistent_cache_available.assert_called()
            if memory: focal._persistent.get.assert_not_called()

    def test_missing_focal_or_descriptor_remains_unknown_without_decode(self):
        focal, visual, sampler = self.fixture()
        focal._persistent.get.return_value=None
        self.assertIsNone(visual.cached_focal_observation_at(0,1.))
        sampler.feature_descriptor_at.assert_not_called()
        focal, visual, sampler = self.fixture(value=FocalSelectorGeometry(0,1.,None))
        self.assertIsNone(visual.cached_focal_observation_at(0,1.))
        sampler.feature_descriptor_at.assert_not_called()
        focal, visual, sampler = self.fixture()
        sampler.feature_descriptor_at.return_value=None
        self.assertIsNone(visual.cached_focal_observation_at(0,1.))
        self.assertTrue(all(call.kwargs['cache_only'] for call in sampler.feature_descriptor_at.call_args_list))
        focal.selector_geometry.assert_not_called()

    def test_source_change_or_cache_disconnect_escapes_even_after_a_hit(self):
        focal, visual, sampler = self.fixture()
        self.assertIsNotNone(visual.cached_focal_observation_at(0,1.))
        focal._identity.return_value=(1,2,3,5)
        with self.assertRaises(OSError): visual.cached_focal_observation_at(0,1.)
        focal, visual, sampler = self.fixture()
        sampler.assert_required_persistent_cache_available.side_effect=OSError('SSD')
        with self.assertRaises(OSError): visual.cached_focal_observation_at(0,1.)
        sampler.feature_descriptor_at.assert_not_called()

    def test_actual_body_track_retimes_its_query_not_its_independent_anchors(self):
        focal, visual, _ = self.fixture(memory=True,value=geometry(body_only=True))
        result=visual.cached_focal_observation_at(0,1.01).geometry
        self.assertEqual(result.body_track.source_seconds,1.01)
        self.assertEqual(result.body_track.observation.source_seconds,1.01)
        self.assertEqual(result.body_track.support_seconds,(.8,1.2))
        self.assertIsNone(result.face_bbox_xywh)

    def test_wrong_source_geometry_and_untyped_payload_cannot_seed(self):
        for value in (geometry(1),object()):
            focal, visual, _=self.fixture(value=value)
            with self.assertRaises((TypeError,ValueError)):
                visual.cached_focal_observation_at(0,1.)
        with self.assertRaises((TypeError,ValueError)):
            CachedMemberObservation(_descriptor(.2),FocalSelectorGeometry(0,1.,None))


class CachedBoundaryPublicationTests(unittest.TestCase):
    def store(self):
        callback=Mock(side_effect=AssertionError('cold observation is forbidden'))
        return MemberBoundaryObservationStore((plan(),),observe=callback,prepare=callback,
                                              observe_samples=callback)

    def test_cache_seeding_preserves_source_master_clock_and_unknown_slots(self):
        store=self.store()
        def cached(source,seconds):
            if source != 0 or seconds != plan().local_times_by_source[0][1]: return None
            return CachedMemberObservation(_descriptor(.2),replace(geometry(),source_seconds=seconds))
        count=seed_cached_boundary_samples(store,SimpleNamespace(cached_focal_observation_at=cached))
        self.assertEqual(count,1)
        sample=store.observed_sample(0,0,1)
        self.assertEqual(sample.master_seconds,plan().master_times[1])
        self.assertEqual(sample.source_local_seconds,plan().local_times_by_source[0][1])
        with self.assertRaises(KeyError):store.observed_sample(0,0,0)

    def test_content_ineligible_samples_do_not_read_any_cached_evidence(self):
        store=self.store()
        callback=Mock(return_value=None)
        self.assertEqual(seed_cached_boundary_samples(store,SimpleNamespace(cached_focal_observation_at=callback)),0)
        queried=tuple((c.args[0],c.args[1]) for c in callback.call_args_list)
        expected=tuple((source,seconds) for source in range(4)
            for seconds,valid in zip(plan().local_times_by_source[source],plan().eligibility_by_source[source]) if valid)
        self.assertEqual(queried,expected)
        self.assertFalse(store.observed_sample_demands)

    def test_wrong_clock_is_rejected_and_cannot_partly_publish_a_source(self):
        store=self.store()
        callback=Mock(return_value=CachedMemberObservation(_descriptor(.2),geometry()))
        with self.assertRaises(ValueError):
            seed_cached_boundary_samples(store,SimpleNamespace(cached_focal_observation_at=callback))
        self.assertFalse(store.observed_sample_demands)


class SeededSearchBudgetTests(unittest.TestCase):
    def search(self,budget=3):
        return MemberObservationSearch(((True,)*3,)*3,maximum_observations=budget,maximum_solver_calls=5)

    def test_cached_path_is_still_solved_using_measured_policy(self):
        search=self.search()
        demand=(TransitionDemand(0,0,1),TransitionDemand(1,1,2))
        search.seed_observations(tuple(observed(d) for d in demand))
        observer=Mock(side_effect=AssertionError('complete cached path must not infer'))
        proposer=Mock(return_value=(0,1,2))
        result=search.run(propose=proposer,observe=observer)
        self.assertEqual(result.observation_count,0)
        self.assertEqual(result.reused_observation_count,2)
        self.assertEqual(result.solver_call_count,1)
        observer.assert_not_called()
        proposer.assert_called_once()

    def test_cached_partial_pair_still_reserves_refinement_work(self):
        search=self.search(budget=1)
        search.seed_observations(tuple(replace(observed(d),complete=False)
            for d in (TransitionDemand(0,0,1),TransitionDemand(1,1,2))))
        refine=Mock(side_effect=lambda old:replace(old,complete=True))
        with self.assertRaises(MemberObservationSearchIncomplete):
            search.run(propose=lambda _:(0,1,2),observe=Mock(),refine=refine)
        self.assertEqual(refine.call_count,1)

    def test_duplicate_or_foreign_seed_does_not_overwrite_or_partly_publish(self):
        valid=observed(TransitionDemand(0,0,1))
        for items in ((valid,valid),(valid,observed(TransitionDemand(4,0,1))),(valid,None)):
            search=self.search()
            with self.assertRaises(ValueError):search.seed_observations(items)
            self.assertFalse(search.observations)
        search=self.search()
        search.seed_observations((valid,))
        with self.assertRaises(ValueError):search.seed_observations((valid,))
        self.assertEqual(search.observations,(valid,))


if __name__=='__main__':unittest.main()
