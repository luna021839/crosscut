from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.member_endpoint_exclusions import (
    MemberEndpointExclusion, observed_unresolved_endpoints, probe_unresolved_endpoint,
)
from crosscut.member_boundary_observations import MemberBoundaryObservationStore
from crosscut.member_observation_search import MemberObservationSearch, ObservationState, TransitionDemand, TransitionObservation
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.direct_member_visual import _boundary_sample_with_focal_geometry, _observe_member_boundary_pair, DirectMemberVisualConfig
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_member_boundary_observations import plan
from engine.tests.test_source_first_focal_geometry import geometry


class MemberEndpointExclusionTests(unittest.TestCase):
    def selective_store(self, modes=lambda offset, source, slot: 'missing' if source == 2 else 'face'):
        full, observer, _ = self.store(modes)
        slots = Mock(side_effect=lambda p, source, indices:
            {slot: observer(p, source)[slot] for slot in indices})
        prepare = Mock()
        store = MemberBoundaryObservationStore(full.plans, observe=Mock(),
            observe_samples=slots, prepare=prepare)
        return store, slots, prepare

    def test_endpoint_probe_only_completes_the_explicitly_missing_side(self):
        for left, right, role, slot in ((2,0,'outgoing',1), (0,2,'incoming',2)):
            with self.subTest(role=role):
                store, observer, prepare = self.selective_store()
                for source, indices in ((left,(0,1)), (right,(2,3))):
                    store.requested_samples(0, source, indices)
                observer.reset_mock()
                certificate = probe_unresolved_endpoint(store, boundary_index=3,
                    outgoing_source_index=left, incoming_source_index=right)
                self.assertEqual((certificate.source_index, certificate.role), (2,role))
                self.assertEqual(certificate.offsets, (0,-2,2))
                self.assertEqual(observer.call_count, 2)
                self.assertEqual({call.args[1:] for call in observer.call_args_list}, {(2,(slot,))})
                prepared = prepare.call_args.args[0]
                self.assertEqual(prepared, tuple((2,p.local_times_by_source[2][slot])
                    for p in store.plans if p.offset_frames != 0))
                self.assertNotIn((-2, right if role=='outgoing' else left, 2 if role=='outgoing' else 1),
                    store.observed_sample_demands)

    def test_endpoint_probe_never_treats_unobserved_legacy_or_one_recovery_as_absence(self):
        for recovery in ('face','body','legacy'):
            store, _, _ = self.selective_store(lambda offset,source,slot:
                recovery if offset==2 or source!=2 else 'missing')
            for source, indices in ((2,(0,1)), (0,(2,3))):
                store.requested_samples(0,source,indices)
            self.assertIsNone(probe_unresolved_endpoint(store,boundary_index=0,
                outgoing_source_index=2,incoming_source_index=0))
        for mode in ('unobserved','legacy','face','body','hold'):
            store, observer, prepare = self.selective_store(lambda *_: 'missing' if mode=='hold' else mode)
            if mode!='unobserved':
                store.requested_samples(0,2,(0,1,2,3))
            observer.reset_mock()
            self.assertIsNone(probe_unresolved_endpoint(store,boundary_index=0,
                outgoing_source_index=2,incoming_source_index=2 if mode=='hold' else 0))
            observer.assert_not_called(); prepare.assert_not_called()

    def test_progressive_pair_can_return_a_real_endpoint_certificate_without_fake_pair_cost(self):
        store, _, _ = self.selective_store()
        result = _observe_member_boundary_pair(store,outgoing_source_index=2,incoming_source_index=0,
            source_frame_geometries=(direct_member_source_frame_geometry((2160,3840)),)*4,
            continuous_official_portrait_fancam_contract=True,priority_boundary=True,
            config=DirectMemberVisualConfig(),stop_after_first_eligible=True,
            endpoint_boundary_index=5)
        self.assertIsInstance(result,MemberEndpointExclusion)
        self.assertIn(result,observed_unresolved_endpoints(store,5))
        self.assertNotIn((-2,0,2),store.observed_sample_demands)
        # The complete legacy scorer must reject every allowed offset, with
        # no usable effect. The certificate never stands in for paired cost.
        measured = _observe_member_boundary_pair(store,outgoing_source_index=2,incoming_source_index=0,
            source_frame_geometries=(direct_member_source_frame_geometry((2160,3840)),)*4,
            continuous_official_portrait_fancam_contract=True,priority_boundary=True,
            config=DirectMemberVisualConfig())
        self.assertEqual(len(measured),len(store.plans))
        self.assertTrue(all('cross_source_focal_subject_unresolved' in item.mismatch.severe_flags
            and item.mismatch.cinematic_bridge is None for item in measured))

    def test_certified_endpoint_probe_charges_budget_and_needs_read_only_confirmation(self):
        certificate=MemberEndpointExclusion(0,0,'outgoing',(0,),(1.,),(2.,))
        for confirmed,budget in ((False,2),(True,1),(True,2)):
            search=MemberObservationSearch(((True,)*3,)*2,maximum_observations=budget,
                maximum_solver_calls=3)
            proposals=iter(((0,1),(1,2),(1,2)))
            observer=Mock(side_effect=lambda demand: certificate if demand.outgoing_source_index==0
                else TransitionObservation(demand,True,.2,('actual paired samples',)))
            args=dict(propose=lambda _:next(proposals),observe=observer,
                endpoint_exclusions=lambda _: (certificate,) if confirmed else ())
            if not confirmed:
                with self.assertRaisesRegex(ValueError,'confirmed'):
                    search.run(**args)
            elif budget==1:
                from crosscut.member_observation_search import MemberObservationSearchIncomplete
                with self.assertRaises(MemberObservationSearchIncomplete) as caught:
                    search.run(**args)
                self.assertEqual(caught.exception.diagnostics['observation_count'],1)
                self.assertEqual(caught.exception.diagnostics['endpoint_only_observation_count'],1)
                self.assertFalse(search.observations)
            else:
                result=search.run(**args)
                self.assertEqual(result.endpoint_only_observation_count,1)
                self.assertEqual(result.observation_count,2)
                self.assertEqual(result.endpoint_excluded_transition_count,2)
                self.assertEqual(len(search.observations),1)
                self.assertEqual(result.source_indices,(1,2))

    def test_endpoint_probe_io_failure_propagates_without_a_negative_certificate(self):
        store, _, prepare = self.selective_store()
        store.requested_samples(0,2,(0,1))
        prepare.side_effect=OSError('SSD disconnected')
        with self.assertRaisesRegex(OSError,'SSD disconnected'):
            probe_unresolved_endpoint(store,boundary_index=0,
                outgoing_source_index=2,incoming_source_index=0)
        self.assertEqual(observed_unresolved_endpoints(store,0),())

    def test_endpoint_probe_retains_an_actual_cached_rejection_without_counting_it_twice(self):
        certificate=MemberEndpointExclusion(0,0,'outgoing',(0,),(1.,),(2.,))
        failed=TransitionDemand(0,0,1)
        measured=TransitionObservation(failed,False,.8,('all actual cached offsets',))
        search=MemberObservationSearch(((True,)*3,)*2,maximum_observations=2,
            maximum_solver_calls=3)
        proposals=iter(((0,1),(1,2),(1,2)))
        result=search.run(propose=lambda _:next(proposals),
            observe=lambda demand:certificate if demand==failed else TransitionObservation(demand,True,.2,('pixels',)),
            reuse_observed=lambda demand:(measured,) if demand==failed else (),
            endpoint_exclusions=lambda _:(certificate,))
        self.assertIn(measured,search.observations)
        self.assertNotIn(failed,search.endpoint_excluded_demands)
        self.assertEqual(result.endpoint_only_observation_count,1)
        self.assertEqual(result.endpoint_excluded_transition_count,1)
        self.assertEqual(result.reused_observation_count,1)
        self.assertEqual(result.unobserved_transition_count,3)

    def test_unrelated_endpoint_certificate_cannot_reject_the_requested_pair(self):
        for certificate in (MemberEndpointExclusion(1,0,'outgoing',(0,),(1.,),(2.,)),
                            MemberEndpointExclusion(0,2,'incoming',(0,),(1.,),(2.,))):
            search=MemberObservationSearch(((True,)*3,)*2,maximum_observations=2,
                maximum_solver_calls=3)
            with self.assertRaisesRegex(ValueError,'requested pair'):
                search.run(propose=lambda _:(0,1),observe=lambda _:certificate,
                    endpoint_exclusions=lambda _:(certificate,))
            self.assertFalse(search.observations)
            self.assertFalse(search.endpoint_excluded_demands)

    def test_partial_roles_prove_only_the_completely_observed_endpoint_side(self):
        full, observer, _ = self.store()
        selective = Mock(side_effect=lambda p, source, slots:
            {slot: full.samples_for(p.offset_frames, source)[slot] for slot in slots})
        store = MemberBoundaryObservationStore(full.plans, observe=Mock(), observe_samples=selective)
        for p in store.plans:
            store.requested_samples(p.offset_frames, 2, (0, 1))
        selective.reset_mock(); observer.reset_mock()
        result = observed_unresolved_endpoints(store, 4)
        self.assertEqual([item.role for item in result], ['outgoing'])
        self.assertFalse(store.observed_demands)
        selective.assert_not_called(); observer.assert_not_called()
        # One incoming offset is still not a whole-domain negative.
        store.requested_samples(0, 2, (2,))
        self.assertEqual([item.role for item in observed_unresolved_endpoints(store, 4)], ['outgoing'])

    def store(self, modes=lambda offset, source, slot: 'missing' if source == 2 else 'face'):
        plans = tuple(replace(plan(offset), eligibility_by_source=((True,)*5,)*4) for offset in (0, -2, 2))
        def observe(p, source):
            result = []
            for slot, (master, local) in enumerate(zip(p.master_times, p.local_times_by_source[source])):
                mode = modes(p.offset_frames, source, slot)
                if mode == 'missing':
                    focal = FocalSelectorGeometry(source, local, None)
                elif mode == 'body':
                    focal = geometry(source, body_only=True)
                    track = replace(focal.body_track, source_seconds=local,
                        observation=replace(focal.body_track.observation, source_seconds=local),
                        support_seconds=(local-.12, local+.12))
                    focal = replace(focal, source_seconds=local, body_track=track)
                else:
                    focal = replace(geometry(source), source_seconds=local)
                provider = (SimpleNamespace() if mode == 'legacy'
                    else SimpleNamespace(focal_geometry_at=lambda *_: focal))
                result.append(_boundary_sample_with_focal_geometry(master_seconds=master,
                    source_local_seconds=local, source_index=source, descriptor=_descriptor(.2), provider=provider))
            return tuple(result)
        observer, prepare = Mock(side_effect=observe), Mock()
        return MemberBoundaryObservationStore(plans, observe=observer, prepare=prepare), observer, prepare

    def materialize(self, store, source=2):
        for p in store.plans:
            store.samples_for(p.offset_frames, source)

    def test_complete_explicit_negative_excludes_only_this_side_boundary_not_holds(self):
        store, observer, prepare = self.store()
        self.materialize(store)
        observer.reset_mock()
        outgoing, incoming = observed_unresolved_endpoints(store, 4)
        observer.assert_not_called()
        prepare.assert_not_called()
        self.assertEqual((outgoing.role, incoming.role), ('outgoing', 'incoming'))
        self.assertTrue(outgoing.excludes(TransitionDemand(4, 2, 0)))
        self.assertFalse(outgoing.excludes(TransitionDemand(4, 0, 2)))
        self.assertTrue(incoming.excludes(TransitionDemand(4, 0, 2)))
        self.assertFalse(incoming.excludes(TransitionDemand(5, 0, 2)))
        self.assertFalse(incoming.excludes(TransitionDemand(4, 2, 2)))
        self.assertEqual(outgoing.offsets, (0, -2, 2))

    def test_missing_offsets_or_one_recovered_offset_do_not_prove_exclusion(self):
        store, observer, _ = self.store()
        store.samples_for(0, 2)
        self.assertEqual(observed_unresolved_endpoints(store, 0), ())
        self.assertEqual(observer.call_count, 1)
        for recovery in ('face', 'body', 'legacy'):
            store, _, _ = self.store(lambda offset, source, slot:
                recovery if offset == 2 else 'missing')
            self.materialize(store)
            self.assertEqual(observed_unresolved_endpoints(store, 0), ())

    def test_valid_tracked_body_and_legacy_authority_are_not_missing_subject_proof(self):
        for mode in ('body', 'face', 'legacy'):
            store, _, _ = self.store(lambda *_: mode)
            self.materialize(store)
            self.assertEqual(observed_unresolved_endpoints(store, 0), ())

    def test_roles_are_distinct_and_tampered_source_or_clock_is_rejected(self):
        store, _, _ = self.store(lambda offset, source, slot: 'missing' if slot == 1 else 'face')
        self.materialize(store)
        self.assertEqual([item.role for item in observed_unresolved_endpoints(store, 0)], ['outgoing'])
        original = store._samples[(0, 2)]
        for changes in ({'master_seconds': 100.}, {'source_local_seconds': 100.},
                        {'focal_geometry': FocalSelectorGeometry(1, original[1].source_local_seconds, None)}):
            store._samples[(0, 2)] = (original[0], replace(original[1], **changes), *original[2:])
            with self.assertRaisesRegex(ValueError, 'source/clock'):
                observed_unresolved_endpoints(store, 0)
        store._samples[(0, 2)] = original

    def test_certified_pairs_match_actual_scorer_at_every_offset_without_bridge(self):
        store, _, _ = self.store()
        self.materialize(store)
        for certificate in observed_unresolved_endpoints(store, 0):
            for other in (0, 1, 3):
                left, right = (2, other) if certificate.role == 'outgoing' else (other, 2)
                candidates = _observe_member_boundary_pair(store, outgoing_source_index=left,
                    incoming_source_index=right,
                    source_frame_geometries=(direct_member_source_frame_geometry((2160,3840)),)*4,
                    continuous_official_portrait_fancam_contract=True, priority_boundary=True,
                    config=DirectMemberVisualConfig())
                for candidate in candidates:
                    self.assertIn('cross_source_focal_subject_unresolved', candidate.mismatch.severe_flags)
                    self.assertIsNone(candidate.mismatch.cinematic_bridge)

    def test_search_keeps_exclusions_separate_from_paired_measurements(self):
        search = MemberObservationSearch(((True,)*3,)*3, maximum_observations=10,
            maximum_solver_calls=5, allow_same_source_holds=True)
        certificate = MemberEndpointExclusion(0, 0, 'outgoing', (0,), (1.,), (2.,))
        proposals = iter(((0, 1, 2), (1, 2, 0), (1, 2, 0)))
        observer = Mock(side_effect=lambda demand: TransitionObservation(demand,
            demand != TransitionDemand(0, 0, 1), .2, ('paired',)))
        result = search.run(propose=lambda _: next(proposals), observe=observer,
            endpoint_exclusions=lambda demand: (certificate,) if demand.boundary_index == 0 else ())
        self.assertEqual(result.endpoint_excluded_transition_count, 1)
        self.assertEqual(result.observation_count, 3)
        self.assertNotIn(TransitionDemand(0, 0, 2), [item.demand for item in search.observations])
        self.assertEqual(search.proposal_graph()[0][0][2].state, ObservationState.ENDPOINT_INELIGIBLE)
        self.assertEqual(search.proposal_graph()[0][0][0].state, ObservationState.UNOBSERVED)
        self.assertEqual(search.proposal_graph()[1][0][2].state, ObservationState.UNOBSERVED)

    def test_bad_callback_cannot_hide_eligible_evidence_or_another_boundary(self):
        for certificate in (MemberEndpointExclusion(0, 0, 'outgoing', (0,), (1.,), (2.,)),
                            MemberEndpointExclusion(1, 0, 'outgoing', (0,), (1.,), (2.,))):
            search = MemberObservationSearch(((True,)*3,)*3, maximum_observations=10, maximum_solver_calls=5)
            with self.assertRaises(ValueError):
                search.run(propose=lambda _: (0, 1, 2),
                    observe=lambda demand: TransitionObservation(demand, True, .2, ('paired',)),
                    endpoint_exclusions=lambda _: (certificate,))


if __name__ == '__main__': unittest.main()
