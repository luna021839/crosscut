from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.member_endpoint_exclusions import MemberEndpointExclusion, observed_unresolved_endpoints
from crosscut.member_boundary_observations import MemberBoundaryObservationStore
from crosscut.member_observation_search import MemberObservationSearch, ObservationState, TransitionDemand, TransitionObservation
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.direct_member_visual import _boundary_sample_with_focal_geometry, _observe_member_boundary_pair, DirectMemberVisualConfig
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_member_boundary_observations import plan
from engine.tests.test_source_first_focal_geometry import geometry


class MemberEndpointExclusionTests(unittest.TestCase):
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
