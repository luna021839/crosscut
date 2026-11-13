from dataclasses import replace
import json
import unittest

from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_reference_consensus import (
    FocalReferenceConsensus, SoloFaceObservation, build_focal_reference_consensus,
)


def face(embedding):
    return FocalFaceCandidate((.4,.2,.15,.1),tuple(embedding))


def seeds(embedding, count=4, start=0.):
    return tuple(SoloFaceObservation(start+i*3.,face(embedding)) for i in range(count))


class FocalReferenceConsensusTests(unittest.TestCase):
    def test_frozen_bank_prefers_supported_face_not_an_unrelated_pair(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        self.assertIsNotNone(bank)
        target,neighbour=face((.9,.1)),face((0.,1.))
        self.assertIs(bank.select((neighbour,target)),target)
        self.assertIsNone(bank.select((neighbour,)))
        self.assertIs(bank.select((target,)),target)

    def test_selection_cannot_drift_the_bank(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        before=bank.references
        for _ in range(40):
            bank.select((face((0.,1.)),))
        self.assertEqual(bank.references,before)

    def test_occlusion_and_ambiguous_candidates_do_not_invent_target(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        self.assertIsNone(bank.select(()))
        self.assertIsNone(bank.select((face((1.,0.)),face((1.,.01)))))

    def test_one_matching_reference_does_not_authorize_a_query(self):
        observations=(SoloFaceObservation(0.,face((1.,0.,0.))),
                      SoloFaceObservation(3.,face((0.,1.,0.))),
                      SoloFaceObservation(6.,face((0.,0.,1.))))
        bank=FocalReferenceConsensus(observations,3,0)
        self.assertIsNone(bank.select((face((1.,0.,0.)),)))

    def test_repeated_or_adjacent_frames_cannot_create_independent_support(self):
        observations=tuple(SoloFaceObservation(i*.01,face((1.,0.))) for i in range(20))
        self.assertIsNone(build_focal_reference_consensus(observations))
        self.assertIsNone(build_focal_reference_consensus(seeds((1.,0.),2)))

    def test_two_competing_performer_clusters_stay_ambiguous(self):
        self.assertIsNone(build_focal_reference_consensus(
            seeds((1.,0.),3)+seeds((0.,1.),3,20.)))
        self.assertIsNone(build_focal_reference_consensus(
            seeds((1.,0.),3)+seeds((0.,1.),2,20.)))

    def test_dominant_repeated_solo_track_can_win(self):
        bank=build_focal_reference_consensus(seeds((1.,0.),6)+seeds((0.,1.),2,30.))
        self.assertIsNotNone(bank)
        self.assertEqual(len(bank.references),6)
        self.assertEqual(bank.alternate_support_count,2)

    def test_scalar_audit_contains_no_embeddings_or_name(self):
        bank=build_focal_reference_consensus(seeds((.98321,.11623)))
        payload=bank.to_dict()
        self.assertFalse(payload['identity_name_inferred'])
        self.assertFalse(payload['embeddings_persisted'])
        self.assertNotIn('.98321',json.dumps(payload))
        self.assertNotIn('embedding',repr(bank.references))

    def test_unbounded_bootstrap_is_rejected(self):
        with self.assertRaises(ValueError):
            build_focal_reference_consensus(seeds((1.,0.),65))

    def test_invalid_geometry_and_times_are_rejected(self):
        for seconds in (float('nan'),float('inf'),-1.,True,None,'1'):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                SoloFaceObservation(seconds,face((1.,0.)))
        with self.assertRaises(ValueError):
            build_focal_reference_consensus(seeds((1.,0.)),minimum_seed_similarity=2.)

    def test_zero_norm_and_incompatible_embeddings_are_unsupported(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        self.assertIsNone(bank.select((face((0.,0.)), face((1.,0.,0.)))))
        self.assertIsNone(build_focal_reference_consensus(seeds((0.,0.))))

    def test_invalid_counts_thresholds_and_seed_types(self):
        for kwargs in ({'observation_count':True}, {'alternate_support_count':1.2},
                       {'minimum_query_similarity':True}, {'references':list(seeds((1.,0.)))}):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                replace(build_focal_reference_consensus(seeds((1.,0.))),**kwargs)
        with self.assertRaises(TypeError):
            build_focal_reference_consensus((None,))
        with self.assertRaises(ValueError):
            build_focal_reference_consensus(seeds((1.,0.)),minimum_seed_similarity=True)

    def test_temporal_bridge_requires_two_independently_bank_matched_frames(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        exact=face((.342,.94))
        bridge=face((.7,.714))
        self.assertIsNone(bank.select((exact,)))
        nearby=(SoloFaceObservation(9.9,bridge),SoloFaceObservation(9.8,bridge))
        self.assertIs(bank.recover_from_adjacent_observations((exact,),query_seconds=10.,
                      adjacent_observations=nearby),exact)
        for invalid in (nearby[:1], (nearby[0],nearby[0]),
                        (SoloFaceObservation(9.,bridge),SoloFaceObservation(9.1,bridge))):
            self.assertIsNone(bank.recover_from_adjacent_observations((exact,),query_seconds=10.,
                              adjacent_observations=invalid))

    def test_temporal_bridge_does_not_fabricate_or_switch_to_distant_person(self):
        bank=build_focal_reference_consensus(seeds((1.,0.)))
        bridge=face((.7,.714))
        nearby=(SoloFaceObservation(9.9,bridge),SoloFaceObservation(9.8,bridge))
        for candidates in ((),(face((0.,1.)),),
                           (replace(face((.342,.94)),bbox_xywh=(.9,.8,.05,.05)),)):
            self.assertIsNone(bank.recover_from_adjacent_observations(candidates,query_seconds=10.,
                              adjacent_observations=nearby))
        self.assertIsNone(bank.select((face((.342,.94)),)))


if __name__=='__main__':
    unittest.main()
