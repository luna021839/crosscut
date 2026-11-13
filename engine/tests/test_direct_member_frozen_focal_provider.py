"""The production focal source bank cannot be overwritten by cut-local pairs."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.direct_member_editorial_motion import DirectMemberDenseFaceProvider
from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_reference_consensus import SoloFaceObservation, build_focal_reference_consensus
from crosscut.visual_feature_cache import CanonicalFrameIdentity


TARGET=FocalFaceCandidate((.4,.2,.15,.1),(1.,0.))
NEIGHBOUR=FocalFaceCandidate((.6,.2,.15,.1),(0.,1.))


class FakeSampler:
    def __init__(self,*args,**kwargs):
        self.frame_decode_count=0
        self.capture=SimpleNamespace(get=lambda key: 30. if key==5 else 3000.)
    def frame_identity(self,seconds):
        return CanonicalFrameIdentity.from_seconds(seconds,frame_rate=30.)
    def close(self):
        pass


class FrozenFocalProviderTests(unittest.TestCase):
    def provider(self):
        with patch('crosscut.direct_member_editorial_motion._VideoSampler',FakeSampler), \
             patch('crosscut.direct_member_editorial_motion._enable_dense_sequential_frame_reads'):
            return DirectMemberDenseFaceProvider((Path('left'),Path('right')),
                                                require_focal_consensus=True)

    def test_bootstrap_is_once_and_pair_registration_cannot_drift(self):
        provider=self.provider()
        calls=[]
        provider._focal_candidates=lambda source,seconds: calls.append((source,seconds)) or (TARGET,)
        bank=provider._ensure_focal_consensus(0)
        self.assertIsNotNone(bank)
        self.assertEqual(len(calls),21)
        self.assertIs(provider._ensure_focal_consensus(0),bank)
        self.assertEqual(len(calls),21)
        provider._register_focal_reference(0,NEIGHBOUR)
        self.assertTrue(all(face==TARGET for face in provider._focal_references[0]))

    def test_interleaved_bootstrap_recovers_source_without_weakening_evidence(self):
        provider=self.provider()
        calls=[]
        def candidates(source,seconds):
            calls.append(seconds)
            return () if len(calls)<=21 else (TARGET,)
        provider._focal_candidates=candidates
        self.assertIsNotNone(provider._ensure_focal_consensus(0))
        self.assertEqual(len(calls),41)

    def test_exhausted_bootstrap_is_bounded_and_never_falls_back_to_crowd(self):
        provider=self.provider()
        calls=[]
        provider._focal_candidates=lambda source,seconds: calls.append(seconds) or (TARGET,NEIGHBOUR)
        self.assertIsNone(provider.sample(0,10.))
        self.assertEqual(len(calls),42)
        self.assertIsNone(provider.sample(0,10.))
        self.assertEqual(len(calls),42)

    def test_unrelated_crosscut_pair_cannot_override_known_focal_bank(self):
        provider=self.provider()
        bank=build_focal_reference_consensus(tuple(SoloFaceObservation(float(i*3),TARGET) for i in range(4)))
        provider._frozen_focal_banks={0:bank,1:bank}
        provider._focal_candidates=lambda source,seconds: (NEIGHBOUR,)
        self.assertEqual(provider.paired_samples(0,10.,1,10.),(None,None))
        self.assertEqual(provider.focal_boundary_reference_capture(),((0,1,None,None),))

    def test_temporal_recovery_uses_exact_observation_and_frozen_seeds(self):
        provider=self.provider()
        bank=build_focal_reference_consensus(tuple(SoloFaceObservation(float(i*3),TARGET) for i in range(4)))
        provider._frozen_focal_banks={0:bank}
        exact=FocalFaceCandidate(TARGET.bbox_xywh,(.342,.94))
        adjacent=FocalFaceCandidate(TARGET.bbox_xywh,(.7,.714))
        provider._focal_candidates=lambda source,seconds: (exact,) if abs(seconds-10.)<1e-7 else (adjacent,)
        self.assertIs(provider._frozen_focal_candidate(0,10.,direction=1),exact)
        self.assertEqual(provider._focal_temporal_recovery_count,1)
        self.assertIs(provider._frozen_focal_banks[0],bank)
        self.assertIsNone(bank.select((exact,)))


if __name__=='__main__':
    unittest.main()
