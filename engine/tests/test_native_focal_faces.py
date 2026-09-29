from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.native_focal_faces import NativeFocalFaceProvider
from crosscut.visual_score import _DetectedFace


class NativeFocalFaceTests(unittest.TestCase):
    def test_alignment_uses_native_pixels_not_detection_thumbnail(self):
        provider=NativeFocalFaceProvider((Path('source'),))
        native=np.zeros((1080,1920,3),np.uint8)
        face=_DetectedFace((300.,100.,100.,120.),((320.,140.),(375.,140.),(350.,165.),(325.,190.),(375.,190.)),.99)
        with patch.object(provider,'frame_at',return_value=native), \
             patch('crosscut.native_focal_faces._detect_faces',return_value=((face,),'yunet')) as detect, \
             patch('crosscut.native_focal_faces.extract_sface_embedding',return_value=(1.,0.)) as encode:
            candidates=provider.candidates_at(0,1.)
            self.assertEqual(detect.call_args.args[0].shape,(540,960,3))
            self.assertIs(encode.call_args.args[0],native)
            self.assertEqual(len(candidates),1)
            self.assertTrue(provider.candidate_observation_complete(0, 1.))
            self.assertIs(provider.candidates_at(0,1.),candidates)
            self.assertEqual(encode.call_count,1)
            self.assertEqual(detect.call_count, 1)

    def test_boolean_query_cannot_alias_cached_source(self):
        provider=NativeFocalFaceProvider((Path('left'),Path('right')))
        provider._cache[(1,1.)]=()
        with self.assertRaises(ValueError):provider.candidates_at(True,1.)

    def test_unknown_detection_backend_cannot_create_face_identity(self):
        provider=NativeFocalFaceProvider((Path('source'),))
        with patch.object(provider,'frame_at',return_value=np.zeros((128,128,3),np.uint8)), \
             patch('crosscut.native_focal_faces._detect_faces',return_value=((),'haar')), \
             patch('crosscut.native_focal_faces.extract_sface_embedding') as encode:
            self.assertEqual(provider.candidates_at(0,1.),())
            self.assertFalse(provider.candidate_observation_complete(0, 1.))
            encode.assert_not_called()

    def test_empty_yunet_result_is_complete_but_failed_decode_is_not(self):
        provider = NativeFocalFaceProvider((Path('source'),))
        with patch.object(provider, 'frame_at', return_value=np.zeros((128,128,3),np.uint8)), \
             patch('crosscut.native_focal_faces._detect_faces', return_value=((), 'yunet')):
            self.assertEqual(provider.candidates_at(0, 1.), ())
            self.assertTrue(provider.candidate_observation_complete(0, 1.))
        with patch.object(provider, 'frame_at', return_value=None):
            self.assertEqual(provider.candidates_at(0, 2.), ())
            self.assertFalse(provider.candidate_observation_complete(0, 2.))
        self.assertFalse(provider.candidate_observation_complete(False, 1.))
        self.assertFalse(provider.candidate_observation_complete(0, True))
        self.assertFalse(provider.candidate_observation_complete(0, 3.))

    def test_partial_identity_extraction_preserves_candidates_without_absence_proof(self):
        provider = NativeFocalFaceProvider((Path('source'),))
        face = _DetectedFace((30.,10.,10.,12.), ((32.,14.),(37.5,14.),(35.,16.5),(32.5,19.),(37.5,19.)), .99)
        with patch.object(provider, 'frame_at', return_value=np.zeros((128,128,3),np.uint8)), \
             patch('crosscut.native_focal_faces._detect_faces', return_value=((face, face), 'yunet')), \
             patch('crosscut.native_focal_faces.extract_sface_embedding', side_effect=((1.,0.), ())):
            self.assertEqual(len(provider.candidates_at(0, 1.)), 1)
            self.assertFalse(provider.candidate_observation_complete(0, 1.))

    def test_completion_evidence_has_the_same_cache_lifetime_as_candidates(self):
        provider = NativeFocalFaceProvider((Path('source'),))
        for frame in range(4096):
            provider._cache[(0, float(frame))] = ()
            provider._candidate_completeness[(0, float(frame))] = True
        with patch.object(provider, 'frame_at', return_value=None):
            provider.candidates_at(0, 4096.)
        self.assertEqual(len(provider._cache), 4096)
        self.assertEqual(len(provider._candidate_completeness), 4096)
        self.assertFalse(provider.candidate_observation_complete(0, 0.))
        provider.close()
        self.assertFalse(provider._candidate_completeness)


if __name__=='__main__':unittest.main()
