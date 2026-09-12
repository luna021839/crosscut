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
            self.assertIs(provider.candidates_at(0,1.),candidates)
            self.assertEqual(encode.call_count,1)

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
            encode.assert_not_called()


if __name__=='__main__':unittest.main()
