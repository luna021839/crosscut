"""Alignment/normalization regressions; recognition accuracy requires media."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

from crosscut.native_face_descriptor import (ARCFACE_TARGET,NativeArcFaceR50,
    arcface_input,face_similarity_affine)


class NativeFaceDescriptorTests(unittest.TestCase):
    def args(self):
        return dict(bbox_xywh=(.1,.1,.8,.8),landmarks_xy=tuple(map(tuple,ARCFACE_TARGET/112)))

    def test_known_rotation_scale_translation_replays_all_five_landmarks(self):
        for angle,scale,translation in ((0.,1.,(0.,0.)),(.7,2.,(300.,140.)),(-.9,.4,(-12.,40.))):
            c,s=np.cos(angle),np.sin(angle);rotation=np.asarray(((c,-s),(s,c)))
            source=ARCFACE_TARGET@rotation.T*scale+translation
            matrix=face_similarity_affine(source)
            actual=np.column_stack((source,np.ones(5)))@matrix.T
            np.testing.assert_allclose(actual,ARCFACE_TARGET,rtol=0.,atol=1e-10)
            self.assertGreater(np.linalg.det(matrix[:,:2]),0.)

    def test_rgb_normalization_matches_pinned_model_without_changing_original(self):
        pixels=np.empty((112,112,3),np.uint8);pixels[:]=(0,127,255)
        before=pixels.copy();blob=arcface_input(pixels,**self.args())
        self.assertEqual(blob.shape,(1,3,112,112));self.assertEqual(blob.dtype,np.float32)
        np.testing.assert_allclose(blob[0,:,55,55],(1.,-0.5/127.5,-1.),atol=1e-7)
        self.assertTrue(blob.flags.c_contiguous);np.testing.assert_array_equal(pixels,before)

    def test_degenerate_or_nonfinite_landmarks_do_not_create_a_descriptor(self):
        for points in (np.zeros((5,2)),np.ones((4,2)),np.full((5,2),np.nan),np.column_stack((np.arange(5),np.arange(5)))):
            with self.subTest(points=points),self.assertRaises(ValueError):face_similarity_affine(points)
        pixels=np.zeros((112,112,3),np.uint8)
        for change in ({'bbox_xywh':(-.1,.1,.8,.8)},{'landmarks_xy':((-1.,.5),)*5}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                arcface_input(pixels,**{**self.args(),**change})

    def test_one_inference_returns_a_normalized_512d_vector(self):
        backend=Mock(get_inputs=Mock(return_value=[SimpleNamespace(name='input',type='tensor(float)',shape=[None,3,112,112])]),
            get_outputs=Mock(return_value=[SimpleNamespace(name='output',shape=[None,512])]),
            run=Mock(return_value=[np.arange(1,513,dtype=np.float32)[None]]))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.onnx';path.write_bytes(b'synthetic-not-executed')
            with patch('crosscut.native_face_descriptor.ARCFACE_R50_SHA256',hashlib.sha256(path.read_bytes()).hexdigest()), \
                    patch('onnxruntime.InferenceSession',return_value=backend) as factory:
                model=NativeArcFaceR50(path)
            result=model.embedding(np.zeros((112,112,3),np.uint8),**self.args())
        self.assertEqual(len(result),512);self.assertAlmostEqual(np.linalg.norm(result),1.)
        self.assertEqual(backend.run.call_count,1)
        self.assertEqual(factory.call_args.kwargs['providers'],['CPUExecutionProvider'])

    def test_wrong_asset_is_rejected_before_model_loading(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.onnx';path.write_bytes(b'wrong')
            with patch('onnxruntime.InferenceSession') as factory,self.assertRaisesRegex(ValueError,'pinned'):
                NativeArcFaceR50(path)
            factory.assert_not_called()

    def test_invalid_model_outputs_cannot_become_zero_similarity_success(self):
        model=object.__new__(NativeArcFaceR50);model.input_name='i';model.output_name='o'
        for value in (np.zeros((1,512)),np.ones((1,128)),np.full((1,512),np.nan)):
            model.session=Mock(run=Mock(return_value=[value]))
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):
                model.embedding(np.zeros((112,112,3),np.uint8),**self.args())


if __name__=='__main__':unittest.main()
