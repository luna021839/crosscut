from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

from crosscut.native_person_onnx import (
    PersonDetectorConfig, NativePersonOnnxDetector, native_letterbox, person_rows)


class NativePersonOnnxTests(unittest.TestCase):
    def tensor(self, transform, boxes):
        count = sum((transform.input_width//s)*(transform.input_height//s) for s in (8,16,32))
        result = np.zeros((1,84,count), np.float32)
        for index, (x,y,w,h,score,other) in enumerate(boxes):
            result[0,:4,index] = (x,y,w,h)
            result[0,4,index] = score
            result[0,5,index] = other
        return result

    def test_landscape_letterbox_color_and_native_coordinates(self):
        pixels = np.full((90,160,3), (10,20,30), np.uint8)
        tensor, transform = native_letterbox(pixels, maximum_side=320)
        self.assertEqual(tensor.shape, (1,3,192,320))
        self.assertEqual((transform.left,transform.top,transform.scale),(0,6,2))
        np.testing.assert_allclose(tensor[0,:,6,0],np.asarray((30,20,10))/255,atol=1e-7)
        np.testing.assert_allclose(tensor[0,:,0,0],114/255,atol=1e-7)
        prediction = self.tensor(transform, [(100,86,80,120,.8,0)])
        rows = person_rows(prediction,transform,config=PersonDetectorConfig(maximum_side=320))
        np.testing.assert_allclose(rows, [[30,10,70,70,.8]], atol=1e-5)

    def test_portrait_and_odd_padding_do_not_swap_native_axes(self):
        for height,width in ((160,90),(101,153),(153,101)):
            _, t = native_letterbox(np.zeros((height,width,3),np.uint8),maximum_side=320)
            self.assertEqual(t.input_width%32,0);self.assertEqual(t.input_height%32,0)
            xywh = (width*.5*t.scale+t.left,height*.5*t.scale+t.top,width*.4*t.scale,height*.6*t.scale)
            rows = person_rows(self.tensor(t,[(*xywh,.9,0)]),t,config=PersonDetectorConfig())
            np.testing.assert_allclose(rows[0,:4], [width*.3,height*.2,width*.7,height*.8],atol=1e-4)

    def test_nms_single_label_and_threshold_remain_explicit(self):
        _, t = native_letterbox(np.zeros((160,160,3),np.uint8),maximum_side=320)
        prediction = self.tensor(t,[(80,80,40,80,.9,0),(80,80,40,80,.8,0),
            (220,80,40,80,.7,.95),(80,220,40,80,.35,0),(220,220,40,80,.6,0)])
        rows = person_rows(prediction,t,config=PersonDetectorConfig())
        self.assertEqual(len(rows),2)
        np.testing.assert_allclose(rows[:,4],[.9,.6],atol=1e-6)

    def test_more_than_twelve_rectangles_are_not_dropped_or_called_a_cast(self):
        _, t = native_letterbox(np.zeros((320,320,3),np.uint8),maximum_side=320)
        prediction = self.tensor(t,[(10+20*i,100,10,30,.9,0) for i in range(14)])
        self.assertEqual(len(person_rows(prediction,t,config=PersonDetectorConfig())),14)

    def test_native_clipping_and_padding_only_have_explicit_pixel_meaning(self):
        _, t = native_letterbox(np.zeros((90,160,3),np.uint8),maximum_side=320)
        prediction = self.tensor(t,[(0,80,40,60,.9,0),(220,1,20,2,.9,0)])
        rows = person_rows(prediction,t,config=PersonDetectorConfig())
        self.assertEqual(len(rows),1);self.assertEqual(rows[0,0],0)

    def test_invalid_pixels_configs_and_predictions_fail_before_fake_evidence(self):
        for value in (True,0,31,33,1312):
            with self.assertRaises(ValueError): PersonDetectorConfig(maximum_side=value)
        for value in (True,0,1,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): PersonDetectorConfig(confidence=value)
        for pixels in (np.zeros((2,2),np.uint8), np.zeros((2,2,3),np.float32),np.zeros((0,2,3),np.uint8)):
            with self.assertRaises(ValueError):native_letterbox(pixels,maximum_side=320)
        _, t = native_letterbox(np.zeros((32,32,3),np.uint8),maximum_side=32)
        good = self.tensor(t,[(16,16,10,10,.8,0)])
        for bad in (good.astype(np.float64),good[:,1:,:],good[:,:,:-1],good*np.nan):
            with self.assertRaises(ValueError):person_rows(bad,t,config=PersonDetectorConfig())
        for index,value in ((4,1.1),(4,-.1),(2,-2)):
            bad=good.copy();bad[0,index,0]=value
            with self.assertRaises(ValueError):person_rows(bad,t,config=PersonDetectorConfig())

    def test_adapter_owns_one_actual_forward_and_cache_fingerprint(self):
        _, t=native_letterbox(np.zeros((32,32,3),np.uint8),maximum_side=32)
        result=self.tensor(t,[(16,16,10,10,.8,0)])
        with patch('crosscut.native_person_onnx.OnnxCpuNetwork') as factory:
            factory.return_value.forward.return_value=result
            factory.return_value.fingerprint.return_value={'model_sha256':'a'*64}
            model=NativePersonOnnxDetector(Path('explicit.onnx'),expected_sha256='a'*64,
                                           config=PersonDetectorConfig(maximum_side=32))
            rows=model.infer(np.zeros((32,32,3),np.uint8))
            self.assertEqual(len(rows),1)
            factory.return_value.forward.assert_called_once_with()
            self.assertFalse(model.fingerprint()['cast_identity_or_pose_certified'])
            self.assertEqual(model.fingerprint()['maximum_side'],32)
            self.assertEqual(model.fingerprint()['inference_calls_per_observation'],1)
            factory.assert_called_once_with(Path('explicit.onnx'),expected_sha256='a'*64,
                                           output_names=('output0',),intra_op_threads=4)


if __name__ == '__main__':
    unittest.main()
