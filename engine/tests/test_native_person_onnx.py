from pathlib import Path
from unittest.mock import patch
import unittest
import hashlib
import shutil
import subprocess
import tempfile
from time import monotonic

import numpy as np

from crosscut.native_person_onnx import (
    PersonDetectorConfig, NativePersonOnnxDetector, native_letterbox, person_rows,
    register_native_person_model, MODEL_NAME)


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


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class NativePersonSessionTests(unittest.TestCase):
    """Real session/chart decoder, explicit fake network (not human evidence)."""
    def setUp(self):
        from crosscut.group_joint_observations import GroupJointObservationSession
        from crosscut.group_joint_source_camera import JointGroupWorkBudget
        from crosscut.group_source_domain import GroupSourceDomain
        from crosscut.render_frame_schedule import probe_render_frame_clock
        self.temporary=tempfile.TemporaryDirectory(prefix='crosscut-person-session-')
        self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name);self.model=root/'fixture.onnx'
        self.model.write_bytes(b'explicit fake network for registration/budget tests')
        self.sha=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.video=root/'chart.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=96x54:r=30',
            '-frames:v','8','-c:v','libx264','-pix_fmt','yuv420p',str(self.video)],
            capture_output=True,check=True,timeout=10.)
        clock=probe_render_frame_clock(self.video)
        domain=GroupSourceDomain((self.video,),(0,),(0.,),(None,),((),),(clock,))
        self.budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_model_calls=1)
        self.session=GroupJointObservationSession(domain=domain,budget=self.budget)
        self.addCleanup(self.session.close)
        _,transform=native_letterbox(np.zeros((54,96,3),np.uint8),maximum_side=320)
        prediction=NativePersonOnnxTests().tensor(transform,[(100,100,40,80,.8,0)])
        self.network_patch=patch('crosscut.native_person_onnx.OnnxCpuNetwork')
        self.factory=self.network_patch.start();self.addCleanup(self.network_patch.stop)
        self.factory.return_value.forward.return_value=prediction
        self.factory.return_value.fingerprint.return_value={'model_sha256':self.sha,'test_fixture':True}
        self.digest_patch=patch('crosscut.native_person_onnx.PINNED_YOLO11L_ONNX_SHA256',self.sha)
        self.digest_patch.start();self.addCleanup(self.digest_patch.stop)

    def register(self):
        return register_native_person_model(self.session,self.model,config=PersonDetectorConfig(maximum_side=320))

    def test_actual_session_names_charges_and_reuses_one_inference(self):
        from crosscut.group_joint_source_camera import JointGroupBudgetExhausted
        name=self.register();self.assertEqual(name,MODEL_NAME)
        first=self.session.observe_native(0,2,model=name)
        first[0,0]=999
        repeated=self.session.observe_native(0,2,model=name)
        self.assertNotEqual(repeated[0,0],999)
        self.assertEqual(self.budget.model_calls,{MODEL_NAME:1})
        self.factory.return_value.forward.assert_called_once_with()
        self.assertEqual(self.budget.counts['new_decodes'],1)
        with self.assertRaises(JointGroupBudgetExhausted):self.session.observe_native(0,3,model=name)
        self.assertFalse(self.session.model_adapter_preimages[name]['cast_identity_or_pose_certified'])
        self.assertNotIn('nanodet_people',self.session.models)

    def test_model_changed_after_loading_is_not_registered(self):
        original=self.factory.return_value.fingerprint
        def replace_asset():
            self.model.write_bytes(b'replaced fixture model')
            return original.return_value
        self.factory.return_value.fingerprint.side_effect=replace_asset
        with self.assertRaisesRegex(ValueError,'bytes loaded'):self.register()
        self.assertEqual(self.session.models,{})

    def test_duplicate_registration_cannot_silently_replace_config_or_cache(self):
        name=self.register();before=self.session.model_adapters[name]
        with self.assertRaisesRegex(ValueError,'already registered'):
            register_native_person_model(self.session,self.model,config=PersonDetectorConfig(maximum_side=640))
        self.assertEqual(self.session.model_adapters[name],before)
        self.assertEqual(self.factory.call_count,1)


if __name__ == '__main__':
    unittest.main()
