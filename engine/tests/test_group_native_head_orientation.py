"""Synthetic analysis-inverse/budget regressions, not performer quality QA."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

from crosscut.group_native_head_orientation import (SUPPORTED_ANALYSIS_ROLLS,
    inverse_mapped_head,register_native_head_orientation_models,observe_native_head_orientation_alternatives)
from crosscut.group_joint_source_camera import JointGroupWorkBudget,JointGroupBudgetExhausted
from tests import test_group_joint_temporal_pose as pose_fixtures


class NativeHeadOrientationTests(unittest.TestCase):
    def row(self):
        return np.asarray((390.,140.,150.,180.,425.,190.,475.,190.,450.,230.,425.,270.,475.,270.,.92))

    def mapped(self,angle=30.,**kw):
        args=dict(row=self.row(),working_size_wh=(1000,500),
            analysis_affine=cv2.getRotationMatrix2D((499.5,249.5),angle,1.),
            analysis_roll_degrees=angle,observation_id=0)
        args.update(kw)
        return inverse_mapped_head(**args)

    def test_all_frozen_rolls_round_trip_landmarks_in_non_square_native_geometry(self):
        for angle in SUPPORTED_ANALYSIS_ROLLS:
            value=self.mapped(angle)
            affine=np.asarray(value.source_to_analysis_pixel_affine)
            points=np.asarray(value.head.landmarks_xy)*[1000,500]
            replay=np.column_stack((points,np.ones(5)))@affine.T
            np.testing.assert_allclose(replay,self.row()[4:14].reshape(5,2),rtol=0.,atol=1e-10)
            corners=np.asarray(value.inverse_mapped_face_roi_corners_xy)*[1000,500]
            back=np.column_stack((corners,np.ones(4)))@affine.T
            np.testing.assert_allclose(back,((390.,140.),(540.,140.),(540.,320.),(390.,320.)),atol=1e-10)
            self.assertFalse(value.to_dict()['named_or_semantic_identity_verified'])
            self.assertTrue(value.to_dict()['detector_roi_not_physical_whole_head_extent'])

    def test_off_native_roi_is_not_clamped_or_relabelled_a_valid_extent(self):
        row=self.row();row[:4]=(-100.,-100.,70.,60.)
        value=self.mapped(30.,row=row)
        self.assertTrue(any(v<0. or v>1. for p in value.inverse_mapped_face_roi_corners_xy for v in p))
        self.assertTrue(value.to_dict()['source_coordinates_not_clamped'])
        self.assertFalse(value.to_dict()['candidate_admitted'])

    def test_declared_roll_affine_center_and_scale_cannot_be_forged(self):
        for affine in (cv2.getRotationMatrix2D((500.,250.),30.,1.),
                cv2.getRotationMatrix2D((499.5,249.5),-30.,1.),
                cv2.getRotationMatrix2D((499.5,249.5),30.,1.1),np.zeros((2,3)),np.full((2,3),np.nan)):
            with self.subTest(affine=affine),self.assertRaises(ValueError):
                self.mapped(analysis_affine=affine)

    def test_malformed_detector_rows_and_working_types_are_rejected(self):
        for args in ({'row':np.zeros(15)},{'row':np.ones(14)},
                {'working_size_wh':(True,500)},{'analysis_roll_degrees':True},
                {'observation_id':True},{'working_size_wh':(0,500)}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                self.mapped(**args)

    def fake_session(self,budget,pixels):
        domain=pose_fixtures.NativeTemporalPoseTests().domain()
        session=SimpleNamespace(budget=budget,domain=domain,models={},model_adapters={},callbacks={},
            _assert_source_identity=Mock(return_value=(1,2,3,4,5)),
            native_pixels=Mock(return_value=pixels))
        def register(name,path,observer,adapter_fingerprint):
            session.models[name]=(Path(path),observer,hashlib.sha256(Path(path).read_bytes()).hexdigest())
            session.model_adapters[name]='a'*64;session.callbacks[name]=observer
        session.register_model=Mock(side_effect=register)
        def observe(source,native,model):
            budget.spend('model_calls',model=model)
            return session.callbacks[model](pixels)
        session.observe_native=Mock(side_effect=observe)
        return session

    def test_one_registered_counted_call_per_actual_detector_variant_keeps_original_pixels(self):
        pixels=np.zeros((500,1000,3),np.uint8);pixels[200,500]=(12,34,56)
        before=hashlib.sha256(memoryview(pixels)).hexdigest()
        budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_model_calls=4)
        session=self.fake_session(budget,pixels)
        detectors=[Mock(detect=Mock(return_value=(None,self.row()[None,:]))) for _ in range(4)]
        with TemporaryDirectory() as tmp:
            asset=Path(tmp)/'synthetic-unopened-head.onnx';asset.write_bytes(b'synthetic-model-not-loadable')
            with patch('crosscut.group_native_head_orientation.YUNET_EXPECTED_SHA256',hashlib.sha256(asset.read_bytes()).hexdigest()), \
                    patch('crosscut.group_native_head_orientation.cv2.FaceDetectorYN.create',side_effect=detectors):
                names=register_native_head_orientation_models(session,asset)
                result=observe_native_head_orientation_alternatives(session,request_source_index=0,native_frame_index=0)
        self.assertEqual(len(names),4)
        self.assertEqual(budget.counts['model_calls'],4)
        self.assertEqual(tuple(budget.model_calls),(names))
        self.assertTrue(all(d.detect.call_count==1 for d in detectors))
        self.assertEqual(result['native_pixel_sha256'],before)
        self.assertEqual(hashlib.sha256(memoryview(pixels)).hexdigest(),before)
        self.assertEqual(sum(len(v['alternatives']) for v in result['variants']),4)
        self.assertFalse(result['hypotheses_deduplicated_or_subject_selected'])
        self.assertFalse(result['source_pixels_or_clock_or_render_modified'])

    def test_missing_or_overflow_detector_is_not_a_pruned_success(self):
        pixels=np.zeros((500,1000,3),np.uint8)
        budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
        session=self.fake_session(budget,pixels)
        detectors=[Mock(detect=Mock(return_value=(None,None))),
            Mock(detect=Mock(return_value=(None,np.tile(self.row(),(25,1)))))]
        with TemporaryDirectory() as tmp:
            asset=Path(tmp)/'synthetic.onnx';asset.write_bytes(b'test')
            with patch('crosscut.group_native_head_orientation.YUNET_EXPECTED_SHA256',hashlib.sha256(asset.read_bytes()).hexdigest()), \
                    patch('crosscut.group_native_head_orientation.cv2.FaceDetectorYN.create',side_effect=detectors):
                register_native_head_orientation_models(session,asset,angles=(-30.,30.))
                result=observe_native_head_orientation_alternatives(session,request_source_index=0,native_frame_index=0,angles=(-30.,30.))
        self.assertEqual(result['variants'][0]['alternatives'],[])
        self.assertFalse(result['variants'][0]['unavailable'])
        self.assertTrue(result['variants'][1]['unavailable'])
        self.assertEqual(result['variants'][1]['alternatives'],[])
        self.assertEqual(budget.counts['model_calls'],2)

    def test_variant_budget_cannot_hide_four_invocations_in_one_debit(self):
        pixels=np.zeros((500,1000,3),np.uint8)
        budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_model_calls=1)
        session=self.fake_session(budget,pixels)
        detector=Mock(detect=Mock(return_value=(None,None)))
        with TemporaryDirectory() as tmp:
            asset=Path(tmp)/'synthetic.onnx';asset.write_bytes(b'test')
            with patch('crosscut.group_native_head_orientation.YUNET_EXPECTED_SHA256',hashlib.sha256(asset.read_bytes()).hexdigest()), \
                    patch('crosscut.group_native_head_orientation.cv2.FaceDetectorYN.create',return_value=detector):
                register_native_head_orientation_models(session,asset)
                with self.assertRaisesRegex(JointGroupBudgetExhausted,'model_calls'):
                    observe_native_head_orientation_alternatives(session,request_source_index=0,native_frame_index=0)
        self.assertEqual(budget.counts['model_calls'],1)
        self.assertEqual(detector.detect.call_count,1)

    def test_bad_angles_and_unpinned_model_fail_before_registration(self):
        session=Mock(budget=Mock())
        for angles in ((),[],(-30.,-30.),(0.,),(True,),(-90.,),SUPPORTED_ANALYSIS_ROLLS+(30.,)):
            with self.subTest(angles=angles),self.assertRaises(ValueError):
                register_native_head_orientation_models(session,Path('unopened.onnx'),angles=angles)
        with TemporaryDirectory() as tmp:
            asset=Path(tmp)/'bad.onnx';asset.write_bytes(b'not-pinned')
            with self.assertRaisesRegex(ValueError,'pinned YuNet'):
                register_native_head_orientation_models(session,asset)
        session.register_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
