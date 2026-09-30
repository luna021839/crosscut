from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from unittest.mock import patch,Mock
import unittest
import numpy as np

from crosscut.native_global_face import observe_native_global_face,NativeGlobalFace
from crosscut.visual_score import _DetectedFace
from crosscut.closeup_gesture import CloseupGestureProvider
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.group_focal_attention import GLOBAL_FOCAL_BASIS
from engine.tests.test_global_action_cut_camera import person,pose,observation


def face(box):
    x,y,w,h=box
    points=((x+w*.3,y+h*.35),(x+w*.7,y+h*.35),(x+w*.5,y+h*.55),
        (x+w*.35,y+h*.75),(x+w*.65,y+h*.75))
    return _DetectedFace(tuple(v*(854 if i%2==0 else 480) for i,v in enumerate(box)),
        tuple((x*854,y*480) for x,y in points),.96)


class NativeGlobalFaceTests(unittest.TestCase):
    def test_foreground_is_selected_from_native_set_not_a_legacy_background(self):
        foreground=(.40,.22,.12,.22);background=(.8,.3,.025,.06)
        frame=np.zeros((1080,1920,3),np.uint8)
        with patch('crosscut.native_global_face.detect_roll_aware_faces',return_value=(face(background),face(foreground))) as detect:
            result=observe_native_global_face(frame)
        np.testing.assert_allclose(result.bbox_xywh,foreground)
        self.assertEqual(result.face_count,2)
        self.assertEqual(detect.call_args.args[0].shape,(480,854,3))
        self.assertEqual(detect.call_args.kwargs['angles'],(0.,))
        self.assertEqual(len(result.geometry),10)

    def test_upright_ambiguity_cannot_become_a_larger_rotated_box_guess(self):
        with patch('crosscut.native_global_face.detect_roll_aware_faces',return_value=
                (face((.1,.2,.12,.2)),face((.6,.2,.13,.2)))) as detect:
            result=observe_native_global_face(np.zeros((480,854,3),np.uint8))
        self.assertIsNone(result.bbox_xywh)
        self.assertIsNone(result.geometry)
        self.assertEqual(detect.call_count,1,'area margin is unchanged, no rotation gamble on actual ambiguity')

    def test_roll_recovery_is_bounded_to_an_empty_upright_set(self):
        for recovered in ((),(face((.3,.2,.2,.3)),)):
            with self.subTest(recovered=bool(recovered)),patch('crosscut.native_global_face.detect_roll_aware_faces',side_effect=((),recovered)) as detect:
                result=observe_native_global_face(np.zeros((480,854,3),np.uint8))
            self.assertEqual(detect.call_count,2)
            self.assertEqual(result.bbox_xywh is not None,bool(recovered))


class NativeGlobalArmOwnerTests(unittest.TestCase):
    def provider(self):
        provider=CloseupGestureProvider.__new__(CloseupGestureProvider)
        provider.sources=(Path('a.mp4'),);provider.identities=('same',);provider._identity=lambda p:'same'
        provider.capture=SimpleNamespace(frame_at_native_index=lambda *a:np.zeros((480,854,3),np.uint8))
        provider.pose=Mock();provider.pose.describe.return_value=pose(person())
        provider.cache=OrderedDict();provider.verify_native_global_focal=True
        return provider

    def native(self):
        row=observation(0)
        return NativeGlobalFace(2,row.face_box_xywh,row.face_landmarks_xy,0.,row.face_eye_scale,None)

    def test_corrected_foreground_arms_must_own_the_new_face_not_the_cached_background(self):
        provider=self.provider();body=person();foreground=self.native()
        descriptor=SimpleNamespace(primary_face_bbox=(.8,.2,.03,.07),formation_people=(body,))
        clock=VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)
        with patch('crosscut.native_global_face.observe_native_global_face',return_value=foreground), \
                patch('crosscut.group_focal_attention.resolve_group_focal_subject') as central:
            result=provider.observe(0,1.,clock,descriptor)
        self.assertEqual(result[2],GLOBAL_FOCAL_BASIS)
        self.assertEqual(provider.pose.describe.call_args.kwargs['face_bbox_xywh'],foreground.bbox_xywh)
        self.assertEqual(provider.pose.describe.call_args.kwargs['bbox_xywh'],body.bbox_xywh)
        central.assert_not_called()
        self.assertEqual(descriptor.primary_face_bbox,(.8,.2,.03,.07),'persistent evidence is untouched')

    def test_native_foreground_with_no_unique_body_cannot_borrow_background_arms(self):
        for people in ((),(person(),person())):
            provider=self.provider();descriptor=SimpleNamespace(primary_face_bbox=(.8,.2,.03,.07),formation_people=people)
            with self.subTest(people=len(people)),patch('crosscut.native_global_face.observe_native_global_face',return_value=self.native()):
                result=provider.observe(0,1.,VerifiedRenderFrameClock(Fraction(30),3000,'a'*64),descriptor)
            self.assertIsNone(result);provider.pose.describe.assert_not_called()


if __name__=='__main__':unittest.main()
