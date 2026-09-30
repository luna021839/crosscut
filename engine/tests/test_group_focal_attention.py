from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.boundary_subject_alignment import _default_observation_provider
from crosscut.closeup_gesture import CloseupGestureProvider
from crosscut.group_focal_attention import (
    CENTRAL_FOCAL_BASIS, select_central_focal_face)
from crosscut.pose import FormationPersonDescriptor
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_score import _DetectedFace
from tests.test_boundary_subject_alignment import _frame
from tests.test_closeup_gesture import observation as arm_observation
from tests.test_visual_boundary_repair import _descriptor


def face(box):
    x, y, w, h = box
    return _DetectedFace(tuple(v * scale for v, scale in zip(box, (854,480,854,480))),
        tuple(((x+dx*w)*854, (y+dy*h)*480) for dx,dy in
              ((.3,.3),(.7,.3),(.5,.5),(.3,.8),(.7,.8))), .95)


def person(box):
    x,y,w,h = box
    return FormationPersonDescriptor(box, (x+w/2,y+h/2), .95)


class GroupFocalAttentionTests(unittest.TestCase):
    def setUp(self):
        self.center = face((.46,.30,.09,.16))
        self.peripheral = face((.05,.25,.10,.16))
        self.body = person((.34,.25,.32,.74))

    def select(self, faces, people=None):
        return select_central_focal_face(faces, (self.body,) if people is None else people,
                                        frame_width=854, frame_height=480)

    def test_large_peripheral_face_does_not_erase_central_attention(self):
        result = self.select((self.peripheral, self.center))
        self.assertIsNotNone(result)
        self.assertEqual(2, result.detected_face_count)
        self.assertEqual(self.body, result.person)
        self.assertAlmostEqual(.46, result.bbox_xywh[0])

    def test_equal_central_candidates_stay_ambiguous(self):
        other = face((.32,.31,.085,.16))
        self.assertIsNone(self.select((self.center, other)))

    def test_missing_owner_cannot_fall_through_to_smaller_visible_person(self):
        small = face((.33,.32,.04,.08))
        small_body = person((.29,.28,.12,.68))
        self.assertIsNone(self.select((self.center,small),(small_body,)))
        self.assertIsNone(self.select((self.center,), (self.body,self.body)))

    def test_tiny_wide_face_or_missing_landmarks_cannot_authorize_medium_framing(self):
        self.assertIsNone(self.select((face((.46,.30,.03,.07)),)))
        self.assertIsNone(self.select((replace(self.center, landmarks_xy=None),)))
        self.assertIsNone(self.select((self.peripheral,)))

    def test_focal_gesture_extension_does_not_overwrite_geometry_planner(self):
        original = replace(_descriptor(.5), face_count=2, primary_face_bbox=None,
            primary_face_landmarks_xy=None, primary_face_eye_scale=None,
            formation_people=(self.body,))
        with patch('crosscut.visual_score.describe_frame', return_value=original), \
             patch('crosscut.group_focal_attention.observe_central_focal_face') as focal:
            result = _default_observation_provider(_frame(0))
        focal.assert_not_called()
        self.assertIsNone(original.primary_face_bbox)
        self.assertIsNone(result.face_box_xywh)
        self.assertFalse(result.face_is_unambiguous_dominant)

    def test_exact_gesture_can_measure_central_face_without_forging_identity(self):
        focal = self.select((self.peripheral,self.center))
        descriptor = SimpleNamespace(primary_face_bbox=None,formation_people=(self.body,))
        with TemporaryDirectory() as directory:
            source=Path(directory)/'a.mp4'
            source.touch()
            with patch('crosscut.boundary_subject_alignment._CaptureProvider') as capture, \
                 patch('crosscut.closeup_gesture.BodyPose2DAnalyzer') as pose, \
                 patch('crosscut.group_focal_attention.observe_central_focal_face',return_value=focal):
                capture.return_value.frame_at_native_index.return_value=_frame(0)
                pose.return_value.describe.return_value=arm_observation()
                provider=CloseupGestureProvider((source,),Path('pose.onnx'))
                clock=VerifiedRenderFrameClock(Fraction(30),300,'a'*64)
                try:
                    evidence=provider.compare((0,1.,clock,descriptor),(0,2.,clock,descriptor))
                    again=provider.compare((0,1.,clock,descriptor),(0,2.,clock,descriptor))
                    self.assertEqual(evidence,again)
                    self.assertEqual(2,pose.return_value.describe.call_count)
                    self.assertEqual((30,60),(evidence.outgoing_native_frame,evidence.incoming_native_frame))
                    self.assertEqual(CENTRAL_FOCAL_BASIS,evidence.outgoing_focal_basis)
                    self.assertFalse(evidence.to_dict()['cross_camera_identity_proven'])
                    self.assertFalse(evidence.to_dict()['temporal_phase_proven'])
                finally:
                    provider.close()

    def test_primary_face_with_missing_body_does_not_substitute_a_central_person(self):
        descriptor=SimpleNamespace(primary_face_bbox=(.1,.1,.1,.1),formation_people=(self.body,))
        with TemporaryDirectory() as directory:
            source=Path(directory)/'a.mp4'; source.touch()
            with patch('crosscut.boundary_subject_alignment._CaptureProvider') as capture, \
                 patch('crosscut.group_focal_attention.observe_central_focal_face') as central:
                provider=CloseupGestureProvider((source,),Path('pose.onnx'))
                try:
                    self.assertIsNone(provider.observe(0,1.,
                        VerifiedRenderFrameClock(Fraction(30),300,'a'*64),descriptor))
                    central.assert_not_called()
                    capture.return_value.frame_at_native_index.assert_not_called()
                finally:
                    provider.close()
