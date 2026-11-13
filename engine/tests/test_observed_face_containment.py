import unittest
from unittest.mock import patch
import numpy as np
from crosscut.observed_face_containment import recover_observed_face_step


class ObservedFaceContainmentTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((120,200,3),dtype=np.uint8)
        self.box=(.4,.2,.15,.3)
        self.landmarks=((.43,.3),(.5,.3),(.47,.35),(.44,.4),(.5,.4))

    def check_case(self, *, left=None, right=None, observed=None, embeddings=None):
        observations=((self.box,self.landmarks),)
        with patch('crosscut.observed_face_containment._observations',side_effect=[
                (self.frame, observations if left is None else left),
                (self.frame, observations if right is None else right)]), patch(
                'crosscut.observed_face_containment.extract_sface_embedding',
                side_effect=embeddings or [(1.,0.),(1.,0.)]):
            return recover_observed_face_step(self.frame,self.frame,self.box,
                       (self.box,) if observed is None else observed)

    def test_measured_same_local_face_can_recover(self):
        result=self.check_case()
        self.assertEqual(result.box,self.box)
        self.assertEqual(result.appearance_similarity,1.)

    def test_missing_face_in_either_frame_is_not_invented(self):
        self.assertIsNone(self.check_case(left=()))
        self.assertIsNone(self.check_case(right=()))
        self.assertIsNone(self.check_case(observed=()))

    def test_another_identity_at_same_location_is_rejected(self):
        self.assertIsNone(self.check_case(embeddings=[(1.,0.),(0.,1.)]))
        self.assertIsNone(self.check_case(embeddings=[(),(1.,0.)]))

    def test_background_face_cannot_be_found_by_appearance(self):
        self.assertIsNone(self.check_case(right=(((.05,.1,.15,.3),self.landmarks),)))

    def test_ambiguous_detected_or_native_faces_are_rejected(self):
        observation=(self.box,self.landmarks)
        self.assertIsNone(self.check_case(left=(observation,observation)))
        self.assertIsNone(self.check_case(right=(observation,observation)))
        self.assertIsNone(self.check_case(observed=(self.box,self.box)))

    def test_native_observation_must_agree(self):
        self.assertIsNone(self.check_case(observed=((.5,.2,.15,.3),)))

    def test_abrupt_scale_change_is_rejected(self):
        larger=(.4,.2,.19,.36)
        self.assertIsNone(self.check_case(right=((larger,self.landmarks),),observed=(larger,)))

    def test_invalid_pixels_and_boxes_are_rejected_before_models(self):
        with patch('crosscut.observed_face_containment._observations') as detect:
            for before, box, observations in ((None,self.box,(self.box,)),
                    (self.frame,(.4,.2,0.,.3),(self.box,)),
                    (self.frame,self.box,((.4,.2,float('nan'),.3),))):
                self.assertIsNone(recover_observed_face_step(before,self.frame,box,observations))
            detect.assert_not_called()

    def test_detector_resolution_is_explicit_bounded_and_does_not_bypass_identity(self):
        observations=((self.box,self.landmarks),)
        for size in (480,960):
            with patch('crosscut.observed_face_containment._observations',side_effect=[
                    (self.frame,observations),(self.frame,observations)]) as detect, \
                    patch('crosscut.observed_face_containment.extract_sface_embedding',
                        side_effect=[(1.,0.),(0.,1.)]):
                self.assertIsNone(recover_observed_face_step(self.frame,self.frame,self.box,
                    (self.box,),maximum_detection_width=size))
            self.assertEqual([call.kwargs for call in detect.call_args_list],
                [{'maximum_width':size}]*2)
        with patch('crosscut.observed_face_containment._observations') as detect:
            for size in (True,0,479,480.,961,1920):
                with self.subTest(size=size),self.assertRaises(ValueError):
                    recover_observed_face_step(self.frame,self.frame,self.box,(self.box,),
                        maximum_detection_width=size)
            detect.assert_not_called()

    def test_high_resolution_path_keeps_the_original_pixels_when_already_small_enough(self):
        from crosscut.observed_face_containment import _observations
        frame=np.zeros((480,854,3),np.uint8)
        with patch('crosscut.observed_face_containment._detect_faces',return_value=((),'yunet')) as detect:
            working,observations=_observations(frame,maximum_width=960)
        self.assertIs(working,frame)
        self.assertEqual(detect.call_args.args[0].shape,frame.shape)
        self.assertEqual(observations,())


if __name__=='__main__':
    unittest.main()
