from dataclasses import replace
import unittest
from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.closeup_gesture import arm_articulation, compare_arm_articulation


def observation():
    return BodyPose2DEvidence('observed', (.1,.1,.8,.9), (1600,900),
        ((.4,.4,.9),(.6,.4,.9),(.3,.6,.9),(.7,.6,.9),
         (.3,.3,.9),(.7,.3,.9),(.4,1.1,.1),(.6,1.1,.1)), True)


class CloseupGestureTests(unittest.TestCase):
    def pair(self, a, b):
        return compare_arm_articulation(a,b,outgoing_native_frame=31,incoming_native_frame=60)

    def test_unseen_hips_do_not_hide_four_visible_arm_bones(self):
        a = observation()
        self.assertIsNone(a.as_formation_person(.9))
        evidence = self.pair(a,a)
        self.assertAlmostEqual(0., evidence.mismatch)
        self.assertEqual((31,60), (evidence.outgoing_native_frame,evidence.incoming_native_frame))
        self.assertFalse(evidence.to_dict()['temporal_phase_proven'])

    def test_uniform_pixel_scale_and_translation_are_invariant(self):
        a = observation()
        values = tuple((x*.6+.1,y*.6+.1,c) for x,y,c in a.upper_body_xyc)
        b = replace(a, upper_body_xyc=values, frame_size_wh=(3200,1800))
        self.assertAlmostEqual(0.,self.pair(a,b).mismatch)

    def test_real_pixel_aspect_not_normalized_xy_angle(self):
        a=observation()
        b=replace(a, frame_size_wh=(900,1600))
        self.assertGreater(self.pair(a,b).mismatch,.05)

    def test_single_wrong_hand_is_not_averaged_away(self):
        a=observation(); values=list(a.upper_body_xyc); values[4]=(.3,.9,.9)
        b=replace(a,upper_body_xyc=tuple(values))
        evidence=self.pair(a,b)
        self.assertAlmostEqual(1.,evidence.mismatch)
        self.assertAlmostEqual(1.,sum(evidence.bone_mismatches))

    def test_missing_ambiguous_or_offscreen_arm_is_unknown_not_perfect(self):
        a=observation()
        bad=[None,replace(a,face_anchor_consistent=False),replace(a,face_anchor_consistent=None),
             replace(a,reason='pose_inference_failed')]
        for value in ((.3,.3,.29),(1.1,.3,.9),(float('nan'),.3,.9),(.3,.6,.9)):
            points=list(a.upper_body_xyc); points[4]=value
            bad.append(replace(a,upper_body_xyc=tuple(points)))
        for b in bad:
            with self.subTest(b=b): self.assertIsNone(self.pair(a,b))

    def test_wrong_native_id_cannot_be_published(self):
        for index in (True,-1,1.5):
            with self.assertRaises(ValueError):
                compare_arm_articulation(observation(),observation(),
                    outgoing_native_frame=index,incoming_native_frame=1)
