from dataclasses import replace
import math
import unittest
from unittest.mock import Mock

from crosscut.closeup_gesture import (
    CloseupGestureEvidence, CloseupGestureProvider, ObservedArmMotion,
    compare_arm_articulation, measure_arm_motion,
)
from crosscut.group_focal_attention import GLOBAL_FOCAL_BASIS
from tests.test_closeup_gesture import observation


def wrist(angle):
    value=observation()
    points=list(value.upper_body_xyc)
    points[4]=(.3+math.cos(angle)*100/1600,.6+math.sin(angle)*100/900,.9)
    return replace(value,upper_body_xyc=tuple(points))


class CloseupArmMotionTests(unittest.TestCase):
    def measure(self, angles, **kwargs):
        return measure_arm_motion(tuple(wrist(a) for a in angles),
            **dict(dict(native_frames=(0,3,7,10),
                        output_intervals_seconds=(.1,1/30,.1)),**kwargs))

    def test_continuing_motion_beats_identical_still_pose_with_reverse_motion(self):
        continuous=self.measure((0.,.3,.4,.7))
        backwards=self.measure((0.,.3,.3,0.))
        self.assertAlmostEqual(0.,continuous.cost)
        self.assertGreater(backwards.cost,.06)
        static=compare_arm_articulation(wrist(.3),wrist(.3),
            outgoing_native_frame=3,incoming_native_frame=7)
        self.assertAlmostEqual(0.,static.mismatch)
        improved=compare_arm_articulation(wrist(.3),wrist(.4),
            outgoing_native_frame=3,incoming_native_frame=7)
        self.assertGreater(improved.mismatch,static.mismatch)
        self.assertTrue(replace(improved,motion=continuous).improves_over(
            replace(static,motion=backwards)))

    def test_angle_wrap_does_not_turn_small_motion_into_full_rotation(self):
        evidence=self.measure((math.pi-.3,math.pi,math.pi+.1,math.pi+.4))
        self.assertAlmostEqual(0.,evidence.cost)

    def test_output_time_not_native_id_distance_controls_motion_units(self):
        first=self.measure((0.,.3,.4,.7))
        other=self.measure((0.,.3,.4,.7),native_frames=(0,6,70,76))
        self.assertAlmostEqual(first.cost,other.cost)

    def test_smaller_velocity_penalty_cannot_hide_larger_predicted_pose_jump(self):
        before=CloseupGestureEvidence(3,7,(.2,)*4,motion=ObservedArmMotion(
            (0,3,7,10),(.1,1/30,.1),(.6,)*4,(1.3,)*4))
        after=replace(before,motion=ObservedArmMotion((0,3,7,10),(.1,1/30,.1),(.8,)*4,(.2,)*4))
        self.assertLess(after.ranking_cost,before.ranking_cost)
        self.assertFalse(after.improves_over(before))

    def test_translation_and_pixel_scale_preserve_measurement(self):
        observations=tuple(wrist(a) for a in (0.,.3,.4,.7))
        transformed=tuple(replace(o,upper_body_xyc=tuple((x*.6+.1,y*.6+.1,c)
            for x,y,c in o.upper_body_xyc),frame_size_wh=(3200,1800)) for o in observations)
        result=measure_arm_motion(transformed,native_frames=(0,3,7,10),
            output_intervals_seconds=(.1,1/30,.1))
        self.assertAlmostEqual(0.,result.cost)

    def test_missing_frame_and_missing_common_bones_are_not_zero_cost(self):
        observations=[wrist(0.)]*4
        observations[0]=None
        self.assertIsNone(measure_arm_motion(observations,native_frames=(0,3,7,10),
            output_intervals_seconds=(.1,1/30,.1)))
        left,right=list(wrist(0.).upper_body_xyc),list(wrist(0.).upper_body_xyc)
        for i in (0,2,4): left[i]=(*left[i][:2],.1)
        for i in (1,3,5): right[i]=(*right[i][:2],.1)
        self.assertIsNone(measure_arm_motion((replace(wrist(0.),upper_body_xyc=tuple(left)),
            wrist(0.),wrist(0.),replace(wrist(0.),upper_body_xyc=tuple(right))),
            native_frames=(0,3,7,10),output_intervals_seconds=(.1,1/30,.1)))

    def test_dropping_a_motion_bone_or_switching_objective_cannot_improve(self):
        before=CloseupGestureEvidence(3,7,(.2,)*4,motion=ObservedArmMotion(
            (0,3,7,10),(.1,1/30,.1),(.5,.5,.5,None),(.1,.1,.1,None)))
        hidden=replace(before,motion=ObservedArmMotion((0,3,7,10),(.1,1/30,.1),
            (.01,None,.01,None),(.01,None,.01,None)))
        static=replace(before,motion=None,bone_mismatches=(.0,)*4)
        self.assertFalse(hidden.improves_over(before))
        self.assertFalse(static.improves_over(before))
        self.assertFalse(before.preserves_observations_of(static))

    def test_reversed_cloned_invalid_clocks_and_nonfinite_costs_rejected(self):
        for ids in ((3,3,7,10),(4,3,7,10),(0,3,10,7),(True,3,7,10)):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                self.measure((0.,.3,.4,.7),native_frames=ids)
        for intervals in ((0.,1/30,.1),(.1,float('nan'),.1),(.1,True,.1)):
            with self.subTest(intervals=intervals),self.assertRaises(ValueError):
                self.measure((0.,.3,.4,.7),output_intervals_seconds=intervals)
        with self.assertRaises(ValueError):
            ObservedArmMotion((0,3,7,10),(.1,1/30,.1),
                (.1,float('nan'),None,None),(.1,.2,None,None))

    def test_payload_is_scalar_auditable_and_does_not_claim_phase_approval(self):
        motion=self.measure((0.,.3,.3,0.))
        evidence=CloseupGestureEvidence(3,7,(0.,)*4,motion=motion)
        payload=evidence.to_dict()
        self.assertEqual(evidence,CloseupGestureEvidence.from_dict(payload))
        for key in ('calibrated','quality_approved','temporal_phase_proven','cross_camera_identity_proven'):
            self.assertIs(payload['motion'][key],False)
        self.assertNotIn('upper_body_xyc',repr(payload))
        self.assertNotIn('pose_embedding',repr(payload))
        payload['motion']['cost']=0.
        with self.assertRaises(ValueError): CloseupGestureEvidence.from_dict(payload)

    def test_motion_cannot_be_attached_to_different_endpoint_frames(self):
        with self.assertRaises(ValueError):
            CloseupGestureEvidence(4,7,(0.,)*4,motion=self.measure((0.,.3,.4,.7)))

    def test_malformed_payload_cannot_invent_quality_or_crash_as_attribute_error(self):
        for bad in (None,[],{},dict(self.measure((0.,.3,.4,.7)).to_dict(),calibrated=0),
                    dict(self.measure((0.,.3,.4,.7)).to_dict(),quality_approved=True),
                    dict(self.measure((0.,.3,.4,.7)).to_dict(),cost=False)):
            with self.subTest(payload=bad),self.assertRaises(ValueError):
                ObservedArmMotion.from_dict(bad)

    def test_provider_binds_four_samples_and_refuses_source_run_crossing(self):
        provider=object.__new__(CloseupGestureProvider)
        provider.observe=Mock(side_effect=[(n,wrist(a),GLOBAL_FOCAL_BASIS)
            for n,a in zip((0,3,7,10),(0.,.3,.4,.7))])
        samples=((0,0.,None,None),(0,.1,None,None),(1,7/30,None,None),(1,10/30,None,None))
        evidence=provider.compare_motion(*samples,output_fps=30.)
        self.assertAlmostEqual(0.,evidence.motion.cost)
        self.assertEqual(4,provider.observe.call_count)
        with self.assertRaises(ValueError):
            provider.compare_motion((2,0.,None,None),*samples[1:],output_fps=30.)


if __name__=='__main__': unittest.main()
