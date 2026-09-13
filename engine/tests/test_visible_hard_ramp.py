import unittest
from crosscut.render import EditorialTransformKeyframe as K,EditorialTransformTrack as T
from crosscut.visible_hard_ramp import shorten_hard_ramp_to_observations
from crosscut.direct_member_editorial_motion import build_direct_member_editorial_motion_plan
from crosscut.render import RenderTransition
from engine.tests import test_direct_member_editorial_motion as fixture

class VisibleHardRampTests(unittest.TestCase):
    def test_outgoing_gap_moves_only_identity_plateau_and_keeps_boundary(self):
        track=T((K(0.,1.),K(1.25,1.),K(2.,1.3)))
        calls=[]
        def observed(i):calls.append(i);return i not in (37,38,39)
        result=shorten_hard_ramp_to_observations(track,fps=30.,frame_count=61,frame_available=observed)
        self.assertIsNotNone(result)
        self.assertEqual(result.final_keyframe,track.final_keyframe)
        self.assertEqual(result.keyframes[1].time_seconds,40/30)
        self.assertGreaterEqual(min(calls),37)

    def test_incoming_gap_ends_return_before_unobserved_frame(self):
        track=T((K(0.,1.3),K(.75,1.),K(2.,1.)))
        result=shorten_hard_ramp_to_observations(track,fps=30.,frame_count=61,frame_available=lambda i:i!=22)
        self.assertEqual(result.keyframes[0],track.keyframes[0])
        self.assertEqual(result.keyframes[1].time_seconds,.7)

    def test_endpoint_gap_and_too_fast_ramp_remain_unresolved(self):
        track=T((K(0.,1.),K(1.25,1.),K(2.,1.3)))
        for missing in (60,47):
            self.assertIsNone(shorten_hard_ramp_to_observations(track,fps=30.,frame_count=61,frame_available=lambda i:i!=missing))
        self.assertIsNone(shorten_hard_ramp_to_observations(track,fps=30.,frame_count=61,frame_available=lambda i:True))

    def test_actual_planner_keeps_zoom_for_short_gap_at_ramp_start(self):
        f=fixture.DirectMemberEditorialMotionTests();selection,renders,boxes=f._basic_inputs(duration=8.)
        def provider(s,t):return None if s==0 and 7.20<=t<=7.30 else f._sample(boxes[s])
        plan=build_direct_member_editorial_motion_plan(visual_selection=selection,render_segments=renders,
            transitions=(RenderTransition(),),common_master_start=0.,geometry_provider=provider,fps=30.)
        self.assertEqual(plan.boundary_match_count,1)
        self.assertIsNotNone(plan.track_for(0))
        self.assertGreater(plan.track_for(0).final_keyframe.zoom,1.05)

if __name__=='__main__':unittest.main()
