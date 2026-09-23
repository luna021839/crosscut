from dataclasses import replace
import unittest
from unittest.mock import patch

from crosscut.direct_member_editorial_motion import build_direct_member_editorial_motion_plan
from crosscut.render import RenderTransition,TransitionKind,DIRECT_MEMBER_POSE_DISSOLVE_REASON
from engine.tests.member_roll_fixture import direct_inputs,sample,focal


class DirectMemberRollTests(unittest.TestCase):
    def solve(self,provider=focal,inputs=None):
        selection,segments,transitions=inputs or direct_inputs()
        return build_direct_member_editorial_motion_plan(visual_selection=selection,render_segments=segments,
            transitions=transitions,common_master_start=0.,geometry_provider=sample,
            pose_geometry_provider=provider,fps=30.)

    def test_actual_dense_planner_authors_supported_roll_after_following(self):
        plan=self.solve()
        self.assertGreater(plan.roll_event_count,0,plan.to_dict())
        self.assertGreater(plan.track_for(0).final_keyframe.roll_degrees,.5)
        self.assertLess(plan.track_for(1).keyframes[0].roll_degrees,-.5)
        row=plan.member_roll_audit[0]
        self.assertTrue(row['accepted'],row)
        self.assertLess(row['pose_rms_after'],row['pose_rms_before'])
        with self.assertRaises(TypeError):row['accepted']=False

    def test_missing_or_wrong_source_pose_preserves_all_original_camera_tracks(self):
        base=self.solve(None)
        for provider in (lambda s,t:None,lambda s,t:focal(s,t+.02),
                         lambda s,t:replace(focal(s,t),pose_backend='formation_pose')):
            with self.subTest(provider=provider):
                result=self.solve(provider)
                self.assertEqual(result.segments,base.segments)
                self.assertEqual(result.roll_event_count,0)

    def test_missing_dense_body_rejects_only_rotation(self):
        base=self.solve(None)
        def missing(s,t):
            value=focal(s,t)
            return replace(value,person=None,pose_screen_xyc=()) if s==0 and 4.3<t<4.6 else value
        result=self.solve(missing)
        self.assertEqual(result.segments,base.segments)
        self.assertFalse(result.member_roll_audit[0]['accepted'])

    def test_dissolve_is_owned_by_joint_solver_and_off_keeps_roll_available(self):
        selection,segments,transitions=direct_inputs()
        effect=RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON)
        on=self.solve(inputs=(selection,segments,(effect,)))
        off=self.solve(inputs=(selection,segments,transitions))
        self.assertEqual(on.roll_event_count,0)
        self.assertGreater(off.roll_event_count,0)

    def test_native_cut_window_prevents_roll_without_reselection(self):
        selection,segments,transitions=direct_inputs()
        selection.span_validity[0].sources[0].native_scene_cut_source_seconds=(4.99,)
        base=self.solve(None,inputs=(selection,segments,transitions))
        result=self.solve(inputs=(selection,segments,transitions))
        self.assertEqual(result.segments,base.segments)
        self.assertEqual(result.roll_event_count,0)

    def test_cancellation_is_not_hidden_as_missing_evidence(self):
        def cancel(s,t):raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):self.solve(cancel)

    def test_boundary_keeps_stricter_face_floor_than_dense_interior(self):
        base=self.solve(None)
        # A 99% visible face clears the existing 98% dense-frame floor, but
        # must not replace the 99.5% face requirement at the cut itself.
        with patch('crosscut.direct_member_roll.rotated_visible_fraction',return_value=.99):
            proposed=self.solve()
        self.assertEqual(proposed.segments,base.segments)
        self.assertEqual(proposed.roll_event_count,0)
