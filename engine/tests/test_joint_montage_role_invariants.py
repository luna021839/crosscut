from dataclasses import replace
from unittest import TestCase

from crosscut.joint_visual_candidate_adapter import (
    JointDecodedMontageFeedback, JointDecodedMontageSegmentObservation,
    _decoded_montage_role_control_inputs,
)
from crosscut.timeline import ShotType
from engine.tests.test_joint_visual_candidate_adapter import _candidate_set


class JointMontageRoleInvariantTests(TestCase):
    def test_repeated_unknown_midpoint_clears_shadow_group_wide_flag_only(self):
        workspace=_candidate_set(role_conditioned=True).optimizer_workspace
        lattice=[list(row) for row in workspace.candidates_by_segment]
        target=replace(lattice[0][0],shot_type=ShotType.GROUP_WIDE,is_group_wide=True)
        lattice[0][0]=target
        feedback=JointDecodedMontageFeedback(observations=tuple(
            JointDecodedMontageSegmentObservation.from_candidate(target,
                render_variant_signature=signature*64,
                sampled_proxy_shot_classes=('wide','unknown','wide'))
            for signature in ('a','b')))
        shadow,metadata,audit=_decoded_montage_role_control_inputs(lattice,
            workspace.subject_metadata_by_candidate,feedback)
        self.assertIs(shadow[0][0].shot_type,ShotType.MEDIUM)
        self.assertFalse(shadow[0][0].is_group_wide)
        self.assertTrue(target.is_group_wide)
        self.assertIs(target.shot_type,ShotType.GROUP_WIDE)
        self.assertEqual(metadata[target.candidate_id].shot_scale_bases[1],'unknown_no_subject_geometry')
        self.assertEqual(audit['hard_no_good_count'],0)

    def test_observed_wide_role_never_invents_group_identity_fallback(self):
        workspace=_candidate_set(role_conditioned=True).optimizer_workspace
        lattice=workspace.candidates_by_segment
        target=lattice[0][0]
        feedback=JointDecodedMontageFeedback(observations=tuple(
            JointDecodedMontageSegmentObservation.from_candidate(target,
                render_variant_signature=signature*64,
                sampled_proxy_shot_classes=('wide','wide','wide'))
            for signature in ('a','b')))
        shadow,_,_=_decoded_montage_role_control_inputs(lattice,
            workspace.subject_metadata_by_candidate,feedback)
        self.assertIs(shadow[0][0].shot_type,ShotType.WIDE)
        self.assertFalse(shadow[0][0].is_group_wide)
