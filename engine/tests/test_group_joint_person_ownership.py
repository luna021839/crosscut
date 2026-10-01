from dataclasses import replace
import unittest

from crosscut.body_pose_2d import pose_input_geometry
from crosscut.group_joint_native_observer import NativeGroupHead
from crosscut.group_joint_person_ownership import (OwnedPersonObservation,bind_person_ownership)
from crosscut.group_joint_native_joint_flow import native_joint_pixel_flows,native_joint_assignment_cost
from engine.tests.test_group_joint_formation import person
from engine.tests import test_group_joint_native_joint_flow as flow_fixtures


def owned(p):
    _,_,affine,crop=pose_input_geometry((1000,800),p.box_xywh)
    return OwnedPersonObservation(p.observation_id,p.box_xywh,p.upper_body_xyc,p.confidence,
        pose_crop_xywh=crop,source_pixel_to_pose_input_affine=tuple(tuple(map(float,row)) for row in affine),
        pose_reason='observed')


class PersonOwnershipTests(unittest.TestCase):
    def test_normal_arms_outside_tight_box_are_not_discarded(self):
        p=owned(person(0,.5))
        self.assertLess(p.upper_body_xyc[4][0],p.box_xywh[0])
        rows,_=bind_person_ownership((p,),(),(1000,800))
        self.assertEqual(rows[0].body_ownership_state,'observed')
        self.assertEqual(dict(rows[0].part_states)['left_arm'],'observed')
        self.assertEqual(dict(rows[0].part_states)['head_extent'],'unknown')

    def test_duplicate_pose_keeps_both_hypotheses_as_ambiguous(self):
        p=owned(person(0,.5));q=replace(p,observation_id=1,box_xywh=(.46,.2,.1,.4))
        rows,_=bind_person_ownership((p,q),(),(1000,800))
        self.assertEqual(tuple(p.body_ownership_state for p in rows),('ambiguous','ambiguous'))
        self.assertEqual(rows[0].competing_person_observation_ids,(1,))
        self.assertEqual(len(rows),2)

    def test_real_overlap_with_separate_torso_geometry_is_not_nms_merged(self):
        a=owned(person(0,.48));b=owned(person(1,.52))
        rows,_=bind_person_ownership((a,b),(),(1000,800))
        self.assertEqual(tuple(p.body_ownership_state for p in rows),('observed','observed'))

    def test_other_person_pose_outside_roi_cannot_be_owned(self):
        p=replace(owned(person(0,.5)),box_xywh=(.1,.2,.1,.4))
        rows,_=bind_person_ownership((p,),(),(1000,800))
        self.assertEqual(rows[0].body_ownership_state,'unknown')

    def test_face_anchor_binds_only_one_body_and_never_certifies_whole_head(self):
        p=owned(person(0,.5));raw=((.5,.3,.9),)*17
        p=replace(p,raw_pose_xyc=raw)
        head=NativeGroupHead(7,(.47,.27,.06,.06),((.5,.3),)*5,.95)
        rows,owners=bind_person_ownership((p,),(head,),(1000,800))
        self.assertEqual(rows[0].unique_head_observation_id,7)
        self.assertEqual(owners[7],[0])
        self.assertEqual(dict(rows[0].part_states)['head_extent'],'unknown')
        wrong=replace(head,box_xywh=(.7,.27,.06,.06))
        rows,_=bind_person_ownership((p,),(wrong,),(1000,800))
        self.assertIsNone(rows[0].unique_head_observation_id)

    def test_disjoint_roi_same_skeleton_is_not_a_pixel_identity_link(self):
        fixture=flow_fixtures.NativeJointFlowTests()
        previous,current,subject,moved=fixture.fixture()
        flow=native_joint_pixel_flows(previous,current,(subject,))[0]
        self.assertIsNotNone(flow)
        wrong=replace(moved,box_xywh=(.1,.2,.1,.4))
        self.assertIsNone(native_joint_assignment_cost(flow,wrong,(1000,800)))

    def test_ambiguous_owned_skeleton_does_not_track_background_texture(self):
        fixture=flow_fixtures.NativeJointFlowTests()
        previous,current,subject,_=fixture.fixture()
        ambiguous=replace(owned(subject),body_ownership_state='ambiguous')
        self.assertEqual(native_joint_pixel_flows(previous,current,(ambiguous,)),(None,))
