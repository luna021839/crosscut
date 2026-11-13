"""Turned-face camera registration uses observed common joints, not fake faces."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
from crosscut.joint_dissolve_pose import OverlapPoseObservation
from crosscut.joint_dissolve_registration import (
    joint_pose_anchor_observations, fit_joint_dissolve_registration,
    OverlapFaceObservation, OverlapPoseAnchorObservation,
)
from crosscut.focal_body_track import FocalBodyGeometryCertificate
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.pose import FormationPersonDescriptor
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
import engine.tests.test_automatic_joint_dissolve as automatic_tests


def points(side):
    base=((.38,.35),(.62,.35),(.33,.46),(.67,.46),(.30,.55),(.70,.55),(.43,.65),(.57,.65))
    return tuple((x if side==0 else .5+(x-.5)*.85,
                  y if side==0 else .5+(y-.5)*.85,.9) for x,y in base)


def window():
    return tuple(tuple(OverlapPoseObservation(side,3+n/30,10+side*10+n/30,points(side))
                       for n in range(7)) for side in range(2))


class JointPoseAnchorTests(unittest.TestCase):
    def test_measured_pose_registration_never_claims_face_detection(self):
        poses=window()
        anchors,evidence=joint_pose_anchor_observations(poses)
        self.assertFalse(evidence['face_detection_claimed'])
        self.assertTrue(all(isinstance(p,OverlapPoseAnchorObservation) for seq in anchors for p in seq))
        tracks,audit=fit_joint_dissolve_registration(*anchors,fps=30.,maximum_zooms=(1.5,1.5),pose_observations=poses)
        self.assertIsNotNone(tracks,audit)
        self.assertEqual(audit['raw_observation_count'],0)
        self.assertEqual(audit['pixel_tracked_observation_count'],0)
        self.assertEqual(audit['pose_anchor_observation_count'],14)
        self.assertEqual(audit['registration_basis'],'observed_focal_common_joint_extent')
        self.assertLess(audit['focal_pose_registration']['after']['joint_rms_distance'],
                        audit['focal_pose_registration']['before']['joint_rms_distance'])

    def test_one_missing_pose_cannot_be_interpolated(self):
        poses=window()
        anchors,evidence=joint_pose_anchor_observations(((None,*poses[0][1:]),poses[1]))
        self.assertIsNone(anchors)
        self.assertEqual(evidence['status'],'exact_pose_missing')

    def test_same_joint_ids_required_across_both_windows(self):
        poses=window()
        def hide(p,ids):
            return replace(p,points_xyc=tuple((x,y,0. if i in ids else c)
                for i,(x,y,c) in enumerate(p.points_xyc)))
        changed=((hide(poses[0][0],(0,1)),*poses[0][1:]),
                 (hide(poses[1][0],(2,3)),*poses[1][1:]))
        anchors,_=joint_pose_anchor_observations(changed)
        self.assertTrue(all(p.common_joint_indices==(4,5,6,7) for seq in anchors for p in seq))
        changed=(changed[0],(hide(changed[1][0],(2,3,4)),*changed[1][1:]))
        self.assertIsNone(joint_pose_anchor_observations(changed)[0])

    def test_mixed_basis_and_modified_anchor_are_not_valid_evidence(self):
        poses=window();anchors,_=joint_pose_anchor_observations(poses)
        face=OverlapFaceObservation(0,3.,10.,anchors[0][0].box)
        for left in ((face,*anchors[0][1:]),
                     (replace(anchors[0][0],box=(.2,.2,.2,.2)),*anchors[0][1:]),
                     (replace(anchors[0][0],source_seconds=10.1),*anchors[0][1:])):
            with self.assertRaises(ValueError):
                fit_joint_dissolve_registration(left,anchors[1],fps=30.,maximum_zooms=(1.5,1.5),pose_observations=poses)

    def inputs(self):
        fixture=automatic_tests.AutomaticJointDissolveTests()
        args=fixture.inputs()
        def geometry(source,local):
            box=(.25,.15,.50,.70)
            person=FormationPersonDescriptor(box,(.5,.5),.95,points(source))
            track=FocalBodyGeometryCertificate(source,local,box,(local-.2,local+.2),.2)
            return FocalSelectorGeometry(source,local,None,person=person,body_track=track,
                pose_backend='rtmpose_body7',pose_screen_xyc=points(source))
        args['focal_provider']=SimpleNamespace(selector_geometry=geometry,
            sample=lambda s,t:SourceFirstFocalGeometryProvider._body_sample(geometry(s,t)))
        return args,fixture

    def test_automatic_stage_materializes_and_densely_reviews_body_only_targets(self):
        args,fixture=self.inputs()
        segments,effects,audit=register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'],1,audit)
        row=audit['boundaries'][0]
        self.assertTrue(row['containment']['passed'])
        self.assertTrue(all(r['face_visible_fraction'] is None for r in row['containment']['frames']))
        fixture.plan(args,segments,effects)
        for before,after in zip(args['render_segments'],segments):
            self.assertEqual(replace(after,editorial_transform=None,outgoing_overlap_transform=None),before)

    def test_missing_exact_pose_keeps_effect_unregistered(self):
        args,_=self.inputs();resolve=args['focal_provider'].selector_geometry
        args['focal_provider'].selector_geometry=lambda s,t:replace(resolve(s,t),pose_screen_xyc=()) if s==1 else resolve(s,t)
        segments,effects,audit=register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'],0)
        self.assertEqual(segments,args['render_segments'])
        self.assertEqual(effects,args['transitions'])

    def test_native_cut_on_second_source_prevents_measurement_or_solve(self):
        args,_=self.inputs()
        args['visual_selection'].span_validity[0].sources[1].native_scene_cut_source_seconds=(20.1,)
        with patch('crosscut.automatic_joint_dissolve.fit_joint_dissolve_registration') as fit:
            _,_,audit=register_automatic_member_dissolves(**args)
        fit.assert_not_called()
        self.assertEqual(audit['source_measurement_queries'],0)
        self.assertEqual(audit['boundaries'][0]['reason'],'native_cut_in_overlap')

    def test_failed_containment_cannot_authorize_pose_camera(self):
        args,_=self.inputs()
        with patch('crosscut.automatic_joint_dissolve.review_joint_dissolve_containment',return_value={'passed':False}):
            segments,effects,audit=register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'],0)
        self.assertEqual(segments,args['render_segments'])
        self.assertEqual(effects,args['transitions'])


if __name__=='__main__':unittest.main()
