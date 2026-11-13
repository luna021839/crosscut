from dataclasses import replace
import unittest
import numpy as np

from crosscut.joint_dissolve_pose import (
    OverlapPoseObservation, source_focal_overlap_pose,
    paired_overlap_pose_arrays, pose_registration_metrics,
)
from crosscut.joint_dissolve_registration import OverlapFaceObservation, fit_joint_dissolve_registration
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.pose import FormationPersonDescriptor
from crosscut.direct_member_transform import direct_member_source_frame_geometry


def window():
    left=tuple(OverlapFaceObservation(0,10+i/30,20+i/30,(.42,.2,.16,.16)) for i in range(3))
    right=tuple(OverlapFaceObservation(1,10+i/30,30+i/30,(.32+i*.01,.25,.12,.12)) for i in range(3))
    poses=[]
    for side,seq in enumerate((left,right)):
        points=[]
        for i,obs in enumerate(seq):
            base=np.array([[.35+.04*j,.4+.04*j] for j in range(8)])
            if side:base=(base-[.5,.28])*.75+[.38+i*.01,.31]
            points.append(OverlapPoseObservation(obs.source_index,obs.master_seconds,obs.source_seconds,
                tuple((float(x),float(y),.9) for x,y in base)))
        poses.append(tuple(points))
    return (left,right),tuple(poses)


class JointDissolvePoseTests(unittest.TestCase):
    def test_joint_pose_fit_keeps_face_limits_and_measured_pose_nonregression(self):
        faces,poses=window()
        tracks,audit=fit_joint_dissolve_registration(*faces,fps=30.,maximum_zooms=(1.5,1.5),pose_observations=poses)
        self.assertIsNotNone(tracks,audit)
        report=audit['focal_pose_registration']
        self.assertLess(report['after']['joint_rms_distance'],report['before']['joint_rms_distance'])
        self.assertEqual(report['common_joint_indices'],list(range(8)))
        self.assertLessEqual(audit['maximum_position_mismatch'],.080000001)
        self.assertLessEqual(audit['maximum_scale_mismatch'],.060000001)
        self.assertFalse(audit['time_shift_applied'])
        self.assertFalse(report['quality_approval'])

    def test_face_only_repair_cannot_displace_already_matching_skeletons(self):
        faces,poses=window()
        poses=(poses[0],tuple(replace(p,points_xyc=poses[0][i].points_xyc) for i,p in enumerate(poses[1])))
        face_only,_=fit_joint_dissolve_registration(*faces,fps=30.,maximum_zooms=(1.5,1.5))
        self.assertIsNotNone(face_only)
        fitted,audit=fit_joint_dissolve_registration(*faces,fps=30.,maximum_zooms=(1.5,1.5),pose_observations=poses)
        self.assertIsNone(fitted,audit)
        self.assertFalse(audit['infeasibility_proven'])

    def test_pose_and_face_must_bind_same_source_and_exact_clocks(self):
        faces,poses=window()
        for field,value in (('source_index',7),('source_seconds',20.01),('master_seconds',10.01)):
            changed=((replace(poses[0][0],**{field:value}),*poses[0][1:]),poses[1])
            with self.assertRaises(ValueError):paired_overlap_pose_arrays(changed,faces)

    def test_common_joint_ids_do_not_change_across_frames_or_sides(self):
        faces,poses=window()
        def hidden(pose,indices):
            return replace(pose,points_xyc=tuple((x,y,0. if i in indices else c)
                for i,(x,y,c) in enumerate(pose.points_xyc)))
        changed=((hidden(poses[0][0],{0,1}),*poses[0][1:]),
                 (poses[1][0],hidden(poses[1][1],{2,3}),poses[1][2]))
        points,audit=paired_overlap_pose_arrays(changed,faces)
        self.assertEqual(points.shape,(2,3,4,2))
        self.assertEqual(audit['common_joint_indices'],[4,5,6,7])
        changed=(changed[0],(*changed[1][:2],hidden(changed[1][2],{4})))
        points,audit=paired_overlap_pose_arrays(changed,faces)
        self.assertIsNone(points)
        self.assertEqual(audit['status'],'insufficient_common_joints')

    def test_missing_pose_is_reported_without_interpolation_or_fake_pose_claim(self):
        faces,poses=window()
        changed=((None,*poses[0][1:]),poses[1])
        points,audit=paired_overlap_pose_arrays(changed,faces)
        self.assertIsNone(points)
        self.assertEqual(audit['status'],'exact_pose_missing')
        baseline,_=fit_joint_dissolve_registration(*faces,fps=30.,maximum_zooms=(1.5,1.5))
        tracks,audit=fit_joint_dissolve_registration(*faces,fps=30.,maximum_zooms=(1.5,1.5),pose_observations=changed)
        self.assertEqual(tracks,baseline)
        self.assertEqual(audit['focal_pose_registration']['status'],'exact_pose_missing')

    def test_invalid_confidence_or_numeric_data_is_rejected(self):
        _,poses=window()
        for point in ((.4,.5,float('nan')),(.4,.5,1.1),(.4,True,.9),(.4,.5)):
            with self.assertRaises(ValueError):
                replace(poses[0][0],points_xyc=(point,*poses[0][0].points_xyc[1:]))

    def test_source_mapping_uses_screen_not_torso_normalized_points(self):
        body=FormationPersonDescriptor((.3,.08,.4,.8),(.5,.48),.95,
            tuple((-.2+.06*j,-.3+.05*j,.9) for j in range(8)))
        value=FocalSelectorGeometry(1,3.,(.45,.1,.1,.1),person=body,
            pose_backend='rtmpose_body7',pose_screen_xyc=tuple((.4+.03*j,.3+.05*j,.9) for j in range(8)))
        geometry=direct_member_source_frame_geometry((1920,1080),output_size=(1080,1920))
        args=dict(source_index=5,expected_geometry_source_index=1,master_seconds=2.,source_seconds=3.,frame_geometry=geometry)
        out=source_focal_overlap_pose(value,**args)
        self.assertEqual(out.source_index,5)
        self.assertAlmostEqual(out.points_xyc[0][0],(.4-geometry.fill_crop_xywh[0])/geometry.fill_crop_xywh[2])
        with self.assertRaises(ValueError):source_focal_overlap_pose(value,**{**args,'expected_geometry_source_index':5})
        self.assertIsNone(source_focal_overlap_pose(replace(value,pose_backend='formation_pose'),**args))


if __name__=='__main__':unittest.main()
