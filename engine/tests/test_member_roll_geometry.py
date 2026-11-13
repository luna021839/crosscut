from dataclasses import replace
import unittest
import numpy as np

from crosscut.joint_dissolve_registration import fit_joint_dissolve_registration, review_joint_dissolve_containment
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.render import RenderSegment, RenderTransition, EditorialTransformTrack, EditorialTransformKeyframe
from engine.tests.member_roll_fixture import windows, sample, SIZE


class MemberRollGeometryTests(unittest.TestCase):
    def test_measured_joint_rotation_is_authored(self):
        faces, poses = windows()
        tracks, audit = fit_joint_dissolve_registration(*faces, fps=30.,
            maximum_zooms=(1.5,1.5), pose_observations=poses)
        self.assertIsNotNone(tracks, audit)
        self.assertGreater(max(abs(p.roll_degrees) for t in tracks for p in t.keyframes), .5)

    def test_rotated_face_and_body_containment_is_measured(self):
        track = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.12,.5,.5,1.2),))
        geometry = (direct_member_source_frame_geometry((360,640),output_size=SIZE),)*2
        checked = review_joint_dissolve_containment(
            segments=(RenderSegment(0,2.,1.,0.,editorial_transform=track),RenderSegment(1,4.,1.,1.)),
            transitions=(RenderTransition(),), source_geometries=geometry, geometry_provider=sample, fps=30.)
        self.assertTrue(checked['passed'], checked['frames'][:1])

    def test_pose_rotation_requires_common_confident_temporally_consistent_joints(self):
        from crosscut.member_roll_geometry import estimate_pair_roll
        _, poses = windows()
        roll, audit = estimate_pair_roll(poses, fps=30., output_size=SIZE)
        self.assertAlmostEqual(roll, -2.4, places=6)
        cases = [None, (poses[0], (None,*poses[1][1:])),
            tuple(tuple(replace(p,points_xyc=tuple((x,y,.1) for x,y,c in p.points_xyc)) for p in seq) for seq in poses),
            (poses[0][:2],poses[1][:2])]
        for case in cases:
            with self.subTest(case=str(case)[:50]):
                self.assertIsNone(estimate_pair_roll(case,fps=30.,output_size=SIZE)[0])
        _, no_tilt = windows(degrees=0.)
        self.assertIsNone(estimate_pair_roll(no_tilt,fps=30.,output_size=SIZE)[0])
        _, excess = windows(degrees=5.)
        self.assertIsNone(estimate_pair_roll(excess,fps=30.,output_size=SIZE)[0])

    def test_box_visibility_is_polygon_area_and_respects_portrait_aspect(self):
        from crosscut.member_roll_geometry import project_points, box_visible_fraction
        state = EditorialTransformKeyframe(0.,1.12,.5,.5,1.2)
        projected = project_points(((.8,.5),), state, output_size=SIZE)[0]
        self.assertGreater(projected[1],.5)
        self.assertAlmostEqual(projected[1]-.5, .3*1.12*np.sin(np.deg2rad(1.2))*SIZE[0]/SIZE[1])
        self.assertAlmostEqual(box_visible_fraction((.4,.3,.1,.1),state,output_size=SIZE),1.)
        fraction = box_visible_fraction((.0,.0,.2,.3),state,output_size=SIZE)
        self.assertGreater(fraction,0.)
        self.assertLess(fraction,1.)

    def test_automatic_joint_registration_keeps_roll_through_nominal_and_overlap(self):
        from types import SimpleNamespace
        from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
        from crosscut.render import TransitionKind,DIRECT_MEMBER_POSE_DISSOLVE_REASON
        from engine.tests.member_roll_fixture import direct_inputs,focal
        selection,segments,_=direct_inputs()
        result,effects,audit=register_automatic_member_dissolves(render_segments=segments,
            transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON),),
            visual_selection=selection,eligible_request_source_indices=(2,0),common_master_start=0.,
            focal_provider=SimpleNamespace(sample=sample,selector_geometry=focal),fps=30.)
        self.assertEqual(audit['applied_count'],1,audit)
        self.assertTrue(audit['boundaries'][0]['registration']['roll']['applied'])
        self.assertGreater(abs(result[0].outgoing_overlap_transform.keyframes[0].roll_degrees),.5)
        self.assertGreater(abs(result[1].editorial_transform.keyframes[0].roll_degrees),.5)
        self.assertTrue(audit['boundaries'][0]['containment']['passed'])

    def test_failed_rotated_containment_keeps_the_existing_dissolve_solution(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
        from crosscut.joint_dissolve_registration import _fit_joint_dissolve_geometry
        from crosscut.render import TransitionKind,DIRECT_MEMBER_POSE_DISSOLVE_REASON
        from engine.tests.member_roll_fixture import direct_inputs,focal
        selection,segments,_=direct_inputs()
        args=dict(render_segments=segments,
            transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON),),
            visual_selection=selection,eligible_request_source_indices=(2,0),common_master_start=0.,
            focal_provider=SimpleNamespace(sample=sample,
                selector_geometry=lambda s,t:replace(focal(s,0.),source_seconds=t)),fps=30.)
        def without_roll(*a,**kw):
            kw.pop('output_size',None);kw.pop('enable_rotation',None)
            return _fit_joint_dissolve_geometry(*a,**kw)
        with patch('crosscut.automatic_joint_dissolve.fit_joint_dissolve_registration',side_effect=without_roll):
            baseline=register_automatic_member_dissolves(**args)
        self.assertEqual(baseline[2]['applied_count'],1,baseline[2])
        real_review=review_joint_dissolve_containment
        def fail_rotation(**kw):
            if any(s.editorial_transform and any(abs(p.roll_degrees)>1e-9 for p in s.editorial_transform.keyframes)
                   for s in kw['segments']):
                return dict(passed=False,frames=[])
            return real_review(**kw)
        with patch('crosscut.automatic_joint_dissolve.review_joint_dissolve_containment',side_effect=fail_rotation):
            actual=register_automatic_member_dissolves(**args)
        self.assertEqual(actual[:2],baseline[:2])
        self.assertEqual(actual[2]['applied_count'],1)
        self.assertTrue(actual[2]['boundaries'][0]['roll_refinement_rejected'])

    def test_articulation_reflection_and_stale_clock_cannot_author_roll(self):
        from crosscut.member_roll_geometry import estimate_pair_roll
        _,poses=windows()
        variants=[]
        variant=tuple(replace(p,points_xyc=tuple((x+(.08 if j in (4,5) else 0),y,c)
            for j,(x,y,c) in enumerate(p.points_xyc))) for p in poses[1])
        variants.append((poses[0],variant))
        variants.append((poses[0],tuple(replace(p,points_xyc=tuple((1-x,y,c) for x,y,c in p.points_xyc)) for p in poses[1])))
        variants.append((poses[0],tuple(replace(p,master_seconds=p.master_seconds+(.01 if i else 0)) for i,p in enumerate(poses[1]))))
        for variant in variants:
            with self.subTest(kind=variants.index(variant)):
                self.assertIsNone(estimate_pair_roll(variant,fps=30.,output_size=SIZE)[0])


if __name__ == '__main__':
    unittest.main()
