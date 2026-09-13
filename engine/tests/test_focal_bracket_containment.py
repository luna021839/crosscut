from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.focal_bracket_containment import measure_bracketed_body_containment
from crosscut.direct_member_editorial_motion import DirectMemberFaceSample
from crosscut.focal_selector_geometry import FOCAL_SELECTOR_BODY_BASIS
from crosscut.direct_member_target_face import FOCAL_FACE_BASIS
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.joint_dissolve_registration import review_joint_dissolve_containment
from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
from crosscut.render import RenderSegment, RenderTransition, TransitionKind, EditorialTransformTrack, EditorialTransformKeyframe
from engine.tests import test_automatic_joint_dissolve as automatic_fixture


def sample(box=(.3, .16, .4, .6)):
    return DirectMemberFaceSample(box, 0, 'none', box, .9, 'source_local_focal_body_track', FOCAL_SELECTOR_BODY_BASIS)


def support(source=0, local=10., boxes=((.3, .16, .4, .6),), anchors=None, cuts=()):
    return measure_bracketed_body_containment(source_index=source, source_seconds=local,
        observed_query_boxes=boxes,
        anchors=anchors or ((source, local-1/30, sample()), (source, local+1/30, sample())),
        native_scene_cut_seconds=cuts)


class FocalBracketContainmentTests(unittest.TestCase):
    def test_selects_only_an_observed_query_box(self):
        boxes=((.05,.1,.2,.7),(.3,.16,.4,.6))
        result=support(boxes=boxes)
        self.assertEqual(result.bbox_xywh, boxes[1])
        self.assertEqual(result.selected_index, 1)

    def test_competing_nearly_identical_boxes_do_not_make_identity_certain(self):
        self.assertIsNone(support(boxes=((.3,.16,.4,.6),(.31,.16,.4,.6))))

    def test_disagreeing_anchors_cannot_pick_the_more_convenient_one(self):
        anchors=((0,9.96,sample()),(0,10.04,sample((.05,.1,.2,.7))))
        self.assertIsNone(support(anchors=anchors))

    def test_missing_detection_native_cut_or_wrong_source_never_recovers(self):
        self.assertIsNone(support(boxes=()))
        self.assertIsNone(support(cuts=(10.01,)))
        self.assertIsNone(support(anchors=((1,9.96,sample()),(1,10.04,sample()))))
        self.assertIsNone(support(anchors=((0,9.8,sample()),(0,10.2,sample()))))

    def test_anonymous_or_face_only_anchors_do_not_authorize_body(self):
        for invalid in (replace(sample(), alignment_geometry_basis_override=None),
                        DirectMemberFaceSample((.45,.2,.1,.1), 1, 'yunet', None, 0., 'none', FOCAL_FACE_BASIS)):
            self.assertIsNone(support(anchors=((0,9.96,invalid),(0,10.04,invalid))))

    def test_containment_recovery_never_authorizes_overlap_pose(self):
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(1.,1.1)))
        segment=RenderSegment(0,10.,1.,0.,editorial_transform=camera,outgoing_overlap_transform=camera)
        calls=[]
        def measured(source,local):
            calls.append(local)
            return support(source,local)
        review=review_joint_dissolve_containment(segments=(segment,),transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,3),),
            source_geometries={0:direct_member_source_frame_geometry((1080,1920),output_size=(1080,1920))},
            geometry_provider=lambda *a:None,bracketed_body_containment_provider=measured,fps=30.)
        self.assertFalse(review['passed'])
        self.assertTrue(all(row['passed'] for row in review['frames'] if row['scope']=='nominal'))
        self.assertTrue(all(not row['passed'] for row in review['frames'] if row['scope']=='overlap'))
        self.assertTrue(all(t<11. for t in calls))

    def test_automatic_stage_uses_exact_compact_cached_body_and_strict_flanks(self):
        args=automatic_fixture.AutomaticJointDissolveTests().inputs();provider=args['focal_provider'];original=provider.sample
        missing=20.+.5*.99
        provider.sample=lambda s,t: None if s==1 and abs(t-missing)<1e-8 else original(s,t)
        provider.faces=SimpleNamespace(_samplers=(None,None),_identity_key=lambda s,sampler,t:(s,t))
        provider.bodies=SimpleNamespace(_frame_cache={(1,missing):SimpleNamespace(source_index=1,
            observations=(SimpleNamespace(person=SimpleNamespace(bbox_xywh=(.3,.16,.4,.6))),))})
        _,_,audit=register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'],1)
        rows=[r for r in audit['boundaries'][0]['containment']['frames'] if r['reason']=='bracketed_observed_body_containment']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['source_local_seconds'],missing)
        self.assertFalse(rows[0]['face_detection_claimed'])
        self.assertFalse(rows[0]['cross_source_pose_authority'])

    def test_source_or_clock_mismatched_support_is_not_reused(self):
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(1.,1.1)))
        for provider in (lambda s,t:support(s+1,t), lambda s,t:support(s,t+.01)):
            review=review_joint_dissolve_containment(segments=(RenderSegment(0,10.,1.,0.,editorial_transform=camera),),
                transitions=(),source_geometries={0:direct_member_source_frame_geometry((1080,1920),output_size=(1080,1920))},
                geometry_provider=lambda *a:None,bracketed_body_containment_provider=provider,fps=30.)
            self.assertFalse(review['passed'])
            self.assertFalse(any(r['reason']=='bracketed_observed_body_containment' for r in review['frames']))

    def test_visible_body_does_not_excuse_cropped_upper_body(self):
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2,.5,.58),EditorialTransformKeyframe(1.,1.2,.5,.58)))
        review=review_joint_dissolve_containment(segments=(RenderSegment(0,10.,1.,0.,editorial_transform=camera),),
            transitions=(),source_geometries={0:direct_member_source_frame_geometry((1080,1920),output_size=(1080,1920))},
            geometry_provider=lambda *a:None,bracketed_body_containment_provider=lambda s,t:support(s,t),fps=30.)
        rows=[r for r in review['frames'] if r['reason']=='bracketed_observed_body_containment']
        self.assertTrue(rows)
        self.assertTrue(all(r['body_visible_fraction']>.72 for r in rows))
        self.assertTrue(all(r['upper_body_visible_fraction']<.98 and not r['passed'] for r in rows))
        self.assertFalse(review['passed'])

    def test_rotated_camera_does_not_use_unrotated_bracket_bounds(self):
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1,roll_degrees=.5),EditorialTransformKeyframe(1.,1.1,roll_degrees=.5)))
        calls=[]
        def provider(s,t):calls.append((s,t));return support(s,t)
        review=review_joint_dissolve_containment(segments=(RenderSegment(0,10.,1.,0.,editorial_transform=camera),),
            transitions=(),source_geometries={0:direct_member_source_frame_geometry((1080,1920),output_size=(1080,1920))},
            geometry_provider=lambda *a:None,bracketed_body_containment_provider=provider,fps=30.)
        self.assertFalse(review['passed'])
        self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
