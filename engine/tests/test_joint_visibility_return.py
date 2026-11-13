from dataclasses import replace
import unittest
from unittest.mock import patch

from crosscut.joint_dissolve_registration import shorten_joint_incoming_return,review_joint_dissolve_containment
from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe,RenderTransition,TransitionKind
from engine.tests import test_automatic_joint_dissolve as automatic_fixture


def camera(zoom=1.08,center=.53):
    points=tuple(EditorialTransformKeyframe(n/30.,zoom,center,.5) for n in range(8))
    return EditorialTransformTrack((*points,EditorialTransformKeyframe(7/30.+.75,1.),
        EditorialTransformKeyframe(3.,1.)),reason='joint_dissolve_incoming_ramp_pending_containment_review')


def failure(seconds=.4,segment=1,scope='nominal'):
    return dict(passed=False,frames=[dict(passed=False,segment_index=segment,scope=scope,
        output_local_seconds=seconds,reason='focal_geometry_missing')])


class JointVisibilityReturnTests(unittest.TestCase):
    def adjust(self,track=None,review=None):
        return shorten_joint_incoming_return(track or camera(),RenderTransition(TransitionKind.FACE_DISSOLVE,7),
            review or failure(),fps=30.)

    def test_return_preserves_entire_blend_and_first_pure_frame(self):
        before=camera();after,audit=self.adjust(before)
        self.assertTrue(audit['adjusted'])
        self.assertEqual(audit['return_frame_count'],4)
        for n in range(8):self.assertEqual(before.evaluate(n/30.),after.evaluate(n/30.))
        self.assertEqual(after.evaluate(.4).zoom,1.)
        self.assertEqual(after.evaluate(.4).center_x,.5)
        self.assertEqual(after.final_keyframe,before.final_keyframe)
        self.assertFalse(audit['quality_approval'])

    def test_unknown_overlap_or_outgoing_camera_never_waives_review(self):
        for review in (failure(.2),failure(segment=0),failure(scope='overlap'),failure(.3),
                dict(passed=False,frames=[])):
            self.assertIsNone(self.adjust(review=review)[0])

    def test_large_fast_return_is_not_silently_accepted(self):
        self.assertIsNone(self.adjust(camera(zoom=1.5))[0])
        self.assertIsNone(self.adjust(camera(zoom=1.15,center=.56))[0])

    def test_no_overwrite_of_unowned_camera(self):
        self.assertIsNone(self.adjust(replace(camera(),reason='user_camera'))[0])
        nonidentity=replace(camera(),keyframes=(*camera().keyframes[:-1],
            EditorialTransformKeyframe(3.,1.01)))
        self.assertIsNone(self.adjust(nonidentity)[0])

    def test_automatic_stage_rechecks_once_without_repeating_registration(self):
        fixture=automatic_fixture.AutomaticJointDissolveTests();args=fixture.inputs()
        original=args['focal_provider'].sample
        # One exact nominal frame after the blend, on the incoming source clock.
        missing_local=20.+.5*.99
        args['focal_provider'].sample=lambda source,local: (
            None if source==1 and abs(local-missing_local)<1e-8 else original(source,local))
        reviews=[]
        def review(**kwargs):
            kwargs['pixel_containment_provider']=None
            result=review_joint_dissolve_containment(**kwargs)
            reviews.append(result)
            return result
        from crosscut.joint_dissolve_registration import fit_joint_dissolve_registration
        with patch('crosscut.automatic_joint_dissolve.review_joint_dissolve_containment',side_effect=review),\
             patch('crosscut.automatic_joint_dissolve.fit_joint_dissolve_registration',wraps=fit_joint_dissolve_registration) as fits:
            _,_,audit=register_automatic_member_dissolves(**args)
        self.assertEqual(fits.call_count,1)
        self.assertEqual(len(reviews),2)
        self.assertFalse(reviews[0]['passed'])
        self.assertTrue(reviews[1]['passed'])
        self.assertEqual(audit['applied_count'],1)
        self.assertTrue(audit['boundaries'][0]['visibility_timed_return']['adjusted'])


if __name__=='__main__':unittest.main()
