from dataclasses import replace
import unittest

from crosscut.boundary_subject_alignment import _fixed_crop_at, _project_box, _TransformState
from crosscut.render import (EditorialTransformKeyframe, EditorialTransformTrack, NormalizedCrop,
    PanCropKeyframe, PanCropTrack, RenderSegment)
from crosscut.shot_context_camera import retain_opposite_cut_camera


class OppositeCutCameraRetentionTests(unittest.TestCase):
    def fixture(self):
        count=43;fps=30.
        pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.02,.10,.8,.8)),
            PanCropKeyframe(count/fps,NormalizedCrop(.10,.05,.8,.8))))
        old=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.12),
            EditorialTransformKeyframe((count-1)/fps,1.14,.51,.50)))
        segment=RenderSegment(2,12.,count/fps,5.,pan_crop=pan,editorial_transform=old)
        native=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.4,.60,.55),
            EditorialTransformKeyframe((count-1)/fps,1.4,.60,.55)))
        return segment,native,dict(head_envelopes=((.45,.35,.12,.17),)*count,fps=fps)

    def test_retains_all_four_nonneutral_pan_and_editorial_states_on_either_side(self):
        segment,native,args=self.fixture();count=len(args['head_envelopes'])
        for outgoing in (False,True):
            with self.subTest(outgoing=outgoing):
                result=retain_opposite_cut_camera(segment,native,**args,outgoing=outgoing)
                for n in range(count):
                    time=n/args['fps'];protected=n<4 if outgoing else n>=count-4
                    if protected:
                        self.assertEqual(result.editorial_transform.evaluate(time),segment.editorial_transform.evaluate(time))
                        for a,b in zip(_fixed_crop_at(result.pan_crop,time),_fixed_crop_at(segment.pan_crop,time)):
                            self.assertAlmostEqual(a,b,places=12)
                for name in ('source_index','source_start','source_duration','master_start','duration',
                        'look_adjustment','boundary_look_track'):
                    self.assertEqual(getattr(result,name),getattr(segment,name))

    def test_reparameterized_candidate_keeps_the_exact_native_viewport_before_return(self):
        segment,native,args=self.fixture()
        result=retain_opposite_cut_camera(segment,native,**args)
        # The old pan does not contain this desired viewport. Its position
        # changes, while its width and the native camera projection do not.
        self.assertNotEqual(_fixed_crop_at(result.pan_crop,0.),_fixed_crop_at(segment.pan_crop,0.))
        self.assertEqual(result.pan_crop.fixed_crop_size,segment.pan_crop.fixed_crop_size)
        for n in range(26):
            time=n/args['fps'];x,y,w,h=_fixed_crop_at(result.pan_crop,time)
            key=result.editorial_transform.evaluate(time);wanted=native.evaluate(time)
            self.assertAlmostEqual(key.zoom/w,wanted.zoom)
            self.assertAlmostEqual(x+w*key.center_x,wanted.center_x)
            self.assertAlmostEqual(y+h*key.center_y,wanted.center_y)
            head=args['head_envelopes'][n]
            projected=_project_box(((head[0]-x)/w,(head[1]-y)/h,head[2]/w,head[3]/h),
                _TransformState(key.zoom,key.center_x,key.center_y,0.))[0]
            expected=_project_box(head,_TransformState(wanted.zoom,wanted.center_x,wanted.center_y,0.))[0]
            for a,b in zip(projected,expected):self.assertAlmostEqual(a,b)

    def test_no_base_pan_remains_no_base_pan_and_return_is_not_a_one_frame_reset(self):
        segment,native,args=self.fixture();segment=replace(segment,pan_crop=None)
        result=retain_opposite_cut_camera(segment,native,**args)
        self.assertIsNone(result.pan_crop)
        start=len(args['head_envelopes'])-4-14
        self.assertEqual(result.editorial_transform.evaluate(start/30.),native.evaluate(start/30.))
        steps=[abs(result.editorial_transform.evaluate(n/30.).zoom-
            result.editorial_transform.evaluate((n-1)/30.).zoom) for n in range(start+1,40)]
        self.assertGreater(max(steps),0.)
        self.assertLess(steps[-1],max(steps)/10)
        self.assertLess(steps[0],max(steps)/10)

    def test_short_or_dense_expression_domains_and_invalid_policy_fail(self):
        segment,native,args=self.fixture()
        for extra in ({'preserved_boundary_frames':True},{'preserved_boundary_frames':3},
                {'preserved_boundary_frames':15},{'ramp_frames':31},{'outgoing':1},
                {'fps':float('nan')},{'head_envelopes':args['head_envelopes'][:21]},
                {'head_envelopes':args['head_envelopes']*2}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                retain_opposite_cut_camera(segment,native,**dict(args,**extra))

    def test_unknown_heads_and_pixel_budget_are_not_relaxed_for_the_better_endpoint(self):
        segment,native,args=self.fixture()
        for head in (None,(-.01,.2,.1,.1),(.01,.01,.1,.1)):
            heads=list(args['head_envelopes']);heads[10]=head
            with self.subTest(head=head),self.assertRaises(ValueError):
                retain_opposite_cut_camera(segment,native,**dict(args,head_envelopes=heads))
        wider=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))
        with self.assertRaisesRegex(ValueError,'outside the fixed-size'):
            retain_opposite_cut_camera(segment,wider,**args)

    def test_rotation_and_base_aspect_are_not_silently_flattened(self):
        segment,native,args=self.fixture()
        with self.assertRaisesRegex(ValueError,'unrolled'):
            retain_opposite_cut_camera(replace(segment,editorial_transform=None),native,**args)
        rolled=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.4,.5,.5,1.),))
        for old,new in ((segment,rolled),(replace(segment,editorial_transform=rolled),native)):
            with self.assertRaisesRegex(ValueError,'unrolled'):
                retain_opposite_cut_camera(old,new,**args)
        pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.1,.1,.7,.8)),))
        with self.assertRaisesRegex(ValueError,'aspect'):
            retain_opposite_cut_camera(replace(segment,pan_crop=pan),native,**args)


if __name__=='__main__':unittest.main()
