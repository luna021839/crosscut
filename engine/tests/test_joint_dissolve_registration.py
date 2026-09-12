from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest
import shutil
import subprocess
import tempfile
from crosscut.joint_dissolve_registration import (OverlapFaceObservation,fit_joint_dissolve_registration,
    materialize_joint_dissolve_registration,review_joint_dissolve_containment)
from crosscut.render import (RenderSegment,RenderPlan,RenderTransition,TransitionKind,
    EditorialTransformKeyframe as Keyframe,EditorialTransformTrack as Track,
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY,
    _dissolve_handle_video_chain)
from crosscut.ive import AspectRatio


def observations():
    left=tuple(OverlapFaceObservation(0,1+i/30,1+i/30,(.42,.2,.16,.16)) for i in range(7))
    right=tuple(OverlapFaceObservation(1,1+i/30,2+i/30,(.32+i*.01,.25,.12,.12)) for i in range(7))
    return left,right


class JointRegistrationTests(unittest.TestCase):
    def test_measured_moving_registration_fits_without_time_shift(self):
        tracks,audit=fit_joint_dissolve_registration(*observations(),fps=30.,maximum_zooms=(1.5,1.5))
        self.assertIsNotNone(tracks,audit)
        self.assertLessEqual(audit['maximum_position_mismatch'],.08000001)
        self.assertLessEqual(audit['maximum_scale_mismatch'],.06000001)
        self.assertNotEqual(tracks[1].keyframes[0].center_x,tracks[1].keyframes[-1].center_x)
        self.assertFalse(audit['quality_approval'])
        self.assertFalse(audit['time_shift_applied'])

    def test_unbound_timestamps_and_sources_are_not_interpolated(self):
        left,right=observations()
        for changed in ((replace(right[0],master_seconds=1.1),*right[1:]),
                        (replace(right[0],source_index=0),*right[1:])):
            with self.assertRaises(ValueError):
                fit_joint_dissolve_registration(left,changed,fps=30.,maximum_zooms=(1.5,1.5))
        with self.assertRaises(TypeError):
            fit_joint_dissolve_registration(left,(None,*right[1:]),fps=30.,maximum_zooms=(1.5,1.5))

    def test_unrecoverable_viewport_does_not_authorize_camera(self):
        left,right=observations()
        tracks,audit=fit_joint_dissolve_registration(left,right,fps=30.,maximum_zooms=(1.,1.))
        self.assertIsNone(tracks)
        self.assertFalse(audit['quality_approval'])


class OverlapCameraRenderTests(unittest.TestCase):
    def plan(self):
        nominal=Track((Keyframe(0.,1.),Keyframe(2.,1.1)),reason='nominal')
        outgoing=Track((Keyframe(0.,1.1),Keyframe(.2,1.2)),reason='measured')
        incoming=Track((Keyframe(0.,1.1),Keyframe(.2,1.2),Keyframe(2.,1.)),reason='measured')
        return RenderPlan(sources=(Path('/tmp/a.mp4'),Path('/tmp/b.mp4')),
            segments=(RenderSegment(0,0.,2.,0.,editorial_transform=nominal,outgoing_overlap_transform=outgoing),
                      RenderSegment(1,2.,2.,2.,editorial_transform=incoming)),
            transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                transform_overlap_authority=MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY),),
            reference_source=0,aspect_ratio=AspectRatio.PORTRAIT,output=Path('/tmp/out.mp4'),width=180,height=320,fps=30.,
            member_pose_dissolve_requested=True,member_pose_dissolve_effective=True,member_pose_dissolve_count=1)

    def test_renderer_consumes_overlap_camera_not_fixed_endpoint(self):
        plan=self.plan()
        with patch('crosscut.render._video_chain',return_value='verified') as chain:
            _dissolve_handle_video_chain(0,plan.segments[0],width=180,height=320,fps=30.,extension_frames=7)
        self.assertIs(chain.call_args.kwargs['editorial_transform'],plan.segments[0].outgoing_overlap_transform)
        self.assertEqual(chain.call_args.kwargs['source_start'],2.)
        self.assertIn('outgoing_overlap_transform',plan.to_json()['segments'][0])

    def test_unbound_discontinuous_or_wrong_length_handle_is_rejected(self):
        plan=self.plan();left,right=plan.segments
        for changed in (replace(left,outgoing_overlap_transform=Track((Keyframe(0.,1.3),Keyframe(.2,1.3)),reason='reset')),
                        replace(left,outgoing_overlap_transform=Track((Keyframe(0.,1.1),Keyframe(1.,1.2)),reason='long')),
                        replace(left,editorial_transform=None)):
            with self.assertRaises(ValueError):replace(plan,segments=(changed,right))
        with self.assertRaises(ValueError):replace(plan,segments=(replace(left,outgoing_overlap_transform=None),right))
        with self.assertRaises(ValueError):replace(plan,segments=(left,replace(right,outgoing_overlap_transform=left.outgoing_overlap_transform)))

    def test_source_measured_authority_does_not_claim_pixel_review(self):
        plan=self.plan()
        self.assertIn('pending_pixel_review',plan.transitions[0].transform_overlap_authority)
        with self.assertRaises(ValueError):replace(plan,transitions=(replace(plan.transitions[0],transform_overlap_authority=None),))

    @unittest.skipUnless(shutil.which('ffmpeg'),'requires actual renderer')
    def test_outgoing_handle_camera_changes_actual_static_source_pixels(self):
        ffmpeg=shutil.which('ffmpeg')
        with tempfile.TemporaryDirectory(prefix='joint-overlap-pixels-') as temp:
            source=Path(temp)/'static.mp4'
            subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',
                'color=black:s=180x320:r=30:d=4,drawbox=x=60:y=80:w=20:h=20:color=white:t=fill',
                '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',str(source)],check=True,capture_output=True)
            plan=self.plan()
            graph=_dissolve_handle_video_chain(0,plan.segments[0],width=180,height=320,fps=30.,extension_frames=7)
            pixels=subprocess.run([ffmpeg,'-v','error','-i',str(source),'-filter_complex',graph,
                '-map','[vh0]','-frames:v','7','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],check=True,capture_output=True).stdout
            stride=180*320*3
            self.assertEqual(len(pixels),stride*7)
            first=pixels[:stride];last=pixels[-stride:]
            self.assertNotEqual(first,last)
            self.assertGreater(sum(v>220 for v in last),sum(v>220 for v in first))

    def test_materialization_binds_actual_source_clocks_and_containment(self):
        from engine.tests.test_member_dissolve_camera import face
        from crosscut.direct_member_transform import direct_member_source_frame_geometry
        left,right=observations()
        left=tuple(replace(s,master_seconds=s.master_seconds+1,source_seconds=s.source_seconds+1) for s in left)
        right=tuple(replace(s,master_seconds=s.master_seconds+1) for s in right)
        tracks,audit=fit_joint_dissolve_registration(left,right,fps=30.,maximum_zooms=(1.5,1.5))
        self.assertIsNotNone(tracks,audit)
        original=(RenderSegment(0,0.,2.,0.),RenderSegment(1,2.,2.,2.))
        effect=RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON)
        args=dict(left_segment=original[0],right_segment=original[1],left_observations=left,
            right_observations=right,tracks=tracks,transition=effect,common_master_start=0.,fps=30.)
        a,b,t=materialize_joint_dissolve_registration(**args)
        self.assertIs(a.outgoing_overlap_transform,tracks[0])
        with self.assertRaises(ValueError):
            materialize_joint_dissolve_registration(**{**args,'right_segment':replace(original[1],source_start=2.2)})
        geometry=(direct_member_source_frame_geometry((2160,3840),output_size=(1080,1920)),)*2
        checked=review_joint_dissolve_containment(segments=(a,b),transitions=(t,),source_geometries=geometry,geometry_provider=face,fps=30.)
        self.assertTrue(checked['passed'],checked)
        missing=review_joint_dissolve_containment(segments=(a,b),transitions=(t,),source_geometries=geometry,geometry_provider=lambda s,t:None,fps=30.)
        self.assertFalse(missing['passed'])
        self.assertFalse(checked['quality_approval'])

    def test_pixel_containment_support_must_match_exact_source_and_clock(self):
        from crosscut.focal_pixel_containment_support import FocalPixelContainmentSupport
        from crosscut.direct_member_transform import direct_member_source_frame_geometry
        plan=self.plan()
        geometry=(direct_member_source_frame_geometry((2160,3840),output_size=(1080,1920)),)*2
        def support(s,t):
            return FocalPixelContainmentSupport(s,t,(.4,.3,.1,.1),(t-.01,t+.01),.9)
        args=dict(segments=plan.segments,transitions=plan.transitions,source_geometries=geometry,
            geometry_provider=lambda s,t:None,fps=30.,legacy_face_only_maximum_zoom=1.2)
        good=review_joint_dissolve_containment(**args,pixel_containment_provider=support)
        self.assertTrue(good['passed'],good)
        stale=review_joint_dissolve_containment(**args,pixel_containment_provider=lambda s,t:support(s,t+.001))
        self.assertFalse(stale['passed'])
        wrong=review_joint_dissolve_containment(**args,pixel_containment_provider=lambda s,t:support(s+1,t))
        self.assertFalse(wrong['passed'])

    def test_pixel_support_rejects_native_cut_and_long_unobserved_interval(self):
        from crosscut.focal_pixel_containment_support import FocalPixelContainmentSupport,measure_focal_pixel_containment
        box=(.4,.3,.1,.1)
        with self.assertRaises(ValueError):FocalPixelContainmentSupport(0,1.,box,(0.,2.),.9)
        self.assertIsNone(measure_focal_pixel_containment(source_index=0,frames=(None,None,None),
            source_times=(1.,1.03,1.06),focal_boxes=(box,None,box),query_index=1,fps=30.,native_scene_cut_seconds=(1.02,)))


if __name__=='__main__':unittest.main()
