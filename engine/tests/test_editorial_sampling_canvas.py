from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from crosscut.render import (EditorialTransformKeyframe,EditorialTransformTrack,RenderPlan,
    _editorial_transform_filters,_has_identity_geometry,_video_post_clock_filters)
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut.post_render_qa import _boundary_editorial_track_from_serialized
from crosscut.post_render_perceptual_quality import _reference_view_for_span
from crosscut.prescribed_render_geometry import PrescribedGeometrySampler
from crosscut.shot_context_camera import retain_opposite_cut_camera,slice_shot_camera_track
from engine.tests import test_opposite_cut_camera_retention as fixtures


class EditorialSamplingCanvasTests(unittest.TestCase):
    def track(self):
        return EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
            EditorialTransformKeyframe(1.,1.14)),sampling_canvas_zoom=1.52)

    def test_explicit_canvas_is_bounded_and_not_crop_authority(self):
        track=self.track()
        self.assertEqual(track.effective_sampling_canvas_zoom,1.52)
        self.assertEqual(track.evaluate(0.).zoom,1.1)
        filters=_editorial_transform_filters(track,width=320,height=180,fps=30.)
        self.assertEqual(filters[0],'scale=488:274:force_original_aspect_ratio=increase')
        for canvas in (True,.99,1.13,2.01,float('nan'),float('inf')):
            with self.subTest(canvas=canvas),self.assertRaises(ValueError):
                replace(track,sampling_canvas_zoom=canvas)

    def test_legacy_json_and_pixels_remain_unchanged_by_default(self):
        track=replace(self.track(),sampling_canvas_zoom=None)
        self.assertNotIn('sampling_canvas_zoom',track.to_dict())
        self.assertEqual(track.effective_sampling_canvas_zoom,1.14)
        segment,_,_=fixtures.OppositeCutCameraRetentionTests().fixture()
        segment=replace(segment,source_index=0,master_start=0.)
        plan=RenderPlan((Path('s.mp4'),),(segment,),0,'16:9',Path('o.mp4'),fps=30.)
        self.assertNotIn('sampling_canvas_zoom',plan.to_json()['segments'][0]['editorial_transform'])
        self.assertEqual(render_plan_from_json(json.loads(json.dumps(plan.to_json()))),plan)

    def test_plan_motion_and_qa_roundtrip_keep_the_explicit_canvas(self):
        from crosscut.boundary_subject_alignment import BoundarySubjectAlignmentSegment
        from crosscut.editorial_motion import EditorialMotionSegment
        from crosscut.pipeline import _final_render_segments_payload
        segment,_,_=fixtures.OppositeCutCameraRetentionTests().fixture()
        segment=replace(segment,source_index=0,master_start=0.,editorial_transform=self.track())
        plan=RenderPlan((Path('s.mp4'),),(segment,),0,'16:9',Path('o.mp4'),fps=30.)
        self.assertEqual(render_plan_from_json(json.loads(json.dumps(plan.to_json()))),plan)
        raw=json.loads(json.dumps(plan.to_json()))['segments'][0]['editorial_transform']
        self.assertEqual(_boundary_editorial_track_from_serialized(raw,label='fixture'),self.track())
        self.assertEqual(EditorialMotionSegment(0,0,self.track()).to_dict()['sampling_canvas_zoom'],1.52)
        self.assertEqual(BoundarySubjectAlignmentSegment(0,0,self.track()).to_dict()['sampling_canvas_zoom'],1.52)
        self.assertEqual(_final_render_segments_payload(plan.segments,common_master_start=0.)[0]
            ['editorial_sampling_canvas_zoom'],1.52)

    def test_pixel_reference_consumes_the_same_sampling_canvas_as_the_renderer(self):
        from unittest.mock import patch
        segment,_,_=fixtures.OppositeCutCameraRetentionTests().fixture();segment=replace(segment,editorial_transform=self.track())
        span=PrescribedGeometrySampler(None,segment,master_start=0.).span
        pixels=np.zeros((108,192,3),np.uint8)
        with patch('crosscut.render_geometry.reproject_editorial_frame',return_value=pixels) as reproject:
            _reference_view_for_span(pixels,span,relative_time=.2,size=(192,108))
        self.assertEqual(reproject.call_args.kwargs['maximum_zoom'],1.52)
        bad=replace(span,editorial_transform=dict(span.editorial_transform,sampling_canvas_zoom=True))
        with self.assertRaises(ValueError):_reference_view_for_span(pixels,bad,relative_time=.2,size=(192,108))

    def test_camera_slices_keep_the_whole_curve_raster_and_do_not_pixel_coalesce(self):
        for track in (self.track(),replace(self.track(),sampling_canvas_zoom=None)):
            sliced=slice_shot_camera_track(track,frame_offset=0,frame_count=8,fps=30.)
            self.assertEqual(sliced.effective_sampling_canvas_zoom,track.effective_sampling_canvas_zoom)
        neutral=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),),sampling_canvas_zoom=1.52)
        segment,_,_=fixtures.OppositeCutCameraRetentionTests().fixture()
        self.assertFalse(_has_identity_geometry(replace(segment,pan_crop=None,editorial_transform=neutral)))

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_actual_ffmpeg_pixels_keep_the_complete_protected_window(self):
        segment,native,args=fixtures.OppositeCutCameraRetentionTests().fixture()
        # The old, larger peak is outside the protected tail. It still decides
        # the renderer's raster there, and must not disappear after editing.
        old=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.52),
            EditorialTransformKeyframe((43-1)/30.,1.14,.51,.50)))
        segment=replace(segment,editorial_transform=old)
        result=retain_opposite_cut_camera(segment,native,**args)
        self.assertEqual(result.editorial_transform.sampling_canvas_zoom,1.52)
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=s=640x360:r=30:d=1.5','-c:v','libx264','-crf','10',str(source)],
                check=True,capture_output=True,timeout=30)
            def pixels(version):
                filters=_video_post_clock_filters(crop=None,pan_crop=version.pan_crop,
                    editorial_transform=version.editorial_transform,look_adjustment=None,
                    boundary_look_track=None,width=320,height=180,fps=30.,frame_count=43,
                    input_clock_is_canonical=True)
                raw=subprocess.run(['ffmpeg','-v','error','-i',str(source),'-vf',','.join(filters),
                    '-frames:v','43','-an','-pix_fmt','rgb24','-f','rawvideo','-'],
                    check=True,capture_output=True,timeout=30).stdout
                return np.frombuffer(raw,np.uint8).reshape(43,180,320,3)
            before,after=pixels(segment),pixels(result)
            np.testing.assert_array_equal(before[-4:],after[-4:])
            self.assertFalse(np.array_equal(before[0],after[0]))
