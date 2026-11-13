"""Synthetic FFmpeg pixels -> actual source-conformance/final MEMBER QA.

Sources are generated calibration pictures, not IVE media. Pose/identity and
sync observations use explicit synthetic providers; codecs, pixels, source
clock binding, artifact hashing and final QA execution are real.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

import cv2
import numpy as np

from crosscut.first_result_controller import FirstResultController
from crosscut.member_camera_worker import validate_member_camera_clock
from crosscut.optional_native_worker import run_isolated_python_call
from crosscut.render import render
from crosscut.render_reference_clock import load_render_reference_clock
from crosscut.review_first_runtime import ReviewFirstRuntime,classify_review_first_qa
from crosscut.member_post_render_qa import evaluate_automatic_member_post_render_qa
from tests.test_member_post_render_qa import _fixture


def calibration_sources(paths):
    height,width=640,360
    yy,xx=np.indices((height,width))
    gray=(75+((xx//16+yy//16)%2)*80).astype(np.uint8)
    image=np.repeat(gray[:,:,None],3,axis=2)
    for centre in ((100,180),(260,180),(100,460),(260,460)):
        cv2.circle(image,centre,13,(20,20,240),-1)
    for index,path in enumerate(paths):
        angle=2.4 if index%2 else 0.
        frame=cv2.warpAffine(image,cv2.getRotationMatrix2D((width/2,height/2),-angle,1.),
            (width,height),borderMode=cv2.BORDER_REFLECT)
        png=path.with_suffix('.png')
        cv2.imwrite(str(png),frame)
        subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(png),
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','13',
            '-c:v','libx264','-preset','ultrafast','-crf','10','-pix_fmt','yuv420p','-c:a','aac',str(path)],
            check=True,capture_output=True,timeout=30)


def decoded(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
    finally:cap.release()
    return np.asarray(frames)


def bind_final_manifest(artifacts,manifest,plan,motion):
    """Build real clock proof with explicit synthetic motion observations."""
    from tests.test_member_post_render_qa import _identity_timewarp,_motion_descriptor
    from crosscut.timewarp import TimeWarpControlPoint
    from crosscut.cut_local_sync import analyze_cut_local_sync,MEMBER_ACTIVATION
    from crosscut.reference_match import DecodedTargetFrames
    warp=replace(_identity_timewarp(),master_duration_seconds=13.,source_duration_seconds=13.,
        control_points=(TimeWarpControlPoint(0.,0.),TimeWarpControlPoint(13.,13.)))
    payloads=[dict(warp.to_dict(),source_index=i,status='local_monotonic_timewarp') for i in range(len(plan.sources))]
    points=tuple(i/30 for i in range(391))
    by_source={p.resolve():i for i,p in enumerate(plan.sources)}
    def decode(path,indices,_config):
        values={}
        for n in sorted(set(indices)):
            frame=np.zeros((8,8,3),dtype=np.float32)
            frame[0,0,0]=n;frame[0,0,1]=by_source[path.resolve()]
            values[n]=frame
        return DecodedTargetFrames(frame_rate=30.,declared_frame_count=len(points),
            requested_indices=tuple(sorted(values)),frames=values)
    manifest['local_timewarps']=payloads
    manifest['render']['cut_local_sync']=analyze_cut_local_sync(plan,source_timewarps=(warp,)*len(plan.sources),
        timewarp_payloads=payloads,activation=MEMBER_ACTIVATION,pts_provider=lambda p,t:points,
        decode_provider=decode,descriptor_provider=lambda frame:_motion_descriptor(frame,source_motion_shift_frames=0))
    manifest['render']['editorial_motion']=motion.to_dict()
    usage=plan.source_usage_payload()
    manifest['render']['final_source_usage']=usage
    manifest['member_direct_full_frame']['final_source_usage']=usage
    for item,segment in zip(manifest['member_direct_full_frame']['segments'],plan.segments):
        item.update(editorial_transform_applied=segment.editorial_transform is not None,
            source_local_start=segment.source_start,source_local_end=segment.source_end)
    artifacts.render_plan_path.write_text(json.dumps(plan.to_json()))
    artifacts.analysis_path.write_text(json.dumps(manifest))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg/ffprobe required')
class MemberRollPixelTests(unittest.TestCase):
    def test_child_roll_is_visible_in_pixels_and_final_qa_binds_that_exact_plan(self):
        with TemporaryDirectory() as temp:
            artifacts,manifest=_fixture(Path(temp))
            paths=tuple(Path(p) for p in manifest['request']['sources'])
            calibration_sources(paths)
            _,baseline,proposal=run_isolated_python_call('tests.member_roll_fixture','worker_proposal',
                dict(sources=paths,output=artifacts.video_path,dissolve=False,segment_duration=3.),deadline=monotonic()+20.)
            validate_member_camera_clock(proposal,baseline)
            owner=FirstResultController(ReviewFirstRuntime(enabled=True),sources=paths,
                source_durations=(13.,)*4,master_start=0.,duration=12.)
            owner.install_baseline((baseline,None,None,{},{}),plan=baseline,authority='verified_member_source_primitive')
            accepted=owner.run_optional('observed_roll',lambda:proposal,
                validate=lambda value:validate_member_camera_clock(value,baseline))
            owner.promote(accepted,plan=accepted[0],authority='registered_member_camera_proposal')
            plan=owner.incumbent[0]
            self.assertEqual(proposal[1].roll_event_count,4)
            render(plan,video_codec='libx264')
            owner.render_complete()
            actual=decoded(plan.output)
            self.assertEqual(len(actual),plan.output_frame_count)
            # Same source, crop, zoom and pan, changing ONLY roll. The actual
            # production renderer must produce different pixels on rolled ticks.
            no_roll=replace(plan,output=Path(temp)/'no-roll.mp4',segments=tuple(replace(s,
                editorial_transform=replace(s.editorial_transform,keyframes=tuple(replace(p,roll_degrees=0.)
                    for p in s.editorial_transform.keyframes))) for s in plan.segments))
            render(no_roll,video_codec='libx264')
            flat=decoded(no_roll.output)
            self.assertEqual(actual.shape,flat.shape)
            self.assertGreater(float(np.abs(actual[89].astype(float)-flat[89]).mean()),1.)
            # Independent CV projection used by the actual source-fidelity QA
            # must prefer the requested angle to missing and reversed roll.
            from crosscut.post_render_perceptual_quality import _parse_plan,_reference_view_for_span
            serialized=json.loads(json.dumps(plan.to_json()))
            spans=_parse_plan(serialized)[-1]
            original=decoded(paths[0])[89]
            reference=_reference_view_for_span(original,spans[0],relative_time=89/30,size=(plan.width,plan.height))
            for analysis_size in ((256,455),(90,160)):
                reference_at_analysis_size=_reference_view_for_span(original,spans[0],relative_time=89/30,
                    size=analysis_size,render_size=(plan.width,plan.height))
                np.testing.assert_array_equal(reference_at_analysis_size,
                    cv2.resize(reference,analysis_size,interpolation=cv2.INTER_AREA))
            flat_spans=_parse_plan(json.loads(json.dumps(no_roll.to_json())))[-1]
            wrong=_reference_view_for_span(original,flat_spans[0],relative_time=89/30,size=(plan.width,plan.height))
            error=lambda image:float(np.abs(actual[89].astype(float)-image).mean())
            self.assertLess(error(reference),error(wrong))
            flipped=deepcopy(serialized)
            for segment in flipped['segments']:
                for point in segment['editorial_transform']['keyframes']:point['roll_degrees']*=-1
            reversed_reference=_reference_view_for_span(original,_parse_plan(flipped)[-1][0],relative_time=89/30,
                size=(plan.width,plan.height))
            self.assertLess(error(reference),error(reversed_reference))
            clock=load_render_reference_clock(plan.output,serialized,(89,90,269,270))
            self.assertEqual(set(clock.frames),{89,90,269,270})
            self.assertEqual(clock.frames[90].source_index,1)
            bind_final_manifest(artifacts,manifest,plan,proposal[1])
            # Pose/motion descriptors are synthetic; final FFmpeg full decode,
            # perceptual source-view comparison, fingerprints and gates are real.
            from tests.member_continuity_fixture import member_continuity_report
            continuity=lambda path,payload,**kw:member_continuity_report(payload)
            report=evaluate_automatic_member_post_render_qa(artifacts,rendered_continuity_evaluator=continuity)
            payload=report.to_dict()
            self.assertTrue(payload['render_receipt']['all_eligible_sources_pixel_proven'],payload['issues'])
            self.assertTrue(payload['metrics']['content_stability']['artifact_content_stable'])
            self.assertTrue(payload['metrics']['perceptual_quality']['metrics']['source_reference_clock']['verified_reference_count']>0)
            self.assertFalse(any('contract' in issue['code'] or 'clock' in issue['code'] or 'provenance' in issue['code']
                                 for issue in payload['issues']),payload['issues'])
            positive=classify_review_first_qa(payload,content_binding_passed=True,review_first_enabled=True,member=True)
            self.assertTrue(positive.video_output_exposed,payload['issues'])
            print(json.dumps(dict(test='member_roll_pixels_dissolve_off',frame_count=len(actual),
                rotated_segments=proposal[1].roll_event_count,pixel_mae_correct=error(reference),
                pixel_mae_dropped_roll=error(wrong),pixel_mae_reversed_roll=error(reversed_reference),
                final_qa_passed=report.passed,final_status=positive.status.value,
                final_qa_issue_codes=[i['code'] for i in payload['issues']])))
            # Dropping a serialized angle while keeping the old encoded file
            # cannot pass the independent final plan/video clock binding.
            damaged=deepcopy(serialized)
            damaged['segments'][0]['editorial_transform']['keyframes'][-1]['roll_degrees']=0.
            artifacts.render_plan_path.write_text(json.dumps(damaged))
            rejected=evaluate_automatic_member_post_render_qa(artifacts,
                rendered_continuity_evaluator=continuity).to_dict()
            self.assertFalse(rejected['passed'])
            self.assertTrue(any(i['code']=='perceptual_render_source_clock_invalid'
                for i in rejected['metrics']['perceptual_quality']['issues']),rejected['issues'])
            disposition=classify_review_first_qa(rejected,content_binding_passed=True,
                review_first_enabled=True,member=True)
            self.assertFalse(disposition.video_output_exposed)

    def test_registered_overlap_roll_changes_real_dissolve_pixels_and_round_trips(self):
        from crosscut.render_plan_serialization import render_plan_from_json
        with TemporaryDirectory() as temp:
            artifacts,manifest=_fixture(Path(temp))
            paths=tuple(Path(p) for p in manifest['request']['sources'])
            calibration_sources(paths)
            _,base,proposal=run_isolated_python_call('tests.member_roll_fixture','worker_proposal',
                dict(sources=paths,output=artifacts.video_path,dissolve=True,segment_duration=3.),deadline=monotonic()+20.)
            plan=proposal[0]
            validate_member_camera_clock(proposal,base)
            self.assertTrue(plan.member_pose_dissolve_effective)
            restored=render_plan_from_json(json.loads(json.dumps(plan.to_json())))
            self.assertEqual(restored.segments,plan.segments)
            render(plan,video_codec='libx264')
            actual=decoded(plan.output)
            self.assertEqual(len(actual),360)
            def without_roll(track):
                return replace(track,keyframes=tuple(replace(p,roll_degrees=0.) for p in track.keyframes)) if track else None
            flat_plan=replace(plan,output=Path(temp)/'flat-dissolve.mp4',segments=tuple(replace(s,
                editorial_transform=without_roll(s.editorial_transform),
                outgoing_overlap_transform=without_roll(s.outgoing_overlap_transform)) for s in plan.segments))
            render(flat_plan,video_codec='libx264')
            flat=decoded(flat_plan.output)
            self.assertEqual(actual.shape,flat.shape)
            for frame in (*range(90,97),*range(270,277)):
                self.assertGreater(float(np.abs(actual[frame].astype(float)-flat[frame]).mean()),1.)
            bind_final_manifest(artifacts,manifest,plan,proposal[1])
            from tests.member_continuity_fixture import member_continuity_report
            report=evaluate_automatic_member_post_render_qa(artifacts,
                rendered_continuity_evaluator=lambda path,payload,**kw:member_continuity_report(payload))
            payload=report.to_dict()
            self.assertTrue(payload['render_receipt']['all_eligible_sources_pixel_proven'],payload['issues'])
            self.assertFalse(any('contract' in i['code'] or 'clock' in i['code'] or 'provenance' in i['code']
                for i in payload['issues']),payload['issues'])
            disposition=classify_review_first_qa(payload,content_binding_passed=True,review_first_enabled=True,member=True)
            self.assertTrue(disposition.video_output_exposed,payload['issues'])
            print(json.dumps(dict(test='member_roll_pixels_dissolve_on',frame_count=len(actual),
                rotated_segments=proposal[1].roll_event_count,rotated_overlap_frames=14,
                final_qa_passed=report.passed,final_status=disposition.status.value,
                final_qa_issue_codes=[i['code'] for i in payload['issues']],
                continuity_failures=payload['metrics']['rendered_continuity'].get('failure_reasons'),
                perceptual_issue_codes=[i['code'] for i in payload['metrics']['perceptual_quality']['issues']])))
