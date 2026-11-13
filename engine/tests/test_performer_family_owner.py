"""Retained-family normal-path integration, with explicit synthetic sensors.

Media, native PTS, dense camera planner, scope refresh, look fit, matched H264
admission, parent replay, CAS and final QA are REAL production calls. The only
model substitutes are color-landmark sensors on generated geometric charts;
these are NOT people, pose-model inference, or professional quality evidence.
No final-result wrapper or validator is replaced with a passed flag.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut import group_performer_family_owner as owner
from crosscut.boundary_subject_alignment import build_boundary_subject_alignment_plan
from crosscut.boundary_look_match import (analyze_boundary_look_match,
    verify_boundary_look_match_render_domain, BoundaryLookRenderAdmissionConfig)
from crosscut.editorial_motion import build_editorial_motion_plan
from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.first_result_controller import FirstResultController
from crosscut.ive import AspectRatio
from crosscut.optional_native_worker import isolated_optional, optional_native_scope
from crosscut.pipeline import _review_first_group_bundle, _validate_review_first_group_bundle
from crosscut.render import (RenderSegment, RenderTransition, PanCropTrack, PanCropKeyframe,
    NormalizedCrop, render, segment_frame_counts)
from crosscut.render_frame_schedule import probe_render_frame_clock
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor, VisualTimelinePlan
from crosscut.exact_native_frame_reader import ExactNativeFrameReader
from engine.tests.test_boundary_subject_alignment import _choice

_REAL_PROPOSE = owner.propose_performer_family.__wrapped__
_REAL_FINAL = owner._final_live_replay.__wrapped__


def box_from_chart(frame):
    if frame is None:return None
    # Explicit green chart region, not a human/face detector.
    b,g,r=cv2.split(frame)
    mask=((g.astype(float)-r)>45)&((g.astype(float)-b)>45)&(g>130)
    ys,xs=np.nonzero(mask)
    if len(xs)<20:return None
    x0,x1,y0,y1=int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())
    h,w=frame.shape[:2]
    return (x0/w,y0/h,(x1-x0+1)/w,(y1-y0+1)/h)


def describe_chart(frame, **kwargs):
    box=box_from_chart(frame)
    if box is None:
        return FrameVisualDescriptor(.5,.5,.0,ShotType.MEDIUM,0,(.1,.2,.3),
            pose_backend='synthetic_fixture_no_pose',shot_scale_basis='unknown_no_subject_geometry')
    # Non-identifying geometry and scalar image statistics from decoded chart.
    x,y,w,h=box
    return FrameVisualDescriptor(.9,.9,h,ShotType.MEDIUM,1,(.1,.2,.3),
        primary_face_bbox=box,color_embedding=(float(frame.mean())/255.,.2,.3),
        face_detector_backend='synthetic_color_chart_not_a_face',
        primary_face_head_roll_radians=0.,primary_face_eye_scale=w*.42,
        detected_person_count=1,pose_backend='synthetic_fixture_no_pose',
        shot_scale_basis='synthetic_color_chart_geometry')


class ChartProvider:
    def __init__(self, sources, model_path, **kwargs):
        self.sources=sources;self.handles={};self.readers={};self.capture=self
    def frame_at_native_index(self, index, tick, clock):
        if index not in self.readers:
            capture=cv2.VideoCapture(str(self.sources[index]));self.handles[index]=capture
            self.readers[index]=ExactNativeFrameReader(capture,clock)
        return self.readers[index].frame_at(tick)
    def compare_motion(self,*args,**kwargs):return None  # No invented arm evidence.
    def close(self):
        for capture in self.handles.values():capture.release()


def chart_motion(frames,samples,provider,*,output_fps,required_basis=None):
    from crosscut.observed_head_motion import HeadMotionEvidence,measure_head_motion
    # Explicit sensor substitute: the generated chart represents a fixed
    # frontal signature. This is NOT measured YuNet/pose output. Test only
    # production evidence plumbing, not the scientific sensor's correctness.
    # Absence remains None; arms are never supplied by this fixture.
    boxes=[box_from_chart(frame) for frame in frames]
    signatures=[(0.,0.,0.,0.,0.) if box is not None else None for box in boxes]
    measured=measure_head_motion(signatures,
        native_frames=tuple(s.native_frame_index for s in samples),
        output_intervals_seconds=(3/output_fps,1/output_fps,3/output_fps),
        same_source_face_paths_pixel_verified=all(box is not None for box in boxes))
    return HeadMotionEvidence(measured) if measured is not None else None


@contextmanager
def chart_sensors():
    with ExitStack() as stack:
        stack.enter_context(patch('crosscut.visual_score.describe_frame',side_effect=describe_chart))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.measured_output_face_box',side_effect=box_from_chart))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.measure_projected_motion',side_effect=chart_motion))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.observe_projected_boundary',side_effect=chart_observation))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.CloseupGestureProvider',ChartProvider))
        stack.enter_context(patch('crosscut.closeup_gesture.CloseupGestureProvider',ChartProvider))
        yield


def chart_observation(frames,samples,provider,*,output_fps,required_basis=None):
    from crosscut.group_motion_camera_refinement import ProjectedBoundaryObservation, _box_discontinuity
    boxes=[box_from_chart(frame) for frame in frames]
    faces=(_box_discontinuity(boxes[1],boxes[2])
        if boxes[1] is not None and boxes[2] is not None else None)
    motion=chart_motion(frames,samples,provider,output_fps=output_fps,required_basis=required_basis)
    return ProjectedBoundaryObservation(motion,faces)


@isolated_optional()
def chart_propose(context):
    # Importable static entry: child process recreates sensors, but runs the
    # unmodified complete production owner, not a parent-passed candidate.
    with chart_sensors():
        return _REAL_PROPOSE(context)


@isolated_optional()
def chart_final(plan_json,receipt,deadline):
    with chart_sensors():
        return _REAL_FINAL(plan_json,receipt,deadline)


@isolated_optional()
def chart_unknown_propose(context):
    from crosscut.group_motion_camera_refinement import ProjectedBoundaryObservation
    with chart_sensors(), patch('crosscut.group_motion_camera_refinement.measure_projected_motion',return_value=None), \
            patch('crosscut.group_motion_camera_refinement.observe_projected_boundary',return_value=ProjectedBoundaryObservation()):
        return _REAL_PROPOSE(context)


def encode_chart(path,*,variant,frames=360):
    width,height=320,180
    pixels=np.empty((frames,height,width,3),np.uint8)
    for tick,frame in enumerate(pixels):
        # Camera response mismatch; similar texture supports actual look fit.
        yy,xx=np.mgrid[:height,:width]
        texture=12*np.sin(xx*.43)+9*np.sin(yy*.37)
        for channel,offset in enumerate((28,38,48)):
            frame[:,:,channel]=np.clip(offset+xx*.42+texture+variant*18,0,255)
        cx,cy=160-(variant if tick<180 else 0),76
        # Prior motion lives at edge 2; prior color lives at edge 3. Their
        # synthetic geometry problems are independent. At the color-only
        # boundary (native tick 270), chart geometry is exactly equal; earlier
        # edges retain genuine nonzero camera mismatch for both old/new motion.
        halfw,halfh=((36,30) if variant==0 or tick>=240 else (32,27))
        cv2.rectangle(frame,(cx-halfw,cy-halfh),(cx+halfw-1,cy+halfh-1),(52+variant*18,192+variant*18,52+variant*18),-1)
        # Visible symmetric chart marks, no likeness of an actual person.
        cv2.circle(frame,(cx-8,cy-8),2,(22,125,22),-1)
        cv2.circle(frame,(cx+8,cy-8),2,(22,125,22),-1)
        cv2.line(frame,(cx-8,cy+9),(cx+8,cy+9),(22,125,22),2)
        for bit in range(9):
            if tick&(1<<bit):frame[2:4,2+3*bit:4+3*bit]=100
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','320x180','-r','30','-i','pipe:0',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-threads','1',
        '-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-bf','0','-c:a','aac','-y',str(path)]
    out=subprocess.run(cmd,input=pixels.tobytes(),capture_output=True,timeout=35)
    if out.returncode:raise RuntimeError(out.stderr.decode())
    return probe_render_frame_clock(path)


def build_fixture(root,sources,*,boundary=0):
    duration=10.;n=5
    mapping=tuple(range(len(sources)));offsets=(1.,)*len(sources);warps=(None,)*len(sources)
    choices=tuple(replace(_choice(i,i%2,start=2.*i,end=2.*(i+1)),
        source_local_start=2.*i+1,source_local_end=2.*(i+1)+1,
        entry_shot_scale=.4,exit_shot_scale=.4) for i in range(n))
    timeline=VisualTimelinePlan(choices,1.)
    request=SimpleNamespace(sources=sources,reference_source=0,aspect_ratio=AspectRatio.LANDSCAPE,
        source_cache_directory=root/'cache')
    section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,duration,.5,.5,1,4,15.,.4,.1,False,False),))
    with patch('crosscut.pipeline.output_size',return_value=(320,180)):
        bundle=_review_first_group_bundle(timeline=timeline,request=request,eligible_indices=mapping,
            offsets=offsets,timewarps=warps,common_start=0.,section_plan=section,
            phrase_times=(0.,2.,4.,6.,8.,10.),output=root/'candidate.mp4',
            reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
    # Retain a real non-neutral source face crop, independently of editorial
    # camera, and an actual older dense motion on a DISTANT boundary.
    pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.01,.01,.98,.98)),
        PanCropKeyframe(2.,NormalizedCrop(.01,.01,.98,.98))))
    face=FaceMatchPlan(tuple(FaceMatchSegment(i,i%2,True,None,.33,.34,1/.98,1/.98,pan,
        alignment_subject='face',alignment_purpose='synthetic_source_chart_fit') for i in range(n)),.34,1.18,(.5,.5))
    transitions=tuple(RenderTransition(tier='C') for _ in range(n-1))
    clocks=tuple(probe_render_frame_clock(p) for p in sources)
    with chart_sensors():
        alignment=build_boundary_subject_alignment_plan(visual_timeline=timeline,sources=sources,
            transitions=transitions,fps=30.,output_width=320,output_height=180,
            face_match_plan=face,render_frame_clocks=clocks,boundary_indices=(2,))
    motion=build_editorial_motion_plan(visual_timeline=timeline,transitions=transitions,
        fps=30.,boundary_subject_alignment_plan=alignment,face_match_plan=face)
    motion=replace(motion,eligible_request_indices=mapping)
    plan=replace(bundle.render_plan,fps=30.,transitions=transitions,segments=tuple(
        RenderSegment(i%2,2.*i+1,2.,2.*i,pan_crop=pan,
            editorial_transform=motion.track_for(i)) for i in range(n)))
    # Real full old look fit AND actual renderer/decode admission. The fixture
    # is not positive unless an old applied correction really exists.
    from crosscut.boundary_look_match import _tracks_from_decisions
    from crosscut.source_look_match import SourceLookAdjustment
    look=analyze_boundary_look_match(plan,group_16x9_safe=True)
    decisions=tuple(replace(d,
        applied=d.applied and d.boundary_index==3,
        reason='fixture_manually_authored_brightness_only_recipe',
        # An authored HALF-strength brightness recipe gives the retained
        # family a genuinely encoded-admitted baseline across FFmpeg builds.
        # No production admission threshold or verification result is changed.
        outgoing_adjustment=SourceLookAdjustment(brightness=d.outgoing_adjustment.brightness*.5)
            if d.boundary_index==3 else SourceLookAdjustment(),
        incoming_adjustment=SourceLookAdjustment(brightness=d.incoming_adjustment.brightness*.5)
            if d.boundary_index==3 else SourceLookAdjustment()) for d in look.decisions)
    look=replace(look,decisions=decisions,segment_tracks=_tracks_from_decisions(plan,decisions,look.analysis_config))
    look=verify_boundary_look_match_render_domain(plan,look,
        config=BoundaryLookRenderAdmissionConfig(proxy_width=320,proxy_height=180),
        maximum_verified_boundaries=4,clipping_backoff_enabled=True)
    plan=look.apply(plan)
    bundle=replace(bundle,render_plan=plan,face_match_plan=face,editorial_motion_plan=motion,
        boundary_look_match=look,group_transition_plan=replace(bundle.group_transition_plan,transitions=transitions))
    _validate_review_first_group_bundle(bundle,eligible_indices=mapping,offsets=offsets,
        timewarps=warps,common_start=0.,phrase_times=bundle.phrase_times)
    controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=sources,
        source_durations=(12.,)*len(sources),master_start=0.,duration=duration,journal_path=root/'controller.json')
    controller.install_baseline(bundle,plan=plan,authority='deterministic_full_song_visual_selector')
    model=root/'fixture-model.bin';model.write_bytes(b'explicit chart sensor fixture: no ONNX inference')
    return bundle,controller,request,model


def test_fixed_clock_keeps_identical_pan_object():
    from crosscut.group_cut_camera_proposal import retime_base_pan
    pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.1,.1,.8,.8)),))
    # Exact zero-shift should not churn a still-valid source face recipe.
    assert retime_base_pan(pan,relative_shift=0.,duration=2.) == pan


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local native tools needed')
class PerformerFamilyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory();cls.root=Path(cls.storage.name)
        cls.sources=(cls.root/'a.mp4',cls.root/'b.mp4')
        for i,path in enumerate(cls.sources):encode_chart(path,variant=i)
        cls.bundle,cls.controller,cls.request,cls.model=build_fixture(cls.root,cls.sources)
        with optional_native_scope(monotonic()+120.),patch.object(owner,'installed_pose_model',return_value=cls.model):
            cls.seed_context=owner.freeze_context(controller=cls.controller,request=cls.request,
                eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),sync_scores=(.93,.91))
            cls.seed_candidate=chart_propose(cls.seed_context)
        if isinstance(cls.seed_candidate,owner.PerformerFamilyRejection):
            raise AssertionError(cls.seed_candidate.error)

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()

    def new_controller(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=self.sources,
            source_durations=(12.,12.),master_start=0.,duration=10.,journal_path=root/'controller.json')
        controller.install_baseline(self.bundle,plan=self.bundle.render_plan,authority='deterministic_full_song_visual_selector')
        return controller,root

    def context(self,controller):
        with patch.object(owner,'installed_pose_model',return_value=self.model):
            return owner.freeze_context(controller=controller,request=self.request,eligible_indices=(0,1),
                offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),sync_scores=(.93,.91))

    def test_rolled_refresh_uses_actual_render_canvas_for_face_and_look(self):
        from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
        context, _ = self.sample()
        track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.12, roll_degrees=1.2),
            EditorialTransformKeyframe(2., 1.12, roll_degrees=1.2)))
        original = self.bundle.render_plan
        plan = replace(original, width=1280, height=720, segments=(
            replace(original.segments[0], editorial_transform=track), *original.segments[1:]))
        session = owner.GroupJointObservationSession(domain=context.domain,
            budget=owner.JointGroupWorkBudget(absolute_deadline=monotonic()+30., maximum_new_decodes=10))
        try:
            sample = owner.native_frame_clocks_for_output(plan, context.domain.clocks_by_request, [20])[20]
            with patch.object(owner, '_reference_view_for_span', wraps=owner._reference_view_for_span) as projector:
                image, observation = owner._project(session, plan, sample)
                self.assertEqual(projector.call_args.kwargs['render_size'], (1280,720))
                raw = session.native_pixels(sample.source_index, sample.native_frame_index)
                span = owner.PrescribedGeometrySampler(SimpleNamespace(), plan.segments[0], master_start=0.).span
                actual = owner._reference_view_for_span(raw, span, relative_time=20/30.,
                    size=(320,180), render_size=(1280,720))
                analysis_canvas_only = owner._reference_view_for_span(raw, span, relative_time=20/30., size=(320,180))
                self.assertTrue(np.array_equal(image,actual))
                self.assertFalse(np.array_equal(image,analysis_canvas_only))
                cache = owner.ProjectedLookCache(session, context.domain)
                cache.measure_many_for_segment(plan,0,(1.+20/30.,))
                self.assertEqual(projector.call_args.kwargs['render_size'], (1280,720))
                self.assertEqual(cache.evidence[-1]['projected_pixels_sha256'],observation['projected_pixels_sha256'])
        finally:
            session.close()

    def test_positive_retained_families_diagnostic(self):
        self.assertGreater(self.bundle.face_match_plan.applied_count,0)
        self.assertTrue(any(s.track for s in self.bundle.editorial_motion_plan.segments))
        self.assertGreater(self.bundle.boundary_look_match.applied_count,0)
        controller,root=self.new_controller()
        context,candidate=self.sample()
        self.assertIsNotNone(candidate.performer_family_review)
        owner.validate_bundle_binding(candidate)
        receipt=candidate.to_dict()['performer_family_review']
        (root/'diagnostic.json').write_text(json.dumps(receipt,indent=2))
        print('POSITIVE',receipt['changed_cells'],receipt['refreshed_cells'],receipt['incident_edges'])

    def test_positive_full_owner_independent_replay_cas_and_final_source_consumer(self):
        controller,root=self.new_controller()
        with patch.object(owner,'installed_pose_model',return_value=self.model), \
                patch.object(owner,'propose_performer_family',new=chart_propose):
            result=owner.attempt_performer_family(controller=controller,request=self.request,
                eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),
                sync_scores=(.93,.91),archive_path=root/'transaction.json')
        self.assertIsNotNone(result,controller.to_dict())
        self.assertIs(controller.incumbent,result)
        receipt=result.to_dict()['performer_family_review']
        self.assertEqual(receipt['refreshed_cells'],[0,1,2])
        for index in (3,4):
            self.assertEqual(result.render_plan.segments[index],self.bundle.render_plan.segments[index])
            self.assertEqual(result.face_match_plan.segments[index],self.bundle.face_match_plan.segments[index])
            self.assertEqual(result.editorial_motion_plan.segments[index],self.bundle.editorial_motion_plan.segments[index])
            self.assertEqual(result.visual_timeline.choices[index],self.bundle.visual_timeline.choices[index])
        self.assertEqual(result.face_match_plan.applied_count,5)
        self.assertGreater(result.boundary_look_match.applied_count,0)
        self.assertEqual(result.boundary_look_match.decisions[3],self.bundle.boundary_look_match.decisions[3])
        archive=json.loads((root/'transaction.json').read_text())
        self.assertEqual(archive['original_bundle'],self.bundle.to_dict())
        self.assertEqual(archive['candidate_bundle'],result.to_dict())
        with patch.object(owner,'_final_live_replay',new=chart_final):
            metrics=owner.replay_final_sources(result.render_plan.to_json(),receipt)
        self.assertTrue(owner.qa_metrics_valid(metrics),metrics)
        self.assertFalse(metrics['quality_approved'])
        print('PARENT_FINAL',metrics)

    def sample(self):
        return replace(self.seed_context,deadline=monotonic()+120.),self.seed_candidate

    def test_nonempty_family_and_unknown_flag_mutations_are_rejected(self):
        context,candidate=self.sample()
        receipt=candidate.to_dict()['performer_family_review']
        variants=[replace(candidate,performer_family_review=None),
            replace(candidate,face_match_plan=None),replace(candidate,editorial_motion_plan=None),
            replace(candidate,boundary_look_match=None),
            replace(candidate,render_plan=replace(candidate.render_plan,segments=(
                replace(candidate.render_plan.segments[0],source_start=1.1),*candidate.render_plan.segments[1:])))]
        for key in ('quality_approved','body_phase_certified','decoded_output_quality_certified','cross_camera_identity_certified','source_changes_supported'):
            variants.append(replace(candidate,performer_family_review={**receipt,key:True}))
        for i,value in enumerate(variants):
            with self.subTest(mutation=i),self.assertRaises(ValueError):owner.validate_bundle_binding(value)
        for name in ('_review_first_group_motion_camera_bundle','_review_first_group_shot_context_bundle',
                '_review_first_group_boundary_look_bundle'):
            import crosscut.pipeline as pipeline
            kwargs=dict(request=None,eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),
                scene_cut_times=((),()),phrase_times=candidate.phrase_times)
            if name.endswith('shot_context_bundle'):kwargs=dict(scene_cut_times=((),()))
            if name.endswith('boundary_look_bundle'):kwargs=dict(group_16x9_safe=True)
            with self.subTest(stage=name),self.assertRaisesRegex(ValueError,'fresh complete replay'):
                getattr(pipeline,name)(candidate,**kwargs)

    def test_rehashed_stale_face_meaning_fails_fresh_independent_parent(self):
        context,candidate=self.sample()
        face=replace(candidate.face_match_plan,segments=(replace(candidate.face_match_plan.segments[0],
            alignment_purpose='forged_source_recipe_meaning'),*candidate.face_match_plan.segments[1:]))
        altered=replace(candidate,face_match_plan=face)
        receipt=candidate.to_dict()['performer_family_review'];receipt['metadata_sha256']=owner.digest(owner.metadata_payload(altered))
        altered=replace(altered,performer_family_review=receipt)
        owner.validate_bundle_binding(altered) # Self-consistent hash is NOT sufficient.
        with optional_native_scope(monotonic()+100.),patch.object(owner,'propose_performer_family',new=chart_propose):
            with self.assertRaisesRegex(ValueError,'replay disagrees'):owner.validate_parent(altered,context)

    def test_actual_live_source_and_model_mutations_invalidate_frozen_context(self):
        import os
        controller,root=self.new_controller()
        # Mutation fixtures own their inodes. Restoring bytes and mtime cannot
        # restore ctime, which is intentionally part of production freshness.
        # Never poison the shared, previously admitted seed for later tests.
        sources=tuple(root/f'mutation-source-{i}.mp4' for i in range(len(self.sources)))
        for source,target in zip(self.sources,sources,strict=True):shutil.copyfile(source,target)
        bundle,controller,request,model=build_fixture(root,sources)
        with optional_native_scope(monotonic()+60.),patch.object(owner,'installed_pose_model',return_value=model):
            context=owner.freeze_context(controller=controller,request=request,eligible_indices=(0,1),
                offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),sync_scores=(.93,.91))
        for path in (sources[0],model):
            previous=path.read_bytes();stat=path.stat()
            try:
                path.write_bytes(previous+b'changed-after-freeze')
                with self.subTest(path=path.name),self.assertRaisesRegex(ValueError,'preimage changed'):
                    owner._fresh(context)
            finally:
                path.write_bytes(previous);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        sibling=model.parent/'fixture-secondary-model.onnx'
        try:
            sibling.write_bytes(b'non-weight fault injection to test absent-asset binding')
            with self.assertRaisesRegex(ValueError,'preimage changed'):owner._fresh(context)
        finally:sibling.unlink()
        with self.assertRaisesRegex(ValueError,'preimage changed'):
            owner._fresh(replace(context,code_dependencies=(('changed-code','0'*64),)))

    def test_missing_model_does_not_download_or_erase_existing_families(self):
        controller,root=self.new_controller();token=controller.transaction_token()
        with patch.object(owner,'propose_performer_family',side_effect=AssertionError('missing model cannot propose')) as proposer, \
                patch('urllib.request.urlopen',side_effect=AssertionError('no network')) as network:
            result=owner.attempt_performer_family(controller=controller,request=self.request,
                eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),
                sync_scores=(.93,.91),archive_path=root/'missing.json')
        self.assertIsNone(result);proposer.assert_not_called();network.assert_not_called()
        self.assertEqual(controller.transaction_token(),token);self.assertIs(controller.incumbent,self.bundle)
        archive=json.loads((root/'missing.json').read_text())
        self.assertEqual(archive['original_bundle'],self.bundle.to_dict())
        self.assertIn('no download attempted',archive['error'])

    def test_archive_failure_precedes_native_proposal_and_preserves_owner(self):
        controller,root=self.new_controller();token=controller.transaction_token()
        with patch('crosscut.pipeline._write_analysis_manifest_atomically',side_effect=OSError('injected archive failure')), \
                patch.object(owner,'propose_performer_family',side_effect=AssertionError('unarchived proposal forbidden')) as proposer:
            result=owner.attempt_performer_family(controller=controller,request=self.request,
                eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),
                sync_scores=(.93,.91),archive_path=root/'bad.json')
        self.assertIsNone(result);proposer.assert_not_called()
        self.assertEqual(controller.transaction_token(),token)
        self.assertIs(controller.incumbent,self.bundle)

    def test_stale_cas_token_rejects_before_any_fresh_sensor_work(self):
        controller,root=self.new_controller();token=controller.transaction_token()
        controller.promote(self.bundle,plan=self.bundle.render_plan,authority='newer_existing_owner')
        with patch.object(owner,'validate_parent',side_effect=AssertionError('stale parent cannot run')) as parent:
            with self.assertRaisesRegex(ValueError,'baseline changed before parent'):
                controller.compare_and_promote(self.bundle,plan=self.bundle.render_plan,authority=owner.CONTRACT,
                    token=token,validate=parent)
        parent.assert_not_called();self.assertIs(controller.incumbent,self.bundle)

    def test_same_source_clock_and_base_face_recipe_are_pinned(self):
        context,candidate=self.sample()
        old=self.bundle.render_plan
        variants=[replace(candidate.render_plan,segments=(replace(candidate.render_plan.segments[0],source_index=1),
            *candidate.render_plan.segments[1:])),
            replace(candidate.render_plan,segments=(replace(candidate.render_plan.segments[0],pan_crop=None),
            *candidate.render_plan.segments[1:])),
            replace(candidate.render_plan,fps=29.97)]
        for i,plan in enumerate(variants):
            with self.subTest(mutation=i),self.assertRaises(ValueError):owner.changed_scope(old,plan,context.domain)
        self.assertEqual(owner.changed_scope(old,candidate.render_plan,context.domain),((1,),(0,1,2),(0,1)))

    def test_actual_fresh_look_failure_preserves_previous_active_recipe(self):
        from crosscut.group_joint_observations import GroupJointObservationSession
        from crosscut.group_joint_source_camera import JointGroupWorkBudget
        controller,root=self.new_controller()
        with optional_native_scope(monotonic()+60.):context=self.context(controller)
        session=GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=monotonic()+60.,maximum_new_decodes=160))
        try:
            # Prior brightness-only edge 3 was actually encode-verified. The
            # full fresh automatic color fit on these charts is a real negative,
            # not a fake passed/failed reviewer return value.
            with self.assertRaisesRegex(ValueError,'existing active look failed fresh admission'):
                owner.refresh_look(self.bundle,self.bundle.render_plan,(3,),session,context.domain)
        finally:session.close()
        self.assertGreater(self.bundle.boundary_look_match.applied_count,0)
        self.assertIs(controller.incumbent,self.bundle)


    def test_unknown_native_motion_rejects_with_full_ipc_evidence_not_zero(self):
        controller,root=self.new_controller();token=controller.transaction_token()
        with patch.object(owner,'installed_pose_model',return_value=self.model), \
                patch.object(owner,'propose_performer_family',new=chart_unknown_propose):
            result=owner.attempt_performer_family(controller=controller,request=self.request,
                eligible_indices=(0,1),offsets=(1.,1.),timewarps=(None,None),scene_cut_times=((),()),
                sync_scores=(.93,.91),archive_path=root/'unknown.json')
        self.assertIsNone(result);self.assertEqual(controller.transaction_token(),token)
        archive=json.loads((root/'unknown.json').read_text())
        evidence=archive['producer_rejection_evidence']
        self.assertIsNotNone(evidence)
        attempts=evidence['candidate_lineage'][-1]['attempts']
        rows=[r for r in attempts if r.get('stage')=='baseline_motion']
        self.assertTrue(rows)
        self.assertTrue(all(r['motion_cost'] is None for r in rows))
        self.assertEqual(archive['original_bundle']['face_match'],self.bundle.face_match_plan.to_dict())
        self.assertEqual(archive['original_bundle']['editorial_motion'],self.bundle.editorial_motion_plan.to_dict())
        self.assertEqual(archive['original_bundle']['boundary_look_match'],self.bundle.boundary_look_match.to_dict())

    def test_quality_uncertainty_cannot_be_removed_to_obtain_approval(self):
        from crosscut.review_first_runtime import classify_review_first_qa,ReviewFirstStatus
        from engine.tests.test_review_first_runtime import _group_integrity_metrics
        candidate=self.seed_candidate;receipt=candidate.to_dict()['performer_family_review']
        with patch.object(owner,'_final_live_replay',new=chart_final):
            measured=owner.replay_final_sources(candidate.render_plan.to_json(),receipt)
        metrics={**_group_integrity_metrics(),'performer_family_review':measured}
        report={'passed':False,'metrics':metrics,'issues':[{'code':owner.QUALITY_CODE}]}
        def status(value):
            return classify_review_first_qa(value,content_binding_passed=True,review_first_enabled=True,member=False)
        self.assertEqual(status(report).status,ReviewFirstStatus.REVIEW_READY)
        self.assertFalse(status(report).delivery_completion_message_allowed)
        removed={**report,'passed':True,'issues':[]}
        self.assertEqual(status(removed).status,ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertIn('performer_family_unapproved_quality_issue_missing',status(removed).technical_issue_codes)
        forged={**measured,'quality_approved':True};forged.pop('evidence_sha256')
        forged['evidence_sha256']=owner.digest(forged)
        self.assertEqual(status({**report,'metrics':{**metrics,'performer_family_review':forged}}).status,
            ReviewFirstStatus.INPUT_UNUSABLE)
