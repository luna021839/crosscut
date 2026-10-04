"""NON-HUMAN chart fixtures. Model/derivative outputs are explicit substitutes.

Real: source decoders/PTS, selected-landmark LK, projection, scene graph,
whole-camera assembly, retained-family refresh, look encode, independent worker
replay, CAS and final QA. Not established: human inference, real-camera
classification, named actor/phase/head extent or professional quality.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from itertools import combinations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
from unittest.mock import patch
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import cv2
import numpy as np

from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut import group_scene_source_owner as scene
from crosscut import group_scene_observations as sensors_module
from crosscut.optional_native_worker import isolated_optional,optional_native_scope
from crosscut.group_joint_native_observer import NativeGroupHead
from crosscut.source_derivative_diversity import (SourceDerivativeDiversityResult,
    SourceDerivativePairEvidence,SourceDerivativeConfig,DERIVATIVE_DISTINCT)
from engine.tests.test_source_changing_family_owner import source_chart_sensors,chart_native
from engine.tests.test_performer_family_owner import encode_chart,build_fixture,chart_sensors,box_from_chart

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
    NormalizedCrop, EditorialTransformTrack, EditorialTransformKeyframe, render, segment_frame_counts)
from crosscut.render_frame_schedule import probe_render_frame_clock
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor, VisualTimelinePlan
from crosscut.exact_native_frame_reader import ExactNativeFrameReader
from engine.tests.test_boundary_subject_alignment import _choice


_REAL_PROPOSE=family.propose_performer_family.__wrapped__
_REAL_FINAL=source.final_live_replay.__wrapped__


class FixedEndpointInventoryTests(unittest.TestCase):
    def fixture(self, failed=(), interrupted=None):
        states=tuple(SimpleNamespace(key=str(i)*64,index=i) for i in range(4))
        calls=[]
        def hypothesis(value):
            return dict(analysis_roll=None,anchor_head_ids=[value.index]*2)
        def endpoint(value,*,outgoing):
            calls.append(value.index)
            if value.index==interrupted:
                raise scene.JointGroupBudgetExhausted('EXPLICIT_FIXED_PROBE_BUDGET_INTERRUPTION')
            if value.index in failed:
                raise sensors_module.SceneHypothesisRejection('EXPLICIT_OBSERVED_ENDPOINT_UNAVAILABLE')
            return ()
        return states,SimpleNamespace(hypothesis=hypothesis,endpoint=endpoint),calls

    def run_inventory(self,failed=(),interrupted=None):
        states,sensors,calls=self.fixture(failed,interrupted)
        inventory=scene.fixed_endpoint_inventory(states,sensors=sensors,entry=False)
        record={}
        return inventory,sensors,calls,record

    def test_failed_first_two_do_not_starve_already_discovered_third(self):
        inventory,sensors,calls,record=self.run_inventory(failed=(0,1))
        result=scene.measure_fixed_endpoint_inventory(inventory,sensors=sensors,outgoing=True,record=record)
        self.assertEqual(tuple(value.index for value in result),(2,3))
        self.assertEqual(calls,[0,1,2,3])
        self.assertEqual(record['outcome'],'USABLE')
        self.assertTrue(record['inventory_exhausted'])
        self.assertEqual(record['next_cursor'],4)
        self.assertFalse(record['new_budget_retry_started'])

    def test_all_observed_failures_are_not_an_unexecuted_suffix(self):
        inventory,sensors,calls,record=self.run_inventory(failed=(0,1,2,3))
        result=scene.measure_fixed_endpoint_inventory(inventory,sensors=sensors,outgoing=True,record=record)
        self.assertEqual(result,())
        self.assertEqual(calls,[0,1,2,3])
        self.assertEqual(record['outcome'],'OBSERVED_UNAVAILABLE')
        self.assertTrue(record['inventory_exhausted'])
        self.assertEqual(record['unexecuted_suffix_count'],0)

    def test_budget_before_third_retains_cursor_and_never_restarts(self):
        inventory,sensors,calls,record=self.run_inventory(failed=(0,1),interrupted=2)
        with self.assertRaises(scene.JointGroupBudgetExhausted):
            scene.measure_fixed_endpoint_inventory(inventory,sensors=sensors,outgoing=True,record=record)
        self.assertEqual(calls,[0,1,2])
        self.assertEqual(record['outcome'],'BUDGET_PARTIAL')
        self.assertEqual(record['next_cursor'],2)
        self.assertEqual(record['completed_measurement_count'],2)
        self.assertFalse(record['inventory_exhausted'])
        self.assertEqual(record['unexecuted_suffix_count'],1)
        self.assertEqual(record['pending_candidate_count'],2)
        self.assertEqual(record['interrupted_current_state_key'],inventory[2].key)
        self.assertEqual(record['attempts'][-1]['status'],'BUDGET_PARTIAL')
        self.assertFalse(record['new_budget_retry_started'])

    def test_two_usable_stops_with_explicit_unmeasured_suffix_not_exhaustiveness(self):
        inventory,sensors,calls,record=self.run_inventory()
        result=scene.measure_fixed_endpoint_inventory(inventory,sensors=sensors,outgoing=True,record=record)
        self.assertEqual(tuple(value.index for value in result),(0,1))
        self.assertEqual(calls,[0,1])
        self.assertEqual(record['outcome'],'USABLE')
        self.assertFalse(record['inventory_exhausted'])
        self.assertEqual(record['unexecuted_suffix_count'],2)

    def test_empty_inventory_is_not_a_measurement_of_negative_evidence(self):
        _,sensors,_,record=self.run_inventory()
        result=scene.measure_fixed_endpoint_inventory((),sensors=sensors,outgoing=True,record=record)
        self.assertEqual(result,())
        self.assertEqual(record['outcome'],'NOT_STARTED')
        self.assertEqual(record['attempts'],[])
        self.assertEqual(record['completed_measurement_count'],0)

    def test_contract_failure_does_not_consume_the_next_fixed_endpoint(self):
        values,sensors,calls,record=self.run_inventory()
        def bad_contract(value,**kwargs):
            calls.append(value.index)
            raise ValueError('EXPLICIT_BAD_FIXED_ENDPOINT_REFERENCE')
        sensors.endpoint=bad_contract
        with self.assertRaises(ValueError):
            scene.measure_fixed_endpoint_inventory(values,sensors=sensors,outgoing=True,record=record)
        self.assertEqual(calls,[0])
        self.assertEqual(record['outcome'],'PRODUCER_FAILED')
        self.assertEqual(record['completed_measurement_count'],0)


def scene_derivative_fixture(count=8):
    pairs=tuple(SourceDerivativePairEvidence(a,b,DERIVATIVE_DISTINCT,
        'EXPLICIT_NONHUMAN_CAMERA_RELATION_SENSOR_FIXTURE_NOT_ACTUAL_DERIVATIVE_ANALYSIS',
        5,0,5,0,None,None,None,None,None,None,()) for a,b in combinations(range(count),2))
    value=SourceDerivativeDiversityResult((1.,3.,5.,7.,9.),(),pairs,tuple(range(count)),
        0,True,True,'',SourceDerivativeConfig())
    return replace(value,audit_sha256=family.digest(value._payload(include_sha256=False)))


def encode_scene_chart(path,*,variant,frames=360,small_focal_marker=False,joint_textures=False):
    """Distinct non-human raster patterns; eight files are not copied aliases."""
    base=path.with_suffix('.seed.mp4')
    encode_chart(base,variant=variant%2,frames=frames)
    cap=cv2.VideoCapture(str(base));images=[]
    yy,xx=np.mgrid[:180,:320]
    while True:
        ok,image=cap.read()
        if not ok:break
        # Added blue geometric marker, kept separate from green descriptor.
        cv2.rectangle(image,(32,43),(102,101),(190,65,30),-1)
        for x,y in ((59,62),(75,62),(67,70),(59,79),(75,79)):
            cv2.circle(image,(x,y),2,(30,20,10),-1)
        if 2<=variant<6:
            # Independent source chart background spatial pattern. This does
            # not prove the real derivative classifier; its typed output is
            # explicitly substituted by scene_derivative_fixture.
            image[120:,:,:]=np.clip(40+(xx[120:]*((variant%3)+1)+yy[120:]*(variant+1))%100,0,255)[:,:,None]
        if small_focal_marker:
            # Actual fixture pixels, not detector/geometry/admission outputs.
            # A distant late-shot mismatch must remain visible after tick 240.
            cv2.rectangle(image,(110,34),(208,115),(45,45,45),-1)
            cv2.rectangle(image,(131,52),(188,99),(52,192,52),-1)
            for x,y in ((152,68),(168,68),(152,85),(168,85)):
                cv2.circle(image,(x,y),2,(22,125,22),-1)
        if joint_textures and (marker:=box_from_chart(image)) is not None:
            x,y,w,h=marker;cx=int(round((x+w/2)*320));cy=int(round((y+h/2)*180))
            for px,py in ((cx-8,cy-8),(cx+8,cy-8),(cx-8,cy+9),(cx+8,cy+9)):
                for dx in (-4,-1,2):
                    for dy in (-4,-1,2):
                        color=(15,90,15) if (dx+dy)%2 else (45,115,45)
                        image[py+dy:py+dy+2,px+dx:px+dx+2]=color
        # A per-file raster tag is not a claim of eight independent real cameras.
        image[-1,-(variant+1):,:]=(variant*20,20,40)
        images.append(image)
    cap.release();base.unlink()
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','320x180','-r','30','-i','pipe:0',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-threads','1',
        '-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-bf','0','-c:a','aac','-y',str(path)]
    result=subprocess.run(cmd,input=np.asarray(images).tobytes(),capture_output=True,timeout=40)
    if result.returncode:raise RuntimeError(result.stderr.decode())


def build_scene_fixture(root,sources,*,boundary=0,authored_camera_unit=None,boundary_look_enabled=True):
    from crosscut.source_look_match import SourceLookAdjustment
    duration=16.;n=8
    mapping=tuple(range(len(sources)));offsets=(1.,)*len(sources);warps=(None,)*len(sources)
    choices=tuple(replace(_choice(i,i,start=2.*i,end=2.*(i+1)),
        source_local_start=2.*i+1,source_local_end=2.*(i+1)+1,
        entry_shot_scale=.4,exit_shot_scale=.4) for i in range(n))
    timeline=VisualTimelinePlan(choices,1.)
    request=SimpleNamespace(sources=sources,reference_source=0,aspect_ratio=AspectRatio.LANDSCAPE,
        source_cache_directory=root/'cache')
    section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,duration,.5,.5,1,4,15.,.4,.1,False,False),))
    with patch('crosscut.pipeline.output_size',return_value=(320,180)):
        bundle=_review_first_group_bundle(timeline=timeline,request=request,eligible_indices=mapping,
            offsets=offsets,timewarps=warps,common_start=0.,section_plan=section,
            phrase_times=tuple(float(i*2) for i in range(n+1)),output=root/'candidate.mp4',
            reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
    # Retain a real non-neutral source face crop, independently of editorial
    # camera, and an actual older dense motion on a DISTANT boundary.
    pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.01,.01,.98,.98)),
        PanCropKeyframe(2.,NormalizedCrop(.01,.01,.98,.98))))
    face=FaceMatchPlan(tuple(FaceMatchSegment(i,i,True,None,.33,.34,1/.98,1/.98,pan,
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
    if authored_camera_unit is not None:
        # A deliberately bad authored chart camera, NOT a successful NN fit.
        # Boundary index 2 above owns incoming shot 3, not shot 2. Removing
        # None on shot 2 used to leave KEEP while the test claimed two edits.
        assert type(authored_camera_unit) is int and 0<=authored_camera_unit<n
        authored=EditorialTransformTrack((
            EditorialTransformKeyframe(0.,1.18,.57,.5),
            EditorialTransformKeyframe(2.,1.18,.57,.5)),
            reason='EXPLICIT_NONHUMAN_AUTHORED_BAD_CAMERA_NOT_NN_EVIDENCE')
        motion=replace(motion,segments=tuple(replace(item,track=authored,
            reasons=(*item.reasons,'explicit_authored_nonhuman_camera'))
            if item.segment_index==authored_camera_unit else item for item in motion.segments))
        # Constant framing is not an animated zoom/pan target event. Existing
        # frame-visible motion counters retain their original exact semantics.
    plan=replace(bundle.render_plan,fps=30.,transitions=transitions,segments=tuple(
        RenderSegment(i,2.*i+1,2.,2.*i,pan_crop=pan,
            look_adjustment=SourceLookAdjustment(brightness=.002+i*.0001),
            editorial_transform=motion.track_for(i)) for i in range(n)))
    # Real full old look fit AND actual renderer/decode admission. The fixture
    # is not positive unless an old applied correction really exists.
    from crosscut.boundary_look_match import _tracks_from_decisions
    from crosscut.source_look_match import SourceLookAdjustment
    look=None
    if boundary_look_enabled:
        look=analyze_boundary_look_match(plan,group_16x9_safe=True)
        original=look
        # Author a small non-neutral recipe, and require REAL matched encode
        # admission. Bounded fixture authoring only; no gate/result is mocked.
        for strength in (.125,.25,.0625,.375):
            decisions=tuple(replace(d,
                applied=d.applied and d.boundary_index==6,
                reason='fixture_authored_brightness_recipe_actual_encode_required',
                outgoing_adjustment=SourceLookAdjustment(brightness=d.outgoing_adjustment.brightness*strength)
                    if d.boundary_index==6 else SourceLookAdjustment(),
                incoming_adjustment=SourceLookAdjustment(brightness=d.incoming_adjustment.brightness*strength)
                    if d.boundary_index==6 else SourceLookAdjustment()) for d in original.decisions)
            look=replace(original,decisions=decisions,
                segment_tracks=_tracks_from_decisions(plan,decisions,original.analysis_config))
            look=verify_boundary_look_match_render_domain(plan,look,
                config=BoundaryLookRenderAdmissionConfig(proxy_width=320,proxy_height=180),
                maximum_verified_boundaries=4,clipping_backoff_enabled=True)
            if look.applied_count: break
        plan=look.apply(plan)
    bundle=replace(bundle,render_plan=plan,face_match_plan=face,editorial_motion_plan=motion,
        boundary_look_match=look,group_transition_plan=replace(bundle.group_transition_plan,transitions=transitions))
    _validate_review_first_group_bundle(bundle,eligible_indices=mapping,offsets=offsets,
        timewarps=warps,common_start=0.,phrase_times=bundle.phrase_times)
    controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=sources,
        source_durations=(18.,)*len(sources),master_start=0.,duration=duration,journal_path=root/'controller.json')
    controller.install_baseline(bundle,plan=plan,authority='deterministic_full_song_visual_selector')
    model=root/'fixture-model.bin';model.write_bytes(b'explicit chart sensor fixture: no ONNX inference')
    return bundle,controller,request,model


def scene_native(session,*,request_source_index,native_frame_index,head_analysis_roll=None):
    row=chart_native(session,request_source_index=request_source_index,native_frame_index=native_frame_index)
    if not row.heads:return row
    # The two chart shapes have similar eye area, so legacy single foreground
    # selection is ambiguous. These are NOT actual YuNet detections of humans.
    points=tuple((x/320,y/180) for x,y in ((59,62),(75,62),(67,70),(59,79),(75,79)))
    second=NativeGroupHead(1,(32/320,43/180,71/320,59/180),points,.99)
    return replace(row,heads=(*row.heads,second),reason='EXPLICIT_TWO_GEOMETRIC_MARKERS_NOT_HUMAN_SCENE')


@contextmanager
def scene_chart_sensors():
    with source_chart_sensors(),patch.object(source,'observe_native_group',side_effect=scene_native):
        yield


@isolated_optional()
def scene_chart_propose(context):
    with scene_chart_sensors():return _REAL_PROPOSE(context)


@isolated_optional()
def scene_chart_final(plan_json,receipt,deadline):
    with scene_chart_sensors():return _REAL_FINAL(plan_json,receipt,deadline)


@isolated_optional()
def scene_chart_selected_replay(context,candidate):
    from crosscut.group_scene_selected_replay import replay_selected_scene
    with scene_chart_sensors():return replay_selected_scene.__wrapped__(context,candidate)


@isolated_optional()
def camera_chart_selected_replay(context,candidate):
    from crosscut.group_scene_selected_replay import replay_selected_scene
    with source_chart_sensors():return replay_selected_scene.__wrapped__(context,candidate)


def camera_only_chart_window(prepared,limit):
    """Explicit legal larger/smaller/larger chart scope, not production policy."""
    return tuple((i,item) for i,item in enumerate(prepared)
        if len(item[0])==3 and item[0][1].eligible_source_index==2)


def authored_global_chart_union(context,mutable_units=(2,6),*,clipped_interior=False):
    """Fixture numeric input, not separate local completions or NN authority.

    Endpoint cameras are fitted from decoded chart geometry. This fixture
    authoring session is separate from the measured execution transaction;
    neither its work nor chart detectors are claimed as production evidence.
    """
    from crosscut.group_scene_owned_program import (freeze_owned_scene_program,
        own_scene_intervals,camera_unit_from_plan,assemble_owned_scene_union)
    from crosscut.group_scene_execution_scope import execution_path_from_union
    from crosscut.group_joint_observations import GroupJointObservationSession
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    program=freeze_owned_scene_program(context,context.domain)
    allocation=own_scene_intervals(program,
        mutable_unit_ids=tuple(program.units[i].unit_id for i in mutable_units),
        maximum_mutable_units=6,maximum_mutable_frames=540)
    original=context.incumbent.render_plan
    original_cameras=tuple(camera_unit_from_plan(program=program,unit_id=unit_id,
        plan=original,domain=context.domain) for unit_id in allocation.mutable_unit_ids)
    baseline=execution_path_from_union(context,context.domain,assemble_owned_scene_union(
        context=context,domain=context.domain,allocation=allocation,cameras=original_cameras))
    segments=list(original.segments)
    with GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=context.deadline,**source.WORK_LIMITS)) as session:
        sensors=sensors_module.SceneSensors(session,source.installed_native_assets(context))
        states=tuple(sensors.bind(s,sensors.discover(s,rolls=(None,),endpoint_scope='whole')[0])
            for s in baseline)
        for index in baseline.mutable_indices:
            value=states[index];left,right=states[index-1],states[index+1]
            original_unit_index=program.units.index(baseline.scoped_units[index])
            if original_unit_index==2:
                # Remove the genuinely non-neutral original boundary-2 camera.
                # Admission must measure the gain; it is not asserted here.
                camera=source._hypothesis(context,value,value.eligible_source_index,None)
                first_cell=baseline.scoped_units[index].role_program['original_shot']['first_segment_index']
                segments[first_cell:first_cell+len(camera.segments)]=camera.segments
                continue
            targets=(scene._target(sensors.frame(left,left.end_output_frame-1)),
                scene._target(sensors.frame(right,right.first_output_frame)))
            first,entry=sensors.mutable_camera_inputs(value,value.first_output_frame)
            last,exit=sensors.mutable_camera_inputs(value,value.end_output_frame-1)
            a,ta,_=source._at(value,value.first_output_frame)
            b,tb,_=source._at(value,value.end_output_frame-1)
            boxes=(source._base_box(scene._native_target(first),a,ta),
                source._base_box(scene._native_target(last),b,tb))
            envelopes=(tuple(source._base_box(box,a,ta) for box in entry),
                tuple(source._base_box(box,b,tb) for box in exit))
            proposal=source.propose_endpoint_focal_face_roi_camera(source_face_boxes=boxes,
                fixed_neighbor_face_boxes=targets,frame_count=len(value.native_frame_indices),
                fps=original.fps,required_endpoint_envelopes=envelopes,maximum_zoom=2.)
            track=proposal.track
            if clipped_interior and index==baseline.mutable_indices[-1]:
                keys=[]
                for time in (0.,.1,.3,.9,1.4,56/30,59/30):
                    old=track.evaluate(time)
                    keys.append(EditorialTransformKeyframe(time,2.,.26,.5) if time==.9 else
                        EditorialTransformKeyframe(time,old.zoom,old.center_x,old.center_y,old.roll_degrees))
                track=EditorialTransformTrack(tuple(keys))
            camera=source._hypothesis(context,value,value.eligible_source_index,track)
            first_cell=baseline.scoped_units[index].role_program['original_shot']['first_segment_index']
            segments[first_cell:first_cell+len(camera.segments)]=camera.segments
    selected=replace(original,segments=tuple(segments))
    cameras=tuple(camera_unit_from_plan(program=program,unit_id=unit_id,
        plan=selected,domain=context.domain) for unit_id in allocation.mutable_unit_ids)
    return assemble_owned_scene_union(context=context,domain=context.domain,
        allocation=allocation,cameras=cameras)


def render_and_decode_chart(testcase,plan,*,capture_ticks=()):
    """Actual encoder/complete decode. Never a human editorial quality gate."""
    render(plan,video_codec='libx264',video_bitrate='1M',
        video_maxrate='2M',video_buffer_size='2M',maximum_runtime_seconds=30.)
    cap=cv2.VideoCapture(str(plan.output));decoded=0;captured={}
    try:
        while True:
            ok,pixels=cap.read()
            if not ok:break
            testcase.assertEqual(pixels.shape[:2],(180,320))
            if decoded in capture_ticks:captured[decoded]=pixels.copy()
            decoded+=1
    finally:cap.release()
    testcase.assertEqual(decoded,plan.output_frame_count)
    testcase.assertEqual(tuple(sorted(captured)),tuple(sorted(capture_ticks)))
    return captured


class ScenePureContractTests(unittest.TestCase):
    def test_obligation_observation_failure_and_malformed_contract_remain_distinct(self):
        from crosscut.group_scene_hypothesis_outcomes import SceneRoleObservationRejection,SceneHypothesisRejection
        role=sensors_module.GroupOutputRole.FOCAL_HEAD
        raw=dict(heads=[dict(observation_id=0)],people=[])
        with self.assertRaises(SceneRoleObservationRejection):
            sensors_module.scene_role_obligations(raw,role,0,required_part_names=('left_arm',))
        for invalid,names,head in ((dict(heads=[dict(observation_id=0),dict(observation_id=0)],people=[]),('left_arm',),0),
                (raw,('missing_arm',),0),(raw,('left_arm',),False)):
            with self.subTest(raw=invalid,names=names,head=head):
                try:sensors_module.scene_role_obligations(invalid,role,head,required_part_names=names)
                except ValueError as exc:self.assertNotIsInstance(exc,SceneHypothesisRejection)
                else:self.fail('malformed original duty/raw ID must remain a contract failure')

    def test_edge_contract_failures_propagate_from_frame_endpoint_and_actual_composition_projection(self):
        role=sensors_module.GroupOutputRole
        left=SimpleNamespace(key='a'*64,role=role.COMPOSITION,first_output_frame=0,end_output_frame=8,
            output_fps=30.,segments=(SimpleNamespace(source_index=0),),source_clock_identity='b'*64,
            native_frame_indices=tuple(range(8)))
        right=SimpleNamespace(key='c'*64,role=role.FOCAL_HEAD,first_output_frame=8,end_output_frame=16,
            output_fps=30.)
        def row(state,tick):
            return dict(output_tick=tick,scene_geometry=dict(occupancy=.2,attention_xy=[.5,.5],
                subject_height_quantiles=[.2,.3,.4],horizontal_span=.4),
                # Explicit corrupt raw source, detected by the ACTUAL motion function.
                native_observation=dict(request_source_index=9,source_clock_identity='b'*64,
                    native_frame_index=tick))
        for stage in ('frame','endpoint','composition'):
            def frame(state,tick):
                if stage=='frame':raise ValueError('EXPLICIT_BAD_FROZEN_FRAME_CONTRACT')
                return row(state,tick)
            def endpoint(state,*,outgoing):
                if stage=='endpoint':raise ValueError('EXPLICIT_BAD_ENDPOINT_CONTRACT')
                return dict(output_ticks=[state.end_output_frame-4,state.end_output_frame-1]
                    if outgoing else [state.first_output_frame,state.first_output_frame+3])
            with self.subTest(stage=stage),self.assertRaises(ValueError) as caught:
                sensors_module.compare_scene_edge(SimpleNamespace(frame=frame,endpoint=endpoint),left,right)
            self.assertNotIsInstance(caught.exception,sensors_module.SceneHypothesisRejection)
        from crosscut.group_scene_hypothesis_outcomes import SceneHypothesisRejection
        sensor=SimpleNamespace(frame=lambda *a:(_ for _ in ()).throw(SceneHypothesisRejection('measured support missing')))
        measured=sensors_module.compare_scene_edge(sensor,left,right)
        self.assertFalse(measured['available'])
        self.assertEqual(measured['reason'],'measured support missing')

    def test_unknown_and_same_family_are_not_independent(self):
        result=scene_derivative_fixture(3)
        binding={'audit':result.to_dict()}
        self.assertEqual(scene.relation(binding,0,1),DERIVATIVE_DISTINCT)
        binding['audit']['pairs']=[]
        self.assertEqual(scene.relation(binding,0,1),'uncertain')
        binding['audit']['family_ids']=[0,0,1]
        self.assertEqual(scene.relation(binding,0,1),'confirmed_same_camera_derivative')

    def test_known_components_cannot_be_lost_or_bought_by_better_others(self):
        before={'metrics':{'arm_predicted_0':.2,'occupancy':.1}}
        self.assertFalse(sensors_module.preserves_known(before,{'available':True,'metrics':{'occupancy':0.}}))
        self.assertFalse(sensors_module.preserves_known(before,{'available':True,'metrics':{'occupancy':0.,'arm_predicted_0':.3}}))
        self.assertTrue(sensors_module.preserves_known(before,{'available':True,'metrics':{'occupancy':.1,'arm_predicted_0':.1}}))

    def test_fallback_uses_same_distinct_pair_invariant(self):
        binding={'audit':scene_derivative_fixture(4).to_dict()}
        scene.validate_bound_derivative_path(binding,[0,1,2,3])
        for indices in ([0,1,1,3],[0,1,8],[0,False,2]):
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                scene.validate_bound_derivative_path(binding,indices)

    def test_incomplete_audit_cannot_be_relabelled_distinct(self):
        for field in ('production_ready','clock_contract_complete'):
            audit=scene_derivative_fixture(4).to_dict();audit[field]=False
            payload=dict(audit);payload.pop('audit_sha256');audit['audit_sha256']=family.digest(payload)
            with self.subTest(field=field),self.assertRaises(ValueError):
                scene.validate_bound_derivative_path({'audit':audit},[0,1,2,3])

    def test_rehashed_unknown_pair_is_not_verified_independence(self):
        audit=scene_derivative_fixture(4).to_dict();audit['pairs'][0]['status']='uncertain'
        payload=dict(audit);payload.pop('audit_sha256');audit['audit_sha256']=family.digest(payload)
        with self.assertRaises(ValueError):scene.validate_bound_derivative_path({'audit':audit},[0,1,2,3])


class SceneLocalDenseRejectionIdentityTests(unittest.TestCase):
    """NONMEDIA dependency identity tests, not manufactured native admission."""
    def fixture(self):
        from fractions import Fraction
        from crosscut.group_source_domain import GroupSourceDomain
        from crosscut.render_frame_schedule import VerifiedRenderFrameClock
        from crosscut.group_joint_source_camera import GroupSourceCameraState,JointGroupWorkBudget,GroupOutputRole
        storage=tempfile.TemporaryDirectory();self.addCleanup(storage.cleanup)
        path=Path(storage.name)/'NONMEDIA-scope-input.bin';path.write_bytes(b'explicit nonmedia identity control')
        domain=GroupSourceDomain((path,),(0,),(0.,),(None,),((),),
            (VerifiedRenderFrameClock(Fraction(30),8,'a'*64,'synthetic_nonmedia'),))
        state=GroupSourceCameraState(1,0,8,0,(RenderSegment(0,0.,8/30,0.),),tuple(range(8)),
            GroupOutputRole.FOCAL_HEAD,'a'*64,output_size=(320,180))
        h=dict(span_key=list(source.NativeFamilySensors.span_key(state)),analysis_roll=None,
            endpoint_scope='whole',anchor_output_ticks=[0,3,4,7],anchor_head_ids=[0]*4)
        state=replace(state,editorial_contract_sha256=family.digest(h))
        sensor=sensors_module.SceneSensors.__new__(sensors_module.SceneSensors)
        sensor.session=SimpleNamespace(domain=domain,closed=False,models={},model_adapters={},
            budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.))
        sensor.assets={};sensor.hypotheses={state.editorial_contract_sha256:h}
        sensor.original_duty_program=dict(program_sha256='d'*64,units=[dict(
            first_output_frame=0,end_output_frame=8,role=state.role.value,required_parts=['left_arm'])])
        sensor.raw_rows={(0,0,None):SimpleNamespace(sha256='c'*64)};sensor.state_dense_rejections={}
        sensor._remember_state_dense_rejection(state,[((0,0,None),'c'*64)],
            sensors_module.SceneHypothesisRejection('EXPLICIT_NONMEDIA_CACHE_DEPENDENCY_CONTROL'))
        return sensor,state

    def test_local_failure_is_reused_only_with_unchanged_camera_duty_and_native_observation(self):
        for change in ('none','camera','duty','native','hypothesis','closed','budget_owner'):
            sensor,state=self.fixture()
            if change=='camera':state=replace(state,segments=(replace(state.segments[0],
                editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),))),))
            elif change=='duty':sensor.original_duty_program['units'][0]['required_parts'].append('right_arm')
            elif change=='native':sensor.raw_rows[0,0,None]=SimpleNamespace(sha256='e'*64)
            elif change=='hypothesis':sensor.hypotheses[state.editorial_contract_sha256]['anchor_head_ids'][0]=1
            elif change=='closed':sensor.session.closed=True
            elif change=='budget_owner':sensor.session.budget=SimpleNamespace()
            # The legacy mutable selector is path[1:-1]; no observations run.
            path=(None,state,None)
            with self.subTest(change=change):
                if change in ('hypothesis','closed','budget_owner'):
                    with self.assertRaises(ValueError):sensor.cached_state_dense_rejection(path)
                else:
                    result=sensor.cached_state_dense_rejection(path)
                    self.assertEqual(result is not None,change=='none')
                    if result is not None:self.assertFalse(result['new_native_budget_granted'])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local encode/decode tools required')
class SceneSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.paths=tuple(cls.root/f'scene-{i}.mp4' for i in range(8))
        for i,path in enumerate(cls.paths):encode_scene_chart(path,variant=i,frames=540)
        cls.bundle,cls.controller,cls.request,cls.model=build_scene_fixture(cls.root,cls.paths)
        with optional_native_scope(monotonic()+600),patch.object(family,'installed_pose_model',return_value=cls.model):
            cls.context=family.freeze_context(controller=cls.controller,request=cls.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,source_derivatives=scene_derivative_fixture())
        with optional_native_scope(cls.context.deadline),scene_chart_sensors():
            cls.result=scene.propose_scene_changes(cls.context)
        value=(dict(error=cls.result.error,evidence=family.plain(cls.result.evidence))
            if isinstance(cls.result,family.PerformerFamilyRejection) else cls.result.to_dict())
        destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
        if destination:
            target=Path(destination);target.mkdir(parents=True,exist_ok=True)
            (target/'scene-development-result.json').write_text(json.dumps(value,indent=2))
            (target/'scene-baseline.json').write_text(json.dumps(cls.bundle.to_dict(),indent=2))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_source_scene_normal_family_positive(self):
        if isinstance(self.result,family.PerformerFamilyRejection):self.fail(self.result.error)
        family.validate_bundle_binding(self.result)
        receipt=family.plain(self.result.performer_family_review)
        self.assertEqual(receipt['source_camera_refresh']['contract'],sensors_module.DENSE_CONTRACT)
        self.assertFalse(receipt['quality_approved'])
        self.assertFalse(receipt['source_camera_refresh']['body_phase_certified'])
        self.assertTrue(any(a.source_index!=b.source_index for a,b in zip(self.bundle.render_plan.segments,self.result.render_plan.segments)))
        self.assertTrue(all(s.look_adjustment is not None for s in self.bundle.render_plan.segments))
        self.assertGreater(self.bundle.boundary_look_match.applied_count,0)
        self.assertGreater(self.result.boundary_look_match.applied_count,0)
        for row in receipt['source_camera_refresh']['mutable_shots']:
            self.assertTrue(all(len(v['scene_heads'])==2 for v in row['rows']))
        generated=receipt['search']['generated_numeric_states']
        for slot in {v['slot'] for v in generated}:
            self.assertEqual({v['source'] for v in generated if v['slot']==slot},set(range(8)))


    def test_all_alternatives_observed_before_single_foreground_gate(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        receipt=family.plain(self.result.performer_family_review)
        dense=receipt['source_camera_refresh']
        rows=dense['scene_discovery']['native_rows']
        self.assertEqual({r['request_source_index'] for r in rows},set(range(8)))
        self.assertTrue(any(not b['focal_assignment_unique'] for b in receipt['search']['baseline_components'].values()))

    def test_same_source_camera_reaches_normal_completion_parent_and_final(self):
        # Restrict numeric proposals to the original source, not the measured
        # geometry/gain/dense/descriptor/look gates. The actual normal owner
        # must find a real camera edit; chart model outputs are explicit fixtures.
        original_hypothesis=source._hypothesis
        def same_source(context,incumbent,index,track=None):
            if index!=incumbent.eligible_source_index:
                raise ValueError('EXPLICIT_TEST_CAMERA_ONLY_NUMERIC_SOURCE_DOMAIN')
            return original_hypothesis(context,incumbent,index,track)
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),source_chart_sensors(), \
                patch.object(scene,'time_stratified_window_frontier',side_effect=camera_only_chart_window), \
                patch.object(source,'_hypothesis',side_effect=same_source):
            candidate=scene.propose_scene_changes(context)
            if isinstance(candidate,family.PerformerFamilyRejection):
                evidence=family.plain(candidate.evidence)
                summary=dict(error=candidate.error,complete_alternative_errors=sorted({
                    row.get('error','') for row in evidence.get('candidate_alternatives',())}),
                    graph_counts=[(row['window'],row.get('path_count')) for row in evidence.get('attempts',())
                        if row.get('stage')=='scene_graph'],
                    edge_prefix=[dict(baseline={str(key):value.get('metrics') for key,value
                            in graph.get('baseline_components',{}).items()},
                        edges=[{key:edge.get(key) for key in ('available','metrics','reason','source_relation')}
                            for edge in graph.get('edge_observations',())[:8]])
                        for graph in evidence.get('attempts',()) if graph.get('stage')=='scene_graph'][:1])
                self.fail(json.dumps(summary))
            receipt=family.plain(candidate.performer_family_review)
            self.assertEqual([s.source_index for s in candidate.render_plan.segments],
                [s.source_index for s in context.incumbent.render_plan.segments])
            operations=receipt['selected_scene_program']['edit_operations']
            self.assertTrue(operations['camera_only_program'])
            self.assertTrue(operations['distinct_camera_transition_count_not_inferred'])
            self.assertFalse(operations['request_budget_restarted'])
            self.assertTrue(any(row['operation']=='CAMERA' for row in operations['actual_shot_operations']))
            family.validate_bundle_binding(candidate)
            from crosscut.group_scene_selected_replay import replay_selected_scene
            with patch.object(__import__('crosscut.group_scene_selected_replay',fromlist=['replay_selected_scene']),
                    'replay_selected_scene',camera_chart_selected_replay):
                proof=source.validate_parent(candidate,context)
            self.assertTrue(proof['fresh_complete_family_equal'])
            self.assertTrue(proof['source_and_camera_edit_operations_independently_rederived'])
            self.assertFalse(proof['quality_approved'])
            render_and_decode_chart(self,candidate.render_plan)
            metrics=_REAL_FINAL(candidate.render_plan.to_json(),receipt,monotonic()+90.)
            self.assertTrue(family.qa_metrics_valid(metrics))
            self.assertFalse(metrics['quality_approved'])

    def test_same_source_keep_does_not_reach_dense_or_restart_work(self):
        original_hypothesis=source._hypothesis
        def keep(context,incumbent,index,track=None):
            if index!=incumbent.eligible_source_index:
                raise ValueError('EXPLICIT_TEST_KEEP_NUMERIC_SOURCE_DOMAIN')
            return original_hypothesis(context,incumbent,index,None)
        context=replace(self.context,deadline=monotonic()+90.)
        with optional_native_scope(context.deadline),source_chart_sensors(), \
                patch.object(scene,'time_stratified_window_frontier',side_effect=camera_only_chart_window), \
                patch.object(source,'_hypothesis',side_effect=keep):
            result=scene.propose_scene_changes(context)
        self.assertIsInstance(result,family.PerformerFamilyRejection)
        evidence=family.plain(result.evidence)
        self.assertEqual(evidence['candidate_alternatives'],[])
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        self.assertFalse(evidence['fallback_work_restart_allowed'])

    def test_old_source_only_eligibility_starves_the_valid_camera_control(self):
        # Exact old eligibility criterion, explicitly a local counterfactual.
        # Use the SAME numeric/sensor/geometry scope as the positive case above.
        original_hypothesis=source._hypothesis
        def same_source(context,incumbent,index,track=None):
            if index!=incumbent.eligible_source_index:
                raise ValueError('EXPLICIT_TEST_CAMERA_ONLY_NUMERIC_SOURCE_DOMAIN')
            return original_hypothesis(context,incumbent,index,track)
        def old_criterion(original,proposed):
            return any(a.eligible_source_index!=b.eligible_source_index
                for a,b in zip(original,proposed,strict=True))
        context=replace(self.context,deadline=monotonic()+90.)
        with optional_native_scope(context.deadline),source_chart_sensors(), \
                patch.object(scene,'time_stratified_window_frontier',side_effect=camera_only_chart_window), \
                patch.object(source,'_hypothesis',side_effect=same_source), \
                patch.object(scene,'path_has_numeric_edit',side_effect=old_criterion):
            result=scene.propose_scene_changes(context)
        self.assertIsInstance(result,family.PerformerFamilyRejection)
        evidence=family.plain(result.evidence)
        self.assertEqual(evidence['candidate_alternatives'],[])
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)

    def test_camera_only_endpoint_match_cannot_hide_clipped_interior(self):
        original_hypothesis=source._hypothesis
        def clipped(context,incumbent,index,track=None):
            if index!=incumbent.eligible_source_index:
                raise ValueError('EXPLICIT_TEST_CAMERA_ONLY_NUMERIC_SOURCE_DOMAIN')
            if track is not None:
                # Keep all four cut-anchor frames unchanged. The actual
                # renderer camera has a bad interior, not a patched dense gate.
                times=(0.,.1,.3,.9,1.4,56/30,59/30)
                keys=[]
                for time in times:
                    old=track.evaluate(time)
                    keys.append(EditorialTransformKeyframe(time,2.,.26,.5) if time==.9 else
                        EditorialTransformKeyframe(time,old.zoom,old.center_x,old.center_y,old.roll_degrees))
                track=EditorialTransformTrack(tuple(keys))
            return original_hypothesis(context,incumbent,index,track)
        context=replace(self.context,deadline=monotonic()+90.)
        with optional_native_scope(context.deadline),source_chart_sensors(), \
                patch.object(scene,'time_stratified_window_frontier',side_effect=camera_only_chart_window), \
                patch.object(source,'_hypothesis',side_effect=clipped):
            result=scene.propose_scene_changes(context)
        self.assertIsInstance(result,family.PerformerFamilyRejection)
        evidence=family.plain(result.evidence)
        self.assertGreater(evidence['native_work']['counts']['dense_paths'],0,
            'negative control never reached interior checking: '+json.dumps({
                'alternatives':[row.get('error') for row in evidence['candidate_alternatives']],
                'graph_counts':[(row['window'],row.get('path_count')) for row in evidence['attempts']
                    if row.get('stage')=='scene_graph']}))
        self.assertTrue(any('clip' in row.get('error','') or 'crop' in row.get('error','')
            for row in evidence['candidate_alternatives']))
        self.assertFalse(evidence['fallback_work_restart_allowed'])

    def test_normal_owner_freezes_original_program_before_any_donor_native_query(self):
        from crosscut.group_joint_source_camera import joint_state_from_render_plan
        from crosscut.group_joint_state_factory import fixed_neighbor_state_from_render_plan
        from crosscut.group_scene_original_duties import duty_at
        original_raw=sensors_module.SceneSensors.raw
        donors=[]
        def observe(sensor,state,tick,roll):
            h=sensor.hypotheses.get(state.editorial_contract_sha256)
            scope='whole' if h is None else h['endpoint_scope']
            builder=joint_state_from_render_plan if scope=='whole' else fixed_neighbor_state_from_render_plan
            incumbent=builder(plan=self.context.incumbent.render_plan,domain=self.context.domain,
                slot=state.slot,first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,
                role=state.role)
            if replace(state,editorial_contract_sha256=None).key!=incumbent.key:
                program=getattr(sensor,'original_duty_program',None)
                self.assertIsNotNone(program,'donor native work preceded original duty freeze')
                ref=duty_at(program,state,tick)
                donors.append(ref['program_sha256'])
            return original_raw(sensor,state,tick,roll)
        with optional_native_scope(monotonic()+120),scene_chart_sensors(), \
                patch.object(sensors_module.SceneSensors,'raw',new=observe):
            result=scene.propose_scene_changes(self.context)
        self.assertNotIsInstance(result,family.PerformerFamilyRejection,getattr(result,'error',None))
        self.assertTrue(donors,'normal source-changing graph did not execute any donor work')

    def test_binding_rejects_ipc_source_clock_static_and_flag_tampering(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        for field in ('source','clock','static','quality','native_pixels'):
            with self.subTest(field=field):
                if field in ('source','clock','static'):
                    plan=self.result.render_plan;segments=list(plan.segments)
                    i=family.plain(self.result.performer_family_review)['changed_cells'][0]
                    updates={'source':{'source_index':(segments[i].source_index+1)%8},
                        'clock':{'source_start':segments[i].source_start+1/30},
                        'static':{'look_adjustment':None}}[field]
                    segments[i]=replace(segments[i],**updates)
                    candidate=replace(self.result,render_plan=replace(plan,segments=tuple(segments)))
                else:
                    receipt=deepcopy(family.plain(self.result.performer_family_review))
                    if field=='quality':receipt['quality_approved']=True
                    else:receipt['source_camera_refresh']['mutable_shots'][0]['rows'][0]['native_observation']['native_pixel_sha256']='0'*64
                    candidate=replace(self.result,performer_family_review=receipt)
                with self.assertRaises(ValueError):family.validate_bundle_binding(candidate)

    def test_dense_consumer_cannot_drop_background_head_or_invent_head_extent(self):
        original=family.plain(self.result.performer_family_review)['source_camera_refresh']
        for field in ('head_drop','visibility','anatomy','hypothesis','duplicate',
                'head_duplicate','visibility_nan','headroom_nan','missing_tick','duplicate_tick',
                'row_role','legacy_contract'):
            with self.subTest(field=field):
                dense=deepcopy(original);row=dense['mutable_shots'][0]['rows'][0]
                if field=='head_drop':row['scene_heads']=row['scene_heads'][:1]
                elif field=='visibility':next(h for h in row['scene_heads'] if h['required_by_role'])['visible_fraction']=.5
                elif field=='anatomy':row['scene_heads'][0]['physical_head_extent']=[0,0,1,1]
                elif field=='hypothesis':row['scene_hypothesis_sha256']='f'*64
                elif field=='duplicate':dense['source_derivative_binding']['audit']['pairs']=[]
                elif field=='head_duplicate':row['scene_heads'].append(deepcopy(row['scene_heads'][0]))
                elif field=='visibility_nan':row['scene_heads'][0]['visible_fraction']=float('nan')
                elif field=='headroom_nan':next(h for h in row['scene_heads']
                    if h['observation_id']==row['selected_frame_local_head_id'])['headroom_visible_fraction']=float('nan')
                elif field=='missing_tick':dense['mutable_shots'][0]['rows'].pop()
                elif field=='duplicate_tick':dense['mutable_shots'][0]['rows'][1]['output_tick']=row['output_tick']
                elif field=='row_role':row['role']='formation'
                elif field=='legacy_contract':dense['contract']='source_native_scene_whole_camera_refresh_v2'
                with self.assertRaises(ValueError):sensors_module.validate_scene_dense(dense)

    def test_fixed_subcell_neighbor_complete_refresh_and_final_native_replay(self):
        # Explicit chart cut-time fixture: topology/clock/producer behavior,
        # not a human cut detector or actual professional editing approval.
        with optional_native_scope(monotonic()+120),scene_chart_sensors(), \
                patch.object(family,'installed_pose_model',return_value=self.model):
            context=family.freeze_context(controller=self.controller,request=self.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((2.,),(),(),(),(),(),(),()),sync_scores=(.93,.91)*4,
                source_derivatives=scene_derivative_fixture())
            # Allocate the first valid partial-neighbor window explicitly in
            # this regression; normal scheduler behavior is separately tested.
            with patch.object(scene,'time_stratified_window_frontier',
                    side_effect=lambda prepared,limit:tuple(enumerate(prepared[:1]))):
                candidate=scene.propose_scene_changes(context)
            self.assertNotIsInstance(candidate,family.PerformerFamilyRejection,
                getattr(candidate,'error',None))
            family.validate_bundle_binding(candidate)
            family.validate_serialized_metadata(candidate.to_dict())
            receipt=family.plain(candidate.performer_family_review)
            self.assertEqual(receipt['source_camera_refresh']['window_bounds'][0],[30,60])
            self.assertEqual(len(candidate.render_plan.segments),len(self.bundle.render_plan.segments))
            self.assertEqual(candidate.render_plan.segments[0],self.bundle.render_plan.segments[0])
            metrics=_REAL_FINAL(candidate.render_plan.to_json(),receipt,monotonic()+90)
            self.assertTrue(family.qa_metrics_valid(metrics))
            self.assertFalse(metrics['quality_approved'])

    def test_internal_native_epoch_keeps_numeric_sampling_but_is_not_a_full_family(self):
        # The requested complete epoch owner is NOT implemented. Assert the
        # exact retained boundary rather than dressing a numeric split as a pass.
        from crosscut.group_interval_materialization import split_group_render_intervals
        from crosscut.render_boundary_clock import native_frame_clocks_for_output
        domain=replace(self.context.domain,scene_cut_times=((),(4.,),(),(),(),(),(),()))
        plan=self.bundle.render_plan
        scaffold=split_group_render_intervals(plan=plan,domain=domain,split_output_frames=(90,))
        before=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(plan.output_frame_count))
        after=native_frame_clocks_for_output(scaffold.plan,domain.clocks_by_request,range(plan.output_frame_count))
        self.assertEqual([(r.source_index,r.native_frame_index) for r in before.values()],
            [(r.source_index,r.native_frame_index) for r in after.values()])
        self.assertFalse(scaffold.to_dict()['full_bundle_created'])
        self.assertIs(scaffold.plan.segments[-1],plan.segments[-1])
        from crosscut.group_joint_source_camera import joint_state_from_render_plan,GroupOutputRole
        with self.assertRaises(ValueError):joint_state_from_render_plan(plan=plan,domain=domain,
            slot=1,first_output_frame=60,end_output_frame=90,role=GroupOutputRole.FOCAL_HEAD)

    def test_real_native_multihead_observation_does_not_require_dominant_or_arms(self):
        from crosscut.group_joint_observations import GroupJointObservationSession
        from crosscut.group_joint_source_camera import JointGroupWorkBudget,GroupOutputRole
        shots=source.visible_group_shots(self.bundle.render_plan,self.context.domain)[:4]
        states=source._states(self.bundle.render_plan,self.context.domain,shots,source._shot_roles(self.context,shots))
        with optional_native_scope(monotonic()+90),scene_chart_sensors():
            session=GroupJointObservationSession(domain=self.context.domain,
                budget=JointGroupWorkBudget(absolute_deadline=monotonic()+90,**source.WORK_LIMITS))
            try:
                sensors=sensors_module.SceneSensors(session,source.installed_native_assets(self.context))
                raw=sensors.raw(states[0],0,None)
                with self.assertRaises(ValueError):source._dominant(raw,states[0].role)
                hypotheses=sensors.discover(states[0],rolls=(None,))
                self.assertGreaterEqual(len(hypotheses),2)
                bound=sensors.bind(states[0],hypotheses[0]);row=sensors.frame(bound,0)
                self.assertEqual(len(row['scene_heads']),2)
                arm_state=replace(states[0],role=GroupOutputRole.FOCAL_HEAD_ARMS)
                arm_h=sensors.discover(arm_state,rolls=(None,))
                with self.assertRaisesRegex(ValueError,'required arms'):
                    sensors.frame(sensors.bind(arm_state,arm_h[0]),0)
            finally:session.close()


    def test_budget_failure_retains_raw_eight_source_evidence_without_candidate(self):
        limits=dict(source.WORK_LIMITS,maximum_states=1)
        with optional_native_scope(monotonic()+90),scene_chart_sensors(),patch.object(source,'WORK_LIMITS',limits):
            value=scene.propose_scene_changes(self.context)
        self.assertIsInstance(value,family.PerformerFamilyRejection)
        evidence=family.plain(value.evidence)
        self.assertEqual({r['request_source_index'] for r in evidence['scene_observations']['native_rows']},set(range(8)))
        self.assertLessEqual(evidence['native_work']['counts']['states'],1)
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        self.assertIs(self.controller.incumbent,self.bundle)

    def test_failed_fixed_geometry_skips_original_mutable_native_work(self):
        # Explicit NONHUMAN failed incident endpoint. The unchanged endpoint
        # cannot be repaired by any donor; do not spend the entire mutable
        # original lifecycle before learning that fact. Decoding/discovery and
        # original duty construction stay real and use ONE unchanged budget.
        original=sensors_module.SceneSensors.discover
        mutable=[];fixed=[];budgets=set()
        def discover(sensor,state,**kwargs):
            budgets.add(id(sensor.session.budget))
            scope=kwargs.get('endpoint_scope','whole')
            (mutable if scope=='whole' else fixed).append(
                (state.first_output_frame,state.end_output_frame,scope))
            return original(sensor,state,**kwargs)
        def unavailable(sensor,state,*,outgoing):
            raise ValueError('EXPLICIT_NONHUMAN_FIXED_REQUIRED_GEOMETRY_UNAVAILABLE')
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),scene_chart_sensors(), \
                patch.object(sensors_module.SceneSensors,'discover',discover), \
                patch.object(sensors_module.SceneSensors,'endpoint',unavailable):
            result=scene.propose_scene_changes(context)
        self.assertIsInstance(result,family.PerformerFamilyRejection)
        self.assertTrue(fixed)
        self.assertEqual(mutable,[],
            'failed unchanged endpoints still triggered original mutable sensing')
        self.assertEqual(len(budgets),1)
        evidence=family.plain(result.evidence)
        self.assertEqual(evidence['native_work']['counts']['states'],0)
        self.assertEqual(evidence['native_work']['counts']['edges'],0)
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        programs=evidence['scene_observations']['original_programme_records']['native_duty_programmes']
        self.assertTrue(programs,'fixed preflight erased its original duty evidence')
        self.assertTrue(all(len(program['units'])==1 for program in programs.values()))
        self.assertTrue(all(not program['donor_observations_define_duties']
            for program in programs.values()))
        self.assertFalse(evidence['quality_approved'])
        self.assertIs(self.controller.incumbent,self.bundle)

    def test_failed_right_preflight_keeps_both_original_units_without_mutable_query(self):
        original_discover=sensors_module.SceneSensors.discover
        original_endpoint=sensors_module.SceneSensors.endpoint
        mutable=[];right_failures=[]
        def discover(sensor,state,**kwargs):
            if kwargs.get('endpoint_scope','whole')=='whole':mutable.append(state.key)
            return original_discover(sensor,state,**kwargs)
        def endpoint(sensor,state,*,outgoing):
            if outgoing:return original_endpoint(sensor,state,outgoing=outgoing)
            right_failures.append(state.key)
            raise ValueError('EXPLICIT_NONHUMAN_RIGHT_FIXED_ENDPOINT_UNAVAILABLE')
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),scene_chart_sensors(), \
                patch.object(sensors_module.SceneSensors,'discover',discover), \
                patch.object(sensors_module.SceneSensors,'endpoint',endpoint):
            result=scene.propose_scene_changes(context)
        self.assertIsInstance(result,family.PerformerFamilyRejection)
        self.assertTrue(right_failures,'right-side control was never reached')
        self.assertEqual(mutable,[])
        evidence=family.plain(result.evidence)
        records=evidence['scene_observations']['original_programme_records']
        programs=records['native_duty_programmes']
        self.assertTrue(all(len(program['units'])==1 for program in programs.values()))
        scopes={branch['original_hypothesis']['endpoint_scope']
            for program in programs.values() for branch in program['units'][0]['branches']}
        self.assertEqual(scopes,{'entry','exit'})
        self.assertIn(records['active_native_duty_programme_sha256'],programs)
        self.assertEqual(evidence['native_work']['counts']['edges'],0)
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        self.assertFalse(evidence['quality_approved'])

    def test_normal_numeric_paths_bind_explicit_global_original_units(self):
        from crosscut.group_scene_owned_program import freeze_owned_scene_program
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        receipt=family.plain(self.result.performer_family_review)
        search=receipt['search'];programme=freeze_owned_scene_program(self.context,self.context.domain)
        allocation=search['explicit_global_numeric_ownership']
        bounds={tuple(interval) for interval in allocation['mutable_output_intervals']}
        expected=tuple(unit.unit_id for unit in programme.units
            if (unit.first_output_frame,unit.end_output_frame) in bounds)
        self.assertEqual(tuple(allocation['mutable_unit_ids']),expected)
        self.assertEqual(allocation['original_program_sha256'],programme.program_sha256)
        self.assertFalse(allocation['positional_fixed_endpoints_assumed'])
        self.assertFalse(search['global_search_and_completion_owner_implemented'])
        selected=receipt['selected_scene_program']
        self.assertEqual(search['global_numeric_union_plan_sha256'],family.digest(selected['numeric_plan']))
        self.assertEqual(search['global_numeric_union_dependency_sha256'],
            selected['dependency_support']['support_sha256'])
        self.assertEqual(len(search['global_numeric_camera_sha256']),len(expected))
        self.assertFalse(receipt['quality_approved'])

    def test_fixed_preflight_versions_do_not_change_complete_original_duties(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        family.validate_bundle_binding(self.result)
        receipt=family.plain(self.result.performer_family_review)
        search=receipt['search'];dense=receipt['source_camera_refresh']
        complete=dense['original_duty_program']
        refs=search['fixed_preflight_programme_sha256']
        self.assertEqual(len(refs),2)
        self.assertTrue(search['fixed_endpoints_before_original_mutable_observations'])
        programs=dense['scene_discovery']['original_programme_records']['native_duty_programmes']
        for index,ref in zip((0,-1),refs,strict=True):
            self.assertEqual(len(programs[ref]['units']),1)
            self.assertEqual(programs[ref]['units'][0],complete['units'][index])
        self.assertIn(complete['program_sha256'],programs)
        self.assertGreater(len(complete['units']),2)
        for span in dense['mutable_shots']:
            for row in span['rows']:
                self.assertEqual(row['original_duty_reference']['program_sha256'],
                    complete['program_sha256'])

    def test_one_unknown_mutable_window_does_not_end_later_complete_search(self):
        # Explicit non-human missing-hypothesis fixture. There is still work
        # budget: this is NOT the state-budget exhaustion condition.
        original=sensors_module.SceneSensors.discover
        blocked=[]
        later=[]
        def discover(sensor,state,**kwargs):
            if kwargs.get('endpoint_scope','whole')=='whole':
                bounds=(state.first_output_frame,state.end_output_frame)
                if not blocked:blocked.append(bounds)
                if bounds==blocked[0]:return ()
                later.append(bounds)
            return original(sensor,state,**kwargs)
        with optional_native_scope(monotonic()+120),scene_chart_sensors(), \
                patch.object(sensors_module.SceneSensors,'discover',discover):
            candidate=scene.propose_scene_changes(self.context)
        self.assertTrue(blocked and later)
        self.assertNotIsInstance(candidate,family.PerformerFamilyRejection,
            getattr(candidate,'error',None))
        family.validate_bundle_binding(candidate)
        self.assertFalse(family.plain(candidate.performer_family_review)['quality_approved'])

    def test_initial_fixed_probe_batch_cannot_hide_later_complete_window_with_budget_left(self):
        # Explicit missing fixed hypothesis, not an actual human-source failure.
        # Decode/refresh/assembly stay real; all attempts share ONE native budget.
        original=sensors_module.SceneSensors.discover
        blocked=[];later=[];budgets=set()
        def discover(sensor,state,**kwargs):
            budgets.add(id(sensor.session.budget))
            if kwargs.get('endpoint_scope','whole')=='exit':
                bounds=(state.first_output_frame,state.end_output_frame)
                if not blocked:blocked.append(bounds)
                if bounds==blocked[0]:return ()
                later.append(bounds)
            return original(sensor,state,**kwargs)
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),scene_chart_sensors(), \
                patch.object(scene,'MAX_FIXED_WINDOW_PROBES',1), \
                patch.object(sensors_module.SceneSensors,'discover',discover):
            candidate=scene.propose_scene_changes(context)
        self.assertTrue(blocked and later)
        self.assertEqual(len(budgets),1)
        self.assertNotIsInstance(candidate,family.PerformerFamilyRejection,getattr(candidate,'error',None))
        family.validate_bundle_binding(candidate)
        receipt=family.plain(candidate.performer_family_review)
        self.assertEqual(receipt['search']['fixed_probe_batch_size'],1)
        self.assertTrue(receipt['search']['fixed_probe_batches_share_native_session'])
        self.assertLessEqual(receipt['native_work']['counts']['model_calls'],source.WORK_LIMITS['maximum_model_calls'])
        self.assertFalse(receipt['quality_approved'])

    def test_spent_first_graph_does_not_hide_later_source_camera_window(self):
        # NONHUMAN edge-unavailability control. Unlike the missing-hypothesis
        # test above, the first graph actually spends its full state frontier.
        # Real chart decoding, pixel paths, family refresh and look encode stay
        # active. This is allocation evidence, not actual IVE media quality.
        real_discover=sensors_module.SceneSensors.discover
        real_edge=scene.compare_scene_edge
        blocked=[];later=[]
        def discover(sensor,state,**kwargs):
            if kwargs.get('endpoint_scope','whole')=='whole':
                bounds=(state.first_output_frame,state.end_output_frame)
                if not blocked:blocked.append(bounds)
                elif bounds!=blocked[0]:later.append(bounds)
            return real_discover(sensor,state,**kwargs)
        def edge(sensor,left,right):
            row=real_edge(sensor,left,right)
            if blocked and any((s.first_output_frame,s.end_output_frame)==blocked[0]
                    for s in (left,right)):
                return dict(row,available=False,
                    reason='EXPLICIT_NONHUMAN_FIRST_WINDOW_UNAVAILABLE_EDGE_CONTROL')
            return row
        context=replace(self.context,deadline=monotonic()+180.)
        with optional_native_scope(context.deadline),scene_chart_sensors(), \
                patch.object(sensors_module.SceneSensors,'discover',discover), \
                patch.object(scene,'compare_scene_edge',edge):
            candidate=scene.propose_scene_changes(context)
        evidence=family.plain(candidate.evidence) if isinstance(candidate,family.PerformerFamilyRejection) else {}
        self.assertTrue(blocked and later,dict(blocked=blocked,later=later,
            error=getattr(candidate,'error',None),native_work=evidence.get('native_work'),
            last_attempt=evidence.get('attempts',[None])[-1]))
        self.assertNotIsInstance(candidate,family.PerformerFamilyRejection,
            getattr(candidate,'error',None))
        family.validate_bundle_binding(candidate)
        receipt=family.plain(candidate.performer_family_review)
        self.assertGreater(receipt['native_work']['counts']['states'],24)
        self.assertLessEqual(receipt['native_work']['counts']['states'],72)
        self.assertLessEqual(receipt['native_work']['counts']['edges'],384)
        frontiers=receipt['native_work']['state_edge_frontiers']
        self.assertGreaterEqual(len(frontiers),2)
        self.assertTrue(all(row['spent']['states']<=24 and row['spent']['edges']<=128
            for row in frontiers))
        self.assertEqual(sum(row['spent']['states'] for row in frontiers),
            receipt['native_work']['counts']['states'])
        self.assertTrue(receipt['search']['state_edge_frontiers_explicit'])
        self.assertFalse(receipt['search']['window_fairness_guaranteed'])
        self.assertFalse(receipt['quality_approved'])

    def test_no_fixed_heads_preserves_numeric_donors_without_spending_native_inference_on_them(self):
        def unknown(*a,**kw):return replace(scene_native(*a,**kw),heads=())
        with optional_native_scope(monotonic()+90),scene_chart_sensors(),patch.object(source,'observe_native_group',side_effect=unknown):
            value=scene.propose_scene_changes(self.context)
        self.assertIsInstance(value,family.PerformerFamilyRejection)
        evidence=family.plain(value.evidence)
        native=evidence['scene_observations']['native_rows']
        self.assertTrue(native)
        # Unchanged endpoints are the only observations needed to reject this
        # window. All donors remain numeric choices, NOT negative detections.
        self.assertTrue(all(row['endpoint_scope'] in ('entry','exit')
            for row in evidence['scene_observations']['discoveries']))
        self.assertEqual({s['source'] for row in evidence['numeric_source_inventory']
            for s in row['states']},set(range(8)))
        skipped=[a for a in evidence['attempts'] if a.get('stage')=='scene_fixed_hypotheses']
        self.assertTrue(skipped)
        self.assertTrue(all(a['donor_native_observations_skipped'] and
            not a['donor_absence_inferred'] for a in skipped))
        self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        self.assertFalse(evidence['quality_approved'])
        self.assertIs(self.controller.incumbent,self.bundle)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local encode/decode tools required')
class GlobalSceneSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.paths=tuple(cls.root/f'global-{i}.mp4' for i in range(8))
        for i,path in enumerate(cls.paths):
            encode_scene_chart(path,variant=i,frames=540,small_focal_marker=i==6)
        cls.bundle,cls.controller,cls.request,cls.model=build_scene_fixture(cls.root,cls.paths,
            authored_camera_unit=2)
        with optional_native_scope(monotonic()+600.),patch.object(family,'installed_pose_model',return_value=cls.model):
            cls.context=family.freeze_context(controller=cls.controller,request=cls.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,source_derivatives=scene_derivative_fixture())

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def external_reference(self, *, absent=True):
        from crosscut.group_external_original_reference import GroupExternalOriginalReference
        bundle=replace(self.bundle)
        if absent:
            bundle=replace(bundle,boundary_look_match=None,render_plan=replace(bundle.render_plan,
                segments=tuple(replace(s,boundary_look_track=None) for s in bundle.render_plan.segments)))
        return GroupExternalOriginalReference(bundle,self.context.domain,((420,480),))

    def test_external_absent_policy_stays_original_after_an_unprotected_camera_promotion(self):
        from crosscut.first_result_controller import FirstResultController
        from crosscut.review_first_runtime import ReviewFirstRuntime
        reference=self.external_reference();baseline=reference.original_bundle;plan=baseline.render_plan
        controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=plan.sources,
            source_durations=(18.,)*8,master_start=plan.master_audio_start,duration=plan.duration,
            external_original_reference=reference)
        controller.install_baseline(baseline,plan=plan,authority='EXPLICIT_NONHUMAN_REFERENCE')
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))
        rows=list(plan.segments);rows[0]=replace(rows[0],editorial_transform=track)
        candidate=replace(baseline,render_plan=replace(plan,segments=tuple(rows)))
        controller.promote(candidate,plan=candidate.render_plan,authority='NUMERIC_POLICY_ONLY_NOT_NATIVE_ADMISSION')
        self.assertFalse(controller.boundary_look_creation_allowed)
        self.assertEqual(controller.external_original_reference.to_dict(),reference.to_dict())
        rows[-1]=replace(rows[-1],editorial_transform=track)
        protected=replace(candidate,render_plan=replace(plan,segments=tuple(rows)))
        with self.assertRaisesRegex(ValueError,'protected'):
            controller.promote(protected,plan=protected.render_plan,authority='PROTECTED_NEGATIVE_CONTROL')
        with self.assertRaisesRegex(ValueError,'boundary-look policy'):
            controller.promote(self.bundle,plan=self.bundle.render_plan,authority='IMPLICIT_NEW_LOOK_NEGATIVE')
        shifted=replace(candidate,render_plan=replace(candidate.render_plan,master_audio_start=.1))
        with self.assertRaisesRegex(ValueError,'clock authority'):
            reference.validate_candidate(shifted,shifted.render_plan)
        altered_flag=replace(candidate,render_plan=replace(candidate.render_plan,
            member_pose_dissolve_requested=True,member_pose_dissolve_effective=False,
            member_pose_dissolve_count=0))
        with self.assertRaisesRegex(ValueError,'clock authority'):
            reference.validate_candidate(altered_flag,altered_flag.render_plan)
        with self.assertRaisesRegex(ValueError,'verified source/interval/frame domain'):
            controller.promote(shifted,plan=shifted.render_plan,authority='CHANGED_MASTER_CLOCK_NEGATIVE')
        self.assertIs(controller.incumbent,candidate)

    def test_external_original_rejects_absent_metadata_with_active_tracks_and_changed_original(self):
        from crosscut.group_external_original_reference import GroupExternalOriginalReference
        from crosscut.render import BoundaryLookKeyframe,BoundaryLookTrack
        # This is a contract-corruption control, not look-fit quality evidence.
        # A real fit is allowed to retain no applied tracks. Manufacture the
        # contradictory actual renderer recipe explicitly instead of assuming
        # that every chart fit must author a correction in every environment.
        rows=list(self.bundle.render_plan.segments)
        rows[0]=replace(rows[0],boundary_look_track=BoundaryLookTrack((
            BoundaryLookKeyframe(0.,brightness=.003),)))
        corrupted=replace(self.bundle,boundary_look_match=None,
            render_plan=replace(self.bundle.render_plan,segments=tuple(rows)))
        with self.assertRaisesRegex(ValueError,'renderer tracks remain'):
            GroupExternalOriginalReference(corrupted,self.context.domain,((420,480),))
        reference=self.external_reference()
        # Explicit corruption of one copied original, not the class fixture.
        object.__setattr__(reference.original_bundle,'diagnostics',('EXPLICIT_MUTATED_ROOT',))
        with self.assertRaisesRegex(ValueError,'reference changed'):
            reference.to_dict()

    def test_serialized_external_policy_rederives_original_root_tracks_and_protected_camera(self):
        from crosscut.group_external_original_reference import validate_external_original_reference_payload
        reference=self.external_reference();record=reference.to_dict();original=reference.original_bundle
        def validate(value,plan):
            validate_external_original_reference_payload(value,plan=plan,look=None,
                source_domain_payload=self.context.domain.to_dict(),
                source_identities=[list(v) for v in self.context.source_identities],
                live_clocks=self.context.domain.clocks_by_request)
        validate(record,original.render_plan)
        for field in ('mode','original_numeric_plan','original_metadata','encoded_pixel_identity_verified',
                'source_clock_authority','protected_output_intervals'):
            value=json.loads(json.dumps(record))
            if field=='mode':value[field]='ACTIVE_EXISTING'
            elif field=='original_numeric_plan':value[field]['segments'][0]['source_start']+=.1
            elif field=='original_metadata':value[field]['diagnostics'].append('EXPLICIT_MUTATED_SERIALIZED_ROOT')
            elif field=='encoded_pixel_identity_verified':value[field]=True
            elif field=='source_clock_authority':value[field]='SELF_CLAIMED_NATIVE_AUTHORITY'
            else:
                value[field]=[[420,480],[450,480]]
                preimage={key:value[key] for key in ('original_plan_sha256','original_metadata_sha256',
                    'source_domain_sha256','source_identities','original_boundary_look','protected_output_intervals')}
                value['policy_sha256']=family.digest(preimage)
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate(value,original.render_plan)
        rows=list(original.render_plan.segments)
        rows[-1]=replace(rows[-1],editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),)))
        with self.assertRaisesRegex(ValueError,'protected'):
            validate(record,replace(original.render_plan,segments=tuple(rows)))

    def test_external_reference_requires_the_actual_original_source_clock_domain(self):
        reference=self.external_reference(absent=False)
        reference.validate_source_domain(self.context.domain)
        with self.assertRaisesRegex(ValueError,'source/clock domain'):
            reference.validate_source_domain(replace(self.context.domain,audio_offsets=(1.2,)*8))
        with self.assertRaisesRegex(ValueError,'source/clock domain'):
            reference.validate_source_domain(replace(self.context.domain,
                render_frame_clocks=(replace(self.context.domain.render_frame_clocks[0],
                    packet_grid_sha256='0'*64),*self.context.domain.render_frame_clocks[1:])))

    def run_real_obligation_control(self, *, six_binary=False, corrupt_reference=False):
        """Real dense/obligation producers; only NON-HUMAN detectors are replaced.

        A's declared analysis variant loses its arm owner at one internal,
        non-anchor native frame. B retains the SAME original required arm.
        No dense, obligation, utility, completion or admission function is
        patched to create failure/success. This is not human model evidence.
        """
        from crosscut import group_scene_global_execution as execution
        from crosscut.group_joint_person_ownership import OwnedPersonObservation
        cls=type(self)
        if not hasattr(cls,'arm_context'):
            root=cls.root/'arm-obligations';root.mkdir()
            paths=tuple(root/f'arm-{i}.mp4' for i in range(8))
            for i,path in enumerate(paths):
                encode_scene_chart(path,variant=i,frames=540,small_focal_marker=i==6,joint_textures=True)
            bundle,controller,request,model=build_scene_fixture(root,paths,
                authored_camera_unit=2,boundary_look_enabled=False)
            with optional_native_scope(monotonic()+180.),patch.object(family,'installed_pose_model',return_value=model):
                cls.arm_context=family.freeze_context(controller=controller,request=request,
                    eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                    scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,source_derivatives=scene_derivative_fixture())
        real_discover=sensors_module.SceneSensors.discover
        real_frame=sensors_module.SceneSensors.frame
        raw_calls=[];dense_calls=[]
        def native(session, *, request_source_index, native_frame_index, head_analysis_roll=None):
            row=chart_native(session,request_source_index=request_source_index,
                native_frame_index=native_frame_index,head_analysis_roll=head_analysis_roll)
            if row.heads:
                head=row.heads[0];points=head.landmarks_xy
                # Geometric-marker joints on actual decoded textured pixels.
                # Ownership/pose are explicit synthetic detector substitutes.
                joints=tuple((*points[i],.99) for i in (0,1,3,4,4,3,3,4))
                missing=request_source_index==2 and head_analysis_roll==30. and native_frame_index==157
                owner=OwnedPersonObservation(0,head.box_xywh,joints,.99,
                    candidate_head_observation_ids=() if missing else (0,),
                    unique_head_observation_id=None if missing else 0,
                    body_ownership_state='unknown' if missing else 'observed',
                    part_states=(('left_arm','unknown' if missing else 'observed'),
                        ('right_arm','unknown')))
                row=replace(row,people=(owner,),reason='EXPLICIT_SYNTHETIC_ARM_OWNER_'+str(head_analysis_roll))
                if missing:raw_calls.append(dict(sensor_session=id(session),native=native_frame_index))
            return row
        def discover(sensor,state,**kwargs):
            selected=sensor.original_duty_program is not None
            if selected and state.first_output_frame==120:
                values=real_discover(sensor,state,rolls=(30.,None),endpoint_scope='whole')
            else:values=real_discover(sensor,state,**kwargs)
            if six_binary and selected and values and state.first_output_frame!=120:
                duplicate=dict(values[0],explicit_nonhuman_second_inventory_option=True)
                key=family.digest(duplicate);sensor.hypotheses[key]=duplicate
                record=next(r for r in sensor.discoveries.values() if values[0] is sensor.hypotheses[r['hypothesis_ids'][0]])
                if key not in record['hypothesis_ids']:record['hypothesis_ids'].append(key)
                return (values[0],duplicate)
            return values
        def frame(sensor,state,tick,**kwargs):
            if corrupt_reference and sensor.original_duty_program is not None and state.first_output_frame==120 \
                    and state.segments[0].editorial_transform is None:
                sensor.hypotheses.pop(state.editorial_contract_sha256,None)
            return real_frame(sensor,state,tick,**kwargs)
        # Trace calls without replacing dense's implementation/exception.
        real_dense=sensors_module.SceneSensors.dense
        def traced_dense(sensor,path):
            dense_calls.append(dict(sensor=id(sensor),budget=id(sensor.session.budget),
                duty=sensor.original_duty_program['program_sha256']))
            return real_dense(sensor,path)
        context=replace(cls.arm_context,deadline=monotonic()+180.)
        with optional_native_scope(context.deadline),source_chart_sensors():
            union=authored_global_chart_union(context)
            with patch.object(source,'observe_native_group',side_effect=native), \
                    patch.object(sensors_module.SceneSensors,'discover',discover), \
                    patch.object(sensors_module.SceneSensors,'frame',frame), \
                    patch.object(sensors_module.SceneSensors,'dense',traced_dense), \
                    patch.object(source,'complete_source_candidate',wraps=source.complete_source_candidate) as completion:
                candidate=execution.complete_global_scene_program.__wrapped__(context,union)
        return candidate,dense_calls,raw_calls,completion.call_count

    def obligation_control_failure(self,candidate):
        if not isinstance(candidate,family.PerformerFamilyRejection):return None
        evidence=family.plain(candidate.evidence)
        return json.dumps(dict(error=candidate.error,outcome=evidence.get('outcome'),
            attempts=[dict(status=a.get('status'),error=a.get('error'),
                edge_reasons=[r.get('reason') for r in a.get('observed_incident_rows',())],
                decision=a.get('editorial_decision')) for a in evidence.get('attempts',())]))

    def test_real_required_owner_failure_advances_without_erasing_original_arm_duty(self):
        candidate,calls,raw_calls,completed=self.run_real_obligation_control()
        self.assertFalse(isinstance(candidate,family.PerformerFamilyRejection),self.obligation_control_failure(candidate))
        self.assertEqual(completed,1)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(len(raw_calls),1)
        receipt=family.plain(candidate.performer_family_review)
        self.assertEqual(receipt['native_work']['counts']['dense_paths'],2)
        duty=receipt['search']['original_duty_program']
        unit=next(u for u in duty['units'] if u['first_output_frame']==120)
        self.assertTrue(all('left_arm' in r['required_part_names'] for r in unit['rows']))
        self.assertFalse(receipt['quality_approved'])
        family.validate_bundle_binding(candidate)

    def test_real_frozen_hypothesis_contract_failure_is_terminal_without_pair_cache_or_completion(self):
        candidate,calls,_,completed=self.run_real_obligation_control(corrupt_reference=True)
        self.assertIsInstance(candidate,family.PerformerFamilyRejection)
        evidence=family.plain(candidate.evidence)
        self.assertEqual(evidence['outcome'],'TERMINAL_FAILURE')
        self.assertEqual(calls,[])
        self.assertEqual(completed,0)
        self.assertEqual(evidence['conditional_frontier']['infeasible_pairs'],[])
        self.assertEqual(evidence['conditional_frontier']['completed_combination_count'],0)

    def test_real_state_local_owner_failure_does_not_starve_binary_frontier_with_four_dense_slots(self):
        candidate,calls,raw_calls,completed=self.run_real_obligation_control(six_binary=True)
        self.assertFalse(isinstance(candidate,family.PerformerFamilyRejection),self.obligation_control_failure(candidate))
        self.assertEqual(completed,1)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(len(raw_calls),1)
        receipt=family.plain(candidate.performer_family_review)
        self.assertEqual(receipt['native_work']['limits']['dense_paths'],4)
        self.assertEqual(receipt['native_work']['counts']['dense_paths'],2)
        self.assertTrue(any(a['status']=='CACHED_OBSERVED_STATE_DENSE_REJECT'
            for a in receipt['search']['prior_complete_rejections']))
        self.assertFalse(receipt['quality_approved'])

    def test_original_no_boundary_look_reaches_completion_parent_and_final_without_fabricating_look(self):
        from crosscut.group_scene_global_execution import complete_global_scene_program
        storage=tempfile.TemporaryDirectory();self.addCleanup(storage.cleanup);root=Path(storage.name)
        bundle,controller,request,model=build_scene_fixture(root,self.paths,
            authored_camera_unit=2,boundary_look_enabled=False)
        self.assertIsNone(bundle.boundary_look_match)
        self.assertTrue(all(s.boundary_look_track is None for s in bundle.render_plan.segments))
        self.assertTrue(all(s.look_adjustment is not None for s in bundle.render_plan.segments))
        with optional_native_scope(monotonic()+180.),patch.object(family,'installed_pose_model',return_value=model):
            context=family.freeze_context(controller=controller,request=request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,source_derivatives=scene_derivative_fixture())
        from crosscut.group_external_original_reference import GroupExternalOriginalReference
        reference=GroupExternalOriginalReference(bundle,context.domain,((420,480),))
        context=replace(context,external_original_reference=reference)
        with optional_native_scope(context.deadline),source_chart_sensors(), \
                patch('crosscut.boundary_look_match.analyze_boundary_look_match',
                    side_effect=AssertionError('An originally absent family must not be fabricated')) as fit, \
                patch('crosscut.boundary_look_match.verify_boundary_look_match_render_domain',
                    side_effect=AssertionError('No matched-look proof exists in the absent family')) as look_encode:
            union=authored_global_chart_union(context)
            candidate=complete_global_scene_program.__wrapped__(context,union)
            self.assertFalse(isinstance(candidate,family.PerformerFamilyRejection),getattr(candidate,'error',None))
            self.assertIsNone(candidate.boundary_look_match)
            self.assertEqual([s.look_adjustment for s in candidate.render_plan.segments],
                [s.look_adjustment for s in bundle.render_plan.segments])
            self.assertTrue(all(s.boundary_look_track is None for s in candidate.render_plan.segments))
            family.validate_bundle_binding(candidate)
            receipt=family.plain(candidate.performer_family_review)
            self.assertEqual(receipt['external_original_reference'],reference.to_dict())
            audit=receipt['look_refresh']
            self.assertEqual(audit['retained_family']['original']['mode'],'ORIGINAL_ABSENT')
            self.assertEqual(audit['retained_family']['final']['mode'],'ORIGINAL_ABSENT')
            self.assertEqual(audit['refreshed_edges'],[])
            self.assertFalse(audit['matched_production_renderer_used'])
            self.assertFalse(audit['quality_approved'])
            with patch.object(__import__('crosscut.group_scene_selected_replay',fromlist=['replay_selected_scene']),
                    'replay_selected_scene',camera_chart_selected_replay):
                proof=source.validate_parent(candidate,context)
            self.assertTrue(proof['fresh_complete_family_equal'])
            metrics=_REAL_FINAL(candidate.render_plan.to_json(),receipt,monotonic()+90.)
            self.assertTrue(family.qa_metrics_valid(metrics))
            self.assertFalse(metrics['quality_approved'])
            fit.assert_not_called();look_encode.assert_not_called()
        probes=(150,390)
        original_pixels=render_and_decode_chart(self,replace(bundle.render_plan,
            output=root/'no-look-original.mp4'),capture_ticks=probes)
        selected_pixels=render_and_decode_chart(self,candidate.render_plan,capture_ticks=probes)
        for tick in probes:
            self.assertGreater(float(np.mean(np.abs(original_pixels[tick].astype(np.int16)
                -selected_pixels[tick].astype(np.int16)))),.25)
        destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
        if destination:
            target=Path(destination);target.mkdir(parents=True,exist_ok=True)
            (target/'global-original-no-look-proof.json').write_text(json.dumps(dict(
                fixture='EXPLICIT_NON_HUMAN_CHART_NOT_MODEL_OR_EDITING_QUALITY_PROOF',
                original_bundle=bundle.to_dict(),completion=receipt,parent=proof,final_native=metrics,
                actual_original_control_render_decoded_frames=bundle.render_plan.output_frame_count,
                actual_full_render_decoded_frames=candidate.render_plan.output_frame_count,
                old_static_source_recipes_preserved=True,no_boundary_look_fit_or_encode_performed=True,
                human_media_rendered=False,quality_approved=False),indent=2))

    def test_distant_global_camera_union_reaches_one_completion_parent_and_final(self):
        from crosscut.group_scene_global_execution import complete_global_scene_program
        from crosscut.group_scene_selected_replay import GLOBAL_CONTRACT
        from crosscut.group_scene_edit_operations import derive_scene_edit_operations
        context=replace(self.context,deadline=monotonic()+180.)
        with optional_native_scope(context.deadline),source_chart_sensors():
            union=authored_global_chart_union(context)
            operations=derive_scene_edit_operations(original=context.incumbent.render_plan,
                proposed=union.plan,domain=context.domain,
                mutable_intervals=union.allocation.mutable_output_intervals)
            expected_operations=[(120,180,'CAMERA'),(360,420,'CAMERA')]
            self.assertEqual([(row['first_output_frame'],row['end_output_frame'],row['operation'])
                for row in operations['actual_shot_operations'] if row['mutable']],expected_operations,
                'Two scoped mutables are not proof of two actual camera edits')
            candidate=complete_global_scene_program.__wrapped__(context,union)
            if isinstance(candidate,family.PerformerFamilyRejection):
                evidence=family.plain(candidate.evidence)
                self.fail(json.dumps(dict(error=candidate.error,
                    counts=evidence.get('native_work',{}).get('counts'),
                    attempts=evidence.get('attempts',()))))
            family.validate_bundle_binding(candidate)
            receipt=family.plain(candidate.performer_family_review)
            program=receipt['selected_scene_program']
            self.assertEqual(program['contract'],GLOBAL_CONTRACT)
            self.assertEqual([(row['first_output_frame'],row['end_output_frame'],row['operation'])
                for row in program['edit_operations']['actual_shot_operations'] if row['mutable']],
                expected_operations)
            scope=program['global_execution_scope']
            self.assertEqual(len(program['selected_states']),6)
            self.assertEqual(scope['mutable_state_indices'],[1,4])
            self.assertEqual(scope['incident_state_pair_indices'],[[0,1],[1,2],[3,4],[4,5]])
            self.assertEqual(len(receipt['source_camera_refresh']['mutable_shots']),2)
            self.assertEqual(receipt['search']['atomic_numeric_programmes_completed'],1)
            self.assertFalse(receipt['search']['local_complete_bundles_concatenated'])
            self.assertFalse(receipt['quality_approved'])
            probes=(150,390)
            original_pixels=render_and_decode_chart(self,replace(context.incumbent.render_plan,
                output=self.root/'global-original-control.mp4'),capture_ticks=probes)
            selected_pixels=render_and_decode_chart(self,candidate.render_plan,capture_ticks=probes)
            pixel_changes=[]
            for tick in probes:
                difference=float(np.mean(np.abs(original_pixels[tick].astype(np.int16)
                    -selected_pixels[tick].astype(np.int16))))
                self.assertGreater(difference,.25,'Actual encoded pixels must change in BOTH mutable intervals')
                pixel_changes.append(dict(output_tick=tick,mean_absolute_decoded_pixel_difference=difference,
                    visible_geometric_change_certified=False,professional_quality_certified=False))
            self.assertEqual([s.source_index for s in candidate.render_plan.segments],
                [s.source_index for s in context.incumbent.render_plan.segments])
            with patch.object(__import__('crosscut.group_scene_selected_replay',fromlist=['replay_selected_scene']),
                    'replay_selected_scene',camera_chart_selected_replay):
                proof=source.validate_parent(candidate,context)
            self.assertTrue(proof['fresh_complete_family_equal'])
            self.assertTrue(proof['source_and_camera_edit_operations_independently_rederived'])
            self.assertFalse(proof['quality_approved'])
            metrics=_REAL_FINAL(candidate.render_plan.to_json(),receipt,monotonic()+90.)
            self.assertTrue(family.qa_metrics_valid(metrics))
            self.assertFalse(metrics['quality_approved'])
            destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
            if destination:
                target=Path(destination);target.mkdir(parents=True,exist_ok=True)
                (target/'global-chart-execution-proof.json').write_text(json.dumps(dict(
                    fixture='EXPLICIT_NON_HUMAN_CHART_NOT_MODEL_OR_EDITING_QUALITY_PROOF',
                    completion=receipt,parent=proof,final_native=metrics,
                    actual_full_render_decoded_frames=candidate.render_plan.output_frame_count,
                    actual_original_control_render_decoded_frames=context.incumbent.render_plan.output_frame_count,
                    actual_encoded_mutable_pixel_changes=pixel_changes,
                    actual_output_size=[candidate.render_plan.width,candidate.render_plan.height],
                    human_media_rendered=False,quality_approved=False),indent=2))

    def test_distant_global_camera_union_cannot_hide_clipped_interior(self):
        from crosscut.group_scene_global_execution import complete_global_scene_program
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),source_chart_sensors():
            union=authored_global_chart_union(context,clipped_interior=True)
            candidate=complete_global_scene_program.__wrapped__(context,union)
        self.assertIsInstance(candidate,family.PerformerFamilyRejection)
        evidence=family.plain(candidate.evidence)
        self.assertGreater(evidence['native_work']['counts']['dense_paths'],0)
        self.assertTrue('clip' in candidate.error or 'crop' in candidate.error,candidate.error)
        self.assertFalse(evidence['fallback_work_restart_allowed'])

    def run_two_conditional_chart_control(self,*,terminal=None):
        """Explicit A-fails/B-valid control; real B native completion stays on.

        A is a labelled NON-HUMAN sensor alternative with identical endpoint
        observations; only its dense failure is injected. It is not evidence
        of two real human hypotheses or a trained model's editing ability.
        """
        from crosscut import group_scene_global_execution as execution
        from crosscut.group_scene_hypothesis_outcomes import SceneHypothesisRejection
        real_discover=sensors_module.SceneSensors.discover
        real_dense=sensors_module.SceneSensors.dense
        calls=[]
        def discover(sensor,state,**kwargs):
            values=real_discover(sensor,state,**kwargs)
            if (values and state.first_output_frame==120
                    and state.segments[0].editorial_transform is None):
                bad=dict(values[0],explicit_nonhuman_dense_failure_control=True)
                identity=family.digest(bad)
                sensor.hypotheses[identity]=bad
                record=next(r for r in sensor.discoveries.values()
                    if values[0] is sensor.hypotheses[r['hypothesis_ids'][0]])
                if identity not in record['hypothesis_ids']:record['hypothesis_ids'].insert(0,identity)
                return (bad,values[0])
            return values
        def dense(sensor,path):
            calls.append(dict(sensor=id(sensor),budget=id(sensor.session.budget),
                duty=sensor.original_duty_program['program_sha256']))
            if any(sensor.hypothesis(s).get('explicit_nonhuman_dense_failure_control') for s in path):
                if terminal=='contract':raise ValueError('EXPLICIT_TERMINAL_ARCHIVE_CONTRACT_CONTROL')
                if terminal=='budget':sensor.session.budget.limits['states']=1
                raise SceneHypothesisRejection('EXPLICIT_A_ENDPOINTS_PASS_DENSE_REJECT')
            return real_dense(sensor,path)
        context=replace(self.context,deadline=monotonic()+180.)
        with optional_native_scope(context.deadline),source_chart_sensors():
            union=authored_global_chart_union(context)
            with patch.object(sensors_module.SceneSensors,'discover',discover), \
                    patch.object(sensors_module.SceneSensors,'dense',dense), \
                    patch.object(execution,'MAX_CONDITIONAL_PATHS',
                        1 if terminal=='truncated' else execution.MAX_CONDITIONAL_PATHS), \
                    patch.object(source,'complete_source_candidate',wraps=source.complete_source_candidate) as completion:
                candidate=execution.complete_global_scene_program.__wrapped__(context,union)
        return candidate,calls,completion.call_count

    def test_global_dense_reject_advances_same_session_to_second_conditional_then_completes_once(self):
        candidate,calls,completed=self.run_two_conditional_chart_control()
        self.assertFalse(isinstance(candidate,family.PerformerFamilyRejection),getattr(candidate,'error',None))
        self.assertEqual(completed,1)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])
        family.validate_bundle_binding(candidate)
        receipt=family.plain(candidate.performer_family_review)
        self.assertEqual(receipt['native_work']['counts']['dense_paths'],2)
        self.assertEqual(receipt['search']['atomic_numeric_programmes_completed'],1)
        self.assertFalse(receipt['quality_approved'])

    def test_global_budget_before_second_conditional_retains_unknown_pending_cursor_without_restart(self):
        candidate,calls,completed=self.run_two_conditional_chart_control(terminal='budget')
        self.assertIsInstance(candidate,family.PerformerFamilyRejection)
        evidence=family.plain(candidate.evidence)
        self.assertEqual(evidence['outcome'],'BUDGET_PARTIAL')
        self.assertEqual(completed,0)
        self.assertEqual(len(calls),1)
        cursor=evidence['conditional_frontier']
        self.assertEqual(cursor['completed_combination_count'],1)
        self.assertEqual(cursor['pending_combination_count'],1)
        self.assertEqual(cursor['next_rank_vector'],[0,1,0,0,0,0])
        self.assertFalse(cursor['inventory_exhausted'])
        self.assertFalse(evidence['fallback_work_restart_allowed'])
        self.assertFalse(evidence['quality_approved'])

    def test_global_contract_error_is_terminal_not_an_observed_hypothesis_rejection(self):
        candidate,calls,completed=self.run_two_conditional_chart_control(terminal='contract')
        self.assertIsInstance(candidate,family.PerformerFamilyRejection)
        evidence=family.plain(candidate.evidence)
        self.assertEqual(evidence['outcome'],'TERMINAL_FAILURE')
        self.assertEqual(completed,0)
        self.assertEqual(len(calls),1)
        self.assertEqual(evidence['conditional_frontier']['completed_combination_count'],0)
        self.assertFalse(evidence['fallback_work_restart_allowed'])

    def test_global_search_cap_is_not_exhaustive_no_gain_and_keeps_second_conditional_pending(self):
        candidate,calls,completed=self.run_two_conditional_chart_control(terminal='truncated')
        self.assertIsInstance(candidate,family.PerformerFamilyRejection)
        evidence=family.plain(candidate.evidence)
        self.assertEqual(evidence['outcome'],'SEARCH_TRUNCATED')
        self.assertEqual(completed,0)
        self.assertEqual(len(calls),1)
        cursor=evidence['conditional_frontier']
        self.assertEqual(cursor['completed_combination_count'],1)
        self.assertEqual(cursor['pending_combination_count'],1)
        self.assertEqual(cursor['next_rank_vector'],[0,1,0,0,0,0])
        self.assertFalse(cursor['inventory_exhausted'])
        self.assertFalse(evidence['fallback_work_restart_allowed'])
        self.assertFalse(evidence['quality_approved'])

    def test_real_chart_mixed_endpoint_hypothesis_emits_typed_local_temporal_rejection(self):
        from crosscut.group_joint_observations import GroupJointObservationSession
        from crosscut.group_joint_source_camera import JointGroupWorkBudget
        from crosscut.group_scene_execution_scope import execution_path_from_union
        from crosscut.group_scene_hypothesis_outcomes import SceneHypothesisRejection
        context=replace(self.context,deadline=monotonic()+120.)
        with optional_native_scope(context.deadline),source_chart_sensors():
            union=authored_global_chart_union(context)
        path=execution_path_from_union(context,context.domain,union)
        with optional_native_scope(context.deadline),scene_chart_sensors(), \
                GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
                    absolute_deadline=context.deadline,**source.WORK_LIMITS)) as session:
            sensors=sensors_module.SceneSensors(session,source.installed_native_assets(context))
            values=[]
            for index,state in enumerate(path):
                hypotheses=sensors.discover(state,rolls=(None,),endpoint_scope='whole')
                if index==path.mutable_indices[0]:
                    hypothesis=next(h for h in hypotheses if h['anchor_head_ids']==[0,0,1,1])
                else:hypothesis=next(h for h in hypotheses if h['anchor_head_ids']==[0,0,0,0])
                values.append(sensors.bind(state,hypothesis))
            with self.assertRaisesRegex(SceneHypothesisRejection,'one whole temporal path'):
                sensors.dense(path.with_states(values))


class SceneFormationPixelTests(unittest.TestCase):
    def fixture(self,flat=False):
        from engine.tests.test_group_joint_native_joint_flow import NativeJointFlowTests
        from crosscut.group_joint_source_camera import JointGroupWorkBudget
        a,b,p,q=NativeJointFlowTests().fixture()
        def move(person,delta,i):
            x,y,w,h=person.box_xywh
            return replace(person,observation_id=i,box_xywh=(x+delta,y,w,h),
                upper_body_xyc=tuple((x+delta,y,c) for x,y,c in person.upper_body_xyc))
        previous=np.maximum(np.roll(a,-200,axis=1),np.roll(a,200,axis=1))
        current=np.maximum(np.roll(b,-200,axis=1),np.roll(b,200,axis=1))
        if flat:previous[:]=0;current[:]=0
        rows=(SimpleNamespace(request_source_index=0,native_frame_index=6,frame_size_wh=(1000,800),
                source_clock_identity='c'*64,source_file_identity=(1,2,3,4,5),
                native_pixel_sha256=source.sha256(memoryview(previous)).hexdigest(),
                people=(move(p,-.2,0),move(p,.2,1)),sha256='a'*64),
            SimpleNamespace(request_source_index=0,native_frame_index=9,frame_size_wh=(1000,800),
                source_clock_identity='c'*64,source_file_identity=(1,2,3,4,5),
                native_pixel_sha256=source.sha256(memoryview(current)).hexdigest(),
                people=(move(q,-.2,10),move(q,.2,11)),sha256='b'*64))
        sensors=SimpleNamespace(link_cache={},session=SimpleNamespace(
            budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10),
            native_pixels=lambda source,tick:previous if tick==6 else current),
            # EXPLICIT head association fixture. Even a perfect head mapping
            # must not convert missing arm-joint pixel paths into a certificate.
            _head_links=lambda *args:{'unique_links':[[0,10],[1,11]]})
        return sensors,rows

    def test_real_joint_lk_observes_two_body_cohorts_without_a_single_face(self):
        sensors,rows=self.fixture()
        evidence=sensors_module.native_body_links(sensors,*rows)
        self.assertTrue(evidence['complete_required_joint_paths'])
        self.assertEqual(evidence['assignments'],[[0,0],[1,1]])
        self.assertEqual(len(evidence['joint_pixel_flow']),2)
        self.assertFalse(evidence['body_phase_certified'])
        self.assertFalse(evidence['quality_approved'])

    def test_required_body_path_uses_every_intermediate_original_raster(self):
        sensors,rows=self.fixture()
        reads=[];pixels=sensors.session.native_pixels
        def record(source_id,native):
            reads.append(native)
            return pixels(source_id,native)
        sensors.session.native_pixels=record
        evidence=sensors_module.native_body_links(sensors,*rows)
        self.assertTrue(evidence['complete_required_joint_paths'])
        self.assertEqual(sorted(reads),[6,7,8,9])
        self.assertEqual(evidence['native_frame_indices'],[6,7,8,9])
        self.assertTrue(evidence['native_steps_complete'])
        self.assertEqual(evidence['native_pixel_sha256'][0],rows[0].native_pixel_sha256)
        self.assertEqual(evidence['native_pixel_sha256'][-1],rows[1].native_pixel_sha256)

    def test_matching_endpoint_joints_cannot_skip_an_occluded_middle_raster(self):
        sensors,rows=self.fixture()
        pixels=sensors.session.native_pixels
        sensors.session.native_pixels=lambda source_id,native:(
            np.zeros_like(pixels(source_id,native)) if native==7 else pixels(source_id,native))
        evidence=sensors_module.native_body_links(sensors,*rows)
        self.assertFalse(evidence['complete_required_joint_paths'])
        self.assertEqual(evidence['assignments'],[])

    def test_required_body_path_rejects_missing_middle_or_changed_endpoint_pixels(self):
        for mismatch in ('middle_missing','endpoint_hash','clock','file','raster','gap'):
            with self.subTest(mismatch=mismatch):
                sensors,rows=self.fixture()
                if mismatch=='middle_missing':
                    pixels=sensors.session.native_pixels
                    sensors.session.native_pixels=lambda source_id,native:(
                        None if native==7 else pixels(source_id,native))
                elif mismatch=='endpoint_hash':rows[1].native_pixel_sha256='0'*64
                elif mismatch=='clock':rows[1].source_clock_identity='d'*64
                elif mismatch=='file':rows[1].source_file_identity=(1,2,3,4,6)
                elif mismatch=='raster':rows[1].frame_size_wh=(999,800)
                else:rows[1].native_frame_index=20
                with self.assertRaises(ValueError):sensors_module.native_body_links(sensors,*rows)

    def test_unread_native_steps_are_not_certified_when_no_joint_texture_exists(self):
        sensors,rows=self.fixture(flat=True)
        evidence=sensors_module.native_body_links(sensors,*rows)
        self.assertFalse(evidence['complete_required_joint_paths'])
        self.assertFalse(evidence['native_steps_complete'])
        self.assertEqual(evidence['native_frame_indices'],[6,9])

    def test_perfect_head_hypotheses_cannot_certify_unknown_joint_motion(self):
        sensors,rows=self.fixture(flat=True)
        evidence=sensors_module.native_body_links(sensors,*rows)
        self.assertEqual(evidence['costs'],[[None,None],[None,None]])
        self.assertFalse(evidence['complete_required_joint_paths'])
        self.assertTrue(evidence['head_body_hypotheses_are_not_joint_path_authority'])

    def focal_fixture(self,flat=False):
        from dataclasses import asdict
        from crosscut.group_joint_person_ownership import OwnedPersonObservation
        sensors,rows=self.fixture(flat=flat)
        for row in rows:
            row.people=tuple(OwnedPersonObservation(**asdict(p),
                candidate_head_observation_ids=(p.observation_id,),
                unique_head_observation_id=p.observation_id,
                body_ownership_state='observed',
                part_states=(('left_arm','observed'),('right_arm','observed')))
                for p in row.people)
        return sensors,rows

    def test_focal_joint_path_does_not_require_disappearing_background_body(self):
        sensors,rows=self.focal_fixture()
        rows[1].people=rows[1].people[:1]
        focal=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        formation=sensors_module.native_body_links(sensors,*rows)
        self.assertTrue(focal['complete_required_joint_paths'])
        self.assertEqual(focal['required_body_scope'],'selected_focal_owner')
        self.assertEqual(focal['required_body_observation_ids'],[[0],[10]])
        self.assertEqual(len(focal['joint_pixel_flow']),2)
        self.assertFalse(formation['complete_required_joint_paths'])
        self.assertEqual(formation['required_body_observation_ids'],[[0,1],[10]])

    def test_background_joint_path_cannot_replace_missing_selected_owner(self):
        sensors,rows=self.focal_fixture()
        rows[1].people=rows[1].people[1:]
        evidence=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        self.assertEqual(evidence['assignments'],[[1,0]])
        self.assertFalse(evidence['complete_required_joint_paths'])
        self.assertEqual(evidence['required_body_observation_ids'],[[0],[]])

    def test_single_known_arm_has_its_own_pixel_mask_without_inventing_the_other(self):
        sensors,rows=self.focal_fixture()
        for row in rows:
            row.people=tuple(replace(person,
                upper_body_xyc=tuple((x,y,0. if i in (3,5) else c)
                    for i,(x,y,c) in enumerate(person.upper_body_xyc)),
                part_states=(('left_arm','observed'),('right_arm','unknown')))
                for person in row.people)
        one=sensors_module.native_body_links(sensors,*rows,
            selected_head_ids=(0,10),selected_part_names=('left_arm',))
        both=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        self.assertTrue(one['complete_required_joint_paths'])
        self.assertEqual(one['required_part_names'],['left_arm'])
        self.assertFalse(both['complete_required_joint_paths'])

    def test_selected_head_link_cannot_replace_missing_joint_pixels(self):
        sensors,rows=self.focal_fixture(flat=True)
        evidence=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        self.assertFalse(evidence['complete_required_joint_paths'])
        self.assertEqual(evidence['assignments'],[])

    def test_selected_head_joint_contradiction_is_not_hidden_by_scoping(self):
        sensors,rows=self.focal_fixture()
        sensors._head_links=lambda *args:{'unique_links':[[0,11],[1,10]]}
        evidence=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        self.assertTrue(evidence['head_joint_assignment_contradiction'])
        self.assertFalse(evidence['complete_required_joint_paths'])

    def test_selected_temporal_scope_rejects_ambiguous_owners_and_invalid_ids(self):
        sensors,rows=self.focal_fixture()
        rows[1].people=tuple(replace(p,candidate_head_observation_ids=(10,),
            unique_head_observation_id=10) for p in rows[1].people)
        evidence=sensors_module.native_body_links(sensors,*rows,selected_head_ids=(0,10))
        self.assertFalse(evidence['complete_required_joint_paths'])
        for ids in ((False,10),(0,),[0,10]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                sensors_module.native_body_links(sensors,*rows,selected_head_ids=ids)

    def test_reversed_or_foreign_source_is_not_a_body_path(self):
        sensors,rows=self.fixture()
        with self.assertRaises(ValueError):sensors_module.native_body_links(sensors,rows[1],rows[0])
        foreign=SimpleNamespace(**{**vars(rows[1]),'request_source_index':1})
        with self.assertRaises(ValueError):sensors_module.native_body_links(sensors,rows[0],foreign)
