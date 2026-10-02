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
from engine.tests.test_performer_family_owner import encode_chart,build_fixture,chart_sensors

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


_REAL_PROPOSE=family.propose_performer_family.__wrapped__
_REAL_FINAL=source.final_live_replay.__wrapped__


def scene_derivative_fixture(count=8):
    pairs=tuple(SourceDerivativePairEvidence(a,b,DERIVATIVE_DISTINCT,
        'EXPLICIT_NONHUMAN_CAMERA_RELATION_SENSOR_FIXTURE_NOT_ACTUAL_DERIVATIVE_ANALYSIS',
        5,0,5,0,None,None,None,None,None,None,()) for a,b in combinations(range(count),2))
    value=SourceDerivativeDiversityResult((1.,3.,5.,7.,9.),(),pairs,tuple(range(count)),
        0,True,True,'',SourceDerivativeConfig())
    return replace(value,audit_sha256=family.digest(value._payload(include_sha256=False)))


def encode_scene_chart(path,*,variant,frames=360):
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
        # A per-file raster tag is not a claim of eight independent real cameras.
        image[-1,-(variant+1):,:]=(variant*20,20,40)
        images.append(image)
    cap.release();base.unlink()
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','320x180','-r','30','-i','pipe:0',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-threads','1',
        '-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-bf','0','-c:a','aac','-y',str(path)]
    result=subprocess.run(cmd,input=np.asarray(images).tobytes(),capture_output=True,timeout=40)
    if result.returncode:raise RuntimeError(result.stderr.decode())


def build_scene_fixture(root,sources,*,boundary=0):
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
    plan=replace(bundle.render_plan,fps=30.,transitions=transitions,segments=tuple(
        RenderSegment(i,2.*i+1,2.,2.*i,pan_crop=pan,
            look_adjustment=SourceLookAdjustment(brightness=.002+i*.0001),
            editorial_transform=motion.track_for(i)) for i in range(n)))
    # Real full old look fit AND actual renderer/decode admission. The fixture
    # is not positive unless an old applied correction really exists.
    from crosscut.boundary_look_match import _tracks_from_decisions
    from crosscut.source_look_match import SourceLookAdjustment
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


class ScenePureContractTests(unittest.TestCase):
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
                people=(move(p,-.2,0),move(p,.2,1)),sha256='a'*64),
            SimpleNamespace(request_source_index=0,native_frame_index=9,frame_size_wh=(1000,800),
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
