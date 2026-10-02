"""Headless NON-HUMAN charts, actual native producer/flow/family consumers.

Only model inference/descriptors and the typed derivative result are synthetic.
Source decoding, body-only native producer, native-step LK, projected geometry,
scene policy/search, metadata refresh, look encoding and consumers are real.
No human/NN/actor/phase/quality claim follows from these tests.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
from unittest.mock import patch
import hashlib
import json
import os
import tempfile
import subprocess
import unittest

import cv2
import numpy as np

from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut import group_scene_source_owner as scene
from crosscut import group_scene_observations as observations
from crosscut import group_scene_composition as composition
from crosscut import group_scene_editorial_policy as policy
from crosscut.group_scene_intent import role_from_cells, declared_shot_intent, validate_intent
from crosscut.group_scene_role_obligations import scene_role_obligations
from crosscut.group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget
from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.optional_native_worker import isolated_optional, optional_native_scope
from crosscut.timeline import ShotType
from crosscut.group_motion_camera_refinement import ProjectedBoundaryObservation
from crosscut.visual_score import FrameVisualDescriptor
from crosscut import pipeline
from engine.tests import test_scene_source_owner as charts
from engine.tests.test_source_changing_family_owner import source_chart_sensors
from engine.tests.test_performer_family_owner import chart_sensors


def chart_detector(pixels):
    """Explicit two-color rectangle model substitute, NOT person inference."""
    b,g,r = (pixels[:,:,i].astype(float) for i in range(3))
    rows=[]
    for mask in (((g-r)>45)&((g-b)>45)&(g>130), ((b-g)>60)&((b-r)>60)):
        yy,xx=np.nonzero(mask)
        if len(xx)<20: continue
        rows.append((float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1),.99))
    return tuple(sorted(rows,key=lambda row:row[0]))


def encode_wide_chart(path, *, variant, frames=540):
    """Texture-bearing NON-HUMAN ROIs; flat charts are separately negative.

    The prior face fixture has too few corners for the *unchanged* 12-corner
    body-support LK gate. Add visible geometric texture, not fake flow output.
    Leave later retained focal/look rasters unchanged before re-encoding.
    """
    seed=path.with_suffix('.untextured.mp4')
    charts.encode_scene_chart(seed,variant=variant,frames=frames)
    capture=cv2.VideoCapture(str(seed));images=[]
    while True:
        ok,image=capture.read()
        if not ok:break
        if len(images)<300:
            for left,top,right,bottom,_ in chart_detector(image):
                # Marks stay attached to the observed moving rectangle.
                for y in range(int(top)+8,int(bottom)-5,9):
                    for x in range(int(left)+8,int(right)-5,9):
                        cv2.rectangle(image,(x,y),(x+2,y+2),(12,18,12),-1)
        images.append(image)
    capture.release();seed.unlink()
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','320x180','-r','30','-i','pipe:0',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-threads','1',
        '-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-bf','0','-c:a','aac','-y',str(path)]
    encoded=subprocess.run(cmd,input=np.asarray(images).tobytes(),capture_output=True,timeout=40)
    if encoded.returncode:raise RuntimeError(encoded.stderr.decode())


def describe_headless_chart(frame, **kwargs):
    rows=chart_detector(frame);h,w=frame.shape[:2]
    height=max(((p[3]-p[1])/h for p in rows),default=0.)
    return FrameVisualDescriptor(.9,.9,height,ShotType.WIDE,0,(.1,.2,.3),
        color_embedding=(float(frame.mean())/255.,.2,.3),
        face_detector_backend='EXPLICIT_CHART_NO_FACE_OBSERVATION',
        detected_person_count=len(rows),pose_backend='EXPLICIT_RECTANGLE_NO_JOINT_OBSERVATION',
        shot_scale_basis='EXPLICIT_CHART_NATIVE_BODY_SUPPORT')


@contextmanager
def wide_chart_sensors():
    with source_chart_sensors(), ExitStack() as stack:
        def register(session, **assets):
            session.register_model('nanodet_people',assets['person_model_path'],chart_detector,
                adapter_fingerprint=dict(contract='EXPLICIT_NONHUMAN_RECTANGLE_SENSOR_NOT_NANODET_INFERENCE',
                    fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        stack.enter_context(patch.object(source,'register_joint_native_models',side_effect=register))
        stack.enter_context(patch.object(source,'observe_native_group',side_effect=ValueError(
            'head/pose native path must not be requested by this all-composition candidate')))
        stack.enter_context(patch('crosscut.visual_score.describe_frame',side_effect=describe_headless_chart))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.measured_output_face_box',return_value=None))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.measure_projected_motion',return_value=None))
        stack.enter_context(patch('crosscut.group_motion_camera_refinement.observe_projected_boundary',
            return_value=ProjectedBoundaryObservation()))
        yield


def build_wide_fixture(root,sources,**kwargs):
    # Author the baseline FIRST: four wide output cells with no focal duty;
    # four later retained focal crops and an encoded-admitted distant look.
    # This is not a donor-dependent weakening of an already frozen candidate.
    bundle,_,request,model=charts.build_scene_fixture(root,sources,**kwargs)
    timeline=replace(bundle.visual_timeline,choices=tuple(replace(c,shot_type=ShotType.WIDE,
        entry_shot_scale=.35,exit_shot_scale=.35) if i<4 else c
        for i,c in enumerate(bundle.visual_timeline.choices)))
    face=replace(bundle.face_match_plan,segments=tuple(replace(f,applied=False,
        fallback_reason='EXPLICIT_PREDECLARED_WIDE_BODY_COMPOSITION',pan_crop=None,
        observed_face_height=None,target_face_height=None,required_zoom=None,applied_zoom=1.,
        alignment_subject=None) if i<4 else f for i,f in enumerate(bundle.face_match_plan.segments)))
    clocks=tuple(charts.probe_render_frame_clock(p) for p in sources)
    with chart_sensors():
        alignment=charts.build_boundary_subject_alignment_plan(visual_timeline=timeline,sources=sources,
            transitions=bundle.render_plan.transitions,fps=30.,output_width=320,output_height=180,
            face_match_plan=face,render_frame_clocks=clocks,boundary_indices=(4,))
    motion=charts.build_editorial_motion_plan(visual_timeline=timeline,transitions=bundle.render_plan.transitions,
        fps=30.,boundary_subject_alignment_plan=alignment,face_match_plan=face)
    motion=replace(motion,eligible_request_indices=tuple(range(8)))
    plan=replace(bundle.render_plan,segments=tuple(replace(s,pan_crop=face.segments[i].pan_crop,
        editorial_transform=motion.track_for(i)) for i,s in enumerate(bundle.render_plan.segments)))
    from crosscut.timeline_review import review_timeline
    review=review_timeline(timeline,available_source_count=8)
    bundle=replace(bundle,render_plan=plan,visual_timeline=timeline,face_match_plan=face,
        timeline_review=review,pre_render_score=float(review.weighted_score),editorial_motion_plan=motion,
        stage_edit_exact_shot_clock=pipeline._deterministic_exact_shot_clock_from_timeline(timeline,
            section_plan=bundle.section_plan,phrase_times=bundle.phrase_times))
    pipeline._validate_review_first_group_bundle(bundle,eligible_indices=tuple(range(8)),offsets=(1.,)*8,
        timewarps=(None,)*8,common_start=0.,phrase_times=bundle.phrase_times)
    controller=charts.FirstResultController(charts.ReviewFirstRuntime(enabled=True),sources=sources,
        source_durations=(18.,)*8,master_start=0.,duration=16.,journal_path=root/'wide-controller.json')
    controller.install_baseline(bundle,plan=plan,authority='deterministic_full_song_visual_selector')
    return bundle,controller,request,model


@isolated_optional()
def wide_chart_propose(context):
    with wide_chart_sensors():
        return family.propose_performer_family.__wrapped__(context)


@isolated_optional()
def wide_chart_final(plan_json,receipt,deadline):
    with wide_chart_sensors():
        return source.final_live_replay.__wrapped__(plan_json,receipt,deadline)


class WidePolicyTests(unittest.TestCase):
    def cell(self,kind='wide',subject=None,active=False):
        return dict(segment_index=0,shot_type=kind,alignment_subject=subject,face_applied=active)

    def edge(self, **values):
        return dict(available=True,required_semantics_available=True,
            metrics=dict.fromkeys(policy.GEOMETRY_METRICS,.2)|values)

    def test_wide_without_focal_or_pose_declaration_is_composition(self):
        self.assertEqual(role_from_cells([self.cell()]),GroupOutputRole.COMPOSITION)

    def test_active_focal_duty_cannot_be_weakened_by_wide_label(self):
        self.assertEqual(role_from_cells([self.cell(subject='face',active=True)]),GroupOutputRole.FOCAL_HEAD)

    def test_unapplied_explicit_focal_duty_does_not_become_optional(self):
        self.assertEqual(role_from_cells([self.cell(subject='face',active=False)]),GroupOutputRole.FOCAL_HEAD)

    def test_active_untyped_crop_cannot_turn_into_composition_duty(self):
        with self.assertRaisesRegex(ValueError,'active output crop'):
            role_from_cells([self.cell(active=True)])

    def test_explicit_upper_body_and_formation_duties_remain(self):
        for subject,role in [('upper_body',GroupOutputRole.FOCAL_HEAD_ARMS),('formation',GroupOutputRole.FORMATION)]:
            self.assertEqual(role_from_cells([self.cell(subject=subject)]),role)

    def test_mixed_shot_roles_need_owned_split(self):
        with self.assertRaisesRegex(ValueError,'mixed'):
            role_from_cells([self.cell(),self.cell('medium')])

    def test_neighbor_pair_does_not_rewrite_whole_shot_intent(self):
        context=SimpleNamespace(incumbent=SimpleNamespace(
            visual_timeline=SimpleNamespace(choices=[SimpleNamespace(shot_type='wide')]),
            face_match_plan=SimpleNamespace(segments=[SimpleNamespace(alignment_subject=None,applied=False)]),
            editorial_motion_plan=SimpleNamespace(boundary_subject_alignment_plan=SimpleNamespace(
                decisions=[SimpleNamespace(eligible=True,boundary_index=0,subject_kind='upper_body')]))))
        shot=SimpleNamespace(first_segment_index=0,last_segment_index=0,first_output_frame=0,end_output_frame=60)
        self.assertEqual(source._shot_roles(context,[shot]),(GroupOutputRole.COMPOSITION,))

    def test_unknown_intent_is_not_source_dependent_fallback(self):
        with self.assertRaises(ValueError):role_from_cells([self.cell('unresolved')])

    def test_geometry_tradeoff_can_have_better_aggregate_utility(self):
        before=self.edge(occupancy=.05,attention=.8,scale=.8,span=.8,screen_motion=.8)
        after=self.edge(occupancy=.10,attention=.1,scale=.1,span=.1,screen_motion=.1)
        self.assertFalse(observations.preserves_known(before,after))
        verdict=policy.evaluate_composition_path(dict(edges=[before],state_keys=['baseline']),[after])
        self.assertTrue(verdict['admitted']);self.assertFalse(verdict['required_semantics_traded'])

    def test_unknown_required_evidence_is_not_purchasable(self):
        before=self.edge();after=self.edge(occupancy=0,attention=0,scale=0,span=0,screen_motion=0)
        after['required_semantics_available']=False
        self.assertFalse(policy.evaluate_composition_path(dict(edges=[before]),[after])['admitted'])

    def test_nan_or_missing_composition_metric_rejects(self):
        for value in (float('nan'),float('inf'),-1.,None):
            after=self.edge(attention=value)
            self.assertFalse(policy.evaluate_composition_path(dict(edges=[self.edge()]),[after])['admitted'])
        after=self.edge();del after['metrics']['scale']
        with self.assertRaises(ValueError):policy.composition_cost(after)

    def test_known_arm_metric_cannot_be_bought_with_framing(self):
        before=self.edge(arm_velocity=.1);after=self.edge(occupancy=0,attention=0,scale=0,span=0,screen_motion=0)
        self.assertFalse(policy.evaluate_composition_path(dict(edges=[before]),[after])['admitted'])

    def test_unknown_baseline_is_newly_observable_not_numerically_improved(self):
        verdict=policy.evaluate_composition_path(dict(edges=[None],state_keys=None),[self.edge()])
        self.assertTrue(verdict['admitted']);self.assertEqual(verdict['comparison_kind'],'newly_observable')
        self.assertIsNone(verdict['known_baseline_cost'])

    def test_baseline_never_combines_incompatible_middle_hypotheses(self):
        state=lambda key:SimpleNamespace(key=key)
        groups=[[state('L')],[state('A'),state('B')],[state('R')]]
        costs={('L','A'):.1,('A','R'):.9,('L','B'):.9,('B','R'):.1}
        reference=policy.coherent_path_reference(groups,lambda a,b:self.edge(
            **dict.fromkeys(policy.GEOMETRY_METRICS,costs[a.key,b.key])))
        self.assertIn(reference['state_keys'],[['L','A','R'],['L','B','R']])
        self.assertEqual([r['metrics']['attention'] for r in reference['edges']],[.1,.9])
        self.assertEqual(len(reference['alternatives']),2)

    def test_composition_scope_does_not_invent_required_arms(self):
        raw=dict(heads=[],people=[dict(observation_id=i,part_states=[]) for i in range(2)])
        scope=scene_role_obligations(raw,GroupOutputRole.COMPOSITION,None)
        self.assertEqual(scope.required_parts,());self.assertEqual(scope.required_head_ids,())
        self.assertEqual(scope.required_person_ids,(0,1))


class WideOwnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.paths=tuple(cls.root/f'headless-chart-{i}.mp4' for i in range(8))
        for i,path in enumerate(cls.paths):encode_wide_chart(path,variant=i,frames=540)
        cls.bundle,cls.controller,cls.request,cls.model=build_wide_fixture(cls.root,cls.paths)
        with optional_native_scope(monotonic()+600),patch.object(family,'installed_pose_model',return_value=cls.model):
            cls.context=family.freeze_context(controller=cls.controller,request=cls.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,source_derivatives=charts.scene_derivative_fixture())
        with optional_native_scope(cls.context.deadline),wide_chart_sensors():
            cls.result=scene.propose_scene_changes(cls.context)
        destination=os.environ.get('WIDE_TEST_ARTIFACT_DIR')
        if destination:
            Path(destination).mkdir(parents=True,exist_ok=True)
            value=(dict(error=cls.result.error,evidence=family.plain(cls.result.evidence))
                if isinstance(cls.result,family.PerformerFamilyRejection) else cls.result.to_dict())
            (Path(destination)/'wide-owner-result.json').write_text(json.dumps(value,indent=2))
            (Path(destination)/'wide-baseline.json').write_text(json.dumps(cls.bundle.to_dict(),indent=2))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def dense(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection,getattr(self.result,'error',None))
        return family.plain(self.result.editorial_motion_plan.source_camera_refresh)

    def test_headless_body_producer_reaches_complete_source_changed_family(self):
        dense=self.dense();family.validate_bundle_binding(self.result)
        self.assertTrue(all(r==GroupOutputRole.COMPOSITION.value for r in dense['roles']))
        self.assertEqual(dense['contract'],observations.DENSE_CONTRACT)
        self.assertTrue(any(a.source_index!=b.source_index for a,b in zip(
            self.bundle.render_plan.segments,self.result.render_plan.segments)))
        rows=[r for s in dense['mutable_shots'] for r in s['rows']]
        self.assertGreaterEqual(len(rows),60)
        self.assertTrue(all(not r['scene_heads'] and not r['projected_observed_arms'] for r in rows))
        self.assertTrue(all(len(r['scene_people'])==2 for r in rows))
        self.assertTrue(all(r['raw_face_roi_xywh'] is None for r in rows))
        self.assertTrue(all(r['native_observation']['contract']==composition.RAW_CONTRACT for r in rows))
        self.assertTrue(all(all(v is None for v in h['anchor_head_ids']) for h in dense['scene_hypotheses'].values()))
        receipt=family.plain(self.result.performer_family_review)
        self.assertTrue(receipt['search']['editorial_decision']['admitted'])
        self.assertEqual(set(receipt['native_work']['model_calls']),{'nanodet_people'})
        self.assertLessEqual(receipt['native_work']['counts']['dense_paths'],4)
        for key in ('cross_camera_identity_certified','body_phase_certified','quality_approved'):
            self.assertIs(dense[key],False)
        # Retained active focal crops/look away from this local wide repair.
        self.assertGreater(self.bundle.boundary_look_match.applied_count,0)
        self.assertEqual(self.result.face_match_plan.segments[4:],self.bundle.face_match_plan.segments[4:])
        self.assertEqual(self.result.render_plan.segments[4:],self.bundle.render_plan.segments[4:])

    def test_duplicate_projected_body_rejected(self):
        dense=self.dense();dense['mutable_shots'][0]['rows'][0]['scene_people'].append(
            deepcopy(dense['mutable_shots'][0]['rows'][0]['scene_people'][0]))
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_nan_visibility_rejected(self):
        dense=self.dense();dense['mutable_shots'][0]['rows'][0]['scene_people'][0]['visible_fraction']=float('nan')
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_lost_dense_native_support_path_rejected(self):
        dense=self.dense();dense['mutable_shots'][0]['rows'][1]['composition_support_link']=None
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_body_path_cannot_certify_articulated_motion(self):
        dense=self.dense();dense['mutable_shots'][0]['rows'][1]['composition_support_link']['articulated_motion_certified']=True
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_duplicate_assignment_or_wrong_native_pair_rejected(self):
        for mode in ('duplicate','clock'):
            dense=self.dense();link=dense['mutable_shots'][0]['rows'][1]['composition_support_link']
            if mode=='duplicate':link['assignments'][1]=link['assignments'][0]
            else:link['native_frame_indices'][0]-=1
            with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_rehashed_wide_role_cannot_hide_retained_focal_intent(self):
        dense=self.dense();key=dense['hypothesis_keys'][1];h=dense['scene_hypotheses'].pop(key)
        h['shot_intent']['cells'][0].update(alignment_subject='face',face_applied=True)
        new=family.digest(h);dense['scene_hypotheses'][new]=h;dense['hypothesis_keys'][1]=new
        for row in dense['mutable_shots'][0]['rows']:row['scene_hypothesis_sha256']=new
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_missing_exact_tick_and_fake_native_alias_rejected(self):
        dense=self.dense();dense['mutable_shots'][0]['rows'].pop(2)
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)
        dense=self.dense();dense['mutable_shots'][0]['rows'][1]['native_repeated_alias']=True
        with self.assertRaises(ValueError):observations.validate_scene_dense(dense)

    def test_final_live_replay_reobserves_actual_headless_body_paths(self):
        self.dense()
        with optional_native_scope(self.context.deadline):
            metrics=wide_chart_final(self.result.render_plan.to_json(),
                family.plain(self.result.performer_family_review),self.context.deadline)
        self.assertGreaterEqual(metrics['dense_native_output_tick_count'],60)
        self.assertIs(metrics['quality_approved'],False)
