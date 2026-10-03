"""Real source-clock/camera/family transactions on explicit NON-HUMAN charts.

Only source/screen descriptors, head/pose model inference and native detector
variants are substituted. Decoders, schedule, state product, family refresh,
matched look encoder, independent workers, controller CAS, and QA are real.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut.group_joint_native_observer import NativeGroupHead,NativeGroupObservation
from crosscut.optional_native_worker import isolated_optional,optional_native_scope
from engine.tests.test_performer_family_owner import (encode_chart,build_fixture,
    box_from_chart,chart_sensors,ChartProvider)

_REAL_PROPOSE=family.propose_performer_family.__wrapped__
_REAL_SOURCE_FINAL=source.final_live_replay.__wrapped__


def chart_native(session,*,request_source_index,native_frame_index,head_analysis_roll=None):
    session.budget.spend('model_calls',model='EXPLICIT_SYNTHETIC_CHART_NOT_YUNET_OR_RTMPOSE')
    pixels=session.native_pixels(request_source_index,native_frame_index)
    box=box_from_chart(pixels);heads=()
    if box is not None:
        x,y,w,h=box;cx=x+w/2;cy=y+h/2
        points=((cx-8/320,cy-8/180),(cx+8/320,cy-8/180),(cx,cy),
            (cx-8/320,cy+9/180),(cx+8/320,cy+9/180))
        heads=(NativeGroupHead(0,box,points,.99),)
    clock=session.domain.clocks_by_request[request_source_index]
    return NativeGroupObservation(request_source_index,native_frame_index,
        (pixels.shape[1],pixels.shape[0]),family.digest(clock.to_dict()),
        source._file_identity(session.domain.request_sources[request_source_index]),
        source.sha256(memoryview(pixels)).hexdigest(),(('CHART_SENSOR','a'*64),),(),heads,
        'EXPLICIT_SYNTHETIC_CHART_NOT_HUMAN_MODEL_EVIDENCE',(('CHART_SENSOR','b'*64),))


@contextmanager
def source_chart_sensors(model=None):
    with ExitStack() as stack:
        stack.enter_context(chart_sensors())
        stack.enter_context(patch.object(source,'installed_native_assets',side_effect=lambda c:dict(
            person_model_path=c.model_path,pose_model_path=c.model_path,head_model_path=c.model_path)))
        stack.enter_context(patch.object(source,'register_joint_native_models',return_value=None))
        stack.enter_context(patch.object(source,'register_native_head_orientation_models',return_value=None))
        stack.enter_context(patch.object(source,'observe_native_group',side_effect=chart_native))
        # Explicit detector/window sensor substitute, NOT production YuNet or
        # real landmark-flow evidence. Native window policy has its own tests.
        def chart_window(session,*,request_source_index,native_frame_indices):
            rows=[chart_native(session,request_source_index=request_source_index,
                native_frame_index=tick) for tick in native_frame_indices]
            return SimpleNamespace(selected=all(row.heads for row in rows),
                selected_analysis_roll_degrees=None,reason='explicit_synthetic_chart_window',
                to_dict=lambda:dict(sensor='EXPLICIT_NON_HUMAN_CHART_WINDOW',
                    observations=[row.to_dict() for row in rows],quality_approved=False))
        stack.enter_context(patch.object(source,'observe_native_head_window_policy',side_effect=chart_window))
        yield


@isolated_optional()
def source_chart_propose(context):
    with source_chart_sensors():
        return _REAL_PROPOSE(context)


@isolated_optional()
def source_chart_final(plan_json,receipt,deadline):
    with source_chart_sensors():
        return _REAL_SOURCE_FINAL(plan_json,receipt,deadline)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local tools required')
class SourceChangingFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory();cls.root=Path(cls.storage.name)
        cls.sources=tuple(cls.root/f'source-{i}.mp4' for i in range(8))
        for i,p in enumerate(cls.sources):
            if i<2:encode_chart(p,variant=i)
            else:shutil.copyfile(cls.sources[i%2],p)
        cls.bundle,cls.controller,cls.request,cls.model=build_fixture(cls.root,cls.sources)
        with optional_native_scope(monotonic()+300.),patch.object(family,'installed_pose_model',return_value=cls.model):
            cls.context=family.freeze_context(controller=cls.controller,request=cls.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4)
        with optional_native_scope(monotonic()+300.),source_chart_sensors():
            cls.result=source.propose_source_changes(cls.context)
        if isinstance(cls.result,family.PerformerFamilyRejection):
            (cls.root/'source-debug-rejection.json').write_text(json.dumps(
                dict(error=cls.result.error,evidence=family.plain(cls.result.evidence)),indent=2))

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()

    def test_eight_sources_produce_actual_source_change_and_all_active_families(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection,
            getattr(self.result,'error',None))
        result=self.result;receipt=result.to_dict()['performer_family_review']
        family.validate_bundle_binding(result)
        self.assertEqual(result.candidate_origin,source.CONTRACT)
        self.assertTrue(receipt['source_changes_supported'])
        self.assertTrue(any(a.source_index!=b.source_index for a,b in
            zip(self.bundle.render_plan.segments,result.render_plan.segments)))
        self.assertEqual(result.face_match_plan.applied_count,5)
        self.assertGreater(result.boundary_look_match.applied_count,0)
        self.assertEqual(result.boundary_look_match.decisions[3],self.bundle.boundary_look_match.decisions[3])
        self.assertIsNotNone(result.editorial_motion_plan.source_camera_refresh)
        self.assertFalse(receipt['quality_approved'])
        self.assertFalse(receipt['source_camera_refresh']['semantic_raw_v2_admission'])
        generated=receipt['search']['generated_numeric_states']
        for slot in {r['slot'] for r in generated}:
            self.assertEqual({r['source'] for r in generated if r['slot']==slot},set(range(8)))
        print('SOURCE_POSITIVE',receipt['changed_cells'],receipt['refreshed_cells'],receipt['native_work'])

    def test_zero_candidate_rejection_keeps_unavailable_native_discovery_evidence(self):
        def unavailable(session,*,request_source_index,native_frame_indices):
            # Actual decoded chart observations, explicitly not human NN data.
            rows=[chart_native(session,request_source_index=request_source_index,
                native_frame_index=tick).to_dict() for tick in native_frame_indices]
            return SimpleNamespace(selected=False,selected_analysis_roll_degrees=None,
                reason='explicit_chart_discovery_unavailable',to_dict=lambda:dict(
                    sensor='EXPLICIT_NON_HUMAN_CHART_WINDOW',observations=rows,quality_approved=False))
        with optional_native_scope(self.context.deadline),source_chart_sensors(), \
                patch.object(source,'observe_native_head_window_policy',side_effect=unavailable):
            rejected=source.propose_source_changes(self.context)
        self.assertIsInstance(rejected,family.PerformerFamilyRejection)
        evidence=family.plain(rejected.evidence)
        self.assertEqual(evidence['candidate_alternatives'],[])
        self.assertTrue(evidence['pre_camera_native_declarations'])
        for row in evidence['pre_camera_native_declarations']:
            self.assertFalse(row['available'])
            self.assertTrue(row['native_head_window_policy']['observations'])
            self.assertFalse(row['native_head_window_policy']['quality_approved'])
        self.assertFalse(evidence['quality_approved'])

    def test_full_owner_independent_parent_single_cas_and_live_final(self):
        from crosscut.first_result_controller import FirstResultController
        from crosscut.review_first_runtime import ReviewFirstRuntime
        controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=self.sources,
            source_durations=(12.,)*8,master_start=0.,duration=10.,journal_path=self.root/'source-cas.json')
        controller.install_baseline(self.bundle,plan=self.bundle.render_plan,authority='deterministic_full_song_visual_selector')
        token=controller.transaction_token();calls=[]
        def propose(context):
            candidate=source_chart_propose(context);calls.append(candidate);return candidate
        with patch.object(family,'installed_pose_model',return_value=self.model), \
                patch.object(family,'propose_performer_family',side_effect=propose), \
                patch.object(controller,'compare_and_promote',wraps=controller.compare_and_promote) as cas:
            candidate=family.attempt_performer_family(controller=controller,request=self.request,
                eligible_indices=tuple(range(8)),offsets=(1.,)*8,timewarps=(None,)*8,
                scene_cut_times=((),)*8,sync_scores=(.93,.91)*4,archive_path=self.root/'source-transaction.json')
        self.assertIsNotNone(candidate,controller.to_dict())
        self.assertEqual(candidate.candidate_origin,source.CONTRACT)
        self.assertEqual(len(calls),2)
        self.assertEqual(cas.call_count,1)
        self.assertEqual(controller.transaction_token().revision,token.revision+1)
        # Durable archives retain the lossless native wire view. Compare the
        # complete expanded schema, not the storage packet to a public bundle.
        archive=family.plain(json.loads((self.root/'source-transaction.json').read_text()))
        self.assertEqual(archive['original_bundle'],self.bundle.to_dict())
        self.assertEqual(archive['candidate_bundle'],candidate.to_dict())
        receipt=candidate.to_dict()['performer_family_review']
        with patch.object(source,'final_live_replay',new=source_chart_final):
            metrics=family.replay_final_sources(candidate.render_plan.to_json(),receipt)
        self.assertTrue(family.qa_metrics_valid(metrics),metrics)
        self.assertEqual(metrics['dense_native_output_tick_count'],120)
        self.assertFalse(metrics['quality_approved'])
        import os
        destination=os.environ.get('SOURCE_TEST_ARTIFACT_DIR')
        if destination:
            root=Path(destination);root.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.root/'source-transaction.json',root/'source-transaction.json')
            (root/'FINAL_LIVE_REPLAY.json').write_text(json.dumps(metrics,indent=2))
            (root/'SOURCE_SELECTED_PLAN.json').write_text(json.dumps(candidate.render_plan.to_json(),indent=2))
            (root/'SOURCE_SELECTED_BUNDLE.json').write_text(json.dumps(candidate.to_dict(),indent=2))
        print('SOURCE_PARENT_CAS_FINAL',metrics)

    def test_resigned_metadata_still_needs_independent_sensor_replay(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        receipt=self.result.to_dict()['performer_family_review']
        receipt['final_observations']['samples'][0]['native_pixels_sha256']='0'*64
        candidate=replace(self.result,performer_family_review=receipt)
        # A structurally well-formed changed observation is not proof. Independent
        # parent native/metadata replay MUST disagree even though binding passes.
        family.validate_bundle_binding(candidate)
        with optional_native_scope(self.context.deadline),patch.object(family,'propose_performer_family',new=source_chart_propose):
            with self.assertRaisesRegex(ValueError,'independent complete sensor/family replay disagrees'):
                source.validate_parent(candidate,self.context)

    def test_missing_dense_tick_cannot_be_resigned_as_complete(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        dense=self.result.to_dict()['editorial_motion']['source_camera_refresh']
        dense['mutable_shots'][0]['rows'].pop(10)
        candidate=replace(self.result,editorial_motion_plan=replace(self.result.editorial_motion_plan,
            source_camera_refresh=dense))
        receipt=candidate.to_dict()['performer_family_review'];receipt['source_camera_refresh']=dense
        receipt['metadata_sha256']=family.digest(family.metadata_payload(candidate))
        candidate=replace(candidate,performer_family_review=receipt)
        with self.assertRaisesRegex(ValueError,'omitted an output tick'):family.validate_bundle_binding(candidate)

    def test_metadata_flags_and_dense_family_cannot_disappear(self):
        self.assertNotIsInstance(self.result,family.PerformerFamilyRejection)
        with self.assertRaises(ValueError):family.validate_bundle_binding(replace(self.result,performer_family_review=None))
        with self.assertRaises(ValueError):family.validate_bundle_binding(replace(self.result,editorial_motion_plan=replace(
            self.result.editorial_motion_plan,source_camera_refresh=None)))
        for name in ('quality_approved','cross_camera_identity_certified','body_phase_certified','decoded_output_quality_certified'):
            receipt=self.result.to_dict()['performer_family_review'];receipt[name]=True
            with self.subTest(name=name),self.assertRaises(ValueError):
                family.validate_bundle_binding(replace(self.result,performer_family_review=receipt))
