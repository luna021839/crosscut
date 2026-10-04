"""Normal analyze -> source-changing family -> REAL encode -> ORIGINAL QA.

NON-HUMAN charts and substituted upstream/head descriptors are explicit.
This test does NOT import test_pipeline or any excluded capacity/MEMBER tests.
No original post-render QA, exposure rule, threshold or verdict is replaced.
"""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from crosscut import pipeline
from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.media import probe_media
from crosscut.render import render as actual_render, write_plan
from crosscut.source_derivative_diversity import SourceDerivativeConfig, SourceDerivativeDiversityResult
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from engine.tests.test_scene_source_owner import encode_scene_chart, build_scene_fixture, scene_derivative_fixture
from engine.tests.test_scene_source_owner import scene_chart_propose, scene_chart_final, scene_chart_sensors
from engine.tests.test_scene_source_owner import scene_chart_selected_replay
from crosscut.group_scene_observations import DENSE_CONTRACT

_REAL_PROXY=pipeline.render_and_review_proxy
_REAL_FINAL_QA=pipeline._enforce_automatic_stage_edit_post_render_qa


class Reporter:
    def __init__(self): self.rows=[]
    def log(self,*args,**kwargs): self.rows.append(('log',args,kwargs))
    def status(self,*args,**kwargs): self.rows.append(('status',args,kwargs))
    def progress(self,*args,**kwargs): self.rows.append(('progress',args,kwargs))
    def emit(self,*args,**kwargs): self.rows.append(('emit',args,kwargs))



@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local tools required')
class SceneSourceNormalAnalyzeTests(unittest.TestCase):
    def run_normal(self,*,corrupt_parent=False,external_original=False,absent_look=False,
            learned_completion_reject=False):
        with ExitStack() as stack:
            # macOS exposes /var as /private/var. AnalysisRequest canonicalizes
            # source paths, so the baseline fixture must use that SAME domain;
            # do not weaken the real controller's source-identity comparison.
            root=Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
            paths=tuple(root/f'source-{i}.mp4' for i in range(8))
            for i,path in enumerate(paths):
                encode_scene_chart(path,variant=i,frames=540)
            bundle,_,_,model=build_scene_fixture(root,paths,boundary_look_enabled=not absent_look)
            request=pipeline.AnalysisRequest.from_dict(dict(song='XOXZ',edit_mode='group',
                aspect_ratio='16:9',source_mode='automatic',review_first_enabled=True,
                sources=[str(p) for p in paths],output_directory=str(root/'out'),
                source_cache_directory=str(root/'cache'),experimental_pixel_camera_review=False))
            self.assertEqual(request.sources,paths)
            reference=None
            if external_original:
                from crosscut.group_external_original_reference import GroupExternalOriginalReference
                from crosscut.group_source_domain import GroupSourceDomain
                from crosscut.render_frame_schedule import probe_render_frame_clock
                reference=GroupExternalOriginalReference(bundle,GroupSourceDomain(paths,tuple(range(8)),
                    (1.,)*8,(None,)*8,((),)*8,tuple(probe_render_frame_clock(p) for p in paths)),((420,480),))
            self.original_reference_record=None if reference is None else reference.to_dict()
            alignment=AlignmentResult(reference=str(paths[0]),sample_rate=4000,
                alignments=tuple(SourceAlignment(source=str(p),is_reference=i==0,
                    offset_samples=4000,offset_seconds=1.,score=(.93,.91)[i%2],overlap_samples=44000,
                    overlap_seconds=11.,score_margin=1.,overlap_ratio=1.) for i,p in enumerate(paths)))
            draft=GroupDraftTimeline(source_count=8,common_master_start=0.,common_master_end=16.,
                beat_period_seconds=.5,audio_offsets=(1.,)*8,sync_scores=(.93,.91)*4,
                segments=tuple(DraftSegment(segment_index=i,source_index=i,canonical_master_start=i*2.,
                    canonical_master_end=(i+1)*2.,output_start_seconds=i*2.,source_local_start=i*2.+1.,
                    source_local_end=(i+1)*2.+1.,audio_offset_seconds=1.,sync_score=(.93,.91)[i%2],
                    estimated_beat_count=4.,start_boundary_reasons=('range_edge',),end_boundary_reasons=('phrase',))
                    for i in range(8)))
            rhythm=SimpleNamespace(bpm=120.,beats=(object(),),phrase_candidates=tuple(
                SimpleNamespace(start_seconds=i*2.,end_seconds=(i+1)*2.) for i in range(8)),
                beat_times_seconds=tuple(i*2. for i in range(9)),to_dict=lambda:{'explicit_chart_rhythm':True})
            # Strictly parsed synthetic montage profile, NOT human corpus evidence.
            from engine.tests.test_professional_montage_policy import (_profile,
                _install_positive_duration_role,_write)
            from crosscut.professional_montage_policy import load_professional_montage_policy
            profile=_profile()
            for role in ('build','bridge_like'):
                _install_positive_duration_role(profile,role_name=role,
                    close_values=[.20]*20,nonwide_values=[.60]*20)
            policy_path=root/'EXPLICIT_SYNTHETIC_MONTAGE_PROFILE.json'
            _write(policy_path,profile);policy=load_professional_montage_policy(policy_path)
            model_context=None
            if learned_completion_reject:
                from engine.tests.test_stage_edit_single_completion import synthetic_complete_generation
                generated,_=synthetic_complete_generation(root,sources=paths,section_plan=bundle.section_plan,
                    duration=16,source_offsets=(1.,)*8,width=320,height=180,fps=30.,prefer_long_spans=True)
                generated.write_json(root/'NONHUMAN-generation-audit.json')
                model_context=dict(bundle=None,motion=None,generation=generated,snapshot=None,
                    causal_audit={'explicit_synthetic_heads':True},policy=policy,motion_profile=None,
                    profile_payload=None,montage_config=pipeline._rendered_montage_config_from_policy(
                        policy,require_complete_role_duration_policy=True),
                    audit_path=root/'NONHUMAN-generation-audit.json')
                self.learned_materialization=stack.enter_context(patch(
                    'crosscut.pipeline.materialize_stage_edit_visual_candidate',
                    side_effect=ValueError('EXPLICIT_SYNTHETIC_SELECTED_COMPLETION_REJECTED')))
            fixtures=dict(_load_stage_edit_montage_policy=policy,_align_sequentially=(alignment,None),
                _common_master_range=(0.,16.),analyze_rhythm=rhythm,build_section_plan=bundle.section_plan,
                build_group_draft=draft,build_visual_timeline=bundle.visual_timeline,
                repair_visual_timeline_boundaries=VisualBoundaryRepairPlan(bundle.visual_timeline,(bundle.visual_timeline,),(),1,()),
                _review_first_registered_group_generation=model_context,analyze_source_derivative_diversity=scene_derivative_fixture(),
                detect_scene_changes_in_video_cached=((),{'detection_policy':'explicit_chart_fixture'}),
                analyze_source_look_match=None,build_boundary_affine_plan=None,build_boundary_subject_alignment_plan=None,
                build_editorial_motion_plan=None,build_group_transition_plan=None)
            for name,value in fixtures.items():stack.enter_context(patch('crosscut.pipeline.'+name,return_value=value))
            stack.enter_context(patch('crosscut.pipeline.probe_media',side_effect=probe_media))
            stack.enter_context(patch('crosscut.pipeline.output_size',return_value=(320,180)))
            stack.enter_context(patch('crosscut.pipeline.build_face_match_plan',side_effect=ValueError('chart baseline supplies explicit existing face family')))
            stack.enter_context(patch('crosscut.pipeline._review_first_group_bundle',side_effect=lambda **kw:
                replace(bundle,render_plan=replace(bundle.render_plan,output=kw['output']))))
            self.look_authoring=stack.enter_context(patch('crosscut.pipeline._review_first_group_boundary_look_bundle',
                wraps=pipeline._review_first_group_boundary_look_bundle))
            stack.enter_context(patch.object(family,'installed_pose_model',return_value=model))
            calls=[]
            def proposer(context):
                candidate=scene_chart_propose(context);calls.append(candidate)
                if corrupt_parent and len(calls)==1 and not isinstance(candidate,family.PerformerFamilyRejection):
                    receipt=candidate.to_dict()['performer_family_review']
                    receipt['final_observations']['samples'][0]['native_pixels_sha256']='0'*64
                    return replace(candidate,performer_family_review=receipt)
                return candidate
            stack.enter_context(patch.object(family,'propose_performer_family',side_effect=proposer))
            from crosscut import group_scene_selected_replay as selected_replay
            stack.enter_context(patch.object(selected_replay,'replay_selected_scene',new=scene_chart_selected_replay))
            stack.enter_context(patch.object(source,'final_live_replay',new=scene_chart_final))
            stack.enter_context(patch('crosscut.pipeline._apply_bounded_human_target_proxy_rank'))
            def montage(**kw):
                analysis=pipeline._stage_edit_montage_analysis_payload(**{k:kw[k] for k in (
                    'bundle','proxy_review_payload','section_plan','source_derivative_diversity','source_index_mapping')})
                return {'passed':False,'weighted_score':45.,'explicit_chart_measurement':True},analysis
            stack.enter_context(patch('crosscut.pipeline._review_stage_edit_bundle_montage',side_effect=montage))
            # No local sync modification/admission is claimed by this fixture.
            stack.enter_context(patch('crosscut.pipeline.analyze_cut_local_sync',return_value=dict(
                passed=False,retained_cross_source_boundary_count=4,measured_boundary_count=0,
                unknown_boundary_count=4,residual_p95_frames=None,explicit_chart_unmeasured=True)))
            rendered=[]
            def render_real(plan,**kw):
                result=actual_render(plan,video_codec='libx264',video_bitrate='1M',video_maxrate='2M',
                    video_buffer_size='2M',**kw)
                write_plan(plan,plan.output.with_suffix('.plan.json'));rendered.append(plan)
                return result
            def proxy_real(plan,output,**kw):
                new,review=_REAL_PROXY(plan,Path(output),**kw);rendered.append(new);return new,review
            stack.enter_context(patch('crosscut.pipeline.render',side_effect=render_real))
            stack.enter_context(patch('crosscut.pipeline.render_and_review_proxy',side_effect=proxy_real))
            stack.enter_context(patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')))
            self.assertIs(pipeline._enforce_automatic_stage_edit_post_render_qa,_REAL_FINAL_QA)
            failure=None;reporter=Reporter()
            with scene_chart_sensors():
                try:pipeline.analyze(request,reporter,review_rendered_proxies=True,
                    group_external_original_reference=reference)
                except RuntimeError as exc:failure=str(exc)
            manifests=[];qa=[]
            for path in request.output_directory.rglob('*.json'):
                if any(part.startswith('.') for part in path.relative_to(request.output_directory).parts):continue
                try:value=json.loads(path.read_text())
                except (ValueError,OSError):continue
                if isinstance(value,dict) and 'rendered_candidate_selection' in value and 'request' in value:
                    manifests.append((path,value))
                if isinstance(value,dict) and 'verdict' in value and 'issues' in value and 'metrics' in value:qa.append((path,value))
            selected=next(((p,v) for p,v in manifests if 'app_result_exposure' in v
                and v.get('artifact_role')!='pending_qa'),manifests[0] if manifests else (None,None))
            destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
            if destination:
                target=Path(destination)/('normal-scene-negative' if corrupt_parent else 'normal-scene-positive')
                target.mkdir(parents=True,exist_ok=True)
                for p in request.output_directory.rglob('*.json'):
                    relative=p.relative_to(request.output_directory)
                    if any(part.startswith('.') for part in relative.parts):continue
                    dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                for p in request.output_directory.rglob('*.mp4'):
                    dest=target/'CHART_MEDIA'/p.name
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                (target/'RUN.json').write_text(json.dumps(dict(runtime_error=failure,proposer_calls=len(calls),
                    original_qa_unmodified=True,sensor_evidence='EXPLICIT_NON_HUMAN_CHART_SUBSTITUTES',
                    model_inference_performed=False,declared_synthetic_derivative_families=8,actual_derivative_classification_performed=False,
                    real_encoded_outputs=[dict(path=str(p.output),frames=p.output_frame_count,
                        width=p.width,height=p.height) for p in rendered],logs=reporter.rows),indent=2))
            self.assertIsNotNone(selected[1],(failure,reporter.rows[-20:]))
            return bundle,selected[1],qa,rendered,calls,failure

    def test_normal_learned_failure_completes_one_selected_rank_not_eight_then_keeps_original_qa(self):
        _,manifest,reports,rendered,_,failure=self.run_normal(learned_completion_reject=True)
        self.assertEqual(self.learned_materialization.call_count,1,failure)
        selected=self.learned_materialization.call_args.args[0]
        receipt=manifest['review_first_visual_selection']['complete_path_numeric_selection']
        self.assertEqual(receipt['selected_rank'],selected.rank)
        self.assertEqual(len(receipt['compared_complete_paths']),8)
        self.assertEqual(receipt['maximum_selected_completions'],1)
        self.assertFalse(receipt['native_scene_selection_performed'])
        self.assertFalse(receipt['quality_approved'])
        self.assertGreaterEqual(len(rendered),2,failure)
        self.assertTrue(reports,failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])

    def test_external_original_absence_is_retained_before_normal_authoring_through_final_replay(self):
        baseline,manifest,reports,rendered,calls,failure=self.run_normal(external_original=True,absent_look=True)
        self.assertEqual(self.look_authoring.call_count,0)
        self.assertEqual(len(calls),1,failure)
        self.assertGreaterEqual(len(rendered),2,failure)
        chosen=manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertEqual(chosen['candidate_origin'],source.CONTRACT,json.dumps(dict(runtime_error=failure,
            proposal_errors=[c.error if isinstance(c,family.PerformerFamilyRejection) else None for c in calls],
            attempts=manifest.get('review_first_visual_selection',{}).get('first_result_controller',{}).get('attempts'))))
        self.assertIsNone(chosen['boundary_look_match'])
        self.assertTrue(all(s.boundary_look_track is None for p in rendered for s in p.segments))
        receipt=chosen['performer_family_review']
        self.assertEqual(receipt['external_original_reference'],self.original_reference_record)
        self.assertEqual(receipt['external_original_reference']['mode'],'ABSENT_PRESERVED')
        self.assertEqual(receipt['look_refresh']['refreshed_edges'],[])
        self.assertFalse(receipt['look_refresh']['matched_production_renderer_used'])
        controller=manifest['review_first_visual_selection']['first_result_controller']
        self.assertEqual(controller['external_original_reference'],self.original_reference_record)
        self.assertTrue(reports,failure)
        metrics=reports[-1][1]['metrics'].get('performer_family_review')
        self.assertTrue(family.qa_metrics_valid(metrics),metrics)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        self.assertTrue(all(s.look_adjustment is not None for s in baseline.render_plan.segments))

    def test_fresh_normal_absent_baseline_still_allows_explicit_look_authoring(self):
        _,_,_,rendered,_,failure=self.run_normal(absent_look=True)
        self.assertEqual(self.look_authoring.call_count,1,failure)
        self.assertGreaterEqual(len(rendered),2,failure)

    def test_source_change_reaches_real_normal_renderer_and_original_failing_qa(self):
        baseline,manifest,reports,rendered,calls,failure=self.run_normal()
        self.assertEqual(len(calls),1,failure)
        self.assertGreaterEqual(len(rendered),2,failure)
        chosen=manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertEqual(chosen['candidate_origin'],source.CONTRACT,failure)
        self.assertEqual(chosen['performer_family_review']['source_camera_refresh']['contract'],DENSE_CONTRACT)
        self.assertGreater(baseline.boundary_look_match.applied_count,0)
        self.assertGreater(chosen['boundary_look_match']['applied_boundary_count'],0)
        self.assertTrue(all(s.look_adjustment is not None for s in baseline.render_plan.segments))
        for key in ('face_match','editorial_motion','boundary_look_match'):
            self.assertEqual(chosen[key],manifest['render'][key])
        self.assertTrue(any(a.source_index!=b.source_index for a,b in zip(baseline.render_plan.segments,rendered[-1].segments)))
        self.assertTrue(reports,failure)
        report=reports[-1][1];metrics=report['metrics'].get('performer_family_review')
        self.assertTrue(family.qa_metrics_valid(metrics),metrics)
        self.assertEqual(metrics['contract'],source.CONTRACT)
        self.assertIn(family.QUALITY_CODE,{i['code'] for i in report['issues']})
        self.assertFalse({i['code'] for i in report['issues']} & {
            'editorial_motion_aggregate_structural_bounds',
            'editorial_motion_zoom_event_count_matches_reason_events'})
        self.assertFalse(report['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        print('NORMAL_SCENE_POSITIVE',failure,[(p.output_frame_count,p.width,p.height) for p in rendered],metrics)

    def test_tampered_child_preserves_original_full_families_through_real_normal_renderer(self):
        baseline,manifest,reports,rendered,calls,failure=self.run_normal(corrupt_parent=True)
        self.assertEqual(len(calls),1,failure)
        self.assertGreaterEqual(len(rendered),2,failure)
        chosen=manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertNotEqual(chosen['candidate_origin'],source.CONTRACT)
        for key in ('face_match','editorial_motion','boundary_look_match'):
            self.assertEqual(baseline.to_dict()[key],chosen[key])
        self.assertEqual([s.source_index for s in baseline.render_plan.segments],
            [s.source_index for s in rendered[-1].segments])
        self.assertEqual(baseline.render_plan.segments,rendered[-1].segments)
        self.assertTrue(reports,failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        print('NORMAL_SCENE_NEGATIVE',failure)
