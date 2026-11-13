"""Actual analyze -> retained owner -> encoded render -> unmodified exposure QA.

Upstream music/acquisition and prior full-family baseline are fixtures. Native
sensor outputs use explicitly simulated charts, never actual person inference.
No final QA/exposure function or quality threshold is mocked to a passed flag.
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
from crosscut import group_performer_family_owner as owner
from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.media import probe_media
from crosscut.render import render as actual_render, write_plan
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from engine.tests.test_performer_family_owner import encode_chart, build_fixture, chart_propose, chart_final, chart_sensors
from engine.tests.test_pipeline import _MemoryReporter, _source_derivative_diversity_fixture

_REAL_FINAL_QA = pipeline._enforce_automatic_stage_edit_post_render_qa
_REAL_PROXY = pipeline.render_and_review_proxy


class PerformerNormalAnalyzeTests(unittest.TestCase):
    def run_normal(self, *, corrupt_parent=False):
        from engine.tests.test_group_learned_production_transaction import GroupLearnedProductionTransactionTests
        executable_path=os.environ.get('PATH',os.defpath)
        destination=os.environ.get('PERFORMER_TEST_ARTIFACT_DIR')
        h=GroupLearnedProductionTransactionTests();h.setUp();self.addCleanup(h.doCleanups)
        h.stack.enter_context(patch.dict(os.environ,{'PATH':executable_path}))
        sources=h.sources[:8]
        for i,path in enumerate(sources):
            if i<2:encode_chart(path,variant=i)
            else:shutil.copyfile(sources[i%2],path)
        bundle,_,_,model=build_fixture(h.root,sources)
        h.request=replace(h.request,sources=sources,source_cache_directory=h.root/'cache',
            experimental_pixel_camera_review=False)
        alignment=AlignmentResult(reference=str(sources[0]),sample_rate=4000,
            alignments=tuple(SourceAlignment(source=str(path),is_reference=i==0,
                offset_samples=4000,offset_seconds=1.,score=(.93,.91)[i%2],overlap_samples=44000,
                overlap_seconds=11.,score_margin=1.,overlap_ratio=1.) for i,path in enumerate(sources)))
        draft=GroupDraftTimeline(source_count=8,common_master_start=0.,common_master_end=10.,
            beat_period_seconds=.5,audio_offsets=(1.,)*8,sync_scores=(.93,.91)*4,
            segments=tuple(DraftSegment(segment_index=i,source_index=i%2,canonical_master_start=i*2.,
                canonical_master_end=(i+1)*2.,output_start_seconds=i*2.,source_local_start=i*2.+1,
                source_local_end=(i+1)*2.+1,audio_offset_seconds=1.,sync_score=(.93,.91)[i%2],
                estimated_beat_count=4.,start_boundary_reasons=('range_edge',),end_boundary_reasons=('phrase',))
                for i in range(5)))
        rhythm=SimpleNamespace(bpm=120.,beats=(object(),),phrase_candidates=tuple(
            SimpleNamespace(start_seconds=i*2.,end_seconds=(i+1)*2.) for i in range(5)),
            beat_times_seconds=(0.,2.,4.,6.,8.,10.),to_dict=lambda:{'fixture':True})
        for name,value in {'_align_sequentially':(alignment,None),'_common_master_range':(0.,10.),'analyze_rhythm':rhythm,
            'build_section_plan':bundle.section_plan,'build_group_draft':draft,
            'build_visual_timeline':bundle.visual_timeline,
            'repair_visual_timeline_boundaries':VisualBoundaryRepairPlan(bundle.visual_timeline,(bundle.visual_timeline,),(),1,()),
            '_review_first_registered_group_generation':None,
            'analyze_source_derivative_diversity':_source_derivative_diversity_fixture(8)}.items():
            h.stack.enter_context(patch('crosscut.pipeline.'+name,return_value=value))
        h.stack.enter_context(patch('crosscut.pipeline.probe_media',side_effect=probe_media))
        h.stack.enter_context(patch('crosscut.pipeline.output_size',return_value=(320,180)))
        def retained_baseline(**kwargs):
            return replace(bundle,render_plan=replace(bundle.render_plan,output=kwargs['output']))
        h.stack.enter_context(patch('crosscut.pipeline._review_first_group_bundle',side_effect=retained_baseline))
        h.stack.enter_context(patch.object(owner,'installed_pose_model',return_value=model))
        calls=[]
        def proposer(context):
            candidate=chart_propose(context)
            calls.append(candidate)
            if corrupt_parent and len(calls)==1 and not isinstance(candidate,owner.PerformerFamilyRejection):
                receipt=candidate.to_dict()['performer_family_review']
                receipt['final_observations']['samples'][0]['native_pixels_sha256']='0'*64
                return replace(candidate,performer_family_review=receipt)
            return candidate
        h.stack.enter_context(patch.object(owner,'propose_performer_family',side_effect=proposer))
        h.stack.enter_context(patch.object(owner,'_final_live_replay',new=chart_final))
        h.stack.enter_context(patch('crosscut.pipeline._enforce_automatic_stage_edit_post_render_qa',new=_REAL_FINAL_QA))
        # This wrapper delegates to the REAL renderer, preserving actual errors.
        rendered=[]
        def render_real(plan,**kwargs):
            result=actual_render(plan,video_codec='libx264',video_bitrate='1M',video_maxrate='2M',
                video_buffer_size='2M',**kwargs)
            write_plan(plan,plan.output.with_suffix('.plan.json'));rendered.append(plan)
            return result
        h.render.side_effect=render_real
        def proxy_real(plan,output,**kwargs):
            proxy_plan,review=_REAL_PROXY(plan,Path(output),**kwargs)
            rendered.append(proxy_plan)
            return proxy_plan,review
        h.proxy.side_effect=proxy_real
        failure=None
        # No assets may be acquired by any unexpected downstream path.
        h.stack.enter_context(patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden in chart integration')))
        with chart_sensors():
            try:pipeline.analyze(h.request,_MemoryReporter(),review_rendered_proxies=True)
            except RuntimeError as exc:failure=str(exc)
        output=h.request.output_directory
        manifests=[]
        for path in output.glob('*.json'):
            try:payload=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if isinstance(payload,dict) and 'rendered_candidate_selection' in payload and 'request' in payload:
                manifests.append((path,payload))
        self.assertTrue(manifests, (failure,[p.name for p in output.glob('*')]))
        # Prefer the original analysis, not the QA frozen analysis snapshot.
        chosen=next(((p,v) for p,v in manifests if 'app_result_exposure' in v
            and v.get('artifact_role')!='pending_qa'),manifests[0])
        path,manifest=chosen
        qa=[]
        for item in output.glob('*.json'):
            try:value=json.loads(item.read_text())
            except (OSError,ValueError):continue
            if isinstance(value,dict) and 'verdict' in value and 'issues' in value and 'metrics' in value:
                qa.append((item,value))
        if destination:
            target=Path(destination)/('normal-negative' if corrupt_parent else 'normal-positive')
            target.mkdir(parents=True,exist_ok=True)
            for item in output.rglob('*.json'):
                relative=item.relative_to(output)
                if not any(part.startswith('.') for part in relative.parts):
                    dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,dest)
            (target/'RUN.json').write_text(json.dumps(dict(runtime_error=failure,
                original_full_families=dict(face=bundle.face_match_plan.applied_count,
                    look=bundle.boundary_look_match.applied_count,
                    motion=sum(s.track is not None for s in bundle.editorial_motion_plan.segments)),
                actual_encoded_outputs=[dict(path=str(p.output),frames=p.output_frame_count,width=p.width,height=p.height)
                    for p in rendered],proposer_call_count=len(calls),
                final_exposure_wrapper='UNMODIFIED_PRODUCTION_FUNCTION',sensor_evidence='EXPLICIT_SYNTHETIC_SUBSTITUTES'),indent=2))
        return bundle,manifest,qa,rendered,calls,failure

    def test_positive_normal_analyze_real_encoded_qa_unapproved(self):
        bundle,manifest,reports,rendered,calls,failure=self.run_normal()
        self.assertGreaterEqual(len(calls),2, failure)
        self.assertGreaterEqual(len(rendered),2, failure)
        complete=manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertEqual(complete['candidate_origin'],owner.CONTRACT, failure)
        self.assertEqual(complete['face_match'],manifest['render']['face_match'])
        self.assertEqual(complete['editorial_motion'],manifest['render']['editorial_motion'])
        self.assertEqual(complete['boundary_look_match'],manifest['render']['boundary_look_match'])
        from crosscut.review_first_selection_contract import review_first_release_binding_valid
        self.assertTrue(review_first_release_binding_valid(manifest['rendered_candidate_selection'],manifest['render']))
        self.assertTrue(reports, failure)
        report=reports[-1][1]
        self.assertTrue(owner.qa_metrics_valid(report['metrics'].get('performer_family_review')),report['metrics'].get('performer_family_review'))
        self.assertIn(owner.QUALITY_CODE,{i['code'] for i in report['issues']})
        self.assertFalse(report['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        print('NORMAL_ANALYZE_POSITIVE',manifest['artifact_role'],failure,sorted(i['code'] for i in report['issues']))

    def test_negative_normal_analyze_replay_rejection_keeps_existing_families(self):
        bundle,manifest,reports,rendered,calls,failure=self.run_normal(corrupt_parent=True)
        self.assertGreaterEqual(len(calls),2,failure)
        complete=manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertIsNone(complete['performer_family_review'])
        self.assertEqual(complete['face_match'],bundle.face_match_plan.to_dict())
        self.assertEqual(complete['editorial_motion'],bundle.editorial_motion_plan.to_dict())
        self.assertEqual(complete['boundary_look_match'],bundle.boundary_look_match.to_dict())
        self.assertTrue(reports,failure)
        attempts=manifest['review_first_visual_selection']['first_result_controller']['attempts']
        self.assertTrue(any(a['stage']=='normal_performer_retained_family_owner' and a['status']=='rejected'
            and 'replay disagrees' in str(a) for a in attempts),attempts)
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        print('NORMAL_ANALYZE_NEGATIVE',manifest['artifact_role'],failure)
