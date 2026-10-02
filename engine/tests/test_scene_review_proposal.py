"""No human footage/inference. Real native clocks, child/parent, FFmpeg and QA.

The typed derivative audit and original upstream descriptors are explicit
NONHUMAN fixtures. The new trial itself uses no detector/model substitutes.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from crosscut import group_scene_review_proposal as trial
from crosscut import group_performer_family_owner as family
from crosscut.optional_native_worker import optional_native_scope
from engine.tests.test_wide_composition_owner import build_wide_fixture, encode_wide_chart
from engine.tests.test_scene_source_owner import scene_derivative_fixture
from engine.tests.test_scene_normal_analyze import Reporter


class SceneReviewPolicyTests(unittest.TestCase):
    def test_clock_publication_rejects_stale_video_or_plan_without_overwriting(self):
        from crosscut.render_clock_artifact import render_clock_path, RENDER_CLOCK_ARTIFACT_CONTRACT
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);staging=root/'trial.unpublished.mp4';video=root/'trial.mp4'
            plan={'output':str(staging),'fixture':'clock-publication-only'}
            clock=dict(artifact_contract=RENDER_CLOCK_ARTIFACT_CONTRACT,
                video_content_sha256='a'*64,quality_approval=False,plan=plan)
            render_clock_path(staging).write_text(json.dumps(clock))
            for candidate_plan,video_hash in ((dict(plan,fixture='different'),'a'*64),(plan,'b'*64)):
                with self.subTest(candidate_plan=candidate_plan,video_hash=video_hash),self.assertRaises(ValueError):
                    trial._publish_review_clock(staging,video,candidate_plan,video_hash)
                self.assertFalse(render_clock_path(video).exists())
            result=trial._publish_review_clock(staging,video,plan,'a'*64)
            self.assertEqual(json.loads(render_clock_path(video).read_text())['plan'],dict(plan,output=str(video)))
            self.assertEqual(result['render_clock_sha256'],family._file_hash(render_clock_path(video)))
            with self.assertRaisesRegex(ValueError,'already exists'):
                trial._publish_review_clock(staging,video,plan,'a'*64)

    def test_flat_spatial_support_is_unknown_not_zero(self):
        value=trial.scene_descriptor(np.full((54,96,3),80,np.uint8))
        self.assertIsNone(value['spatial_texture'])
        edge=trial.scene_edge(value,value)
        self.assertEqual(edge['unknown_metric_count'],1)
        self.assertIsNone(edge['metrics']['spatial_texture'])
        self.assertGreater(edge['ranking_cost'],0)
        self.assertIsNone(edge['semantics']['detected_person_count'])
        self.assertFalse(edge['quality_improvement_established'])

    def test_nonfinite_metric_rejected(self):
        value=trial.scene_descriptor(np.full((54,96,3),80,np.uint8))
        value['mean_bgr'][0]=float('nan')
        with self.assertRaises(ValueError): trial.scene_edge(value,value)

    def test_no_person_sensor_in_review_contract(self):
        self.assertIsNone(trial.UNKNOWN['stage_cohort'])
        self.assertIsNone(trial.UNKNOWN['arm_containment'])
        self.assertIsNone(trial.UNKNOWN['beat_phase'])
        self.assertFalse(trial.UNKNOWN['semantic_authority'])

    def test_descriptor_rejects_non_image(self):
        with self.assertRaises(ValueError): trial.scene_descriptor(np.ones((5,5),np.float32))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local FFmpeg required')
class SceneReviewProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name).resolve()
        cls.sources=tuple(cls.root/f'source-{i}.mp4' for i in range(8))
        for i,p in enumerate(cls.sources): encode_wide_chart(p,variant=i,frames=540)
        cls.bundle,cls.controller,cls.request,cls.model=build_wide_fixture(cls.root,cls.sources)
        cls.args=dict(controller=cls.controller,eligible_indices=tuple(range(8)),offsets=(1.,)*8,
            timewarps=(None,)*8,scene_cut_times=((),)*8,source_derivatives=scene_derivative_fixture())
        with optional_native_scope(monotonic()+180):
            cls.context=trial.freeze_context(**cls.args)
            cls.proposal,cls.reason=trial.propose(cls.context)
        if cls.proposal is None: raise AssertionError(cls.reason)
        cls.plan,cls.receipt=cls.proposal

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def context_now(self):
        return replace(self.context,deadline=monotonic()+180)

    def test_real_producer_and_independent_parent_without_models(self):
        context=self.context_now()
        with optional_native_scope(context.deadline):
            result=trial.replay(context,self.plan,self.receipt)
        self.assertEqual(result['counts']['model_calls'],0)
        self.assertLessEqual(result['counts']['new_decodes'],1200)
        self.assertEqual(result['counts']['dense_paths'],2)
        self.assertFalse(self.receipt['quality_approved'])
        self.assertFalse(self.receipt['refreshed_retained_family_authority'])
        self.assertTrue(any(a.source_index!=b.source_index for a,b in zip(self.plan.segments,self.bundle.render_plan.segments)))
        self.assertEqual(self.plan.segments[4:],self.bundle.render_plan.segments[4:])
        self.assertEqual(self.controller.incumbent,self.bundle)

    def test_exact_dense_order_not_set_or_cardinality(self):
        bad=deepcopy(self.receipt);bad['dense'][0],bad['dense'][1]=bad['dense'][1],bad['dense'][0]
        with self.assertRaises(ValueError):trial.replay.__wrapped__(self.context_now(),self.plan,bad)

    def test_duplicate_dense_tick_rejected(self):
        bad=deepcopy(self.receipt);bad['dense'][1]=bad['dense'][0]
        with self.assertRaises(ValueError):trial.replay.__wrapped__(self.context_now(),self.plan,bad)

    def test_changed_raw_hash_rejected(self):
        bad=deepcopy(self.receipt);bad['dense'][0]['after']['raw_pixels_sha256']='0'*64
        with self.assertRaises(ValueError):trial.replay.__wrapped__(self.context_now(),self.plan,bad)

    def test_unknown_cannot_be_claimed_known(self):
        bad=deepcopy(self.receipt);bad['semantics']['stage_cohort']=[]
        with self.assertRaisesRegex(ValueError,'uncertainty'):trial._rederive(self.context_now(),self.plan,bad)

    def test_approval_and_metadata_laundering_rejected(self):
        for key in ('quality_approved','delivery_completion_message_allowed','refreshed_retained_family_authority'):
            bad=deepcopy(self.receipt);bad[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):trial._rederive(self.context_now(),self.plan,bad)

    def test_plan_payload_integer_alias_is_not_the_canonical_selected_plan(self):
        segments=list(self.plan.segments)
        # Equal numeric value is not identical typed producer/consumer JSON.
        segments[0]=replace(segments[0],source_start=int(segments[0].source_start))
        altered=replace(self.plan,segments=tuple(segments))
        self.assertEqual(family.plan_payload(altered),family.plan_payload(self.plan))
        self.assertNotEqual(family.digest(family.plan_payload(altered)),family.digest(family.plan_payload(self.plan)))
        with self.assertRaisesRegex(ValueError,'independently assembled'):
            trial._rederive(self.context_now(),altered,self.receipt)

    def test_incumbent_plan_under_new_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError,'independently assembled'):
            trial._rederive(self.context_now(),self.bundle.render_plan,self.receipt)

    def test_stronger_duty_rederived_from_original(self):
        from crosscut.group_actual_shots import visible_group_shots
        context=self.context_now();shots=visible_group_shots(context.incumbent.render_plan,context.domain)[:3]
        index=shots[1].first_segment_index
        for subject in ('face','upper_body','formation'):
            face=context.incumbent.face_match_plan
            segments=list(face.segments);segments[index]=replace(segments[index],alignment_subject=subject)
            altered=replace(context,incumbent=replace(context.incumbent,face_match_plan=replace(face,segments=tuple(segments))))
            with self.subTest(subject=subject),self.assertRaises(ValueError):trial._intents_and_guard(altered,shots)

    def test_source_identity_changed_rejected(self):
        context=replace(self.context_now(),source_identities=())
        with self.assertRaisesRegex(ValueError,'source preimage'):trial._fresh(context)

    def test_deadline_and_cumulative_work_not_reset(self):
        with self.assertRaises(ValueError):trial._fresh(replace(self.context,deadline=monotonic()-1))
        counts=dict(self.receipt['work']['counts'],new_decodes=1201)
        with self.assertRaisesRegex(ValueError,'cumulative'):trial._session(self.context_now(),counts)

    def test_real_encode_full_decode_and_selected_pixel_witness(self):
        context=self.context_now()
        output=self.root/'review-actual.mp4'
        with optional_native_scope(context.deadline):
            result=trial.render_review(context,self.plan,self.receipt,output)
        self.assertTrue(output.is_file())
        self.assertTrue(all(result['technical_qa']['final_media']['checks'].values()))
        self.assertFalse(result['technical_qa']['professional_qa_passed'])
        first,end=self.receipt['bounds'][1]
        self.assertEqual([r['output_tick'] for r in result['changed_pixel_witness']['rows']],list(range(first,end)))
        self.assertEqual(result['proxy_plan']['width'],854)
        self.assertEqual(result['proxy_plan']['height'],480)
        self.assertEqual(result['video_sha256'],family._file_hash(output))
        import os
        destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
        if destination:
            root=Path(destination)/'new-scene-review';root.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(output,root/'NONHUMAN-review.mp4')
            (root/'RESULT.json').write_text(json.dumps(family.plain(result),indent=2))
            (root/'RECEIPT.json').write_text(json.dumps(family.plain(self.receipt),indent=2))

    def test_corrupt_encoded_file_not_exposed(self):
        output=self.root/'corrupt.mp4';output.write_bytes(b'not-a-video')
        from crosscut.rendered_proxy_review import derive_proxy_render_plan,RenderedProxyReview
        proxy=derive_proxy_render_plan(self.plan,output)
        review=RenderedProxyReview(dict(schema_version=3,geometry_provider_failure_count=0,
            reviewed_boundary_count=0,boundaries=[],introduced_cross_source_technical_issue_reasons=[]))
        with self.assertRaisesRegex(ValueError,'full-decode'):
            trial._technical_qa(self.context_now(),proxy,review)

    def test_nested_technical_failure_is_not_editorial_unknown(self):
        from crosscut.rendered_proxy_review import RenderedProxyReview
        review=RenderedProxyReview(dict(schema_version=3,geometry_provider_failure_count=0,
            reviewed_boundary_count=0,boundaries=[],introduced_cross_source_technical_issue_reasons=['black_frame_run']))
        with self.assertRaisesRegex(ValueError,'nested technical'):
            trial._technical_qa(self.context_now(),self.plan,review)

    def test_float_tick_and_boolean_semantic_flag_rejected(self):
        bad=deepcopy(self.receipt);bad['bounds'][0][0]=float(bad['bounds'][0][0])
        with self.assertRaises(ValueError):trial._rederive(self.context_now(),self.plan,bad)
        bad=deepcopy(self.receipt);bad['semantics']['semantic_authority']=0
        with self.assertRaises(ValueError):trial._rederive(self.context_now(),self.plan,bad)
        bad=deepcopy(self.receipt);bad['dense'][0]['output_tick']=float(bad['dense'][0]['output_tick'])
        with self.assertRaises(ValueError):trial.replay.__wrapped__(self.context_now(),self.plan,bad)

    def test_known_selected_arms_cannot_be_dropped_by_wide_label(self):
        from crosscut.group_actual_shots import visible_group_shots
        from crosscut.closeup_gesture import CloseupGestureEvidence
        context=self.context_now();shots=visible_group_shots(context.incumbent.render_plan,context.domain)[:3]
        evidence=CloseupGestureEvidence(59,60,(.1,.1,None,None))
        repairs=SimpleNamespace(repairs=(SimpleNamespace(boundary_index=0,
            selected_visual_evidence=SimpleNamespace(closeup_gesture=evidence)),))
        context=replace(context,incumbent=replace(context.incumbent,boundary_repairs=repairs))
        with self.assertRaisesRegex(ValueError,'selected arm'):trial._intents_and_guard(context,shots)

    def test_native_broadcaster_cut_donor_still_rejected(self):
        from crosscut.group_actual_shots import visible_group_shots
        context=self.context_now();shots=visible_group_shots(context.incumbent.render_plan,context.domain)
        bounds=self.receipt['bounds']
        window=next(shots[i:i+3] for i in range(len(shots)-2)
            if [[s.first_output_frame,s.end_output_frame] for s in shots[i:i+3]]==bounds)
        donor=self.receipt['donor_eligible_index']
        cuts=[() for _ in context.domain.eligible_indices]
        row=self.receipt['dense'][len(self.receipt['dense'])//2]['after']
        cuts[donor]=(row['native_frame_index']/30.,)
        context=replace(context,domain=replace(context.domain,scene_cut_times=tuple(cuts)))
        with self.assertRaises(ValueError):trial._path(context,window,donor)

    def test_unknown_derivative_pair_is_not_independence(self):
        from crosscut.group_actual_shots import visible_group_shots
        context=self.context_now();shots=visible_group_shots(context.incumbent.render_plan,context.domain)[:3]
        with self.assertRaises(ValueError):trial._path(replace(context,source_derivative_binding=None),shots,6)

    def test_nonidentity_request_eligible_mapping(self):
        from crosscut.group_actual_shots import visible_group_shots
        from crosscut.group_scene_source_owner import freeze_derivatives
        context=self.context_now();order=(7,0,6,1,5,2,4,3)
        domain=replace(context.domain,eligible_indices=order,
            render_frame_clocks=tuple(context.domain.render_frame_clocks[i] for i in order))
        identities=tuple(context.source_identities[i] for i in order)
        context=replace(context,domain=domain,source_identities=identities,
            source_derivative_binding=freeze_derivatives(scene_derivative_fixture(),domain,identities))
        shots=visible_group_shots(context.incumbent.render_plan,domain)[:3]
        _,path,assembly,_=trial._path(context,shots,order.index(6))
        self.assertEqual(path[1].eligible_source_index,order.index(6))
        self.assertTrue(all(s.source_index==6 for s in path[1].segments))
        self.assertEqual(assembly[0].sources,self.bundle.render_plan.sources)

    def test_parent_decode_allowance_does_not_trust_low_child_counter(self):
        counts=dict(self.receipt['work']['counts'],new_decodes=0)
        session=trial._session(self.context_now(),counts)
        try:
            self.assertEqual(session.budget.limits['new_decodes'],500)
            self.assertEqual(session.budget.limits['edges'],4)
        finally:session.close()

    def test_incumbent_renamed_video_rejected_by_real_encoded_witness(self):
        from crosscut.rendered_proxy_review import derive_proxy_render_plan
        from crosscut.render import render
        context=self.context_now();first,end=self.receipt['bounds'][1]
        full=derive_proxy_render_plan(self.bundle.render_plan,self.root/'renamed-incumbent.mp4')
        candidate=derive_proxy_render_plan(self.plan,self.root/'not-encoded-full.mp4')
        selected=trial._witness_plan(context,candidate,first,end,self.root/'selected-witness.mp4')
        original=trial._witness_plan(context,full,first,end,self.root/'original-witness.mp4')
        for plan in (full,selected,original):
            render(plan,video_codec='libx264',video_bitrate='8M',video_maxrate='12M',video_buffer_size='16M',
                audio_bitrate='128k',maximum_runtime_seconds=trial._remaining(context))
        with self.assertRaisesRegex(ValueError,'pixels differ|match incumbent'):
            trial.compare_decoded_witness(full.output,selected.output,first,end,context,original.output)
        # A zero-change reference cannot masquerade as a new proposal either.
        with self.assertRaisesRegex(ValueError,'not distinguishable'):
            trial.compare_decoded_witness(full.output,original.output,first,end,context,original.output)
