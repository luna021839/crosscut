"""Executable synthetic native-video tests, no weights/network/holdout data.

The synthetic camera evidence is measured from encoded pixels. The analyze
harness scripts music/acquisition and unrelated learned/montage dependencies;
new owner, native clock/pixel reads, replay/CAS and QA consumer stay real.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.first_result_controller import FirstResultController
from crosscut.group_normal_interval_owner import (
    CONTRACT, MOTION_BASIS, UNKNOWN, QUALITY_CODE, NormalIntervalContext,
    attempt_normal_interval, freeze_context, propose_normal_interval, validate_parent,
    validate_bundle_binding, validate_serialized_metadata, consume_normal_interval_qa,
    replay_final_sources, qa_metrics_valid, validate_epoch_clock, _pixel_descriptor, _rank, _scaffold, _overlay,
    _dependencies, _digest, _plan_payload, _metadata_payload,
)
from crosscut.optional_native_worker import optional_native_scope
from crosscut.pipeline import _review_first_group_bundle, _validate_review_first_group_bundle
from crosscut.render import segment_frame_counts, EditorialTransformKeyframe, EditorialTransformTrack
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import probe_render_frame_clock
from crosscut.review_first_runtime import ReviewFirstRuntime, classify_review_first_qa, ReviewFirstStatus
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.ive import AspectRatio
from crosscut.visual_score import VisualTimelinePlan
from crosscut.visual_boundary_repair import _thaw_audit_value
from engine.tests.test_pipeline import _visual_choice


def generate_source(path, *, cut_frames=(39, 143), frame_count=360, uniform=False):
    """Three real raster epochs; R0 is slightly wider and right-displaced.

    Integer frame codes in a corner make accidental repeated/neighbor reads
    distinguishable. The test never labels a rectangle as a face or body.
    """
    data = np.zeros((frame_count, 180, 320, 3), np.uint8)
    for index, frame in enumerate(data):
        frame[:] = 24
        if not uniform:
            mutable = cut_frames[0] <= index < cut_frames[1]
            half = 32 if mutable else 36
            cx = 168 if mutable else 160
            cv2.rectangle(frame, (cx-half, 90-half), (cx+half, 90+half), (205, 205, 205), -1)
            cv2.line(frame, (cx-half, 90), (cx+half, 90), (80, 140, 190), 2)
            for bit in range(9):
                if index & (1 << bit):
                    frame[3:5, 3+bit*3:5+bit*3] = 110
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', '320x180', '-r', '30000/1001', '-i', 'pipe:0', '-f', 'lavfi',
        '-i', 'anullsrc=r=48000:cl=stereo', '-shortest', '-c:v', 'libx264',
        '-threads', '1', '-preset', 'ultrafast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-bf', '0', '-c:a', 'aac', '-y', str(path)]
    completed = subprocess.run(command, input=data.tobytes(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30)
    if completed.returncode:
        raise AssertionError(completed.stderr.decode())
    clock = probe_render_frame_clock(path)
    if clock is None:
        raise AssertionError('synthetic H264 source lacks a real verified native clock')
    return clock


def make_controller(root, sources, *, fps=29.97, source_start=0., duration=8.):
    section = SectionPlan((SectionDirective(0, SectionRole.VERSE_LIKE, 0., duration,
        .5, .9, 1, 4, 13.5, .34, 0., False, False),))
    choices = tuple(_visual_choice(i, 0, i*duration/2, (i+1)*duration/2) for i in range(2))
    choices = tuple(replace(c, source_local_start=c.source_local_start+source_start,
        source_local_end=c.source_local_end+source_start) for c in choices)
    request = SimpleNamespace(sources=sources, reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE)
    with patch('crosscut.pipeline.output_size', return_value=(320, 180)):
        baseline = _review_first_group_bundle(timeline=VisualTimelinePlan(choices, 2.),
            request=request, eligible_indices=tuple(range(len(sources))), offsets=(source_start,)*len(sources),
            timewarps=(None,)*len(sources), common_start=0., section_plan=section,
            phrase_times=(0., duration/2, duration), output=root/'out.mp4',
            reporter=SimpleNamespace(log=lambda *args, **kwargs: None), warnings=[])
    baseline = replace(baseline, render_plan=replace(baseline.render_plan, fps=fps))
    controller = FirstResultController(ReviewFirstRuntime(enabled=True), sources=sources,
        source_durations=(12.012,)*len(sources), master_start=0., duration=duration,
        journal_path=root/'controller.json')
    controller.install_baseline(baseline, plan=baseline.render_plan,
        authority='deterministic_full_song_visual_selector')
    return controller


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires installed FFmpeg, never downloads')
class NormalIntervalOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.root = Path(cls.storage.name)
        cls.source = cls.root/'source.mp4'
        cls.clock = generate_source(cls.source)
        cls.copy = cls.root/'source-copy.mp4'
        shutil.copyfile(cls.source, cls.copy)
        cls.sources = (cls.source, cls.copy)
        cls.cuts = tuple(n/float(cls.clock.frame_rate) for n in (39,143))

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.controller = make_controller(self.work, self.sources)

    def context(self, controller=None):
        controller = controller or self.controller
        return freeze_context(controller=controller, eligible_indices=(0,1), offsets=(0.,0.),
            timewarps=(None,None), scene_cut_times=(self.cuts,self.cuts))

    def candidate(self):
        with optional_native_scope(monotonic()+40.):
            context = self.context()
            return context, propose_normal_interval(context)

    def test_real_native_positive_split_complete_replay_cas_and_final_source_qa(self):
        baseline = self.controller.incumbent
        candidate = attempt_normal_interval(controller=self.controller, eligible_indices=(0,1),
            offsets=(0.,0.), timewarps=(None,None), scene_cut_times=(self.cuts,self.cuts),
            archive_path=self.work/'proposal.json')
        self.assertIsNotNone(candidate, self.controller.to_dict())
        self.assertIs(self.controller.incumbent, candidate)
        self.assertGreater(len(candidate.render_plan.segments),len(baseline.render_plan.segments))
        receipt = candidate.to_dict()['normal_interval_review']
        self.assertEqual(receipt['uncertain_semantics'],UNKNOWN)
        self.assertFalse(receipt['quality_approved'])
        self.assertTrue(all(c.shot_scale is None for c in candidate.visual_timeline.choices))
        self.assertEqual(len(candidate.visual_timeline.choices), len(candidate.editorial_motion_plan.segments))
        self.assertEqual(len(candidate.stage_edit_exact_shot_clock['shots']), len(candidate.render_plan.segments))
        changed = receipt['changed_segment_indices']
        self.assertTrue(changed)
        for i, segment in enumerate(candidate.render_plan.segments):
            self.assertEqual(segment.editorial_transform is not None, i in changed)
        original = native_frame_clocks_for_output(baseline.render_plan, {0:self.clock,1:self.clock},
            range(baseline.render_plan.output_frame_count))
        selected = native_frame_clocks_for_output(candidate.render_plan, {0:self.clock,1:self.clock},
            range(candidate.render_plan.output_frame_count))
        self.assertEqual([(r.source_index,r.native_frame_index) for r in original.values()],
            [(r.source_index,r.native_frame_index) for r in selected.values()])
        self.assertEqual(sum(segment_frame_counts(candidate.render_plan)), baseline.render_plan.output_frame_count)
        archive = json.loads((self.work/'proposal.json').read_text())
        self.assertEqual(archive['original_bundle'], baseline.to_dict())
        self.assertEqual(archive['candidate_bundle'], candidate.to_dict())
        metrics = replay_final_sources(candidate.render_plan.to_json(), receipt)
        self.assertTrue(qa_metrics_valid(metrics))
        self.assertGreater(metrics['sampled_native_pixel_count'], 4)

    def test_uniform_pixels_unknown_not_zero_cost_and_no_adoption(self):
        self.assertIsNone(_pixel_descriptor(np.zeros((90,160,3),np.uint8)))
        with optional_native_scope(monotonic()+40.):
            context = self.context()
            _,window,_,_ = _scaffold(context)
        self.assertIsNone(_rank([],window))
        uniform = self.work/'uniform.mp4'
        generate_source(uniform,uniform=True)
        controller = make_controller(self.work,(uniform,uniform))
        before = controller.transaction_token()
        result = attempt_normal_interval(controller=controller,eligible_indices=(0,1),offsets=(0.,0.),
            timewarps=(None,None),scene_cut_times=(self.cuts,self.cuts),archive_path=self.work/'none.json')
        self.assertIsNone(result)
        self.assertEqual(controller.transaction_token(),before)
        self.assertIn('UNKNOWN is not zero',str(controller.to_dict()))

    def test_no_three_actual_epochs_does_not_use_canonical_cells_as_shots(self):
        token = self.controller.transaction_token()
        result = attempt_normal_interval(controller=self.controller,eligible_indices=(0,1),offsets=(0.,0.),
            timewarps=(None,None),scene_cut_times=((),()),archive_path=self.work/'none.json')
        self.assertIsNone(result)
        self.assertEqual(self.controller.transaction_token(),token)
        self.assertIn('three observed actual epochs',str(self.controller.to_dict()))

    def test_live_replacement_rejects_parent_without_touching_incumbent(self):
        source = self.work/'replace.mp4'
        shutil.copyfile(self.source,source)
        controller = make_controller(self.work,(source,source))
        with optional_native_scope(monotonic()+40.):
            context = self.context(controller)
            candidate = propose_normal_interval(context)
            token = controller.transaction_token()
            source.write_bytes(source.read_bytes()+b'changed-after-observation')
            with self.assertRaisesRegex(ValueError,'preimage changed'):
                controller.compare_and_promote(candidate,plan=candidate.render_plan,authority=CONTRACT,
                    token=context.token,validate=lambda value:validate_parent(value,context))
        self.assertEqual(controller.transaction_token(),token)

    def test_unsupported_authority_is_preserved_not_deleted(self):
        for family, value in (('dynamic_crop_application',{'some':'retained'}),
                ('cut_local_sync',{'passed':False}),('output_geometry_deferral',{'keep':True})):
            with self.subTest(family=family):
                controller = make_controller(self.work,self.sources)
                kept = replace(controller.incumbent,**{family:value})
                controller.promote(kept,plan=kept.render_plan,authority='existing')
                result = attempt_normal_interval(controller=controller,eligible_indices=(0,1),offsets=(0.,0.),
                    timewarps=(None,None),scene_cut_times=(self.cuts,self.cuts),archive_path=self.work/'none.json')
                self.assertIsNone(result)
                self.assertIs(controller.incumbent,kept)
                self.assertIn(family,str(controller.to_dict()))

    def test_downstream_source_camera_motion_clock_unknown_and_receipt_faults_reject(self):
        context,candidate = self.candidate()
        receipt = candidate.to_dict()['normal_interval_review']
        variants = [replace(candidate,normal_interval_review=None),
            replace(candidate,render_plan=replace(candidate.render_plan,segments=(
                replace(candidate.render_plan.segments[0],source_start=.01),*candidate.render_plan.segments[1:]))),
            replace(candidate,editorial_motion_plan=replace(candidate.editorial_motion_plan,zoom_event_count=999)),
            replace(candidate,visual_timeline=replace(candidate.visual_timeline,choices=(
                replace(candidate.visual_timeline.choices[0],shot_scale=.0),*candidate.visual_timeline.choices[1:])))]
        for name,value in (('quality_approved',True),('uncertain_semantics',{**UNKNOWN,'motion_phase':0.})):
            variants.append(replace(candidate,normal_interval_review={**receipt,name:value}))
        for altered in variants:
            with self.subTest(altered=altered.cadence_profile),self.assertRaises(ValueError):
                validate_bundle_binding(altered)
        for name in ('_review_first_group_motion_camera_bundle','_review_first_group_shot_context_bundle',
                '_review_first_group_boundary_look_bundle'):
            import crosscut.pipeline as pipeline
            kwargs = dict(request=None,eligible_indices=(0,1),offsets=(0.,0.),timewarps=(None,None),
                scene_cut_times=(self.cuts,self.cuts),phrase_times=context.incumbent.phrase_times)
            if name.endswith('shot_context_bundle'):kwargs={'scene_cut_times':(self.cuts,self.cuts)}
            if name.endswith('boundary_look_bundle'):kwargs={'group_16x9_safe':True}
            with self.subTest(owner=name),self.assertRaisesRegex(ValueError,'fresh complete replay'):
                getattr(pipeline,name)(candidate,**kwargs)

    def test_forged_parent_rank_with_self_consistent_metadata_is_reobserved(self):
        context,candidate = self.candidate()
        receipt = candidate.to_dict()['normal_interval_review']
        receipt['rankings'][1]['contrast_moment_cost']=-100.
        forged=replace(candidate,normal_interval_review=receipt)
        # Complete-bundle hash alone cannot establish scalar pixel observations.
        validate_bundle_binding(forged)
        with optional_native_scope(context.deadline),self.assertRaisesRegex(ValueError,'replay disagrees'):
            validate_parent(forged,context)

    def test_stale_tokens_cannot_replace_newer_incumbent(self):
        context,candidate=self.candidate()
        self.controller.promote(context.incumbent,plan=context.incumbent.render_plan,authority='newer')
        with self.assertRaisesRegex(ValueError,'baseline changed'):
            self.controller.compare_and_promote(candidate,plan=candidate.render_plan,authority=CONTRACT,
                token=context.token,validate=lambda value:None)
        self.assertIs(self.controller.incumbent,context.incumbent)

    def test_final_qa_consumer_checks_independent_plan_render_and_selection_copies(self):
        _,candidate=self.candidate()
        complete=candidate.to_dict()
        analysis=dict(visual_timeline=complete['visual_timeline'],render=dict(
            normal_interval_review=complete['normal_interval_review'],candidate_origin=CONTRACT,
            editorial_motion=complete['editorial_motion'],group_transitions=complete['group_transitions'],
            final_render_segments=complete['final_render_segments']))
        selection=dict(selected_path_authority=CONTRACT,complete_candidates=[complete])
        metrics=consume_normal_interval_qa(analysis,candidate.render_plan.to_json(),selection)
        self.assertTrue(qa_metrics_valid(metrics))
        for family in ('editorial_motion','group_transitions','final_render_segments','normal_interval_review'):
            corrupted=deepcopy(analysis);corrupted['render'][family]=None
            with self.subTest(family=family),self.assertRaises(ValueError):
                consume_normal_interval_qa(corrupted,candidate.render_plan.to_json(),selection)
        poisoned=deepcopy(metrics);poisoned['uncertain_semantics']['identity']=0.
        self.assertFalse(qa_metrics_valid(poisoned))


    def test_quality_disposition_is_review_ready_never_auto_approved(self):
        from engine.tests.test_review_first_runtime import _group_integrity_metrics
        _,candidate=self.candidate()
        metrics=replay_final_sources(candidate.render_plan.to_json(),candidate.to_dict()['normal_interval_review'])
        report=dict(passed=False,issues=[dict(code=QUALITY_CODE)],
            metrics={**_group_integrity_metrics(),'normal_interval_review':metrics})
        classify=lambda value: classify_review_first_qa(value,content_binding_passed=True,
            review_first_enabled=True,member=False)
        disposition=classify(report)
        self.assertEqual(disposition.status,ReviewFirstStatus.REVIEW_READY)
        self.assertFalse(disposition.delivery_completion_message_allowed)
        # Missing quality issue cannot turn a source/pixel receipt into approval.
        omitted=deepcopy(report);omitted.update(passed=True,issues=[])
        self.assertEqual(classify(omitted).status,ReviewFirstStatus.INPUT_UNUSABLE)
        for fault in ('unknown_as_zero','technical_fault','binding_lost'):
            damaged=deepcopy(report)
            if fault=='unknown_as_zero':
                damaged['metrics']['normal_interval_review']['uncertain_semantics']['motion_phase']=0.
            elif fault=='binding_lost':
                damaged['metrics']['normal_interval_review'].pop('evidence_sha256')
            else:
                damaged['issues'].append(dict(code='full_decode_failed'))
            with self.subTest(fault=fault):
                self.assertEqual(classify(damaged).status,ReviewFirstStatus.INPUT_UNUSABLE)

    def test_epoch_clock_replay_and_canonical_seam_not_an_actual_shot(self):
        _,candidate=self.candidate()
        timeline=candidate.visual_timeline.to_dict()
        cuts=validate_epoch_clock(candidate.render_plan,timeline,{0:self.cuts,1:self.cuts},(0,1))
        self.assertEqual(len(cuts),2)
        # R0 straddles the original 4-second cell seam. It is still one epoch.
        self.assertEqual(len({c.shot_id for c in candidate.visual_timeline.choices}),3)
        receipt=candidate.to_dict()['normal_interval_review']
        changed=receipt['changed_segment_indices']
        self.assertGreater(len(changed),1)
        self.assertEqual(len({candidate.visual_timeline.choices[i].shot_id for i in changed}),1)
        for field in ('actual_events','source_cut_times','plan_sha256','native_clocks'):
            damaged=deepcopy(timeline)
            damaged['actual_epoch_clock'][field]=[] if field!='plan_sha256' else '0'*64
            with self.subTest(field=field),self.assertRaises((ValueError,IndexError,KeyError,TypeError)):
                validate_epoch_clock(candidate.render_plan,damaged,{0:self.cuts,1:self.cuts},(0,1))

    def test_different_output_rate_and_shifted_source_use_runtime_ticks(self):
        controller=make_controller(self.work,self.sources,fps=24.,source_start=.2)
        with optional_native_scope(monotonic()+40.):
            context=freeze_context(controller=controller,eligible_indices=(0,1),offsets=(.2,.2),
                timewarps=(None,None),scene_cut_times=(self.cuts,self.cuts))
            candidate=propose_normal_interval(context)
            validate_parent(candidate,context)
        receipt=candidate.to_dict()['normal_interval_review']
        self.assertNotEqual(receipt['window'][1]['first_output_frame'],39)
        self.assertEqual(candidate.render_plan.fps,24.)
        original=native_frame_clocks_for_output(context.incumbent.render_plan,{0:self.clock,1:self.clock},
            range(context.incumbent.render_plan.output_frame_count))
        changed=native_frame_clocks_for_output(candidate.render_plan,{0:self.clock,1:self.clock},
            range(candidate.render_plan.output_frame_count))
        self.assertEqual([(r.source_index,r.native_frame_index) for r in original.values()],
            [(r.source_index,r.native_frame_index) for r in changed.values()])

    def test_deadline_and_archive_failure_keep_complete_incumbent(self):
        before=self.controller.incumbent
        with optional_native_scope(monotonic()+40.):
            context=self.context()
            with self.assertRaises(Exception):
                propose_normal_interval(replace(context,deadline=monotonic()-1.))
        with patch('crosscut.pipeline._write_analysis_manifest_atomically',side_effect=OSError('archive disk fault')):
            result=attempt_normal_interval(controller=self.controller,eligible_indices=(0,1),offsets=(0.,0.),
                timewarps=(None,None),scene_cut_times=(self.cuts,self.cuts),archive_path=self.work/'broken.json')
        self.assertIsNone(result)
        self.assertIs(self.controller.incumbent,before)
        self.assertIn('archive disk fault',str(self.controller.to_dict()))

    def test_reentrant_parent_cannot_publish_over_a_new_incumbent(self):
        context,candidate=self.candidate()
        def validator(value):
            with optional_native_scope(context.deadline):
                validate_parent(value,context)
            self.controller.promote(context.incumbent,plan=context.incumbent.render_plan,authority='newer')
        with self.assertRaisesRegex(ValueError,'baseline changed'):
            self.controller.compare_and_promote(candidate,plan=candidate.render_plan,authority=CONTRACT,
                token=context.token,validate=validator)
        self.assertIs(self.controller.incumbent,context.incumbent)




@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires installed FFmpeg')
class NormalAnalyzeTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory()
        cls.source=Path(cls.storage.name)/'native.mp4'
        cls.clock=generate_source(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        from engine.tests.test_group_learned_production_transaction import GroupLearnedProductionTransactionTests
        # The inherited mock-only fixture clears all environment variables.
        # These tests perform real subprocess work, so retain only the user's
        # executable search path (Homebrew is not in os.defpath on macOS).
        executable_path = os.environ.get('PATH', os.defpath)
        self.h=GroupLearnedProductionTransactionTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.h.stack.enter_context(patch.dict(os.environ, {'PATH': executable_path}))
        self.h.request = replace(self.h.request, experimental_pixel_camera_review=True)
        for source in self.h.sources:
            shutil.copyfile(self.source,source)
        from crosscut.media import probe_media
        from crosscut.vision import SceneChange
        self.h.stack.enter_context(patch('crosscut.pipeline.probe_media',side_effect=probe_media))
        self.h.stack.enter_context(patch('crosscut.pipeline.output_size',return_value=(320,180)))
        self.h.stack.enter_context(patch('crosscut.pipeline._review_first_registered_group_generation',return_value=None))
        # Observed raster cut inventory from the encoded fixture, not a worker
        # success flag. Find rectangle extent discontinuities independently.
        cap=cv2.VideoCapture(str(self.h.sources[0]));last=None;changes=[];index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            mask=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)>150
            value=int(mask.sum())
            if last is not None and abs(value-last)>400:
                changes.append(SceneChange(index,1.,1.,1.))
            last=value;index+=1
        cap.release()
        self.assertEqual([c.frame_index for c in changes],[39,143])
        actual_cuts=tuple(c.frame_index/float(self.clock.frame_rate) for c in changes)
        def cut_bound(timeline):
            return replace(timeline,choices=tuple(replace(c,
                internal_scene_cut_master_seconds=tuple(t for t in actual_cuts
                    if c.canonical_master_start+1e-6<t<c.canonical_master_end-1e-6)) for c in timeline.choices))
        self.h.baseline=cut_bound(self.h.baseline)
        self.h.learned=cut_bound(self.h.learned)
        self.h.materialized=replace(self.h.materialized,visual_plan=self.h.learned)
        from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
        self.h.stack.enter_context(patch('crosscut.pipeline.build_visual_timeline',return_value=self.h.baseline))
        self.h.stack.enter_context(patch('crosscut.pipeline.repair_visual_timeline_boundaries',return_value=
            VisualBoundaryRepairPlan(self.h.baseline,(self.h.baseline,),(),1,())))
        self.h.stack.enter_context(patch('crosscut.pipeline.materialize_stage_edit_visual_candidate',return_value=self.h.materialized))
        self.h.stack.enter_context(patch('crosscut.pipeline.detect_scene_changes_in_video_cached',
            return_value=(tuple(changes),{'detection_policy':'synthetic_decoded_rectangle_extent_changes'})))
        from crosscut.render import render as actual_render, write_plan
        self.rendered=[]
        def render_real(plan, **kwargs):
            codec=actual_render(plan,video_codec='libx264',video_bitrate='1M',video_maxrate='2M',
                video_buffer_size='2M',**kwargs)
            write_plan(plan,plan.output.with_suffix('.plan.json'))
            self.rendered.append(plan)
            return codec
        self.h.render.side_effect=render_real
        def real_proxy_artifact(plan,output,**kwargs):
            # Real native-clock renderer, but no claimed model/montage quality.
            projected=replace(plan,output=Path(output))
            render_real(projected,maximum_runtime_seconds=45.)
            return plan,SimpleNamespace(payload=dict(passed=False,weighted_score=45.,
                boundary_issue_indices=[0],proxy={},
                measurement_scope='synthetic_transport_only_editorial_review_not_measured'))
        self.h.proxy.side_effect=real_proxy_artifact
        self.qa_metrics=None;self.full_qa_report=None
        from engine.tests.test_group_learned_production_transaction import _ReachedFinalQA
        from crosscut.post_render_qa import evaluate_post_render_qa,PostRenderArtifactSet
        def final_consumer(**kwargs):
            self.h.manifest=kwargs['manifest']
            self.qa_metrics=consume_normal_interval_qa(self.h.manifest,
                self.h.render.call_args.args[0].to_json(),self.h.manifest['rendered_candidate_selection'])
            # Full independent artifact evaluator with real FFmpeg decode.
            # The optional perceptual provider is explicitly unavailable, not
            # filled with fabricated passing face/body/quality measurements.
            def no_perceptual(*args,**kw):
                raise ValueError('synthetic integration has no semantic/perceptual model evidence')
            self.full_qa_report=evaluate_post_render_qa(
                PostRenderArtifactSet.from_analysis_path(kwargs['manifest_path']),
                perceptual_evaluator=no_perceptual,trusted_registered_stage_edit=False)
            raise _ReachedFinalQA()
        self.h.stack.enter_context(patch('crosscut.pipeline._enforce_automatic_stage_edit_post_render_qa',
            side_effect=final_consumer))

    def test_positive_analyze_real_render_and_full_qa_consume_selected_variable_topology(self):
        plan=self.h.reach_final_qa()
        selection=self.h.manifest['rendered_candidate_selection']
        self.assertEqual(selection['selected_path_authority'],CONTRACT)
        self.assertGreater(len(plan.segments),2)
        self.assertTrue(qa_metrics_valid(self.qa_metrics))
        self.assertIsNotNone(self.full_qa_report)
        report=self.full_qa_report.to_dict()
        self.assertTrue(qa_metrics_valid(report['metrics']['normal_interval_review']))
        self.assertIn(QUALITY_CODE,{row['code'] for row in report['issues']})
        self.assertFalse(report['passed'])
        # Full QA must replay numeric inventory, not mislabel it as heuristic
        # authored gestures or admit unsafe camera values.
        codes={row['code'] for row in report['issues']}
        self.assertNotIn('editorial_transform_safety_bound_exceeded',codes)
        self.assertNotIn('editorial_motion_zoom_event_count_matches_reason_events',codes)
        self.assertFalse(any(code.startswith('editorial_motion_') for code in codes),codes)
        self.assertEqual(len(self.rendered),2)
        for rendered in self.rendered:
            cap=cv2.VideoCapture(str(rendered.output));count=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                count+=1
            cap.release()
            self.assertEqual(count,rendered.output_frame_count)
        motion=selection['complete_candidates'][0]['editorial_motion']
        self.assertEqual(motion,self.h.manifest['render']['editorial_motion'])
        attempts=self.h.manifest['review_first_visual_selection']['first_result_controller']['attempts']
        self.assertTrue(any(a['stage']=='normal_actual_epoch_camera_owner' for a in attempts))
        self.assertTrue(any('fresh complete replay' in str(a) for a in attempts))

    def test_default_request_never_lets_contrast_owner_suppress_pose_and_look(self):
        self.h.request = replace(self.h.request, experimental_pixel_camera_review=False)
        owner = self.h.stack.enter_context(patch(
            'crosscut.group_normal_interval_owner.attempt_normal_interval',
            side_effect=AssertionError('contrast-only owner must not run in ordinary jobs')))
        self.h.reach_final_qa()
        owner.assert_not_called()
        self.assertIsNone(self.qa_metrics)
        attempts = self.h.manifest['review_first_visual_selection']['first_result_controller']['attempts']
        stages = {row['stage'] for row in attempts}
        self.assertIn('joint_motion_camera_refinement', stages)
        self.assertIn('two_sided_shot_context_framing', stages)
        self.assertIn('boundary_look_match_render_admission', stages)
        self.assertNotIn('normal_actual_epoch_camera_owner', stages)

    def test_negative_analyze_parent_replay_failure_renders_original_complete_owner(self):
        original=propose_normal_interval
        def corrupt_after_observation(context):
            candidate=original(context)
            receipt=candidate.to_dict()['normal_interval_review']
            receipt['raw_samples'][0]['native_sha256']='0'*64
            return replace(candidate,normal_interval_review=receipt)
        # Corrupt both worker and parent via the wrapper would be a vacuous test.
        # Parent's independent read is therefore tested with first-call-only
        # corruption; the second actual worker observation remains unchanged.
        count=[0]
        def once(context):
            count[0]+=1
            return corrupt_after_observation(context) if count[0]==1 else original(context)
        self.h.stack.enter_context(patch('crosscut.group_normal_interval_owner.propose_normal_interval',side_effect=once))
        plan=self.h.reach_final_qa()
        self.assertEqual(len(plan.segments),2)
        self.assertTrue(all(s.editorial_transform is None for s in plan.segments))
        self.assertIsNone(self.qa_metrics)
        complete=self.h.manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertIsNone(complete['normal_interval_review'])
        attempts=self.h.manifest['review_first_visual_selection']['first_result_controller']['attempts']
        self.assertTrue(any(a['stage']=='normal_actual_epoch_camera_owner' and a['status']=='rejected'
            and 'replay disagrees' in str(a) for a in attempts))
        self.assertEqual(complete['editorial_motion'],self.h.manifest['render']['editorial_motion'])

    def test_learned_incumbent_still_attempts_normal_owner_and_preserves_v2_authority(self):
        # This branch uses the established learned mock fixture. It tests call
        # placement and non-erasure, not new native/semantic evidence.
        self.h.stack.enter_context(patch('crosscut.pipeline._review_first_registered_group_generation',
            return_value=self.h.context))
        plan=self.h.reach_final_qa()
        self.assertEqual(tuple(s.source_index for s in plan.segments),(1,2))
        self.assertIsNone(self.qa_metrics)
        attempts=self.h.manifest['review_first_visual_selection']['first_result_controller']['attempts']
        self.assertTrue(any(a['stage']=='normal_actual_epoch_camera_owner'
            and 'dynamic_crop' in str(a) for a in attempts))
        complete=self.h.manifest['rendered_candidate_selection']['complete_candidates'][0]
        self.assertEqual(complete['stage_edit_dynamic_crop_version'],2)
        self.assertTrue(all(s.editorial_transform is not None for s in plan.segments))


if __name__=='__main__':
    unittest.main()
