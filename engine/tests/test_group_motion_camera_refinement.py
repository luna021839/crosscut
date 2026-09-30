from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.boundary_subject_alignment import BoundarySubjectObservation,build_boundary_subject_alignment_plan
from crosscut.closeup_gesture import CloseupGestureEvidence, ObservedArmMotion
from crosscut.observed_head_motion import HeadMotionEvidence,ObservedHeadMotion,HEAD_MOTION_BASIS
from crosscut.group_cut_camera_proposal import build_group_cut_camera_proposal
from crosscut.group_motion_camera_refinement import (
    GroupMotionCameraRefinement, camera_motion_rank, refine_group_motion_camera,
    focal_boundary_shortlist,
    measured_motion_improves,
    measured_output_face_box,
    retimed_incumbent_camera_preview,
)
from crosscut.ive import AspectRatio
from crosscut.pipeline import (
    _review_first_group_bundle, _review_first_group_motion_camera_bundle,
    _validate_review_first_group_bundle,
)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.visual_score import VisualTimelinePlan
from crosscut.timeline import ShotType
from engine.tests.test_boundary_subject_alignment import _choice, _frame, _observations
from engine.tests.test_post_render_qa_current_group_contract import produced_plan


class MotionCameraRankTests(unittest.TestCase):
    def gesture(self, pose=.2, velocity=.1):
        return CloseupGestureEvidence(13, 20, (.1, .2, None, None),
            motion=ObservedArmMotion((10, 13, 20, 23), (.1, 1/30., .1),
                (pose, pose, None, None), (velocity, velocity, None, None)))

    def test_missing_evidence_is_not_a_perfect_numeric_score(self):
        gesture = self.gesture()
        for value, faces in ((None, (.0, .0)), (replace(gesture, motion=None), (.0, .0)),
                             (gesture, None), (gesture, (float('nan'), .1)),
                             (gesture, (.1, True)), (gesture, (-.1, .1))):
            self.assertIsNone(camera_motion_rank(value, faces))

    def test_output_face_reacquisition_does_not_invent_dominance_or_replace_known_geometry(self):
        frame=np.zeros((1080,1920,3),np.uint8);box=(.4,.2,.1,.15)
        with patch('crosscut.group_motion_camera_refinement._default_geometry_provider',return_value=
                    SimpleNamespace(face_bbox_xywh=box)), \
                patch('crosscut.shot_context_camera.observe_shot_head') as observe:
            self.assertEqual(measured_output_face_box(frame),box);observe.assert_not_called()
        for dominant in (False,True):
            observation=BoundarySubjectObservation(face_count=2,face_box_xywh=box,
                face_is_unambiguous_dominant=dominant)
            with patch('crosscut.group_motion_camera_refinement._default_geometry_provider',return_value=
                        SimpleNamespace(face_bbox_xywh=None)), \
                    patch('crosscut.shot_context_camera.observe_shot_head',return_value=observation) as observe:
                self.assertEqual(measured_output_face_box(frame),box if dominant else None)
                self.assertEqual(observe.call_args.args[0].shape,(480,854,3))

    def test_joint_rank_accounts_for_framing_and_keeps_existing_face_limits(self):
        better_motion = self.gesture(pose=.1, velocity=.05)
        worse_motion = self.gesture()
        self.assertGreater(camera_motion_rank(better_motion, (.2, .2)),
                           camera_motion_rank(worse_motion, (.01, .01)))
        self.assertIsNone(camera_motion_rank(better_motion, (.401, .0)))
        self.assertIsNone(camera_motion_rank(better_motion, (.0, .251)))
        self.assertIsNotNone(camera_motion_rank(worse_motion, (.5,.3),enforce_face_limits=False))

    def test_motion_comparison_preserves_all_temporally_observed_bones(self):
        before=replace(self.gesture(),bone_mismatches=(.1,.2,.3,None))
        better=self.gesture(pose=.1,velocity=.05)
        self.assertFalse(better.improves_over(before),'static comparator remains unchanged')
        self.assertTrue(measured_motion_improves(better,before),
            'a third endpoint-only bone had no four-frame motion value to lose')
        before=replace(before,motion=replace(before.motion,predicted_jump_radians=(.2,.2,.2,None),
            velocity_jump_per_output_tick_radians=(.1,.1,.1,None)))
        self.assertFalse(measured_motion_improves(better,before),
            'even a low-cost proposal cannot drop a genuinely observed motion bone')
        self.assertFalse(measured_motion_improves(replace(better,motion=None),before))
        self.assertFalse(measured_motion_improves(self.gesture(pose=.25,velocity=0.),self.gesture()),
            'lower velocity cost cannot buy a larger predicted pose jump')

    def test_bound_is_checked_before_any_media_decode(self):
        with patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock') as probe:
            for options in ({'maximum_boundaries': 5}, {'maximum_boundaries': True},
                            {'maximum_camera_candidates': 3}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    refine_group_motion_camera(plan=None, timeline=None, audio_offsets=(),
                        source_timewarps=(), scene_cut_times=(), model_path='unused', **options)
            probe.assert_not_called()

    def test_ranking_preview_keeps_absolute_incumbent_curve_and_sampling_raster(self):
        outgoing=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),
            EditorialTransformKeyframe(.7,1.2),EditorialTransformKeyframe(1.,1.)))
        incoming=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),
            EditorialTransformKeyframe(.5,1.),EditorialTransformKeyframe(1.,1.)))
        segments=(RenderSegment(0,0.,1.,0.,editorial_transform=outgoing),
            RenderSegment(1,1.,1.,1.,editorial_transform=incoming),RenderSegment(0,2.,1.,2.))
        plan=RenderPlan((Path('0.mp4'),Path('1.mp4')),segments,0,'16:9',Path('unused.mp4'),fps=30.)
        numeric=replace(plan,segments=(replace(segments[0],duration=1.2,editorial_transform=None),
            replace(segments[1],master_start=1.2,source_start=1.2,duration=.8,editorial_transform=None),segments[2]))
        preview=retimed_incumbent_camera_preview(plan,numeric,0)
        self.assertEqual(preview.segments[2],plan.segments[2])
        for index,shift in ((0,0.),(1,.2)):
            original=plan.segments[index].editorial_transform
            track=preview.segments[index].editorial_transform
            for time in (0.,.25,.5,.7):
                a,b=track.evaluate(time),original.evaluate(time+shift)
                for field in ('zoom','center_x','center_y','roll_degrees'):
                    self.assertAlmostEqual(getattr(a,field),getattr(b,field))
            self.assertEqual(track.effective_sampling_canvas_zoom,original.effective_sampling_canvas_zoom)
            self.assertIn('not_dense_authority',track.reason)
        self.assertIsNotNone(preview.segments[1].editorial_transform.sampling_canvas_zoom)
        self.assertIsNone(plan.segments[1].editorial_transform.sampling_canvas_zoom)
        self.assertEqual(preview.output_frame_count,plan.output_frame_count)
        for bad in (True,-1,2):
            with self.assertRaises(ValueError):retimed_incumbent_camera_preview(plan,numeric,bad)

    def test_bounded_candidate_frontier_does_not_limit_sensing_to_the_first_twelve_scales(self):
        choices=tuple(replace(_choice(i,i%2,start=float(i),end=float(i+1)),
            shot_type=ShotType.MEDIUM,shot_scale=.8-i*.01,
            exit_shot_type=ShotType.MEDIUM,exit_shot_scale=.8-i*.01,
            entry_shot_type=ShotType.MEDIUM,entry_shot_scale=.8-i*.01) for i in range(30))
        timeline=VisualTimelinePlan(choices,1.)
        self.assertNotIn(15,focal_boundary_shortlist(timeline))
        self.assertIn(15,focal_boundary_shortlist(timeline,maximum_candidates=32))
        self.assertEqual(len(focal_boundary_shortlist(timeline,maximum_candidates=20)),20)
        for value in (True,0,49):
            with self.assertRaises(ValueError):focal_boundary_shortlist(timeline,maximum_candidates=value)
        with patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock') as probe:
            for options in ({'maximum_candidate_boundaries':49},{'maximum_candidate_boundaries':True},
                    {'maximum_measured_boundaries':0},{'maximum_measured_boundaries':True},
                    {'maximum_candidate_boundaries':4,'maximum_measured_boundaries':5}):
                with self.subTest(options=options),self.assertRaises(ValueError):
                    refine_group_motion_camera(plan=None,timeline=None,audio_offsets=(),source_timewarps=(),
                        scene_cut_times=(),model_path='unused',**options)
            probe.assert_not_called()

    def test_unknown_motion_does_not_consume_the_complete_measured_frontier(self):
        choices=tuple(replace(_choice(i,i%2,start=float(i),end=float(i+1)),
            entry_shot_type=ShotType.MEDIUM,exit_shot_type=ShotType.MEDIUM,
            entry_shot_scale=.8-i*.01,exit_shot_scale=.8-i*.01) for i in range(16))
        plan=RenderPlan((Path('0.mp4'),Path('1.mp4')),
            tuple(RenderSegment(i%2,float(i),1.,float(i)) for i in range(16)),0,
            '16:9',Path('unused.mp4'),fps=30.,transitions=(RenderTransition(),)*15)
        calls=0
        def compare(*_args,**_kwargs):
            nonlocal calls
            calls+=1
            return None if calls<=12 else self.gesture()
        sampler=SimpleNamespace(feature_descriptor_at=lambda *a,**k:SimpleNamespace(),close=lambda:None)
        capture=SimpleNamespace(frame_at_native_index=lambda *a:np.zeros((90,160,3),np.uint8))
        provider=SimpleNamespace(compare_motion=compare,capture=capture,close=lambda:None)
        with patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock',return_value=
                    VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)), \
                patch('crosscut.group_motion_camera_refinement.CloseupGestureProvider',return_value=provider), \
                patch('crosscut.group_motion_camera_refinement._VideoSampler',return_value=sampler), \
                patch('crosscut.group_motion_camera_refinement._default_geometry_provider',return_value=
                    SimpleNamespace(face_bbox_xywh=(.4,.25,.1,.15))) as observed, \
                patch('crosscut.group_motion_camera_refinement.measured_motion_improves',return_value=False):
            result=refine_group_motion_camera(plan=plan,timeline=VisualTimelinePlan(choices,1.),
                audio_offsets=(0.,0.),source_timewarps=(None,None),scene_cut_times=((),()),
                model_path='unused',maximum_measured_boundaries=1)
        baseline=[r for r in result.attempts if r['stage']=='baseline_motion']
        self.assertEqual([r['boundary_index'] for r in baseline],list(range(13)))
        self.assertFalse(any(r['complete_measured_candidate'] for r in baseline[:12]))
        self.assertTrue(baseline[12]['complete_measured_candidate'])
        observed.assert_called();self.assertEqual(observed.call_count,2,'unknown motion did not trigger output face sensing')
        budget=next(r for r in result.attempts if r['stage']=='bounded_measured_focal_frontier')
        self.assertEqual(budget['measured_boundary_indices'],[12])
        self.assertEqual(result.proposals,())
        self.assertEqual(result.plan,plan)

    def test_head_fallback_stays_in_its_basis_and_native_cuts_precede_all_sensing(self):
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,i,start=2.*i,end=2.*(i+1)),
            entry_shot_scale=.5,exit_shot_scale=.5) for i in range(2)),1.)
        plan=RenderPlan((Path('0.mp4'),Path('1.mp4')),
            (RenderSegment(0,0.,2.,0.),RenderSegment(1,2.,2.,2.)),0,'16:9',Path('unused.mp4'),
            fps=30.,transitions=(RenderTransition(),))
        for arm_available,native_cut in ((False,False),(True,False),(False,True)):
            arm_calls=[];head_calls=[];descriptor_calls=[]
            def arm(*args,**kwargs):
                arm_calls.append(args)
                return self.gesture() if arm_available else None
            def head(*args,**kwargs):
                head_calls.append(args)
                frames=tuple(int(row[1]*30.+.5) for row in args)
                return HeadMotionEvidence(ObservedHeadMotion(frames,(.1,1/30.,.1),.2,.1))
            def descriptor(*args,**kwargs):
                descriptor_calls.append(args)
                return SimpleNamespace()
            provider=SimpleNamespace(compare_motion=arm,compare_head_motion=head,close=lambda:None,
                capture=SimpleNamespace(frame_at_native_index=lambda *a:np.zeros((90,160,3),np.uint8)))
            with self.subTest(arm_available=arm_available,native_cut=native_cut), \
                    patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock',return_value=
                        VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)), \
                    patch('crosscut.group_motion_camera_refinement.CloseupGestureProvider',return_value=provider), \
                    patch('crosscut.group_motion_camera_refinement._VideoSampler',return_value=
                        SimpleNamespace(feature_descriptor_at=descriptor,close=lambda:None)), \
                    patch('crosscut.group_motion_camera_refinement.measured_output_face_box',return_value=(.4,.25,.1,.15)):
                result=refine_group_motion_camera(plan=plan,timeline=timeline,audio_offsets=(0.,0.),
                    source_timewarps=(None,None),scene_cut_times=((1.95,) if native_cut else (),()),
                    model_path='unused')
            baseline=next(row for row in result.attempts if row['stage']=='baseline_motion')
            self.assertEqual(result.proposals,())
            if native_cut:
                self.assertFalse(arm_calls or head_calls or descriptor_calls)
                self.assertFalse(baseline['complete_measured_candidate'])
            elif arm_available:
                self.assertTrue(arm_calls)
                self.assertFalse(head_calls)
                self.assertNotEqual(baseline['motion_evidence_basis'],HEAD_MOTION_BASIS)
            else:
                self.assertEqual(len(arm_calls),1,'head candidate offsets may not switch to newly visible arms')
                self.assertEqual(len(head_calls),7)
                self.assertEqual(baseline['motion_evidence_basis'],HEAD_MOTION_BASIS)
                self.assertTrue(baseline['complete_measured_candidate'])

    def test_action_recovery_is_one_explicit_retry_inside_existing_camera_slots(self):
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,i,start=2.*i,end=2.*(i+1)),
            entry_shot_scale=.5,exit_shot_scale=.5) for i in range(2)),1.)
        plan=RenderPlan((Path('0.mp4'),Path('1.mp4')),
            (RenderSegment(0,0.,2.,0.),RenderSegment(1,2.,2.,2.)),0,'16:9',Path('unused.mp4'),
            fps=30.,transitions=(RenderTransition(),))
        for reason,recoverable in (('shot_policy_family_mismatch',True),
                ('exact_subject_would_be_clipped',True),('broadcaster_native_internal_cut',False)):
            calls=[];measurements=0
            def measure(*_args,**_kwargs):
                nonlocal measurements
                measurements+=1
                return self.gesture(pose=.6 if measurements==1 else .1)
            def camera(**kwargs):
                calls.append(kwargs)
                raise ValueError('camera proposal lacks dense local subject alignment: '+
                    ('per_frame_subject_containment_failed' if kwargs.get('global_action_framing') else reason))
            provider=SimpleNamespace(compare_motion=measure,close=lambda:None,
                capture=SimpleNamespace(frame_at_native_index=lambda *a:np.zeros((90,160,3),np.uint8)))
            with self.subTest(reason=reason), \
                    patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock',return_value=
                        VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)), \
                    patch('crosscut.group_motion_camera_refinement.CloseupGestureProvider',return_value=provider), \
                    patch('crosscut.group_motion_camera_refinement._VideoSampler',return_value=
                        SimpleNamespace(feature_descriptor_at=lambda *a,**k:SimpleNamespace(),close=lambda:None)), \
                    patch('crosscut.group_motion_camera_refinement._default_geometry_provider',return_value=
                        SimpleNamespace(face_bbox_xywh=(.4,.25,.1,.15))), \
                    patch('crosscut.group_motion_camera_refinement.build_group_cut_camera_proposal',side_effect=camera):
                result=refine_group_motion_camera(plan=plan,timeline=timeline,audio_offsets=(0.,0.),
                    source_timewarps=(None,None),scene_cut_times=((),()),model_path=Path('exact-model'))
            self.assertEqual(len(calls),4 if recoverable else 2)
            self.assertEqual(sum(bool(call.get('global_action_framing')) for call in calls),2 if recoverable else 0)
            self.assertEqual(result.proposals,())
            for call in calls:
                if call.get('global_action_framing'):
                    self.assertEqual(call['action_framing_model_path'],Path('exact-model'))
                    self.assertNotIn('local_focal_attention',call)

    def test_shortlist_uses_cut_endpoint_sizes_not_interior_labels(self):
        choices=tuple(replace(_choice(i,i%2,start=float(i),end=float(i+1)),
            shot_type=ShotType.CLOSE_UP,shot_scale=.9,
            exit_shot_type=ShotType.CLOSE_UP,exit_shot_scale=.3,
            entry_shot_type=ShotType.CLOSE_UP,entry_shot_scale=.3) for i in range(16))
        choices=(*choices[:13],replace(choices[13],shot_type=ShotType.MEDIUM,
            shot_scale=.35,exit_shot_scale=.6),replace(choices[14],
            entry_shot_type=ShotType.MEDIUM,entry_shot_scale=.5),choices[15])
        targets=focal_boundary_shortlist(VisualTimelinePlan(choices,1.))
        self.assertEqual(targets[0],13)
        self.assertEqual(len(targets),12)

    def test_protected_contexts_are_not_decoded_and_invalid_domains_fail_early(self):
        plan,_=produced_plan(Path('/synthetic'))
        timeline=VisualTimelinePlan(tuple(_choice(i,i,start=float(i),end=float(i+1))
            for i in range(2)),1.)
        arguments=dict(plan=plan,timeline=timeline,audio_offsets=(0.,0.),
            source_timewarps=(None,None),scene_cut_times=((),()),model_path='unused')
        with patch('crosscut.group_motion_camera_refinement.probe_render_frame_clock') as probe, \
                patch('crosscut.group_motion_camera_refinement.CloseupGestureProvider') as provider, \
                patch('crosscut.group_motion_camera_refinement.focal_boundary_shortlist',return_value=(0,)):
            result=refine_group_motion_camera(**arguments,protected_segment_indices=(1,))
            self.assertEqual(result.plan,plan)
            self.assertEqual(result.timeline,timeline)
            self.assertEqual(result.proposals,())
            self.assertEqual(result.attempts[0]['stage'],'protected_context')
            for protected in ((True,),(-1,),(2,),(1,1)):
                with self.subTest(protected=protected),self.assertRaises(ValueError):
                    refine_group_motion_camera(**arguments,protected_segment_indices=protected)
            probe.assert_not_called();provider.assert_not_called()


class MotionCameraBundleTests(unittest.TestCase):
    def fixture(self):
        plan, motion = produced_plan(Path('/synthetic'))
        timeline = VisualTimelinePlan(tuple(_choice(index, index,
            start=float(index), end=float(index+1)) for index in range(2)), 1.)
        section = SectionPlan((SectionDirective(0, SectionRole.BUILD, 0., 2., .5, .5,
            1, 4, 15., .4, .1, False, False),))
        request = SimpleNamespace(sources=plan.sources, reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE)
        bundle = _review_first_group_bundle(timeline=timeline, request=request,
            eligible_indices=(0, 1), offsets=(0., 0.), timewarps=(None, None),
            common_start=0., section_plan=section, phrase_times=(0., 1., 2.),
            output=plan.output, reporter=SimpleNamespace(log=lambda *_args, **_kwargs: None), warnings=[])
        bundle = replace(bundle, render_plan=plan, editorial_motion_plan=motion,
            group_transition_plan=replace(bundle.group_transition_plan, transitions=plan.transitions))
        delta = 2/30.
        left, right = plan.segments
        pair = (replace(left, duration=left.duration+delta, editorial_transform=None),
            replace(right, master_start=right.master_start+delta,
                source_start=right.source_start+delta, duration=right.duration-delta,
                editorial_transform=None))
        def align(**arguments):
            return build_boundary_subject_alignment_plan(**arguments,
                frame_provider=lambda index, _seconds: _frame(index),
                observation_provider=_observations(source0_box=(.35, .23, .225, .27),
                    source1_box=(.40, .26, .20, .24)), source_resolutions=((1920, 1080),)*2)
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',
                   side_effect=align):
            proposal = build_group_cut_camera_proposal(plan=plan, timeline=timeline,
                boundary_index=0, shifted_segments=pair,
                source_clocks={i: VerifiedRenderFrameClock(Fraction(30), 3000, 'a'*64)
                    for i in range(2)}, scene_cut_times=((), ()))
        result = GroupMotionCameraRefinement(proposal.plan, proposal.clock_timeline,
            (proposal,), (), .5)
        return bundle, request, result

    def run_bundle(self, bundle, request, result):
        with patch('crosscut.group_motion_camera_refinement.refine_group_motion_camera',
                   return_value=result):
            return _review_first_group_motion_camera_bundle(bundle, request=request,
                eligible_indices=(0, 1), offsets=(0., 0.), timewarps=(None, None),
                scene_cut_times=((), ()), phrase_times=(0., 1., 2.), model_path=Path('unused'))

    def test_complete_retime_and_one_sided_camera_promote_atomically(self):
        bundle, request, result = self.fixture()
        before = bundle.render_plan.to_json()
        changed = self.run_bundle(bundle, request, result)
        self.assertIsNotNone(changed)
        _validate_review_first_group_bundle(changed, eligible_indices=(0, 1),
            offsets=(0., 0.), timewarps=(None, None), common_start=0.)
        self.assertEqual(changed.render_plan, result.plan)
        self.assertEqual(changed.visual_timeline, result.timeline)
        self.assertNotEqual(changed.stage_edit_exact_shot_clock, bundle.stage_edit_exact_shot_clock)
        self.assertFalse(changed.candidate_lineage[-1]['quality_approved'])
        self.assertIsNone(changed.boundary_repairs)
        self.assertEqual(bundle.render_plan.to_json(), before)
        self.assertIsNone(changed.render_plan.segments[0].editorial_transform)
        self.assertEqual(changed.editorial_motion_plan.authored_event_counts,
                         bundle.editorial_motion_plan.authored_event_counts)
        from crosscut.post_render_qa import _transform_metrics, _IssueCollector, PostRenderQAConfig
        import json
        collector=_IssueCollector()
        metrics=_transform_metrics({'editorial_motion':changed.editorial_motion_plan.to_dict()},
            json.loads(json.dumps(changed.render_plan.to_json())),PostRenderQAConfig(),collector)
        self.assertTrue(all(metrics['count_reconciliation_checks'].values()),
                        metrics['count_reconciliation_checks'])

    def test_empty_optional_pass_retains_failure_evidence_without_editing_the_plan(self):
        bundle,request,_=self.fixture()
        result=GroupMotionCameraRefinement(bundle.render_plan,bundle.visual_timeline,(),
            ({'stage':'complete_camera_candidate','boundary_index':0,'admitted':False,
              'reason':'per_frame_head_unobserved'},),.4)
        after=self.run_bundle(bundle,request,result)
        self.assertEqual(after.render_plan,bundle.render_plan)
        self.assertEqual(after.visual_timeline,bundle.visual_timeline)
        self.assertEqual(after.editorial_motion_plan,bundle.editorial_motion_plan)
        self.assertEqual(after.stage_edit_exact_shot_clock,bundle.stage_edit_exact_shot_clock)
        self.assertIs(after.timeline_review,bundle.timeline_review)
        audit=after.candidate_lineage[-1]
        self.assertFalse(audit['applied'])
        self.assertEqual(audit['changed_boundary_count'],0)
        self.assertEqual(audit['attempts'][0]['reason'],'per_frame_head_unobserved')
        self.assertFalse(audit['quality_approved'])

    def test_stale_clock_or_incomplete_dense_receipt_cannot_promote(self):
        bundle, request, result = self.fixture()
        proposal = result.proposals[0]
        invalid_alignment = replace(proposal.alignment, decisions=(
            replace(proposal.alignment.decisions[0], validated_output_frame_count=0),))
        candidates = (replace(result, timeline=bundle.visual_timeline),
            replace(result, proposals=(replace(proposal, alignment=invalid_alignment),)))
        before = bundle.render_plan.to_json()
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.run_bundle(bundle, request, candidate)
        self.assertEqual(bundle.render_plan.to_json(), before)

    def test_later_look_or_learned_clock_authority_is_not_silently_replaced(self):
        bundle, request, result = self.fixture()
        for values in ({'boundary_look_match': object()}, {'cut_local_sync': {}},
                       {'dynamic_crop_application': {}}):
            with self.subTest(values=values):
                self.assertIsNone(self.run_bundle(replace(bundle, **values), request, result))

    def test_experimental_subject_authority_cannot_be_relabelled_as_the_incumbent(self):
        from crosscut.boundary_subject_alignment import FOCAL_EXPERIMENT_BASIS, ACTION_EXPERIMENT_BASIS, SHOT_CAMERA_EXPERIMENT_BASIS
        bundle, request, result = self.fixture()
        proposal = result.proposals[0]
        for basis in (FOCAL_EXPERIMENT_BASIS,ACTION_EXPERIMENT_BASIS,SHOT_CAMERA_EXPERIMENT_BASIS,'unknown_new_basis'):
            changed = replace(result,proposals=(replace(proposal,
                alignment=replace(proposal.alignment,analysis_basis=basis)),))
            with self.subTest(basis=basis), self.assertRaisesRegex(ValueError,'authority'):
                self.run_bundle(bundle,request,changed)
        old_basis = replace(bundle,editorial_motion_plan=replace(bundle.editorial_motion_plan,
            boundary_subject_alignment_plan=replace(bundle.editorial_motion_plan.boundary_subject_alignment_plan,
                analysis_basis='exact_cut_frame_shot_adaptive_subject_alignment_v4')))
        with self.assertRaisesRegex(ValueError,'authority'):
            self.run_bundle(old_basis,request,result)

    def test_explicit_global_action_receipt_keeps_its_own_mixed_plan_header(self):
        from crosscut.boundary_subject_alignment import GLOBAL_ACTION_BASIS
        bundle,request,result=self.fixture();proposal=result.proposals[0]
        action=replace(proposal.alignment,analysis_basis=GLOBAL_ACTION_BASIS,global_action_framing=True)
        changed=self.run_bundle(bundle,request,replace(result,proposals=(replace(proposal,alignment=action),)))
        alignment=changed.editorial_motion_plan.boundary_subject_alignment_plan
        self.assertEqual(alignment.analysis_basis,GLOBAL_ACTION_BASIS)
        self.assertTrue(alignment.global_action_framing)
        self.assertFalse(alignment.to_dict()['global_action_framing']['central_face_authority'])
        self.assertFalse(changed.candidate_lineage[-1]['quality_approved'])

    def test_unrelated_retime_keeps_independently_bound_context_in_real_bundle(self):
        import engine.tests.test_shot_context_camera as cameras
        from crosscut.editorial_motion import build_editorial_motion_plan
        from crosscut.shot_context_refinement import attach_shot_context_proposal
        from crosscut.shot_context_binding import bound_shot_context_cameras
        from crosscut.post_render_qa import _bound_local_alignment_tracks
        import json
        fixture=cameras.ShotContextProposalTests();args=fixture.fixture();sources=args['plan'].sources
        source_order=(0,2,0,1,2)
        timeline=VisualTimelinePlan(tuple(_choice(i,source,start=float(i),end=float(i+1))
            for i,source in enumerate(source_order)),1.)
        transitions=tuple(args['plan'].transitions[0] for _ in range(4))
        def align(**arguments):
            return build_boundary_subject_alignment_plan(**arguments,
                frame_provider=lambda index,_seconds:_frame(index),
                observation_provider=_observations(source0_box=(.35,.23,.225,.27),
                    source1_box=(.40,.26,.20,.24)),source_resolutions=((1920,1080),)*3)
        alignment=align(visual_timeline=timeline,sources=sources,transitions=transitions,
            fps=30.,boundary_indices=(0,))
        motion=build_editorial_motion_plan(visual_timeline=timeline,transitions=transitions,
            fps=30.,boundary_subject_alignment_plan=alignment)
        plan=replace(args['plan'],segments=tuple(RenderSegment(source,float(i),1.,float(i),
            editorial_transform=motion.track_for(i)) for i,source in enumerate(source_order)),
            transitions=transitions)
        context=fixture.run_proposal(dict(args,plan=plan,segment_index=3))
        motion=attach_shot_context_proposal(plan,motion,context);plan=context.plan
        request=SimpleNamespace(sources=sources,reference_source=0,aspect_ratio=AspectRatio.LANDSCAPE)
        section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,5.,.5,.5,
            1,4,15.,.4,.1,False,False),))
        bundle=_review_first_group_bundle(timeline=timeline,request=request,
            eligible_indices=(0,1,2),offsets=(0.,)*3,timewarps=(None,)*3,common_start=0.,
            section_plan=section,phrase_times=(0.,5.),output=plan.output,
            reporter=SimpleNamespace(log=lambda *_a,**_k:None),warnings=[])
        bundle=replace(bundle,render_plan=plan,editorial_motion_plan=motion,
            group_transition_plan=replace(bundle.group_transition_plan,transitions=transitions))
        delta=2/30.;left,right=plan.segments[:2]
        pair=(replace(left,duration=1.+delta,editorial_transform=None),
            replace(right,master_start=1.+delta,source_start=1.+delta,duration=1.-delta,
                editorial_transform=None))
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',side_effect=align):
            proposal=build_group_cut_camera_proposal(plan=plan,timeline=timeline,boundary_index=0,
                shifted_segments=pair,source_clocks={i:VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)
                    for i in range(3)},scene_cut_times=((),)*3,preserved_boundary_frames=4)
        result=GroupMotionCameraRefinement(proposal.plan,proposal.clock_timeline,(proposal,),(),.2)
        with patch('crosscut.group_motion_camera_refinement.refine_group_motion_camera',return_value=result) as refine:
            changed=_review_first_group_motion_camera_bundle(bundle,request=request,
                eligible_indices=(0,1,2),offsets=(0.,)*3,timewarps=(None,)*3,
                scene_cut_times=((),)*3,phrase_times=(0.,5.),model_path=Path('unused'))
        self.assertEqual(refine.call_args.kwargs['protected_segment_indices'],(2,3,4))
        raw=changed.editorial_motion_plan.to_dict()
        self.assertEqual(set(bound_shot_context_cameras(raw,changed.render_plan)),{3})
        self.assertEqual(plan.segments[2:],changed.render_plan.segments[2:])
        old_bound=_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json())))
        new_bound=_bound_local_alignment_tracks(raw,json.loads(json.dumps(changed.render_plan.to_json())))
        self.assertTrue(old_bound[0]<=new_bound[0]);self.assertTrue(old_bound[1]<=new_bound[1])
        self.assertEqual(changed.candidate_lineage[-1]['unchanged_shot_context_receipts_rebound'],1)
        self.assertFalse(changed.candidate_lineage[-1]['quality_approved'])

    def test_invalid_existing_context_is_rejected_before_expensive_refinement(self):
        bundle,request,result=self.fixture()
        corrupted=replace(bundle,editorial_motion_plan=replace(bundle.editorial_motion_plan,
            shot_context_cameras=({'contract':'unverified_context'},)))
        with patch('crosscut.group_motion_camera_refinement.refine_group_motion_camera') as refine:
            with self.assertRaisesRegex(ValueError,'prior'):
                _review_first_group_motion_camera_bundle(corrupted,request=request,
                    eligible_indices=(0,1),offsets=(0.,0.),timewarps=(None,None),
                    scene_cut_times=((),()),phrase_times=(0.,2.),model_path=Path('unused'))
            refine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
