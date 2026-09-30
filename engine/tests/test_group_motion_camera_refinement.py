from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.boundary_subject_alignment import build_boundary_subject_alignment_plan
from crosscut.closeup_gesture import CloseupGestureEvidence, ObservedArmMotion
from crosscut.group_cut_camera_proposal import build_group_cut_camera_proposal
from crosscut.group_motion_camera_refinement import (
    GroupMotionCameraRefinement, camera_motion_rank, refine_group_motion_camera,
    focal_boundary_shortlist,
    measured_motion_improves,
)
from crosscut.ive import AspectRatio
from crosscut.pipeline import (
    _review_first_group_bundle, _review_first_group_motion_camera_bundle,
    _validate_review_first_group_bundle,
)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
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
        from crosscut.boundary_subject_alignment import FOCAL_EXPERIMENT_BASIS, ACTION_EXPERIMENT_BASIS
        bundle, request, result = self.fixture()
        proposal = result.proposals[0]
        for basis in (FOCAL_EXPERIMENT_BASIS,ACTION_EXPERIMENT_BASIS,'unknown_new_basis'):
            changed = replace(result,proposals=(replace(proposal,
                alignment=replace(proposal.alignment,analysis_basis=basis)),))
            with self.subTest(basis=basis), self.assertRaisesRegex(ValueError,'authority'):
                self.run_bundle(bundle,request,changed)
        old_basis = replace(bundle,editorial_motion_plan=replace(bundle.editorial_motion_plan,
            boundary_subject_alignment_plan=replace(bundle.editorial_motion_plan.boundary_subject_alignment_plan,
                analysis_basis='exact_cut_frame_shot_adaptive_subject_alignment_v4')))
        with self.assertRaisesRegex(ValueError,'authority'):
            self.run_bundle(old_basis,request,result)


if __name__ == '__main__':
    unittest.main()
