from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.closeup_gesture import CloseupGestureEvidence
from crosscut.pipeline import _review_first_closeup_gesture_repair
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.timeline import ShotType
from crosscut.visual_boundary_repair import (repair_visual_timeline_boundaries, VisualBoundaryRepairPlan,
    _gesture_preserves_visual_comparison)
from crosscut.visual_score import VisualTimelinePlan
from tests.test_visual_boundary_repair import _choice, _descriptor


class CloseupGestureRepairTests(unittest.TestCase):
    def test_arm_preference_cannot_trade_away_existing_visual_cost_or_coverage(self):
        before=dict(total_cost=.2,base_total_cost=.2,focus_mismatch=.1)
        self.assertTrue(_gesture_preserves_visual_comparison(dict(before,total_cost=.1),before))
        for after in (dict(before,total_cost=.3),dict(before,total_cost=.1,base_total_cost=.3),
                      dict(before,total_cost=.1,focus_mismatch=None),dict(before,total_cost=float('nan'))):
            self.assertFalse(_gesture_preserves_visual_comparison(after,before))

    def setUp(self):
        self.timeline=VisualTimelinePlan(tuple(replace(_choice(i,i%2),shot_type=ShotType.CLOSE_UP)
            for i in range(6)),1.)
        self.arguments=dict(visual_timeline=self.timeline, sources=(Path('a'),Path('b')),
            audio_offsets=(0.,0.),scene_cut_times=((),()), output_fps=30.,boundary_indices=(1,),
            render_frame_clocks=(VerifiedRenderFrameClock(Fraction(30),1000,'a'*64),)*2,
            descriptor_provider=lambda source,seconds: _descriptor(.5),
            closeup_gesture_model_path=Path('exact-model.onnx'))

    @staticmethod
    def compare(left,right):
        first=round(right[1]*30)
        cost=.01 if first==58 else .5
        return CloseupGestureEvidence(round(left[1]*30),first,(cost,)*4)

    def test_exact_gesture_moves_only_requested_boundary_and_keeps_old_pose_basis(self):
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=self.compare
            result=repair_visual_timeline_boundaries(**self.arguments)
            provider.return_value.close.assert_called_once()
        self.assertEqual(1,result.changed_count)
        repair=result.repairs[0]
        self.assertEqual(-2,repair.result.selected_frame_offset)
        self.assertEqual('actual_pose_pair_unavailable',repair.selected_visual_evidence.pose_basis)
        self.assertIsNone(repair.selected_visual_evidence.pose_mismatch)
        self.assertEqual(.01,repair.selected_visual_evidence.selection_cost)
        self.assertEqual(.01,repair.result.selected.visual_cost)
        self.assertEqual(repair.selected_visual_evidence.base_total_cost,
            repair.selected_visual_evidence.total_cost)
        self.assertIn('closeup_gesture',repair.selected_visual_evidence.to_dict())
        for i,(old,new) in enumerate(zip(self.timeline.choices,result.timeline.choices)):
            self.assertEqual(old.source_index,new.source_index)
            if i not in (1,2): self.assertEqual(old,new)
        self.assertEqual(sum(c.duration_seconds for c in self.timeline.choices),
            sum(c.duration_seconds for c in result.timeline.choices))
        from crosscut.post_render_qa import _validate_temporal_pose_motion_evidence
        configuration=result.to_dict()['temporal_pose_motion_configuration']
        _, errors=_validate_temporal_pose_motion_evidence(repair.selected_visual_evidence.to_dict(),
            configurations_by_fingerprint={configuration['configuration_fingerprint_sha256']:configuration})
        self.assertEqual((),errors)  # Articulation is NOT a forged dynamic-motion bonus.

    def test_no_observed_baseline_cannot_be_claimed_as_improvement(self):
        def compare(left,right):
            return None if round(right[1]*30)==60 else self.compare(left,right)
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=compare
            result=repair_visual_timeline_boundaries(**self.arguments)
        self.assertEqual(0,result.changed_count)
        self.assertEqual(self.timeline.choices,result.timeline.choices)
        self.assertIn('no_proven_articulation_improvement',result.failures[0]['reason'])

    def test_partial_incumbent_can_improve_only_on_same_or_superset_of_bones(self):
        def compare(left,right):
            first=round(right[1]*30)
            values={58:(None,.001,.001,None),59:(.15,.15,.15,None)}.get(first,(.5,.5,.5,None))
            return CloseupGestureEvidence(round(left[1]*30),first,values)
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=compare
            result=repair_visual_timeline_boundaries(**self.arguments)
        self.assertEqual(1,result.changed_count)
        repair=result.repairs[0]
        # The superficially cheapest -2 drops a visible arm. The optimizer
        # must actually try the next admissible candidate, not reject the
        # whole optional pass after having selected the unsupported winner.
        self.assertEqual(-1,repair.result.selected_frame_offset)
        self.assertEqual(frozenset((0,1,2)),repair.selected_visual_evidence.closeup_gesture.observed_bones)
        original=next(row for row in repair.candidate_evidence_audit if row['frame_offset']==0)
        self.assertEqual(3,original['visual_evidence']['closeup_gesture']['bone_count'])
        hidden=next(row for row in repair.candidate_evidence_audit if row['frame_offset']==-2)
        self.assertTrue(hidden['production_transition_gate_passed'])
        self.assertFalse(hidden['gesture_preserves_incumbent_observations'])
        # Validate this again at publication, independently of the optimizer.
        audit=repair.to_dict()['candidate_evidence_audit']
        before=next(row for row in audit if row['frame_offset']==0)
        before['visual_evidence']['closeup_gesture']=CloseupGestureEvidence(59,60,(.5,)*4).to_dict()
        with self.assertRaisesRegex(ValueError,'incumbent gesture observations'):
            replace(repair,candidate_evidence_audit=tuple(audit))
        audit=repair.to_dict()['candidate_evidence_audit']
        before=next(row for row in audit if row['frame_offset']==0)
        before['visual_evidence']['total_cost']=-1.
        with self.assertRaisesRegex(ValueError,'regresses incumbent visual evidence'):
            replace(repair,candidate_evidence_audit=tuple(audit))

    def test_newly_visible_bad_arm_cannot_hide_behind_old_partial_mask(self):
        def compare(left,right):
            first=round(right[1]*30)
            values=(.01,.01,.01,.8) if first==58 else (.5,.5,.5,None)
            return CloseupGestureEvidence(round(left[1]*30),first,values)
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=compare
            result=repair_visual_timeline_boundaries(**self.arguments)
        self.assertEqual(0,result.changed_count)
        self.assertEqual(tuple((c.source_index,c.canonical_master_start,c.canonical_master_end,
                                c.source_local_start,c.source_local_end) for c in self.timeline.choices),
                         tuple((c.source_index,c.canonical_master_start,c.canonical_master_end,
                                c.source_local_start,c.source_local_end) for c in result.timeline.choices))

    def test_unverified_clock_or_missing_gesture_keeps_complete_incumbent(self):
        for clock in (None,self.arguments['render_frame_clocks']):
            arguments=dict(self.arguments,render_frame_clocks=clock)
            with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
                provider.return_value.compare.return_value=None
                result=repair_visual_timeline_boundaries(**arguments)
                if clock is None: provider.assert_not_called()
            self.assertEqual(self.timeline.choices,result.timeline.choices)

    def test_perfect_arm_cost_cannot_bypass_raw_geometry_gate(self):
        arguments=dict(self.arguments,descriptor_provider=lambda source,seconds:
            _descriptor(.95 if source==0 and round(seconds*30)==58 else .5))
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=self.compare
            result=repair_visual_timeline_boundaries(**arguments)
        self.assertFalse(any(r.result.selected_frame_offset==-2 for r in result.repairs))
        self.assertEqual(self.timeline.choices,result.timeline.choices)

    def test_wrapper_calls_one_bounded_refinement_and_retains_success_lineage(self):
        earlier=VisualBoundaryRepairPlan(self.timeline,(self.timeline,),(),0,())
        kwargs={k:v for k,v in self.arguments.items() if k not in ('visual_timeline','boundary_indices')}
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=self.compare
            result=_review_first_closeup_gesture_repair(earlier,**kwargs)
        self.assertIsNotNone(result)
        self.assertEqual(2,len(result.attempt_input_timelines))
        self.assertIn('single_exact_closeup',result.analysis_basis)
        self.assertEqual(tuple(c.source_index for c in self.timeline.choices),
            tuple(c.source_index for c in result.timeline.choices))

    def test_no_asset_access_for_wide_only_path(self):
        timeline=replace(self.timeline,choices=tuple(replace(c,shot_type=ShotType.WIDE)
            for c in self.timeline.choices))
        earlier=VisualBoundaryRepairPlan(timeline,(timeline,),(),0,())
        with patch('crosscut.pipeline.repair_visual_timeline_boundaries') as repair, \
             patch('crosscut.source_first_model_assets.ensure_source_first_model_assets') as assets:
            self.assertIsNone(_review_first_closeup_gesture_repair(earlier,sources=(Path('a'),Path('b'))))
        repair.assert_not_called(); assets.assert_not_called()

    def test_medium_foreground_pairs_receive_same_bounded_exact_audition(self):
        timeline=replace(self.timeline, choices=tuple(replace(c,shot_type=ShotType.MEDIUM)
                                                     for c in self.timeline.choices))
        earlier=VisualBoundaryRepairPlan(timeline,(timeline,),(),0,())
        kwargs={k:v for k,v in self.arguments.items() if k not in ('visual_timeline','boundary_indices')}
        with patch('crosscut.visual_boundary_repair.CloseupGestureProvider') as provider:
            provider.return_value.compare.side_effect=self.compare
            result=_review_first_closeup_gesture_repair(earlier,**kwargs)
        self.assertIsNotNone(result)
        self.assertEqual(tuple(c.source_index for c in timeline.choices),
                         tuple(c.source_index for c in result.timeline.choices))

    def test_more_medium_pairs_do_not_increase_twelve_boundary_work_limit(self):
        choices=tuple(replace(_choice(i,i%2),shot_type=ShotType.MEDIUM) for i in range(30))
        choices=choices[:7]+tuple(replace(c,shot_type=ShotType.CLOSE_UP) for c in choices[7:9])+choices[9:]
        timeline=VisualTimelinePlan(choices,1.)
        earlier=VisualBoundaryRepairPlan(timeline,(timeline,),(),0,())
        with patch('crosscut.pipeline.repair_visual_timeline_boundaries',return_value=earlier) as repair:
            result=_review_first_closeup_gesture_repair(earlier,sources=(Path('a'),Path('b')),
                                                      closeup_gesture_model_path=Path('pose.onnx'))
        self.assertIsNone(result)
        targets=repair.call_args.kwargs['boundary_indices']
        self.assertEqual(12,len(targets))
        self.assertIn(7,targets)
