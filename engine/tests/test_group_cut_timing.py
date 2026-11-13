from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.group_cut_timing import propose_numeric_group_cut_timing,native_flash_timing_frontier
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import (RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,
    EditorialTransformKeyframe,TransitionKind,segment_frame_counts)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.group_actual_shots import visible_group_cut_inventory


class GroupCutTimingTests(unittest.TestCase):
    def fixture(self,fps=30.,native_rate=Fraction(30)):
        sources=tuple(Path(f'unopened-cut-source-{i}.mp4') for i in range(4))
        clock=VerifiedRenderFrameClock(native_rate,10000,'a'*64)
        domain=GroupSourceDomain(sources,(0,2,3),(0.,)*3,(None,)*3,((),)*3,(clock,)*3)
        parts=(RenderSegment(0,0.,1.,0.),RenderSegment(2,1.,1.,1.),
            RenderSegment(2,2.,1.,2.),RenderSegment(3,3.,1.,3.),RenderSegment(3,4.,1.,4.))
        transitions=(RenderTransition(),RenderTransition(tier='D',reason='same_native_hold'),
            RenderTransition(),RenderTransition(tier='D',reason='same_native_hold'))
        plan=RenderPlan(sources,parts,0,'16:9',Path('not-rendered-cut-preview.mp4'),fps=fps,transitions=transitions)
        return plan,domain

    def test_positive_move_changes_only_four_tick_source_owners_and_whole_actual_rates(self):
        plan,domain=self.fixture();result=propose_numeric_group_cut_timing(plan=plan,domain=domain,
            boundary_index=2,timing_offset_frames=4)
        self.assertEqual((result.old_cut_output_tick,result.new_cut_output_tick),(90,94))
        self.assertEqual(result.changed_source_owner_ticks,(90,91,92,93))
        self.assertEqual(segment_frame_counts(result.plan),(30,30,34,26,30))
        self.assertEqual(result.plan.segments[0],plan.segments[0]);self.assertEqual(result.plan.segments[1],plan.segments[1])
        self.assertEqual(result.plan.segments[4],plan.segments[4]);self.assertFalse(result.to_dict()['candidate_admitted'])

    def test_negative_move_preserves_fixed_samples_before_and_after_the_new_cut(self):
        plan,domain=self.fixture();result=propose_numeric_group_cut_timing(plan=plan,domain=domain,
            boundary_index=2,timing_offset_frames=-4)
        before=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(150))
        after=native_frame_clocks_for_output(result.plan,domain.clocks_by_request,range(150))
        self.assertEqual((result.old_cut_output_tick,result.new_cut_output_tick),(90,86))
        for tick in (*range(86),*range(90,150)):
            self.assertEqual((before[tick].source_index,before[tick].native_frame_index),
                (after[tick].source_index,after[tick].native_frame_index))

    def test_fractional_5994_to_2997_and_repeated_native_23976_to_2997_are_replayed(self):
        for rate in (Fraction(60000,1001),Fraction(24000,1001)):
            for offset in (-4,4):
                plan,domain=self.fixture(29.97,rate)
                result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=offset)
                self.assertEqual(result.plan.output_frame_count,plan.output_frame_count)
                self.assertEqual(result.new_cut_output_tick-result.old_cut_output_tick,offset)

    def test_actual_native_cut_inside_right_cell_keeps_its_exact_output_tick(self):
        plan,domain=self.fixture();domain=replace(domain,scene_cut_times=((),(),(3.5,)))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)
        self.assertTrue(result.actual_native_camera_events_preserved)

    def test_clipping_an_existing_native_camera_cut_is_not_permitted(self):
        plan,domain=self.fixture();domain=replace(domain,scene_cut_times=((),(),(3.05,)))
        with self.assertRaisesRegex(ValueError,'actual native camera cut'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)

    def test_explicit_repair_collapses_five_frame_outgoing_native_flash(self):
        plan,domain=self.fixture()
        domain=replace(domain,scene_cut_times=((),(85/30,),()))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
            timing_offset_frames=-5,collapse_incident_native_flash=True)
        self.assertEqual((result.old_cut_output_tick,result.new_cut_output_tick),(90,85))
        self.assertEqual(result.changed_source_owner_ticks,tuple(range(85,90)))
        self.assertEqual([(e.kind,e.first_incoming_output_frame)
            for e in visible_group_cut_inventory(result.plan,domain)],
            [('source_switch',30),('source_switch',85)])
        row=result.to_dict()
        self.assertFalse(row['actual_native_camera_events_preserved'])
        self.assertTrue(row['unaffected_native_camera_events_preserved'])
        self.assertEqual(row['collapsed_native_camera_event']['native_cut_frame'],85)
        self.assertFalse(row['candidate_admitted'])

    def test_explicit_repair_collapses_incoming_flash_at_its_native_cut(self):
        plan,domain=self.fixture()
        domain=replace(domain,scene_cut_times=((),(),(95/30,)))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
            timing_offset_frames=5,collapse_incident_native_flash=True)
        self.assertEqual(result.changed_source_owner_ticks,tuple(range(90,95)))
        self.assertEqual([(e.kind,e.first_incoming_output_frame)
            for e in visible_group_cut_inventory(result.plan,domain)],
            [('source_switch',30),('source_switch',95)])

    def test_flash_repair_does_not_waive_other_native_camera_events(self):
        plan,domain=self.fixture()
        # Extending outgoing source 2 would add a different native cut at 93.
        domain=replace(domain,scene_cut_times=((),(93/30,),(95/30,)))
        with self.assertRaisesRegex(ValueError,'unrelated actual cut'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
                timing_offset_frames=5,collapse_incident_native_flash=True)

    def test_flash_repair_requires_exact_adjacent_native_anchor_not_nearby_timestamp(self):
        plan,domain=self.fixture()
        domain=replace(domain,scene_cut_times=((),(85/30,),()))
        with self.assertRaisesRegex(ValueError,'exact adjacent native flash'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
                timing_offset_frames=-4,collapse_incident_native_flash=True)

    def test_flash_repair_preserves_every_native_sample_outside_owned_interval(self):
        plan,domain=self.fixture()
        domain=replace(domain,scene_cut_times=((),(85/30,),()))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
            timing_offset_frames=-5,collapse_incident_native_flash=True)
        before=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(150))
        after=native_frame_clocks_for_output(result.plan,domain.clocks_by_request,range(150))
        for tick in (*range(85),*range(90,150)):
            self.assertEqual((before[tick].source_index,before[tick].native_frame_index),
                (after[tick].source_index,after[tick].native_frame_index))

    def test_native_flash_policy_requires_explicit_boolean(self):
        plan,domain=self.fixture()
        for policy in (None,1,'yes'):
            with self.assertRaisesRegex(TypeError,'boolean'):
                propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,
                    timing_offset_frames=5,collapse_incident_native_flash=policy)

    def test_native_flash_frontier_uses_exact_renderer_ticks_and_not_source_quick_cuts(self):
        plan,domain=self.fixture()
        self.assertEqual(native_flash_timing_frontier(plan=plan,domain=domain),())
        domain=replace(domain,scene_cut_times=((),(85/30,),(95/30,)))
        hints=native_flash_timing_frontier(plan=plan,domain=domain)
        self.assertEqual([(h.boundary_index,h.timing_offset_frames) for h in hints],[(2,-5),(2,5)])
        self.assertTrue(all(h.to_dict()['decoded_pixels_reclassified'] is False for h in hints))
        self.assertEqual(native_flash_timing_frontier(plan=plan,domain=domain,maximum_flash_frames=4),())
        for limit in (True,0,9):
            with self.assertRaises(ValueError):native_flash_timing_frontier(plan=plan,domain=domain,
                maximum_flash_frames=limit)

    def test_normal_motion_camera_offset_list_keeps_exact_odd_frame_flash_hypothesis(self):
        from crosscut.group_motion_camera_refinement import boundary_timing_hypotheses
        plan,domain=self.fixture();domain=replace(domain,scene_cut_times=((),(85/30,),()))
        hints=native_flash_timing_frontier(plan=plan,domain=domain)
        rows=boundary_timing_hypotheses(boundary_index=2,native_flashes=hints,allow_retiming=True)
        self.assertEqual(rows[0],(-5,True))
        self.assertEqual(rows[1:],tuple((offset,False) for offset in (-6,-4,-2,2,4,6)))
        self.assertEqual(boundary_timing_hypotheses(boundary_index=2,native_flashes=hints,
            allow_retiming=False),())
        self.assertTrue(all(not flag for _,flag in boundary_timing_hypotheses(boundary_index=0,
            native_flashes=hints,allow_retiming=True)))

    def test_even_frame_exact_flash_does_not_keep_a_duplicate_ordinary_hypothesis(self):
        from crosscut.group_motion_camera_refinement import boundary_timing_hypotheses
        plan,domain=self.fixture();domain=replace(domain,scene_cut_times=((),(86/30,),()))
        hints=native_flash_timing_frontier(plan=plan,domain=domain)
        rows=boundary_timing_hypotheses(boundary_index=2,native_flashes=hints,allow_retiming=True)
        self.assertEqual(rows[0],(-4,True))
        self.assertEqual(sum(offset==-4 for offset,_ in rows),1)

    def test_moving_cut_cannot_introduce_a_new_outgoing_native_camera_event(self):
        plan,domain=self.fixture();domain=replace(domain,scene_cut_times=((),(3.05,),()))
        with self.assertRaisesRegex(ValueError,'actual native camera cut'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)

    def test_same_source_internal_cell_is_not_a_real_cross_source_cut(self):
        plan,domain=self.fixture()
        with self.assertRaisesRegex(ValueError,'actual cross-source'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=1,timing_offset_frames=4)

    def test_zero_boolean_fractional_large_or_invalid_boundary_is_rejected(self):
        plan,domain=self.fixture()
        for offset in (0,True,1.5,9,-9):
            with self.assertRaises(ValueError):propose_numeric_group_cut_timing(plan=plan,domain=domain,
                boundary_index=2,timing_offset_frames=offset)
        for boundary in (-1,True,4):
            with self.assertRaises(ValueError):propose_numeric_group_cut_timing(plan=plan,domain=domain,
                boundary_index=boundary,timing_offset_frames=4)

    def test_too_short_cell_and_terminal_clone_are_rejected(self):
        plan,domain=self.fixture()
        small=replace(plan,segments=(replace(plan.segments[0],duration=2/30),
            replace(plan.segments[1],master_start=2/30,duration=1-2/30),
            *tuple(replace(s,master_start=s.master_start-1) for s in plan.segments[2:])),
            transitions=(RenderTransition(),RenderTransition(),*plan.transitions[2:]))
        with self.assertRaises(ValueError):propose_numeric_group_cut_timing(plan=small,domain=domain,
            boundary_index=0,timing_offset_frames=-1)
        short=replace(domain,render_frame_clocks=(*domain.render_frame_clocks[:2],VerifiedRenderFrameClock(Fraction(30),145,'b'*64)))
        with self.assertRaisesRegex(ValueError,'terminal clone'):
            propose_numeric_group_cut_timing(plan=plan,domain=short,boundary_index=2,timing_offset_frames=4)

    def test_declared_phase_receipt_cannot_be_borrowed_across_new_bounds(self):
        from tests import test_shot_video_clock as clock_fixture
        from crosscut.group_joint_state_factory import assemble_single_group_source_camera_state
        from crosscut.shot_video_clock import attach_authorized_shot_video_clocks
        args,context,_,phase,clock=clock_fixture.ShotVideoClockTests().prepare()
        path=args['observations'].path
        plan,_=assemble_single_group_source_camera_state(plan=context.incumbent_plan,
            domain=args['domain'],path=(path[0],phase,path[2]))
        plan=attach_authorized_shot_video_clocks(plan=plan,context=context,clocks=(clock,))
        domain=args['domain']
        with self.assertRaisesRegex(ValueError,'renewed external shot-clock authority'):
            propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)

    def test_absolute_camera_and_sampling_canvas_are_retained_after_trimming_peak(self):
        plan,domain=self.fixture();track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.3),
            EditorialTransformKeyframe(.2,1.15),EditorialTransformKeyframe(1.,1.1)))
        plan=replace(plan,segments=(*plan.segments[:3],replace(plan.segments[3],editorial_transform=track),
            replace(plan.segments[4],editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),)))))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)
        after=result.plan.segments[3].editorial_transform
        self.assertEqual(after.effective_sampling_canvas_zoom,track.effective_sampling_canvas_zoom)
        for frame in (0,3,20,25):
            a=after.evaluate(frame/plan.fps);b=track.evaluate((frame+4)/plan.fps)
            for name in ('zoom','center_x','center_y','roll_degrees'):self.assertAlmostEqual(getattr(a,name),getattr(b,name))

    def test_source_phase_is_retained_not_reset_to_the_global_audio_alignment(self):
        plan,domain=self.fixture();plan=replace(plan,segments=tuple(replace(s,source_start=s.source_start+.1)
            if s.source_index==2 else s for s in plan.segments))
        result=propose_numeric_group_cut_timing(plan=plan,domain=domain,boundary_index=2,timing_offset_frames=4)
        self.assertEqual(result.plan.segments[1].source_start,1.1)
        self.assertEqual(result.plan.segments[2].source_start,2.1)
        self.assertFalse(result.to_dict()['identity_phase_lip_sync_or_pose_certified'])

    def test_long_plan_cannot_allocate_an_unbounded_native_sample_dictionary(self):
        plan,domain=self.fixture()
        long=replace(plan,segments=(RenderSegment(0,0.,1000.,0.),RenderSegment(2,1000.,1000.,1000.)),
            transitions=(RenderTransition(),))
        with patch('crosscut.group_cut_timing.native_frame_clocks_for_output') as replay, \
                self.assertRaisesRegex(ValueError,'bounded full-event'):
            propose_numeric_group_cut_timing(plan=long,domain=domain,boundary_index=0,timing_offset_frames=4)
        replay.assert_not_called()

    def test_long_whole_renderer_units_decline_before_clock_allocation(self):
        plan,domain=self.fixture()
        long=replace(plan,segments=(RenderSegment(0,0.,15.,0.),RenderSegment(2,15.,15.,15.)),
            transitions=(RenderTransition(),))
        with patch('crosscut.group_cut_timing.native_frame_clocks_for_output') as replay, \
                self.assertRaisesRegex(ValueError,'two-whole-unit replay budget'):
            propose_numeric_group_cut_timing(plan=long,domain=domain,boundary_index=0,timing_offset_frames=4)
        replay.assert_not_called()


if __name__=='__main__':unittest.main()
