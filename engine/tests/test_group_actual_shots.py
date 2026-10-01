from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.group_actual_shots import (visible_group_cut_inventory,visible_group_shots,
    assert_native_cut_events_preserved,bounded_actual_shot_windows)
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


class ActualGroupShotTests(unittest.TestCase):
    def fixture(self,durations=(1.,1.),cuts=(1.,),native_rate=30):
        sources=(Path('a.mp4'),Path('b.mp4'),Path('c.mp4'));segments=[];start=0.
        for duration in durations:
            segments.append(RenderSegment(0,start,duration,start));start+=duration
        plan=RenderPlan(sources=sources,segments=tuple(segments),reference_source=0,
            aspect_ratio='16:9',output=Path('unused.mp4'),fps=30.,
            transitions=(RenderTransition(tier='C'),)*(len(segments)-1))
        domain=GroupSourceDomain(sources,(0,1,2),(0.,)*3,(None,)*3,(cuts,(),()),
            (VerifiedRenderFrameClock(Fraction(native_rate),1000,'a'*64),)*3)
        return plan,domain

    def test_native_cut_on_cell_seam_is_one_actual_cut_not_two_empty_intervals(self):
        plan,domain=self.fixture()
        events=visible_group_cut_inventory(plan,domain)
        self.assertEqual([(e.kind,e.first_incoming_output_frame,e.native_cut_frame) for e in events],
            [('native_camera_cut',30,30)])
        self.assertEqual([s.frame_count for s in visible_group_shots(plan,domain)],[30,30])

    def test_cell_split_or_camera_geometry_does_not_manufacture_cuts(self):
        for durations in ((2.,),(.4,.4,1.2)):
            plan,domain=self.fixture(durations)
            self.assertEqual([(e.kind,e.first_incoming_output_frame) for e in visible_group_cut_inventory(plan,domain)],
                [('native_camera_cut',30)])
            self.assertEqual([s.frame_count for s in visible_group_shots(plan,domain)],[30,30])
        plan,domain=self.fixture(cuts=())
        plan=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],
            editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),)))))
        self.assertEqual(visible_group_cut_inventory(plan,domain),())
        self.assertEqual(len(visible_group_shots(plan,domain)),1)

    def test_actual_native_frame_grid_not_fractional_cut_timestamp_is_authority(self):
        plan,domain=self.fixture(cuts=(1.01,),native_rate=60)
        event=visible_group_cut_inventory(plan,domain)[0]
        self.assertEqual((event.native_cut_frame,event.first_incoming_output_frame),(61,31))
        self.assertEqual((event.outgoing_native_frame,event.incoming_native_frame),(60,62))

    def test_seek_gap_is_not_fake_observation_of_native_cuts_in_unrendered_content(self):
        plan,domain=self.fixture(cuts=(1.5,))
        plan=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],source_start=2.)))
        self.assertEqual([(e.kind,e.first_incoming_output_frame) for e in visible_group_cut_inventory(plan,domain)],
            [('same_source_clock_jump',30)])

    def test_source_switches_are_distinct_from_native_cuts_and_split_true_shots(self):
        plan,domain=self.fixture(cuts=())
        plan=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],source_index=2)))
        events=visible_group_cut_inventory(plan,domain)
        self.assertEqual([(e.kind,e.incoming_source_index) for e in events],[('source_switch',2)])
        self.assertEqual([s.source_index for s in visible_group_shots(plan,domain)],[0,2])

    def test_same_native_cut_count_does_not_authorize_different_output_tick(self):
        plan,domain=self.fixture((2.,),cuts=(1.,))
        changed=replace(plan,segments=(replace(plan.segments[0],source_start=.1),))
        self.assertEqual(len(visible_group_cut_inventory(plan,domain)),len(visible_group_cut_inventory(changed,domain)))
        with self.assertRaisesRegex(ValueError,'identity or output tick'):
            assert_native_cut_events_preserved(plan,changed,domain)

    def test_search_window_boundaries_follow_native_cuts_inside_a_cell(self):
        plan,domain=self.fixture((2.,),cuts=(.4,.8,1.2,1.6))
        windows=bounded_actual_shot_windows(plan,domain)
        self.assertEqual(len(windows),2)
        self.assertEqual([s.frame_count for s in windows[1]],[12,12,12,12])
        self.assertEqual([(s.first_output_frame,s.end_output_frame) for s in windows[1]],
            [(0,12),(12,24),(24,36),(36,48)])
        self.assertTrue(all(s.first_segment_index==s.last_segment_index==0 for s in windows[1]))
        for limits in (dict(maximum_mutable_shots=3),dict(maximum_windows=True),dict(maximum_mutable_frames=0)):
            with self.assertRaises(ValueError):bounded_actual_shot_windows(plan,domain,**limits)

    def test_cuts_before_first_visible_frame_and_after_last_are_not_invented_events(self):
        plan,domain=self.fixture((2.,),cuts=(.5,1.,2.,3.))
        plan=replace(plan,segments=(replace(plan.segments[0],source_start=1.),))
        self.assertEqual([(e.first_incoming_output_frame,e.native_cut_frame)
            for e in visible_group_cut_inventory(plan,domain)],[(30,60)])

    def test_missing_eligible_clock_and_terminal_hold_are_not_actual_shots(self):
        plan,domain=self.fixture()
        domain=replace(domain,eligible_indices=(0,),audio_offsets=(0.,),source_timewarps=(None,),
            scene_cut_times=((1.,),),render_frame_clocks=domain.render_frame_clocks[:1])
        with self.assertRaises(ValueError):visible_group_cut_inventory(replace(plan,segments=(
            plan.segments[0],replace(plan.segments[1],source_index=2))),domain)
        short=replace(domain,render_frame_clocks=(VerifiedRenderFrameClock(Fraction(30),59,'b'*64),))
        with self.assertRaisesRegex(ValueError,'terminal-cloned'):
            visible_group_cut_inventory(plan,short)


if __name__=='__main__':unittest.main()
