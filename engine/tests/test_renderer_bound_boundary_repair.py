from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.boundary_repair import BoundaryRepairRequest, repair_boundary
from crosscut.render import (RenderSegment,
    quantized_segment_frame_counts, _video_render_units, RenderPlan)
from crosscut.render_boundary_clock import hard_cut_boundary_sample_clocks
from crosscut.render_frame_schedule import VerifiedRenderFrameClock, sampler_frame_schedule
from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_visual_boundary_repair import _choice, _descriptor


NATIVE_FPS = 30000 / 1001


def clock_fixture():
    origin = 26.703685592197097
    output_starts = (0., 77.55080565587744, 78.74191570027648, 79.9663924447089, 81.19086918914129)
    endpoints = ((10., 87.55080565587744),
                 (77.55725941686345, 78.74843678861319),
                 (78.74843678861319, 79.97298274644496),
                 (149.497247055783, 150.72169969843506))
    choices = tuple(replace(_choice(i, source),
        canonical_master_start=origin+output_starts[i],
        canonical_master_end=origin+output_starts[i+1],
        source_local_start=endpoints[i][0], source_local_end=endpoints[i][1])
        for i, source in enumerate((2, 0, 0, 1)))
    return VisualTimelinePlan(choices, 1.)


class RendererBoundBoundaryRepairTests(unittest.TestCase):
    def test_search_snaps_to_output_grid_and_records_zero_offset_time_change(self):
        result = repair_boundary(BoundaryRepairRequest(
            nominal_boundary_seconds=1.01, outgoing_segment_start_seconds=0.,
            incoming_segment_end_seconds=2., output_fps=30.,
            output_timeline_origin_seconds=0.), visual_costs_by_frame_offset={0: .1})
        self.assertEqual(result.selected_frame_offset, 0)
        self.assertEqual(result.selected_boundary_seconds, 1.)
        self.assertTrue(result.changed)
        self.assertEqual(result.policy, 'fixed_source_output_tick_search_renderer_bound_flanks_v2')
        for origin in (True, float('nan'), -.1, .1):
            with self.subTest(origin=origin), self.assertRaises((TypeError, ValueError)):
                BoundaryRepairRequest(nominal_boundary_seconds=1., outgoing_segment_start_seconds=0.,
                    incoming_segment_end_seconds=2., output_fps=30., output_timeline_origin_seconds=origin)

    def test_frame_sampling_matches_actual_renderer_coalesced_schedule(self):
        timeline = clock_fixture()
        origin = timeline.choices[0].canonical_master_start
        segments = tuple(RenderSegment(c.source_index, c.source_local_start, c.duration_seconds,
            c.canonical_master_start-origin, source_duration=c.source_duration_seconds) for c in timeline.choices)
        plan = RenderPlan((Path('a'), Path('b'), Path('c')), segments, 0, '16:9', Path('unused.mp4'))
        counts = quantized_segment_frame_counts(segments, plan.fps)
        units = _video_render_units(plan, counts)
        expected = []
        for unit in units:
            if unit.last_segment_index != 2 and unit.first_segment_index != 3:
                continue
            rate = ((unit.segments[-1].source_end-unit.segments[0].source_start)
                    / sum(s.duration for s in unit.segments))
            schedule = sampler_frame_schedule(source_start=unit.segments[0].source_start,
                source_playback_rate=rate, native_fps=NATIVE_FPS, output_fps=plan.fps,
                frame_count=unit.frame_count, source_frame_count=10000).source_frame_indices
            expected.extend((schedule[-4], schedule[-1]) if unit.last_segment_index == 2 else (schedule[0], schedule[3]))
        clocks = {i: VerifiedRenderFrameClock(Fraction(30000,1001),10000,'a'*64) for i in range(3)}
        samples = hard_cut_boundary_sample_clocks(plan, clocks, boundary_index=2)
        self.assertEqual([sample.native_frame_index for sample in samples], expected)
        self.assertEqual(expected, [2393, 2396, 4480, 4483])

    def test_subframe_music_origin_no_longer_hides_clean_native_camera_join(self):
        timeline = clock_fixture()
        clocks = (VerifiedRenderFrameClock(Fraction(30000, 1001), 10000, 'a'*64),)*3
        def provider(source, seconds):
            frame = int(seconds*NATIVE_FPS+.5)
            # These two original camera cuts straddle the music timestamp.
            # The outgoing final wide frame and incoming earlier camera must
            # both be absent from the new edit, not swapped for one another.
            bad = (source == 0 and frame >= 2396) or (source == 1 and frame < 4480)
            return _descriptor(.9 if bad else .5)
        arguments = dict(visual_timeline=timeline, sources=(Path('a'),Path('b'),Path('c')),
            audio_offsets=(0.,0.,0.), scene_cut_times=((),(4480/NATIVE_FPS,),()),
            boundary_indices=(2,), descriptor_provider=provider)
        # Use exact source clocks from this real-valued fixture without making
        # an artificial global offset fit all sources. The scalar map offsets
        # are constant across this tiny correction (subframe drift negligible).
        nominal = timeline.choices[2].canonical_master_end
        arguments['audio_offsets'] = (timeline.choices[2].source_local_end-nominal,
            timeline.choices[3].source_local_start-nominal, 0.)
        previous = repair_visual_timeline_boundaries(**arguments)
        self.assertEqual(previous.reviewed_cross_source_count, 0)
        updated = repair_visual_timeline_boundaries(**arguments, render_frame_clocks=clocks)
        self.assertEqual(updated.reviewed_cross_source_count, 1, updated.failures)
        decision = updated.repairs[0]
        self.assertEqual(decision.result.selected_frame_offset, -1)
        self.assertEqual(decision.result.selected.interior_scene_cut_count, 0)
        self.assertTrue(all(item['sample_clock_basis']=='renderer_bound_output_ticks_v2'
            for item in decision.candidate_evidence_audit if item['visual_evidence']))
        choices = updated.timeline.choices
        origin = choices[0].canonical_master_start
        segments = tuple(RenderSegment(c.source_index,c.source_local_start,c.duration_seconds,
            c.canonical_master_start-origin,source_duration=c.source_duration_seconds) for c in choices)
        plan = RenderPlan((Path('a'),Path('b'),Path('c')),segments,0,'16:9',Path('unused.mp4'))
        samples = hard_cut_boundary_sample_clocks(plan,dict(enumerate(clocks)),boundary_index=2)
        self.assertEqual(samples[1].native_frame_index, 2395)
        self.assertEqual(samples[2].native_frame_index, 4480)

    def test_unverified_clocks_do_not_claim_renderer_parity(self):
        timeline = VisualTimelinePlan((_choice(0,0),_choice(1,1)), 1.)
        repaired = repair_visual_timeline_boundaries(visual_timeline=timeline,
            sources=(Path('a'),Path('b')),audio_offsets=(0.,0.),scene_cut_times=((),()),
            descriptor_provider=lambda *_: _descriptor(.5),render_frame_clocks=(None,None))
        self.assertTrue(repaired.repairs)
        self.assertTrue(all(item['sample_clock_basis']=='unverified_legacy_midpoint_v1'
            for item in repaired.repairs[0].candidate_evidence_audit if item['visual_evidence']))


if __name__ == '__main__':
    unittest.main()
