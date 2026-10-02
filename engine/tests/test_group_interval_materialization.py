from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch

from crosscut.group_actual_shots import visible_group_cut_inventory
from crosscut.group_interval_materialization import (
    split_group_render_intervals, refresh_split_group_render_observations)
from crosscut.group_joint_source_camera import GroupOutputRole, joint_state_from_render_plan
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import (BoundaryLookKeyframe, BoundaryLookTrack,
    EditorialTransformKeyframe, EditorialTransformTrack, NormalizedCrop,
    PanCropKeyframe, PanCropTrack, RenderPlan, RenderSegment, RenderTransition,
    TransitionKind, render, segment_frame_counts, _base_crop_values, _boundary_look_values)
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock
from crosscut.visual_score import VisualTimelinePlan
from tests import test_group_joint_state_factory as factory_fixture
from tests.test_group_joint_render_observations import JointSampler
from tests.test_sparse_source_lattice import Sampler
from tests.test_visual_boundary_repair import _choice


class GroupIntervalMaterializationTests(unittest.TestCase):
    def fixture(self, fps=29.97, native_fps=Fraction(60000, 1001)):
        plan, domain, _ = factory_fixture.JointStateFactoryTests().fixture(fps, native_fps)
        timeline = VisualTimelinePlan(tuple(replace(_choice(i, i), quality_score=.99,
            face_match_candidate_failure='old_anchor_failure') for i in range(4)), 1.,
            candidate_review_summaries=({'passed': True},))
        return plan, domain, timeline

    def test_new_inside_cell_boundary_preserves_all_global_native_frames(self):
        for fps, native in ((30., Fraction(30)), (30., Fraction(60)),
                (29.97, Fraction(30000, 1001)), (29.97, Fraction(60000, 1001))):
            with self.subTest(fps=fps, native=native):
                plan, domain, _ = self.fixture(fps, native)
                result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
                self.assertEqual(segment_frame_counts(result.plan), (30, 15, 15, 30, 30))
                self.assertEqual(result.split_old_segment_indices, (1,))
                self.assertEqual(result.refreshed_segment_indices, (0, 1, 2, 3))
                before = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(plan.output_frame_count))
                after = native_frame_clocks_for_output(result.plan, domain.clocks_by_request, range(plan.output_frame_count))
                self.assertEqual([(x.source_index, x.native_frame_index) for x in before.values()],
                    [(x.source_index, x.native_frame_index) for x in after.values()])
                self.assertEqual(result.plan.master_audio_start, plan.master_audio_start)
                self.assertEqual(result.plan.duration, plan.duration)
                self.assertEqual(result.plan.reference_source, plan.reference_source)
                self.assertIs(result.plan.segments[-1], plan.segments[-1])
                self.assertFalse(result.to_dict()['full_bundle_created'])

    def test_actual_native_cut_inside_cell_becomes_a_boundable_epoch(self):
        plan, domain, _ = self.fixture(30., Fraction(30))
        domain = replace(domain, scene_cut_times=((), (1.5,), (), ()))
        with self.assertRaisesRegex(ValueError, 'owned splitting'):
            joint_state_from_render_plan(plan=plan, domain=domain, slot=1,
                first_output_frame=30, end_output_frame=45, role=GroupOutputRole.FOCAL_HEAD_ARMS)
        before = visible_group_cut_inventory(plan, domain)
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        state = joint_state_from_render_plan(plan=result.plan, domain=domain, slot=1,
            first_output_frame=30, end_output_frame=45, role=GroupOutputRole.FOCAL_HEAD_ARMS)
        self.assertEqual(len(state.native_frame_indices), 15)
        self.assertEqual(visible_group_cut_inventory(result.plan, domain), before)
        self.assertIn(45, [e.first_incoming_output_frame for e in before if e.kind == 'native_camera_cut'])

    def test_fractional_cell_keeps_its_original_shared_unit_origin(self):
        sources = (Path('a.mp4'), Path('b.mp4'))
        clock = VerifiedRenderFrameClock(Fraction(60), 8000, 'a'*64)
        domain = GroupSourceDomain(sources, (0, 1), (0., 0.), (None, None), ((), ()), (clock, clock))
        plan = RenderPlan(sources=sources, segments=(
            RenderSegment(0, .0143, .084, 0.), RenderSegment(0, .0983, .334, .084),
            RenderSegment(1, .021, 1., .418)), reference_source=0, aspect_ratio='16:9',
            output=Path('unused.mp4'), fps=30.)
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(7,))
        before = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(12))
        after = native_frame_clocks_for_output(result.plan, domain.clocks_by_request, range(12))
        self.assertEqual([r.native_frame_index for r in before.values()], [r.native_frame_index for r in after.values()])
        self.assertEqual(after[7].unit_first_segment_index, 0)
        self.assertNotEqual(after[7].native_frame_index,
            round(result.plan.segments[2].source_start*60))

    def test_camera_pan_look_and_sampling_canvas_do_not_restart_at_split(self):
        plan, domain, _ = self.fixture(30., Fraction(60))
        camera = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.18, .48, .49),
            EditorialTransformKeyframe(.4, 1.3, .52, .5), EditorialTransformKeyframe(1., 1.1, .5, .5)),
            sampling_canvas_zoom=1.4)
        pan = PanCropTrack((PanCropKeyframe(0., NormalizedCrop(.02, .02, .96, .96)),
            PanCropKeyframe(1., NormalizedCrop(.13, .12, .8, .8))))
        look = BoundaryLookTrack((BoundaryLookKeyframe(0., .02, 1.02, 1.03),
            BoundaryLookKeyframe(.4, -.01, .99, .99), BoundaryLookKeyframe(1.)))
        segment = replace(plan.segments[1], editorial_transform=camera, pan_crop=pan, boundary_look_track=look)
        plan = replace(plan, segments=(plan.segments[0], segment, *plan.segments[2:]))
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        for tick in range(15):
            old_t, new_t = (15+tick)/30., tick/30.
            old, new = segment.editorial_transform.evaluate(old_t), result.plan.segments[2].editorial_transform.evaluate(new_t)
            for name in ('zoom', 'center_x', 'center_y', 'roll_degrees'):
                self.assertAlmostEqual(getattr(old, name), getattr(new, name))
            self.assertEqual(result.plan.segments[2].editorial_transform.effective_sampling_canvas_zoom, 1.4)
            for left, right in zip(_base_crop_values(segment, old_t), _base_crop_values(result.plan.segments[2], new_t)):
                self.assertAlmostEqual(left, right)
            for left, right in zip(_boundary_look_values(look, old_t),
                    _boundary_look_values(result.plan.segments[2].boundary_look_track, new_t)):
                self.assertAlmostEqual(left, right)
        self.assertAlmostEqual(result.plan.segments[1].editorial_transform.final_keyframe.zoom,
            result.plan.segments[2].editorial_transform.keyframes[0].zoom)

    def test_repeat_or_existing_boundaries_do_not_manufacture_new_cuts(self):
        plan, domain, timeline = self.fixture()
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(30, 60))
        self.assertIs(result.plan, plan)
        with patch('crosscut.visual_score._VideoSampler') as sampler:
            observed = refresh_split_group_render_observations(scaffold=result, timeline=timeline,
                domain=domain, sync_scores=(1.,)*4)
        self.assertIs(observed.timeline, timeline)
        sampler.assert_not_called()

    def test_split_refreshes_real_new_descriptor_times_and_reindexes_untouched_rows(self):
        plan, domain, timeline = self.fixture(30., Fraction(30))
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        Sampler.instances = []; Sampler.unreadable_source = None
        with patch('crosscut.visual_score._VideoSampler', JointSampler):
            observed = refresh_split_group_render_observations(scaffold=result, timeline=timeline,
                domain=domain, sync_scores=(1.,)*4)
        self.assertEqual(len(observed.timeline.choices), 5)
        self.assertEqual(tuple(c.segment_index for c in observed.timeline.choices), tuple(range(5)))
        self.assertEqual(tuple(c.source_index for c in observed.timeline.choices), (0, 1, 1, 2, 3))
        for choice, segment in zip(observed.timeline.choices, result.plan.segments):
            self.assertAlmostEqual(choice.canonical_master_start, segment.master_start)
            self.assertAlmostEqual(choice.canonical_master_end, segment.master_start+segment.duration)
            self.assertAlmostEqual(choice.source_local_start, segment.source_start)
            self.assertAlmostEqual(choice.source_local_end, segment.source_end)
        for choice in observed.timeline.choices[:4]:
            self.assertAlmostEqual(choice.quality_score, .8)
            self.assertNotEqual(choice.face_match_candidate_failure, 'old_anchor_failure')
        self.assertEqual(observed.timeline.choices[4], replace(timeline.choices[3], segment_index=4))
        self.assertEqual(observed.timeline.candidate_review_summaries, ())
        self.assertIs(observed.incumbent_timeline, timeline)
        self.assertTrue(all(s.closed for s in Sampler.instances))
        self.assertFalse(observed.to_dict()['candidate_admitted'])

    def test_old_boundary_receipt_is_not_relabelled_as_fresh_after_index_shift(self):
        plan, domain, timeline = self.fixture(30., Fraction(30))
        original = replace(timeline.choices[-1], transition_evidence_authority_from_previous=object(),
            transition_evidence_contract_from_previous='old_index_bound_authority')
        timeline = replace(timeline, choices=(*timeline.choices[:-1], original))
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        with patch('crosscut.visual_score._VideoSampler', JointSampler):
            observed = refresh_split_group_render_observations(scaffold=result, timeline=timeline,
                domain=domain, sync_scores=(1.,)*4)
        self.assertEqual(observed.unrebound_boundary_authority_old_indices, (3,))
        self.assertIs(observed.incumbent_timeline.choices[-1], original)
        self.assertIsNone(observed.timeline.choices[-1].transition_evidence_authority_from_previous)

    def test_disjoint_split_windows_do_not_reanalyse_or_overwrite_the_unchanged_gap(self):
        from crosscut.group_source_span_proposal import refresh_fixed_group_render_observations
        plan, domain, timeline = self.fixture(30., Fraction(30))
        plan = replace(plan,segments=tuple(replace(plan.segments[i%4],
            source_start=float(i),master_start=float(i)) for i in range(10)),
            transitions=tuple(RenderTransition() for _ in range(9)))
        timeline = replace(timeline,choices=tuple(replace(_choice(i,i%4),
            quality_score=.99,face_match_candidate_failure='unchanged-gap-history') for i in range(10)))
        scaffold = split_group_render_intervals(plan=plan,domain=domain,split_output_frames=(45,225))
        self.assertEqual(scaffold.refreshed_segment_indices,(0,1,2,3,7,8,9,10))
        Sampler.instances=[];Sampler.unreadable_source=None
        with patch('crosscut.visual_score._VideoSampler',JointSampler), \
                patch('crosscut.group_interval_materialization.refresh_fixed_group_render_observations',
                    wraps=refresh_fixed_group_render_observations) as refresh:
            observed = refresh_split_group_render_observations(scaffold=scaffold,timeline=timeline,
                domain=domain,sync_scores=(1.,)*4)
        ranges=[(call.kwargs['first_segment_index'],call.kwargs['last_segment_index'])
            for call in refresh.call_args_list]
        self.assertEqual(ranges,[(0,3),(7,10)])
        for item in scaffold.cell_map:
            if item.new_segment_index not in scaffold.refreshed_segment_indices:
                self.assertEqual(observed.timeline.choices[item.new_segment_index],
                    replace(timeline.choices[item.old_segment_index],segment_index=item.new_segment_index))
        self.assertAlmostEqual(observed.timeline.choices[1].quality_score,.8)
        self.assertAlmostEqual(observed.timeline.choices[8].quality_score,.8)
        self.assertTrue(all(s.closed for s in Sampler.instances))

    def test_split_refresh_rejects_a_consumer_that_overwrites_an_unrequested_cell(self):
        from crosscut.group_source_span_proposal import refresh_fixed_group_render_observations
        plan, domain, timeline = self.fixture(30., Fraction(30))
        scaffold = split_group_render_intervals(plan=plan,domain=domain,split_output_frames=(45,))
        def corrupt(**kwargs):
            result = refresh_fixed_group_render_observations(**kwargs)
            last = replace(result.choices[-1],quality_score=.123)
            return replace(result,choices=(*result.choices[:-1],last))
        Sampler.instances=[];Sampler.unreadable_source=None
        with patch('crosscut.visual_score._VideoSampler',JointSampler), \
                patch('crosscut.group_interval_materialization.refresh_fixed_group_render_observations',side_effect=corrupt), \
                self.assertRaisesRegex(ValueError,'outside its declared local window'):
            refresh_split_group_render_observations(scaffold=scaffold,timeline=timeline,
                domain=domain,sync_scores=(1.,)*4)
        self.assertEqual(timeline.choices[-1].quality_score,.99)
        self.assertTrue(all(s.closed for s in Sampler.instances))

    def test_unreadable_split_anchor_never_reuses_old_descriptors_or_expands_donors(self):
        plan, domain, timeline = self.fixture(30., Fraction(30))
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        Sampler.instances = []; Sampler.unreadable_source = 2
        try:
            with patch('crosscut.visual_score._VideoSampler', JointSampler), self.assertRaisesRegex(ValueError, 'readable'):
                refresh_split_group_render_observations(scaffold=result, timeline=timeline,
                    domain=domain, sync_scores=(1.,)*4)
            self.assertTrue(all(s.closed for s in Sampler.instances))
            self.assertEqual(timeline.choices[1].quality_score, .99)
        finally:
            Sampler.unreadable_source = None

    def test_bad_ticks_and_topology_budget_fail_before_media(self):
        plan, domain, _ = self.fixture()
        for ticks in ((True,), (0,), (120,), (45, 45), (60, 45), (31,), [45]):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                split_group_render_intervals(plan=plan, domain=domain, split_output_frames=ticks)
        with self.assertRaisesRegex(ValueError, 'additional-cell budget'):
            split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(40, 50), maximum_additional_cells=1)
        with self.assertRaisesRegex(ValueError, 'replay budget'):
            split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,), maximum_replay_frames=60)

    def test_stale_or_forged_scaffold_and_timeline_reject_before_sampler(self):
        plan, domain, timeline = self.fixture()
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        bad = replace(result, split_old_segment_indices=())
        stale = replace(timeline, choices=(replace(timeline.choices[0], source_index=3), *timeline.choices[1:]))
        with patch('crosscut.visual_score._VideoSampler') as sampler:
            for scaffold, candidate_timeline in ((bad, timeline), (result, stale)):
                with self.assertRaises(ValueError):
                    refresh_split_group_render_observations(scaffold=scaffold, timeline=candidate_timeline,
                        domain=domain, sync_scores=(1.,)*4)
        sampler.assert_not_called()

    def test_existing_timewarp_rate_is_preserved_at_every_sample(self):
        plan, domain, _ = self.fixture(29.97, Fraction(60000, 1001))
        plan = replace(plan, segments=(plan.segments[0],
            replace(plan.segments[1], source_duration=1.0123), *plan.segments[2:]))
        result = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        for segment in result.plan.segments[1:3]:
            self.assertAlmostEqual(segment.source_playback_rate, 1.0123)

    def test_terminal_clone_and_overlap_are_not_split_as_real_observed_epochs(self):
        plan, domain, _ = self.fixture(30., Fraction(30))
        domain = replace(domain, render_frame_clocks=(VerifiedRenderFrameClock(Fraction(30), 100, 'a'*64),)*4)
        with self.assertRaisesRegex(ValueError, 'terminal clones'):
            split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        plan, domain, _ = self.fixture(30., Fraction(30))
        plan = replace(plan, transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE, 4), *plan.transitions[1:]))
        with self.assertRaisesRegex(ValueError, 'hard-cut'):
            split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))

    def test_static_isolated_entry_round_trips_noop_without_promoting_a_bundle(self):
        from crosscut.optional_native_worker import optional_native_scope
        plan, domain, timeline = self.fixture(30., Fraction(30))
        scaffold = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=())
        with optional_native_scope(monotonic()+10.):
            result = refresh_split_group_render_observations(scaffold=scaffold,
                timeline=timeline, domain=domain, sync_scores=(1.,)*4)
        self.assertEqual(result.scaffold, scaffold)
        self.assertEqual(result.timeline, timeline)
        self.assertFalse(result.to_dict()['candidate_admitted'])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class GroupIntervalRendererPixelTests(unittest.TestCase):
    def test_split_and_original_emit_the_same_native_frame_ids_and_camera_pixels(self):
        import numpy as np
        import cv2
        with tempfile.TemporaryDirectory(prefix='crosscut-epoch-pixel-test-') as directory:
            root = Path(directory)
            source = root/'source.mp4'
            frames = np.zeros((90, 90, 160, 3), np.uint8)
            for n, frame in enumerate(frames):
                for bit in range(8):
                    frame[:, bit*20:(bit+1)*20] = 235 if n >> bit & 1 else 16
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '160x90', '-framerate', '60', '-i', 'pipe:0', '-f', 'lavfi', '-i',
                'anullsrc=r=48000:cl=mono', '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', str(source)], input=frames.tobytes(), capture_output=True, check=True)
            clock = probe_render_frame_clock(source)
            domain = GroupSourceDomain((source,), (0,), (0.,), (None,), ((),), (clock,))
            for transformed in (False, True):
                with self.subTest(transformed=transformed):
                    track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.12),
                        EditorialTransformKeyframe(.22, 1.3, .51, .5),
                        EditorialTransformKeyframe(.8, 1.16, .49, .5))) if transformed else None
                    old = RenderPlan(sources=(source,), segments=(RenderSegment(0, .0143, .8, 0.,
                        editorial_transform=track),), reference_source=0, aspect_ratio='16:9',
                        width=160, height=90, fps=30., output=root/f'original-{transformed}.mp4')
                    split = split_group_render_intervals(plan=old, domain=domain, split_output_frames=(10,))
                    new = replace(split.plan, output=root/f'split-{transformed}.mp4')
                    for plan in (old, new):
                        render(plan, video_codec='libx264', maximum_runtime_seconds=20.)
                    decoded = []
                    for path in (old.output, new.output):
                        cap = cv2.VideoCapture(str(path)); values = []
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            values.append(frame)
                        cap.release(); decoded.append(np.asarray(values))
                    self.assertEqual(decoded[0].shape, (24, 90, 160, 3))
                    self.assertEqual(decoded[1].shape, decoded[0].shape)
                    # The camera math/sampling canvas is unchanged. H.264
                    # output is decoded and compared, not certified by a PID.
                    self.assertLess(float(np.abs(decoded[0].astype(float)-decoded[1]).mean()), .05)
                    self.assertLess(float(np.abs(decoded[0].astype(float)-decoded[1]).max()), 3.)


if __name__ == '__main__':
    unittest.main()
