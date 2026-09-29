from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crosscut.boundary_subject_alignment import (
    BoundarySubjectObservation, _CaptureProvider, _alignment_native_frames,
    _frame_counts, _native_cut_in_boundary_window, build_boundary_subject_alignment_plan,
)
from crosscut.render import RenderPlan, RenderSegment, RenderTransition, TransitionKind, render
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice, _frame


def timeline():
    origin = 4.007
    choices = (
        _choice(0,0,start=origin,end=origin+.484),
        _choice(1,0,start=origin+.484,end=origin+.968),
        _choice(2,1,start=origin+.968,end=origin+1.568))
    return VisualTimelinePlan(tuple(replace(c, source_local_start=start,
        source_local_end=start+c.duration_seconds)
        for c,start in zip(choices, (.0143,.4983,2.35))), 1.)


def decode_id(frame):
    height,width=frame.shape[:2]
    return sum((frame[height//3:2*height//3,
        round((bit+.4)*width/8):round((bit+.6)*width/8)].mean()>125) << bit
        for bit in range(8))


class AlignmentClockTests(unittest.TestCase):
    def test_native_shot_entry_is_not_an_interior_camera_flash(self):
        original=timeline().choices[2]
        choice=replace(original, source_local_start=2.344,
            source_local_end=2.344+original.duration_seconds,
            internal_scene_cut_master_seconds=(original.canonical_master_start+.006,))
        arguments=dict(outgoing=False,frame_count=18,fps=30.,window_frames=14)
        self.assertTrue(_native_cut_in_boundary_window(choice,**arguments))
        exact=dict(arguments,native_frame_window=(141,171),native_frame_rate=60.)
        self.assertFalse(_native_cut_in_boundary_window(choice,**exact))
        interior=replace(choice,internal_scene_cut_master_seconds=(
            original.canonical_master_start+.006+1/60.,))
        self.assertTrue(_native_cut_in_boundary_window(interior,**exact))
        # An unlocalized failure/near-cut flag is not excused by a clock.
        self.assertTrue(_native_cut_in_boundary_window(replace(choice,entry_scene_cut_near=True),**exact))
        self.assertTrue(_native_cut_in_boundary_window(replace(choice,
            internal_scene_cut_master_seconds=(),face_match_candidate_failure='internal_camera_cut'),**exact))

    def test_segment_local_clock_is_not_the_coalesced_renderer_frame(self):
        visual=timeline()
        clocks={i:VerifiedRenderFrameClock(Fraction(60),200,'a'*64) for i in range(2)}
        counts=_frame_counts(visual.choices,30.)
        starts, samples=_alignment_native_frames(visual.choices,(Path('a'),Path('b')),
            (RenderTransition(),)*2, counts,30.,clocks)
        self.assertEqual(counts,(15,14,18))
        actual=samples[starts[1]+counts[1]-1]
        # Fractional segment duration changes phase after the logical split.
        legacy=visual.choices[1].source_local_start+(counts[1]-1)/30.
        self.assertNotEqual(int(legacy*60+.5), actual.native_frame_index)
        self.assertEqual(actual.native_frame_index,57)
        self.assertEqual(actual.unit_first_segment_index,0)

    def test_unknown_native_clock_cannot_authorize_or_even_decode_alignment(self):
        visual=timeline()
        provider=Mock(return_value=_frame(0))
        result=build_boundary_subject_alignment_plan(visual_timeline=visual,
            sources=(Path('a'),Path('b')),transitions=(RenderTransition(),)*2,fps=30.,
            frame_provider=provider,render_frame_clocks=(None,None))
        self.assertEqual(result.eligible_count,0)
        self.assertEqual(result.decisions[0].rejection_reason,'unverified_renderer_cut_frame')
        provider.assert_not_called()
        self.assertEqual(result.sample_clock_basis,'partially_verified_renderer_output_ticks_v2')

    def test_overlap_and_terminal_clone_are_not_pure_native_alignment_evidence(self):
        visual=timeline()
        clocks={i:VerifiedRenderFrameClock(Fraction(60),200,'a'*64) for i in range(2)}
        counts=_frame_counts(visual.choices,30.)
        transitions=(RenderTransition(),RenderTransition(TransitionKind.FACE_DISSOLVE,4))
        starts,samples=_alignment_native_frames(visual.choices,(Path('a'),Path('b')),
            transitions,counts,30.,clocks)
        self.assertTrue(all(starts[2]+index not in samples for index in range(4)))
        clocks[1]=VerifiedRenderFrameClock(Fraction(60),175,'b'*64)
        _,samples=_alignment_native_frames(visual.choices,(Path('a'),Path('b')),
            (RenderTransition(),)*2,counts,30.,clocks)
        self.assertNotIn(sum(counts)-1,samples)

    def test_capture_does_not_merge_exact_timestamp_queries_or_keep_unbounded_4k_pixels(self):
        capture=Mock()
        capture.isOpened.return_value=True
        capture.read.return_value=(True,np.zeros((100,100,3),np.uint8))
        with patch('crosscut.boundary_subject_alignment.cv2.VideoCapture',return_value=capture):
            provider=_CaptureProvider((Path('a'),),maximum_cache_bytes=40000)
            try:
                provider.frame_at(0,.00832)
                provider.frame_at(0,.00834)
                self.assertEqual([call.args[0] for call in capture.set.call_args_list],
                    [cv2.CAP_PROP_POS_MSEC]*2)
                self.assertAlmostEqual(capture.set.call_args_list[0].args[1],8.32)
                self.assertAlmostEqual(capture.set.call_args_list[1].args[1],8.34)
                self.assertEqual(provider._cache_bytes,30000)
            finally: provider.close()
        self.assertEqual(provider._cache_bytes,0)

    def test_bad_native_seek_position_fails_without_neighboring_timestamp_fallback(self):
        capture=Mock()
        capture.isOpened.return_value=True
        capture.get.side_effect=lambda key: {cv2.CAP_PROP_FPS:60.,cv2.CAP_PROP_FRAME_COUNT:200,
            cv2.CAP_PROP_POS_FRAMES:19.,cv2.CAP_PROP_POS_MSEC:18/60*1000.}[key]
        capture.read.return_value=(True,np.zeros((90,160,3),np.uint8))
        with patch('crosscut.boundary_subject_alignment.cv2.VideoCapture',return_value=capture):
            provider=_CaptureProvider((Path('a'),))
            try:
                self.assertIsNone(provider.frame_at_native_index(0,20,
                    VerifiedRenderFrameClock(Fraction(60),200,'a'*64)))
                capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES,20)
            finally: provider.close()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class AlignmentClockPixelTests(unittest.TestCase):
    def test_dense_alignment_decodes_exact_same_native_ids_as_actual_final_render(self):
        frames=np.zeros((200,90,160,3),np.uint8)
        for index in range(200):
            for bit in range(8): frames[index,:,bit*20:(bit+1)*20,:]=235 if index>>bit&1 else 16
        with tempfile.TemporaryDirectory(prefix='crosscut-alignment-clock-test-') as directory:
            root=Path(directory)
            source=root/'native-ids.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','160x90','-framerate','60','-i','pipe:0','-f','lavfi','-i',
                'anullsrc=r=48000:cl=mono','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',
                '-c:a','aac','-shortest',str(source)],input=frames.tobytes(),capture_output=True,check=True)
            visual=timeline()
            # The source camera starts on the first visible incoming native
            # frame even though its event is a few milliseconds after the
            # fractional nominal trim. It must not veto this entire ramp.
            incoming=visual.choices[2]
            visual=replace(visual,choices=(*visual.choices[:2],replace(incoming,
                source_local_start=2.344,source_local_end=2.344+incoming.duration_seconds,
                internal_scene_cut_master_seconds=(incoming.canonical_master_start+.006,))))
            observed=[]
            def observations(frame):
                native=decode_id(frame)
                observed.append(native)
                box=(.40,.26,.20,.24) if native<128 else (.3875,.23,.225,.27)
                return BoundarySubjectObservation(face_count=1,face_box_xywh=box,face_eye_scale=box[3]*.42)
            # No injected frame reader: actual source clock probing and native
            # decoding feed every dense subject observation in the producer.
            decoded_requests=[]
            read=_CaptureProvider.frame_at_native_index
            def checked_read(provider, source_index, index, clock):
                frame=read(provider,source_index,index,clock)
                self.assertIsNotNone(frame)
                self.assertEqual(decode_id(frame),index)
                decoded_requests.append((source_index,index))
                return frame
            with patch.object(_CaptureProvider,'frame_at_native_index',checked_read):
                aligned=build_boundary_subject_alignment_plan(visual_timeline=visual,
                    sources=(source,source),transitions=(RenderTransition(),)*2,fps=30.,
                    observation_provider=observations,source_resolutions=((1920,1080),)*2)
            self.assertEqual(aligned.eligible_count,1)
            decision=aligned.decisions[0]
            self.assertEqual(decision.outgoing_native_frame_index,57)
            self.assertEqual(decision.incoming_native_frame_index,141)
            self.assertEqual(aligned.sample_clock_basis,'renderer_bound_output_ticks_v2')
            self.assertGreater(decision.validated_output_frame_count,0)
            plan=RenderPlan((source,source),tuple(RenderSegment(c.source_index,c.source_local_start,
                c.duration_seconds,c.canonical_master_start-visual.choices[0].canonical_master_start,
                source_duration=c.source_duration_seconds,editorial_transform=aligned.track_for(c.segment_index))
                for c in visual.choices),0,'16:9',root/'actual.mp4',fps=30.,width=160,height=90)
            render(plan,video_codec='libx264')
            raw=subprocess.run(['ffmpeg','-v','error','-i',str(plan.output),'-an','-f','rawvideo',
                '-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
            actual=np.frombuffer(raw,np.uint8).reshape((-1,90,160,3))
            clocks={i:probe_render_frame_clock(source) for i in range(2)}
            expected=native_frame_clocks_for_output(plan,clocks,range(plan.output_frame_count))
            self.assertEqual(decode_id(actual[decision.outgoing_output_frame_index]),57)
            self.assertEqual(decode_id(actual[decision.incoming_output_frame_index]),141)
            self.assertTrue(set(observed)<={sample.native_frame_index for sample in expected.values()})
            self.assertTrue(set(decoded_requests)<={(s.source_index,s.native_frame_index)
                for s in expected.values()})
            for index,sample in expected.items():
                self.assertEqual(decode_id(actual[index]),sample.native_frame_index)


if __name__ == '__main__': unittest.main()
