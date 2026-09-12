from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.render import (
    EditorialTransformKeyframe, EditorialTransformTrack,
    RenderPlan, RenderSegment, render,
)
from crosscut.render_frame_schedule import (
    NativeFrameSchedule, probe_render_frame_clock,
    render_frame_clock_from_probe, sampler_frame_schedule,
)


class RenderFrameScheduleTests(unittest.TestCase):
    def test_b_frame_packet_order_is_not_presentation_order(self):
        probe = {'streams':[{'codec_name':'h264','avg_frame_rate':'30/1',
            'r_frame_rate':'30/1','time_base':'1/30000','start_time':'0','nb_frames':'4'}],
            'packets':[{'pts':i*1000,'duration':1000} for i in (0,3,1,2)]}
        clock = render_frame_clock_from_probe(probe)
        self.assertEqual(clock.frame_rate, Fraction(30))
        self.assertEqual(clock.frame_count, 4)
        for field, value in (('start_time','0.1'),('avg_frame_rate','29/1'),('codec_name','hevc')):
            changed = deepcopy(probe)
            changed['streams'][0][field] = value
            self.assertIsNone(render_frame_clock_from_probe(changed))
        for pts in ([0,1000,1000,3000], [0,1000,2000,4000]):
            changed = deepcopy(probe)
            changed['packets'] = [{'pts':p,'duration':1000} for p in pts]
            self.assertIsNone(render_frame_clock_from_probe(changed))

    def test_vp9_quantized_pts_must_be_bijective_with_native_index_cells(self):
        probe = {'streams':[{'codec_name':'vp9','avg_frame_rate':'30000/1001',
            'r_frame_rate':'30000/1001','time_base':'1/1000','start_time':'0','nb_frames':'5'}],
            'packets':[{'pts':i,'duration':33} for i in (0,33,67,100,133)]}
        clock = render_frame_clock_from_probe(probe)
        self.assertIsNotNone(clock)
        self.assertEqual(clock.grid_basis,'bijective_nearest_index_presentation_grid')
        for bad in (33,84,120):
            changed = deepcopy(probe)
            changed['packets'][2]['pts'] = bad
            self.assertIsNone(render_frame_clock_from_probe(changed))

    def test_schedule_retains_duplicates_skips_and_compacts_linear_runs(self):
        schedule = NativeFrameSchedule((4,4,5,7,7,8))
        self.assertEqual([schedule.first_output_tick(i) for i in range(4,10)], [0,2,3,3,5,6])
        long = NativeFrameSchedule(tuple(range(100,6100)))
        self.assertLess(len(long.ffmpeg_tick_expression()), 150)
        self.assertIn('+N-6099',long.ffmpeg_tick_expression())
        with self.assertRaises(ValueError): NativeFrameSchedule((2,1))
        with self.assertRaises(ValueError): NativeFrameSchedule((False,1))

    def test_terminal_padding_is_explicit_and_never_an_out_of_domain_observation(self):
        args = dict(source_start=0.,source_playback_rate=1.01,native_fps=30.,
                    output_fps=30.,frame_count=60,source_frame_count=60)
        with self.assertRaisesRegex(ValueError,'exceeds source'):
            sampler_frame_schedule(**args)
        padded = sampler_frame_schedule(**args,maximum_terminal_hold_frames=30)
        self.assertEqual(padded.terminal_hold_output_indices,(59,))
        self.assertEqual(padded.source_frame_indices[-1],59)
        with self.assertRaisesRegex(ValueError,'exceeds source'):
            sampler_frame_schedule(**{**args,'source_start':3.},maximum_terminal_hold_frames=30)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class RenderFrameSchedulePixelTests(unittest.TestCase):
    def test_public_renderer_selects_native_ids_with_fractional_starts_and_camera(self):
        frames = np.zeros((100,90,160,3),np.uint8)
        for index in range(100):
            for bit in range(8):
                frames[index,:,bit*20:(bit+1)*20,:] = 235 if (index>>bit)&1 else 16
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for native,codec in ((24.,'libx264'),(30.,'libx264'),(60.,'libx264'),(30000/1001,'libvpx-vp9')):
                source = root/f'native-{native}-{codec}.mp4'
                codec_options = (['-preset','medium','-crf','0'] if codec=='libx264'
                    else ['-lossless','1','-deadline','realtime','-cpu-used','4','-video_track_timescale','1000'])
                subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24',
                    '-video_size','160x90','-framerate',str(native),'-i','pipe:0',
                    '-f','lavfi','-i','anullsrc=r=48000:cl=mono','-c:v',codec,
                    *codec_options,'-pix_fmt','yuv420p','-c:a','aac',
                    '-shortest',str(source)],input=frames.tobytes(),check=True,capture_output=True)
                clock = probe_render_frame_clock(source)
                self.assertIsNotNone(clock)
                for case,(fps,start_frame,speed,camera) in enumerate((
                    (30.,10.09,1.,False),(60.,10.5,.99,False),
                    (30.,11.51,1.01,True),(60.,11.5,1.,True),
                )):
                    with self.subTest(native=native,codec=codec,fps=fps,start_frame=start_frame,camera=camera):
                        start, duration = start_frame/native,30/fps
                        track = EditorialTransformTrack((
                            EditorialTransformKeyframe(0.,1.05,.5,.5,-.15),
                            EditorialTransformKeyframe(duration,1.04,.502,.499,.15),
                        ),reason='frame_schedule_temporal_parity') if camera else None
                        output = root/f'out-{native}-{case}.mp4'
                        segment = RenderSegment(0,start,duration,0.,
                            source_duration=duration*speed,editorial_transform=track)
                        plan = RenderPlan((source,),(segment,),0,AspectRatio.LANDSCAPE,
                            output,master_audio_start=start,width=160,height=90,fps=fps)
                        render(plan,video_codec='libx264')
                        raw_pixels = subprocess.run(['ffmpeg','-v','error','-i',str(output),
                            '-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],check=True,capture_output=True).stdout
                        decoded = np.frombuffer(raw_pixels,np.uint8).reshape((-1,90,160,3))
                        observed = tuple(sum((frame[40:50,bit*20+8:bit*20+12,:].mean()>125)<<bit for bit in range(8)) for frame in decoded)
                        expected = sampler_frame_schedule(source_start=start,
                            source_playback_rate=segment.source_playback_rate,native_fps=float(clock.frame_rate),
                            output_fps=fps,frame_count=30,source_frame_count=clock.frame_count)
                        self.assertEqual(observed,expected.source_frame_indices)
                        evidence = json.loads(output.with_suffix('.mp4.frame-clock.json').read_text())
                        self.assertEqual(evidence['timestamp_fallback_source_indices'],[])
                        self.assertFalse(evidence['quality_approval'])


if __name__ == '__main__': unittest.main()
