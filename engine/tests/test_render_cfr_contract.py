"""Decode actual short renders: count, PTS, last-frame duration and CFR metadata."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from crosscut.ive import AspectRatio
from crosscut.render import (RenderPlan, RenderSegment, RenderTransition, TransitionKind,
    SourceLookAdjustment, render)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class RenderCFRContractTests(unittest.TestCase):
    def test_short_render_clock_matrix(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=128x72:r=60:d=2',
                '-f','lavfi','-i','anullsrc=r=48000:cl=mono','-c:v','libx264','-crf','10',
                '-c:a','aac','-shortest',str(source)], check=True, capture_output=True)
            vfr = root/'vfr.mp4'
            subprocess.run(['ffmpeg','-v','error','-i',str(source),'-vf',
                r'setpts=if(lt(N\,30)\,N/(60*TB)\,(0.5+(N-30)/30)/TB)',
                '-fps_mode:v','vfr','-c:v','libx264','-c:a','copy',str(vfr)],check=True,capture_output=True)
            for fps in (30., 30000/1001, 60000/1001):
                for kind in ('hard', 'dissolve', 'consecutive', 'shared', 'fallback', 'vfr'):
                    with self.subTest(fps=fps, kind=kind):
                        duration = 9/fps
                        segments = (RenderSegment(0,.0143,duration,0.),
                            RenderSegment(0 if kind=='shared' else 1,.0143+duration,duration,duration,
                                look_adjustment=SourceLookAdjustment(brightness=.01) if kind=='shared' else None))
                        transitions = (RenderTransition(TransitionKind.FACE_DISSOLVE,3),) if kind=='dissolve' else ()
                        if kind == 'consecutive':
                            segments = tuple(RenderSegment(i%2,.0143+i*6/fps,6/fps,i*6/fps) for i in range(3))
                            transitions = tuple(RenderTransition(TransitionKind.FACE_DISSOLVE,3) for _ in range(2))
                        chosen_source = vfr if kind=='vfr' else source
                        plan = RenderPlan((chosen_source,chosen_source),segments,0,AspectRatio.LANDSCAPE,
                            root/f'{fps}-{kind}.mp4',width=128,height=72,fps=fps,transitions=transitions)
                        render(plan,video_codec='libx264',use_verified_frame_schedule=kind!='fallback')
                        if kind == 'vfr':
                            clock=json.loads(plan.output.with_suffix('.mp4.frame-clock.json').read_text())
                            self.assertEqual(clock['timestamp_fallback_source_indices'],[0,1])
                        result = json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                            '-show_streams','-show_frames','-of','json',str(plan.output)],check=True,capture_output=True).stdout)
                        stream, frames = result['streams'][0],result['frames']
                        self.assertEqual(len(frames),18)
                        self.assertAlmostEqual(float(Fraction(stream['avg_frame_rate'])),fps,places=4)
                        self.assertAlmostEqual(float(stream['duration']),18/fps,delta=.00005, msg={key:stream.get(key) for key in ('avg_frame_rate','r_frame_rate','duration','start_time','time_base','nb_frames')})
                        times = [float(f['best_effort_timestamp_time']) for f in frames]
                        for i,t in enumerate(times): self.assertAlmostEqual(t,i/fps,delta=.00005)
                        self.assertAlmostEqual(float(frames[-1].get('duration_time',frames[-1].get('pkt_duration_time'))),1/fps,delta=.00005)


if __name__ == '__main__': unittest.main()
