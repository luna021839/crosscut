from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
from crosscut.seek_locality import SeekLocalityReader, probe_constant_frame_clock


class ActualH264LocalityTests(unittest.TestCase):
    def test_bframes_open_gop_and_vfr_against_timestamp_and_sequential_decode(self):
        ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        if not ffmpeg or not ffprobe:
            self.skipTest('ffmpeg and ffprobe required')
        with tempfile.TemporaryDirectory(prefix='crosscut-h264-locality-') as directory:
            for label, rate, options, eligible in (
                ('bframes', '30000/1001', ['-bf','3'], True),
                ('open-gop', '30', ['-bf','3','-x264-params','open-gop=1'], True),
                ('vfr', '30', ['-vf', r'select=not(eq(mod(n\,5)\,0))','-fps_mode','vfr'], False),
            ):
                path = Path(directory)/(label+'.mp4')
                subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',
                                f'testsrc2=size=128x96:rate={rate}:duration=4',
                                '-c:v','libx264','-preset','veryfast','-g','30',
                                *options,'-an',str(path)],check=True,capture_output=True,timeout=20)
                clock = probe_constant_frame_clock(path,ffprobe=ffprobe)
                self.assertEqual(clock is not None,eligible,label)
                sequential = cv2.VideoCapture(str(path))
                expected = []
                while True:
                    ok, frame = sequential.read()
                    if not ok:
                        break
                    expected.append(frame)
                sequential.release()
                baseline = cv2.VideoCapture(str(path))
                accelerated = cv2.VideoCapture(str(path))
                try:
                    reader = SeekLocalityReader(accelerated,clock=clock,process_frame=lambda f:f)
                    fps = baseline.get(cv2.CAP_PROP_FPS)
                    queries = [i+part for i in range(2,110,3) for part in (0.,.499999,.5,.500001)]
                    queries += list(reversed(queries))
                    for point in queries:
                        seconds = point/fps
                        baseline.set(cv2.CAP_PROP_POS_MSEC,seconds*1000.)
                        ok, old = baseline.read()
                        actual = reader.frame_at(seconds)
                        if not ok:
                            self.assertIsNone(actual)
                            continue
                        np.testing.assert_array_equal(actual,old,err_msg=f'{label}: {point}')
                        if eligible:
                            index = int(((seconds*1000.)/1000.)*fps+.5)
                            np.testing.assert_array_equal(actual,expected[index],err_msg=f'{label} sequential: {point}')
                    if eligible:
                        self.assertTrue(reader.enabled)
                        self.assertEqual(reader.fallback_count,0)
                        self.assertLess(reader.seek_count,len(queries)//2)
                    else:
                        self.assertFalse(reader.enabled)
                        self.assertEqual(reader.seek_count,len(queries))
                finally:
                    baseline.release()
                    accelerated.release()


if __name__ == '__main__':
    unittest.main()
