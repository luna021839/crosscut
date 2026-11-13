from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.exact_native_frame_reader import ExactNativeFrameReader
from crosscut.render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock


class Capture:
    def __init__(self, *, basis='exact_cfr_presentation_grid', backend='FFMPEG'):
        self.clock = VerifiedRenderFrameClock(Fraction(30), 300, 'a'*64, basis)
        self.position = 0
        self.backend = backend
        self.reads, self.grabs, self.seeks = [], [], []
        self.bad_pts = self.bad_position = self.failed_index = None

    def getBackendName(self): return self.backend

    def get(self, key):
        if key == cv2.CAP_PROP_FPS: return 30.
        if key == cv2.CAP_PROP_FRAME_COUNT: return 300.
        if key == cv2.CAP_PROP_POS_FRAMES:
            return self.position + int(self.position-1 == self.bad_position)
        if key == cv2.CAP_PROP_POS_MSEC:
            value = (self.position-1)/30.*1000.
            if self.position-1 == self.bad_pts: value += 1000./30.
            if self.clock.grid_basis == 'bijective_nearest_index_presentation_grid': value=round(value)
            return value
        raise AssertionError(key)

    def set(self, key, index):
        assert key == cv2.CAP_PROP_POS_FRAMES
        self.position = index
        self.seeks.append(index)
        return True

    def read(self):
        index = self.position
        self.position += 1
        self.reads.append(index)
        return index != self.failed_index, np.full((8,8,3), index % 256, np.uint8)

    def grab(self):
        index = self.position
        self.position += 1
        self.grabs.append(index)
        return index != self.failed_index


class ExactNativeReaderTests(unittest.TestCase):
    def test_forward_window_skips_only_rgb_and_preserves_exact_requested_ids(self):
        cap = Capture()
        reader = ExactNativeFrameReader(cap,cap.clock,maximum_forward_frames=48)
        for index in (20,21,40,39,180):
            self.assertEqual(reader.frame_at(index)[0,0,0], index)
        self.assertEqual(cap.seeks,[20,39,180])
        self.assertEqual(cap.reads,[20,21,40,39,180])
        self.assertEqual(cap.grabs,list(range(22,40)))
        self.assertEqual(reader.diagnostics()['pixels_retained'],0)

    def test_millisecond_vp9_clock_has_only_proven_half_cell_tolerance(self):
        cap = Capture(basis='bijective_nearest_index_presentation_grid')
        reader = ExactNativeFrameReader(cap,cap.clock)
        self.assertIsNotNone(reader.frame_at(1))
        self.assertIsNotNone(reader.frame_at(8))
        cap.bad_pts=9
        self.assertIsNone(reader.frame_at(10))
        self.assertEqual(cap.reads,[1,8])
        self.assertFalse(reader.sequential_enabled)

    def test_bad_intermediate_pts_position_or_failed_grab_cannot_borrow_target(self):
        for problem in ('bad_pts','bad_position','failed_index'):
            cap=Capture(); setattr(cap,problem,23)
            reader=ExactNativeFrameReader(cap,cap.clock)
            self.assertIsNotNone(reader.frame_at(20))
            self.assertIsNone(reader.frame_at(25))
            self.assertEqual(cap.reads,[20])
            self.assertEqual(cap.seeks,[20])
            self.assertFalse(reader.diagnostics()['neighboring_frame_fallback'])
            self.assertEqual(reader.invalid_read_count,1)

    def test_external_capture_move_needs_explicit_invalidation(self):
        cap=Capture(); reader=ExactNativeFrameReader(cap,cap.clock)
        reader.frame_at(20)
        cap.position=70
        self.assertIsNone(reader.frame_at(21))
        reader.invalidate_position()
        self.assertIsNotNone(reader.frame_at(21))
        self.assertEqual(cap.seeks,[20,21])

    def test_unsupported_backend_and_zero_window_use_validated_index_seeks(self):
        for backend,window in (('AVFOUNDATION',120),('FFMPEG',0)):
            cap=Capture(backend=backend); reader=ExactNativeFrameReader(cap,cap.clock,maximum_forward_frames=window)
            reader.frame_at(20); reader.frame_at(22)
            self.assertEqual(cap.seeks,[20,22])
            self.assertEqual(cap.grabs,[])

    def test_invalid_requests_and_clock_are_not_silently_clamped(self):
        cap=Capture(); reader=ExactNativeFrameReader(cap,cap.clock)
        for index in (-1,300,True,2.2):
            with self.assertRaises(ValueError): reader.frame_at(index)
        for window in (-1,121,True):
            with self.assertRaises(ValueError): ExactNativeFrameReader(cap,cap.clock,maximum_forward_frames=window)
        with self.assertRaises(ValueError): ExactNativeFrameReader(cap,None)
        unknown=VerifiedRenderFrameClock(Fraction(30),300,'a'*64,'unverified_grid')
        with self.assertRaises(ValueError): ExactNativeFrameReader(cap,unknown)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class ExactNativeReaderPixelTests(unittest.TestCase):
    def test_actual_h264_b_frames_and_vp9_match_independent_seeks_and_encoded_ids(self):
        frames=np.zeros((160,90,160,3),np.uint8)
        for index in range(len(frames)):
            for bit in range(8): frames[index,:,bit*20:(bit+1)*20,:]=235 if index>>bit&1 else 16
        with tempfile.TemporaryDirectory(prefix='crosscut-exact-native-') as directory:
            for codec,opts in (('libx264',('-crf','0','-bf','3')),
                               ('libvpx-vp9',('-lossless','1','-deadline','realtime','-cpu-used','8'))):
                path=Path(directory)/(codec+'.mp4')
                subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','bgr24',
                    '-video_size','160x90','-framerate','30000/1001','-i','pipe:0',
                    '-c:v',codec,*opts,'-g','120','-pix_fmt','yuv420p',str(path)],
                    input=frames.tobytes(),capture_output=True,check=True)
                clock=probe_render_frame_clock(path)
                self.assertIsNotNone(clock)
                cap,independent=cv2.VideoCapture(str(path)),cv2.VideoCapture(str(path))
                try:
                    reader=ExactNativeFrameReader(cap,clock)
                    for index in (10,11,16,42,43,120,119,121,145,159):
                        actual=reader.frame_at(index)
                        self.assertIsNotNone(actual,(codec,index))
                        independent.set(cv2.CAP_PROP_POS_FRAMES,index)
                        ok,expected=independent.read()
                        self.assertTrue(ok)
                        np.testing.assert_array_equal(actual,expected)
                        decoded=sum((actual[30:60,bit*20+8:bit*20+12].mean()>125)<<bit for bit in range(8))
                        self.assertEqual(decoded,index)
                    self.assertLess(reader.seek_count,4)
                    self.assertGreater(reader.grab_count,80)
                    self.assertEqual(reader.invalid_read_count,0)
                finally:
                    cap.release(); independent.release()


if __name__=='__main__': unittest.main()
