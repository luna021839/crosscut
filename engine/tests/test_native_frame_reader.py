from copy import deepcopy
from fractions import Fraction
import unittest

from crosscut.native_frame_reader import native_constant_frame_clock
from crosscut.seek_locality import ConstantFrameClock, SeekLocalityReader
from engine.tests.test_seek_locality import FakeCapture, payload


class NativeFrameReaderTests(unittest.TestCase):
    def h264(self):
        value=payload()
        value['streams'][0]['codec_name']='h264'
        return value

    def test_h264_b_frame_packet_order_does_not_change_presentation_grid(self):
        value=self.h264()
        value['packets']=value['packets'][::2]+value['packets'][1::2]
        self.assertEqual(native_constant_frame_clock(value),ConstantFrameClock(Fraction(30000,1001),12))

    def test_vfr_start_offset_missing_or_duplicate_packet_never_uses_index_alias(self):
        original=self.h264()
        cases=[]
        value=deepcopy(original); value['packets'][4]['pts']+=1; cases.append(value)
        value=deepcopy(original); value['packets'][4]['pts']=value['packets'][3]['pts']; cases.append(value)
        value=deepcopy(original); value['packets'].pop(); cases.append(value)
        value=deepcopy(original); value['streams'][0]['start_time']='0.01'; cases.append(value)
        value=deepcopy(original); value['streams'][0]['avg_frame_rate']='29/1'; cases.append(value)
        value=deepcopy(original); value['packets'][3]['duration']=True; cases.append(value)
        value=deepcopy(original); value['packets'][3]['pts']=True; cases.append(value)
        for value in cases:
            with self.subTest(value=value):
                self.assertIsNone(native_constant_frame_clock(value))

    def test_unknown_codec_and_incomplete_metadata_stay_unsupported(self):
        value=self.h264(); value['streams'][0]['codec_name']='hevc'
        self.assertIsNone(native_constant_frame_clock(value))
        self.assertIsNone(native_constant_frame_clock({}))
        value=self.h264(); value['streams'][0]['nb_frames']=True
        self.assertIsNone(native_constant_frame_clock(value))
        self.assertIsNotNone(native_constant_frame_clock(payload()))

    def test_original_frame_retention_is_one_and_decoder_drift_falls_back(self):
        cap=FakeCapture()
        reader=SeekLocalityReader(cap,clock=ConstantFrameClock(Fraction(30),300),
                                  process_frame=lambda frame:frame,maximum_cached_frames=1)
        for index in range(20,31):
            frame=reader.frame_at(index/30)
            self.assertEqual(int(frame[0,0,0]),index)
            self.assertEqual(len(reader.frames),1)
        self.assertEqual(reader.seek_count,1)
        cap.bad_pts=True
        reader.frame_at(31/30)
        self.assertFalse(reader.enabled)


if __name__=='__main__':
    unittest.main()
