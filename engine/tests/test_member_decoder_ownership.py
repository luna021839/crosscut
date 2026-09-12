"""Exercise the real sampler/provider composition, not a decoder in isolation."""
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.direct_member_editorial_motion import DirectMemberDenseFaceProvider, _enable_dense_sequential_frame_reads
from crosscut.seek_locality import ConstantFrameClock, SeekLocalityReader
from crosscut.visual_score import _VideoSampler, _working_frame
from engine.tests.test_seek_locality import FakeCapture


class OwnedSampler(_VideoSampler):
    def __init__(self):
        self.capture = FakeCapture()
        self.frame_rate = 30.0
        self.cache = OrderedDict()
        self.frame_decode_count = 0
        self.closed = False
        self._seek_locality = SeekLocalityReader(
            self.capture, clock=ConstantFrameClock(Fraction(30), 300),
            process_frame=_working_frame,
        )

    def close(self):
        self.closed = True


class MemberDecoderOwnershipTests(unittest.TestCase):
    def test_legacy_wrapper_reproduces_loss_of_decoder_ownership(self):
        sampler = OwnedSampler()
        _enable_dense_sequential_frame_reads(sampler)
        for index in (30,31,33):
            sampler.frame_at(index/30)
        self.assertFalse(sampler._seek_locality.enabled)
        self.assertEqual(1,sampler._seek_locality.fallback_count)

    def test_member_constructor_keeps_one_decoder_owner_and_exact_pixels(self):
        sampler = OwnedSampler()
        with patch('crosscut.direct_member_editorial_motion._VideoSampler',return_value=sampler):
            provider = DirectMemberDenseFaceProvider((Path('fixture.mp4'),))
        try:
            for index in (30,31,33,32,35,36,40,38,45):
                frame = provider._samplers[0].frame_at(index/30)
                np.testing.assert_array_equal(frame,np.full((2,3,3),index,dtype=np.int32))
            self.assertTrue(sampler._seek_locality.enabled)
            self.assertEqual(0,sampler._seek_locality.fallback_count)
            self.assertEqual(1,sampler._seek_locality.seek_count)
        finally:
            provider.close()
        self.assertTrue(sampler.closed)


if __name__ == '__main__':
    unittest.main()
