"""Bounded decoder locality for an already-verified native frame index.

Unlike a timestamp sampler this reader never rounds a query, borrows a nearby
frame, or falls back to another clock. The caller owns the capture and source
identity. This object retains no pixels and skips RGB materialization only for
intermediate frames whose presentation/index positions are also checked.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .render_frame_schedule import VerifiedRenderFrameClock


class ExactNativeFrameReader:
    def __init__(self, capture, clock: VerifiedRenderFrameClock, *, maximum_forward_frames=120):
        if not isinstance(clock, VerifiedRenderFrameClock) or clock.grid_basis not in {
            'exact_cfr_presentation_grid', 'bijective_nearest_index_presentation_grid',
        }:
            raise ValueError('exact native reads require a verified supported clock')
        if type(maximum_forward_frames) is not int or not 0 <= maximum_forward_frames <= 120:
            raise ValueError('native forward window must be 0...120 frames')
        self.capture, self.clock = capture, clock
        self.maximum_forward_frames = maximum_forward_frames
        self.next_index = None
        self.seek_count = self.read_count = self.grab_count = self.invalid_read_count = 0
        try:
            self.sequential_enabled = capture.getBackendName() == 'FFMPEG'
        except (AttributeError, cv2.error):
            self.sequential_enabled = False

    def invalidate_position(self):
        """The owning caller explicitly performed a different capture read."""
        self.next_index = None

    def _invalid(self):
        self.next_index = None
        self.sequential_enabled = False
        self.invalid_read_count += 1
        return None

    def _position_matches(self, index):
        rate = float(self.clock.frame_rate)
        tolerance = (1e-6 if self.clock.grid_basis == 'exact_cfr_presentation_grid'
                     else .5 / rate - 1e-9)
        return (math.isclose(self.capture.get(cv2.CAP_PROP_POS_FRAMES), index+1,
                            rel_tol=0., abs_tol=1e-6)
                and math.isclose(self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000., index/rate,
                                 rel_tol=0., abs_tol=tolerance))

    def frame_at(self, index: int) -> np.ndarray | None:
        if type(index) is not int or not 0 <= index < self.clock.frame_count:
            raise ValueError('native frame is outside the verified source')
        capture = self.capture
        if (not math.isclose(capture.get(cv2.CAP_PROP_FPS), float(self.clock.frame_rate),
                             rel_tol=1e-9, abs_tol=0.)
                or capture.get(cv2.CAP_PROP_FRAME_COUNT) != self.clock.frame_count):
            return self._invalid()
        if self.next_index is not None and not math.isclose(
                capture.get(cv2.CAP_PROP_POS_FRAMES), self.next_index, rel_tol=0., abs_tol=1e-6):
            return self._invalid()
        if (not self.sequential_enabled or self.next_index is None
                or not 0 <= index-self.next_index <= self.maximum_forward_frames):
            self.seek_count += 1
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                return self._invalid()
            self.next_index = index
        while self.next_index <= index:
            current = self.next_index
            if current == index:
                self.read_count += 1
                success, pixels = capture.read()
                if not success or not isinstance(pixels, np.ndarray) or not pixels.size:
                    return self._invalid()
            else:
                self.grab_count += 1
                if not capture.grab():
                    return self._invalid()
            if not self._position_matches(current):
                return self._invalid()
            self.next_index = current+1
        return pixels

    def diagnostics(self):
        return dict(contract='verified_native_index_bounded_forward_decode_v1',
            sequential_enabled=self.sequential_enabled,
            maximum_forward_frames=self.maximum_forward_frames,
            seek_count=self.seek_count, read_count=self.read_count,
            skipped_rgb_frame_count=self.grab_count, invalid_read_count=self.invalid_read_count,
            neighboring_frame_fallback=False, pixels_retained=0)
