"""Run-local, exact-clock camera feedback; not decoded-QA or source rejection."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MemberCameraSearchFeedback:
    proposals: tuple[tuple[float, int, int, float], ...] = ()

    def __post_init__(self):
        keys = set()
        for cut, left, right, cost in self.proposals:
            if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in (cut, cost))
                    or cut < 0 or cost <= 0
                    or any(isinstance(v, bool) or not isinstance(v, int)
                           or v < 0 for v in (left, right)) or left == right):
                raise ValueError('camera feedback requires exact clocks and source pairs')
            key = (cut, left, right)
            if key in keys:
                raise ValueError('camera feedback repeats an exact proposal')
            keys.add(key)

    def penalty(self, cut, left, right):
        # Float arithmetic tolerance, NOT a neighborhood of nearby video frames.
        return max((cost for time, a, b, cost in self.proposals
                    if (left, right) == (a, b) and abs(cut-time) <= 1e-7), default=0.)

    def frame_offsets(self, *, canonical_cut, ordinary_offsets, frame_step):
        """Fill previously skipped frames ONLY in a failed boundary's old window.

        Keep the musical search window unchanged. A two-frame coarse search
        must not keep proposing its sole failed sample while never examining
        the interleaved frames. Dense source/pose/camera checks still apply.
        """
        offsets = tuple(ordinary_offsets)
        if (not offsets or 0 not in offsets
                or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)
                or not math.isfinite(frame_step) or frame_step <= 0
                or not math.isfinite(canonical_cut)):
            raise ValueError('camera refinement needs a valid original frame window')
        radius = max(abs(n) for n in offsets)
        affected = any(abs(canonical_cut-time) <= radius*frame_step+1e-7
                       for time, *_ in self.proposals)
        if not affected:
            return offsets
        return tuple(sorted(range(-radius, radius+1), key=lambda n: (abs(n), n)))
