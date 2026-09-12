"""Short pixel-continuity handles, not face recognition or cut authorization.

The cut's already selected box is followed through nearby frames. Each step
needs strict forward/backward optical flow; congruent face geometry refreshes
the box, with at most two consecutive detector misses. A different face with
a high recognition cosine cannot replace the tracked box from elsewhere in
the image. No pixels, keypoints, or identity embeddings are serialized.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .boundary_subject_alignment import _optical_flow_box


Box = tuple[float, float, float, float]


def _valid_box(box: Box) -> bool:
    return (isinstance(box, (tuple, list)) and len(box) == 4
            and all(not isinstance(v, bool) and isinstance(v, (int, float))
                    and math.isfinite(v) for v in box)
            and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
            and box[0]+box[2] <= 1.000001 and box[1]+box[3] <= 1.000001)


def _iou(a: Box, b: Box) -> float:
    w = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    h = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return w*h/max(1e-12, a[2]*a[3]+b[2]*b[3]-w*h)


@dataclass(frozen=True, slots=True)
class ContainmentSample:
    index: int
    box: Box
    basis: str
    consecutive_detector_misses: int


@dataclass(frozen=True, slots=True)
class CutAnchoredContainment:
    samples: tuple[ContainmentSample, ...]
    complete: bool
    failure: str | None
    failed_index: int | None


def track_cut_handle(
    *, frames: Sequence[np.ndarray | None], times: Sequence[float],
    face_boxes: Sequence[Sequence[Box]], anchor_box: Box, anchor_at_end: bool,
    output_fps: float, scene_break_indices: Sequence[int] = (),
    maximum_handle_seconds: float = .65, maximum_detector_misses: int = 2,
) -> CutAnchoredContainment:
    """Return measured containment support only inside one short source shot.

``scene_break_indices`` identifies the first frame after a native source cut.
Every interval with such a cut is rejected. A caller must bind the source hash,
actual decoded frame clocks, selected endpoint and renderer crop independently.
The anchor must be an actual observed face; this function never invents one.
    """
    if not frames or len(frames) != len(times) or len(frames) != len(face_boxes):
        raise ValueError('handle frames, times and face boxes must align')
    if not isinstance(anchor_at_end, bool) or not _valid_box(anchor_box):
        raise ValueError('handle anchor is invalid')
    if isinstance(output_fps, bool) or not isinstance(output_fps, (int, float)) or not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError('output fps must be positive and finite')
    if isinstance(maximum_handle_seconds, bool) or not isinstance(maximum_handle_seconds, (int, float)) or not math.isfinite(maximum_handle_seconds) or not 0 < maximum_handle_seconds <= .75:
        raise ValueError('containment must stay a short boundary handle')
    if isinstance(maximum_detector_misses, bool) or not isinstance(maximum_detector_misses, int) or not 0 <= maximum_detector_misses <= 2:
        raise ValueError('at most two consecutive detector misses are supported')
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('handle clocks must be finite and strictly ordered')
    if any(not _valid_box(b) for boxes in face_boxes for b in boxes):
        raise ValueError('observed face geometry is invalid')
    if times[-1]-times[0] > maximum_handle_seconds+1e-9:
        return CutAnchoredContainment((), False, 'handle_duration_exceeded', None)
    if any(b-a > 2./output_fps+1e-9 for a,b in zip(times,times[1:])):
        return CutAnchoredContainment((), False, 'handle_frame_gap', None)
    if any(isinstance(i, bool) or not isinstance(i,int) or not 1 <= i < len(frames) for i in scene_break_indices):
        raise ValueError('scene break index is outside the handle')
    if scene_break_indices:
        return CutAnchoredContainment((), False, 'native_scene_cut', min(scene_break_indices))
    order = tuple(reversed(range(len(frames)))) if anchor_at_end else tuple(range(len(frames)))
    anchor_index = order[0]
    if not any(_iou(anchor_box,b) >= .9 for b in face_boxes[anchor_index]):
        return CutAnchoredContainment((), False, 'anchor_not_observed', anchor_index)
    samples: list[ContainmentSample] = []
    previous = None
    box = anchor_box
    misses = 0
    for index in order:
        frame = frames[index]
        if not isinstance(frame,np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            return CutAnchoredContainment(tuple(samples), False, 'frame_unavailable', index)
        if previous is None:
            basis = 'observed_cut_anchor'
        else:
            box = _optical_flow_box(previous, frame, box)
            if box is None:
                return CutAnchoredContainment(tuple(samples), False, 'pixel_flow_failed', index)
            ranked = sorted(((_iou(box,b),b) for b in face_boxes[index]), reverse=True)
            if ranked and ranked[0][0] >= .35:
                if len(ranked)>1 and ranked[1][0] >= .35 and ranked[0][0]-ranked[1][0] < .1:
                    return CutAnchoredContainment(tuple(samples), False, 'face_geometry_ambiguous', index)
                box = ranked[0][1]
                misses = 0
                basis = 'pixel_flow_plus_observed_geometry'
            else:
                misses += 1
                if misses > maximum_detector_misses:
                    return CutAnchoredContainment(tuple(samples), False, 'detector_gap_exceeded', index)
                basis = 'bounded_pixel_flow_containment'
        samples.append(ContainmentSample(index, box, basis, misses))
        previous = frame
    return CutAnchoredContainment(tuple(sorted(samples,key=lambda s:s.index)), True, None, None)


def track_around_cut_anchor(
    *, frames: Sequence[np.ndarray | None], times: Sequence[float],
    face_boxes: Sequence[Sequence[Box]], anchor_box: Box, anchor_index: int,
    output_fps: float, scene_break_indices: Sequence[int] = (),
) -> CutAnchoredContainment:
    """Track both ways from the actual measured cut frame, never a new face.

The complete handle, not each half, remains bounded to .65 seconds. Renderer
endpoints may be half a frame outside the measured cut frame; both must retain
pixel continuity with that same observed anchor. Failure samples are ordered.
    """
    if (isinstance(anchor_index, bool) or not isinstance(anchor_index, int)
            or not 0 <= anchor_index < len(frames)):
        raise ValueError('cut anchor index is outside the handle')
    if not times or len(frames) != len(times) or len(frames) != len(face_boxes):
        raise ValueError('handle frames, times and face boxes must align')
    if any(isinstance(t, bool) or not isinstance(t, (int, float))
           or not math.isfinite(t) for t in times):
        raise ValueError('handle clocks must be finite numbers')
    # Validate the entire clock and scene domain before splitting. The anchor
    # remains observed even when it is not either outer renderer endpoint.
    if times[-1] - times[0] > .65 + 1e-9:
        return CutAnchoredContainment((), False, 'handle_duration_exceeded', None)
    if scene_break_indices:
        # Reuse public validation; this rejects before reading any flow pixels.
        return track_cut_handle(frames=frames, times=times, face_boxes=face_boxes,
            anchor_box=anchor_box, anchor_at_end=False, output_fps=output_fps,
            scene_break_indices=scene_break_indices)
    collected: dict[int, ContainmentSample] = {}
    for start, end, backwards in ((0, anchor_index + 1, True),
                                  (anchor_index, len(frames), False)):
        result = track_cut_handle(frames=frames[start:end], times=times[start:end],
            face_boxes=face_boxes[start:end], anchor_box=anchor_box,
            anchor_at_end=backwards, output_fps=output_fps)
        for sample in result.samples:
            index = sample.index + start
            collected[index] = ContainmentSample(index, sample.box, sample.basis,
                                                  sample.consecutive_detector_misses)
        if not result.complete:
            return CutAnchoredContainment(tuple(collected[i] for i in sorted(collected)),
                False, result.failure,
                result.failed_index + start if result.failed_index is not None else None)
    return CutAnchoredContainment(tuple(collected[i] for i in sorted(collected)), True, None, None)
