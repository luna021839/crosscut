"""Observed-body containment between two independently resolved focal frames.

This does not authorize cross-source pose matching or publish a target identity.
The query box is an actual detection, never an interpolated or union box.
"""
from dataclasses import dataclass
import math

from .direct_member_editorial_motion import DirectMemberFaceSample
from .direct_member_target_face import FOCAL_FACE_BASIS
from .focal_selector_geometry import FOCAL_SELECTOR_BODY_BASIS


def _valid_box(box):
    return (len(box) == 4 and all(isinstance(v, (float, int)) and not isinstance(v, bool)
            and math.isfinite(v) for v in box) and min(box[:2]) >= 0
            and min(box[2:]) > 0 and box[0]+box[2] <= 1.000001 and box[1]+box[3] <= 1.000001)


def _iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    area = max(0., min(a[0]+a[2], b[0]+b[2])-x)*max(0., min(a[1]+a[3], b[1]+b[3])-y)
    return area/max(1e-12, a[2]*a[3]+b[2]*b[3]-area)


@dataclass(frozen=True)
class FocalBracketContainmentSupport:
    source_index: int
    source_seconds: float
    observed_query_boxes: tuple[tuple[float, float, float, float], ...]
    selected_index: int
    anchor_source_seconds: tuple[float, float]
    anchor_boxes: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]

    def __post_init__(self):
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int) or self.source_index < 0
                or isinstance(self.selected_index, bool) or not isinstance(self.selected_index, int)
                or not 0 <= self.selected_index < len(self.observed_query_boxes)
                or not 1 <= len(self.observed_query_boxes) <= 32
                or len(self.anchor_source_seconds) != 2 or len(self.anchor_boxes) != 2
                or not all(_valid_box(b) for b in (*self.observed_query_boxes, *self.anchor_boxes))
                or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                           for t in (self.source_seconds, *self.anchor_source_seconds))
                or not 0 <= self.anchor_source_seconds[0] < self.source_seconds < self.anchor_source_seconds[1]
                or self.anchor_source_seconds[1]-self.anchor_source_seconds[0] > .20+1e-9):
            raise ValueError('body containment needs source-bound observed boxes and short independent brackets')
        for anchor in self.anchor_boxes:
            scores = [_iou(box, anchor) for box in self.observed_query_boxes]
            score = scores[self.selected_index]
            competitor = max((v for i, v in enumerate(scores) if i != self.selected_index), default=0.)
            if score < .75 or score-competitor < .15:
                raise ValueError('both focal anchors must independently prefer the same observed body')

    @property
    def bbox_xywh(self):
        return self.observed_query_boxes[self.selected_index]


def measure_bracketed_body_containment(*, source_index, source_seconds,
        observed_query_boxes, anchors, native_scene_cut_seconds=()):
    """Anchors are (source index, local seconds, strict focal sample) triples.

    Both anchors must come from the existing strict source-local authority,
    never from this recovery route. A native cut or ambiguous correspondence
    leaves the query unresolved. All detections, including rivals, participate.
    """
    if len(anchors) != 2 or any(a[0] != source_index for a in anchors):
        return None
    times = tuple(a[1] for a in anchors)
    if any(times[0] < t <= times[1] for t in native_scene_cut_seconds):
        return None
    boxes = []
    for _, _, sample in anchors:
        if (not isinstance(sample, DirectMemberFaceSample)
                or sample.alignment_geometry_basis not in (FOCAL_FACE_BASIS, FOCAL_SELECTOR_BODY_BASIS)
                or sample.subject_bbox_xywh is None or sample.subject_confidence < .30):
            return None
        boxes.append(sample.subject_bbox_xywh)
    candidates = tuple(tuple(b) for b in observed_query_boxes)
    for index in range(len(candidates)):
        try:
            return FocalBracketContainmentSupport(source_index, source_seconds, candidates,
                index, times, tuple(boxes))
        except ValueError:
            continue
    return None
