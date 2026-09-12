"""Source-authorized body correspondence measured in decoded output pixels.

Experimental QA path, not production activation. A face-hidden source target
must already own a verified source-local track. Output boxes and joints are
detected anew; no source geometry is presented as an output measurement.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from .body_pose_2d import BodyPose2DEvidence
from .focal_selector_geometry import FocalSelectorGeometry
from .pose import FormationPersonDescriptor
from .timeline import _upper_body_shape_mismatch


def _encoder_id(encoder) -> str:
    return hashlib.sha256(json.dumps(encoder.fingerprint(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _cosine(left, right):
    if len(left) != 512 or len(right) != 512:
        raise ValueError('body appearance requires the exact 512-component model basis')
    if any(not math.isfinite(v) for v in (*left, *right)):
        raise ValueError('body appearance must be finite')
    denominator = math.sqrt(sum(v*v for v in left) * sum(v*v for v in right))
    if denominator <= 0:
        raise ValueError('body appearance cannot be zero')
    return sum(a*b for a,b in zip(left,right)) / denominator


@dataclass(frozen=True, slots=True)
class SourceBodyReference:
    source_index: int
    source_seconds: float
    source_body_bbox_xywh: tuple[float, float, float, float]
    source_authority: FocalSelectorGeometry = field(repr=False)
    source_frame_size_wh: tuple[int, int]
    encoder_sha256: str
    appearance: tuple[float, ...] = field(repr=False)

    def __post_init__(self):
        geometry = self.source_authority
        if (not isinstance(geometry, FocalSelectorGeometry) or geometry.person is None
                or (geometry.face_bbox_xywh is None and geometry.body_track is None)
                or geometry.source_index != self.source_index
                or geometry.source_seconds != self.source_seconds
                or geometry.person.bbox_xywh != self.source_body_bbox_xywh):
            raise ValueError('body reference must own its exact source person/frame')
        _cosine(self.appearance, self.appearance)
        if len(self.encoder_sha256) != 64:
            raise ValueError('body reference encoder identity is missing')
        if (len(self.source_frame_size_wh) != 2 or min(self.source_frame_size_wh) <= 0
                or geometry.pose_backend != 'rtmpose_body7' or self.source_pose is None):
            raise ValueError('body reference requires observed source RTMPose, not a generic skeleton')

    @property
    def source_pose(self):
        return BodyPose2DEvidence('observed', self.source_body_bbox_xywh,
            self.source_frame_size_wh, self.source_authority.pose_screen_xyc
        ).as_formation_person(self.source_authority.person.detection_confidence)


def build_source_body_reference(frame, geometry, encoder) -> SourceBodyReference:
    if not isinstance(geometry, FocalSelectorGeometry) or geometry.person is None:
        raise ValueError('source target body is unresolved')
    features, = encoder.describe(frame, (geometry.person.bbox_xywh,))
    return SourceBodyReference(geometry.source_index, geometry.source_seconds,
        geometry.person.bbox_xywh, geometry, (frame.shape[1], frame.shape[0]),
        _encoder_id(encoder), tuple(features))


@dataclass(frozen=True, slots=True)
class RenderedBodyMeasurement:
    reason: str
    person: FormationPersonDescriptor | None = None
    pose: BodyPose2DEvidence | None = None
    similarity: float | None = None
    margin: float | None = None
    candidate_count: int = 0
    source_output_pose_mismatch: float | None = None

    def audit(self) -> dict[str, Any]:
        return {'contract': 'source_authorized_output_body_measurement_v1',
            'reason': self.reason, 'similarity': self.similarity, 'margin': self.margin,
            'candidate_count': self.candidate_count,
            'observed_output_body_bbox_xywh': self.person.bbox_xywh if self.person else None,
            'output_pose_measured': self.pose is not None,
            'observed_output_pose_xyc': self.pose.upper_body_xyc if self.pose else (),
            'source_output_pose_mismatch': self.source_output_pose_mismatch,
            'source_geometry_copied_to_output': False,
            'face_geometry_synthesized': False, 'appearance_embeddings_persisted': False,
            'production_activation_authorized': False}


def measure_referenced_output_body(frame, reference, people, encoder, pose):
    if not isinstance(reference, SourceBodyReference):
        raise TypeError('output body measurement requires a source body reference')
    if _encoder_id(encoder) != reference.encoder_sha256:
        raise ValueError('source and output body encoders differ')
    people = tuple(people)
    if not people:
        return RenderedBodyMeasurement('output_body_unobserved')
    features = encoder.describe(frame, tuple(p.bbox_xywh for p in people))
    if len(features) != len(people):
        raise ValueError('output body appearance count differs from detections')
    scores = sorted(((_cosine(reference.appearance, row), i)
                     for i,row in enumerate(features)), reverse=True)
    similarity, index = scores[0]
    margin = similarity-scores[1][0] if len(scores)>1 else None
    # Same minimum/margin as the source-local frozen-face body anchor gate.
    # This is not a learned identity probability or cross-song quality proof.
    if similarity < .68 or (margin is not None and margin < .06):
        return RenderedBodyMeasurement('output_target_body_ambiguous',
            similarity=similarity, margin=margin, candidate_count=len(people))
    measured = pose.describe(frame, bbox_xywh=people[index].bbox_xywh)
    normalized = measured.as_formation_person(people[index].detection_confidence)
    if measured.reason != 'observed' or normalized is None:
        return RenderedBodyMeasurement('output_target_pose_unobserved',
            similarity=similarity, margin=margin, candidate_count=len(people))
    mismatch = _upper_body_shape_mismatch(reference.source_pose, normalized)
    if mismatch is None or mismatch > .30:
        return RenderedBodyMeasurement('output_pose_source_ownership_unresolved',
            similarity=similarity, margin=margin, candidate_count=len(people),
            source_output_pose_mismatch=mismatch)
    return RenderedBodyMeasurement('observed', normalized, measured,
        similarity, margin, len(people), mismatch)
