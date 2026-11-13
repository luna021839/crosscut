"""Source-bound focal geometry for proposals, separate from raw descriptors.

This module receives a frozen-bank match; it does not identify people. The
eight torso-normalized formation joints keep their own basis and cannot be
substituted into the legacy 60-component dominant-pose cache. No face/body
appearance vectors are serialized. Final decoded video QA remains mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .direct_member_target_face import FOCAL_FACE_BASIS, FocalFaceCandidate
from .focal_pose import face_associated_person
from .focal_body_track import FocalBodyTrackResult, FocalBodyGeometryCertificate
from .pose import FormationPersonDescriptor
from .timeline import _upper_body_shape_mismatch


FOCAL_SELECTOR_FACE_BASIS = FOCAL_FACE_BASIS
FOCAL_SELECTOR_POSE_BASIS = 'focal_associated_eight_joint_shape_v1'
FOCAL_SELECTOR_BODY_BASIS = 'frozen_source_focal_body_track_geometry'
# Existing dense renderer limit, shared with early focal proposals. Generic
# silhouettes retain their separate stricter framing/vertical requirements.
FOCAL_MEMBER_MINIMUM_SUBJECT_VISIBLE_FRACTION = .720


@dataclass(frozen=True, slots=True)
class FocalSelectorGeometry:
    source_index: int
    source_seconds: float
    face_bbox_xywh: tuple[float, float, float, float] | None
    face_scale: float | None = None
    person: FormationPersonDescriptor | None = None
    body_track: FocalBodyTrackResult | FocalBodyGeometryCertificate | None = field(default=None, repr=False)
    pose_backend: str = 'formation_pose'
    pose_screen_xyc: tuple[tuple[float, float, float], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                or self.source_index < 0 or isinstance(self.source_seconds, bool)
                or not isinstance(self.source_seconds, (float, int))
                or not math.isfinite(self.source_seconds) or self.source_seconds < 0):
            raise ValueError('focal selector observation must be source/time bound')
        box = self.face_bbox_xywh
        if box is not None:
            if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
                raise ValueError('focal face box must be observed normalized geometry')
        if self.face_scale is not None and (isinstance(self.face_scale, bool)
                or not isinstance(self.face_scale, (float, int))
                or not math.isfinite(self.face_scale) or self.face_scale <= 0):
            raise ValueError('focal landmark scale must be positive')
        track_box = (self.body_track.body_bbox_xywh if isinstance(self.body_track,FocalBodyGeometryCertificate)
                     else self.body_track.observation.person.bbox_xywh
                     if isinstance(self.body_track,FocalBodyTrackResult) and self.body_track.observation else None)
        if self.body_track is not None and (not isinstance(self.body_track, (FocalBodyTrackResult,FocalBodyGeometryCertificate))
                or track_box is None or self.body_track.source_index != self.source_index
                or not math.isclose(self.body_track.source_seconds, self.source_seconds, abs_tol=1e-9, rel_tol=0)
                or self.body_track.reason != 'observed_target_body_track'
                or len(set(self.body_track.support_seconds)) < 2
                or self.person is None or track_box != self.person.bbox_xywh):
            raise ValueError('focal body track must bind this exact observed source body')
        if box is None and (self.face_scale is not None or self.person is not None and self.body_track is None):
            raise ValueError('missing focal face cannot borrow another body')
        if self.person is not None and (not isinstance(self.person, FormationPersonDescriptor)
                or box is not None and face_associated_person(box, (self.person,)) is None):
            raise ValueError('focal pose body must contain its own observed face')
        if self.pose_backend not in ('formation_pose', 'rtmpose_body7'):
            raise ValueError('focal pose backend is not supported')
        if self.pose_screen_xyc and (len(self.pose_screen_xyc)!=8
                or any(len(p)!=3 or any(not math.isfinite(v) for v in p)
                       or not 0<=p[2]<=1 for p in self.pose_screen_xyc)):
            raise ValueError('focal screen pose is invalid')

    def to_dict(self) -> dict:
        return {'contract': 'frozen_source_selector_geometry_v1',
                'alignment_basis': FOCAL_SELECTOR_FACE_BASIS,
                'source_index': self.source_index, 'source_seconds': self.source_seconds,
                'face_bbox_xywh': self.face_bbox_xywh,
                'face_landmark_scale': self.face_scale,
                'associated_body_bbox_xywh': self.person.bbox_xywh if self.person else None,
                'associated_pose_basis': FOCAL_SELECTOR_POSE_BASIS if self.person and self.person.upper_body_keypoints_xyc else None,
                'associated_pose_backend': self.pose_backend,
                'body_track': self.body_track.to_dict() if self.body_track else None,
                'appearance_embeddings_persisted': False,
                'proposal_geometry_not_final_render_authority': True}


def focal_selector_geometry(
    *, source_index: int, source_seconds: float,
    candidate: FocalFaceCandidate | None,
    people: Sequence[FormationPersonDescriptor],
) -> FocalSelectorGeometry:
    person = face_associated_person(candidate.bbox_xywh, people) if candidate else None
    return FocalSelectorGeometry(source_index, source_seconds,
        candidate.bbox_xywh if candidate else None,
        (candidate.eye_scale or candidate.landmark_scale) if candidate else None,
        person)


def focal_pose_shape_mismatch(
    left: FocalSelectorGeometry | None, right: FocalSelectorGeometry | None,
) -> float | None:
    if left is None or right is None or left.person is None or right.person is None or left.pose_backend != right.pose_backend:
        return None
    return _upper_body_shape_mismatch(left.person, right.person)


def common_focal_pose_vectors(
    observations: Sequence[FocalSelectorGeometry | None],
) -> tuple[tuple[float, ...], ...] | None:
    """Use the SAME visible joint indices for an entire cut-adjacent window.

    Independently picking visible joints on each side can compare an elbow's
    velocity with a shoulder's. Missing joints are never zero-filled. Four
    common joints are the same minimum as the existing formation shape metric.
    """
    if len(observations) < 2 or len(observations) > 4:
        raise ValueError('focal motion window must contain two through four observations')
    if any(o is None or o.person is None or len(o.person.upper_body_keypoints_xyc) != 8
           for o in observations):
        return None
    if len({o.pose_backend for o in observations}) != 1:
        return None
    points = [o.person.upper_body_keypoints_xyc for o in observations]
    visible = [j for j in range(8) if min(p[j][2] for p in points) >= .20]
    if len(visible) < 4:
        return None
    return tuple(tuple(float(p[j][axis]) for j in visible for axis in (0, 1)) for p in points)
