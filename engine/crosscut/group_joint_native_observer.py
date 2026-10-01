"""Source-native head/person/body rows before any virtual camera is searched.

Rows contain real model/ROI/native/file/pixel identities and remain replayable.
Detection IDs are frame-local, never persistent subject or member identities.
There is no acquisition, Haar fallback, hidden descriptor model or approval.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2

from .group_joint_formation import ObservedFormationPerson
from .group_joint_formation_observer import register_formation_models
from .group_joint_observations import _file_identity
from .visual_score import YUNET_EXPECTED_SHA256, _working_frame


@dataclass(frozen=True, slots=True)
class NativeGroupHead:
    observation_id: int
    box_xywh: tuple
    landmarks_xy: tuple
    confidence: float


@dataclass(frozen=True, slots=True)
class NativeGroupObservation:
    request_source_index: int
    native_frame_index: int
    frame_size_wh: tuple[int, int]
    source_clock_identity: str
    source_file_identity: tuple
    native_pixel_sha256: str | None
    model_content_sha256: tuple[tuple[str, str], ...]
    people: tuple[ObservedFormationPerson, ...]
    heads: tuple[NativeGroupHead, ...]
    reason: str

    def to_dict(self):
        return dict(contract='actual_source_native_person_body_head_observation_rows_v1', **asdict(self),
            detection_ids_are_frame_local=True, persistent_subject_correspondence_verified=False,
            named_member_identity_certified=False, motion_certified=False,
            visibility_certified=False, quality_approved=False)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def register_joint_native_models(session, *, person_model_path, pose_model_path, head_model_path):
    """Each actual model call debits the same named worker budget."""
    register_formation_models(session, person_model_path=person_model_path, pose_model_path=pose_model_path)
    head_model_path = Path(head_model_path)
    session.budget.check()
    with head_model_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != YUNET_EXPECTED_SHA256:
        raise ValueError('joint native head model differs from its pinned asset')
    detector = cv2.FaceDetectorYN.create(str(head_model_path), '', (960, 540), .76, .3, 5000)
    def heads(pixels):
        working = _working_frame(pixels, maximum_width=960)
        height, width = working.shape[:2]
        detector.setInputSize((width, height))
        _, rows = detector.detect(working)
        if rows is None:
            return ()
        if len(rows) > 24:
            return None
        result = []
        for index, row in enumerate(rows):
            x, y, w, h = (float(v) for v in row[:4])
            points = tuple((float(row[n]/width), float(row[n+1]/height)) for n in range(4, 14, 2))
            result.append(NativeGroupHead(index, (x/width, y/height, w/width, h/height),
                points, float(row[14])))
        return tuple(result)
    session.register_model('yunet_head_faces', head_model_path, heads)


def observe_native_group(session, *, request_source_index, native_frame_index):
    """Bound all original geometry, even if it is missing or later cropped.

    Camera variants do not re-detect/select a smaller cohort. Source-relative
    temporal tracking and role declaration must consume these original rows.
    """
    source, native = request_source_index, native_frame_index
    pixels = session.native_pixels(source, native)
    local = session.domain.eligible_index(source)
    clock = session.domain.render_frame_clocks[local]
    clock_identity = hashlib.sha256(json.dumps(clock.to_dict(), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    identity = _file_identity(session.domain.request_sources[source])
    models = tuple((name, session.models[name][2]) for name in
        ('nanodet_people', 'rtmpose_body7', 'yunet_head_faces'))
    def snapshot(reason, people=(), heads=()):
        return NativeGroupObservation(source, native,
            (pixels.shape[1], pixels.shape[0]) if pixels is not None else (0, 0), clock_identity, identity,
            hashlib.sha256(memoryview(pixels)).hexdigest() if pixels is not None else None,
            models, tuple(people), tuple(heads), reason)
    if pixels is None:
        return snapshot('exact_native_decode_unavailable')
    rows = session.observe_native(source, native, model='nanodet_people')
    if rows is None:
        return snapshot('person_model_unavailable')
    if len(rows) > 12:
        return snapshot('person_detection_overflow_not_truncated')
    height, width = pixels.shape[:2]
    people = []
    for index, row in enumerate(rows):
        left, top, right, bottom, confidence = map(float, row)
        box = (left/width, top/height, (right-left)/width, (bottom-top)/height)
        if (box[2] <= 0 or box[3] <= 0 or box[0] < 0 or box[1] < 0
                or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
            return snapshot('invalid_source_native_body_roi')
        body = session.observe_native(source, native, model='rtmpose_body7', arguments={'bbox_xywh': list(box)})
        points = body.upper_body_xyc if body is not None and body.reason == 'observed' else ()
        people.append(ObservedFormationPerson(index, box, points, confidence))
    heads = session.observe_native(source, native, model='yunet_head_faces')
    if heads is None:
        return snapshot('head_model_unavailable_or_detection_overflow', people)
    return snapshot('observed_native_model_rows' if people else 'no_people_detected', people, heads)
