"""Source-native head/person/body rows before any virtual camera is searched.

Rows contain real model/ROI/native/file/pixel identities and remain replayable.
Detection IDs are frame-local, never persistent subject or member identities.
There is no acquisition, Haar fallback, hidden descriptor model or approval.
"""
from dataclasses import asdict, dataclass,replace
import hashlib
import json
import math
from pathlib import Path

import cv2

from .group_joint_formation import ObservedFormationPerson
from .group_joint_formation_observer import register_formation_models
from .group_joint_person_ownership import OwnedPersonObservation,bind_person_ownership,ownership_policy_fingerprint
from .group_joint_observations import _file_identity
from .visual_score import YUNET_EXPECTED_SHA256, _working_frame


@dataclass(frozen=True, slots=True)
class NativeGroupHead:
    observation_id: int
    box_xywh: tuple
    landmarks_xy: tuple
    confidence: float
    candidate_body_observation_ids: tuple[int,...] = ()
    extent_basis: str = 'face_roi_not_whole_head'

    def __post_init__(self):
        if (type(self.observation_id) is not int or self.observation_id<0
                or not isinstance(self.box_xywh,tuple) or len(self.box_xywh)!=4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in self.box_xywh)
                or min(self.box_xywh[2:])<=0
                or type(self.confidence) not in (int,float) or not 0<=self.confidence<=1
                or not isinstance(self.landmarks_xy,tuple) or len(self.landmarks_xy)!=5
                or any(not isinstance(p,tuple) or len(p)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in self.landmarks_xy)
                or self.extent_basis!='face_roi_not_whole_head'):
            raise ValueError('native face observation must retain its actual ROI/landmarks, not whole-head authority')


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
    model_adapter_fingerprints: tuple[tuple[str,str],...] = ()
    person_ownership_policy_sha256: str | None = None
    raw_person_detector_json: str | None = None

    def __post_init__(self):
        sha=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if (type(self.request_source_index) is not int or self.request_source_index<0
                or type(self.native_frame_index) is not int or self.native_frame_index<0
                or not isinstance(self.frame_size_wh,tuple) or len(self.frame_size_wh)!=2
                or any(type(v) is not int or v<0 for v in self.frame_size_wh)
                or self.native_pixel_sha256 is not None
                    and (not sha(self.native_pixel_sha256) or min(self.frame_size_wh)<1)
                or not sha(self.source_clock_identity)
                or not isinstance(self.source_file_identity,tuple) or len(self.source_file_identity)!=5
                or any(type(v) is not int for v in self.source_file_identity)
                or not isinstance(self.people,tuple) or len(self.people)>12
                or any(not isinstance(p,ObservedFormationPerson) for p in self.people)
                or len({p.observation_id for p in self.people})!=len(self.people)
                or not isinstance(self.heads,tuple) or len(self.heads)>24
                or any(not isinstance(head,NativeGroupHead) for head in self.heads)
                or len({head.observation_id for head in self.heads})!=len(self.heads)):
            raise ValueError('native observation requires bounded source/native/pixel/raw-model identities')
        if self.person_ownership_policy_sha256 is not None and not sha(self.person_ownership_policy_sha256):
            raise ValueError('native part ownership must bind its actual postprocessing policy')
        if self.person_ownership_policy_sha256 is None and any(
                isinstance(p, OwnedPersonObservation) and p.upper_body_ownership_state != 'unknown' for p in self.people):
            raise ValueError('new upper-body ownership cannot borrow a legacy observation policy')
        for values in (self.model_content_sha256,self.model_adapter_fingerprints):
            if (not isinstance(values,tuple) or any(not isinstance(row,tuple) or len(row)!=2
                        or not isinstance(row[0],str) or not row[0] or not sha(row[1]) for row in values)
                    or len({row[0] for row in values})!=len(values)):
                raise ValueError('native model and adapter identities must be immutable named SHA-256 values')
        if self.model_adapter_fingerprints and tuple(n for n,_ in self.model_adapter_fingerprints)!=tuple(n for n,_ in self.model_content_sha256):
            raise ValueError('every native model must bind the same named adapter set')
        if self.raw_person_detector_json is not None:
            from .group_native_detector_result import validate_detector_result, MAX_RAW_RESULT_BYTES
            if (not isinstance(self.raw_person_detector_json, str)
                    or len(self.raw_person_detector_json) > MAX_RAW_RESULT_BYTES*3+4096):
                raise ValueError('native raw detector receipt must be frozen JSON text')
            validate_detector_result(json.loads(self.raw_person_detector_json), asdict(self))

    def to_dict(self):
        payload = asdict(self)
        raw_detector = payload.pop('raw_person_detector_json')
        contract = 'actual_source_native_owned_person_body_face_observation_rows_v3'
        if self.person_ownership_policy_sha256 is None:
            # Legacy raw preimages stay byte-semantically reproducible, without
            # upgrading them to new partial-body ownership evidence.
            contract = 'actual_source_native_owned_person_body_face_observation_rows_v2'
            payload.pop('person_ownership_policy_sha256')
            for person in payload['people']:
                person.pop('upper_body_ownership_state', None)
        if raw_detector is not None:
            from .group_native_detector_result import detector_semantics_complete
            contract = 'actual_source_native_owned_person_body_face_observation_rows_v4'
            result = json.loads(raw_detector)
            payload['raw_person_detector_result'] = result
            payload['person_detector_semantics_complete'] = detector_semantics_complete(result, payload)
        return dict(contract=contract, **payload,
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
    session.register_model('yunet_head_faces', head_model_path, heads,
        adapter_fingerprint={'contract':'yunet_native_face_not_whole_head_v1',
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'opencv_version':cv2.__version__,'working_maximum_width':960,
            'confidence':.76,'nms':.3,'top_k':5000})


def observe_native_group(session, *, request_source_index, native_frame_index,
        head_analysis_roll=None):
    """Bound all original geometry, even if it is missing or later cropped.

    Camera variants do not re-detect/select a smaller cohort. Source-relative
    temporal tracking and role declaration must consume these original rows.
    A caller may predeclare ONE analysis-only detector roll for a source
    window. Its inverse-mapped heads replace, never concatenate with, the
    unrotated detector's alternatives. The actual named model/adapter is part
    of the raw row hash. Missing results never fall back to another variant.
    """
    from .group_native_head_orientation import (
        OrientedNativeHeadAlternative, native_head_analysis_spec)
    head_model, id_base = native_head_analysis_spec(head_analysis_roll)
    required_models = ('nanodet_people', 'rtmpose_body7', head_model)
    if any(name not in session.models or name not in session.model_adapters
            for name in required_models):
        raise ValueError('native group needs its predeclared registered detector variant')
    source, native = request_source_index, native_frame_index
    pixels = session.native_pixels(source, native)
    local = session.domain.eligible_index(source)
    clock = session.domain.render_frame_clocks[local]
    clock_identity = hashlib.sha256(json.dumps(clock.to_dict(), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    identity = _file_identity(session.domain.request_sources[source])
    models = tuple((name, session.models[name][2]) for name in required_models)
    adapters=tuple((name,session.model_adapters[name]) for name,_ in models)
    raw_detector_json = None
    def snapshot(reason, people=(), heads=()):
        return NativeGroupObservation(source, native,
            (pixels.shape[1], pixels.shape[0]) if pixels is not None else (0, 0), clock_identity, identity,
            hashlib.sha256(memoryview(pixels)).hexdigest() if pixels is not None else None,
            models, tuple(people), tuple(heads), reason,adapters,ownership_policy_fingerprint(), raw_detector_json)
    if pixels is None:
        return snapshot('exact_native_decode_unavailable')
    heads=session.observe_native(source,native,model=head_model)
    head_missing=heads is None
    if head_missing:
        heads=()
    elif head_analysis_roll is not None:
        if (not isinstance(heads, tuple) or len(heads) > 24
                or any(not isinstance(value, OrientedNativeHeadAlternative)
                    or value.model_name != head_model
                    or value.analysis_roll_degrees != head_analysis_roll
                    or not 0 <= value.head.observation_id < 24 for value in heads)):
            raise ValueError('native oriented heads differ from the frozen detector variant')
        heads = tuple(replace(value.head, observation_id=id_base + value.head.observation_id)
            for value in heads)
    rows = session.observe_native(source, native, model='nanodet_people')
    from .group_native_detector_result import capture_detector_result, bounded_detector_rows
    result = capture_detector_result(rows, binding=dict(request_source_index=source, native_frame_index=native,
        source_clock_identity=clock_identity, source_file_identity=list(identity),
        native_pixel_sha256=hashlib.sha256(memoryview(pixels)).hexdigest()),
        model_content_sha256=session.models['nanodet_people'][2],
        model_adapter_sha256=session.model_adapters['nanodet_people'])
    raw_detector_json = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if rows is None:
        return snapshot('person_model_unavailable',heads=heads)
    if result['row_count'] is not None and result['row_count'] > 12:
        return snapshot('person_detection_overflow_not_truncated',heads=heads)
    try:
        rows = bounded_detector_rows(result)
    except ValueError:
        return snapshot('invalid_or_unretained_native_body_detector_result',heads=heads)
    height, width = pixels.shape[:2]
    people = []
    for index, row in enumerate(rows):
        left, top, right, bottom, confidence = map(float, row)
        box = (left/width, top/height, (right-left)/width, (bottom-top)/height)
        if (any(not math.isfinite(v) for v in (*box, confidence)) or not 0 <= confidence <= 1
                or box[2] <= 0 or box[3] <= 0 or box[0] < 0 or box[1] < 0
                or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
            return snapshot('invalid_source_native_body_roi',heads=heads)
        candidates=tuple(head for head in heads if box[0]<=head.box_xywh[0]+head.box_xywh[2]/2<=box[0]+box[2]
            and box[1]<=head.box_xywh[1]+head.box_xywh[3]/2<=box[1]+box[3])
        face_box=None
        if len(candidates)==1 and all(v>=0 for v in candidates[0].box_xywh[:2]) \
                and candidates[0].box_xywh[0]+candidates[0].box_xywh[2]<=1 \
                and candidates[0].box_xywh[1]+candidates[0].box_xywh[3]<=1:
            face_box=candidates[0].box_xywh
        # One native body tensor is shared across head analysis variants.
        # Keep the face-only consistency view separate; dropping it from a
        # complete result cache would incorrectly share True/False outcomes.
        body = session.observe_native(source, native, model='rtmpose_body7',
            arguments={'bbox_xywh':list(box)})
        from .body_pose_2d import pose_face_anchor_consistency
        face_consistent=pose_face_anchor_consistency(body,face_box)
        points = body.upper_body_xyc if body is not None and body.reason == 'observed' else ()
        people.append(OwnedPersonObservation(index,box,points,confidence,
            pose_crop_xywh=None if body is None else body.pose_crop_xywh,
            source_pixel_to_pose_input_affine=() if body is None else body.source_pixel_to_pose_input_affine,
            raw_pose_xyc=() if body is None else body.raw_keypoints_xyc,
            pose_reason='pose_model_unavailable' if body is None else body.reason,
            face_anchor_consistent=face_consistent))
    people,owners=bind_person_ownership(tuple(people),heads,(width,height))
    heads=tuple(replace(head,candidate_body_observation_ids=tuple(owners[head.observation_id])) for head in heads)
    return snapshot('head_model_unavailable_or_detection_overflow' if head_missing else
        'observed_native_model_rows' if people else 'no_people_detected',people,heads)
