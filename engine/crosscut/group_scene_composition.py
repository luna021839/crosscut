"""Head-independent native body-support/composition observations.

Detector rectangles and their measured pixel support are NOT articulated body
motion, named performers, cast membership, anatomical extent or phase. The
conservative v1 cohort is every bounded native detector rectangle, selected
before camera fitting; births/deaths/occlusion remain unsupported, not erased.
No focal detector/pose inference is called by this producer. Not-requested
modalities are different from a detector reporting zero faces or no arms.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .group_joint_native_observer import NativeGroupObservation
from .group_joint_formation import ObservedFormationPerson
from .group_joint_observations import _file_identity
from .boundary_subject_alignment import _optical_flow_box
from .cut_anchored_containment import _iou

RAW_CONTRACT = 'native_detector_body_support_only_v1'
LINK_CONTRACT = 'native_body_roi_support_pixel_path_v1'
MAX_NATIVE_GAP = 12


@dataclass(frozen=True, slots=True)
class NativeCompositionObservation(NativeGroupObservation):
    def __post_init__(self):
        NativeGroupObservation.__post_init__(self)
        if self.heads or any(p.upper_body_xyc or p.temporal_motion is not None for p in self.people):
            raise ValueError('composition-only native rows cannot invent head/pose observations')
        if tuple(k for k, _ in self.model_content_sha256) != ('nanodet_people',):
            raise ValueError('composition-only rows must name the actual detector call')

    def to_dict(self):
        value = NativeGroupObservation.to_dict(self)
        value.update(contract=RAW_CONTRACT, heads_state='not_requested',
            articulated_pose_state='not_requested',
            body_extent_basis='detector_rectangle_not_anatomical_extent',
            semantic_cast_membership_certified=False)
        return value


def observe_native_composition(session, *, request_source_index, native_frame_index):
    name = 'nanodet_people'
    if name not in session.models or name not in session.model_adapters:
        raise ValueError('composition needs the registered, pinned native person detector')
    pixels = session.native_pixels(request_source_index, native_frame_index)
    clock = session.domain.render_frame_clocks[session.domain.eligible_index(request_source_index)]
    clock_hash = hashlib.sha256(json.dumps(clock.to_dict(), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    def result(reason, people=()):
        return NativeCompositionObservation(request_source_index, native_frame_index,
            (pixels.shape[1], pixels.shape[0]) if pixels is not None else (0, 0),
            clock_hash, _file_identity(session.domain.request_sources[request_source_index]),
            None if pixels is None else hashlib.sha256(memoryview(pixels)).hexdigest(),
            ((name, session.models[name][2]),), tuple(people), (), reason,
            ((name, session.model_adapters[name]),))
    if pixels is None:
        return result('exact_native_decode_unavailable')
    rows = session.observe_native(request_source_index, native_frame_index, model=name)
    if rows is None:
        return result('person_detector_unavailable')
    if len(rows) > 12:
        return result('person_detection_overflow_not_truncated')
    height, width = pixels.shape[:2]
    people = []
    for index, row in enumerate(rows):
        if len(row) != 5:
            return result('invalid_native_body_detector_row')
        left, top, right, bottom, confidence = map(float, row)
        box = left/width, top/height, (right-left)/width, (bottom-top)/height
        if (any(not math.isfinite(v) for v in (*box, confidence))
                or not 0 <= confidence <= 1 or min(box[2:]) <= 0
                or min(box[:2]) < 0 or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
            return result('invalid_source_native_body_roi')
        people.append(ObservedFormationPerson(index, box, (), confidence))
    return result('observed_body_support_only' if people else 'no_people_detected', people)


def body_support_links(sensors, previous, current):
    """Follow each native ROI through ACTUAL intervening native rasters.

    Existing containment LK thresholds are unchanged and used only for ROI
    support, never promoted into arm-joint or endpoint action alignment. Each
    call is an adjacent native-raster step; a three-output-tick endpoint is not
    passed to a nearby-frame helper as a single long-baseline observation.
    """
    if (previous.request_source_index != current.request_source_index
            or previous.source_clock_identity != current.source_clock_identity
            or previous.source_file_identity != current.source_file_identity
            or previous.frame_size_wh != current.frame_size_wh
            or current.native_frame_index <= previous.native_frame_index):
        raise ValueError('composition motion needs distinct chronological same-source native rows')
    key = ('composition_support', previous.sha256, current.sha256)
    if key in sensors.link_cache:
        return sensors.link_cache[key]
    value = dict(contract=LINK_CONTRACT, raw_observation_sha256=[previous.sha256, current.sha256],
        native_frame_indices=[previous.native_frame_index, current.native_frame_index],
        request_source_index=previous.request_source_index,
        required_body_observation_ids=[[p.observation_id for p in r.people] for r in (previous, current)],
        predicted_boxes=[], overlaps=[], assignments=[], native_pixel_sha256=[],
        complete_support_paths=False, articulated_motion_certified=False,
        named_identity_certified=False, semantic_cast_membership_certified=False,
        body_phase_certified=False, quality_approved=False)
    gap = current.native_frame_index-previous.native_frame_index
    if gap > MAX_NATIVE_GAP or min(len(previous.people), len(current.people)) < 2:
        value['reason'] = 'unsupported_native_gap_or_fewer_than_two_body_supports'
        sensors.link_cache[key] = value
        return value
    frames = [sensors.session.native_pixels(previous.request_source_index, n)
        for n in range(previous.native_frame_index, current.native_frame_index+1)]
    if any(p is None for p in frames):
        value['reason'] = 'intermediate_native_pixels_unavailable'
        sensors.link_cache[key] = value
        return value
    value['native_pixel_sha256'] = [hashlib.sha256(memoryview(p)).hexdigest() for p in frames]
    predicted = []
    for person in previous.people:
        box = person.box_xywh
        for a, b in zip(frames, frames[1:]):
            sensors.session.budget.spend('pose_comparisons')
            box = _optical_flow_box(a, b, box)
            if box is None:
                break
        predicted.append(box)
    overlaps = np.asarray([[0. if b is None else _iou(b, p.box_xywh)
        for p in current.people] for b in predicted], dtype=float)
    aa, bb = linear_sum_assignment(1.-overlaps)
    pairs = []
    for a, b in zip(aa, bb):
        score = float(overlaps[a, b])
        rivals = [float(overlaps[a, j]) for j in range(len(current.people)) if j != b]
        reverse = [float(overlaps[i, b]) for i in range(len(previous.people)) if i != a]
        # Same ROI correspondence limits as the existing formation observer.
        if score < .60 or any(score-v < .10 for v in (*rivals, *reverse)):
            continue
        pairs.append([previous.people[a].observation_id, current.people[b].observation_id])
    value.update(predicted_boxes=[None if b is None else list(b) for b in predicted],
        overlaps=overlaps.tolist(), assignments=pairs,
        complete_support_paths=len(pairs) == len(previous.people) == len(current.people),
        reason='measured_roi_support_not_joint_or_identity_evidence')
    sensors.session.budget.spend('pose_evidence_bytes', len(json.dumps(value, sort_keys=True).encode()))
    sensors.link_cache[key] = value
    return value


def validate_raw_composition(native):
    if (native.get('contract') != RAW_CONTRACT or native.get('heads_state') != 'not_requested'
            or native.get('articulated_pose_state') != 'not_requested'
            or native.get('semantic_cast_membership_certified') is not False
            or native.get('heads') != [] or len(native.get('people', ())) < 2
            or any(p.get('upper_body_xyc') or p.get('temporal_motion') is not None
                for p in native.get('people', ()) )):
        raise ValueError('body-only role borrowed face/pose authority or lost its raw cohort')


def validate_support_link(link, previous, current):
    digest = lambda r: hashlib.sha256(json.dumps(r, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    ids = [[p['observation_id'] for p in r['people']] for r in (previous, current)]
    if (not isinstance(link, dict) or link.get('contract') != LINK_CONTRACT
            or link.get('complete_support_paths') is not True
            or link.get('raw_observation_sha256') != [digest(previous), digest(current)]
            or link.get('required_body_observation_ids') != ids
            or link.get('native_frame_indices') != [previous['native_frame_index'], current['native_frame_index']]
            or link.get('request_source_index') != previous['request_source_index']
            or previous['request_source_index'] != current['request_source_index']
            or any(link.get(k) is not False for k in ('articulated_motion_certified',
                'named_identity_certified', 'semantic_cast_membership_certified', 'body_phase_certified', 'quality_approved'))):
        raise ValueError('composition support link has wrong raw pair, scope or authority')
    pairs = link.get('assignments', [])
    if (len(pairs) != len(ids[0]) or len(ids[0]) != len(ids[1])
            or sorted(p[0] for p in pairs) != sorted(ids[0])
            or sorted(p[1] for p in pairs) != sorted(ids[1])):
        raise ValueError('composition support assignment is not one-to-one and complete')
    gap = current['native_frame_index']-previous['native_frame_index']
    hashes = link.get('native_pixel_sha256', [])
    if (not 0 < gap <= MAX_NATIVE_GAP or len(hashes) != gap+1
            or hashes[0] != previous['native_pixel_sha256'] or hashes[-1] != current['native_pixel_sha256']):
        raise ValueError('composition support skipped an actual native raster')
