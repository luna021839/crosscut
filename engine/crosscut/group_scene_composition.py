"""Head-independent native body-support/composition observations.

Detector rectangles and their measured pixel support are NOT articulated body
motion, named performers, cast membership, anatomical extent or phase. The
raw cohort is every bounded native detector rectangle, retained before camera
fitting. The v2 support link uses only uniquely pixel-tracked pairs; unmatched
births/deaths/occlusion remain explicit and are not negative cast evidence.
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
LINK_CONTRACT = 'native_body_roi_support_pixel_path_v2'
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
    identity = _file_identity(session.domain.request_sources[request_source_index])
    raw_detector_json = None
    def result(reason, people=()):
        return NativeCompositionObservation(request_source_index, native_frame_index,
            (pixels.shape[1], pixels.shape[0]) if pixels is not None else (0, 0),
            clock_hash, identity,
            None if pixels is None else hashlib.sha256(memoryview(pixels)).hexdigest(),
            ((name, session.models[name][2]),), tuple(people), (), reason,
            ((name, session.model_adapters[name]),), raw_person_detector_json=raw_detector_json)
    if pixels is None:
        return result('exact_native_decode_unavailable')
    rows = session.observe_native(request_source_index, native_frame_index, model=name)
    from .group_native_detector_result import capture_detector_result, bounded_detector_rows
    record = capture_detector_result(rows, binding=dict(request_source_index=request_source_index,
        native_frame_index=native_frame_index, source_clock_identity=clock_hash,
        source_file_identity=list(identity), native_pixel_sha256=hashlib.sha256(memoryview(pixels)).hexdigest()),
        model_content_sha256=session.models[name][2], model_adapter_sha256=session.model_adapters[name])
    raw_detector_json = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if rows is None:
        return result('person_detector_unavailable')
    if record['row_count'] is not None and record['row_count'] > 12:
        return result('person_detection_overflow_not_truncated')
    try:
        rows = bounded_detector_rows(record)
    except ValueError:
        return result('invalid_or_unretained_native_body_detector_result')
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
    key = (LINK_CONTRACT, previous.sha256, current.sha256)
    if key in sensors.link_cache:
        return sensors.link_cache[key]
    value = dict(contract=LINK_CONTRACT, raw_observation_sha256=[previous.sha256, current.sha256],
        native_frame_indices=[previous.native_frame_index, current.native_frame_index],
        request_source_index=previous.request_source_index,
        required_body_observation_ids=[[p.observation_id for p in r.people] for r in (previous, current)],
        support_scope='observed_native_roi_subset_not_semantic_cast',
        observed_body_observation_ids=[[], []],
        unmatched_body_observation_ids=[[p.observation_id for p in r.people] for r in (previous, current)],
        predicted_boxes=[], overlaps=[], assignments=[], native_pixel_sha256=[],
        observed_support_paths=False, complete_support_paths=False, articulated_motion_certified=False,
        named_identity_certified=False, semantic_cast_membership_certified=False,
        body_phase_certified=False, quality_approved=False)
    gap = current.native_frame_index-previous.native_frame_index
    if gap > MAX_NATIVE_GAP or min(len(previous.people), len(current.people)) < 2:
        value['reason'] = 'unsupported_native_gap_or_fewer_than_two_body_supports'
        sensors.link_cache[key] = value
        return value
    source = previous.request_source_index
    previous_pixels = sensors.session.native_pixels(source, previous.native_frame_index)
    if previous_pixels is None:
        value['reason'] = 'intermediate_native_pixels_unavailable'
        sensors.link_cache[key] = value
        return value
    hashes = [hashlib.sha256(memoryview(previous_pixels)).hexdigest()]
    predicted = [person.box_xywh for person in previous.people]
    # Stream one adjacent native pair for ALL still-live ROI paths. Keeping
    # the entire gap in a local list defeated the session's byte LRU: thirteen
    # 4K rasters can remain alive even after its cache evicts them. The same
    # frames, hashes, per-person sequence and LK/assignment thresholds remain;
    # only loop order and these local raster lifetimes change. Do not flush the
    # shared cache or immutable model evidence to achieve the local bound.
    for native in range(previous.native_frame_index+1, current.native_frame_index+1):
        current_pixels = sensors.session.native_pixels(source, native)
        if current_pixels is None:
            value['reason'] = 'intermediate_native_pixels_unavailable'
            sensors.link_cache[key] = value
            return value
        hashes.append(hashlib.sha256(memoryview(current_pixels)).hexdigest())
        for index, box in enumerate(predicted):
            if box is None:
                continue
            sensors.session.budget.spend('pose_comparisons')
            predicted[index] = _optical_flow_box(previous_pixels, current_pixels, box)
        previous_pixels = current_pixels
    value['native_pixel_sha256'] = hashes
    if hashes[0] != previous.native_pixel_sha256 or hashes[-1] != current.native_pixel_sha256:
        value['reason'] = 'native_support_pixels_differ_from_raw_observation'
        sensors.link_cache[key] = value
        return value
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
    matched = [[pair[side] for pair in pairs] for side in (0, 1)]
    value.update(predicted_boxes=[None if b is None else list(b) for b in predicted],
        overlaps=overlaps.tolist(), assignments=pairs,
        observed_body_observation_ids=matched,
        unmatched_body_observation_ids=[[p.observation_id for p in row.people
            if p.observation_id not in matched[side]] for side, row in enumerate((previous, current))],
        observed_support_paths=len(pairs) >= 2,
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
            or link.get('observed_support_paths') is not True
            or type(link.get('complete_support_paths')) is not bool
            or link.get('support_scope') != 'observed_native_roi_subset_not_semantic_cast'
            or link.get('raw_observation_sha256') != [digest(previous), digest(current)]
            or link.get('required_body_observation_ids') != ids
            or link.get('native_frame_indices') != [previous['native_frame_index'], current['native_frame_index']]
            or link.get('request_source_index') != previous['request_source_index']
            or previous['request_source_index'] != current['request_source_index']
            or any(previous.get(k) != current.get(k) for k in (
                'source_clock_identity', 'source_file_identity', 'frame_size_wh'))
            or any(link.get(k) is not False for k in ('articulated_motion_certified',
                'named_identity_certified', 'semantic_cast_membership_certified', 'body_phase_certified', 'quality_approved'))):
        raise ValueError('composition support link has wrong raw pair, scope or authority')
    pairs = link.get('assignments', [])
    if (not isinstance(pairs, list) or len(pairs) < 2
            or any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p) for p in pairs)
            or any(len(set(side)) != len(side) for side in ids)):
        raise ValueError('composition support needs at least two unique observed ROI pairs')
    matched = [[p[side] for p in pairs] for side in (0, 1)]
    if (any(len(set(side)) != len(side) for side in matched)
            or any(not set(matched[n]) <= set(ids[n]) for n in (0, 1))
            or link.get('observed_body_observation_ids') != matched
            or link.get('unmatched_body_observation_ids') != [
                [v for v in ids[n] if v not in matched[n]] for n in (0, 1)]
            or link['complete_support_paths'] != (len(pairs) == len(ids[0]) == len(ids[1]))):
        raise ValueError('composition support assignment lost raw/unmatched IDs or forged completeness')
    overlaps = link.get('overlaps')
    predicted = link.get('predicted_boxes')
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    if (not isinstance(overlaps, list) or len(overlaps) != len(ids[0])
            or any(not isinstance(row, list) or len(row) != len(ids[1])
                or any(not finite(v) or not 0 <= v <= 1 for v in row) for row in overlaps)
            or not isinstance(predicted, list) or len(predicted) != len(ids[0])
            or any(box is not None and (not isinstance(box, list) or len(box) != 4
                or any(not finite(v) for v in box) or min(box[2:]) <= 0) for box in predicted)):
        raise ValueError('composition support geometry does not retain all native ROI rows')
    for before, after in pairs:
        a, b = ids[0].index(before), ids[1].index(after)
        score = overlaps[a][b]
        if (predicted[a] is None or score < .60
                or abs(score-_iou(predicted[a], current['people'][b]['box_xywh'])) > 1e-9
                or any(score-v < .10 for j, v in enumerate(overlaps[a]) if j != b)
                or any(score-row[b] < .10 for i, row in enumerate(overlaps) if i != a)):
            raise ValueError('composition support pair is not uniquely measured at the unchanged ROI limits')
    gap = current['native_frame_index']-previous['native_frame_index']
    hashes = link.get('native_pixel_sha256', [])
    if (not 0 < gap <= MAX_NATIVE_GAP or len(hashes) != gap+1
            or hashes[0] != previous['native_pixel_sha256'] or hashes[-1] != current['native_pixel_sha256']):
        raise ValueError('composition support skipped an actual native raster')
