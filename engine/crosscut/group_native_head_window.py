"""Freeze one native head detector variant BEFORE source/camera comparison.

An upright background detection must not suppress a foreground tilted head.
All five predeclared sensor variants see the SAME original window. Variants
remain distinct evidence; their boxes are never concatenated into people or
expanded into a physical-head extent. The selected window is only a replayable
discovery policy, not member identity, body ownership or motion admission.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .group_focal_attention import focal_landmarks_track
from .boundary_subject_alignment import _optical_flow_box
from .cut_anchored_containment import _iou
from .group_joint_native_observer import NativeGroupHead
from .group_native_head_orientation import (
    SUPPORTED_ANALYSIS_ROLLS, OrientedNativeHeadAlternative, native_head_analysis_spec)
from .roll_aware_face_geometry import same_measured_face
from .visual_score import _DetectedFace, _working_frame


ANALYSIS_VARIANTS = (None, *SUPPORTED_ANALYSIS_ROLLS)
MAXIMUM_WINDOW_EVIDENCE_BYTES = 2 * 1024 * 1024


def _policy_sources():
    names=('group_native_head_window.py', 'group_focal_attention.py',
        'roll_aware_face_geometry.py', 'boundary_subject_alignment.py',
        'cut_anchored_containment.py', 'visual_score.py')
    root=Path(__file__).parent
    return tuple((name,hashlib.sha256((root/name).read_bytes()).hexdigest()) for name in names)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class NativeHeadWindowPolicy:
    request_source_index: int
    native_frame_indices: tuple[int, int, int]
    selected_analysis_roll_degrees: float | None
    selected: bool
    reason: str
    variants: tuple[dict, ...]
    source_clock_sha256: str
    source_file_identity: tuple
    pixel_sha256: tuple[str, ...]
    distinct_trajectory_classes: tuple[tuple[int, ...], ...]
    source_code_sha256: tuple[tuple[str, str], ...]

    def to_dict(self):
        return dict(contract='pre_camera_native_head_detector_window_policy_v1',
            **asdict(self), variant_policy_declared_before_pixels=True,
            selected_before_donor_camera_scores=True, variants_merged_as_people=False,
            original_source_pixels_or_clock_modified=False,
            dominance_basis='original_pixel_eye_span_squared_not_rotated_box_envelope',
            same_source_tracking_basis='existing_native_landmark_or_box_pixel_flow_policy_no_threshold_change',
            eye_area_ranking_requires_landmark_path_not_box_path=True,
            foreground_attention_is_not_semantic_identity=True,
            body_ownership_verified=False, full_head_extent_verified=False,
            temporal_phase_certified=False, candidate_admitted=False,
            quality_approved=False)

    @property
    def sha256(self):
        return _sha(self.to_dict())


def _valid(head, width, height):
    points = np.asarray(head.landmarks_xy, dtype=float)
    return (points.shape == (5, 2) and np.isfinite(points).all()
        and bool(((points >= 0) & (points <= 1)).all())
        and np.linalg.norm((points[0]-points[1])*[width, height]) >= 2.)


def _power(head, width, height):
    points = np.asarray(head.landmarks_xy, dtype=float)*[width, height]
    return float(np.linalg.norm(points[0]-points[1])**2 / height**2)


def _foreground(heads, width, height):
    # The same established 1.8 area margin, applied to source-pixel eye-area
    # rather than rotated face-box envelopes. This orders discovery only.
    ordered = sorted((head for head in heads if _valid(head, width, height)),
        key=lambda head: (-_power(head, width, height), head.observation_id))
    if not ordered or len(ordered) > 1 and _power(ordered[0], width, height) < \
            1.8*_power(ordered[1], width, height):
        return None
    return ordered[0]


def _same_window(first, second, sizes):
    # This suppresses duplicate detector hypotheses, not physical subjects.
    # Keep the original landmarks/boxes of ONE retained variant unchanged.
    for left, right, (width, height) in zip(first, second, sizes, strict=True):
        def pixels(head):
            x, y, w, h = head.box_xywh
            return _DetectedFace((x*width, y*height, w*width, h*height),
                tuple((a*width, b*height) for a, b in head.landmarks_xy), head.confidence)
        if not same_measured_face(pixels(left), pixels(right)):
            return False
    return True


def observe_native_head_window_policy(session, *, request_source_index,
        native_frame_indices, analysis_variants=ANALYSIS_VARIANTS):
    """Three native rasters, counted registered models, no camera score input.

    Missing or competing foreground tracks leave selection unavailable, while
    every raw variant stays in the result. No source is removed from eligibility.
    This function never derives arm/body duties from a face detector success.
    """
    variants = tuple(analysis_variants)
    indices = tuple(native_frame_indices)
    if (not isinstance(analysis_variants, tuple) or not 1 <= len(variants) <= 5
            or len(set(variants)) != len(variants)
            or any(v is not None and (type(v) not in (int, float)
                or not math.isfinite(v) or v not in SUPPORTED_ANALYSIS_ROLLS) for v in variants)
            or not isinstance(native_frame_indices, tuple) or len(indices) != 3
            or any(type(n) is not int or n < 0 for n in indices)
            or not indices[0] <= indices[1] <= indices[2]):
        raise ValueError('native head window requires three ordered native IDs and a frozen bounded variant set')
    local = session.domain.eligible_index(request_source_index)
    clock = session.domain.render_frame_clocks[local]
    if indices[-1] >= clock.frame_count:
        raise ValueError('native head window is outside its verified source clock')
    identity = session._assert_source_identity(request_source_index)
    specs = tuple(native_head_analysis_spec(v) for v in variants)
    for model, _ in specs:
        if model not in session.models or model not in session.model_adapters:
            raise ValueError('all native head window sensor variants must be registered before observation')
        session._assert_model_assets(model)
    clock_sha = _sha(clock.to_dict())
    code_sha = _policy_sources()
    if len(set(indices)) != 3:
        return NativeHeadWindowPolicy(request_source_index, indices, None, False,
            'unique_native_window_samples_unavailable', (), clock_sha, identity, (), (), code_sha)
    cuts = session.domain.scene_cut_times[local]
    if any(indices[0] < math.ceil(time*float(clock.frame_rate)-1e-9) <= indices[-1]
            for time in cuts):
        return NativeHeadWindowPolicy(request_source_index, indices, None, False,
            'window_crosses_actual_native_camera_cut', (), clock_sha, identity, (), (), code_sha)
    records = [dict(analysis_roll_degrees=angle, model_name=model,
        model_sha256=session.models[model][2], adapter_sha256=session.model_adapters[model],
        heads_by_native_frame=[], unavailable_native_frame_indices=[],
        foreground_head_ids=[], measured_eye_area=[], adjacent_pixel_links=[], adjacent_link_basis=[],
        stable_foreground_available=False)
        for angle, (model, _) in zip(variants, specs, strict=True)]
    histories = [[] for _ in variants]
    previous_working = None
    sizes = []
    pixels_sha = []
    for native in indices:
        session.budget.check()
        pixels = session.native_pixels(request_source_index, native)
        pixels_sha.append(None if pixels is None else hashlib.sha256(memoryview(pixels)).hexdigest())
        if pixels is None:
            for record, history in zip(records, histories, strict=True):
                record['heads_by_native_frame'].append([])
                record['unavailable_native_frame_indices'].append(native)
                history.append(None)
            previous_working = None
            sizes.append((0, 0))
            continue
        height, width = pixels.shape[:2]
        sizes.append((width, height))
        working = _working_frame(pixels, maximum_width=960)
        for index, ((model, id_base), angle) in enumerate(zip(specs, variants, strict=True)):
            observed = session.observe_native(request_source_index, native, model=model)
            heads = ()
            if observed is None:
                records[index]['unavailable_native_frame_indices'].append(native)
            elif angle is None:
                if (not isinstance(observed, tuple) or len(observed) > 24
                        or any(not isinstance(h, NativeGroupHead) for h in observed)):
                    raise ValueError('upright window observer returned unbound head variants')
                heads = observed
            else:
                if (not isinstance(observed, tuple) or len(observed) > 24
                        or any(not isinstance(h, OrientedNativeHeadAlternative)
                            or h.model_name != model or h.analysis_roll_degrees != angle
                            or not 0 <= h.head.observation_id < 24 for h in observed)):
                    raise ValueError('oriented window observer differs from its frozen native variant')
                heads = tuple(replace(h.head, observation_id=id_base+h.head.observation_id) for h in observed)
            records[index]['heads_by_native_frame'].append([asdict(h) for h in heads])
            head = _foreground(heads, width, height)
            prior = histories[index][-1] if histories[index] else None
            if histories[index]:
                basis = None
                if (prior is not None and head is not None and previous_working is not None
                        and previous_working.shape == working.shape):
                    if focal_landmarks_track(previous_working, working, prior.landmarks_xy, head.landmarks_xy):
                        basis = 'native_face_landmark_path_not_whole_head_or_part_ownership'
                    else:
                        box = _optical_flow_box(previous_working, working, prior.box_xywh)
                        if box is not None and _iou(box, head.box_xywh) >= .60:
                            basis = 'native_face_box_path_not_whole_head_or_part_ownership'
                records[index]['adjacent_pixel_links'].append(basis is not None)
                records[index]['adjacent_link_basis'].append(basis)
            histories[index].append(head)
            records[index]['foreground_head_ids'].append(None if head is None else head.observation_id)
            records[index]['measured_eye_area'].append(None if head is None else _power(head, width, height))
        # The only owned extra rasters are two <=960px working views; release
        # the old native cache rather than retain a three-frame 4K window.
        previous_working = working.copy()
        session.retain_native_pixels(((request_source_index, native),))
        session.budget.check()
    if session._assert_source_identity(request_source_index) != identity:
        raise ValueError('native source changed during head window observation')
    if _policy_sources() != code_sha:
        raise ValueError('native head window selection implementation changed during observation')
    feasible = []
    for index, record in enumerate(records):
        record['stable_foreground_available'] = bool(len(histories[index]) == 3
            and all(h is not None for h in histories[index])
            and len(record['adjacent_pixel_links']) == 2
            and all(record['adjacent_pixel_links'])
            and not record['unavailable_native_frame_indices'])
        # A box path observes a face ROI moving, NOT the detector's regressed
        # eyes/nose/mouth. Inverse-rotated variants can disagree about those
        # landmarks while tracking the same box. Do not rank eye-area with a
        # box-only receipt or let such predictions erase a measured landmark
        # window. Keep the raw box alternative explicit for a later ROI owner.
        record['landmark_geometry_window_available'] = bool(
            record['stable_foreground_available'] and all(basis ==
                'native_face_landmark_path_not_whole_head_or_part_ownership'
                for basis in record['adjacent_link_basis']))
        record['box_only_geometry_is_not_eye_area_ranking_evidence'] = bool(
            record['stable_foreground_available'] and not record['landmark_geometry_window_available'])
        if record['landmark_geometry_window_available']:
            feasible.append(index)
    classes = []
    for index in feasible:
        # Require agreement with EVERY member, not a transitive A~B~C merge
        # that can fuse two distinct performers through a noisy middle head.
        existing = next((group for group in classes if all(_same_window(
            histories[index], histories[other], sizes) for other in group)), None)
        if existing is None:
            classes.append([index])
        else:
            existing.append(index)
    ranked = sorted(classes, key=lambda group: -max(min(records[i]['measured_eye_area']) for i in group))
    selected_index = None
    reason = 'no_stable_native_foreground_head_window'
    if ranked:
        power = lambda group: max(min(records[i]['measured_eye_area']) for i in group)
        if len(ranked) > 1 and power(ranked[0]) < 1.8*power(ranked[1]):
            reason = 'competing_native_foreground_trajectories_remain_ambiguous'
        else:
            # Prefer upright if it observes the SAME winning head; otherwise
            # preserve the earliest frozen variant, never inflate its box.
            selected_index = min(ranked[0])
            reason = 'native_eye_area_and_two_measured_pixel_links_ranked_window'
    if len(json.dumps(records, separators=(',', ':'), allow_nan=False).encode()) > MAXIMUM_WINDOW_EVIDENCE_BYTES:
        raise ValueError('native head window evidence exceeded its explicit byte budget')
    return NativeHeadWindowPolicy(request_source_index, indices,
        None if selected_index is None else variants[selected_index], selected_index is not None,
        reason, tuple(records), clock_sha, identity, tuple(pixels_sha),
        tuple(tuple(group) for group in classes), code_sha)
