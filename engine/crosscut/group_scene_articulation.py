"""Observed partial articulation for composition ranking, never actor identity.

Body-only scene rows stay body-only. A separate pinned RTMPose observation
retains every detector ROI and missing pose. Native joint-local pixel paths
associate an observable subset over each three-output-tick endpoint. Missing,
ambiguous and occluded observations remain archived/unmatched; they are not
renamed as absent performers or complete whole-shot tracks.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .group_joint_native_observer import NativeGroupObservation
from .group_joint_person_ownership import OwnedPersonObservation, bind_person_ownership, ownership_policy_fingerprint
from .group_joint_formation import (ObservedFormationTemporalMotion, DeclaredFormationView,
    compare_projected_formation, _bones)
from .group_joint_native_joint_flow import native_joint_pixel_flow_stream, native_joint_assignment_cost
from .group_joint_source_camera import GroupOutputRole

RAW_CONTRACT = 'native_composition_parallel_articulated_observations_v1'
CONTRACT = 'composition_partial_articulated_incident_ranking_v1'
METRICS = ('composition_pose_shape', 'composition_pose_velocity')
MAX_NATIVE_GAP = 12


@dataclass(frozen=True, slots=True)
class NativeCompositionPoseObservation(NativeGroupObservation):
    def __post_init__(self):
        NativeGroupObservation.__post_init__(self)
        if (self.heads or tuple(name for name,_ in self.model_content_sha256) != ('nanodet_people','rtmpose_body7')
                or any(not isinstance(p,OwnedPersonObservation) or p.unique_head_observation_id is not None
                    or p.face_anchor_consistent is True for p in self.people)):
            raise ValueError('parallel composition pose cannot borrow a focal face/identity detector')

    def to_dict(self):
        value = NativeGroupObservation.to_dict(self)
        value.update(contract=RAW_CONTRACT, heads_state='not_requested',
            semantic_cast_membership_certified=False, named_member_identity_certified=False)
        return value


def observe_composition_pose(session, body_support):
    """Actual registered pose calls; no face/global-focal authority is borrowed."""
    name = 'rtmpose_body7'
    if name not in session.models or name not in session.model_adapters:
        return None
    people = []
    for person in body_support.people:
        body = session.observe_native(body_support.request_source_index, body_support.native_frame_index,
            model=name, arguments={'bbox_xywh': list(person.box_xywh)})
        observed = body is not None and body.reason == 'observed'
        people.append(OwnedPersonObservation(person.observation_id, person.box_xywh,
            body.upper_body_xyc if observed else (), person.confidence,
            pose_crop_xywh=None if body is None else body.pose_crop_xywh,
            source_pixel_to_pose_input_affine=() if body is None else body.source_pixel_to_pose_input_affine,
            raw_pose_xyc=() if body is None else body.raw_keypoints_xyc,
            pose_reason='pose_model_unavailable' if body is None else body.reason,
            face_anchor_consistent=None if body is None else body.face_anchor_consistent))
    people, _ = bind_person_ownership(tuple(people), (), body_support.frame_size_wh)
    return NativeCompositionPoseObservation(body_support.request_source_index, body_support.native_frame_index,
        body_support.frame_size_wh, body_support.source_clock_identity, body_support.source_file_identity,
        body_support.native_pixel_sha256,
        (*body_support.model_content_sha256, (name, session.models[name][2])), people, (),
        'parallel_pose_observed_no_focal_identity' if people else body_support.reason,
        (*body_support.model_adapter_fingerprints, (name, session.model_adapters[name])),
        ownership_policy_fingerprint(), raw_person_detector_json=body_support.raw_person_detector_json)


def _raw(sensors, state, tick):
    if state.role != GroupOutputRole.COMPOSITION:
        return sensors.raw(state, tick, sensors.hypothesis(state)['analysis_roll'])
    base = sensors.raw(state, tick, None)
    key = base.sha256
    if key not in sensors.articulation_raw_rows:
        row = None
        if ('rtmpose_body7' in sensors.session.models
                and 'rtmpose_body7' in sensors.session.model_adapters):
            row = sensors.session.budget.normalized_native_views.measure(
                label='native_composition_parallel_pose_view',
                input_binding=dict(request_source_index=base.request_source_index,
                    native_frame_index=base.native_frame_index,source_clock_identity=base.source_clock_identity,
                    body_support_sha256=base.sha256),
                producer=lambda:observe_composition_pose(sensors.session,base))
        sensors.articulation_raw_rows[key] = row
    return sensors.articulation_raw_rows[key]


def _project(person, state, tick):
    from .group_source_changing_owner import project_box, project_point
    box, _ = project_box(person.box_xywh, state, tick)
    points = tuple((*project_point(p[:2], state, tick), p[2]) for p in person.upper_body_xyc)
    # A projected off-screen joint is missing in this output, not clipped into
    # an observed position. Its complete original pose remains in the raw row.
    points = tuple((x, y, c if 0 <= x < 1 and 0 <= y < 1 else 0.) for x, y, c in points)
    return replace(person, box_xywh=tuple(box), upper_body_xyc=points)


def endpoint_articulation(sensors, state, *, outgoing):
    if type(outgoing) is not bool:
        raise ValueError('articulation endpoint side must be explicit')
    ticks = (state.end_output_frame-4, state.end_output_frame-1) if outgoing else (
        state.first_output_frame, state.first_output_frame+3)
    key = state.key, outgoing
    if key in sensors.articulation_endpoint_cache:
        return sensors.articulation_endpoint_cache[key]
    raw = tuple(_raw(sensors, state, tick) for tick in ticks)
    evidence = dict(contract=CONTRACT, state_key=state.key, output_ticks=list(ticks),
        raw_observations=[None if row is None else row.to_dict() for row in raw],
        raw_observation_sha256=[None if row is None else row.sha256 for row in raw],
        native_pixel_sha256=[], native_frame_indices=[], native_steps_complete=False,
        joint_pixel_flows=[], assignments=[], projected_observed_subset=[],
        unmatched_observation_ids=[[], []], complete_semantic_cohort=False,
        whole_shot_tracking_certified=False, named_identity_certified=False,
        cross_camera_identity_certified=False, body_phase_certified=False, quality_approved=False)
    pixel_hashes = {}
    def finish(view, reason):
        evidence.update(native_frame_indices=sorted(pixel_hashes),
            native_pixel_sha256=[pixel_hashes[n] for n in sorted(pixel_hashes)])
        result = view, json.loads(json.dumps(dict(evidence, reason=reason), allow_nan=False))
        sensors.articulation_endpoint_cache[key] = result
        return result
    if any(row is None for row in raw):
        return finish(None, 'registered_pose_observation_unavailable')
    a, b = raw
    evidence['unmatched_observation_ids'] = [[p.observation_id for p in row.people] for row in raw]
    gap = b.native_frame_index-a.native_frame_index
    if (a.request_source_index != b.request_source_index or a.source_file_identity != b.source_file_identity
            or a.source_clock_identity != b.source_clock_identity or a.frame_size_wh != b.frame_size_wh
            or not 0 < gap <= MAX_NATIVE_GAP or min(len(a.people), len(b.people)) < 2):
        return finish(None, 'source_local_articulated_endpoint_unavailable')
    def pixels(native):
        image = sensors.session.native_pixels(a.request_source_index, native)
        if image is None:
            raise ValueError('articulation requires every intervening actual native raster')
        pixel_hashes[native] = sha256(memoryview(image)).hexdigest()
        return image
    first, last = pixels(a.native_frame_index), pixels(b.native_frame_index)
    if [pixel_hashes[a.native_frame_index], pixel_hashes[b.native_frame_index]] != [a.native_pixel_sha256, b.native_pixel_sha256]:
        raise ValueError('articulation raw observation pixels changed')
    # Stream intermediate original rasters. Two endpoints remain live; the
    # native session owns its bounded LRU, not a local list of thirteen 4K frames.
    flows = native_joint_pixel_flow_stream(first, last, a.people,
        intermediate_pixels=(pixels(n) for n in range(a.native_frame_index+1, b.native_frame_index)))
    evidence['joint_pixel_flows'] = [None if flow is None else flow.to_dict() for flow in flows]
    evidence['native_steps_complete'] = len(pixel_hashes) == gap+1
    costs = [[native_joint_assignment_cost(flow, person, b.frame_size_wh) for person in b.people]
        for flow in flows]
    matrix = np.asarray([[1e6 if value is None else value for value in row] for row in costs])
    sensors.session.budget.spend('pose_comparisons', int(matrix.size))
    pairs = []
    aa, bb = linear_sum_assignment(matrix)
    for i, j in zip(aa, bb):
        cost = costs[i][j]
        if cost is None or cost > .12:
            continue
        rivals = [v for k, v in enumerate(costs[i]) if k != j and v is not None]
        reverse = [row[j] for k, row in enumerate(costs) if k != i and row[j] is not None]
        if any(v-cost < .025 for v in (*rivals, *reverse)):
            continue
        x, y = _project(a.people[i], state, ticks[0]), _project(b.people[j], state, ticks[1])
        bones_a, bones_b = _bones(x, state.output_size), _bones(y, state.output_size)
        tracked = {row[0] for row in flows[i].joints}
        joints = {'left_upper_arm': {0,2}, 'right_upper_arm': {1,3},
            'left_forearm': {2,4}, 'right_forearm': {3,5}}
        common = tuple(name for name in bones_a if name in bones_b and joints[name] <= tracked)
        if len(common) < 2:
            continue
        velocity = tuple((name, *map(float, (bones_b[name]-bones_a[name])/(3/state.output_fps)))
            for name in common)
        motion = ObservedFormationTemporalMotion(state.key, ticks,
            (a.native_frame_index, b.native_frame_index), state.output_fps,
            (x.observation_id, y.observation_id), velocity, True)
        endpoint = y if outgoing else x
        pairs.append((int(i), int(j), replace(endpoint, temporal_motion=motion)))
    evidence.update(costs=costs, assignments=[[a.people[i].observation_id, b.people[j].observation_id]
        for i, j, _ in pairs], unmatched_observation_ids=[
            [p.observation_id for i, p in enumerate(a.people) if i not in {v[0] for v in pairs}],
            [p.observation_id for j, p in enumerate(b.people) if j not in {v[1] for v in pairs}]])
    if len(pairs) < 2:
        return finish(None, 'fewer_than_two_observable_joint_pixel_paths')
    # Only the observed subset is a correspondence hypothesis. The raw/context
    # inventory above is not erased and is not called an absent semantic cast.
    people = tuple(p for _, _, p in pairs)
    evidence['projected_observed_subset'] = [asdict(p) for p in people]
    view = DeclaredFormationView(people, tuple(p.observation_id for p in people), state.output_size)
    return finish(view, 'observed_partial_native_joint_paths_not_complete_cast')


def compare_composition_articulation(sensors, left, right):
    outgoing, a = endpoint_articulation(sensors, left, outgoing=True)
    incoming, b = endpoint_articulation(sensors, right, outgoing=False)
    result = dict(contract=CONTRACT, endpoints=[a,b], available=False, metrics=None,
        correspondence=None, named_identity_certified=False, cross_camera_identity_certified=False,
        body_phase_certified=False, complete_semantic_cohort=False, quality_approved=False)
    if outgoing is None or incoming is None:
        return dict(result, reason='partial_articulation_unavailable')
    match = compare_projected_formation(outgoing, incoming, use_temporal_correspondence=True)
    result['correspondence'] = match.to_dict()
    if not match.observed:
        return dict(result, reason=match.status)
    aa = {p.observation_id:p for p in outgoing.people}
    bb = {p.observation_id:p for p in incoming.people}
    velocities = []
    for (i,j), common in zip(match.pairs, match.common_bones, strict=True):
        va = {name:np.asarray((x,y)) for name,x,y in aa[i].temporal_motion.bone_velocities}
        vb = {name:np.asarray((x,y)) for name,x,y in bb[j].temporal_motion.bone_velocities}
        for name in common:
            if name in va and name in vb:
                velocities.append(min(1., float(np.linalg.norm(va[name]-vb[name]))/
                    max(1., math.sqrt(float(np.dot(va[name],va[name])+np.dot(vb[name],vb[name]))))))
    if not velocities:
        return dict(result, reason='matched_articulated_velocity_unavailable')
    return dict(result, available=True, metrics=dict(composition_pose_shape=match.pose_mismatch,
        composition_pose_velocity=float(np.mean(velocities))), reason='observed_unlabeled_partial_articulation')


def validate_articulation_evidence(value, edge_metrics, native_assets):
    """Structure/authority binding only; final original-pixel replay is mandatory."""
    from .group_performer_family_owner import digest
    from .group_native_detector_result import validate_serialized_detector_result
    if (not isinstance(value, dict) or value.get('contract') != CONTRACT
            or type(value.get('available')) is not bool or not isinstance(edge_metrics, dict)
            or any(value.get(k) is not False for k in ('named_identity_certified',
                'cross_camera_identity_certified', 'body_phase_certified',
                'complete_semantic_cohort', 'quality_approved'))):
        raise ValueError('partial composition articulation cannot acquire identity/cohort/quality authority')
    digest(value)
    endpoints = value.get('endpoints')
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError('partial articulation requires both endpoint observations')
    for endpoint in endpoints:
        if (not isinstance(endpoint, dict) or endpoint.get('contract') != CONTRACT
                or any(endpoint.get(k) is not False for k in ('complete_semantic_cohort',
                    'whole_shot_tracking_certified', 'named_identity_certified',
                    'cross_camera_identity_certified', 'body_phase_certified', 'quality_approved'))):
            raise ValueError('partial articulation endpoint has unauthorized authority')
        ticks = endpoint.get('output_ticks')
        raw = endpoint.get('raw_observations')
        if (not isinstance(ticks, list) or len(ticks) != 2 or any(type(t) is not int or t < 0 for t in ticks)
                or ticks[1]-ticks[0] != 3 or not isinstance(raw, list) or len(raw) != 2
                or endpoint.get('raw_observation_sha256') != [None if r is None else digest(r) for r in raw]):
            raise ValueError('partial articulation raw pair/output clock changed')
        if not value['available']:
            continue
        if any(not isinstance(r, dict) for r in raw):
            raise ValueError('observed articulation cannot use unknown raw rows')
        a, b = raw
        first, end = a['native_frame_index'], b['native_frame_index']
        if (any(type(t) is not int for t in (first,end)) or not 0 < end-first <= MAX_NATIVE_GAP
                or a['request_source_index'] != b['request_source_index']
                or a['source_file_identity'] != b['source_file_identity']
                or a['source_clock_identity'] != b['source_clock_identity']
                or endpoint.get('native_steps_complete') is not True
                or endpoint.get('native_frame_indices') != list(range(first,end+1))):
            raise ValueError('observed articulation skipped actual same-source native steps')
        hashes = endpoint.get('native_pixel_sha256')
        if (not isinstance(hashes, list) or len(hashes) != end-first+1
                or hashes[0] != a['native_pixel_sha256'] or hashes[-1] != b['native_pixel_sha256']):
            raise ValueError('observed articulation native raster hashes differ')
        for row in raw:
            validate_serialized_detector_result(row)
            models, adapters = dict(row['model_content_sha256']), dict(row['model_adapter_fingerprints'])
            if not {'nanodet_people','rtmpose_body7'} <= set(models) or set(models) != set(adapters):
                raise ValueError('articulation did not retain every actual model/adapter identity')
            for name, content in models.items():
                asset = ('person_model_path' if name == 'nanodet_people' else
                    'pose_model_path' if name == 'rtmpose_body7' else
                    'head_model_path' if name.startswith('yunet_') else None)
                if asset is None or not isinstance(native_assets, dict) or content != native_assets.get(asset,{}).get('sha256'):
                    raise ValueError('articulation model differs from frozen native assets')
        ids = [{p['observation_id'] for p in r['people']} for r in raw]
        pairs = endpoint.get('assignments')
        subset = endpoint.get('projected_observed_subset')
        if (not isinstance(pairs, list) or len(pairs) < 2 or any(not isinstance(p,list) or len(p)!=2
                or any(type(i) is not int for i in p) or p[0] not in ids[0] or p[1] not in ids[1] for p in pairs)
                or len({p[0] for p in pairs}) != len(pairs) or len({p[1] for p in pairs}) != len(pairs)
                or not isinstance(subset, list) or len(subset) != len(pairs)):
            raise ValueError('articulation observable subset has duplicate/missing native assignments')
        for person, pair in zip(subset, pairs, strict=True):
            motion = person.get('temporal_motion') or {}
            if (motion.get('same_source_pixel_path_verified') is not True
                    or motion.get('state_key') != endpoint['state_key']
                    or motion.get('output_frames') != ticks
                    or motion.get('native_frames') != [first,end]
                    or motion.get('observation_ids') != pair
                    or person.get('observation_id') not in pair):
                raise ValueError('articulation subset motion does not own its source-local pair')
    if value['available']:
        metrics = value.get('metrics')
        if (not isinstance(metrics, dict) or set(metrics) != set(METRICS)
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1 for v in metrics.values())
                or any(edge_metrics.get(k) != metrics[k] for k in METRICS)
                or (value.get('correspondence') or {}).get('status') != 'observed_partial_correspondence'):
            raise ValueError('observed articulation metrics/correspondence changed')
    elif value.get('metrics') is not None or any(k in edge_metrics for k in METRICS):
        raise ValueError('unknown articulation cannot become a measured zero score')
