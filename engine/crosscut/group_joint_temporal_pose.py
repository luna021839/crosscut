"""Raw-17 temporal measurements independent of LK feature survival.

These are source-local association HYPOTHESES, not persistent/named identity,
occlusion, surface motion, semantic phase or production admission. All owner
alternatives and missing duties survive. Native seconds and actual output
seconds are separate bases; a repeated native raster is not another pose.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
from itertools import product
import json
import math

import numpy as np

from .group_joint_native_observer import NativeGroupObservation
from .group_joint_person_ownership import (OwnedPersonObservation, ownership_policy_fingerprint,
    bind_person_ownership)
from .group_joint_source_camera import GroupSourceCameraState
from .group_source_domain import GroupSourceDomain


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


class TemporalPoseUnavailable(ValueError):
    """Insufficient sampling/cut/bounded work, distinct from corrupt binding."""


class PoseJointObservationState(str, Enum):
    VISIBLE_SUPPORTED = 'visible_supported'
    SOURCE_OCCLUDED = 'source_occluded'
    OUT_OF_SOURCE_FRAME = 'out_of_source_frame'
    DETECTOR_UNKNOWN = 'detector_unknown'
    OWNERSHIP_AMBIGUOUS = 'ownership_ambiguous'
    PREDICTED_ONLY = 'predicted_only'


@dataclass(frozen=True, slots=True)
class FrozenPoseMotionDutiesV1:
    """Caller-declared sensor duties; NOT an independently verified role.

    Confidence is a raw measurement policy, never a calibrated probability of
    semantic correctness. Discovery's sensor duties cannot replace the normal
    parent's persistent/semantic subject and extent obligations.
    """
    reference: str
    required_joint_indices: tuple[int, ...]
    minimum_joint_confidence: float

    def __post_init__(self):
        if (not isinstance(self.reference, str) or not self.reference.strip()
                or len(self.reference) > 512 or not isinstance(self.required_joint_indices, tuple)
                or not self.required_joint_indices or len(set(self.required_joint_indices)) != len(self.required_joint_indices)
                or any(type(i) is not int or not 0 <= i < 17 for i in self.required_joint_indices)
                or type(self.minimum_joint_confidence) not in (int, float)
                or not math.isfinite(self.minimum_joint_confidence)
                or not .3 <= self.minimum_joint_confidence <= 1):
            raise ValueError('pose duties require an explicit reference, immutable raw joints and confidence policy')

    @property
    def sha256(self):
        return _sha(asdict(self))


@dataclass(frozen=True, slots=True)
class NativePoseAnchorV1:
    observation: NativeGroupObservation
    person_observation_id: int | None
    measurement_seconds: float
    joint_states: tuple[PoseJointObservationState, ...]
    torso_local_xy: tuple[tuple[float, float] | None, ...]

    def __post_init__(self):
        if (not isinstance(self.observation, NativeGroupObservation)
                or self.person_observation_id is not None and (type(self.person_observation_id) is not int
                    or self.person_observation_id not in {p.observation_id for p in self.observation.people})
                or type(self.measurement_seconds) not in (int, float) or not math.isfinite(self.measurement_seconds)
                or self.measurement_seconds < 0 or not isinstance(self.joint_states, tuple)
                or len(self.joint_states) != 17 or any(not isinstance(s, PoseJointObservationState) for s in self.joint_states)
                or not isinstance(self.torso_local_xy, tuple) or len(self.torso_local_xy) != 17
                or any(p is not None and (not isinstance(p, tuple) or len(p) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)) for p in self.torso_local_xy)):
            raise ValueError('native pose anchor needs exact raw identity and finite per-joint states/geometry')

    def to_dict(self):
        return dict(observation_sha256=self.observation.sha256,
            request_source_index=self.observation.request_source_index,
            native_frame_index=self.observation.native_frame_index,
            person_observation_id=self.person_observation_id,
            measurement_seconds=self.measurement_seconds,
            joint_states=[s.value for s in self.joint_states],
            torso_local_xy=self.torso_local_xy,
            geometry_for_ambiguous_owner_is_not_visible_subject_evidence=True)


@dataclass(frozen=True, slots=True)
class NativePoseTrackEvidenceV1:
    duties: FrozenPoseMotionDutiesV1
    source_domain: GroupSourceDomain
    anchors: tuple[NativePoseAnchorV1, ...]
    time_basis: str
    state: GroupSourceCameraState | None = None
    output_ticks: tuple[int, ...] | None = None
    _digest: str = field(init=False, repr=False, compare=False)
    _domain_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.duties, FrozenPoseMotionDutiesV1) or not isinstance(self.anchors, tuple)
                or any(not isinstance(a, NativePoseAnchorV1) for a in self.anchors)):
            raise ValueError('temporal pose evidence needs immutable typed duties and raw anchors')
        rows = tuple(a.observation for a in self.anchors)
        clock = _bound_rows(self.source_domain, rows)
        if self.time_basis == 'verified_native_presentation_seconds_not_output_phase_authority':
            if self.state is not None or self.output_ticks is not None:
                raise ValueError('source-native pose cannot borrow output clock authority')
            expected_times = tuple(row.native_frame_index/float(clock.frame_rate) for row in rows)
        elif self.time_basis == 'actual_complete_state_output_seconds':
            _validate_output_join(self.state, self.source_domain, rows, self.output_ticks)
            expected_times = tuple(tick/self.state.output_fps for tick in self.output_ticks)
        else:
            raise ValueError('unknown temporal pose time basis')
        for anchor, expected in zip(self.anchors, expected_times, strict=True):
            person = next((p for p in anchor.observation.people if p.observation_id == anchor.person_observation_id), None)
            states, xy = _joint_geometry(anchor.observation, person, self.duties.minimum_joint_confidence)
            if anchor.measurement_seconds != expected or anchor.joint_states != states or anchor.torso_local_xy != xy:
                raise ValueError('temporal pose states/geometry/time do not replay from raw observations')
        if any(a.observation.native_frame_index == b.observation.native_frame_index
                and a.person_observation_id != b.person_observation_id for a, b in zip(self.anchors, self.anchors[1:])):
            raise ValueError('repeated native pose cannot silently change its owner')
        object.__setattr__(self, '_domain_digest', _sha(self.source_domain.to_dict()))
        object.__setattr__(self, '_digest', _sha(self.to_dict()))

    @property
    def state_key(self):
        return None if self.state is None else self.state.key

    def to_dict(self):
        return dict(contract='bound_raw17_torso_local_temporal_pose_hypothesis_v1',
            duties=asdict(self.duties), duty_sha256=self.duties.sha256,
            source_domain_sha256=self._domain_digest,
            raw_observation_sha256=[a.observation.sha256 for a in self.anchors],
            anchors=[a.to_dict() for a in self.anchors], time_basis=self.time_basis,
            state_key=self.state_key, output_ticks=self.output_ticks,
            association_basis='supplied_all_raw_owner_alternatives_not_persistent_identity',
            lk_feature_survival_prerequisite=False,
            pose_or_velocity_imputed=False, missing_duties_removed=False,
            raw_score_basis='rtmpose_simcc_raw_peak_not_probability_v1',
            raw_scores_clipped=False, legacy_body8_score_clipping_replayed=True,
            source_occlusion_inferred_from_detector_failure=False,
            source_native_face_roi_is_not_whole_head_extent=True,
            camera_projection_or_required_extent_certified=False,
            source_local_identity_certified=False, semantic_duties_verified=False,
            phase_certified=False, quality_approved=False)

    @property
    def sha256(self):
        return self._digest


def _joint_geometry(row, person, confidence):
    unknown = (PoseJointObservationState.DETECTOR_UNKNOWN,)*17
    if (not isinstance(person, OwnedPersonObservation) or person.pose_reason != 'observed'
            or len(person.raw_pose_xyc) != 17 or person.pose_crop_xywh is None
            or not person.source_pixel_to_pose_input_affine or row.native_pixel_sha256 is None):
        return unknown, (None,)*17
    x, y, w, h = person.pose_crop_xywh
    ambiguous = (bool(person.competing_person_observation_ids)
        or 'ambiguous' in (person.body_ownership_state, person.upper_body_ownership_state))
    owned = 'observed' in (person.body_ownership_state, person.upper_body_ownership_state)
    states, pixels = [], []
    for px, py, score in person.raw_pose_xyc:
        point = None
        # RTMPose's raw SimCC peak is not a unit-interval probability.
        # Keep >1 exactly; legacy body8 clips only its compatibility scores.
        if score < confidence:
            status = PoseJointObservationState.DETECTOR_UNKNOWN
        elif not 0 <= px < 1 or not 0 <= py < 1:
            status = PoseJointObservationState.OUT_OF_SOURCE_FRAME
        elif not x <= px <= x+w or not y <= py <= y+h:
            status = PoseJointObservationState.DETECTOR_UNKNOWN
        else:
            point = np.asarray((px*row.frame_size_wh[0], py*row.frame_size_wh[1]))
            status = (PoseJointObservationState.OWNERSHIP_AMBIGUOUS if ambiguous else
                PoseJointObservationState.VISIBLE_SUPPORTED if owned else
                PoseJointObservationState.DETECTOR_UNKNOWN)
        states.append(status); pixels.append(point)
    # Actual shoulders, pixel aspect and signed anatomical axis. No box-sized
    # torso, cropped hips, mirrored arm or missing shoulder is manufactured.
    left, right = pixels[5], pixels[6]
    if left is None or right is None:
        return tuple(states), (None,)*17
    axis = right-left; length = float(np.linalg.norm(axis))
    if length < 2.:
        return tuple(states), (None,)*17
    ux = axis/length; uy = np.asarray((-ux[1], ux[0])); origin = (left+right)/2
    return tuple(states), tuple(None if point is None else (
        float(np.dot(point-origin, ux)/length), float(np.dot(point-origin, uy)/length)) for point in pixels)


def _bound_rows(domain, rows):
    if (not isinstance(domain, GroupSourceDomain) or not isinstance(rows, tuple)
            or not 3 <= len(rows) <= 14 or any(not isinstance(row, NativeGroupObservation) for row in rows)):
        raise ValueError('temporal raw pose needs three to fourteen typed chronological native anchors')
    first = rows[0]; source = first.request_source_index
    clock = domain.clocks_by_request.get(source)
    if clock is None:
        raise ValueError('temporal pose source is excluded from the native clock domain')
    grid = _sha(clock.to_dict())
    sizes = {row.frame_size_wh for row in rows if row.native_pixel_sha256 is not None}
    if len(sizes) > 1:
        raise ValueError('temporal pose native raster dimensions changed')
    for row in rows:
        if (row.source_clock_identity != grid or row.request_source_index != source
                or row.native_frame_index >= clock.frame_count
                or (row.source_file_identity, row.model_content_sha256,
                    row.model_adapter_fingerprints, row.person_ownership_policy_sha256)
                    != (first.source_file_identity, first.model_content_sha256,
                        first.model_adapter_fingerprints, first.person_ownership_policy_sha256)
                or not row.model_adapter_fingerprints or not row.model_content_sha256
                or row.person_ownership_policy_sha256 != ownership_policy_fingerprint()):
            raise ValueError('temporal pose raw source/grid/file/raster/model/adapter/ownership binding changed')
        for person in row.people:
            if (isinstance(person, OwnedPersonObservation) and len(person.raw_pose_xyc) == 17
                    and tuple((person.raw_pose_xyc[n][0],person.raw_pose_xyc[n][1],
                        max(0.,min(1.,person.raw_pose_xyc[n][2]))) for n in (5,6,7,8,9,10,11,12)) != person.upper_body_xyc):
                raise ValueError('temporal pose raw17 and declared eight-joint ROI basis differ')
        rebound, owners = bind_person_ownership(row.people, row.heads, row.frame_size_wh)
        if rebound != row.people or any(h.candidate_body_observation_ids != tuple(owners[h.observation_id]) for h in row.heads):
            raise ValueError('temporal pose raw ownership alternatives do not independently replay')
    if any(a.native_frame_index > b.native_frame_index for a, b in zip(rows, rows[1:])):
        raise ValueError('temporal native anchors are not chronological')
    if any(a.native_frame_index == b.native_frame_index and a.sha256 != b.sha256
            for a, b in zip(rows, rows[1:])):
        raise ValueError('repeated native pose has contradictory raw evidence')
    if len({row.native_frame_index for row in rows}) < 3:
        raise TemporalPoseUnavailable('three distinct native samples required; repeated output ticks are not new poses')
    fps = float(clock.frame_rate)
    if any(first.native_frame_index < math.ceil(cut*fps-1e-9) <= rows[-1].native_frame_index
            for cut in domain.cuts_for_request(source)):
        raise TemporalPoseUnavailable('temporal pose cannot join across an actual native broadcaster cut')
    return clock


def _windows(*, domain, rows, choices, duties, times, time_basis, state=None,
        output_ticks=None, maximum_hypotheses=64):
    _bound_rows(domain, rows)
    if (not isinstance(duties, FrozenPoseMotionDutiesV1) or not isinstance(choices, tuple)
            or len(choices) != len(rows) or any(not isinstance(values, tuple) or not values
                or len(set(values)) != len(values) or any(v is not None and (type(v) is not int or v < 0)
                    for v in values) for values in choices)
            or type(maximum_hypotheses) is not int or not 1 <= maximum_hypotheses <= 256
            or len(times) != len(rows) or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or any(a > b for a, b in zip(times, times[1:]))):
        raise ValueError('temporal pose requires retained finite all-owner hypotheses and frozen duties')
    groups = []
    for n, row in enumerate(rows):
        if n == 0 or row.native_frame_index != rows[n-1].native_frame_index:
            groups.append(n)
        elif choices[n] != choices[n-1]:
            raise ValueError('repeated native pose cannot change its raw owner alternatives')
    count = math.prod(len(choices[n]) for n in groups)
    if count > maximum_hypotheses:
        raise TemporalPoseUnavailable('temporal pose hypothesis overflow, not success-biased truncation')
    people = [{p.observation_id: p for p in row.people} for row in rows]
    if any(value is not None and value not in by_id for by_id, values in zip(people, choices) for value in values):
        raise ValueError('temporal pose association names an absent raw person')
    result = []
    for unique_ids in product(*(choices[n] for n in groups)):
        ids = []; group = -1
        for n in range(len(rows)):
            if n in groups:
                group += 1
            ids.append(unique_ids[group])
        anchors = tuple(NativePoseAnchorV1(row, pid, float(time), *_joint_geometry(row,
            people[n].get(pid), duties.minimum_joint_confidence))
            for n, (row, pid, time) in enumerate(zip(rows, ids, times, strict=True)))
        result.append(NativePoseTrackEvidenceV1(duties, domain, anchors,
            time_basis, state, output_ticks))
    return tuple(result)


def native_pose_windows_from_rows(*, domain, rows, choices, duties, maximum_hypotheses=64):
    clock = _bound_rows(domain, rows)
    return _windows(domain=domain, rows=rows, choices=choices, duties=duties,
        times=tuple(row.native_frame_index/float(clock.frame_rate) for row in rows),
        time_basis='verified_native_presentation_seconds_not_output_phase_authority',
        maximum_hypotheses=maximum_hypotheses)


def independent_native_pose_hypotheses(*, domain, rows, duties, initial_person_ids=None,
        maximum_hypotheses=64):
    """All raw source-local person associations, not conditioned on LK success.

    The optional FIRST-anchor cohort is external, frozen before donor search.
    Later missing rows retain None. These alternatives include unresolved
    identity switches; a small pose cost is never permission to pick one.
    """
    _bound_rows(domain, rows)
    choices = tuple(tuple(p.observation_id for p in row.people) or (None,) for row in rows)
    if initial_person_ids is not None:
        if (not isinstance(initial_person_ids, tuple) or not initial_person_ids
                or any(n is not None and (type(n) is not int or n not in choices[0]) for n in initial_person_ids)
                or len(set(initial_person_ids)) != len(initial_person_ids)):
            raise ValueError('independent pose cohort needs actual externally frozen first-anchor person IDs')
        choices = (initial_person_ids, *choices[1:])
    return native_pose_windows_from_rows(domain=domain, rows=rows, choices=choices, duties=duties,
        maximum_hypotheses=maximum_hypotheses)


def _validate_output_join(state, domain, rows, output_ticks):
    if (not isinstance(state, GroupSourceCameraState) or not isinstance(output_ticks, tuple)
            or len(output_ticks) != len(rows) or any(type(n) is not int
                or not state.first_output_frame <= n < state.end_output_frame for n in output_ticks)
            or any(a >= b for a, b in zip(output_ticks, output_ticks[1:]))):
        raise ValueError('render pose needs actual chronological ticks of a complete source-camera state')
    state.bind_source(domain)
    source = domain.request_index(state.eligible_source_index)
    if any(row.request_source_index != source or row.native_frame_index !=
            state.native_frame_indices[tick-state.first_output_frame] for row, tick in zip(rows, output_ticks)):
        raise ValueError('render pose raw rows differ from the actual whole-state output/native join')


def render_pose_windows_from_rows(*, state, domain, rows, output_ticks, choices, duties,
        maximum_hypotheses=64):
    _validate_output_join(state, domain, rows, output_ticks)
    return _windows(domain=domain, rows=rows, choices=choices, duties=duties,
        times=tuple(tick/state.output_fps for tick in output_ticks),
        time_basis='actual_complete_state_output_seconds', state=state,
        output_ticks=output_ticks, maximum_hypotheses=maximum_hypotheses)


def _unique_anchors(window):
    result = []
    for anchor in window.anchors:
        if not result or anchor.observation.native_frame_index != result[-1].observation.native_frame_index:
            result.append(anchor)
    return tuple(result)


def _motion(window, joint):
    anchors = _unique_anchors(window)
    points = tuple(a.torso_local_xy[joint] for a in anchors)
    if any(p is None for p in points):
        return None
    times = np.asarray([a.measurement_seconds for a in anchors], dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0):
        return None
    values = np.asarray(points)
    velocity = np.diff(values, axis=0)/dt[:, None]
    reversals = sum(float(np.dot(a, b)) < 0 for a, b in zip(velocity, velocity[1:]))
    return values, velocity, reversals


def _cosine(a, b):
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if norm < 1e-12 else float(np.clip(np.dot(a, b)/norm, -1., 1.))


def compare_pose_phase_windows(reference, candidate):
    """Multi-anchor phase ranking measurements, NEVER a calibrated gate.

    Pairwise same-order anchor comparison is meaningful only for separately
    aligned music times. The caller's alignment is not certified here. Every
    required joint is reported; ambiguous geometry cannot become visible duty.
    All owner-pair alternatives must be retained by the calling discovery path.
    """
    if (not isinstance(reference, NativePoseTrackEvidenceV1) or not isinstance(candidate, NativePoseTrackEvidenceV1)
            or reference.duties != candidate.duties or reference.source_domain != candidate.source_domain
            or reference.time_basis != candidate.time_basis
            or len(_unique_anchors(reference)) != len(_unique_anchors(candidate))):
        raise ValueError('pose phase comparison requires the same frozen duties/time basis and unique anchor count')
    rows = []
    for joint in reference.duties.required_joint_indices:
        a, b = _motion(reference, joint), _motion(candidate, joint)
        supported = all(anchor.joint_states[joint] == PoseJointObservationState.VISIBLE_SUPPORTED
            for anchor in (*reference.anchors, *candidate.anchors))
        raw = dict(joint_index=joint, required=True, all_source_support_observed=supported,
            reference_states=[anchor.joint_states[joint].value for anchor in reference.anchors],
            candidate_states=[anchor.joint_states[joint].value for anchor in candidate.anchors],
            multi_anchor_residual=None, maximum_velocity_difference_per_second=None,
            velocity_direction_cosines=None, reference_reversals=None, candidate_reversals=None)
        if a is not None and b is not None:
            pa, va, ra = a; pb, vb, rb = b
            raw.update(multi_anchor_residual=[float(v) for v in np.linalg.norm(pa-pb, axis=1)],
                maximum_velocity_difference_per_second=float(np.max(np.linalg.norm(va-vb, axis=1))),
                velocity_direction_cosines=[_cosine(x, y) for x, y in zip(va, vb)],
                reference_reversals=ra, candidate_reversals=rb)
        rows.append(raw)
    return dict(contract='raw17_multi_anchor_cross_source_pose_motion_measurements_v1',
        reference_track_sha256=reference.sha256, candidate_track_sha256=candidate.sha256,
        duty_sha256=reference.duties.sha256, time_basis=reference.time_basis, joints=rows,
        raw_geometry_may_remain_ownership_ambiguous=True,
        all_required_source_support_observed=all(row['all_source_support_observed'] for row in rows),
        missing_is_zero_cost=False, minimum_over_convenient_owner_used=False,
        alignment_certified=False, source_local_association_certified=False,
        cross_source_identity_certified=False, temporal_phase_certified=False,
        calibrated=False, candidate_admitted=False, quality_approved=False)


def compare_pose_boundary_windows(outgoing, incoming):
    """True-output-second moving-cut residuals across actual complete states."""
    if (not isinstance(outgoing, NativePoseTrackEvidenceV1) or not isinstance(incoming, NativePoseTrackEvidenceV1)
            or outgoing.time_basis != 'actual_complete_state_output_seconds'
            or incoming.time_basis != outgoing.time_basis or outgoing.duties != incoming.duties
            or incoming.source_domain != outgoing.source_domain
            or outgoing.state.output_fps != incoming.state.output_fps
            or outgoing.state.end_output_frame != incoming.state.first_output_frame
            or outgoing.output_ticks[-1] != outgoing.state.end_output_frame-1
            or incoming.output_ticks[0] != incoming.state.first_output_frame
            or incoming.anchors[0].measurement_seconds <= outgoing.anchors[-1].measurement_seconds):
        raise ValueError('moving cut comparison needs externally bound chronological output windows and unchanged duties')
    gap = incoming.anchors[0].measurement_seconds-outgoing.anchors[-1].measurement_seconds
    measured = []
    for joint in outgoing.duties.required_joint_indices:
        a, b = _motion(outgoing, joint), _motion(incoming, joint)
        item = dict(joint_index=joint, required=True, predicted_pose_jump=None,
            velocity_difference_per_output_second=None, velocity_direction_cosine=None,
            outgoing_reversals=None, incoming_reversals=None,
            all_source_support_observed=all(anchor.joint_states[joint] == PoseJointObservationState.VISIBLE_SUPPORTED
                for anchor in (*outgoing.anchors, *incoming.anchors)))
        if a is not None and b is not None:
            pa, va, ra = a; pb, vb, rb = b
            item.update(predicted_pose_jump=float(np.linalg.norm(pa[-1]+va[-1]*gap-pb[0])),
                velocity_difference_per_output_second=float(np.linalg.norm(va[-1]-vb[0])),
                velocity_direction_cosine=_cosine(va[-1], vb[0]), outgoing_reversals=ra, incoming_reversals=rb)
        measured.append(item)
    return dict(contract='raw17_actual_output_seconds_moving_cut_measurements_v1',
        outgoing_track_sha256=outgoing.sha256, incoming_track_sha256=incoming.sha256,
        outgoing_state_key=outgoing.state_key, incoming_state_key=incoming.state_key,
        boundary_gap_output_seconds=gap, duty_sha256=outgoing.duties.sha256, joints=measured,
        lk_feature_survival_prerequisite=False, missing_is_zero_cost=False,
        all_required_source_support_observed=all(row['all_source_support_observed'] for row in measured),
        surface_or_extent_certified=False, source_local_association_certified=False,
        cross_source_identity_certified=False, temporal_phase_certified=False,
        calibrated=False, candidate_admitted=False, quality_approved=False)
