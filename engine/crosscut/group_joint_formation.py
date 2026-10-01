"""Observed partial formation correspondence in projected output coordinates.

Required foreground cohorts are declared BEFORE comparing cameras. Extra
detections are not silently zipped into them; occlusion remains explicitly
unmatched. This is geometry/choreography evidence, not named-member identity,
four-tick motion, containment, cut approval or professional-quality proof.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class ObservedFormationPerson:
    observation_id: int
    box_xywh: tuple[float, float, float, float]
    upper_body_xyc: tuple[tuple[float, float, float], ...]
    confidence: float

    def __post_init__(self):
        box = self.box_xywh
        if (type(self.observation_id) is not int or self.observation_id < 0
                or not isinstance(box, tuple) or len(box) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                or box[2] <= 0 or box[3] <= 0
                or type(self.confidence) not in (int, float)
                or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
                or not isinstance(self.upper_body_xyc, tuple)
                or len(self.upper_body_xyc) not in (0, 8)
                or any(not isinstance(p, tuple) or len(p) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in p)
                    or not 0 <= p[2] <= 1 for p in self.upper_body_xyc)):
            raise ValueError('formation person requires explicit observed screen geometry and eight-joint basis')
        # Off-screen points/boxes must remain distinguishable from visible ones.
        # Clipping them here would manufacture containment evidence.


@dataclass(frozen=True, slots=True)
class DeclaredFormationView:
    people: tuple[ObservedFormationPerson, ...]
    required_observation_ids: tuple[int, ...]
    output_size: tuple[int, int] = (1920, 1080)

    def __post_init__(self):
        ids = tuple(p.observation_id for p in self.people)
        if (not isinstance(self.people, tuple) or len(self.people) > 12
                or len(set(ids)) != len(ids)
                or not isinstance(self.required_observation_ids, tuple)
                or len(set(self.required_observation_ids)) != len(self.required_observation_ids)
                or not set(self.required_observation_ids) <= set(ids)
                or any(type(n) is not int for n in self.required_observation_ids)
                or not isinstance(self.output_size, tuple) or len(self.output_size) != 2
                or any(type(n) is not int or n < 1 for n in self.output_size)):
            raise ValueError('formation role needs an explicit predeclared bounded foreground cohort')

    @property
    def required(self):
        by_id = {p.observation_id: p for p in self.people}
        return tuple(by_id[n] for n in self.required_observation_ids)


@dataclass(frozen=True, slots=True)
class ProjectedFormationMatch:
    status: str
    pairs: tuple[tuple[int, int], ...]
    unmatched_outgoing: tuple[int, ...]
    unmatched_incoming: tuple[int, ...]
    outgoing_coverage: float
    incoming_coverage: float
    assignment_margin: float | None
    pose_mismatch: float | None
    height_distribution_mismatch: float | None
    horizontal_span_mismatch: float | None
    group_center_distance: float | None
    paired_subject_location_mismatch: float | None
    framing_cost: float | None
    common_bones: tuple[tuple[str, ...], ...] = ()

    @property
    def observed(self):
        return self.status == 'observed_partial_correspondence'

    def to_dict(self):
        from dataclasses import asdict
        return dict(asdict(self), contract='predeclared_projected_partial_formation_correspondence_v1',
            correspondence_basis='eight_joint_pose_and_cohort_relative_location_not_x_order_zip',
            named_member_identity_certified=False, motion_certified=False,
            visibility_certified=False, quality_approved=False, raw_family_gate=False)


_BONES = (('left_upper_arm', 0, 2), ('right_upper_arm', 1, 3),
          ('left_forearm', 2, 4), ('right_forearm', 3, 5))


def _bones(person, size):
    if not person.upper_body_xyc or person.confidence < .35:
        return {}
    points = np.asarray(person.upper_body_xyc)
    result = {}
    for name, a, b in _BONES:
        if min(points[a, 2], points[b, 2]) < .3:
            continue
        delta = (points[b, :2] - points[a, :2]) * np.asarray(size)
        length = float(np.linalg.norm(delta))
        if length >= 2.:
            result[name] = delta / length
    return result


def _locations(people):
    boxes = np.asarray([p.box_xywh for p in people], dtype=float)
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    center = np.median(centers, axis=0)
    scale = max(1e-6, float(np.median(boxes[:, 3])))
    return (centers - center) / scale


def _unknown(status, left, right, *, pairs=(), margin=None, common=()):
    aa, bb = {a for a, _ in pairs}, {b for _, b in pairs}
    return ProjectedFormationMatch(status, tuple(pairs),
        tuple(p.observation_id for p in left if p.observation_id not in aa),
        tuple(p.observation_id for p in right if p.observation_id not in bb),
        len(aa) / len(left) if left else 0., len(bb) / len(right) if right else 0.,
        margin, None, None, None, None, None, None, tuple(common))


def compare_projected_formation(outgoing, incoming, *, minimum_pairs=2,
        minimum_cohort_coverage=2/3, maximum_pair_cost=.45, minimum_assignment_margin=.015):
    """Partial bipartite assignment with explicit ambiguity, never forced zip.

    Pair reachability uses observed pose and COHORT-relative positions, not
    face-height thresholds or a raw shot-family label. Camera compatibility
    then uses paired height distributions/span/center/locations in the actual
    output. The constants are a diagnostic policy, not a quality waiver/SLA.
    Unknown pose, insufficient overlap and ambiguous assignment stay unknown.
    """
    if (not isinstance(outgoing, DeclaredFormationView)
            or not isinstance(incoming, DeclaredFormationView)
            or outgoing.output_size != incoming.output_size
            or type(minimum_pairs) is not int or minimum_pairs < 2
            or any(type(v) not in (int, float) or not math.isfinite(v)
                or not 0 < v <= 1 for v in (minimum_cohort_coverage, maximum_pair_cost,
                    minimum_assignment_margin))):
        raise ValueError('projected formation comparison needs common output coordinates and bounded policy')
    left, right = outgoing.required, incoming.required
    if min(len(left), len(right)) < minimum_pairs:
        return _unknown('required_cohort_unobserved', left, right)
    positions_a, positions_b = _locations(left), _locations(right)
    bones_a = [_bones(p, outgoing.output_size) for p in left]
    bones_b = [_bones(p, incoming.output_size) for p in right]
    # An unmatched dummy is cheaper than fabricating a bad person match.
    # Padding BOTH sides permits occlusion without pairing a background extra.
    n, m = len(left), len(right)
    costs = np.full((n+m, n+m), 10., dtype=float)
    costs[:n, m:] = maximum_pair_cost/2
    costs[n:, :m] = maximum_pair_cost/2
    costs[n:, m:] = 0.
    details = {}
    for i in range(n):
        for j in range(m):
            common = tuple(name for name, _, _ in _BONES if name in bones_a[i] and name in bones_b[j])
            if len(common) < 2:
                continue
            pose = float(np.mean([np.linalg.norm(bones_a[i][name]-bones_b[j][name])/2 for name in common]))
            location = min(1., float(np.linalg.norm(positions_a[i]-positions_b[j]))/3)
            value = .75*pose + .25*location
            costs[i, j] = value
            details[i, j] = (pose, common)
    rows, cols = linear_sum_assignment(costs)
    accepted = tuple((int(i), int(j)) for i, j in zip(rows, cols)
        if i < n and j < m and (int(i), int(j)) in details and costs[i, j] < maximum_pair_cost)
    pairs = tuple((left[i].observation_id, right[j].observation_id) for i, j in accepted)
    common = tuple(details[i, j][1] for i, j in accepted)
    if len(pairs) < minimum_pairs or min(len(pairs)/n, len(pairs)/m) < minimum_cohort_coverage:
        return _unknown('required_partial_correspondence_unobserved', left, right, pairs=pairs, common=common)
    best = float(costs[rows, cols].sum())
    alternatives = []
    for i, j in accepted:
        trial = costs.copy(); trial[i, j] = 10.
        a, b = linear_sum_assignment(trial)
        alternate_pairs = {(int(x), int(y)) for x, y in zip(a, b)
            if x < n and y < m and (int(x), int(y)) in details
                and trial[x, y] < maximum_pair_cost}
        changed_pairs = len(set(accepted)-alternate_pairs)
        # Measure uncertainty per CHANGED assignment. Dividing a local swap's
        # margin by every unrelated good match makes a six-person cohort less
        # certain than the exact same ambiguous pair in a three-person cohort.
        alternatives.append((float(trial[a, b].sum())-best)/max(1, changed_pairs))
    margin = min(alternatives)
    if margin < minimum_assignment_margin:
        return _unknown('ambiguous_partial_correspondence', left, right, pairs=pairs, margin=margin, common=common)
    boxes_a = np.asarray([left[i].box_xywh for i, _ in accepted])
    boxes_b = np.asarray([right[j].box_xywh for _, j in accepted])
    centers_a, centers_b = boxes_a[:, :2]+boxes_a[:, 2:]/2, boxes_b[:, :2]+boxes_b[:, 2:]/2
    # Quantiles, not max(height): one nearby dancer cannot stand for a group.
    heights_a, heights_b = (np.quantile(boxes[:, 3], (.25, .5, .75)) for boxes in (boxes_a, boxes_b))
    heights = float(np.mean(np.abs(np.log(heights_a/heights_b))))
    spans = [float(max(boxes[:, 0]+boxes[:, 2])-min(boxes[:, 0])) for boxes in (boxes_a, boxes_b)]
    span = abs(math.log(max(spans[0], 1e-6)/max(spans[1], 1e-6)))
    center = float(np.linalg.norm(np.median(centers_a, axis=0)-np.median(centers_b, axis=0)))
    location = float(np.sqrt(np.mean(np.sum((centers_a-centers_b)**2, axis=1))))
    pose = float(np.mean([details[i, j][0] for i, j in accepted]))
    framing = min(1., .35*heights + .25*span + .2*center + .2*location)
    return ProjectedFormationMatch('observed_partial_correspondence', pairs,
        tuple(p.observation_id for p in left if p.observation_id not in {a for a, _ in pairs}),
        tuple(p.observation_id for p in right if p.observation_id not in {b for _, b in pairs}),
        len(pairs)/n, len(pairs)/m, margin, pose, heights, span, center, location, framing, common)
