"""Short source-local target tracks established BEFORE cross-source cuts.

A positive face match alone may jump to another performer when the target
bends or is occluded. A body score alone can confuse similar costumes. This
module requires distinct frozen-face/body anchors and connected observed body
trajectories. It may recover an observed body with no visible face; it never
manufactures a face box or updates the frozen identity gallery.

An unambiguous frozen face with independent exact-frame pose ownership is also
an anchor, as in direct-frame admission. This does not waive the .60 clothing
floor or the appearance/motion edges needed to reach an occluded query.

All inference inputs are transient. Serialized results contain geometry and
scalar evidence, not face/body vectors or named-identity probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .direct_member_target_face import FocalFaceCandidate
from .face_identity import cosine_similarity
from .pose import FormationPersonDescriptor


FOCAL_BODY_ANCHOR_POLICY = 'frozen_clothing_or_unique_exact_face_pose_with_observed_body_edges_v2'


def _cos(left, right) -> float:
    value = cosine_similarity(left, right)
    return -1.0 if value is None else float(value)


def body_reference_similarity(embedding, references) -> float:
    scores = sorted((_cos(embedding, r) for r in references), reverse=True)
    return sum(scores[:3])/min(3, len(scores)) if scores else -1.0


def _iou(left, right) -> float:
    x, y, w, h = left
    rx, ry, rw, rh = right
    intersection = max(0., min(x+w, rx+rw)-max(x, rx))*max(0., min(y+h, ry+rh)-max(y, ry))
    return intersection/max(1e-12, w*h+rw*rh-intersection)


@dataclass(frozen=True, slots=True)
class FocalBodyObservation:
    source_index: int
    source_seconds: float
    person: FormationPersonDescriptor
    appearance: tuple[float, ...] = field(repr=False)
    reference_similarity: float
    face: FocalFaceCandidate | None = field(default=None, repr=False)
    frozen_face_match: bool = False
    pose_screen_xyc: tuple[tuple[float, float, float], ...] = field(default=(), repr=False)
    face_pose_consistent: bool | None = None

    def __post_init__(self):
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                or self.source_index < 0 or isinstance(self.source_seconds, bool)
                or not isinstance(self.source_seconds, (int, float))
                or not math.isfinite(self.source_seconds) or self.source_seconds < 0
                or not isinstance(self.person, FormationPersonDescriptor)
                or not isinstance(self.appearance, tuple) or not self.appearance
                or any(not math.isfinite(v) for v in self.appearance)
                or not math.isfinite(self.reference_similarity)
                or not -1.000001 <= self.reference_similarity <= 1.000001
                or not isinstance(self.frozen_face_match, bool)
                or self.frozen_face_match and self.face is None
                or self.face_pose_consistent is not None and not isinstance(self.face_pose_consistent, bool)
                or self.face_pose_consistent is True and (self.face is None or not self.pose_screen_xyc)
                or self.pose_screen_xyc and (len(self.pose_screen_xyc) != 8
                    or any(len(p)!=3 or any(not math.isfinite(v) for v in p)
                           or not 0<=p[2]<=1 for p in self.pose_screen_xyc))):
            raise ValueError('source-local body observation is invalid')


@dataclass(frozen=True, slots=True)
class FocalBodyTrackResult:
    source_index: int
    source_seconds: float
    observation: FocalBodyObservation | None = field(repr=False)
    reason: str
    support_seconds: tuple[float, ...] = ()
    path_margin: float | None = None

    def to_dict(self) -> dict:
        observation = self.observation
        return {'contract': 'source_local_frozen_face_body_track_v1',
            'anchor_policy': FOCAL_BODY_ANCHOR_POLICY,
            'source_index': self.source_index, 'source_seconds': self.source_seconds,
            'reason': self.reason, 'anchor_seconds': self.support_seconds,
            'path_margin': self.path_margin,
            'body_bbox_xywh': observation.person.bbox_xywh if observation else None,
            'face_bbox_xywh': observation.face.bbox_xywh if observation and observation.face else None,
            'body_reference_similarity': observation.reference_similarity if observation else None,
            'face_geometry_extrapolated': False, 'gallery_updated': False,
            'appearance_embeddings_persisted': False,
            'final_cut_quality_authority': False}


@dataclass(frozen=True, slots=True)
class FocalBodyGeometryCertificate:
    """Scalar snapshot of an observed track; contains no appearance vector."""
    source_index: int
    source_seconds: float
    body_bbox_xywh: tuple[float, float, float, float]
    support_seconds: tuple[float, ...]
    path_margin: float | None
    reason: str = 'observed_target_body_track'

    def __post_init__(self):
        x,y,w,h=self.body_bbox_xywh
        if (isinstance(self.source_index,bool) or not isinstance(self.source_index,int) or self.source_index<0
                or not math.isfinite(self.source_seconds) or self.source_seconds<0
                or any(not math.isfinite(v) for v in self.body_bbox_xywh)
                or min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001
                or len(set(self.support_seconds))<2 or len(self.support_seconds)>65
                or any(not math.isfinite(t) or t<0 or abs(t-self.source_seconds)>2.1 for t in self.support_seconds)
                or self.path_margin is not None and (not math.isfinite(self.path_margin) or self.path_margin<0)
                or self.reason!='observed_target_body_track'):
            raise ValueError('focal body geometry certificate is invalid')

    @classmethod
    def from_observed_track(cls, result):
        if not isinstance(result,FocalBodyTrackResult) or result.observation is None:
            raise ValueError('a body certificate requires an actually observed track')
        return cls(result.source_index,result.source_seconds,result.observation.person.bbox_xywh,
            result.support_seconds,result.path_margin,result.reason)

    def to_dict(self):
        return {'contract':'source_local_focal_body_geometry_certificate_v1',
            'source_index':self.source_index,'source_seconds':self.source_seconds,
            'body_bbox_xywh':self.body_bbox_xywh,'anchor_seconds':self.support_seconds,
            'path_margin':self.path_margin,'reason':self.reason,
            'face_geometry_extrapolated':False,'appearance_embeddings_persisted':False,
            'final_cut_quality_authority':False}


@dataclass(frozen=True, slots=True)
class FocalBodyFrame:
    source_index: int
    source_seconds: float
    observations: tuple[FocalBodyObservation, ...] = field(repr=False)

    def __post_init__(self):
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                or self.source_index < 0 or isinstance(self.source_seconds, bool)
                or not isinstance(self.source_seconds, (int, float))
                or not math.isfinite(self.source_seconds) or self.source_seconds < 0
                or not isinstance(self.observations, tuple) or len(self.observations)>32
                or any(not isinstance(o, FocalBodyObservation) or o.source_index != self.source_index
                       or o.source_seconds != self.source_seconds for o in self.observations)):
            raise ValueError('source-local body frame is invalid')


def deduplicate_body_observations(observations: Sequence[FocalBodyObservation]) -> tuple[FocalBodyObservation, ...]:
    retained = []
    for item in sorted(observations, key=lambda o:(-o.person.detection_confidence, o.person.bbox_xywh)):
        if any(_iou(item.person.bbox_xywh, old.person.bbox_xywh) >= .65
               and _cos(item.appearance, old.appearance) >= .90 for old in retained):
            continue
        retained.append(item)
    return tuple(retained)


def focal_body_anchor_indices(observations: Sequence[FocalBodyObservation], *,
                              allow_exact_pose_anchors: bool = True) -> tuple[int, ...]:
    """Shared necessary-condition authority for a deduplicated observed frame.

    Preserve every legacy face/clothing anchor. A new pose-owned route needs
    exactly ONE frozen-face body in the frame, real pose agreement and the same
    clothing floor as every trajectory edge. Similar costumes cannot veto an
    independently owned face, but ambiguous ownership and missing pose cannot
    use this route. Neither route alone approves a trajectory or a cut.
    """
    frozen = tuple(i for i, item in enumerate(observations) if item.frozen_face_match)
    anchors = []
    for i, item in enumerate(observations):
        margin = item.reference_similarity-max((other.reference_similarity
            for j, other in enumerate(observations) if j != i), default=-1.)
        clothing_anchor = item.frozen_face_match and item.reference_similarity >= .68 and margin >= .06
        pose_anchor = (allow_exact_pose_anchors and frozen == (i,) and item.face_pose_consistent is True
                       and item.reference_similarity >= .60)
        if clothing_anchor or pose_anchor:
            anchors.append(i)
    return tuple(anchors)


def _resolve_focal_body_track(
    frames: Sequence[FocalBodyFrame], *, source_index: int,
    query_seconds: float, maximum_window_seconds: float = 1.25,
    allow_exact_pose_anchors: bool,
) -> FocalBodyTrackResult:
    """Resolve a bounded observed path supported by at least two face/body anchors.

    This is an offline source-local graph, not a causal training feature. A
    frame with no observed body breaks the path; there is no interpolated
    person through an occluder. All candidate query bodies are evaluated in
    the same graph before ranking, so encounter order cannot pick identity.
    """
    if (isinstance(source_index, bool) or not isinstance(source_index, int)
            or source_index < 0 or isinstance(query_seconds, bool)
            or not isinstance(query_seconds, (int, float)) or not math.isfinite(query_seconds)
            or query_seconds < 0 or not 0 < maximum_window_seconds <= 2.
            or len(frames) > 65):
        raise ValueError('focal body track window is invalid')
    selected_frames = []
    for frame in frames:
        if not isinstance(frame, FocalBodyFrame) or frame.source_index != source_index:
            raise ValueError('body track cannot mix source evidence')
        if abs(frame.source_seconds-query_seconds) <= maximum_window_seconds+1e-9:
            selected_frames.append(frame)
    selected_frames.sort(key=lambda frame:frame.source_seconds)
    if len({frame.source_seconds for frame in selected_frames}) != len(selected_frames):
        raise ValueError('duplicate frame evidence cannot strengthen a body track')
    prepared = [deduplicate_body_observations(frame.observations) for frame in selected_frames]
    query = next((i for i, f in enumerate(selected_frames) if abs(f.source_seconds-query_seconds)<1e-7), None)
    fail = lambda reason:FocalBodyTrackResult(source_index, query_seconds, None, reason)
    if query is None or not prepared[query]:
        return fail('query_body_unobserved')

    anchors = [(i, j) for i, frame in enumerate(prepared)
               for j in focal_body_anchor_indices(frame,allow_exact_pose_anchors=allow_exact_pose_anchors)]
    if len({prepared[i][j].source_seconds for i, j in anchors}) < 2:
        return fail('independent_face_body_anchors_missing')

    def edge(left, right):
        delta = abs(right.source_seconds-left.source_seconds)
        if not .01 <= delta <= .26:
            return None
        if min(left.reference_similarity, right.reference_similarity) < .60:
            return None
        similarity = _cos(left.appearance, right.appearance)
        if similarity < .70:
            return None
        lb, rb = left.person.bbox_xywh, right.person.bbox_xywh
        center = math.dist(left.person.centroid_xy, right.person.centroid_xy)
        scale = abs(math.log((lb[2]*lb[3])/(rb[2]*rb[3])))
        # Distance is scaled by observed body size, not face size: bending
        # changes the face much more than the source-local body trajectory.
        normalized_center = center/max(.08, (math.sqrt(lb[2]*lb[3])+math.sqrt(rb[2]*rb[3]))/2)
        if normalized_center > .60 or scale > .85 or (_iou(lb, rb) < .12 and normalized_center > .35):
            return None
        if left.pose_screen_xyc and right.pose_screen_xyc:
            common = [(a, b) for a, b in zip(left.pose_screen_xyc, right.pose_screen_xyc)
                      if min(a[2], b[2])>=.30 and all(0<=v<1 for v in (*a[:2], *b[:2]))]
            if len(common)>=4:
                weight = sum(min(a[2],b[2]) for a,b in common)
                displacement = math.sqrt(sum(min(a[2],b[2])*math.dist(a[:2],b[:2])**2
                    for a,b in common)/weight)
                pose_motion = displacement/max(.08, (math.sqrt(lb[2]*lb[3])+math.sqrt(rb[2]*rb[3]))/2)
                return .30*(1-similarity)+.10*normalized_center+.10*scale+.50*pose_motion
        return .55*(1-similarity)+.25*normalized_center+.20*scale

    reachable = {j:[] for j in range(len(prepared[query]))}
    for ai, aj in anchors:
        # The query itself cannot count as its own temporal corroboration.
        if ai == query:
            continue
        step = 1 if ai < query else -1
        costs = {aj:(0., 0.)}
        for fi in range(ai+step, query+step, step):
            previous = prepared[fi-step]
            current = prepared[fi]
            next_costs = {}
            for j, observation in enumerate(current):
                routes = []
                for k, (total_cost, maximum_cost) in costs.items():
                    transition = edge(previous[k], observation)
                    if transition is not None:
                        routes.append((total_cost+transition, max(maximum_cost, transition)))
                if routes:
                    steps = abs(fi-ai)
                    next_costs[j] = min(routes, key=lambda cost:cost[0]/steps+.5*cost[1])
            costs = next_costs
            if not costs:
                break
        for j, (total_cost, maximum_cost) in costs.items():
            # A long window cannot dilute a one-frame identity jump into a
            # tiny mean motion cost. Retain the worst observed transition.
            reachable[j].append((prepared[ai][aj].source_seconds,
                total_cost/max(1, abs(query-ai))+.5*maximum_cost))
    ranked = []
    for j, observation in enumerate(prepared[query]):
        supports = reachable[j]
        times = sorted({t for t, _ in supports})
        if len(times) < 2 or times[-1]-times[0] < .08:
            continue
        # A single side is allowed at source edges, but evidence is still
        # two independently observed anchors, not one propagated match.
        best_costs = sorted(cost for _, cost in supports)[:2]
        score = observation.reference_similarity-sum(best_costs)/len(best_costs)
        ranked.append((score, j, tuple(times)))
    ranked.sort(key=lambda row:(-row[0], row[1]))
    if not ranked:
        return fail('target_trajectory_unresolved')
    margin = ranked[0][0]-ranked[1][0] if len(ranked)>1 else None
    if margin is not None and margin < .06:
        return fail('competing_target_trajectories')
    _, j, times = ranked[0]
    return FocalBodyTrackResult(source_index, query_seconds, prepared[query][j],
        'observed_target_body_track', times, margin)


def resolve_focal_body_track(
    frames: Sequence[FocalBodyFrame], *, source_index: int,
    query_seconds: float, maximum_window_seconds: float = 1.25,
) -> FocalBodyTrackResult:
    """Preserve a resolved legacy track; inspect pose-owned anchors only on failure.

    Both passes consume the SAME in-memory observations. No extra decoding,
    model inference or bank updates happen here. This strict incumbent-first
    rule also preserves already published positive source geometry; negatives
    are never persistent and are reconsidered under the new observer identity.
    """
    arguments=dict(source_index=source_index,query_seconds=query_seconds,
                   maximum_window_seconds=maximum_window_seconds)
    incumbent=_resolve_focal_body_track(frames,allow_exact_pose_anchors=False,**arguments)
    if incumbent.observation is not None or incumbent.reason=='query_body_unobserved':
        return incumbent
    return _resolve_focal_body_track(frames,allow_exact_pose_anchors=True,**arguments)
