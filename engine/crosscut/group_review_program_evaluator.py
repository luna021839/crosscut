"""A frozen, shared editorial REVIEW objective, never professional approval.

Geometry is style, not candidate eligibility. Required original polyline
support is a separate three-state contract. Native pixel features, anonymous
source-local tracks and two incident contexts remain separate evidence.
"""
from dataclasses import asdict, dataclass, field
import json
import math

from .group_original_visible_support import ARM_INDICES, support_preserved_coordinates
from .group_performer_family_owner import digest
from .group_review_camera_repair import (effective_face_detail_pixels,
    geometry_cost, shape_cost)

CONTRACT='shared_frozen_measured_group_review_objective_v3'


@dataclass(frozen=True,slots=True)
class BoundReviewPartDuties:
    """Immutable snapshot with one root hash, not a hash of the film per tick."""
    token_json:str
    identity:str=field(init=False)
    first_tick:int=field(init=False)
    row_json:tuple[str,...]=field(init=False,repr=False)

    def __post_init__(self):
        if not isinstance(self.token_json,str) or len(self.token_json)>32*1024*1024:
            raise ValueError('bound review duties need a bounded serialized snapshot')
        token=json.loads(self.token_json)
        if token.get('contract')!='frozen_original_review_part_duties_v1' or not isinstance(token.get('rows'),dict):
            raise ValueError('bound review original duties need the explicit frozen contract')
        ticks=sorted(int(t) for t in token['rows'])
        if (not ticks or len(ticks)>16384 or ticks[0]<0
                or ticks!=list(range(ticks[0],ticks[-1]+1))
                or set(token['rows'])!={str(t) for t in ticks}):
            raise ValueError('bound review duty snapshot needs exact contiguous output ticks')
        rows=tuple(json.dumps(token['rows'][str(t)],sort_keys=True,separators=(',',':'),allow_nan=False) for t in ticks)
        object.__setattr__(self,'identity',digest(token))
        object.__setattr__(self,'first_tick',ticks[0])
        object.__setattr__(self,'row_json',rows)

    def row(self,tick):
        index=tick-self.first_tick
        return None if type(tick) is not int or not 0<=index<len(self.row_json) else json.loads(self.row_json[index])


def bind_review_part_duties(token):
    return token if isinstance(token,BoundReviewPartDuties) else BoundReviewPartDuties(
        json.dumps(token,sort_keys=True,separators=(',',':'),allow_nan=False))


@dataclass(frozen=True,slots=True)
class ReviewObjective:
    readability_weight:float=3.5
    seam_mean_weight:float=.40
    seam_worst_weight:float=.35
    pose_weight:float=.30
    missing_weight:float=2.5
    ambiguity_weight:float=.45
    blur_weight:float=.35
    temporal_weight:float=.40
    intervention_weight:float=.06
    reference_detail_pixels:float=100.
    maximum_shortlist_sources:int=2
    minimum_appearance_cosine:float=.12
    appearance_ambiguity_margin:float=.025

    def __post_init__(self):
        values=asdict(self)
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values())
                or self.reference_detail_pixels<=0 or not 1<=self.maximum_shortlist_sources<=2
                or type(self.maximum_shortlist_sources) is not int):
            raise ValueError('review objective requires explicit finite nonnegative weights and bounded work')

    def to_dict(self):
        return dict(contract=CONTRACT,**asdict(self),
            soft_geometry_is_eligibility=False,unknown_imputed_as_zero=False,
            missing_policy='explicit_penalty_and_unknown_metric_not_dropped_rows',
            tie_policy='content_native_clock_identity_then_camera_digest',
            calibrated_expert_quality_predictor=False,quality_approved=False)

    @property
    def identity(self):return digest(self.to_dict())


def consume_frozen_part_duties(token,measurement,*,candidate_head_ids,key):
    """Consume EVERY remaining anonymous branch, not convenient donor limbs.

    A token is independently frozen by its producer before donor sensing.
    Known required support may be full or partial; None means explicit full.
    Empty measurements never prove a known duty. Source motion UNKNOWN alone
    is not a reason to reject a review. This consumer is not a cast solver.
    """
    bound=isinstance(token,BoundReviewPartDuties)
    if not bound and token.get('contract')!='frozen_original_review_part_duties_v1':
        raise ValueError('review original duties need the explicit frozen contract')
    raw=measurement['native_observation'];tick=measurement['output_tick']
    required=token.row(tick) if bound else token['rows'].get(str(tick))
    if required is None:raise ValueError('original duty token does not own this output tick')
    projected={p['observation_id']:p for p in measurement['projection']['people']}
    transform=lambda p:(.5+(p[0]-key.center_x)*key.zoom,.5+(p[1]-key.center_y)*key.zoom)
    inside=lambda p:all(-1e-9<=v<=1+1e-9 for v in p)
    failures=[];unknown=[]
    if required['head_required'] and not candidate_head_ids:unknown.append('required_head_missing_or_unmatched')
    for identity in candidate_head_ids:
        heads=[h for h in measurement['projection']['heads'] if h['observation_id']==identity]
        if required['head_required'] and not heads:
            unknown.append('required_head_unobserved');continue
        if required['head_required']:
            x,y,w,h=heads[0]['projected_box_xywh']
            if heads[0]['visible_fraction']<.995 or not all(inside(transform(p)) for p in ((x,y),(x+w,y+h))):
                failures.append('required_face_roi_crop')
        owners=[p for p in raw['people'] if p.get('unique_head_observation_id')==identity]
        for name,support in required['parts'].items():
            if len(owners)!=1 or dict(owners[0]['part_states']).get(name)!='observed':
                unknown.append('required_'+name+'_observation_or_ownership_unknown');continue
            p=owners[0];indices=ARM_INDICES[name]
            points=[p['upper_body_xyc'][i][:2] for i in indices]
            projected_points=[transform(projected[p['observation_id']]['projected_upper_body_xyc'][i][:2]) for i in indices]
            if support is None:
                # Explicit whole arm is not purchased with inferred joints.
                if any(not all(-1e-9<=v<=1+1e-9 for v in q) for q in (*points,*projected_points)):
                    failures.append('explicit_required_'+name+'_outside')
            elif not support_preserved_coordinates(support,native_points=points,
                    projected_points=projected_points):
                failures.append('original_visible_'+name+'_support_loss')
    if required['parts'] and not candidate_head_ids:unknown.append('required_arm_correspondence_unknown')
    state='FAIL' if failures else 'UNKNOWN' if unknown else 'PASS'
    return dict(state=state,failures=failures,unknown=unknown,token_sha256=token.identity if bound else digest(token),
        output_tick=tick,remaining_candidate_branches=list(candidate_head_ids),
        silhouette_or_occlusion_certified=False,quality_approved=False)


def freeze_empty_part_duties(*,original_role_program,first,end):
    """Explicit nonfocal part scope; composition inventory remains separate.

    This cannot erase a focal/formation role. The current bounded review owns
    canonical COMPOSITION units only; this token certifies no anatomical part
    duty, not matched original cast or normal composition admission.
    """
    roles={e['role'] for e in original_role_program['epochs']}
    if roles!={'observed_body_composition'}:
        raise ValueError('nonfocal review cannot discard original focal/formation duties')
    return dict(contract='frozen_original_review_part_duties_v1',
        original_role_program_sha256=digest(original_role_program),
        rows={str(t):dict(head_required=False,parts={}) for t in range(first,end)},
        anatomical_parts_required=False,original_composition_inventory_certified=False,
        original_motion_certified=False,quality_approved=False)


def _longest_missing(values):
    longest=current=0
    for value in values:
        current=current+1 if value is None else 0;longest=max(longest,current)
    return longest


def evaluate_review_program(*,objective,descriptors,pixel_features,left,right,
        support_verdicts,association,authority,intervention=0.):
    if (not isinstance(objective,ReviewObjective) or not descriptors
            or len(descriptors)!=len(pixel_features) or len(descriptors)!=len(support_verdicts)
            or authority not in ('recorded_nonauthoritative','selected_fresh','parent_fresh','final_fresh')):
        raise ValueError('shared review evaluation needs bounded aligned observations and explicit authority')
    states=[v['state'] for v in support_verdicts]
    if not set(states)<={'PASS','FAIL','UNKNOWN'}:raise ValueError('invalid original duty verdict')
    hard='FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'PASS'
    details=[effective_face_detail_pixels(d) for d in descriptors]
    missing=[d is None for d in details]
    observed=[d for d in details if d is not None]
    fraction=sum(missing)/len(details);longest=_longest_missing(details)/len(details)
    # Missing values stay None in the metric. The explicitly declared risk
    # penalty is NOT an invented perfect measurement or certified absence.
    readability=None if not observed else sum(max(0.,1.-min(d,objective.reference_detail_pixels)/
        objective.reference_detail_pixels)**2 for d in observed)/len(observed)
    costs=(geometry_cost(left,descriptors[0]),geometry_cost(descriptors[-1],right))
    poses=(shape_cost(left,descriptors[0]),shape_cost(descriptors[-1],right))
    known=[v for v in costs if v is not None];known_pose=[v for v in poses if v is not None]
    blur=[f.get('detail_risk') for f in pixel_features if f is not None and f.get('detail_risk') is not None]
    centers=[None if d is None else d.head_center for d in descriptors]
    jitter=[]
    for a,b,c in zip(centers,centers[1:],centers[2:]):
        if a is not None and b is not None and c is not None:
            jitter.append(math.hypot(c[0]-2*b[0]+a[0],(c[1]-2*b[1]+a[1])*9/16))
    uncertainty=(costs.count(None)+poses.count(None))/4+association.get('ambiguity_risk',1.)
    score=(objective.readability_weight*(1. if readability is None else readability)
        +objective.seam_mean_weight*(sum(known)/len(known) if known else 0.)
        +objective.seam_worst_weight*(max(known) if known else 0.)
        +objective.pose_weight*(sum(known_pose)/len(known_pose) if known_pose else 0.)
        +objective.missing_weight*(fraction+longest)
        +objective.ambiguity_weight*uncertainty
        +objective.blur_weight*(sum(blur)/len(blur) if blur else 1.)
        +objective.temporal_weight*(sum(jitter)/len(jitter) if jitter else 0.)
        +objective.intervention_weight*intervention)
    return dict(contract=CONTRACT,objective_sha256=objective.identity,authority=authority,
        hard_original_part_state=hard,eligible_review=hard=='PASS',soft_rank=score,
        incident_seam_costs=list(costs),incident_pose_costs=list(poses),
        worst_observed_seam=None if not known else max(known),
        effective_detail_pixels=details,readability_risk=readability,
        actor_missing_fraction=fraction,longest_actor_missing_fraction=longest,
        native_patch_detail_risk=None if not blur else sum(blur)/len(blur),
        temporal_second_difference=None if not jitter else sum(jitter)/len(jitter),
        association=association,unknown_imputed_as_zero=False,
        style_geometry_is_hard_gate=False,cross_source_identity_certified=False,
        motion_nonregression_certified=False,normal_candidate_admitted=False,quality_approved=False)
