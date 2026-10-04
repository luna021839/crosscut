"""Finite whole-edit search over already REALIZED review variants.

This module neither infers people nor changes a camera after selection.
Original ownership, actual native evidence and final quality remain separate.
"""
from dataclasses import dataclass
import json
import math

from .group_performer_family_owner import digest


@dataclass(frozen=True, slots=True)
class RealizedReviewVariant:
    first: int
    end: int
    payload_json: str
    utility: float
    changed_frames: int

    def __post_init__(self):
        if (type(self.first) is not int or type(self.end) is not int
                or not 0<=self.first<self.end<=36000
                or type(self.changed_frames) is not int
                or not 0<=self.changed_frames<=self.end-self.first
                or type(self.utility) not in (int,float) or not math.isfinite(self.utility)):
            raise ValueError('realized variant needs finite exact output bounds')
        value=json.loads(self.payload_json)
        if (not isinstance(value,dict) or value.get('quality_approved') is not False
                or not all(k in value for k in
                    ('camera_identity','sampler_identity','actor_path_identity','source_identity'))):
            raise ValueError('realized variant must bind its actual camera, sampler and actor path')

    @property
    def payload(self):return json.loads(self.payload_json)

    @property
    def identity(self):
        return digest(dict(first=self.first,end=self.end,payload=self.payload))

    def assert_unchanged(self,payload):
        if digest(payload)!=digest(self.payload):
            raise ValueError('selected realized camera/sampler/actor payload changed after selection')


def select_realized_review_path(layers,edge_at,*,output_frame_count,
        maximum_changed_frames=1600,maximum_beam=48,maximum_states=16000,
        maximum_edges=60000,work_port=None):
    """Choose one compatible path; boundary ticks are part of each variant.

    An edge is measured on these exact left/right cameras. UNKNOWN may remain
    exploratory, but is never converted to measured quality or admission.
    """
    if (not isinstance(layers,(list,tuple)) or not 1<=len(layers)<=256
            or any(not isinstance(row,(list,tuple)) or not row or len(row)>256 for row in layers)
            or any(not isinstance(v,RealizedReviewVariant) for row in layers for v in row)
            or any(type(v) is not int or not 1<=v<=cap for v,cap in
                ((output_frame_count,36000),(maximum_changed_frames,16384),
                 (maximum_beam,64),(maximum_states,20000),(maximum_edges,100000)))):
        raise ValueError('realized whole-path search needs finite explicit bounds')
    beam=[dict(end=0,changed=0,utility=0.,variants=(),edges=(),tie=())]
    cache={};states=0;edges=0;dominated=0
    for layer in layers:
        new=[]
        for parent in beam:
            for variant in layer:
                if parent['end']!=variant.first:continue
                changed=parent['changed']+variant.changed_frames
                if changed>maximum_changed_frames:continue
                states+=1
                if states>maximum_states:raise ValueError('realized selection state budget exceeded')
                if work_port is not None:work_port.charge('states')
                edge=None
                if parent['variants']:
                    left=parent['variants'][-1];key=(left.identity,variant.identity,variant.first)
                    if key not in cache:
                        edges+=1
                        if edges>maximum_edges:raise ValueError('realized selection edge budget exceeded')
                        if work_port is not None:work_port.charge('edges')
                        value=edge_at(left,variant)
                        if (not isinstance(value,dict) or type(value.get('allowed')) is not bool
                                or value.get('left_identity')!=left.identity
                                or value.get('right_identity')!=variant.identity
                                or value.get('boundary_tick')!=variant.first
                                or type(value.get('cost')) not in (int,float)
                                or not math.isfinite(value['cost']) or value['cost']<0):
                            raise ValueError('realized edge is not bound to its actual neighbors')
                        cache[key]=value
                    edge=cache[key]
                    if not edge['allowed']:continue
                score=parent['utility']+variant.utility-(0. if edge is None else edge['cost'])
                new.append(dict(end=variant.end,changed=changed,utility=score,
                    variants=parent['variants']+(variant,),edges=parent['edges']+(() if edge is None else (edge,)),
                    tie=parent['tie']+(variant.identity,)))
        # All future costs depend on the actual last variant and remaining
        # changed-frame budget. Keeping many worse prefixes of that identical
        # state wastes the finite beam and hides genuinely different cameras.
        unique={}
        for p in new:
            key=(p['variants'][-1].identity,p['changed'])
            old=unique.get(key)
            if old is None or (-p['utility'],p['tie'])<(-old['utility'],old['tie']):unique[key]=p
        dominated+=len(new)-len(unique)
        new=list(unique.values())
        new.sort(key=lambda p:(-p['utility'],p['changed'],p['tie']))
        beam=new[:maximum_beam]
        if not beam:raise ValueError('no compatible realized path; do not synthesize a passing edit')
    finalists=[p for p in beam if p['end']==output_frame_count]
    if not finalists:raise ValueError('realized path omitted the final output frames')
    best=finalists[0]
    return dict(variants=best['variants'],edges=best['edges'],utility=best['utility'],
        changed_frames=best['changed'],visited_states=states,measured_edges=edges,
        dominated_identical_suffix_states_removed=dominated,
        numerical_realized_path_only=True,quality_approved=False)


def incident_tick_union(first,end,original_cut_ticks,candidate_cut_ticks,*,output_frame_count):
    """Entry, EXIT and every original/new event use one common incident set."""
    if (any(type(t) is not int for t in (first,end,output_frame_count))
            or not 0<=first<end<=output_frame_count):
        raise ValueError('incident review needs an exact bounded interval')
    cuts=tuple(original_cut_ticks)+tuple(candidate_cut_ticks)
    if any(type(t) is not int or not 0<t<output_frame_count for t in cuts):
        raise ValueError('incident review borrowed an absent actual cut')
    return tuple(sorted({t for t in (first,end,*cuts)
        if first<=t<=end and 0<t<output_frame_count}))


def require_paired_motion_adoption(*,comparison,support_states,motion_verdicts):
    """A consumer gate, not a quality badge. Near duration cannot pay motion."""
    if (comparison.get('verdict') not in ('BETTER','WORSE','INCONCLUSIVE')
            or any(s not in ('PASS','FAIL','UNKNOWN') for s in support_states)
            or any(s not in ('NONREGRESSING','REGRESSING','UNMEASURED','UNCHANGED')
                for s in motion_verdicts)):
        raise ValueError('paired adoption needs explicit measured verdicts')
    if comparison['verdict']!='BETTER' or 'FAIL' in support_states or any(
            s in ('REGRESSING','UNMEASURED') for s in motion_verdicts):
        raise ValueError('paired motion/support comparison does not permit encode')
    return dict(numeric_review_encode_permitted=True,
        unresolved_original_support=sum(s=='UNKNOWN' for s in support_states),
        quality_approved=False,normal_complete_bundle_created=False)
