"""Finite whole-edit search over already REALIZED review variants.

This module neither infers people nor changes a camera after selection.
Original ownership, actual native evidence and final quality remain separate.
"""
from dataclasses import dataclass,field
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
    _identity: str=field(init=False,repr=False)

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
        object.__setattr__(self,'_identity',digest(dict(first=self.first,end=self.end,payload=value)))

    @property
    def payload(self):return json.loads(self.payload_json)

    @property
    def identity(self):
        return self._identity

    @property
    def evaluation_identity(self):
        return digest(dict(physical_variant=self.identity,utility=self.utility,
            changed_frames=self.changed_frames,policy='near4.5_blur0.45_path0.5_duration_v1'))

    def assert_unchanged(self,payload):
        if digest(payload)!=digest(self.payload):
            raise ValueError('selected realized camera/sampler/actor payload changed after selection')


def validate_restored_review_variant(variant,metadata,selected,*,camera_identity,sampler_identity,
        expected_source_identity,output_frame_count):
    """Recompute what the consumer uses, not only the duplicated JSON hash."""
    payload=variant.payload
    if (metadata.get('bindings')!=payload.get('bindings')
            or metadata.get('source')!=payload.get('source')
            or metadata.get('phase')!=payload.get('phase_frames')
            or metadata.get('keep')!=payload.get('keep')
            or tuple(metadata.get('phase_scope',()))!=(max(0,variant.first-6),
                min(output_frame_count,variant.end+6))):
        raise ValueError('cached model metadata disagrees with sealed actor/source/phase/scope')
    bindings=payload['bindings'];ticks=tuple(b['output_tick'] for b in bindings)
    if tuple(sorted(set(ticks)))!=ticks or tuple(selected)!=ticks:
        raise ValueError('restored actor binding inventory changed')
    for binding in bindings:
        view=selected[binding['output_tick']]
        if binding['state']=='UNMATCHED':
            if view is not None or binding['head_id'] is not None or binding['native_frame'] is not None:
                raise ValueError('restored unmatched actor changed to a selected occurrence')
        elif (binding['state']!='HEAD_OCCURRENCE' or view is None
                or view['descriptor'].frame_local_head_id!=binding['head_id']
                or view['row']['native_observation']['native_frame_index']!=binding['native_frame']):
            raise ValueError('restored selected actor changed its sealed occurrence/native parent')
    actor_identity=digest([(t,None if v is None else v['descriptor'].frame_local_head_id,
        None if v is None else digest(v['row']['native_observation'])) for t,v in selected.items()])
    if (actor_identity!=payload['actor_path_identity'] or camera_identity!=payload['camera_identity']
            or sampler_identity!=payload['sampler_identity'] or expected_source_identity!=payload['source_identity']):
        raise ValueError('restored actual actor/raw/camera/sampler identity disagrees with sealed payload')
    near=sum(0. if v is None else v['near'] for v in selected.values())/len(selected)
    blur=sum(1. if v is None or v['detail_risk'] is None else v['detail_risk'] for v in selected.values())/len(selected)
    path=payload['actor_path'];path_cost=0. if path is None else path['cost']/len(selected)
    # FPS is supplied in metadata by the restoring caller; it is checked
    # against the immutable plan, never accepted from the cached score.
    fps=metadata.get('validated_plan_fps')
    if type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0:
        raise ValueError('restored evaluation requires the actual plan clock')
    utility=(variant.end-variant.first)/fps*(4.5*near-.45*blur-.5*path_cost)
    if (abs(near-payload['near_fraction'])>1e-9 or abs(utility-variant.utility)>1e-8
            or variant.changed_frames!=(0 if payload['keep'] else variant.end-variant.first)
            or metadata.get('evaluation_identity',variant.evaluation_identity)!=variant.evaluation_identity):
        raise ValueError('cached selection utility or changed-frame accounting changed')
    return dict(actual_restored_actor_verified=True,evaluation_identity=variant.evaluation_identity,
        policy='near4.5_blur0.45_path0.5_duration_v1',legacy_evaluation_recomputed=
            'evaluation_identity' not in metadata,quality_approved=False)


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


def require_paired_motion_adoption(*,comparison,expected_manifest=None,keyed_receipts=None,
        support_states=None,motion_verdicts=None):
    """Only a complete plan-bound keyed comparison can permit review encoding.

    Legacy generic strings cannot prove inventory, original UNKNOWN provenance
    or an unchanged sampler/camera/actor. Keep the argument names solely to
    give old callers an explicit rejection, never a permissive compatibility
    fallback. Numeric review encoding still does not establish expert quality.
    """
    from .group_review_adoption_manifest import decide_keyed_review_adoption
    if (expected_manifest is None or keyed_receipts is None or support_states is not None
            or motion_verdicts is not None):
        raise ValueError('paired adoption needs explicit measured keyed receipts and a frozen manifest; '
            'generic verdict strings cannot permit encode')
    decision=decide_keyed_review_adoption(manifest=expected_manifest,receipts=keyed_receipts,
        comparison=comparison)
    if decision['state']!='REVIEW_ENCODE_ELIGIBLE':
        raise ValueError('paired motion/support comparison does not permit encode: '+decision['state'])
    return decision
