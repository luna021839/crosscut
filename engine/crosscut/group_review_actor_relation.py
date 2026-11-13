"""Time-specific original actor obligations, separate from mutual donors.

An anonymous continuation is not preservation of a known original actor.
The sealed original occurrences precede candidate body or camera costs.
"""
from dataclasses import dataclass
import json
import math
from .group_performer_family_owner import digest
from .group_review_native_window import cosine
from .group_review_motion_seam import review_actor_association_valid,review_window_connected


def _sha(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


@dataclass(frozen=True,slots=True)
class FrozenReviewFocalHandoff:
    token_json:str

    def __post_init__(self):
        p=self.payload
        if not isinstance(p,dict):raise ValueError('handoff declaration is absent')
        interval=p.get('transition_interval')
        if (p.get('contract')!='independently_declared_focal_handoff_v1'
                or not all(_sha(p.get(k)) for k in ('from_focal_scope','to_focal_scope',
                    'original_actor_inventory_identity','original_preservation_duty_identity'))
                or p['from_focal_scope']==p['to_focal_scope']
                or not isinstance(interval,list) or len(interval)!=2
                or any(type(t) is not int or t<0 for t in interval) or interval[0]>=interval[1]
                or not isinstance(p.get('editorial_reason'),str) or not p['editorial_reason']
                or p.get('permitted_transition_relation')!='EXPLICIT_FOCAL_CHANGE'):
            raise ValueError('handoff requires distinct focal scopes, interval and original obligations')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def _valid_handoff(request):
    try:declaration=FrozenReviewFocalHandoff(json.dumps(request['handoff_declaration']))
    except (KeyError,TypeError,ValueError):return False
    p=declaration.payload
    return (request.get('handoff_declaration_identity')==declaration.identity
        and p['transition_interval'][0]<=request['tick']<p['transition_interval'][1])


@dataclass(frozen=True,slots=True)
class FrozenReviewActorRelationDuty:
    token_json:str

    def __post_init__(self):
        p=self.payload;rows=p.get('requests')
        if (p.get('contract')!='predeclared_original_candidate_actor_relation_v1'
                or not _sha(p.get('original_actor_inventory_identity'))
                or not _sha(p.get('independent_control_identity'))
                or not isinstance(rows,list) or not 1<=len(rows)<=2048
                or any(type(r.get('tick')) is not int or r['tick']<0 for r in rows)
                or any(a['tick']>=b['tick'] for a,b in zip(rows,rows[1:]))
                or type(p.get('minimum_cosine')) not in (int,float)
                or not math.isfinite(p['minimum_cosine']) or not -1<=p['minimum_cosine']<=1):
            raise ValueError('actor relation needs exact predeclared time and independent original/control roots')
        for r in rows:
            relation=r.get('relation')
            if (relation not in ('CONTINUE_KNOWN_ORIGINAL','ANONYMOUS_CONTINUATION','EXPLICIT_HANDOFF','UNRESOLVED')
                    or relation in ('CONTINUE_KNOWN_ORIGINAL','EXPLICIT_HANDOFF') and
                    (not _sha(r.get('original_occurrence_identity')) or not isinstance(r.get('original_vector'),list)
                        or len(r['original_vector'])!=128 or any(type(v) not in (int,float)
                            or not math.isfinite(v) for v in r['original_vector']))
                    or relation=='EXPLICIT_HANDOFF' and (not _valid_handoff(r)
                        or r['handoff_declaration']['original_actor_inventory_identity']!=
                            p['original_actor_inventory_identity'])
                    or relation=='ANONYMOUS_CONTINUATION' and r.get('original_actor_known') is not False):
                raise ValueError('anonymous continuity cannot erase a known actor or invent a handoff')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_actor_relation_duty(*,requests,original_actor_inventory_identity,
        independent_control_identity,minimum_cosine,independently_declared_anchor_tick=None):
    if independently_declared_anchor_tick is not None and independently_declared_anchor_tick not in [r['tick'] for r in requests]:
        raise ValueError('actor relation anchor is outside the independently declared original requests')
    return FrozenReviewActorRelationDuty(json.dumps(dict(
        contract='predeclared_original_candidate_actor_relation_v1',requests=list(requests),
        original_actor_inventory_identity=original_actor_inventory_identity,
        independent_control_identity=independent_control_identity,minimum_cosine=minimum_cosine,
        independently_declared_anchor_tick=independently_declared_anchor_tick,
        candidate_body_costs_used=False,named_actor_certified=False,quality_approved=False),
        sort_keys=True,allow_nan=False))


def original_actor_relation_request(tick,view,*,relation='CONTINUE_KNOWN_ORIGINAL',
        handoff_declaration_identity=None,handoff_declaration=None):
    if view is not None and view['row']['output_tick']!=tick:
        raise ValueError('original actor relation borrowed another instant')
    if (not review_actor_association_valid(view) or view.get('original_role') in
            ('required_formation','observed_body_composition')):
        return dict(tick=tick,relation='UNRESOLVED',original_actor_known=False,
            original_occurrence_identity=None,original_vector=None)
    return dict(tick=tick,relation=relation,original_actor_known=True,
        original_occurrence_identity=digest(dict(raw=view['row']['native_observation'],
            selected_head=view['descriptor'].frame_local_head_id)),
        original_vector=list(view['vector']),handoff_declaration_identity=handoff_declaration_identity,
        handoff_declaration=None if handoff_declaration is None else handoff_declaration.payload)


def evaluate_review_actor_relation(*,duty,observations,anonymous_continuation_state='UNKNOWN'):
    if not isinstance(duty,FrozenReviewActorRelationDuty):raise ValueError('actor relation requires its frozen duty')
    p=duty.payload;requests=p['requests']
    if not isinstance(observations,dict) or set(observations)!={r['tick'] for r in requests}:
        raise ValueError('candidate actor relation omitted or borrowed a requested output instant')
    rows=[]
    for r in requests:
        v=observations[r['tick']];sim=None
        if v is not None and v['row']['output_tick']!=r['tick']:
            raise ValueError('actor relation borrowed another candidate instant')
        if not review_actor_association_valid(v):state='UNKNOWN'
        elif r['relation']=='UNRESOLVED':state='UNKNOWN'
        elif r['relation']=='ANONYMOUS_CONTINUATION':state=anonymous_continuation_state
        else:
            sim=cosine(r['original_vector'],v['vector'])
            state='UNKNOWN' if sim is None else 'PASS' if sim>=p['minimum_cosine'] else 'FAIL'
        if state not in ('PASS','FAIL','UNKNOWN'):raise ValueError('anonymous relation needs an actual three-state receipt')
        rows.append(dict(tick=r['tick'],relation=r['relation'],state=state,appearance_cosine=sim,
            original_occurrence_identity=r.get('original_occurrence_identity'),
            candidate_occurrence_identity=None if v is None else digest(dict(raw=v['row']['native_observation'],
                head=v['descriptor'].frame_local_head_id))))
    states=[r['state'] for r in rows]
    known=[r for r in rows if r['relation'] in ('CONTINUE_KNOWN_ORIGINAL','EXPLICIT_HANDOFF')]
    return dict(contract='actual_original_candidate_actor_relation_receipt_v1',
        duty_identity=duty.identity,state='FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'PASS',
        requests=rows,known_original_request_count=len(known),
        known_original_actor_preserved=bool(known) and all(r['state']=='PASS' for r in known),
        known_original_preservation_state='NOT_APPLICABLE' if not known else
            'FAIL' if any(r['state']=='FAIL' for r in known) else
            'UNKNOWN' if any(r['state']=='UNKNOWN' for r in known) else 'PASS',
        anonymous_continuation_is_known_original_preservation=False,named_actor_certified=False,
        motion_suitability_certified=False,quality_approved=False)


def evaluate_review_control_anchored_actor_relation(*,duty,original_observations,observations,
        original_anchor_alternatives,candidate_anchor_alternatives,ambiguity_margin=.025):
    """One controlled cross-source anchor plus two COMPLETE local paths.

    Re-comparing frontal identity embeddings between broadcasters at every
    profile/occluded frame is not the same test as preserving a continuously
    anchored actor. This is explicitly a review correspondence hypothesis, not
    a named identity or an independent pixel-flow claim. Missing/ambiguous
    original bindings or any broken path remain UNKNOWN.
    """
    if not isinstance(duty,FrozenReviewActorRelationDuty):raise ValueError('anchored relation requires its predeclared duty')
    p=duty.payload;requests=p['requests'];ticks=[r['tick'] for r in requests]
    if set(original_observations)!=set(ticks) or set(observations)!=set(ticks):
        raise ValueError('anchored actor relation omitted a time-specific original/candidate request')
    anchor=p.get('independently_declared_anchor_tick')
    if anchor not in ticks or not 0<=ambiguity_margin<=1:
        raise ValueError('anchored actor relation needs an independent original anchor, not a favourable score')
    result=dict(contract='independent_control_anchor_plus_complete_native_actor_paths_v1',state='UNKNOWN',
        duty_identity=duty.identity,anchor_tick=anchor,known_original_actor_preserved=False,
        known_original_request_count=sum(r['relation']=='CONTINUE_KNOWN_ORIGINAL' for r in requests),
        correspondence_is_review_hypothesis=True,pixel_flow_certified=False,named_actor_certified=False,
        motion_suitability_certified=False,quality_approved=False)
    for r in requests:
        v=original_observations[r['tick']];candidate=observations[r['tick']]
        if r['relation']!='CONTINUE_KNOWN_ORIGINAL' or not review_actor_association_valid(v):return result
        actual=digest(dict(raw=v['row']['native_observation'],selected_head=v['descriptor'].frame_local_head_id))
        if (v['row']['output_tick']!=r['tick'] or actual!=r['original_occurrence_identity']
                or list(v['vector'])!=r['original_vector']):
            raise ValueError('anchored relation borrowed another frozen original occurrence')
        if candidate is not None and candidate['row']['output_tick']!=r['tick']:
            raise ValueError('anchored relation borrowed another candidate instant')
    a=[original_observations[t] for t in ticks];b=[observations[t] for t in ticks]
    if not review_window_connected(a) or not review_window_connected(b):
        return dict(result,reason='complete_native_actor_paths_unresolved')
    original=original_observations[anchor];candidate=observations[anchor]
    def binding(v):return digest((v['row']['native_observation'],v['descriptor'].frame_local_head_id,list(v['vector'])))
    def alternatives(values,selected):
        if (not isinstance(values,(list,tuple)) or not 1<=len(values)<=24
                or any(v['row']['output_tick']!=anchor or v['row']['native_observation']!=selected['row']['native_observation'] for v in values)
                or binding(selected) not in {binding(v) for v in values}):
            raise ValueError('actor anchor rivals borrowed another raster or omitted the selected occurrence')
        return values
    left=alternatives(original_anchor_alternatives,original);right=alternatives(candidate_anchor_alternatives,candidate)
    similarity=cosine(original['vector'],candidate['vector'])
    rival_scores=[cosine(original['vector'],v['vector']) for v in right if binding(v)!=binding(candidate)]
    rival_scores += [cosine(candidate['vector'],v['vector']) for v in left if binding(v)!=binding(original)]
    measured_rivals=[s for s in rival_scores if s is not None]
    margin=None if similarity is None or not measured_rivals else similarity-max(measured_rivals)
    result.update(anchor_cosine=similarity,anchor_rival_margin=margin,
        original_complete_path_identity=digest([binding(v) for v in a]),
        candidate_complete_path_identity=digest([binding(v) for v in b]))
    if similarity is None or similarity<p['minimum_cosine']:return dict(result,state='FAIL',reason='independent_identity_anchor_failed')
    if margin is not None and margin<ambiguity_margin:return dict(result,reason='anchor_rivals_unresolved')
    return dict(result,state='PASS',known_original_actor_preserved=True,
        per_time_original_occurrences_consumed=True,all_native_actor_path_edges_consumed=True,
        direct_every_frame_cross_source_frontal_embedding_test=False)
