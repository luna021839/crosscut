"""Time-specific original actor obligations, separate from mutual donors.

An anonymous continuation is not preservation of a known original actor.
The sealed original occurrences precede candidate body or camera costs.
"""
from dataclasses import dataclass
import json
import math
from .group_performer_family_owner import digest
from .group_review_native_window import cosine


def _sha(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


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
                    or relation=='EXPLICIT_HANDOFF' and not _sha(r.get('handoff_declaration_identity'))
                    or relation=='ANONYMOUS_CONTINUATION' and r.get('original_actor_known') is not False):
                raise ValueError('anonymous continuity cannot erase a known actor or invent a handoff')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_actor_relation_duty(*,requests,original_actor_inventory_identity,
        independent_control_identity,minimum_cosine):
    return FrozenReviewActorRelationDuty(json.dumps(dict(
        contract='predeclared_original_candidate_actor_relation_v1',requests=list(requests),
        original_actor_inventory_identity=original_actor_inventory_identity,
        independent_control_identity=independent_control_identity,minimum_cosine=minimum_cosine,
        candidate_body_costs_used=False,named_actor_certified=False,quality_approved=False),
        sort_keys=True,allow_nan=False))


def original_actor_relation_request(tick,view,*,relation='CONTINUE_KNOWN_ORIGINAL',
        handoff_declaration_identity=None):
    if view is None:
        return dict(tick=tick,relation='UNRESOLVED',original_actor_known=False,
            original_occurrence_identity=None,original_vector=None)
    if view['row']['output_tick']!=tick:raise ValueError('original actor relation borrowed another instant')
    return dict(tick=tick,relation=relation,original_actor_known=True,
        original_occurrence_identity=digest(dict(raw=view['row']['native_observation'],
            selected_head=view['descriptor'].frame_local_head_id)),
        original_vector=list(view['vector']),handoff_declaration_identity=handoff_declaration_identity)


def evaluate_review_actor_relation(*,duty,observations,anonymous_continuation_state='UNKNOWN'):
    if not isinstance(duty,FrozenReviewActorRelationDuty):raise ValueError('actor relation requires its frozen duty')
    p=duty.payload;requests=p['requests']
    if not isinstance(observations,dict) or set(observations)!={r['tick'] for r in requests}:
        raise ValueError('candidate actor relation omitted or borrowed a requested output instant')
    rows=[]
    for r in requests:
        v=observations[r['tick']];sim=None
        if v is None or v.get('reacquisition_ambiguous',False):state='UNKNOWN'
        elif v['row']['output_tick']!=r['tick']:raise ValueError('actor relation borrowed another candidate instant')
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
    return dict(contract='actual_original_candidate_actor_relation_receipt_v1',
        duty_identity=duty.identity,state='FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'PASS',
        requests=rows,known_original_actor_preserved=all(r['state']=='PASS' for r in rows
            if r['relation'] in ('CONTINUE_KNOWN_ORIGINAL','EXPLICIT_HANDOFF')),
        anonymous_continuation_is_known_original_preservation=False,named_actor_certified=False,
        motion_suitability_certified=False,quality_approved=False)
