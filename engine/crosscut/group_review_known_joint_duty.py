"""Original visible joints remain obligations outside a match-cut window.

Cut-local action declarations do not authorize dropping an original wrist in
the rest of an edited shot. This inventory is frozen from original views only;
the candidate must supply the same known joint classes at each requested tick.
This is presence/containment evidence, not joint phase or actor identity proof.
"""
from dataclasses import dataclass
import json
import math
from .group_performer_family_owner import digest


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True,slots=True)
class FrozenReviewKnownJointDuty:
    token_json:str

    def __post_init__(self):
        p=self.payload;rows=p.get('requests')
        if (p.get('contract')!='original_time_specific_known_joint_presence_duty_v1'
                or not _sha(p.get('original_actor_inventory_identity'))
                or not _sha(p.get('original_preservation_duty_identity'))
                or not isinstance(rows,list) or not 1<=len(rows)<=2048
                or any(type(r.get('tick')) is not int or r['tick']<0 for r in rows)
                or any(a['tick']>=b['tick'] for a,b in zip(rows,rows[1:]))):
            raise ValueError('known joint duty requires the independently frozen original inventory')
        for r in rows:
            joints=r.get('known_joint_indices')
            if (r.get('original_state') not in ('OBSERVED','UNRESOLVED')
                    or not isinstance(joints,list) or joints!=sorted(set(joints))
                    or any(type(i) is not int or not 0<=i<6 for i in joints)
                    or r['original_state']=='OBSERVED' and not _sha(r.get('original_occurrence_identity'))):
                raise ValueError('known joint duty cannot borrow candidate visibility or erase an original observation')
            if (r.get('unresolved_joint_indices')!=sorted(set(range(6))-set(joints))
                    or r.get('original_actor_state') not in ('OBSERVED','UNRESOLVED')
                    or r['original_state']=='OBSERVED' and not joints):
                raise ValueError('known joint duty cannot promote an empty or unresolved anatomical fact')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_known_joint_duty(original_views,*,original_actor_inventory_identity,
        original_preservation_duty_identity):
    rows=[]
    for tick,v in sorted(original_views.items()):
        if v is not None and v['row']['output_tick']!=tick:
            raise ValueError('original joint duty borrowed another output instant')
        shape=None if v is None else v['descriptor'].pose_shape
        known=sorted({p[0] for p in shape or ()})
        rows.append(dict(tick=tick,original_state='OBSERVED' if known else 'UNRESOLVED',
            original_actor_state='UNRESOLVED' if v is None or v.get('reacquisition_ambiguous',False)
                or v.get('original_role') in ('required_formation','observed_body_composition') else 'OBSERVED',
            original_occurrence_identity=None if v is None else digest(dict(
                raw=v['row']['native_observation'],head=v['descriptor'].frame_local_head_id)),
            known_joint_indices=known,unresolved_joint_indices=sorted(set(range(6))-set(known)),
            known_joint_evidence_identity=None if v is None else v.get('review_joint_evidence_identity')))
    return FrozenReviewKnownJointDuty(json.dumps(dict(
        contract='original_time_specific_known_joint_presence_duty_v1',requests=rows,
        original_actor_inventory_identity=original_actor_inventory_identity,
        original_preservation_duty_identity=original_preservation_duty_identity,
        candidate_observations_used_to_declare=False,quality_approved=False),sort_keys=True,allow_nan=False))


def consume_review_known_joint_duty(*,duty,observations,projected_joint_points):
    if not isinstance(duty,FrozenReviewKnownJointDuty):raise ValueError('known duty must be sealed before selection')
    requests=duty.payload['requests'];ticks={r['tick'] for r in requests}
    if set(observations)!=ticks or set(projected_joint_points)!=ticks:
        raise ValueError('known joint consumer omitted a whole-shot output instant')
    results=[]
    for r in requests:
        tick=r['tick'];v=observations[tick];points=projected_joint_points[tick]
        if v is not None and v['row']['output_tick']!=tick:
            raise ValueError('known joint consumer borrowed another selected instant')
        missing=[];cropped=[]
        if v is None:
            state='UNKNOWN'
        else:
            shape=v['descriptor'].pose_shape
            observed={p[0] for p in shape or ()}
            for joint in r['known_joint_indices']:
                if joint not in observed or joint not in points:missing.append(joint);continue
                xy=points[joint]
                if (not isinstance(xy,(tuple,list)) or len(xy)!=2
                        or any(type(n) not in (int,float) or not math.isfinite(n) or not 0<=n<=1 for n in xy)):
                    cropped.append(joint)
            unresolved=(r['original_state']=='UNRESOLVED' or r['original_actor_state']=='UNRESOLVED'
                or v.get('reacquisition_ambiguous',False))
            state='FAIL' if cropped else 'UNKNOWN' if missing or unresolved else 'PASS'
        results.append(dict(tick=tick,state=state,requested_joint_indices=r['known_joint_indices'],
            unresolved_original_joint_indices=r['unresolved_joint_indices'],original_actor_state=r['original_actor_state'],
            missing_joint_indices=missing,cropped_joint_indices=cropped))
    states=[r['state'] for r in results]
    return dict(contract='actual_time_specific_known_joint_presence_receipt_v1',
        duty_identity=duty.identity,state='FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'PASS',
        requests=results,action_phase_certified=False,actor_relation_certified=False,quality_approved=False)
