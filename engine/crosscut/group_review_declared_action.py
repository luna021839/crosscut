"""Independent visible-action requests when the old match is unmeasured.

Requests are declared from a fixed catalogue before donor costs, never from a
candidate's favourable joint intersection. Original motion and preservation
duties retain separate immutable roots. Measuring a declaration is NOT a
motion improvement, an ownership promotion, encode permission, or adoption.
"""
from dataclasses import dataclass
import json

from .group_performer_family_owner import digest
from .group_review_motion_contract import (FrozenReviewMotionIntent,compare_frozen_review_motion_route,
    frozen_motion_request_inventory)


ACTION_SCHEMAS={
    'upper_body_six_v1':(0,1,2,3,4,5),
    'bilateral_upper_arm_v1':(0,1,2,3),
    'left_arm_with_shoulders_v1':(0,1,2,4),
    'right_arm_with_shoulders_v1':(0,1,3,5),
    'left_elbow_with_shoulders_v1':(0,1,2),
    'right_elbow_with_shoulders_v1':(0,1,3),
}


def _sha(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


@dataclass(frozen=True,slots=True)
class FrozenReviewActionDeclaration:
    token_json:str

    def __post_init__(self):
        p=json.loads(self.token_json);ticks=p.get('ticks',[]);schema=p.get('schema')
        if (p.get('contract')!='independent_declared_visible_action_duty_v1'
                or schema not in ACTION_SCHEMAS or not 4<=len(ticks)<=16
                or any(type(t) is not int or t<0 for t in ticks)
                or any(a>=b for a,b in zip(ticks,ticks[1:]))
                or p.get('requested_joint_indices')!=list(ACTION_SCHEMAS[schema])
                or not all(_sha(p.get(k)) for k in ('original_motion_intent_identity',
                    'original_preservation_duty_identity','declared_scope_identity'))
                or p.get('candidate_joint_success_used_to_declare') is not False
                or p.get('original_known_duties_waived') is not False):
            raise ValueError('declared action needs an independently sealed catalogue request and original roots')
        expected=frozen_motion_request_inventory(ticks,[list(ACTION_SCHEMAS[schema]) for _ in ticks])
        if (any(p.get(k)!=v for k,v in expected.items())
                or p.get('action_kind')!=('BODY' if schema=='upper_body_six_v1' else 'PARTIAL_ACTION')):
            raise ValueError('declared action changed its typed position/velocity inventory')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_action_declaration(*,schema,ticks,original_motion_intent,
        original_preservation_duty_identity,declared_scope_identity):
    if not isinstance(original_motion_intent,FrozenReviewMotionIntent):
        raise ValueError('original motion state must remain separately sealed')
    if tuple(ticks)!=tuple(original_motion_intent.payload['ticks']):
        raise ValueError('declared action changed its original incident tick inventory')
    p=dict(contract='independent_declared_visible_action_duty_v1',schema=schema,ticks=list(ticks),
        requested_joint_indices=list(ACTION_SCHEMAS.get(schema,())),
        original_motion_intent_identity=original_motion_intent.identity,
        original_preservation_duty_identity=original_preservation_duty_identity,
        declared_scope_identity=declared_scope_identity,
        original_intent_kind=original_motion_intent.payload['kind'],
        candidate_joint_success_used_to_declare=False,original_known_duties_waived=False,
        original_unknown_promoted_to_pass=False,quality_approved=False)
    p.update(frozen_motion_request_inventory(ticks,[p['requested_joint_indices'] for _ in ticks]),
        action_kind='BODY' if schema=='upper_body_six_v1' else 'PARTIAL_ACTION')
    return FrozenReviewActionDeclaration(json.dumps(p,sort_keys=True,allow_nan=False))


def compare_declared_review_action_route(left,right,*,fps,declaration,route,original_motion_intent):
    if (not isinstance(declaration,FrozenReviewActionDeclaration)
            or not isinstance(original_motion_intent,FrozenReviewMotionIntent)
            or declaration.payload['original_motion_intent_identity']!=original_motion_intent.identity):
        raise ValueError('declared action borrowed or rewrote the original motion intent')
    p=declaration.payload;indices=p['requested_joint_indices']
    request=dict(contract='frozen_selected_original_motion_intent_v2',ticks=p['ticks'],
        original_actor_identity=declaration.identity,
        kind=p['action_kind'],
        joint_requests=[indices for _ in p['ticks']],known_joint_union=indices,
        original_unmatched_ticks=original_motion_intent.payload['original_unmatched_ticks'],
        request_origin='INDEPENDENT_DECLARATION_NOT_ORIGINAL_OBSERVATION',
        original_motion_intent_identity=original_motion_intent.identity,
        inferred_whole_arm_visibility_asserted=False,quality_approved=False)
    request.update({k:p[k] for k in ('position_requests','velocity_requests')})
    intent=FrozenReviewMotionIntent(json.dumps(request,sort_keys=True,allow_nan=False))
    result=compare_frozen_review_motion_route(left,right,fps=fps,intent=intent,route=route)
    return dict(result,contract='actual_candidate_declared_visible_action_measurement_v1',
        declaration_identity=declaration.identity,declared_action_schema=p['schema'],
        original_motion_intent_identity=original_motion_intent.identity,
        original_intent_kind=original_motion_intent.payload['kind'],
        original_preservation_duty_identity=p['original_preservation_duty_identity'],
        original_motion_state_promoted=False,original_known_duties_waived=False,
        comparative_improvement_claimed=False,review_encode_permitted=False,adoption_permitted=False)
