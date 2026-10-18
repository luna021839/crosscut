"""Non-dilutable opposition of actually visible original-image arm bones.

This is an evidence-relative soft cost, not an arm track or a cut gate. A
missing bone contributes nothing; a matching arm earns no new reward. Only
opposition worse than the existing UNKNOWN .5 baseline adds cost. Original
MODEL_ONLY evidence retains the existing halfway shrinkage.
"""
import numpy as np

ARM_BONES=np.array((0,1,3,4))
ARM_JOINTS=np.array(((0,2),(2,4),(1,3),(3,5)))


def compare(left,right):
    def observed(value):
        rows=np.arange(len(value['endpoint']));endpoint=value['endpoint']
        xy=value['xy'][rows,endpoint]
        visible=(xy[...,0]>=0)&(xy[...,0]<=16/9)&(xy[...,1]>=0)&(xy[...,1]<=1.)
        kind=value['kind'][rows,endpoint][:,ARM_BONES]
        ambiguous=value['ambiguous'][rows,endpoint][:,ARM_BONES]
        known=np.isin(kind,(1,2))&~ambiguous
        known&=value['independent_full_image_body_context'][:,None]
        return value['unit'][rows,endpoint][:,ARM_BONES],kind,known,visible[:,ARM_JOINTS].any(-1)
    a,ka,known_a,shown_a=observed(left);b,kb,known_b,shown_b=observed(right)
    original_known=known_a[:,None]&known_b[None]
    # A digital crop cannot erase an independently measured opposition just
    # by hiding the incoming wrist/elbow. Score a bone entering either view;
    # both wholly hidden has no new visual cost, but earns no match credit.
    known=original_known&(shown_a[:,None]|shown_b[None])
    raw=(1.-np.clip((a[:,None]*b[None]).sum(-1),-1.,1.))/2.
    # RTMO is MODEL_ONLY. Preserve the general pixel-supported convention
    # without inventing certification for these model-derived instances.
    shrink=np.where((ka[:,None]==2)&(kb[None]==2),1.,.5)
    excess=np.where(known,shrink*np.maximum(0.,raw-.5),0.)
    return dict(sparse_arm_contradiction_relative_cost=excess.max(-1),
        sparse_arm_direction_residual=np.where(known,raw,0.).max(-1),
        sparse_arm_direction_measured=known.any(-1),
        sparse_arm_bones_measured=known,
        sparse_arm_action_credit_cap=np.where(original_known,np.maximum(0.,1.-2.*raw),1.).min(-1))


def bound_credit(detail):
    if 'sparse_arm_action_credit_cap' not in detail:return detail
    result=dict(detail);cap=detail['sparse_arm_action_credit_cap']
    for key in ('matched_action_change_density','actual_played_match_advantage'):
        if key in result:result[key]=np.minimum(result[key],cap)
    return result
