"""Non-dilutable opposition and displacement of measured displayed arms.

This is an evidence-relative soft cost, not an arm track or a cut gate. A
missing bone contributes nothing; a matching arm earns no new reward. Only
contradiction worse than the existing UNKNOWN .5 baseline adds cost. Original
MODEL_ONLY evidence retains the existing halfway shrinkage.
"""
import numpy as np

ARM_BONES=np.array((0,1,3,4))
ARM_JOINTS=np.array(((0,2),(2,4),(1,3),(3,5)))
ARM_POINTS=np.arange(6)


def positions(left,right):
    """Parallel bones can still put a measured hand in the wrong place.

    Use the actual per-side rendered joint coordinates, in face-height units.
    No shoulder-relative translation removes a visible jump. Missing or
    ambiguous joints never become an observed mismatch; a crop cannot erase
    a known mismatch from the action credit by hiding the measured joint.
    """
    def observed(value):
        rows=np.arange(len(value['endpoint']));endpoint=value['endpoint']
        xy=value['xy'][rows,endpoint][:,ARM_POINTS]
        kind=value['joint_kind'][rows,endpoint][:,ARM_POINTS]
        ambiguous=value['joint_ambiguous'][rows,endpoint][:,ARM_POINTS]
        known=np.isin(kind,(1,2))&~ambiguous
        known&=value['independent_full_image_body_context'][:,None]
        known&=value['seen'][:,None]
        shown=(xy[...,0]>=0)&(xy[...,0]<=16/9)&(xy[...,1]>=0)&(xy[...,1]<=1.)
        return xy,kind,known,shown
    a,ka,known_a,shown_a=observed(left);b,kb,known_b,shown_b=observed(right)
    original_known=known_a[:,None]&known_b[None]
    known=original_known&(shown_a[:,None]|shown_b[None])
    height=np.maximum(.005,(left['height'][:,None]+right['height'][None,:])/2.)
    displacement=np.linalg.norm(a[:,None]-b[None],axis=-1)/height[...,None]
    raw=displacement/(1.+displacement)
    shrink=np.where((ka[:,None]==2)&(kb[None]==2),1.,.5)
    excess=np.where(known,shrink*np.maximum(0.,raw-.5),0.)
    return dict(sparse_arm_position_relative_cost=excess.max(-1),
        sparse_arm_position_residual=np.where(known,raw,0.).max(-1),
        sparse_arm_position_face_heights=np.where(known,displacement,0.).max(-1),
        sparse_arm_joints_measured=known,
        sparse_arm_position_action_credit_cap=np.where(original_known,np.maximum(0.,1.-2.*raw),1.).min(-1))


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
    result=dict(sparse_arm_contradiction_relative_cost=excess.max(-1),
        sparse_arm_direction_residual=np.where(known,raw,0.).max(-1),
        sparse_arm_direction_measured=known.any(-1),
        sparse_arm_bones_measured=known,
        sparse_arm_action_credit_cap=np.where(original_known,np.maximum(0.,1.-2.*raw),1.).min(-1))
    if all('joint_kind' in value and 'joint_ambiguous' in value for value in (left,right)):
        result.update(positions(left,right))
        result['sparse_arm_contradiction_relative_cost']=np.maximum(
            result['sparse_arm_contradiction_relative_cost'],result['sparse_arm_position_relative_cost'])
        result['sparse_arm_action_credit_cap']=np.minimum(
            result['sparse_arm_action_credit_cap'],result['sparse_arm_position_action_credit_cap'])
    return result


def bound_credit(detail):
    if 'sparse_arm_action_credit_cap' not in detail:return detail
    result=dict(detail);cap=detail['sparse_arm_action_credit_cap']
    for key in ('matched_action_change_density','actual_played_match_advantage'):
        if key in result:result[key]=np.minimum(result[key],cap)
    return result
