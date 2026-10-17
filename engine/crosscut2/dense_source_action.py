"""Use contemporary common-clock arm evidence before source/port selection.

This consumes the same subject/action measurement as cut-moment refinement,
but evaluates the ACTUAL proposed cameras without a hidden camera re-fit.
It does not reward camera motion, invent phase shifts, or certify identities.
"""
import numpy as np
from .subject_cut_moments import _seam_action,_normalized_body,_motion_steps


def motion(a,b,joints):
    """Every actual measured step near the seam, not four aliased endpoints.

    Residual/activity use a three-output-frame displacement unit. A missing
    intermediate face breaks an interval; no wrist is interpolated over it.
    """
    ticks=a['clock'].output_frames
    normal=[]
    for sample in (a,b):
        normal.append(_normalized_body(sample['window'],sample['at']))
    residual,activity,_=_motion_steps(*normal,a['supported']&b['supported'],ticks,joints,
        a['contiguous']&b['contiguous'])
    return residual,activity


def paired(left,right,*,rates,available,basis):
    rates=rates.copy();available=available.copy();basis=basis.copy()
    observed=np.zeros(available.shape,bool);used=np.zeros_like(observed)
    original_a=left.get('independent_full_image_body_context',np.zeros(len(left['dense_subject_contexts']),bool))
    original_b=right.get('independent_full_image_body_context',np.zeros(len(right['dense_subject_contexts']),bool))
    for i,a in enumerate(left['dense_subject_contexts']):
        if a is None:continue
        for j,b in enumerate(right['dense_subject_contexts']):
            if a is None or b is None:continue
            observed[i,j]=True
            independent=original_a[i] and original_b[j]
            # An unusable supplemental ROI must not erase an independently
            # measured full-image source context. Legacy corrected windows
            # retain their previous UNKNOWN replacement semantics.
            if not independent:rates[i,j]=.5;available[i,j]=False;basis[i,j]=0.
            ac,bc=a['clock'],b['clock']
            if ac.fps!=bc.fps or not np.array_equal(ac.output_frames,bc.output_frames):
                raise ValueError('dense subjects must share the actual music clock')
            geometry=dict(common_source_shot_sample_mask=a['same_shot']&b['same_shot'],
                outgoing_camera=left['endpoint_cameras'][i],incoming_camera=right['endpoint_cameras'][j])
            value=_seam_action(a['window'],b['window'],ac,bc,geometry)
            if not value['seam_action_supported']:continue
            dense_motion,activity=motion(a,b,value['seam_arm_joints'])
            if dense_motion is None:continue
            # Source context is root-relative pose + contemporary motion.
            # Played screen/body placement is scored separately in matrix();
            # it must neither disappear nor be counted twice as this context.
            raw=np.mean([value['seam_pose_residual']/(1.+value['seam_pose_residual']),
                         dense_motion/(1.+dense_motion)])
            risk=.5+.5*(raw-.5)
            for channel,wrist in ((0,4),(1,5)):
                if wrist in value['seam_arm_joints']:rates[i,j,channel]=risk
            available[i,j]=True;used[i,j]=True
            movement=activity/(1.+activity)
            basis[i,j]=.5*min(max(0.,1.-raw),movement)
    return rates,available,basis,observed,used
