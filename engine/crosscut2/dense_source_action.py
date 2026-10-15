"""Use contemporary common-clock arm evidence before source/port selection.

This consumes the same subject/action measurement as cut-moment refinement,
but evaluates the ACTUAL proposed cameras without a hidden camera re-fit.
It does not reward camera motion, invent phase shifts, or certify identities.
"""
import numpy as np
from .subject_cut_moments import _seam_action,_normalized_body


def motion(a,b,joints):
    """Every actual measured step near the seam, not four aliased endpoints.

    Residual/activity use a three-output-frame displacement unit. A missing
    intermediate face breaks an interval; no wrist is interpolated over it.
    """
    ticks=a['clock'].output_frames
    normal=[]
    for sample in (a,b):
        normal.append(_normalized_body(sample['window'],sample['at']))
    common=a['supported']&b['supported'];residual=[];activity=[]
    for joint in joints:
        seen=np.flatnonzero(common[:,joint]&(ticks>=-4)&(ticks<=3))
        for first,last in zip(seen[:-1],seen[1:]):
            gap=int(ticks[last]-ticks[first])
            if gap>3 or not a['contiguous'][first:last].all() or not b['contiguous'][first:last].all():continue
            da=normal[0][last,joint]-normal[0][first,joint]
            db=normal[1][last,joint]-normal[1][first,joint]
            residual.append(float(np.linalg.norm(da-db)*3/gap))
            activity.append(float(min(np.linalg.norm(da),np.linalg.norm(db))*3/gap))
    return (None,0.) if not residual else (max(residual),max(activity))


def paired(left,right,*,rates,available,basis):
    rates=rates.copy();available=available.copy();basis=basis.copy()
    observed=np.zeros(available.shape,bool);used=np.zeros_like(observed)
    for i,a in enumerate(left['dense_subject_contexts']):
        if a is None:continue
        for j,b in enumerate(right['dense_subject_contexts']):
            if a is None or b is None:continue
            observed[i,j]=True;rates[i,j]=.5;available[i,j]=False;basis[i,j]=0.
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
