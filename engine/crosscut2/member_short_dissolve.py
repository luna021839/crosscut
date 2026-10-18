"""Assess a short MEMBER overlap from actual source/view measurements.

Only the effect is declined on missing current target faces. This never bans
a source or changes the hard-cut programme. Sparse body evidence remains
explicit; face interpolation never manufactures an anatomical observation.
"""
import numpy as np
from .aspect_viewport import _dimensions


def assess(faces,known,poses,*,output_size):
    boxes=np.asarray(faces,float);available=np.asarray(known,bool);body=np.asarray(poses,float)
    _dimensions(output_size)
    if (boxes.shape!=(3,2,4) or available.shape!=(3,2) or body.shape!=(3,2,8,3)
            or not np.isfinite(boxes).all() or not np.isfinite(body).all()
            or np.any(body[...,2]<0) or np.any(body[...,2]>1)):
        raise ValueError('three real two-source face/pose overlap samples required')
    if not available.all() or np.any(boxes[...,2:]<=0):
        return dict(eligible=False,reason='current_target_face_not_observed_on_both_sources',
            missing_current_faces=int((~available).sum()),pose_invented=False)
    centers=boxes[...,:2]+boxes[...,2:]/2;metric=np.array([output_size[0]/output_size[1],1.])
    heights=boxes[...,3];mean_height=heights.mean(1)
    position=np.linalg.norm((centers[:,0]-centers[:,1])*metric,axis=-1)/mean_height
    scale=np.abs(np.log(heights[:,0]/heights[:,1]))
    first=boxes[...,:2];last=first+boxes[...,2:]
    visible=np.prod(np.maximum(0.,np.minimum(1.,last)-np.maximum(0.,first))/boxes[...,2:],axis=-1)
    bones=np.array([(0,2),(2,4),(1,3),(3,5)])
    vector=(body[...,bones[:,1],:2]-body[...,bones[:,0],:2])*metric
    length=np.linalg.norm(vector,axis=-1)
    measured=(body[...,bones,2]>=.3).all(-1)&(length>.005)
    common=measured[:,0]&measured[:,1]
    unit=vector/np.maximum(.001,length)[...,None]
    direction=np.linalg.norm(unit[:,0]-unit[:,1],axis=-1)/2
    worst=None if not common.any() else float(direction[common].max())
    eligible=bool(max(position)<=.7 and max(scale)<=.3 and min(visible.ravel())>=.95
        and max(heights.ravel())<=.30 and (worst is None or worst<=.42))
    return dict(eligible=eligible,reason='short_measured_overlap' if eligible else 'measured_overlap_discrepancy',
        maximum_face_position_in_face_heights=float(max(position)),maximum_log_face_scale=float(max(scale)),
        minimum_visible_face_fraction=float(min(visible.ravel())),maximum_visible_arm_direction_mismatch=worst,
        measured_future_bone_pairs=int(common.sum()),missing_future_bones_are_not_matches=True,
        score=float(max(position)+max(scale)+(.5 if worst is None else worst)),
        pose_invented=False,normal_speed_AV_approval=False)
