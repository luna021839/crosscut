"""A[-1] -> B[0] displayed face geometry; no virtual-side or H baseline.

GROUP focal uses the actual physical crop. MEMBER requires separately measured
target identity; an absent/unknown target never borrows another visible face.
No claim of pose, motion, identity or human-preference calibration is made.
"""
import numpy as np


def extract(p,n,*,intent='group',target_measured=None):
    if intent not in ('group','member'):raise ValueError('explicit group/member focal intent required')
    if intent=='group':
        heads=np.asarray([row['displayed_focal_heads'][n] for row in p.data])
        measured=heads[...,2]>.019
    else:
        heads=p.head[:,n]
        measured=np.zeros(heads.shape[:2],bool) if target_measured is None else np.asarray(target_measured)
        if measured.shape!=heads.shape[:2] or measured.dtype!=bool:
            raise ValueError('measured member-target identity mask required')
        measured=measured&(heads[...,2]>.019)
    camera=p.camera[:,n]
    center=.5+camera[:,:,:1]*(heads[...,:2]-camera[:,:,1:])
    height=heads[...,2]*camera[:,:,0]
    seen=measured[:,0,None]&measured[:,1][None,:]
    scale=abs(np.log(np.maximum(1e-9,height[:,0,None])/np.maximum(1e-9,height[:,1][None,:])))
    average=np.maximum(.005,(height[:,0,None]+height[:,1][None,:])/2.)
    position=np.linalg.norm((center[:,0,None]-center[None,:,1])*[16/9,1.],axis=-1)/average
    return dict(measured=seen,log_scale=np.where(seen,scale,0.),
        face_height_displacement=np.where(seen,position,0.),played_face_heights=height)
