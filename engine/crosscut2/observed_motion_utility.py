"""Source-own movement utility only in actual measured six-frame windows.

Max union, not sum, prevents repeated/overlapping observations earning extra
reward. No activity is interpolated through an otherwise unobserved long hold.
"""
import numpy as np
from .event_ports import rho


def integrals(p,*,check=lambda:None,maximum_bytes=200*1024**2):
    q,N=len(p.states),len(p.nodes);final=int(p.nodes[-1]);needed=q*final*8
    if needed>maximum_bytes:raise ValueError('physical motion-window workspace exceeds200MiB')
    reward=np.zeros((q,final));scoped=np.asarray([d.get('pixel_part_scope',np.zeros(N,bool)) for d in p.data],bool)
    for n,tick in enumerate(p.nodes):
        check();body=p.body[:,n]
        valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
        geometry=valid[...,p.pairs].all(-1)
        physical=np.asarray([d['physical_window_valid'][n] for d in p.data],bool)
        for w,(early,late,start,end) in enumerate(((0,1,tick-7,tick-1),(2,3,tick,tick+6))):
            first,last=max(0,int(start)),min(final,int(end))
            if first>=last:continue
            good=geometry[:,early]&geometry[:,late]&physical[:,w,None]
            angular=rho(np.linalg.norm(p.velocity[:,n,w],axis=-1))
            delta=(p.projected[:,n,late,p.destinations,:2]-p.projected[:,n,early,p.destinations,:2])*[16/9,1.]
            height=np.maximum(.005,p.head[:,n,w,2]*p.camera[:,n,w,0])
            screen=rho(np.linalg.norm(delta,axis=-1)/height[:,None]/(6/p.fps))
            evidence=np.minimum(np.minimum(p.phase_certainty[:,n,early],p.phase_certainty[:,n,late]),p.motion_certainty[:,n,w])
            cap=np.where(evidence<=0,0.,np.where(scoped[:,n,None]&(evidence==1.),1.,.5))
            activity=np.where(good,np.maximum(angular,screen)*cap,0.).max(-1)
            amplitude=p.availability[:,n,w]*activity
            reward[:,first:last]=np.maximum(reward[:,first:last],amplitude[:,None])
    values=np.asarray([reward[:,a:b].sum(1)/p.fps for a,b in zip(p.nodes[:-1],p.nodes[1:])]).T
    return values,dict(contract='own_motion_visibility_max_union_physical_windows_v1',
        workspace_bytes=needed,source_NN=0,unobserved_hold_motion_extrapolated=False,
        duplicate_observation_reward_added=False,uncertain_geometry_activity_cap=.5)
