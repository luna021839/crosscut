"""Pair-specific H_A/K_AB/H_B view on common part/window evidence.

Raw observations stay separate. Unsupported or contradicted pose comparisons
receive the SAME neutral prior in all three options, never a hard opposition
veto. A corroborated path is not anatomical ownership certification. This is
an explicit untrained editorial admission prior, NOT global DP dominance.
"""
import numpy as np
from .connection_observations import extract as raw_extract
from .connection_prior import quality,activity
from .motion_direction import compare as compare_direction
from .event_ports import rho


def scores(p,n):
    raw=raw_extract(p,n);q=len(p.states);shape=(q,q)
    central=np.asarray([d['known_center_native_cut'][n] for d in p.data],bool)
    face_seen=np.zeros(shape,bool)
    for side in (0,1):
        present=p.head[:,n,side,2]>.019
        seen=present[:,None]&present[None,:]
        seen&=p.continuous[:,n][None,:] if side==0 else p.continuous[:,n][:,None]
        face_seen|=seen
    fallback=(central[:,None]|central[None,:])&(p.head[:,n,0,2]>.019)[:,None]&(p.head[:,n,1,2]>.019)[None,:]
    face_seen|=fallback
    common_face=face_seen&np.diag(face_seen)[:,None]&np.diag(face_seen)[None,:]
    scope=np.asarray([d.get('pixel_part_scope',np.zeros(len(p.nodes),bool))[n] for d in p.data],bool)
    body=p.body[:,n];valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    geometry=valid[...,p.pairs].all(-1)
    physical=np.asarray([d['physical_window_valid'][n] for d in p.data],bool)
    continuous=p.continuous[:,n]&~central
    common_source=scope[:,None]&scope[None,:]&continuous[:,None]&continuous[None,:]
    phase=np.zeros(shape);motion=np.zeros(shape);phase_seen=np.zeros(shape,bool);motion_seen=np.zeros(shape,bool)
    for side,phase_at in ((0,1),(1,2)):
        height=p.head[:,n,side,2]*p.camera[:,n,side,0]
        scale=np.maximum(.005,(height[:,None]+height[None,:])/2.)
        for k,(_,joint) in enumerate(p.pairs):
            good=geometry[:,phase_at,k]&(p.phase_certainty[:,n,phase_at,k]==1.)
            common=common_source&common_face&good[:,None]&good[None,:]
            if not common.any():continue
            xy=p.projected[:,n,phase_at,joint,:2]
            screen=np.linalg.norm((xy[:,None]-xy[None,:])*[16/9,1.],axis=-1)/scale
            angle=np.linalg.norm(p.bones[:,None,n,phase_at,k]-p.bones[None,:,n,phase_at,k],axis=-1)/2.
            phase=np.maximum(phase,np.where(common,rho(np.maximum(screen,angle)),0.));phase_seen|=common
    for w,(early,late) in enumerate(((0,1),(2,3))):
        height=p.head[:,n,w,2]*p.camera[:,n,w,0]
        scale=np.maximum(.005,(height[:,None]+height[None,:])/2.)
        delta=(p.projected[:,n,late,:,:2]-p.projected[:,n,early,:,:2])*[16/9,1.]
        for k,joint in enumerate(p.destinations):
            good=geometry[:,early,k]&geometry[:,late,k]&physical[:,w]
            good&=(p.phase_certainty[:,n,early,k]==1.)&(p.phase_certainty[:,n,late,k]==1.)&(p.motion_certainty[:,n,w,k]==1.)
            common=common_source&common_face&good[:,None]&good[None,:]
            if not common.any():continue
            a,b=p.velocity[:,None,n,w,k],p.velocity[None,:,n,w,k]
            direction,_,_=compare_direction(a,b);np.fill_diagonal(direction,0.)
            na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
            speed=rho(abs(np.log((na+.15)/(nb+.15))))
            screen=rho(np.linalg.norm(delta[:,None,joint]-delta[None,:,joint],axis=-1)/scale)
            motion=np.maximum(motion,np.where(common,np.maximum(np.maximum(direction,speed),screen),0.));motion_seen|=common
    own,connection=activity(p,n)
    K=quality(position=np.where(common_face,raw['position'],.5),scale=np.where(common_face,raw['scale'],.5),
        phase=np.where(phase_seen,phase,.5),motion=np.where(motion_seen,motion,.5),
        crop=raw['crop'],visibility=raw['visibility'],own=own,connection=connection,
        music=p.strength[n],style=raw['style'],identity_conflict=raw['identity_conflict'])
    holds=[]
    for left in (True,False):
        expand=lambda x:x[:,None] if left else x[None,:]
        holds.append(quality(position=np.where(common_face,expand(np.diag(raw['position'])),.5),
            scale=np.where(common_face,expand(np.diag(raw['scale'])),.5),phase=np.where(phase_seen,0.,.5),
            motion=np.where(motion_seen,0.,.5),crop=expand(np.diag(raw['crop'])),
            visibility=expand(np.diag(raw['visibility'])),own=expand(np.diag(own)),
            connection=0.,music=p.strength[n],style=0.,identity_conflict=expand(np.diag(raw['identity_conflict']))))
    return K,np.minimum(*holds),dict(common_face=common_face,common_phase=phase_seen,common_motion=motion_seen)
