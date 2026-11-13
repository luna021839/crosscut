"""Exact projection of raw v6 channels consumed by the connection prior.

Do not materialize all channels or unused per-evidence QxQxpart tensors. This
does not change the prior, missing values or raw residuals. The full extractor
remains the independent reference in equivalence tests/annotation exports.
"""
import numpy as np
from .event_ports import rho
from .motion_direction import compare as compare_direction


def extract(p,n):
    q=len(p.states);shape=(q,q);raw=np.zeros(shape+(4,),np.float32)
    body=p.body[:,n];valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    geometry=valid[...,p.pairs].all(-1)
    face_seen=np.zeros(shape,bool);phase_seen=np.zeros(shape,bool);motion_seen=np.zeros(shape,bool)
    phase_max=np.zeros(shape);motion_max=np.zeros(shape)
    for side,phase in ((0,1),(1,2)):
        head=p.head[:,n,side];camera=p.camera[:,n,side]
        center=.5+camera[:,:1]*(head[:,:2]-camera[:,1:]);height=head[:,2]*camera[:,0]
        scale=np.maximum(.005,(height[:,None]+height[None,:])/2.)
        seen=head[:,2]>.019;usable=seen[:,None]&seen[None,:]
        usable&=p.continuous[:,n][None,:] if side==0 else p.continuous[:,n][:,None]
        face_seen|=usable
        position=rho(np.linalg.norm((center[:,None]-center[None,:])*[16/9,1.],axis=-1)/scale)
        ratio=rho(abs(np.log(np.maximum(.001,height[:,None])/np.maximum(.001,height[None,:]))))
        raw[:,:,0]=np.maximum(raw[:,:,0],np.where(usable,position,0.))
        raw[:,:,1]=np.maximum(raw[:,:,1],np.where(usable,ratio,0.))
        for k,(_,b) in enumerate(p.pairs):
            common=usable&geometry[:,phase,k,None]&geometry[:,phase,k][None,:]
            point=p.projected[:,n,phase,b,:2]
            screen=np.linalg.norm((point[:,None]-point[None,:])*[16/9,1.],axis=-1)/scale
            angle=np.linalg.norm(p.bones[:,None,n,phase,k]-p.bones[None,:,n,phase,k],axis=-1)/2.
            phase_max=np.maximum(phase_max,np.where(common,rho(np.maximum(screen,angle)),0.));phase_seen|=common
    central=np.asarray([r['known_center_native_cut'][n] for r in p.data],bool)
    uninterrupted=np.eye(q,dtype=bool)&~central[:,None]
    no_phase=~phase_seen;fallback=~face_seen&(central[:,None]|central[None,:])
    h1,h2=p.head[:,n,0],p.head[:,n,1];c1,c2=p.camera[:,n,0],p.camera[:,n,1]
    ha,hb=h1[:,2]*c1[:,0],h2[:,2]*c2[:,0]
    scale=np.maximum(.005,(ha[:,None]+hb[None,:])/2.)
    exposed=(h1[:,2]>.019)[:,None]&(h2[:,2]>.019)[None,:]
    ca=.5+c1[:,:1]*(h1[:,:2]-c1[:,1:]);cb=.5+c2[:,:1]*(h2[:,:2]-c2[:,1:])
    raw[:,:,0]=np.maximum(raw[:,:,0],np.where(fallback&exposed,
        rho(np.linalg.norm((ca[:,None]-cb[None,:])*[16/9,1.],axis=-1)/scale),0.))
    raw[:,:,1]=np.maximum(raw[:,:,1],np.where(fallback&exposed,
        rho(abs(np.log(np.maximum(.001,ha[:,None])/np.maximum(.001,hb[None,:])))),0.))
    face_seen|=fallback&exposed
    for k,(_,b) in enumerate(p.pairs):
        common=no_phase&exposed&geometry[:,1,k,None]&geometry[:,2,k][None,:]&~uninterrupted
        a1,b1=p.projected[:,n,1,b,:2],p.projected[:,n,2,b,:2]
        screen=np.linalg.norm((a1[:,None]-b1[None,:])*[16/9,1.],axis=-1)/scale
        angle=np.linalg.norm(p.bones[:,None,n,1,k]-p.bones[None,:,n,2,k],axis=-1)/2.
        phase_max=np.maximum(phase_max,np.where(common,rho(np.maximum(screen,angle)),0.));phase_seen|=common
    physical=np.asarray([r['physical_window_valid'][n] for r in p.data],bool)
    for window,(early,late) in enumerate(((0,1),(2,3))):
        height=p.head[:,n,window,2]*p.camera[:,n,window,0]
        scale=np.maximum(.005,(height[:,None]+height[None,:])/2.)
        screen=(p.projected[:,n,late,:,:2]-p.projected[:,n,early,:,:2])*[16/9,1.]
        for k,joint in enumerate(p.destinations):
            valid=geometry[:,early,k]&geometry[:,late,k]&physical[:,window]
            common=valid[:,None]&valid[None,:]
            common&=p.continuous[:,n][None,:] if window==0 else p.continuous[:,n][:,None]
            a,b=p.velocity[:,None,n,window,k],p.velocity[None,:,n,window,k]
            na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
            direction,_,_=compare_direction(a,b);np.fill_diagonal(direction,0.)
            speed=rho(abs(np.log((na+.15)/(nb+.15))))
            camera_motion=rho(np.linalg.norm(screen[:,None,joint]-screen[None,:,joint],axis=-1)/scale)
            motion_max=np.maximum(motion_max,np.where(common,np.maximum(np.maximum(direction,speed),camera_motion),0.))
            motion_seen|=common
    raw[:,:,2]=phase_max;raw[:,:,3]=motion_max
    values=[np.where(seen,raw[:,:,k],.5) for k,seen in enumerate((face_seen,face_seen,phase_seen,motion_seen))]
    values=[np.where(uninterrupted,0.,v) for v in values]
    style=np.zeros(shape)
    for region in (0,1):
        style+=.5*np.sqrt(np.maximum(0.,1.-np.einsum('ik,jk->ij',p.colour[:,n,0,region],p.colour[:,n,1,region],optimize=False)))
    style=np.where(p.colour_valid[:,n,0,None]&p.colour_valid[:,n,1][None,:],style,0.).astype(np.float32)
    ma,mb=p.member_hint[:,n,0],p.member_hint[:,n,1]
    return dict(zip(('position','scale','phase','motion'),values),
        crop=np.maximum(p.crop[:,n,0,None],p.crop[:,n,1][None,:]),
        visibility=np.minimum(p.availability[:,n,0,None],p.availability[:,n,1][None,:]),
        style=style,identity_conflict=(ma[:,None]>=0)&(mb[None,:]>=0)&(ma[:,None]!=mb[None,:]))
