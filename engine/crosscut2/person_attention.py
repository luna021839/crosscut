"""Visible rectangle competition for a physically pinned face and viewport.

Person rectangles are not masks, depth, ownership or identity certificates.
All plausible face-containing rectangles remain candidates; the most generous
target area gives a LOWER BOUND on competition, rather than choosing a small
box to invent occlusion. A missing/ambiguous anchor is UNKNOWN (.5), not a
clean view. Confidence is not used to rank people.
"""
import numpy as np


def competition(rectangles, heads, cameras):
    """Broadcast (...,persons,5), (...,4), (...,3) original measurements.

    Zero padding represents absent measurements, not an empty scene. A face
    must overlap an upper rectangle anchor; a rectangle merely enclosing a
    background face cannot establish ownership. Every compatible box is kept
    and the largest displayed target area is used optimistically.
    """
    boxes=np.asarray(rectangles,float);head=np.asarray(heads,float);camera=np.asarray(cameras,float)
    if (boxes.ndim<2 or boxes.shape[-1]!=5 or head.shape[-1:]!=(4,) or camera.shape[-1:]!=(3,)
            or not all(np.isfinite(v).all() for v in (boxes,head,camera))
            or np.any((boxes<0)|(boxes>1)) or np.any(camera[...,0]<1.-1e-9)):
        raise ValueError('finite normalized person rectangles, actual face and viewport required')
    active=(boxes[...,4]>0)&(boxes[...,2]>boxes[...,0])&(boxes[...,3]>boxes[...,1])
    center=.5+camera[...,None,:1]*(boxes[...,:2]-camera[...,None,1:])
    corner=.5+camera[...,None,:1]*(boxes[...,2:4]-camera[...,None,1:])
    area=np.prod(np.maximum(0.,np.minimum(1.,corner)-np.maximum(0.,center)),axis=-1)
    area=np.where(active,area,0.)
    half=head[...,2,None]*np.array([.45*9/16,.5])
    low,high=head[...,:2]-half,head[...,:2]+half
    overlap=np.prod(np.maximum(0.,np.minimum(boxes[...,2:4],high[...,None,:])-
        np.maximum(boxes[...,:2],low[...,None,:])),axis=-1)
    fraction=overlap/np.maximum(1e-12,np.prod(2*half,axis=-1)[...,None])
    top=head[...,1,None]-boxes[...,1]
    plausible=active&(fraction>=.6)&(top>=-.5*head[...,2,None])&(top<=2.*head[...,2,None])
    target=np.max(np.where(plausible,area,0.),axis=-1,initial=0.)
    maximum=np.max(area,axis=-1,initial=0.)
    face_center=.5+camera[...,:1]*(head[...,:2]-camera[...,1:])
    face_half=half*camera[...,:1]
    visible=(np.minimum(1.,face_center+face_half)>np.maximum(0.,face_center-face_half)).all(-1)
    measured=(head[...,2]>.019)&(head[...,3]>=.75)&visible&(target>0)&(maximum>0)
    deficit=np.clip(1.-target/np.maximum(1e-12,maximum),0.,1.)
    return dict(deficit=np.where(measured,deficit,.5),measured=measured,
        target_area_upper_bound=target,maximum_visible_person_area=maximum,
        compatible_rectangles=np.sum(plausible&(area>0),axis=-1))
