"""Opaque local clothing reveal, separate from face and stage transitions.

Only a narrow moving material boundary is translucent. A completed region
contains the actual incoming garment, not a continuing two-outfit exposure.
No new clothes, body joints or background pixels are generated.
"""
import cv2
import numpy as np

from .member_wardrobe_layer import _warp,phases


def smooth(progress,start,end):
    value=float(np.clip((progress-start)/(end-start),0.,1.))
    return value*value*(3.-2.*value)


def reveal(garment,progress):
    """Top-to-bottom replacement through actual garment support, not skin."""
    garment=np.asarray(garment,np.float32)
    if garment.ndim!=2 or not np.isfinite(garment).all() or garment.min()<0 or garment.max()>1:
        raise ValueError('finite measured garment mask required')
    phase,_=phases(progress)
    yy,_=np.nonzero(garment>.2)
    if not len(yy):return np.zeros_like(garment)
    rank=(np.arange(garment.shape[0])[:,None]-yy.min())/max(1,yy.max()-yy.min())
    # The transition band spans only 10% of clothing height. Everything
    # behind it is opaque incoming material; everything ahead stays original.
    band=.1
    alpha=np.clip((phase*(1+band)-rank)/band,0.,1.)
    return garment*alpha.astype(np.float32)


def composite(outgoing,incoming,*,incoming_garment,outgoing_person,incoming_person,incoming_to_outgoing,progress,
        incoming_face_landmarks=None):
    if (outgoing.dtype!=np.uint8 or incoming.dtype!=np.uint8 or outgoing.shape!=incoming.shape
            or outgoing.ndim!=3 or outgoing.shape[2]!=3):raise ValueError('matching actual BGR8 pictures required')
    h,w=outgoing.shape[:2]
    masks=[np.asarray(v,np.float32) for v in (incoming_garment,outgoing_person,incoming_person)]
    if any(v.shape!=(h,w) or not np.isfinite(v).all() or v.min()<0 or v.max()>1 for v in masks):
        raise ValueError('measured same-grid garment/person masks required')
    matrix=np.asarray(incoming_to_outgoing,np.float32)
    if matrix.shape!=(2,3) or not np.isfinite(matrix).all() or np.linalg.det(matrix[:,:2])<=0:
        raise ValueError('actual orientation-preserving foreground registration required')
    phases(progress)
    if progress==0:return outgoing.copy()
    if progress==1:return incoming.copy()
    release=smooth(progress,.38,.94)
    person_phase=smooth(progress,.68,.9)
    stage_phase=smooth(progress,.68,1.)
    identity=np.eye(2,3,dtype=np.float32)
    warps=(identity+(cv2.invertAffineTransform(matrix)-identity)*release,
        identity+(matrix-identity)*(1-release))
    sigma=max(.6,min(w,h)*.003);prepared=[];supports=[]
    for picture,person,warp in zip((outgoing,incoming),masks[1:],warps,strict=True):
        moved=_warp(person,warp,(w,h))
        support=cv2.GaussianBlur(np.maximum(person,moved),(0,0),sigma)
        prepared.append(picture.astype(np.float32)*(1-support[...,None])
            +_warp(picture,warp,(w,h)).astype(np.float32)*support[...,None])
        supports.append(support)
    clothes=_warp(masks[0],warps[1],(w,h))
    region=clothes
    if incoming_face_landmarks is not None:
        points=np.asarray(incoming_face_landmarks,float)
        if points.shape!=(5,2) or not np.isfinite(points).all():raise ValueError('actual five-point facial camera guide required')
        points=points@warps[1][:,:2].T+warps[1][:,2]
        neck=max(points[3:,1])+max(3.,np.linalg.norm(points[0]-points[1])*.35)
        below=np.clip((np.arange(h)[:,None]-neck)/max(3.,w*.01),0.,1.)
        # The outfit change includes newly exposed skin (e.g. a different
        # neckline). Restricting it to B's fabric leaves A's shirt floating
        # inside those real skin openings. This is a compositing guide, not a
        # head-extent/identity observation or a new semantic measurement.
        region=np.maximum(region,_warp(masks[2],warps[1],(w,h))*below)
    clothing_alpha=reveal(region,progress)
    # Skin/head and stage have distinct clocks. Releasing face registration
    # no longer waits for the final background-fade pictures.
    person=np.maximum(*supports)
    alpha=stage_phase+(person_phase-stage_phase)*person
    alpha=np.maximum(alpha,clothing_alpha)[...,None]
    return np.rint(np.clip(prepared[0]*(1-alpha)+prepared[1]*alpha,0,255)).astype(np.uint8)
