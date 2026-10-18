"""Framing evidence for face-unavailable MEMBER context, never an actor ID.

An independently observed torso can support an ordinary context shot. It
cannot supply a missing face, named-member match, pose-match credit or mask.
Missing body observations remain UNKNOWN, unlike a completed empty sensor.
"""
import numpy as np


def framing_loss(people,crops):
    """Finite soft content cost in each actually proposed normalized crop.

    Four measured torso points keep a visible back view distinct from a
    camera pointed at a bench or legs. Partial occlusion is continuous; no
    source, zoom or cut is excluded. Native joint coordinates are not clipped.
    """
    crops=np.asarray(crops,float)
    if (crops.ndim!=2 or crops.shape[1]!=4 or not np.isfinite(crops).all()
            or np.any(crops[:,:2]<0) or np.any(crops[:,2:]<=0)
            or np.any(crops[:,:2]+crops[:,2:]>1.+1e-6)):
        raise ValueError('actual finite source-normalized camera crops required')
    if people is None:return np.full(len(crops),.9),np.zeros(len(crops),bool)
    points=np.asarray(people['keypoints17'],float);scores=np.asarray(people['scores'],float)
    if (points.shape!=(len(scores),17,3) or scores.ndim!=1 or not np.isfinite(points).all()
            or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))
            or np.any((points[...,2]<0)|(points[...,2]>1))):
        raise ValueError('original full-image COCO17 observations required')
    if not len(points):return np.full(len(crops),4.),np.ones(len(crops),bool)
    torso=points[:,[5,6,11,12]]
    native=(torso[...,2]>=.3)&(torso[...,:2]>=0).all(-1)&(torso[...,:2]<=1).all(-1)
    projected=(torso[None,...,:2]-crops[:,None,None,:2])/crops[:,None,None,2:]
    shown=native[None]&((projected>=0)&(projected<=1)).all(-1)
    shoulders=shown[...,:2].mean(-1);hips=shown[...,2:].mean(-1)
    presence=(np.sqrt(shoulders*hips)*scores[None]).max(-1)
    # .5 is the explicit neutral presence level, not calibrated probability.
    # A genuinely observed empty/torso-free frame is worse than an unmeasured
    # one; a body-only context is still less desirable than a clear face.
    loss=.9+3.1*np.maximum(0.,1.-2.*presence)-.35*np.maximum(0.,2.*presence-1.)
    return loss,np.ones(len(crops),bool)
