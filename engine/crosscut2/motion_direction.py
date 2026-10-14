"""Direction needs TWO non-degenerate moving vectors, not just one.

The neutral .5 for an undefined direction is a declared prior, not measured
opposition. Speed and displayed displacement remain independent raw errors.
This does not certify the pose's anatomy or scale errors by reliability.
"""
import numpy as np


def compare(a,b):
    na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
    moving=np.maximum(na,nb)>=.3
    observed=moving&(np.minimum(na,nb)>=.15)
    cosine=np.clip((a*b).sum(-1)/np.maximum(1e-9,na*nb),-1,1)
    residual=np.where(moving,np.where(observed,(1.-cosine)/2.,.5),0.)
    return residual,moving,observed
