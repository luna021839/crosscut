"""Additive source-native detections, not a relabelling of an old face bank."""
import numpy as np


def slots(original,detected,*,source_aspect):
    old=np.asarray(original,float);new=np.asarray(detected,float)
    if (old.shape!=(8,4) or new.ndim!=2 or new.shape[1]!=4 or not np.isfinite(old).all()
            or not np.isfinite(new).all() or not np.isfinite(source_aspect) or source_aspect<=0):
        raise ValueError('finite original8-head bank and actual source-aspect detections required')
    available=list(np.flatnonzero(np.all(old==0.,axis=1)))
    existing=[v.copy() for v in old if v[2]>.019 and v[3]>=.75];result=[]
    for k in np.argsort(-new[:,2],kind='stable'):
        face=new[k]
        if face[2]<=.019 or not .75<=face[3]<=1.:continue
        duplicate=any(np.linalg.norm((face[:2]-v[:2])*[source_aspect,1.])/max(.02,face[2],v[2])<=.35
            and abs(np.log(face[2]/v[2]))<=np.log(2.) for v in existing)
        if duplicate or not available:continue
        result.append((int(available.pop(0)),int(k)));existing.append(face)
    return tuple(result)
