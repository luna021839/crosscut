"""Two actual MEMBER viewport cameras, without a16:9 metric assumption.

No pose is rotated, mirrored, relabelled or invented. Existing cameras remain
candidates. Face geometry, clipped measured arms and native sampling effort
are scored on the same realized physical view returned to the renderer.
"""
import numpy as np
from .aspect_viewport import _dimensions


def score(first,second,face_a,face_b,pose_a,pose_b,*,output_size,sampling=(1.,1.)):
    cameras=np.asarray([first,second],float);faces=np.asarray([face_a,face_b],float)
    points=np.asarray([pose_a,pose_b],float)
    _dimensions(output_size);sampling=np.asarray(sampling,float)
    if (cameras.shape!=(2,3) or faces.shape!=(2,4) or points.shape!=(2,8,3)
            or not all(np.isfinite(v).all() for v in (cameras,faces,points))
            or np.any(faces[:,2:]<=0) or sampling.shape!=(2,) or not np.isfinite(sampling).all() or min(sampling)<=0
            or np.any((cameras[:,0]<1)|(cameras[:,0]>2.4))
            or np.any(cameras[:,1:]<.5/cameras[:,:1]-1e-9)
            or np.any(cameras[:,1:]>1.-.5/cameras[:,:1]+1e-9)
            or np.any(np.linalg.norm(cameras[:,1:]-.5,axis=1)>.25+1e-9)):
        raise ValueError('actual finite member view cameras and observed face/screen-pose geometry required')
    centers=faces[:,:2]+faces[:,2:]/2
    shown=.5+cameras[:,:1]*(centers-cameras[:,1:]);height=faces[:,3]*cameras[:,0]
    aspect=output_size[0]/output_size[1]
    position=np.linalg.norm((shown[0]-shown[1])*[aspect,1.])/max(.005,float(height.mean()))
    scale=abs(np.log(height[0]/height[1]))/np.log(1.5)
    geometry=max(position,scale)
    half=faces[:,2:]*cameras[:,:1]/2
    extent=np.maximum(0.,np.minimum(1.,shown+half)-np.maximum(0.,shown-half))
    face_crop=float((1.-np.prod(extent/np.maximum(1e-12,2*half),axis=1)).sum())
    visible=(points[:,:6,2]>=.3)&(points[:,:6,:2]>=0).all(-1)&(points[:,:6,:2]<=1).all(-1)
    projected=.5+cameras[:,None,:1]*(points[:,:6,:2]-cameras[:,None,1:])
    lost=visible&((projected<0).any(-1)|(projected>1).any(-1))
    crop=float((lost.sum(-1)/np.maximum(1,visible.sum(-1))).mean())
    effort=float(np.maximum(0.,np.log2(cameras[:,0]/sampling)).mean())
    tight=float(np.maximum(0.,height-.28).sum())
    cost=3.*geometry+3.*face_crop+1.5*crop+.3*effort+6.*tight+.05*float(np.log(cameras[:,0]).sum())
    return dict(cost=cost,geometry=geometry,position=position,log_scale=scale,
        clipped_face_fraction=face_crop,added_arm_crop_fraction=crop,added_sampling_effort=effort)


def fit(face_a,face_b,pose_a,pose_b,*,existing,output_size,caps=(2.,2.),sampling=(1.,1.)):
    faces=np.asarray([face_a,face_b],float);old=np.asarray(existing,float)
    if (faces.shape!=(2,4) or old.shape!=(2,3) or not np.isfinite(faces).all()
            or not np.isfinite(old).all() or np.any(faces[:,2:]<=0)
            or any(not 1<=v<=2.4 for v in caps)):
        raise ValueError('two observed faces, original cameras and finite member zoom caps required')
    center=faces[:,:2]+faces[:,2:]/2;ha,hb=faces[:,3]
    low=max(ha,hb);high=max(ha*caps[0],hb*caps[1])
    targets=np.unique(np.r_[np.linspace(low,high,13),ha,hb])
    candidates=[old.copy()]
    for target in targets:
        z=np.clip(target/faces[:,3],1.,caps)
        lo=1.+z[:,None]*(center-1.);hi=z[:,None]*center
        lower=lo.max(0);upper=hi.min(0)
        shared=np.where(lower<=upper,np.clip(center.mean(0),lower,upper),(lower+upper)/2)
        displayed=np.clip(shared,lo,hi);pan=center+(.5-displayed)/z[:,None]
        delta=pan-.5;radius=np.linalg.norm(delta,axis=1)
        pan=.5+delta*np.minimum(1.,.25/np.maximum(1e-12,radius))[:,None]
        candidates.append(np.c_[z,pan])
    measured=[score(v[0],v[1],face_a,face_b,pose_a,pose_b,output_size=output_size,sampling=sampling) for v in candidates]
    at=min(range(len(candidates)),key=lambda i:measured[i]['cost'])
    return candidates[at],dict(before=measured[0],after=measured[at],original_camera_retained=at==0,
        output_aspect=output_size[0]/output_size[1],source_or_clock_or_pose_changed=False)
