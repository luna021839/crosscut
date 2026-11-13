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


class ObservedBodyContext:
    """Numeric content quadrature and camera penalties for any member/song.

    `by_output[source][frame]` supplies (original native ordinal, people row).
    `face_queries[source][native]` says whether that independently observed
    frame has the requested member. Neither input is inferred by this class.
    """
    def __init__(self, by_output, face_queries, *, fps, radius_seconds=.5):
        if (len(by_output)!=len(face_queries) or not len(by_output)
                or not np.isfinite([fps,radius_seconds]).all() or fps<=0 or radius_seconds<=0):
            raise ValueError('bound per-source body/face queries and finite quadrature radius required')
        self.by_output=by_output;self.queries=face_queries;self.fps=float(fps);self.radius=float(radius_seconds)*fps
        for rows,queries in zip(by_output,face_queries,strict=True):
            if any(type(n) is not int or n<0 or type(v) is not bool for n,v in queries.items()):
                raise ValueError('current native target hints must retain observed boolean states')
            for frame,(native,people) in rows.items():
                if type(frame) is not int or type(native) is not int or min(frame,native)<0:
                    raise ValueError('body context needs explicit output/native frame bindings')
                framing_loss(people,[[0.,0.,1.,1.]])
        self.metadata=dict(completed_original_body_queries=sum(map(len,by_output)),
            maximum_quadrature_radius_seconds=float(radius_seconds),source_NN=0,source_rasters=0,
            interpolated_body_observations_claimed=False,named_actor_identity_or_pose_credit_supplied=False,
            measured_empty_content_cost=4.,unmeasured_context_cost=.9,source_veto=False)

    def content(self,source,crops,unary,natives):
        crops=np.asarray(crops);natives=np.asarray(natives);ticks=np.arange(len(crops))
        result=np.array(unary,copy=True);rows=self.by_output[source]
        if not rows:return result
        frames=np.array(sorted(rows),int)
        distance=np.abs(ticks[:,None]-frames[None,:]);at=distance.argmin(1)
        active=distance[np.arange(len(crops)),at]<=self.radius
        keys=np.array(sorted(self.queries[source]),int)
        if len(keys):
            qdistance=np.abs(natives[:,None]-keys[None,:]);nearest=qdistance.argmin(1)
            positive=np.array([self.queries[source][int(n)] for n in keys])
            body_native=np.array([rows[int(f)][0] for f in frames])[at]
            active&=~positive[nearest]
            active&=np.abs(natives-body_native)<=qdistance[np.arange(len(crops)),nearest]+1
        for k,frame in enumerate(frames):
            use=np.flatnonzero(active&(at==k))
            if not len(use):continue
            loss,_=framing_loss(rows[int(frame)][1],crops[use])
            weight=1.-distance[use,k]/(self.radius+1.)
            result[use]+=(loss-.9)*weight
        return result

    def penalty(self,shot,*,start,end):
        from .member_programme import camera_path
        known=[(f,body) for f,(_,body) in self.by_output[shot['source']].items()
            if shot['start']<=f<shot['start']+shot['count']]
        if not known:return 0.
        ticks,keys=camera_path(shot,start,end);total=0.
        for frame,body in known:
            t=frame-shot['start'];camera=np.array([np.interp(t,ticks,keys[:,a]) for a in range(3)])
            base=shot['base'][t];extent=base[2:]/camera[0]
            crop=np.r_[np.clip(base[:2]+base[2:]*camera[1:]-extent/2,0.,1.-extent),extent]
            actual,_=framing_loss(body,[crop]);full,_=framing_loss(body,[[0.,0.,1.,1.]])
            total+=max(0.,float(actual[0]-full[0]))
        return total
