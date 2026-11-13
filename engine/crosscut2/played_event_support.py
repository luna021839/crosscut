"""Support of events ACTUALLY played, not all candidate cameras' events.

With authored holds >=12 frames and support <=6 frames, another authored
event cannot shorten K support. Source-pair neighbours suffice. A held native
event needs first/last boundary corrections, including both on a single event.
"""
import math
import numpy as np
from .connection_prior import physical_weights


class PlayedEventSupport:
    def __init__(self,events,*,final,fps,minimum_hold_frames=12):
        if (type(final) is not int or final<2 or not math.isfinite(fps) or fps<=0
                or minimum_hold_frames<12):
            raise ValueError('positive clock and >=12frame authored holds required')
        self.final,self.fps=final,fps;self.events=[]
        for values in events:
            raw=np.asarray(values)
            if raw.size and (raw.ndim!=1 or raw.dtype.kind not in 'iu' or np.any(raw<=0) or np.any(raw>=final)):
                raise ValueError('interior integer source-local native events required')
            self.events.append(np.unique(raw.astype(int)))

    def cut_weights(self,tick,sources):
        if not 0<tick<self.final:raise ValueError('interior authored cut required')
        before=[];after=[]
        for xs in self.events:
            left=np.searchsorted(xs,tick,side='left');right=np.searchsorted(xs,tick,side='right')
            before.append((tick-xs[left-1])/2. if left else float(tick))
            after.append((xs[right]-tick)/2. if right<len(xs) else float(self.final-tick))
        sources=np.asarray(sources,int)
        if sources.ndim!=1 or np.any(sources<0) or np.any(sources>=len(self.events)):
            raise ValueError('state-source indices escape native registry')
        return np.minimum(6.,np.minimum(np.asarray(before)[sources,None],np.asarray(after)[None,sources]))/self.fps


class NativeHoldCosts:
    """STATE-sized costs; intentionally not the SOURCE-sized fragment API."""
    def __init__(self,support,sources,rates):
        self.support=support;self.sources=np.asarray(sources,int);self.groups=[]
        if len(rates)!=len(self.sources):raise ValueError('one rate row per camera/actor state required')
        for s,xs in enumerate(support.events):
            indices=np.flatnonzero(self.sources==s)
            values=np.asarray([rates[i] for i in indices],float).reshape(len(indices),len(xs))
            if not np.isfinite(values).all():raise ValueError('finite own-native rates required')
            weights=physical_weights(xs,final=support.final,fps=support.fps)
            w=np.asarray([weights[int(t)] for t in xs])
            prefix=np.pad(np.cumsum(values*w,axis=1),((0,0),(1,0)))
            self.groups.append((indices,xs,values,w,prefix))

    def __call__(self,first,final):
        if not 0<=first<final<=self.support.final:raise ValueError('valid played hold required')
        result=np.zeros(len(self.sources))
        for indices,xs,rates,weights,prefix in self.groups:
            left=np.searchsorted(xs,first,side='right');right=np.searchsorted(xs,final,side='left')
            if left==right:continue
            value=prefix[:,right]-prefix[:,left]
            start_cap=(xs[left]-first)/2./self.support.fps if first else weights[left]
            end_cap=(final-xs[right-1])/2./self.support.fps if final<self.support.final else weights[right-1]
            if right-left==1:
                w=min(weights[left],start_cap,end_cap)
                value+=(w-weights[left])*rates[:,left]
            else:
                value+=(min(weights[left],start_cap)-weights[left])*rates[:,left]
                value+=(min(weights[right-1],end_cap)-weights[right-1])*rates[:,right-1]
            result[indices]=value
        return result
