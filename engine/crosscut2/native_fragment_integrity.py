"""Prevent an authored boundary from creating a 1-2 frame native-shot flash.

Only known strictly internal native events count. Unplayed events, exact
coincidences, file endpoints and untouched native/native fragments are not
new authored fragments. This is not semantic cut recall or aesthetic approval.
"""
import numpy as np


class NativeFragmentIntegrity:
    def __init__(self,support,*,minimum_frames):
        if type(minimum_frames) is not int or minimum_frames<1:
            raise ValueError('explicit positive integer native fragment support required')
        self.support,self.minimum_frames=support,minimum_frames

    def allowed(self,first,final,sources):
        if not 0<=first<final<=self.support.final:
            raise ValueError('valid played hold required')
        source_allowed=np.ones(len(self.support.events),bool)
        for source,xs in enumerate(self.support.events):
            left=np.searchsorted(xs,first,side='right')
            right=np.searchsorted(xs,final,side='left')
            if left==right:continue
            if first and xs[left]-first<self.minimum_frames:source_allowed[source]=False
            if final<self.support.final and final-xs[right-1]<self.minimum_frames:source_allowed[source]=False
        sources=np.asarray(sources)
        if sources.ndim!=1 or sources.dtype.kind not in 'iu' or np.any(sources<0) or np.any(sources>=len(source_allowed)):
            raise ValueError('state-source indices escape actual native registry')
        return source_allowed[sources]
