"""Optional strict compiled v2 window statistics, never an aesthetic model.

No fast-math, object-mode fallback, parallel reduction, fitted weights or
frame interpolation. The reference remains sequence_context.windows().
Compilation settings follow the upstream Numba JIT contract:
https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
"""
from functools import lru_cache
import numpy as np
from .sequence_context import PRIMITIVES,SUMMARIES


def _statistics(frames,values,known,ends,fps,history):
    B,T,P=values.shape;result=np.zeros((B,len(ends),P*9))
    for e in range(len(ends)):
        end=int(ends[e]);first=max(0,end-history)
        a=np.searchsorted(frames,first);z=np.searchsorted(frames,end)
        duration=end-first;pair_domain=max(1,duration-1)
        for b in range(B):
            for p in range(P):
                count=0;total=0.;low=np.inf;high=-np.inf;last=0.;latest=-1
                pairs=0;variation=0.;maximum_rate=0.
                for t in range(a,z):
                    if not known[b,t,p]:continue
                    value=values[b,t,p];total+=value;count+=1
                    low=min(low,value);high=max(high,value);last=value;latest=int(frames[t])
                    if t>a and known[b,t-1,p] and frames[t]-frames[t-1]==1:
                        rate=abs(value-values[b,t-1,p])*fps
                        variation+=rate;maximum_rate=max(maximum_rate,rate);pairs+=1
                k=p*9
                if count:
                    result[b,e,k]=total/count;result[b,e,k+1]=low;result[b,e,k+2]=high
                    result[b,e,k+4]=last;result[b,e,k+5]=(end-1-latest)/duration
                else:result[b,e,k+5]=1.
                result[b,e,k+3]=count/duration
                result[b,e,k+6]=variation/max(1,pairs);result[b,e,k+7]=maximum_rate
                result[b,e,k+8]=pairs/pair_domain
    return result


@lru_cache(maxsize=1)
def _compiled():
    # No import-time compilation, install, persistent JIT cache or silent
    # execution-backend fallback. An explicit caller must handle absence.
    from numba import njit
    if len(SUMMARIES)!=9:raise ValueError('compiled kernel requires the exact v2 nine-summary schema')
    return njit(cache=False,nogil=True,fastmath=False,parallel=False)(_statistics)


def features(frames,values,observed,ends,*,fps,context_frames):
    frames=np.asarray(frames);ends=np.asarray(ends);values=np.asarray(values);observed=np.asarray(observed)
    if (frames.ndim!=1 or frames.dtype.kind not in 'iu' or frames.min(initial=0)<0
            or frames.max(initial=0)>np.iinfo(np.int64).max
            or ends.ndim!=1 or ends.dtype.kind not in 'iu' or ends.min(initial=1)<1
            or ends.max(initial=0)>np.iinfo(np.int64).max
            or values.ndim!=3 or values.shape[1:]!=(len(frames),len(PRIMITIVES)) or not len(values)
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()
            or observed.shape!=values.shape or observed.dtype!=bool
            or not np.isfinite(fps) or fps<=0 or type(context_frames) is not int or context_frames<1):
        raise ValueError('finite explicit v2 numeric batch, signed actual ticks and validity masks required')
    frames=np.ascontiguousarray(frames,dtype=np.int64);ends=np.ascontiguousarray(ends,dtype=np.int64)
    if np.any(np.diff(frames)<=0) or np.any(np.diff(ends)<=0):
        raise ValueError('chronological unique output ticks required')
    function=_compiled()
    result=function(frames,np.ascontiguousarray(values,dtype=np.float64),
        np.ascontiguousarray(observed),ends,float(fps),context_frames)
    if not function.nopython_signatures:raise RuntimeError('compiled statistics did not execute in nopython mode')
    return result


def receipt():
    from importlib.metadata import version
    function=_compiled()
    return dict(backend='numba_strict_v2_window_statistics',numba_version=version('numba'),
        nopython_specializations=len(function.nopython_signatures),fastmath=False,parallel_reduction=False,
        persistent_compiler_cache=False,silent_numpy_fallback=False,kernel_is_not_an_aesthetic_model=True,
        professional_quality_approved=False)
