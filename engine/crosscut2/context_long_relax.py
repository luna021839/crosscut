"""Strict batched relaxation of PROVED history-independent suffixes.

Caller supplies every allowed baseline alternative and all retained labels.
Only prefixes whose future physical suffix is exactly the SAME may reduce
to their best rank. No beam, clock, camera or editorial gate is changed.
"""
from functools import lru_cache
import numpy as np


def _relax_into(base,handled,in_cost,in_cuts,head,middle,tail,benefit,same_episode,origin,width,
                costs,cuts,backs,bits,slots,credits,contexts):
    P,Q,C=handled.shape;L=in_cost.shape[-1];shape=(P,Q,C,2)
    costs[:]=np.inf;cuts[:]=0;backs[:]=-1;bits[:]=-1;slots[:]=-1;credits[:]=0.;contexts[:]=0.
    for p in range(P):
        for q in range(Q):
            for code in range(C):
                if not handled[p,q,code]:continue
                for bit in range(2):
                    for k in range(L):
                        if not np.isfinite(in_cost[p,bit,k]):continue
                        sequence=(head[p,bit,k,code]+middle[p,code])+tail[q,code]
                        raw=(in_cost[p,bit,k]+base[p,q,code])+sequence
                        count=in_cuts[p,bit,k];back=(((origin*width+p)*2+bit)*L+k)
                        variants=2 if bit==0 and benefit[p,q]>0. else 1
                        for claim in range(variants):
                            paid=benefit[p,q] if claim else 0.
                            outbit=(1 if bit or claim else 0) if same_episode else 0
                            value=raw-paid;old=costs[p,q,code,outbit]
                            better=(value<old or (value==old and (count<cuts[p,q,code,outbit] or
                                (count==cuts[p,q,code,outbit] and (back<backs[p,q,code,outbit] or
                                (back==backs[p,q,code,outbit] and paid>credits[p,q,code,outbit]))))))
                            if better:
                                costs[p,q,code,outbit]=value;cuts[p,q,code,outbit]=count
                                backs[p,q,code,outbit]=back;credits[p,q,code,outbit]=paid
                                bits[p,q,code,outbit]=bit;slots[p,q,code,outbit]=k
                                contexts[p,q,code,outbit]=sequence
    return costs,cuts,backs,bits,slots,credits,contexts


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    return njit(cache=False,nogil=True,fastmath=False,parallel=False)(_relax_into)


def relax(base,handled,in_cost,in_cuts,head,middle,tail,benefit,*,same_episode,origin,width,kernel,out=None):
    if kernel not in ('numpy','numba'):raise ValueError('explicit strict relaxation backend required')
    P,Q,C=handled.shape;L=in_cost.shape[-1]
    if (C!=4 or base.shape[:2]!=(P,Q) or in_cost.shape!=(P,2,L) or in_cuts.shape!=in_cost.shape
            or head.shape!=(P,2,L,C) or middle.shape!=(P,C) or tail.shape!=(Q,C)
            or benefit.shape!=(P,Q) or handled.dtype!=bool):
        raise ValueError('complete baseline domain and every retained prefix required')
    shape=(P,Q,C,2);dtypes=(np.float64,np.int64,np.int64,np.int64,np.int64,np.float64,np.float64)
    if out is None:out=tuple(np.empty(shape,dtype) for dtype in dtypes)
    if (not isinstance(out,tuple) or len(out)!=7 or any(a.shape!=shape or a.dtype!=dtype
            or not a.flags.writeable for a,dtype in zip(out,dtypes))):
        raise ValueError('seven complete writable numeric relaxation outputs required')
    function=_compiled() if kernel=='numba' else _relax_into
    return function(base[:,:,:4],handled,in_cost,in_cuts,head,middle,tail,benefit,
        bool(same_episode),int(origin),int(width),*out)
