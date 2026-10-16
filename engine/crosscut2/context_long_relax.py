"""Strict batched relaxation of PROVED history-independent suffixes.

Caller supplies every allowed baseline alternative and all retained labels.
Only prefixes whose future physical suffix is exactly the SAME may reduce
to their best rank. No beam, clock, camera or editorial gate is changed.
"""
from functools import lru_cache
import numpy as np


def _relax(base,handled,in_cost,in_cuts,head,middle,tail,benefit,same_episode,origin,width):
    P,Q,C=handled.shape;L=in_cost.shape[-1];shape=(P,Q,C,2)
    costs=np.full(shape,np.inf);cuts=np.zeros(shape,np.int64);backs=np.full(shape,-1,np.int64)
    bits=np.full(shape,-1,np.int64);slots=np.full(shape,-1,np.int64)
    credits=np.zeros(shape);contexts=np.zeros(shape)
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
    return njit(cache=False,nogil=True,fastmath=False,parallel=False)(_relax)


def relax(base,handled,in_cost,in_cuts,head,middle,tail,benefit,*,same_episode,origin,width,kernel):
    if kernel not in ('numpy','numba'):raise ValueError('explicit strict relaxation backend required')
    P,Q,C=handled.shape;L=in_cost.shape[-1]
    if (C!=4 or base.shape[:2]!=(P,Q) or in_cost.shape!=(P,2,L) or in_cuts.shape!=in_cost.shape
            or head.shape!=(P,2,L,C) or middle.shape!=(P,C) or tail.shape!=(Q,C)
            or benefit.shape!=(P,Q) or handled.dtype!=bool):
        raise ValueError('complete baseline domain and every retained prefix required')
    function=_compiled() if kernel=='numba' else _relax
    return function(base[:,:,:4],handled,in_cost,in_cuts,head,middle,tail,benefit,bool(same_episode),int(origin),int(width))
