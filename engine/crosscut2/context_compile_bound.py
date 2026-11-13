"""Exact existing-beam rank rejection BEFORE compiling physical inputs.

Requires an explicit proved nonnegative context cost. No aesthetic threshold,
beam change, source/camera quota or approximate score is introduced. A recipe
is unnecessary only when even its proved floor cannot enter ANY original paid
state. The floor is zero, or the unchanged exact middle of an already proved
long reference/local-intent hold. Later offers cannot worsen a full beam's
worst rank or open a free slot.
"""
from functools import lru_cache
import numpy as np


def _rank(destination,q,bit,k):
    return (destination[0][q,bit,k],destination[1][q,bit,k],destination[2][q,bit,k],
        destination[3][q,bit,k],-destination[4][q,bit,k])


def _mask(base,allowed,in_cost,in_cuts,benefit,destination,same_episode,origin,width,floors,zero_floor,
          out,vacant,worst):
    P,Q,C=allowed.shape;L=in_cost.shape[-1];out[:]=False;vacant[:]=False;worst[:]=-1
    for q in range(Q):
        for bit in range(2):
            for k in range(L):
                if destination[6][q,bit,k]==0:vacant[q,bit]=True
                elif worst[q,bit]<0 or _rank(destination,q,bit,k)>_rank(destination,q,bit,worst[q,bit]):
                    worst[q,bit]=k
    for p in range(P):
        for q in range(Q):
            for code in range(C):
                if not allowed[p,q,code]:continue
                possible=False
                for bit in range(2):
                    for k in range(L):
                        if not np.isfinite(in_cost[p,bit,k]):continue
                        # Identical original FP association and complete rank.
                        floor=0. if zero_floor else floors[p,q,code]
                        raw=(in_cost[p,bit,k]+base[p,q,code])+floor
                        count=in_cuts[p,bit,k];back=(((origin*width+p)*2+bit)*L+k)
                        variants=2 if bit==0 and benefit[p,q]>0. else 1
                        for claim in range(variants):
                            paid=benefit[p,q] if claim else 0.
                            outbit=(1 if bit or claim else 0) if same_episode else 0
                            rank=(raw-paid,count,back,code,-paid)
                            if vacant[q,outbit] or rank<_rank(destination,q,outbit,worst[q,outbit]):
                                possible=True;break
                        if possible:break
                    if possible:break
                out[p,q,code]=possible


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    from numba.extending import register_jitable
    register_jitable(_rank)
    return njit(cache=False,fastmath=False,parallel=False,nogil=True)(_mask)


def viable(base,allowed,in_cost,in_cuts,benefit,destination,*,same_episode,origin,width,workspace,kernel,floors=None):
    P,Q,C=allowed.shape;L=in_cost.shape[-1]
    if (kernel not in ('numpy','numba') or base.shape!=allowed.shape or allowed.dtype!=bool
            or in_cost.shape!=(P,2,L) or in_cuts.shape!=in_cost.shape or benefit.shape!=(P,Q)
            or len(destination)!=7 or any(a.shape!=(Q,2,L) for a in destination)
            or (floors is not None and (floors.shape!=allowed.shape or floors.dtype!=np.float64
                or not np.isfinite(floors).all() or np.any(floors<0.)))):
        raise ValueError('complete original rank, paid states and camera domain required')
    out=workspace.empty(allowed.shape,bool,owner='proved-zero-context-precompile-domain')
    vacant=workspace.empty((Q,2),bool,owner='precompile-rank-free-slots')
    worst=workspace.empty((Q,2),np.int64,owner='precompile-rank-worst-slots')
    function=_compiled() if kernel=='numba' else _mask
    function(base,allowed,in_cost,in_cuts,benefit,destination,bool(same_episode),int(origin),int(width),
        base if floors is None else floors,floors is None,out,vacant,worst)
    return out
