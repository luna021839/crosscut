"""Strict packed beam operations over already compiled physical suffix IDs.

The beam is the original bounded approximation, not exact global search.
Only a candidate that can enter the existing rank frontier constructs a
suffix. Codes, costs, backpointers and paid bits never become identity keys.
"""
from functools import lru_cache
import numpy as np
from .playback_suffix_arena import _extend,_resolve,_release_root


def _rank(fields,k):
    return fields[0][k],fields[1][k],fields[2][k],fields[3][k],-fields[4][k]


def _room(fields,rank):
    free=-1;worst=-1
    for k in range(len(fields[0])):
        if fields[6][k]==0:
            if free<0:free=k
        elif worst<0 or _rank(fields,k)>_rank(fields,worst):worst=k
    if free<0 and rank>=_rank(fields,worst):return -1
    return free if free>=0 else worst


def _place(fields,values,handle,target,roots,cells,frames):
    equivalent=0;discarded=0
    rank=values[0],values[1],values[2],values[3],-values[4]
    for k in range(len(fields[0])):
        if fields[6][k]==handle:
            equivalent=1
            if rank>=_rank(fields,k):
                _release_root(roots,cells,frames,handle)
                return discarded,equivalent,0,0
            target=k;break
    if not equivalent and fields[6][target]!=0:discarded=1
    if fields[6][target]!=0:_release_root(roots,cells,frames,fields[6][target])
    fields[0][target]=values[0];fields[1][target]=values[1];fields[2][target]=values[2]
    fields[3][target]=values[3];fields[4][target]=values[4];fields[5][target]=values[5];fields[6][target]=handle
    return discarded,equivalent,0,0


def _offer_extended(fields,values,parent,current,first,final,H,mask,roots,cells,frames,scratch):
    target=_room(fields,(values[0],values[1],values[2],values[3],-values[4]))
    if target<0:return 1,0,1,0
    handle,error=_extend(roots,cells,frames,parent,current,first,final,H,mask,*scratch)
    if error:return 0,0,0,error
    return _place(fields,values,handle,target,roots,cells,frames)


def _offer_existing(fields,values,handle,roots,cells,frames):
    target=_room(fields,(values[0],values[1],values[2],values[3],-values[4]))
    if target<0:return 1,0,1,0
    roots[2][_resolve(roots,handle)]+=1
    return _place(fields,values,handle,target,roots,cells,frames)


@lru_cache(maxsize=1)
def _compiled():
    # Registers the pure state primitives as nopython-callable helpers first.
    from .playback_suffix_arena import _compiled as state_kernels
    state_kernels()
    from numba import njit
    from numba.extending import register_jitable
    for f in (_rank,_room,_place,_offer_extended,_offer_existing):register_jitable(f)
    options=dict(cache=False,nogil=True,fastmath=False,parallel=False)
    return tuple(njit(**options)(f) for f in (_offer_extended,_offer_existing))


class CompiledBeam:
    def __init__(self,arena):
        self.arena=arena;self.extended,self.existing=(_offer_extended,_offer_existing) if arena.kernel=='numpy' else _compiled()
        self.discarded=self.equivalent=self.rank_rejected=0

    def offer(self,fields,values,parent,current):
        a=self.arena
        if current._arena is not a:raise ValueError('compiled current lease is not owned by this beam arena')
        with a._lock,np.errstate(over='ignore'):
            result=self.extended(fields,values,np.uint64(parent),current._ids,current.first,current.final,a.H,a.mask,
                a._roots,a._cells,a._frames,a._scratch)
        self._count(result)

    def cut(self,fields,values,handle):
        a=self.arena
        with a._lock:result=self.existing(fields,values,np.uint64(handle),a._roots,a._cells,a._frames)
        self._count(result)

    def _count(self,result):
        discarded,equivalent,rank_rejected,error=result
        if error==1:raise MemoryError('live suffix capacity exceeded; compiled beam did not reduce the domain')
        if error:raise ValueError('compiled beam received inconsistent physical intervals')
        self.discarded+=discarded;self.equivalent+=equivalent;self.rank_rejected+=rank_rejected
