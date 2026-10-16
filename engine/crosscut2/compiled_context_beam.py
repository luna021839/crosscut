"""Strict packed beam operations over already compiled physical suffix IDs.

The beam is the original bounded approximation, not exact global search.
Only a candidate that can enter the existing rank frontier constructs a
suffix. Codes, costs, backpointers and paid bits never become identity keys.
"""
from functools import lru_cache
import numpy as np
from .playback_suffix_arena import _extend,_resolve,_release_root,_LOW,SuffixRoot


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


def _live(table,handle):
    slot=np.int64(handle&_LOW)-1
    return (slot>=0 and slot<len(table[2]) and table[2][slot]>0
        and table[3][slot]==(handle>>np.uint64(32)))


def _long_batch(destination,values,hints,cursor,roots,cells,frames):
    """Resume in original p/q/code/paid-bit order; stop only for a missing ID.

    The scalar relaxer has already kept the best original lane for each
    identical proved future. No raw input/model/camera callback enters this
    kernel. Hints are weak: a freed/recycled generation is a cache miss.
    """
    costs,counts,backs,bits,slots,credits,contexts=values;P,Q,C,_=costs.shape
    discarded=equivalent=rejected=0
    while cursor<P*Q*C*2:
        bit=cursor%2;at=cursor//2;code=at%C;at//=C;q=at%Q;p=at//Q
        if not np.isfinite(costs[p,q,code,bit]):cursor+=1;continue
        fields=(destination[0][q,bit],destination[1][q,bit],destination[2][q,bit],
            destination[3][q,bit],destination[4][q,bit],destination[5][q,bit],destination[6][q,bit])
        record=(costs[p,q,code,bit],counts[p,q,code,bit],backs[p,q,code,bit],
            code,credits[p,q,code,bit],contexts[p,q,code,bit])
        target=_room(fields,(record[0],record[1],record[2],record[3],-record[4]))
        if target<0:discarded+=1;rejected+=1;cursor+=1;continue
        handle=hints[q,code]
        if not _live(roots,handle):
            hints[q,code]=0
            return cursor,discarded,equivalent,rejected,p,q,code,bits[p,q,code,bit],slots[p,q,code,bit]
        roots[2][_resolve(roots,handle)]+=1
        d,e,r,error=_place(fields,record,handle,target,roots,cells,frames)
        discarded+=d;equivalent+=e;rejected+=r;cursor+=1
    return cursor,discarded,equivalent,rejected,-1,-1,-1,-1,-1


@lru_cache(maxsize=1)
def _compiled():
    # Registers the pure state primitives as nopython-callable helpers first.
    from .playback_suffix_arena import _compiled as state_kernels
    state_kernels()
    from numba import njit
    from numba.extending import register_jitable
    for f in (_rank,_room,_place,_offer_extended,_offer_existing,_live):register_jitable(f)
    options=dict(cache=False,nogil=True,fastmath=False,parallel=False)
    return tuple(njit(**options)(f) for f in (_offer_extended,_offer_existing,_long_batch))


class CompiledBeam:
    def __init__(self,arena):
        self.arena=arena
        self.extended,self.existing,self._long=(_offer_extended,_offer_existing,_long_batch) if arena.kernel=='numpy' else _compiled()
        self.discarded=self.equivalent=self.rank_rejected=0
        self.long_kernel_calls=self.long_suffix_misses=0

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

    def long_batch(self,destination,values,hints,build):
        """Only a rank-viable cache miss may build a proved physical suffix.

        build returns ONE owned root; the weak table never owns it. Even an
        exception, rank rejection or replacement releases this temporary.
        """
        a=self.arena;cursor=0;pending=None
        try:
            while True:
                with a._lock:
                    result=self._long(destination,values,hints,cursor,a._roots,a._cells,a._frames)
                self.long_kernel_calls+=1
                if pending is not None:a.release(pending);pending=None
                cursor,d,e,r,p,q,code,bit,slot=map(int,result);self._count((d,e,r,0))
                if p<0:break
                pending=build(p,q,code,bit,slot);self.long_suffix_misses+=1
                if not isinstance(pending,SuffixRoot) or pending.arena!=a.owner:
                    pending=None
                    raise ValueError('long batch requires an owned physical root in this arena')
                hints[q,code]=pending.handle
        finally:
            if pending is not None:a.release(pending)

    def _count(self,result):
        discarded,equivalent,rank_rejected,error=result
        if error==1:raise MemoryError('live suffix capacity exceeded; compiled beam did not reduce the domain')
        if error:raise ValueError('compiled beam received inconsistent physical intervals')
        self.discarded+=discarded;self.equivalent+=equivalent;self.rank_rejected+=rank_rejected
