"""Original packed rank/paid-port semantics over fixed-slot backing IDs."""
from functools import lru_cache
import numpy as np
from .bounded_suffix_spool import _resolve,_retain,_release
from .compiled_context_beam import CompiledBeam,_rank,_room
from .playback_suffix_arena import SuffixRoot


def _place(fields,record,handle,target,meta):
    equivalent=discarded=0;rank=(record[0],record[1],record[2],record[3],-record[4])
    for k in range(len(fields[0])):
        if fields[6][k]==handle:
            equivalent=1
            if rank>=_rank(fields,k):_release(meta,handle);return discarded,equivalent,0,0
            target=k;break
    if not equivalent and fields[6][target]!=0:discarded=1
    if fields[6][target]!=0:_release(meta,fields[6][target])
    fields[0][target]=record[0];fields[1][target]=record[1];fields[2][target]=record[2]
    fields[3][target]=record[3];fields[4][target]=record[4];fields[5][target]=record[5];fields[6][target]=handle
    return discarded,equivalent,0,0


def _offer(fields,record,handle,meta):
    target=_room(fields,(record[0],record[1],record[2],record[3],-record[4]))
    if target<0:return 1,0,1,0
    _retain(meta,handle);return _place(fields,record,handle,target,meta)


def _live(meta,handle):
    slot=np.int64(handle&np.uint64(0xffffffff))-1
    return slot>=0 and slot<len(meta[0]) and meta[0][slot]>0 and meta[1][slot]==(handle>>np.uint64(32))


def _long(destination,values,hints,cursor,meta):
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
        if not _live(meta,handle):
            hints[q,code]=0
            return cursor,discarded,equivalent,rejected,p,q,code,bits[p,q,code,bit],slots[p,q,code,bit]
        _retain(meta,handle);d,e,r,error=_place(fields,record,handle,target,meta)
        discarded+=d;equivalent+=e;rejected+=r;cursor+=1
    return cursor,discarded,equivalent,rejected,-1,-1,-1,-1,-1


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    from numba.extending import register_jitable
    for f in (_resolve,_retain,_release,_rank,_room,_place,_offer,_live):register_jitable(f)
    return tuple(njit(cache=False,fastmath=False,parallel=False,nogil=True)(f) for f in (_offer,_long,_room))


class SpooledBeam(CompiledBeam):
    def __init__(self,arena):
        self.arena=arena;self.existing,self._long,self.room=(_offer,_long,_room) if arena.kernel=='numpy' else _compiled()
        self.discarded=self.equivalent=self.rank_rejected=self.long_kernel_calls=self.long_suffix_misses=0

    def offer(self,fields,values,parent,current):
        a=self.arena
        if current._arena is not a:raise ValueError('current lease belongs to another spooled arena')
        # Store before publishing or releasing any old label. IO failures
        # leave the original packed record and root references unchanged.
        with a._lock:target=self.room(fields,(values[0],values[1],values[2],values[3],-values[4]))
        if target<0:self._count((1,0,1,0));return
        root=a.append(SuffixRoot(a.owner,int(parent)),current)
        try:self.cut(fields,values,root.handle)
        finally:a.release(root)

    def cut(self,fields,values,handle):
        a=self.arena
        with a._lock:result=self.existing(fields,values,np.uint64(handle),a.store._meta)
        self._count(result)

    def long_batch(self,destination,values,hints,build):
        a=self.arena;cursor=0;pending=None
        try:
            while True:
                with a._lock:result=self._long(destination,values,hints,cursor,a.store._meta)
                self.long_kernel_calls+=1
                if pending is not None:a.release(pending);pending=None
                cursor,d,e,r,p,q,code,bit,slot=map(int,result);self._count((d,e,r,0))
                if p<0:break
                pending=build(p,q,code,bit,slot);self.long_suffix_misses+=1
                if not isinstance(pending,SuffixRoot) or pending.arena!=a.owner:
                    pending=None;raise ValueError('long batch requires a root owned by this backing store')
                hints[q,code]=pending.handle
        finally:
            if pending is not None:a.release(pending)
