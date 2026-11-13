"""Exact pre-history numeric input interning for ONE frozen hold arena.

Handles must come from content/observation and pre-camera intent registries,
not member names or an unverified source number. This component neither
creates those registries nor certifies them. Per-arena IDs must NOT escape
into persistent suffix states without their original keys/model binding.
The current descriptor is single-source GROUP +3-parameter camera only;
roll/hybrid/dissolve require explicit versioned composite inputs, never a
blend silently encoded as just its first source.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class InputBatch:
    model_binding: tuple
    keys: np.ndarray
    local_ids: np.ndarray
    unique_first_rows: np.ndarray
    probes: int
    maximum_active_bytes: int
    backend: str

    @property
    def unique_count(self):return len(self.unique_first_rows)


def pack(*,model_binding,frames,native,cameras,source_observation_handles,intent_handles,out=None):
    frames=np.asarray(frames);native=np.asarray(native);camera=np.asarray(cameras,np.float64)
    sources=np.asarray(source_observation_handles);intents=np.asarray(intent_handles);N=len(frames)
    if (not isinstance(model_binding,tuple) or len(model_binding)!=2
            or any(not isinstance(v,str) or not v for v in model_binding)
            or frames.shape!=(N,) or frames.dtype.kind not in 'iu' or frames.min(initial=0)<0
            or native.shape!=(N,) or native.dtype.kind not in 'iu' or native.min(initial=0)<0
            or camera.shape!=(N,3) or not np.isfinite(camera).all()
            or sources.shape!=(N,) or sources.dtype.kind not in 'iu' or sources.min(initial=0)<0
            or intents.shape!=(N,2) or intents.dtype.kind not in 'iu' or intents.min(initial=0)<0):
        raise ValueError('frozen exact clocks/cameras and bound numeric cue/intent handles required')
    keys=np.empty((N,8),np.uint64) if out is None else np.asarray(out)
    if keys.shape!=(N,8) or keys.dtype!=np.uint64 or not keys.flags.c_contiguous or not keys.flags.writeable:
        raise ValueError('writable exact-shape packed input output buffer required')
    keys[:,0]=frames;keys[:,1]=native;keys[:,2]=sources;keys[:,3:5]=intents
    keys[:,5:]=np.ascontiguousarray(camera).view(np.uint64)
    keys.setflags(write=False)
    return tuple(model_binding),keys


def _deduplicate(keys,slots,mask):
    N=len(keys);ids=np.empty(N,np.int64);first=np.empty(N,np.int64);count=0;probes=0
    for row in range(N):
        h=np.uint64(14695981039346656037)
        for k in range(8):h=(h^keys[row,k])*np.uint64(1099511628211)
        position=np.int64((h&mask)&np.uint64(len(slots)-1))
        while True:
            probes+=1;old=slots[position]
            if old<0:
                slots[position]=count;first[count]=row;ids[row]=count;count+=1;break
            same=True;other=first[old]
            for k in range(8):
                if keys[other,k]!=keys[row,k]:same=False;break
            if same:ids[row]=old;break
            position=(position+1)&(len(slots)-1)
    return ids,first[:count],probes


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    return njit(cache=False,nogil=True,fastmath=False,parallel=False)(_deduplicate)


def intern(packed,*,maximum_active_bytes=16*2**20,kernel='numpy'):
    binding,keys=packed;keys=np.asarray(keys)
    if (not isinstance(binding,tuple) or len(binding)!=2 or keys.ndim!=2 or keys.shape[1]!=8
            or keys.dtype!=np.uint64 or keys.flags.writeable or not keys.flags.c_contiguous
            or type(maximum_active_bytes) is not int or maximum_active_bytes<1 or kernel not in ('numpy','numba')):
        raise ValueError('immutable packed input arena and explicit scratch budget/backend required')
    N=len(keys);capacity=1
    while capacity<max(2,2*N):capacity*=2
    # Includes original key buffer, table, returned IDs and full backing array
    # of unique row indices. There is no hidden N*labels metadata allocation.
    active=keys.nbytes+capacity*8+N*16
    if active>maximum_active_bytes:raise MemoryError('input arena budget exceeded; no candidates are removed')
    slots=np.full(capacity,-1,np.int64);function=_compiled() if kernel=='numba' else _deduplicate
    with np.errstate(over='ignore'):
        ids,first,probes=function(keys,slots,np.uint64(0xffffffffffffffff))
    ids.setflags(write=False);first.setflags(write=False)
    return InputBatch(binding,keys,ids,first,int(probes),active,kernel)
