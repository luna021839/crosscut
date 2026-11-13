"""Bounded interning of exact physical inputs, never candidate elimination.

Global six-frame cells are storage units only: every native ordinal, camera
sample, intended target and encoded frame is retained. Python key equality
checks original input bytes after hashing, including artificial collisions.
"""
from collections import OrderedDict
from dataclasses import dataclass
import sys
import numpy as np


def bytes_used(value):
    if isinstance(value,np.ndarray):return sys.getsizeof(value)+value.nbytes
    if isinstance(value,(tuple,list)):
        return sys.getsizeof(value)+sum(bytes_used(v) for v in value)
    return sys.getsizeof(value)


@dataclass(frozen=True,slots=True)
class InputKey:
    source: str
    observation_version: int
    model_schema: tuple
    frames: bytes
    native: bytes
    cameras: bytes
    intents: tuple


@dataclass(frozen=True,slots=True)
class Block:
    key: InputKey
    frames: np.ndarray
    encoded: np.ndarray
    physical_frames: tuple


class BlockPool:
    def __init__(self,*,maximum_bytes=32*2**20):
        if type(maximum_bytes) is not int or maximum_bytes<1:raise ValueError('explicit playback cache bound required')
        self.maximum_bytes=maximum_bytes;self.cache=OrderedDict();self.bytes=0;self.peak_bytes=0
        self.requests=self.hits=self.projections=self.frames_encoded=self.evictions=0

    def compile(self,*,source,observation_version,model_schema,frames,native,cameras,intents,project,encode):
        frames=np.asarray(frames,np.int64);native=np.asarray(native,np.int64);cameras=np.asarray(cameras,np.float64)
        T=len(frames)
        if (frames.ndim!=1 or native.shape!=frames.shape or cameras.shape!=(T,3) or len(intents)!=T
                or np.any(np.diff(frames)<=0) or not np.isfinite(cameras).all()):
            raise ValueError('exact chronological playback inputs and per-frame intents required')
        if not T:return ()
        # A selected hold can begin inside a global storage cell. Those thin
        # fragments stay exact and flatten to the same per-frame identity on
        # virtual split/rejoin; no resampling or padded observation is used.
        starts=np.r_[0,np.flatnonzero((np.diff(frames//6)!=0)|(np.diff(frames)!=1))+1]
        stops=np.r_[starts[1:],T];blocks=[];missing=[]
        for a,b in zip(starts,stops):
            key=InputKey(source,int(observation_version),tuple(model_schema),frames[a:b].tobytes(),
                native[a:b].tobytes(),cameras[a:b].tobytes(),tuple(intents[a:b]))
            self.requests+=1
            if key in self.cache:
                self.hits+=1;block,size=self.cache.pop(key);self.cache[key]=(block,size);blocks.append(block)
            else:missing.append((len(blocks),int(a),int(b),key));blocks.append(None)
        if missing:
            rows=np.concatenate([np.arange(a,b) for _,a,b,_ in missing])
            evidence=project(frames[rows],native[rows],cameras[rows])
            if not np.array_equal(evidence.frames,frames[rows]):raise AssertionError('projection changed exact playback ticks')
            encoded=np.asarray(encode(evidence),float)
            if encoded.ndim!=2 or len(encoded)!=len(rows) or not np.isfinite(encoded).all():
                raise ValueError('one finite ordered embedding per actual frame required')
            self.projections+=1;self.frames_encoded+=len(rows);offset=0
            for position,a,b,key in missing:
                ticks=frames[a:b].copy();value=encoded[offset:offset+b-a].copy();offset+=b-a
                ticks.setflags(write=False);value.setflags(write=False)
                physical=tuple((source,int(observation_version),int(t),int(n),c.tobytes(),intent)
                    for t,n,c,intent in zip(frames[a:b],native[a:b],cameras[a:b],intents[a:b]))
                block=Block(key,ticks,value,physical)
                size=(sys.getsizeof(block)+sys.getsizeof(key)+bytes_used((key.source,key.observation_version,
                    key.model_schema,key.frames,key.native,key.cameras,key.intents,physical,ticks,value)))
                while self.cache and self.bytes+size>self.maximum_bytes:
                    _,(_,old)=self.cache.popitem(last=False);self.bytes-=old;self.evictions+=1
                if size<=self.maximum_bytes:self.cache[key]=(block,size);self.bytes+=size
                self.peak_bytes=max(self.peak_bytes,self.bytes);blocks[position]=block
        return tuple(blocks)

    def receipt(self):
        return dict(contract='exact_intent_native_camera_block_interner_v1',storage_cell_frames=6,
            per_frame_samples_preserved=True,hash_collision_resolved_by_original_bytes=True,
            requests=self.requests,cache_hits=self.hits,projection_batches=self.projections,
            frames_encoded=self.frames_encoded,evictions=self.evictions,
            retained_blocks=len(self.cache),retained_bytes=self.bytes,peak_retained_bytes=self.peak_bytes,
            maximum_bytes=self.maximum_bytes,candidates_or_camera_codes_removed=False)


def flatten(blocks,*,first=None,final=None):
    frames=np.concatenate([b.frames for b in blocks]) if blocks else np.zeros(0,np.int64)
    width=blocks[0].encoded.shape[1] if blocks else 0
    values=np.concatenate([b.encoded for b in blocks]) if blocks else np.zeros((0,width))
    keys=sum((b.physical_frames for b in blocks),())
    keep=np.ones(len(frames),bool)
    if first is not None:keep&=frames>=first
    if final is not None:keep&=frames<final
    return frames[keep],values[keep],tuple(k for k,active in zip(keys,keep) if active)
