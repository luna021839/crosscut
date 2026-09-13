"""Bounded, forward-decoded reference pixels shared by a crop batch.

Exact integer frames are unchanged. A missing older request restarts decoding
from frame zero; there is no timestamp seek, approximate frame or stale-source
reuse. Pixels remain in RAM only and never become serialized supervision.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path

import cv2

from .reference_match import (
    DecodedTargetFrames, ReferenceMatchConfig, _requested_indices, _working_frame,
)


@dataclass
class _Stream:
    capture: object
    identity: tuple[int, ...]
    frame_rate: float
    declared_count: int
    last_index: int = -1


def _identity(path: Path) -> tuple[int, ...]:
    info=path.stat()
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


class SequentialReferenceFrameCache:
    def __init__(self, *, maximum_bytes: int = 256 * 1024**2, maximum_open_sources: int = 12):
        if any(type(v) is not int or v < 1 for v in (maximum_bytes,maximum_open_sources)):
            raise ValueError("reference frame cache limits must be positive integers")
        self.maximum_bytes=maximum_bytes
        self.maximum_open_sources=maximum_open_sources
        self._streams=OrderedDict()
        self._identities={}
        self._pixels=OrderedDict()
        self.accounted_bytes=0
        self.decoded_frames=0
        self.cache_hits=0
        self.restarts=0
        self.opens=0
        self.closed=False

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def close(self):
        for stream in self._streams.values():
            stream.capture.release()
        self._streams.clear()
        self._pixels.clear()
        self._identities.clear()
        self.accounted_bytes=0
        self.closed=True

    def _open(self,path,identity):
        while len(self._streams)>=self.maximum_open_sources:
            _,old=self._streams.popitem(last=False)
            old.capture.release()
        capture=cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot decode reference-match source: {path}")
        rate=float(capture.get(cv2.CAP_PROP_FPS))
        count=float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(rate) or rate<=0:
            capture.release()
            raise ValueError("invalid source frame rate")
        stream=_Stream(capture,identity,rate,int(round(count)) if math.isfinite(count) and count>0 else 0)
        self._streams[path]=stream
        self.opens+=1
        return stream

    def decode(self,path,target_indices,*,config=None):
        if self.closed:
            raise RuntimeError("reference frame cache is closed")
        requested=_requested_indices(target_indices)
        if not requested:
            return DecodedTargetFrames(frame_rate=0.,declared_frame_count=0,requested_indices=(),frames={})
        settings=config or ReferenceMatchConfig()
        path=Path(path).resolve(strict=True)
        identity=_identity(path)
        if path in self._identities and self._identities[path]!=identity:
            raise RuntimeError("reference frame cache source changed")
        self._identities[path]=identity
        stream=self._streams.get(path)
        if stream is None:
            stream=self._open(path,identity)
        self._streams.move_to_end(path)
        results={}
        missing=[]
        for index in requested:
            key=(path,identity,settings.working_width,index)
            cached=self._pixels.get(key)
            if cached is None:
                missing.append(index)
            else:
                self.cache_hits+=1
                self._pixels.move_to_end(key)
                results[index]=cached.copy()  # Callers cannot poison the RAM pool.
        if missing and missing[0]<=stream.last_index:
            stream.capture.release()
            self._streams.pop(path)
            stream=self._open(path,identity)
            self.restarts+=1
        if missing:
            wanted=set(missing)
            while stream.last_index<missing[-1]:
                ok,frame=stream.capture.read()
                if not ok or frame is None:
                    break
                stream.last_index+=1
                self.decoded_frames+=1
                index=stream.last_index
                if index not in wanted:
                    continue
                working=_working_frame(frame,settings.working_width)
                results[index]=working.copy()
                size=working.nbytes+256
                if size<=self.maximum_bytes:
                    while self._pixels and self.accounted_bytes+size>self.maximum_bytes:
                        _,old=self._pixels.popitem(last=False)
                        self.accounted_bytes-=old.nbytes+256
                    working.setflags(write=False)
                    self._pixels[(path,identity,settings.working_width,index)]=working
                    self.accounted_bytes+=size
        if _identity(path)!=identity:
            raise RuntimeError("reference frame cache source changed during decode")
        return DecodedTargetFrames(frame_rate=stream.frame_rate,declared_frame_count=stream.declared_count,
                                   requested_indices=requested,frames={i:results[i] for i in requested if i in results})

    def diagnostics(self):
        return dict(contract="bounded_forward_only_exact_reference_frame_cache_v1",
                    decoded_frames=self.decoded_frames,cache_hits=self.cache_hits,restarts=self.restarts,
                    opens=self.opens,open_sources=len(self._streams),accounted_bytes=self.accounted_bytes,
                    maximum_bytes=self.maximum_bytes,persisted_pixels=False,random_seeks=0)
