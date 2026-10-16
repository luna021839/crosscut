"""One explicit allocation owner for packed playback state and its leases.

Accounting is not RSS. Buffers remain charged until their actual NumPy owner
dies, including when a caller keeps a small view after closing a lease.
Allocation refusal never removes a source, person, label or camera code.
"""
from math import prod
import sys
from threading import RLock
import weakref
import numpy as np


class PlaybackWorkspace:
    # Bounded Python bookkeeping, not a per-candidate dictionary. At most 128
    # active claims, each with a short name and weak finalizer, fit this reserve.
    bookkeeping_bytes=64*2**10
    maximum_claims=128

    def __init__(self,limit_bytes):
        if type(limit_bytes) is not int or limit_bytes<self.bookkeeping_bytes:
            raise ValueError('explicit playback workspace including bookkeeping required')
        self.limit_bytes=limit_bytes;self._claims={};self._serial=0;self._lock=RLock()
        self._frozen_pins=[]
        self.current_bytes=self.peak_bytes=self.bookkeeping_bytes

    def claim(self,amount,owner):
        if type(amount) is not int or amount<0 or not isinstance(owner,str) or not 1<=len(owner)<=128:
            raise ValueError('bounded named allocation claim required')
        with self._lock:
            if len(self._claims)>=self.maximum_claims or self.current_bytes+amount>self.limit_bytes:
                raise MemoryError('playback workspace refused allocation before creation; no domain reduction')
            self._serial+=1;key=self._serial;self._claims[key]=(amount,owner)
            self.current_bytes+=amount;self.peak_bytes=max(self.peak_bytes,self.current_bytes)
            return key

    def release(self,key):
        with self._lock:
            if key not in self._claims:raise ValueError('allocation claim is not live')
            self.current_bytes-=self._claims.pop(key)[0]

    def empty(self,shape,dtype,*,owner):
        shape=tuple(shape);dtype=np.dtype(dtype)
        if (any(type(n) is not int or n<0 for n in shape) or dtype.kind not in 'biuf'
                or len(shape)>4):raise ValueError('bounded plain numeric array required')
        probe=np.empty((0,)*len(shape),dtype=dtype)
        header=sys.getsizeof(probe)-probe.nbytes
        key=self.claim(prod(shape)*dtype.itemsize+header,owner)
        try:value=np.empty(shape,dtype=dtype)
        except BaseException:self.release(key);raise
        weakref.finalize(value,self.release,key)
        return value

    def adopt_group(self,amount,buffers,*,owner):
        """Conservatively charge pre-existing frozen inputs until ALL die.

        This is not a pre-allocation guarantee for their earlier preparation.
        A group is never discounted just because one small view is evicted.
        Weak callback bookkeeping is explicitly included in the claim.
        """
        buffers=tuple(buffers)
        if not buffers or any(not isinstance(a,np.ndarray) for a in buffers):
            raise ValueError('actual frozen input buffers required')
        key=self.claim(amount+len(buffers)*512,owner)
        return self.pin_group(key,amount,buffers)

    def pin_frozen_inputs(self,amount,buffers,*,owner):
        """One lifetime for immutable source inputs retained for this workspace.

        The workspace strongly owns the actual arrays, not a weak callback
        for each tiny source array. This claim is NEVER discounted mid-job;
        external small views cannot make it undercount. Only plain arrays are
        pinned, not the context/callback graph or this workspace's own buffers.
        """
        buffers=tuple(buffers)
        if (type(amount) is not int or amount<0 or not buffers
                or any(not isinstance(value,np.ndarray) for value in buffers)):
            raise ValueError('actual immutable prepared source inputs required')
        key=self.claim(amount+sys.getsizeof(buffers)+1024,owner)
        try:
            for value in buffers:value.setflags(write=False)
            self._frozen_pins.append(buffers)
        except BaseException:self.release(key);raise
        return key

    def pin_group(self,key,amount,buffers):
        """Convert a pre-allocation reservation to actual buffer ownership."""
        buffers=tuple(buffers)
        if type(amount) is not int or amount<0 or not buffers:
            raise ValueError('actual grouped numeric ownership required')
        with self._lock:
            old,name=self._claims[key];new=amount+len(buffers)*512;change=new-old
            if self.current_bytes+change>self.limit_bytes:
                raise MemoryError('producer exceeded its preflight byte reservation')
            self._claims[key]=(new,name);self.current_bytes+=change;self.peak_bytes=max(self.peak_bytes,self.current_bytes)
        remaining=[len(buffers)]
        def released():
            remaining[0]-=1
            if not remaining[0]:self.release(key)
        for value in buffers:weakref.finalize(value,released)
        return key

    def receipt(self):
        with self._lock:
            return dict(limit_bytes=self.limit_bytes,current_bytes=self.current_bytes,peak_bytes=self.peak_bytes,
                bookkeeping_reservation_bytes=self.bookkeeping_bytes,active_claims=len(self._claims),
                immutable_source_groups_pinned_for_workspace_lifetime=len(self._frozen_pins),
                actual_buffer_lifetime_accounted=True,whole_process_RSS_included=False,
                owners=tuple((name,size) for size,name in self._claims.values()))
