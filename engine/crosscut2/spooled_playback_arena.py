"""Explicit opt-in fixed-slot numeric suffix storage for compiled playback.

Inputs are compiled BEFORE history expansion. Current leases are transient;
only a beam-viable append interns a persistent root. During selection a single
bound origin bank supplies past raw words and embeddings without rereading
the file per prefix or reconstructing any camera/NI/observation input.
No normal app, trained model or whole-runtime/media approval is implied.
"""
import sys
import numpy as np
from .bounded_suffix_spool import BoundedSuffixSpool,StoredPrefixBank
from .playback_suffix_arena import SuffixRoot


class _CurrentLease:
    def __init__(self,arena,keys,values,first,final):
        self._arena=arena;self.keys=keys;self.values=values;self.first=first;self.final=final
        arena.current_leases+=1;arena.peak_current_leases=max(arena.peak_current_leases,arena.current_leases)
    def close(self):
        if self._arena is not None:self._arena.current_leases-=1
        self._arena=None;self.keys=self.values=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


class SpooledPlaybackArena:
    def __init__(self,*,namespace,embedding_width,history_frames,root_capacity,workspace,
                 directory,maximum_spool_bytes,kernel='numpy',check=lambda:None,reserve_disk=True):
        if kernel not in ('numpy','numba'):raise ValueError('explicit strict numeric metadata kernel required')
        self.namespace=namespace;self.width=embedding_width;self.H=history_frames;self.workspace=workspace
        self.kernel=kernel;self.closed=False;self.current_leases=self.peak_current_leases=0
        self.origin_banks_bound=self.root_appends=self.probe_calls=0;self._bank=None;self._order=self._sorted=None
        self.store=BoundedSuffixSpool(namespace=namespace,embedding_width=embedding_width,history_frames=history_frames,
            capacity=root_capacity,workspace=workspace,directory=directory,maximum_spool_bytes=maximum_spool_bytes,
            check=check,reserve_disk=reserve_disk)
        self.owner=self.store.owner;self._lock=self.store._lock
        try:
            self._keys=workspace.empty((history_frames,8),np.uint64,owner='spooled-suffix-assembly-raw-inputs')
            self._values=workspace.empty((history_frames,embedding_width),np.float64,owner='spooled-suffix-assembly-embeddings')
            self._empty_keys=workspace.empty((0,8),np.uint64,owner='spooled-empty-input')
            self._empty_values=workspace.empty((0,embedding_width),np.float64,owner='spooled-empty-embedding')
        except BaseException:self.store.close();raise

    def _handle(self,root):
        if self.closed or not isinstance(root,SuffixRoot) or root.arena!=self.owner:
            raise ValueError('root belongs to another or closed spooled arena')
        self.store._resolve(root);return np.uint64(root.handle)

    def empty(self):
        if self.closed:raise ValueError('spooled arena is closed')
        return self.store.intern(self._empty_keys,self._empty_values,final=0)

    def admit(self,keys,encoded,*,first=None):
        keys=np.asarray(keys);encoded=np.asarray(encoded)
        if (self.closed or keys.dtype!=np.uint64 or keys.ndim!=2 or keys.shape[1]!=8 or not 0<len(keys)<=self.H
                or encoded.dtype!=np.float64 or encoded.shape!=(len(keys),self.width)
                or not keys.flags.c_contiguous or not encoded.flags.c_contiguous or not np.isfinite(encoded).all()
                or np.any(np.diff(keys[:,0])!=1) or keys[-1,0]>=2**63-1):
            raise ValueError('complete exact compiled current tail required')
        begin=int(keys[0,0]);final=int(keys[-1,0])+1;first=begin if first is None else first
        if type(first) is not int or first<0 or first>begin or begin!=max(first,final-self.H):
            raise ValueError('current tail does not cover its exact recent interval')
        raw=self.workspace.empty(keys.shape,np.uint64,owner='transient-spooled-current-raw-inputs');raw[:]=keys
        values=self.workspace.empty(encoded.shape,np.float64,owner='transient-spooled-current-embeddings');values[:]=encoded
        raw.setflags(write=False);values.setflags(write=False)
        return _CurrentLease(self,raw,values,first,final)

    def probe(self,keys,out,found):
        if (self.closed or keys.dtype!=np.uint64 or keys.shape!=(len(keys),8)
                or out.shape!=(len(keys),self.width) or out.dtype!=np.float64
                or found.shape!=(len(keys),) or found.dtype!=bool or not found.flags.writeable):
            raise ValueError('explicit current lookup buffers required')
        # There is no hidden resident frame cache. Exact frozen baseline
        # reuse remains in CompiledPlayback; lost live-frame reuse is counted
        # and must pass the actual descriptor IO/compute gate, not assumed free.
        found[:]=False;self.probe_calls+=1

    def bind_origin(self,handles,bank):
        if self._bank is not None or not isinstance(bank,StoredPrefixBank) or bank.values.shape[0]!=len(handles):
            raise ValueError('one complete decoded raw+embedding origin bank required')
        w=self.workspace;reservation=w.claim(len(handles)*8+1024,'spooled-origin-root-to-lane-order')
        try:order=np.argsort(handles,kind='quicksort')
        except BaseException:w.release(reservation);raise
        w.pin_group(reservation,sys.getsizeof(order),[order])
        sorted_handles=w.empty((len(handles),),np.uint64,owner='spooled-origin-sorted-root-handles')
        for at,lane in enumerate(order):sorted_handles[at]=handles[lane]
        self._order=order;self._sorted=sorted_handles;self._bank=bank;self.origin_banks_bound+=1

    def unbind_origin(self):self._bank=self._order=self._sorted=None

    def append(self,parent,current):
        handle=self._handle(parent)
        if not isinstance(current,_CurrentLease) or current._arena is not self:
            raise ValueError('owned transient compiled current lease required')
        with self._lock:
            slot=self.store._resolve(parent)
            if int(self.store.ends[slot])!=current.first:raise ValueError('prefix/current interval mismatch')
            first=max(0,current.final-self.H);count=current.final-first;at=0
            keys=self._keys[:count];values=self._values[:count]
            if first<current.first:
                if self._bank is None:raise ValueError('short extension requires a bound decoded origin bank')
                where=int(np.searchsorted(self._sorted,handle))
                if where==len(self._sorted) or self._sorted[where]!=handle:raise ValueError('prefix not retained in origin bank')
                lane=int(self._order[where]);start=int(np.searchsorted(self._bank.frames,first))
                old_keys=self._bank.keys[lane,start:];old_values=self._bank.values[lane,start:]
                at=len(old_keys);keys[:at]=old_keys;values[:at]=old_values
            start=int(np.searchsorted(current.keys[:,0],first));length=len(current.keys)-start
            if at+length!=count:raise ValueError('assembled suffix has a gap or overlapping output tick')
            keys[at:]=current.keys[start:];values[at:]=current.values[start:]
            root=self.store.intern(keys,values,final=current.final);self.root_appends+=1;return root

    def retain(self,root):self._handle(root);return self.store.retain(root)
    def release(self,root):self._handle(root);self.store.release(root)
    def decode_handles(self,handles,*,end):return self.store.decode_handles(handles,end=end)
    def decode(self,roots,*,end):
        handles=self.workspace.empty((len(roots),),np.uint64,owner='spooled-decoded-root-handles')
        for at,root in enumerate(roots):handles[at]=self._handle(root)
        return self.decode_handles(handles,end=end)

    def receipt(self):
        r=self.store.receipt()
        return dict(contract='fixed_slot_exact_raw_suffix_with_one_bound_origin_bank_v1',kernel=self.kernel,
            history_frames=self.H,namespace=self.namespace,backing=r,
            origin_banks_bound=self.origin_banks_bound,root_append_calls=self.root_appends,
            frame_probe_calls=self.probe_calls,hidden_resident_frame_cache=False,
            tables=dict(roots=dict(capacity=self.store.capacity,live=self.store.live,peak_live=self.store.peak_live),
                transient_current_leases=dict(live=self.current_leases,peak_live=self.peak_current_leases)),
            workspace=self.workspace.receipt(),source_or_camera_candidates_removed=False,
            normal_app_activation_approved=False,whole_runtime_or_media_approved=False)

    def close(self):
        if self.closed:return
        if self._bank is not None or self.current_leases:raise ValueError('cannot close a live current or origin bank')
        self.store.close();self.closed=True
        self._keys=self._values=self._empty_keys=self._empty_values=None
