"""Opt-in fixed-capacity backing for exact precompiled playback suffixes.

This primitive is NOT connected to a selector, the app or a fitted model.
One caller-selected directory holds one temporary fixed-slot file. Live
roots own slots, not an append-only history of attempted candidates. Raw
words decide identity after digest collisions; embeddings are checked too.
Only already compiled numeric inputs enter this store: no source/clock,
camera reconstruction, observation projection, encoder or media callback.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import RLock
import numpy as np
from .playback_suffix_arena import SuffixRoot,_owners
from .playback_workspace import PlaybackWorkspace


class StoredSuffix:
    def __init__(self,keys,values):self.keys=keys;self.values=values
    def close(self):self.keys=self.values=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


class StoredPrefixBank:
    def __init__(self,frames,keys,values):self.frames=frames;self.keys=keys;self.values=values
    def close(self):self.frames=self.keys=self.values=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def capacity_requirements(*,embedding_width,history_frames,capacity):
    """No file/array creation; resource BOUNDS, not a measured runtime pass."""
    if (type(embedding_width) is not int or not 1<=embedding_width<=16
            or type(history_frames) is not int or not 1<=history_frames<=480
            or type(capacity) is not int or not 1<=capacity<2**31):
        raise ValueError('bounded exact suffix dimensions required')
    buckets=1
    while buckets<2*capacity:buckets*=2
    shapes=((capacity,),(capacity,),(capacity,),(capacity,),(capacity,4),(capacity,4),(buckets,))
    header=sys.getsizeof(np.empty(0,np.uint64));slot=history_frames*(8+embedding_width)*8
    metadata=sum(int(np.prod(s))*8+header+16*(len(s)-1) for s in shapes)
    return dict(index_buckets=buckets,slot_bytes=slot,backing_bytes=capacity*slot,
        metadata_bytes=metadata,io_scratch_bytes=2*slot+4096+header,
        total_managed_index_and_io_bytes=metadata+2*slot+4096+header)


class BoundedSuffixSpool:
    """Explicit storage/byte limits; no payload eviction or domain reduction.

    The caller must budget both the managed workspace AND maximum disk bytes.
    Free space is checked before creation, not guaranteed against other jobs.
    Production consumers still need bank integration and whole-runtime tests.
    """
    def __init__(self,*,namespace,embedding_width,history_frames,capacity,
                 workspace,directory,maximum_spool_bytes,check=lambda:None):
        if (not isinstance(namespace,tuple) or not namespace or len(namespace)>32
                or any(not isinstance(v,str) or not 1<=len(v)<=128 for v in namespace)
                or type(embedding_width) is not int or not 1<=embedding_width<=16
                or type(history_frames) is not int or not 1<=history_frames<=480
                or type(capacity) is not int or not 1<=capacity<2**31
                or not isinstance(workspace,PlaybackWorkspace) or not callable(check)
                or type(maximum_spool_bytes) is not int or maximum_spool_bytes<1):
            raise ValueError('explicit finite namespace, dimensions and memory/disk limits required')
        directory=Path(directory)
        if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
            raise ValueError('explicit existing nonsymlink backing directory required')
        self.owner=next(_owners);self.width=embedding_width;self.H=history_frames;self.capacity=capacity
        self.workspace=workspace;self.directory=directory.resolve(strict=True);self.device=self.directory.stat().st_dev
        requirements=capacity_requirements(embedding_width=embedding_width,history_frames=history_frames,capacity=capacity)
        self.slot_bytes=requirements['slot_bytes'];self.maximum_bytes=requirements['backing_bytes']
        if self.maximum_bytes>maximum_spool_bytes:raise MemoryError('fixed suffix backing exceeds declared disk budget')
        buckets=requirements['index_buckets']
        specs=((capacity,),(capacity,),(capacity,),(capacity,),(capacity,4),(capacity,4),(buckets,))
        # readv fills an actual owned buffer: retained views/tracebacks keep
        # its allocation charged even after close. No untracked pread bytes.
        self.scratch_bytes=2*self.slot_bytes+4096
        if workspace.current_bytes+requirements['total_managed_index_and_io_bytes']>workspace.limit_bytes:
            raise MemoryError('fixed suffix metadata/scratch refused before file or array creation')
        check()
        if shutil.disk_usage(self.directory).free<self.maximum_bytes:
            raise OSError('insufficient free space for the explicit fixed suffix backing bound')
        self.check=check;self.binding=hashlib.sha256(json.dumps(namespace,separators=(',',':')).encode()).digest()
        self._lock=RLock();self.closed=False;self.fd=None;self.path=None;self._io=None
        self.free_count=capacity;self.live=self.peak_live=self.read_bytes=self.written_bytes=self.probes=0
        self.arrays=[]
        try:
            for shape in specs:self.arrays.append(workspace.empty(shape,np.uint64,owner='bounded-suffix-spool-index'))
            self.refs,self.generations,self.free,self.ends,self.digests,self.checksums,self.buckets=self.arrays
            self.refs[:]=0;self.generations[:]=1;self.free[:]=np.arange(capacity-1,-1,-1,dtype=np.uint64)
            self.ends[:]=0;self.digests[:]=0;self.checksums[:]=0;self.buckets[:]=0
            self._io=workspace.empty((self.scratch_bytes,),np.uint8,owner='bounded-suffix-spool-IO-scratch')
            self.fd,name=tempfile.mkstemp(prefix='ive-crosscut-owned-suffix-',suffix='.bin',dir=self.directory)
            self.path=Path(name);stat=os.fstat(self.fd);self.file_identity=(stat.st_dev,stat.st_ino)
            if stat.st_dev!=self.device:raise OSError('suffix backing directory identity changed')
            os.ftruncate(self.fd,self.maximum_bytes)
        except BaseException:
            self._close_file()
            self._io=None;self.arrays=[]
            raise

    def _live(self):
        if self.closed:raise ValueError('suffix backing is closed')
        self.check()
        if self.directory.stat().st_dev!=self.device:raise OSError('suffix backing device changed')

    def _resolve(self,root):
        if self.closed:raise ValueError('suffix backing is closed')
        if not isinstance(root,SuffixRoot) or root.arena!=self.owner:raise ValueError('suffix root belongs to another store')
        slot=(root.handle&0xffffffff)-1;generation=root.handle>>32
        if slot<0 or slot>=self.capacity or not self.refs[slot] or int(self.generations[slot])!=generation:
            raise ValueError('stale suffix backing handle')
        return slot

    def _digest(self,keys):
        h=hashlib.sha256(self.binding)
        if len(keys):h.update(memoryview(keys).cast('B'))
        return h.digest()

    def _checksum(self,keys,values):
        h=hashlib.sha256(self.binding)
        if len(keys):h.update(memoryview(keys).cast('B'));h.update(memoryview(values).cast('B'))
        return h.digest()

    def _home(self,digest):return int.from_bytes(digest[:8],'little')&(len(self.buckets)-1)

    def _write(self,buffer,offset):
        if not len(buffer):return
        view=memoryview(buffer).cast('B');at=0
        while at<len(view):
            self._live();n=os.pwrite(self.fd,view[at:],offset+at)
            if n<=0:raise OSError('suffix backing write made no progress')
            self.written_bytes+=n;at+=n

    def _read(self,count,offset,into):
        self._live();data=self._io[into:into+count];at=0;os.lseek(self.fd,offset,os.SEEK_SET)
        while at<count:
            self._live();n=os.readv(self.fd,[memoryview(data[at:])])
            if n<=0:raise OSError('short exact suffix backing read')
            self.read_bytes+=n;at+=n
        return data

    def _payload(self,slot):
        count=min(self.H,int(self.ends[slot]));start=slot*self.slot_bytes
        keys=self._read(count*64,start,0)
        values=self._read(count*self.width*8,start+self.H*64,self.H*64)
        h=hashlib.sha256(self.binding);h.update(keys);h.update(values)
        if h.digest()!=self.checksums[slot].tobytes():raise OSError('exact suffix backing checksum changed')
        return keys,values

    def intern(self,keys,values,*,final):
        keys=np.asarray(keys);values=np.asarray(values)
        if (type(final) is not int or not 0<=final<2**63 or keys.dtype!=np.uint64 or keys.shape!=(min(final,self.H),8)
                or values.dtype!=np.float64 or values.shape!=(len(keys),self.width)
                or not keys.flags.c_contiguous or not values.flags.c_contiguous or not np.isfinite(values).all()
                or not np.array_equal(keys[:,0],np.arange(max(0,final-self.H),final,dtype=np.uint64))):
            raise ValueError('complete exact precompiled raw suffix and finite embeddings required')
        with self._lock:
            self._live();digest=self._digest(keys);words=np.frombuffer(digest,np.uint64);at=self._home(digest)
            while self.buckets[at]:
                self.probes+=1;slot=int(self.buckets[at])-1
                if int(self.ends[slot])==final and np.array_equal(self.digests[slot],words):
                    raw,encoded=self._payload(slot)
                    if not len(keys) or memoryview(raw)==memoryview(keys).cast('B'):
                        if len(keys) and memoryview(encoded)!=memoryview(values).cast('B'):
                            raise ValueError('same exact raw input has different model embeddings')
                        self.refs[slot]+=np.uint64(1)
                        return SuffixRoot(self.owner,(int(self.generations[slot])<<32)|slot+1)
                at=(at+1)&(len(self.buckets)-1)
            if not self.free_count:raise MemoryError('live suffix backing capacity exceeded; no candidates removed')
            slot=int(self.free[self.free_count-1]);start=slot*self.slot_bytes
            # Publish nothing until both pieces were written successfully.
            self._write(keys,start);self._write(values,start+self.H*64)
            self.free_count-=1;self.refs[slot]=1;self.ends[slot]=final;self.digests[slot]=words
            self.checksums[slot]=np.frombuffer(self._checksum(keys,values),np.uint64);self.buckets[at]=slot+1
            self.live+=1;self.peak_live=max(self.peak_live,self.live)
            return SuffixRoot(self.owner,(int(self.generations[slot])<<32)|slot+1)

    def retain(self,root):
        with self._lock:self.refs[self._resolve(root)]+=np.uint64(1)
        return root

    def release(self,root):
        with self._lock:
            # Cleanup must still work after the caller's deadline/cancel
            # callback fails; it owns only metadata and performs no disk IO.
            slot=self._resolve(root);self.refs[slot]-=np.uint64(1)
            if self.refs[slot]:return
            mask=len(self.buckets)-1;position=self._home(self.digests[slot].tobytes())
            while int(self.buckets[position])!=slot+1:position=(position+1)&mask
            hole=position;position=(hole+1)&mask
            while self.buckets[position]:
                other=int(self.buckets[position])-1;home=self._home(self.digests[other].tobytes())
                if ((hole-home)&mask)<((position-home)&mask):self.buckets[hole]=self.buckets[position];hole=position
                position=(position+1)&mask
            self.buckets[hole]=0;self.live-=1
            if self.generations[slot]<0xffffffff:
                self.generations[slot]+=np.uint64(1);self.free[self.free_count]=slot;self.free_count+=1

    def read(self,root):
        with self._lock:
            self._live();slot=self._resolve(root);count=min(self.H,int(self.ends[slot]));w=self.workspace
            keys=w.empty((count,8),np.uint64,owner='loaded-exact-suffix-keys')
            values=w.empty((count,self.width),np.float64,owner='loaded-exact-suffix-embeddings')
            raw,encoded=self._payload(slot)
            keys[:]=np.frombuffer(raw,np.uint64).reshape(keys.shape)
            values[:]=np.frombuffer(encoded,np.float64).reshape(values.shape)
            keys.setflags(write=False);values.setflags(write=False)
            return StoredSuffix(keys,values)

    def decode_handles(self,handles,*,end):
        """One bank per finalized origin; numeric IO, never source replay.

        All roots must remain owned by their label bank during this call.
        Both raw keys and embeddings are needed for exact future extension;
        neither a scalar cost nor an embedding alone certifies input equality.
        """
        handles=np.asarray(handles)
        if handles.ndim!=1 or handles.dtype!=np.uint64 or type(end) is not int or not 0<=end<2**63:
            raise ValueError('packed owned handles and one exact prefix end required')
        with self._lock:
            self._live();count=min(self.H,end);B=len(handles);w=self.workspace
            ticks=w.empty((count,),np.int64,owner='spooled-prefix-bank-ticks');ticks[:]=np.arange(end-count,end)
            keys=w.empty((B,count,8),np.uint64,owner='spooled-prefix-bank-raw-inputs')
            values=w.empty((B,count,self.width),np.float64,owner='spooled-prefix-bank-embeddings')
            for lane,handle in enumerate(handles):
                slot=self._resolve(SuffixRoot(self.owner,int(handle)))
                if int(self.ends[slot])!=end:raise ValueError('prefix bank mixes different finalized ends')
                raw,encoded=self._payload(slot)
                keys[lane]=np.frombuffer(raw,np.uint64).reshape(count,8)
                values[lane]=np.frombuffer(encoded,np.float64).reshape(count,self.width)
            ticks.setflags(write=False);keys.setflags(write=False);values.setflags(write=False)
            return StoredPrefixBank(ticks,keys,values)

    def receipt(self):
        with self._lock:
            if self.closed:raise ValueError('suffix backing is closed')
            try:stat=os.fstat(self.fd)
            except OSError:stat=None
            return dict(contract='fixed_slot_exact_precompiled_suffix_backing_prototype_v1',
                capacity=self.capacity,live_roots=self.live,peak_live_roots=self.peak_live,
                backing_logical_bytes=None if stat is None else stat.st_size,
                backing_allocated_bytes=None if stat is None else stat.st_blocks*512,
                declared_maximum_backing_bytes=self.maximum_bytes,read_bytes=self.read_bytes,
                written_bytes=self.written_bytes,collision_probes=self.probes,workspace=self.workspace.receipt(),
                append_only_candidate_history=False,raw_input_compared_after_digest_collision=True,
                normal_app_or_selector_connected=False,whole_runtime_or_media_quality_approved=False)

    def _close_file(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.path is not None and self.path.exists():
            stat=self.path.lstat()
            if self.path.is_symlink() or (stat.st_dev,stat.st_ino)!=self.file_identity:
                raise OSError('suffix backing pathname replaced; refusing to remove another file')
            self.path.unlink()

    def close(self):
        with self._lock:
            if self.closed:return
            if self.live:raise ValueError('cannot close suffix backing with live roots')
            self._close_file();self.closed=True
            self._io=None;self.arrays=[]
            self.refs=self.generations=self.free=self.ends=self.digests=self.checksums=self.buckets=None
