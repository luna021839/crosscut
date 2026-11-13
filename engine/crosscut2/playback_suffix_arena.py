"""Packed, collision-safe, reference-owned physical suffix DAG.

Raw frame keys are admitted BEFORE prefix expansion. State extension only
uses those exact frame token IDs, six-frame cell IDs and root IDs. No camera,
clock, observation projection, model encoding or raw-key packing is available
inside extension/decode. A root owns cells; a cell owns frame tokens. Slots
are generation checked and zero-reference keys leave their interner at once.

This component is not a selector, a fitted model or a whole-runtime claim.
Its namespace must be bound by the consumer to exact model/source/observation
and pre-camera intent registries; arbitrary numeric source handles do not
certify physical inputs. Composite/hybrid/dissolve inputs need their own schema.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
import sys
from threading import RLock
import numpy as np
from .playback_workspace import PlaybackWorkspace

_owners=count(1)
_LOW=np.uint64(0xffffffff)


def _resolve(table,handle):
    slot=np.int64(handle&_LOW)-1
    if (slot<0 or slot>=len(table[2]) or table[2][slot]==0
            or table[3][slot]!=(handle>>np.uint64(32))):
        raise ValueError('stale physical suffix arena handle')
    return slot


def _hash(row,mask):
    h=np.uint64(14695981039346656037)
    for word in row:h=(h^word)*np.uint64(1099511628211)
    return h&mask


def _intern(table,row,mask):
    keys,values,refs,generations,slots,hashes,free,control=table
    h=_hash(row,mask);position=np.int64(h&np.uint64(len(slots)-1))
    control[2]+=1
    while slots[position]>=0:
        control[5]+=1;slot=slots[position];same=hashes[slot]==h
        if same:
            for k in range(len(row)):
                if keys[slot,k]!=row[k]:same=False;break
        if same:
            refs[slot]+=1;control[3]+=1
            return (generations[slot]<<np.uint64(32))|np.uint64(slot+1),False
        position=(position+1)&(len(slots)-1)
    if control[0]==0:return np.uint64(0),False
    control[0]-=1;slot=free[control[0]];slots[position]=slot
    keys[slot]=row;hashes[slot]=h;refs[slot]=1;control[1]+=1
    control[4]=max(control[4],control[1])
    return (generations[slot]<<np.uint64(32))|np.uint64(slot+1),True


def _unref(table,handle):
    keys,values,refs,generations,slots,hashes,free,control=table
    slot=_resolve(table,handle);refs[slot]-=1
    if refs[slot]:return False,slot
    # Back-shift deletion: no accumulating tombstones, including under a
    # deliberately forced hash collision and thousands of replacements.
    position=np.int64(hashes[slot]&np.uint64(len(slots)-1))
    while slots[position]!=slot:position=(position+1)&(len(slots)-1)
    hole=position;position=(hole+1)&(len(slots)-1)
    while slots[position]>=0:
        other=slots[position];home=np.int64(hashes[other]&np.uint64(len(slots)-1))
        if ((hole-home)&(len(slots)-1))<((position-home)&(len(slots)-1)):
            slots[hole]=other;hole=position
        position=(position+1)&(len(slots)-1)
    slots[hole]=-1;control[1]-=1
    # Never wrap a generation and make an old handle live again.
    if generations[slot]<_LOW:
        generations[slot]+=np.uint64(1);free[control[0]]=slot;control[0]+=1
    return True,slot


def _release_cell(cells,frames,handle):
    freed,slot=_unref(cells,handle)
    if freed:
        for at in range(np.int64(cells[0][slot,0])):_unref(frames,cells[0][slot,at+1])


def _release_root(roots,cells,frames,handle):
    freed,slot=_unref(roots,handle)
    if freed:
        for at in range(np.int64(roots[0][slot,1])):_release_cell(cells,frames,roots[0][slot,at+2])


def _release_frames(frames,handles):
    for handle in handles:_unref(frames,handle)


def _admit_frames(table,keys,encoded,handles,mask):
    error=0;processed=0;saved=table[1].view(np.uint64);incoming=encoded.view(np.uint64)
    for at in range(len(keys)):
        handle,new=_intern(table,keys[at],mask)
        if handle==0:error=1;break
        handles[at]=handle;processed=at+1;slot=_resolve(table,handle)
        if new:table[1][slot]=encoded[at]
        else:
            for d in range(encoded.shape[1]):
                if saved[slot,d]!=incoming[at,d]:error=2;break
        if error:break
    if error:
        for at in range(processed):_unref(table,handles[at])
    return error


def _probe_frames(table,keys,out,found,mask):
    for at in range(len(keys)):
        row=keys[at];h=_hash(row,mask);position=np.int64(h&np.uint64(len(table[4])-1));found[at]=False
        while table[4][position]>=0:
            slot=table[4][position];same=table[5][slot]==h
            if same:
                for k in range(len(row)):
                    if table[0][slot,k]!=row[k]:same=False;break
            if same:
                out[at]=table[1][slot];found[at]=True;break
            position=(position+1)&(len(table[4])-1)


def _extend(roots,cells,frames,parent,current,first,final,H,mask,flat,cell_row,root_row,held_cells):
    parent_slot=_resolve(roots,parent)
    if np.int64(roots[0][parent_slot,0])!=first:return np.uint64(0),2
    want=max(0,final-H);size=0
    if want<first:
        for c in range(np.int64(roots[0][parent_slot,1])):
            cell=_resolve(cells,roots[0][parent_slot,c+2])
            for f in range(np.int64(cells[0][cell,0])):
                token=cells[0][cell,f+1];tick=np.int64(frames[0][_resolve(frames,token),0])
                if tick>=want:
                    if size>=H:return np.uint64(0),2
                    flat[size]=token;size+=1
    for token in current:
        tick=np.int64(frames[0][_resolve(frames,token),0])
        if tick>=want:
            if size>=H:return np.uint64(0),2
            flat[size]=token;size+=1
    if size!=final-want:return np.uint64(0),2
    for at in range(size):
        if np.int64(frames[0][_resolve(frames,flat[at]),0])!=want+at:return np.uint64(0),2
    root_row[:]=0;root_row[0]=final;held=0;at=0;error=0
    while at<size:
        tick=want+at;length=min(size-at,6-tick%6);cell_row[:]=0;cell_row[0]=length
        for k in range(length):cell_row[k+1]=flat[at+k]
        child,new=_intern(cells,cell_row,mask)
        if child==0:error=1;break
        held_cells[held]=child;root_row[held+2]=child;held+=1
        if new:
            for k in range(length):frames[2][_resolve(frames,cell_row[k+1])]+=1
        at+=length
    result=np.uint64(0)
    if not error:
        root_row[1]=held;result,new=_intern(roots,root_row,mask)
        if result==0:error=1
        elif new:
            for k in range(held):cells[2][_resolve(cells,held_cells[k])]+=1
    for k in range(held):_release_cell(cells,frames,held_cells[k])
    return result,error


def _decode(roots,cells,frames,handles,first,final,out):
    for lane in range(len(handles)):
        root=_resolve(roots,handles[lane])
        if np.int64(roots[0][root,0])!=final:raise ValueError('prefix bank mixes different finalized origins')
        seen=0
        for c in range(np.int64(roots[0][root,1])):
            cell=_resolve(cells,roots[0][root,c+2])
            for f in range(np.int64(cells[0][cell,0])):
                token=_resolve(frames,cells[0][cell,f+1]);tick=np.int64(frames[0][token,0])
                if tick!=first+seen:raise ValueError('physical suffix has a missing or reordered output tick')
                out[lane,seen]=frames[1][token];seen+=1
        if seen!=final-first:raise ValueError('physical prefix bank does not cover its exact recent interval')


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    from numba.extending import register_jitable
    for function in (_resolve,_hash,_intern,_unref,_release_cell,_release_root,_release_frames,
            _admit_frames,_extend,_decode,_probe_frames):register_jitable(function)
    options=dict(cache=False,nogil=True,fastmath=False,parallel=False)
    return tuple(njit(**options)(f) for f in (_admit_frames,_extend,_release_root,_release_frames,_decode,_probe_frames))


@dataclass(frozen=True,slots=True)
class SuffixRoot:
    arena: int
    handle: int


class _DecodedLease:
    def __init__(self,frames,values):self._frames=frames;self._values=values
    @property
    def frames(self):
        if self._frames is None:raise ValueError('prefix bank lease is closed')
        return self._frames
    @property
    def values(self):
        if self._values is None:raise ValueError('prefix bank lease is closed')
        return self._values
    def close(self):self._frames=self._values=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


class _FrameLease:
    def __init__(self,arena,handles,first,final):
        self._arena=arena;self._ids=handles;self.first=first;self.final=final
    def close(self):
        if self._arena is not None:
            arena=self._arena
            with arena._lock:arena._functions[3](arena._frames,self._ids)
            self._ids=None;self._arena=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


class PlaybackSuffixArena:
    def __init__(self,*,namespace,embedding_width,history_frames=120,frame_capacity=4096,
                 cell_capacity=1024,root_capacity=256,workspace=None,maximum_bytes=32*2**20,
                 kernel='numpy',hash_mask=0xffffffffffffffff):
        if (not isinstance(namespace,tuple) or not namespace or len(namespace)>32
                or any(not isinstance(v,str) or not 1<=len(v)<=128 for v in namespace)
                or type(embedding_width) is not int or not 1<=embedding_width<=16
                or type(history_frames) is not int or not 1<=history_frames<=480
                or any(type(n) is not int or not 1<=n<2**31 for n in (frame_capacity,cell_capacity,root_capacity))
                or kernel not in ('numpy','numba') or type(hash_mask) is not int or not 0<=hash_mask<2**64):
            raise ValueError('bounded exact physical suffix arena contract required')
        self.namespace=namespace;self.owner=next(_owners);self.width=embedding_width;self.H=history_frames
        self.workspace=PlaybackWorkspace(maximum_bytes) if workspace is None else workspace
        if not isinstance(self.workspace,PlaybackWorkspace):raise ValueError('one playback allocation owner required')
        self.kernel=kernel;self.mask=np.uint64(hash_mask);self._lock=RLock();self.closed=False
        self.maximum_cells=(history_frames+10)//6
        self._functions=(_admit_frames,_extend,_release_root,_release_frames,_decode,_probe_frames) if kernel=='numpy' else _compiled()
        # Preflight ALL fixed arrays before creating any of them. The same
        # workspace owns hash slots, free lists, generations and scratch.
        specs=[]
        for n,key_width,value_width in ((frame_capacity,8,embedding_width),
                (cell_capacity,7,0),(root_capacity,self.maximum_cells+2,0)):
            cap=1
            while cap<2*n:cap*=2
            specs.append(((n,key_width),(n,value_width),(n,),(n,),(cap,),(n,),(n,),(6,)))
        probe=np.empty((0,),np.uint64);header=sys.getsizeof(probe)
        planned=sum(sum(int(np.prod(s))*8+header+16*(len(s)-1) for s in table) for table in specs)
        planned+=sum(n*8+header for n in (history_frames,7,self.maximum_cells+2,self.maximum_cells))
        if self.workspace.current_bytes+planned>self.workspace.limit_bytes:
            raise MemoryError('packed suffix arena refused before fixed-array allocation')
        def table(shapes,name):
            arrays=tuple(self.workspace.empty(s,np.int64 if k in (4,6,7) else
                np.float64 if k==1 else np.uint64,owner=name) for k,s in enumerate(shapes))
            arrays[2][:]=0;arrays[3][:]=1;arrays[4][:]=-1;arrays[6][:]=np.arange(len(arrays[2]))
            arrays[7][:]=0;arrays[7][0]=len(arrays[2]);return arrays
        self._frames=table(specs[0],'physical-frame-table');self._cells=table(specs[1],'canonical-cell-table')
        self._roots=table(specs[2],'canonical-root-table')
        self._scratch=tuple(self.workspace.empty((n,),np.uint64,owner='suffix-extension-scratch') for n in
            (history_frames,7,self.maximum_cells+2,self.maximum_cells))

    def _handle(self,root):
        if self.closed or not isinstance(root,SuffixRoot) or root.arena!=self.owner:
            raise ValueError('root belongs to another or closed physical suffix arena')
        return np.uint64(root.handle)

    def empty(self):
        if self.closed:raise ValueError('physical suffix arena is closed')
        with self._lock,np.errstate(over='ignore'):
            row=self._scratch[2];row[:]=0;handle,_=_intern(self._roots,row,self.mask)
            if not handle:raise MemoryError('live root capacity exceeded; no labels removed')
            return SuffixRoot(self.owner,int(handle))

    def admit(self,keys,encoded,*,first=None):
        keys=np.asarray(keys);encoded=np.asarray(encoded)
        if (self.closed or keys.ndim!=2 or keys.shape[1]!=8 or keys.dtype!=np.uint64
                or not keys.flags.c_contiguous or not len(keys) or len(keys)>self.H
                or encoded.shape!=(len(keys),self.width) or encoded.dtype!=np.float64
                or not encoded.flags.c_contiguous or not np.isfinite(encoded).all()
                or np.any(np.diff(keys[:,0])!=1) or keys[-1,0]>=2**63-1):
            raise ValueError('precompiled contiguous exact tail keys and encoded frames required')
        a=int(keys[0,0]);final=int(keys[-1,0])+1;first=a if first is None else first
        if type(first) is not int or first<0 or first>a or a!=max(first,final-self.H):
            raise ValueError('compiled descriptor must retain its complete exact recent hold tail')
        handles=self.workspace.empty((len(keys),),np.uint64,owner='compiled-current-frame-lease')
        with self._lock,np.errstate(over='ignore'):
            error=self._functions[0](self._frames,keys,encoded,handles,self.mask)
        if error==1:raise MemoryError('live frame capacity exceeded; no camera alternatives removed')
        if error:raise ValueError('same exact input key produced different model embeddings')
        handles.setflags(write=False)
        return _FrameLease(self,handles,first,final)

    def append(self,parent,current):
        handle=self._handle(parent)
        if not isinstance(current,_FrameLease) or current._arena is not self:
            raise ValueError('current descriptor lease belongs to another or closed arena')
        with self._lock,np.errstate(over='ignore'):
            result,error=self._functions[1](self._roots,self._cells,self._frames,handle,current._ids,
                current.first,current.final,self.H,self.mask,*self._scratch)
        if error==1:raise MemoryError('live cell/root capacity exceeded; no beam or source reduction')
        if error:raise ValueError('physical suffix/current descriptor interval mismatch')
        return SuffixRoot(self.owner,int(result))

    def probe(self,keys,out,found):
        """Compile-phase lookup by original words; no future state is merged."""
        if (self.closed or not isinstance(keys,np.ndarray) or keys.dtype!=np.uint64 or keys.ndim!=2 or keys.shape[1]!=8
                or out.shape!=(len(keys),self.width) or out.dtype!=np.float64
                or found.shape!=(len(keys),) or found.dtype!=bool or not out.flags.writeable or not found.flags.writeable):
            raise ValueError('exact compile-phase lookup buffers required')
        with self._lock,np.errstate(over='ignore'):self._functions[5](self._frames,keys,out,found,self.mask)

    def retain(self,root):
        handle=self._handle(root)
        with self._lock:self._roots[2][_resolve(self._roots,handle)]+=1
        return root

    def release(self,root):
        handle=self._handle(root)
        with self._lock:self._functions[2](self._roots,self._cells,self._frames,handle)

    def decode(self,roots,*,end):
        if type(end) is not int or end<0:raise ValueError('exact finalized prefix end required')
        handles=self.workspace.empty((len(roots),),np.uint64,owner='prefix-root-handle-lease')
        for at,root in enumerate(roots):handles[at]=self._handle(root)
        return self.decode_handles(handles,end=end)

    def decode_handles(self,handles,*,end):
        """Arena-local packed IDs for its owning consumer, not cross-arena IDs."""
        handles=np.asarray(handles)
        if (self.closed or handles.ndim!=1 or handles.dtype!=np.uint64
                or type(end) is not int or end<0):raise ValueError('owned exact prefix root handles required')
        first=max(0,end-self.H);ticks=self.workspace.empty((end-first,),np.int64,owner='prefix-bank-clock')
        values=self.workspace.empty((len(handles),end-first,self.width),np.float64,owner='prefix-bank-embeddings')
        ticks[:]=np.arange(first,end)
        with self._lock:self._functions[4](self._roots,self._cells,self._frames,handles,first,end,values)
        ticks.setflags(write=False);values.setflags(write=False)
        return _DecodedLease(ticks,values)

    def receipt(self):
        if self.closed:raise ValueError('physical suffix arena is closed')
        return dict(contract='owned_exact_frame_cell_suffix_DAG_v1',kernel=self.kernel,
            history_frames=self.H,storage_cell_frames=6,per_frame_input_preserved=True,
            identity_not_based_on_embeddings=True,zero_reference_keys_removed=True,
            generation_checked_handles=True,namespace=self.namespace,
            tables={name:dict(capacity=len(t[2]),live=int(t[7][1]),peak_live=int(t[7][4]),
                requests=int(t[7][2]),hits=int(t[7][3]),hash_probes=int(t[7][5])) for name,t in
                (('frames',self._frames),('cells',self._cells),('roots',self._roots))},
            workspace=self.workspace.receipt(),source_or_camera_candidates_removed=False,
            whole_runtime_or_media_approved=False)

    def close(self):
        with self._lock:
            if self.closed:return
            if any(t[7][1] for t in (self._roots,self._cells,self._frames)):
                raise ValueError('cannot discard an arena with live roots or compiled frame leases')
            self.closed=True;self._roots=self._cells=self._frames=self._scratch=None
