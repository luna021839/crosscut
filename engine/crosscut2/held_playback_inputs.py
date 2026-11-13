"""Freeze actual hold geometry once, BEFORE contextual history expansion.

This records already-computed direct paths and shared baseline endpoint
patches. It changes no admission, target, camera, source clock or base cost.
Only one hold arena is owned by ArcCosts; retained DP labels must use compiled
tokens, not retain these per-candidate camera arrays or reconstruct old holds.
"""
from dataclasses import dataclass
from types import MappingProxyType
import sys
import numpy as np
from .programme_source_clock import indices


def frozen(value,dtype=None):
    out=np.array(value,dtype=dtype,copy=True);out.setflags(write=False);return out


@dataclass(frozen=True)
class HeldInputs:
    first_node: int
    final_node: int
    first_frame: int
    final_frame: int
    allowed: np.ndarray
    source_of_entry: tuple
    native: dict
    baselines: dict
    patches: dict
    direct: dict
    stored_bytes: int
    shared_baseline_bytes: int

    def sample(self,entry,exit,code,frames,*,camera_out=None):
        frames=np.asarray(frames)
        if (frames.ndim!=1 or frames.dtype.kind not in 'iu' or np.any(np.diff(frames)<=0)
                or frames.min(initial=self.first_frame)<self.first_frame
                or frames.max(initial=self.first_frame)>=self.final_frame
                or type(code) is not int or not 0<=code<6
                or not 0<=entry<self.allowed.shape[0] or not 0<=exit<self.allowed.shape[1]
                or not self.allowed[entry,exit,code]):
            raise ValueError('only exact in-domain samples of an already admitted camera alternative are available')
        camera=np.empty((len(frames),3),float) if camera_out is None else np.asarray(camera_out)
        if camera.shape!=(len(frames),3) or camera.dtype!=np.float64 or not camera.flags.writeable:
            raise ValueError('writable float64 camera output buffer required')
        source=self.source_of_entry[entry];at=frames-self.first_frame
        if code>=4:
            ps,qs,values=self.direct[source,code]
            p=int(np.flatnonzero(ps==entry)[0]);q=int(np.flatnonzero(qs==exit)[0])
            camera[:]=values[p*len(qs)+q,at]
        else:
            camera[:]=self.baselines[source][code,at]
            if self.final_frame-self.first_frame>=38:
                ps,qs,left,right=self.patches[source,code]
                p=int(np.flatnonzero(ps==entry)[0]);q=int(np.flatnonzero(qs==exit)[0])
                early=at<19;late=at>=self.final_frame-self.first_frame-19
                camera[early]=left[p,at[early]]
                camera[late]=right[q,at[late]-(self.final_frame-self.first_frame-19)]
        return self.native[source][at],camera


class HoldRecorder:
    def __init__(self,arcs,i,j):
        self.arcs=arcs;self.i=i;self.j=j;self.first,self.final=map(int,arcs.nodes[[i,j]])
        self.direct={}

    def record_direct(self,source,code,ps,qs,values):
        # This is an ArcCosts-owned newly computed array, never a user input.
        # Its actual floating-point values are reused, not re-fitted/replayed.
        values.setflags(write=False)
        self.direct[source,code]=(frozen(ps,np.int64),frozen(qs,np.int64),values)

    def seal(self,allowed):
        from .boundary_arc_costs import cameras
        arcs=self.arcs;length=self.final-self.first;native={};baselines={};patches={};owned=0
        a=np.array([p.source for p in arcs.entries[self.i]])
        b=np.array([p.source for p in arcs.exits[self.j]])
        if (np.shape(allowed)!=(len(a),len(b),6) or np.asarray(allowed).dtype!=bool
                or np.any(allowed&(a[:,None,None]!=b[None,:,None]))):
            raise ValueError('captured mask must preserve the complete same-source camera domain')
        for source,cat in enumerate(arcs.cats):
            ps=np.array([k for k,p in enumerate(arcs.entries[self.i]) if p.source==source],int)
            qs=np.array([k for k,p in enumerate(arcs.exits[self.j]) if p.source==source],int)
            if not len(ps) or not len(qs) or not allowed[np.ix_(ps,qs,np.arange(6))].any():continue
            clock=cat.bank.get('programme_render_clock')
            if clock is None:raise ValueError('captured playback requires the actual renderer clock, not guessed native indices')
            a,b=clock['source_times'][[self.i,self.j]];rate=float((b-a)/(length/arcs.fps))
            native[source]=frozen(indices(a,rate,np.arange(length),fps=arcs.fps,native_fps=clock['fps']))
            if native[source].min()<0 or native[source].max()>=clock['count']:
                raise ValueError('captured native playback escapes its bound source')
            # The current arena is an immutable snapshot, not a borrowed view
            # into a later mutable baseline/cache. No per-history copies exist.
            baselines[source]=arcs.dense[source][:,self.first:self.final]
            if length>=38:
                entry=np.asarray([arcs.entries[self.i][k].strip for k in ps])
                leave=np.asarray([arcs.exits[self.j][k].strip for k in qs])
                for code in range(4):
                    first=arcs.dense[source][code][self.first:self.first+38]
                    last=arcs.dense[source][code][self.final-38:self.final]
                    left,_=cameras(entry,np.broadcast_to(first[-7:],entry.shape),first,cap=cat.bank['zoom_cap'])
                    right,_=cameras(np.broadcast_to(last[:7],leave.shape),leave,last,cap=cat.bank['zoom_cap'])
                    patches[source,code]=(frozen(ps),frozen(qs),frozen(left[:,:19]),frozen(right[:,-19:]))
            owned+=native[source].nbytes+sys.getsizeof(baselines[source])
        for group in (*patches.values(),*self.direct.values()):owned+=sum(v.nbytes for v in group)
        mask=frozen(allowed,bool);owned+=mask.nbytes
        return HeldInputs(self.i,self.j,self.first,self.final,mask,
            tuple(p.source for p in arcs.entries[self.i]),MappingProxyType(native),MappingProxyType(baselines),
            MappingProxyType(patches),MappingProxyType(dict(self.direct)),owned,arcs.captured_baseline_bytes)
