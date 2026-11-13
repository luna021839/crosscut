"""Actual-source/native-clock context consumed by the camera-port selector.

Freeze native image targets BEFORE proposed cameras. Only exact observed
native ordinals may supply faces/poses; a repeated displayed native image is
real repeated exposure, not interpolation over a gap. No source pixels/NN,
fitting, music timestamps or source numbers enter the preference features.
Full-frame/background motion remains UNKNOWN until actually observed.
"""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import re
import sys
from time import monotonic
import numpy as np
from .scene_targets import TargetScene,from_original_view,FEATURE,FORMATION,UNKNOWN
from .physical_display_observations import focal_metadata
from .sequence_context import Evidence,PRIMITIVES,project,merge,windows,batched_window_features
from .programme_source_clock import indices

OBSERVATION_CONTRACT='intent_bound_exact_native_target_context_v2'


def _native_scene(scene,k):
    """Original-image role, not the role of an anchor seven frames away.

    Float32 canonicalization is the existing observation storage precision,
    not a new aesthetic/identity tolerance. Bodies transfer only through an
    exact co-observed head geometry at this SAME source/native image.
    """
    native=int(scene.native_indices[k]);raw=scene.original_heads[k].copy()
    raw[~scene.original_observed[k]]=0.
    _,at,_,kind=focal_metadata(raw[None,None],np.array([[[1.,.5,.5]]]))
    found=np.flatnonzero((raw[:,2]>.019)&(raw[:,3]>=.75));role=UNKNOWN
    if int(kind[0,0])==1:found=np.array([int(at[0,0])]);role=FEATURE
    elif int(kind[0,0])==2:role=FORMATION
    if len(found):
        # A native formation is an unordered observed set, not six mandatory
        # members or a new face chosen by the eventual zoom/crop.
        found=np.asarray(sorted(found,key=lambda h:tuple(raw[h,:3].astype(np.float32))))
        heads=raw[found].astype(np.float32)[None];count=len(found);known=np.ones((1,count),bool)
        body=np.zeros((1,count,8,3),np.float32);evidence=np.zeros((1,count,8),np.uint8)
        ambiguity=np.zeros((1,count,8),bool);quality=np.full((1,count),.5);qknown=np.zeros((1,count),bool)
        for h,head in enumerate(heads[0]):
            same=scene.head_observed[k]&(scene.heads[k,:,:3].astype(np.float32)==head[:3]).all(-1)
            matched=np.flatnonzero(same)
            if len(matched)!=1:continue
            j=int(matched[0]);body[0,h]=scene.body[k,j];evidence[0,h]=scene.body_evidence[k,j]
            ambiguity[0,h]=scene.body_ambiguity[k,j];quality[0,h]=scene.roi_readability[k,j]
            qknown[0,h]=scene.roi_measured[k,j]
        identity=heads[0,:,:3].tobytes()
    else:
        # Partial arm/body observations can survive a missing face. Do not
        # lend them to a different visible actor or fabricate a head raster.
        heads=scene.heads[k:k+1].astype(np.float32);known=np.zeros(heads.shape[:2],bool)
        body=scene.body[k:k+1];evidence=scene.body_evidence[k:k+1];ambiguity=scene.body_ambiguity[k:k+1]
        quality=np.full(known.shape,.5);qknown=np.zeros(known.shape,bool)
        identity=scene.track.encode() if evidence.any() else b'no_face_no_body_observation'
    key=hashlib.sha256(str(role).encode()+identity).hexdigest()
    return TargetScene(scene.binding,'native-'+str(native)+'-'+key,np.array([native]),np.array([native]),role,
        heads,known,body,evidence,ambiguity,quality,qknown,raw[None],scene.original_observed[k:k+1])


def _dense_scenes(cat,binding):
    if cat.subjects is None:return
    seen=set()
    for items in cat.subjects.by_native.values():
        for window,_ in items:
            if id(window) in seen:continue
            seen.add(id(window));n=len(window.ordinals)
            original=np.zeros((n,max(1+len(v) for v in window.competitors),4))
            for i,others in enumerate(window.competitors):
                original[i,0]=window.heads[i];original[i,1:1+len(others)]=others
            known=((original[...,2]>.019)&(original[...,3]>=.75))
            head_known=window.associated[:,None]
            kinds=np.where(window.support[:,None],1,0).astype(np.uint8)
            yield TargetScene(binding,window.track,window.ordinals,window.ordinals,FEATURE,
                window.heads[:,None],head_known,window.body[:,None],kinds,np.zeros(kinds.shape,bool),
                np.full(head_known.shape,.5),np.zeros(head_known.shape,bool),original,known)


def _native_cut_scenes(native,binding):
    observed=np.asarray(native.get('observed',np.ones(len(native['frames']),bool)))
    ordinals=native.get('native_indices');keys=native.get('native_event_keys')
    for k in range(len(native['frames'])):
        if not observed[k]:continue
        if ordinals is not None:pair=np.asarray(ordinals[k],int)
        elif keys is not None and len(keys[k])==1:pair=np.array([int(keys[k][0])-1,int(keys[k][0])])
        else:continue  # No guessed output/native FPS equivalence.
        for side,ordinal in enumerate(pair):
            if ordinal<0:continue
            raw=np.asarray(native['heads'][k,side]);valid=(raw[:,2]>.019)&(raw[:,3]>=.75);count=len(raw)
            yield TargetScene(binding,'actual-native-cut-'+str(ordinal),np.array([ordinal]),np.array([ordinal]),UNKNOWN,
                raw[None],valid[None],np.zeros((1,count,8,3)),np.zeros((1,count,8),np.uint8),np.zeros((1,count,8),bool),
                np.full((1,count),.5),np.zeros((1,count),bool),raw[None],valid[None])


class _SourceField:
    """Shared raw native observations and exact camera projection tables."""
    def __init__(self,cat,native,binding,check,*,intent=None,ordered=False):
        originals=([from_original_view(cat,n,side,source_binding=binding)
            for n in range(len(cat.nodes)) for side in (0,1)] if intent is None else [intent.scene])
        native_scenes=list(_native_cut_scenes(native,binding)) if intent is None else []
        if intent is None:originals.extend(_dense_scenes(cat,binding))
        originals.extend(native_scenes)
        self.native_cut_endpoint_rows=len(native_scenes)
        observed=np.asarray(native.get('observed',np.ones(len(native['frames']),bool)))
        keys=native.get('native_event_keys');ordinals=native.get('native_indices')
        self.native_cut_events_missing_ordinal=sum(bool(known) and ordinals is None and
            (keys is None or len(keys[k])!=1) for k,known in enumerate(observed))
        groups={};parts=[];ordered_parts=[];self.normalized_rows=0
        if ordered:
            from .ordered_context import project as ordered_project,merge as ordered_merge
        for scene in originals:
            check()
            for k in range(len(scene.frames)):
                if intent is None:actual=_native_scene(scene,k)
                else:
                    ordinal=int(scene.native_indices[k])
                    if not intent.first_native<=ordinal<intent.final_native:continue
                    actual=TargetScene(scene.binding,scene.track,np.array([ordinal]),np.array([ordinal]),scene.role,
                        **{name:getattr(scene,name)[k:k+1] for name in ('heads','head_observed','body',
                            'body_evidence','body_ambiguity','roi_readability','roi_measured','original_heads','original_observed')})
                self.normalized_rows+=1
                evidence=project(actual,np.array([[1.,.5,.5]]));parts.append(evidence)
                if ordered:ordered_parts.append(ordered_project(actual,np.array([[1.,.5,.5]])))
                groups.setdefault(int(actual.native_indices[0]),[]).append(actual)
        self.static=merge(parts);T=len(self.static.frames)
        self.ordered_static=ordered_merge(ordered_parts) if ordered else None
        head_rows=[];body_rows=[];head_valid=np.zeros(T,bool);body_valid=np.zeros(T,bool)
        for i,ordinal in enumerate(self.static.frames):
            faces=[];points=[]
            for scene in groups[int(ordinal)]:
                if scene.head_observed.any():
                    face=scene.heads[0,scene.head_observed[0]].copy()
                    face[:,3]=1.  # geometry identity; validity was already observed
                    faces.append(face)
                projected=scene.project(np.array([[1.,.5,.5]]));known=projected['body_observed'][0]
                if known.any():
                    p=scene.body[0,:,:,:2][known]
                    p=p[np.lexsort((p[:,1],p[:,0]))];points.append(p)
            same=lambda values:bool(values) and all(np.array_equal(values[0],v) for v in values[1:])
            head_valid[i]=same(faces);body_valid[i]=same(points)
            head_rows.append(faces[0] if head_valid[i] else np.zeros((0,4)))
            body_rows.append(points[0] if body_valid[i] else np.zeros((0,2)))
        H=max([len(v) for v in head_rows]+[1]);B=max([len(v) for v in body_rows]+[1])
        self.heads=np.zeros((T,H,4),np.float32);self.heads_known=np.zeros((T,H),bool)
        self.body=np.zeros((T,B,2),np.float32);self.body_known=np.zeros((T,B),bool)
        for i,(heads,body) in enumerate(zip(head_rows,body_rows)):
            self.heads[i,:len(heads)]=heads;self.heads_known[i,:len(heads)]=True
            self.body[i,:len(body)]=body;self.body_known[i,:len(body)]=True
        self.head_consistent=head_valid;self.body_consistent=body_valid
        self.bytes=sum(v.nbytes for v in (self.static.frames,self.static.values,self.static.observed,self.static.conflicted,
            self.heads,self.heads_known,self.body,self.body_known,head_valid,body_valid))
        if ordered:self.bytes+=sum(getattr(self.ordered_static,n).nbytes for n in ('frames','values',
            'observed','conflicted','person_splits','people','people_observed','people_conflicted'))
        for v in (self.heads,self.heads_known,self.body,self.body_known,head_valid,body_valid):v.setflags(write=False)

    def project(self,frames,native,cameras):
        T=len(frames);values=np.zeros((T,len(PRIMITIVES)));known=np.zeros(values.shape,bool)
        conflicts=np.zeros(values.shape,bool);scopes=[None]*T
        at=np.searchsorted(self.static.frames,native);safe=np.minimum(at,max(0,len(self.static.frames)-1))
        hit=(at<len(self.static.frames))&(self.static.frames[safe]==native)
        rows=np.flatnonzero(hit);where=at[hit]
        values[rows]=self.static.values[where];known[rows]=self.static.observed[where];conflicts[rows]=self.static.conflicted[where]
        for row,k in zip(rows,where):scopes[int(row)]=self.static.target_scopes[int(k)]
        for label in ('feature','formation'):
            for axis,coordinate in (('x',1),('y',2)):
                k=PRIMITIVES.index(label+'_center_'+axis)
                values[:,k]=.5+cameras[:,0]*(values[:,k]-cameras[:,coordinate])
            values[:,PRIMITIVES.index(label+'_height')]*=cameras[:,0]
        if len(rows):
            head=self.heads[where];z=cameras[rows,0]
            center=.5+z[:,None,None]*(head[...,:2]-cameras[rows,None,1:])
            half=head[...,2,None]*z[:,None,None]*[.45*9/16,.5]
            extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
            fraction=np.prod(extent/np.maximum(1e-12,2*half),axis=-1)
            mask=self.heads_known[where];count=mask.sum(-1)
            for name,raw in (('target_visible_fraction',fraction),('known_head_crop_fraction',fraction<1.-1e-9)):
                k=PRIMITIVES.index(name);values[rows,k]=np.where(mask,raw,0.).sum(-1)/np.maximum(1,count)
                known[rows,k]&=self.head_consistent[where]&(count>0)
            body=.5+z[:,None,None]*(self.body[where]-cameras[rows,None,1:])
            mask=self.body_known[where];count=mask.sum(-1);k=PRIMITIVES.index('known_body_crop_fraction')
            outside=(body<0).any(-1)|(body>1).any(-1)
            values[rows,k]=(outside&mask).sum(-1)/np.maximum(1,count)
            known[rows,k]&=self.body_consistent[where]&(count>0)
        for name,value in (('additional_log_zoom',np.log(cameras[:,0])),
                ('additional_center_x',cameras[:,1]-.5),('additional_center_y',cameras[:,2]-.5)):
            k=PRIMITIVES.index(name);values[:,k]=value;known[:,k]=True;conflicts[:,k]=False
        known&=~conflicts
        return Evidence(np.asarray(frames),values,known,tuple(scopes),conflicts)

    def project_ordered(self,frames,native,cameras):
        from .ordered_context import lookup,recamera
        if self.ordered_static is None:raise ValueError('ordered observations were not prepared')
        evidence=lookup(self.ordered_static,frames,native)
        return recamera(evidence,cameras,self.project(frames,native,cameras))


class PlayedContext:
    def __init__(self,model,*,nodes,entries,exits,catalogues,arcs,native_ports,fps,source_bindings,check,kernel='numpy'):
        if (not isinstance(source_bindings,(list,tuple)) or len(source_bindings)!=len(catalogues)
                or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in source_bindings)):
            raise ValueError('actual content-bound physical sources required for sequence context')
        self.model=model;self.nodes=np.asarray(nodes,int);self.entries=entries;self.exits=exits
        self.cats=catalogues;self.arcs=arcs;self.fps=float(fps);self.check=check;self.final=int(nodes[-1])
        self.history_frames=int(np.ceil(model.context_seconds*fps));self.interval_frames=model.interval_frames
        if kernel not in ('numpy','numba'):raise ValueError('explicit numpy reference or strict numba context kernel required')
        self.kernel=kernel;self._window_features=batched_window_features;kernel_started=monotonic()
        if kernel=='numba':
            from .sequence_context_kernel import features as compiled_features
            self._window_features=compiled_features
            for readonly in (False,True):
                frames=np.zeros(0,np.int64);raw=np.zeros((1,0,len(PRIMITIVES)));known=np.zeros(raw.shape,bool)
                if readonly:
                    for value in (frames,raw,known):value.setflags(write=False)
                check();self._window_features(frames,raw,known,np.zeros(0,np.int64),fps=fps,context_frames=self.history_frames)
        self.kernel_preparation_seconds=monotonic()-kernel_started
        self._prepare_intents(native_ports,source_bindings)
        self.fields=[];self.reference=[];self.base_prefix=[];self.ends=np.r_[np.arange(6,self.final,6),self.final]
        self.durations=np.diff(np.r_[0,self.ends])/fps
        self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=32*2**20
        self.clock_cache=OrderedDict();self.fast=0;self.fallback=0;self.window_calls=0;self.cache_hits=0
        self.identical_input_hits=0
        for s,(cat,binding,native) in enumerate(zip(catalogues,source_bindings,native_ports,strict=True)):
            check();clock=cat.bank.get('programme_render_clock')
            if (clock is None or len(clock['source_times'])!=len(nodes) or clock['fps']<=0 or clock['count']<1):
                raise ValueError('exact programme renderer clock required before sequence context')
            field=_SourceField(cat,native,binding,check);self.fields.append(field)
            times=np.interp(np.arange(self.final),self.nodes,np.asarray(clock['source_times']))
            reference=(((times*1000.)/1000.)*clock['fps']+.5).astype(np.int64);self.reference.append(reference)
            baseline=[]
            for camera in arcs.dense[s]:
                evidence=field.project(np.arange(self.final),reference,np.asarray(camera))
                report=windows(evidence,final_frame=self.final,fps=fps)
                baseline.append(np.r_[0.,np.cumsum(model.costs(report['features'])*report['interval_seconds'])])
            self.base_prefix.append(np.asarray(baseline))
        self.source_table_bytes=sum(f.bytes for f in (*self.fields,*self.intent_fields.values()))+sum(v.nbytes for v in self.reference+self.base_prefix)
        self.workspace_reserved_bytes=self.source_table_bytes+self.cache_limit+16*2**20
        if self.workspace_reserved_bytes>=180*2**20:raise MemoryError('source context leaves no declared label workspace')

    def _prepare_intents(self,native_ports,source_bindings,*,ordered=False):
        self.intents={};self.intent_fields={};self.intent_bindings={};buckets={}
        for ports in (self.entries,self.exits):
            for row in ports:
                for port in row:
                    raw=port.intent
                    if raw is None or not raw.pinned:continue
                    if id(raw) not in self.intents:
                        bound,digest=raw.bind(source_bindings[port.source]);bucket=buckets.setdefault(digest,[])
                        match=next((k for k,(old,_) in enumerate(bucket) if bound.same_input(old)),None)
                        if match is None:
                            key=digest+':'+str(len(bucket));bucket.append((bound,key))
                        else:key=bucket[match][1]
                        self.intents[id(raw)]=bound;self.intent_bindings[id(raw)]=key
                        if key not in self.intent_fields:
                            self.intent_fields[key]=_SourceField(self.cats[port.source],native_ports[port.source],
                                source_bindings[port.source],self.check,intent=bound,ordered=ordered)

    def _native(self,ref,frames):
        s=self.entries[ref.first][ref.entry].source;clock=self.cats[s].bank['programme_render_clock']
        first,final=map(int,self.nodes[[ref.first,ref.final]]);a,b=clock['source_times'][[ref.first,ref.final]]
        rate=float((b-a)/((final-first)/self.fps))
        return indices(a,rate,np.asarray(frames)-first,fps=self.fps,native_fps=clock['fps'])

    def _part(self,ref,first,final):
        a=max(first,int(self.nodes[ref.first]));b=min(final,int(self.nodes[ref.final]))
        frames=np.arange(a,b);s=self.entries[ref.first][ref.entry].source
        camera=self.arcs.sample_camera(ref.first,ref.final,ref.entry,ref.exit,ref.camera,frames)
        native=self._native(ref,frames)
        return self._project_intents(ref,frames,native,camera)

    def _project_intents(self,ref,frames,native,cameras):
        return self._intent_projection(ref,frames,native,cameras,ordered=False)

    def _intent_projection(self,ref,frames,native,cameras,*,ordered):
        if ordered:
            from .ordered_context import OrderedEvidence as Result,merge as combine
            scope_name='scopes';project_name='project_ordered'
        else:Result=Evidence;combine=merge;scope_name='target_scopes';project_name='project'
        s=self.entries[ref.first][ref.entry].source
        original=getattr(self.fields[s],project_name)(frames,native,cameras)
        overlays=[];claimed=np.zeros(len(frames),bool)
        for port in (self.entries[ref.first][ref.entry],self.exits[ref.final][ref.exit]):
            if port.intent is None or not port.intent.pinned:continue
            intent=self.intents[id(port.intent)];key=self.intent_bindings[id(port.intent)]
            active=(native>=intent.first_native)&(native<intent.final_native)
            if not active.any():continue
            value=getattr(self.intent_fields[key],project_name)(frames[active],native[active],cameras[active])
            # The target plan applies throughout its finite associated window,
            # but absent face/body rasters do NOT become observations. Keep a
            # known intended role separately from target visibility.
            raw=value.values.copy();known=value.observed.copy()
            for name,kind in (('role_feature',FEATURE),('role_formation',FORMATION),('role_unknown',UNKNOWN)):
                k=PRIMITIVES.index(name);raw[:,k]=float(intent.scene.role==kind);known[:,k]=True
            value=(replace(value,values=raw,observed=known,scopes=(key,)*len(value.frames)) if ordered else
                Result(value.frames,raw,known,(key,)*len(value.frames),value.conflicted))
            overlays.append(value);claimed|=active
        if not overlays:return original
        keep=~claimed
        base=(original.take(np.flatnonzero(keep)) if ordered else Result(original.frames[keep],original.values[keep],
            original.observed[keep],tuple(x for x,k in zip(getattr(original,scope_name),keep) if k),original.conflicted[keep]))
        return combine([base,*overlays])

    def _memo(self,key,make):
        if key in self.cache:
            self.cache_hits+=1;value,size=self.cache.pop(key);self.cache[key]=(value,size);return value
        value=make();size=sys.getsizeof(key)+sys.getsizeof(value)+64+sum(sys.getsizeof(v) for v in key)
        if isinstance(value,Evidence):
            size+=sum(v.nbytes for v in (value.frames,value.values,value.observed,value.conflicted))
            size+=sys.getsizeof(value.target_scopes)+sum(sys.getsizeof(s) for s in value.target_scopes)
        while self.cache and self.cache_bytes+size>self.cache_limit:
            _,(_,old)=self.cache.popitem(last=False);self.cache_bytes-=old
        if size<=self.cache_limit:self.cache[key]=(value,size);self.cache_bytes+=size
        return value

    def _history(self,history,first,end):
        return self._memo(('history',history,first,end),lambda:merge([
            self._part(r,first,end) for r in history if self.nodes[r.final]>first and self.nodes[r.first]<end]))

    def _batch_head_cost(self,histories,ref,ends):
        if not len(ends):return np.zeros(len(histories))
        first=max(0,int(ends[0])-self.history_frames);final=int(ends[-1]);begin=int(self.nodes[ref.first])
        current=self._part(ref,first,final)
        h=hashlib.sha256(ends.tobytes())
        for a in (current.frames,np.where(current.observed,current.values,0.),current.observed):h.update(a.tobytes())
        signature=h.digest();keys=[('evaluated-head',history,first,final,begin,signature) for history in histories]
        result=np.empty(len(histories));missing=[]
        for b,key in enumerate(keys):
            if key in self.cache:
                result[b]=self._memo(key,lambda:None);self.identical_input_hits+=1
            else:missing.append(b)
        if not missing:return result
        # Reuse only exactly identical observed scoring inputs. Camera codes,
        # physical prefix refs and their future-equivalence states are NOT
        # merged/dropped by this bounded scalar-cost cache.
        frames=np.arange(first,final);B=len(missing)
        values=np.zeros((B,len(frames),len(PRIMITIVES)));known=np.zeros(values.shape,bool)
        at=current.frames-first
        values[:,at]=current.values;known[:,at]=current.observed
        for b,original in enumerate(missing):
            history=histories[original]
            past=self._history(history,first,begin);at=past.frames-first
            values[b,at]=past.values;known[b,at]=past.observed
        features=self._window_features(frames,values,known,ends,fps=self.fps,context_frames=self.history_frames)
        durations=self.durations[np.searchsorted(self.ends,ends)]
        self.window_calls+=len(ends)*B
        scores=self.model.costs(features)@durations
        for b,score in zip(missing,scores):result[b]=self._memo(keys[b],lambda v=float(score):v)
        return result

    def _cost(self,refs,ends):
        if not len(ends):return 0.
        first=max(0,int(ends[0])-self.history_frames);final=int(ends[-1])
        parts=[self._part(r,first,final) for r in refs if self.nodes[r.final]>first and self.nodes[r.first]<final]
        evidence=merge(parts);self.window_calls+=len(ends)
        if self.kernel=='numba':
            if not np.array_equal(ends,self.ends[(self.ends>=ends[0])&(self.ends<=final)]):
                raise AssertionError('context windows moved with a cut')
            features=self._window_features(evidence.frames,evidence.values[None],evidence.observed[None],ends,
                fps=self.fps,context_frames=self.history_frames)[0]
            return float(self.model.costs(features)@self.durations[np.searchsorted(self.ends,ends)])
        report=windows(evidence,final_frame=self.final,fps=self.fps,score_after_frame=int(ends[0])-1,
            score_until_frame=final)
        if not np.array_equal(report['interval_final'],ends):raise AssertionError('context windows moved with a cut')
        return float(self.model.costs(report['features'])@report['interval_seconds'])

    def _reference_clock(self,ref):
        s=self.entries[ref.first][ref.entry].source;key=(s,ref.first,ref.final)
        if key in self.clock_cache:return self.clock_cache[key]
        first,final=map(int,self.nodes[[ref.first,ref.final]]);frames=np.arange(first,final)
        equal=np.array_equal(self._native(ref,frames),self.reference[s][first:final])
        self.clock_cache[key]=equal
        if len(self.clock_cache)>4096:self.clock_cache.popitem(last=False)
        return equal

    def _long_parts(self,ref):
        first,final=map(int,self.nodes[[ref.first,ref.final]])
        if ref.camera<4 and final-first>=self.history_frames+38 and self._reference_clock(ref):
            fresh=self.ends[(self.ends>first)&(self.ends<=final)]
            head=fresh[fresh<first+self.history_frames+19]
            tail=fresh[fresh>final-19];middle=fresh[(fresh>=first+self.history_frames+19)&(fresh<=final-19)]
            s=self.entries[ref.first][ref.entry].source;mid=0.
            if len(middle):
                a,b=np.searchsorted(self.ends,[middle[0],middle[-1]])
                mid=float(self.base_prefix[s][ref.camera,b+1]-self.base_prefix[s][ref.camera,a])
            tail_key=('tail',s,ref.final,ref.exit,ref.camera)
            suffix=self._memo(tail_key,lambda:self._cost((ref,),tail))
            return head,mid,suffix
        return None

    def lower_bound(self,ref):
        # Softplus rates are nonnegative. An exact history-independent middle
        # and tail therefore bound this arc for EVERY incoming history, not
        # just a cheaply pooled prefix. No new editorial criterion is used.
        parts=self._long_parts(ref)
        return 0. if parts is None else float(parts[1]+parts[2])

    def incremental(self,histories,ref):
        self.check();first,final=map(int,self.nodes[[ref.first,ref.final]])
        fresh=self.ends[(self.ends>first)&(self.ends<=final)];parts=self._long_parts(ref)
        if parts is not None:
            self.fast+=1;head,mid,suffix=parts
            return self._batch_head_cost(histories,ref,head)+mid+suffix
        self.fallback+=1
        # Every camera code becomes independent of its incoming history after
        # four seconds. Score that suffix once, not once per retained label.
        head=fresh[fresh<first+self.history_frames];later=fresh[fresh>=first+self.history_frames]
        tail=self._memo(('independent',ref),lambda:self._cost((ref,),later)) if len(later) else 0.
        return self._batch_head_cost(histories,ref,head)+tail

    def equivalent_key(self,history,end):
        def make():
            evidence=self._history(history,max(0,end-self.history_frames),end);h=hashlib.sha256()
            for a in (evidence.frames,evidence.values,evidence.observed,evidence.conflicted):h.update(a.tobytes())
            h.update(json.dumps(evidence.target_scopes,separators=(',',':')).encode());return h.digest()
        return self._memo(('physical-key',history,end),make)

    def replay(self,refs):
        # Independent full-stream calculation, not the long-arc prefix/cache
        # decomposition used by incremental search. Actual scored cameras and
        # exact source ordinals are regenerated from the chosen frozen codes.
        evidence=merge([self._part(r,int(self.nodes[r.first]),int(self.nodes[r.final])) for r in refs])
        report=windows(evidence,final_frame=self.final,fps=self.fps)
        costs=self.model.costs(report['features'])*report['interval_seconds']
        per_hold=[float(costs[(self.ends>self.nodes[r.first])&(self.ends<=self.nodes[r.final])].sum()) for r in refs]
        return per_hold,float(costs.sum())

    def receipt(self):
        self.model.verify()
        execution=None
        if self.kernel=='numba':
            from .sequence_context_kernel import receipt
            execution=receipt()
        return dict(observation_contract=OBSERVATION_CONTRACT,source_NN_calls=0,new_source_rasters=0,
            exact_native_observation_lookup_no_interpolation=True,pre_camera_targets=True,
            head_and_body_not_borrowed_between_people=True,background_flow_measured=False,
            selected_pre_camera_intents_consumed=True,finite_pinned_intents=len(self.intent_fields),
            global_singer_or_dense_intent_plan_completed=False,
            original_native_cut_observations_consumed=any(f.native_cut_endpoint_rows for f in self.fields),
            native_cut_endpoint_rows_per_source=[f.native_cut_endpoint_rows for f in self.fields],
            observed_native_cut_events_missing_ordinal_per_source=[f.native_cut_events_missing_ordinal for f in self.fields],
            source_table_bytes=self.source_table_bytes,
            cache_bytes=self.cache_bytes,workspace_reserved_bytes=self.workspace_reserved_bytes,
            long_reference_clock_factorizations=self.fast,direct_context_evaluations=self.fallback,
            observed_window_evaluations=self.window_calls,cache_hits=self.cache_hits,
            identical_observed_head_input_cache_hits=self.identical_input_hits,
            context_kernel=self.kernel,kernel_preparation_seconds=self.kernel_preparation_seconds,
            compiled_kernel_execution=execution,
            normal_speed_AV_performed=False,professional_quality_approved=False)
