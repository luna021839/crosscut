"""Exact playback-block consumer of an explicitly fitted ordered head.

Camera/intent observations are projected and frame-encoded once per retained
physical input, not per history. Labels/codes/base costs remain independent.
The strict small temporal kernel keeps chronological arithmetic rather than
reassociating near-tie history/current dot products. No fit or film approval.
"""
from collections import OrderedDict
import hashlib
import json
import re
import sys
from time import monotonic
import numpy as np
from .context_port_dp import ArcRef
from .played_context import PlayedContext,_SourceField,OBSERVATION_CONTRACT
from .playback_blocks import BlockPool,bytes_used,flatten
from .ordered_context import SCHEMA_SHA256
from .playback_intent import exact_array_input


class BlockPlayedContext(PlayedContext):
    def __init__(self,model,*,nodes,entries,exits,catalogues,arcs,native_ports,fps,source_bindings,check,kernel='numpy'):
        if (not isinstance(source_bindings,(list,tuple)) or len(source_bindings)!=len(catalogues)
                or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in source_bindings)):
            raise ValueError('actual content-bound physical sources required for ordered sequence context')
        if kernel not in ('numpy','numba') or float(fps)!=model.clock_fps:
            raise ValueError('explicit ordered kernel and exact fitted output clock required')
        self.model=model;self.nodes=np.asarray(nodes,int);self.entries=entries;self.exits=exits
        self.cats=catalogues;self.arcs=arcs;self.fps=float(fps);self.check=check;self.final=int(nodes[-1])
        self.history_frames=model.context_frames;self.interval_frames=6;self.kernel=kernel
        self.source_bindings=tuple(source_bindings);self.model_key=(SCHEMA_SHA256,model.bound_content_sha256)
        self.ends=np.r_[np.arange(6,self.final,6),self.final];self.durations=np.diff(np.r_[0,self.ends])/fps
        self.pool=BlockPool();self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=8*2**20
        self.clock_cache=OrderedDict();self.fast=self.fallback=self.window_calls=self.cache_hits=self.identical_input_hits=0
        self.head_tables={};self.head_table_bytes=0;self.suffix_end=None;self.suffix_versions={};self.suffix_hints={}
        self._prepare_intents(native_ports,source_bindings,ordered=True)
        self.fields=[];self.reference=[];self.base_rates=[];self.reference_encoded=[];self.field_versions=[];versions={}
        kernel_started=monotonic()
        check();model.rates(np.zeros(0,np.int64),np.zeros((1,0,model.embedding_width)),np.zeros(0,np.int64),kernel=kernel)
        from .context_long_relax import relax
        relax(np.zeros((0,0,6)),np.zeros((0,0,4),bool),np.zeros((0,2,8)),np.zeros((0,2,8),np.int64),
            np.zeros((0,2,8,4)),np.zeros((0,4)),np.zeros((0,4)),np.zeros((0,0)),
            same_episode=True,origin=0,width=1,kernel=kernel)
        self.kernel_preparation_seconds=monotonic()-kernel_started
        for cat,binding,native in zip(catalogues,source_bindings,native_ports,strict=True):
            check();clock=cat.bank.get('programme_render_clock')
            if clock is None or len(clock['source_times'])!=len(nodes) or clock['fps']<=0 or clock['count']<1:
                raise ValueError('exact programme renderer clock required before ordered context')
            field=_SourceField(cat,native,binding,check,ordered=True);self.fields.append(field)
            data=field.ordered_static;h=hashlib.sha256()
            arrays=('frames','values','observed','conflicted','person_splits','people','people_observed','people_conflicted')
            for name in arrays:h.update(getattr(data,name).tobytes())
            h.update(json.dumps(data.scopes,separators=(',',':')).encode());digest=h.digest()
            bucket=versions.setdefault((binding,digest),[]);version=None
            for old,number in bucket:
                if old.scopes==data.scopes and all(exact_array_input(getattr(old,n),getattr(data,n)) for n in
                        arrays):version=number;break
            if version is None:version=len(self.field_versions);bucket.append((data,version))
            self.field_versions.append(version)
            times=np.interp(np.arange(self.final),self.nodes,np.asarray(clock['source_times']))
            self.reference.append((((times*1000.)/1000.)*clock['fps']+.5).astype(np.int64))
        for s,field in enumerate(self.fields):
            rates=[];encoded_rows=[];frames=np.arange(self.final);native=self.reference[s]
            for camera in arcs.dense[s]:
                check();blocks=self._compile(s,frames,native,np.asarray(camera),((),)*len(frames),
                    lambda t,n,c,field=field:field.project_ordered(t,n,c))
                ticks,encoded,_=flatten(blocks)
                rates.append(model.rates(ticks,encoded[None],self.ends,kernel=kernel)[0])
                if arcs.capture_playback_inputs:
                    encoded.setflags(write=False);encoded_rows.append(encoded)
            self.base_rates.append(np.asarray(rates))
            self.reference_encoded.append(tuple(encoded_rows))
        self.source_table_bytes=(sum(f.bytes for f in (*self.fields,*self.intent_fields.values()))
            +sum(v.nbytes for v in self.reference+self.base_rates)
            +sum(v.nbytes for rows in self.reference_encoded for v in rows)
            +sum(v.nbytes for v in model.parameters.values()))
        # Explicit fixed-label head table plus one destination's exact suffix
        # identities and bounded projection/relaxation temporaries.
        self.head_table_reservation=len(nodes)*max(map(len,entries))*2*8*4*9+8*2**20
        self.workspace_reserved_bytes=(self.source_table_bytes+self.pool.maximum_bytes+self.cache_limit
            +self.head_table_reservation+24*2**20)
        if self.workspace_reserved_bytes>=180*2**20:
            raise MemoryError('ordered source tables leave no declared label workspace')

    def _compile(self,s,frames,native,cameras,intents,project):
        # Bound scene/ragged-person temporary tensors independently of film duration.
        # Batches align to global120-frame cells; the interner retains every
        # per-frame clock/camera/intent inside its six-frame storage blocks.
        if not len(frames):return ()
        starts=np.r_[0,np.flatnonzero(np.diff(frames//120)!=0)+1];stops=np.r_[starts[1:],len(frames)]
        result=[]
        for a,b in zip(starts,stops):
            self.check();result.extend(self.pool.compile(source=self.source_bindings[s],
                observation_version=self.field_versions[s],model_schema=self.model_key,
                frames=frames[a:b],native=native[a:b],cameras=cameras[a:b],intents=intents[a:b],
                project=project,encode=self.model.encode))
        return tuple(result)

    def _blocks(self,ref,first,final):
        first=max(first,int(self.nodes[ref.first]));final=min(final,int(self.nodes[ref.final]))
        frames=np.arange(first,final);s=self.entries[ref.first][ref.entry].source
        native=self._native(ref,frames)
        camera=self.arcs.sample_camera(ref.first,ref.final,ref.entry,ref.exit,ref.camera,frames)
        active=[[] for _ in frames]
        for port in (self.entries[ref.first][ref.entry],self.exits[ref.final][ref.exit]):
            if port.intent is None or not port.intent.pinned:continue
            intent=self.intents[id(port.intent)];key=self.intent_bindings[id(port.intent)]
            for k in np.flatnonzero((native>=intent.first_native)&(native<intent.final_native)):active[int(k)].append(key)
        intents=tuple(tuple(sorted(set(v))) for v in active)
        return self._compile(s,frames,native,camera,intents,
            lambda t,n,c:self._intent_projection(ref,t,n,c,ordered=True))

    def _memo(self,key,make):
        if key in self.cache:
            self.cache_hits+=1;value,size=self.cache.pop(key);self.cache[key]=(value,size);return value
        value=make();size=bytes_used(key)+bytes_used(value)+64
        while self.cache and self.cache_bytes+size>self.cache_limit:
            _,(_,old)=self.cache.popitem(last=False);self.cache_bytes-=old
        if size<=self.cache_limit:self.cache[key]=(value,size);self.cache_bytes+=size
        return value

    def _encoded_history(self,history,first,final):
        def build():
            blocks=tuple(b for ref in history if self.nodes[ref.final]>first and self.nodes[ref.first]<final
                for b in self._blocks(ref,first,final))
            ticks,value,key=flatten(blocks)
            if not len(ticks):value=np.zeros((0,self.model.embedding_width))
            return ticks,value,key
        return self._memo(('encoded-history',history,first,final),build)

    def _batch_head_cost(self,histories,ref,ends):
        if not len(ends):return np.zeros(len(histories))
        first=max(0,int(ends[0])-self.history_frames);final=int(ends[-1]);begin=int(self.nodes[ref.first])
        current,encoded,current_key=flatten(self._blocks(ref,first,final))
        result=np.empty(len(histories));missing=[];past=[];keys=[]
        for b,history in enumerate(histories):
            ticks,values,physical=self._encoded_history(history,first,begin)
            key=('ordered-window-cost',first,final,ends.tobytes(),physical,current_key)
            keys.append(key);past.append((ticks,values))
            if key in self.cache:
                result[b]=self._memo(key,lambda:None);self.identical_input_hits+=1
            else:missing.append(b)
        if not missing:return result
        frames=np.arange(first,final);batch=np.zeros((len(missing),len(frames),self.model.embedding_width))
        for at,b in enumerate(missing):
            ticks,values=past[b];batch[at,ticks-first]=values;batch[at,current-first]=encoded
        # Only small embeddings are repeated across labels. Native/target
        # lookup and the large masked frame encoder were already shared.
        rates=self.model.rates(frames,batch,ends,kernel=self.kernel)
        durations=self.durations[np.searchsorted(self.ends,ends)]
        self.window_calls+=len(ends)*len(missing)
        for b,value in zip(missing,(rates*durations).sum(-1)):result[b]=self._memo(keys[b],lambda v=float(value):v)
        return result

    def _cost(self,refs,ends):
        if not len(ends):return 0.
        first=max(0,int(ends[0])-self.history_frames);final=int(ends[-1])
        blocks=tuple(b for ref in refs if self.nodes[ref.final]>first and self.nodes[ref.first]<final
            for b in self._blocks(ref,first,final))
        frames,encoded,key=flatten(blocks)
        def evaluate():
            self.window_calls+=len(ends)
            rates=self.model.rates(frames,encoded[None],ends,kernel=self.kernel)[0]
            return float((rates*self.durations[np.searchsorted(self.ends,ends)]).sum())
        return self._memo(('independent-ordered-cost',ends.tobytes(),key),evaluate)

    def _local_intents(self,ref):
        s=self.entries[ref.first][ref.entry].source;first,final=map(int,self.nodes[[ref.first,ref.final]])
        clock=self.reference[s]
        for side,port in enumerate((self.entries[ref.first][ref.entry],self.exits[ref.final][ref.exit])):
            if port.intent is None or not port.intent.pinned:continue
            intent=self.intents[id(port.intent)]
            a,b=np.searchsorted(clock,[intent.first_native,intent.final_native])
            a=max(first,int(a));b=min(final,int(b))
            if a<b and ((side==0 and b>first+19) or (side==1 and a<final-19)):return False
        return True

    def _port_local(self,port,first,final,side):
        if port.intent is None or not port.intent.pinned:return True
        intent=self.intents[id(port.intent)];clock=self.reference[port.source]
        a,b=np.searchsorted(clock,[intent.first_native,intent.final_native])
        a=max(first,int(a));b=min(final,int(b))
        return a>=b or (b<=first+19 if side==0 else a>=final-19)

    def long_candidates(self,i,j,base,allowed,prefix,in_cost,in_cuts,benefit,*,same_episode,width):
        """All baselines × every retained prefix, with exact suffix reduction.

        Once a reference-clock hold exceeds H+38, its entry patch/intent is
        outside the future suffix. The tail depends only on actual source,
        end port and family. This is physical equality, not cost dominance
        across different recent films or a smaller declared beam/domain.
        """
        first,final=map(int,self.nodes[[i,j]])
        if final-first<self.history_frames+38:return None
        P,Q=allowed.shape[:2];L=in_cost.shape[-1]
        if L!=8:return None  # Other declared label domains use the reference path, never a reduced domain.
        handled=allowed[:,:,:4].copy();by_source={}
        for s in range(len(self.cats)):
            ps=[p for p,v in enumerate(self.entries[i]) if v.source==s]
            qs=[q for q,v in enumerate(self.exits[j]) if v.source==s]
            exact=bool(ps and qs) and self._reference_clock(ArcRef(i,j,ps[0],qs[0],0))
            by_source[s]=exact
        for p,port in enumerate(self.entries[i]):
            handled[p]&=by_source[port.source] and self._port_local(port,first,final,0)
        for q,port in enumerate(self.exits[j]):handled[:,q]&=self._port_local(port,first,final,1)
        if not handled.any():return None
        signature=tuple((p,b,tuple(slots.tolist()),tuple(histories)) for (p,b),(slots,histories) in prefix.items())
        if i not in self.head_tables:
            head=np.full((P,2,L,4),np.nan);self.head_tables[i]=(head,signature)
            # Histories/ArcRefs are already owned and charged by the DP.
            # Charge only the new table and tuple-of-pointers structure here.
            self.head_table_bytes+=head.nbytes+sys.getsizeof(signature)+sum(sys.getsizeof(row)
                +sys.getsizeof(row[2])+sys.getsizeof(row[3]) for row in signature)
            if self.head_table_bytes>self.head_table_reservation:
                raise MemoryError('explicit ordered head-table reservation exceeded')
        head,old_signature=self.head_tables[i]
        if signature!=old_signature:raise ValueError('finalized incoming labels changed under shared ordered heads')
        head_ends=self.ends[(self.ends>first)&(self.ends<first+self.history_frames+19)]
        for p,code in zip(*np.nonzero(handled.any(axis=1))):
            q=int(np.flatnonzero(handled[p,:,code])[0]);ref=ArcRef(i,j,int(p),q,int(code))
            for bit in (0,1):
                slots,histories=prefix[int(p),bit];missing=np.isnan(head[p,bit,slots,code])
                if missing.any():
                    values=self._batch_head_cost([h for h,active in zip(histories,missing) if active],ref,head_ends)
                    head[p,bit,slots[missing],code]=values
        middle=np.zeros((P,4));tail=np.zeros((Q,4))
        mid_ends=self.ends[(self.ends>=first+self.history_frames+19)&(self.ends<=final-19)]
        tail_ends=self.ends[(self.ends>final-19)&(self.ends<=final)]
        at=np.searchsorted(self.ends,mid_ends)
        for s in range(len(self.cats)):
            ps=[p for p,v in enumerate(self.entries[i]) if v.source==s]
            if ps:
                for code in range(4):middle[ps,code]=(self.base_rates[s][code,at]*self.durations[at]).sum()
        for q,code in zip(*np.nonzero(handled.any(axis=0))):
            p=int(np.flatnonzero(handled[:,q,code])[0]);ref=ArcRef(i,j,p,int(q),int(code))
            s=self.entries[i][p].source
            tail[q,code]=self._memo(('ordered-tail',s,j,int(q),int(code)),lambda ref=ref:self._cost((ref,),tail_ends))
        from .context_long_relax import relax
        self.check();values=relax(base,handled,in_cost,in_cuts,head,middle,tail,benefit,
            same_episode=same_episode,origin=i,width=width,kernel=self.kernel)
        return handled,values

    def _long_parts(self,ref):
        first,final=map(int,self.nodes[[ref.first,ref.final]])
        if ref.camera>=4 or final-first<self.history_frames+38 or not self._reference_clock(ref) or not self._local_intents(ref):return None
        fresh=self.ends[(self.ends>first)&(self.ends<=final)]
        head=fresh[fresh<first+self.history_frames+19];tail=fresh[fresh>final-19]
        middle=fresh[(fresh>=first+self.history_frames+19)&(fresh<=final-19)]
        s=self.entries[ref.first][ref.entry].source
        at=np.searchsorted(self.ends,middle)
        # Sum the original per-window rates, not subtract two long prefix
        # sums. Keep the same temporal-kernel arithmetic at near ties.
        mid=float((self.base_rates[s][ref.camera,at]*self.durations[at]).sum())
        suffix=self._memo(('ordered-tail',s,ref.final,ref.exit,ref.camera),lambda:self._cost((ref,),tail))
        return head,mid,suffix

    def equivalent_key(self,history,end):
        if self.suffix_end!=end:
            self.suffix_end=end;self.suffix_versions={};self.suffix_hints={}
        ref=history[-1] if history else None;hint=None
        if (ref is not None and ref.camera<4 and self.nodes[ref.final]-self.nodes[ref.first]>=self.history_frames+38
                and self._reference_clock(ref) and self._local_intents(ref)):
            hint=(self.entries[ref.first][ref.entry].source,ref.final,ref.exit,ref.camera)
            if hint in self.suffix_hints:return self.suffix_hints[hint]
            physical=flatten(self._blocks(ref,max(0,end-self.history_frames),end))[2]
        else:physical=self._encoded_history(history,max(0,end-self.history_frames),end)[2]
        if physical not in self.suffix_versions:self.suffix_versions[physical]=len(self.suffix_versions)
        number=self.suffix_versions[physical]
        if hint is not None:self.suffix_hints[hint]=number
        return number

    def replay(self,refs):
        # Independent exact clock/camera projection and frame encoding; do
        # not consult physical-block IDs, embeddings or cached scalar costs.
        frames=[];encoded=[]
        for ref in refs:
            first,final=map(int,self.nodes[[ref.first,ref.final]])
            for a in range(first,final,120):
                t=np.arange(a,min(final,a+120));native=self._native(ref,t)
                camera=self.arcs.sample_camera(ref.first,ref.final,ref.entry,ref.exit,ref.camera,t)
                evidence=self._intent_projection(ref,t,native,camera,ordered=True)
                frames.append(t);encoded.append(self.model.encode(evidence))
        ticks=np.concatenate(frames);values=np.concatenate(encoded)
        rates=self.model.rates(ticks,values[None],self.ends,kernel='numpy')[0];costs=rates*self.durations
        parts=[float(costs[(self.ends>self.nodes[r.first])&(self.ends<=self.nodes[r.final])].sum()) for r in refs]
        return parts,float(costs.sum())

    def receipt(self):
        self.model.verify()
        return dict(observation_contract=OBSERVATION_CONTRACT,ordered_schema_sha256=SCHEMA_SHA256,
            context_kernel=self.kernel,context_frames=self.history_frames,storage=self.pool.receipt(),
            source_NN_calls=0,new_source_rasters=0,pre_camera_targets=True,
            selected_pre_camera_intents_consumed=True,finite_pinned_intents=len(self.intent_fields),
            ordered_native_and_displayed_body_geometry=True,background_flow_measured=False,
            source_table_bytes=self.source_table_bytes,cache_bytes=self.cache_bytes,
            workspace_reserved_bytes=self.workspace_reserved_bytes,kernel_preparation_seconds=self.kernel_preparation_seconds,
            exact_shared_head_table_bytes=self.head_table_bytes,head_table_reservation=self.head_table_reservation,
            exact_long_suffix_batched_relaxation=True,
            observed_window_evaluations=self.window_calls,cache_hits=self.cache_hits,
            identical_observed_head_input_cache_hits=self.identical_input_hits,
            long_reference_clock_factorizations=self.fast,direct_context_evaluations=self.fallback,
            temporal_operation_order='chronological_scalar_no_history_current_reassociation',
            global_singer_or_dense_intent_plan_completed=False,stage_edit_transformer_used=False,
            normal_speed_AV_performed=False,professional_quality_approved=False)
