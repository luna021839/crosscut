"""Opt-in held-input compiler for the owned exact suffix executor.

Reuses frozen observation preparation, not its history cache. Every current
camera/NI/intent/embedding recipe is compiled before looking at prefix lanes.
Only a prefix bank's small embeddings enter temporal scoring. No real fit,
new source analysis, whole-runtime pass or professional quality is implied.
"""
from dataclasses import is_dataclass,fields as dataclass_fields
from collections.abc import Mapping
import hashlib
import json
import sys
import weakref
import numpy as np
from .context_port_dp import ArcRef
from .block_played_context import BlockPlayedContext
from .playback_input_batch import pack
from .playback_suffix_arena import PlaybackSuffixArena
from .playback_workspace import PlaybackWorkspace


def _owned_inputs(values,exclude=()):
    seen=set();buffers=[]
    def size(value):
        identity=id(value)
        if identity in seen:return 0
        if isinstance(exclude,np.ndarray):
            at=int(np.searchsorted(exclude,identity))
            if at<len(exclude) and int(exclude[at])==identity:return 0
        elif identity in exclude:return 0
        seen.add(identity);used=sys.getsizeof(value)
        if isinstance(value,np.ndarray):
            buffers.append(value)
            if isinstance(value.base,np.ndarray):used+=size(value.base)
        elif isinstance(value,Mapping):used+=sum(size(k)+size(v) for k,v in value.items())
        elif isinstance(value,(tuple,list)):used+=sum(size(v) for v in value)
        elif is_dataclass(value):used+=sum(size(getattr(value,f.name)) for f in dataclass_fields(value))
        elif hasattr(value,'__dict__'):used+=size(value.__dict__)
        return used
    return size(values),buffers


class _ProducerReservation:
    def __init__(self,compiler,i,j):
        self.compiler=compiler;self.i=i;self.j=j;self.key=None;self.finalizer=None
    def __enter__(self):
        c=self.compiler;old=c.context;i,j=self.i,self.j;L=int(old.nodes[j]-old.nodes[i]);counts=[]
        direct=0;direct_counts=[];maximum_heads=1
        for s,cat in enumerate(old.cats):
            P=sum(p.source==s for p in old.entries[i]);Q=sum(q.source==s for q in old.exits[j]);counts.append(P*Q)
            adjacent=old.arcs.opportunity_episodes is not None and old.arcs.opportunity_episodes[j]-old.arcs.opportunity_episodes[i]<=2
            whole=adjacent and L>=14 and not np.any((old.arcs.support.events[s]>old.nodes[i])&(old.arcs.support.events[s]<old.nodes[j]))
            direct+=P*Q*L*24*(int(L<38)+int(whole))
            direct_counts.append(P*Q if L<38 or whole else 0)
            maximum_heads=max(maximum_heads,cat.bank['heads'].shape[-2])
        P=len(old.entries[i]);Q=len(old.exits[j])
        # Long baseline families use shared interiors plus38-frame endpoint
        # geometry, NOT a full-length path batch. Only actual direct4/5
        # recipes reserve full-length content/geometry scratch.
        amount=(direct+max(direct_counts,default=0)*L*24*(64+16*maximum_heads)
            +max(counts,default=0)*38*24*8
            +(P+Q)*4*38*24+len(old.cats)*L*8+P*Q*6*64+64*2**10)
        self.key=c.arena.workspace.claim(int(amount),'hold-producer-preflight-and-owned-inputs')
        self.finalizer=weakref.finalize(self,c.arena.workspace.release,self.key)
        c.maximum_producer_reservation=max(c.maximum_producer_reservation,int(amount))
        return self
    def observe(self,base,allowed,benefit):
        c=self.compiler;arcs=c.context.arcs
        amount,buffers=_owned_inputs([base,allowed,benefit,arcs.last_camera_alternatives,
            arcs.last_codes,arcs.last_playback_inputs],exclude=c.source_buffer_ids)
        c.arena.workspace.pin_group(self.key,amount,buffers);self.finalizer.detach();self.key=None
    def __exit__(self,*unused):
        # On failure before observe(), keep the conservative claim until this
        # guard (and any traceback retaining producer temporaries) actually dies.
        return False


class _Origin:
    def __init__(self,compiler,i,root_ids,costs):
        w=compiler.arena.workspace;P,_,L=root_ids.shape;count=int(np.count_nonzero(root_ids));self.i=i
        self.lanes=w.empty((P,2,L),np.int64,owner='origin-prefix-lane-map');self.lanes[:]=-1
        self.handles=w.empty((count,),np.uint64,owner='origin-prefix-roots')
        at=0
        for p in range(P):
            for bit in (0,1):
                for k in range(L):
                    if root_ids[p,bit,k]:
                        self.lanes[p,bit,k]=at;self.handles[at]=root_ids[p,bit,k];at+=1
        self.compiler=compiler
        self.decoded=compiler.arena.decode_handles(self.handles,end=int(compiler.context.nodes[i]))
        self.head=w.empty((P,2,L,4),np.float64,owner='origin-exact-baseline-heads');self.head[:]=np.nan
        self.present=np.isfinite(costs)
        compiler.origin_decode_calls+=1
    def __enter__(self):self.compiler.active_origin=self;return self
    def __exit__(self,*unused):
        self.compiler.active_origin=None
        self.decoded.close();self.decoded=None;self.lanes=self.handles=self.head=self.present=None


class _Current:
    def __init__(self,hold,p,q,code):
        c=hold.compiler;old=c.context;self.hold=hold;self.ref=ArcRef(hold.i,hold.j,p,q,code)
        self.first,self.final=hold.first,hold.final;self.source=old.entries[hold.i][p].source
        if not hold.reference_clock[self.source]:c.nonreference_recipes+=1
        fresh=old.ends[(old.ends>self.first)&(old.ends<=self.final)]
        self.long=(code<4 and self.final-self.first>=old.history_frames+38
            and hold.reference_clock[self.source] and old._port_local(old.entries[hold.i][p],self.first,self.final,0)
            and old._port_local(old.exits[hold.j][q],self.first,self.final,1))
        self.middle=0.;self.tail=0.;self.lower_bound=0.;self.lease=None
        if self.long:
            self.head_ends=fresh[fresh<self.first+old.history_frames+19]
            middle=fresh[(fresh>=self.first+old.history_frames+19)&(fresh<=self.final-19)]
            self.tail_ends=fresh[fresh>self.final-19]
            at=np.searchsorted(old.ends,middle)
            self.middle=float((old.base_rates[self.source][code,at]*old.durations[at]).sum())
            # The sparse recipe still includes EVERY frame needed by head,
            # tail windows and exact future H suffix. Baseline middle uses
            # the already prepared actual reference-clock frame sequence.
            tail_first=max(self.first,int(self.tail_ends[0])-old.history_frames) if len(self.tail_ends) else self.final
            bank=c.active_origin
            need_head=not np.isfinite(bank.head[p,:,:,code][bank.present[p]]).all()
            ranges=[np.arange(self.first,int(self.head_ends[-1]))] if len(self.head_ends) and need_head else []
            ranges.append(np.arange(min(tail_first,max(self.first,self.final-old.history_frames)),self.final))
            selected=np.unique(np.concatenate(ranges))
        else:
            self.head_ends=fresh[fresh<self.first+old.history_frames]
            self.tail_ends=fresh[fresh>=self.first+old.history_frames]
            selected=np.arange(self.first,self.final)
        w=c.arena.workspace
        self.frames=w.empty((len(selected),),np.int64,owner='compiled-current-ticks');self.frames[:]=selected
        self.camera=w.empty((len(selected),3),np.float64,owner='compiled-current-camera')
        self.native,_=hold.inputs.sample(p,q,code,self.frames,camera_out=self.camera)
        self.keys=w.empty((len(selected),8),np.uint64,owner='compiled-current-raw-keys')
        self.encoded=w.empty((len(selected),old.model.embedding_width),np.float64,owner='compiled-current-embeddings')
        active=w.empty((len(selected),2),np.int64,owner='compiled-current-intent-handles');active[:]=0
        # At most two ACTIVE pre-camera intents. Canonical IDs were assigned
        # only after the original bind/same_input collision comparison.
        for side,port in enumerate((old.entries[hold.i][p],old.exits[hold.j][q])):
            if port.intent is None or not port.intent.pinned:continue
            intent=old.intents[id(port.intent)];key=old.intent_bindings[id(port.intent)]
            hit=(self.native>=intent.first_native)&(self.native<intent.final_native)
            active[hit,side]=c.intent_handles[key]
        active.sort(axis=1);active[active[:,0]==active[:,1],0]=0
        source_handles=w.empty((len(selected),),np.int64,owner='compiled-current-source-handles')
        source_handles[:]=c.source_handles[self.source]
        pack(model_binding=old.model_key,frames=self.frames,native=self.native,cameras=self.camera,
            source_observation_handles=source_handles,intent_handles=active,out=self.keys)
        c.recipe_compilations+=1;c.camera_sample_calls+=1
        # A fixed-size projection lease is reserved before any ragged tensor
        # is allocated. Chunking is storage/evaluation, never frame thinning.
        found=w.empty((min(120,len(self.frames)),),bool,owner='compile-frame-lookup-mask')
        for a in range(0,len(self.frames),120):
            b=min(len(self.frames),a+120);old.check()
            hit=found[:b-a];c.arena.probe(self.keys[a:b],self.encoded[a:b],hit)
            live_hits=int(hit.sum())
            # Source baseline reuse requires exact NI/camera bits and NO
            # active target intent, never coincidentally equal embeddings.
            neutral=(active[a:b]==0).all(-1)&(self.native[a:b]==old.reference[self.source][self.frames[a:b]])
            for family,baseline in enumerate(old.arcs.dense[self.source]):
                equal=neutral&~hit&(self.camera[a:b].view(np.uint64)==
                    np.ascontiguousarray(baseline[self.frames[a:b]],dtype=np.float64).view(np.uint64)).all(-1)
                if equal.any():
                    self.encoded[a:b][equal]=old.reference_encoded[self.source][family][self.frames[a:b][equal]]
                    hit[equal]=True;c.reference_frame_reuses+=int(equal.sum())
            c.live_frame_reuses+=live_hits
            missing=np.flatnonzero(~hit)
            if not len(missing):continue
            c.encode_missing(self,a+missing)
        if len(self.tail_ends):self.tail=self._own_cost(self.tail_ends)
        self.lower_bound=(self.middle+self.tail) if self.long else self.tail
        suffix=int(np.searchsorted(self.frames,max(self.first,self.final-old.history_frames)))
        # Own only a <=H frame tail; admit copies raw keys/embeddings, so this
        # small persistent state never pins the current hold's larger arrays.
        self.lease=c.arena.admit(self.keys[suffix:],self.encoded[suffix:],first=self.first)

    def _own_cost(self,ends):
        old=self.hold.compiler.context
        first=max(0,int(ends[0])-old.history_frames);final=int(ends[-1])
        keep=(self.frames>=first)&(self.frames<final)
        w=self.hold.compiler.arena.workspace
        claim=w.claim(int(keep.sum())*(old.model.embedding_width+1)*8+len(ends)*64+4096,'independent-temporal-window-scratch')
        try:
            rates=old.model.rates(self.frames[keep],self.encoded[keep][None],ends,kernel=old.kernel)[0]
            self.hold.compiler.window_evaluations+=len(ends)
            result=float((rates*old.durations[np.searchsorted(old.ends,ends)]).sum());del rates
            return result
        finally:w.release(claim)

    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
    def close(self):
        if self.lease is not None:self.lease.close();self.lease=None
        self.frames=self.camera=self.native=self.keys=self.encoded=None


class _Hold:
    def __init__(self,compiler,i,j,base,allowed):
        self.compiler=compiler;self.i=i;self.j=j;old=compiler.context
        self.first,self.final=map(int,old.nodes[[i,j]])
        self.inputs=old.arcs.last_playback_inputs
        if (self.inputs is None or (self.inputs.first_node,self.inputs.final_node)!=(i,j)
                or not np.array_equal(self.inputs.allowed,allowed)):
            raise ValueError('current frozen actual-camera/clock hold does not match the scored domain')
        self.reference_clock={s:np.array_equal(ni,old.reference[s][self.first:self.final]) for s,ni in self.inputs.native.items()}
        compiler.hold_compilations+=1
        compiler.native_schedule_compilations+=len(self.inputs.native)
    def current(self,p,q,code):return _Current(self,p,q,code)
    def __enter__(self):return self
    def __exit__(self,*unused):
        if self.compiler.context.arcs.last_playback_inputs is self.inputs:self.compiler.context.arcs.last_playback_inputs=None
        self.compiler.context.arcs.last_camera_alternatives=None
        self.compiler.context.arcs.last_codes=None
        self.inputs=None


class CompiledPlayback:
    """Explicit adapter, not the normal app's default or an AV-qualified model."""
    @staticmethod
    def namespace(context):
        registry=dict(sources=list(zip(context.source_bindings,context.field_versions)),
            intents=sorted(context.intent_fields))
        digest=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return ('single-source-three-camera-ordered-input-v1',*context.model_key,digest)

    def __init__(self,context,*,arena=None,maximum_bytes=200*2**20,label_limit=8):
        if not isinstance(context,BlockPlayedContext) or not context.arcs.capture_playback_inputs:
            raise ValueError('prepared ordered observations and opt-in captured actual playback producer required')
        if getattr(context.model,'pointwise_frame_encoder',False) is not True:
            raise ValueError('lossless frame tiling requires an explicit pointwise encoder contract')
        self.context=context;self.origin_decode_calls=self.hold_compilations=self.recipe_compilations=0
        self.camera_sample_calls=self.projection_batches=self.encoded_frames=self.window_evaluations=0
        self.reference_frame_reuses=self.live_frame_reuses=0;self.active_origin=None
        self.native_schedule_compilations=self.nonreference_recipes=self.maximum_producer_reservation=0
        self.projection_tile_splits=self.maximum_projection_claim=0
        w=PlaybackWorkspace(maximum_bytes) if arena is None else arena.workspace
        expected=self.namespace(context)
        if arena is not None and (arena.namespace!=expected or arena.width!=context.model.embedding_width
                or arena.H!=context.history_frames or arena.kernel!=context.kernel):
            raise ValueError('suffix namespace must bind this exact model, source/observation and intent registries')
        if any(len(rows)!=4 for rows in context.reference_encoded):
            raise ValueError('captured producer must retain its exact prepared baseline frame embeddings')
        owned=[context.fields,context.intent_fields,context.intents,context.reference,context.base_rates,context.reference_encoded,context.arcs.dense,
            context.model.parameters,context.source_bindings,context.model_key,context.ends,context.durations]
        amount,buffers=_owned_inputs(owned)
        w.pin_frozen_inputs(amount,buffers,owner='frozen-ordered-context-source-inputs')
        # A compact sorted numeric exclusion index, not53k Python ints/weak
        # callbacks copied into a new set on every hold accounting pass.
        self.source_buffer_ids=w.empty((len(buffers),),np.uint64,owner='frozen-input-allocation-index')
        for i,value in enumerate(buffers):self.source_buffer_ids[i]=id(value)
        self.source_buffer_ids.sort();self.source_buffer_ids.setflags(write=False)
        self.source_input_bytes=amount
        # These are recomputable caches, NOT owners of persistent suffixes.
        # Their preparation role is over; the new executor never calls them.
        context.pool.cache.clear();context.pool.bytes=0;context.cache.clear();context.cache_bytes=0
        context.head_tables.clear();context.suffix_versions.clear();context.suffix_hints.clear()
        if arena is None:
            W=max(map(len,context.entries+context.exits));N=len(context.nodes)
            roots=sum(map(len,context.exits))*2*label_limit+W*2*label_limit+4
            # Finite small domains can reserve their full live-label bound.
            # Larger domains still have an explicit fixed capacity/byte
            # refusal, not a silently expanded budget or reduced beam.
            frames=min(196608,(roots+1)*context.history_frames)
            cells=min((roots+1)*(context.history_frames+10)//6,frames//3+4*W*N)
            arena=PlaybackSuffixArena(namespace=expected,embedding_width=context.model.embedding_width,
                history_frames=context.history_frames,frame_capacity=frames,cell_capacity=cells,
                root_capacity=roots,workspace=w,kernel=context.kernel)
        self.arena=arena;self.label_limit=label_limit
        self.source_handles=tuple(v+1 for v in context.field_versions)
        self.intent_handles={key:i+1 for i,key in enumerate(sorted(context.intent_fields))}
        maximum_people=max([int(np.diff(f.ordered_static.person_splits).max(initial=0)) for f in
            (*context.fields,*context.intent_fields.values())]+[1])
        # All known scene/person projection and mask/set-encoder copies for
        # at most120 rows, including both overlay intents. This conservative
        # reservation is charged before projection; it is not a RSS reading.
        self.projection_reservation_bytes=120*(64+max(1,maximum_people)*81)*8*32

    def projection_requirement(self,ref,native):
        """Count actual ragged rows before allocation, including BOTH overlays.

        Original projection is computed even on claimed frames, so its
        temporaries are included too. No actor/evidence slot is truncated.
        """
        old=self.context;s=old.entries[ref.first][ref.entry].source
        def units(field,ordinals):
            data=field.ordered_static;T=len(ordinals);people=hits=0
            if len(data.frames):
                at=np.searchsorted(data.frames,ordinals);safe=np.minimum(at,len(data.frames)-1)
                hit=(at<len(data.frames))&(data.frames[safe]==ordinals);hits=int(hit.sum())
                people=int(np.diff(data.person_splits)[at[hit]].sum())
            padding=hits*(field.heads.shape[1]*4+field.body.shape[1]*2)
            return 64*T+81*people+4*padding
        count=units(old.fields[s],native)
        for port in (old.entries[ref.first][ref.entry],old.exits[ref.final][ref.exit]):
            if port.intent is None or not port.intent.pinned:continue
            intent=old.intents[id(port.intent)];key=old.intent_bindings[id(port.intent)]
            active=(native>=intent.first_native)&(native<intent.final_native)
            if active.any():count+=units(old.intent_fields[key],native[active])
        return int(count*8*32+64*2**10)

    def encode_missing(self,current,used):
        old=self.context;w=self.arena.workspace;pending=[used]
        while pending:
            used=pending.pop();old.check();required=self.projection_requirement(current.ref,current.native[used])
            if required>w.limit_bytes-w.current_bytes:
                if len(used)<2:
                    raise MemoryError('one complete observed frame exceeds available workspace; no people or cues removed')
                middle=len(used)//2;pending.extend((used[middle:],used[:middle]));self.projection_tile_splits+=1
                continue
            claim=w.claim(required,'actual-ragged-projection-encoder-temporaries')
            try:
                evidence=old._intent_projection(current.ref,current.frames[used],current.native[used],current.camera[used],ordered=True)
                values=old.model.encode(evidence)
                current.encoded[used]=values;self.projection_batches+=1;self.encoded_frames+=len(used)
                self.maximum_projection_claim=max(self.maximum_projection_claim,required)
                del evidence,values
            finally:w.release(claim)

    def open_origin(self,i,roots,costs):return _Origin(self,i,roots,costs)
    def reserve_hold(self,i,j):return _ProducerReservation(self,i,j)
    def compile_hold(self,i,j,base,allowed):return _Hold(self,i,j,base,allowed)

    def score(self,bank,current,p,bit,slots):
        old=self.context;ends=current.head_ends
        if not len(ends):return np.full(len(slots),current.lower_bound)
        code=current.ref.camera
        cached=bank.head[p,bit,slots,code] if current.long else np.full(len(slots),np.nan)
        missing=np.flatnonzero(np.isnan(cached));head=cached.copy()
        if len(missing):
            w=self.arena.workspace
            claim=w.claim(64*2**10+len(missing)*old.history_frames*old.model.embedding_width*8,
                'ordered-head-temporal-kernel-scratch')
            try:self._score_head(bank,current,p,bit,slots,missing,head)
            finally:w.release(claim)
        return (head+current.middle)+current.tail if current.long else head+current.tail

    def _score_head(self,bank,current,p,bit,slots,missing,head):
        old=self.context;ends=current.head_ends;code=current.ref.camera
        lanes=bank.lanes[p,bit,np.asarray(slots)[missing]]
        if np.any(lanes<0):raise ValueError('context scoring requested an unretained prefix lane')
        first=max(0,int(ends[0])-old.history_frames);final=int(ends[-1]);w=self.arena.workspace
        frames=w.empty((final-first,),np.int64,owner='temporal-head-ticks');frames[:]=np.arange(first,final)
        batch=w.empty((len(missing),len(frames),old.model.embedding_width),np.float64,owner='temporal-head-embeddings')
        batch[:]=0.
        past=bank.decoded.frames;keep=(past>=first)&(past<current.first)
        for at,lane in enumerate(lanes):batch[at,past[keep]-first]=bank.decoded.values[lane,keep]
        keep=(current.frames>=current.first)&(current.frames<final)
        batch[:,current.frames[keep]-first]=current.encoded[keep]
        rates=old.model.rates(frames,batch,ends,kernel=old.kernel)
        durations=old.durations[np.searchsorted(old.ends,ends)]
        head[missing]=(rates*durations).sum(-1);self.window_evaluations+=len(ends)*len(missing)
        if current.long:bank.head[p,bit,np.asarray(slots)[missing],code]=head[missing]

    def receipt(self):
        self.context.model.verify()
        return dict(contract='pre_history_compiled_actual_hold_ordered_context_v1',
            origin_prefix_banks_decoded=self.origin_decode_calls,holds_compiled=self.hold_compilations,
            current_recipes_compiled=self.recipe_compilations,held_camera_samples=self.camera_sample_calls,
            projection_batches=self.projection_batches,frames_encoded=self.encoded_frames,
            exact_reference_frame_reuses=self.reference_frame_reuses,exact_live_frame_reuses=self.live_frame_reuses,
            native_schedule_compilations=self.native_schedule_compilations,nonreference_recipes=self.nonreference_recipes,
            maximum_hold_producer_preallocation_reservation=self.maximum_producer_reservation,
            window_evaluations=self.window_evaluations,source_input_bytes=self.source_input_bytes,
            projection_reservation_bytes=self.projection_reservation_bytes,
            projection_reservation_policy='actual_ragged_rows_with_lossless_pointwise_frame_tiling',
            projection_tile_splits=self.projection_tile_splits,maximum_actual_projection_claim=self.maximum_projection_claim,
            history_camera_or_native_reconstruction_calls=0,source_NN_calls=0,new_source_rasters=0,
            per_frame_physical_input_preserved=True,workspace=self.arena.workspace.receipt(),
            owned_suffix_state=self.arena.receipt(),normal_app_activation_approved=False,
            whole_runtime_or_media_quality_approved=False)
