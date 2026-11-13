"""Finite-history semi-Markov search over exact boundary camera alternatives.

Unlike the rejected cost/debt frontier, a cheaper prefix does not dominate a
different recent movie. Only future-equivalent suffixes at the SAME port and
paid bit may merge. Other pruning is an explicit bounded beam approximation.
Labels contain backpointers and shared thin arc references, never pixel or
per-frame camera arrays. The caller owns source/role/clock evidence and a
fixed-time contextual scorer, not a new per-cut reward.
"""
from dataclasses import dataclass
import sys
import numpy as np
from .boundary_port_dp import _ports, _vector, _mask, _matrix


@dataclass(frozen=True, slots=True)
class ArcRef:
    first: int
    final: int
    entry: int
    exit: int
    camera: int


@dataclass(slots=True)
class _Candidate:
    cost: float
    cuts: int
    back: int
    code: int
    credit: float
    context: float
    history: tuple

    def rank(self):
        return self.cost, self.cuts, self.back, self.code, -self.credit


def solve(nodes, entries, exits, *, start_cost, hold_cost, cut_cost,
          context_cost, episodes, credit, history_frames, label_limit=8,
          history_key=None, context_lower_bound=None, long_context_batch=None, minimum_frames=12, check=lambda: None,
          workspace_limit_bytes=200*2**20, start_allowed=None,
          final_cost=None, final_allowed=None):
    """Score all retained camera codes jointly with the previous context.

    hold_cost(i,j) -> (finite cost, bool allowed), [entry,exit,camera].
    context_cost(histories, ArcRef) -> one finite incremental cost per prefix.
    That callback must charge only newly completed GLOBAL fixed-time windows,
    including original camera changes, and use at most history_frames of past.
    history_key(suffix,end_frame), if supplied, must encode future-equivalent
    physical camera/target/role/clock input, NOT a pooled feature or low cost.
    With no key callback, only identical thin suffix arc references merge.
    context_lower_bound(ref), if supplied, must be a proved lower bound on
    EVERY prefix's extra cost. It only skips candidates unable to enter the
    already-declared beam, never applies scalar dominance across histories.
    long_context_batch, if supplied, may merge prefix candidates ONLY where
    the caller proves their future physical suffixes equal. It must preserve
    every declared camera cell, paid-bit transition and original rank rule.
    """
    nodes=np.asarray(nodes);episodes=np.asarray(episodes)
    if (nodes.ndim!=1 or len(nodes)<2 or nodes.dtype.kind not in 'iu' or nodes[0]!=0
            or np.any(np.diff(nodes)<=0) or episodes.shape!=nodes.shape or episodes.dtype.kind not in 'iu'
            or np.any(episodes<0) or np.any(np.diff(episodes)<0)
            or type(history_frames) is not int or history_frames<1
            or type(label_limit) is not int or not 1<=label_limit<=32
            or type(minimum_frames) is not int or minimum_frames<1
            or type(workspace_limit_bytes) is not int or workspace_limit_bytes<1
            or not callable(context_cost) or not callable(credit)
            or (context_lower_bound is not None and not callable(context_lower_bound))
            or (long_context_batch is not None and not callable(long_context_batch))
            or (history_key is not None and not callable(history_key))):
        raise ValueError('ordered physical nodes/episodes and explicit bounded context required')
    N=len(nodes);entries=_ports(entries,N,'entry');exits=_ports(exits,N,'exit')
    width=max(map(len,entries+exits));L=label_limit
    start=_vector(start_cost,len(entries[0]),'start_cost')
    begin=_mask(start_allowed,len(entries[0]),'start_allowed')
    final=np.zeros(len(exits[-1])) if final_cost is None else _vector(final_cost,len(exits[-1]),'final_cost')
    finish=_mask(final_allowed,len(exits[-1]),'final_allowed')
    # cost, cuts, backpointer, camera, paid credit, context delta, history id.
    def empty(p):
        shape=(p,2,L)
        return [np.full(shape,np.inf),np.zeros(shape,np.int64),np.full(shape,-1,np.int64),
            np.full(shape,-1,np.int64),np.zeros(shape),np.zeros(shape),np.full(shape,-1,np.int64)]
    incoming=[empty(len(p)) for p in entries];outgoing=[empty(len(p)) for p in exits]
    arrays=sum(a.nbytes for states in (incoming,outgoing) for state in states for a in state)
    histories=[()];history_ids={():0};arc_objects=set();object_bytes=0
    peak=arrays;discarded=equivalent=hold_calls=cut_calls=batches=evaluations=camera_count=rank_rejected=bound_rejected=0
    worst_cache={};long_batched_holds=long_merged_prefixes=0

    def reserve(transient=0):
        nonlocal peak
        managed=(arrays+object_bytes+sys.getsizeof(histories)+sys.getsizeof(history_ids)
            +sys.getsizeof(arc_objects)+transient)
        if managed>workspace_limit_bytes:
            raise MemoryError('explicit context labels/history workspace exceeded; no hidden domain reduction')
        peak=max(peak,managed)

    def intern(history):
        nonlocal object_bytes
        found=history_ids.get(history)
        if found is not None:return found
        found=len(histories);histories.append(history);history_ids[history]=found
        object_bytes+=sys.getsizeof(history)+sys.getsizeof(found)
        for ref in history:
            identity=id(ref)
            if identity not in arc_objects:
                arc_objects.add(identity)
                object_bytes+=sys.getsizeof(identity)+sys.getsizeof(ref)+sum(sys.getsizeof(v) for v in
                    (ref.first,ref.final,ref.entry,ref.exit,ref.camera))
        reserve();return found

    def worst(pool):
        if len(pool)<L:return None
        key=id(pool)
        if key not in worst_cache:
            k=max(pool,key=lambda k:pool[k].rank());worst_cache[key]=(k,pool[k].rank())
        return worst_cache[key]

    def retain(pool,value,end):
        nonlocal discarded,equivalent,rank_rejected
        bound=worst(pool)
        if bound is not None and value.rank()>=bound[1]:
            # This cannot replace ANY retained label, including an exact
            # future-equivalent one. Skip expensive physical suffix hashing;
            # this is the SAME declared finite beam, not cost dominance over
            # different histories or a reduction of the camera domain.
            discarded+=1;rank_rejected+=1;return
        key=value.history if history_key is None else history_key(value.history,end)
        old=pool.get(key)
        if old is not None:
            equivalent+=1
            if value.rank()<old.rank():pool[key]=value;worst_cache.pop(id(pool),None)
            return
        if len(pool)<L:pool[key]=value;return
        discarded+=1
        if value.rank()<bound[1]:del pool[bound[0]];pool[key]=value;worst_cache.pop(id(pool),None)

    def save(pools,state):
        # Pool construction ends before any child can point at these slots.
        # Shared histories are interned ONLY for retained, finalized labels.
        for p,by_bit in enumerate(pools):
            for bit,pool in enumerate(by_bit):
                for slot,v in enumerate(sorted(pool.values(),key=lambda x:x.rank())):
                    values=(v.cost,v.cuts,v.back,v.code,v.credit,v.context,intern(v.history))
                    for field,value in zip(state,values):field[p,bit,slot]=value

    def pools(p):return [[{} for _ in range(2)] for _ in range(p)]
    def suffix(history,ref,end):
        return tuple(r for r in (*history,ref) if int(nodes[r.final])>end-history_frames)
    reserve()
    incoming[0][0][:,0,0]=np.where(begin,start,np.inf)
    incoming[0][6][:,0,0]=np.where(begin,0,-1)
    for j in range(1,N):
        check();worst_cache.clear();out_pool=pools(len(exits[j]))
        for i in range(j):
            check()
            if nodes[j]-nodes[i]<minimum_frames and j!=N-1:continue
            if not np.isfinite(incoming[i][0]).any():continue
            base,allowed=hold_cost(i,j);base=np.asarray(base);allowed=np.asarray(allowed)
            shape=(len(entries[i]),len(exits[j]))
            if (base.ndim!=3 or base.shape[:2]!=shape or base.dtype.kind not in 'iuf'
                    or allowed.shape!=base.shape or allowed.dtype!=bool or not np.isfinite(base).all()):
                raise ValueError('all finite per-camera hold alternatives and a boolean mask required')
            allowed=allowed&(entries[i][:,None,None]==exits[j][None,:,None])
            benefit=np.asarray(credit(i,j),float)
            if benefit.shape==(shape[0],):benefit=np.broadcast_to(benefit[:,None],shape)
            if benefit.shape!=shape or not np.isfinite(benefit).all() or np.any(benefit<0):
                raise ValueError('actual finite nonnegative once-only opportunity credit required')
            reserve(base.nbytes+allowed.nbytes+benefit.nbytes+len(exits[j])*2*L*(256+16*history_frames//minimum_frames))
            hold_calls+=1;camera_count+=int(allowed.sum());prefix={}
            for p in range(shape[0]):
                for bit in (0,1):
                    slots=np.flatnonzero(np.isfinite(incoming[i][0][p,bit]))
                    prefix[p,bit]=(slots,[histories[int(incoming[i][6][p,bit,k])] for k in slots])
            handled=np.zeros(allowed.shape,bool)
            if long_context_batch is not None:
                batch=long_context_batch(i,j,base,allowed,prefix,incoming[i][0],incoming[i][1],benefit,
                    same_episode=episodes[i]==episodes[j],width=width)
                if batch is not None:
                    subset,values=batch
                    if subset.shape!=allowed.shape[:2]+(4,) or subset.dtype!=bool or np.any(subset&~allowed[:,:,:4]):
                        raise ValueError('long context batch changed the declared camera domain')
                    handled[:,:,:4]=subset;long_batched_holds+=1
                    costs,counts,backs,bits,slots,claims,extras=values
                    expected=allowed.shape[:2]+(4,2)
                    if any(v.shape!=expected for v in values):raise ValueError('complete long context candidate matrices required')
                    present=np.isfinite(costs)
                    if (np.any(present&~subset[:,:,:,None]) or not np.isfinite(claims[present]).all()
                            or not np.isfinite(extras[present]).all() or np.any(claims[present]<0)
                            or np.any((bits[present]<0)|(bits[present]>1))
                            or np.any((slots[present]<0)|(slots[present]>=L))):
                        raise ValueError('valid unchanged long prefix/paid-bit transitions required')
                    for p,q,code,outbit in zip(*np.nonzero(np.isfinite(costs))):
                        p,q,code,outbit=map(int,(p,q,code,outbit));bit=int(bits[p,q,code,outbit]);k=int(slots[p,q,code,outbit])
                        history=histories[int(incoming[i][6][p,bit,k])];ref=ArcRef(i,j,p,q,code)
                        h=suffix(history,ref,int(nodes[j]));paid=float(claims[p,q,code,outbit])
                        retain(out_pool[q][outbit],_Candidate(float(costs[p,q,code,outbit]),int(counts[p,q,code,outbit]),
                            int(backs[p,q,code,outbit]),code,paid,float(extras[p,q,code,outbit]),h),int(nodes[j]))
                    long_merged_prefixes+=int(subset.sum())
            for p,q,code in zip(*np.nonzero(allowed)):
                p,q,code=int(p),int(q),int(code);ref=ArcRef(i,j,p,q,code)
                if handled[p,q,code]:continue
                lower=None if context_lower_bound is None else float(context_lower_bound(ref))
                if lower is not None and not np.isfinite(lower):raise ValueError('finite explicit context lower bound required')
                for bit in (0,1):
                    slots,past=prefix[p,bit]
                    if not len(slots):continue
                    if lower is not None:
                        retained=[]
                        for at,k in enumerate(slots):
                            cost=float(incoming[i][0][p,bit,k]+base[p,q,code]+lower)
                            cuts=int(incoming[i][1][p,bit,k]);back=(((i*width+p)*2+bit)*L+int(k))
                            viable=False
                            for claim in (False,True) if bit==0 and benefit[p,q]>0 else (False,):
                                paid=float(benefit[p,q]) if claim else 0.
                                outbit=(1 if bit or claim else 0) if episodes[i]==episodes[j] else 0
                                limit=worst(out_pool[q][outbit])
                                viable|=limit is None or (cost-paid,cuts,back,code,-paid)<limit[1]
                            if viable:retained.append(at)
                        bound_rejected+=len(slots)-len(retained)
                        slots=slots[retained];past=[past[k] for k in retained]
                        if not len(slots):continue
                    extra=np.asarray(context_cost(past,ref),float)
                    if extra.shape!=(len(slots),) or not np.isfinite(extra).all():
                        raise ValueError('one finite incremental fixed-time context cost per prefix required')
                    if lower is not None and np.any(extra<lower):raise ValueError('context cost violated its explicit lower bound')
                    batches+=1;evaluations+=len(slots)
                    for k,history,sequence in zip(slots,past,extra):
                        k=int(k);h=suffix(history,ref,int(nodes[j]))
                        cost=float(incoming[i][0][p,bit,k]+base[p,q,code]+sequence)
                        cuts=int(incoming[i][1][p,bit,k]);back=(((i*width+p)*2+bit)*L+k)
                        for claim in (False,True) if bit==0 and benefit[p,q]>0 else (False,):
                            paid=float(benefit[p,q]) if claim else 0.
                            outbit=(1 if bit or claim else 0) if episodes[i]==episodes[j] else 0
                            retain(out_pool[q][outbit],_Candidate(cost-paid,cuts,back,code,paid,float(sequence),h),int(nodes[j]))
        save(out_pool,outgoing[j])
        if j==N-1 or not np.isfinite(outgoing[j][0]).any():continue
        check();cut,allowed=_matrix(cut_cost(j),(len(exits[j]),len(entries[j])),'cut')
        allowed=allowed&(exits[j][:,None]!=entries[j][None,:]);cut_calls+=1;in_pool=pools(len(entries[j]))
        for q,p in zip(*np.nonzero(allowed)):
            for bit in (0,1):
                for k in np.flatnonzero(np.isfinite(outgoing[j][0][q,bit])):
                    history=histories[int(outgoing[j][6][q,bit,k])]
                    value=_Candidate(float(outgoing[j][0][q,bit,k]+cut[q,p]),
                        int(outgoing[j][1][q,bit,k])+1,int(q)*L+int(k),-1,0.,0.,history)
                    retain(in_pool[p][bit],value,int(nodes[j]))
        reserve(cut.nbytes+allowed.nbytes+len(entries[j])*2*L*(256+16*history_frames//minimum_frames))
        save(in_pool,incoming[j])
    total=np.where(finish[:,None,None],outgoing[-1][0]+final[:,None,None],np.inf)
    if not np.isfinite(total).any():raise ValueError('no complete contextual port programme')
    order=np.lexsort((np.arange(total.size),outgoing[-1][1].ravel(),total.ravel()))
    at=int(order[0]);q,bit,k=map(int,np.unravel_index(at,total.shape))
    objective=float(total.ravel()[at]);cuts=int(outgoing[-1][1][q,bit,k]);j=N-1
    runs=[];codes=[];claims=[];context=[]
    while True:
        encoded=int(outgoing[j][2][q,bit,k]);oldslot=encoded%L;encoded//=L
        oldbit=encoded%2;encoded//=2;i,p=divmod(encoded,width)
        if i<0 or i>=j:raise AssertionError('context backtrace is not acyclic')
        runs.append((i,j,p,q));codes.append(int(outgoing[j][3][q,bit,k]))
        paid=float(outgoing[j][4][q,bit,k]);claims.append(dict(episode=int(episodes[i]),credit=paid,claimed=paid>0))
        context.append(float(outgoing[j][5][q,bit,k]))
        if i==0:break
        previous=int(incoming[i][2][p,oldbit,oldslot]);q,k=divmod(previous,L);j=i;bit=oldbit
    claims=list(reversed(claims));paid=[v['episode'] for v in claims if v['claimed']]
    if len(paid)!=len(set(paid)):raise AssertionError('context search paid one physical episode twice')
    receipt=dict(contract='bounded_fixed_time_context_boundary_port_DP_v1',nodes=N,
        hold_calls=hold_calls,cut_calls=cut_calls,camera_alternatives_scored=camera_count,
        context_callback_batches=batches,context_prefix_evaluations=evaluations,
        label_limit_per_port_paid_bit=L,labels_discarded_by_bound_events=discarded,
        equivalent_histories_merged=equivalent,shared_suffix_histories=len(histories),history_frames=history_frames,
        expensive_history_keys_skipped_by_existing_beam_rank=rank_rejected,
        prefix_context_evaluations_skipped_by_explicit_lower_bound=bound_rejected,
        explicit_context_lower_bound_used=context_lower_bound is not None,
        exact_long_suffix_batch_enabled=long_context_batch is not None,
        exact_long_suffix_batched_holds=long_batched_holds,
        exact_long_suffix_camera_cells=long_merged_prefixes,
        history_merge_policy='identical_suffix_refs' if history_key is None else 'caller_supplied_future_equivalent_physical_suffix',
        scalar_cost_or_debt_dominance=False,exact_global_optimum_claimed=False,
        no_beam_pruning_in_this_finite_domain=discarded==0 and bound_rejected==0,
        label_array_bytes=arrays,peak_managed_workspace_bytes=peak,workspace_limit_bytes=workspace_limit_bytes,
        callback_workspace_and_whole_process_RSS_included=False,full_frame_arrays_per_label=False,
        selected_arc_codes=list(reversed(codes)),opportunity_claims=claims,
        selected_context_costs=list(reversed(context)),opportunity_credit_once_only=True)
    return tuple(reversed(runs)),(objective,cuts),receipt
