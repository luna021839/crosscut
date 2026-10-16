"""Origin-major semi-Markov DP with opaque exact physical playback state.

An explicit compiler owns compile_hold/current/score and real input evidence;
this solver does not infer clocks or fabricate context. Each origin bank is
decoded once and kept across every future hold. Pending destination beams own
their roots until displaced/finalized, while compact backtraces own no payload.
This is an opt-in executor; legacy reference and normal app gates are unchanged.
"""
import numpy as np
from .boundary_port_dp import _ports,_vector,_mask,_matrix
from .compiled_context_beam import CompiledBeam
from .playback_suffix_arena import SuffixRoot,PlaybackSuffixArena
from .spooled_playback_arena import SpooledPlaybackArena


def solve(nodes,entries,exits,*,start_cost,hold_cost,cut_cost,compiler,episodes,credit,
          history_frames,label_limit=8,minimum_frames=12,check=lambda:None,
          start_allowed=None,final_cost=None,final_allowed=None):
    nodes=np.asarray(nodes);episodes=np.asarray(episodes)
    arena=getattr(compiler,'arena',None)
    if (nodes.ndim!=1 or len(nodes)<2 or nodes.dtype.kind not in 'iu' or nodes[0]!=0
            or np.any(np.diff(nodes)<=0) or episodes.shape!=nodes.shape or episodes.dtype.kind not in 'iu'
            or np.any(episodes<0) or np.any(np.diff(episodes)<0)
            or type(history_frames) is not int or history_frames<1
            or type(label_limit) is not int or not 1<=label_limit<=32
            or type(minimum_frames) is not int or minimum_frames<1
            or not callable(credit) or not isinstance(arena,(PlaybackSuffixArena,SpooledPlaybackArena)) or arena.H!=history_frames):
        raise ValueError('ordered nodes/episodes and an owned exact playback compiler required')
    N=len(nodes);entries=_ports(entries,N,'entry');exits=_ports(exits,N,'exit')
    W=max(map(len,entries+exits));L=label_limit;workspace=arena.workspace
    beam=compiler.make_beam() if callable(getattr(compiler,'make_beam',None)) else CompiledBeam(arena)
    start=_vector(start_cost,len(entries[0]),'start_cost');begin=_mask(start_allowed,len(entries[0]),'start_allowed')
    final=np.zeros(len(exits[-1])) if final_cost is None else _vector(final_cost,len(exits[-1]),'final_cost')
    finish=_mask(final_allowed,len(exits[-1]),'final_allowed')
    # Packed whole-record arrays: only seven allocation owners, not N dicts
    # of candidate histories. Their root field is cleared on payload release.
    shape=(2,N*W,2,L);fields=[]
    for k,dtype in enumerate((np.float64,np.int64,np.int64,np.int64,np.float64,np.float64,np.uint64)):
        array=workspace.empty(shape,dtype,owner='packed-DP-labels');array[:]=np.inf if k==0 else -1 if k in (2,3) else 0
        fields.append(array)
    arrays=sum(a.nbytes for a in fields);hold_calls=cut_calls=cameras=batches=evaluations=bound_rejected=0
    origin_banks=0

    def state(side,n,size):return tuple(f[side,n*W:n*W+size] for f in fields)
    def group(state,p,bit):return tuple(f[p,bit] for f in state)
    def sort(state):
        for p in range(len(state[0])):
            for bit in (0,1):
                lane=group(state,p,bit)
                order=sorted(range(L),key=lambda k:(lane[0][k],lane[1][k],lane[2][k],lane[3][k],-lane[4][k]))
                for column in lane:column[:]=column[order]
    def release(state):
        for handle in state[6].ravel():
            if handle:arena.release(SuffixRoot(arena.owner,int(handle)))
        state[6][:]=0

    incoming0=state(0,0,len(entries[0]))
    try:
        for p in range(len(entries[0])):
            if begin[p]:
                incoming0[0][p,0,0]=start[p];incoming0[6][p,0,0]=arena.empty().handle
        for i in range(N):
            check();incoming=state(0,i,len(entries[i]));outgoing=state(1,i,len(exits[i]))
            if i:
                sort(outgoing)
                if i==N-1:break
                if np.isfinite(outgoing[0]).any():
                    cut,allowed=_matrix(cut_cost(i),(len(exits[i]),len(entries[i])),'cut')
                    allowed=allowed&(exits[i][:,None]!=entries[i][None,:]);cut_calls+=1
                    for q,p in zip(*np.nonzero(allowed)):
                        for bit in (0,1):
                            for k in np.flatnonzero(np.isfinite(outgoing[0][q,bit])):
                                values=(float(outgoing[0][q,bit,k]+cut[q,p]),int(outgoing[1][q,bit,k])+1,
                                    int(q)*L+int(k),-1,0.,0.)
                                beam.cut(group(incoming,int(p),bit),values,outgoing[6][q,bit,k])
                    sort(incoming)
                release(outgoing)
            if not np.isfinite(incoming[0]).any():continue
            # Compiler bank contains all finalized port/bit groups once. No
            # history tuple, camera replay or raw input reconstruction enters
            # any future-prefix evaluation in this executor.
            with compiler.open_origin(i,incoming[6],incoming[0]) as bank:
                origin_banks+=1
                for j in range(i+1,N):
                    check()
                    if nodes[j]-nodes[i]<minimum_frames and j!=N-1:continue
                    with compiler.reserve_hold(i,j) as producer:
                        base,allowed=hold_cost(i,j);base=np.asarray(base);allowed=np.asarray(allowed)
                        expected=(len(entries[i]),len(exits[j]))
                        if (base.ndim!=3 or base.shape[:2]!=expected or base.dtype.kind not in 'iuf'
                                or allowed.shape!=base.shape or allowed.dtype!=bool or not np.isfinite(base).all()):
                            raise ValueError('all finite unchanged per-camera alternatives and masks required')
                        allowed=allowed&(entries[i][:,None,None]==exits[j][None,:,None])
                        benefit=np.asarray(credit(i,j),float)
                        if benefit.shape==(expected[0],):benefit=np.broadcast_to(benefit[:,None],expected)
                        if benefit.shape!=expected or not np.isfinite(benefit).all() or np.any(benefit<0):
                            raise ValueError('actual finite nonnegative once-only opportunity credit required')
                        producer.observe(base,allowed,benefit)
                        hold_calls+=1;cameras+=int(allowed.sum());destination=state(1,j,len(exits[j]))
                        with compiler.compile_hold(i,j,base,allowed) as hold:
                            general=allowed;batch=None
                            if callable(getattr(compiler,'long_candidates',None)):
                                batch=compiler.long_candidates(hold,bank,base,allowed,incoming[0],incoming[1],benefit,
                                    same_episode=episodes[i]==episodes[j],width=W)
                            if batch is not None:
                                handled,values=batch
                                def build_suffix(p,q,code,bit,slot):
                                    with hold.current(p,q,code,part='suffix') as current:
                                        return arena.append(SuffixRoot(arena.owner,int(incoming[6][p,bit,slot])),current.lease)
                                beam.long_batch(destination,values,compiler.long_roots[j],build_suffix)
                                general=workspace.empty(allowed.shape,bool,owner='unfactored-current-camera-domain')
                                general[:]=allowed;general[:,:,:4]&=~handled
                                batch=handled=values=None
                            for p,q,code in zip(*np.nonzero(general)):
                                p,q,code=map(int,(p,q,code))
                                with hold.current(p,q,code) as current:
                                    lower=float(current.lower_bound)
                                    if not np.isfinite(lower):raise ValueError('proved finite context lower bound required')
                                    for bit in (0,1):
                                        slots=np.flatnonzero(np.isfinite(incoming[0][p,bit]));viable=[]
                                        for k in slots:
                                            raw=float((incoming[0][p,bit,k]+base[p,q,code])+lower)
                                            back=(((i*W+p)*2+bit)*L+int(k));count=int(incoming[1][p,bit,k])
                                            possible=False
                                            for claim in (False,True) if bit==0 and benefit[p,q]>0 else (False,):
                                                paid=float(benefit[p,q]) if claim else 0.
                                                outbit=(1 if bit or claim else 0) if episodes[i]==episodes[j] else 0
                                                target=group(destination,q,outbit);present=target[6]!=0
                                                worst=max(((target[0][x],target[1][x],target[2][x],target[3][x],-target[4][x])
                                                    for x in np.flatnonzero(present)),default=None)
                                                possible|=not present.all() or (raw-paid,count,back,code,-paid)<worst
                                            if possible:viable.append(int(k))
                                        bound_rejected+=len(slots)-len(viable)
                                        if not viable:continue
                                        extras=np.asarray(compiler.score(bank,current,p,bit,viable),float)
                                        if extras.shape!=(len(viable),) or not np.isfinite(extras).all() or np.any(extras<lower):
                                            raise ValueError('one finite exact context increment per retained prefix required')
                                        batches+=1;evaluations+=len(viable)
                                        for k,extra in zip(viable,extras):
                                            raw=float((incoming[0][p,bit,k]+base[p,q,code])+extra)
                                            count=int(incoming[1][p,bit,k]);back=(((i*W+p)*2+bit)*L+k)
                                            for claim in (False,True) if bit==0 and benefit[p,q]>0 else (False,):
                                                paid=float(benefit[p,q]) if claim else 0.
                                                outbit=(1 if bit or claim else 0) if episodes[i]==episodes[j] else 0
                                                values=(raw-paid,count,back,code,paid,float(extra))
                                                beam.offer_lazy(group(destination,q,outbit),values,incoming[6][p,bit,k],current)
                            batch=general=values=None
                    base=allowed=benefit=None
            release(incoming)
        last=state(1,N-1,len(exits[-1]));total=np.where(finish[:,None,None],last[0]+final[:,None,None],np.inf)
        if not np.isfinite(total).any():raise ValueError('no complete compiled contextual port programme')
        order=np.lexsort((np.arange(total.size),last[1].ravel(),total.ravel()))
        at=int(order[0]);q,bit,k=map(int,np.unravel_index(at,total.shape));objective=float(total.ravel()[at])
        cuts=int(last[1][q,bit,k]);j=N-1;runs=[];codes=[];claims=[];context=[]
        while True:
            outgoing=state(1,j,len(exits[j]));encoded=int(outgoing[2][q,bit,k]);oldslot=encoded%L;encoded//=L
            oldbit=encoded%2;encoded//=2;i,p=divmod(encoded,W)
            if i<0 or i>=j:raise AssertionError('compiled context backtrace is not acyclic')
            runs.append((i,j,p,q));codes.append(int(outgoing[3][q,bit,k]));paid=float(outgoing[4][q,bit,k])
            claims.append(dict(episode=int(episodes[i]),credit=paid,claimed=paid>0));context.append(float(outgoing[5][q,bit,k]))
            if i==0:break
            incoming=state(0,i,len(entries[i]));previous=int(incoming[2][p,oldbit,oldslot])
            q,k=divmod(previous,L);j=i;bit=oldbit
        claims=list(reversed(claims));paid_episodes=[c['episode'] for c in claims if c['claimed']]
        if len(paid_episodes)!=len(set(paid_episodes)):raise AssertionError('compiled DP paid one physical episode twice')
        receipt=dict(contract='origin_major_owned_exact_suffix_context_DP_v1',nodes=N,hold_calls=hold_calls,
            cut_calls=cut_calls,camera_alternatives_scored=cameras,context_callback_batches=batches,
            context_prefix_evaluations=evaluations,label_limit_per_port_paid_bit=L,
            labels_discarded_by_bound_events=beam.discarded,equivalent_histories_merged=beam.equivalent,
            expensive_history_keys_skipped_by_existing_beam_rank=beam.rank_rejected,
            prefix_context_evaluations_skipped_by_explicit_lower_bound=bound_rejected,
            finalized_origin_banks_decoded=origin_banks,history_frames=history_frames,
            factored_long_beam_kernel_calls=beam.long_kernel_calls,
            factored_long_rank_viable_suffix_compilations=beam.long_suffix_misses,
            history_merge_policy='exact_physical_suffix_only_within_same_port_and_paid_bit',
            scalar_cost_or_debt_dominance=False,exact_global_optimum_claimed=False,
            selected_arc_codes=list(reversed(codes)),opportunity_claims=claims,
            selected_context_costs=list(reversed(context)),opportunity_credit_once_only=True,
            label_array_bytes=arrays,workspace=workspace.receipt(),suffix_state=arena.receipt(),
            code_or_source_domain_reduced=False,normal_app_activation_approved=False)
        return tuple(reversed(runs)),(objective,cuts),receipt
    finally:
        # Includes future pending roots, current incoming roots and final roots
        # on success, cancellation, allocation refusal or callback exceptions.
        for side in (0,1):
            for n in range(N):release(state(side,n,len(entries[n]) if side==0 else len(exits[n])))
