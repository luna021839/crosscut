"""Bounded cost/debt labels over camera alternatives, ports and paid bits.

Transfers consume no pixels. Dominated labels are removed; if a frontier has
more than8 labels, fixed evenly spaced frontier ranks retain its cheapest and
least-debt ends. This bounded search is NOT an exact global-optimum claim.
"""
import numpy as np
from .scene_readout import apply


def _prune(fields,limit):
    cost,debt,cuts,*_=fields
    order=np.lexsort((debt,cuts,cost),axis=-1)
    values=[np.take_along_axis(x,order,axis=-1) for x in fields]
    cost,debt,cuts,*_=values
    previous=np.concatenate((np.full(debt.shape[:-1]+(1,),np.inf),np.minimum.accumulate(debt,axis=-1)[...,:-1]),axis=-1)
    keep=np.isfinite(cost)&(debt<previous)
    counts=keep.sum(-1);rank=np.cumsum(keep,axis=-1)-1
    desired=np.where(counts[...,None]<=limit,np.arange(limit),
        np.rint((counts[...,None]-1)*np.linspace(0,1,limit))).astype(int)
    selected=np.argmax(keep[...,None]& (rank[...,None]==desired[...,None,:]),axis=-2)
    valid=np.arange(limit)<counts[...,None]
    output=[np.take_along_axis(x,selected,axis=-1) for x in values]
    output[0]=np.where(valid,output[0],np.inf)
    output[1]=np.where(valid,output[1],np.inf)
    return output,int(np.maximum(0,counts-limit).sum())


def solve(nodes,entries,exits,*,start_cost,hold_cost,cut_cost,cut_transfer,episodes,credit,
          minimum_frames=12,label_limit=8,check=lambda:None,workspace_limit_bytes=200*2**20,
          start_allowed=None,final_cost=None,final_allowed=None):
    nodes=np.asarray(nodes);episodes=np.asarray(episodes)
    if (nodes.ndim!=1 or len(nodes)<2 or nodes.dtype.kind not in 'iu' or np.any(np.diff(nodes)<=0)
            or episodes.shape!=nodes.shape or episodes.dtype.kind not in 'iu' or np.any(np.diff(episodes)<0)
            or type(label_limit) is not int or not 2<=label_limit<=32
            or type(minimum_frames) is not int or minimum_frames<1):
        raise ValueError('ordered nodes/episodes and bounded readout labels required')
    entries=[np.asarray(v,int) for v in entries];exits=[np.asarray(v,int) for v in exits]
    if len(entries)!=len(nodes) or len(exits)!=len(nodes) or any(v.ndim!=1 or np.any(v<0) for v in entries+exits):
        raise ValueError('one nonnegative entry/exit source domain per node required')
    width=max(map(len,entries+exits));N=len(nodes);L=label_limit
    # cost, debt, cut count, predecessor, camera code, claimed credit. No
    # source/frame/camera arrays are cloned when a label is added.
    def empty(n):
        shape=(n,2,L)
        return [np.full(shape,np.inf),np.full(shape,np.inf),np.zeros(shape,np.int64),
            np.full(shape,-1,np.int64),np.full(shape,-1,np.int64),np.zeros(shape)]
    incoming=[empty(len(v)) for v in entries];outgoing=[empty(len(v)) for v in exits]
    persistent=sum(a.nbytes for states in (incoming,outgoing) for state in states for a in state)
    if persistent>workspace_limit_bytes:raise MemoryError('readout label workspace exceeds allowance')
    start=np.asarray(start_cost,float);begin=np.ones(len(entries[0]),bool) if start_allowed is None else np.asarray(start_allowed,bool)
    if start.shape!=begin.shape or start.shape!=(len(entries[0]),) or not np.isfinite(start).all():
        raise ValueError('finite source-bound start costs required')
    incoming[0][0][:,0,0]=np.where(begin,start,np.inf);incoming[0][1][:,0,0]=0.
    pruned=0;peak=persistent;hold_calls=0;camera_alternatives=0
    def reserve(count):
        nonlocal peak
        need=persistent+128*count
        if need>workspace_limit_bytes:raise MemoryError('readout candidate workspace exceeds allowance')
        peak=max(peak,need)
    def merge(target,rows,bit,fields):
        nonlocal pruned
        old=[v[rows,bit] for v in target]
        selected,removed=_prune([np.concatenate((a,b),axis=-1) for a,b in zip(old,fields)],L)
        pruned+=removed
        for array,values in zip(target,selected):array[rows,bit]=values
    for j in range(1,N):
        check()
        for i in range(j):
            check()
            if nodes[j]-nodes[i]<minimum_frames and j!=N-1:continue
            if not np.isfinite(incoming[i][0]).any():continue
            base,allowed,transfer=hold_cost(i,j)
            base,allowed,transfer=np.asarray(base,float),np.asarray(allowed),np.asarray(transfer,float)
            shape=(len(entries[i]),len(exits[j]))
            if (base.ndim!=3 or base.shape[:2]!=shape or allowed.shape!=base.shape or allowed.dtype!=bool
                    or transfer.shape!=base.shape+(4,) or not np.isfinite(base).all() or not np.isfinite(transfer).all()
                    or np.any(transfer[...,0]<0) or np.any(transfer[...,1:]<0)):
                raise ValueError('finite per-camera costs/affine readout transfers required')
            hold_calls+=1;K=base.shape[-1];camera_alternatives+=int(allowed.sum())
            benefit=np.asarray(credit(i,j),float)
            if benefit.shape==(shape[0],):benefit=np.broadcast_to(benefit[:,None],shape)
            if benefit.shape!=shape or not np.isfinite(benefit).all() or np.any(benefit<0):
                raise ValueError('nonnegative supported once-only credit required')
            same=episodes[i]==episodes[j]
            for source in np.intersect1d(entries[i],exits[j]):
                p=np.flatnonzero(entries[i]==source);q=np.flatnonzero(exits[j]==source)
                bc=base[np.ix_(p,q,np.arange(K))];ok=allowed[np.ix_(p,q,np.arange(K))]
                if not ok.any():continue
                tr=transfer[np.ix_(p,q,np.arange(K))];reserve(len(p)*L*len(q)*K)
                for bit,claim in ((0,False),(0,True),(1,False)):
                    outbit=(1 if bit or claim else 0) if same else 0
                    reward=benefit[np.ix_(p,q)] if claim else np.zeros((len(p),len(q)))
                    if claim and not (reward>0).any():continue
                    d=np.where(np.isfinite(incoming[i][0][p,bit]),incoming[i][1][p,bit],0.)[:,:,None,None]
                    new_debt,extra=apply(tr[:,None],d)
                    value=incoming[i][0][p,bit,:,None,None]+bc[:,None]+extra-reward[:,None,:,None]
                    mask=ok[:,None]&np.isfinite(incoming[i][0][p,bit,:,None,None])
                    if claim:mask&=reward[:,None,:,None]>0
                    value=np.where(mask,value,np.inf);new_debt=np.where(mask,new_debt,np.inf)
                    shape=value.shape
                    cuts=np.broadcast_to(incoming[i][2][p,bit,:,None,None],shape)
                    back=np.broadcast_to((((i*width+p)*2+bit)[:,None]*L+np.arange(L))[:,:,None,None],shape)
                    codes=np.broadcast_to(np.arange(K),shape)
                    paid=np.broadcast_to(reward[:,None,:,None],shape)
                    fields=[x.transpose(2,0,1,3).reshape(len(q),-1) for x in (value,new_debt,cuts,back,codes,paid)]
                    merge(outgoing[j],q,outbit,fields)
        if j==N-1:continue
        base,allowed=cut_cost(j);transfer=np.asarray(cut_transfer(j),float)
        base,allowed=np.asarray(base,float),np.asarray(allowed)
        shape=(len(exits[j]),len(entries[j]))
        if (base.shape!=shape or allowed.shape!=shape or allowed.dtype!=bool or transfer.shape!=shape+(4,)
                or not np.isfinite(base).all() or not np.isfinite(transfer).all()):
            raise ValueError('source-bound cut costs and shared native/authored readout transfer required')
        allowed=allowed&(exits[j][:,None]!=entries[j][None,:]);reserve(shape[0]*shape[1]*L)
        for bit in (0,1):
            d=np.where(np.isfinite(outgoing[j][0][:,bit]),outgoing[j][1][:,bit],0.)[:,:,None]
            new_debt,extra=apply(transfer[:,None],d)
            value=outgoing[j][0][:,bit,:,None]+base[:,None]+extra
            value=np.where(allowed[:,None]&np.isfinite(outgoing[j][0][:,bit,:,None]),value,np.inf)
            new_debt=np.where(np.isfinite(value),new_debt,np.inf)
            cuts=np.broadcast_to(outgoing[j][2][:,bit,:,None]+1,value.shape)
            back=np.broadcast_to((np.arange(shape[0])[:,None]*L+np.arange(L))[:,:,None],value.shape)
            fields=[x.transpose(2,0,1).reshape(shape[1],-1) for x in (value,new_debt,cuts,back)]
            fields.extend([np.full(fields[0].shape,-1,np.int64),np.zeros(fields[0].shape)])
            selected,removed=_prune(fields,L);pruned+=removed
            for array,values in zip(incoming[j],selected):array[:,bit]=values
    endcost=outgoing[-1][0].copy()
    if final_cost is not None:endcost+=np.asarray(final_cost,float)[:,None,None]
    if final_allowed is not None:endcost=np.where(np.asarray(final_allowed,bool)[:,None,None],endcost,np.inf)
    order=np.lexsort((outgoing[-1][1].ravel(),outgoing[-1][2].ravel(),endcost.ravel()))
    at=int(order[0]);objective=float(endcost.ravel()[at])
    if not np.isfinite(objective):raise ValueError('no complete readout programme')
    q,bit,slot=map(int,np.unravel_index(at,endcost.shape));cuts=int(outgoing[-1][2][q,bit,slot]);j=N-1
    runs=[];codes=[];claims=[];readout=[]
    while True:
        code=int(outgoing[j][4][q,bit,slot]);amount=float(outgoing[j][5][q,bit,slot])
        encoded=int(outgoing[j][3][q,bit,slot]);oldslot=encoded%L;encoded//=L
        oldbit=encoded%2;encoded//=2;i,p=divmod(encoded,width)
        runs.append((i,j,p,q));codes.append(code)
        claims.append(dict(episode=int(episodes[i]),credit=amount,claimed=amount>0))
        readout.append(dict(entry_debt=float(incoming[i][1][p,oldbit,oldslot]),
            exit_debt=float(outgoing[j][1][q,bit,slot])))
        if not i:break
        previous=int(incoming[i][3][p,oldbit,oldslot]);q,slot=divmod(previous,L);j=i;bit=oldbit
    claims=list(reversed(claims));paid=[v['episode'] for v in claims if v['claimed']]
    if len(paid)!=len(set(paid)):raise AssertionError('readout search paid one episode twice')
    receipt=dict(nodes=N,entry_ports=sum(map(len,entries)),exit_ports=sum(map(len,exits)),
        hold_calls=hold_calls,camera_alternatives_scored=camera_alternatives,
        label_limit_per_port_paid_bit=L,frontier_labels_discarded_by_bound=pruned,
        exact_global_optimum_claimed=False,dominance='cost_then_cut_count_and_residual_debt',
        bounded_frontier_policy='even_ranks_including_cheapest_and_least_debt',
        persistent_bytes=persistent,peak_workspace_bytes=peak,workspace_limit_bytes=workspace_limit_bytes,
        full_frame_arrays_per_label=False,selected_arc_codes=list(reversed(codes)),
        opportunity_claims=claims,current_event_paid_bit=True,opportunity_credit_once_only=True,
        selected_readout_states=list(reversed(readout)),burden_reset_at_episode_or_native_cut=False)
    return tuple(reversed(runs)),(objective,cuts),receipt
