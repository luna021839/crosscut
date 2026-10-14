"""Exact fixed-state DP: event preference FIRST, visibility/support SECOND.

The caller supplies actual frozen, same-event H/K losses. No default aesthetic
model, giant multiplier, source quota, maximum hold or calibration is invented.
Internal native-event losses belong to preference_intervals; grid H is replaced
by K at a cut, never counted twice. All supplied costs must be finite.
"""
import math
import numpy as np


def choose(nodes,*,fps,unary,through,preference_holds,preference_intervals,
           preference_ports,state_sources,source_events,minimum_hold,native_cost,
           check=lambda:None,fragment_in_primary=False):
    nodes=np.asarray(nodes);sources=np.asarray(state_sources,int)
    q,N=len(sources),len(nodes)
    if (N<2 or not q or nodes.dtype.kind not in 'iu' or nodes[0]!=0 or np.any(np.diff(nodes)<=0)
            or not math.isfinite(fps) or fps<=0 or not math.isfinite(minimum_hold) or minimum_hold<=0
            or unary.shape!=(q,N-1) or through.shape!=(q,N)
            or preference_holds.shape!=(q,N) or preference_intervals.shape!=(q,N-1)
            or not all(np.isfinite(v).all() for v in (unary,through,preference_holds,preference_intervals))
            or np.any(sources<0) or np.any(sources>=len(source_events))):
        raise ValueError('finite fixed-state lexicographic preference graph required')
    visibility=np.pad(np.cumsum(unary*np.diff(nodes)/fps,axis=1),((0,0),(1,0)))
    native=np.pad(np.cumsum(through,axis=1),((0,0),(1,0)))
    holds=np.pad(np.cumsum(preference_holds,axis=1),((0,0),(1,0)))
    intervals=np.pad(np.cumsum(preference_intervals,axis=1),((0,0),(1,0)))
    primary=np.full((N,q),np.inf);secondary=np.full((N,q),np.inf)
    predecessor={};entries={};different=sources[:,None]!=sources[None,:]
    def incoming(n):
        if n in entries:return entries[n]
        check();matrix=np.asarray(preference_ports(n))
        if matrix.shape!=(q,q) or not np.isfinite(matrix).all():raise ValueError('finite same-event K matrix required')
        a=primary[n,:,None]+matrix;b=np.broadcast_to(secondary[n,:,None],(q,q)).copy()
        a=np.where(different,a,np.inf);b=np.where(different,b,np.inf)
        old=np.lexsort((b,a),axis=0)[0];at=np.arange(q)
        entries[n]=(a[old,at],b[old,at],old)
        return entries[n]
    for end in range(1,N):
        for start in range(end-1,-1,-1):
            check();duration=(nodes[end]-nodes[start])/fps
            if duration<minimum_hold and end!=N-1:continue
            # Only strictly internal grid events consume H. The entry event
            # consumes K once; the final endpoint belongs to the next hold.
            p=intervals[:,end]-intervals[:,start]+holds[:,end]-holds[:,start+1]
            costs=np.asarray(native_cost(start,end),float)
            if costs.shape!=(len(source_events),) or not np.isfinite(costs).all():
                raise ValueError('finite source-local support costs required')
            if fragment_in_primary:p=p+costs[sources]
            s=(visibility[:,end]-visibility[:,start]+native[:,end]-native[:,start+1]
                +(0. if fragment_in_primary else costs[sources]+max(0.,minimum_hold-duration)))
            if start==0:
                a,b,old=p,s,np.full(q,-1,int)
            else:
                ep,es,old=incoming(start);a,b=ep+p,es+s+1/fps
            better=(a<primary[end])|((a==primary[end])&(b<secondary[end]))
            primary[end,better]=a[better];secondary[end,better]=b[better]
            for state in np.flatnonzero(better):predecessor[end,int(state)]=(start,int(old[state]))
    state=int(np.lexsort((secondary[-1],primary[-1]))[0])
    objective=(float(primary[-1,state]),float(secondary[-1,state]))
    if not all(math.isfinite(v) for v in objective):raise ValueError('no complete preference programme')
    end=N-1;runs=[]
    while end:
        first,old=predecessor[end,state];runs.append((first,end,state));end,state=first,old
    return tuple(reversed(runs)),objective
