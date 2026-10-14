"""Exact fixed-state C-W DP. No editorial maximum hold or duration birth fee.

Primary: integrated unary + real through/native events + (risk - opportunity).
Secondary: same-show/short-return diversity, only on EXACT primary ties.
Stable state/predecessor iteration is the deterministic final tie convention.
"""
import math
import numpy as np
from .stage_mix import native_event_cost


def choose_opportunities(nodes, *, fps, unary, ports, state_sources, source_events,
                         same_show, through, check=lambda:None, minimum_hold=.9,
                         event_seconds=False,native_cost=None):
    nodes,state_sources = np.asarray(nodes),np.asarray(state_sources,int)
    q,sources = len(state_sources),len(source_events)
    N = len(nodes)
    if (N < 2 or not q or not sources or nodes.dtype.kind not in 'iu' or nodes[0] != 0
            or np.any(np.diff(nodes) <= 0) or not math.isfinite(fps) or fps <= 0
            or not math.isfinite(minimum_hold) or minimum_hold<=0
            or unary.shape != (q,N-1) or through.shape != (q,N) or same_show.shape != (q,q)
            or not all(np.isfinite(v).all() for v in (unary,through,same_show))
            or (not event_seconds and np.any(through<0)) or np.any(same_show<0)
            or np.any(state_sources<0) or np.any(state_sources>=sources)
            or any(np.asarray(v).ndim!=1 or not np.isfinite(v).all()
                   or np.any(np.diff(v)<=0) or np.any(v<0) or np.any(v>nodes[-1]) for v in source_events)):
        raise ValueError('finite complete source-bound opportunity graph required')
    prefix = np.pad(np.cumsum(unary*np.diff(nodes)/fps,axis=1),((0,0),(1,0)))
    crossing = np.pad(np.cumsum(through,axis=1),((0,0),(1,0)))
    primary = np.full((N,q,sources+1),np.inf)
    secondary = np.full_like(primary,np.inf)
    previous,cache = {},{}
    by_source = [np.flatnonzero(state_sources==s) for s in range(sources)]
    def incoming(start,short):
        if (start,short) in cache:return cache[start,short]
        check()
        matrix = np.asarray(ports(start))
        if matrix.shape != (q,q) or not np.isfinite(matrix).all():
            raise ValueError('finite opportunity port matrix required')
        for short_class in (False,True):
            p,d = np.full((q,sources),np.inf),np.full((q,sources),np.inf)
            old,prior = np.full((q,sources),-1,int),np.full((q,sources),-1,int)
            for new_source,targets in enumerate(by_source):
                if not len(targets):continue
                shifted = secondary[start].copy()
                if short_class:shifted[:,new_source] += .7
                order = np.lexsort((shifted,primary[start]),axis=1)[:,0]
                pp = primary[start,np.arange(q),order]
                dd = shifted[np.arange(q),order]
                for old_source,indices in enumerate(by_source):
                    if old_source == new_source or not len(indices):continue
                    jp = pp[indices,None]+matrix[np.ix_(indices,targets)]
                    jd = dd[indices,None]+same_show[np.ix_(indices,targets)]
                    arg = np.lexsort((jd,jp),axis=0)[0]
                    picked = indices[arg]
                    p[targets,old_source] = jp[arg,np.arange(len(targets))]
                    d[targets,old_source] = jd[arg,np.arange(len(targets))]
                    old[targets,old_source] = picked
                    prior[targets,old_source] = order[picked]
            cache[start,short_class] = p,d,old,prior
        return cache[start,short]
    for end in range(1,N):
        check()
        for start in range(end-1,-1,-1):
            check()  # same absolute selection deadline, not only port callbacks
            duration = (nodes[end]-nodes[start])/fps
            if duration < minimum_hold and end != N-1:continue
            hold = (prefix[:,end]-prefix[:,start]+crossing[:,end]-crossing[:,start+1]
                    +(1. if event_seconds else 2.)*max(0.,minimum_hold-duration))
            event = np.array([native_event_cost(nodes[start],nodes[end],v,fps) for v in source_events]) if native_cost is None else np.asarray(native_cost(start,end),float)
            if event.shape!=(sources,) or not np.isfinite(event).all():raise ValueError('finite per-source native hold costs required')
            hold += event[state_sources]
            if start == 0:
                primary[end,:,sources] = hold;secondary[end,:,sources] = 0.
                for state in range(q):previous[end,state,sources] = (0,-1,-1)
                continue
            p,d,old,prior = incoming(start,duration<3.)
            candidate = p+hold[:,None]+(1./fps if event_seconds else .35)
            current = primary[end,:,:sources]
            better = (candidate<current)|((candidate==current)&(d<secondary[end,:,:sources]))
            primary[end,:,:sources][better] = candidate[better]
            secondary[end,:,:sources][better] = d[better]
            for state,source in zip(*np.nonzero(better)):
                previous[end,int(state),int(source)] = (start,int(old[state,source]),int(prior[state,source]))
    chosen = np.lexsort((secondary[-1].ravel(),primary[-1].ravel()))[0]
    state,prior = np.unravel_index(chosen,primary[-1].shape)
    score,diversity = float(primary[-1,state,prior]),float(secondary[-1,state,prior])
    if not math.isfinite(score):raise ValueError('no complete physically covering opportunity programme')
    runs,end = [],N-1
    while end:
        start,old,before = previous[end,int(state),int(prior)]
        runs.append((start,end,int(state)))
        end,state,prior = start,old,before
    return tuple(reversed(runs)),score,diversity
