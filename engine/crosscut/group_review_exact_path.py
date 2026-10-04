"""Finite exact adjacency and Pareto path selection over sealed review variants.

No across-suffix beam pruning or score-based top-two preselection is used.
Bounds are declared before edge evaluation. Truncation is NOT infeasibility.
Physical/motion and full output checks remain the caller's responsibility.
"""
import math
from .group_review_realized_path import RealizedReviewVariant


def nonblocking_beam_diagnostic(exact, run_beam, *, constraints_equivalent):
    """An optional diagnostic cannot undo a complete exact selection."""
    if exact.get('state')!='COMPLETE' or type(constraints_equivalent) is not bool:
        raise ValueError('beam diagnostic needs complete exact result and explicit constraint parity')
    result=dict(exact_utility=exact['utility'],exact_changed_frames=exact['changed_frames'],
        exact_result_retained=True,quality_approved=False)
    if not constraints_equivalent:
        return dict(result,state='NOT_COMPARABLE_CONSTRAINTS',beam_lost_utility=None)
    try:
        approximate=run_beam()
        return dict(result,state='COMPARABLE',beam_utility=approximate['utility'],
            beam_changed_frames=approximate['changed_frames'],
            beam_lost_utility=exact['utility']-approximate['utility'])
    except Exception as error:
        return dict(result,state='DIAGNOSTIC_FAILED',error_type=type(error).__name__,beam_lost_utility=None)


def precompute_review_edge_graph(layers,edge_at,*,maximum_edges=60000,work_port=None):
    if (not isinstance(layers,(list,tuple)) or not 1<=len(layers)<=256
            or any(not row or len(row)>256 or any(not isinstance(v,RealizedReviewVariant) for v in row) for row in layers)
            or type(maximum_edges) is not int or not 1<=maximum_edges<=100000):
        raise ValueError('exact review graph needs a finite declared variant/edge domain')
    pairs=[(a,b) for left,right in zip(layers,layers[1:]) for a in left for b in right if a.end==b.first]
    if len(pairs)>maximum_edges:
        return dict(state='TRUNCATED',preflight_compatible_edges=len(pairs),edges={},reason='edge_domain_budget')
    cache={};reasons={}
    for a,b in pairs:
        key=(a.identity,b.identity)
        if key in cache:continue
        if work_port is not None:work_port.charge('edges')
        edge=edge_at(a,b)
        if (not isinstance(edge,dict) or type(edge.get('allowed')) is not bool
                or edge.get('left_identity')!=a.identity or edge.get('right_identity')!=b.identity
                or edge.get('boundary_tick')!=b.first or type(edge.get('cost')) not in (int,float)
                or not math.isfinite(edge['cost']) or edge['cost']<0):
            raise ValueError('exact graph edge changed its actual neighbor identity')
        cache[key]=edge
        reason=edge.get('final_reason','ALLOWED' if edge['allowed'] else 'VETO')
        reasons[reason]=reasons.get(reason,0)+1
    return dict(state='COMPLETE',preflight_compatible_edges=len(pairs),edges=cache,
        measured_edges=len(cache),final_edge_reasons=reasons,quality_approved=False)


def review_two_sided_reachability(layers,graph,*,maximum_changed_frames):
    """Minimum change budget via BOTH ends, not a media-quality certificate."""
    if graph.get('state')!='COMPLETE':return dict(state='TRUNCATED')
    forward={};backward={};edges=graph['edges']
    for i,row in enumerate(layers):
        for v in row:
            previous=[forward[a.identity] for a in layers[i-1] if a.identity in forward
                and (a.identity,v.identity) in edges and edges[a.identity,v.identity]['allowed']] if i else [0] if v.first==0 else []
            if previous:forward[v.identity]=min(previous)+v.changed_frames
    for i in range(len(layers)-1,-1,-1):
        for v in layers[i]:
            following=[backward[b.identity] for b in layers[i+1] if b.identity in backward
                and (v.identity,b.identity) in edges and edges[v.identity,b.identity]['allowed']] if i<len(layers)-1 else [0]
            if following:backward[v.identity]=min(following)+v.changed_frames
    rows=[]
    for i,row in enumerate(layers):
        for v in row:
            total=None if v.identity not in forward or v.identity not in backward else (
                forward[v.identity]+backward[v.identity]-v.changed_frames)
            rows.append(dict(layer=i,identity=v.identity,first=v.first,end=v.end,changed_frames=v.changed_frames,
                forward_minimum_changed=forward.get(v.identity),backward_minimum_changed=backward.get(v.identity),
                complete_path_minimum_changed=total,reachable_with_change_budget=total is not None and total<=maximum_changed_frames))
    return dict(state='COMPLETE',variants=rows,minimum_change_reachability_only=True,
        visible_cut_budget_or_motion_quality_certified=False,quality_approved=False)


def select_exact_realized_review_path(layers,graph,*,output_frame_count,
        maximum_changed_frames=1600,maximum_visible_cuts=140,maximum_states=120000,
        internal_cut_counts=None,work_port=None):
    if (graph.get('state')!='COMPLETE' or any(type(v) is not int or v<1 for v in
            (output_frame_count,maximum_changed_frames,maximum_visible_cuts,maximum_states))
            or maximum_states>500000 or output_frame_count>36000):
        raise ValueError('exact selection needs a complete bounded graph and explicit budgets')
    counts={} if internal_cut_counts is None else internal_cut_counts
    if any(type(v) is not int or v<0 for v in counts.values()):
        raise ValueError('exact cut budget needs actual unique native event counts')
    fronts={};states=0;dominated=0;edges=graph['edges']
    for i,row in enumerate(layers):
        current={}
        for v in row:
            parents=[p for a in layers[i-1] for p in fronts.get(a.identity,())
                if (a.identity,v.identity) in edges and edges[a.identity,v.identity]['allowed']] if i else [
                    dict(changed=0,cuts=0,utility=0.,variants=(),edges=(),tie=())] if v.first==0 else []
            choices=[]
            for p in parents:
                edge=None if not p['variants'] else edges[p['variants'][-1].identity,v.identity]
                changed=p['changed']+v.changed_frames
                cuts=p['cuts']+counts.get(v.identity,0)+(0 if edge is None else edge.get('visible_boundary_event_count',0))
                if changed>maximum_changed_frames or cuts>maximum_visible_cuts:continue
                states+=1
                if states>maximum_states:return dict(state='TRUNCATED',visited_states=states,
                    reason='exact_state_budget',no_feasible_path_claimed=False,quality_approved=False)
                if work_port is not None:work_port.charge('states')
                choices.append(dict(changed=changed,cuts=cuts,utility=p['utility']+v.utility-(0. if edge is None else edge['cost']),
                    variants=p['variants']+(v,),edges=p['edges']+(() if edge is None else (edge,)),tie=p['tie']+(v.identity,)))
            choices.sort(key=lambda p:(p['changed'],p['cuts'],-p['utility'],p['tie']))
            keep=[]
            for p in choices:
                if any(q['changed']<=p['changed'] and q['cuts']<=p['cuts'] and q['utility']>=p['utility']
                        and (q['utility']>p['utility'] or q['tie']<=p['tie']) for q in keep):
                    dominated+=1;continue
                keep=[q for q in keep if not (p['changed']<=q['changed'] and p['cuts']<=q['cuts']
                    and p['utility']>=q['utility'] and (p['utility']>q['utility'] or p['tie']<=q['tie']))]
                keep.append(p)
            if keep:current[v.identity]=keep
        fronts=current
        if not fronts:return dict(state='NO_FEASIBLE_PATH',visited_states=states,quality_approved=False)
    final=[p for values in fronts.values() for p in values if p['variants'][-1].end==output_frame_count]
    if not final:return dict(state='NO_FEASIBLE_PATH',visited_states=states,quality_approved=False)
    best=min(final,key=lambda p:(-p['utility'],p['changed'],p['cuts'],p['tie']))
    return dict(state='COMPLETE',**{k:best[k] for k in ('variants','edges','utility')},
        changed_frames=best['changed'],visible_cuts=best['cuts'],visited_states=states,
        measured_edges=graph['measured_edges'],dominated_resource_suffix_states_removed=dominated,
        across_suffix_beam_pruning=False,quality_approved=False,numerical_realized_path_only=True)
