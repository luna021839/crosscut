"""Bounded connected anonymous appearance hypotheses for GROUP reviews.

One candidate owns its whole source-local head sequence. Entry and exit may
not independently pick different people. These are review hypotheses over
recorded measurements, never named identity, LK authority or quality approval.
Missing/ambiguous associations survive explicitly, without max-face recovery.
"""
import math

from .group_performer_family_owner import digest
from .group_review_camera_repair import shape_cost
from .group_review_native_window import cosine
from .group_review_source_equivalence import physical_review_file_identity


def _content_key(view):
    if view is None:
        return 'unmatched'
    d = view['descriptor']
    # Detector head numbers are frame-local, not cross-frame identity or ties.
    return digest(dict(vector=view['vector'], member_hint=view.get('member_hint'),center=d.head_center,
        height=d.face_height, width=d.face_width, shape=d.pose_shape))


def _address(view):
    row = view['row']
    raw = row['native_observation']
    camera = (('native_run', view['native_camera_unit_index'])
        if type(view.get('native_camera_unit_index')) is int
        else row['projection']['camera_sha256'])
    return (row['output_tick'], raw['request_source_index'], raw['native_frame_index'],
        raw['source_clock_identity'], camera)


def _native_head_geometry(view):
    head_id = view['descriptor'].frame_local_head_id
    heads = [h for h in view['row']['native_observation'].get('heads', ())
        if h.get('observation_id') == head_id and 'box_xywh' in h]
    if len(heads) == 1:
        x, y, w, h = heads[0]['box_xywh']
        return (x + w/2, y + h/2), h
    descriptor = view['descriptor']
    return descriptor.head_center, descriptor.face_height


def _link(left, right, *, maximum_native_gap, minimum_cosine,observed_native_camera_cut=False):
    if left is None or right is None:
        return dict(cost=1.5, state='unmatched', cosine=None, linked=False)
    a,b = _address(left),_address(right)
    if a[1] != b[1] or a[3] != b[3] or a[0] >= b[0] or a[2] > b[2]:
        raise ValueError('source-local actor path borrowed another source/clock or reversed time')
    if (a[4] != b[4] and not observed_native_camera_cut) or b[2]-a[2] > maximum_native_gap:
        return dict(cost=2.,state='unmeasured_camera_or_gap_bridge',cosine=None,linked=False)
    if a[2] == b[2] and left['row']['native_observation'] != right['row']['native_observation']:
        raise ValueError('same native frame has conflicting actor-path observations')
    if a[2] == b[2] and left['descriptor'].frame_local_head_id != right['descriptor'].frame_local_head_id:
        return None  # Two occurrences in ONE raster are not a temporal association.
    if physical_review_file_identity(left) != physical_review_file_identity(right):
        raise ValueError('source-local actor path changed source file identity')
    sim = cosine(left['vector'],right['vector'])
    if sim is None or sim < minimum_cosine:
        return None  # The real unmatched alternative remains in the graph.
    shape = shape_cost(left['descriptor'],right['descriptor'])
    from .ive_local_arcface_gallery import local_member_hint_conflict
    member_conflict=local_member_hint_conflict(left.get('member_hint'),right.get('member_hint'))
    reacquired=a[4]!=b[4]
    # Identity embeddings alone can prefer another performer doing the same
    # dance. A camera-local hypothesis also owns its measured head trajectory.
    # A real observed camera cut is a separate reacquisition, not a spatial
    # jump penalized as a continuous local track.
    spatial=0.
    if not reacquired:
        ac,ah=_native_head_geometry(left);bc,bh=_native_head_geometry(right)
        distance=math.hypot(bc[0]-ac[0],(bc[1]-ac[1])*9/16)
        # A long future sequence of a different foreground person must not
        # amortize an instantaneous cross-frame teleport. This deliberately
        # broad native-raster displacement bound scales with observed gap and
        # head extent. Real camera cuts are handled by explicit reacquisition.
        # An excluded link leaves an unmatched slot, not an identity verdict.
        if distance>.06*max(1,b[2]-a[2])+.4*(ah+bh):
            return None
        spatial=.4*distance/(.12+(ah+bh)/2)
        spatial+=.2*abs(math.log(bh/ah))
    return dict(cost=2.*max(0.,1.-sim)+spatial+(.25 if shape is None else .15*shape)+(.25 if reacquired else 0.)
            +(2. if member_conflict else 0.),
        measured_camera_local_spatial_cost=spatial,
        local_photo_member_hint_conflict=member_conflict,
        state='appearance_reacquisition_at_observed_native_cut' if reacquired else 'appearance_link_hypothesis',
        cosine=sim,linked=True,continuous_camera_local_track_hypothesis=not reacquired,
        observed_native_camera_cut=observed_native_camera_cut)


def anonymous_review_actor_paths(frames, *, maximum_paths=2, maximum_heads=4,
        maximum_native_gap=12, minimum_cosine=.25, ambiguity_margin=.025,
        maximum_beam=32, work_port=None,known_native_camera_cut_edges=(),distinct_anchor_hypotheses=False):
    """Return a small set of complete sequences, retaining unmatched states.

    Each ordered frame supplies explicit, actually observed head views. A head
    view uses its own descriptor/vector/pose, not another largest body. The
    caller can put each connected sequence into the whole-edit beam, making
    the actual previous/next selected actor part of that beam's seam cost.
    """
    if (not isinstance(frames,(list,tuple)) or not 2 <= len(frames) <= 32
            or any(type(v) is not int or not 1 <= v <= cap for v,cap in
                ((maximum_paths,4),(maximum_heads,8),(maximum_beam,64),(maximum_native_gap,12)))
            or not math.isfinite(minimum_cosine) or not 0 <= minimum_cosine <= 1
            or not math.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1
            or type(distinct_anchor_hypotheses) is not bool):
        raise ValueError('anonymous actor paths require explicit finite bounds')
    if (not isinstance(known_native_camera_cut_edges,tuple)
            or tuple(sorted(set(known_native_camera_cut_edges)))!=known_native_camera_cut_edges
            or any(type(i) is not int or not 1<=i<len(frames) for i in known_native_camera_cut_edges)):
        raise ValueError('camera reacquisition needs explicit sampled edges with observed native cuts')
    rows=[]
    # A hypothesis key includes its embedding. Hash each immutable measured
    # view once per call, not once for every beam parent/child/rival.
    content_keys={}
    def key_for(view):
        if view is None:return 'unmatched'
        key=id(view)
        if key not in content_keys:content_keys[key]=_content_key(view)
        return content_keys[key]
    heads_truncated=False
    beam_truncated=False
    previous=None
    for frame in frames:
        if not isinstance(frame,(list,tuple)) or len(frame) > 24:
            raise ValueError('unbounded actual head inventory')
        keys=set()
        for view in frame:
            d=view['descriptor']
            if d is None or d.frame_local_head_id is None or view['vector'] is None:
                raise ValueError('head path needs actual explicit head/appearance hypotheses')
            address=_address(view)
            if previous is not None and (address[0] <= previous[0]
                    or address[1] != previous[1] or address[3] != previous[3]):
                raise ValueError('actor frames changed source, clock or output order')
            if keys and address != _address(frame[0]):
                raise ValueError('same actor frame contains different observation addresses')
            key=key_for(view)
            if key in keys:
                raise ValueError('duplicate indistinguishable actor hypotheses')
            keys.add(key)
        if frame:
            previous=_address(frame[0])
        ordered=sorted(frame,key=lambda v:(-(v['descriptor'].face_height or 0.)
            *(v['descriptor'].face_width or 0.),key_for(v)))
        heads_truncated |= len(ordered)>maximum_heads
        rows.append(tuple(ordered[:maximum_heads])+(None,))
    beam=[dict(cost=0.,views=(),links=(),tie=())]
    visited=0
    for frame_index,choices in enumerate(rows):
        cache={}
        parents={key_for(p['views'][-1]):p['views'][-1] for p in beam if p['views']}
        if work_port is not None:
            work_port.charge('association_nodes',len(choices))
            work_port.charge('association_edges',len(parents)*len(choices))
        for key,left in parents.items():
            for right in choices:
                cache[key,key_for(right)]=_link(left,right,
                    maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
                    observed_native_camera_cut=frame_index in known_native_camera_cut_edges)
        next_beam=[]
        for parent in beam:
            for view in choices:
                link=None if not parent['views'] else cache[key_for(parent['views'][-1]),key_for(view)]
                if parent['views'] and link is None:
                    continue
                cost=(1.5 if view is None else 0.)+(0. if link is None else link['cost'])
                next_beam.append(dict(cost=parent['cost']+cost,views=parent['views']+(view,),
                    links=parent['links']+(() if link is None else (link,)),
                    tie=parent['tie']+(key_for(view),)))
                visited+=1
        next_beam.sort(key=lambda p:(p['cost'],p['tie']))
        if distinct_anchor_hypotheses:
            # Preserve histories for each pre-score first-frame actor. Two
            # nearly identical histories of one actor must not crowd every
            # other focal hypothesis out of the small candidate inventory.
            groups={}
            for p in next_beam:groups.setdefault(p['tie'][0],[]).append(p)
            beam_truncated |= any(len(g)>maximum_beam for g in groups.values())
            beam=sorted((p for g in groups.values() for p in g[:maximum_beam]),key=lambda p:(p['cost'],p['tie']))
        else:
            beam_truncated |= len(next_beam)>maximum_beam
            beam=next_beam[:maximum_beam]
    if distinct_anchor_hypotheses:
        anchor_order=tuple(key_for(v) for v in rows[0] if v is not None)
        chosen=[min((p for p in beam if p['tie'][0]==anchor),key=lambda p:(p['cost'],p['tie']))
            for anchor in anchor_order if any(p['tie'][0]==anchor for p in beam)][:maximum_paths]
    else:chosen=beam[:maximum_paths]
    results=[]
    for path in chosen:
        # A different clearly connected performer is an editorial alternative,
        # not uncertainty in this actor's temporal association. Competing
        # histories that share an observed endpoint can change the middle
        # identity and must remain ambiguous downstream.
        rivals=[p for p in beam if p['tie']!=path['tie'] and any(
            path['views'][i] is not None and p['views'][i] is not None
            and key_for(path['views'][i])==key_for(p['views'][i])
            for i in (0,len(path['views'])-1))]
        gap=None if not rivals else min(p['cost']-path['cost'] for p in rivals)
        complete=all(v is not None for v in path['views']) and all(l['linked'] for l in path['links'])
        results.append(dict(observations=path['views'],cost=path['cost'],
            frame_local_head_ids=tuple(None if v is None else v['descriptor'].frame_local_head_id for v in path['views']),
            source_local_link_hypotheses=path['links'],
            all_sampled_frames_connected=complete,
            competing_path_margin=gap,ambiguous=bool(gap is not None and gap < ambiguity_margin),
            ambiguity_scope='selected_path_endpoint_conditioned_history',
            missing_frames=sum(v is None for v in path['views']),
            hypothesis_sha256=digest(path['tie']),visited_states=visited,
            head_inventory_truncated=heads_truncated,association_beam_truncated=beam_truncated,
            complete_competing_inventory_measured=not (heads_truncated or beam_truncated),
            appearance_hypothesis_is_measured_LK_authority=False,
            continuous_camera_local_track_hypothesis=all(l.get('continuous_camera_local_track_hypothesis',False)
                for l in path['links']),
            named_actor_certified=False,motion_certified=False,quality_approved=False))
    return tuple(results)


def _fixed_anchor_chain(rows, seed, anchor_index, *, key, maximum_native_gap,
        minimum_cosine, known_native_camera_cut_edges, work_port):
    """Exact anchor-conditioned paths with last-observed state over occlusion.

    A NULL slot cannot reset identity. Both temporal directions retain the last
    actual occurrence and must pay a measured link to reacquire another head.
    Once the maximum native gap expires, that branch stays unmatched. Missing
    anatomy is never interpolated; per-frame whole-path marginals still expose
    ambiguous reacquisition without erasing independently known earlier frames.
    """
    def extend(order):
        # State = last observed row/head, plus whether THIS frame is observed.
        layers=[{(anchor_index,key(seed)):(seed,seed)}]
        forward=[{(anchor_index,key(seed)):0.}];parents=[{}];edges=[]
        for i in order:
            layer={};costs={};back={};links=[]
            for token,(last,current) in layers[-1].items():
                for candidate in rows[i]:
                    if candidate is None:
                        # Retaining the last observation distinguishes A,NULL
                        # from B,NULL. Identical NULL states must not merge them.
                        target=token;last_after=last;edge_cost=3.
                        measured=next((v for v in rows[i] if v is not None),None)
                        if last is None or measured is not None and abs(
                                _address(measured)[2]-_address(last)[2])>maximum_native_gap:
                            target=(-1,'expired');last_after=None
                    else:
                        if last is None:continue
                        a,b=(candidate,last) if i<anchor_index else (last,candidate)
                        cut=any(min(i,token[0])<c<=max(i,token[0]) for c in known_native_camera_cut_edges)
                        link=_link(a,b,maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
                            observed_native_camera_cut=cut)
                        if link is None or not link['linked']:continue
                        edge_cost=link['cost']+(1.5 if current is None else 0.)
                        target=(i,key(candidate));last_after=candidate
                    # Distinguish a currently observed occurrence from a NULL
                    # descendant of that occurrence in this same layer.
                    layer[target]=(last_after,candidate)
                    cost=forward[-1][token]+edge_cost
                    links.append((token,target,edge_cost))
                    if target not in costs or cost<costs[target]:costs[target]=cost;back[target]=token
            if work_port is not None:
                work_port.charge('association_nodes',len(layer));work_port.charge('association_edges',len(links))
            if not layer:return None
            layers.append(layer);forward.append(costs);parents.append(back);edges.append(links)
        if not order:return dict(cost=0.,views={},margins={},visited=1)
        token=min(forward[-1],key=lambda t:(forward[-1][t],t));best=forward[-1][token]
        selected=[token]
        for i in range(len(order),0,-1):selected.append(parents[i][selected[-1]])
        selected.reverse();suffix=[{t:0. for t in layer} for layer in layers]
        for i in range(len(order)-1,-1,-1):
            suffix[i]={t:math.inf for t in layers[i]}
            for source,target,cost in edges[i]:
                suffix[i][source]=min(suffix[i][source],cost+suffix[i+1][target])
        views={};margins={}
        for n,i in enumerate(order,1):
            chosen=layers[n][selected[n]][1];views[i]=chosen
            alternative=min((forward[n][t]+suffix[n][t] for t,(_,v) in layers[n].items()
                if key(v)!=key(chosen)),default=math.inf)
            margins[i]=None if not math.isfinite(alternative) else max(0.,alternative-best)
        return dict(cost=best,views=views,margins=margins,visited=sum(map(len,layers)))
    left=extend(tuple(range(anchor_index-1,-1,-1)));right=extend(tuple(range(anchor_index+1,len(rows))))
    if left is None or right is None:return None
    values={anchor_index:seed,**left['views'],**right['views']}
    margins={anchor_index:None,**left['margins'],**right['margins']}
    views=tuple(values[i] for i in range(len(rows)))
    return dict(cost=left['cost']+right['cost'],views=views,tie=tuple(key(v) for v in views),
        frame_margins=tuple(margins[i] for i in range(len(rows))),visited_states=left['visited']+right['visited']-1)


def anchored_review_actor_paths(frames, *, independent_anchor, anchor_frame_index,
        maximum_paths=2, maximum_heads=8, maximum_native_gap=12,
        minimum_cosine=.25, minimum_anchor_cosine=.25, ambiguity_margin=.025,
        maximum_beam=32, work_port=None, known_native_camera_cut_edges=(),
        maximum_frames=32, fixed_anchor_occurrence=False):
    """Condition on an independent original occurrence BEFORE the path cap.

    Extend each actual candidate anchor backward and forward. First-frame face
    area never chooses which performer gets compared with the original. Close
    anchor rivals, missing observations and budget exclusions remain explicit;
    this is an appearance hypothesis, not named identity or adoption authority.
    """
    if (type(maximum_frames) is not int or not 2<=maximum_frames<=512
            or type(fixed_anchor_occurrence) is not bool
            or not isinstance(frames,(list,tuple)) or not 2<=len(frames)<=maximum_frames
            or type(anchor_frame_index) is not int or not 0<=anchor_frame_index<len(frames)
            or independent_anchor is None or independent_anchor.get('vector') is None
            or any(type(v) is not int or not 1<=v<=cap for v,cap in
                ((maximum_paths,4),(maximum_heads,8),(maximum_beam,64),(maximum_native_gap,12)))
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1
                for v in (minimum_cosine,minimum_anchor_cosine,ambiguity_margin))
            or not isinstance(known_native_camera_cut_edges,tuple)
            or tuple(sorted(set(known_native_camera_cut_edges)))!=known_native_camera_cut_edges
            or any(type(i) is not int or not 1<=i<len(frames) for i in known_native_camera_cut_edges)):
        raise ValueError('anchor-conditioned paths require an independent measured anchor and finite bounds')
    keys={};rows=[];heads_truncated=False;previous=None
    def key(v):
        if v is None:return 'unmatched'
        if id(v) not in keys:keys[id(v)]=_content_key(v)
        return keys[id(v)]
    for frame in frames:
        if not isinstance(frame,(list,tuple)) or len(frame)>24:
            raise ValueError('unbounded actual anchor-conditioned head inventory')
        for v in frame:
            if v.get('descriptor') is None or v['descriptor'].frame_local_head_id is None or v.get('vector') is None:
                raise ValueError('anchor-conditioned paths require actual head and appearance observations')
            address=_address(v)
            if (previous is not None and (address[0]<=previous[0] or address[1]!=previous[1] or address[3]!=previous[3])
                    or address!=_address(frame[0])):
                raise ValueError('anchor-conditioned actor frames changed source, clock or output order')
        if frame:previous=_address(frame[0])
        if len({key(v) for v in frame})!=len(frame):raise ValueError('duplicate indistinguishable actor hypotheses')
        # If a bound really excludes observations, it cannot certify complete
        # competing inventory. The independent anchor, not face area, ranks it.
        ordered=sorted(frame,key=lambda v:(-(cosine(independent_anchor['vector'],v['vector']) or 0.),key(v)))
        heads_truncated |= len(ordered)>maximum_heads
        rows.append(tuple(ordered[:maximum_heads])+(None,))
    anchor_rows=[v for v in rows[anchor_frame_index] if v is not None]
    if anchor_rows and independent_anchor['row']['output_tick']!=anchor_rows[0]['row']['output_tick']:
        raise ValueError('independent original anchor borrowed another output instant')
    scores={key(v):cosine(independent_anchor['vector'],v['vector']) for v in anchor_rows}
    eligible=[v for v in anchor_rows if scores[key(v)] is not None and scores[key(v)]>=minimum_anchor_cosine]
    if fixed_anchor_occurrence:
        # Source-local continuation starts from ONE actually selected native
        # occurrence. It must not choose a different seed because its face
        # embedding scores better, nor use a cross-source reference here.
        eligible=[v for v in eligible if
            v['row']['native_observation']==independent_anchor['row']['native_observation']
            and v['descriptor'].frame_local_head_id==independent_anchor['descriptor'].frame_local_head_id]
        if len(eligible)!=1:
            raise ValueError('fixed source-local anchor occurrence is absent or ambiguous')
    eligible.sort(key=lambda v:(-scores[key(v)],key(v)))
    anchor_budget_excluded=len(eligible)>maximum_paths
    results=[]
    for seed in eligible[:maximum_paths]:
        fixed = (_fixed_anchor_chain(rows,seed,anchor_frame_index,key=key,
            maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
            known_native_camera_cut_edges=known_native_camera_cut_edges,work_port=work_port)
            if fixed_anchor_occurrence else None)
        if fixed_anchor_occurrence and fixed is None:continue
        initial=[None]*len(rows);initial[anchor_frame_index]=seed
        beam=[dict(cost=0.,views=tuple(initial),tie=(key(seed),))];visited=1;beam_truncated=False
        order=() if fixed is not None else tuple(range(anchor_frame_index-1,-1,-1))+tuple(range(anchor_frame_index+1,len(rows)))
        for i in order:
            backward=i<anchor_frame_index;neighbor=i+1 if backward else i-1
            next_beam=[]
            if work_port is not None:
                work_port.charge('association_nodes',len(rows[i]))
                work_port.charge('association_edges',len(beam)*len(rows[i]))
            for parent in beam:
                for v in rows[i]:
                    a,b=(v,parent['views'][neighbor]) if backward else (parent['views'][neighbor],v)
                    edge_index=neighbor if backward else i
                    link=_link(a,b,maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
                        observed_native_camera_cut=edge_index in known_native_camera_cut_edges)
                    if link is None:continue
                    views=list(parent['views']);views[i]=v
                    next_beam.append(dict(cost=parent['cost']+link['cost']+(1.5 if v is None else 0.),
                        views=tuple(views),tie=tuple(key(x) for x in views)))
                    visited+=1
            next_beam.sort(key=lambda p:(p['cost'],p['tie']))
            # Link costs are Markov and nonnegative: all future extensions
            # depend on this measured endpoint, not the discarded history.
            # The exact two best histories PER endpoint suffice for the best
            # complete path and its nearest competing history. This is not a
            # global heuristic beam hiding relevant alternatives. At the left
            # -> right switch all histories share the independently fixed seed.
            endpoint_groups={}
            for p in next_beam:endpoint_groups.setdefault(key(p['views'][i]),[]).append(p)
            exact=sorted((p for group in endpoint_groups.values() for p in group[:2]),key=lambda p:(p['cost'],p['tie']))
            beam_truncated |= len(exact)>maximum_beam;beam=exact[:maximum_beam]
        if not beam:continue
        chosen=fixed if fixed is not None else beam[0]
        rivals=[p for p in beam[1:] if p['tie']!=chosen['tie']]
        path_margin=(min((v for v in fixed['frame_margins'] if v is not None),default=None)
            if fixed is not None else None if not rivals else min(p['cost']-chosen['cost'] for p in rivals))
        if fixed is not None:visited=fixed['visited_states']
        anchor_rivals=[] if fixed_anchor_occurrence else [s for k,s in scores.items() if k!=key(seed) and s is not None]
        anchor_margin=None if not anchor_rivals else scores[key(seed)]-max(anchor_rivals)
        links=tuple(_link(a,b,maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
            observed_native_camera_cut=i in known_native_camera_cut_edges)
            for i,(a,b) in enumerate(zip(chosen['views'],chosen['views'][1:]),1))
        complete=all(v is not None for v in chosen['views']) and all(l is not None and l['linked'] for l in links)
        ambiguous=bool(independent_anchor.get('reacquisition_ambiguous',False)
            or anchor_margin is not None and anchor_margin<ambiguity_margin
            or path_margin is not None and path_margin<ambiguity_margin)
        results.append(dict(observations=chosen['views'],cost=chosen['cost'],
            frame_local_head_ids=tuple(None if v is None else v['descriptor'].frame_local_head_id for v in chosen['views']),
            source_local_link_hypotheses=links,all_sampled_frames_connected=complete,
            competing_path_margin=path_margin,anchor_rival_margin=anchor_margin,anchor_cosine=scores[key(seed)],
            independent_anchor_identity=digest((independent_anchor['row']['native_observation'],
                independent_anchor['descriptor'].frame_local_head_id,list(independent_anchor['vector']))),
            independent_anchor_frame_index=anchor_frame_index,ambiguous=ambiguous,
            fixed_anchor_occurrence=fixed_anchor_occurrence,
            frame_competing_margins=None if fixed is None else fixed['frame_margins'],
            ambiguous_frame_indices=None if fixed is None else tuple(i for i,v in enumerate(fixed['frame_margins'])
                if independent_anchor.get('reacquisition_ambiguous',False) or v is not None and v<ambiguity_margin),
            maximum_frames=maximum_frames,
            ambiguity_scope='independent_original_anchor_and_bidirectional_path_history',
            missing_frames=sum(v is None for v in chosen['views']),hypothesis_sha256=digest(chosen['tie']),
            visited_states=visited,head_inventory_truncated=heads_truncated,association_beam_truncated=beam_truncated,
            temporal_optimizer=('exact_fixed_anchor_last_observed_gap_min_marginals_v2' if fixed is not None
                else 'exact_top_two_histories_per_conditioned_endpoint_v1'),
            anchor_path_budget_excluded=anchor_budget_excluded,
            complete_competing_inventory_measured=not(heads_truncated or beam_truncated or anchor_budget_excluded),
            budget_exclusion_is_actor_unsuitability=False,appearance_hypothesis_is_measured_LK_authority=False,
            named_actor_certified=False,motion_certified=False,quality_approved=False))
    return tuple(results)


def connected_source_actor_views(frames, *, ticks, native_units, anchor, maximum_frames=240,
        work_port=None):
    """Follow one selected occurrence through its physical camera run.

    Returns actual observed views only. Missing frames and other native camera
    runs stay unmatched; callers may choose a new shot there. The same path
    supplies action, camera support and rendered-frame inspection.
    """
    if (not isinstance(ticks,tuple) or ticks!=tuple(sorted(set(ticks)))
            or len(frames)!=len(ticks) or set(native_units)!=set(ticks)
            or type(maximum_frames) is not int or not 2<=maximum_frames<=512
            or not 1<=len(ticks)<=maximum_frames):
        raise ValueError('connected actor needs its bounded ordered physical request inventory')
    if anchor is None:return {t:None for t in ticks},dict(state='ANCHOR_UNOBSERVED')
    at=anchor['row']['output_tick']
    if at not in ticks:raise ValueError('selected anchor is outside the actor path request')
    unit=native_units[at]
    indices=[i for i,t in enumerate(ticks) if native_units[t]==unit]
    if indices!=list(range(min(indices),max(indices)+1)):
        raise ValueError('physical camera run reappears non-contiguously')
    selected_ticks=tuple(ticks[i] for i in indices)
    for i in indices:
        for v in frames[i]:
            raw=v['row']['native_observation']
            address=(raw['request_source_index'],raw['source_clock_identity'],
                digest(physical_review_file_identity(v)),v.get('native_camera_unit_index'))
            if v['row']['output_tick']!=ticks[i] or address!=tuple(unit):
                raise ValueError('actor path view changed its physical source file clock or run')
    selected={t:None for t in ticks}
    if len(indices)==1:
        candidates=[v for v in frames[indices[0]] if v['row']['native_observation']==anchor['row']['native_observation']
            and v['descriptor'].frame_local_head_id==anchor['descriptor'].frame_local_head_id]
        if len(candidates)!=1:raise ValueError('fixed source-local anchor occurrence is absent or ambiguous')
        selected[at]=candidates[0]
        return selected,dict(state='SINGLE_OBSERVED_OCCURRENCE',connected_motion_observed=False)
    path=anchored_review_actor_paths([frames[i] for i in indices],independent_anchor=anchor,
        anchor_frame_index=selected_ticks.index(at),maximum_paths=1,maximum_heads=8,
        maximum_frames=maximum_frames,fixed_anchor_occurrence=True,work_port=work_port)[0]
    ambiguous_indices=set(path['ambiguous_frame_indices'])
    for i,(t,v) in enumerate(zip(selected_ticks,path['observations'])):
        if v is not None:
            selected[t]=dict(v,reacquisition_ambiguous=bool(v.get('reacquisition_ambiguous',False) or i in ambiguous_indices),
                source_actor_path_identity=path['hypothesis_sha256'])
    receipt={k:v for k,v in path.items() if k!='observations'}
    receipt.update(state='CONNECTED_SOURCE_LOCAL_HYPOTHESIS',requested_ticks=list(ticks),
        physical_run_ticks=list(selected_ticks),other_physical_run_ticks=[t for t in ticks if t not in selected_ticks])
    return selected,receipt
