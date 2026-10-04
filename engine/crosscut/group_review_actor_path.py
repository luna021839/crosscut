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


def _content_key(view):
    if view is None:
        return 'unmatched'
    d = view['descriptor']
    # Detector head numbers are frame-local, not cross-frame identity or ties.
    return digest(dict(vector=view['vector'], center=d.head_center,
        height=d.face_height, width=d.face_width, shape=d.pose_shape))


def _address(view):
    row = view['row']
    raw = row['native_observation']
    return (row['output_tick'], raw['request_source_index'], raw['native_frame_index'],
        raw['source_clock_identity'], row['projection']['camera_sha256'])


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
    if left['row']['native_observation'].get('source_file_identity') != right['row']['native_observation'].get('source_file_identity'):
        raise ValueError('source-local actor path changed source file identity')
    sim = cosine(left['vector'],right['vector'])
    if sim is None or sim < minimum_cosine:
        return None  # The real unmatched alternative remains in the graph.
    shape = shape_cost(left['descriptor'],right['descriptor'])
    reacquired=a[4]!=b[4]
    return dict(cost=2.*max(0.,1.-sim)+(.25 if shape is None else .15*shape)+(.25 if reacquired else 0.),
        state='appearance_reacquisition_at_observed_native_cut' if reacquired else 'appearance_link_hypothesis',
        cosine=sim,linked=True,continuous_camera_local_track_hypothesis=not reacquired,
        observed_native_camera_cut=observed_native_camera_cut)


def anonymous_review_actor_paths(frames, *, maximum_paths=2, maximum_heads=4,
        maximum_native_gap=12, minimum_cosine=.25, ambiguity_margin=.025,
        maximum_beam=32, work_port=None,known_native_camera_cut_edges=()):
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
            or not math.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1):
        raise ValueError('anonymous actor paths require explicit finite bounds')
    if (not isinstance(known_native_camera_cut_edges,tuple)
            or tuple(sorted(set(known_native_camera_cut_edges)))!=known_native_camera_cut_edges
            or any(type(i) is not int or not 1<=i<len(frames) for i in known_native_camera_cut_edges)):
        raise ValueError('camera reacquisition needs explicit sampled edges with observed native cuts')
    rows=[]
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
            key=_content_key(view)
            if key in keys:
                raise ValueError('duplicate indistinguishable actor hypotheses')
            keys.add(key)
        if frame:
            previous=_address(frame[0])
        ordered=sorted(frame,key=lambda v:(-(v['descriptor'].face_height or 0.)
            *(v['descriptor'].face_width or 0.),_content_key(v)))
        heads_truncated |= len(ordered)>maximum_heads
        rows.append(tuple(ordered[:maximum_heads])+(None,))
    beam=[dict(cost=0.,views=(),links=(),tie=())]
    visited=0
    for frame_index,choices in enumerate(rows):
        cache={}
        parents={_content_key(p['views'][-1]):p['views'][-1] for p in beam if p['views']}
        if work_port is not None:
            work_port.charge('association_nodes',len(choices))
            work_port.charge('association_edges',len(parents)*len(choices))
        for key,left in parents.items():
            for right in choices:
                cache[key,_content_key(right)]=_link(left,right,
                    maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
                    observed_native_camera_cut=frame_index in known_native_camera_cut_edges)
        next_beam=[]
        for parent in beam:
            for view in choices:
                link=None if not parent['views'] else cache[_content_key(parent['views'][-1]),_content_key(view)]
                if parent['views'] and link is None:
                    continue
                cost=(1.5 if view is None else 0.)+(0. if link is None else link['cost'])
                next_beam.append(dict(cost=parent['cost']+cost,views=parent['views']+(view,),
                    links=parent['links']+(() if link is None else (link,)),
                    tie=parent['tie']+(_content_key(view),)))
                visited+=1
        next_beam.sort(key=lambda p:(p['cost'],p['tie']))
        beam_truncated |= len(next_beam)>maximum_beam
        beam=next_beam[:maximum_beam]
    chosen=beam[:maximum_paths]
    results=[]
    for path in chosen:
        gap=None if len(beam)<2 else beam[1]['cost']-beam[0]['cost']
        complete=all(v is not None for v in path['views']) and all(l['linked'] for l in path['links'])
        results.append(dict(observations=path['views'],cost=path['cost'],
            frame_local_head_ids=tuple(None if v is None else v['descriptor'].frame_local_head_id for v in path['views']),
            source_local_link_hypotheses=path['links'],
            all_sampled_frames_connected=complete,
            competing_path_margin=gap,ambiguous=gap is None or gap < ambiguity_margin,
            missing_frames=sum(v is None for v in path['views']),
            hypothesis_sha256=digest(path['tie']),visited_states=visited,
            head_inventory_truncated=heads_truncated,association_beam_truncated=beam_truncated,
            complete_competing_inventory_measured=not (heads_truncated or beam_truncated),
            appearance_hypothesis_is_measured_LK_authority=False,
            continuous_camera_local_track_hypothesis=all(l.get('continuous_camera_local_track_hypothesis',False)
                for l in path['links']),
            named_actor_certified=False,motion_certified=False,quality_approved=False))
    return tuple(results)
