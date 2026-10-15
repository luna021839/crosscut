"""Prepare pair-closed raw person evidence BEFORE the played-port DP.

Only a bounded subset of camera-conditioned UNKNOWN comparisons is observed.
Raw frame attempts, closed endpoint pairs and viewport-valid extents are NOT
interchangeable coverage claims. No same-person or aesthetic labels are made.
"""
from time import monotonic
import numpy as np
from crosscut.native_person_onnx import PINNED_YOLO11L_ONNX_SHA256
from .person_rectangle_bank import observe_person_rectangles
from .person_pair_requests import reserve_complete_pairs
from .physical_display_observations import focal_metadata


def _key(port):
    observation=port.observation
    return int(port.source),int(observation['native_indices'][1 if observation['side']==0 else 0])


def _face_known(port):
    head=port.observation['head']
    return bool(head[2]>.019 and head[3]>=.75)


def comparison_pairs(entries,exits,native_ports,bound,dense):
    """Native UNKNOWN flanks first, then interleaved canonical-node requests.

    Source count and timestamps never become editorial bonuses. Interleaving
    only spreads the finite observation allowance; it does not force cuts.
    """
    pairs=[];seen=set()
    def add(pair):
        if pair not in seen:seen.add(pair);pairs.append(pair)
    if not len(native_ports)==len(bound)==len(dense):raise ValueError('person native domains differ')
    native_pairs=[]
    for s,(bank,record) in enumerate(zip(native_ports,bound)):
        if record['source']!=s:raise ValueError('person source binding order differs')
        for i,tick in enumerate(bank['frames']):
            rows=[r for r in record['native_event_rows'] if r['output_frame']==int(tick)]
            actual={tuple(map(int,r['exposed_native_ordinals'])) for r in rows}
            if len(actual)!=1:raise ValueError('one actual exposed native pair per registry tick required')
            a,b=next(iter(actual))
            if not 0<=a<b<record['frame_count']:raise ValueError('person native pair escapes the actual source clock')
            cameras=np.asarray([path[[int(tick)-1,int(tick)]] for path in dense[s]])
            _,index,_,_=focal_metadata(bank['heads'][i:i+1],cameras)
            if not bank['observed'][i] or (index<0).any():
                native_pairs.append(((s,a),(s,b)))
    for pair in native_pairs:add(pair)
    by_node=[]
    for incoming,outgoing in zip(entries[1:-1],exits[1:-1]):
        physical={}
        for a in outgoing:
            for b in incoming:
                if a.source==b.source or (_face_known(a) and _face_known(b)):continue
                pair=(_key(a),_key(b));physical.setdefault(pair,None)
        by_node.append(sorted(physical))
    # One request per node per pass is observation scheduling, not an EDL quota.
    for index in range(max(map(len,by_node),default=0)):
        for values in by_node:
            if index<len(values):add(values[index])
    return pairs,len(set(native_pairs))


def bind_rectangles(catalogues,entries,exits,native_ports,bound,rectangles):
    """Bind actual ordinals, never substitute a nearby face/body observation."""
    for catalogue,bank in zip(catalogues,rectangles):catalogue.bank['person_rectangles_by_native']=bank
    for node in (*entries,*exits):
        for port in node:
            s,f=_key(port)
            port.observation['person_rectangles']=rectangles[s].get(f)
            port.observation['person_native_ordinal']=f
    for s,(bank,record) in enumerate(zip(native_ports,bound)):
        mapping={};ordinals=[];attempted=[];measured=[]
        for tick in bank['frames']:
            keys={tuple(map(int,r['exposed_native_ordinals'])) for r in record['native_event_rows']
                  if r['output_frame']==int(tick)}
            if len(keys)!=1:raise ValueError('person native observation has no unambiguous physical pair')
            a,b=next(iter(keys));left,right=rectangles[s].get(a),rectangles[s].get(b)
            mapping[int(tick)]=(left,right);ordinals.append((a,b))
            attempted.append(left is not None and right is not None)
            measured.append(left is not None and right is not None and len(left)>0 and len(right)>0)
        bank.update(person_rectangles=mapping,person_native_ordinals=np.asarray(ordinals,int).reshape(-1,2),
            person_native_pair_attempted=np.asarray(attempted,bool),
            person_native_pair_measured=np.asarray(measured,bool))


def prepare_pair_observer(*,sources,bound,cache,model,maximum_new_frames=256,maximum_seconds=120.):
    """Factory for the explicit experimental pre-selection observation phase."""
    if type(maximum_new_frames) is not int or not 0<=maximum_new_frames<=256:
        raise ValueError('person pair preparation allows at most256new forwards')
    def prepare(*,nodes,catalogues,entries,exits,dense,native_ports,check):
        started=monotonic();check()
        pairs,native_count=comparison_pairs(entries,exits,native_ports,bound,dense)
        requests=[{} for _ in sources]
        for pair in pairs:
            for s,f in pair:requests[s][f]=0
        cached,warm=observe_person_rectangles(sources=sources,bound=bound,requests=requests,
            cache=cache,model=model,check=check,maximum_new_frames=0,maximum_seconds=maximum_seconds)
        available={(s,f) for s,rows in enumerate(cached) for f in rows}
        reservation=reserve_complete_pairs(pairs,available,maximum_new_frames=maximum_new_frames)
        new_requests=[{} for _ in sources]
        for s,f in reservation['new_frames']:new_requests[s][f]=0
        remaining=maximum_seconds-(monotonic()-started)
        cold=[]
        if reservation['new_frame_count'] and remaining>=1.:
            new,cold=observe_person_rectangles(sources=sources,bound=bound,requests=new_requests,
                cache=cache,model=model,check=check,maximum_new_frames=maximum_new_frames,maximum_seconds=remaining)
            for old,extra in zip(cached,new):old.update(extra)
        bind_rectangles(catalogues,entries,exits,native_ports,bound,cached)
        complete=sum(all(f in cached[s] for s,f in pair) for pair in pairs)
        nonempty=sum(all(f in cached[s] and len(cached[s][f])>0 for s,f in pair) for pair in pairs)
        check()
        return dict(contract='camera_conditioned_pair_closed_person_observations_v1',
            targeted_physical_pairs=len(pairs),targeted_unknown_native_pairs=native_count,
            requested_physical_endpoint_frames=sum(map(len,requests)),
            reused_physical_endpoint_frames=sum(r['reused_frames'] for r in warm),
            new_person_forwards=sum(r['new_person_forwards'] for r in cold),
            new_face_pose_forwards=0,requested_forward_limit=maximum_new_frames,
            maximum_observation_seconds=maximum_seconds,elapsed_seconds=monotonic()-started,
            closed_endpoint_attempt_pairs=complete,nonempty_raw_rectangle_pairs=nonempty,
            actual_viewport_valid_pair_count_not_claimed_by_raw_counts=True,
            all_targeted_pairs_attempted=complete==len(pairs),
            reservation=reservation,warm_cache_receipts=warm,new_observation_receipts=cold,
            model_sha256=PINNED_YOLO11L_ONNX_SHA256,raw_rectangles_projected_only_by_consumer=True,
            original_face_body_caches_invalidated=False,person_NN_in_DP=False,
            actor_identity_or_professional_quality_certified=False,
            source_hash_and_model_hash_IO_still_performed=True)
    return prepare
