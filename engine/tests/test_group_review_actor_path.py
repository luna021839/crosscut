from copy import deepcopy
import pytest
from crosscut.group_review_actor_path import anonymous_review_actor_paths,anchored_review_actor_paths
from crosscut.group_review_camera_repair import FramingDescriptor


def view(tick,actor,head_id=0,*,native=None,camera='camera-a'):
    vector=[0.]*128
    vector[actor]=1.
    native=tick if native is None else native
    d=FramingDescriptor((.4+actor*.1,.35),.3,.15,None,head_id,False)
    return dict(descriptor=d,vector=vector,near=1.,appearance=None,row=dict(output_tick=tick,
        native_observation=dict(request_source_index=2,native_frame_index=native,source_clock_identity='clock'),
        projection=dict(camera_sha256=camera)))


def test_one_internal_sequence_cannot_independently_choose_a_entry_and_b_exit():
    paths=anonymous_review_actor_paths([[view(0,0)],[view(2,0)],[view(4,1)]])
    assert not any(p['all_sampled_frames_connected'] for p in paths)
    assert all(p['missing_frames'] for p in paths)
    assert all(not p['named_actor_certified'] for p in paths)


def test_competing_people_keep_distinct_connected_sequences():
    frames=[[view(t,0,17),view(t,1,5)] for t in (0,2,4)]
    paths=anonymous_review_actor_paths(frames)
    assert len(paths)==2 and all(p['all_sampled_frames_connected'] for p in paths)
    assert all(not p['ambiguous'] for p in paths)
    assert {p['frame_local_head_ids'] for p in paths}=={(17,17,17),(5,5,5)}


def test_shared_endpoints_do_not_hide_ambiguous_middle_actor_history():
    middle_a=view(2,0,7);middle_b=view(2,1,8)
    middle_b['vector']=list(middle_a['vector'])
    from dataclasses import replace
    middle_b['descriptor']=replace(middle_b['descriptor'],head_center=middle_a['descriptor'].head_center,
        face_width=.15001)
    paths=anonymous_review_actor_paths([[view(0,0,7)],[middle_a,middle_b],[view(4,0,7)]])
    assert all(p['ambiguous'] for p in paths)
    assert all(p['competing_path_margin']==pytest.approx(0.) for p in paths)


def test_local_spatial_hypothesis_does_not_jump_to_a_more_similar_background_face():
    from dataclasses import replace
    left=view(0,0,7);near=view(2,0,8);far=view(2,0,9)
    near['vector']=[.98,.2]+[0.]*126
    far['descriptor']=replace(far['descriptor'],head_center=(.95,.85))
    p=anonymous_review_actor_paths([[left],[near,far]])[0]
    assert p['frame_local_head_ids']==(7,8)


def test_two_frozen_anchor_actors_are_not_two_histories_of_one_actor():
    frames=[[view(t,0,7),view(t,1,9)] for t in (0,2,4)]
    paths=anonymous_review_actor_paths(frames,distinct_anchor_hypotheses=True)
    assert {p['frame_local_head_ids'][0] for p in paths}=={7,9}
    assert all(not p['ambiguous'] for p in paths)


def test_beam_does_not_rehash_embeddings_for_every_parent_and_rival(monkeypatch):
    import crosscut.group_review_actor_path as module
    original=module._content_key;calls=[]
    def counted(v):
        calls.append(id(v));return original(v)
    monkeypatch.setattr(module,'_content_key',counted)
    frames=[[view(t,i,i) for i in range(4)] for t in (0,2,4,6)]
    paths=anonymous_review_actor_paths(frames,maximum_heads=4,distinct_anchor_hypotheses=True)
    assert len(calls)==16 and len(set(calls))==16
    assert paths and all(p['all_sampled_frames_connected'] for p in paths)


def test_detector_head_id_permutation_does_not_change_content_or_rank():
    frames=[[view(t,0,17),view(t,1,5)] for t in (0,2,4)]
    renamed=[[view(t,0,100+t),view(t,1,200+t)] for t in (0,2,4)]
    a,b=anonymous_review_actor_paths(frames),anonymous_review_actor_paths(renamed)
    assert [(p['hypothesis_sha256'],p['cost']) for p in a]==[(p['hypothesis_sha256'],p['cost']) for p in b]


def test_missing_exit_is_not_largest_face_reacquisition():
    paths=anonymous_review_actor_paths([[view(0,0)],[view(2,0)],[]])
    assert paths[0]['frame_local_head_ids'][-1] is None
    assert not paths[0]['all_sampled_frames_connected']


def test_large_gap_or_native_cut_is_an_unmeasured_bridge_not_linked_actor():
    for frames in ([[view(0,0)],[view(13,0)]],
                   [[view(0,0)],[view(2,0,camera='camera-b')]]):
        p=anonymous_review_actor_paths(frames)[0]
        assert not p['all_sampled_frames_connected']
        assert p['source_local_link_hypotheses'][0]['state']=='unmeasured_camera_or_gap_bridge'


def test_same_native_alias_preserves_output_time_but_cannot_change_raw_measurement():
    a,b=view(0,0,native=8),view(1,0,native=8)
    assert anonymous_review_actor_paths([[a],[b]])[0]['all_sampled_frames_connected']
    b['row']['native_observation']['changed']='different detector result'
    with pytest.raises(ValueError,match='conflicting'):
        anonymous_review_actor_paths([[a],[b]])


def test_other_source_or_clock_cannot_be_borrowed():
    for key,value in (('request_source_index',3),('source_clock_identity','other')):
        b=view(2,0)
        b['row']['native_observation'][key]=value
        with pytest.raises(ValueError,match='source, clock'):
            anonymous_review_actor_paths([[view(0,0)],[b]])


def test_frame_head_bound_is_explicit_without_changing_complete_inventory():
    frames=[[view(t,i,i) for i in range(8)] for t in (0,2)]
    original=deepcopy(frames)
    paths=anonymous_review_actor_paths(frames,maximum_heads=2)
    assert frames==original and all(p['visited_states']<=32 for p in paths)
    assert all(p['head_inventory_truncated'] and not p['complete_competing_inventory_measured'] for p in paths)


def test_one_native_alias_cannot_link_two_different_occurrences_even_with_similar_faces():
    a,b=view(0,0,7,native=8),view(1,0,9,native=8)
    b['vector']=[.8,.6]+[0.]*126
    paths=anonymous_review_actor_paths([[a],[b]])
    assert not any(p['all_sampled_frames_connected'] for p in paths)
    assert all(p['missing_frames'] for p in paths)


def test_source_file_identity_is_not_hidden_by_equal_clock_and_source_number():
    a,b=view(0,0),view(1,0)
    a['row']['native_observation']['source_file_identity']=[1,2,3,4,5]
    b['row']['native_observation']['source_file_identity']=[1,2,3,4,6]
    with pytest.raises(ValueError,match='file identity'):
        anonymous_review_actor_paths([[a],[b]])


def test_observed_native_cut_can_reacquire_appearance_without_claiming_a_continuous_track():
    frames=[[view(0,0)],[view(2,0,camera='camera-b')],[view(4,0,camera='camera-b')]]
    p=anonymous_review_actor_paths(frames,known_native_camera_cut_edges=(1,))[0]
    assert p['all_sampled_frames_connected']
    assert not p['continuous_camera_local_track_hypothesis']
    assert not p['appearance_hypothesis_is_measured_LK_authority']
    assert p['source_local_link_hypotheses'][0]['state']=='appearance_reacquisition_at_observed_native_cut'


def test_a_known_native_cut_does_not_turn_another_person_or_large_gap_into_a_match():
    for frames in ([[view(0,0)],[view(2,1,camera='camera-b')]],
                   [[view(0,0)],[view(13,0,camera='camera-b')]]):
        assert not any(p['all_sampled_frames_connected'] for p in
            anonymous_review_actor_paths(frames,known_native_camera_cut_edges=(1,)))


def test_independent_anchor_recovers_fifth_small_face_before_path_budget():
    from dataclasses import replace
    frames=[[view(t,i,10+i) for i in range(5)] for t in (0,2,4)]
    for frame in frames:frame[4]['descriptor']=replace(frame[4]['descriptor'],face_height=.04,face_width=.02)
    old=anonymous_review_actor_paths(frames,maximum_paths=4,maximum_heads=8,distinct_anchor_hypotheses=True)
    assert not any(p['frame_local_head_ids']==(14,14,14) for p in old)
    anchor=view(2,4,99);anchor['row']['native_observation']['request_source_index']=7
    paths=anchored_review_actor_paths(frames,independent_anchor=anchor,anchor_frame_index=1)
    assert paths[0]['frame_local_head_ids']==(14,14,14)
    assert paths[0]['all_sampled_frames_connected'] and not paths[0]['ambiguous']
    assert not paths[0]['named_actor_certified']


def test_anchor_conditioning_preserves_close_rivals_not_a_false_match():
    frames=[[view(t,0,7),view(t,1,8)] for t in (0,2,4)]
    for frame in frames:frame[1]['vector']=list(frame[0]['vector'])
    paths=anchored_review_actor_paths(frames,independent_anchor=view(2,0),anchor_frame_index=1)
    assert len(paths)==2 and all(p['ambiguous'] for p in paths)
    assert all(p['anchor_rival_margin']==pytest.approx(0.) for p in paths)


def test_actual_interval_outside_missing_data_does_not_reject_anchored_path():
    full=[[],[view(1,0)],[view(2,0)],[view(3,0)],[]]
    paths=anchored_review_actor_paths(full[1:4],independent_anchor=view(2,0),anchor_frame_index=1)
    assert paths[0]['all_sampled_frames_connected']
    paths=anchored_review_actor_paths(full,independent_anchor=view(2,0),anchor_frame_index=2)
    assert not paths[0]['all_sampled_frames_connected'] and paths[0]['missing_frames']==2


def test_missing_inside_anchor_scope_and_unobserved_native_cut_stay_unresolved():
    for frames in ([[view(0,0)],[view(1,0)],[]],
                   [[view(0,0)],[view(1,0)],[view(2,0,camera='cut')]]):
        p=anchored_review_actor_paths(frames,independent_anchor=view(1,0),anchor_frame_index=1)[0]
        assert not p['all_sampled_frames_connected']


def test_anchor_budget_loss_is_explicit_and_does_not_certify_inventory():
    frames=[[view(t,i,i) for i in range(3)] for t in (0,2,4)]
    for frame in frames:
        for v in frame:v['vector']=[1.]+[0.]*127
    p=anchored_review_actor_paths(frames,independent_anchor=view(2,0),anchor_frame_index=1,maximum_paths=1)[0]
    assert p['anchor_path_budget_excluded'] and not p['complete_competing_inventory_measured']
    assert p['ambiguous'] and not p['budget_exclusion_is_actor_unsuitability']


def test_anchor_from_another_instant_is_not_borrowed():
    with pytest.raises(ValueError,match='another output instant'):
        anchored_review_actor_paths([[view(0,0)],[view(2,0)]],independent_anchor=view(3,0),anchor_frame_index=0)


def test_exact_anchor_endpoint_dynamic_program_matches_exhaustive_history_costs():
    from itertools import product
    from crosscut.group_review_actor_path import _link
    from dataclasses import replace
    frames=[[view(t,0,7),view(t,1,8)] for t in range(5)]
    for frame in frames:
        frame[1]['vector']=[.95,.05]+[0.]*126
        frame[1]['descriptor']=replace(frame[1]['descriptor'],head_center=(.405,.35))
    seed=frames[2][0];enumerated=[]
    for choices in product(*(frame+[None] if i!=2 else [seed] for i,frame in enumerate(frames))):
        links=[_link(a,b,maximum_native_gap=12,minimum_cosine=.25) for a,b in zip(choices,choices[1:])]
        if any(l is None for l in links):continue
        enumerated.append(sum(l['cost'] for l in links)+1.5*sum(v is None for v in choices))
    enumerated.sort()
    paths=anchored_review_actor_paths(frames,independent_anchor=view(2,0),anchor_frame_index=2)
    selected=next(p for p in paths if p['frame_local_head_ids'][2]==7)
    assert selected['cost']==pytest.approx(enumerated[0])
    assert selected['competing_path_margin']==pytest.approx(enumerated[1]-enumerated[0])
    assert not selected['association_beam_truncated']


def test_long_complete_anchor_path_does_not_lose_inventory_to_unmatched_history_explosion():
    frames=[[view(t,0,7),view(t,1,8)] for t in range(25)]
    p=anchored_review_actor_paths(frames,independent_anchor=view(12,0),anchor_frame_index=12)[0]
    assert p['all_sampled_frames_connected'] and p['complete_competing_inventory_measured']
    assert p['visited_states']<2000


def test_fixed_native_occurrence_extends_full_shot_without_framewise_embedding_winner():
    from dataclasses import replace
    from crosscut.group_review_actor_path import connected_source_actor_views
    from crosscut.group_performer_family_owner import digest
    ticks=tuple(range(80));frames=[]
    for t in ticks:
        near=view(t,0,2*t);far=view(t,0,2*t+1)
        near['descriptor']=replace(near['descriptor'],head_center=(.35+t*.002,.40))
        far['descriptor']=replace(far['descriptor'],head_center=(.83-t*.001,.72))
        near['vector']=[.98,.2]+[0.]*126
        if t==40:near['vector']=[1.]+[0.]*127
        far['vector']=[1.]+[0.]*127
        for v in (near,far):v['native_camera_unit_index']=0
        frames.append([near,far])
    units={t:(2,'clock',digest(None),0) for t in ticks}
    selected,receipt=connected_source_actor_views(frames,ticks=ticks,native_units=units,anchor=frames[40][0])
    assert [selected[t]['descriptor'].frame_local_head_id for t in ticks]==[2*t for t in ticks]
    assert receipt['fixed_anchor_occurrence'] and receipt['all_sampled_frames_connected']
    assert receipt['visited_states']<8000 and not receipt['named_actor_certified']


def test_connected_actor_keeps_missing_slots_and_other_camera_runs_unmatched():
    from crosscut.group_review_actor_path import connected_source_actor_views
    from crosscut.group_performer_family_owner import digest
    ticks=tuple(range(6));frames=[[view(t,0,t)] for t in ticks];frames[3]=[]
    for t,frame in enumerate(frames):
        for v in frame:v['native_camera_unit_index']=int(t>=4)
    units={t:(2,'clock',digest(None),int(t>=4)) for t in ticks}
    selected,receipt=connected_source_actor_views(frames,ticks=ticks,native_units=units,anchor=frames[1][0])
    assert selected[1] is not None
    assert all(selected[t] is None for t in (3,4,5))
    assert receipt['other_physical_run_ticks']==[4,5]
    assert not receipt['all_sampled_frames_connected']


def test_fixed_occurrence_cannot_borrow_a_cross_source_anchor():
    anchor=view(1,0);anchor['row']['native_observation']['request_source_index']=99
    with pytest.raises(ValueError,match='anchor occurrence'):
        anchored_review_actor_paths([[view(t,0)] for t in range(3)],independent_anchor=anchor,
            anchor_frame_index=1,fixed_anchor_occurrence=True)


def test_frame_ambiguity_does_not_erase_independent_known_anchor_and_close_neighbors():
    from dataclasses import replace
    from crosscut.group_review_actor_path import connected_source_actor_views
    from crosscut.group_performer_family_owner import digest
    ticks=tuple(range(8));frames=[[view(t,0,t)] for t in ticks]
    # Two equally good reacquisitions after an unobserved frame. Neither is
    # evidence that the independently fixed first half suddenly became unknown.
    frames[5]=[]
    for t in (6,7):
        other=view(t,0,t+100)
        other['descriptor']=replace(other['descriptor'],head_center=(.401,.35))
        frames[t].append(other)
    for frame in frames:
        for v in frame:v['native_camera_unit_index']=0
    units={t:(2,'clock',digest(None),0) for t in ticks}
    selected,receipt=connected_source_actor_views(frames,ticks=ticks,native_units=units,anchor=frames[2][0])
    assert receipt['ambiguous']
    assert not any(selected[t]['reacquisition_ambiguous'] for t in range(5))
    assert selected[5] is None
    assert all(selected[t]['reacquisition_ambiguous'] for t in (6,7))
    assert set(receipt['ambiguous_frame_indices'])=={6,7}


def test_same_anchor_instant_is_an_occurrence_not_fabricated_motion():
    from crosscut.group_review_actor_path import connected_source_actor_views
    from crosscut.group_performer_family_owner import digest
    anchor=view(8,0,7);anchor['native_camera_unit_index']=3
    selected,receipt=connected_source_actor_views([[anchor]],ticks=(8,),
        native_units={8:(2,'clock',digest(None),3)},anchor=anchor)
    assert selected[8] is anchor and receipt['state']=='SINGLE_OBSERVED_OCCURRENCE'
    assert not receipt['connected_motion_observed']


def test_fixed_min_marginals_equal_exhaustive_whole_paths():
    from itertools import product
    from dataclasses import replace
    from crosscut.group_review_actor_path import _link
    frames=[[view(t,0,7),view(t,1,8)] for t in range(4)]
    for frame in frames:
        frame[1]['vector']=[.95,.05]+[0.]*126
        frame[1]['descriptor']=replace(frame[1]['descriptor'],head_center=(.43,.35))
    seed=frames[1][0];possibilities=[]
    for sequence in product(*(frame+[None] if i!=1 else [seed] for i,frame in enumerate(frames))):
        links=[_link(a,b,maximum_native_gap=12,minimum_cosine=.25) for a,b in zip(sequence,sequence[1:])]
        if any(link is None for link in links):continue
        possibilities.append((sum(link['cost'] for link in links)+1.5*sum(v is None for v in sequence),sequence))
    path=anchored_review_actor_paths(frames,independent_anchor=seed,anchor_frame_index=1,
        fixed_anchor_occurrence=True)[0]
    assert path['cost']==pytest.approx(min(cost for cost,_ in possibilities))
    for i,selected in enumerate(path['observations']):
        alternate=min((cost for cost,sequence in possibilities if sequence[i] is not selected),default=None)
        margin=path['frame_competing_margins'][i]
        assert margin is None if alternate is None else margin==pytest.approx(alternate-path['cost'])


def test_native_run_link_ignores_editorial_query_boundaries_and_uses_native_head_geometry():
    from dataclasses import replace
    from crosscut.group_review_actor_path import _link
    a,b=view(0,0),view(1,0,camera='new-editorial-crop')
    for v in (a,b):
        v['native_camera_unit_index']=4
        v['row']['native_observation']['heads']=[dict(observation_id=v['descriptor'].frame_local_head_id,
            box_xywh=(.35,.25,.1,.2))]
    b['descriptor']=replace(b['descriptor'],head_center=(.9,.8),face_height=.4)
    linked=_link(a,b,maximum_native_gap=12,minimum_cosine=.25)
    assert linked['linked'] and linked['measured_camera_local_spatial_cost']==pytest.approx(0.)


@pytest.mark.parametrize('anchor_index',[0,5])
def test_fixed_anchor_cannot_reset_identity_through_one_missing_frame(anchor_index):
    frames=[[view(0,0)],[],*[ [view(t,1)] for t in range(2,6)]]
    if anchor_index==5:frames=list(reversed(frames))
    for t,frame in enumerate(frames):
        for v in frame:
            v['row']['output_tick']=t;v['row']['native_observation']['native_frame_index']=t
    anchor=frames[anchor_index][0]
    path=anchored_review_actor_paths(frames,independent_anchor=anchor,anchor_frame_index=anchor_index,
        fixed_anchor_occurrence=True)[0]
    assert sum(v is not None for v in path['observations'])==1
    assert path['observations'][anchor_index] is anchor


def test_short_occlusion_reacquires_the_actual_same_actor_without_filling_missing_pose():
    frames=[[view(0,0)],[],[view(2,0)],[view(3,0)]]
    path=anchored_review_actor_paths(frames,independent_anchor=frames[0][0],anchor_frame_index=0,
        fixed_anchor_occurrence=True)[0]
    assert path['frame_local_head_ids']==(0,None,0,0)
    assert not path['all_sampled_frames_connected'] and path['missing_frames']==1


def test_long_future_track_cannot_amortize_a_one_frame_teleport():
    from dataclasses import replace
    frames=[[view(t,0,t)] for t in range(40)]
    for t,frame in enumerate(frames):
        v=frame[0];v['descriptor']=replace(v['descriptor'],face_height=.1,face_width=.06,
            head_center=(.39,.43) if t==0 else (.70,.67))
    path=anchored_review_actor_paths(frames,independent_anchor=frames[0][0],anchor_frame_index=0,
        fixed_anchor_occurrence=True,maximum_frames=64)[0]
    assert path['observations'][1] is None


def test_occlusion_expanded_state_costs_and_marginals_match_exhaustive_connected_paths():
    from itertools import product
    from crosscut.group_review_actor_path import _link
    frames=[[view(0,0)],[],[view(2,0,1),view(2,1,2)],[view(3,0,3),view(3,1,4)]]
    seed=frames[0][0];possibilities=[]
    for tail in product(*(frame+[None] for frame in frames[1:])):
        sequence=(seed,*tail);last=seed;current=seed;cost=0.;valid=True
        for v in tail:
            if v is None:cost+=3.
            else:
                link=_link(last,v,maximum_native_gap=12,minimum_cosine=.25)
                if link is None or not link['linked']:valid=False;break
                cost+=link['cost']+(1.5 if current is None else 0.);last=v
            current=v
        if valid:possibilities.append((cost,sequence))
    path=anchored_review_actor_paths(frames,independent_anchor=seed,anchor_frame_index=0,
        fixed_anchor_occurrence=True)[0]
    assert path['cost']==pytest.approx(min(c for c,_ in possibilities))
    for i,v in enumerate(path['observations']):
        alternative=min((c for c,s in possibilities if s[i] is not v),default=None)
        if alternative is None:assert path['frame_competing_margins'][i] is None
        else:assert path['frame_competing_margins'][i]==pytest.approx(alternative-path['cost'])


def test_expired_occlusion_does_not_reacquire_an_actor_without_observed_bridge():
    frames=[[view(0,0)],*[[] for _ in range(12)],[view(13,0)],[view(14,0)]]
    path=anchored_review_actor_paths(frames,independent_anchor=frames[0][0],anchor_frame_index=0,
        fixed_anchor_occurrence=True)[0]
    assert all(v is None for v in path['observations'][1:])
