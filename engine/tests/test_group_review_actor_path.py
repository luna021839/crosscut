from copy import deepcopy
import pytest
from crosscut.group_review_actor_path import anonymous_review_actor_paths
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
    assert all(p['ambiguous'] for p in paths)
    assert {p['frame_local_head_ids'] for p in paths}=={(17,17,17),(5,5,5)}


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
