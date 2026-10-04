from copy import deepcopy
from dataclasses import replace
import pytest
from crosscut.group_review_reference_actor import freeze_original_review_actor_inventory
from crosscut.group_review_camera_repair import FramingDescriptor
from crosscut.group_performer_family_owner import digest

FILE=[1,2,3,4,5]
UNIT=(2,'clock',digest(FILE),7)

def view(tick,actor,head_id,*,center=.5,height=.2,camera='camera',native=None):
    vector=[0.]*128;vector[actor]=1.
    box=[center-.05,.25,.1,height]
    descriptor=FramingDescriptor((center,.25+height/2),height,.1,None,head_id,False)
    return dict(vector=vector,descriptor=descriptor,row=dict(output_tick=tick,
        native_observation=dict(request_source_index=2,native_frame_index=tick if native is None else native,
            source_clock_identity='clock',source_file_identity=FILE,
            heads=[dict(observation_id=head_id,box_xywh=box)]),
        projection=dict(camera_sha256=camera)))

def freeze(frames,units=None):
    return freeze_original_review_actor_inventory(frames,native_units={t:UNIT for t in frames}
        if units is None else units,original_program_identity='original')

def test_larger_side_face_cannot_replace_original_centered_actor_in_same_native_shot():
    frames={t:[view(t,0,t+10),view(t,1,t+20,center=.15,height=.45)] for t in range(5)}
    selected,receipt=freeze(frames)
    assert [v['descriptor'].frame_local_head_id for v in selected.values()]==list(range(10,15))
    assert not receipt['donor_costs_used'] and not receipt['named_actor_certified']
    assert receipt['groups'][0]['seed_tick']==2

def test_frame_local_detector_id_changes_do_not_change_the_reference_actor():
    frames={0:[view(0,0,4),view(0,1,9,center=.9)],
        1:[view(1,0,12),view(1,1,4,center=.9)],2:[view(2,0,9),view(2,1,12,center=.9)]}
    selected,_=freeze(frames)
    assert [selected[t]['descriptor'].frame_local_head_id for t in frames]==[4,12,9]

def test_editorial_camera_jumps_do_not_become_native_actor_reacquisition():
    frames={t:[view(t,0,7,camera='original-cell-'+str(t))] for t in range(5)}
    selected,receipt=freeze(frames)
    assert all(v is not None for v in selected.values())
    assert all(not r['link']['observed_native_camera_cut'] for r in receipt['groups'][0]['links'])
    assert all(selected[t]['row']['projection']['camera_sha256']=='original-cell-'+str(t) for t in frames)

def test_missing_target_stays_unmatched_without_largest_face_reselection():
    frames={0:[view(0,0,7)],1:[view(1,1,9,center=.9)],2:[view(2,0,5)]}
    selected,_=freeze(frames)
    assert selected[1] is not None  # Independent midpoint seed is the only observed face.
    assert selected[0] is None and selected[2] is None
    assert selected[1]['descriptor'].frame_local_head_id==9

def test_missing_slot_can_be_bridged_only_from_unchanged_direct_anchor_within_native_gap():
    frames={0:[view(0,0,7)],1:[],2:[view(2,0,8)],20:[view(20,0,9)]}
    selected,_=freeze(frames)
    assert selected[1] is None and selected[20] is None
    assert selected[0] is not None and selected[2] is not None

def test_ambiguous_identity_is_retained_not_certified_from_centered_face():
    a=view(0,0,8);b=view(0,0,9)
    frames={0:[a,b],1:[view(1,0,7)],2:[view(2,0,6)]}
    selected,_=freeze(frames)
    assert selected[0]['reacquisition_ambiguous']

def test_raw_sources_are_unchanged_and_physical_mismatch_is_rejected():
    frames={0:[view(0,0,7)],1:[view(1,0,8)]};before=deepcopy(frames)
    freeze(frames);assert frames==before
    frames[1][0]['row']['native_observation']['source_clock_identity']='wrong'
    with pytest.raises(ValueError,match='borrowed'):freeze(frames)

def test_new_physical_native_camera_unit_gets_its_own_independent_seed():
    frames={0:[view(0,0,7)],1:[view(1,0,8)],2:[view(2,1,9)],3:[view(3,1,10)]}
    units={t:UNIT if t<2 else (*UNIT[:3],8) for t in frames}
    selected,receipt=freeze(frames,units)
    assert all(v is not None for v in selected.values()) and len(receipt['groups'])==2


def test_distant_original_visits_do_not_erase_independently_scoped_local_actor_requests():
    frames={0:[view(0,0,7)],1:[view(1,0,8)],100:[view(100,1,9)],101:[view(101,1,10)]}
    selected,receipt=freeze_original_review_actor_inventory(frames,
        native_units={t:UNIT for t in frames},original_program_identity='original',
        original_attention_scopes={t:'original-visit-a' if t<100 else 'original-visit-b' for t in frames})
    assert all(v is not None for v in selected.values())
    assert len(receipt['groups'])==2 and not receipt['attention_scopes_are_physical_camera_cuts']


def test_attention_scope_cannot_be_omitted_only_after_actor_link_failure():
    frames={0:[view(0,0,7)],1:[view(1,0,8)]}
    with pytest.raises(ValueError,match='scopes'):
        freeze_original_review_actor_inventory(frames,native_units={t:UNIT for t in frames},
            original_program_identity='original',original_attention_scopes={0:'a'})
