"""Explicit two-actor chart inputs; native row-local IDs are not identities."""
from copy import deepcopy
from dataclasses import replace

import pytest

from crosscut import group_scene_focal_pose_motion as pose
from crosscut.group_joint_person_ownership import bind_person_ownership
from engine.tests import test_scene_focal_pose_motion as fixtures


def two_actor_world():
    sensors,state,base,rows,_=fixtures.scene_world()
    for native,row in tuple(rows.items()):
        people=[];heads=[]
        for actor,shift in ((0,-.2),(1,.2)):
            person=row.people[0]
            raw=tuple((x+shift,y,c) for x,y,c in person.raw_pose_xyc)
            people.append(replace(person,observation_id=100+actor,
                box_xywh=(.05 if actor==0 else .55,.2,.4,.65),raw_pose_xyc=raw,
                upper_body_xyc=tuple(raw[n] for n in (5,6,7,8,9,10,11,12)),
                candidate_head_observation_ids=(),unique_head_observation_id=None,
                competing_person_observation_ids=(),part_states=()))
            head=row.heads[0]
            identity=(native%2 if actor==0 else 1-native%2)
            heads.append(replace(head,observation_id=identity,
                box_xywh=(head.box_xywh[0]+shift,*head.box_xywh[1:]),
                landmarks_xy=tuple((x+shift,y) for x,y in head.landmarks_xy),
                candidate_body_observation_ids=()))
        owned,owners=bind_person_ownership(tuple(people),tuple(heads),row.frame_size_wh)
        heads=tuple(replace(h,candidate_body_observation_ids=tuple(owners[h.observation_id])) for h in heads)
        rows[native]=replace(row,people=owned,heads=heads)
    # Explicit synthetic face continuation, not a learned identity claim.
    def links(a,b):
        pairs=[[a.native_frame_index%2,b.native_frame_index%2],
            [1-a.native_frame_index%2,1-b.native_frame_index%2]]
        return dict(raw_pair=[a.sha256,b.sha256],measured_links=pairs,
            unique_links=pairs,cross_camera_identity_certified=False)
    sensors._head_links=links
    states=[];hypotheses=[]
    for actor in (0,1):
        h=deepcopy(base)
        h['anchor_head_ids']=[(tick%2 if actor==0 else 1-tick%2) for tick in h['anchor_output_ticks']]
        h['raw_observation_sha256']=[rows[tick].sha256 for tick in h['anchor_output_ticks']]
        sensors.hypotheses[pose.digest(h)]=h
        states.append(sensors.bind(state,h));hypotheses.append(h)
    return sensors,tuple(states),tuple(hypotheses),rows


def view(sensors,state,rows,first,last):
    return pose.focal_pose_view(sensors,state,rows[first],rows[last],output_ticks=(first,last),
        selected_head_ids=(0,1),required_part_names=('left_arm','right_arm'))


@pytest.mark.parametrize('order',[(0,1),(1,0)])
def test_primary_native_anchor_positions_own_the_cached_person_path(order):
    sensors,states,hypotheses,rows=two_actor_world()
    results={}
    for actor in order:
        first,last=(0,1) if actor==0 else (1,2)
        results[actor]=view(sensors,states[actor],rows,first,last)
        fixtures.validate_view(results[actor],sensors,states[actor],hypotheses[actor])
        assert results[actor]['required_body_observation_ids']==[[100+actor],[100+actor]]
        assert [s['head_observation_id'] for s in results[actor]['actual_output_samples']]==[0,1]
    assert len(sensors.focal_pose_native_cache)==2
    for actor in (0,1):
        cold,cold_states,cold_hypotheses,cold_rows=two_actor_world()
        first,last=(0,1) if actor==0 else (1,2)
        expected=view(cold,cold_states[actor],cold_rows,first,last)
        assert results[actor]==expected


@pytest.mark.parametrize('order',[(0,1),(1,0)])
def test_normal_dense_is_order_independent_with_alternating_frame_local_head_ids(order):
    sensors,states,hypotheses,_=two_actor_world()
    for actor in order:
        dense=sensors.dense((states[actor],states[actor],states[actor]))[0]
        for row in dense['rows'][1:]:
            value=row['required_body_temporal_link']
            fixtures.validate_view(value,sensors,states[actor],hypotheses[actor])
            assert value['required_body_observation_ids']==[[100+actor],[100+actor]]


def test_corrupted_internal_cached_primary_path_is_rejected_before_export():
    sensors,states,_,rows=two_actor_world()
    view(sensors,states[0],rows,0,1)
    measured=next(iter(sensors.focal_pose_native_cache.values()))
    # Internal cache data is not trusted merely because a key exists.
    for key in tuple(sensors.focal_pose_native_cache):
        sensors.focal_pose_native_cache[key]=replace(measured,head_ids=(1,0,1))
    with pytest.raises(ValueError,match='cached.*owner'):
        view(sensors,states[0],rows,0,1)
