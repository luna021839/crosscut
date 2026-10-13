import numpy as np
import pytest
from crosscut2.fixed_holds import State, clamped_camera, make_states
from crosscut2.hold_utility import known_switch, soft_hold_utility


def bank():
    heads=np.zeros((4,2,8,4),np.float32)
    heads[:,:,0]=[.45,.3,.12,.95];heads[:,:,1]=[.65,.4,.10,.95]
    vectors=np.zeros((4,2,3,128),np.float32);vectors[:,:,0,0]=1;vectors[:,:,1,1]=1
    return dict(heads=heads,vectors=vectors,native_indices=np.arange(8).reshape(4,2),zoom_cap=1.35)


def total(row,nodes):
    return float(row['unary'] @ np.diff(nodes)/10 + row['through'][1:-1].sum())


def test_known_foreground_a_b_a_is_not_free_like_stable_a():
    stable,switched=bank(),bank()
    switched['vectors'][1,:,[0,1]]=switched['vectors'][1,:,[1,0]]
    nodes=np.array([0,10,20,30])
    states,rows,_=make_states(nodes,[stable,switched],None,fps=10)
    a=rows[states.index(State(0,None,'native'))];b=rows[states.index(State(1,None,'native'))]
    assert total(b,nodes) > total(a,nodes)+1.
    assert b['owner'][1,0]!=b['owner'][0,1]


def test_known_switch_never_turns_unknown_into_actor_zero():
    assert known_switch(np.array([-1,-1,0,2]),np.array([-1,0,0,1])).tolist()==[False,False,False,True]


def test_named_owner_detector_order_change_does_not_pay_a_switch_event():
    original,reordered=bank(),bank()
    reordered['heads'][1,:,[0,1]]=reordered['heads'][1,:,[1,0]]
    reordered['vectors'][1,:,[0,1]]=reordered['vectors'][1,:,[1,0]]
    states,rows,_=make_states(np.array([0,10,20,30]),[original,reordered],None,fps=10)
    a=rows[states.index(State(0,0,'native'))];b=rows[states.index(State(1,0,'native'))]
    assert not a['interval_owner_switch'].any() and not b['interval_owner_switch'].any()
    assert not a['owner_through'].any() and not b['owner_through'].any()


def test_focal_size_is_continuous_not_a_two_level_step():
    a=soft_hold_utility(.021)['scale'];b=soft_hold_utility(.05)['scale'];c=soft_hold_utility(.069)['scale']
    assert a>b>c
    assert abs(soft_hold_utility(.069999)['scale']-soft_hold_utility(.070001)['scale']) < .001


def test_unattempted_and_failed_body_states_are_finite_uncertainty_not_visibility():
    values=[soft_hold_utility(.16,body_states=[state]) for state in range(4)]
    assert values[2]['visibility_uncertainty']>values[0]['visibility_uncertainty']>values[1]['visibility_uncertainty']
    assert all(np.isfinite(list(value.values())).all() for value in values)
    assert all('occlusion_certified' not in value for value in values)


def test_comfortable_native_face_is_not_zoomed_to_fill_point_twenty():
    camera=clamped_camera(np.array([.5,.3,.16,.95]),2.,np.zeros((2,8,3)))
    assert camera.zoom==1.


def test_cached_body_failures_reach_even_native_candidate_hold_utility():
    banks=[bank(),bank()];nodes=np.array([0,10,20,30])
    poses=[np.zeros((4,4,3,8,3)) for _ in banks]
    observed=np.ones((4,4,3),np.uint8);failed=observed.copy();failed[1:3,:,0]=2
    states,rows,_=make_states(nodes,banks,poses,fps=10,body_states=[observed,failed])
    a=rows[states.index(State(0,None,'native'))];b=rows[states.index(State(1,None,'native'))]
    assert total(b,nodes)>total(a,nodes)
    assert b['utility_parts']['visibility_uncertainty'].max()>0


def test_unknown_sources_keep_finite_native_paths():
    missing=bank();missing['heads'][:]=0;missing['vectors'][:]=0
    states,rows,_=make_states(np.array([0,10,20,30]),[missing],None,fps=10)
    assert State(0,None,'native') in states
    assert all(np.isfinite(row['unary']).all() for row in rows)


def test_selected_cost_decomposition_matches_single_and_multi_source_dp():
    from crosscut2.fixed_holds import select_fixed_holds
    nodes=np.array([0,10,20,30])
    for count in (1,2):
        _,_,_,score,receipt=select_fixed_holds(nodes,[bank() for _ in range(count)],None,fps=10,
            source_events=[np.array([15]) for _ in range(count)],broadcast_slots=None,check=lambda:None)
        assert receipt['selected_hold_cost_decomposition'][-1]['cumulative_cost']==pytest.approx(score)
