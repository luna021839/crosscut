import numpy as np
import pytest
from crosscut2.fixed_holds import State, clamped_camera, make_states, port_cost
from crosscut2.hold_utility import known_switch, soft_hold_utility,native_readability_cost


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


def test_optional_alignment_camera_can_fit_size_without_blind_native_zoom_reward():
    head=np.array([.5,.3,.16,.95]);support=np.zeros((2,8,3))
    native=clamped_camera(head,2.,support)
    aligned=clamped_camera(head,2.,support,alignment_target=.24)
    assert native.zoom==1. and aligned.zoom>1.
    assert soft_hold_utility(.16,aligned.zoom)['scale']==soft_hold_utility(.16)['scale']


def test_signage_framing_with_low_subject_receives_finite_native_position_cost():
    center=soft_hold_utility(.16,projected_center=(.5,.3))
    low=soft_hold_utility(.16,projected_center=(.5,.85))
    assert low['framing_position']>center['framing_position']


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
        assert receipt['whole_song_quadratic_tensor_allocated'] is False
        assert receipt['port_matrix_nodes_built'] == (len(nodes)-2 if count>1 else 0)
        assert receipt['cost_tensor_bytes'] == (receipt['fixed_states']**2*4 if count>1 else 0)
        assert receipt['actor_owned_alignment_families_available'] is True


def test_dp_consumes_the_same_missing_port_and_broadcast_cost_it_reports(monkeypatch):
    import crosscut2.fixed_holds as fixed
    banks=[bank(),bank()];nodes=np.array([0,10,20,30])
    banks[1]['heads'][1,1]=0;banks[1]['vectors'][1,1]=0
    for value in banks:
        value['native_readability']=np.tile([.08,.25,.1,100.],(4,2,8,1))
        value['native_readability_valid']=value['heads'][:,:,:,2]>.019
    captured={};original=fixed.coupled_semi_markov
    def observe(*args,**kwargs):
        producer=kwargs['seam_cost']
        def capture(node):
            matrix=producer(node)
            captured[node]=matrix.copy()
            return matrix
        kwargs['seam_cost']=capture
        return original(*args,**kwargs)
    monkeypatch.setattr(fixed,'coupled_semi_markov',observe)
    _,states,rows,_,_=fixed.select_fixed_holds(nodes,banks,None,fps=10,
        source_events=[np.array([],int) for _ in banks],broadcast_slots=['mbc','mbc'],check=lambda:None,
        boundary_readability=True)
    a=states.index(State(0,None,'native'));b=states.index(State(1,None,'native'))
    reported=fixed.port_cost(1,rows[a],rows[b],fps=10)
    assert reported['role']=='unobserved'
    assert captured[1][a,b]==pytest.approx(reported['soft_cost']+.9)
    for n in (1,2):
        for a,sa in enumerate(states):
            for b,sb in enumerate(states):
                if sa.source==sb.source:continue
                expected=fixed.port_cost(n,rows[a],rows[b],fps=10)['soft_cost']+.9
                assert captured[n][a,b]==pytest.approx(expected)


def test_overlarge_source_face_is_not_free_just_because_digital_zoom_is_one():
    comfortable=soft_hold_utility(.20)['scale']
    extreme=soft_hold_utility(.42)['scale']
    assert extreme-comfortable > 1.


def test_missing_actor_is_not_replaced_with_an_observed_other_actor():
    value=bank();value['vectors'][1,:,0]=value['vectors'][1,:,1]
    nodes=np.array([0,10,20,30])
    states,rows,_=make_states(nodes,[value],None,fps=10)
    row=rows[states.index(State(0,0,'native'))]
    assert row['indices'][1].tolist()==[-1,-1]
    assert row['owner'][1].tolist()==[-1,-1]
    assert np.all(row['utility_parts']['ownership_uncertainty'][1] > 0)
    assert not row['heads'][1].any()


def test_native_pixel_quality_reaches_the_actual_hold_and_dp_cost_without_source_exclusion():
    banks=[bank(),bank()];nodes=np.array([0,10,20,30])
    for source,value in enumerate(banks):
        value['native_readability']=np.tile([.15,.7 if source==0 else .15,0.,100.],(4,2,8,1))
        value['native_readability_valid']=value['heads'][:,:,:,2]>.019
    states,rows,_=make_states(nodes,banks,None,fps=10)
    a=rows[states.index(State(0,None,'native'))];b=rows[states.index(State(1,None,'native'))]
    assert total(b,nodes)>total(a,nodes)+1.
    assert b['utility_parts']['native_readability'].max()>0 and np.isfinite(b['unary']).all()


def test_missing_quality_is_not_imputed_and_native_saturation_is_finite():
    assert native_readability_cost(None)==0
    assert 0 < native_readability_cost([0,0,1,100]) <= 2
    with pytest.raises(ValueError):native_readability_cost([.1,float('nan'),0,100])


def test_digital_zoom_does_not_change_the_same_native_readability_cost():
    observed=[.1,.35,0,100]
    assert soft_hold_utility(.12,1.,native_readability=observed)['native_readability']==(
        soft_hold_utility(.12,1.35,native_readability=observed)['native_readability'])


def test_readability_at_a_boundary_is_not_diluted_into_long_hold_average():
    nodes=np.array([0,10,20,30]);banks=[bank(),bank()]
    for value in banks:
        value['native_readability']=np.tile([.15,.7,0.,100.],(4,2,8,1))
        value['native_readability_valid']=value['heads'][:,:,:,2]>.019
    banks[1]['native_readability'][1,1,0,1]=.05
    states,rows,_=make_states(nodes,banks,None,fps=10,boundary_readability=True)
    left=rows[states.index(State(0,0,'native'))]
    right=rows[states.index(State(1,0,'native'))]
    clean=port_cost(1,left,left,fps=10)
    blurred=port_cost(1,left,right,fps=10)
    expected=native_readability_cost([.15,.05,0.,100.])
    assert blurred['boundary_native_readability_cost']==pytest.approx(expected)
    assert blurred['soft_cost']-clean['soft_cost']==pytest.approx(expected)
    assert port_cost(2,left,right,fps=10)['boundary_native_readability_cost']==0


def test_unapproved_boundary_readability_experiment_is_off_by_default():
    nodes=np.array([0,10,20,30]);values=[bank(),bank()]
    for value in values:
        value['native_readability']=np.tile([.05,.1,.1,100.],(4,2,8,1))
        value['native_readability_valid']=value['heads'][:,:,:,2]>.019
    states,rows,_=make_states(nodes,values,None,fps=10)
    left=rows[states.index(State(0,0,'native'))];right=rows[states.index(State(1,0,'native'))]
    assert port_cost(1,left,right,fps=10)['boundary_native_readability_cost']==0
    assert not left['boundary_native_readability_enabled']


def test_readability_event_does_not_invent_occlusion_or_exclude_unknown_sources():
    value=bank();nodes=np.array([0,10,20,30])
    value['heads'][1,1]=0;value['vectors'][1,1]=0
    states,rows,_=make_states(nodes,[value],None,fps=10,boundary_readability=True)
    row=rows[states.index(State(0,None,'native'))]
    result=port_cost(1,row,row,fps=10)
    assert result['role']=='unobserved'
    assert result['boundary_native_readability_cost']==0
    assert np.isfinite(result['soft_cost'])
