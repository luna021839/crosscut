from itertools import product

import numpy as np
import pytest

from crosscut2.action_geometry import action_crop_cost, compare_action
from crosscut2.fixed_holds import State, clamped_camera, coupled_semi_markov, local_actor_hypotheses, make_states, port_cost


def bank_fixture(n=4):
    heads = np.zeros((n,2,8,4),np.float32)
    heads[:,:,0] = [.45,.3,.12,.95]
    heads[:,:,1] = [.65,.4,.10,.95]
    vectors = np.zeros((n,2,3,128),np.float32)
    vectors[:,:,0,0] = 1; vectors[:,:,1,1] = 1
    return dict(heads=heads,vectors=vectors,native_indices=np.arange(n*2).reshape(n,2),zoom_cap=1.35)


def test_source_actor_hypothesis_survives_detector_order_change_and_null_reacquisition():
    bank = bank_fixture()
    bank['heads'][1,:,[0,1]] = bank['heads'][1,:,[1,0]]
    bank['vectors'][1,:,[0,1]] = bank['vectors'][1,:,[1,0]]
    bank['vectors'][2] = 0
    labels,count = local_actor_hypotheses(bank)
    assert count == 2
    assert labels[1,0,1] == labels[0,0,0]
    assert labels[1,0,0] == labels[0,0,1]
    assert (labels[2] == -1).all()
    assert labels[3,0,0] == labels[0,0,0]


def test_early_adjacent_frame_singletons_do_not_exhaust_persistent_actor_budget():
    bank = bank_fixture(6)
    bank['vectors'][:] = 0
    bank['vectors'][0,:,0,0] = 1
    bank['vectors'][1,:,0,1] = 1
    bank['vectors'][2:,:,0,2] = 1
    labels,count = local_actor_hypotheses(bank,maximum_clusters=2)
    assert count == 1
    assert (labels[:2,:,0] == -1).all()
    assert (labels[2:,:,0] == 0).all()


def test_actor_hypothesis_budget_does_not_delete_any_physical_source():
    bank = bank_fixture()
    bank['vectors'][:] = 0
    states,rows,counts = make_states(np.array([0,10,20,30]),[bank,bank],None,fps=10)
    assert counts == [0,0]
    assert {s.source for s in states if s.actor is None and s.family == 'native'} == {0,1}
    assert all(s.actor is None for s in states)
    assert all(np.isfinite(d['unary']).all() for d in rows)


def test_actor_owned_states_keep_the_same_alignment_options_as_unknown_owner():
    bank = bank_fixture()
    bank['zoom_cap'] = 2.
    bank['heads'][:,:,0,2] = .16
    states, rows, _ = make_states(np.array([0,10,20,30]), [bank], None, fps=10)
    expected = {'native', 'action_fit', 'align_medium', 'align_close'}
    assert {s.family for s in states if s.actor == 0} == expected
    owned = rows[states.index(State(0,0,'align_close'))]
    unknown = rows[states.index(State(0,None,'align_close'))]
    assert np.all(owned['owner'] == 0)
    assert np.all(owned['cameras'][:,:,0] > 1.4)
    assert np.array_equal(owned['cameras'], unknown['cameras'])


def test_lazy_port_matrices_are_exact_and_built_once_for_both_duration_classes():
    nodes = np.arange(0,81,10)
    rng = np.random.default_rng(17)
    unary = rng.uniform(0,1,(4,8))
    seams = rng.uniform(0,4,(len(nodes),4,4)).astype(np.float32)
    calls = []
    def matrix(node):
        calls.append(node)
        return seams[node]
    kwargs = dict(fps=10,unary=unary,state_sources=np.array([0,0,1,1]))
    expected = coupled_semi_markov(nodes,seam_cost=seams,**kwargs)
    actual = coupled_semi_markov(nodes,seam_cost=matrix,**kwargs)
    assert actual == expected
    assert len(calls) == len(set(calls)) == len(nodes)-2


@pytest.mark.parametrize('bad', [np.zeros((3,4)), np.full((4,4),np.nan)])
def test_lazy_port_matrices_validate_shape_and_finiteness_before_use(bad):
    with pytest.raises(ValueError,match='finite fixed-state node port matrix'):
        coupled_semi_markov(np.arange(0,41,10),fps=10,unary=np.zeros((4,4)),
            seam_cost=lambda node: bad,state_sources=np.array([0,0,1,1]))


def test_lazy_ports_keep_default_ten_source_full_hypothesis_domain():
    sources = np.repeat(np.arange(10),13*4)
    calls = []
    def matrix(node):
        calls.append(node)
        return np.zeros((len(sources),len(sources)),np.float32)
    runs,score = coupled_semi_markov(np.array([0,30,60,90]),fps=10,
        unary=np.zeros((len(sources),3)),seam_cost=matrix,state_sources=sources)
    assert len(sources) == 520
    assert len(runs) == 3 and np.isfinite(score)
    assert calls == [1,2]


def test_camera_family_preserves_observed_inside_hold_wrists():
    head = np.array([.5,.3,.12,.95])
    support = np.zeros((2,8,3))
    support[:,:,0] = [.4,.6,.3,.7,.05,.95,.4,.6]
    support[:,:,1] = [.45,.45,.5,.5,.5,.5,.8,.8]
    support[:,:,2] = .9
    camera = clamped_camera(head,2.,support)
    assert action_crop_cost([camera.zoom],[[camera.x,camera.y]],support)[0] < .05
    assert camera.zoom < 2.


def test_torso_only_alignment_is_not_an_arm_action_match():
    pose = np.zeros((2,8,3))
    pose[:,[0,1,6,7]] = [[.4,.3,.9],[.6,.3,.9],[.4,.7,.9],[.6,.7,.9]]
    matched = compare_action(pose,pose,seconds=.2)
    assert matched.common_joints == 4
    assert matched.phase_loss is matched.velocity_loss is None
    assert matched.cost == 2.


def test_dp_keeps_more_expensive_same_source_camera_with_better_exit():
    nodes = np.array([0,10,20])
    unary = np.array([[0.,9.],[.8,9.],[9.,0.]])
    seams = np.full((3,3,3),100.)
    seams[1,1,2] = 0.
    runs,_ = coupled_semi_markov(nodes,fps=10,unary=unary,seam_cost=seams,
        state_sources=np.array([0,0,1]),minimum_hold=.9,maximum_hold=1.1,target_hold=1.)
    assert [r[2] for r in runs] == [1,2]


def test_fixed_state_dp_matches_brute_force_on_a_small_complete_graph():
    nodes = np.array([0,10,20,30])
    unary = np.array([[0.,.3,.7],[.2,.1,.5],[.4,0.,.1],[.3,.2,0.]])
    rng = np.random.default_rng(90)
    seams = rng.uniform(0,.3,(4,4,4))
    through = rng.uniform(0,.4,(4,4))
    through[:,[0,-1]] = 0
    sources = np.array([0,0,1,1])
    def enumerate_paths(start,history,score):
        if start == 3: return [(history,score)]
        results = []
        for end in range(start+1,min(3,start+2)+1):
            d = end-start
            for state in range(4):
                if history and sources[state] == sources[history[-1][2]]: continue
                cost = float(unary[state,start:end].sum()) + .23*(d-1.4)**2
                cost += float(through[state,start+1:end].sum())
                if history:
                    cost += .35 + seams[start,history[-1][2],state]
                    if len(history)>=2 and sources[state] == sources[history[-2][2]]: cost += .7
                results += enumerate_paths(end,history+[(start,end,state)],score+cost)
        return results
    expected = min(enumerate_paths(0,[],0.),key=lambda row:row[1])
    actual,score = coupled_semi_markov(nodes,fps=10,unary=unary,seam_cost=seams,
        state_sources=sources,minimum_hold=.9,maximum_hold=2.1,target_hold=1.4,through=through)
    assert score == pytest.approx(expected[1])
    assert actual == tuple(expected[0])


def test_all_unknown_action_leaves_actor_camera_graph_and_numeric_cost_identical():
    banks = [bank_fixture(),bank_fixture()]
    zero = [np.zeros((4,4,3,8,3),np.float32) for _ in banks]
    nodes = np.array([0,10,20,30])
    states,data,_ = make_states(nodes,banks,zero,fps=10)
    for left,right in product(range(len(states)),repeat=2):
        if states[left].source == states[right].source: continue
        a = port_cost(1,data[left],data[right],fps=10,action_enabled=False)
        b = port_cost(1,data[left],data[right],fps=10,action_enabled=True)
        assert a['soft_cost'] == b['soft_cost']
        assert b['conditional_pose_loss'] is None


def test_single_source_keeps_full_coverage_without_made_up_cross_source_cuts():
    runs,_ = coupled_semi_markov(np.arange(0,101,10),fps=10,
        unary=np.array([np.ones(10),np.full(10,2.)]),seam_cost=np.zeros((11,2,2)),
        state_sources=np.array([0,0]))
    assert runs == ((0,10,0),)


def test_single_source_never_materializes_irrelevant_cross_source_ports():
    def unused(node):
        raise AssertionError('single-source programme must not ask for seam matrices')
    runs,_ = coupled_semi_markov(np.arange(0,101,10),fps=10,
        unary=np.array([np.ones(10),np.full(10,2.)]),seam_cost=unused,
        state_sources=np.array([0,0]))
    assert runs == ((0,10,0),)


def test_equal_centroid_and_face_scale_cannot_hide_a_different_formation_layout():
    a, b = bank_fixture(), bank_fixture()
    for bank, xs in ((a, [.35, .5, .65]), (b, [.1, .5, .9])):
        bank['heads'][:] = 0
        bank['heads'][:,:,:3] = [[x, .4, .07, .95] for x in xs]
        bank['vectors'][:] = 0
    states, rows, _ = make_states(np.array([0,10,20,30]), [a,b], None, fps=10)
    first=rows[states.index(State(0,None,'native'))]
    second=rows[states.index(State(1,None,'native'))]
    same = port_cost(1, first, first, fps=10)
    different = port_cost(1, first, second, fps=10)
    assert same['role'] == different['role'] == 'formation'
    assert same['formation_layout_loss'] == 0
    assert different['formation_layout_loss'] > 2
    assert different['soft_cost'] > same['soft_cost'] + 2


def test_unknown_appearance_does_not_disable_observed_focal_camera_fitting():
    bank = bank_fixture(); bank['vectors'][:] = 0
    bank['heads'][:,:,0,2]=.08;bank['heads'][:,:,1,2]=.06
    states, rows, _ = make_states(np.array([0,10,20,30]), [bank], None, fps=10)
    assert [(s.actor, s.family) for s in states][:2] == [(None, 'native'), (None, 'action_fit')]
    assert all(s.actor is None for s in states)
    assert np.all(rows[0]['cameras'] == [1., .5, .5])
    assert rows[1]['cameras'][:,:,0].max() > 1.


def test_first_hold_does_not_bypass_the_declared_maximum_duration():
    nodes = np.array([0,10,20,30])
    runs,_ = coupled_semi_markov(nodes,fps=10,unary=np.array([[-10.,-10.,-10.],[0.,0.,0.]]),
        seam_cost=np.zeros((4,2,2)),state_sources=np.array([0,1]),minimum_hold=.9,
        maximum_hold=1.1,target_hold=1.)
    assert all((nodes[end]-nodes[start])/10 <= 1.1 for start,end,_ in runs)


def _moving_pose(wrist_y):
    return np.array([[.40,.40,.9],[.50,.40,.9],[.38,.50,.9],[.52,.50,.9],
        [.38,wrist_y,.9],[.52,wrist_y,.9],[.41,.70,.9],[.49,.70,.9]])


def test_same_clock_motion_prefers_synchronized_hit_over_out_of_phase_snapshot():
    nodes = np.array([0,10,20,30])
    banks = [bank_fixture() for _ in range(4)]
    poses = [np.zeros((4,4,3,8,3)) for _ in banks]
    for source, ys in ((0,(.60,.45,.45,.60)),(1,(.60,.45,.45,.60)),(2,(.30,.45,.45,.30))):
        poses[source][1,:,0] = [_moving_pose(y) for y in ys]
    states,data,_ = make_states(nodes,banks,poses,fps=10)
    row = lambda source: data[states.index(State(source,None,'native'))]
    hit, wrong, missing = (port_cost(1,row(0),row(source),fps=10)['soft_cost'] for source in (1,2,3))
    assert hit < missing < wrong
    assert wrong - hit > 1.


def test_unobserved_actor_side_does_not_insert_a_one_frame_native_camera_pulse():
    bank = bank_fixture()
    bank['heads'][:,:,0,2]=.08;bank['heads'][:,:,1,2]=.06
    bank['vectors'][1,1,0] = 0
    states,rows,_ = make_states(np.array([0,10,20,30]),[bank],None,fps=10)
    row = rows[states.index(State(0,0,'action_fit'))]
    assert row['cameras'][1,1,0] == pytest.approx(1.35)
    assert row['through'][1] == pytest.approx(0.)


def test_dp_charges_a_one_frame_camera_snap_only_inside_a_hold():
    runs,_ = coupled_semi_markov(np.array([0,10,20]),fps=10,
        unary=np.array([[0.,0.],[.1,.1],[5.,5.]]), seam_cost=np.full((3,3,3),100.),
        state_sources=np.array([0,0,1]),minimum_hold=.9,maximum_hold=2.1,target_hold=2.,
        through=np.array([[0.,3.,0.],[0.,0.,0.],[0.,0.,0.]]))
    assert runs == ((0,2,1),)


def test_authored_cut_does_not_pay_a_through_node_camera_cost():
    runs,score = coupled_semi_markov(np.array([0,10,20]),fps=10,
        unary=np.array([[0.,9.],[9.,0.]]),seam_cost=np.zeros((3,2,2)),
        state_sources=np.array([0,1]),minimum_hold=.9,maximum_hold=1.1,target_hold=1.,
        through=np.array([[0.,100.,0.],[0.,100.,0.]]))
    assert [r[2] for r in runs] == [0,1]
    assert score == pytest.approx(.35)


def test_native_cut_inside_pose_window_is_not_choreographic_velocity():
    bank = bank_fixture()
    poses = np.zeros((4,4,3,8,3))
    poses[1,:,0] = [_moving_pose(y) for y in (.60,.45,.45,.60)]
    states,rows,_ = make_states(np.array([0,10,20,30]),[bank],[poses],fps=10,
        source_events=[np.array([8])])
    row = rows[states.index(State(0,None,'native'))]
    assert not row['motion_valid'][1,0].any()
    assert row['motion_valid'][1,1,2:6].all()


def test_same_location_different_appearance_does_not_certify_unrealized_window():
    bank = bank_fixture()
    bank['vectors'][1,1,0] = 0
    bank['vectors'][1,1,0,2] = 1
    states,rows,_ = make_states(np.array([0,10,20,30]),[bank],None,fps=10)
    row = rows[states.index(State(0,None,'native'))]
    assert not row['continuous'][1]
