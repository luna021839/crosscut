import numpy as np
from crosscut2.fixed_holds import State, make_states, port_cost
from crosscut2.action_geometry import complete_arm_motion


def fixture():
    heads=np.zeros((4,2,8,4));heads[:,:,0]=[.45,.3,.12,.95]
    vectors=np.zeros((4,2,3,128));vectors[:,:,0,0]=1
    return dict(heads=heads,vectors=vectors,native_indices=np.arange(8).reshape(4,2),zoom_cap=1.35)


def pose(y):
    points=np.array([[.4,.4,.9],[.5,.4,.9],[.38,.5,.9],[.52,.5,.9],
        [.38,y,.9],[.52,y,.9],[.41,.7,.9],[.49,.7,.9]])
    points[[1,3,5,6,7]]=0
    return points


def comparisons(*, full_left=False, events=None, missing_joint=None):
    poses=[np.zeros((4,4,3,8,3)) for _ in range(4)]
    for s,ys in ((0,(.60,.45,.45,.60)),(1,(.60,.45,.45,.60)),(2,(.30,.45,.45,.30))):
        poses[s][1,:,0]=[pose(y) for y in ys]
        if missing_joint is not None:poses[s][1,:,0,missing_joint]=0
    if full_left:
        poses[0][1,:,0,1]=[.5,.4,.9]
    states,rows,_=make_states(np.array([0,10,20,30]),[fixture() for _ in poses],poses,fps=10,
        source_events=events)
    row=lambda s:rows[states.index(State(s,None,'native'))]
    return [port_cost(1,row(0),row(s),fps=10) for s in (1,2,3)],row(0)


def test_single_arm_synchronized_hit_beats_missing_and_contrary_same_clock():
    records,_=comparisons()
    hit,wrong,missing=records
    assert hit['soft_cost']<missing['soft_cost']<wrong['soft_cost']
    assert hit['synchronous_velocity_loss']==0.
    assert wrong['synchronous_velocity_loss']>.9
    assert hit['synchronous_basis_per_window']==['directed_complete_arm_segments']*2


def test_full_body_and_partial_arm_choose_common_arm_basis_not_mixed_vectors():
    records,_=comparisons(full_left=True)
    assert records[0]['soft_cost']<records[2]['soft_cost']<records[1]['soft_cost']
    assert records[0]['synchronous_basis_per_window']==['directed_complete_arm_segments']*2


def test_missing_elbow_or_wrist_does_not_invent_single_arm_motion():
    for joint in (2,4):
        records,_=comparisons(missing_joint=joint)
        assert records[0]['synchronous_velocity_loss'] is None
        assert records[1]['synchronous_velocity_loss'] is None


def test_zero_length_bone_is_not_a_stationary_observed_arm():
    points=pose(.5)
    velocity,valid=complete_arm_motion(points,points,seconds=.2)
    assert not valid.any()


def test_native_cut_masks_arm_clock_window_like_full_body():
    records,row=comparisons(events=[np.array([8]) for _ in range(4)])
    assert not row['arm_valid'][1,0].any()
    assert row['arm_valid'][1,1].sum()==2
    assert records[0]['synchronous_basis_per_window'][0] is None
