import numpy as np

from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.played_partial_pose import features,matrix
from crosscut2.source_clock_action import endpoint_correspondence


def catalogue(*,wrong=False,swap=False,cut=()):
    nodes=np.array([0,20,40]);heads=np.zeros((3,2,8,4));vectors=np.zeros((3,2,3,128))
    heads[:,:,0]=[.45,.3,.2,.99];heads[:,:,1]=[.75,.3,.12,.99]
    vectors[:,:,0,0]=1.;vectors[:,:,1,1]=1.
    base=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
                   [.3,.55,.9],[.7,.55,.9],[.43,.7,.9],[.57,.7,.9]])
    poses=np.zeros((3,4,3,8,3));poses[:,:,0]=base;poses[:,:,1]=base+[.2,0,0]
    for at,y in enumerate([.65,.4,.4,.15 if wrong else .65]):poses[:,at,0,[4,5],1]=y
    if swap:
        heads[:,1,:2]=heads[:,1,:2][:,::-1].copy();vectors[:,1,:2]=vectors[:,1,:2][:,::-1].copy()
        poses[:,2:,:2]=poses[:,2:,:2][:,:,::-1].copy()
    bank=dict(heads=heads,vectors=vectors,zoom_cap=2.,native_cut_frames=np.asarray(cut),
              pose_native_indices=np.array([[0,1,2,3],[13,19,20,26],[30,36,37,39]]))
    return SourceCatalogue(nodes,bank,poses,[],dict(presence=np.zeros((3,2,1))),[],fps=30.)


def obs(cat,side):return cat.observation(1,side,np.tile([1.,.5,.5],(7,1)))


def test_actual_catalogue_supplies_synchronous_context_and_reversals_match():
    a,b,wrong=catalogue(),catalogue(),catalogue(wrong=True)
    good=matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    bad=matrix(features([obs(a,0)],fps=30.),features([obs(wrong,1)],fps=30.),details=True)
    assert good['same_clock_source_action_used'][0,0]
    assert good['pose_rate'][0,0]<bad['pose_rate'][0,0]
    assert good['face_rate'][0,0]==bad['face_rate'][0,0]==0.


def test_face_order_change_selects_the_same_subject_on_both_source_sides():
    cat=catalogue(swap=True);a,b=obs(cat,0),obs(cat,1)
    assert a['head_index']==0 and b['head_index']==1
    assert a['source_clock_subject_associated'] and b['source_clock_subject_associated']
    np.testing.assert_array_equal(a['source_clock_body'],b['source_clock_body'])
    np.testing.assert_array_equal(a['source_clock_body'][:,4,1],np.array([.65,.4,.4,.65],np.float32))


def test_native_camera_cut_or_missing_appearance_does_not_invent_common_subject():
    cat=catalogue(cut=(20,))
    assert obs(cat,0)['source_clock_observed_sample_mask'].tolist()==[True,True,False,False]
    assert obs(cat,1)['source_clock_observed_sample_mask'].tolist()==[False,False,True,True]
    cat=catalogue();vectors=cat.bank['vectors'].copy();vectors[:]=0.
    assert (endpoint_correspondence(cat.bank['heads'],vectors)==-1).all()


def test_incoming_native_cut_keeps_the_real_shared_after_interval_for_moving_match():
    left,right=catalogue(),catalogue(cut=(20,))
    value=matrix(features([obs(left,0)],fps=30.),features([obs(right,1)],fps=30.),details=True)
    assert value['same_clock_source_action_used'][0,0]
    assert not obs(right,1)['source_clock_observed_sample_mask'][:2].any()


def test_missing_or_contradicted_common_action_keeps_finite_uncertainty_not_perfect_cost():
    cat=catalogue();cat.evidence[:]=0
    a,b=obs(cat,0),obs(cat,1)
    value=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    assert not value['same_clock_source_action_used'][0,0]
    assert value['pose_rate'][0,0]==.5


def set_style(cat,colour):
    cat.style['colour']=np.zeros(cat.bank['heads'].shape[:3]+(2,40))
    cat.style['colour_valid']=np.ones(cat.bank['heads'].shape[:3],bool)
    cat.style['colour'][...,colour]=1.


def test_change_credit_needs_measured_moving_configuration_appearance_style_and_visibility():
    a,b=catalogue(),catalogue();set_style(a,0);set_style(b,1)
    def detail():return matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    measured=detail()
    assert measured['matched_action_change_density'][0,0]>0
    set_style(b,0)
    assert detail()['matched_action_change_density'][0,0]==0  # identical style, not a constant cut bonus
    set_style(b,1);b.body[:]=b.body[:,1:2]
    assert detail()['matched_action_change_density'][0,0]==0  # no jointly observed movement
    b=catalogue();set_style(b,1);b.style['colour_valid'][:]=False
    assert detail()['matched_action_change_density'][0,0]==0


def test_worse_action_cannot_earn_the_same_change_credit_for_a_pretty_new_stage():
    a,good,bad=catalogue(),catalogue(),catalogue(wrong=True)
    for c,colour in [(a,0),(good,1),(bad,1)]:set_style(c,colour)
    def density(b):return matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)['matched_action_change_density'][0,0]
    assert density(good)>density(bad)


def test_recurrent_other_member_does_not_get_same_actor_change_credit_or_exclude_cut():
    a,b=catalogue(),catalogue();set_style(a,0);set_style(b,1)
    a.member_hints[:]=0;b.member_hints[:]=1
    value=matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    assert value['allowed'][0,0] and value['same_clock_source_action_used'][0,0]
    assert value['matched_action_change_density'][0,0]==0.


def test_preference_cues_limit_change_not_multiply_as_independent_probabilities():
    a,b=catalogue(),catalogue();set_style(a,0);set_style(b,1)
    a.quality[:]=.36;b.quality[:]=.36
    value=matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    # Measured articulated activity remains capped as MODEL_ONLY. It must
    # not be discounted again by multiplying five uncalibrated factors.
    density=value['matched_action_change_density'][0,0]
    assert .15<density<=.5
