import numpy as np
import pytest

from crosscut2.source_member_scores import MemberScoreSeeds,bank_signature,from_queries,pair_risk
from crosscut2.featured_member_state import bind_hints
from crosscut2.focal_appearance import discrepancy
from crosscut2.played_partial_pose import features,matrix
from crosscut2.boundary_arc_costs import ArcCosts
from test_crosscut2_focal_appearance_risk import vector,observation
from test_crosscut2_event_holds import inputs

NAMES=tuple('abcdef')


def profile(member=0,best=.43):
    result=np.array([.1,.12,.08,.15,.09,.11]);result[member]=best
    return result


def seed(bank,*,member=0,best=.43):
    votes=[dict(member_scores=dict(zip(NAMES,profile(member,best))))]
    return from_queries(bank,[(int(bank['native_indices'][1,1]),bank['heads'][1,1,0])],votes,
        [-1],gallery_identity='same-original-gallery',member_names=NAMES)


def test_individual_unknown_gallery_scores_survive_without_new_forced_labels():
    _,banks,_,_=inputs(count=2);bank=banks[0];scores=seed(bank)
    raw,strength=scores.project(bank['vectors'])
    assert scores.cluster_hints.tolist()==[-1] and strength[:,:,0].min()>0
    np.testing.assert_allclose(raw[:,:,0],np.broadcast_to(profile(),raw[:,:,0].shape))
    assert not raw[:,:,1:].any() and not strength[:,:,1:].any()
    assert not scores.vectors.flags.writeable and not scores.scores.flags.writeable
    assert '0.43' not in repr(scores)
    changed=dict(bank,heads=bank['heads'].copy());changed['heads'][0,0,0,0]+=.01
    with pytest.raises(ValueError,match='physical source'):scores.verify(changed)


def test_missing_queries_ties_and_duplicate_observations_do_not_create_certainty():
    _,banks,_,_=inputs(count=2);bank=banks[0];q=vector()
    a=MemberScoreSeeds(np.array([-1]),np.array([q]),np.array([profile()]),bank_signature(bank),'gallery')
    duplicate=MemberScoreSeeds(np.array([-1]),np.array([q,q]),np.array([profile(),profile()]),bank_signature(bank),'gallery')
    for x,y in zip(a.project(q),duplicate.project(q),strict=True):np.testing.assert_array_equal(x,y)
    conflicting=MemberScoreSeeds(np.array([-1]),np.array([q,q]),np.array([profile(),profile(1)]),bank_signature(bank),'gallery')
    assert conflicting.project(q)[1]==0.
    assert a.project(np.zeros(128))[1]==0.
    empty=from_queries(bank,[(int(bank['native_indices'][1,1]),bank['heads'][1,1,0])],[None],[-1],
        gallery_identity='gallery',member_names=NAMES)
    assert len(empty.vectors)==0 and not empty.project(bank['vectors'])[1].any()


def test_source_and_native_score_fields_share_the_same_queries_and_no_fourth_face_borrow():
    _,banks,_,_=inputs(count=2)
    ports=[dict(frames=np.array([18]),heads=b['heads'][1:2].copy(),vectors=b['vectors'][1:2].copy()) for b in banks]
    old_vectors=banks[0]['vectors'].copy();old_heads=banks[0]['heads'].copy()
    faces,native=bind_hints(banks,[seed(banks[0]),seed(banks[1],member=1)],ports)
    assert (faces[0]==-1).all() and (native[0]['member_hints']==-1).all()
    assert banks[0]['member_score_support'][:,:,0].min()>0
    np.testing.assert_allclose(native[0]['member_score_profiles'][0],banks[0]['member_score_profiles'][1])
    assert not banks[0]['member_score_support'][:,:,3:].any()
    np.testing.assert_array_equal(old_vectors,banks[0]['vectors']);np.testing.assert_array_equal(old_heads,banks[0]['heads'])
    assert 'member_score_profiles' not in ports[0]


def test_actual_profiles_replace_the_same_hint_evidence_once_without_changing_admission():
    a,b=profile(),profile(1)
    same,_,used=discrepancy(vector(),vector(),member_a=np.array(-1),member_b=np.array(-1),
        profile_a=a,profile_b=a,support_a=1.,support_b=1.,details=True)
    different,_,_=discrepancy(vector(),vector(),member_a=np.array(-1),member_b=np.array(-1),
        profile_a=a,profile_b=b,support_a=1.,support_b=1.,details=True)
    assert used==1. and same==pytest.approx(0.,abs=1e-15) and different>same
    old=discrepancy(vector(),vector(-1),member_a=np.array(0),member_b=np.array(0))[0]
    actual,_,used=discrepancy(vector(),vector(-1),member_a=np.array(0),member_b=np.array(0),
        profile_a=a,profile_b=b,support_a=1.,support_b=1.,details=True)
    assert actual>old and used==1.
    left=observation(0,vector());right=observation(1,vector())
    for row,p in ((left,a),(right,b)):
        row.update(individual_member_score_profile=p,individual_member_score_support=1.,individual_member_score_gallery='same')
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    assert detail['individual_member_score_influence'][0,0]==1. and detail['allowed'][0,0]
    right['individual_member_score_gallery']='other-gallery'
    other=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    assert other['individual_member_score_influence'][0,0]==0. and other['appearance_rate'][0,0]==0.


def test_weak_current_face_support_cannot_receive_a_blanket_same_cluster_discount():
    a=vector();b=np.zeros(128);b[1]=1.
    old,_=discrepancy(a,b,member_a=np.array(0),member_b=np.array(0))
    actual,measured,influence=discrepancy(a,b,member_a=np.array(0),member_b=np.array(0),
        profile_a=profile(),profile_b=profile(),support_a=0.,support_b=1.,details=True)
    assert old==.25 and actual==.5 and measured and influence==0.
    missing,measured,influence=discrepancy(np.zeros(128),np.zeros(128),
        member_a=np.array(0),member_b=np.array(0),profile_a=profile(),profile_b=profile(),
        support_a=0.,support_b=0.,details=True)
    assert missing==.5 and not measured and influence==0.


def test_current_profile_confidence_is_monotonic_and_not_double_counted_with_cluster_labels():
    a=vector();b=np.zeros(128);b[1]=1.
    values=[]
    for support in (0.,.2,.5,1.):
        hinted,_,influence=discrepancy(a,b,member_a=np.array(0),member_b=np.array(0),
            profile_a=profile(),profile_b=profile(),support_a=support,support_b=1.,details=True)
        no_hint,_,_=discrepancy(a,b,member_a=np.array(-1),member_b=np.array(-1),
            profile_a=profile(),profile_b=profile(),support_a=support,support_b=1.,details=True)
        assert hinted==no_hint and influence==support
        values.append(hinted)
    assert np.all(np.diff(values)<=0.) and values[0]==.5 and values[-1]==pytest.approx(.25)


def test_flat_missing_or_weaker_profiles_are_not_positive_identity_proof():
    risk,strength=pair_risk(profile(),np.ones(6)*.4,1.,1.)
    assert strength==0.
    _,strength=pair_risk(profile(),profile(),.2,.8)
    assert strength==.2
    _,strength=pair_risk(profile(),profile(),1.,1.,compatible=False)
    assert strength==0.


def test_actual_native_and_authored_boundaries_consume_equal_score_risk():
    heads=np.zeros((1,2,8,4));heads[:,:,0]=[.5,.3,.2,.99]
    v=np.zeros((1,2,3,128));v[0,0,0]=vector();v[0,1,0]=vector(-1.)
    q=np.zeros((1,2,8,6));q[:,:,0]=profile()
    w=np.zeros((1,2,8));w[:,:,0]=.7
    bank=dict(frames=np.array([20]),heads=heads,vectors=v,
        member_score_profiles=q,member_score_support=w,member_score_gallery_identity='same')
    arcs=ArcCosts.__new__(ArcCosts);arcs.native=[bank]
    camera=np.tile([1.,.5,.5],(1,1,1))
    native=arcs.native_rates(0,np.array([20]),camera,camera)
    left,right=observation(0,vector()),observation(1,vector(-1.))
    for row in (left,right):
        row.update(individual_member_score_profile=profile(),individual_member_score_support=.7,individual_member_score_gallery='same')
    authored=matrix(features([left],fps=30.),features([right],fps=30.))[0]
    assert native[0,0]==pytest.approx(authored[0,0])
    heads[:,:,3]=[.5,.3,.4,.99]
    assert arcs.native_rates(0,np.array([20]),camera,camera)[0,0]==pytest.approx(1.)


def test_cli_requires_explicit_photo_inputs_before_any_acquisition(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--individual-member-scores'])
    with pytest.raises(ValueError,match='explicit local photo assets'):main()


def test_full_interval_consumer_receives_individual_profiles_before_source_and_camera_selection():
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    nodes,banks,_,_=args
    for bank in banks:
        bank['programme_render_clock']=dict(fps=30.,count=int(nodes[-1]),source_times=nodes/30.)
    kw['identity_hints']=[seed(banks[0]),seed(banks[1],member=1)]
    result=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True))
    receipt=result[4]
    assert all(row['supported_face_occurrences']>0 for row in receipt['individual_member_score_fields'])
    assert all(row['raw_scores_are_not_probabilities'] and not row['forced_member_labels']
        for row in receipt['individual_member_score_fields'])
    assert result[3]==pytest.approx(sum(sum(row['components'].values())
        for row in receipt['selected_hold_cost_decomposition']))
    assert not receipt['professional_quality_approved']
