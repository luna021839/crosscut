import numpy as np

from crosscut2.featured_member_state import bind_hints,consensus
from crosscut2.focal_appearance import discrepancy
from test_crosscut2_event_holds import inputs


def members(banks,ids):
    result=[]
    for bank,member in zip(banks,ids):
        value=np.full(bank['heads'].shape[:3],-1,int);value[:,:,0]=member;result.append(value)
    return result


def test_multiple_source_native_views_supply_soft_featured_member_not_a_source_ban():
    _,banks,_,_=inputs(count=3)
    factors,receipt=consensus(banks,members(banks,[0,0,1]))
    assert receipt['framing_consensus_nodes']>0
    assert factors[0][1,1,0]>factors[2][1,1,0]>0
    assert not receipt['source_exclusion'] and not receipt['named_identity_certified']


def test_duplicate_broadcast_family_does_not_manufacture_independent_consensus():
    _,banks,_,_=inputs(count=3)
    for bank in banks:bank['consensus_family']='same-performance'
    factors,receipt=consensus(banks,members(banks,[0,0,0]))
    assert receipt['source_families']==1 and receipt['maximum_confidence']==0
    assert all(np.all(v==1.) for v in factors)


def test_no_member_evidence_does_not_force_a_member_or_remove_usable_sources():
    _,banks,_,_=inputs(count=3)
    factors,receipt=consensus(banks,members(banks,[-1,-1,-1]))
    assert receipt['maximum_confidence']==0 and all(np.all(v==1.) for v in factors)


def test_current_face_profiles_supply_featured_votes_when_cluster_hints_are_unknown_or_wrong():
    _,banks,_,_=inputs(count=3);names=members(banks,[-1,1,0])
    for bank,member in zip(banks,[0,0,1],strict=True):
        shape=bank['heads'].shape[:3];scores=np.full(shape+(6,),.1);scores[...,member]=.6
        bank.update(member_score_profiles=scores,member_score_support=np.ones(shape),
            member_score_gallery_identity='same-gallery')
    factors,receipt=consensus(banks,names)
    assert receipt['current_gallery_score_fields_used_before_camera']
    assert factors[0][1,1,0]==factors[1][1,1,0]>factors[2][1,1,0]>0.
    assert factors[2][1,1,0]<.75  # the current weighted vote is not attenuated a second time
    assert not receipt['source_exclusion']
    for bank in banks:bank['member_score_support'][:]=0.
    original,_,before=consensus(banks,names,return_evidence=True)
    for bank in banks:
        del bank['member_score_profiles'];del bank['member_score_support']
    legacy,_,after=consensus(banks,names,return_evidence=True)
    for a,b in zip(original,legacy,strict=True):np.testing.assert_allclose(a,b)
    for key in before:np.testing.assert_allclose(before[key],after[key])


def test_current_profiles_do_not_create_extra_family_votes_or_certainty_from_flat_scores():
    _,banks,_,_=inputs(count=3);names=members(banks,[-1,-1,-1])
    for bank in banks:
        shape=bank['heads'].shape[:3];scores=np.full(shape+(6,),.1);scores[...,0]=.6
        bank.update(member_score_profiles=scores,member_score_support=np.ones(shape),
            member_score_gallery_identity='same-gallery',consensus_family='same-performance')
    _,receipt=consensus(banks,names)
    assert receipt['maximum_confidence']==0.
    for s,bank in enumerate(banks):bank['member_score_profiles'][:]=.4;bank['consensus_family']=s
    factors,receipt=consensus(banks,names)
    assert receipt['maximum_confidence']==0. and all(np.all(value==1.) for value in factors)


def test_explicit_cohort_excludes_auxiliary_role_votes_but_keeps_the_physical_source_domain():
    import pytest
    _,banks,_,_=inputs(count=5);names=members(banks,[1,0,1,0,1])
    factors,receipt,evidence=consensus(banks,names,return_evidence=True,source_cohort=(1,3))
    assert len(factors)==5 and receipt['voting_source_indices']==[1,3]
    assert receipt['source_families']==2 and receipt['physical_source_domain_count']==5
    assert evidence['posterior'][...,0].min()>evidence['posterior'][...,1].max()
    before={key:value.copy() for key,value in evidence.items()}
    for s in (0,2,4):names[s][:]=5;banks[s]['heads'][:]=0.
    _,_,changed=consensus(banks,names,return_evidence=True,source_cohort=(1,3))
    for key in before:np.testing.assert_array_equal(before[key],changed[key])
    for invalid in ((),(1,1),(5,),(-1,),(True,)):
        with pytest.raises(ValueError,match='physical featured-vote'):consensus(banks,names,source_cohort=invalid)


def test_hint_binding_keeps_fourth_face_unknown_and_adds_native_hint_without_reinterpreting_vectors():
    _,banks,_,_=inputs(count=2)
    ports=[dict(frames=np.array([18]),heads=b['heads'][1:2].copy(),vectors=b['vectors'][1:2].copy()) for b in banks]
    faces,native=bind_hints(banks,[np.array([0]),np.array([1])],ports)
    assert (faces[0][:,:,0]==0).all() and (faces[1][:,:,0]==1).all()
    assert (faces[0][:,:,3:]==-1).all() and native[0]['member_hints'][0,0,0]==0
    assert 'member_hints' not in ports[0]
    np.testing.assert_array_equal(native[0]['vectors'],ports[0]['vectors'])


def test_recurrent_named_hint_is_soft_evidence_not_a_gate_or_identity_certificate():
    a=np.eye(128)[0];b=np.zeros(128);b[:2]=[.42,np.sqrt(1-.42**2)]
    raw,_=discrepancy(a,b)
    same,_=discrepancy(a,b,member_a=np.array(0),member_b=np.array(0))
    different,_=discrepancy(a,b,member_a=np.array(0),member_b=np.array(1))
    assert 0<same<raw<different<1.
