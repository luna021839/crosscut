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
