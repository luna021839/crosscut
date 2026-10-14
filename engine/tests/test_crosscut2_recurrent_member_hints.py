import numpy as np
from crosscut.ive import IVEMember
from crosscut2.recurrent_member_hints import select_faces,associate,consensus,valid_query_geometry


def test_only_original_quality_picks_independent_faces_before_member_comparison():
    heads=np.zeros((5,2,8,4));heads[:,:,0]=[.5,.3,.2,.95];heads[4,:,0,2]=.3
    labels=np.full((5,2,3),-1);labels[:,:,0]=0
    bank=dict(heads=heads,native_indices=np.array([[10,11],[15,16],[30,31],[45,46],[60,61]]))
    requests,groups=select_faces(bank,labels,1,native_fps=30.)
    assert len(requests)==3 and requests[0][0]==60
    assert all(abs(a[0]-b[0])>=8 for i,a in enumerate(requests) for b in requests[i+1:])
    assert groups==[[0,1,2]]


def test_one_strong_face_or_conflicting_members_is_not_consensus():
    members=tuple(IVEMember)
    known=lambda m:dict(state='LIKELY_MEMBER',member=m.value)
    unknown=dict(state='UNKNOWN',member=None)
    assert consensus([known(members[0]),unknown,unknown])==-1
    assert consensus([known(members[0]),known(members[1]),known(members[0])])==-1
    assert consensus([known(members[0]),known(members[0]),unknown])==0


def test_foreign_or_ambiguous_detector_face_cannot_substitute_for_cached_head():
    head=np.array([.5,.3,.2,.95])
    face=np.zeros(15);face[:4]=[.45*960,.2*540,.1*960,.2*540]
    assert associate(np.array([face]),head) is not None
    shifted=face.copy();shifted[0]+=.3*960
    assert associate(np.array([shifted]),head) is None
    assert associate(np.array([face,face.copy()]),head) is None


def test_partially_offscreen_face_is_unknown_without_source_abort_or_clamped_landmarks():
    box=(-.01,.2,.2,.3);points=((.03,.25),(.12,.25),(.08,.3),(.04,.4),(.11,.4))
    assert not valid_query_geometry(box,points)
    assert valid_query_geometry((0.,.2,.2,.3),points)
