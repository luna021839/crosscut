import numpy as np
import pytest

from crosscut2.focal_appearance import discrepancy,direct_discrepancy,direct_cosine
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.played_partial_pose import features,matrix
from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
from test_crosscut2_event_holds import inputs


def test_actual_pair_replaces_its_duplicate_gallery_vote_but_retains_independent_SFace():
    s=np.eye(128)[0];a,b=np.eye(512)[:2]
    arguments=dict(member_a=0,member_b=1,profile_a=np.array([.9,0,0,0,0,0]),
        profile_b=np.array([0,.9,0,0,0,0]),support_a=1.,support_b=1.)
    value,known,influence=discrepancy(s,s,**arguments,direct_a=a,direct_b=b,
        direct_valid_a=True,direct_valid_b=True,details=True)
    assert value==.25 and known and influence==0.
    before=discrepancy(s,s,**arguments,details=True)
    missing=discrepancy(s,s,**arguments,direct_a=a,direct_b=np.zeros(512),
        direct_valid_a=True,direct_valid_b=False,details=True)
    assert before==missing
    different_space=discrepancy(s,s,**arguments,direct_a=a,direct_b=b,
        direct_valid_a=True,direct_valid_b=True,direct_compatible=False,details=True)
    assert before==different_space


def test_missing_SFace_can_use_a_real_query_without_borrowing_or_inventing_one():
    a=np.eye(512)[0]
    value,known,_=discrepancy(np.zeros(128),np.zeros(128),direct_a=a,direct_b=a,
        direct_valid_a=True,direct_valid_b=True,details=True)
    assert value==0. and known
    with pytest.raises(ValueError,match='512D'):
        direct_discrepancy(np.ones(128),np.ones(128),True,True)
    with pytest.raises(ValueError,match='unit'):
        direct_discrepancy(a*2,a,True,True)


def test_broadcast_pair_matrix_uses_exact_typed_vector_families():
    a=np.eye(512)[:3];risk,known,cosine=direct_discrepancy(a[:,None],a[None,:],
        np.ones((3,1),bool),np.ones((1,3),bool))
    np.testing.assert_allclose(cosine,np.eye(3))
    assert known.all()
    np.testing.assert_allclose(risk,(1.-np.eye(3))/2.)


def catalogues():
    nodes,banks,poses,styles=inputs();cats=[]
    for s,bank in enumerate(banks):
        shape=bank['heads'].shape[:-1]
        bank['direct_face_vectors']=np.tile(np.eye(512,dtype=np.float32)[s],shape+(1,))
        bank['direct_face_vectors'].setflags(write=False)
        bank['direct_face_known']=bank['heads'][...,2]>.019
        bank['direct_face_descriptor_family']='f'*64
        cats.append(SourceCatalogue(nodes,bank,poses[s],[],styles[s],[],fps=30.,direct_face_appearance=True))
    return nodes,cats,poses,styles


def test_source_catalogue_delivers_only_the_pinned_current_face_to_the_actual_cost():
    nodes,cats,_,_=catalogues()
    observations=[c.observation(1,side,np.tile([1.,.5,.5],(7,1))) for c,side in zip(cats,(0,1))]
    assert all(o['direct_face_known'] for o in observations)
    assert direct_cosine(*observations)==0.
    assert all(not o['direct_face_vector'].flags.writeable for o in observations)
    measured=matrix(*(features([o],fps=30.) for o in observations),details=True)
    legacy=[{k:v for k,v in o.items() if not k.startswith('direct_face_')} for o in observations]
    baseline=matrix(*(features([o],fps=30.) for o in legacy),details=True)
    assert measured['appearance_rate'][0,0]>baseline['appearance_rate'][0,0]
    assert measured['allowed'][0,0]==baseline['allowed'][0,0]


def test_opt_in_flag_reaches_the_whole_interval_consumer_without_authorizing_inference(monkeypatch):
    import crosscut2.focal_appearance as appearance
    nodes,cats,poses,styles=catalogues();count=[];original=appearance.direct_discrepancy
    def observed(*args,**kwargs):count.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(appearance,'direct_discrepancy',observed)
    with pytest.raises(ValueError,match='Direct face'):
        PlayedPosePortPrior(direct_face_appearance=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,direct_face_appearance=True)
    for cat in cats:
        cat.bank['programme_render_clock']=dict(fps=30.,count=54,source_times=nodes/30.)
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4))) for _ in cats]
    result=select(nodes,[c.bank for c in cats],poses,styles,fps=30.,source_events=[[],[]],
        body_states=None,native_ports=native,music=np.array([0.,1.,1.,0.]),prior=prior)
    assert count and result[4]['event_preference']['fixed_connection_prior']['direct_face_appearance_used']
