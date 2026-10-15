import numpy as np
import pytest
from crosscut2.event_states import build
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.boundary_port_domain import make
from test_crosscut2_event_holds import inputs


def prepared():
    nodes,banks,poses,styles=inputs()
    states,rows,_=build(nodes,banks,poses,styles,fps=30.,source_events=[[],[]],body_states=None)
    catalogues=[];paths=[]
    for s in range(2):
        raw=[d for state,d in zip(states,rows) if state.source==s and state.family=='native']
        catalogues.append(SourceCatalogue(nodes,banks[s],poses[s],raw,styles[s],[],fps=30.))
        families={}
        for state,d in zip(states,rows):
            if state.source==s:families.setdefault(state.family,(d['ticks'],d['curve']))
        paths.append(list(families.values()))
    return nodes,catalogues,paths


def test_catalogue_uses_physical_head_and_never_creates_motion_on_stationary_source():
    nodes,catalogues,_=prepared();cat=catalogues[0]
    sample=cat.observation(1,0,np.tile([1.,.5,.5],(7,1)))
    assert sample['head_index']==0 and sample['body'][:,0,2].min()>.3
    native=cat.rates(np.arange(54),np.tile([1.,.5,.5],(54,1)))
    curves=np.tile([1.,.5,.5],(2,54,1));curves[1,:,0]=1.1
    batched=cat.rates(np.arange(54),curves)
    assert batched.shape==(2,54)
    np.testing.assert_allclose(native,batched[0])
    assert np.isfinite(batched).all()


def test_proposal_pairs_retained_and_finite_global_domain_bound():
    nodes,catalogues,paths=prepared()
    entry,exit,dense,report=make(nodes,catalogues,paths,fps=30.)
    assert max(map(len,entry))<=16 and max(map(len,exit))<=16
    assert report['paired_endpoints_kept_together']
    for n in range(1,len(nodes)-1):
        left={p for v in exit[n] for p in v.proposal_ids if p[0]=='paired'}
        right={p for v in entry[n] for p in v.proposal_ids if p[0]=='paired'}
        assert left==right and len(left)<=8
    assert np.asarray(dense).shape==(2,4,54,3)


def test_pinned_fourth_face_cannot_borrow_another_face_body_or_descriptor():
    _,catalogues,_=prepared();cat=catalogues[0]
    cat.bank['heads'][1,0,3]=[.5,.3,.12,.99]
    sample=cat.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=3)
    assert sample['head_index']==3 and sample['target_pinned_before_camera']
    assert not sample['body'].any() and not sample['appearance_vector'].any()
    assert sample['displayed_focal_head_index']!=3
    assert not sample['target_is_displayed_focal']
    with pytest.raises(ValueError,match='actually observed'):
        cat.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=7)


def test_actor_pair_retrieval_can_find_non_native_focal_before_the_camera():
    nodes,catalogues,paths=prepared()
    a,b=catalogues
    a.bank['heads'][:,:,0]=[.75,.3,.25,.99]
    b.bank['heads'][:,:,0]=[.30,.3,.25,.99]
    b.bank['heads'][:,:,1]=[.75,.3,.25,.99]
    b.bank['vectors'][:,:,0]=0.;b.bank['vectors'][:,:,0,1]=1.
    b.bank['vectors'][:,:,1]=a.bank['vectors'][:,:,0]
    b.body[:,:,1]=b.body[:,:,0];b.evidence[:,:,1]=b.evidence[:,:,0]
    assert b.observation(1,1,np.tile([1.,.5,.5],(7,1)))['head_index']==0
    entry,exit,_,report=make(nodes,catalogues,paths,fps=30.)
    pair=[p for p in entry[1] if p.source==1 and any(i[0]=='paired' and i[2:4]==(0,1) for i in p.proposal_ids)]
    assert len(pair)==1 and pair[0].observation['head_index']==1
    assert pair[0].observation['target_is_displayed_focal']
    assert report['retained_pairs_using_non_native_focal']>0
    assert len(entry[1])<=4*len(catalogues)+8
