import numpy as np
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
