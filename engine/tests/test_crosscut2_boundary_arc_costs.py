import numpy as np
import pytest
from crosscut2.boundary_camera_arc import build as scalar
from crosscut2.boundary_arc_costs import cameras,ArcCosts
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.event_holds import fragment_costs
from crosscut2.boundary_port_domain import make
from test_crosscut2_display_source_catalogue import prepared


@pytest.mark.parametrize('length',[1,5,12,13,14,20,37,38,54,120])
def test_batch_geometry_matches_independent_opus_scalar_and_actual_strips(length):
    base=np.tile([1.25,.5,.5],(length,1));a=np.tile([1.25,.5,.5],(7,1));b=a.copy()
    if length>=14:a[:,0]=1.1;b[:,0]=1.4
    batch,valid=cameras(a[None],b[None],base,cap=1.5)
    k=min(7,length);one=scalar(a[:k],b[-k:],base,cap=1.5)
    assert valid[0] and one is not None
    np.testing.assert_allclose(batch[0],one.values,rtol=0,atol=2e-14)
    np.testing.assert_array_equal(batch[0,:k],a[:k])
    np.testing.assert_array_equal(batch[0,-k:],b[-k:])


def arc_fixture(with_native):
    nodes,cats,paths=prepared()
    entry,exit,dense,_=make(nodes,cats,paths,fps=30.)
    events=[np.array([10,25,40],int),np.array([15,30],int)] if with_native else [np.array([],int)]*2
    native=[]
    for s,xs in enumerate(events):
        heads=np.tile(cats[s].bank['heads'][1],(len(xs),1,1,1))
        if len(xs):heads[-1,1,0,2]*=.8
        native.append(dict(frames=xs,heads=heads,observed=np.ones(len(xs),bool)))
    support=PlayedEventSupport(events,final=54,fps=30.)
    fragments=fragment_costs(nodes,events,fps=30.,support_seconds=12/30.)
    return nodes,entry,exit,ArcCosts(nodes,entry,exit,cats,dense,native,support,fragments,fps=30.),fragments


@pytest.mark.parametrize('with_native',[False,True])
def test_long_prefix_and_short_arc_costs_equal_complete_replayed_paths(with_native):
    nodes,entry,exit,arcs,fragments=arc_fixture(with_native)
    compared=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            values,allowed,codes=arcs(i,j)
            assert np.isfinite(values).all()
            for p,q in zip(*np.nonzero(allowed)):
                source=entry[i][p].source;path=arcs.replay(i,j,p,q,int(codes[p,q]))
                actual=arcs.direct_cost(source,int(nodes[i]),path[None])[0]+fragments(i,j)[source]
                assert values[p,q]==pytest.approx(actual,rel=1e-10,abs=1e-10)
                compared+=1
    assert compared>0


def test_incompatible_twelve_frame_overlap_is_masked_not_silently_clamped():
    base=np.tile([1.2,.5,.5],(12,1));a=base[:7].copy();b=base[-7:].copy();b[:,0]=1.3
    _,valid=cameras(a[None],b[None],base,cap=1.5)
    assert not valid[0]
