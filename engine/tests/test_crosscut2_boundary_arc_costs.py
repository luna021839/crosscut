import numpy as np
import pytest
from crosscut2.boundary_camera_arc import build as scalar
from crosscut2.boundary_arc_costs import cameras,ArcCosts
from crosscut2.boundary_arc_costs import whole_viewports
from crosscut2.boundary_arc_costs import camera_samples,camera_samples_many
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.event_holds import fragment_costs
from crosscut2.boundary_port_domain import make
from test_crosscut2_display_source_catalogue import prepared


@pytest.mark.parametrize('length',[12,14,20,37,38,39,90,173])
def test_sparse_batch_samples_are_bit_exact_with_existing_scalar_codes(length):
    base=np.tile([1.2,.5,.5],(length,1))
    entry=np.tile([1.2,.5,.5],(3,7,1));leave=entry.copy()
    if length>=14:
        entry[:,0:,0]=np.array([1.15,1.2,1.25])[:,None]
        leave[:,0:,0]=np.array([1.2,1.25,1.3])[:,None]
    at=np.unique(np.r_[0,min(6,length-1),np.arange(7,length-7,3),length-1]).astype(int)
    for code in range(6):
        if code==5 and length<14:continue
        values=camera_samples_many(entry,leave,base,at,code=code)
        for k in range(len(entry)):
            np.testing.assert_array_equal(values[k],camera_samples(entry[k],leave[k],base,at,code=code))


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


def arc_fixture(with_native,**kwargs):
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
    return nodes,entry,exit,ArcCosts(nodes,entry,exit,cats,dense,native,support,fragments,fps=30.,**kwargs),fragments


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


@pytest.mark.parametrize('length',[14,20,38,74])
def test_whole_viewport_batch_matches_scalar_without_forced_zoom_pulse(length):
    from crosscut2.subject_moment_sequence import hold_curve
    a=np.tile([1.35,.52,.5],(7,1));b=np.tile([1.2,.5,.5],(7,1))
    result,valid=whole_viewports(a[None],b[None],length=length,cap=1.35)
    one=hold_curve(0,length,([0,length-1],[[1.,.5,.5]]*2),a,b,cap=1.35,family='whole_viewport')
    assert valid[0] and one is not None
    np.testing.assert_allclose(result[0],one.values,rtol=0.,atol=2e-14)


def test_adjacent_opportunity_whole_camera_is_scored_before_DP_and_replayed_exactly():
    nodes,entry,exit,arcs,fragments=arc_fixture(False)
    arcs.opportunity_episodes=np.array([0,0,1,2])
    total=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            cost,allowed,codes=arcs(i,j)
            for p,q in zip(*np.nonzero(allowed)):
                path=arcs.replay(i,j,p,q,int(codes[p,q]));source=entry[i][p].source
                assert cost[p,q]==pytest.approx(arcs.direct_cost(source,int(nodes[i]),path[None])[0]+fragments(i,j)[source])
                total+=int(codes[p,q]==5)
    assert arcs.whole_viewport_paths_scored>0


@pytest.mark.parametrize('with_native',[False,True])
def test_all_context_camera_codes_equal_actual_replay_without_rejected_debt(with_native):
    nodes,entry,exit,arcs,fragments=arc_fixture(with_native,preserve_camera_alternatives=True)
    arcs.opportunity_episodes=np.array([0,0,1,2]);compared=0
    assert arcs.readouts is None
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            values,valid=arcs.camera_alternatives(i,j)
            best,allowed,codes=arcs(i,j)
            assert np.isfinite(values).all() and values.shape==valid.shape==best.shape+(6,)
            np.testing.assert_array_equal(valid.any(-1),allowed)
            for p,q in zip(*np.nonzero(allowed)):
                assert best[p,q]==pytest.approx(values[p,q][valid[p,q]].min())
            for p,q,code in zip(*np.nonzero(valid)):
                path=arcs.replay(i,j,p,q,int(code));source=entry[i][p].source
                actual=arcs.direct_cost(source,int(nodes[i]),path[None])[0]+fragments(i,j)[source]
                assert values[p,q,code]==pytest.approx(actual,rel=1e-10,abs=1e-10)
                frames=np.arange(nodes[i],nodes[j])
                # Bit exactness matters for points exactly on the viewport.
                np.testing.assert_array_equal(arcs.sample_camera(i,j,p,q,int(code),frames),path)
                compared+=1
    assert compared>0
