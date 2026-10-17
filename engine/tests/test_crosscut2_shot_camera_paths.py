import numpy as np
import pytest
from crosscut2.shot_camera_paths import pair_proposals,make_paths,representatives
from crosscut2.fixed_holds import _curve_at


def test_already_close_native_faces_keep_actual_feasible_132_camera_pair():
    a=np.array([.5,.4,.25,.95]);b=np.array([.5,.4,.33,.95])
    pairs=pair_proposals(a,b,caps=(1.35,1.35))
    ca,cb=pairs[0]
    assert ca.zoom==pytest.approx(1.32) and cb.zoom==1.
    assert a[2]*ca.zoom==pytest.approx(b[2]*cb.zoom)
    assert np.allclose(ca.project(a[:2]),cb.project(b[:2]))
    for left,right in pair_proposals([.1,.2,.1,.95],[.85,.7,.4,.95],caps=(1.35,2.)):
        assert 1<=left.zoom<=1.35 and 1<=right.zoom<=2.
        for c in (left,right):
            assert .5/c.zoom<=c.x<=1.-.5/c.zoom and .5/c.zoom<=c.y<=1.-.5/c.zoom


def fixture():
    nodes=np.array([0,20,40,60]);banks=[]
    for size in (.25,.33):
        h=np.zeros((4,2,8,4));h[:,:,0]=[.5,.4,size,.95]
        banks.append(dict(heads=h,zoom_cap=1.35))
    return nodes,banks


def test_feasible_camera_survives_whole_source_paths_without_invented_motion():
    nodes,banks=fixture();paths,report=make_paths(nodes,banks,[np.array([],int)]*2)
    assert any(np.isclose(values[:,0],1.32).all() for _,values in paths[0])
    assert report['paths_per_source']==4
    for source in paths:
        for ticks,values in source:
            actual=_curve_at(ticks,values,np.arange(60))
            assert np.array_equal(actual,np.tile(actual[0],(60,1)))


def test_native_cut_is_only_place_frozen_camera_may_jump_and_no_silent_budget_trim():
    nodes,banks=fixture();banks[1]['heads'][2:,:,0,2]=.20
    paths,_=make_paths(nodes,banks,[np.array([30]),np.array([30])])
    for source in paths:
        for ticks,values in source:
            actual=_curve_at(ticks,values,np.arange(60))
            assert np.allclose(actual[:30],actual[0]) and np.allclose(actual[30:],actual[30])
    with pytest.raises(ValueError,match='budget exceeded'):make_paths(nodes,banks,[np.array([],int)]*2,maximum_candidates=1)
    a,error=representatives([(1.32,.5,.48),(1.335,.5,.48),(1.35,.5,.48)])
    assert len(a)==3 and error==0.


def test_fine_edit_instants_need_not_replace_an_existing_physical_camera_domain():
    old,banks=fixture();new=np.array([0,10,20,30,40,50,60])
    expanded=[]
    for s,bank in enumerate(banks):
        h=np.zeros((len(new),2,8,4));h[np.searchsorted(new,old)]=bank['heads']
        h[[1,3,5],:,0]=[.2 if s==0 else .8,.3,.1 if s==0 else .4,.99]
        expanded.append(dict(bank,heads=h))
    before,_=make_paths(old,banks,[[],[]])
    changed,_=make_paths(new,expanded,[[],[]])
    def signature(paths):return tuple((tuple(t),tuple(v.ravel())) for row in paths for t,v in row)
    assert signature(changed)!=signature(before)  # Reproduces the old k-centre domain change.
    preserved,receipt=make_paths(new,expanded,[[],[]],proposal_nodes=old)
    assert signature(preserved)==signature(before)
    assert receipt['proposal_grid_nodes']==len(old)
    assert receipt['decision_grid_nodes']==len(new)


@pytest.mark.parametrize('invalid',[[0,15,60],[0,20],[False,20,60],[0.,20.,60.],np.array([0,40,20,60],np.uint64)])
def test_camera_proposal_grid_must_be_an_explicit_exact_subset(invalid):
    nodes,banks=fixture()
    with pytest.raises(ValueError,match='proposal grid'):
        make_paths(nodes,banks,[[],[]],proposal_nodes=invalid)


def test_real_programme_consumes_the_explicit_domain_without_restricting_edit_nodes(monkeypatch):
    from crosscut2 import played_pose_port_programme as programme
    from test_crosscut2_pair_camera_options import counterexample
    _,_,_,_,arcs,_=counterexample(False)
    grid=arcs.nodes[[0,1,-1]];original=programme.make_paths;seen=[]
    def recorded(nodes,banks,events,**kwargs):
        seen.append((nodes.copy(),kwargs['proposal_nodes']))
        return original(nodes,banks,events,**kwargs)
    monkeypatch.setattr(programme,'make_paths',recorded)
    prior=programme.PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,camera_proposal_nodes=grid)
    _,_,_,_,receipt,_=programme.select(arcs.nodes,[c.bank for c in arcs.cats],[c.body for c in arcs.cats],
        [c.style for c in arcs.cats],fps=30.,source_events=[[]]*4,body_states=None,
        native_ports=arcs.native,music=np.array([0.,1.,1.,0.]),prior=prior)
    np.testing.assert_array_equal(seen[0][0],arcs.nodes)
    assert seen[0][1]==tuple(grid)
    assert receipt['camera_domain']['proposal_grid_nodes']==len(grid)
    assert receipt['camera_domain']['decision_grid_nodes']==len(arcs.nodes)
    with pytest.raises(ValueError,match='played interval'):
        programme.PlayedPosePortPrior(camera_proposal_nodes=grid)
