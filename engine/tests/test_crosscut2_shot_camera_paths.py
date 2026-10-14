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
