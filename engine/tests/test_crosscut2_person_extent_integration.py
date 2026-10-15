import copy
import numpy as np
import pytest
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_partial_pose import features,matrix
from crosscut2.physical_display_observations import UNMEASURED
from crosscut2.person_extent import height_interval,interval_gap
from test_crosscut2_played_partial_pose import observation
from test_crosscut2_boundary_arc_costs import arc_fixture


def boxes(height):return np.array([[.3,.05,.7,.05+height,.9]],np.float32)


def authored(*,known=False,right_boxes=True,enabled=True,left_zoom=1.):
    a,b=observation(0),observation(1)
    for o in (a,b):
        o['body'][:]=0.;o['evidence'][:]=UNMEASURED
        if not known:o['head'][:]=0.
    a['person_rectangles']=boxes(.2);b['person_rectangles']=boxes(.8) if right_boxes else None
    a['camera_strip'][:,0]=left_zoom
    return matrix(features([a],fps=30.,person_extents=enabled),
        features([b],fps=30.,person_extents=enabled),details=True)


def native(*,known=False,right_boxes=True,enabled=True,left_zoom=1.):
    arcs=ArcCosts.__new__(ArcCosts);arcs.person_extents=enabled
    heads=np.zeros((1,2,8,4));heads[:,:,0]=[.5,.3,.25,.99] if known else [0,0,0,0]
    arcs.native=[dict(frames=np.array([10]),heads=heads,observed=np.array([known]),
        person_rectangles={10:(boxes(.2),boxes(.8) if right_boxes else None)})]
    return arcs.native_rates(0,np.array([10]),np.array([[[left_zoom,.5,.5]]]),np.array([[[1.,.5,.5]]]))[0,0]


def test_unknown_native_authored_same_extent_risk_without_new_admission_gate():
    d=authored()
    assert d['allowed'][0,0] and not d['measured_face'][0,0]
    assert d['face_rate'][0,0]==1. and d['pose_rate'][0,0]==.5
    assert d['person_extent_measured'][0,0] and d['geometry_rate'][0,0]>3.
    assert native()==pytest.approx(d['geometry_rate'][0,0]+.5)


@pytest.mark.parametrize('known',[False,True])
@pytest.mark.parametrize('right_boxes',[False,True])
def test_known_face_and_missing_extent_preserve_original_contract(known,right_boxes):
    d=authored(known=known,right_boxes=right_boxes)
    old=authored(known=known,right_boxes=right_boxes,enabled=False)
    if known or not right_boxes:
        assert np.array_equal(d['geometry_rate'],old['face_rate'])
        assert native(known=known,right_boxes=right_boxes)==native(known=known,right_boxes=right_boxes,enabled=False)
    assert np.array_equal(d['allowed'],old['allowed'])


def test_actual_endpoint_camera_changes_same_shared_risk_not_pose_evidence():
    d=authored(left_zoom=1.35)
    assert native(left_zoom=1.35)==pytest.approx(d['geometry_rate'][0,0]+.5)
    assert d['pose_rate'][0,0]==.5 and d['allowed'][0,0]
    assert not (d['phase_evidence_kind']>0).any()


def test_extent_native_prefix_patch_and_direct_replay_stay_exact():
    nodes,entry,exit,old,fragments=arc_fixture(True)
    native_banks=copy.deepcopy(old.native)
    for bank in native_banks:
        bank['observed'][:]=False;bank['heads'][:]=0.
        bank['person_rectangles']={int(t):(boxes(.2),boxes(.8)) for t in bank['frames']}
    arcs=ArcCosts(nodes,entry,exit,old.cats,old.dense,native_banks,old.support,fragments,fps=30.,person_extents=True)
    assert any((rates>1.5).any() for _,rates,_,_ in arcs.native_tables)
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            values,allowed,codes=arcs(i,j)
            for p,q in zip(*np.nonzero(allowed)):
                s=entry[i][p].source;path=arcs.replay(i,j,p,q,int(codes[p,q]))
                actual=arcs.direct_cost(s,int(nodes[i]),path[None])[0]+fragments(i,j)[s]
                assert values[p,q]==pytest.approx(actual,rel=1e-10,abs=1e-10)
