import numpy as np
import pytest
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_partial_pose import features,matrix
from crosscut2.physical_display_observations import UNMEASURED
from test_crosscut2_played_partial_pose import observation


def native_rate(a,b,*,measured=True,before=(1.,.5,.5),after=(1.,.5,.5)):
    arcs=ArcCosts.__new__(ArcCosts)
    heads=np.zeros((1,2,8,4));heads[0,0,0]=a;heads[0,1,0]=b
    arcs.native=[dict(frames=np.array([10]),heads=heads,observed=np.array([measured]))]
    return arcs.native_rates(0,np.array([10]),np.array([[before]]),np.array([[after]]))[0,0]


@pytest.mark.parametrize('ratio',[1.,1.2,1.45])
def test_native_and_authored_face_geometry_have_the_same_units_with_unmeasured_pose(ratio):
    a,b=observation(0),observation(1)
    for o in (a,b):o['evidence'][:]=UNMEASURED;o['body'][:]=0.
    b['head'][2]*=ratio
    result=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    assert result['allowed'][0,0] and result['pose_rate'][0,0]==.5
    expected=result['face_rate'][0,0]+result['pose_rate'][0,0]
    assert native_rate(a['head'],b['head'])==pytest.approx(expected)


def test_observed_tenfold_native_scale_jump_is_not_flattened_near_one():
    a=np.array([.5,.4,.04,.99]);b=a.copy();b[2]=.4
    bad=native_rate(a,b);unknown=native_rate(a,b,measured=False)
    assert np.isfinite(bad) and bad>5.
    assert unknown==pytest.approx(1.5) and bad>unknown
    left,right=observation(0),observation(1)
    for o,head in ((left,a),(right,b)):
        o['head']=head;o['body'][:]=0.;o['evidence'][:]=UNMEASURED
    authored=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    assert not authored['allowed'][0,0]  # admission unchanged, risk not clipped
    assert authored['face_rate'][0,0]+authored['pose_rate'][0,0]==pytest.approx(bad)


def test_native_event_uses_the_actual_camera_on_both_played_sides():
    a=np.array([.5,.4,.2,.99]);b=a.copy()
    assert native_rate(a,b)==pytest.approx(.5)
    # Transforming only the outgoing side creates a real scale discrepancy.
    changed=native_rate(a,b,before=(1.35,.5,.5))
    assert changed==pytest.approx(.5+abs(np.log(1.35))/np.log(1.5))
