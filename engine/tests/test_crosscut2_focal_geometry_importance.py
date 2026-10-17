import numpy as np
import pytest
from crosscut2.editorial_roles import event_energy,geometry_importance
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_partial_pose import features,matrix
from test_crosscut2_focal_appearance_risk import observation,vector


@pytest.mark.parametrize('bad',[True,0.,.5,8.01,float('nan'),float('inf'),'3',[3]])
def test_importance_requires_an_explicit_finite_editorial_value(bad):
    with pytest.raises(ValueError,match='geometry importance'):geometry_importance(bad)


def test_zero_geometry_match_gains_no_new_reward_or_pacing_bonus():
    detail=dict(geometry_rate=np.array([0.,1.,2.]),pose_rate=np.full(3,.4),appearance_rate=np.full(3,.2))
    a=event_energy(detail,.1,fps=30.,point_geometry=True)
    b=event_energy(detail,.1,fps=30.,point_geometry=True,geometry_weight=3.)
    assert a[0]==b[0] and np.all(b>=a)
    np.testing.assert_allclose(b-a,2.*6/30*detail['geometry_rate'])
    # No change to partial body, identity, flow, or admission measurements.
    assert detail['pose_rate'].tolist()==[.4]*3


def test_native_and_authored_size_position_discrepancies_use_the_same_importance():
    left,right=observation(0,vector()),observation(1,vector())
    right['head']=np.array([.65,.4,.12,.99])
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    heads=np.zeros((1,2,8,4));heads[0,0,0]=left['head'];heads[0,1,0]=right['head']
    vectors=np.zeros((1,2,3,128));vectors[:,:,0]=vector()
    arcs=ArcCosts.__new__(ArcCosts);arcs.native=[dict(frames=np.array([20]),heads=heads,vectors=vectors)]
    arcs.point_geometry=True;arcs.geometry_weight=3.;arcs.fps=30.
    camera=np.tile([1.,.5,.5],(1,1,1))
    pose,point=arcs.native_rates(0,np.array([20]),camera,camera,components=True)
    authored=event_energy(detail,.1,fps=30.,point_geometry=True,geometry_weight=3.)
    np.testing.assert_allclose(.1*pose+point,authored)
    assert detail['allowed'].dtype==bool  # Increasing importance is never a new admission rule.


def test_new_importance_is_opt_in_and_recorded_by_the_actual_consumer():
    assert PlayedPosePortPrior().geometry_weight==1.
    with pytest.raises(ValueError,match='played-interval'):
        PlayedPosePortPrior(geometry_weight=3.)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,geometry_weight=3.)
    assert prior.artifact['focal_geometry_weight']==3.
    assert not prior.artifact['geometry_weight_changes_source_or_cut_admission']
