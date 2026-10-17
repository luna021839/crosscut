import numpy as np
import pytest

from crosscut2.editorial_roles import appearance_importance,event_energy
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_partial_pose import features,matrix
from crosscut2.direct_face_queries import native_fields,query_key
from test_crosscut2_focal_appearance_risk import observation,vector


@pytest.mark.parametrize('bad',[True,0.,.5,8.01,float('nan'),float('inf'),'3',[3]])
def test_appearance_importance_is_an_explicit_finite_editorial_choice(bad):
    with pytest.raises(ValueError,match='appearance importance'):appearance_importance(bad)


def test_importance_does_not_change_evidence_admission_or_reward_a_zero_discrepancy():
    detail=dict(geometry_rate=np.array([.1,.1]),pose_rate=np.array([.4,.4]),
        appearance_rate=np.array([0.,.4]),allowed=np.array([True,False]))
    before=event_energy(detail,.1,fps=30.,point_geometry=True)
    after=event_energy(detail,.1,fps=30.,point_geometry=True,appearance_weight=3.)
    np.testing.assert_allclose(after-before,2.*6/30*detail['appearance_rate'])
    assert before[0]==after[0] and detail['allowed'].tolist()==[True,False]
    assert PlayedPosePortPrior().appearance_weight==1.
    with pytest.raises(ValueError,match='played-interval'):
        PlayedPosePortPrior(appearance_weight=3.)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,appearance_weight=3.)
    assert prior.artifact['focal_appearance_weight']==3.
    assert not prior.artifact['appearance_weight_changes_source_or_cut_admission']


def test_native_and_authored_cuts_consume_the_same_actual_queries_and_importance():
    left,right=observation(0,vector()),observation(1,vector())
    for m,obs in enumerate((left,right)):
        obs.update(direct_face_vector=np.eye(512,dtype=np.float32)[m],direct_face_known=True,
            direct_face_descriptor_family='f'*64)
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    heads=np.zeros((1,2,8,4));heads[0,0,0]=left['head'];heads[0,1,0]=right['head']
    vectors=np.zeros((1,2,3,128));vectors[:,:,0]=vector()
    direct=np.zeros((1,2,8,512),np.float32);direct[0,:,0]=np.eye(512,dtype=np.float32)[:2]
    known=np.zeros((1,2,8),bool);known[0,:,0]=True
    bank=dict(frames=np.array([20]),heads=heads,vectors=vectors,direct_face_vectors=direct,
        direct_face_known=known,direct_face_descriptor_family='f'*64)
    arcs=ArcCosts.__new__(ArcCosts);arcs.native=[bank];arcs.fps=30.
    arcs.point_geometry=True;arcs.appearance_weight=3.;arcs.direct_face_appearance=True
    camera=np.tile([1.,.5,.5],(1,1,1))
    pose,point=arcs.native_rates(0,np.array([20]),camera,camera,components=True)
    authored=event_energy(detail,.1,fps=30.,point_geometry=True,appearance_weight=3.)
    np.testing.assert_allclose(.1*pose+point,authored)
    assert float(point[0,0])==pytest.approx(.15)
    # Having cached fields alone must not enable the optional consumer.
    arcs.direct_face_appearance=False
    _,off=arcs.native_rates(0,np.array([20]),camera,camera,components=True)
    assert float(off[0,0])==pytest.approx(0.)


def test_native_queries_join_exact_physical_ordinals_and_unknown_events_stay_unknown():
    head=np.array([.5,.4,.1,.99],np.float32)
    port=dict(contract='physical_native_event_registry_v1',heads=np.tile(head,(2,2,8,1)),
        observed=np.array([True,False]),native_event_keys=[[11],[20,21]])
    values={query_key(11,head):(np.eye(512,dtype=np.float32)[0],True),
        query_key(0,head):(np.eye(512,dtype=np.float32)[1],True)}
    vectors,known,native=native_fields(port,values)
    assert native[0].tolist()==[10,11] and known[0,1].all() and not known[0,0].any()
    assert not known[1].any() and not vectors[1].any()
    assert not vectors.flags.writeable and not known.flags.writeable
    port['observed'][1]=True
    with pytest.raises(ValueError,match='one exact physical cut ordinal'):native_fields(port,values)


def test_old_numeric_native_snapshots_need_actual_cut_clock_not_a_relabelled_contract():
    head=np.array([.5,.4,.1,.99],np.float32)
    port=dict(frames=np.array([30]),heads=np.tile(head,(1,2,8,1)),
        observed=np.array([True]),native_event_keys=[[11]])
    values={query_key(11,head):(np.eye(512,dtype=np.float32)[0],True)}
    with pytest.raises(ValueError,match='physical exact-adjacent'):native_fields(port,values)
    bound=dict(native_event_rows=[dict(output_frame=30,source_native_first_incoming=11,
        exposed_native_ordinals=[10,11])])
    _,known,natives=native_fields(port,values,bound=bound)
    assert known[0,1].all() and natives.tolist()==[[10,11]] and 'contract' not in port
    bound['native_event_rows'][0]['exposed_native_ordinals']=[9,11]
    with pytest.raises(ValueError,match='actual physical cut/clock'):native_fields(port,values,bound=bound)
