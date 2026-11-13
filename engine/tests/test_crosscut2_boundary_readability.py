import numpy as np
import pytest
from crosscut2.boundary_readability import pair, native_fields
from crosscut2.editorial_roles import event_energy
from crosscut2.played_partial_pose import features, matrix
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from test_crosscut2_source_clock_action import catalogue, obs
from test_crosscut2_boundary_arc_costs import arc_fixture


def test_intrinsic_original_detail_not_camera_size_role_or_body_competition():
    cat=catalogue();a,b=obs(cat,0),obs(cat,1)
    for o,q in ((a,.9),(b,.3)):
        o.update(target_native_ROI_readability=q,displayed_native_quality_measured=True,
            displayed_native_quality_proxy=.01)
    detail=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    assert detail['endpoint_readability_rate'][0,0]==pytest.approx(.7)
    assert detail['endpoint_readability_measured'][0,0]
    before=event_energy(detail,.01,fps=30.,point_geometry=True)
    after=event_energy(detail,.01,fps=30.,point_geometry=True,boundary_readability=True)
    assert (after-before)[0,0]==pytest.approx(.2*.7)
    # The point cost cannot shrink when adjacent cuts shorten the pose window.
    after2=event_energy(detail,.4,fps=30.,point_geometry=True,boundary_readability=True)
    before2=event_energy(detail,.4,fps=30.,point_geometry=True)
    assert (after2-before2)[0,0]==pytest.approx(.2*.7)


def test_missing_does_not_fabricate_detail_or_add_an_admission_gate():
    risk,known=pair(np.array([1.,1.,.2]),[1.,0.,1.],np.array([True,True,True]),np.array([True,False,False]))
    np.testing.assert_allclose(risk,[0.,.5,.8])
    assert known.tolist()==[True,False,False]
    cat=catalogue();a,b=obs(cat,0),obs(cat,1)
    old=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    for o in (a,b):o.update(target_native_ROI_readability=.01,displayed_native_quality_measured=True)
    new=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    np.testing.assert_array_equal(old['allowed'],new['allowed'])
    for key in ('geometry_rate','pose_rate','appearance_rate'):
        np.testing.assert_array_equal(old[key],new[key])
    np.testing.assert_array_equal(event_energy(old,.2,fps=30.,point_geometry=True),
        event_energy(new,.2,fps=30.,point_geometry=True))


def test_clear_moving_match_keeps_existing_motion_benefit_and_is_not_forced_static():
    base=dict(geometry_rate=0.,appearance_rate=.1,pose_rate=.2,endpoint_readability_rate=0.)
    moving=dict(base,focal_image_motion_relative_cost=-.2)
    assert event_energy(moving,.2,fps=30.,point_geometry=True,boundary_readability=True)<event_energy(
        base,.2,fps=30.,point_geometry=True,boundary_readability=True)


def fixture():
    head=np.array([.5,.4,.2,.99],np.float32)
    heads=np.zeros((1,2,8,4),np.float32);heads[:,:,0]=head
    metrics=np.zeros_like(heads);metrics[:,:,0]=[.1,.25,0.,150.]
    valid=np.zeros(heads.shape[:-1],bool);valid[:,:,0]=True
    bank=dict(heads=heads,native_indices=np.array([[10,11]]),native_readability=metrics,native_readability_valid=valid)
    port=dict(heads=heads.copy(),native_indices=np.array([[10,11]]),observed=np.array([True]))
    return bank,port


def test_native_requires_same_actual_box_and_ordinal_not_nearest_frame_or_face():
    bank,port=fixture();before=bank['native_readability'].copy()
    values,known=native_fields(bank,port)
    assert known.sum()==2 and values[0,0,0]<1.
    port['native_indices']+=1
    values,known=native_fields(bank,port)
    assert known.sum()==1 and known[0,0,0]
    port['heads'][0,0,0,0]+=1e-6
    values,known=native_fields(bank,port)
    assert not known.any() and (values==.5).all()
    np.testing.assert_array_equal(before,bank['native_readability'])
    del port['native_indices']
    assert not native_fields(bank,port)[1].any()


def test_native_and_authored_share_point_units_and_complete_arc_replay():
    nodes,entry,exit,old,fragments=arc_fixture(True)
    from crosscut2.boundary_arc_costs import ArcCosts
    arcs=ArcCosts(nodes,entry,exit,old.cats,old.dense,old.native,old.support,fragments,
        fps=30.,point_geometry=True,boundary_readability=True)
    paths=np.asarray(old.dense[0]);events=old.support.events[0]
    _,newpoints=arcs.native_rates(0,events,paths[:,events-1],paths[:,events],components=True)
    # The old fixture has no ROI-native binding: explicit UNKNOWN .5, not 0.
    old.point_geometry=True
    _,points=old.native_rates(0,events,paths[:,events-1],paths[:,events],components=True)
    np.testing.assert_allclose(newpoints-points,.2*.5)
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            costs,allowed,codes=arcs(i,j)
            for p,q in zip(*np.nonzero(allowed)):
                path=arcs.replay(i,j,p,q,int(codes[p,q]));s=entry[i][p].source
                assert costs[p,q]==pytest.approx(arcs.direct_cost(s,int(nodes[i]),path[None])[0]+fragments(i,j)[s])


def test_option_requires_real_interval_consumer_and_preserves_default():
    assert not PlayedPosePortPrior().boundary_readability
    with pytest.raises(ValueError,match='actual played-interval'):
        PlayedPosePortPrior(boundary_readability=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,boundary_readability=True)
    assert prior.artifact['endpoint_native_ROI_readability_cost']
    assert not prior.artifact['endpoint_readability_changes_source_or_cut_admission']


def test_cli_rejects_unused_option_before_reading_any_source(monkeypatch,tmp_path):
    from crosscut2.stage_mix import main
    argv=['stage_mix','--boundary-readability']
    for key in ('source-plan','analysis','cache','output','models'):
        argv.extend(['--'+key,str(tmp_path/key)])
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='endpoint readability requires'):
        main()


def test_exact_native_observation_pays_identical_authored_boundary_cost():
    from crosscut2.boundary_arc_costs import ArcCosts
    bank,port=fixture();port['frames']=np.array([18])
    arcs=ArcCosts.__new__(ArcCosts)
    arcs.native=[port];arcs.native_readability=[native_fields(bank,port)]
    arcs.fps=30.;arcs.boundary_readability=True;arcs.point_geometry=True
    camera=np.tile([1.,.5,.5],(2,1,1))
    _,point=arcs.native_rates(0,np.array([18]),camera,camera,components=True)
    arcs.boundary_readability=False
    _,old=arcs.native_rates(0,np.array([18]),camera,camera,components=True)
    q,valid=native_fields(bank,port)
    risk,_=pair(q[0,0,0],q[0,1,0],valid[0,0,0],valid[0,1,0])
    np.testing.assert_allclose(point-old,.2*risk)
