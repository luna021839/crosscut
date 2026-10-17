from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2 import detailed_pose as fine
from crosscut2.subject_window_bank import SubjectWindow,save
from crosscut2.partial_pose import face_proposals,face_guided_hypothesis
from crosscut2.subject_action_match import SubjectClock


def fixture(tmp_path):
    head=np.array([.5,.3,.2,.99]);raw=np.zeros((14,133,3),np.float32)
    raw[:,:5,:2]=[[.5,.3],[.47,.28],[.53,.28],[.43,.31],[.57,.31]];raw[:,:5,2]=.9
    raw[:,[5,6,7,8,9,10],:2]=[[.43,.43],[.57,.43],[.4,.5],[.6,.5],[.4,.58],[.6,.58]]
    raw[:,[5,6,7,8,9,10],2]=.9
    for root,x in ((91,.4),(112,.6)):
        raw[:,root,:2]=[x,.58];raw[:,root,2]=.9
        for finger in range(5):
            for joint in range(4):raw[:,root+1+finger*4+joint]=[x+.006*(finger-2),.58-.008*(joint+1),.9]
    body=[];support=[]
    for r in raw:
        p=face_guided_hypothesis(SimpleNamespace(reason='observed',raw_keypoints_xyc=r[:17]),head,np.zeros((0,4)))
        body.append(p.points);support.append(p.supported)
    window=SubjectWindow(0,'source-sha','actual-track',np.arange(20,34),np.tile(head,(14,1)),
        np.tile(np.eye(128)[0],(14,1)),np.array(body),np.array(support),np.ones(14,bool),tuple(np.zeros((0,4)) for _ in raw))
    path=tmp_path/'actual-window.npz';save(window,path,provenance={'producer':'historical-unchanged-producer'})
    sidecar=path.with_name(path.stem+'-raw133.npz')
    data=dict(raw133=raw,native=window.ordinals,rois=np.tile(face_proposals(head)[0],(14,1)),
        raster_sha256=['0'*64]*14,producer='historical-unchanged-producer')
    np.savez_compressed(sidecar,**data)
    return window,path,sidecar,data


def observation(w,side,*,camera=(1.,.5,.5)):
    context=dict(window=w,at=np.arange(14),clock=SubjectClock(w.source,w.binding,np.arange(-7,7),w.ordinals,0,30.,'clock'),
        detailed_points=w.detailed_points,detailed_support=w.detailed_support,same_shot=np.ones(14,bool),
        supported=w.support,contiguous=np.ones(13,bool))
    return dict(side=side,camera_strip=np.tile(camera,(7,1)),dense_subject_context=context)


def test_reads_bound_raw_without_changing_original_body_or_old_cache_bytes(tmp_path):
    original,path,_,_=fixture(tmp_path);before=path.read_bytes();actual=fine.attach(original,path)
    assert isinstance(actual,fine.DetailedWindow) and actual.detailed_support.all()
    np.testing.assert_array_equal(actual.body,original.body);assert path.read_bytes()==before
    assert not actual.detailed_points.flags.writeable and not actual.detailed_support.flags.writeable
    value,seen,channels,measured=fine.matrix([fine.describe(observation(actual,0))],[fine.describe(observation(actual,1))])
    np.testing.assert_allclose(channels,.25);assert seen.all() and measured.all() and value[0,0]==pytest.approx(.25)


def test_fingers_change_cost_with_identical_body_wrist_and_face_endpoints(tmp_path):
    original,path,_,_=fixture(tmp_path);a=fine.attach(original,path);points=a.detailed_points.copy()
    points[7,6:26,1]-=.12;b=replace(a,detailed_points=points)
    same=fine.matrix([fine.describe(observation(a,0))],[fine.describe(observation(a,1))])[0][0,0]
    different=fine.matrix([fine.describe(observation(a,0))],[fine.describe(observation(b,1))])[0][0,0]
    np.testing.assert_array_equal(a.body,b.body);assert different>same


def test_opposite_finger_directions_are_not_diluted_by_the_matching_face_and_other_hand(tmp_path):
    original,path,_,_=fixture(tmp_path);a=fine.attach(original,path);points=a.detailed_points.copy()
    points[7,6:26,1]=2*points[7,5,1]-points[7,6:26,1]
    b=replace(a,detailed_points=points)
    risk,seen,channels,_=fine.matrix([fine.describe(observation(a,0))],[fine.describe(observation(b,1))])
    assert seen[0,0] and channels[0,0,1]>.6 and risk[0,0]>.6
    assert channels[0,0,0]==pytest.approx(.25) and channels[0,0,2]==pytest.approx(.25)


def test_proposal_configuration_does_not_punish_a_screen_displacement_that_camera_can_solve(tmp_path):
    original,path,_,_=fixture(tmp_path);w=fine.attach(original,path)
    a=fine.describe(observation(w,0));b=fine.describe(observation(w,1));b['xy']=b['xy']+[.3,0.]
    assert fine.matrix([a],[b])[0][0,0]>.25
    assert fine.matrix([a],[b],configuration_only=True)[0][0,0]==pytest.approx(.25)


def test_missing_hand_is_unknown_and_cropping_it_cannot_create_a_perfect_match(tmp_path):
    original,path,_,_=fixture(tmp_path);a=fine.attach(original,path)
    left=fine.describe(observation(a,0));right=fine.describe(observation(a,1))
    right['known'][5:26]=False
    _,_,channels,seen=fine.matrix([left],[right]);assert channels[0,0,1]==.5 and not seen[0,0,1]
    right=fine.describe(observation(a,1));right['visible'][5:26]=False
    _,_,channels,seen=fine.matrix([left],[right]);assert channels[0,0,1]==.75 and seen[0,0,1]
    left['visible'][5:26]=False
    assert fine.matrix([left],[right])[2][0,0,1]==.5


@pytest.mark.parametrize('change',['producer','roi','body','native'])
def test_wrong_producer_face_roi_body_or_native_never_relabels_a_raw_snapshot(tmp_path,change):
    original,path,sidecar,data=fixture(tmp_path)
    if change=='producer':data['producer']='other-producer'
    if change=='roi':data['rois'][:,0]+=.01
    if change=='body':data['raw133'][:,5,0]+=.01
    if change=='native':data['native']=data['native']+100
    np.savez_compressed(sidecar,**data)
    with pytest.raises(ValueError):fine.attach(original,path)


def test_optional_cost_does_not_invent_pose_or_add_an_unknown_cut_tax(tmp_path):
    from crosscut2.played_partial_pose import features,matrix
    from crosscut2.editorial_roles import event_energy
    from test_crosscut2_played_partial_pose import observation as body_observation
    a,b=body_observation(0),body_observation(1)
    old=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    new=matrix(features([a],fps=30.,detailed_pose=True),features([b],fps=30.,detailed_pose=True),details=True)
    assert new['detailed_pose_relative_cost'][0,0]==0. and not new['detailed_pose_measured'][0,0]
    np.testing.assert_array_equal(old['allowed'],new['allowed'])
    np.testing.assert_array_equal(event_energy(old,.2,fps=30.,point_geometry=True),event_energy(new,.2,fps=30.,point_geometry=True))


def test_actual_played_head_hands_reach_cost_without_erasing_body_or_role_interpretation(tmp_path):
    from crosscut2.played_partial_pose import features,matrix
    from crosscut2.editorial_roles import event_energy,apply
    from test_crosscut2_played_partial_pose import observation as body_observation
    original,path,_,_=fixture(tmp_path);w=fine.attach(original,path)
    a,b=body_observation(0),body_observation(1)
    a.update(observation(w,0));b.update(observation(w,1))
    old=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    new=matrix(features([a],fps=30.,detailed_pose=True),features([b],fps=30.,detailed_pose=True),details=True)
    assert new['detailed_pose_measured'][0,0]
    np.testing.assert_array_equal(old['allowed'],new['allowed'])
    np.testing.assert_array_equal(old['pose_rate'],new['pose_rate'])
    for strength in (0.,1.):
        assert (event_energy(apply(new,strength),.2,fps=30.,point_geometry=True)
            -event_energy(apply(old,strength),.2,fps=30.,point_geometry=True))[0,0]==pytest.approx(-.05)


def test_dense_catalogue_retains_exact_points_and_masks_conflicting_detail_only():
    from dataclasses import fields
    from crosscut2.dense_subject_catalogue import DenseSubjects
    from test_crosscut2_dense_subject_catalogue import prepared
    cat,w,indices=prepared();points=np.zeros((len(w.ordinals),47,3));points[...,2]=.9
    support=np.tile(w.associated[:,None],(1,47))
    a=fine.DetailedWindow(**{f.name:getattr(w,f.name) for f in fields(SubjectWindow)},detailed_points=points,detailed_support=support)
    changed=points.copy();changed[:,6,0]=.1
    b=replace(a,track='other-actual-query',detailed_points=changed)
    cat.subjects=DenseSubjects(source=0,binding='0',indices=indices,windows=[a,b],bank=cat.bank,fps=30.)
    sample=cat.subjects.sample(1,0,0)
    assert sample is not None and not sample['detailed_support'][:,6].any()
    assert sample['detailed_support'][:,:6].any()
    np.testing.assert_array_equal(sample['window'].body,w.body)


def test_detailed_pose_configuration_is_explicit_and_has_a_real_interval_consumer(monkeypatch):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    from crosscut2.stage_mix import main
    assert not PlayedPosePortPrior().detailed_pose
    with pytest.raises(ValueError,match='actual played interval'):PlayedPosePortPrior(detailed_pose=True)
    p=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,detailed_pose=True)
    assert p.artifact['detailed_head_hand_pose'] and not p.artifact['detailed_pose_changes_admission']
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache','unused','--output','unused','--models','unused','--detailed-pose'])
    with pytest.raises(ValueError,match='actual played interval'):main()
