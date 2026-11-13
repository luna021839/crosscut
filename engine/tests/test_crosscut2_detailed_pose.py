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


def test_opposed_measured_hand_bounds_match_bonus_without_a_new_bonus_or_source_veto():
    detail=dict(detailed_pose_channels=np.array([[[.25,.75,.25]]]),
        detailed_pose_channel_measured=np.ones((1,1,3),bool),
        matched_action_change_density=np.array([[.6]]),actual_played_match_advantage=np.array([[.8]]),allowed=np.array([[True]]))
    result=fine.bound_connection_credit(detail)
    assert result['matched_action_change_density'][0,0]==0. and result['allowed'][0,0]
    assert detail['matched_action_change_density'][0,0]==.6
    missing=dict(detail,detailed_pose_channel_measured=np.zeros((1,1,3),bool))
    assert fine.bound_connection_credit(missing)['matched_action_change_density'][0,0]==.6
    matching=dict(detail,detailed_pose_channels=np.full((1,1,3),.25))
    assert fine.bound_connection_credit(matching)['matched_action_change_density'][0,0]==.6
    weak=dict(detail,detailed_pose_channels=np.array([[[.25,.4,.25]]]))
    assert fine.bound_connection_credit(weak)['matched_action_change_density'][0,0]==pytest.approx(.4)


def test_articulated_body_recomputation_cannot_restore_a_contradicted_finger_bonus():
    from crosscut2.played_interval_utility import articulated_seam_evidence
    detail=dict(pose_rate=np.array([[.1]]),matched_action_change_density=np.array([[.6]]),
        matched_action_evidence_density=np.array([[.6]]),detailed_pose_channels=np.array([[[.25,.75,.25]]]),
        detailed_pose_channel_measured=np.ones((1,1,3),bool))
    result=articulated_seam_evidence(detail,[{}],[{}])
    assert result['matched_action_change_density'][0,0]==0.


@pytest.mark.parametrize('measured',[False,True])
def test_raw_opportunity_and_pair_pruning_share_configuration_credit_cap(monkeypatch,measured):
    from crosscut2 import boundary_port_domain as ports
    from test_crosscut2_motion_pair_candidates import motion_catalogues
    nodes,cats,paths=motion_catalogues(False)
    original=ports.matrix;seen=[];fitted=[]
    def body_credit(*a,**kw):
        result=original(*a,**kw);shape=result['pose_rate'].shape
        return dict(result,pose_rate=np.full(shape,.1),
            matched_action_change_density=np.full(shape,.6),matched_action_evidence_density=np.full(shape,.6))
    def configuration(left,right,*,configuration_only=False):
        assert configuration_only  # Native placement can be corrected by a later crop.
        shape=(len(left),len(right));channels=np.full(shape+(3,),.25 if measured else .5)
        if measured:channels[...,1]=.75  # Contradicted fingers despite the matching body.
        return channels.max(-1),np.full(shape,measured),channels,np.full(shape+(3,),measured)
    class Opportunities:
        def observe(self,n,left,right,detail,*sources):seen.append(detail)
        # This fixture verifies raw configuration pruning only. Actual-view
        # re-evaluation is exercised with real Opportunities in its own test.
        def observe_played_pair(self,n,left,right,**kw):fitted.append((n,left.source,right.source))
        def density(self,*a):return 0.
    monkeypatch.setattr(ports,'matrix',body_credit)
    monkeypatch.setattr(fine,'matrix',configuration)
    _,_,_,receipt=ports.make(nodes,cats,paths,fps=30.,detailed_pose=True,
        articulated_actions=True,opportunities=Opportunities(),retain_pair_cameras=True)
    assert seen and fitted and all(a!=b for _,a,b in fitted)
    for detail in seen:
        np.testing.assert_allclose(detail['matched_action_change_density'],0. if measured else .6)
        assert 'detailed_pose_action_credit_cap' in detail
        np.testing.assert_allclose(detail['detailed_pose_action_credit_cap'],0. if measured else 1.)
    assert receipt['detailed_configuration_bounds_raw_opportunity_credit']


def test_fine_moment_reads_only_exact_physical_endpoint_and_never_retimes(tmp_path):
    original,path,_,_=fixture(tmp_path);w=fine.attach(original,path)
    for side in (0,1):
        o=observation(w,side);actual=fine.describe_moment(w,o['dense_subject_context']['clock'],0,side,[1.,.5,.5])
        expected=fine.describe(o)
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
        assert not fine.describe_moment(original,o['dense_subject_context']['clock'],0,side,[1.,.5,.5])['known'].any()
        wrong=replace(o['dense_subject_context']['clock'],binding='other-source')
        with pytest.raises(ValueError,match='source binding'):fine.describe_moment(w,wrong,0,side,[1.,.5,.5])
        shifted=replace(o['dense_subject_context']['clock'],phase_frames=1)
        with pytest.raises(ValueError,match='retime'):fine.describe_moment(w,shifted,0,side,[1.,.5,.5])
