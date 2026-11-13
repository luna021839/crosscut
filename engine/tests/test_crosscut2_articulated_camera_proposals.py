import numpy as np
import pytest
from crosscut2.articulated_camera_proposals import proposal,prepare,prepared_proposal
from crosscut2.shot_camera_paths import pair_proposals,make_paths
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.boundary_port_domain import make
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from test_crosscut2_event_holds import inputs


def observations(ratio=1.2):
    body=np.array([[.4,.44,.9],[.6,.44,.9],[.35,.55,.9],[.65,.55,.9],
        [.3,.58,.9],[.7,.58,.9],[.43,.75,.9],[.57,.75,.9]])
    head=np.array([.5,.3,.1,.99]);result=[]
    for side,scale in ((0,1.),(1,ratio)):
        h=head.copy();h[:2]=.5+scale*(h[:2]-.5)
        b=body.copy();b[:,:2]=.5+scale*(b[:,:2]-.5)
        result.append(dict(side=side,head=h,body=np.tile(b,(2,1,1)),
            evidence=np.ones((2,8),np.uint8),ownership_ambiguity=np.zeros((2,8),bool)))
    return result


def error(pair,observations):
    left,right=observations;a,b=pair
    return np.linalg.norm(a.project(left['body'][1,:6,:2])-b.project(right['body'][0,:6,:2]))


def test_observed_arm_scale_adds_a_view_not_present_in_common_face_size_proposals():
    samples=observations();original=[s['body'].copy() for s in samples]
    pair=proposal(*samples,caps=(2.,2.))
    assert pair[0].zoom==pytest.approx(1.2) and pair[1].zoom==1.
    assert error(pair,samples)<1e-12
    old=pair_proposals(samples[0]['head'],samples[1]['head'],caps=(2.,2.))
    assert all(error(p,samples)>.03 for p in old)
    for before,sample in zip(original,samples):np.testing.assert_array_equal(before,sample['body'])


@pytest.mark.parametrize('reason',['missing','contradicted','ambiguous','offscreen','weak'])
def test_no_corresponding_complete_arm_leaves_old_camera_options_available(reason):
    samples=observations();sample=samples[1]
    if reason=='missing':sample['evidence'][0,[4,5]]=0
    if reason=='contradicted':sample['evidence'][0,[4,5]]=3
    if reason=='ambiguous':sample['ownership_ambiguity'][0,[4,5]]=True
    if reason=='offscreen':sample['body'][0,[4,5],0]=1.2
    if reason=='weak':sample['body'][0,[4,5],2]=.1
    assert proposal(*samples,caps=(2.,2.)) is None
    assert len(pair_proposals(samples[0]['head'],sample['head'],caps=(2.,2.)))==3


def test_only_the_played_native_endpoint_is_fitted_and_legal_viewport_bounds_remain():
    samples=observations(1.8)
    samples[0]['body'][0,:,:2]=0.;samples[1]['body'][1,:,:2]=1.
    pair=proposal(*samples,caps=(1.35,2.))
    assert pair[0].zoom==pytest.approx(1.35) and pair[1].zoom==1.
    for cam,cap in zip(pair,(1.35,2.)):
        assert 1<=cam.zoom<=cap
        assert np.linalg.norm(np.array([cam.x,cam.y])-.5)<=.25+1e-12
        assert all(.5/cam.zoom-1e-12<=v<=1.-.5/cam.zoom+1e-12 for v in (cam.x,cam.y))


def test_prepared_endpoint_copies_are_immutable_and_do_not_reobserve_source_arrays():
    samples=observations();prepared=[prepare(s) for s in samples]
    expected=proposal(*samples,caps=(2.,2.))
    for sample in samples:
        sample['body'][:]=0.;sample['head'][:]=0.;sample['evidence'][:]=0
    assert prepared_proposal(*prepared,caps=(2.,2.))==expected
    for p in prepared:
        assert all(not a.flags.writeable for a in (p.head,p.points,p.known))
    assert prepared_proposal(None,prepared[1],caps=(2.,2.)) is None


@pytest.mark.parametrize('ratio',[.7,1.,1.2,1.8])
def test_preparation_does_not_change_pair_cameras(ratio):
    samples=observations(ratio)
    assert prepared_proposal(*(prepare(s) for s in samples),caps=(1.35,2.))==proposal(*samples,caps=(1.35,2.))


def catalogues():
    nodes,banks,poses,styles=inputs();cats=[]
    for s,bank in enumerate(banks):
        bank['zoom_cap']=2.
        if s:
            bank['heads'][:,:,0,:2]=.5+1.2*(bank['heads'][:,:,0,:2]-.5)
            poses[s][:,:,0,:,:2]=.5+1.2*(poses[s][:,:,0,:,:2]-.5)
        cats.append(SourceCatalogue(nodes,bank,poses[s],[],styles[s],[],fps=30.,editorial_content=True))
    paths,_=make_paths(nodes,banks,[[],[]])
    return nodes,cats,paths


def test_actual_domain_keeps_original_views_and_exposes_scored_paired_arm_views_before_DP():
    nodes,cats,paths=catalogues()
    old=make(nodes,cats,paths,fps=30.,articulated_actions=True,retain_pair_cameras=True)
    new=make(nodes,cats,paths,fps=30.,articulated_actions=True,retain_pair_cameras=True,body_camera_proposals=True)
    receipt=new[3]
    assert receipt['additional_observed_face_arm_camera_options']>0
    assert receipt['retained_observed_face_arm_camera_options']>0
    assert receipt['primary_pair_camera_options_per_actor_pair_upper_bound']==5
    physical=lambda ports:{(p.source,p.geometry_hash,p.observation['head_index']) for p in ports}
    for before,after in ((old[0],new[0]),(old[1],new[1])):
        for a,b in zip(before,after):assert physical(a)<=physical(b)
    for incoming,outgoing in zip(new[0],new[1]):
        assert max(len(incoming),len(outgoing))<=receipt['global_port_limit_per_side']
        assert {k for p in incoming for k in p.proposal_ids if k[0]=='paired'}=={
            k for p in outgoing for k in p.proposal_ids if k[0]=='paired'}


def test_preparation_progress_reports_real_nodes_without_changing_any_port():
    nodes,cats,paths=catalogues();events=[]
    settings=dict(fps=30.,articulated_actions=True,retain_pair_cameras=True,body_camera_proposals=True)
    before=make(nodes,cats,paths,**settings)
    after=make(nodes,cats,paths,**settings,progress=events.append)
    assert events[-1]['completed_nodes']==len(nodes)==events[-1]['total_nodes']
    assert events[-1]['observed_face_arm_options']>0
    assert events[-1]['actor_pairs_compared']>0
    for original,reported in zip(before[:2],after[:2]):
        for first,last in zip(original,reported):
            assert [(p.source,p.geometry_hash,p.proposal_ids) for p in first]==[
                (p.source,p.geometry_hash,p.proposal_ids) for p in last]


def test_articulated_camera_domain_is_explicit_and_requires_the_actual_interval_consumer():
    assert not PlayedPosePortPrior().body_camera_proposals
    with pytest.raises(ValueError,match='Articulated cameras'):PlayedPosePortPrior(body_camera_proposals=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        retain_pair_cameras=True,body_camera_proposals=True,focal_motion=True)
    assert prior.artifact['maximum_cameras_per_primary_actor_pair']==5
    assert prior.artifact['global_port_limit_per_side']=='four_per_source_plus_fifty_explicit_expansion'
    assert not prior.artifact['all_old_physical_candidates_retained']
    nodes,cats,paths=catalogues()
    with pytest.raises(ValueError,match='retained cameras'):
        make(nodes,cats,paths,fps=30.,body_camera_proposals=True)


def test_whole_interval_selector_consumes_body_cameras_and_freezes_its_real_choices():
    from crosscut2.played_pose_port_programme import select
    nodes,cats,_=catalogues()
    for cat in cats:
        cat.bank['programme_render_clock']=dict(fps=30.,count=54,source_times=nodes/30.)
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4))) for _ in cats]
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        retain_pair_cameras=True,body_camera_proposals=True)
    _,_,_,_,receipt,_=select(nodes,[c.bank for c in cats],[c.body for c in cats],
        [c.style for c in cats],fps=30.,source_events=[[],[]],body_states=None,
        native_ports=native,music=np.array([0.,1.,1.,0.]),prior=prior)
    assert receipt['boundary_port_domain']['articulated_body_camera_proposals_enabled']
    assert receipt['boundary_port_domain']['additional_observed_face_arm_camera_options']>0
    assert receipt['frozen_selected_camera_arcs']


def test_CLI_rejects_body_camera_flag_without_the_actual_consumer(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache','unused','--output','unused','--models','unused','--body-camera-proposals'])
    with pytest.raises(ValueError,match='Articulated cameras'):main()
