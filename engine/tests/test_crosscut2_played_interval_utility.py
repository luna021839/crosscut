from dataclasses import replace
import numpy as np
import pytest

from crosscut2.played_interval_utility import DeliverySamples,PlayedIntervalUtility,delivery_amount,articulated_seam_evidence
from crosscut2.observed_reframing import displayed_energy,displayed_energy_many


def samples(length=90,weights=None):
    frames=np.arange(length,dtype=np.int64)
    body=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
        [.15,.55,.9],[.85,.55,.9],[.43,.7,.9],[.57,.7,.9]])
    return DeliverySamples(frames,frames,np.tile([.5,.3,.2,.99],(length,1)),
        np.tile(body,(length,1,1)),np.ones((length,8),bool),
        tuple(np.zeros((0,4)) for _ in frames),np.ones(length) if weights is None else weights)


def utility(delivery,*,nodes=None,images=None):
    length=max((int(v.frames.max())+1 for v in delivery if len(v.frames)),default=90)
    nodes=np.array([0,length//3,2*length//3,length]) if nodes is None else np.asarray(nodes,int)
    clocks=[dict(fps=30.,count=length+10,source_times=nodes/30.) for _ in delivery]
    return PlayedIntervalUtility(nodes=nodes,clocks=clocks,delivery=delivery,
        image_records=[[] for _ in delivery] if images is None else images,fps=30.)


def image_pair(first,last,dx=0.,state='OBSERVED_IMAGE_REFRAMING'):
    points=np.array([(x,y) for x in (.2,.4,.6,.75) for y in (.2,.4,.6,.75)])
    affine=np.eye(3);affine[0,2]=dx
    return dict(output_frames=[first,last],native_pair=[first,last],
        observation=dict(state=state,affine=affine.tolist(),
            inlier_samples=np.column_stack((points,points+[dx,0.])).tolist()))


def test_same_entry_after_seven_frames_can_deliver_different_actual_action_and_room():
    value=samples();field=utility([value]);open_view=np.tile([1.,.5,.5],(90,1));tight=open_view.copy()
    tight[7:,0]=2.
    assert np.array_equal(open_view[:7],tight[:7])
    a=field.replay(0,0,90,open_view);b=field.replay(0,0,90,tight)
    assert a['observed_action_delivery_credit']>b['observed_action_delivery_credit']
    assert a['observed_action_delivery_credit']==pytest.approx(3.)
    assert not field.receipt()['entry_density_times_hold_duration']


def test_refiner_played_arm_direction_is_consumed_before_cut_choice_not_refit_after():
    from types import SimpleNamespace
    from crosscut2.dense_subject_catalogue import DenseSubjects
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs
    a,_,_=prepared(0,bump=.08);b,w,clock=prepared(1,bump=.08)
    body=w.body.copy();body[:,[2,3,4,5],:2]=2*body[:,[0,1,0,1],:2]-body[:,[2,3,4,5],:2]
    other=replace(w,body=body)
    b.subjects=DenseSubjects(source=1,binding='1',indices=clock,windows=[other],bank=b.bank,fps=30.)
    left=[SimpleNamespace(observation=obs(a,0))];right=[SimpleNamespace(observation=obs(b,1))]
    before=np.array([[.1]])
    actual=articulated_seam_evidence(dict(pose_rate=before),left,right)
    assert actual['articulated_arm_direction_measured'][0,0]
    assert actual['articulated_arm_direction'][0,0]>.9
    assert actual['pose_rate'][0,0]>.9 and before[0,0]==.1
    missing=SimpleNamespace(observation=dict(right[0].observation))
    missing.observation.pop('dense_subject_context')
    untouched=articulated_seam_evidence(dict(pose_rate=before),left,[missing])
    assert untouched['pose_rate'][0,0]==.1
    assert not untouched['articulated_arm_direction_measured'].any()


def test_good_measured_arms_replace_missing_channel_penalty_without_losing_screen_or_motion():
    from types import SimpleNamespace
    from crosscut2.played_partial_pose import features,matrix
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08)
    left=[SimpleNamespace(observation=obs(a,0))];right=[SimpleNamespace(observation=obs(b,1))]
    before=matrix(features([left[0].observation],fps=30.),
        features([right[0].observation],fps=30.),details=True)
    assert before['pose_rate'][0,0]>.25
    actual=articulated_seam_evidence(before,left,right)
    assert actual['pose_rate'][0,0]<1e-6
    assert actual['articulated_arm_direction_measured'][0,0]
    assert actual['actual_played_match_advantage'][0,0]>.99
    np.testing.assert_allclose(actual['matched_action_change_density'],
        np.minimum(before['matched_action_evidence_density'],actual['actual_played_match_advantage']))
    # The observed silhouette still has to land at the selected screen position.
    moved=SimpleNamespace(observation=dict(right[0].observation))
    moved.observation['camera_strip']=moved.observation['camera_strip'].copy()
    moved.observation['camera_strip'][:,1]+=.1
    displaced=articulated_seam_evidence(before,left,[moved])
    assert displaced['pose_rate'][0,0]>.05
    assert before['pose_rate'][0,0]>.25  # raw sparse evidence is unchanged


@pytest.mark.parametrize('tick',[-3,-2,1,2])
def test_final_articulated_score_preserves_opposite_intermediate_motion_and_DP_choice(tick):
    from crosscut2.dense_subject_catalogue import DenseSubjects
    from crosscut2.played_partial_pose import features,matrix
    from crosscut2.editorial_roles import event_energy
    from crosscut2.boundary_port_dp import solve
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs

    def moving(source,sign):
        cat,window,indices=prepared(source,bump=0.)
        body=window.body.copy()
        body[window.ordinals==100+tick,4:6,0]+=sign*.08
        cat.subjects=DenseSubjects(source=source,binding=str(source),indices=indices,
            windows=[replace(window,body=body)],bank=cat.bank,fps=30.)
        return cat

    a,good,bad=moving(0,1),moving(1,1),moving(2,-1)
    left=[obs(a,0)];right=[obs(cat,1) for cat in (good,bad)]
    before=matrix(features(left,fps=30.),features(right,fps=30.),details=True)
    after=articulated_seam_evidence(before,left,right)
    assert before['pose_rate'][0,0]<before['pose_rate'][0,1]
    assert after['pose_rate'][0,0]<1e-6
    assert after['pose_rate'][0,1]>after['pose_rate'][0,0]+.01
    np.testing.assert_allclose(after['actual_played_match_advantage'],
        np.maximum(0.,1.-2.*after['pose_rate']))

    # All other graph terms are explicit and identical. The .001 cheaper
    # hold must not win by erasing an actually observed contrary movement.
    seam=event_energy(after,np.full((1,2),.2),fps=30.,point_geometry=True)[0]
    domain=[np.arange(3) for _ in range(3)]
    def hold(i,j):
        costs=np.full((3,3),100. if i==0 and j==2 else 0.)
        if i==1:costs[1,1]=.001
        return costs,np.eye(3,dtype=bool),np.zeros((3,3),int)
    def cut(n):
        cost=np.zeros((3,3));allowed=np.zeros((3,3),bool)
        cost[0,1:]=seam;allowed[0,1:]=True
        return cost,allowed
    runs,_,_=solve([0,30,60],domain,domain,start_cost=np.array([0.,100.,100.]),
        hold_cost=hold,cut_cost=cut,minimum_frames=12)
    assert [run[2] for run in runs]==[0,1]


def test_unassociated_intermediate_arm_cannot_invent_motion_credit():
    from crosscut2.dense_subject_catalogue import DenseSubjects
    from crosscut2.played_partial_pose import features,matrix
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs
    cats=[]
    for source in (0,1):
        cat,window,indices=prepared(source,bump=.08)
        associated=window.associated.copy();associated[window.ordinals==101]=False
        support=window.support.copy();support[~associated]=False
        cat.subjects=DenseSubjects(source=source,binding=str(source),indices=indices,
            windows=[replace(window,associated=associated,support=support)],bank=cat.bank,fps=30.)
        cats.append(cat)
    left=[obs(cats[0],0)];right=[obs(cats[1],1)]
    before=matrix(features(left,fps=30.),features(right,fps=30.),details=True)
    after=articulated_seam_evidence(before,left,right)
    assert after['matched_action_change_density'][0,0]==0.


def test_final_motion_integrates_observed_time_instead_of_one_wrist_maximum():
    from crosscut2.subject_cut_moments import _motion_steps
    ticks=np.arange(-7,7);a=np.zeros((14,8,2));b=a.copy()
    b[ticks==1,4,0]=100.
    known=np.ones((14,8),bool);contiguous=np.ones(13,bool)
    maximum,_,rate=_motion_steps(a,b,known,ticks,[2,3,4,5],contiguous)
    assert maximum==300.
    # Only two of28 genuinely observed joint/time intervals disagree.
    assert rate==pytest.approx((300/301)*2/28)
    assert 0.<rate<.08
    # A sustained measured discrepancy has a larger integrated cost, while
    # an entirely absent wrist never becomes measured perfect movement.
    b[:,4,0]=np.arange(14)*100.
    _,_,sustained=_motion_steps(a,b,known,ticks,[2,3,4,5],contiguous)
    assert sustained>rate
    known[:]=False
    assert _motion_steps(a,b,known,ticks,[2,3,4,5],contiguous)==(None,0.,None)


def test_raw_actor_retrieval_consumes_the_same_articulated_action_before_pruning():
    from crosscut2.boundary_port_domain import make
    from test_crosscut2_dense_subject_catalogue import prepared
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08)
    paths=[[(np.array([0,200]),np.tile([1.,.5,.5],(2,1))) for _ in range(4)] for _ in range(2)]
    class Capture:
        detail=None
        def observe(self,n,left,right,detail,*sources):self.detail=detail
        def density(self,*args):return 0.
    old=Capture();new=Capture()
    make(a.nodes,[a,b],paths,fps=30.,maximum_pairs=0,opportunities=old)
    _,_,_,receipt=make(a.nodes,[a,b],paths,fps=30.,maximum_pairs=0,
        opportunities=new,articulated_actions=True)
    assert 'articulated_arm_direction_measured' not in old.detail
    mask=new.detail['articulated_arm_direction_measured']
    assert mask.any() and (new.detail['pose_rate'][mask]<old.detail['pose_rate'][mask]).any()
    assert receipt['actual_articulated_action_used_before_global_pruning']


def test_naturally_matched_original_views_are_explicit_pair_bound_options():
    from crosscut2.boundary_port_domain import make
    from test_crosscut2_dense_subject_catalogue import prepared
    a,_,_=prepared(0,bump=.08);b,w,clock=prepared(1,bump=.08)
    from crosscut2.dense_subject_catalogue import DenseSubjects
    b.bank['heads'][:,:,0,2]=.23
    heads=w.heads.copy();heads[:,2]=.23
    b.subjects=DenseSubjects(source=1,binding='1',indices=clock,
        windows=[replace(w,heads=heads)],bank=b.bank,fps=30.)
    paths=[[(np.array([0,200]),np.tile([1.,.5,.5],(2,1))) for _ in range(4)] for _ in range(2)]
    entry,exit,_,receipt=make(a.nodes,[a,b],paths,fps=30.,articulated_actions=True,
        recycle_static_capacity=True)
    original=np.tile([1.,.5,.5],(7,1))
    native={i for p in entry[1] if np.array_equal(p.strip,original)
        for i in p.proposal_ids if i[0]=='paired'}
    assert native
    assert native & {i for p in exit[1] if np.array_equal(p.strip,original)
        for i in p.proposal_ids if i[0]=='paired'}
    assert receipt['retained_paired_original_boundary_views']>0
    assert receipt['natural_action_original_view_options_added']>0
    assert max(map(len,entry))<=16 and max(map(len,exit))<=16


def test_static_or_unknown_action_does_not_gain_an_original_view_pair_option():
    from crosscut2.boundary_port_domain import make
    from test_crosscut2_dense_subject_catalogue import prepared
    a,_,_=prepared(0,bump=0.);b,_,_=prepared(1,bump=0.)
    paths=[[(np.array([0,200]),np.tile([1.,.5,.5],(2,1))) for _ in range(4)] for _ in range(2)]
    _,_,_,receipt=make(a.nodes,[a,b],paths,fps=30.,articulated_actions=True,
        recycle_static_capacity=True)
    assert receipt['natural_action_original_view_options_added']==0


def test_measured_action_never_revives_a_missing_or_incompatible_change_cue():
    from types import SimpleNamespace
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs
    left=[SimpleNamespace(observation=obs(prepared(0)[0],0))]
    right=[SimpleNamespace(observation=obs(prepared(1)[0],1))]
    before=dict(pose_rate=np.array([[.5]]),matched_action_change_density=np.array([[0.]]),
        matched_action_evidence_density=np.array([[0.]]))
    actual=articulated_seam_evidence(before,left,right)
    assert actual['pose_rate'][0,0]<1e-6
    assert actual['matched_action_change_density'][0,0]==0.
    assert before['pose_rate'][0,0]==.5


@pytest.mark.parametrize('joints,roots',[([2,4],[0,0]),([3,5],[1,1])])
def test_measured_opposed_arm_cannot_keep_the_old_positive_match_benefit(joints,roots):
    from types import SimpleNamespace
    from crosscut2.dense_subject_catalogue import DenseSubjects
    from crosscut2.played_partial_pose import features,matrix
    from test_crosscut2_dense_subject_catalogue import prepared
    from test_crosscut2_source_clock_action import obs
    a,_,_=prepared(0,bump=.08);b,w,clock=prepared(1,bump=.08)
    body=w.body.copy();body[:,joints,:2]=2*body[:,roots,:2]-body[:,joints,:2]
    b.subjects=DenseSubjects(source=1,binding='1',indices=clock,
        windows=[replace(w,body=body)],bank=b.bank,fps=30.)
    left=[SimpleNamespace(observation=obs(a,0))];right=[SimpleNamespace(observation=obs(b,1))]
    before=matrix(features([left[0].observation],fps=30.),
        features([right[0].observation],fps=30.),details=True)
    old=float(before['matched_action_change_density'][0,0])
    assert old>.1 and before['pose_rate'][0,0]<.5
    after=articulated_seam_evidence(before,left,right)
    advantage=np.maximum(0.,1.-2.*after['pose_rate'])
    assert after['pose_rate'][0,0]>=.49999999
    assert after['matched_action_change_density'][0,0]<=advantage[0,0]+1e-12
    np.testing.assert_array_equal(after['actual_played_match_advantage'],advantage)
    assert before['matched_action_change_density'][0,0]==old  # no mutation of raw evidence


def test_split_and_A_B_A_cannot_duplicate_same_output_time_utility():
    value=samples();field=utility([value,value]);curve=np.tile([1.,.5,.5],(90,1))
    full=field.replay(0,0,90,curve)['observed_action_delivery_credit']
    split=sum(field.replay(s,a,b,curve[a:b])['observed_action_delivery_credit']
        for s,a,b in ((0,0,30),(1,30,60),(0,60,90)))
    assert split==pytest.approx(full)
    assert full==3. and not value.frames.flags.writeable


def test_missing_action_is_not_extended_or_substituted_with_competitor_body():
    value=samples();support=value.support.copy();support[7:,2:6]=False
    partial=replace(value,support=support);field=utility([partial])
    assert field.replay(0,0,90,np.tile([1.,.5,.5],(90,1)))['observed_action_delivery_credit']==pytest.approx(7/30.)
    # Raw confidence stays high: the measured support mask controls evidence.
    assert (partial.body[7:,:,2]>.3).all()
    other=replace(value,competitors=tuple(np.array([[.75,.3,.3,.99]]) for _ in range(90)))
    camera=np.tile([2.,.75,.5],(1,90,1))
    assert delivery_amount(other,np.arange(90),camera).mean()<.4


def test_actual_candidate_native_mismatch_removes_only_unsupported_credit():
    value=samples();shifted=replace(value,native=value.native+1);field=utility([shifted])
    result=field.replay(0,0,90,np.tile([1.,.5,.5],(90,1)))
    assert result['adjustment']==0. and result['observed_action_delivery_credit']==0.


def test_real_catalogue_uses_full_actual_window_not_only_incoming_seven_frames():
    from test_crosscut2_dense_subject_catalogue import prepared
    from crosscut2.editorial_opportunities import Opportunities
    cats=[prepared(s)[0] for s in range(2)]
    for cat in cats:
        cat.nodes=np.array([0,100,200]);cat.final=200
        cat.bank['programme_render_clock']=dict(fps=30.,count=201,source_times=cat.nodes/30.)
    event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a','b'])
    event.bindings[(1,0,1,0,0)]=.8;event.utility[1]=.8
    field=PlayedIntervalUtility.from_catalogues(cats,event)
    for table in field.delivery:
        assert table.frames.tolist()==list(range(100,115))
        assert table.native.tolist()==list(range(100,115))
    assert field.receipt()['targets'][0]['selected_before_camera']
    cats[1].observed_image_records=(dict(image_pair(100,101),source_sha256='wrong-source'),)
    with pytest.raises(ValueError,match='physical source binding'):
        PlayedIntervalUtility.from_catalogues(cats,event)


def test_original_native_cut_stops_fixed_target_without_excluding_other_source():
    from test_crosscut2_dense_subject_catalogue import prepared
    from crosscut2.editorial_opportunities import Opportunities
    cats=[prepared(0)[0],prepared(1,cut=(104,))[0]]
    for cat in cats:
        cat.nodes=np.array([0,100,200]);cat.final=200
        cat.bank['programme_render_clock']=dict(fps=30.,count=201,source_times=cat.nodes/30.)
    event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a','b'])
    event.bindings[(1,0,1,0,0)]=.8;event.utility[1]=.8
    field=PlayedIntervalUtility.from_catalogues(cats,event)
    assert len(field.delivery[0].frames)==15
    assert field.delivery[1].frames.tolist()==[100,101,102,103]


def test_fixed_delivery_uses_validated_body_context_not_the_unmasked_old_track(monkeypatch):
    from test_crosscut2_dense_subject_catalogue import prepared
    from crosscut2.editorial_opportunities import Opportunities
    cats=[prepared(s)[0] for s in range(2)]
    context=cats[0].subjects.sample(1,0,0)
    support=context['window'].support.copy();support[:,[0,2,4]]=False
    validated=dict(context,window=replace(context['window'],support=support))
    calls=[]
    for source,cat in enumerate(cats):
        cat.nodes=np.array([0,100,200]);cat.final=200
        cat.bank['programme_render_clock']=dict(fps=30.,count=201,source_times=cat.nodes/30.)
        def reader(n,side,head,source=source):
            calls.append((source,n,side,head))
            return validated if source==0 else None
        monkeypatch.setattr(cat,'body_subject_context',reader)
        monkeypatch.setattr(cat.subjects,'sample',lambda *a:pytest.fail('raw body bypassed validation'))
    event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a','b'])
    event.bindings[(1,0,1,0,0)]=.8;event.utility[1]=.8
    field=PlayedIntervalUtility.from_catalogues(cats,event)
    assert calls==[(0,1,0,0),(1,1,1,0)]
    assert len(field.delivery[0].frames)==15 and not len(field.delivery[1].frames)
    assert not field.delivery[0].support[:,[0,2,4]].any()
    assert field.delivery[0].support[:,[1,3,5]].all()


def test_batch_image_energy_preserves_scalar_support_and_compensation():
    row=image_pair(0,1,.05);obs=row['observation']
    before=np.array([[1.25,.5,.5],[1.25,.5,.5],[2.,.75,.5],[1.-2e-16,.5,.5]])
    after=before.copy();after[1,1]=.55
    energy,known=displayed_energy_many(obs,before,after,seconds=.2)
    for i in range(len(before)):
        expected=displayed_energy(obs,before[i],after[i],seconds=.2)
        assert bool(known[i])==(expected is not None)
        if expected is not None:assert energy[i]==pytest.approx(expected,abs=1e-15)
    assert energy[0]>0 and energy[1]==pytest.approx(0,abs=1e-20)


def test_composed_motion_replaces_digital_only_on_supported_actual_native_pairs():
    value=samples(3);empty=replace(value,weights=np.zeros(3))
    rows=[image_pair(0,1,.04),image_pair(1,2,-.04)]
    field=utility([empty],nodes=[0,1,2,3],images=[rows])
    fixed=np.tile([1.25,.5,.5],(1,3,1));frames=np.arange(3)
    effect=field.effects(0,frames,frames,fixed)
    assert effect['composed_image_motion_delta'][0]>0  # out-and-back is not stationary
    cancelled=fixed.copy();cancelled[:,1,1]=.54
    effect=field.effects(0,frames,frames,cancelled)
    digital=30*np.sum(np.diff(cancelled[0,:,1:]*[16/9,1.],axis=0)**2)
    assert effect['composed_image_motion_delta'][0]+digital==pytest.approx(0,abs=1e-15)
    assert field.effects(0,frames,frames+10,fixed)['adjustment'][0]==0.
    unknown=utility([empty],nodes=[0,1,2,3],images=[[image_pair(0,1,state='NATIVE_DISCONTINUITY')]])
    assert unknown.effects(0,frames,frames,cancelled)['composed_image_motion_delta'][0]==0.


def test_duplicate_physical_image_event_not_counted_twice_and_conflict_stays_unknown():
    value=replace(samples(3),weights=np.zeros(3));row=image_pair(0,1,.04)
    once=utility([value],nodes=[0,1,2,3],images=[[row]])
    twice=utility([value],nodes=[0,1,2,3],images=[[row,row]])
    camera=np.tile([1.,.5,.5],(1,3,1));frames=np.arange(3)
    assert once.effects(0,frames,frames,camera)['adjustment']==twice.effects(0,frames,frames,camera)['adjustment']
    bad=utility([value],nodes=[0,1,2,3],images=[[row,image_pair(0,1,-.04)]])
    assert bad.conflicts==1 and not bad.images[0]


def test_complete_arc_camera_argmin_and_replay_consume_identical_new_utility():
    from test_crosscut2_boundary_arc_costs import arc_fixture
    nodes,entries,exits,arcs,fragments=arc_fixture(False,preserve_camera_alternatives=True)
    field=utility([samples(54),samples(54)],nodes=nodes)
    arcs.played_interval_utility=field;compared=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            values,valid=arcs.camera_alternatives(i,j)
            best,allowed,codes=arcs(i,j)
            for p,q,code in zip(*np.nonzero(valid)):
                s=entries[i][p].source;curve=arcs.replay(i,j,p,q,int(code))
                effect=field.replay(s,int(nodes[i]),int(nodes[j]),curve)
                exact=arcs.direct_cost(s,int(nodes[i]),curve[None])[0]+fragments(i,j)[s]+effect['adjustment']
                assert values[p,q,code]==pytest.approx(exact,rel=1e-10,abs=1e-10)
                compared+=1
            for p,q in zip(*np.nonzero(allowed)):
                assert best[p,q]==pytest.approx(values[p,q][valid[p,q]].min())
    assert compared>0 and field.calls>0


def test_variable_port_DP_changes_source_and_cut_layout_from_actual_pointwise_delivery():
    from crosscut2.boundary_port_dp import solve
    nodes=np.array([0,30,60,90,120]);domain=[np.arange(3) for _ in nodes]
    medium=samples(120,np.full(120,.5));middle=np.zeros(120);middle[30:90]=1.
    brief=np.zeros(120);brief[60:66]=1.
    field=utility([medium,samples(120,middle),samples(120,brief)],nodes=nodes)
    def hold(i,j):
        cost=np.zeros((3,3))
        for s in range(3):
            curve=np.tile([1.,.5,.5],(nodes[j]-nodes[i],1))
            cost[s,s]=field.replay(s,int(nodes[i]),int(nodes[j]),curve)['adjustment']
        return cost,np.eye(3,dtype=bool)
    args=dict(start_cost=np.zeros(3),start_allowed=np.array([True,False,False]),
        final_allowed=np.array([True,False,False]),hold_cost=hold,
        cut_cost=lambda n:(np.full((3,3),.1),~np.eye(3,dtype=bool)))
    runs,score,receipt=solve(nodes,domain,domain,**args)
    assert [(i,j,p) for i,j,p,q in runs]==[(0,1,0),(1,3,1),(3,4,0)]
    assert score[1]==2 and all(p!=2 for i,j,p,q in runs)
    # No desired cut count or maximum hold is supplied; without useful
    # observed delivery the SAME solver prefers one continuous source.
    args['hold_cost']=lambda i,j:(np.zeros((3,3)),np.eye(3,dtype=bool))
    old,old_score,_=solve(nodes,domain,domain,**args)
    assert len(old)==1 and old_score[1]==0


def test_actual_programme_consumer_uses_pair_bound_claims_not_source_content_bonus():
    from crosscut2.played_pose_port_programme import select,PlayedPosePortPrior
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    nodes,banks,_,_=args
    for bank in banks:
        bank['programme_render_clock']=dict(fps=30.,count=int(nodes[-1]),source_times=nodes/30.)
    result=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True))
    receipt=result[4]
    assert receipt['camera_argmin_includes_actual_played_interval_utility']
    assert receipt['played_interval_utility']['seven_frame_credit_used'] is False
    assert receipt['boundary_port_DP']['current_event_paid_bit']
    assert receipt['boundary_port_DP']['claim_camera_chosen_before_argmin']
    assert not receipt['played_interval_utility']['source_content_delivery_reward_in_base']
    assert result[3]==pytest.approx(sum(sum(p['components'].values()) for p in receipt['selected_hold_cost_decomposition']))
    assert all(p['components']['observed_played_action_delivery_credit']==0 for p in receipt['selected_hold_cost_decomposition'])
    assert not receipt['professional_quality_approved']


@pytest.mark.parametrize('changed_clock',[False,True])
def test_compiled_fixed_interiors_and_patches_equal_direct_all_codes_with_image_and_clock_gaps(changed_clock):
    from test_crosscut2_boundary_arc_costs import arc_fixture
    nodes,entries,exits,arcs,_=arc_fixture(False,preserve_camera_alternatives=True)
    value=samples(54)
    if changed_clock:
        native=value.native.copy();native[5::7]+=1;value=replace(value,native=native)
    field=utility([value,value],nodes=nodes,images=[
        [image_pair(6,7,.02),image_pair(17,18,-.02),image_pair(18,19,.03),image_pair(35,36,.02)],
        [image_pair(30,31,.03),image_pair(40,41,-.03)]])
    arcs.played_interval_utility=field
    calls=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            for source in range(2):
                p=np.array([k for k,v in enumerate(entries[i]) if v.source==source])
                q=np.array([k for k,v in enumerate(exits[j]) if v.source==source])
                if not len(p) or not len(q):continue
                arcs(i,j);allowed=arcs.last_camera_alternatives[3][np.ix_(p,q,np.arange(6))]
                zero=np.zeros(allowed.shape)
                direct=field.direct_adjust(arcs,i,j,source,p,q,zero,allowed)
                compiled=field.adjust(arcs,i,j,source,p,q,zero,allowed)
                np.testing.assert_allclose(compiled[allowed],direct[allowed],rtol=1e-11,atol=1e-11)
                calls+=1
    assert calls and field.prefix_interval_uses>0
    if changed_clock:assert field.clock_correction_rows>0


def test_malformed_actual_camera_is_not_hidden_by_unknown_delivery():
    value=samples();field=utility([value]);camera=np.tile([1.,.5,.5],(1,90,1));camera[0,0,0]=np.nan
    with pytest.raises(ValueError,match='actual sampled'):
        field.effects(0,np.arange(90),np.arange(90),camera)
