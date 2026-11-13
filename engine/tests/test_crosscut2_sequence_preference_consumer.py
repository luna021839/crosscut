"""Synthetic artifacts test code paths, never real human/AV supervision.

These fixtures must not be exported as fitted project models or evidence of
editing quality. Actual model fitting and whole-film approval are separate.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from crosscut2.sequence_context import SCHEMA_SHA256,FEATURE_NAMES,PRIMITIVES,windows,merge
from crosscut2.sequence_preference_model import Model,digest,CONTRACT
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from crosscut2.event_holds import select
from crosscut2.programme_source_clock import attach
from crosscut2.context_port_dp import ArcRef
from crosscut2.boundary_port_domain import make
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.event_holds import fragment_costs
from crosscut2.played_context import PlayedContext,_SourceField
from test_crosscut2_scene_readout import catalogue


def numeric_model(*,zoom_weight=0.,bias=0.,feature_weights=None):
    # Validation fixture only: the self-report/report fields below simulate
    # the artifact contract. They are NOT real user labels or a real fit.
    config=dict(schema_sha256=SCHEMA_SHA256,interval_frames=6,context_seconds=4.)
    weight=np.zeros(len(FEATURE_NAMES));weight[FEATURE_NAMES.index('additional_log_zoom.observed_mean')]=zoom_weight
    for name,value in (feature_weights or {}).items():weight[FEATURE_NAMES.index(name)]=value
    payload=dict(contract=CONTRACT,output_kind='group',fitted=True,feature_schema_sha256=SCHEMA_SHA256,
        feature_names=list(FEATURE_NAMES),window_config=config,
        window_contract_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        center=np.zeros(len(FEATURE_NAMES)).tolist(),scale=np.ones(len(FEATURE_NAMES)).tolist(),
        weights=weight.tolist(),bias=bias,
        training=dict(supervision_kind='human_av_context_preference',self_reported_normal_av=True,
            relative_and_absolute_labels_separate=True,examples_sha256='a'*64,
            train_physical_groups=['unit-fixture-train'],dev_physical_groups=['unit-fixture-dev'],
            absolute_positive_training_options=1,absolute_negative_training_options=1),
        normal_app_activation_approved=False,professional_quality_approved=False)
    payload['content_sha256']=digest(payload)
    return Model(payload)


def prepared(nodes):
    cats=[catalogue(nodes) for _ in range(2)]
    class IdentityWarp:
        def master_to_source(self,t):return t
    clocks=[dict(fps=30.,frame_count=int(nodes[-1]))]*2
    attach([c.bank for c in cats],[IdentityWarp(),IdentityWarp()],clocks,nodes,master_start=0.,fps=30.)
    return cats


def arguments(nodes,cats):
    native=[dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[]) for _ in cats]
    return dict(fps=30.,source_events=[[] for _ in cats],body_states=None,native_ports=native,
        music=np.ones(len(nodes)),connection_prior=PlayedPosePortPrior(editorial_opportunities=True))


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_actual_event_entry_path_consumes_new_context_and_changes_realized_camera_values(kernel):
    if kernel=='numba':pytest.importorskip('numba')
    nodes=np.array([0,18,36,54]);cats=prepared(nodes);kw=arguments(nodes,cats)
    a=select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**kw)
    model=numeric_model(zoom_weight=-100.,bias=10.)
    b=select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**kw,
        sequence_preference_model=model,sequence_source_bindings=['a'*64,'b'*64],sequence_context_kernel=kernel)
    receipt=b[4]
    assert receipt['sequence_preference_consumed'] and not receipt['scene_readout_consumed']
    assert not receipt['pre_DP_scalar_camera_argmin_used']
    assert receipt['sequence_preference_model_sha256']==model.artifact['content_sha256']
    assert receipt['boundary_port_DP']['scalar_cost_or_debt_dominance'] is False
    assert receipt['sequence_preference_context']['source_NN_calls']==0
    assert receipt['sequence_preference_context']['context_kernel']==kernel
    if kernel=='numba':assert receipt['sequence_preference_context']['compiled_kernel_execution']['nopython_specializations']>0
    camera_a=np.concatenate([r['scored_camera_values'] for r in a[2]])
    camera_b=np.concatenate([r['scored_camera_values'] for r in b[2]])
    assert camera_a.shape==camera_b.shape==(54,3) and not np.array_equal(camera_a,camera_b)
    assert receipt['professional_quality_approved'] is False and not receipt['stage_edit_transformer_used']
    parts=receipt['selected_hold_cost_decomposition']
    assert sum(p['components']['fixed_time_sequence_preference'] for p in parts)>0.
    assert sum(p['components']['residual_readout_interaction'] for p in parts)==0.


def test_old_models_missing_binding_or_rejected_debt_cannot_silently_enter_new_path():
    nodes=np.array([0,18,36]);cats=prepared(nodes);kw=arguments(nodes,cats)
    with pytest.raises(ValueError,match='explicit new fixed-time'):
        select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**kw,sequence_preference_model=object())
    with pytest.raises(ValueError,match='actual physical source'):
        select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**kw,sequence_preference_model=numeric_model())
    kw['connection_prior']=PlayedPosePortPrior(editorial_opportunities=True,scene_readout=True)
    with pytest.raises(ValueError,match='replaces rejected E5'):
        select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**kw,
            sequence_preference_model=numeric_model(),sequence_source_bindings=['a'*64,'b'*64])


def test_model_does_not_upgrade_weak_advisory_or_all_negative_evidence_to_fitted_AV():
    artifact=numeric_model().artifact
    for change in ('weak','no_positive','same_event','mutated'):
        bad=deepcopy(artifact)
        if change=='weak':bad['training']['supervision_kind']='opus_silent_still_advisory'
        elif change=='no_positive':bad['training']['absolute_positive_training_options']=0
        elif change=='same_event':bad['contract']='experimental_same_event_preference_ranker_v2'
        else:bad['bias']=1.
        if change!='mutated':bad.pop('content_sha256');bad['content_sha256']=digest(bad)
        with pytest.raises(ValueError,match='new fitted sequence'):Model(bad)
    model=numeric_model();model.artifact['bias']=1.
    with pytest.raises(ValueError,match='parameters changed'):model.verify()
    model=numeric_model();model.artifact['training']['self_reported_normal_av']=False
    model.artifact.pop('content_sha256');model.artifact['content_sha256']=digest(model.artifact)
    with pytest.raises(ValueError,match='parameters changed'):model.verify()
    for name,value in (('context_seconds',2.),('interval_frames',12),('sequence_feature_schema','old_schema')):
        model=numeric_model();setattr(model,name,value)
        with pytest.raises(ValueError,match='parameters changed'):model.verify()


def context_fixture(nodes,*,nonlinear_clock=False,feature_weights=None,kernel='numpy'):
    cats=prepared(nodes);paths=[[(nodes,np.tile([1.+g*.04,.5,.5],(len(nodes),1))) for g in range(4)] for _ in cats]
    cats[1].bank['heads'][:,:,0,0]=.7
    if nonlinear_clock:
        cats[0].bank['programme_render_clock']['source_times']=np.asarray(nodes)/30.+.025*np.sin(np.asarray(nodes)/20.)
    entry,exit,dense,_=make(nodes,cats,paths,fps=30.,coalesce_static_ports=True)
    events=[np.zeros(0,int)]*2;native=[dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[]) for _ in cats]
    support=PlayedEventSupport(events,final=int(nodes[-1]),fps=30.)
    arcs=ArcCosts(nodes,entry,exit,cats,dense,native,support,fragment_costs(nodes,events,fps=30.,support_seconds=.4),
        fps=30.,preserve_camera_alternatives=True)
    model=numeric_model(zoom_weight=3.,bias=.1,feature_weights=feature_weights)
    scorer=PlayedContext(model,nodes=nodes,entries=entry,exits=exit,catalogues=cats,arcs=arcs,
        native_ports=native,fps=30.,source_bindings=['a'*64,'b'*64],check=lambda:None,kernel=kernel)
    return scorer,arcs


@pytest.mark.parametrize('nonlinear',[False,True])
def test_long_context_factorization_or_clock_fallback_equals_independent_whole_stream(nonlinear):
    nodes=np.arange(0,361,30);context,arcs=context_fixture(nodes,nonlinear_clock=nonlinear)
    source=0;entry=context.entries;exit=context.exits
    p=next(k for k,v in enumerate(entry[0]) if v.source==source)
    q=next(k for k,v in enumerate(exit[-1]) if v.source==source)
    ref=ArcRef(0,len(nodes)-1,p,q,0)
    actual=context.incremental([()],ref)[0]
    pieces,whole=context.replay([ref])
    assert actual==pytest.approx(whole,abs=1e-10) and pieces==pytest.approx([whole],abs=1e-10)
    if nonlinear:assert context.fallback==1
    else:assert context.fast==1


def test_native_target_is_frozen_before_camera_competitor_crop():
    nodes=np.array([0,30,60]);cat=prepared(nodes)[0]
    cat.bank['heads'][:,:,0]=[.3,.35,.12,.99];cat.bank['heads'][:,:,1]=[.7,.35,.12001,.99]
    # Construct the real catalogue after fixture preparation.
    from crosscut2.display_source_catalogue import SourceCatalogue
    cat=SourceCatalogue(nodes,cat.bank,None,[],cat.style,[],fps=30.,editorial_content=True)
    native=dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[])
    field=_SourceField(cat,native,'a'*64,lambda:None)
    a=field.project(np.array([30]),np.array([30]),np.array([[1.,.5,.5]]))
    b=field.project(np.array([30]),np.array([30]),np.array([[2.,.7,.5]]))
    k=PRIMITIVES.index('role_unknown')
    assert a.observed[0,k] and a.values[0,k]==b.values[0,k]==1.
    assert not a.observed[0,PRIMITIVES.index('feature_center_x')]
    assert a.target_scopes==b.target_scopes


@pytest.mark.parametrize('camera',[0,1,2,3,4,5])
def test_mixed_source_prefix_and_long_arc_match_whole_replay_for_spatial_and_zoom_context(camera):
    nodes=np.arange(0,421,30)
    context,_=context_fixture(nodes,feature_weights={
        'feature_center_x.observed_mean':2.,'feature_center_x.observed_minimum':-1.,
        'feature_center_x.observed_maximum':1.,'feature_height.observed_mean':3.,
        'additional_log_zoom.observed_minimum':-2.,'additional_log_zoom.observed_maximum':3.})
    def arc(i,j,source,code):
        p=next(k for k,v in enumerate(context.entries[i]) if v.source==source)
        q=next(k for k,v in enumerate(context.exits[j]) if v.source==source)
        return ArcRef(i,j,p,q,code)
    first=arc(0,1,0,1);prefix=arc(1,3,1,2);long=arc(3,len(nodes)-1,0,camera)
    refs=[first,prefix,long];parts=[];history=[]
    for ref in refs:
        parts.append(float(context.incremental([tuple(history)],ref)[0]));history.append(ref)
    whole_parts,whole=context.replay(refs)
    np.testing.assert_allclose(parts,whole_parts,rtol=1e-12,atol=1e-12)
    assert sum(parts)==pytest.approx(whole,abs=1e-11)
    if camera<4:assert context.fast>0
    else:assert context.fast==0


def test_native_extra_target_needs_actual_ordinal_not_output_frame_assumption():
    cat=prepared(np.array([0,30,60]))[0]
    heads=np.zeros((1,2,8,4));heads[:,:,0]=[.8,.3,.2,.99]
    no_clock=dict(frames=np.array([15]),heads=heads,observed=np.ones(1,bool))
    missing=_SourceField(cat,no_clock,'a'*64,lambda:None)
    actual=_SourceField(cat,dict(no_clock,native_event_keys=[[16]]),'a'*64,lambda:None)
    assert missing.native_cut_events_missing_ordinal==1 and missing.native_cut_endpoint_rows==0
    assert actual.native_cut_events_missing_ordinal==0 and actual.native_cut_endpoint_rows==2
    result=actual.project(np.array([15,16]),np.array([15,16]),np.tile([1.,.5,.5],(2,1)))
    k=PRIMITIVES.index('feature_center_x')
    assert result.observed[:,k].all() and result.values[:,k]==pytest.approx([.8,.8])
    unknown=missing.project(np.array([15,16]),np.array([15,16]),np.tile([1.,.5,.5],(2,1)))
    assert not unknown.observed[:,k].any()


def test_actual_dense_window_with_only_one_observed_face_does_not_require_fake_competitors():
    from test_crosscut2_dense_subject_catalogue import prepared as dense_prepared
    cat,window,_=dense_prepared()
    native=dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[])
    field=_SourceField(cat,native,'0',lambda:None)
    result=field.project(np.arange(93,107),np.arange(93,107),np.tile([1.,.5,.5],(14,1)))
    k=PRIMITIVES.index('role_feature')
    assert result.observed[:,k].all() and (result.values[:,k]==1.).all()
    assert result.observed[:,PRIMITIVES.index('feature_height')].all()


def test_batched_distinct_prefixes_and_all_camera_codes_keep_each_real_history_cost():
    nodes=np.arange(0,241,30);context,_=context_fixture(nodes,feature_weights={
        'feature_center_x.observed_adjacent_mean_rate':.4,
        'additional_log_zoom.observed_adjacent_maximum_rate':.3})
    def ref(i,j,s,code):
        p=next(k for k,v in enumerate(context.entries[i]) if v.source==s)
        q=next(k for k,v in enumerate(context.exits[j]) if v.source==s)
        return ArcRef(i,j,p,q,code)
    histories=[(ref(0,2,0,1),),(ref(0,1,1,2),ref(1,2,0,0))]
    for code in range(6):
        arc=ref(2,8,1,code);actual=context.incremental(histories,arc)
        for k,history in enumerate(histories):
            whole,_=context.replay([*history,arc])
            assert actual[k]==pytest.approx(whole[-1],abs=1e-11)
    assert context.cache_hits>0


def test_identical_observed_camera_inputs_reuse_cost_without_erasing_codes_or_histories():
    nodes=np.array([0,30,60,90]);context,arcs=context_fixture(nodes)
    p=next(k for k,v in enumerate(context.entries[0]) if v.source==0)
    q=next(k for k,v in enumerate(context.exits[-1]) if v.source==0)
    refs=[ArcRef(0,3,p,q,code) for code in (0,4,5)]
    for ref in refs[1:]:
        np.testing.assert_array_equal(context._part(ref,0,90).values,context._part(refs[0],0,90).values)
    values=[context.incremental([()],ref)[0] for ref in refs]
    assert values==pytest.approx([values[0]]*3,abs=1e-12)
    assert context.window_calls==15  # all three codes still exist; score input is identical
    assert context.receipt()['identical_observed_head_input_cache_hits']==2
    before=context.window_calls;different=ArcRef(0,3,p,q,1)
    assert context.incremental([()],different)[0]!=pytest.approx(values[0],abs=1e-8)
    assert context.window_calls>before


@pytest.mark.parametrize('nonlinear',[False,True])
def test_history_independent_long_bound_never_exceeds_any_actual_prefix_cost(nonlinear):
    nodes=np.arange(0,421,30);context,_=context_fixture(nodes,nonlinear_clock=nonlinear,
        feature_weights={'feature_center_x.observed_mean':2.,'additional_log_zoom.observed_adjacent_mean_rate':1.})
    def arc(i,j,s,code):
        p=next(k for k,v in enumerate(context.entries[i]) if v.source==s)
        q=next(k for k,v in enumerate(context.exits[j]) if v.source==s)
        return ArcRef(i,j,p,q,code)
    histories=[(arc(0,3,0,0),),(arc(0,1,1,2),arc(1,3,0,1))]
    for code in range(6):
        ref=arc(3,14,0,code);bound=context.lower_bound(ref)
        actual=context.incremental(histories,ref)
        assert (actual>=bound).all() and bound>=0.
        if code<4 and not nonlinear:assert bound>0.
        else:assert bound==0.


def test_cli_rejects_unbound_or_old_sequence_artifact_before_mount_acquisition_or_render(monkeypatch,tmp_path):
    from crosscut2.stage_mix import main
    args=['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--experimental-sequence-preference-model',str(tmp_path/'old.json')]
    monkeypatch.setattr('sys.argv',args)
    with pytest.raises(ValueError,match='requires E4'):main()
    (tmp_path/'old.json').write_text(json.dumps({'contract':'experimental_same_event_preference_ranker_v2'}))
    args.extend(['--editorial-opportunity-exposure','--played-pose-port-camera-prior','--native-part-pixel-support',
        '--joint-motion-event-seconds','--fixed-actor-camera-holds','--native-source-readability',
        '--native-camera-ports','--body-pose-model','unused'])
    with pytest.raises(ValueError,match='new fitted sequence'):main()
