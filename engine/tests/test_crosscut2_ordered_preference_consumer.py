"""Numeric synthetic fixtures only; never real AV labels or trained assets."""
from copy import deepcopy
import numpy as np
import pytest
from crosscut2.ordered_context import NAMES,PERSON_NAMES,SCHEMA_SHA256,project
from crosscut2.ordered_preference_model import OrderedModel,CONTRACT,ARCHITECTURE,load_model
from crosscut2.sequence_preference_model import digest
from crosscut2.context_port_dp import ArcRef
from test_crosscut2_ordered_context import observed_scene
from test_crosscut2_sequence_preference_consumer import context_fixture,prepared,arguments


def numeric_ordered_model(*,fps=30.):
    D=2;F=len(NAMES);H=int(np.ceil(4.*fps))
    config=dict(schema_sha256=SCHEMA_SHA256,interval_frames=6,context_seconds=4.,clock_fps=fps,context_frames=H)
    value=np.zeros((F,D));value[NAMES.index('additional_log_zoom'),0]=2.
    value[NAMES.index('feature_center_x'),1]=.25
    person=np.zeros((len(PERSON_NAMES),D));person[PERSON_NAMES.index('left_wrist.screen_x'),1]=1.
    temporal=np.stack((np.linspace(.001,.003,H),np.linspace(-.001,.002,H)),axis=1)
    payload=dict(contract=CONTRACT,architecture=ARCHITECTURE,output_kind='group',fitted=True,
        feature_schema_sha256=SCHEMA_SHA256,feature_names=dict(scene=list(NAMES),person=list(PERSON_NAMES)),window_config=config,
        window_contract_sha256=digest(config),embedding_width=D,
        center=np.zeros(F).tolist(),scale=np.ones(F).tolist(),encoder_values=value.tolist(),
        encoder_known=np.zeros((F,D)).tolist(),encoder_conflicted=np.zeros((F,D)).tolist(),encoder_bias=np.zeros(D).tolist(),
        person_center=np.zeros(len(PERSON_NAMES)).tolist(),person_scale=np.ones(len(PERSON_NAMES)).tolist(),
        person_values=person.tolist(),person_known=np.zeros(person.shape).tolist(),person_conflicted=np.zeros(person.shape).tolist(),
        person_bias=np.zeros(D).tolist(),set_weights=np.eye(D).tolist(),
        temporal_weights=temporal.tolist(),bias=.1,
        training=dict(supervision_kind='human_av_context_preference',self_reported_normal_av=True,
            relative_and_absolute_labels_separate=True,physical_context_overlap_grouped=True,
            sealed_evaluation_excluded_from_fit=True,examples_sha256='a'*64,
            train_physical_groups=['numeric-unit-train'],dev_physical_groups=['numeric-unit-dev'],
            absolute_positive_training_options=1,absolute_negative_training_options=1),
        normal_app_activation_approved=False,professional_quality_approved=False)
    payload['content_sha256']=digest(payload)
    return OrderedModel(payload)


def ordered_fixture(nodes,*,nonlinear=False,kernel='numpy'):
    old,arcs=context_fixture(nodes,nonlinear_clock=nonlinear)
    model=numeric_ordered_model()
    new=model.prepare(nodes=old.nodes,entries=old.entries,exits=old.exits,catalogues=old.cats,arcs=arcs,
        native_ports=arcs.native,fps=30.,source_bindings=['a'*64,'b'*64],check=lambda:None,kernel=kernel)
    return new


def arc(context,i,j,s,code):
    p=next(k for k,v in enumerate(context.entries[i]) if v.source==s)
    q=next(k for k,v in enumerate(context.exits[j]) if v.source==s)
    return ArcRef(i,j,p,q,code)


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_ordered_causal_kernel_keeps_body_order_and_is_bitwise_same_in_strict_compilation(kernel):
    if kernel=='numba':pytest.importorskip('numba')
    model=numeric_ordered_model();s=observed_scene();camera=np.tile([1.,.5,.5],(120,1))
    data=project(s,camera);encoded=model.encode(data)
    # Order is retained by the positional FIR rather than mean/max pooling.
    encoded[:,1]=np.arange(120)/120.
    ends=np.arange(6,121,6)
    a=model.rates(np.arange(120),encoded[None],ends,kernel='numpy')
    b=model.rates(np.arange(120),encoded[None],ends,kernel=kernel)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,model.rates(np.arange(120),encoded[::-1][None],ends,kernel=kernel))
    with pytest.raises(ValueError,match='pooled'):model.costs(np.zeros((1,189)))


@pytest.mark.parametrize('nonlinear',[False,True])
@pytest.mark.parametrize('code',range(6))
def test_all_codes_mixed_history_real_clocks_and_long_arc_match_independent_ordered_replay(nonlinear,code):
    context=ordered_fixture(np.arange(0,421,30),nonlinear=nonlinear)
    refs=[arc(context,0,1,0,1),arc(context,1,3,1,2),arc(context,3,14,0,code)]
    history=[];costs=[]
    for ref in refs:costs.append(float(context.incremental([tuple(history)],ref)[0]));history.append(ref)
    parts,total=context.replay(refs)
    assert costs==pytest.approx(parts,abs=2e-13) and sum(costs)==pytest.approx(total,abs=2e-13)
    if code<4 and not nonlinear:assert context.fast>0
    receipt=context.receipt()
    assert receipt['ordered_native_and_displayed_body_geometry'] and receipt['source_NN_calls']==0
    assert receipt['storage']['per_frame_samples_preserved']
    assert receipt['temporal_operation_order']=='chronological_scalar_no_history_current_reassociation'
    assert not receipt['professional_quality_approved']


def test_actual_selector_consumes_ordered_model_without_scalar_argmin_or_fake_transformer():
    from crosscut2.event_holds import select
    nodes=np.array([0,18,36,54]);cats=prepared(nodes)
    result=select(nodes,[c.bank for c in cats],None,[c.style for c in cats],**arguments(nodes,cats),
        sequence_preference_model=numeric_ordered_model(),sequence_source_bindings=['a'*64,'b'*64])
    receipt=result[4]
    assert receipt['sequence_preference_consumed'] and not receipt['pre_DP_scalar_camera_argmin_used']
    assert receipt['sequence_preference_context']['ordered_schema_sha256']==SCHEMA_SHA256
    assert receipt['boundary_port_DP']['camera_alternatives_scored']>0
    assert not receipt['stage_edit_transformer_used'] and not receipt['professional_quality_approved']


def test_ordered_artifact_cannot_accept_weak_labels_unsealed_fit_old_schema_or_mutated_parameters():
    model=numeric_ordered_model();assert isinstance(load_model(model.artifact),OrderedModel)
    for name,value in (('supervision_kind','opus_silent_still_advisory'),('physical_context_overlap_grouped',False),
            ('sealed_evaluation_excluded_from_fit',False),('absolute_positive_training_options',0)):
        payload=deepcopy(model.artifact);payload['training'][name]=value;payload.pop('content_sha256')
        payload['content_sha256']=digest(payload)
        with pytest.raises(ValueError,match='genuine grouped AV'):OrderedModel(payload)
    model.artifact['bias']+=.1
    with pytest.raises(ValueError,match='parameters changed'):model.verify()


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_exact_long_batch_keeps_full_domain_paid_bits_cameras_and_near_tie_rank(kernel):
    if kernel=='numba':pytest.importorskip('numba')
    from crosscut2.context_port_dp import solve
    nodes=np.array([0,30,180,210,240]);results=[]
    for batched in (False,True):
        context=ordered_fixture(nodes,kernel=kernel);arcs=context.arcs
        arcs.opportunity_episodes=np.zeros(len(nodes),int)
        def cut(n):
            a=np.array([p.source for p in context.exits[n]]);b=np.array([p.source for p in context.entries[n]])
            return np.zeros((len(a),len(b))),a[:,None]!=b[None,:]
        def holds(i,j):
            cost,allowed=arcs.camera_alternatives(i,j);cost=cost.copy()
            # A sub-tolerance base-cost difference is NOT a context alias.
            if cost.shape[-1]>4:cost[:,:,4]+=np.finfo(float).eps
            return cost,allowed
        runs,objective,receipt=solve(nodes,[[p.source for p in ps] for ps in context.entries],
            [[p.source for p in ps] for ps in context.exits],start_cost=np.zeros(len(context.entries[0])),
            start_allowed=np.array([p.source==0 for p in context.entries[0]]),
            final_allowed=np.array([p.source==1 for p in context.exits[-1]]),
            hold_cost=holds,cut_cost=cut,context_cost=context.incremental,history_key=context.equivalent_key,
            context_lower_bound=context.lower_bound,long_context_batch=context.long_candidates if batched else None,
            episodes=np.array([0,0,1,1,2]),credit=lambda i,j:np.full(len(context.entries[i]),.025 if i else 0.),
            history_frames=context.history_frames,label_limit=8)
        refs=[ArcRef(*run,int(code)) for run,code in zip(runs,receipt['selected_arc_codes'])]
        parts,total=context.replay(refs)
        assert parts==pytest.approx(receipt['selected_context_costs'],abs=2e-13)
        assert sum(receipt['selected_context_costs'])==pytest.approx(total,abs=2e-13)
        if batched:assert receipt['exact_long_suffix_batched_holds']>0
        results.append((runs,objective,receipt['selected_arc_codes'],receipt['opportunity_claims'],
            receipt['camera_alternatives_scored']))
    assert results[0]==results[1]
