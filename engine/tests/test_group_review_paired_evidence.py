"""Equal request grids and deletion-monotonic review comparisons, no media."""
from copy import deepcopy
import pytest
from crosscut.group_review_paired_evidence import (PairedEvidencePlan,
    compare_paired_metric_series, time_normalized_center_acceleration,validate_paired_native_request_inventory)


def world():
    plan=PairedEvidencePlan(100,30.,20,30,'a'*64,'b'*64)
    requests=dict(interior_ticks=plan.interior_ticks,left_ticks=plan.left_ticks,
        right_ticks=plan.right_ticks,appearance_ticks=plan.appearance_ticks)
    before=dict(evidence_plan_sha256=plan.identity, requests=requests,
        metrics=dict(detail=[.6]*10,seam=[.8,.9],pose=[None,None]))
    return plan,before,deepcopy(before)


def compare(p,a,b):return compare_paired_metric_series(plan=p,original=a,candidate=b,
    weights=dict(detail=1.,seam=1.,pose=1.))


def test_identity_candidate_has_zero_measured_delta_and_no_quality_certificate():
    p,a,b=world();result=compare(p,a,b)
    assert result['weighted_candidate_minus_original']==0.
    assert result['verdict']=='INCONCLUSIVE' and not result['quality_approved']


@pytest.mark.parametrize('field', ['interior_ticks','left_ticks','right_ticks','appearance_ticks'])
def test_asymmetric_sampling_or_appearance_requests_are_not_improvement(field):
    p,a,b=world();b['requests']=dict(b['requests'],**{field:()})
    with pytest.raises(ValueError,match='grid|incident|appearance'):compare(p,a,b)


@pytest.mark.parametrize('metric,item',[('seam',1),('detail',4)])
def test_erasing_adverse_measurement_cannot_manufacture_better(metric,item):
    p,a,b=world();b['metrics'][metric][item]=None
    # Other genuinely better observations cannot erase the missing endpoint.
    b['metrics']['detail'][0]=.1
    result=compare(p,a,b)
    assert result['verdict']=='INCONCLUSIVE' and result['lost_known_evidence']


def test_new_pose_coverage_alone_has_no_comparative_improvement():
    p,a,b=world();b['metrics']['pose']=[0.,0.]
    result=compare(p,a,b)
    assert result['verdict']=='INCONCLUSIVE' and result['weighted_candidate_minus_original']==0.
    assert len(result['gained_known_evidence'])==2


def test_measured_readability_gain_can_pay_for_measured_seam_style_tradeoff():
    p,a,b=world();b['metrics']['detail']=[.1]*10;b['metrics']['seam']=[.95,.95]
    assert compare(p,a,b)['verdict']=='BETTER'


def test_constant_velocity_does_not_depend_on_sample_spacing():
    dense=(0,1,2,3,4,5);sparse=(0,2,5)
    make=lambda ts:tuple((.1+t*.003,.3-t*.002) for t in ts)
    assert time_normalized_center_acceleration(dense,make(dense),30.)==pytest.approx([0.]*4,abs=1e-9)
    assert time_normalized_center_acceleration(sparse,make(sparse),30.)==pytest.approx([0.],abs=1e-9)


def test_native_aliases_do_not_delete_output_time_or_request_positions():
    p,a,b=world()
    # Native 4590 may occupy several output ticks; all ten output observations
    # remain in the contract regardless of model-call deduplication.
    a['native_ids']=b['native_ids']=[4590,4590,4591,4591,4592,4592,4593,4593,4594,4594]
    assert len(p.interior_ticks)==10 and compare(p,a,b)['weighted_candidate_minus_original']==0.


def test_union_of_original_and_candidate_cut_appearance_positions_is_identical():
    p=PairedEvidencePlan(100,30.,20,30,'a'*64,'b'*64,extra_appearance_ticks=(21,22,28,29))
    assert {21,22,28,29} <= set(p.appearance_ticks)
    assert p.identity != world()[0].identity
    with pytest.raises(ValueError,match='frozen common'):
        PairedEvidencePlan(100,30.,20,30,'a'*64,'b'*64,extra_appearance_ticks=(5,))


def capture_fixture():
    plan=world()[0];ticks=sorted({*plan.interior_ticks,*plan.left_ticks,*plan.right_ticks})
    capture=dict(full_plan_identity='c'*64,requested_output_ticks=ticks,
        requested_appearance_ticks=list(plan.appearance_ticks),patches={str(t):{} for t in ticks},
        appearance={str(t):[] for t in plan.appearance_ticks},rows={str(t):dict(output_tick=t,
            native_observation=dict(request_source_index=0,native_frame_index=t//2,source_clock_identity='d'*64),
            projection=dict(source_request_index=0,native_frame_index=t//2,source_clock_identity='d'*64)) for t in ticks})
    return plan,capture


def test_actual_capture_keeps_native_alias_output_rows_and_empty_appearance_results():
    p,capture=capture_fixture()
    result=validate_paired_native_request_inventory([p],capture,expected_render_plan_identity='c'*64)
    assert result['requested_output_ticks']==16 and result['actual_capture_keys_verified']
    assert not result['fresh_native_authority_certified']


@pytest.mark.parametrize('domain',['rows','patches','appearance'])
def test_actual_capture_cannot_copy_request_lists_while_omitting_results(domain):
    p,capture=capture_fixture();capture[domain].pop(next(iter(capture[domain])))
    with pytest.raises(ValueError,match='actual capture'):
        validate_paired_native_request_inventory([p],capture,expected_render_plan_identity='c'*64)


def test_actual_capture_cannot_relabel_output_tick_or_native_clock():
    p,capture=capture_fixture();capture['rows']['20']['projection']['native_frame_index']=999
    with pytest.raises(ValueError,match='native clock'):
        validate_paired_native_request_inventory([p],capture,expected_render_plan_identity='c'*64)


def test_incident_union_includes_exit_even_when_internal_metrics_are_identical():
    p=PairedEvidencePlan(100,30.,20,30,'a'*64,'b'*64,
        incident_ticks=(20,23,26,30),extra_appearance_ticks=(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32))
    requests=dict(interior_ticks=p.interior_ticks,left_ticks=p.left_ticks,
        right_ticks=p.right_ticks,appearance_ticks=p.appearance_ticks)
    a=dict(evidence_plan_sha256=p.identity,requests=requests,metrics=dict(detail=[.2]*10,seam=[.2]*4))
    b=deepcopy(a);b['metrics']['seam'][-1]=1.2
    result=compare_paired_metric_series(plan=p,original=a,candidate=b,weights=dict(detail=1.,seam=1.))
    assert result['verdict']=='WORSE'
    assert result['components']['seam']['total_items']==4
    assert p.to_dict()['incident_ticks'][-1]==30
    assert {30,31,32}<=set(p.appearance_ticks)


def test_incident_cut_union_cannot_be_omitted_from_one_series():
    p=PairedEvidencePlan(100,30.,20,30,'a'*64,'b'*64,incident_ticks=(20,23,26,30))
    requests=dict(interior_ticks=p.interior_ticks,left_ticks=p.left_ticks,
        right_ticks=p.right_ticks,appearance_ticks=p.appearance_ticks)
    a=dict(evidence_plan_sha256=p.identity,requests=requests,metrics=dict(seam=[.2]*4))
    b=deepcopy(a);b['metrics']['seam']=[.2]*3
    with pytest.raises(ValueError,match='incident sample domain'):
        compare_paired_metric_series(plan=p,original=a,candidate=b,weights=dict(seam=1.))


def test_legacy_pair_contract_has_no_unrequested_incident_domain():
    assert 'incident_ticks' not in world()[0].to_dict()
