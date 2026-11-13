from copy import deepcopy
import pytest
from crosscut.group_review_paired_evidence import PairedEvidencePlan
from crosscut.group_review_selected_pair_consumer import (selected_pair_side,
    selected_incident_measurements,paired_selected_support)
from engine.tests.test_group_review_motion_seam import view


def fixture():
    plan=PairedEvidencePlan(40,30.,10,15,'a'*64,'b'*64,appearance_stride=1,incident_ticks=(10,15))
    rows={t:view(t,t*.02) for t in range(7,18)}
    for v in rows.values():v['detail_risk']=.2
    states={t:'SOURCE_LOCAL_APPEARANCE_HYPOTHESIS' for t in rows}
    return plan,rows,states


def test_exit_geometry_is_consumed_even_with_identical_interiors():
    plan,rows,states=fixture()
    before=selected_pair_side(plan,rows,association_states=states,full_plan_identity='c'*64)
    changed=deepcopy(rows)
    from dataclasses import replace
    changed[15]['descriptor']=replace(changed[15]['descriptor'],head_center=(.9,.9))
    after=selected_pair_side(plan,changed,association_states=states,full_plan_identity='d'*64)
    assert before['metrics']['closeup_distance']==after['metrics']['closeup_distance']
    assert after['metrics']['incident_geometry'][1]>before['metrics']['incident_geometry'][1]
    assert before['entry_and_exit_consumed']


def test_unmatched_selected_actor_never_uses_another_largest_descriptor():
    plan,rows,states=fixture();rows[12]=None;states[12]='UNMATCHED'
    result=selected_pair_side(plan,rows,association_states=states,full_plan_identity='c'*64)
    assert result['metrics']['closeup_distance'][2] is None
    assert result['metrics']['actor_association_uncertainty'][2]==1.


def test_actual_incident_union_measures_original_continuation_without_fabricated_zero():
    plan,rows,states=fixture()
    result=selected_incident_measurements(plan,rows,actual_cut_ticks={10})
    assert [v['tick'] for v in result]==[10,15]
    assert not result[1]['actual_visible_cut']
    assert result[1]['body']['state']=='OBSERVED_BODY_PHASE_HYPOTHESIS'
    assert not result[1]['continuous_original_assigned_zero']


def test_mixed_source_flank_is_unknown_not_a_temporal_contract_exception():
    plan,rows,states=fixture();rows[13]['row']['native_observation']['request_source_index']=8
    assert selected_incident_measurements(plan,rows,actual_cut_ticks={15})[1]['body']['state']=='INCONCLUSIVE'


def test_physical_run_is_independent_of_authored_camera_and_never_inferred_when_missing():
    plan,rows,states=fixture()
    for t,v in rows.items():v['row']['projection']['camera_sha256']=f'pan-{t}'
    assert selected_incident_measurements(plan,rows,actual_cut_ticks={10})[0]['body']['state']=='OBSERVED_BODY_PHASE_HYPOTHESIS'
    rows[9]['native_camera_unit_index']=1
    assert selected_incident_measurements(plan,rows,actual_cut_ticks={10})[0]['body']['state']=='INCONCLUSIVE'
    rows[9].pop('native_camera_unit_index')
    assert selected_incident_measurements(plan,rows,actual_cut_ticks={10})[0]['body']['state']=='INCONCLUSIVE'


def test_known_support_loss_cannot_use_the_original_unknown_allowance():
    original=[dict(output_tick=3,state='PASS'),dict(output_tick=4,state='UNKNOWN')]
    candidate=[dict(output_tick=3,state='UNKNOWN'),dict(output_tick=4,state='UNKNOWN')]
    result=paired_selected_support(original,candidate)
    assert [v['state'] for v in result]==['FAIL','UNKNOWN']
    assert result[0]['lost_known_support']


def test_missing_exit_or_empty_support_cannot_be_called_verified():
    plan,rows,states=fixture();rows.pop(15)
    with pytest.raises(ValueError,match='entry/exit'):
        selected_pair_side(plan,rows,association_states=states,full_plan_identity='c'*64)
    with pytest.raises(ValueError,match='nonempty'):
        paired_selected_support([],[])
