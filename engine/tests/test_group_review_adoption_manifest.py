from copy import deepcopy
import pytest
from crosscut.group_review_paired_evidence import PairedEvidencePlan
from crosscut.group_review_adoption_manifest import (freeze_review_adoption_manifest,
    validate_keyed_adoption_receipts,decide_keyed_review_adoption)


def fixture(*,old=(10,),new=(10,12),same=()):
    cuts=tuple(sorted({10,15,*old,*new}))
    plan=PairedEvidencePlan(40,30.,10,15,'a'*64,'b'*64,incident_ticks=cuts)
    bindings={t:dict(original='c'*64,candidate='c'*64 if t in same else 'd'*64) for t in cuts}
    m=freeze_review_adoption_manifest(plan,original_plan_identity='e'*64,
        candidate_plan_identity='f'*64,original_cut_ticks=old,candidate_cut_ticks=new,
        incident_bindings=bindings,motion_policy_identity='9'*64,
        support_part_inventory={t:('head',) for t in plan.interior_ticks})
    receipts=[]
    for r in m.payload['requests']:
        receipt=dict(r,manifest_identity=m.identity,evidence_plan_identity=plan.identity,
            original_plan_identity='e'*64,candidate_plan_identity='f'*64)
        if r['domain']=='SUPPORT':receipt.update(original_state='PASS',candidate_state='PASS')
        else:
            verdict=dict(UNCHANGED='UNCHANGED',NEW_CUT='SUPPORTED_NEW_MATCH',
                REPLACED_CUT='NONREGRESSING',REMOVED_CUT='CONTINUITY_SUPPORTED',
                CONTINUOUS_CAMERA_EDIT='CAMERA_CONTINUITY_SUPPORTED')[r['kind']]
            receipt.update(verdict=verdict,motion_policy_identity='9'*64,
                same_time_action_supported=True,independent_suitability_passed=True,
                original_motion_cost=.3,candidate_motion_cost=.2,frozen_nonregression_tolerance=0.,
                actual_sampler_continuity_verified=True)
        receipts.append(receipt)
    comparison=dict(verdict='BETTER',evidence_plan_sha256=plan.identity)
    return m,receipts,comparison


def test_new_cut_is_not_compared_to_fabricated_continuous_original_zero():
    m,receipts,c=fixture();new=next(r for r in receipts if r.get('kind')=='NEW_CUT')
    new.pop('original_motion_cost');new.pop('candidate_motion_cost')
    result=decide_keyed_review_adoption(manifest=m,receipts=receipts,comparison=c)
    assert result['state']=='REVIEW_ENCODE_ELIGIBLE'
    assert not result['quality_approved'] and not result['professional_quality_certified']


@pytest.mark.parametrize('mutation',('empty','exit_missing','duplicate','foreign_plan','foreign_manifest','extra'))
def test_receipt_inventory_cannot_hide_missing_exit_or_borrow_results(mutation):
    m,rows,c=fixture()
    if mutation=='empty':rows=[]
    elif mutation=='exit_missing':rows=[r for r in rows if not(r['domain']=='MOTION' and r['tick']==15)]
    elif mutation=='duplicate':rows.append(deepcopy(rows[0]))
    elif mutation=='foreign_plan':rows[0]['candidate_plan_identity']='1'*64
    elif mutation=='foreign_manifest':rows[0]['manifest_identity']='1'*64
    else:rows.append(dict(rows[0],request_id='1'*64))
    with pytest.raises(ValueError):validate_keyed_adoption_receipts(m,rows)


def test_known_original_support_loss_is_hold_but_original_unknown_remains_unknown():
    m,rows,c=fixture();rows[0]['candidate_state']='UNKNOWN'
    assert decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)['state']=='HOLD'
    rows[0]['original_state']='UNKNOWN'
    result=decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)
    assert result['state']=='REVIEW_ENCODE_ELIGIBLE' and result['unresolved_original_support']==1


def test_changed_actual_window_cannot_claim_unchanged():
    m,rows,c=fixture();r=next(r for r in rows if r['domain']=='MOTION')
    r['kind']='UNCHANGED';r['verdict']='UNCHANGED'
    with pytest.raises(ValueError,match='incident kind'):decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)


@pytest.mark.parametrize('field',('same_time_action_supported','independent_suitability_passed'))
def test_a_low_exploration_cost_or_observation_only_is_not_new_match_suitability(field):
    m,rows,c=fixture();next(r for r in rows if r.get('kind')=='NEW_CUT')[field]=False
    assert decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)['state']=='HOLD'


def test_removed_cut_needs_actual_sampler_continuity_not_an_invented_match_score():
    m,rows,c=fixture(old=(10,12),new=(10,))
    r=next(r for r in rows if r.get('kind')=='REMOVED_CUT');r['actual_sampler_continuity_verified']=False
    assert decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)['state']=='HOLD'


def test_replaced_cut_cannot_lie_about_its_nonregression_verdict():
    m,rows,c=fixture();r=next(r for r in rows if r.get('kind')=='REPLACED_CUT')
    r['candidate_motion_cost']=.8
    assert decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)['state']=='REJECT'


def test_actual_unchanged_window_requires_no_new_motion_claim():
    m,rows,c=fixture(same=(10,15));result=decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)
    assert result['state']=='REVIEW_ENCODE_ELIGIBLE'


def test_expected_manifest_itself_cannot_omit_exit_before_receipts_are_frozen():
    plan=PairedEvidencePlan(40,30.,10,15,'a'*64,'b'*64,incident_ticks=(10,))
    with pytest.raises(ValueError,match='entry/exit'):freeze_review_adoption_manifest(plan,
        original_plan_identity='e'*64,candidate_plan_identity='f'*64,
        original_cut_ticks=(10,),candidate_cut_ticks=(10,),
        incident_bindings={10:dict(original='c'*64,candidate='d'*64)},motion_policy_identity='9'*64,
        support_part_inventory={t:('head',) for t in plan.interior_ticks})


def test_real_consumer_requires_keyed_manifest_and_preserves_review_only_status():
    from crosscut.group_review_realized_path import require_paired_motion_adoption
    m,rows,c=fixture();gate=require_paired_motion_adoption(comparison=c,expected_manifest=m,keyed_receipts=rows)
    assert gate['numeric_review_encode_permitted'] and not gate['quality_approved']
    with pytest.raises(ValueError,match='cannot permit encode'):
        require_paired_motion_adoption(comparison=c,expected_manifest=m,keyed_receipts=rows,
            motion_verdicts=['UNCHANGED'])


def test_unknown_left_arm_cannot_hide_loss_of_previously_known_right_arm():
    plan=PairedEvidencePlan(40,30.,10,11,'a'*64,'b'*64,incident_ticks=(10,11))
    m=freeze_review_adoption_manifest(plan,original_plan_identity='e'*64,candidate_plan_identity='f'*64,
        original_cut_ticks=(),candidate_cut_ticks=(),motion_policy_identity='9'*64,
        incident_bindings={t:dict(original='c'*64,candidate='c'*64) for t in (10,11)},
        support_part_inventory={10:('left_arm','right_arm')})
    rows=[]
    for request in m.payload['requests']:
        row=dict(request,manifest_identity=m.identity,evidence_plan_identity=plan.identity,
            original_plan_identity='e'*64,candidate_plan_identity='f'*64)
        if request['domain']=='SUPPORT':
            row.update(original_state='PASS' if request['part']=='right_arm' else 'UNKNOWN',candidate_state='UNKNOWN')
        else:row.update(verdict='UNCHANGED',motion_policy_identity='9'*64)
        rows.append(row)
    c=dict(verdict='BETTER',evidence_plan_sha256=plan.identity)
    assert decide_keyed_review_adoption(manifest=m,receipts=rows,comparison=c)['state']=='HOLD'
