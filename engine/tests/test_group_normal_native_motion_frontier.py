"""Real frontier/phase/camera/DP/lease over labelled SYNTHETIC input rows."""
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import pytest

from crosscut.group_source_changing_owner import (_balanced_native_motion_tuples,
    realize_normal_native_motion_frontier)
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.first_result_controller import FirstResultController
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.request_work_lease import PhaseAllocation,PHASES,RESOURCES,current_request_work_phase
from tests.test_group_native_motion_inventory import SyntheticSensors
from tests.test_group_native_motion_camera_producer import inputs


def test_declaration_is_deterministic_source_balanced_and_bounded_before_donor_costs():
    sources=(0,2,3,5,6,7,8,9,10,11)
    rows=_balanced_native_motion_tuples(sources,(40,45,50),(-1,0,1),64)
    assert rows==_balanced_native_motion_tuples(sources,(40,45,50),(-1,0,1),64)
    assert len(rows)==len(set(rows))==64
    assert {r[0] for r in rows}=={r[1] for r in rows}==set(sources)
    assert {r[2] for r in rows}=={40,45,50}
    assert all(a!=b for a,b,*_ in rows)
    assert {r[3] for r in rows}=={r[4] for r in rows}=={-1,0,1}
    with pytest.raises(ValueError,match='externally declared'):
        _balanced_native_motion_tuples(sources,(40,),(-1,0,1),65)


def lease_for(args,tmp_path):
    plan=args['context'].incumbent.render_plan
    owner=FirstResultController(ReviewFirstRuntime(enabled=True),sources=plan.sources,
        source_durations=(200.,)*len(plan.sources),master_start=plan.master_audio_start,duration=plan.duration)
    owner.install_baseline(plan,plan=plan,authority='synthetic native frontier accounting only')
    allocations={p:PhaseAllocation(dict(journal_bytes=2*1024**2,native_completion_passes=int(p=='selected_completion'),
        states=4096,edges=4096,association_nodes=200000,association_edges=500000,
        pose_evidence_bytes=16*1024**2,model_calls=0,new_decodes=0),10.) for p in PHASES}
    lease=owner.reserve_request_work(total_limits={k:sum(a.resources[k] for a in allocations.values())
        for k in RESOURCES},allocations=allocations,journal_path=tmp_path/'work.jsonl')
    return owner,lease


@pytest.mark.parametrize('member_conflict',(False,True))
def test_real_selection_stage_keeps_one_completion_and_exact_new_sampler_aliases(tmp_path,member_conflict):
    args=inputs(member_hint_by_source={3:1} if member_conflict else None);owner,lease=lease_for(args,tmp_path)
    def run():
        # JointGroupWorkBudget requires positive local capacities; the active
        # request phase below has authoritative ZERO decode/model allowances.
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.,maximum_model_calls=1,maximum_new_decodes=1)
        sensors=SyntheticSensors(budget)
        source_session=args['views'].sensors.session
        sensors.session=SimpleNamespace(domain=args['domain'],budget=budget,closed=False,
            models=source_session.models,model_adapters=source_session.model_adapters,
            _assert_model_assets=source_session._assert_model_assets,
            _assert_source_identity=source_session._assert_source_identity)
        sensors.recorded_motion_bindings={}
        sensors.member_gallery=getattr(args['views'].sensors,'member_gallery',None)
        for key,supply in args['views'].supplies.items():
            if supply.observation is not None:
                sensors.raw_rows[key]=supply.observation
                sensors.appearance_rows[key[:2]]=supply.appearance
                if supply.member_features is not None:
                    from crosscut.group_performer_family_owner import digest
                    sensors.member_feature_rows[digest(supply.observation.to_dict()),sensors.member_gallery.identity]=supply.member_features
        candidates,report,views=realize_normal_native_motion_frontier(context=args['context'],sensors=sensors,
            first_output_frame=30,end_output_frame=60,cut_ticks=(45,),native_phases=(0,),maximum_tuples=48)
        assert candidates and report['nonregressing_camera_candidates']
        if member_conflict:
            assert any(c.evidence['local_member_hint_conflict_ticks'] for c in candidates)
            assert all(not c.evidence['member_hints_are_hard_eligibility_authority'] for c in candidates)
        assert report['native_completion_consumed'] and report['original_actor_inventory']['donor_costs_used'] is False
        assert report['native_alias_requests']>report['unique_native_requests']
        assert report['demand_basis']=='exact_new_topology_authorized_native_sampler_not_old_discovery_clock'
        assert not report['full_normal_metadata_constructed'] and not report['quality_approved']
        assert len(report['trials'])==report['proposed_joint_tuples']==48
        # This fixture deliberately binds all sources to one synthetic file
        # identity. Aliases are ONE family, not independent evidence.
        assert report['distinct_frozen_source_families']==1
        assert sum(report['unique_native_supply_states'].values())==report['unique_native_requests']
        assert sum(report['native_supply_states'].values())==report['native_alias_requests']
        assert report['original_context_inputs'] and all(not r['private_embeddings_included']
            for r in report['original_context_inputs'])
        assert report['stage_seconds']['cache_census']>=0
        assert all(len(r['endpoint_inputs'])==4 for r in report['trials']
            if r['state']!='INADMISSIBLE_NATIVE_TOPOLOGY')
        assert budget.counts['new_decodes']==budget.counts['model_calls']==0
        assert not sensors.raw_calls and not sensors.appearance_calls
        assert current_request_work_phase().charged['native_completion_passes']==1
        with pytest.raises(ValueError,match='already consumed'):
            realize_normal_native_motion_frontier(context=args['context'],sensors=sensors,
                first_output_frame=30,end_output_frame=60,cut_ticks=(45,),native_phases=(0,),maximum_tuples=1)
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


@pytest.mark.parametrize('stage',('enumeration','balanced_linear_pass'))
def test_large_finite_declaration_checks_cancellation_inside_its_cpu_work(stage):
    calls=0
    # 12*12*3*5*5*4 theoretical rows => 338 enumeration checks.
    stop_at=2 if stage=='enumeration' else 339
    def check():
        nonlocal calls
        calls+=1
        if calls==stop_at:raise InterruptedError('synthetic declaration cancellation')
    with pytest.raises(InterruptedError,match='declaration cancellation'):
        _balanced_native_motion_tuples(tuple(range(12)),(40,45,50),(-2,-1,0,1,2),64,check=check)
    assert calls==stop_at


def test_missing_fixed_context_does_not_enter_a_camera_fit_or_invent_keep_score(tmp_path):
    args=inputs(missing_tick=60);owner,lease=lease_for(args,tmp_path)
    def run():
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.,maximum_model_calls=1,maximum_new_decodes=1)
        sensors=SyntheticSensors(budget);source_session=args['views'].sensors.session
        sensors.session=SimpleNamespace(domain=args['domain'],budget=budget,closed=False,
            models=source_session.models,model_adapters=source_session.model_adapters,
            _assert_model_assets=source_session._assert_model_assets,
            _assert_source_identity=source_session._assert_source_identity)
        sensors.recorded_motion_bindings={}
        for key,supply in args['views'].supplies.items():
            if supply.observation is not None:
                sensors.raw_rows[key]=supply.observation;sensors.appearance_rows[key[:2]]=supply.appearance
        candidates,report,_=realize_normal_native_motion_frontier(context=args['context'],sensors=sensors,
            first_output_frame=30,end_output_frame=60,cut_ticks=(45,),native_phases=(0,),maximum_tuples=16)
        assert not candidates and report['realized_camera_candidates']==0
        missing=[r for r in report['trials'] if r['state']=='UNOBSERVED_ENDPOINT_ANATOMY']
        assert missing and all(60 in r['receipt']['missing_landmark_ticks'] for r in missing)
        assert all(r['receipt']['camera_fitting_started'] is False
            and r['receipt']['measured_keep_incidents'] is None
            and r['receipt']['missing_values_used_as_perfect_keep_score'] is False for r in missing)
        assert report['stage_seconds']['joint_camera_and_incident_measurement']==0
        assert budget.counts['new_decodes']==budget.counts['model_calls']==0
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


@pytest.mark.parametrize('damage',('wrist_jump','cut_side_stop','hold_reversal','unsampled_wrist_loss'))
def test_final_frontier_does_not_count_face_fitted_known_motion_damage_as_nonregression(tmp_path,damage):
    args=inputs(body_damage=damage);owner,lease=lease_for(args,tmp_path)
    def run():
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.,maximum_model_calls=1,maximum_new_decodes=1)
        sensors=SyntheticSensors(budget);source_session=args['views'].sensors.session
        sensors.session=SimpleNamespace(domain=args['domain'],budget=budget,closed=False,
            models=source_session.models,model_adapters=source_session.model_adapters,
            _assert_model_assets=source_session._assert_model_assets,
            _assert_source_identity=source_session._assert_source_identity)
        sensors.recorded_motion_bindings={}
        for key,supply in args['views'].supplies.items():
            if supply.observation is not None:
                sensors.raw_rows[key]=supply.observation;sensors.appearance_rows[key[:2]]=supply.appearance
        candidates,report,_=realize_normal_native_motion_frontier(context=args['context'],sensors=sensors,
            first_output_frame=30,end_output_frame=60,cut_ticks=(45,),native_phases=(0,),maximum_tuples=48)
        target=[r for r in report['trials'] if tuple(r['spec'])==(2,3,45,0,0,0,0)]
        assert len(target)==1 and target[0]['camera_candidates']
        rejected=target[0]['camera_candidates']
        assert all(r['state']=='KNOWN_INCIDENT_OR_HOLD_REGRESSION' for r in rejected)
        assert all(r['known_incident_failures'] or r['known_hold_observation_losses'] or (
            r['known_hold_motion_failures']) for r in rejected)
        rejected_plans={r['selected_plan_identity'] for r in rejected}
        from crosscut.group_performer_family_owner import digest
        assert not rejected_plans.intersection(digest(c.render_plan.to_json()) for c in candidates)
        trials=[c for row in report['trials'] for c in row.get('camera_candidates',())]
        assert report['nonregressing_camera_candidates']==len(candidates)==sum(
            c['state']=='REALIZED_FOR_REVIEW' for c in trials)
        assert report['realized_camera_candidates']>report['nonregressing_camera_candidates']
        assert budget.counts['new_decodes']==budget.counts['model_calls']==0
        assert not sensors.raw_calls and not sensors.appearance_calls and not report['quality_approved']
        assert current_request_work_phase().charged['native_completion_passes']==1
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()
