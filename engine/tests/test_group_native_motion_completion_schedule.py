"""Real numeric settlement/lease tests; no invented detector or NN outputs."""
from time import monotonic

import cv2
import numpy as np
import pytest

from crosscut.first_result_controller import FirstResultController
from crosscut.group_joint_primitive_journal import primitive_reservation_bytes, validate_journal
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_native_motion_completion_schedule import (
    COMPLETION_RESOURCES, NativeMotionCompletionCost, NativeMotionCompletionPackage,
    balanced_completion_schedule, coherent_completion_schedule,
)
from crosscut.group_source_changing_owner import NativeMotionDemand, NativeMotionInventory, NativeMotionSupply
from crosscut.render import RenderPlan, RenderSegment
from crosscut.request_work_lease import PHASES, RESOURCES, PhaseAllocation
from crosscut.review_first_runtime import ReviewFirstRuntime


def lease_owner(tmp_path, byte_capacity):
    plan=RenderPlan((tmp_path/'never-opened.mp4',),(RenderSegment(0,0.,1.,0.),),0,
        '16:9',tmp_path/'never-rendered.mp4')
    owner=FirstResultController(ReviewFirstRuntime(enabled=True),sources=plan.sources,
        source_durations=(1.,),master_start=0.,duration=1.)
    owner.install_baseline(plan,plan=plan,authority='numeric reservation test only')
    allocations={p:PhaseAllocation(dict(journal_bytes=1024*1024,ipc_bytes=1024*1024,
        native_completion_passes=int(p=='selected_completion'),
        pose_comparisons=64,pose_evidence_bytes=byte_capacity),10.) for p in PHASES}
    lease=owner.reserve_request_work(total_limits={k:sum(a.resources[k] for a in allocations.values())
        for k in RESOURCES},allocations=allocations,journal_path=tmp_path/'numeric-work.jsonl')
    return owner,lease


def inventory(size=22):
    # These are genuinely unobserved physical requests, not fake native facts.
    demands=tuple(NativeMotionDemand(i%2,80+i,3*i,f'family-{i%2}','moving_closeup')
        for i in range(size))
    return NativeMotionInventory(tuple(NativeMotionSupply(d,None) for d in demands),(),False,'CACHE_ONLY')


def next_chunk(frozen,budget,excluded,*,balanced=False,limit=22,raw_limit=22,lanes=None):
    reserve=primitive_reservation_bytes(512*1024)
    costs={s.demand.native_key:NativeMotionCompletionCost(0,0,reserve)
        for s in frozen.supplies if s.demand.native_key not in excluded}
    parent=budget.request_work_phase.remaining_capacity(COMPLETION_RESOURCES)
    remaining={k:min(parent[k],budget.limits[k]-budget.counts[k]-budget.reserved[k])
        for k in COMPLETION_RESOURCES}
    args=dict(maximum_observations=limit-len(excluded),maximum_raw_observations=raw_limit-len(excluded),
        resource_costs=costs,remaining_resources=remaining,excluded_native_keys=tuple(excluded))
    if balanced:return balanced_completion_schedule(frozen,lane_counts=lanes,**args)
    return coherent_completion_schedule(frozen,required_native_keys=(),
        packages=(NativeMotionCompletionPackage('one-frozen-package',tuple(
            s.demand.native_key for s in frozen.supplies)),),**args)


def numerical_input(budget,demand,producer=None):
    return budget.measure_primitive(label='actual_nonhuman_opencv_mean',
        input_binding=dict(source=demand.source_index,native=demand.native_frame_index),
        work=dict(pose_comparisons=1),maximum_result_bytes=512*1024,
        producer=(lambda:cv2.mean(np.full((32,32,3),demand.native_frame_index,np.uint8)))
            if producer is None else producer)


@pytest.mark.parametrize('balanced',(False,True))
def test_22_cold_numeric_inputs_advance_after_actual_settlement_in_one_lease(tmp_path,balanced):
    owner,lease=lease_owner(tmp_path,8*1024*1024)
    def run():
        port=lease.ports['selected_completion'];port.charge('native_completion_passes')
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.)
        frozen=inventory();excluded=[];chunks=[];lanes={}
        while len(excluded)<22:
            scheduled,receipt=next_chunk(frozen,budget,excluded,balanced=balanced,lanes=lanes)
            assert scheduled
            chunks.append(len(scheduled));excluded.extend(d.native_key for d in scheduled)
            for demand in scheduled:numerical_input(budget,demand)
            assert not any(budget.reserved.values())
            assert not receipt['successful_observations_or_connected_actors_certified']
        assert chunks==[7,7,7,1]
        assert len(set(excluded))==22 and budget.counts['pose_comparisons']==22
        assert port.charged['pose_comparisons']==22 and port.charged['native_completion_passes']==1
        assert 0<budget.counts['pose_evidence_bytes']<65536
        assert port.charged['pose_evidence_bytes']==budget.counts['pose_evidence_bytes']
        assert budget.counts['model_calls']==budget.counts['new_decodes']==0
        journal=budget.primitive_journal.to_dict();validate_journal(journal)
        assert len(journal['attempts'])==22 and not journal['quality_approved']
        # Admission/exclusion is not success: no native observations were made.
        _,last=next_chunk(frozen,budget,excluded,balanced=balanced,lanes=lanes)
        assert len(last['unresolved_excluded_native_keys'])==22
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_large_actual_payload_still_charges_bytes_and_stops_without_reset(tmp_path):
    owner,lease=lease_owner(tmp_path,2*1024*1024)
    def run():
        port=lease.ports['selected_completion'];port.charge('native_completion_passes')
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.,maximum_pose_evidence_bytes=2*1024*1024)
        frozen=inventory();excluded=[];rng=np.random.default_rng(9123)
        while len(excluded)<22:
            scheduled,_=next_chunk(frozen,budget,excluded)
            if not scheduled:break
            excluded.extend(d.native_key for d in scheduled)
            for demand in scheduled:
                numerical_input(budget,demand,lambda:rng.integers(0,256,300_000,dtype=np.uint8))
        assert 1<len(excluded)<22
        journal=budget.primitive_journal.to_dict();validate_journal(journal)
        actual=journal['charged_blob_bytes']+journal['charged_record_bytes']
        assert actual==budget.counts['pose_evidence_bytes']==port.charged['pose_evidence_bytes']
        assert actual>1024*1024 and not any(budget.reserved.values())
        assert port.charged['native_completion_passes']==1
        assert budget.counts['pose_comparisons']==len(excluded)
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


@pytest.mark.parametrize('failure',(RuntimeError,InterruptedError))
def test_failed_or_cancelled_numeric_work_keeps_charges_and_unexecuted_demands(tmp_path,failure):
    owner,lease=lease_owner(tmp_path,8*1024*1024)
    def run():
        port=lease.ports['selected_completion'];port.charge('native_completion_passes')
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.)
        frozen=inventory();scheduled,_=next_chunk(frozen,budget,[])
        numerical_input(budget,scheduled[0])
        def fail():raise failure('explicit numerical backend termination')
        with pytest.raises(failure):numerical_input(budget,scheduled[1],fail)
        journal=budget.primitive_journal.to_dict();validate_journal(journal)
        assert len(journal['attempts'])==2 and journal['attempts'][1]['status']=='backend_failed'
        assert len(scheduled[2:])==5 and len(frozen.supplies)-len(scheduled)==15
        assert budget.counts['pose_comparisons']==port.charged['pose_comparisons']==2
        assert budget.counts['pose_evidence_bytes']==port.charged['pose_evidence_bytes']>0
        assert not any(budget.reserved.values()) and port.charged['native_completion_passes']==1
        # The frozen requests stay UNKNOWN. No failed input is relabelled done.
        assert all(s.observation is None and s.state=='UNKNOWN_NOT_OBSERVED' for s in frozen.supplies)
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


@pytest.mark.parametrize('limit,raw_limit',((9,9),(22,9)))
def test_chunk_caps_are_global_not_renewed_after_settlement(tmp_path,limit,raw_limit):
    owner,lease=lease_owner(tmp_path,8*1024*1024)
    def run():
        lease.ports['selected_completion'].charge('native_completion_passes')
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+8.)
        frozen=inventory();excluded=[]
        while len(excluded)<min(limit,raw_limit):
            scheduled,_=next_chunk(frozen,budget,excluded,limit=limit,raw_limit=raw_limit)
            assert scheduled;excluded.extend(d.native_key for d in scheduled)
            for demand in scheduled:numerical_input(budget,demand)
        scheduled,receipt=next_chunk(frozen,budget,excluded,limit=limit,raw_limit=raw_limit)
        assert scheduled==() and len(excluded)==9
        assert receipt['successful_observations_or_connected_actors_certified'] is False
        assert budget.counts['pose_comparisons']==9
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_excluded_keys_cannot_silently_change_the_frozen_inventory():
    frozen=inventory(1)
    with pytest.raises(ValueError,match='frozen request inventory'):
        coherent_completion_schedule(frozen,required_native_keys=(),packages=(),maximum_observations=1,
            maximum_raw_observations=1,resource_costs={},remaining_resources=dict.fromkeys(COMPLETION_RESOURCES,0),
            excluded_native_keys=((99,99,None),))


def test_active_union_finishes_across_chunks_even_if_another_package_newly_fits():
    # Resource vectors only. No fake detector or identity observations.
    frozen=inventory(18);keys=tuple(s.demand.native_key for s in frozen.supplies)
    packages=(NativeMotionCompletionPackage('A',keys[:12]),
        NativeMotionCompletionPackage('B',(*keys[:7],*keys[12:])))
    args=dict(required_native_keys=(),packages=packages,maximum_observations=18,
        maximum_raw_observations=18,remaining_resources=dict(new_decodes=100,model_calls=100,
            pose_evidence_bytes=7))
    first,receipt=coherent_completion_schedule(frozen,
        resource_costs={k:NativeMotionCompletionCost(0,0,1) for k in keys},**args)
    assert tuple(d.native_key for d in first)==keys[:7]
    assert receipt['active_package_identity']=='A'
    # A measured resource vector can change after raw settlement. B now fits
    # this memory chunk in full; A must remain active, WITHOUT reserving its
    # entire remaining union in memory or renewing the global input budget.
    excluded=tuple(d.native_key for d in first)
    costs={k:NativeMotionCompletionCost(0,0,2 if k in keys[7:12] else 1)
        for k in keys if k not in excluded}
    ordinary,changed=coherent_completion_schedule(frozen,resource_costs=costs,
        excluded_native_keys=excluded,**args)
    assert changed['selected_packages'][0]['identity']=='B' and len(ordinary)==6
    continuing,settled=coherent_completion_schedule(frozen,resource_costs=costs,
        excluded_native_keys=excluded,active_package_identity='A',**args)
    assert tuple(d.native_key for d in continuing)==keys[7:10]
    assert settled['active_package_identity']=='A'
    assert settled['admitted_resource_upper_bounds']['pose_evidence_bytes']==6
    assert not settled['selected_packages'][0]['entire_missing_input_package_admitted']


def test_precap_unrequested_dependencies_remain_in_the_original_denominator():
    frozen=inventory(1);key=frozen.supplies[0].demand.native_key;omitted=(1,99,None)
    package=NativeMotionCompletionPackage('original-pair',(key,omitted),(omitted,))
    scheduled,receipt=coherent_completion_schedule(frozen,required_native_keys=(),packages=(package,),
        maximum_observations=1,maximum_raw_observations=1,
        resource_costs={key:NativeMotionCompletionCost(0,0,0)},
        remaining_resources=dict.fromkeys(COMPLETION_RESOURCES,0))
    assert tuple(d.native_key for d in scheduled)==(key,)
    row=receipt['selected_packages'][0]
    assert row['original_required_native_input_count']==2
    assert row['unrequested_native_keys']==row['unadmitted_missing_native_keys']==(omitted,)
    assert not row['entire_missing_input_package_admitted']
    assert not receipt['successful_observations_or_connected_actors_certified']


def test_unrequested_exemption_must_exactly_match_the_original_inventory_difference():
    frozen=inventory(1);key=frozen.supplies[0].demand.native_key;omitted=(1,99,None)
    for package in (NativeMotionCompletionPackage('undeclared',(key,omitted)),
            NativeMotionCompletionPackage('silently-waived',(key,),(key,))):
        with pytest.raises(ValueError,match='frozen request inventory'):
            coherent_completion_schedule(frozen,required_native_keys=(),packages=(package,),
                maximum_observations=1,maximum_raw_observations=1,
                resource_costs={key:NativeMotionCompletionCost(0,0,0)},
                remaining_resources=dict.fromkeys(COMPLETION_RESOURCES,0))
