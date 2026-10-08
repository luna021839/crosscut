"""Real inventory/lease accounting with synthetic sensors, not model or media QA."""
from dataclasses import replace
from types import SimpleNamespace
from time import monotonic
import pytest

from crosscut.group_source_changing_owner import NativeFamilySensors,NativeMotionDemand
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_scene_focal_pose_motion import restore_native_row
from crosscut.first_result_controller import FirstResultController
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.request_work_lease import PHASES,RESOURCES,PhaseAllocation
from crosscut.render import RenderPlan,RenderSegment
from engine.tests.test_group_review_recorded_index import native_record


class SyntheticSensors(NativeFamilySensors):
    def __init__(self,budget):
        self.session=SimpleNamespace(budget=budget)
        self.raw_rows={};self.appearance_rows={};self.raw_calls=[];self.appearance_calls=[]
        self.interrupt=False

    def native_at(self,source,native,roll=None,*,cached_only=False):
        key=source,native,roll
        if key not in self.raw_rows and not cached_only:
            self.session.budget.spend('new_decodes')
            self.session.budget.spend('model_calls',14,model='synthetic_raw_observer')
            self.raw_calls.append(key)
            raw,_=native_record()
            self.raw_rows[key]=replace(restore_native_row(raw),request_source_index=source,native_frame_index=native)
        return self.raw_rows.get(key)

    def appearance_at(self,raw,*,cached_only=False):
        key=raw.request_source_index,raw.native_frame_index
        if key not in self.appearance_rows and not cached_only:
            self.session.budget.spend('model_calls',len(raw.heads),model='synthetic_appearance')
            self.appearance_calls.append(key)
            if self.interrupt:raise InterruptedError('synthetic interruption after charged inference')
            self.appearance_rows[key]=dict(synthetic=True,heads=[dict(observation_id=h.observation_id,embedding=None) for h in raw.heads])
        return self.appearance_rows.get(key)


def controller(tmp_path,model_calls=100):
    plan=RenderPlan((tmp_path/'not-read.mp4',),(RenderSegment(0,0.,1.,0.),),0,'16:9',tmp_path/'not-rendered.mp4')
    owner=FirstResultController(ReviewFirstRuntime(enabled=True),sources=plan.sources,
        source_durations=(2.,),master_start=0.,duration=1.)
    owner.install_baseline(plan,plan=plan,authority='synthetic inventory accounting only')
    allocations={p:PhaseAllocation(dict(journal_bytes=1024*1024,
        native_completion_passes=int(p=='selected_completion'),model_calls=model_calls,
        new_decodes=20,pose_evidence_bytes=8*1024*1024),10.) for p in PHASES}
    lease=owner.reserve_request_work(total_limits={k:sum(a.resources[k] for a in allocations.values())
        for k in RESOURCES},allocations=allocations,journal_path=tmp_path/'work.jsonl')
    return owner,lease


def sensors(model_calls=100):
    return SyntheticSensors(JointGroupWorkBudget(absolute_deadline=monotonic()+8.,maximum_model_calls=model_calls))


def test_appearance_only_missing_is_completed_without_repeating_raw_inference(tmp_path):
    owner,lease=controller(tmp_path,1)
    def run():
        s=sensors(1);raw,_=native_record();s.raw_rows[2,80,None]=restore_native_row(raw)
        demands=(NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True),
            NativeMotionDemand(2,80,20,'family','other',require_appearance=True))
        inventory=s.motion_inventory(demands)
        assert all(r.missing_modalities==('APPEARANCE',) for r in inventory.supplies)
        done=s.complete_motion_inventory_once(inventory,maximum_new_observations=2)
        assert s.raw_calls==[] and s.appearance_calls==[(2,80)]
        assert done.attempted_native_keys==((2,80,None),)
        assert all(not r.missing_modalities and r.observation.heads for r in done.supplies)
        with pytest.raises(ValueError,match='already consumed'):
            s.complete_motion_inventory_once(inventory,maximum_new_observations=2)
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()
    assert lease.ports['selected_completion'].charged['native_completion_passes']==1
    assert lease.ports['selected_completion'].charged['model_calls']==1


def test_raw_fourteen_calls_are_not_misrepresented_as_actor_ready_cost(tmp_path):
    owner,lease=controller(tmp_path,14)
    def run():
        s=sensors(14)
        demands=(NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True),)
        result=s.complete_motion_inventory_once(s.motion_inventory(demands),maximum_new_observations=1)
        assert not s.raw_calls and not s.appearance_calls
        assert result.status=='BUDGET_PARTIAL' and result.supplies[0].missing_modalities==('RAW','APPEARANCE')
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_raw_alias_still_completes_appearance_requested_by_another_output_use(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s=sensors()
        demands=(NativeMotionDemand(2,80,8,'family','other'),
            NativeMotionDemand(2,80,12,'family','moving_closeup',require_appearance=True))
        result=s.complete_motion_inventory_once(s.motion_inventory(demands),maximum_new_observations=1)
        assert len(s.raw_calls)==len(s.appearance_calls)==1
        assert all(not r.missing_modalities for r in result.supplies)
        assert result.supplies[1].appearance['heads'][0]['embedding'] is None
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_cancelled_appearance_is_partial_and_cannot_restart_under_a_new_session(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s=sensors();s.interrupt=True
        demands=(NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True),)
        with pytest.raises(InterruptedError):s.complete_motion_inventory_once(s.motion_inventory(demands),maximum_new_observations=1)
        assert s.motion_completion_result.status=='INTERRUPTED_PARTIAL'
        replacement=sensors()
        from crosscut.request_work_lease import RequestWorkExhausted
        with pytest.raises(RequestWorkExhausted):
            replacement.complete_motion_inventory_once(replacement.motion_inventory(demands),maximum_new_observations=1)
    try:
        from crosscut.request_work_lease import RequestWorkExhausted
        with pytest.raises(RequestWorkExhausted):owner.run_work_phase('selected_completion',run)
    finally:lease.close()
