"""Real inventory/lease accounting with synthetic sensors, not model or media QA."""
from dataclasses import replace
from types import SimpleNamespace
from time import monotonic
import pytest

from crosscut.group_source_changing_owner import NativeFamilySensors,NativeMotionDemand
from crosscut.group_native_motion_completion_schedule import NativeMotionCompletionPackage
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_scene_focal_pose_motion import restore_native_row
from crosscut.first_result_controller import FirstResultController
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.request_work_lease import PHASES,RESOURCES,PhaseAllocation
from crosscut.render import RenderPlan,RenderSegment
from engine.tests.test_group_review_recorded_index import native_record
from crosscut.group_review_recorded_index import RecordedGroupReviewIndex
from crosscut.group_joint_observations import _file_identity
from crosscut.group_joint_person_ownership import ownership_policy_fingerprint
from crosscut.group_performer_family_owner import digest
from crosscut.face_identity import SFACE_SHA256
from crosscut.native_face_descriptor import SFACE_MODEL_NAME
from crosscut.group_native_head_orientation import native_head_analysis_spec
import hashlib
import json
import numpy as np


class SyntheticSensors(NativeFamilySensors):
    def __init__(self,budget):
        self.session=SimpleNamespace(budget=budget)
        self.raw_rows={};self.appearance_rows={};self.raw_calls=[];self.appearance_calls=[]
        self.interrupt=False
        self.member_gallery=None;self.member_feature_rows={}
        self.recorded_member_feature_cache=None

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
        new_decodes=20,pose_evidence_bytes=8*1024*1024,identity_reads=128,
        model_hash_bytes=1024*1024),10.) for p in PHASES}
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


def test_coherent_completion_services_fixed_return_before_cheaper_scattered_donors(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s=sensors()
        demands=tuple(NativeMotionDemand(source,native,tick,f'family-{source}','moving_closeup')
            for source,native,tick in ((2,80,8),(2,81,9),(3,90,10),(3,91,11),(5,120,12)))
        packages=(NativeMotionCompletionPackage('pair-2',tuple(d.native_key for d in demands[:2])),
            NativeMotionCompletionPackage('pair-3',tuple(d.native_key for d in demands[2:4])))
        done=s.complete_motion_inventory_once(s.motion_inventory(demands),maximum_new_observations=3,
            required_native_keys=(demands[-1].native_key,),completion_packages=packages)
        assert done.attempted_native_keys==((5,120,None),(2,80,None),(2,81,None))
        assert s.raw_calls==list(done.attempted_native_keys)
        receipt=json.loads(done.completion_schedule_json)
        assert receipt['selected_packages'][0]['entire_missing_input_package_admitted'] is True
        assert receipt['unadmitted_required_native_keys']==[]
        assert not receipt['successful_observations_or_connected_actors_certified']
        assert not receipt['quality_approved']
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_coherent_completion_uses_one_partial_package_instead_of_scattering_remaining_budget(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s=sensors()
        demands=tuple(NativeMotionDemand(source,native,tick,f'family-{source}','moving_closeup')
            for source,native,tick in ((2,80,8),(2,81,9),(3,90,10),(3,91,11)))
        packages=(NativeMotionCompletionPackage('pair-2',tuple(d.native_key for d in demands[:2])),
            NativeMotionCompletionPackage('pair-3',tuple(d.native_key for d in demands[2:])))
        done=s.complete_motion_inventory_once(s.motion_inventory(demands),maximum_new_observations=1,
            completion_packages=packages)
        assert done.attempted_native_keys==((2,80,None),)
        receipt=json.loads(done.completion_schedule_json)
        assert len(receipt['selected_packages'])==1
        assert receipt['selected_packages'][0]['unadmitted_missing_native_keys']==[[2,81,None]]
        assert not receipt['selected_packages'][0]['entire_missing_input_package_admitted']
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_invalid_completion_obligation_is_rejected_before_consuming_the_single_pass(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s=sensors();d=NativeMotionDemand(2,80,8,'family','other')
        inventory=s.motion_inventory((d,))
        with pytest.raises(ValueError,match='frozen request inventory'):
            s.complete_motion_inventory_once(inventory,maximum_new_observations=1,
                required_native_keys=((2,999,None),))
        assert not getattr(s.session,'_native_motion_completion_consumed',False)
        assert not s.raw_calls and not s.appearance_calls
        done=s.complete_motion_inventory_once(inventory,maximum_new_observations=1,
            required_native_keys=(d.native_key,))
        assert done.attempted_native_keys==(d.native_key,)
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_alias_census_reads_each_physical_fact_once_with_union_modalities(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        class CountingSensors(SyntheticSensors):
            def __init__(self,budget):
                super().__init__(budget);self.cache_reads=[];self.archive_reads=[]
            def native_at(self,source,native,roll=None,*,cached_only=False):
                if cached_only:self.cache_reads.append((source,native,roll))
                return super().native_at(source,native,roll,cached_only=cached_only)
            def _recorded_motion_supply(self,demand):
                from crosscut.group_source_changing_owner import NativeMotionSupply
                self.archive_reads.append(demand)
                return NativeMotionSupply(demand,None)
        s=CountingSensors(sensors().session.budget)
        raw=NativeMotionDemand(2,80,8,'family','other')
        app=replace(raw,output_tick=12,opportunity='moving_closeup',require_appearance=True)
        member=replace(raw,output_tick=14,require_member_features=True)
        s.member_gallery=object()  # No rows exist, so no gallery/model method is invoked.
        inventory=s.motion_inventory((raw,app,member)*100)
        assert s.cache_reads==[(2,80,None)]
        assert len(s.archive_reads)==1
        assert s.archive_reads[0].require_appearance and s.archive_reads[0].require_member_features
        assert len(inventory.supplies)==300
        assert inventory.supplies[0].missing_modalities==('RAW',)
        assert inventory.supplies[1].missing_modalities==('RAW','APPEARANCE')
        assert inventory.supplies[2].missing_modalities==('RAW','MEMBER_FEATURES')
        assert not s.raw_calls and not s.appearance_calls
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


def recorded_sensors(tmp_path,*,remount=False,appearance=True,appearance_adapter='f'*64):
    """Real archive/file binding with explicitly synthetic model/pixel methods."""
    path=tmp_path/'source.fixture';path.write_bytes(b'independently retained source bytes')
    identity=_file_identity(path)
    recorded=(identity[0]+1,*identity[1:]) if remount else identity
    clock=SimpleNamespace(to_dict=lambda:dict(test_clock=True),frame_count=100)
    raw,app=native_record()
    model_names=('nanodet_people','rtmpose_body7',native_head_analysis_spec(None)[0])
    pixels=np.zeros((18,32,3),dtype=np.uint8)
    raw=json.loads(json.dumps(replace(restore_native_row(raw),source_file_identity=recorded,
        model_content_sha256=tuple((n,'b'*64) for n in model_names),
        model_adapter_fingerprints=tuple((n,'d'*64) for n in model_names),
        source_clock_identity=digest(clock.to_dict()),person_ownership_policy_sha256=ownership_policy_fingerprint(),
        native_pixel_sha256=hashlib.sha256(memoryview(pixels)).hexdigest()).to_dict()))
    app.update({k:raw[k] for k in ('source_file_identity','source_clock_identity','native_pixel_sha256')})
    app['recorded_native_observation_sha256']=digest(raw)
    index=RecordedGroupReviewIndex();index.add_native_record(raw,app if appearance else None,provenance='explicit fixture manifest')
    s=sensors();s.session.closed=False
    s.session.domain=SimpleNamespace(eligible_indices=(2,),request_sources=(path,path,path),
        clocks_by_request={2:clock})
    s.session._assert_source_identity=lambda src:_file_identity(path)
    s.session._assert_model_assets=lambda name:None
    s.session.models={n:(None,None,sha) for n,sha in raw['model_content_sha256']}
    s.session.models[SFACE_MODEL_NAME]=(None,None,SFACE_SHA256)
    s.session.model_adapters=dict(raw['model_adapter_fingerprints'])
    s.session.model_adapters[SFACE_MODEL_NAME]=appearance_adapter
    def native_pixels(source,native):
        s.session.budget.spend('new_decodes');return pixels
    def observe_native(source,native,*,model,arguments):
        s.session.budget.spend('model_calls',model=model)
        s.appearance_calls.append((source,native))
        return [1.]+[0.]*127
    s.session.native_pixels=native_pixels;s.session.observe_native=observe_native
    s.bind_recorded_motion_index(index,source_identities={2:recorded},
        source_content_sha256={2:hashlib.sha256(path.read_bytes()).hexdigest()})
    return s,index,raw,app,path


def test_persisted_native_and_appearance_service_new_request_with_no_reinference(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s,index,raw,app,path=recorded_sensors(tmp_path,remount=True)
        demands=(NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True),
            NativeMotionDemand(2,80,50,'family','other',require_appearance=True))
        inventory=s.motion_inventory(demands)
        assert all(v.observation.to_dict()==restore_native_row(raw).to_dict() for v in inventory.supplies)
        assert all(v.appearance==app and not v.missing_modalities for v in inventory.supplies)
        assert all(json.loads(v.source_equivalence_json)['state']=='CONTENT_VERIFIED_DEVICE_REMOUNT' for v in inventory.supplies)
        done=s.complete_motion_inventory_once(inventory,maximum_new_observations=10)
        assert not done.attempted_native_keys and not s.raw_calls and not s.appearance_calls
        assert not s.raw_rows  # Archived parents never silently become live detector rows.
        assert s.session.budget.counts['new_decodes']==s.session.budget.counts['model_calls']==0
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_post_remount_completion_cache_services_absent_addresses_without_rewriting_old_parents(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        old,index,raw,app,path=recorded_sensors(tmp_path,remount=True)
        live=replace(restore_native_row(raw),native_frame_index=90,source_file_identity=_file_identity(path))
        live_raw=json.loads(json.dumps(live.to_dict()))
        live_app=json.loads(json.dumps(app))
        live_app.update({k:live_raw[k] for k in ('native_frame_index','source_file_identity')},
            recorded_native_observation_sha256=digest(live_raw))
        index.add_native_record(live_raw,live_app,provenance='explicit one-pass completion fixture')
        fresh=sensors();fresh.session=old.session
        fresh.bind_recorded_motion_index(index,source_identities={2:raw['source_file_identity']},
            source_content_sha256={2:hashlib.sha256(path.read_bytes()).hexdigest()},
            additional_source_identities={2:_file_identity(path)})
        demands=tuple(NativeMotionDemand(2,native,tick,'family','moving_closeup',require_appearance=True)
            for native,tick in ((80,8),(90,9)))
        done=fresh.complete_motion_inventory_once(fresh.motion_inventory(demands),maximum_new_observations=2)
        assert done.supplies[0].observation.source_file_identity==tuple(raw['source_file_identity'])
        assert done.supplies[1].observation.source_file_identity==_file_identity(path)
        assert json.loads(done.supplies[1].source_equivalence_json)['state']=='EXACT_FILE_IDENTITY'
        assert all(not r.missing_modalities for r in done.supplies)
        assert not fresh.raw_calls and not fresh.appearance_calls and not done.attempted_native_keys
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_additional_manifest_cannot_replace_a_conflicting_existing_native_result(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        old,index,raw,app,path=recorded_sensors(tmp_path,remount=True)
        altered=replace(restore_native_row(raw),reason='different_measured_result')
        index.add_native_record(json.loads(json.dumps(altered.to_dict())),
            provenance='explicit conflicting fixture')
        live=replace(restore_native_row(raw),source_file_identity=_file_identity(path))
        index.add_native_record(json.loads(json.dumps(live.to_dict())),provenance='additional live fixture')
        fresh=sensors();fresh.session=old.session
        fresh.bind_recorded_motion_index(index,source_identities={2:raw['source_file_identity']},
            source_content_sha256={2:hashlib.sha256(path.read_bytes()).hexdigest()},
            additional_source_identities={2:_file_identity(path)})
        d=NativeMotionDemand(2,80,8,'family','other',require_appearance=True)
        value=fresh.motion_inventory((d,)).supplies[0]
        assert value.state=='CONFLICTING_NATIVE_RECORDS' and value.observation is None
        assert not value.missing_modalities
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_recorded_appearance_only_remount_completion_keeps_exact_old_raw_parent(tmp_path):
    owner,lease=controller(tmp_path)
    def run():
        s,index,raw,app,path=recorded_sensors(tmp_path,remount=True,appearance=False)
        d=NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True)
        before=s.motion_inventory((d,));assert before.supplies[0].missing_modalities==('APPEARANCE',)
        done=s.complete_motion_inventory_once(before,maximum_new_observations=1)
        value=done.supplies[0]
        assert not s.raw_calls and s.appearance_calls==[(2,80)]
        assert value.appearance['recorded_native_observation_sha256']==digest(raw)
        assert value.appearance['source_file_identity']==raw['source_file_identity']
        assert value.observation.source_file_identity!=_file_identity(path)
        assert not value.appearance['archived_observation_rewritten']
        assert not value.missing_modalities and s.session.budget.counts['model_calls']==1
        assert s.motion_inventory((d,)).supplies[0].appearance==value.appearance
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()


def test_member_only_union_completion_does_not_repeat_raw_or_sface(tmp_path):
    from .test_ive_local_arcface_gallery import gallery,vector
    from crosscut.native_face_descriptor import ARCFACE_MODEL_NAME,ARCFACE_R50_SHA256
    owner,lease=controller(tmp_path)
    def run():
        s,index,raw,app,path=recorded_sensors(tmp_path,remount=True)
        s.member_gallery=gallery()
        s.session.models[ARCFACE_MODEL_NAME]=(None,None,ARCFACE_R50_SHA256)
        s.session.model_adapters[ARCFACE_MODEL_NAME]='b'*64
        called=[]
        def observe_member(source,native,*,model,arguments):
            assert model==ARCFACE_MODEL_NAME
            s.session.budget.spend('model_calls',model=model);called.append((source,native))
            return vector(1)
        s.session.observe_native=observe_member
        d=NativeMotionDemand(2,80,8,'family','moving_closeup',require_appearance=True,
            require_member_features=True)
        missing=NativeMotionDemand(2,90,9,'family','other',require_appearance=True)
        before=s.motion_inventory((d,missing))
        assert before.supplies[0].missing_modalities==('MEMBER_FEATURES',)
        done=s.complete_motion_inventory_once(before,maximum_new_observations=1,
            maximum_new_raw_observations=0)
        assert done.status=='BUDGET_PARTIAL' and done.attempted_native_keys==((2,80,None),)
        assert done.supplies[0].appearance==app and not done.supplies[0].missing_modalities
        assert done.supplies[1].missing_modalities==('RAW','APPEARANCE')
        assert not s.raw_calls and not s.appearance_calls and called==[(2,80)]
        features=done.supplies[0].member_features
        assert features['recorded_native_observation_sha256']==digest(raw)
        assert features['source_file_identity']==raw['source_file_identity']
        assert features['heads'][0]['member_hint']['member']=='가을'
        assert s.session.budget.counts['model_calls']==1
        assert s.motion_inventory((d,)).supplies[0].member_features==features
    try:owner.run_work_phase('selected_completion',run)
    finally:lease.close()
    assert lease.ports['selected_completion'].charged['native_completion_passes']==1


def test_recorded_descriptor_space_change_keeps_heads_but_requires_new_appearance(tmp_path):
    s,index,raw,app,path=recorded_sensors(tmp_path,appearance_adapter='e'*64)
    value=s.motion_inventory((NativeMotionDemand(2,80,8,'family','other',require_appearance=True),)).supplies[0]
    assert value.observation.heads and value.appearance is None
    assert value.missing_modalities==('APPEARANCE',) and value.recorded_input_state=='APPEARANCE_SPACE_MISMATCH'


def test_recorded_wrong_analysis_variant_or_changed_source_is_not_reused(tmp_path):
    s,index,raw,app,path=recorded_sensors(tmp_path)
    d=NativeMotionDemand(2,80,8,'family','other',require_appearance=True)
    s.session.model_adapters[native_head_analysis_spec(None)[0]]='0'*64
    result=s.motion_inventory((d,)).supplies[0]
    assert result.observation is None and result.recorded_input_state=='MISSING_DECLARED_ANALYSIS_VARIANT'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed after'):s.motion_inventory((d,))


def test_recorded_conflicting_same_variant_is_unresolved_not_a_retry_frontier(tmp_path):
    s,index,raw,app,path=recorded_sensors(tmp_path)
    rival=json.loads(json.dumps(replace(restore_native_row(raw),reason='conflicting detector result').to_dict()))
    index.add_native_record(rival,provenance='conflicting same-variant fixture')
    value=s.motion_inventory((NativeMotionDemand(2,80,8,'family','other',require_appearance=True),)).supplies[0]
    assert value.state=='CONFLICTING_NATIVE_RECORDS' and not value.missing_modalities
    assert value.observation is None and not s.raw_calls
