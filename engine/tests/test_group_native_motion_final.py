"""Real codec/clock transport; test patterns never establish editing quality."""
from copy import deepcopy
from hashlib import sha256
import shutil
import subprocess
from time import monotonic

import pytest

from crosscut.group_joint_native_preparation import _plan_identity
from crosscut.group_joint_observations import GroupJointObservationSession, _file_identity
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_native_motion_final import CONTRACT, _final_inputs, native_qa_metrics_valid
from crosscut.group_performer_family_owner import (
    _final_live_replay, digest, plan_payload, replay_final_sources, validate_serialized_metadata, qa_metrics_valid,
)
from crosscut.group_scene_camera_queries import PreparedGroupCameraQueries
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.render_frame_schedule import probe_render_frame_clock


@pytest.fixture
def real_codec_receipt(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("real ffmpeg codec fixture unavailable")
    sources = tuple(tmp_path/f"real-pattern-{i}.mp4" for i in range(4))
    for path in sources:
        subprocess.run([ffmpeg,"-nostdin","-v","error","-f","lavfi","-i",
            "testsrc2=size=320x180:rate=30:duration=1", "-c:v","libx264",
            "-preset","ultrafast","-pix_fmt","yuv420p",str(path)],check=True,timeout=15)
    clocks = tuple(probe_render_frame_clock(sources[i]) for i in (0,2,3))
    assert all(c is not None for c in clocks)
    domain = GroupSourceDomain(sources,(0,2,3),(0.,)*3,(None,)*3,((),)*3,clocks)
    plan = RenderPlan(sources,tuple(RenderSegment((0,2,3)[i%3],i/10,.1,i/10) for i in range(9)),
        0,"16:9",tmp_path/"not-rendered.mp4",width=320,height=180,fps=30,
        transitions=(RenderTransition(),)*8)
    prepared = PreparedGroupCameraQueries(plan,domain)
    session = GroupJointObservationSession(domain=domain,
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.,maximum_new_decodes=9))
    samples = []
    try:
        for tick in range(1,27,3):
            actual = prepared.samples[tick]
            shot = next(s for s in prepared.shots if s.first_output_frame <= tick < s.end_output_frame)
            native = session.native_pixels(actual.source_index,actual.native_frame_index)
            projected = session.projected_pixels(prepared.camera_for(shot),tick)
            samples.append(dict(output_tick=tick,request_source_index=actual.source_index,
                native_frame_index=actual.native_frame_index,
                native_pixel_sha256=sha256(memoryview(native)).hexdigest(),
                projected_pixel_sha256=sha256(memoryview(projected)).hexdigest()))
    finally:
        session.close()
    receipt = dict(contract=CONTRACT,plan_sha256=digest(plan_payload(plan)),
        metadata_sha256=digest(dict(transport_test_patterns_only=True)),source_domain=domain.to_dict(),
        quality_approved=False,raw_parent_admitted=False,body_phase_certified=False,
        cross_camera_identity_certified=False,decoded_output_quality_certified=False,
        source_camera_refresh=dict(contract=CONTRACT,timeline_evidence=dict(
            selected_plan_identity=_plan_identity(plan),samples=samples,
            live_source_binding=dict(contract="native_metadata_live_sources_v1",
                source_domain=domain.to_dict(), source_identities=[list(_file_identity(sources[i])) for i in (0,2,3)]))))
    return plan,receipt


def test_native_receipt_uses_actual_clock_pixel_consumer_not_model_receipt(real_codec_receipt):
    plan,receipt = real_codec_receipt
    # Reproduce the old wrong-consumer failure before any model could run.
    with pytest.raises(KeyError,match="model_path"):
        _final_live_replay.__wrapped__(plan.to_json(),receipt,monotonic()+30.)
    actual = replay_final_sources(plan.to_json(),receipt)
    assert actual["binding_valid"] and actual["source_live_replay"]
    assert actual["sampled_native_pixel_count"] == 9
    assert actual["quality_approved"] is False and actual["body_phase_certified"] is False
    assert actual["cross_camera_identity_certified"] is False
    assert actual["raw_parent_admitted"] is False
    assert native_qa_metrics_valid(actual) and qa_metrics_valid(actual)
    for change in (dict(quality_approved=True),dict(body_phase_certified=True),
            dict(sampled_native_pixel_count=0),dict(final_quality_status='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE')):
        changed=dict(actual,**change);changed.pop('evidence_sha256')
        changed['evidence_sha256']=digest(changed)
        assert not native_qa_metrics_valid(changed) and not qa_metrics_valid(changed)


def test_final_replay_rejects_changed_pixel_parent(real_codec_receipt):
    plan,receipt = real_codec_receipt
    receipt["source_camera_refresh"]["timeline_evidence"]["samples"][0]["projected_pixel_sha256"] = "e"*64
    # The real isolated-worker transport preserves the reason in RuntimeError.
    with pytest.raises(RuntimeError,match="pixel parent changed"):
        replay_final_sources(plan.to_json(),receipt)


def test_missing_live_binding_not_hidden_as_a_model_path_error(real_codec_receipt):
    plan,receipt = real_codec_receipt
    del receipt["source_camera_refresh"]["timeline_evidence"]["live_source_binding"]
    with pytest.raises(ValueError,match="live source/plan preimage"):
        _final_inputs(plan.to_json(),receipt)


def test_changed_mapping_or_duplicate_tick_rejected_before_decode(real_codec_receipt):
    plan,receipt = real_codec_receipt
    changed = deepcopy(receipt)
    changed["source_domain"]["eligible_request_indices"] = [0,1,3]
    with pytest.raises(ValueError):
        _final_inputs(plan.to_json(),changed)
    samples = receipt["source_camera_refresh"]["timeline_evidence"]["samples"]
    samples[1]["output_tick"] = samples[0]["output_tick"]
    with pytest.raises(ValueError,match="sample address"):
        _final_inputs(plan.to_json(),receipt)


def test_native_complete_metadata_serializes_without_backlink_hash_confusion():
    from tests.test_group_native_motion_metadata import numeric_complete
    from crosscut.group_native_motion_metadata import complete_native_motion_metadata
    context,selected,metadata,domain = numeric_complete(phases=(0,0))
    bundle = complete_native_motion_metadata(context=context,selected=selected,metadata=metadata,domain=domain)
    payload = bundle.to_dict()
    receipt = validate_serialized_metadata(payload)
    assert receipt["contract"] == CONTRACT and receipt["quality_approved"] is False
    payload["face_match"]["target_face_height"] = .8
    with pytest.raises(ValueError,match="serialized family"):
        validate_serialized_metadata(payload)
