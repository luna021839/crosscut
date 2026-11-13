"""External-renderer record binding and key cost; NO human media/models."""
from dataclasses import replace
from fractions import Fraction
import pickle
from time import monotonic
from unittest.mock import patch

import numpy as np
import pytest

from crosscut.group_actual_shots import visible_group_cut_inventory,visible_group_shots
from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_scene_camera_queries import PreparedGroupCameraQueries
from crosscut.render import EditorialTransformKeyframe,EditorialTransformTrack
from crosscut import group_scene_camera_queries as queries
from tests.test_scene_role_epoch_program import fixture,compiled


def world():
    context,domain,_=fixture(native_fps=Fraction(30),cut=59/30.)
    plan=context.incumbent.render_plan
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
        EditorialTransformKeyframe(1.,1.4,.54,.48,.8)),sampling_canvas_zoom=1.5)
    context.incumbent.render_plan=replace(plan,width=160,height=90,
        segments=(plan.segments[0],replace(plan.segments[1],editorial_transform=track),*plan.segments[2:]))
    shot=next(s for s in visible_group_shots(context.incumbent.render_plan,domain) if s.first_output_frame==59)
    return domain,compiled(context,domain,shot)


@pytest.mark.parametrize('change',['phase','counts','native','fps','canvas','segments'])
def test_forged_same_renderer_camera_is_rejected_before_projection_or_sensor(change):
    domain,result=world();camera=result.camera
    if change=='phase':
        bad=replace(camera,canonical_segment_first_output_frames=(29,60),
            canonical_segment_output_frame_counts=(31,30))
    elif change=='counts':
        bad=replace(camera,canonical_segment_output_frame_counts=(30,31))
    elif change=='native':bad=replace(camera,native_frame_indices=tuple(n+1 for n in camera.native_frame_indices))
    elif change=='fps':bad=replace(camera,output_fps=29.97)
    elif change=='canvas':bad=replace(camera,output_size=(162,90))
    else:bad=replace(camera,segments=tuple(replace(s,editorial_transform=None) for s in camera.segments))
    assert bad.full_renderer_identity==camera.full_renderer_identity
    query=replace(result.views[0].state,camera=bad)
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(
            absolute_deadline=monotonic()+10.)) as session:
        with patch.object(session,'_assert_source_identity') as identity, \
                patch.object(session,'native_pixels') as native, \
                pytest.raises(ValueError,match='registered'):
            session.projected_pixels(query,59)
        identity.assert_not_called();native.assert_not_called()


def test_registered_camera_and_one_frame_query_share_real_projection_cache_after_ipc_roundtrip():
    domain,result=world()
    from crosscut.optional_native_worker import _encode_message
    camera,query=pickle.loads(_encode_message((result.camera,result.views[0].state)))
    assert camera is query.camera
    assert camera.bind_source(domain) is camera
    assert camera.sample_at(59)[1]==pytest.approx(29/30.)
    frame=np.arange(160*90*3,dtype=np.uint32).reshape(90,160,3).astype(np.uint8)
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(
            absolute_deadline=monotonic()+10.)) as session:
        with patch.object(session,'_assert_source_identity',return_value=(0,0,0,0,0)), \
                patch.object(session,'native_pixels',return_value=frame) as native:
            pixels=session.projected_pixels(query,59)
            assert pixels.shape==(90,160,3)
            assert session.projected_pixels(camera,59) is pixels
            assert session.projected_pixels(query,59) is pixels
            assert native.call_count==1


@pytest.mark.parametrize('scale',[1,2,4])
def test_per_tick_camera_keys_do_not_reserialize_full_preimage(scale):
    context,domain,_=fixture(native_fps=Fraction(30))
    plan=context.incumbent.render_plan
    # One longer camera and multiple independent cameras, same bounded domain.
    plan=replace(plan,segments=tuple(replace(s,duration=scale,source_start=i*scale,
        master_start=i*scale) for i,s in enumerate(plan.segments)))
    calls=[]
    original=queries._digest
    def measured(value):
        if isinstance(value,dict) and value.get('contract')=='physical_camera_full_renderer_observation_v1':
            calls.append(len(value['native_frame_indices']))
        return original(value)
    with patch.object(queries,'_digest',side_effect=measured):
        prepared=PreparedGroupCameraQueries(plan,domain)
        cameras=[prepared.camera_for(s) for s in prepared.shots]
        assert sum(calls)==plan.output_frame_count
        counts=list(calls)
        # Repeat cold/warm camera lookup and every camera key many times. No
        # full-native preimage hashing is allowed after registry construction.
        with patch.object(queries,'_plan_identity',side_effect=AssertionError('full plan rehash')), \
                patch.object(queries,'_json',side_effect=AssertionError('full preimage serialization')):
            for shot,camera in zip(prepared.shots,cameras,strict=True):
                assert prepared.camera_for(shot) is camera
                for tick in range(shot.first_output_frame,shot.end_output_frame):
                    assert camera.key==camera.key
                    camera.sample_at(tick)
                    camera.bind_source(domain)
        assert calls==counts


@pytest.mark.parametrize('native_cut',[None,2.])
def test_continuous_piecewise_rate_is_not_a_source_seek_and_preserves_native_cut(native_cut):
    context,domain,_=fixture(native_fps=Fraction(60000,1001))
    plan=context.incumbent.render_plan
    a=replace(plan.segments[1],source_duration=1.005)
    b=replace(plan.segments[2],source_index=a.source_index,source_start=a.source_end,source_duration=.995)
    plan=replace(plan,segments=(plan.segments[0],a,b,plan.segments[3]))
    domain=replace(domain,scene_cut_times=((),() if native_cut is None else (native_cut,),(),()))
    events=visible_group_cut_inventory(plan,domain)
    assert not any(e.kind=='same_source_clock_jump' for e in events)
    native=[e for e in events if e.kind=='native_camera_cut']
    if native_cut is None:assert native==[]
    else:
        assert [(e.first_incoming_output_frame,e.outgoing_native_frame,e.incoming_native_frame,e.native_cut_frame)
            for e in native]==[(60,118,120,120)]
    prepared=PreparedGroupCameraQueries(plan,domain)
    for shot in prepared.shots:
        camera=prepared.camera_for(shot)
        for tick in range(shot.first_output_frame,shot.end_output_frame):camera.sample_at(tick)
    # Actual seeks are still separate events even if a native cut lies in the
    # omitted interval. No native exception or diversity is manufactured.
    sought=replace(plan,segments=(plan.segments[0],a,replace(b,source_start=b.source_start+.05),plan.segments[3]))
    seam=[e for e in visible_group_cut_inventory(sought,domain) if e.first_incoming_output_frame==60]
    assert [e.kind for e in seam]==['same_source_clock_jump']
