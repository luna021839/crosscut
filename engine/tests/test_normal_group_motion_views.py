"""Real projection/association over synthetic native observations; no media."""
from dataclasses import replace
from fractions import Fraction
import json
from time import monotonic
from types import SimpleNamespace

import pytest

from crosscut.group_source_changing_owner import (NativeMotionDemand,NativeMotionSupply,
    NativeMotionInventory,NativeMotionViews)
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.group_scene_focal_pose_motion import restore_native_row
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_performer_family_owner import digest
from crosscut.native_face_descriptor import NativeFaceDescriptorSpace,SFACE_MODEL_NAME
from crosscut.face_identity import SFACE_SHA256
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
from tests.test_scene_role_epoch_program import fixture
from tests.test_group_review_recorded_index import native_record


def world(*,appearance=True,camera=False,missing_tick=None):
    context,domain,_=fixture(native_fps=Fraction(30))
    context.incumbent.face_match_plan=None
    if camera:
        plan=context.incumbent.render_plan
        segments=list(plan.segments)
        segments[1]=replace(segments[1],editorial_transform=EditorialTransformTrack((
            EditorialTransformKeyframe(0.,1.2,.48,.5),EditorialTransformKeyframe(1.,1.4,.52,.5))))
        context.incumbent.render_plan=replace(plan,segments=tuple(segments))
    original=freeze_owned_scene_program(context,domain)
    compiled=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
        candidate_plan=context.incumbent.render_plan,output_intervals=((30,36),))
    queries={t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    session=SimpleNamespace(domain=domain,budget=budget,closed=False,
        models={SFACE_MODEL_NAME:(None,None,SFACE_SHA256)},model_adapters={SFACE_MODEL_NAME:'f'*64},
        _assert_model_assets=lambda name:None,
        _assert_source_identity=lambda source:restore_native_row(native_record()[0]).source_file_identity)
    sensors=SimpleNamespace(session=session,recorded_motion_bindings={})
    supplies=[]
    for tick,q in queries.items():
        raw,app=native_record();source=domain.request_index(q.eligible_source_index);_,_,native=q.sample_at(tick)
        raw=replace(restore_native_row(raw),request_source_index=source,native_frame_index=native,
            source_clock_identity=q.source_clock_identity)
        payload=json.loads(json.dumps(raw.to_dict()))
        app.update({k:payload[k] for k in ('request_source_index','native_frame_index','source_clock_identity',
            'source_file_identity','native_pixel_sha256')})
        app['recorded_native_observation_sha256']=digest(payload)
        app['descriptor_space']=NativeFaceDescriptorSpace(SFACE_MODEL_NAME,128,SFACE_SHA256,'f'*64).to_dict()
        demand=NativeMotionDemand(source,native,tick,'source-'+str(source),'moving_closeup',require_appearance=True)
        supplies.append(NativeMotionSupply(demand,None if tick==missing_tick else raw,
            app if appearance and tick!=missing_tick else None))
    inventory=NativeMotionInventory(tuple(supplies),(),True,'COMPLETED_ONCE')
    return NativeMotionViews(sensors,inventory),queries,original


def test_missing_appearance_preserves_heads_and_primary_joints_without_association():
    views,queries,_=world(appearance=False)
    q=queries[32];found=views.record(q,32)
    heads=views.candidates(q,32)
    assert found['head_inventory']==(7,)
    assert len(heads)==1 and heads[0]['vector'] is None
    assert heads[0]['actor_association_state']=='APPEARANCE_UNOBSERVED'
    assert heads[0]['descriptor'].frame_local_head_id==7
    assert views.candidates(q,32,association_ready=True)==()
    assert found['row']['native_observation']['heads']
    selected,receipt=views.connected_actor_views(queries,tuple(queries),None)
    assert all(v is None for v in selected.values()) and receipt['state']=='ANCHOR_UNOBSERVED'
    assert views.sensors.session.budget.counts['new_decodes']==0
    assert views.sensors.session.budget.counts['model_calls']==0


def test_actual_camera_projection_and_connected_path_reuse_the_exact_raw_parent():
    views,queries,original=world(camera=True)
    anchor=views.candidates(queries[32],32,association_ready=True)[0]
    ticks=tuple(queries)
    selected,receipt=views.connected_actor_views(queries,ticks,anchor)
    again,same=views.connected_actor_views(queries,ticks,anchor)
    assert again is selected and same is receipt
    assert all(v is not None for v in selected.values())
    assert all(v['source_actor_path_identity']==receipt['hypothesis_sha256'] for v in selected.values())
    row=selected[32]['row'];raw_head=row['native_observation']['heads'][0]['box_xywh']
    rendered_head=row['projection']['heads'][0]['projected_box_xywh']
    assert rendered_head!=pytest.approx(raw_head)
    assert digest(row['native_observation'])==anchor['appearance']['recorded_native_observation_sha256']
    frozen,proof=views.freeze_original_actors(queries,ticks,original_program_identity=original.program_sha256)
    assert set(frozen)==set(ticks) and proof['donor_costs_used'] is False
    assert views.sensors.session.budget.counts['model_calls']==0


def test_missing_raw_slot_is_not_filled_from_a_neighbor_and_closed_session_is_rejected():
    views,queries,_=world(missing_tick=33)
    assert views.record(queries[33],33)['row'] is None
    assert views.candidates(queries[33],33)==()
    anchor=views.candidates(queries[32],32,association_ready=True)[0]
    selected,_=views.connected_actor_views(queries,tuple(queries),anchor)
    assert selected[33] is None
    views.sensors.session.closed=True
    with pytest.raises(ValueError,match='live request session'):
        views.record(queries[32],32)


def test_appearance_alias_conflict_cannot_choose_the_favorable_vector():
    views,_,_=world()
    first=next(iter(views.supplies.values()))
    changed=json.loads(json.dumps(first.appearance))
    changed['heads'][0]['embedding']=[0.,1.]+[0.]*126
    inventory=NativeMotionInventory((first,replace(first,appearance=changed)),(),True,'COMPLETED_ONCE')
    with pytest.raises(ValueError,match='aliases changed their appearance'):
        NativeMotionViews(views.sensors,inventory)


def test_live_typed_appearance_and_json_archive_have_the_same_physical_parent():
    views,queries,_=world()
    old=next(iter(views.supplies.values()))
    live=dict(old.appearance,source_file_identity=old.observation.source_file_identity)
    inventory=NativeMotionInventory((replace(old,appearance=live),old),(),True,'COMPLETED_ONCE')
    actual=NativeMotionViews(views.sensors,inventory)
    found=actual.record(queries[30],30)
    assert found['appearance']['source_file_identity']==found['row']['native_observation']['source_file_identity']
    assert found['appearance']['recorded_native_observation_sha256']==digest(found['row']['native_observation'])
    assert actual.candidates(queries[30],30,association_ready=True)
    assert not inventory.supplies[0].observation.to_dict()['quality_approved']


def test_verified_archived_and_live_mount_parents_share_one_physical_actor_run():
    views,queries,original=world()
    current=views.sensors.session._assert_source_identity(2)
    archived=(current[0]+1,*current[1:])
    binding=json.dumps(dict(state='CONTENT_VERIFIED_DEVICE_REMOUNT',content_sha256='a'*64,
        recorded_file_identity=archived,current_file_identity=current,archived_observations_rewritten=False),
        sort_keys=True,separators=(',',':'))
    views.sensors.recorded_motion_bindings={2:binding}
    supplies=[]
    for supply in views.supplies.values():
        if supply.demand.output_tick%2:
            raw=replace(supply.observation,source_file_identity=archived)
            app=dict(supply.appearance,source_file_identity=list(archived),
                recorded_native_observation_sha256=digest(raw.to_dict()))
            supply=replace(supply,observation=raw,appearance=app,source_equivalence_json=binding)
        supplies.append(supply)
    mixed=NativeMotionViews(views.sensors,NativeMotionInventory(tuple(supplies),(),True,'COMPLETED_ONCE'))
    anchor=mixed.candidates(queries[32],32,association_ready=True)[0]
    selected,receipt=mixed.connected_actor_views(queries,tuple(queries),anchor)
    assert all(v is not None for v in selected.values()) and receipt['all_sampled_frames_connected']
    assert selected[33]['row']['native_observation']['source_file_identity']==list(archived)
    assert selected[32]['row']['native_observation']['source_file_identity']==list(current)
    from crosscut.group_review_motion_seam import review_native_window_continuous
    assert review_native_window_continuous(list(selected.values()))
    frozen,proof=mixed.freeze_original_actors(queries,tuple(queries),
        original_program_identity=original.program_sha256)
    assert all(v is not None for v in frozen.values()) and len(proof['groups'])==1
    assert mixed.sensors.session.budget.counts['new_decodes']==0


def test_unverified_mount_alias_or_changed_raw_parent_cannot_grant_actor_continuity():
    from crosscut.group_review_source_equivalence import physical_review_file_identity
    views,queries,_=world();head=views.candidates(queries[32],32,association_ready=True)[0]
    untyped=dict(head,verified_native_file_binding=dict(physical_file_identity=(1,2,3,4,5)))
    with pytest.raises(ValueError,match='untyped source alias'):
        physical_review_file_identity(untyped)
    changed=dict(head,row=dict(head['row'],native_observation=dict(head['row']['native_observation'],native_frame_index=999)))
    with pytest.raises(ValueError,match='another native raw parent'):
        physical_review_file_identity(changed)
    old=next(iter(views.supplies.values()));identity=old.observation.source_file_identity
    bad=replace(old,observation=replace(old.observation,source_file_identity=(identity[0]+1,*identity[1:])))
    unverified=NativeMotionViews(views.sensors,NativeMotionInventory((bad,),(),True,'COMPLETED_ONCE'))
    # No features: isolate the file-equivalence failure, not a feature-parent mismatch.
    unverified.supplies[old.demand.native_key]=replace(bad,appearance=None)
    with pytest.raises(ValueError,match='sensor-verified source equivalence'):
        unverified.record(queries[30],30)


def test_authored_camera_projects_the_selected_path_without_a_second_actor_search():
    from types import SimpleNamespace as NS
    views,queries,_=world()
    anchor=views.candidates(queries[32],32,association_ready=True)[0]
    selected,receipt=views.connected_actor_views(queries,tuple(queries),anchor)
    selected[33]=None
    selected[34]=dict(selected[34],reacquisition_ambiguous=True)
    before=queries[32].camera._registry.plan
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),))
    plan=replace(before,segments=(before.segments[0],replace(before.segments[1],
        editorial_transform=camera),*before.segments[2:]))
    context=NS(incumbent=NS(render_plan=before,visual_timeline=NS(choices=[
        NS(shot_type=k) for k in ('medium','group_wide','close_up','medium')]),face_match_plan=None))
    original=freeze_owned_scene_program(context,views.domain)
    compiled=compile_common_clock_camera_queries(context=context,domain=views.domain,original_program=original,
        candidate_plan=plan,output_intervals=((30,36),))
    authored={t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    actual=views.reproject_selected_actor_path(authored,selected)
    assert actual[33] is None and actual[34]['reacquisition_ambiguous']
    for tick,value in actual.items():
        if value is None:continue
        assert value['source_actor_path_identity']==receipt['hypothesis_sha256']
        assert value['descriptor'].frame_local_head_id==selected[tick]['descriptor'].frame_local_head_id
        assert value['row']['native_observation']==selected[tick]['row']['native_observation']
    assert actual[32]['descriptor'].face_height>selected[32]['descriptor'].face_height
    assert views.sensors.session.budget.counts['model_calls']==0


def test_authored_camera_cannot_reuse_an_actor_from_another_native_phase():
    views,queries,_=world()
    selected={32:views.candidates(queries[32],32,association_ready=True)[0]}
    context,domain,_=fixture(native_fps=Fraction(30))
    context.incumbent.face_match_plan=None
    original=freeze_owned_scene_program(context,domain)
    before=context.incumbent.render_plan
    changed=replace(before,segments=(before.segments[0],replace(before.segments[1],
        source_start=before.segments[1].source_start+1/30),*before.segments[2:]))
    compiled=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
        candidate_plan=changed,output_intervals=((30,36),))
    phased={t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    with pytest.raises(ValueError,match='native schedule'):
        views.reproject_selected_actor_path(phased,selected)
