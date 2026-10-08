"""Original role epochs are views of one camera, not permission for extra cuts."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from crosscut.group_actual_shots import visible_group_cut_inventory, visible_group_shots
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack, segment_frame_counts
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.shot_context_camera import slice_shot_camera_track
from tests import test_group_joint_state_factory as factory_fixture


def fixture(*, fps=30., native_fps=Fraction(60), subject=None, cut=None):
    plan,domain,_ = factory_fixture.JointStateFactoryTests().fixture(fps,native_fps)
    plan = replace(plan,segments=(plan.segments[0],plan.segments[1],
        replace(plan.segments[2],source_index=plan.segments[1].source_index),plan.segments[3]))
    if cut is not None:
        domain = replace(domain,scene_cut_times=((),(cut,),(),()))
    context = NS(incumbent=NS(render_plan=plan,
        visual_timeline=NS(choices=[NS(shot_type=kind) for kind in
            ('medium','group_wide','close_up','medium')]),
        face_match_plan=NS(segments=[NS(alignment_subject=s,applied=False)
            for s in (None,subject,None,None)])))
    shot = next(s for s in visible_group_shots(plan,domain) if s.first_output_frame==30)
    return context,domain,shot


def compiled(context,domain,shot,**kwargs):
    from crosscut.group_scene_epoch_program import (
        freeze_original_role_epochs,compile_role_epoch_views)
    original = freeze_original_role_epochs(context,shot)
    return compile_role_epoch_views(context=context,domain=domain,shot=shot,
        original_role_program=original,**kwargs)


def samples(plan,domain):
    return [(r.source_index,r.native_frame_index,r.source_request_seconds)
        for r in native_frame_clocks_for_output(plan,domain.clocks_by_request,
            range(plan.output_frame_count)).values()]


def test_mixed_role_camera_is_frozen_as_distinct_original_duty_views():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs
    from crosscut.group_scene_intent import declared_shot_intent,validate_intent
    context,domain,shot = fixture()
    with pytest.raises(ValueError,match='mixed'):
        declared_shot_intent(context,shot)
    program = freeze_original_role_epochs(context,shot)
    assert [(e['first_output_frame'],e['end_output_frame'],e['role'])
        for e in program['epochs']] == [(30,60,'observed_body_composition'),(60,90,'focal_head')]
    assert program['original_output_cell_roles_frozen'] is True
    assert program['original_native_part_lifecycle_frozen'] is False
    result = compiled(context,domain,shot)
    assert [v.state.role for v in result.views] == [GroupOutputRole.COMPOSITION,GroupOutputRole.FOCAL_HEAD]
    for view in result.views:
        validate_intent(view.intent,role=view.state.role.value,first=view.state.first_output_frame,
            end=view.state.end_output_frame)
    assert result.to_dict()['candidate_admitted'] is False
    assert result.to_dict()['full_bundle_created'] is False
    assert samples(result.plan,domain)==samples(context.incumbent.render_plan,domain)


def test_normal_neutral_baseline_without_face_plan_retains_declared_roles():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs
    context,domain,shot=fixture()
    context.incumbent.face_match_plan=None
    program=freeze_original_role_epochs(context,shot)
    assert [e['role'] for e in program['epochs']]==['observed_body_composition','focal_head']
    assert all(c['alignment_subject'] is None and c['face_applied'] is False
        for c in program['original_output_cells'])
    context.incumbent.visual_timeline.choices[1].shot_type='unknown'
    unknown=freeze_original_role_epochs(context,shot)
    assert unknown['epochs'][0]['role'] is None
    assert unknown['epochs'][0]['unresolved_reason']


@pytest.mark.parametrize('fps,native',[(30.,Fraction(30)),(30.,Fraction(60)),
    (29.97,Fraction(30000,1001)),(29.97,Fraction(60000,1001))])
def test_whole_camera_slice_keeps_global_phase_for_each_role_epoch(fps,native):
    context,domain,shot = fixture(fps=fps,native_fps=native)
    original = context.incumbent.render_plan
    camera = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.12,.49,.5),
        EditorialTransformKeyframe(.9,1.36,.53,.48,.6),
        EditorialTransformKeyframe(2.1,1.18,.50,.51,-.4)),sampling_canvas_zoom=1.4)
    counts = segment_frame_counts(original)
    updated = list(original.segments)
    for index,offset in ((1,0),(2,counts[1])):
        updated[index] = replace(updated[index],editorial_transform=slice_shot_camera_track(
            camera,frame_offset=offset,frame_count=counts[index],fps=fps))
    candidate = replace(original,segments=tuple(updated))
    result = compiled(context,domain,shot,candidate_plan=candidate)
    assert len(result.views)==2
    assert samples(result.plan,domain)==samples(candidate,domain)
    for view in result.views:
        for tick in range(view.state.first_output_frame,view.state.end_output_frame):
            segment,phase,_ = view.state.sample_at(tick)
            actual = segment.editorial_transform.evaluate(phase)
            expected = camera.evaluate((tick-shot.first_output_frame)/fps)
            for name in ('zoom','center_x','center_y','roll_degrees'):
                assert getattr(actual,name)==pytest.approx(getattr(expected,name),abs=1e-9)
            assert segment.editorial_transform.effective_sampling_canvas_zoom==1.4


def test_native_cut_inside_cell_is_not_erased_by_a_role_fragment():
    context,domain,shot = fixture(native_fps=Fraction(30),cut=1.5)
    original = context.incumbent.render_plan
    assert (shot.first_output_frame,shot.end_output_frame)==(30,45)
    result = compiled(context,domain,shot)
    assert len(result.views)==1
    assert len(result.views[0].state.native_frame_indices)==15
    assert segment_frame_counts(result.plan)==(30,30,30,30)
    assert result.camera.canonical_segment_output_frame_counts==(30,)
    assert result.to_dict()['render_cells_split_for_observation'] is False
    assert samples(result.plan,domain)==samples(original,domain)
    assert visible_group_cut_inventory(result.plan,domain)==visible_group_cut_inventory(original,domain)


def test_one_frame_original_role_epoch_is_a_query_not_a_new_render_cell():
    context,domain,_ = fixture(native_fps=Fraction(30),cut=59/30)
    context.incumbent.visual_timeline.choices[1].shot_type='close_up'
    context.incumbent.visual_timeline.choices[2].shot_type='group_wide'
    original=context.incumbent.render_plan
    shot=next(s for s in visible_group_shots(original,domain) if s.first_output_frame==59)
    result=compiled(context,domain,shot)
    assert [(v.state.first_output_frame,v.state.end_output_frame,v.state.role) for v in result.views]==[
        (59,60,GroupOutputRole.FOCAL_HEAD),(60,90,GroupOutputRole.COMPOSITION)]
    assert len(result.views[0].state.native_frame_indices)==1
    assert result.views[0].state.camera is result.views[1].state.camera
    assert result.plan is original
    assert samples(result.plan,domain)==samples(original,domain)
    assert visible_group_cut_inventory(result.plan,domain)==visible_group_cut_inventory(original,domain)
    assert result.to_dict()['render_cells_split_for_observation'] is False


def test_subcell_query_projects_the_original_canonical_camera_phase():
    from crosscut.group_source_changing_owner import _at
    context,domain,_=fixture(native_fps=Fraction(30),cut=1.5)
    plan=context.incumbent.render_plan
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
        EditorialTransformKeyframe(1.,1.4,.53,.49,.5)),sampling_canvas_zoom=1.5)
    context.incumbent.render_plan=replace(plan,segments=(plan.segments[0],
        replace(plan.segments[1],editorial_transform=camera),*plan.segments[2:]))
    original=context.incumbent.render_plan
    shot=next(s for s in visible_group_shots(original,domain) if s.first_output_frame==45)
    result=compiled(context,domain,shot)
    assert result.camera.segments==original.segments[1:3]
    segment,phase,native=_at(result.views[0].state,45)
    assert segment is original.segments[1]
    assert phase==pytest.approx(.5)
    assert segment.editorial_transform.evaluate(phase)==camera.evaluate(.5)
    assert native==native_frame_clocks_for_output(original,domain.clocks_by_request,[45])[45].native_frame_index
    assert result.camera.first_output_frame==45
    assert result.views[0].state.first_output_frame==45


def test_role_boundary_is_not_a_license_for_a_new_source_switch():
    context,domain,shot = fixture()
    plan = context.incumbent.render_plan
    segments = list(plan.segments)
    segments[2] = replace(segments[2],source_index=domain.eligible_indices[2])
    with patch('crosscut.group_joint_observations.GroupJointObservationSession') as sensor, \
            pytest.raises(ValueError,match='cut inventory'):
        compiled(context,domain,shot,candidate_plan=replace(plan,segments=tuple(segments)))
    sensor.assert_not_called()


def test_explicit_arms_on_a_wide_are_not_weakened_to_composition():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs
    context,domain,shot = fixture(subject='upper_body')
    program = freeze_original_role_epochs(context,shot)
    assert program['epochs'][0]['role']=='focal_head_and_observed_arms'
    assert compiled(context,domain,shot).views[0].state.role==GroupOutputRole.FOCAL_HEAD_ARMS


def test_unresolved_original_role_does_not_become_context_or_donor_selected_duty():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs
    context,domain,shot = fixture()
    context.incumbent.face_match_plan.segments[1].applied = True
    program = freeze_original_role_epochs(context,shot)
    assert program['epochs'][0]['role'] is None
    assert program['epochs'][0]['unresolved_reason']
    with pytest.raises(ValueError,match='unresolved original'):
        compiled(context,domain,shot)


def test_stale_original_program_is_rederived_from_external_incumbent():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs,compile_role_epoch_views
    context,domain,shot = fixture()
    program = freeze_original_role_epochs(context,shot)
    context.incumbent.face_match_plan.segments[1].alignment_subject = 'upper_body'
    with pytest.raises(ValueError,match='frozen original'):
        compile_role_epoch_views(context=context,domain=domain,shot=shot,original_role_program=program)


def test_complete_candidate_cannot_change_an_unchanged_outer_camera():
    context,domain,shot = fixture()
    plan = context.incumbent.render_plan
    camera = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),EditorialTransformKeyframe(1.,1.2)))
    candidate = replace(plan,segments=(replace(plan.segments[0],editorial_transform=camera),*plan.segments[1:]))
    with pytest.raises(ValueError,match='outside'):
        compiled(context,domain,shot,candidate_plan=candidate)


@pytest.mark.parametrize('change',[{'fps':24.},{'master_audio_start':.5},{'width':1280,'height':720}])
def test_epoch_compiler_preserves_audio_fps_and_output_canvas(change):
    context,domain,shot = fixture()
    with pytest.raises(ValueError,match='output clock'):
        compiled(context,domain,shot,candidate_plan=replace(context.incumbent.render_plan,**change))


def test_role_views_keep_original_cell_provenance_and_full_unchanged_camera_inventory():
    context,domain,shot = fixture()
    result = compiled(context,domain,shot)
    assert result.views[0].intent['epoch_fragment_origin']['original_output_interval']==[30,90]
    assert result.views[1].intent['epoch_fragment_origin']['original_output_interval']==[30,90]
    assert result.views[0].intent['retained_output_cells']==result.views[1].intent['retained_output_cells']
    assert result.views[0].intent['cells']!=result.views[1].intent['cells']
    assert result.to_dict()['epoch_boundaries_are_not_edit_cuts'] is True


def test_common_clock_queries_follow_a_moved_physical_cut_without_moving_original_roles():
    from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
    from crosscut.group_scene_owned_program import freeze_owned_scene_program
    context,domain,_=fixture()
    original=context.incumbent.render_plan
    owned=freeze_owned_scene_program(context,domain)
    rows=list(original.segments)
    rows[0]=replace(rows[0],duration=35/30)
    rows[1]=replace(rows[1],master_start=35/30,source_start=35/30,duration=25/30)
    selected=replace(original,segments=tuple(rows))
    result=compile_common_clock_camera_queries(context=context,domain=domain,original_program=owned,
        candidate_plan=selected,output_intervals=((0,120),))
    assert [(q.first_output_frame,q.end_output_frame,q.role) for q in result.queries]==[
        (0,30,GroupOutputRole.FOCAL_HEAD),(30,35,GroupOutputRole.COMPOSITION),
        (35,60,GroupOutputRole.COMPOSITION),(60,90,GroupOutputRole.FOCAL_HEAD),
        (90,120,GroupOutputRole.FOCAL_HEAD)]
    a,b=result.queries[1:3]
    assert a.original_unit_id==b.original_unit_id==owned.units[1].unit_id
    assert a.camera is not b.camera
    assert result.queries[2].camera is result.queries[3].camera
    assert len(result.cameras)==3
    assert [e.first_incoming_output_frame for e in visible_group_cut_inventory(selected,domain)]==[35,90]
    assert result.plan is selected
    assert result.to_dict()['candidate_admitted'] is False


def test_common_clock_queries_retain_unknown_roles_and_refuse_to_sense_them_as_composition():
    from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
    from crosscut.group_scene_owned_program import freeze_owned_scene_program
    from crosscut.group_scene_observations import SceneSensors
    context,domain,_=fixture()
    context.incumbent.face_match_plan.segments[1].applied=True
    owned=freeze_owned_scene_program(context,domain)
    result=compile_common_clock_camera_queries(context=context,domain=domain,original_program=owned,
        candidate_plan=context.incumbent.render_plan,output_intervals=((30,90),))
    query=result.queries[0]
    assert query.role is None
    assert query.original_role_program['epochs'][query.epoch_index]['unresolved_reason']
    assert query.to_dict()['key_preimage']['role'] is None
    sensor=SceneSensors.__new__(SceneSensors)
    with patch.object(SceneSensors,'raw_native') as native,pytest.raises(ValueError,match='unresolved original'):
        sensor.raw(query,query.first_output_frame,None)
    native.assert_not_called()


@pytest.mark.parametrize('intervals',[((),),((30,30),),((30,60),(45,90)),((True,60),),((0,121),)])
def test_common_clock_query_scope_is_exact_and_cannot_drop_original_duty_addresses(intervals):
    from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
    from crosscut.group_scene_owned_program import freeze_owned_scene_program
    context,domain,_=fixture()
    with pytest.raises(ValueError,match='scope'):
        compile_common_clock_camera_queries(context=context,domain=domain,
            original_program=freeze_owned_scene_program(context,domain),
            candidate_plan=context.incumbent.render_plan,output_intervals=intervals)


def test_role_query_has_no_fabricated_motion_or_legacy_mutable_cell_authority():
    from crosscut.group_scene_camera_queries import GroupSceneRoleQuery
    context,domain,_=fixture(native_fps=Fraction(30),cut=59/30)
    context.incumbent.visual_timeline.choices[1].shot_type='close_up'
    context.incumbent.visual_timeline.choices[2].shot_type='group_wide'
    shot=next(s for s in visible_group_shots(context.incumbent.render_plan,domain) if s.first_output_frame==59)
    result=compiled(context,domain,shot);query=result.views[0].state
    assert isinstance(query,GroupSceneRoleQuery)
    assert query.to_dict()['motion_observed'] is False
    assert query.to_dict()['render_topology_materialized'] is False
    from crosscut.group_scene_observations import scene_anchor_ticks
    with pytest.raises(ValueError,match='too short'):
        scene_anchor_ticks(query.first_output_frame,query.end_output_frame)
    with pytest.raises(ValueError,match='duties'):
        replace(query,role=GroupOutputRole.COMPOSITION)
    with pytest.raises(ValueError,match='non-epoch'):
        query.sample_at(60)


def test_actual_native_decoder_and_encoder_share_full_camera_phase_across_one_frame_role(tmp_path):
    """Real chart codec/projection, no NN substitutions or human quality claim."""
    import shutil
    import cv2
    import numpy as np
    from time import monotonic
    from pathlib import Path
    from crosscut.group_joint_observations import GroupJointObservationSession
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    from crosscut.render_frame_schedule import probe_render_frame_clock
    from crosscut.prescribed_render_geometry import PrescribedGeometrySampler
    from crosscut.post_render_perceptual_quality import _reference_view_for_span
    from crosscut.render import render
    from engine.tests.test_scene_source_owner import encode_scene_chart
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('native chart codec tools unavailable')
    context,domain,_=fixture(native_fps=Fraction(30),cut=59/30)
    paths=tuple(Path(tmp_path/f'EXPLICIT-NON-HUMAN-query-source-{i}.mp4').resolve() for i in range(6))
    # The production encoder opens the full request source list. Observation
    # decoders remain eligible-only; this is not a test of missing inputs.
    for i,path in enumerate(paths):encode_scene_chart(path,variant=i,frames=180)
    domain=replace(domain,request_sources=paths,render_frame_clocks=tuple(
        probe_render_frame_clock(paths[i]) for i in domain.eligible_indices),
        audio_offsets=(.17,-.24,.09,.33))
    plan=replace(context.incumbent.render_plan,sources=paths,width=320,height=180,
        output=Path(tmp_path/'EXPLICIT-NON-HUMAN-full-camera-render.mp4'))
    whole=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1,.49,.5),
        EditorialTransformKeyframe(1.,1.4,.54,.48,.8),
        EditorialTransformKeyframe(2.,1.25,.51,.5,-.6)),sampling_canvas_zoom=1.5)
    rows=list(plan.segments)
    for i,offset in ((1,0),(2,30)):
        rows[i]=replace(rows[i],editorial_transform=slice_shot_camera_track(
            whole,frame_offset=offset,frame_count=30,fps=30.))
    plan=replace(plan,segments=tuple(rows));context.incumbent.render_plan=plan
    context.incumbent.visual_timeline.choices[1].shot_type='close_up'
    context.incumbent.visual_timeline.choices[2].shot_type='group_wide'
    shot=next(s for s in visible_group_shots(plan,domain) if s.first_output_frame==59)
    result=compiled(context,domain,shot)
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+60.,maximum_new_decodes=24)
    session=GroupJointObservationSession(domain=domain,budget=budget)
    projected={};wrong={}
    try:
        for tick in (59,60,74,89):
            query=next(v.state for v in result.views if v.state.first_output_frame<=tick<v.state.end_output_frame)
            projected[tick]=session.projected_pixels(query,tick)
            assert session.projected_pixels(result.camera,tick) is projected[tick]
            segment,phase,native=query.sample_at(tick)
            frame=session.native_pixels(segment.source_index,native)
            sampler=PrescribedGeometrySampler(NS(frame_rate=30.),segment,master_start=segment.master_start)
            expected=_reference_view_for_span(frame,sampler.span,relative_time=phase,size=(320,180))
            np.testing.assert_array_equal(projected[tick],expected)
            wrong[tick]=_reference_view_for_span(frame,sampler.span,relative_time=0.,size=(320,180))
        assert budget.counts['new_decodes']==4
        assert len(session.geometry)==2
    finally:
        session.close()
    render(plan,video_codec='libx264',video_bitrate='2M',video_maxrate='3M',
        video_buffer_size='3M',maximum_runtime_seconds=30.)
    capture=cv2.VideoCapture(str(plan.output));decoded=0;found={}
    try:
        while True:
            ok,pixels=capture.read()
            if not ok:break
            if decoded in projected:found[decoded]=pixels
            decoded+=1
    finally:capture.release()
    assert decoded==plan.output_frame_count
    assert set(found)==set(projected)
    for tick in found:
        error=float(np.mean(np.abs(found[tick].astype(np.int16)-projected[tick].astype(np.int16))))
        assert error<5.,(tick,error)
    # The one-frame query must NOT get a restarted zoom/roll. Compare actual
    # decoded output, not only equivalent metadata or an invented motion flag.
    correct=float(np.mean(np.abs(found[59].astype(np.int16)-projected[59].astype(np.int16))))
    restarted=float(np.mean(np.abs(found[59].astype(np.int16)-wrong[59].astype(np.int16))))
    assert correct<restarted*.5,(correct,restarted)
