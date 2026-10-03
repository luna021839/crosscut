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
            segment = view.state.segments[0]
            actual = segment.editorial_transform.evaluate((tick-view.state.first_output_frame)/fps)
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
    assert segment_frame_counts(result.observation_scaffold.plan)==(30,15,15,30,30)
    assert samples(result.plan,domain)==samples(original,domain)
    assert visible_group_cut_inventory(result.plan,domain)==visible_group_cut_inventory(original,domain)


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
