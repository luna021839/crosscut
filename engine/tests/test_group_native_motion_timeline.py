"""Numeric topology/pixel-statistic tests, not neural or media-quality proof."""
from dataclasses import replace
import json

import numpy as np
import pytest

from crosscut.group_joint_bundle_metadata import SelectedNativeActorFrame, SelectedNativeMotionShot
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_native_motion_programme import NativeMotionSide, realize_native_motion_programme
from crosscut.group_native_motion_timeline import (CONTRACT, metadata_cells, projected_pixel_quality,
    _fresh_choice, _clear_incoming_edge)
from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan
from tests.test_scene_role_epoch_program import fixture


def numeric_metadata(first=30, cut=61, end=90):
    context, domain, _ = fixture()
    result = realize_native_motion_programme(original=context.incumbent.render_plan, domain=domain,
        sides=(NativeMotionSide(first, cut, 2, 1), NativeMotionSide(cut, end, 3, -1)))
    from crosscut.group_review_interval_compile import output_offsets
    offsets = output_offsets(result.render_plan)
    rows = native_frame_clocks_for_output(result.render_plan, domain.clocks_by_request, range(first, end))
    shots = tuple(SelectedNativeMotionShot(a, b, s, (0, 10000), GroupOutputRole.FOCAL_HEAD,
        phase, None, tuple(SelectedNativeActorFrame(t, s, rows[t].native_frame_index, None, None)
            for t in range(a, b)), tuple(range(offsets.index(a), offsets.index(b))))
        for a, b, s, phase in ((first, cut, 2, 1), (cut, end, 3, -1)))
    cells = metadata_cells(original=context.incumbent.render_plan, selected=result.render_plan,
        domain=domain, selected_shots=shots)
    return context, domain, result, shots, cells


@pytest.mark.parametrize('cut', [57, 60, 61, 63])
def test_changed_topology_never_source_mutates_old_choice_measurements(cut):
    context, domain, result, shots, cells = numeric_metadata(cut=cut)
    assert [(c.first_output_frame, c.end_output_frame) for c in cells] == [(0, 30), (30, cut), (cut, 90), (90, 120)]
    assert [c.retained_old_segment_index for c in cells] == [0, None, None, 3]
    assert [c.refresh_source_observations for c in cells] == [False, True, True, False]
    assert [c.refresh_incoming_edge for c in cells] == [False, True, True, True]
    assert len({t for c in cells if c.refresh_source_observations for t in c.taps}) == 6


def test_sliced_immutable_neighbor_gets_fresh_rows_not_full_cell_sample_labels():
    _, _, _, _, cells = numeric_metadata(first=35, end=85)
    assert [c.retained_old_segment_index for c in cells] == [0, None, None, None, None, 3]
    assert [c.taps for c in cells if c.refresh_source_observations] == [
        (30, 32, 34), (35, 47, 60), (61, 72, 84), (85, 87, 89)]


def test_outside_camera_change_is_not_laundered_as_retained_metadata():
    context, domain, result, shots, _ = numeric_metadata()
    plan = result.render_plan
    changed = replace(plan, segments=(replace(plan.segments[0], editorial_transform=EditorialTransformTrack(
        (EditorialTransformKeyframe(0., 1.2),))), *plan.segments[1:]))
    with pytest.raises(ValueError, match='camera/look changed'):
        metadata_cells(original=context.incumbent.render_plan, selected=changed, domain=domain, selected_shots=shots)


def test_outside_native_drift_is_not_laundered_as_retained_metadata():
    context, domain, result, shots, _ = numeric_metadata()
    plan = result.render_plan
    changed = replace(plan, segments=(replace(plan.segments[0], source_start=.5), *plan.segments[1:]))
    with pytest.raises(ValueError, match='selected source/native sampler'):
        metadata_cells(original=context.incumbent.render_plan, selected=changed, domain=domain, selected_shots=shots)


def sample(tick, box=None):
    # Explicit numerical metadata input, not a stub detector/model result.
    pixels = np.indices((90, 160)).sum(axis=0).astype(np.uint8)
    raster = np.repeat(pixels[:, :, None], 3, axis=2)
    return dict(output_tick=tick, pixels=projected_pixel_quality(raster), face_box=box,
        landmarks=None, head_roll=None, eye_scale=None,
        face_backend='NUMERIC_TEST_ONLY_NO_MODEL', person_count=2)


def test_fresh_unknown_actor_is_unknown_not_an_old_closeup_or_successful_zero():
    _, domain, result, _, cells = numeric_metadata()
    cell = cells[1]
    rows = {t: sample(t) for t in cell.taps}
    choice = _fresh_choice(cell, result.render_plan.segments[1], domain, result.render_plan, rows, 'new', ())
    assert choice.shot_type == ShotType.UNKNOWN and choice.shot_scale is None
    assert choice.face_samples_xywh == (None, None, None)
    assert choice.boundary_score is None and choice.face_transition_minimax_mismatch_from_previous is None
    assert choice.dissolve_entry_face_identity_window == ()
    assert choice.face_match_candidate_failure == 'selected_actor_observation_unknown_no_salience_reselection'
    assert choice.quality_score == pytest.approx(rows[cell.taps[0]]['pixels']['quality_score'])
    assert choice.pose_confidences == (0., 0., 0.)
    assert choice.pose_backends == ('native_inventory_no_legacy_pose_embedding',)*3


def test_fresh_row_uses_new_source_clock_and_actual_selected_face_geometry():
    _, domain, result, _, cells = numeric_metadata()
    cell = cells[1]; segment = result.render_plan.segments[1]
    rows = {t: sample(t, (.3, .2, .12, .24)) for t in cell.taps}
    choice = _fresh_choice(cell, segment, domain, result.render_plan, rows, 'selected', ())
    assert choice.source_index == domain.eligible_index(segment.source_index)
    assert (choice.source_local_start, choice.source_local_end) == (segment.source_start, segment.source_end)
    assert choice.shot_type == ShotType.CLOSE_UP
    assert choice.shot_scale == pytest.approx(.24/.42)
    assert choice.face_samples_xywh == ((.3, .2, .12, .24),)*3
    assert choice.face_match_candidate_failure is None


def test_incoming_dependency_refresh_retains_own_face_but_clears_old_donor_proof():
    choice = VisualTimelineChoice(3, 3, 3., 4., 3., 4., .8, .8, .9, .6, ShotType.CLOSE_UP, 'prior',
        face_samples_xywh=((.3, .2, .12, .24),)*3,
        face_transition_minimax_mismatch_from_previous=.05, entry_face_geometry_window=((1., 2.),),
        exit_face_geometry_window=((3., 4.),), transition_evidence_contract_from_previous='OLD_DONOR',
        pose_transition_mismatch_from_previous=.05, layout_transition_mismatch_from_previous=.02,
        dissolve_entry_face_identity_window=((7., 8.),))
    refreshed = _clear_incoming_edge(choice)
    assert refreshed.face_samples_xywh == choice.face_samples_xywh
    assert refreshed.exit_face_geometry_window == choice.exit_face_geometry_window
    assert refreshed.quality_score == choice.quality_score
    assert refreshed.boundary_score is None and refreshed.pose_transition_mismatch_from_previous is None
    assert refreshed.layout_transition_mismatch_from_previous is None
    assert refreshed.entry_face_geometry_window == () and refreshed.dissolve_entry_face_identity_window == ()
    assert refreshed.transition_evidence_contract_from_previous is None


def test_pixel_quality_is_measured_without_claiming_semantic_composition_for_blank_pixels():
    black = projected_pixel_quality(np.zeros((90, 160, 3), np.uint8))
    lit = projected_pixel_quality(np.full((90, 160, 3), 118, np.uint8))
    assert black['composition_score'] is None and lit['composition_score'] is None
    assert black['quality_score'] < lit['quality_score']
    assert black['contrast'] == 0 and lit['sharpness'] == 0
    assert 'not_professional_quality' in lit['quality_basis']
    with pytest.raises(ValueError, match='actual projected BGR'):
        projected_pixel_quality(None)


def test_serialization_does_not_inherit_legacy_detector_pose_or_dissolve_claims():
    timeline = VisualTimelinePlan((), .5, analysis_basis=CONTRACT)
    value = timeline.to_dict()
    assert 'dissolve_safety_window' not in value and 'face_transition_policy' not in value
    assert value['observation_policy']['quality_approved'] is False
    assert value['observation_policy']['body_visibility'] is None
    assert value['observation_policy']['dissolve_proof'] == 'not_produced'
    assert value['observation_policy']['optimizer_score'].startswith('retained_selector_history')
    json.dumps(value, allow_nan=False)
