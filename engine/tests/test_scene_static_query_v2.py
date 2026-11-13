"""Real query orchestration, explicitly substituted nonhuman native rows."""
from dataclasses import replace
from time import monotonic
from unittest.mock import patch
import pytest

from crosscut.group_scene_static_query import observe_static_role_query, reproject_recorded_static_query
from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from engine.tests.test_registered_camera_queries import world
from engine.tests.test_scene_focal_pose_motion import scene_world


def test_one_tick_epoch_is_sensed_once_without_fake_motion_or_camera_phase_restart():
    domain,compiled=world();query=compiled.views[0].state
    assert (query.first_output_frame,query.end_output_frame)==(59,60)
    _,_,_,raws,_=scene_world();raw=raws[0]
    raw=replace(raw,request_source_index=domain.request_index(query.eligible_source_index),
        native_frame_index=query.native_frame_indices[0],source_clock_identity=query.source_clock_identity)
    cache={}
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(
            absolute_deadline=monotonic()+10)) as session:
        with patch('crosscut.group_scene_static_query.observe_native_group',return_value=raw) as native, \
                patch.object(session,'_assert_source_identity'),patch.object(session,'_assert_model_assets'):
            first=observe_static_role_query(session,query,59,cache=cache)
            second=observe_static_role_query(session,query,59,cache=cache)
        native.assert_called_once()
    assert first.to_dict()==second.to_dict()
    result=first.to_dict()
    assert result['output_tick']==59 and result['projection']['canonical_segment_phase_seconds']==pytest.approx(29/30)
    assert not result['motion_observed'] and not result['quality_approved']
    assert result['query']['original_role_program']==query.original_role_program


def test_foreign_camera_is_rejected_before_any_model_call():
    domain,compiled=world();query=compiled.views[0].state
    forged=replace(query,camera=replace(query.camera,native_frame_indices=tuple(n+1 for n in query.camera.native_frame_indices)))
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10)) as session:
        with patch('crosscut.group_scene_static_query.observe_native_group') as native,pytest.raises(ValueError,match='registered'):
            observe_static_role_query(session,forged,59)
        native.assert_not_called()


def test_cached_observations_do_not_cross_native_owners():
    domain,compiled=world();query=compiled.views[0].state
    _,_,_,raws,_=scene_world()
    raw=replace(raws[0],request_source_index=domain.request_index(query.eligible_source_index),
        native_frame_index=query.native_frame_indices[0],source_clock_identity=query.source_clock_identity)
    cache={}
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10)) as first:
        with patch('crosscut.group_scene_static_query.observe_native_group',return_value=raw):
            observe_static_role_query(first,query,59,cache=cache)
    with GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10)) as second:
        with patch('crosscut.group_scene_static_query.observe_native_group') as native,pytest.raises(ValueError,match='another native owner'):
            observe_static_role_query(second,query,59,cache=cache)
        native.assert_not_called()


def recorded_fixture():
    domain,compiled=world();query=compiled.views[0].state
    _,_,_,raws,_=scene_world()
    raw=replace(raws[0],request_source_index=domain.request_index(query.eligible_source_index),
        native_frame_index=query.native_frame_indices[0],source_clock_identity=query.source_clock_identity)
    return domain,query,raw.to_dict()


def test_recorded_numeric_reprojection_never_claims_fresh_native_or_motion():
    domain,query,raw=recorded_fixture()
    with patch('crosscut.group_scene_static_query.observe_native_group') as native:
        row=reproject_recorded_static_query(query,59,raw,domain=domain)
    native.assert_not_called()
    assert row['projection']['canonical_segment_phase_seconds']==pytest.approx(29/30)
    assert row['query']['original_role_program']==query.original_role_program
    for flag in ('one_frame_static_observation','fresh_inference_performed','live_native_authority',
            'motion_observed','actor_identity_certified','candidate_admitted','quality_approved'):
        assert row[flag] is False
    assert row['recorded_observation_reprojected'] is True


@pytest.mark.parametrize('field,value', [('request_source_index',99),('native_frame_index',99),
    ('source_clock_identity','another-clock'),('native_pixel_sha256',None),('quality_approved',True)])
def test_recorded_numeric_reprojection_cannot_borrow_native_source_clock_or_approval(field,value):
    domain,query,raw=recorded_fixture();raw[field]=value
    with pytest.raises(ValueError,match='borrowed'):
        reproject_recorded_static_query(query,59,raw,domain=domain)


def test_recorded_numeric_reprojection_requires_registered_camera():
    domain,query,raw=recorded_fixture()
    forged=replace(query,camera=replace(query.camera,native_frame_indices=tuple(n+1 for n in query.camera.native_frame_indices)))
    with pytest.raises(ValueError,match='registered'):
        reproject_recorded_static_query(forged,59,raw,domain=domain)


def test_compact_review_reprojection_keeps_actual_coordinates_and_native_parent():
    domain,query,raw=recorded_fixture()
    full=reproject_recorded_static_query(query,59,raw,domain=domain)
    compact=reproject_recorded_static_query(query,59,raw,domain=domain,compact=True)
    assert compact['projection']==full['projection']
    assert compact['native_observation']==full['native_observation']
    assert compact['query']['registered_query_sha256']==query.key
    assert compact['query']['camera']['output_size']==list(query.camera.output_size)
    assert not compact['fresh_inference_performed'] and not compact['quality_approved']
