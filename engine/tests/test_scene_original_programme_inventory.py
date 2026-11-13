"""Local programme versions, not controller history or human edit quality.

The real lifecycle freezer uses explicit nonhuman pose/head-link substitutes.
No new media, model inference, acquisition or quality authority is exercised.
"""
from copy import deepcopy

import pytest

from crosscut import group_scene_original_duties as original
from crosscut import group_scene_observations as scene
from engine.tests.test_scene_original_duties import freeze
from engine.tests.test_scene_focal_pose_motion import scene_world


def two_programmes():
    sensors,state,h,_,_=scene_world()
    first=freeze(sensors,state,h)
    sensors.install_original_duty_program(first)
    first_row=deepcopy(scene.family.plain(sensors.frame(state,0)))
    second=original.freeze_original_duty_program(sensors,((state,),),original_plan_identity='b'*64)
    sensors.install_original_duty_program(second)
    second_row=deepcopy(scene.family.plain(sensors.frame(state,0)))
    return sensors,first,second,first_row,second_row


def test_installation_does_not_borrow_the_callers_mutable_programme():
    sensors,state,h,_,_=scene_world();program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program)
    received=deepcopy(program)
    program['units'][0]['rows'][0]['required_part_names'].clear()
    assert sensors.original_duty_program==received


def test_replacing_active_programme_retains_the_prior_frame_reference_version():
    sensors,first,second,old_row,new_row=two_programmes()
    archive=scene.family.plain(sensors.archive())
    records=archive['original_programme_records']
    assert set(records['native_duty_programmes'])=={first['program_sha256'],second['program_sha256']}
    assert records['active_native_duty_programme_sha256']==second['program_sha256']
    assert old_row['original_duty_reference']['program_sha256']==first['program_sha256']
    assert new_row['original_duty_reference']['program_sha256']==second['program_sha256']
    assert records['controller_owned_history_certified'] is False
    assert records['pre_receipt_history_completeness_certified'] is False


def test_same_exact_programme_is_one_storage_version_not_a_new_measurement():
    sensors,state,h,_,calls=scene_world();program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program);before=list(calls)
    sensors.install_original_duty_program(deepcopy(program))
    records=scene.family.plain(sensors.archive())['original_programme_records']
    assert list(records['native_duty_programmes'])==[program['program_sha256']]
    assert calls==before


def test_prior_duty_reference_is_validated_against_its_own_received_version():
    from crosscut.group_scene_programme_archive import validate_programme_records
    sensors,first,second,old_row,new_row=two_programmes()
    archive=scene.family.plain(sensors.archive())
    raw={scene.family.digest(row):row for row in archive['native_rows']}
    records=archive['original_programme_records']
    validate_programme_records(records,raw_inventory=raw,
        reference_roots=(old_row,new_row),required_programme=second)
    assert first['program_sha256']!=second['program_sha256']


@pytest.mark.parametrize('mode',['drop_old','wrong_row','wrong_unit','promote_history','drop_active'])
def test_consistently_rehashed_inventory_cannot_remove_or_borrow_a_received_duty(mode):
    from crosscut.group_scene_programme_archive import validate_programme_records
    sensors,first,second,old_row,new_row=two_programmes()
    archive=scene.family.plain(sensors.archive())
    records=deepcopy(archive['original_programme_records']);row=deepcopy(old_row)
    if mode=='drop_old':records['native_duty_programmes'].pop(first['program_sha256'])
    elif mode=='wrong_row':row['original_duty_reference']['row_sha256']='c'*64
    elif mode=='wrong_unit':row['original_duty_reference']['original_unit_id']='d'*64
    elif mode=='promote_history':records['controller_owned_history_certified']=True
    else:records['native_duty_programmes'].pop(second['program_sha256'])
    raw={scene.family.digest(value):value for value in archive['native_rows']}
    with pytest.raises(ValueError):validate_programme_records(records,raw_inventory=raw,
        reference_roots=(row,new_row),required_programme=second)


def test_mixed_original_roles_survive_locally_without_becoming_a_uniform_duty():
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs
    from engine.tests.test_scene_role_epoch_program import fixture
    context,domain,shot=fixture()
    sensors,_,_,_,_=scene_world()
    program=freeze_original_role_epochs(context,shot)
    identity=sensors.record_original_role_program(program)
    program['epochs'][0]['role']='focal_head'
    stored=scene.family.plain(sensors.archive())['original_programme_records']
    assert [epoch['role'] for epoch in stored['original_role_programmes'][identity]['epochs']]==[
        'observed_body_composition','focal_head']
    assert stored['global_mixed_role_camera_owner_implemented'] is False
    assert stored['original_output_visibility_certified'] is False


@pytest.mark.parametrize('mode',['role','drop_epoch','rename_cell','promote_lifecycle','native_source','stale_plan'])
def test_rehashed_role_programme_cannot_weaken_or_retime_its_original_scope(mode):
    from crosscut.group_scene_epoch_program import freeze_original_role_epochs,validate_original_role_epochs
    from engine.tests.test_scene_role_epoch_program import fixture
    context,domain,shot=fixture();program=freeze_original_role_epochs(context,shot)
    if mode=='role':program['epochs'][0]['role']='focal_head'
    elif mode=='drop_epoch':program['epochs'].pop()
    elif mode=='rename_cell':program['original_output_cells'][1]['segment_index']=99
    elif mode=='promote_lifecycle':program['original_native_part_lifecycle_frozen']=True
    elif mode=='native_source':program['original_shot']['source_index']=5
    else:program['original_plan_identity']='c'*64
    with pytest.raises(ValueError):validate_original_role_epochs(program,original_plan=context.incumbent.render_plan)
