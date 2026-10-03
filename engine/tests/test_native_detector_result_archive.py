"""Returned detector-result retention, with EXPLICIT synthetic model outputs.

No real model, human video, inference quality or parent history is certified.
The actual native producer and result serialization/consumers are exercised.
"""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace as NS

import numpy as np
import pytest

from crosscut import group_joint_native_observer as native
from crosscut import group_scene_composition as composition


def produce(tmp_path, rows, *, body_only):
    source = tmp_path/'EXPLICIT_NONMEDIA_SOURCE.bin'
    source.write_bytes(b'synthetic identity only, never decoded by this fixture')
    pixels = np.zeros((180,320,3), np.uint8)
    clock = NS(to_dict=lambda:dict(contract='EXPLICIT_SYNTHETIC_NATIVE_CLOCK', frame_count=1))
    calls = []
    def observe(source, frame, *, model, arguments=None):
        calls.append(model)
        if model == 'nanodet_people':
            return rows
        if model == 'yunet_head_faces':
            return ()
        raise AssertionError('invalid or overflow detector rows must not cause pose inference')
    names = ('nanodet_people','rtmpose_body7','yunet_head_faces')
    session = NS(models={name:(None,None,'1'*64) for name in names},
        model_adapters={name:'2'*64 for name in names},
        native_pixels=lambda *args:pixels, observe_native=observe,
        domain=NS(request_sources=(source,), eligible_index=lambda source:0,
            render_frame_clocks=(clock,)))
    producer = composition.observe_native_composition if body_only else native.observe_native_group
    row = producer(session, request_source_index=0, native_frame_index=0)
    return row, calls


@pytest.mark.parametrize('body_only', [False, True])
def test_overflow_retains_all_thirteen_exact_returned_rows(tmp_path, body_only):
    rows = np.asarray([[10+i,20,30+i,90,.9] for i in range(13)], dtype=np.float64)
    row, calls = produce(tmp_path, rows, body_only=body_only)
    assert row.reason == 'person_detection_overflow_not_truncated'
    assert row.people == ()
    value = row.to_dict()
    assert 'raw_person_detector_result' in value, 'semantic cap erased actual detector results'
    result = value['raw_person_detector_result']
    assert result['complete_raw_result_retained']
    assert not value['person_detector_semantics_complete']
    assert result['row_count'] == 13
    assert result['model_name'] == 'nanodet_people'
    assert result['model_content_sha256'] == '1'*64
    assert result['model_adapter_sha256'] == '2'*64
    from crosscut.group_native_detector_result import restore_detector_result, validate_detector_result
    validate_detector_result(result, value)
    restored = restore_detector_result(result)
    assert restored.dtype == rows.dtype
    assert restored.shape == rows.shape
    assert restored.tobytes() == rows.tobytes()
    assert calls.count('nanodet_people') == 1
    assert 'rtmpose_body7' not in calls


@pytest.mark.parametrize('body_only', [False, True])
@pytest.mark.parametrize('rows', [
    np.asarray([[10,20,30,90]], dtype=np.float64),
    np.asarray([[10,20,30,90,float('nan')]], dtype=np.float64),
    np.asarray([[10,20,30,90,1.2]], dtype=np.float64),
    np.asarray([[10,20,5,90,.9]], dtype=np.float64),
])
def test_malformed_numeric_results_remain_exact_but_semantically_unavailable(tmp_path, body_only, rows):
    row, calls = produce(tmp_path, rows, body_only=body_only)
    value = row.to_dict()
    assert row.people == ()
    assert row.reason != 'no_people_detected'
    assert not value['person_detector_semantics_complete']
    from crosscut.group_native_detector_result import restore_detector_result
    restored = restore_detector_result(value['raw_person_detector_result'])
    assert restored.dtype == rows.dtype
    assert restored.tobytes() == rows.tobytes()  # Also preserves nonfinite bit patterns.
    assert 'rtmpose_body7' not in calls


@pytest.mark.parametrize('body_only', [False, True])
@pytest.mark.parametrize('rows', [None, np.empty((0,5), dtype=np.float64)])
def test_unavailable_result_is_not_the_same_as_actual_empty_detection(tmp_path, body_only, rows):
    row, _ = produce(tmp_path, rows, body_only=body_only)
    value = row.to_dict()
    result = value['raw_person_detector_result']
    assert result['complete_raw_result_retained']
    assert (result['row_count'] is None) == (rows is None)
    assert value['person_detector_semantics_complete'] == (rows is not None)
    from crosscut.group_native_detector_result import restore_detector_result
    restored = restore_detector_result(result)
    assert (restored is None) == (rows is None)


def test_oversized_numeric_result_is_explicitly_incomplete_not_truncated_to_semantics(tmp_path):
    rows = np.zeros((14000,5), dtype=np.float64)
    row, calls = produce(tmp_path, rows, body_only=True)
    result = row.to_dict()['raw_person_detector_result']
    assert not result['complete_raw_result_retained']
    assert result['row_count'] == 14000
    assert result['raw_byte_count'] == rows.nbytes
    assert result['raw_bytes_sha256'] == sha256(rows.tobytes()).hexdigest()
    assert not row.to_dict()['person_detector_semantics_complete']
    assert calls == ['nanodet_people']


@pytest.mark.parametrize('change', ['bytes','hash','shape','dtype','source','clock','pixel','model','adapter','authority'])
def test_consumer_rejects_result_binding_or_payload_corruption(tmp_path, change):
    rows = np.asarray([[10+i,20,30+i,90,.9] for i in range(13)], dtype=np.float64)
    row, _ = produce(tmp_path, rows, body_only=True)
    value = row.to_dict()
    result = deepcopy(value['raw_person_detector_result'])
    if change == 'bytes': result['payload']['bytes_hex'] = 'ff'+result['payload']['bytes_hex'][2:]
    elif change == 'hash': result['raw_bytes_sha256'] = '0'*64
    elif change == 'shape': result['payload']['shape'][0] += 1
    elif change == 'dtype': result['payload']['dtype'] = '|O'
    elif change == 'source': result['request_source_index'] += 1
    elif change == 'clock': result['source_clock_identity'] = '3'*64
    elif change == 'pixel': result['native_pixel_sha256'] = '4'*64
    elif change == 'model': result['model_content_sha256'] = '5'*64
    elif change == 'adapter': result['model_adapter_sha256'] = '6'*64
    else: result['quality_approved'] = True
    from crosscut.group_native_detector_result import validate_detector_result
    with pytest.raises(ValueError):
        validate_detector_result(result, value)


@pytest.mark.parametrize('rows', [
    ((10.,20.,30.,90.,float('nan')),),
    ((10,20,30,90,.9),(1,2)),
    [('bad',20,30,90,.9)],
    np.zeros((2,3,5), dtype=np.float32),
])
def test_malformed_sequences_or_rank_are_preserved_without_pose_inference(tmp_path, rows):
    row, calls = produce(tmp_path, rows, body_only=True)
    result = row.to_dict()['raw_person_detector_result']
    from crosscut.group_native_detector_result import restore_detector_result
    restored = restore_detector_result(result)
    assert type(restored) is type(rows)
    assert result['complete_raw_result_retained']
    assert not row.to_dict()['person_detector_semantics_complete']
    assert calls == ['nanodet_people']
    if isinstance(rows, np.ndarray):
        assert restored.dtype == rows.dtype and restored.tobytes() == rows.tobytes()
    else:
        assert len(restored) == len(rows)


def test_result_receipt_cannot_be_erased_from_new_serialized_native_contract(tmp_path):
    row, _ = produce(tmp_path, np.empty((0,5)), body_only=False)
    value = row.to_dict()
    assert value['contract'] == 'actual_source_native_owned_person_body_face_observation_rows_v4'
    from crosscut.group_native_detector_result import validate_serialized_detector_result
    validate_serialized_detector_result(value)
    del value['raw_person_detector_result']
    with pytest.raises(ValueError):
        validate_serialized_detector_result(value)


@pytest.mark.parametrize('claim', [False, True])
def test_detector_complete_flag_is_derived_from_actual_rows_not_a_client_claim(tmp_path, claim):
    rows = np.empty((0,5)) if not claim else np.ones((13,5))
    row, _ = produce(tmp_path, rows, body_only=True)
    value = row.to_dict()
    value['person_detector_semantics_complete'] = claim
    from crosscut.group_native_detector_result import validate_serialized_detector_result
    with pytest.raises(ValueError):
        validate_serialized_detector_result(value)
