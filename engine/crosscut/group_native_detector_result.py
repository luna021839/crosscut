"""Bounded, lossless returned detector results before semantic person caps.

These are adapter RETURNS, not network tensors, anatomical masks or identities.
Numeric ndarray bits retain nonfinite malformed results too. Oversized/object
results are explicitly not fully retained, never truncated into valid people.
This received-row record is not a parent-owned append-only inference journal.
"""
from hashlib import sha256
import json
import math
import struct

import numpy as np

CONTRACT = 'returned_native_person_detector_result_v1'
MAX_RAW_RESULT_BYTES = 256*1024
MAX_SEQUENCE_NODES = 4096


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sequence_node(value, count, depth=0):
    count[0] += 1
    if count[0] > MAX_SEQUENCE_NODES or depth > 8:
        raise ValueError('detector sequence exceeds bounded raw-result serialization')
    if type(value) in (tuple, list):
        return dict(type=type(value).__name__, items=[_sequence_node(v, count, depth+1) for v in value])
    if type(value) is float:
        return dict(type='float64', bits=struct.pack('>d', value).hex())
    if type(value) is int:
        return dict(type='int', value=str(value))
    if type(value) is bool:
        return dict(type='bool', value=value)
    if value is None:
        return dict(type='none')
    if type(value) is str and len(value) <= 128:
        return dict(type='str', value=value)
    raise ValueError('detector sequence has an unsupported raw scalar type')


def _restore_node(node, count, depth=0):
    count[0] += 1
    if not isinstance(node, dict) or count[0] > MAX_SEQUENCE_NODES or depth > 8:
        raise ValueError('detector raw sequence is malformed or unbounded')
    kind = node.get('type')
    if kind in ('tuple', 'list'):
        if set(node) != {'type','items'} or not isinstance(node['items'], list):
            raise ValueError('detector raw sequence lost its container')
        values = [_restore_node(v, count, depth+1) for v in node['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'none' and set(node) == {'type'}:
        return None
    if kind == 'float64' and set(node) == {'type','bits'}:
        try:
            return struct.unpack('>d', bytes.fromhex(node['bits']))[0]
        except (ValueError, TypeError, struct.error) as exc:
            raise ValueError('detector raw float bits changed') from exc
    if kind == 'int' and set(node) == {'type','value'}:
        try:
            value = int(node['value'])
        except (ValueError, TypeError) as exc:
            raise ValueError('detector raw integer changed') from exc
        if str(value) == node['value']:
            return value
    if kind == 'bool' and set(node) == {'type','value'} and type(node['value']) is bool:
        return node['value']
    if kind == 'str' and set(node) == {'type','value'} and type(node['value']) is str and len(node['value']) <= 128:
        return node['value']
    raise ValueError('detector raw scalar/container contract changed')


def capture_detector_result(rows, *, binding, model_content_sha256, model_adapter_sha256):
    payload = None; complete = True; raw_hash = None; byte_count = 0
    count = None if rows is None else len(rows) if isinstance(rows, (tuple, list)) or (
        isinstance(rows, np.ndarray) and rows.ndim) else None
    descriptor = dict(python_type=type(rows).__name__)
    if isinstance(rows, np.ndarray):
        descriptor.update(dtype=rows.dtype.str, shape=list(rows.shape))
        byte_count = rows.nbytes
        if not rows.dtype.hasobject:
            # Hash actual C-order bytes even when too large to retain. A hash
            # and shape are NOT represented as complete raw-result retention.
            data = rows.tobytes(order='C'); raw_hash = sha256(data).hexdigest()
            complete = byte_count <= MAX_RAW_RESULT_BYTES and rows.ndim <= 4
            if complete:
                payload = dict(encoding='ndarray_c_order_bits_v1', dtype=rows.dtype.str,
                    shape=list(rows.shape), bytes_hex=data.hex())
        else:
            complete = False  # Never serialize/replay object pointers or pickle.
    elif rows is not None:
        try:
            payload = dict(encoding='typed_python_sequence_v1', value=_sequence_node(rows, [0]))
            data = _encoded(payload); byte_count = len(data); raw_hash = sha256(data).hexdigest()
            if byte_count > MAX_RAW_RESULT_BYTES:
                complete = False; payload = None
        except ValueError:
            complete = False; payload = None
    result = dict(contract=CONTRACT, **binding, model_name='nanodet_people',
        model_content_sha256=model_content_sha256, model_adapter_sha256=model_adapter_sha256,
        row_count=count, raw_byte_count=byte_count, raw_bytes_sha256=raw_hash,
        result_descriptor=descriptor, payload=payload,
        payload_sha256=sha256(_encoded(payload)).hexdigest(),
        complete_raw_result_retained=complete, result_was_none=rows is None,
        pre_receipt_history_completeness_certified=False, quality_approved=False)
    return result


def restore_detector_result(result):
    if not isinstance(result, dict) or result.get('complete_raw_result_retained') is not True:
        raise ValueError('detector raw result was not completely retained')
    payload = result.get('payload')
    if result.get('payload_sha256') != sha256(_encoded(payload)).hexdigest():
        raise ValueError('detector raw payload hash changed')
    if result.get('result_was_none') is True:
        if (payload is not None or result.get('row_count') is not None or result.get('raw_byte_count') != 0
                or result.get('raw_bytes_sha256') is not None):
            raise ValueError('unavailable detector result was relabelled empty')
        return None
    if not isinstance(payload, dict):
        raise ValueError('detector returned result has no raw payload')
    if payload.get('encoding') == 'ndarray_c_order_bits_v1':
        try:
            dtype = np.dtype(payload['dtype']); shape = payload['shape']
            if (dtype.hasobject or dtype.itemsize < 1 or not isinstance(shape, list) or len(shape) > 4
                    or any(type(n) is not int or n < 0 for n in shape)
                    or math.prod(shape)*dtype.itemsize > MAX_RAW_RESULT_BYTES
                    or not isinstance(payload['bytes_hex'], str)
                    or len(payload['bytes_hex']) > MAX_RAW_RESULT_BYTES*2):
                raise ValueError('unsupported/unbounded detector raw array')
            data = bytes.fromhex(payload['bytes_hex'])
            if len(data) != math.prod(shape)*dtype.itemsize:
                raise ValueError('detector raw shape differs from its actual bytes')
            value = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError('detector raw array payload changed') from exc
    elif payload.get('encoding') == 'typed_python_sequence_v1':
        data = _encoded(payload)
        if len(data) > MAX_RAW_RESULT_BYTES:
            raise ValueError('detector raw sequence exceeds its byte limit')
        value = _restore_node(payload.get('value'), [0])
    else:
        raise ValueError('detector raw result encoding changed')
    count = len(value) if isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.ndim) else None
    descriptor = result.get('result_descriptor')
    if (not isinstance(descriptor, dict) or descriptor.get('python_type') != type(value).__name__
            or isinstance(value, np.ndarray) and (descriptor.get('dtype') != value.dtype.str
                or descriptor.get('shape') != list(value.shape))):
        raise ValueError('detector raw descriptor differs from its retained original result')
    if (len(data) != result.get('raw_byte_count') or sha256(data).hexdigest() != result.get('raw_bytes_sha256')
            or count != result.get('row_count')):
        raise ValueError('detector raw result byte/count inventory changed')
    return value


def bounded_detector_rows(result):
    """Valid numeric row shape only; ROI/visibility/ownership remains the caller's."""
    value = restore_detector_result(result)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 5 or value.dtype.kind not in 'iuf':
            raise ValueError('invalid native body detector row')
    elif not isinstance(value, (list, tuple)):
        raise ValueError('invalid native body detector row')
    if len(value) > 12:
        raise ValueError('person detection overflow is not a bounded semantic view')
    rows = []
    for row in value:
        if (not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 5
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(v)) for v in row)):
            raise ValueError('invalid native body detector row')
        rows.append(tuple(float(v) for v in row))
    return tuple(rows)


def detector_semantics_complete(result, native):
    try:
        rows = bounded_detector_rows(result)
        people = native['people']; width, height = native['frame_size_wh']
        if rows is None or min(width, height) < 1 or len(rows) != len(people):
            return False
        for index, (row, person) in enumerate(zip(rows, people, strict=True)):
            left, top, right, bottom, confidence = row
            box = left/width, top/height, (right-left)/width, (bottom-top)/height
            if (min(box[2:]) <= 0 or min(box[:2]) < 0 or box[0]+box[2] > 1.000001
                    or box[1]+box[3] > 1.000001 or not 0 <= confidence <= 1
                    or person['observation_id'] != index or person['confidence'] != confidence
                    or tuple(person['box_xywh']) != box):
                return False
        return True
    except (ValueError, KeyError, TypeError):
        return False


def validate_detector_result(result, native):
    is_sha = lambda v:isinstance(v,str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
    if (not isinstance(result, dict) or result.get('contract') != CONTRACT
            or result.get('model_name') != 'nanodet_people'
            or type(result.get('complete_raw_result_retained')) is not bool
            or type(result.get('result_was_none')) is not bool
            or any(not is_sha(result.get(k)) for k in ('model_content_sha256', 'model_adapter_sha256',
                'source_clock_identity', 'native_pixel_sha256', 'payload_sha256'))
            or any(type(result.get(k)) is not int or result[k] < 0 for k in ('request_source_index', 'native_frame_index'))
            or type(result.get('raw_byte_count')) is not int or result['raw_byte_count'] < 0
            or result.get('row_count') is not None and (type(result['row_count']) is not int or result['row_count'] < 0)
            or any(result.get(k) is not False for k in ('quality_approved', 'pre_receipt_history_completeness_certified'))
            or any(result.get(k) != native.get(k) for k in ('request_source_index', 'native_frame_index',
                'source_clock_identity', 'native_pixel_sha256'))
            or list(result.get('source_file_identity', ())) != list(native.get('source_file_identity', ()))
            or result.get('model_content_sha256') != dict(native.get('model_content_sha256', ())).get('nanodet_people')
            or result.get('model_adapter_sha256') != dict(native.get('model_adapter_fingerprints', ())).get('nanodet_people')):
        raise ValueError('detector raw result differs from its native source/model/authority binding')
    if result['complete_raw_result_retained']:
        restore_detector_result(result)
    elif (result.get('payload') is not None or result.get('result_was_none') is not False
            or result['payload_sha256'] != sha256(_encoded(None)).hexdigest()):
        raise ValueError('incomplete detector raw result acquired available-payload authority')
    if ('person_detector_semantics_complete' in native and
            native['person_detector_semantics_complete'] is not detector_semantics_complete(result, native)):
        raise ValueError('detector semantic view does not retain its actual complete raw row set')


def validate_serialized_detector_result(native):
    present = 'raw_person_detector_result' in native
    if native.get('contract') == 'actual_source_native_owned_person_body_face_observation_rows_v4' and not present:
        raise ValueError('v4 native observation lost its returned detector result')
    if present:
        if type(native.get('person_detector_semantics_complete')) is not bool:
            raise ValueError('native detector semantic completeness is missing')
        validate_detector_result(native['raw_person_detector_result'], native)
