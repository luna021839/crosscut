"""Pre-reserved local numeric primitive returns, before semantic validation.

This worker-owned prefix is NOT the controller's pre-receipt history, an RSS
bound, a whole-request phase lease, or a permanent lossless decoded-raster
archive. Numeric return bits are stored once; each execution/record still costs
work and bytes. Parent/final must execute fresh primitives, not borrow blobs.
"""
import base64
from copy import deepcopy
from hashlib import sha256
import json
import math
import struct
import zlib

import numpy as np

CONTRACT='local_pre_reserved_numeric_primitive_prefix_v1'
MAX_RESULT_BYTES=512*1024
MAX_RECORD_BYTES=4096


def primitive_reservation_bytes(maximum_result_bytes):
    if type(maximum_result_bytes) is not int or not 1<=maximum_result_bytes<=MAX_RESULT_BYTES:
        raise ValueError('primitive result reservation needs its bounded result contract')
    return 2*maximum_result_bytes+512+128+MAX_RECORD_BYTES


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def packed(data):
    return dict(encoding='zlib_lossless_canonical_json_v1',decoded_byte_count=len(data),
        bytes_base64=base64.b64encode(zlib.compress(data)).decode('ascii'))


def unpacked(value,*,limit=MAX_RESULT_BYTES):
    if (not isinstance(value,dict) or set(value)!={'encoding','decoded_byte_count','bytes_base64'}
            or value['encoding']!='zlib_lossless_canonical_json_v1'
            or type(value['decoded_byte_count']) is not int or not 0<=value['decoded_byte_count']<=limit
            or not isinstance(value['bytes_base64'],str) or len(value['bytes_base64'])>2*limit+512):
        raise ValueError('primitive lossless storage envelope changed')
    try:
        data=base64.b64decode(value['bytes_base64'],validate=True)
        decoder=zlib.decompressobj()
        restored=decoder.decompress(data,value['decoded_byte_count']+1)
        if (len(restored)!=value['decoded_byte_count'] or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError('primitive storage lost bytes or exceeded declared decode size')
        return restored
    except (ValueError,zlib.error) as exc:raise ValueError('primitive lossless bytes cannot replay') from exc


def primitive_inputs(row):
    value=json.loads(unpacked(row['input_binding'],limit=2048))
    if not isinstance(value,dict):raise ValueError('primitive input binding is not a mapping')
    return value


def primitive_measurement_ref(label,input_binding,raw_result_sha256):
    """Stable result occurrence, NOT execution identity or history authority."""
    return sha256(encoded(dict(label=label,input_binding_sha256=sha256(encoded(input_binding)).hexdigest(),
        raw_result_sha256=raw_result_sha256))).hexdigest()


def numeric_return(value,*,depth=0):
    if depth>8:raise ValueError('primitive numeric result exceeded declared nesting')
    if value is None:return {'kind':'none'}
    if isinstance(value,np.ndarray):
        if (value.dtype.hasobject or value.dtype.fields is not None or value.dtype.subdtype is not None
                or value.ndim>8 or value.nbytes>MAX_RESULT_BYTES):
            raise ValueError('primitive array exceeded its numeric result contract')
        return dict(kind='ndarray_c_order_bits',dtype=value.dtype.str,shape=list(value.shape),
            bytes_base64=base64.b64encode(value.tobytes(order='C')).decode('ascii'))
    if type(value) in (tuple,list):
        if len(value)>1024:raise ValueError('primitive return sequence is unbounded')
        return dict(kind=type(value).__name__,items=[numeric_return(v,depth=depth+1) for v in value])
    if type(value) is float:return dict(kind='float64_bits',bytes_hex=struct.pack('>d',value).hex())
    if type(value) is int:return dict(kind='int',value=str(value))
    if type(value) is bool:return dict(kind='bool',value=value)
    raise ValueError('primitive has no declared lossless numeric return format')


def restore_numeric_return(value,*,depth=0):
    """Safe numeric replay only. No pickle, model execution or imported class."""
    if depth==0 and isinstance(value,dict) and value.get('encoding')=='zlib_lossless_canonical_json_v1':
        value=json.loads(unpacked(value))
    if not isinstance(value,dict) or depth>8:raise ValueError('invalid primitive numeric blob')
    kind=value.get('kind')
    if kind=='none' and set(value)=={'kind'}:return None
    if kind=='ndarray_c_order_bits' and set(value)=={'kind','dtype','shape','bytes_base64'}:
        try:
            dtype=np.dtype(value['dtype']);shape=value['shape']
            if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
                    or not isinstance(shape,list) or len(shape)>8
                    or any(type(n) is not int or n<0 for n in shape)
                    or math.prod(shape)*dtype.itemsize>MAX_RESULT_BYTES
                    or not isinstance(value['bytes_base64'],str)
                    or len(value['bytes_base64'])>4*((MAX_RESULT_BYTES+2)//3)):
                raise ValueError('invalid bounded primitive array shape/dtype')
            data=base64.b64decode(value['bytes_base64'],validate=True)
            if len(data)!=math.prod(shape)*dtype.itemsize:raise ValueError('primitive array lost bytes')
            return np.frombuffer(data,dtype=dtype).reshape(shape).copy()
        except (ValueError,TypeError,KeyError) as exc:raise ValueError('primitive array cannot replay') from exc
    if kind in ('tuple','list') and set(value)=={'kind','items'}:
        items=value['items']
        if not isinstance(items,list) or len(items)>1024:raise ValueError('unbounded primitive sequence')
        restored=[restore_numeric_return(v,depth=depth+1) for v in items]
        return tuple(restored) if kind=='tuple' else restored
    if kind=='float64_bits' and set(value)=={'kind','bytes_hex'}:
        try:return struct.unpack('>d',bytes.fromhex(value['bytes_hex']))[0]
        except (ValueError,TypeError,struct.error) as exc:raise ValueError('invalid primitive float bits') from exc
    if kind=='bool' and set(value)=={'kind','value'} and type(value['value']) is bool:return value['value']
    if kind=='int' and set(value)=={'kind','value'} and type(value['value']) is str and len(value['value'])<=128:
        try:
            number=int(value['value'])
            if str(number)==value['value']:return number
        except ValueError:pass
    raise ValueError('primitive numeric blob schema changed')


class LocalPrimitiveJournal:
    def __init__(self,budget):
        self.budget=budget;self.attempts=[];self.blobs={};self.blob_bytes=0;self.record_bytes=0

    def measure(self,*,label,input_binding,work,maximum_result_bytes,producer):
        if (not isinstance(label,str) or not 0<len(label)<=128 or not callable(producer)
                or type(maximum_result_bytes) is not int or not 1<=maximum_result_bytes<=MAX_RESULT_BYTES
                or not isinstance(input_binding,dict)):
            raise ValueError('numeric primitive needs bounded named inputs/result contract')
        # Preserve the actual input binding before executing the callback.
        inputs=encoded(input_binding)
        if len(inputs)>2048:raise ValueError('primitive input binding exceeds its reserved record contract')
        row=dict(sequence=len(self.attempts),label=label,input_binding=packed(inputs),
            work=dict(work),maximum_result_bytes=maximum_result_bytes,status='started',
            raw_result_sha256=None,raw_result_retained=False,measurement_ref_sha256=None)
        if len(encoded(row))>MAX_RECORD_BYTES:raise ValueError('primitive BEGIN is unbounded')
        # Bound also the lossless envelope/Base64 expansion. Actual charged
        # bytes are the compact stored wire bytes, not a fictitious RSS bound.
        reservation=self.budget.reserve_primitive(dict(work),primitive_reservation_bytes(maximum_result_bytes))
        try:self.budget.start_primitive(reservation)
        except BaseException:
            self.budget.settle_primitive(reservation,0);raise
        self.attempts.append(row)  # BEFORE the expensive primitive.
        try:
            value=producer()
        except BaseException as exc:
            row.update(status='backend_failed',error_type=type(exc).__name__[:128],
                error_message_retained=False)
            count=len(encoded(row));self.record_bytes+=count
            self.budget.settle_primitive(reservation,count)
            raise
        try:
            payload=encoded(numeric_return(value))
            if len(payload)>maximum_result_bytes:
                raise ValueError('returned primitive exceeded its predeclared byte bound')
        except BaseException as exc:
            row.update(status='return_contract_violated',error_type=type(exc).__name__[:128],
                result_unavailable_after_serialization=True)
            count=len(encoded(row));self.record_bytes+=count
            self.budget.settle_primitive(reservation,count)
            raise
        ref=sha256(payload).hexdigest();stored=self.blobs.get(ref)
        compact=encoded(packed(payload))
        if stored is not None and unpacked(json.loads(stored))!=payload:
            row.update(status='return_contract_violated',error_type='ContentHashCollision',
                result_unavailable_after_serialization=True)
            count=len(encoded(row));self.record_bytes+=count
            self.budget.settle_primitive(reservation,count)
            raise ValueError('primitive content hash collision')
        new_bytes=0 if stored is not None else len(compact)+len(ref)+6
        row.update(status='returned',raw_result_sha256=ref,raw_result_retained=True,
            measurement_ref_sha256=primitive_measurement_ref(label,json.loads(inputs),ref))
        count=len(encoded(row))
        if stored is None:self.blobs[ref]=compact;self.blob_bytes+=new_bytes
        self.record_bytes+=count
        # No deadline check or new spend AFTER execution. Raw bits are now
        # retained even if the next direction/step cannot be admitted.
        self.budget.settle_primitive(reservation,new_bytes+count)
        return value

    def to_dict(self):
        return dict(contract=CONTRACT,attempts=deepcopy(self.attempts),
            blobs={key:json.loads(value) for key,value in self.blobs.items()},
            charged_blob_bytes=self.blob_bytes,charged_record_bytes=self.record_bytes,
            controller_owned=False,pre_receipt_history_completeness_certified=False,
            all_observation_calls_journaled=False,cache_hit_history_complete=False,
            whole_request_phase_reserves_implemented=False,decoded_raster_copies_retained=False,
            process_peak_rss_measured=False,quality_approved=False)


def validate_journal(value):
    """Received reference integrity, explicitly NOT authenticated prior history."""
    if (not isinstance(value,dict) or value.get('contract')!=CONTRACT
            or any(value.get(k) is not False for k in ('controller_owned',
                'pre_receipt_history_completeness_certified','whole_request_phase_reserves_implemented',
                'all_observation_calls_journaled','cache_hit_history_complete',
                'decoded_raster_copies_retained','process_peak_rss_measured','quality_approved'))
            or not isinstance(value.get('attempts'),list) or not isinstance(value.get('blobs'),dict)):
        raise ValueError('primitive journal schema or authority changed')
    referenced=set();record_bytes=0
    for index,row in enumerate(value['attempts']):
        if (not isinstance(row,dict) or row.get('sequence')!=index
                or row.get('status') not in ('returned','backend_failed','return_contract_violated')
                or not isinstance(row.get('input_binding'),dict)
                or not isinstance(row.get('label'),str) or not 0<len(row['label'])<=128
                or not isinstance(row.get('work'),dict) or not row['work']
                or any(type(k) is not str or type(n) is not int or n<0 for k,n in row['work'].items())
                or type(row.get('maximum_result_bytes')) is not int
                or not 1<=row['maximum_result_bytes']<=MAX_RESULT_BYTES
                or len(encoded(row))>MAX_RECORD_BYTES):
            raise ValueError('primitive journal lost ordered BEGIN/terminal records')
        record_bytes+=len(encoded(row));ref=row.get('raw_result_sha256')
        inputs=primitive_inputs(row)
        if row['status']=='returned':
            payload=value['blobs'].get(ref)
            data=None if payload is None else unpacked(payload)
            if (data is None or sha256(data).hexdigest()!=ref
                    or row.get('raw_result_retained') is not True
                    or len(data)>row['maximum_result_bytes']
                    or row.get('measurement_ref_sha256')!=primitive_measurement_ref(row['label'],inputs,ref)):
                raise ValueError('primitive journal omitted or changed a raw result blob')
            restore_numeric_return(payload);referenced.add(ref)
        elif (ref is not None or row.get('raw_result_retained') is not False
                or row.get('measurement_ref_sha256') is not None):
            raise ValueError('failed primitive invented a complete returned result')
    if (referenced!=set(value['blobs']) or value.get('charged_record_bytes')!=record_bytes
            or value.get('charged_blob_bytes')!=sum(len(encoded(payload))+len(key)+6
                for key,payload in value['blobs'].items())):
        raise ValueError('primitive journal changed its reference or byte inventory')
