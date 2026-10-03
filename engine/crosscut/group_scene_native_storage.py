"""Bounded lossless wire views of NORMALIZED native scene observations.

These are not raw NN tensors or raw adapter returns. Reserving a returned
observation view cannot preserve intermediate model returns inside its producer.
The local append-only records are not controller-owned pre-receipt history,
whole-request phase leases, permanent raster storage or an RSS guarantee.
"""
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json

from .group_joint_primitive_journal import encoded,packed,unpacked,MAX_RECORD_BYTES,primitive_reservation_bytes

CONTRACT='lossless_normalized_scene_native_wire_v1'
RECORD_CONTRACT='local_pre_reserved_normalized_native_view_records_v1'
MAX_NATIVE_JSON_BYTES=512*1024
RAW_CONTRACTS=frozenset(('actual_source_native_owned_person_body_face_observation_rows_v2',
    'actual_source_native_owned_person_body_face_observation_rows_v3',
    'actual_source_native_owned_person_body_face_observation_rows_v4',
    'native_detector_body_support_only_v1'))


def _native_payload(value):
    if (not isinstance(value,dict) or value.get('contract') not in RAW_CONTRACTS
            or type(value.get('request_source_index')) is not int
            or type(value.get('native_frame_index')) is not int
            or min(value['request_source_index'],value['native_frame_index'])<0
            or not isinstance(value.get('people'),(list,tuple)) or len(value['people'])>12
            or not isinstance(value.get('heads'),(list,tuple)) or len(value['heads'])>24
            or any(value.get(k) is not False for k in ('quality_approved',
                'named_member_identity_certified','persistent_subject_correspondence_verified',
                'motion_certified','visibility_certified'))):
        raise ValueError('normalized native wire requires the bounded observation/uncertainty schema')
    data=encoded(value)
    if len(data)>MAX_NATIVE_JSON_BYTES:raise ValueError('normalized native observation exceeded its declared wire bound')
    return data


def pack_native_observation(value):
    if isinstance(value,dict) and value.get('contract')==CONTRACT:
        expand_native_observation(value);return value
    payload=_native_payload(value)
    return dict(contract=CONTRACT,observation_sha256=sha256(payload).hexdigest(),
        lossless_json=packed(payload),normalized_view_not_raw_nn_or_adapter_return=True,
        controller_owned=False,pre_receipt_history_certified=False,quality_approved=False)


def expand_native_observation(value):
    if (not isinstance(value,dict) or set(value)!={'contract','observation_sha256','lossless_json',
                'normalized_view_not_raw_nn_or_adapter_return','controller_owned',
                'pre_receipt_history_certified','quality_approved'} or value['contract']!=CONTRACT
            or value['normalized_view_not_raw_nn_or_adapter_return'] is not True
            or any(value[k] is not False for k in ('controller_owned','pre_receipt_history_certified','quality_approved'))):
        raise ValueError('normalized native wire changed its schema or promoted authority')
    data=unpacked(value['lossless_json'],limit=MAX_NATIVE_JSON_BYTES)
    if sha256(data).hexdigest()!=value['observation_sha256']:
        raise ValueError('normalized native observation wire lost its exact canonical bytes')
    restored=json.loads(data)
    if _native_payload(restored)!=data:
        raise ValueError('normalized native wire did not preserve a canonical observation')
    return restored


class NativeObservationView(Mapping):
    """Read-through native mapping retaining ONLY its lossless wire packet.

    A frame-cache row keeps object identity; it does not retain a second full
    native dictionary. A temporary decoded dictionary can exist during access
    or iteration. This is explicitly not a bound on parser/Python process RSS.
    """
    def __init__(self,packet):
        expand_native_observation(packet)
        self._packet=deepcopy(packet)

    def __getitem__(self,key):return expand_native_observation(self._packet)[key]
    def __iter__(self):return iter(expand_native_observation(self._packet))
    def __len__(self):return len(expand_native_observation(self._packet))
    def items(self):return expand_native_observation(self._packet).items()


def pack_native_evidence(value):
    """Only known native observation schemas; unrelated audit nodes are unchanged."""
    if isinstance(value,NativeObservationView):return deepcopy(value._packet)
    if isinstance(value,Mapping):
        if value.get('contract') in RAW_CONTRACTS:return pack_native_observation(dict(value))
        if value.get('contract')==CONTRACT:
            plain={k:expand_native_evidence(v) for k,v in value.items()}
            expand_native_observation(plain);return plain
        return {k:pack_native_evidence(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(pack_native_evidence(v) for v in value)
    if isinstance(value,list):return [pack_native_evidence(v) for v in value]
    return value


def expand_native_evidence(value):
    if isinstance(value,Mapping):
        plain={k:expand_native_evidence(v) for k,v in value.items()}
        return expand_native_observation(plain) if plain.get('contract')==CONTRACT else plain
    if isinstance(value,(list,tuple)):return [expand_native_evidence(v) for v in value]
    return value


class LocalNormalizedNativeViews:
    def __init__(self,budget):
        self.budget=budget;self.attempts=[];self.views={};self.view_bytes=0;self.record_bytes=0

    def measure(self,*,label,input_binding,producer):
        if (label not in ('native_focal_observation_view','native_composition_observation_view')
                or not isinstance(input_binding,dict) or not callable(producer)):
            raise ValueError('normalized native view requires an explicit bounded producer/input')
        inputs=encoded(input_binding)
        if len(inputs)>2048:raise ValueError('normalized view input binding exceeds terminal capacity')
        row=dict(sequence=len(self.attempts),label=label,input_binding=packed(inputs),
            status='started',observation_sha256=None,normalized_view_retained=False,
            intermediate_nn_raw_returns_retained=False)
        if len(encoded(row))>MAX_RECORD_BYTES:raise ValueError('normalized view BEGIN is unbounded')
        reservation=self.budget.reserve_primitive({'pose_comparisons':0},
            primitive_reservation_bytes(MAX_NATIVE_JSON_BYTES))
        try:self.budget.start_primitive(reservation)
        except BaseException:
            self.budget.settle_primitive(reservation,0);raise
        self.attempts.append(row)  # Before producer/decode/NN calls, not after.
        try:value=producer()
        except BaseException as exc:
            row.update(status='producer_failed',error_type=type(exc).__name__[:128])
            size=len(encoded(row));self.record_bytes+=size
            self.budget.settle_primitive(reservation,size);raise
        try:
            packet=pack_native_observation(value.to_dict())
            payload=unpacked(packet['lossless_json'],limit=MAX_NATIVE_JSON_BYTES)
        except BaseException as exc:
            row.update(status='view_contract_violated',error_type=type(exc).__name__[:128],
                normalized_view_unavailable_after_serialization=True)
            size=len(encoded(row));self.record_bytes+=size
            self.budget.settle_primitive(reservation,size);raise
        ref=packet['observation_sha256'];existing=self.views.get(ref)
        if existing is not None and unpacked(existing['lossless_json'],limit=MAX_NATIVE_JSON_BYTES)!=payload:
            row.update(status='view_contract_violated',error_type='ContentHashCollision')
            size=len(encoded(row));self.record_bytes+=size
            self.budget.settle_primitive(reservation,size)
            raise ValueError('normalized native wire content collision')
        addition=0 if existing is not None else len(encoded(packet))+len(ref)+6
        row.update(status='returned',observation_sha256=ref,normalized_view_retained=True)
        record=len(encoded(row))
        if existing is None:self.views[ref]=packet;self.view_bytes+=addition
        self.record_bytes+=record
        self.budget.settle_primitive(reservation,addition+record)  # No postwork deadline/spend.
        return value

    def to_dict(self):
        # Payloads are in archive.native_rows; do not duplicate them here.
        return dict(contract=RECORD_CONTRACT,attempts=deepcopy(self.attempts),
            view_wire_bytes={ref:len(encoded(packet))+len(ref)+6 for ref,packet in self.views.items()},
            charged_view_bytes=self.view_bytes,charged_record_bytes=self.record_bytes,
            controller_owned=False,pre_receipt_history_certified=False,
            intermediate_nn_raw_returns_retained=False,cache_hit_history_complete=False,
            whole_request_phase_reserves_implemented=False,process_peak_rss_measured=False,
            quality_approved=False)


def validate_normalized_view_records(value,raw_inventory):
    if (not isinstance(value,dict) or value.get('contract')!=RECORD_CONTRACT
            or not isinstance(value.get('attempts'),list) or not isinstance(value.get('view_wire_bytes'),dict)
            or any(value.get(k) is not False for k in ('controller_owned','pre_receipt_history_certified',
                'intermediate_nn_raw_returns_retained','cache_hit_history_complete',
                'whole_request_phase_reserves_implemented','process_peak_rss_measured','quality_approved'))):
        raise ValueError('normalized native local record scope or authority changed')
    referenced=set();records=0
    for index,row in enumerate(value['attempts']):
        if (not isinstance(row,dict) or row.get('sequence')!=index
                or row.get('label') not in ('native_focal_observation_view','native_composition_observation_view')
                or row.get('status') not in ('returned','producer_failed','view_contract_violated')
                or row.get('intermediate_nn_raw_returns_retained') is not False
                or len(encoded(row))>MAX_RECORD_BYTES):
            raise ValueError('normalized native view lost its ordered local BEGIN/terminal')
        binding=json.loads(unpacked(row['input_binding'],limit=2048));ref=row.get('observation_sha256')
        if row['status']=='returned':
            native=raw_inventory.get(ref)
            if (native is None or row.get('normalized_view_retained') is not True
                    or binding.get('request_source_index')!=native['request_source_index']
                    or binding.get('native_frame_index')!=native['native_frame_index']
                    or binding.get('source_clock_identity') not in (None,native['source_clock_identity'])):
                raise ValueError('normalized view record lost its committed native/source occurrence')
            referenced.add(ref)
        elif ref is not None or row.get('normalized_view_retained') is not False:
            raise ValueError('failed normalized view invented a complete returned observation')
        records+=len(encoded(row))
    if (referenced!=set(value['view_wire_bytes']) or value.get('charged_record_bytes')!=records
            or any(value['view_wire_bytes'][ref]!=len(encoded(pack_native_observation(raw_inventory[ref])))+len(ref)+6
                for ref in referenced)
            or value.get('charged_view_bytes')!=sum(value['view_wire_bytes'].values())):
        raise ValueError('normalized native view record changed its received byte/reference inventory')
