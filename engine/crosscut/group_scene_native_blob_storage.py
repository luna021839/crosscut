"""Shared lossless NORMALIZED native blobs and exact camera references.

Only local canonical storage is shared. This is not raw NN-result retention,
controller-owned history, a phase lease or a process/IPC memory guarantee.
Public expansion keeps the original observation schema and hash preimages.
"""
from collections.abc import Mapping
from copy import deepcopy

from .group_scene_native_storage import (NativeObservationView, RAW_CONTRACTS,
    CONTRACT as WIRE_CONTRACT, pack_native_observation, expand_native_observation,
    pack_native_evidence, RECORD_CONTRACT)
from .group_joint_primitive_journal import encoded

STORE_KEY = '_scene_normalized_native_blob_store_v1'
STORE_CONTRACT = 'lossless_shared_normalized_scene_native_blob_store_v1'
REFERENCE_CONTRACT = 'normalized_scene_native_observation_reference_v1'
MAX_SHARED_BLOBS = 4096


def native_reference(packet):
    expand_native_observation(packet)
    return _reference_for_sha(packet['observation_sha256'])


def _reference_for_sha(identity):
    return dict(contract=REFERENCE_CONTRACT, observation_sha256=identity,
        normalized_view_not_raw_nn_or_adapter_return=True, controller_owned=False,
        pre_receipt_history_certified=False, quality_approved=False)


class SharedNativeObservationView(NativeObservationView):
    """Read-through reference to a packet ALREADY owned/charged by the session.

    This class does not admit or charge a previously uncommitted observation.
    Packing validates its packet again; sharing is not an immutable-history seal.
    """
    def __init__(self, packet):
        expand_native_observation(packet)
        self._packet = packet


def frame_wire_storage(value, native_views):
    """Charge only new projection/reference bytes when the native blob exists.

    Legacy/explicit substitute producers with no preexisting commit retain
    their full wire charge. Never invent a zero-cost normalized observation.
    """
    stored = pack_native_evidence(value)
    packet = stored['native_observation']
    owned = native_views.views.get(packet['observation_sha256'])
    if owned is None:
        return stored, len(encoded(stored)), NativeObservationView(packet)
    if encoded(owned) != encoded(packet):
        raise ValueError('frame reference differs from its already committed native blob')
    accounted = dict(stored, native_observation=native_reference(owned))
    return stored, len(encoded(accounted)), SharedNativeObservationView(owned)


def pack_shared_native_evidence(value):
    """One blob table per complete artifact; no unresolved standalone refs."""
    from .group_scene_native_storage import expand_native_evidence
    if not isinstance(value, Mapping) or isinstance(value, NativeObservationView):
        raise ValueError('shared native storage needs one complete evidence mapping')
    blobs = {}; occurrences = {}
    def reference(packet):
        ref = packet['observation_sha256']
        existing = blobs.get(ref)
        if existing is not None and encoded(existing) != encoded(packet):
            raise ValueError('shared normalized native blob content collision')
        if existing is None:
            if len(blobs) >= MAX_SHARED_BLOBS:
                raise ValueError('shared normalized native blob inventory exceeded its bound')
            blobs[ref] = deepcopy(packet)
        occurrences[ref] = occurrences.get(ref, 0)+1
        return native_reference(packet)
    def visit(node):
        if isinstance(node, NativeObservationView):
            return reference(node._packet)
        if isinstance(node, Mapping):
            if STORE_KEY in node:
                return visit(expand_native_evidence(node))
            contract = node.get('contract')
            if contract == REFERENCE_CONTRACT:
                raise ValueError('standalone native reference has no owned blob table')
            if contract in RAW_CONTRACTS or contract == WIRE_CONTRACT:
                return reference(pack_native_observation(dict(node)))
            return {key: visit(item) for key, item in node.items()}
        if isinstance(node, tuple): return tuple(visit(item) for item in node)
        if isinstance(node, list): return [visit(item) for item in node]
        return node
    result = visit(value)
    def table(views):
        return dict(contract=STORE_CONTRACT, views=views,
            controller_owned=False, pre_receipt_history_certified=False,
            raw_nn_or_adapter_returns_retained=False, process_peak_rss_measured=False,
            whole_request_phase_reserves_implemented=False, quality_approved=False)
    # A one-use observation must remain inline: a blob key plus a reference is
    # larger than its existing packet. Share only actually repeated content
    # with a positive canonical-byte saving, including table header overhead.
    savings = {ref: (count-1)*len(encoded(blobs[ref]))-
        count*len(encoded(_reference_for_sha(ref)))-len(ref)-6
        for ref, count in occurrences.items()}
    selected = {ref for ref, gain in savings.items() if gain > 0}
    if sum(savings[ref] for ref in selected) <= len(encoded(table({})))+len(STORE_KEY)+6:
        selected = set()
    def materialize(node):
        if isinstance(node, dict):
            if node.get('contract') == REFERENCE_CONTRACT:
                ref = node['observation_sha256']
                return node if ref in selected else deepcopy(blobs[ref])
            return {key: materialize(item) for key, item in node.items()}
        if isinstance(node, tuple): return tuple(materialize(item) for item in node)
        if isinstance(node, list): return [materialize(item) for item in node]
        return node
    result = materialize(result)
    if selected:
        result[STORE_KEY] = table({ref: packet for ref, packet in blobs.items() if ref in selected})
    return result


def expand_shared_native_evidence(value):
    """Validate the complete local table, then reconstruct each exact view."""
    from .group_scene_native_storage import expand_native_evidence
    table = value.get(STORE_KEY)
    if (not isinstance(table, Mapping) or set(table) != {'contract', 'views',
            'controller_owned', 'pre_receipt_history_certified', 'raw_nn_or_adapter_returns_retained',
            'process_peak_rss_measured', 'whole_request_phase_reserves_implemented', 'quality_approved'}
            or table.get('contract') != STORE_CONTRACT
            or not isinstance(table.get('views'), Mapping) or not 1 <= len(table['views']) <= MAX_SHARED_BLOBS
            or any(table.get(key) is not False for key in ('controller_owned',
                'pre_receipt_history_certified', 'raw_nn_or_adapter_returns_retained',
                'process_peak_rss_measured', 'whole_request_phase_reserves_implemented', 'quality_approved'))):
        raise ValueError('shared normalized native table lost its bound or promoted authority')
    restored = {}
    for ref, packet in table['views'].items():
        native = expand_native_observation(dict(packet))
        if ref != packet['observation_sha256']:
            raise ValueError('shared normalized native table changed its content key')
        restored[ref] = native
    used = set()
    def visit(node):
        if isinstance(node, Mapping):
            if STORE_KEY in node:
                return expand_native_evidence(node)
            if node.get('contract') == REFERENCE_CONTRACT:
                ref = node.get('observation_sha256')
                packet = table['views'].get(ref) if isinstance(ref, str) else None
                if packet is None or dict(node) != _reference_for_sha(ref):
                    raise ValueError('native reference lost its committed blob or promoted authority')
                used.add(ref)
                return deepcopy(restored[ref])
            # Inline legacy observations and unrelated audit nodes retain the
            # original expander contract; no borrowed table crosses nesting.
            plain = {key: visit(item) for key, item in node.items()}
            return expand_native_observation(plain) if plain.get('contract') == WIRE_CONTRACT else plain
        if isinstance(node, (list, tuple)): return [visit(item) for item in node]
        return node
    result = {key: visit(item) for key, item in value.items() if key != STORE_KEY}
    if used != set(restored):
        raise ValueError('shared normalized native table contains omitted or unreferenced blobs')
    return result


def pack_owned_native_evidence(value):
    """Use shared storage for committed scene evidence, retain legacy formats."""
    def shared(node):
        if isinstance(node, SharedNativeObservationView): return True
        if isinstance(node, Mapping):
            if STORE_KEY in node: return True
            if node.get('contract') == RECORD_CONTRACT and node.get('attempts'): return True
            return any(shared(item) for item in node.values())
        if isinstance(node, (tuple, list)): return any(shared(item) for item in node)
        return False
    return pack_shared_native_evidence(value) if shared(value) else pack_native_evidence(value)
