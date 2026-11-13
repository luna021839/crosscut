"""Resolve an explicit body-inference backend before opening source evidence.

Auto selects the available Apple GPU once. Runtime inference errors never
silently switch device or reuse the other backend's observation namespace.
"""
import hashlib
import json
import os


def resolve_body_execution_device(requested='auto'):
    if requested == 'auto':
        requested = os.environ.get('IVE_CROSSCUT_BODY_EXECUTION_DEVICE', 'auto')
    if requested not in ('auto', 'cpu', 'mps'):
        raise ValueError('body execution device must be auto, cpu or mps')
    if requested == 'cpu':
        return 'cpu'
    import torch
    available = torch.backends.mps.is_available()
    if requested == 'mps' and not available:
        raise RuntimeError('explicit MPS body execution requires an available Apple GPU')
    return 'mps' if available else 'cpu'


def body_observation_namespace(cpu_namespace, encoder, device):
    if device == 'cpu':
        return cpu_namespace
    if device != 'mps':
        raise ValueError('body observation namespace requires a resolved device')
    return hashlib.sha256(json.dumps({
        'contract': 'device_bound_source_first_observation_v1',
        'cpu_semantics': cpu_namespace,
        'body_execution_device': device,
        'encoder': encoder.fingerprint(),
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
