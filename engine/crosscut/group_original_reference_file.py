"""Content-bound, complete GROUP originals for normal saved/app requests.

This is local renderer/metadata transport, not a quality certificate. A
numeric plan alone cannot become a complete original. Wire type names select
only types reachable from the installed bundle's annotations; they never
import a module, execute code, or select an arbitrary Python constructor.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import get_args, get_type_hints
import types
from typing import Any, Literal, Union

from .group_external_original_reference import GroupExternalOriginalReference

CONTRACT = 'complete_group_external_original_file_v1'
MAX_REFERENCE_BYTES = 32 * 1024 * 1024
MAX_STATE_NODES = 1_000_000
MAX_STATE_DEPTH = 64


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@lru_cache(maxsize=1)
def _types():
    from .pipeline import _GroupRenderCandidateBundle
    from .group_joint_bundle_metadata import GroupNativeMotionProduction
    from .group_source_domain import GroupSourceDomain
    from .group_joint_source_camera import GroupOutputRole
    from .shot_video_clock import FrozenShotVideoClockContext
    from .render import RenderPlan
    from .visual_score import VisualTimelinePlan
    aliases = dict(GroupNativeMotionProduction=GroupNativeMotionProduction,
        GroupSourceDomain=GroupSourceDomain, GroupOutputRole=GroupOutputRole,
        FrozenShotVideoClockContext=FrozenShotVideoClockContext,
        RenderPlan=RenderPlan, VisualTimelinePlan=VisualTimelinePlan)
    registry = {}
    visited = set()

    def visit(annotation):
        if annotation in visited:
            return
        visited.add(annotation)
        for part in get_args(annotation):
            visit(part)
        if not isinstance(annotation, type):
            return
        if not (is_dataclass(annotation) or issubclass(annotation, Enum)):
            return
        if not annotation.__module__.startswith('crosscut.'):
            raise ValueError('original bundle references a non-project state type')
        name = annotation.__module__ + '.' + annotation.__qualname__
        registry[name] = annotation
        if is_dataclass(annotation):
            for child in get_type_hints(annotation, localns=aliases).values():
                visit(child)

    visit(_GroupRenderCandidateBundle)
    visit(GroupSourceDomain)
    return registry


def _matches_type(value, annotation):
    """Check installed field declarations; registered tags are not enough."""
    if annotation in (Any, object):
        return True
    origin = getattr(annotation, '__origin__', None)
    args = get_args(annotation)
    if origin in (Union, types.UnionType) or isinstance(annotation, types.UnionType):
        return any(_matches_type(value, option) for option in args)
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in args)
    if annotation is type(None):
        return value is None
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return type(value) in (int, float)
    if origin in (tuple, list):
        if not isinstance(value, origin):
            return False
        if not args:
            return True
        if origin is list or len(args) == 2 and args[1] is Ellipsis:
            return all(_matches_type(item, args[0]) for item in value)
        return len(args) == len(value) and all(_matches_type(item, expected)
            for item, expected in zip(value, args, strict=True))
    if origin is not None and isinstance(origin, type) and issubclass(origin, Mapping):
        return isinstance(value, Mapping) and (not args or all(
            _matches_type(key, args[0]) and _matches_type(item, args[1]) for key, item in value.items()))
    return isinstance(annotation, type) and isinstance(value, annotation)


@lru_cache(maxsize=None)
def _field_types(cls):
    aliases = {kind.__name__: kind for kind in _types().values()}
    return get_type_hints(cls, localns=aliases)


def _state_codec(*, decode=False):
    registry = _types()
    remaining = MAX_STATE_NODES

    def transform(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_STATE_DEPTH:
            raise ValueError('complete original state exceeds its finite transport bounds')
        child = lambda item: transform(item, depth + 1)
        if decode:
            if value is None or type(value) in (str, bool, int, float):
                if type(value) is float and not math.isfinite(value):
                    raise ValueError('original state contains a nonfinite number')
                return value
            if not isinstance(value, dict):
                raise ValueError('original state is not a tagged installed-type value')
            if set(value) == {'$path'} and isinstance(value['$path'], str):
                return Path(value['$path'])
            if set(value) == {'$fraction'}:
                pair = value['$fraction']
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(item) is not int for item in pair) or pair[1] <= 0):
                    raise ValueError('original fraction is malformed')
                return Fraction(*pair)
            if set(value) == {'$enum', 'value'}:
                cls = registry.get(value['$enum'])
                if cls is None or not issubclass(cls, Enum):
                    raise ValueError('unregistered original enum type')
                return cls(child(value['value']))
            if set(value) == {'$type', 'fields'}:
                cls = registry.get(value['$type'])
                if cls is None or not is_dataclass(cls) or not isinstance(value['fields'], dict):
                    raise ValueError('unregistered original dataclass type')
                names = {field.name for field in fields(cls) if field.init}
                if set(value['fields']) != names:
                    raise ValueError('complete original dataclass lost or added state fields')
                try:
                    arguments = {name: child(item) for name, item in value['fields'].items()}
                    declared = _field_types(cls)
                    if any(not _matches_type(item, declared[name]) for name, item in arguments.items()):
                        raise ValueError('complete original field differs from its installed type')
                    return cls(**arguments)
                except (TypeError, KeyError, AttributeError, OverflowError) as exc:
                    raise ValueError('complete original dataclass is malformed') from exc
            if len(value) == 1:
                key, items = next(iter(value.items()))
                if key in ('$tuple', '$list') and isinstance(items, list):
                    result = [child(item) for item in items]
                    return tuple(result) if key == '$tuple' else result
                if key == '$map' and isinstance(items, list):
                    result = {}
                    for pair in items:
                        if not isinstance(pair, list) or len(pair) != 2:
                            raise ValueError('original mapping entry is malformed')
                        name, item = child(pair[0]), child(pair[1])
                        try:
                            if name in result:
                                raise ValueError('original mapping has a duplicate key')
                            result[name] = item
                        except TypeError as exc:
                            raise ValueError('original mapping key is not immutable') from exc
                    return result
            raise ValueError('unsupported original state tag')
        if isinstance(value, Enum):
            name = type(value).__module__ + '.' + type(value).__qualname__
            if registry.get(name) is not type(value):
                raise ValueError('unregistered original enum type')
            return {'$enum': name, 'value': child(value.value)}
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not math.isfinite(value):
                raise ValueError('original state contains a nonfinite number')
            return value
        if isinstance(value, Path):
            return {'$path': str(value)}
        if isinstance(value, Fraction):
            return {'$fraction': [value.numerator, value.denominator]}
        if is_dataclass(value) and not isinstance(value, type):
            name = type(value).__module__ + '.' + type(value).__qualname__
            if registry.get(name) is not type(value):
                raise ValueError('unregistered original dataclass type')
            return {'$type': name, 'fields': {field.name: child(getattr(value, field.name))
                for field in fields(value) if field.init}}
        if isinstance(value, Mapping):
            pairs = [[child(name), child(item)] for name, item in value.items()]
            pairs.sort(key=lambda pair: _canonical(pair[0]))
            return {'$map': pairs}
        if isinstance(value, (tuple, list)):
            return {'$tuple' if isinstance(value, tuple) else '$list': [child(item) for item in value]}
        # In particular, a private live-owner seal is not a serializable bool.
        # Decline it explicitly rather than removing executable authority.
        raise ValueError('original contains nontransportable live state; do not erase its owner')

    return transform


def original_reference_file_payload(reference, *, song):
    """Freeze every typed field and independently rederive its policy."""
    if not isinstance(reference, GroupExternalOriginalReference) or not isinstance(song, str) or not song.strip():
        raise ValueError('complete original file needs its typed reference and song')
    reference.validate_original()
    state = _state_codec()(reference.original_bundle)
    restored = _state_codec(decode=True)(state)
    check = GroupExternalOriginalReference(restored, reference.source_domain,
        reference.protected_output_intervals)
    if (_canonical(_state_codec()(restored)) != _canonical(state)
            or _canonical(check.to_dict()) != _canonical(reference.to_dict())):
        raise ValueError('original snapshot round-trip changed complete metadata or renderer authority')
    return dict(contract=CONTRACT, group='IVE', song=song.strip(),
        complete_bundle_state=state, complete_source_domain_state=_state_codec()(reference.source_domain),
        external_original_policy=reference.to_dict(),
        professional_quality_approved=False)


def write_original_reference_file(reference, path, *, song):
    """Exclusive local export. Existing originals are never overwritten."""
    payload = original_reference_file_payload(reference, song=song)
    data = (_canonical(payload) + '\n').encode('utf-8')
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError('complete original file exceeds its declared byte bound')
    path = Path(path)
    with path.open('xb') as handle:
        handle.write(data)
    return sha256(data).hexdigest()


def load_original_reference_file(path, *, expected_sha256, song, request_sources):
    """Restore the pinned complete original after automatic materialization."""
    from .pipeline import _GroupRenderCandidateBundle
    from .group_source_domain import GroupSourceDomain
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('complete original needs its explicit lowercase content SHA-256')
    path = Path(path)
    if path.is_symlink():
        raise ValueError('complete original file cannot be a symbolic link')
    with path.open('rb') as handle:
        data = handle.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES or sha256(data).hexdigest() != expected_sha256:
        raise ValueError('complete original file content changed or exceeded its byte bound')

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError('complete original JSON has a duplicate field')
            result[name] = value
        return result

    try:
        payload = json.loads(data, object_pairs_hook=unique_object)
        if (not isinstance(payload, dict) or set(payload) != {'contract', 'group', 'song',
                'complete_bundle_state', 'complete_source_domain_state',
                'external_original_policy', 'professional_quality_approved'}
                or payload['contract'] != CONTRACT or payload['group'] != 'IVE'
                or payload['song'] != song.strip() or payload['professional_quality_approved'] is not False):
            raise ValueError('complete original file has a different song or transport contract')
        bundle = _state_codec(decode=True)(payload['complete_bundle_state'])
        if (not isinstance(bundle, _GroupRenderCandidateBundle)
                or bundle.render_plan.sources != tuple(request_sources)):
            raise ValueError('complete original sources differ from the materialized request order')
        policy = payload['external_original_policy']
        # Numeric replay deliberately uses zero offsets. Normal authoring may
        # not borrow that adapter: retain the actual audio offsets/time warps.
        domain = _state_codec(decode=True)(payload['complete_source_domain_state'])
        if not isinstance(domain, GroupSourceDomain) or domain.request_sources != bundle.render_plan.sources:
            raise ValueError('complete original lost its actual audio/native source domain')
        reference = GroupExternalOriginalReference(bundle, domain,
            tuple(tuple(span) for span in policy['protected_output_intervals']))
        if (_canonical(reference.to_dict()) != _canonical(policy)
                or _canonical(_state_codec()(bundle)) != _canonical(payload['complete_bundle_state'])
                or _canonical(_state_codec()(domain)) != _canonical(payload['complete_source_domain_state'])):
            raise ValueError('complete original policy, source identity or full metadata changed')
        reference.validate_baseline(bundle, bundle.render_plan)
        return reference
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError('complete original file is malformed or incomplete') from exc


def resolve_request_original_reference(request, explicit=None):
    """One normal CLI/app/Python handoff; no parallel reference authority."""
    from .ive import EditMode
    descriptor = getattr(request, 'group_external_original_reference', None)
    if descriptor is None:
        return explicit
    if request.edit_mode is not EditMode.GROUP or not request.review_first_enabled or request.acquire_only:
        raise ValueError('external original request requires a review-first GROUP render')
    reference = load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
        song=request.song, request_sources=request.sources)
    if explicit is not None:
        if (not isinstance(explicit, GroupExternalOriginalReference)
                or _canonical(explicit.to_dict()) != _canonical(reference.to_dict())):
            raise ValueError('explicit original differs from the saved request authority')
    return reference
