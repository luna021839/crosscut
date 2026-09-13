"""Strict round-trip of renderer inputs, including geometry and look tracks."""
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

from .render import (BoundaryLookKeyframe, BoundaryLookTrack, EditorialTransformKeyframe,
    EditorialTransformTrack, NormalizedCrop, PanCropKeyframe, PanCropTrack,
    RenderPlan, RenderSegment, RenderTransition, SourceLookAdjustment)


def render_plan_from_json(payload: Mapping[str, Any]) -> RenderPlan:
    """Recompute all redundant clocks and reject any lossy reconstruction.

    Unlike a boundary-only deserializer this preserves fields that decide
    whether the renderer coalesces identity geometry into a single unit.
    """
    def values(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError(f'{cls.__name__} must be an object')
        return {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
    def track(cls, keyframe_cls, raw):
        if raw is None:
            return None
        args = values(cls, raw)
        args['keyframes'] = tuple(keyframe_cls(**item) for item in raw['keyframes'])
        return cls(**args)
    try:
        args = values(RenderPlan, dict(payload))
        args['sources'] = tuple(Path(p) for p in payload['sources'])
        args['output'] = Path(payload['output'])
        segments = []
        for raw in payload['segments']:
            segment = values(RenderSegment, raw)
            if raw.get('crop') is not None:
                segment['crop'] = NormalizedCrop(**raw['crop'])
            if raw.get('pan_crop') is not None:
                segment['pan_crop'] = PanCropTrack(tuple(PanCropKeyframe(
                    time_seconds=item['time_seconds'], crop=NormalizedCrop(**item['crop']))
                    for item in raw['pan_crop']['keyframes']))
            for name in ('editorial_transform', 'outgoing_overlap_transform'):
                segment[name] = track(EditorialTransformTrack, EditorialTransformKeyframe, raw.get(name))
            segment['boundary_look_track'] = track(BoundaryLookTrack, BoundaryLookKeyframe,
                                                   raw.get('boundary_look_track'))
            if raw.get('look_adjustment') is not None:
                segment['look_adjustment'] = SourceLookAdjustment(**raw['look_adjustment'])
            segments.append(RenderSegment(**segment))
        args['segments'] = tuple(segments)
        args['transitions'] = tuple(RenderTransition(**values(RenderTransition, raw))
                                    for raw in payload['transitions'])
        plan = RenderPlan(**args)
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if canonical(plan.to_json()) != canonical(dict(payload)):
            raise ValueError('render plan round-trip changed serialized input or redundant clocks')
        return plan
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('render plan is incomplete or malformed') from exc
