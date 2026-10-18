"""Strict round-trip of renderer inputs, including geometry and look tracks."""
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

from .render import (BoundaryLookKeyframe, BoundaryLookTrack, EditorialTransformKeyframe,
    EditorialTransformTrack, NormalizedCrop, PanCropKeyframe, PanCropTrack,
    RenderPlan, RenderSegment, RenderTransition, SourceLookAdjustment)


def render_segment_from_json(raw: Mapping[str, Any]) -> RenderSegment:
    """Exact camera/clock preimage shared by plan and received sensor replay."""
    if not isinstance(raw, dict):
        raise ValueError('RenderSegment must be an object')
    try:
        args = {f.name:raw[f.name] for f in fields(RenderSegment) if f.name in raw}
        if raw.get('crop') is not None:
            args['crop'] = NormalizedCrop(**raw['crop'])
        if raw.get('pan_crop') is not None:
            args['pan_crop'] = PanCropTrack(tuple(PanCropKeyframe(
                time_seconds=item['time_seconds'], crop=NormalizedCrop(**item['crop']))
                for item in raw['pan_crop']['keyframes']))
        for name,cls,keyframe in (
                ('editorial_transform',EditorialTransformTrack,EditorialTransformKeyframe),
                ('outgoing_overlap_transform',EditorialTransformTrack,EditorialTransformKeyframe),
                ('boundary_look_track',BoundaryLookTrack,BoundaryLookKeyframe)):
            item=raw.get(name)
            if item is not None:
                values={f.name:item[f.name] for f in fields(cls) if f.name in item}
                values['keyframes']=tuple(keyframe(**point) for point in item['keyframes'])
                args[name]=cls(**values)
        if raw.get('look_adjustment') is not None:
            args['look_adjustment']=SourceLookAdjustment(**raw['look_adjustment'])
        # RenderPlan.to_json omits selected null fields, while a complete state
        # uses asdict. Each caller rechecks its OWN complete canonical preimage;
        # forcing one segment encoding onto the other rejects legitimate plans.
        return RenderSegment(**args)
    except (KeyError,TypeError,OverflowError) as exc:
        raise ValueError('render segment is incomplete or malformed') from exc


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
    def transition(raw):
        args = values(RenderTransition, raw)
        if raw.get('kind') == 'face_dissolve' and raw.get('alpha_curve') is not None:
            args['face_dissolve_curve'] = raw['alpha_curve']
        return RenderTransition(**args)
    try:
        args = values(RenderPlan, dict(payload))
        args['sources'] = tuple(Path(p) for p in payload['sources'])
        args['output'] = Path(payload['output'])
        if 'authorized_shot_video_clocks' in payload:
            from .shot_video_clock import shot_video_clock_from_dict
            args['authorized_shot_video_clocks'] = tuple(shot_video_clock_from_dict(row)
                for row in payload['authorized_shot_video_clocks'])
        args['segments'] = tuple(render_segment_from_json(raw) for raw in payload['segments'])
        args['transitions'] = tuple(transition(raw) for raw in payload['transitions'])
        plan = RenderPlan(**args)
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if canonical(plan.to_json()) != canonical(dict(payload)):
            raise ValueError('render plan round-trip changed serialized input or redundant clocks')
        return plan
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('render plan is incomplete or malformed') from exc
