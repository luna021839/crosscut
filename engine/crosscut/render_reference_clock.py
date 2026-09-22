"""Bind perceptual source references to the clock used by this render."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .render import segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output, RenderedBoundaryNativeFrame
from .render_frame_schedule import FRAME_SCHEDULE_CONTRACT, probe_render_frame_clock
from .render_plan_serialization import render_plan_from_json
from .render_clock_artifact import RENDER_CLOCK_ARTIFACT_CONTRACT, render_clock_path, video_content_sha256


@dataclass(frozen=True)
class RenderReferenceClock:
    frames: Mapping[int, RenderedBoundaryNativeFrame]
    source_fps: Mapping[int, float]
    timestamp_fallback_sources: tuple[int, ...]


def load_render_reference_clock(
    video_path: Path, plan_payload: Mapping[str, Any], output_frame_indices,
    *, probe=probe_render_frame_clock, allow_legacy: bool = False,
) -> RenderReferenceClock | None:
    """Only the evaluated video's adjacent, content-bound clock has authority.

    The sidecar is not pixel approval. Its complete plan must round-trip, and
    its claimed verified native clocks must still match the actual sources.
    Unverified sources remain explicitly on the existing timestamp QA path.
    """
    sidecar = render_clock_path(video_path)
    if not sidecar.exists():
        if allow_legacy and plan_payload.get('frame_clock_contract') is None:
            return None
        raise ValueError('required render clock artifact is missing')
    if sidecar.is_symlink() or not 0 < sidecar.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError('render clock sidecar is not a bounded regular artifact')
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or payload.get('contract') != FRAME_SCHEDULE_CONTRACT
            or payload.get('artifact_contract') != RENDER_CLOCK_ARTIFACT_CONTRACT
            or payload.get('video_content_sha256') != video_content_sha256(video_path)
            or type(payload.get('enabled')) is not bool
            or payload.get('quality_approval') is not False
            or payload.get('plan') != dict(plan_payload)):
        raise ValueError('render clock sidecar does not bind the exact render plan')
    plan = render_plan_from_json(plan_payload)
    raw_clocks = payload.get('source_clocks')
    fallback = payload.get('timestamp_fallback_source_indices')
    if not isinstance(raw_clocks, dict) or not isinstance(fallback, list):
        raise ValueError('render clock source partition is missing')
    used = {segment.source_index for segment in plan.segments}
    if (any(type(i) is not int for i in fallback) or len(set(fallback)) != len(fallback)
            or any(not isinstance(k, str) or k not in {str(i) for i in used} for k in raw_clocks)):
        raise ValueError('render clock source indices are invalid')
    verified = {int(k) for k in raw_clocks}
    if (verified & set(fallback) or verified | set(fallback) != used
            or (not payload['enabled'] and verified)):
        raise ValueError('render clock verified/fallback partition disagrees')
    indices = tuple(output_frame_indices)
    if any(type(i) is not int or not 0 <= i < plan.output_frame_count for i in indices):
        raise ValueError('render clock output request is outside the plan')
    # Pure samples only. A dissolve is not silently compared with one picture.
    counts, cursor, selected, needed = segment_frame_counts(plan), 0, [], set()
    for index, (segment, count) in enumerate(zip(plan.segments, counts, strict=True)):
        incoming = plan.transitions[index - 1] if index else None
        overlap = incoming.frame_count if incoming is not None and incoming.kind.uses_overlap else 0
        for i in indices:
            if cursor <= i < cursor + count and segment.source_index in verified:
                if i < cursor + overlap:
                    raise ValueError('composited output has no single native source reference')
                selected.append(i)
                needed.add(segment.source_index)
        cursor += count
    clocks = {}
    for index in sorted(needed):
        clock = probe(plan.sources[index])
        if clock is None or clock.to_dict() != raw_clocks[str(index)]:
            raise ValueError(f'render clock for source {index} no longer matches its media')
        clocks[index] = clock
    frames = native_frame_clocks_for_output(plan, clocks, selected,
        allow_pure_frames_in_overlap_plan=True)
    return RenderReferenceClock(frames, {i: float(c.frame_rate) for i,c in clocks.items()}, tuple(fallback))
