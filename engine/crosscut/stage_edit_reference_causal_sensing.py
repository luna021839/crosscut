"""Build real causal-sensing evidence on the reference trainer's exact grid.

This is not a legacy snapshot migration: no old feature receives a new schema
stamp. The existing media materializer measures the synchronized original
sources, using the installed private gallery and the same clocks/grid as the
feature job. Human reference edit labels never enter the sensing request.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .ive_causal_sensing_installation import discover_ive_causal_sensing_installation
from .ive_causal_sensing_media_job import (
    IVECausalSensingMediaRequest,
    IVECausalSensingSource,
    ive_causal_sensing_source_availability_evidence_sha256,
    materialize_ive_causal_sensing_from_media,
    reject_attitude_before_ive_causal_sensing_access,
)
from .reference_evidence_builder import ReferenceSourceClock
from .vision import SceneDetectionConfig, detect_scene_changes_in_video_cached

logger = logging.getLogger(__name__)


class ReferenceSensingSource(Protocol):
    source_id: str
    path: Path
    clock: Mapping[str, Any]
    frame_rate: float
    declared_frame_count: int
    content_sha256: str


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def reference_sensing_sources(
    sources: Sequence[ReferenceSensingSource],
    master_times: Sequence[float],
    native_cuts: Mapping[str, Sequence[float]],
) -> tuple[IVECausalSensingSource, ...]:
    """Bind exact source clocks and current+following-frame availability."""
    times = tuple(master_times)
    if not times or times != tuple(sorted(set(times))):
        raise ValueError('reference sensing decision times must be nonempty and canonical')
    ids = [item.source_id for item in sources]
    if len(set(ids)) != len(ids) or set(native_cuts) != set(ids):
        raise ValueError('native cut evidence must cover every unique reference source')
    result = []
    for source in sorted(sources, key=lambda item: item.source_id):
        cuts = tuple(native_cuts[source.source_id])
        typed = IVECausalSensingSource(
            source_id=source.source_id, path=source.path,
            media_content_sha256=source.content_sha256,
            clock=ReferenceSourceClock.from_mapping(source.clock),
            native_cut_source_seconds=cuts,
            native_cut_evidence_sha256=_digest({
                'contract': 'reference_training_measured_native_cuts_v1',
                'source_id': source.source_id,
                'source_content_sha256': source.content_sha256,
                'source_seconds': list(cuts),
            }),
        )
        unavailable = tuple(t for t in times
            if typed.clock.master_to_source(t) < 0
            or round(typed.clock.master_to_source(t) * source.frame_rate) + 1
            >= source.declared_frame_count)
        result.append(replace(typed,
            unavailable_master_times=unavailable,
            availability_frame_rate=source.frame_rate,
            availability_declared_frame_count=source.declared_frame_count,
            unavailable_master_time_evidence_sha256=(
                ive_causal_sensing_source_availability_evidence_sha256(
                    source_id=source.source_id,
                    media_content_sha256=source.content_sha256,
                    source_clock_sha256=typed.clock_sha256,
                    decision_grid_sha256=_digest(list(times)),
                    frame_rate=source.frame_rate,
                    declared_frame_count=source.declared_frame_count,
                    unavailable_master_times=unavailable,
                )
            ),
        ))
    return tuple(result)


def materialize_reference_causal_sensing(
    *, song: str, sources: Sequence[ReferenceSensingSource],
    master_times: Sequence[float], artifact_directory: Path,
    edit_mode: object = 'group', target_member: object | None = None,
):
    reject_attitude_before_ive_causal_sensing_access(song=song,
        source_ids=[item.source_id for item in sources],
        source_paths=[item.path for item in sources])
    installation = discover_ive_causal_sensing_installation()
    cuts = {}
    for source in sources:
        logger.info('Reference causal sensing: native cuts for %s', source.source_id)
        changes, cache = detect_scene_changes_in_video_cached(source.path,
            config=SceneDetectionConfig(min_gap_frames=8), sample_every=2,
            source_cache_directory=artifact_directory / 'native-cut-cache')
        cuts[source.source_id] = tuple(change.frame_index / source.frame_rate
                                       for change in changes)
        logger.info('Reference native cuts: source=%s count=%d cache_reused=%s',
                    source.source_id, len(changes), cache.get('analysis_reused', False))
    request = IVECausalSensingMediaRequest(
        song=song, edit_mode=edit_mode, target_member=target_member,
        sources=reference_sensing_sources(sources, master_times, cuts),
        master_decision_times=tuple(master_times),
        gallery_artifact_path=installation.gallery_artifact_path,
        gallery_content_sha256=installation.gallery_content_sha256,
        artifact_directory=artifact_directory,
        created_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    logger.info('Reference real causal sensing: %d sources, %d exact times',
                len(request.sources), len(request.master_decision_times))
    result = materialize_ive_causal_sensing_from_media(request)
    logger.info('Reference real causal sensing finished; generating current feature snapshots')
    return result
