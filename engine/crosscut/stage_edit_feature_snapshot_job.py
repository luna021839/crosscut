"""Offline, leakage-safe StageEdit feature snapshots from synchronized media.

The reference adapter can recover editorial targets from a finished human
StageMix, but those reverse-match measurements must never enter model inputs.
This job closes that train/serve gap by decoding only the synchronized source
videos at the adapter's causal decision timestamps and materializing the exact
inference-safe StageEdit token schema.

One song's references share a source catalog.  Their visual descriptors are
decoded once per source over the union of requested timestamps, while each
reference receives its own strictly-past causal edit history.  Outputs are
content-addressed and never overwrite a different artifact.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .audio import DEFAULT_ANALYSIS_SAMPLE_RATE, extract_mono_pcm
from .reference_match import ReferenceMatchConfig, decode_target_frames_sequential
from .rhythm import analyze_rhythm
from .section_director import build_section_plan
from .stage_edit_feature_materializer import (
    CausalEditDecision,
    SourceFeatureObservation,
    StageEditCausalSensingArtifacts,
    StageEditFeatureFrame,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
    STAGE_EDIT_PRODUCTION_CAUSAL_SENSING_GATE_CONTRACT,
    materialize_stage_edit_feature_snapshot,
    source_observation_from_visual_cache_payload,
)
from .stage_edit_reference_adapter import build_reference_training_example
from .stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA
from .visual_score import (
    describe_frame,
    frame_visual_numeric_features,
    visual_analyzer_fingerprint_sha256,
)


FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION = 4
FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS = (
    "synchronized_source_causal_stage_edit_feature_snapshot_v4"
)
ACTION_FORMATION_EXTRACTION_CONTRACT = (
    "identity_free_pose60_layout134_formation6x8x3_with_hash_bound_"
    "current_next_temporal_pose_sidecar_and_missing_masks_v2"
)
FEATURE_SNAPSHOT_JOB_INPUT_POLICY: Mapping[str, bool] = {
    "features_available_before_current_editorial_choice": True,
    "strictly_past_causal_edit_history_only": True,
    "reference_match_scores_present": False,
    "selected_or_rejected_source_targets_present": False,
    "raw_frames_persisted": False,
    "identity_embeddings_persisted": False,
    "identity_free_temporal_pose_sidecars_persisted": True,
    "temporal_pose_sidecars_hash_bound_in_snapshot_v4": True,
    "temporal_pose_sidecars_enter_transformer_tensors": False,
    "causal_sensing_exact_timestamp_and_content_hash_bound": True,
    "member_state_unknown_preserved": True,
    "ive_performer_distribution_unknown_preserved": True,
    "causal_sensing_triple_required_for_production": True,
    "inspection_only_without_causal_sensing": False,
}
FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS = (
    FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
    ACTION_FORMATION_EXTRACTION_CONTRACT,
    "reference_match_scores_and_current_or_future_targets_excluded",
    "visual_descriptors_decoded_once_per_source_in_forward_order",
)
DEFAULT_VISUAL_WORKING_WIDTH = 480
DEFAULT_NEGATIVE_SAMPLES_PER_HOLD = 2
DEFAULT_BOUNDARY_GUARD_SECONDS = 0.18
DEFAULT_MAX_SOURCES = 32
# YouTube video IDs can legitimately begin with ``_`` or ``-``.  Keep path
# separators and traversal tokens impossible without rejecting those real IDs.
_SAFE_ARTIFACT_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def map_reference_master_to_source_seconds(
    clock: Mapping[str, Any],
    master_seconds: float,
) -> float:
    """Map an exact reference-master timestamp using the catalog clock.

    Serialized training catalogs intentionally do not reconstruct a permissive
    ``TimeWarpMap`` with guessed duration/configuration fields.  This parser
    accepts only the two explicit mapping contracts written by the reference
    evidence builders and reproduces their checked linear interpolation and
    endpoint extrapolation.
    """

    timestamp = _finite(master_seconds, "master_seconds")
    if timestamp < 0.0:
        raise ValueError("master_seconds cannot be negative")
    if not isinstance(clock, Mapping):
        raise TypeError("source clock must be an object")
    mode = str(clock.get("mapping_mode") or "").strip()
    application = str(clock.get("global_offset_application") or "").strip()
    offset = _finite(clock.get("global_offset_seconds"), "global offset")
    raw_points = clock.get("control_points")
    if not isinstance(raw_points, list):
        raise ValueError("source clock control_points must be an array")

    if mode == "scalar_offset":
        if application != "added_to_master_seconds" or raw_points:
            raise ValueError("scalar source clock contract is inconsistent")
        return timestamp + offset

    if mode != "piecewise_linear" or application != (
        "diagnostic_scalar_baseline_control_points_are_absolute"
    ):
        raise ValueError("unsupported source clock mapping contract")
    if len(raw_points) < 2:
        raise ValueError("piecewise source clock needs at least two points")
    points: list[tuple[float, float]] = []
    for raw in raw_points:
        if not isinstance(raw, Mapping):
            raise ValueError("source clock control point must be an object")
        point = (
            _finite(raw.get("master_seconds"), "control-point master time"),
            _finite(raw.get("source_seconds"), "control-point source time"),
        )
        if points and (point[0] <= points[-1][0] or point[1] <= points[-1][1]):
            raise ValueError("source clock control points must be monotonic")
        points.append(point)
    masters = tuple(point[0] for point in points)
    position = bisect_right(masters, timestamp)
    if position == 0:
        left, right = points[0:2]
    elif position >= len(points):
        left, right = points[-2:]
    else:
        left, right = points[position - 1 : position + 1]
    fraction = (timestamp - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    source_id: str
    path: Path
    clock: Mapping[str, Any]
    frame_rate: float
    declared_frame_count: int
    sync_confidence: float
    sync_margin: float
    sync_residual_normalized: float | None
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _ReferenceScaffold:
    reference_id: str
    times: tuple[float, ...]
    histories: tuple[tuple[CausalEditDecision, ...], ...]
    performance_start_seconds: float
    performance_end_seconds: float


@dataclass(frozen=True, slots=True)
class StageEditFeatureSnapshotJobResult:
    output_directory: Path
    manifest_path: Path
    snapshots: Mapping[str, Path]
    snapshot_sha256: Mapping[str, str]
    statistics: Mapping[str, Any]


def _load_supervision(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    content = Path(path).read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, Mapping):
        raise ValueError(f"reference supervision root must be an object: {path}")
    return content, payload


def _performance_span(payload: Mapping[str, Any]) -> tuple[float, float]:
    raw_shots = payload.get("shot_labels")
    if not isinstance(raw_shots, list):
        raise ValueError("reference supervision has no shot_labels")
    starts: list[float] = []
    ends: list[float] = []
    for shot in raw_shots:
        labels = shot.get("edit_labels") if isinstance(shot, Mapping) else None
        cut = labels.get("cut") if isinstance(labels, Mapping) else None
        if not isinstance(cut, Mapping):
            continue
        try:
            start = _finite(cut.get("cut_master_time"), "shot start")
            end = _finite(cut.get("end_master_time"), "shot end")
        except (TypeError, ValueError):
            continue
        if start >= 0.0 and end > start:
            starts.append(start)
            ends.append(end)
    if not starts:
        raise ValueError("reference supervision has no valid performance span")
    return min(starts), max(ends)


def _canonical_causal_sources(payload: Mapping[str, Any]) -> Mapping[float, str]:
    """Recover proven past source switches even when source loss is masked.

    A detector-owned boundary can prove which source became visible while the
    source-selection target is masked because no usable rejected candidate is
    left.  The reference adapter retains that source in causal context; the
    richer feature snapshot must retain it as well.
    """

    raw_records = payload.get("detected_boundary_records")
    if raw_records is None:
        return {}
    if not isinstance(raw_records, list):
        raise ValueError("detected_boundary_records must be an array")
    by_time: dict[float, str] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("detected boundary record must be an object")
        if not bool(raw.get("is_editorial_source_switch_positive")):
            continue
        if bool(raw.get("loss_masked")):
            raise ValueError("canonical positive source switch cannot be loss masked")
        cut_target = _finite(
            raw.get("cut_target"), "detected-boundary cut target"
        )
        if cut_target != 1.0:
            raise ValueError("canonical positive source switch needs cut target one")
        timestamp = _finite(
            raw.get("cut_master_time"), "detected-boundary cut time"
        )
        source_id = str(raw.get("new_source_id") or "").strip()
        if timestamp < 0.0 or not source_id or source_id == "__UNKNOWN__":
            raise ValueError("canonical causal source switch is invalid")
        previous = by_time.get(timestamp)
        if previous is not None and previous != source_id:
            raise ValueError("canonical causal source switch is ambiguous")
        by_time[timestamp] = source_id
    return by_time


def _reference_scaffold(
    payload: Mapping[str, Any],
    *,
    song_id: str,
    negative_samples_per_hold: int,
    boundary_guard_seconds: float,
    max_sources: int,
) -> _ReferenceScaffold:
    example = build_reference_training_example(
        payload,
        song_id=song_id,
        negative_samples_per_hold=negative_samples_per_hold,
        boundary_guard_seconds=boundary_guard_seconds,
        max_sources=max_sources,
    )
    if len(example.sequence.tokens) != len(example.targets.targets):
        raise ValueError("reference adapter token/target lengths diverged")
    canonical_sources = _canonical_causal_sources(payload)
    history: list[CausalEditDecision] = []
    times: list[float] = []
    histories: list[tuple[CausalEditDecision, ...]] = []
    for token_index, (token, target) in enumerate(
        zip(example.sequence.tokens, example.targets.targets, strict=True)
    ):
        timestamp = float(token.master_time_seconds)
        if times and timestamp <= times[-1]:
            raise ValueError("reference adapter decision times are not unique")
        times.append(timestamp)
        histories.append(tuple(history))
        if target.cut == 1.0 or (
            token_index == 0 and target.selected_source_id is not None
        ):
            causal_source = (
                canonical_sources.get(timestamp)
                if target.cut == 1.0
                else None
            ) or target.selected_source_id
            history.append(
                CausalEditDecision(
                    cut_time_seconds=timestamp,
                    source_id=causal_source,
                    transition_strength=1.0,
                )
            )
    start, end = _performance_span(payload)
    return _ReferenceScaffold(
        reference_id=example.reference_id,
        times=tuple(times),
        histories=tuple(histories),
        performance_start_seconds=start,
        performance_end_seconds=end,
    )


def _sync_residual(clock: Mapping[str, Any]) -> float | None:
    provenance = clock.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    timewarp = provenance.get("local_timewarp")
    if not isinstance(timewarp, Mapping):
        return None
    residuals = timewarp.get("residuals")
    configuration = timewarp.get("configuration")
    if not isinstance(residuals, Mapping) or not isinstance(configuration, Mapping):
        return None
    p95 = residuals.get("p95_seconds")
    maximum = configuration.get("maximum_trend_residual_seconds")
    if p95 is None or maximum is None:
        return None
    p95_value = _finite(p95, "time-warp p95 residual")
    maximum_value = _finite(maximum, "maximum time-warp residual")
    if maximum_value <= 0.0:
        return None
    return min(1.0, max(0.0, p95_value / maximum_value))


def _source_records(catalog: Sequence[Any]) -> tuple[_SourceRecord, ...]:
    records: list[_SourceRecord] = []
    ids: set[str] = set()
    for raw in catalog:
        if not isinstance(raw, Mapping):
            raise ValueError("source catalog entry must be an object")
        source_id = str(raw.get("video_id") or raw.get("source_id") or "").strip()
        path = Path(str(raw.get("path") or "")).expanduser()
        clock = raw.get("clock")
        probe = raw.get("media_probe")
        provenance = raw.get("provenance")
        if (
            not source_id
            or source_id in ids
            or not isinstance(clock, Mapping)
            or not isinstance(probe, Mapping)
            or not isinstance(provenance, Mapping)
        ):
            raise ValueError("source catalog identity is invalid")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source media is missing or unsafe: {path}")
        frame_rate = _finite(probe.get("frame_rate"), "catalog frame rate")
        declared = probe.get("declared_frame_count")
        if (
            frame_rate <= 0.0
            or isinstance(declared, bool)
            or not isinstance(declared, int)
            or declared <= 1
        ):
            raise ValueError("source media probe is invalid")
        expected_size = probe.get("size_bytes")
        if isinstance(expected_size, int) and not isinstance(expected_size, bool):
            if path.stat().st_size != expected_size:
                raise ValueError(f"source media size changed: {path}")
        score = _finite(provenance.get("coarse_score"), "coarse sync score")
        margin = _finite(provenance.get("coarse_margin"), "coarse sync margin")
        records.append(
            _SourceRecord(
                source_id=source_id,
                path=path.resolve(),
                clock=clock,
                frame_rate=frame_rate,
                declared_frame_count=declared,
                sync_confidence=min(1.0, max(0.0, score)),
                sync_margin=min(1.0, max(0.0, margin)),
                sync_residual_normalized=_sync_residual(clock),
                content_sha256=_file_sha256(path),
            )
        )
        ids.add(source_id)
    if len(records) < 2:
        raise ValueError("feature snapshots require at least two synchronized sources")
    return tuple(sorted(records, key=lambda item: item.source_id))


def _source_observations(
    sources: Sequence[_SourceRecord],
    master_times: Sequence[float],
    *,
    working_width: int,
) -> tuple[dict[str, dict[float, SourceFeatureObservation]], str, Mapping[str, Any]]:
    by_source: dict[str, dict[float, SourceFeatureObservation]] = {}
    feature_hash = hashlib.sha256()
    source_stats: dict[str, Any] = {}
    decode_config = ReferenceMatchConfig(working_width=working_width)
    for source in sources:
        index_by_time: dict[float, int | None] = {}
        requested: set[int] = set()
        for timestamp in master_times:
            source_time = map_reference_master_to_source_seconds(
                source.clock, timestamp
            )
            # Match the frame quantization used by reference attribution and
            # crop supervision exactly.  A second half-up convention here can
            # disagree at N + 0.5 and pair a feature with the adjacent label.
            index = int(round(source_time * source.frame_rate))
            # Production causal sensing requires the exact decision frame and
            # its immediate follower.  Keep candidate availability identical
            # to that media contract: a catalog's final native frame is not a
            # usable decision cell even though the current frame alone exists.
            if index < 0 or index + 1 >= source.declared_frame_count:
                index_by_time[timestamp] = None
                continue
            index_by_time[timestamp] = index
            requested.add(index)
            requested.add(index + 1)

        decoded = decode_target_frames_sequential(
            source.path,
            tuple(sorted(requested)),
            config=decode_config,
        )
        if requested and not math.isclose(
            decoded.frame_rate,
            source.frame_rate,
            rel_tol=2.0e-3,
            abs_tol=2.0e-3,
        ):
            raise ValueError(f"source frame rate changed: {source.path}")
        missing = set(requested) - set(decoded.frames)
        # Both current and immediate-following frames are part of the causal
        # availability contract and are therefore mandatory.
        mandatory = set(requested)
        missing_mandatory = tuple(sorted(missing & mandatory))
        if missing_mandatory:
            raise ValueError(
                f"source decode missed decision frames {missing_mandatory[:8]}: "
                f"{source.path}"
            )

        numeric_by_index: dict[int, Mapping[str, Any]] = {}
        for index in sorted(decoded.frames):
            frame = decoded.frames[index]
            following = decoded.frames.get(index + 1)
            descriptor = describe_frame(
                frame,
                following_frame=following,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
            numeric_by_index[index] = frame_visual_numeric_features(descriptor)

        gate = source.sync_confidence >= 0.32 and source.sync_margin >= 0.025
        observations: dict[float, SourceFeatureObservation] = {}
        available_count = 0
        model_pose_count = 0
        formation_layout_count = 0
        formation_people_count = 0
        formation_upper_body_count = 0
        formation_upper_body_person_count = 0
        temporal_pose_sidecar_count = 0
        for timestamp in master_times:
            index = index_by_time[timestamp]
            numeric = numeric_by_index.get(index, {}) if index is not None else {}
            following = (
                numeric_by_index.get(index + 1, {}) if index is not None else {}
            )
            available = index is not None and index in numeric_by_index
            if available:
                available_count += 1
                pose_embedding = numeric.get("pose_embedding")
                if (
                    numeric.get("pose_backend") != "structure_hog_fallback"
                    and isinstance(pose_embedding, (list, tuple))
                    and len(pose_embedding) == 60
                ):
                    model_pose_count += 1
                layout_embedding = numeric.get("person_layout_embedding")
                if (
                    isinstance(layout_embedding, (list, tuple))
                    and len(layout_embedding) == 134
                ):
                    formation_layout_count += 1
                formation_people = numeric.get("formation_people")
                if isinstance(formation_people, (list, tuple)) and formation_people:
                    formation_people_count += 1
                    upper_body_people = sum(
                        isinstance(person, Mapping)
                        and isinstance(
                            person.get("upper_body_keypoints_xyc"), (list, tuple)
                        )
                        and len(person.get("upper_body_keypoints_xyc")) == 8
                        for person in formation_people
                    )
                    formation_upper_body_person_count += upper_body_people
                    if upper_body_people:
                        formation_upper_body_count += 1
            observation = source_observation_from_visual_cache_payload(
                source.source_id,
                numeric,
                following_numeric_features=following,
                interval_seconds=(1.0 / source.frame_rate if available else None),
                sync_confidence=source.sync_confidence,
                sync_residual_normalized=source.sync_residual_normalized,
                available=available,
                hard_gate_passed=available and gate,
            )
            observations[timestamp] = observation
            temporal_pose_sidecar_count += int(
                observation.temporal_pose_sidecar is not None
            )
            feature_hash.update(
                _canonical_bytes(
                    {
                        "source_id": source.source_id,
                        "master_time_seconds": timestamp,
                        "source_frame_index": index,
                        "numeric_features": numeric,
                        "following_numeric_features": following,
                        "available": available,
                        "hard_gate_passed": available and gate,
                    }
                )
            )
        by_source[source.source_id] = observations
        source_stats[source.source_id] = {
            "decision_time_count": len(master_times),
            "available_time_count": available_count,
            "model_pose60_observed_time_count": model_pose_count,
            "formation_layout134_observed_time_count": formation_layout_count,
            "formation_people_observed_time_count": formation_people_count,
            "formation_upper_body_observed_time_count": formation_upper_body_count,
            "formation_upper_body_observed_person_count": (
                formation_upper_body_person_count
            ),
            "temporal_pose60_sidecar_observed_time_count": (
                temporal_pose_sidecar_count
            ),
            "decoded_unique_frame_count": len(decoded.frames),
            "hard_sync_gate_passed": gate,
            "sync_confidence": source.sync_confidence,
            "sync_margin": source.sync_margin,
            "sync_residual_normalized": source.sync_residual_normalized,
        }
        del decoded
        del numeric_by_index
    return by_source, feature_hash.hexdigest(), source_stats


def _write_content_addressed_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    # ``Path.exists`` is false for a dangling symlink.  Check links first so a
    # crafted output/temporary path can never redirect a supposedly immutable
    # artifact write outside the requested directory.
    if path.is_symlink():
        raise FileExistsError(f"content-addressed artifact collision: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"content-addressed artifact collision: {path}")
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink() or temporary.exists():
        raise FileExistsError(f"stale feature-snapshot temporary file: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        # Close the check/open race without following an asynchronously
        # inserted symlink or truncating another writer's partial artifact.
        raise FileExistsError(
            f"stale feature-snapshot temporary file: {temporary}"
        ) from error
    temporary.replace(path)
    return path.resolve()


def build_stage_edit_feature_snapshots(
    supervision_paths: Sequence[Path],
    *,
    song_id: str,
    output_directory: Path,
    expected_source_count: int | None = None,
    negative_samples_per_hold: int = DEFAULT_NEGATIVE_SAMPLES_PER_HOLD,
    boundary_guard_seconds: float = DEFAULT_BOUNDARY_GUARD_SECONDS,
    max_sources: int = DEFAULT_MAX_SOURCES,
    working_width: int = DEFAULT_VISUAL_WORKING_WIDTH,
    style_features: Mapping[str, Any] | None = None,
    causal_sensing: StageEditCausalSensingArtifacts | None = None,
    edit_mode: object = "group",
    target_member: object | None = None,
    inspection_only_without_causal_sensing: bool = False,
) -> StageEditFeatureSnapshotJobResult:
    """Build immutable feature-only snapshots for one complete song corpus."""

    normalized_song = str(song_id).strip()
    paths = tuple(Path(path) for path in supervision_paths)
    destination = Path(output_directory)
    if not normalized_song or not paths:
        raise ValueError("song_id and supervision paths are required")
    if negative_samples_per_hold < 0 or boundary_guard_seconds < 0.0:
        raise ValueError("feature snapshot sampling configuration is invalid")
    if max_sources < 2 or working_width < 320:
        raise ValueError("feature snapshot model/decode configuration is invalid")
    if not isinstance(inspection_only_without_causal_sensing, bool):
        raise TypeError("inspection_only_without_causal_sensing must be a boolean")
    if inspection_only_without_causal_sensing:
        if causal_sensing is not None:
            raise ValueError(
                "inspection-only feature job cannot bind causal sensing"
            )
    else:
        if not isinstance(causal_sensing, StageEditCausalSensingArtifacts):
            raise ValueError(
                "production feature snapshot job requires the complete causal-"
                "sensing artifact triple"
            )
        causal_sensing.binding.require_production_scope(
            edit_mode=edit_mode,
            target_member=target_member,
        )

    loaded = tuple((path, *_load_supervision(path)) for path in paths)
    reference_ids = tuple(
        str(payload.get("reference_id") or "").strip()
        for _path, _content, payload in loaded
    )
    if any(_SAFE_ARTIFACT_ID.fullmatch(value) is None for value in reference_ids):
        raise ValueError("reference supervision ids must be safe artifact ids")
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference supervision ids must be unique")
    catalogs = tuple(payload.get("source_catalog") for _, _, payload in loaded)
    if any(not isinstance(value, list) for value in catalogs):
        raise ValueError("every reference supervision needs a source catalog")
    catalog_sha256 = tuple(_canonical_sha256(value) for value in catalogs)
    if len(set(catalog_sha256)) != 1:
        raise ValueError("song references do not share one exact source catalog")
    sources = _source_records(catalogs[0])
    if expected_source_count is not None and len(sources) != expected_source_count:
        raise ValueError(
            f"expected {expected_source_count} sources but catalog has {len(sources)}"
        )
    if len(sources) > max_sources:
        raise ValueError("source catalog exceeds StageEdit max_sources")

    scaffolds = tuple(
        _reference_scaffold(
            payload,
            song_id=normalized_song,
            negative_samples_per_hold=negative_samples_per_hold,
            boundary_guard_seconds=boundary_guard_seconds,
            max_sources=max_sources,
        )
        for _path, _content, payload in loaded
    )
    global_start = min(item.performance_start_seconds for item in scaffolds)
    global_end = max(item.performance_end_seconds for item in scaffolds)
    master_times = tuple(sorted({time for item in scaffolds for time in item.times}))
    if not master_times:
        raise ValueError("reference corpus has no StageEdit decision times")

    # The identity/scalar-zero source defines the master clock.  Requiring it
    # explicitly avoids analyzing a reference with a shifted broadcaster intro.
    master_candidates = tuple(
        source
        for source in sources
        if source.clock.get("mapping_mode") == "scalar_offset"
        and math.isclose(
            _finite(source.clock.get("global_offset_seconds"), "master offset"),
            0.0,
            abs_tol=1e-9,
        )
    )
    if len(master_candidates) != 1:
        raise ValueError("source catalog needs one exact identity master source")
    master_source = master_candidates[0]
    # ``analyze_rhythm`` reports times relative to the decoded PCM origin,
    # while every StageEdit decision uses the absolute reference-master clock.
    # Decode from master time zero so beat/onset/phrase timestamps remain in
    # the same domain.  Cropping at ``global_start`` without translating the
    # analysis would silently shift every music and section feature whenever a
    # performance starts after a broadcaster intro.
    pcm = extract_mono_pcm(
        master_source.path,
        sample_rate=DEFAULT_ANALYSIS_SAMPLE_RATE,
        start_seconds=None,
        duration_seconds=global_end,
    )
    rhythm = analyze_rhythm(pcm, DEFAULT_ANALYSIS_SAMPLE_RATE)
    section_plan = build_section_plan(
        rhythm,
        start_seconds=global_start,
        end_seconds=global_end,
    )
    observations, visual_features_sha256, source_stats = _source_observations(
        sources,
        master_times,
        working_width=working_width,
    )

    source_identity_payload = [
        {
            "source_id": source.source_id,
            "path": str(source.path),
            "content_sha256": source.content_sha256,
            "clock": source.clock,
            "frame_rate": source.frame_rate,
            "declared_frame_count": source.declared_frame_count,
            "sync_confidence": source.sync_confidence,
            "sync_margin": source.sync_margin,
            "sync_residual_normalized": source.sync_residual_normalized,
        }
        for source in sources
    ]
    provenance_items: list[tuple[str, str]] = [
        ("source_alignment", _canonical_sha256(source_identity_payload)),
        ("visual_feature_cache", visual_features_sha256),
        (
            "visual_score",
            _canonical_sha256(
                {
                    "working_width": working_width,
                    "include_subject_geometry": True,
                    "include_formation_pose": True,
                    "analyzer_fingerprint_sha256": (
                        visual_analyzer_fingerprint_sha256()
                    ),
                }
            ),
        ),
        ("rhythm_analysis", _canonical_sha256(rhythm.to_dict())),
        ("section_plan", _canonical_sha256(section_plan.to_dict())),
    ]
    if style_features is not None:
        provenance_items.append(
            ("style_profile", _canonical_sha256(dict(style_features)))
        )
    provenance = StageEditFeatureProvenance(
        upstream_sha256=tuple(provenance_items),
        analyzer_fingerprint_sha256=visual_analyzer_fingerprint_sha256(),
        contracts=FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
    )

    snapshot_paths: dict[str, Path] = {}
    snapshot_hashes: dict[str, str] = {}
    snapshot_stats: dict[str, Any] = {}
    for scaffold in sorted(scaffolds, key=lambda item: item.reference_id):
        frames = tuple(
            StageEditFeatureFrame(
                master_time_seconds=timestamp,
                source_observations=tuple(
                    observations[source.source_id][timestamp]
                    for source in sources
                ),
                causal_history=history,
            )
            for timestamp, history in zip(
                scaffold.times, scaffold.histories, strict=True
            )
        )
        snapshot = materialize_stage_edit_feature_snapshot(
            f"stageedit-features:{normalized_song}:{scaffold.reference_id}",
            frames,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            rhythm=rhythm,
            section_plan=section_plan,
            style_features=style_features,
            performance_start_seconds=scaffold.performance_start_seconds,
            performance_end_seconds=scaffold.performance_end_seconds,
            provenance=provenance,
            causal_sensing=causal_sensing,
            production_edit_mode=(
                None if inspection_only_without_causal_sensing else edit_mode
            ),
            production_target_member=(
                None if inspection_only_without_causal_sensing else target_member
            ),
            inspection_only_without_causal_sensing=(
                inspection_only_without_causal_sensing
            ),
        )
        name = (
            f"{scaffold.reference_id}-stageedit-feature-snapshot-"
            f"{snapshot.snapshot_sha256[:16]}.json"
        )
        path = _write_content_addressed_json(destination / name, snapshot.to_dict())
        # Re-read through the strict public schema before admitting the artifact.
        restored = StageEditFeatureSnapshot.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if restored.snapshot_sha256 != snapshot.snapshot_sha256:
            raise ValueError("written feature snapshot failed strict round-trip")
        snapshot_paths[scaffold.reference_id] = path
        snapshot_hashes[scaffold.reference_id] = snapshot.snapshot_sha256
        snapshot_stats[scaffold.reference_id] = {
            "token_count": len(snapshot.sequence.tokens),
            "source_count": len(sources),
            "performance_start_seconds": scaffold.performance_start_seconds,
            "performance_end_seconds": scaffold.performance_end_seconds,
        }

    supervision_inputs = [
        {
            "reference_id": str(payload["reference_id"]),
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content, payload in sorted(
            loaded, key=lambda item: str(item[2].get("reference_id"))
        )
    ]
    production_causal_sensing_gate = (
        snapshot.require_production_causal_sensing(
            edit_mode=edit_mode,
            target_member=target_member,
        )
        if not inspection_only_without_causal_sensing
        else {
            "passed": False,
            "contract": STAGE_EDIT_PRODUCTION_CAUSAL_SENSING_GATE_CONTRACT,
            "inspection_only_without_causal_sensing": True,
            "edit_mode": None,
            "target_member": None,
            "member_feature_mode": (
                snapshot.causal_sensing_binding.member_feature_mode
            ),
        }
    )
    manifest_input_policy = dict(FEATURE_SNAPSHOT_JOB_INPUT_POLICY)
    if inspection_only_without_causal_sensing:
        manifest_input_policy.update(
            {
                "causal_sensing_exact_timestamp_and_content_hash_bound": False,
                "member_state_unknown_preserved": False,
                "ive_performer_distribution_unknown_preserved": False,
                "inspection_only_without_causal_sensing": True,
            }
        )
    manifest_unsigned: dict[str, Any] = {
        "schema_version": FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
        "analysis_basis": FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
        "song_id": normalized_song,
        "input_policy": manifest_input_policy,
        "sampling": {
            "negative_samples_per_hold": negative_samples_per_hold,
            "boundary_guard_seconds": boundary_guard_seconds,
            "max_sources": max_sources,
            "visual_working_width": working_width,
        },
        "source_catalog_sha256": catalog_sha256[0],
        "source_count": len(sources),
        "source_content": source_identity_payload,
        "supervision_inputs": supervision_inputs,
        "feature_provenance": snapshot.provenance.to_dict(),
        "causal_sensing_binding": snapshot.causal_sensing_binding.to_dict(),
        "production_causal_sensing_gate": production_causal_sensing_gate,
        "snapshots": [
            {
                "reference_id": reference_id,
                "path": str(snapshot_paths[reference_id]),
                "snapshot_sha256": snapshot_hashes[reference_id],
                **snapshot_stats[reference_id],
            }
            for reference_id in sorted(snapshot_paths)
        ],
        "statistics": {
            "reference_count": len(scaffolds),
            "source_count": len(sources),
            "union_decision_time_count": len(master_times),
            "per_source": source_stats,
        },
    }
    manifest_sha256 = _canonical_sha256(manifest_unsigned)
    manifest = {**manifest_unsigned, "manifest_sha256": manifest_sha256}
    manifest_path = _write_content_addressed_json(
        destination
        / f"stageedit-feature-snapshot-job-manifest-{manifest_sha256[:16]}.json",
        manifest,
    )
    return StageEditFeatureSnapshotJobResult(
        output_directory=destination.resolve(),
        manifest_path=manifest_path,
        snapshots=dict(snapshot_paths),
        snapshot_sha256=dict(snapshot_hashes),
        statistics=manifest_unsigned["statistics"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe StageEdit feature snapshots from media"
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument(
        "--negative-samples-per-hold",
        type=int,
        default=DEFAULT_NEGATIVE_SAMPLES_PER_HOLD,
    )
    parser.add_argument(
        "--boundary-guard-seconds",
        type=float,
        default=DEFAULT_BOUNDARY_GUARD_SECONDS,
    )
    parser.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)
    parser.add_argument(
        "--working-width", type=int, default=DEFAULT_VISUAL_WORKING_WIDTH
    )
    parser.add_argument(
        "--style-features-json",
        type=Path,
        help="Optional JSON object matching the StageEdit style-feature schema",
    )
    parser.add_argument("--edit-mode", choices=("group", "member"), default="group")
    parser.add_argument("--target-member")
    parser.add_argument("--shooting-grammar", type=Path)
    parser.add_argument("--shooting-grammar-content-sha256")
    parser.add_argument("--member-state-graph", type=Path)
    parser.add_argument("--member-state-graph-content-sha256")
    parser.add_argument("--ive-performer-state", type=Path)
    parser.add_argument("--ive-performer-state-content-sha256")
    parser.add_argument(
        "--inspection-only-without-causal-sensing",
        action="store_true",
        help=(
            "Allow an explicitly non-production snapshot with every causal-"
            "sensing column missing. Production validators always reject it."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    style: Mapping[str, Any] | None = None
    if args.style_features_json is not None:
        raw_style = json.loads(args.style_features_json.read_text(encoding="utf-8"))
        if not isinstance(raw_style, Mapping):
            raise ValueError("style feature JSON root must be an object")
        unknown = set(raw_style) - set(DEFAULT_STAGE_EDIT_SCHEMA.style_feature_names)
        if unknown:
            raise ValueError(
                "style feature JSON has unknown keys: " + ", ".join(sorted(unknown))
            )
        style = raw_style
    sensing_values = (
        args.shooting_grammar,
        args.shooting_grammar_content_sha256,
        args.member_state_graph,
        args.member_state_graph_content_sha256,
        args.ive_performer_state,
        args.ive_performer_state_content_sha256,
    )
    if args.inspection_only_without_causal_sensing:
        if any(value is not None for value in sensing_values):
            raise ValueError(
                "inspection-only feature job cannot accept causal-sensing inputs"
            )
        causal_sensing = None
    else:
        if any(value is None for value in sensing_values):
            raise ValueError(
                "production feature job requires all three causal-sensing paths "
                "and content SHA-256 values"
            )
        causal_sensing = StageEditCausalSensingArtifacts(
            shooting_grammar_path=args.shooting_grammar,
            shooting_grammar_content_sha256=(
                args.shooting_grammar_content_sha256
            ),
            member_state_graph_path=args.member_state_graph,
            member_state_graph_content_sha256=(
                args.member_state_graph_content_sha256
            ),
            target_member=(
                args.target_member if args.edit_mode == "member" else None
            ),
            ive_performer_state_path=args.ive_performer_state,
            ive_performer_state_content_sha256=(
                args.ive_performer_state_content_sha256
            ),
        )
    result = build_stage_edit_feature_snapshots(
        args.input,
        song_id=args.song_id,
        output_directory=args.output_directory,
        expected_source_count=args.expected_source_count,
        negative_samples_per_hold=args.negative_samples_per_hold,
        boundary_guard_seconds=args.boundary_guard_seconds,
        max_sources=args.max_sources,
        working_width=args.working_width,
        style_features=style,
        causal_sensing=causal_sensing,
        edit_mode=args.edit_mode,
        target_member=args.target_member,
        inspection_only_without_causal_sensing=(
            args.inspection_only_without_causal_sensing
        ),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "snapshots": {
                    key: str(value) for key, value in result.snapshots.items()
                },
                "snapshot_sha256": dict(result.snapshot_sha256),
                "statistics": dict(result.statistics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ACTION_FORMATION_EXTRACTION_CONTRACT",
    "DEFAULT_BOUNDARY_GUARD_SECONDS",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_NEGATIVE_SAMPLES_PER_HOLD",
    "DEFAULT_VISUAL_WORKING_WIDTH",
    "FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS",
    "FEATURE_SNAPSHOT_JOB_INPUT_POLICY",
    "FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS",
    "FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION",
    "StageEditFeatureSnapshotJobResult",
    "build_stage_edit_feature_snapshots",
    "map_reference_master_to_source_seconds",
]
