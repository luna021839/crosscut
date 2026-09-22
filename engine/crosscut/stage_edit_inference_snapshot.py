"""Production StageEdit feature snapshots from the current synchronized media.

Reference feature jobs start from finished-edit supervision.  The automatic app
has no such artifact: after acquisition it owns only the current source files,
their checked clocks, rhythm, and section plan.  This module materializes the
same target-free token schema directly from those inference-time inputs.

Every output is content addressed.  Source ids bind request order to complete
media SHA-256 identities, while decoded pixels and appearance embeddings are
never written.  A registered model therefore cannot accidentally consume a
snapshot made for a different source pool or clock solution.
"""

from __future__ import annotations
from .optional_native_worker import isolated_optional

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .audio import AlignmentResult, SourceAlignment
from .draft import GroupDraftTimeline
from .rhythm import RhythmAnalysis
from .section_director import SectionPlan
from .stage_edit_feature_materializer import (
    SourceFeatureObservation,
    StageEditCausalSensingArtifacts,
    StageEditFeatureFrame,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
    materialize_stage_edit_feature_snapshot,
    source_observation_from_visual_cache_payload,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_DECISION_HISTORY_OWNER,
    STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT,
    mask_stage_edit_recent_history_for_semimarkov,
    stage_edit_recent_history_mask_contract,
)
from .timewarp import TimeWarpMap
from .visual_score import (
    _VideoSampler,
    frame_visual_numeric_features,
    visual_analyzer_fingerprint_sha256,
)


STAGE_EDIT_INFERENCE_SNAPSHOT_SCHEMA_VERSION = 2
STAGE_EDIT_INFERENCE_SNAPSHOT_CONTRACT = (
    "actual_synchronized_media_inference_snapshot_v2"
)
DEFAULT_INFERENCE_CANDIDATE_SPACING_SECONDS = 0.50
DEFAULT_INFERENCE_VISUAL_WORKING_WIDTH = 480
MINIMUM_PER_SOURCE_AVAILABILITY_RATIO = 0.95
_SAFE_SONG = re.compile(r"[^0-9A-Za-zㄱ-ㆎ가-힣._-]+")


class StageEditInferenceSnapshotError(RuntimeError):
    """The current synchronized media cannot produce a safe model snapshot."""


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


def _stable_file_sha256(path: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise StageEditInferenceSnapshotError(
            f"source changed while hashing: {resolved}"
        )
    return digest.hexdigest()


def _safe_song(value: str) -> str:
    normalized = _SAFE_SONG.sub("-", str(value).strip()).strip("-._")
    return normalized[:96] or "song"


def _canonical_time(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise StageEditInferenceSnapshotError(
            "StageEdit inference candidate time is not finite"
        )
    return round(number, 6)


def build_stage_edit_inference_decision_times(
    *,
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
    maximum_sequence_length: int,
    maximum_spacing_seconds: float = DEFAULT_INFERENCE_CANDIDATE_SPACING_SECONDS,
) -> tuple[float, ...]:
    """Build a dense causal candidate grid in the canonical master clock."""

    if not isinstance(timing_draft, GroupDraftTimeline):
        raise TypeError("timing_draft must use GroupDraftTimeline")
    if not isinstance(rhythm, RhythmAnalysis):
        raise TypeError("rhythm must use RhythmAnalysis")
    if not isinstance(section_plan, SectionPlan):
        raise TypeError("section_plan must use SectionPlan")
    if (
        isinstance(maximum_sequence_length, bool)
        or not isinstance(maximum_sequence_length, int)
        or maximum_sequence_length < 2
    ):
        raise ValueError("maximum_sequence_length must be at least two")
    spacing = float(maximum_spacing_seconds)
    if not math.isfinite(spacing) or not 0.20 <= spacing <= 1.0:
        raise ValueError("maximum inference candidate spacing must be 0.20...1.0")
    start = section_plan.start_seconds
    end = section_plan.end_seconds
    if (
        abs(start - timing_draft.common_master_start) > 1e-5
        or abs(end - timing_draft.common_master_end) > 1e-5
    ):
        raise StageEditInferenceSnapshotError(
            "StageEdit timing draft and section plan clocks differ"
        )

    candidates: set[float] = {_canonical_time(start)}

    def add(value: float) -> None:
        timestamp = _canonical_time(value)
        if start - 1e-6 <= timestamp < end - 1e-6:
            candidates.add(timestamp)

    for section in section_plan.sections[:-1]:
        add(section.end_seconds)
    for segment in timing_draft.segments:
        add(segment.canonical_master_start)
        add(segment.canonical_master_end)
    beats = tuple(
        beat.time_seconds
        for beat in rhythm.beats
        if start - 1e-6 <= beat.time_seconds <= end + 1e-6
    )
    for beat_time in beats:
        add(beat_time)
    for left, right in zip(beats, beats[1:]):
        # Half-beats preserve professional off/down-beat choices without a
        # fixed 4 Hz grid ballooning long songs beyond the 2048-token model.
        add((left + right) / 2.0)
    for phrase in rhythm.phrase_candidates:
        add(phrase.start_seconds)
        add(phrase.end_seconds)

    # Rhythm extraction can have sparse intro/outro evidence.  Fill only those
    # holes so the semi-Markov maximum-duration constraint remains satisfiable.
    cursor = start
    while cursor + spacing < end - 1e-6:
        cursor += spacing
        add(cursor)

    ordered = tuple(sorted(candidates))
    if not ordered or abs(ordered[0] - start) > 1e-5:
        raise StageEditInferenceSnapshotError(
            "StageEdit inference grid does not start at the section plan"
        )
    required_boundaries = tuple(
        _canonical_time(section.end_seconds)
        for section in section_plan.sections[:-1]
    )
    if any(boundary not in ordered for boundary in required_boundaries):
        raise StageEditInferenceSnapshotError(
            "StageEdit inference grid lost a required section boundary"
        )
    if len(ordered) > maximum_sequence_length:
        raise StageEditInferenceSnapshotError(
            "actual song candidate grid exceeds the registered Transformer "
            f"context ({len(ordered)} > {maximum_sequence_length})"
        )
    if any(right - left > spacing + 1e-5 for left, right in zip(ordered, ordered[1:])):
        raise StageEditInferenceSnapshotError(
            "StageEdit inference grid has an unsafe candidate gap"
        )
    if end - ordered[-1] > spacing + 1e-5:
        raise StageEditInferenceSnapshotError(
            "StageEdit inference grid leaves an unsafe terminal gap"
        )
    return ordered


def _source_time(
    master_time: float,
    *,
    alignment: SourceAlignment,
    timewarp: TimeWarpMap | None,
) -> float:
    if timewarp is None:
        return master_time + alignment.offset_seconds
    if not math.isclose(
        timewarp.global_offset_seconds,
        alignment.offset_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise StageEditInferenceSnapshotError(
            "StageEdit timewarp and coarse alignment offset disagree"
        )
    return timewarp.master_to_source(master_time)


def _sync_residual_normalized(timewarp: TimeWarpMap | None) -> float | None:
    if timewarp is None or timewarp.residuals.p95_seconds is None:
        return None
    maximum = timewarp.config.maximum_trend_residual_seconds
    if maximum <= 0.0:
        return None
    return min(1.0, max(0.0, timewarp.residuals.p95_seconds / maximum))


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    destination = Path(path)
    if destination.is_symlink():
        raise FileExistsError(f"immutable StageEdit artifact collision: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != encoded:
            raise FileExistsError(
                f"immutable StageEdit artifact collision: {destination}"
            )
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"stale StageEdit inference temporary file: {temporary}"
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
        temporary.replace(destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return destination.resolve()


@dataclass(frozen=True, slots=True)
class StageEditInferenceSnapshotResult:
    snapshot: StageEditFeatureSnapshot
    snapshot_path: Path
    manifest_path: Path
    source_ids: tuple[str, ...]
    source_content_sha256: tuple[str, ...]
    decision_times: tuple[float, ...]
    statistics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.source_ids) < 2 or len(self.source_ids) != len(
            self.source_content_sha256
        ):
            raise ValueError("inference snapshot source identities are invalid")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("inference snapshot source ids must be unique")
        if len(self.decision_times) != len(self.snapshot.sequence.tokens):
            raise ValueError("inference decision times and snapshot tokens differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": STAGE_EDIT_INFERENCE_SNAPSHOT_CONTRACT,
            "snapshot_path": str(self.snapshot_path),
            "snapshot_sha256": self.snapshot.snapshot_sha256,
            "manifest_path": str(self.manifest_path),
            "source_ids": list(self.source_ids),
            "source_content_sha256": list(self.source_content_sha256),
            "decision_time_count": len(self.decision_times),
            "statistics": dict(self.statistics),
        }


@isolated_optional()
def materialize_stage_edit_inference_snapshot(
    *,
    song_id: str,
    sources: Sequence[Path],
    alignment: AlignmentResult,
    alignment_hard_gates: Sequence[bool],
    source_timewarps: Sequence[TimeWarpMap | None],
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
    output_directory: Path,
    feature_cache_directory: Path | None,
    feature_derived_cache_directory: Path | None = None,
    require_persistent_feature_cache: bool = False,
    maximum_sequence_length: int,
    maximum_sources: int,
    minimum_source_availability_ratio: float = (
        MINIMUM_PER_SOURCE_AVAILABILITY_RATIO
    ),
    causal_sensing: StageEditCausalSensingArtifacts | None = None,
    edit_mode: object = "group",
    target_member: object | None = None,
) -> StageEditInferenceSnapshotResult:
    """Decode target-free model inputs from one actual synchronized source pool."""

    normalized_song = str(song_id).strip()
    normalized_sources = tuple(
        Path(path).expanduser().resolve(strict=True) for path in sources
    )
    alignments = tuple(alignment.alignments)
    hard_gates = tuple(alignment_hard_gates)
    timewarps = tuple(source_timewarps)
    source_count = len(normalized_sources)
    if not normalized_song or source_count < 2:
        raise ValueError("song_id and at least two sources are required")
    if not isinstance(causal_sensing, StageEditCausalSensingArtifacts):
        raise StageEditInferenceSnapshotError(
            "registered automatic StageEdit inference requires the complete "
            "shooting-grammar/member-state/performer-state artifact triple"
        )
    try:
        causal_sensing.binding.require_production_scope(
            edit_mode=edit_mode,
            target_member=target_member,
        )
    except (TypeError, ValueError) as exc:
        raise StageEditInferenceSnapshotError(
            f"registered automatic StageEdit causal sensing is invalid: {exc}"
        ) from exc
    if source_count > maximum_sources:
        raise StageEditInferenceSnapshotError(
            "actual source count exceeds the registered Transformer limit"
        )
    if not (
        len(alignments) == len(hard_gates) == len(timewarps) == source_count
    ):
        raise StageEditInferenceSnapshotError(
            "StageEdit inference source clocks do not align with media"
        )
    if tuple(Path(item.source).resolve() for item in alignments) != normalized_sources:
        raise StageEditInferenceSnapshotError(
            "StageEdit alignment identities differ from current source paths"
        )
    if any(not isinstance(value, bool) for value in hard_gates):
        raise TypeError("alignment_hard_gates must contain booleans")
    if not all(hard_gates):
        raise StageEditInferenceSnapshotError(
            "registered Transformer requires every current source to pass "
            "the synchronization hard gate"
        )
    availability_floor = float(minimum_source_availability_ratio)
    if not math.isfinite(availability_floor) or not 0.90 <= availability_floor <= 1.0:
        raise ValueError("minimum source availability ratio must be 0.90...1.0")

    decision_times = build_stage_edit_inference_decision_times(
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        maximum_sequence_length=maximum_sequence_length,
    )
    cache_directory = (
        Path(feature_cache_directory).expanduser().resolve()
        if feature_cache_directory is not None
        else None
    )
    derived_cache_directory = (
        Path(feature_derived_cache_directory).expanduser()
        if feature_derived_cache_directory is not None
        else None
    )
    if not isinstance(require_persistent_feature_cache, bool):
        raise TypeError("require_persistent_feature_cache must be a boolean")
    if require_persistent_feature_cache and derived_cache_directory is None:
        raise ValueError(
            "require_persistent_feature_cache needs an explicit "
            "feature_derived_cache_directory"
        )
    sampler_arguments: dict[str, Any] = {}
    if cache_directory is not None:
        sampler_arguments["feature_cache_directory"] = cache_directory
    if derived_cache_directory is not None:
        sampler_arguments["feature_derived_cache_directory"] = (
            derived_cache_directory
        )
    if require_persistent_feature_cache:
        sampler_arguments["require_persistent_feature_cache"] = True
    samplers: list[_VideoSampler] = []
    try:
        for path in normalized_sources:
            samplers.append(_VideoSampler(path, **sampler_arguments))
        content_hashes = tuple(
            (
                sampler.persistent_feature_cache.source_sha256
                if sampler.persistent_feature_cache is not None
                else _stable_file_sha256(source)
            )
            for sampler, source in zip(samplers, normalized_sources, strict=True)
        )
        source_ids = tuple(
            f"source-{index:02d}-{digest[:16]}"
            for index, digest in enumerate(content_hashes)
        )
        observations_by_time: dict[float, list[SourceFeatureObservation]] = {
            timestamp: [] for timestamp in decision_times
        }
        feature_identity = hashlib.sha256()
        source_statistics: list[dict[str, Any]] = []
        for source_index, (
            source,
            source_id,
            sampler,
            source_alignment,
            timewarp,
        ) in enumerate(
            zip(
                normalized_sources,
                source_ids,
                samplers,
                alignments,
                timewarps,
                strict=True,
            )
        ):
            available_count = 0
            for timestamp in decision_times:
                local_time = _source_time(
                    timestamp,
                    alignment=source_alignment,
                    timewarp=timewarp,
                )
                following_time = local_time + 1.0 / sampler.frame_rate
                descriptor = None
                following_descriptor = None
                if local_time >= 0.0:
                    descriptor = sampler.feature_descriptor_at(
                        local_time,
                        following_seconds=following_time,
                        include_subject_geometry=True,
                        include_formation_pose=False,
                    )
                    following_descriptor = sampler.feature_descriptor_at(
                        following_time,
                        include_subject_geometry=True,
                        include_formation_pose=False,
                    )
                numeric = (
                    frame_visual_numeric_features(descriptor)
                    if descriptor is not None
                    else {}
                )
                following_numeric = (
                    frame_visual_numeric_features(following_descriptor)
                    if following_descriptor is not None
                    else {}
                )
                available = descriptor is not None
                if available:
                    available_count += 1
                observation = source_observation_from_visual_cache_payload(
                    source_id,
                    numeric,
                    following_numeric_features=following_numeric,
                    interval_seconds=(
                        1.0 / sampler.frame_rate if available else None
                    ),
                    sync_confidence=source_alignment.score,
                    sync_residual_normalized=_sync_residual_normalized(timewarp),
                    available=available,
                    hard_gate_passed=available,
                )
                observations_by_time[timestamp].append(observation)
                feature_identity.update(
                    _canonical_bytes(
                        {
                            "source_id": source_id,
                            "master_time_seconds": timestamp,
                            "source_time_seconds": local_time,
                            "numeric_features": numeric,
                            "following_numeric_features": following_numeric,
                            "available": available,
                            "hard_gate_passed": available,
                        }
                    )
                )
            ratio = available_count / len(decision_times)
            if ratio + 1e-12 < availability_floor:
                raise StageEditInferenceSnapshotError(
                    f"source {source.name} supplied only {ratio:.1%} of "
                    "registered Transformer decision frames"
                )
            source_statistics.append(
                {
                    "source_index": source_index,
                    "source_id": source_id,
                    "path": str(source),
                    "content_sha256": content_hashes[source_index],
                    "frame_rate": sampler.frame_rate,
                    "decision_time_count": len(decision_times),
                    "available_time_count": available_count,
                    "availability_ratio": ratio,
                    "alignment_score": source_alignment.score,
                    "alignment_score_margin": source_alignment.score_margin,
                    "alignment_overlap_ratio": source_alignment.overlap_ratio,
                    "timewarp": (
                        timewarp.to_dict() if timewarp is not None else None
                    ),
                }
            )

        if any(
            sum(item.hard_gate_passed for item in observations_by_time[timestamp])
            < 2
            for timestamp in decision_times
        ):
            raise StageEditInferenceSnapshotError(
                "fewer than two sources survive a StageEdit decision timestamp"
            )

        alignment_identity = {
            "alignment": alignment.to_dict(),
            "alignment_hard_gates": list(hard_gates),
            "source_content": source_statistics,
        }
        provenance = StageEditFeatureProvenance(
            upstream_sha256=(
                ("source_alignment", _canonical_sha256(alignment_identity)),
                ("visual_feature_cache", feature_identity.hexdigest()),
                (
                    "visual_score",
                    _canonical_sha256(
                        {
                            "working_width": DEFAULT_INFERENCE_VISUAL_WORKING_WIDTH,
                            "include_subject_geometry": True,
                            "include_formation_pose": False,
                            "analyzer_fingerprint_sha256": (
                                visual_analyzer_fingerprint_sha256()
                            ),
                        }
                    ),
                ),
                ("rhythm_analysis", _canonical_sha256(rhythm.to_dict())),
                ("section_plan", _canonical_sha256(section_plan.to_dict())),
            ),
            analyzer_fingerprint_sha256=visual_analyzer_fingerprint_sha256(),
            contracts=(
                STAGE_EDIT_INFERENCE_SNAPSHOT_CONTRACT,
                "current_song_sources_and_checked_clocks_only",
                "no_reference_match_or_editorial_choice_labels",
                STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT,
                STAGE_EDIT_DECISION_HISTORY_OWNER,
                "source_content_sha256_bound_ids",
            ),
        )
        frames = tuple(
            StageEditFeatureFrame(
                master_time_seconds=timestamp,
                source_observations=tuple(observations_by_time[timestamp]),
                # No baseline, reference, or learned choice is supplied.  The
                # shared post-materialization policy below also masks the one
                # cold-start statistic emitted by an empty history.
                causal_history=(),
            )
            for timestamp in decision_times
        )
        unmasked_snapshot = materialize_stage_edit_feature_snapshot(
            f"stageedit-inference:IVE:{normalized_song}:{_canonical_sha256(alignment_identity)[:16]}",
            frames,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            rhythm=rhythm,
            section_plan=section_plan,
            style_features=None,
            performance_start_seconds=section_plan.start_seconds,
            performance_end_seconds=section_plan.end_seconds,
            provenance=provenance,
            causal_sensing=causal_sensing,
            production_edit_mode=edit_mode,
            production_target_member=target_member,
        )
        snapshot = StageEditFeatureSnapshot.build(
            mask_stage_edit_recent_history_for_semimarkov(
                unmasked_snapshot.sequence,
                DEFAULT_STAGE_EDIT_SCHEMA,
            ),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            provenance=unmasked_snapshot.provenance,
            causal_sensing_binding=unmasked_snapshot.causal_sensing_binding,
        )
    finally:
        for sampler in samplers:
            sampler.close()

    destination = Path(output_directory).expanduser().resolve()
    artifact_id = f"IVE-{_safe_song(normalized_song)}"
    snapshot_path = _write_immutable_json(
        destination
        / (
            f"{artifact_id}-stageedit-inference-snapshot-"
            f"{snapshot.snapshot_sha256[:16]}.json"
        ),
        snapshot.to_dict(),
    )
    statistics: dict[str, Any] = {
        "source_count": source_count,
        "decision_time_count": len(decision_times),
        "minimum_source_availability_ratio": availability_floor,
        "per_source": source_statistics,
    }
    manifest_unsigned: dict[str, Any] = {
        "schema_version": STAGE_EDIT_INFERENCE_SNAPSHOT_SCHEMA_VERSION,
        "contract": STAGE_EDIT_INFERENCE_SNAPSHOT_CONTRACT,
        "song_id": normalized_song,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot.snapshot_sha256,
        "source_ids": list(source_ids),
        "source_content_sha256": list(content_hashes),
        "decision_times_seconds": list(decision_times),
        "feature_provenance": snapshot.provenance.to_dict(),
        "causal_sensing_binding": snapshot.causal_sensing_binding.to_dict(),
        "production_causal_sensing_gate": (
            snapshot.require_production_causal_sensing(
                edit_mode=edit_mode,
                target_member=target_member,
            )
        ),
        "decision_history_input_contract": (
            stage_edit_recent_history_mask_contract(DEFAULT_STAGE_EDIT_SCHEMA)
        ),
        "statistics": statistics,
        "input_policy": {
            "actual_synchronized_media_only": True,
            "reference_attribution_inputs_present": False,
            "selected_labels_present": False,
            "current_or_future_learned_decisions_present": False,
            "reference_or_baseline_decision_history_present": False,
            "transformer_recent_edit_slice_fully_missing_masked": True,
            "decision_history_owner": STAGE_EDIT_DECISION_HISTORY_OWNER,
            "raw_frames_persisted": False,
            "face_identity_embeddings_persisted": False,
            "source_content_sha256_bound": True,
            "all_sources_passed_alignment_hard_gate": True,
            "causal_sensing_exact_timestamp_and_content_hash_bound": True,
            "member_state_unknown_preserved": True,
            "ive_performer_distribution_unknown_preserved": True,
        },
    }
    manifest_sha = _canonical_sha256(manifest_unsigned)
    manifest = {**manifest_unsigned, "manifest_sha256": manifest_sha}
    manifest_path = _write_immutable_json(
        destination
        / f"{artifact_id}-stageedit-inference-manifest-{manifest_sha[:16]}.json",
        manifest,
    )

    restored = StageEditFeatureSnapshot.from_dict(
        json.loads(snapshot_path.read_text(encoding="utf-8")),
        schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )
    if restored.snapshot_sha256 != snapshot.snapshot_sha256:
        raise StageEditInferenceSnapshotError(
            "written StageEdit inference snapshot failed strict round-trip"
        )
    restored.require_production_causal_sensing(
        edit_mode=edit_mode,
        target_member=target_member,
    )
    return StageEditInferenceSnapshotResult(
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        source_ids=source_ids,
        source_content_sha256=content_hashes,
        decision_times=decision_times,
        statistics=statistics,
    )


__all__ = [
    "DEFAULT_INFERENCE_CANDIDATE_SPACING_SECONDS",
    "MINIMUM_PER_SOURCE_AVAILABILITY_RATIO",
    "STAGE_EDIT_INFERENCE_SNAPSHOT_CONTRACT",
    "STAGE_EDIT_INFERENCE_SNAPSHOT_SCHEMA_VERSION",
    "StageEditInferenceSnapshotError",
    "StageEditInferenceSnapshotResult",
    "build_stage_edit_inference_decision_times",
    "materialize_stage_edit_inference_snapshot",
]
