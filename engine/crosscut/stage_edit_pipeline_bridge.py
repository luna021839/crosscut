"""Explicit production-only StageEdit generation bridge.

The normal visual editor remains the product baseline.  This module provides
one deliberately separate path for retaining learned candidates when a caller
supplies all three identities needed to make the inference auditable:

* an immutable :class:`StageEditFeatureSnapshot`,
* an unambiguous production-ready Dynamic Crop v2 checkpoint, and
* source ids explicitly aligned with the caller's media list.

There is no implicit activation and no fallback to random/untrained weights.
The bridge runs the public head materializer and semi-Markov decoder, then
reprojects every retained shot through the pipeline's checked source clocks.
It never selects or renders a learned candidate in place of the visual
baseline; promotion remains a later rendered-proxy review decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ive import AspectRatio
from .render import aligned_render_segment
from .section_director import SectionPlan
from .source_derivative_diversity import SourceDerivativeFamilyMetadata
from .stage_edit import StageEdit
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_head_materializer import (
    StageEditHeadMaterializationResult,
    materialize_stage_edit_head_frames_v2,
)
from .stage_edit_learned_section_director import (
    LearnedSectionDirectorResult,
    build_learned_section_plan_from_checkpoint,
)
from .stage_edit_semimarkov_beam import (
    StageEditSemiMarkovConfig,
    StageEditSemiMarkovResult,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditReadinessReport,
    StageEditTokenSchema,
    stage_edit_readiness_report,
)
from .timewarp import TimeWarpMap


STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION = 1
STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT = (
    "explicit_snapshot_dynamic_crop_v2_semimarkov_retention_v1"
)


class StageEditPipelineBridgeError(RuntimeError):
    """A requested learned-candidate generation path could not be honored."""


class StageEditPipelineBridgeNotReadyError(StageEditPipelineBridgeError):
    """The explicitly requested snapshot/checkpoint bundle is not ready."""


class StageEditPipelineBridgeInputError(StageEditPipelineBridgeError, ValueError):
    """The request media identities or clocks cannot be mapped exactly."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise StageEditPipelineBridgeInputError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditPipelineBridgeInputError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise StageEditPipelineBridgeInputError(f"{name} must be finite")
    return number


def _strict_source_ids(values: Sequence[str]) -> tuple[str, ...]:
    source_ids = tuple(str(value).strip() for value in values)
    if len(source_ids) < 2 or any(not value for value in source_ids):
        raise StageEditPipelineBridgeInputError(
            "StageEdit generation needs at least two explicit source ids"
        )
    if len(source_ids) != len(set(source_ids)):
        raise StageEditPipelineBridgeInputError(
            "StageEdit generation source ids must be unique"
        )
    return source_ids


def _snapshot_source_ids(snapshot: StageEditFeatureSnapshot) -> frozenset[str]:
    return frozenset(
        candidate.source_id
        for token in snapshot.sequence.tokens
        for candidate in token.source_candidates
    )


@dataclass(frozen=True, slots=True)
class StageEditPipelineBridgePreflight:
    snapshot_path: Path
    checkpoint_directory: Path
    source_ids: tuple[str, ...]
    snapshot: StageEditFeatureSnapshot
    readiness: StageEditReadinessReport

    def __post_init__(self) -> None:
        if not self.snapshot_path.is_absolute() or not self.checkpoint_directory.is_absolute():
            raise ValueError("StageEdit preflight paths must be absolute")
        if _snapshot_source_ids(self.snapshot) != frozenset(self.source_ids):
            raise ValueError("StageEdit preflight source identity changed")
        if not (
            self.readiness.architecture_available
            and self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
        ):
            raise ValueError("StageEdit preflight requires production readiness")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_path": str(self.snapshot_path),
            "snapshot_sha256": self.snapshot.snapshot_sha256,
            "snapshot_sequence_id": self.snapshot.sequence.sequence_id,
            "snapshot_token_count": len(self.snapshot.sequence.tokens),
            "checkpoint_directory": str(self.checkpoint_directory),
            "source_ids": list(self.source_ids),
            "readiness": self.readiness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RetainedStageEditCandidate:
    rank: int
    score: float
    edit: StageEdit
    exact_clock_reprojected: bool
    timewarped_segment_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "exact_clock_reprojected": self.exact_clock_reprojected,
            "timewarped_segment_count": self.timewarped_segment_count,
            "selected_for_render": False,
            "stage_edit": self.edit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StageEditPipelineBridgeResult:
    preflight: StageEditPipelineBridgePreflight
    head_materialization: StageEditHeadMaterializationResult
    semimarkov: StageEditSemiMarkovResult
    retained_candidates: tuple[RetainedStageEditCandidate, ...]
    selection_policy: Mapping[str, Any]
    source_derivative_diversity: SourceDerivativeFamilyMetadata | None = None
    learned_section_director: LearnedSectionDirectorResult | None = None
    contract: str = STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT
    schema_version: int = STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.retained_candidates:
            raise ValueError("StageEdit bridge must retain at least one candidate")
        if self.selection_policy.get("visual_baseline_replaced") is not False:
            raise ValueError("StageEdit bridge cannot replace the visual baseline")
        if self.selection_policy.get("selected_for_render") is not False:
            raise ValueError("StageEdit bridge cannot select an unreviewed candidate")
        if (
            self.source_derivative_diversity is not None
            and not self.source_derivative_diversity.production_ready
        ):
            raise ValueError(
                "StageEdit bridge requires production-ready source derivative metadata"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "activation": self.preflight.to_dict(),
            "selection_policy": dict(self.selection_policy),
            "source_derivative_diversity": (
                self.source_derivative_diversity.to_dict()
                if self.source_derivative_diversity is not None
                else {
                    "production_ready": False,
                    "status": "legacy_identity_default_not_automatic_group_safe",
                }
            ),
            "head_materialization": self.head_materialization.to_dict(),
            "learned_section_director": (
                self.learned_section_director.to_dict()
                if self.learned_section_director is not None
                else None
            ),
            "semimarkov": self.semimarkov.to_dict(),
            "retained_candidate_count": len(self.retained_candidates),
            "retained_candidates": [
                item.to_dict() for item in self.retained_candidates
            ],
        }

    def write_json(self, path: Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(
                f"stale StageEdit bridge temporary file: {temporary}"
            )
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        return destination


def preflight_stage_edit_pipeline_bridge(
    *,
    feature_snapshot_path: Path,
    checkpoint_directory: Path,
    source_ids: Sequence[str],
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditPipelineBridgePreflight:
    """Strict-load all explicit activation identities before expensive work."""

    snapshot_path = Path(feature_snapshot_path).expanduser().resolve()
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    canonical_source_ids = _strict_source_ids(source_ids)
    if not snapshot_path.is_file():
        raise StageEditPipelineBridgeNotReadyError(
            "configured StageEdit feature snapshot does not exist"
        )
    dynamic_metadata = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    if (
        not checkpoint.is_dir()
        or not dynamic_metadata.is_file()
        or (checkpoint / "metadata.json").exists()
    ):
        raise StageEditPipelineBridgeNotReadyError(
            "configured checkpoint must contain only Dynamic Crop v2 metadata"
        )
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("snapshot root must be an object")
        snapshot = StageEditFeatureSnapshot.from_dict(payload, schema=schema)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StageEditPipelineBridgeNotReadyError(
            f"configured StageEdit feature snapshot is invalid: {exc}"
        ) from exc
    observed_source_ids = _snapshot_source_ids(snapshot)
    configured_source_ids = frozenset(canonical_source_ids)
    if observed_source_ids != configured_source_ids:
        missing = sorted(observed_source_ids - configured_source_ids)
        extra = sorted(configured_source_ids - observed_source_ids)
        raise StageEditPipelineBridgeInputError(
            "configured source ids do not exactly match the feature snapshot "
            f"(snapshot_only={missing}, request_only={extra})"
        )
    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps > 0
    ):
        raise StageEditPipelineBridgeNotReadyError(
            "configured Dynamic Crop v2 checkpoint is not production-ready: "
            f"{readiness.status}: {readiness.reason}"
        )
    return StageEditPipelineBridgePreflight(
        snapshot_path=snapshot_path,
        checkpoint_directory=checkpoint,
        source_ids=canonical_source_ids,
        snapshot=snapshot,
        readiness=readiness,
    )


def _aligned_values(
    values: Sequence[Any],
    *,
    source_count: int,
    name: str,
) -> tuple[Any, ...]:
    result = tuple(values)
    if len(result) != source_count:
        raise StageEditPipelineBridgeInputError(
            f"{name} must align with every configured source"
        )
    return result


def _exact_clock_candidate(
    *,
    timeline: Any,
    sources: tuple[Path, ...],
    source_offsets: tuple[float, ...],
    source_durations: tuple[float, ...],
    internal_cuts: tuple[tuple[float, ...], ...],
    timewarps: tuple[TimeWarpMap | None, ...],
    section_plan: SectionPlan,
    config: StageEditSemiMarkovConfig,
) -> RetainedStageEditCandidate:
    """Reproject a decoded candidate and recheck source-local hard bounds."""

    shots = []
    timewarped_count = 0
    for decision, shot in zip(
        timeline.decisions, timeline.edit.shots, strict=True
    ):
        source_index = decision.source_index
        timewarp = timewarps[source_index]
        if timewarp is not None and not math.isclose(
            timewarp.global_offset_seconds,
            source_offsets[source_index],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise StageEditPipelineBridgeInputError(
                "StageEdit source timewarp and alignment offset disagree"
            )
        segment = aligned_render_segment(
            source_index=source_index,
            reference_start=decision.master_start_seconds,
            common_master_start=section_plan.start_seconds,
            source_offset=source_offsets[source_index],
            duration=decision.duration_seconds,
            editorial_transform=decision.editorial_transform,
            timewarp=timewarp,
        )
        if timewarp is not None and not timewarp.scalar_fallback:
            timewarped_count += 1
        if (
            segment.source_start < -1e-6
            or segment.source_end > source_durations[source_index] + 1e-6
        ):
            raise StageEditPipelineBridgeInputError(
                "exact StageEdit source clock leaves available media"
            )
        margin = config.internal_cut_exclusion_margin_seconds
        if any(
            segment.source_start + margin + 1e-6
            < cut
            < segment.source_end - margin - 1e-6
            for cut in internal_cuts[source_index]
        ):
            raise StageEditPipelineBridgeInputError(
                "exact StageEdit source clock crosses a native camera cut"
            )
        shots.append(replace(shot, segment=segment))
    if len(shots) != len(timeline.edit.shots):
        raise AssertionError("StageEdit exact-clock projection lost a shot")
    history = tuple(timeline.edit.candidate_history) + (
        {
            "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
            "visual_baseline_replaced": False,
            "selected_for_render": False,
            "exact_source_clock_reprojection": True,
            "timewarped_segment_count": timewarped_count,
        },
    )
    edit = replace(
        timeline.edit,
        sources=sources,
        shots=tuple(shots),
        candidate_history=history,
    )
    return RetainedStageEditCandidate(
        rank=timeline.rank,
        score=timeline.score,
        edit=edit,
        exact_clock_reprojected=True,
        timewarped_segment_count=timewarped_count,
    )


def run_stage_edit_pipeline_bridge(
    *,
    preflight: StageEditPipelineBridgePreflight,
    sources: Sequence[Path],
    source_durations_seconds: Sequence[float],
    source_start_offsets_seconds: Sequence[float],
    internal_cut_times_seconds: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_quality_log_priors: Sequence[float],
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None = None,
    require_verified_source_derivative_metadata: bool = False,
    section_plan: SectionPlan,
    output: Path,
    reference_source: int,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    width: int = 1920,
    height: int = 1080,
    fps: float = 29.97,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    config: StageEditSemiMarkovConfig = StageEditSemiMarkovConfig(),
) -> StageEditPipelineBridgeResult:
    """Materialize, decode, and retain learned candidates without selecting them."""

    normalized_sources = tuple(Path(value).expanduser().resolve() for value in sources)
    source_count = len(preflight.source_ids)
    if len(normalized_sources) != source_count or len(set(normalized_sources)) != source_count:
        raise StageEditPipelineBridgeInputError(
            "configured StageEdit source ids must align one-to-one with media"
        )
    durations = tuple(
        _finite(value, f"source_durations_seconds[{index}]")
        for index, value in enumerate(
            _aligned_values(
                source_durations_seconds,
                source_count=source_count,
                name="source durations",
            )
        )
    )
    if any(value <= 0.0 for value in durations):
        raise StageEditPipelineBridgeInputError("source durations must be positive")
    offsets = tuple(
        _finite(value, f"source_start_offsets_seconds[{index}]")
        for index, value in enumerate(
            _aligned_values(
                source_start_offsets_seconds,
                source_count=source_count,
                name="source offsets",
            )
        )
    )
    raw_cuts = _aligned_values(
        internal_cut_times_seconds,
        source_count=source_count,
        name="internal source cuts",
    )
    cuts = tuple(
        tuple(
            sorted(
                _finite(value, f"internal_cut_times_seconds[{index}]")
                for value in values
            )
        )
        for index, values in enumerate(raw_cuts)
    )
    timewarps = tuple(
        _aligned_values(
            source_timewarps,
            source_count=source_count,
            name="source timewarps",
        )
    )
    if any(value is not None and not isinstance(value, TimeWarpMap) for value in timewarps):
        raise StageEditPipelineBridgeInputError(
            "source timewarps must be checked TimeWarpMap values"
        )
    quality_priors = tuple(
        _finite(value, f"source_quality_log_priors[{index}]")
        for index, value in enumerate(
            _aligned_values(
                source_quality_log_priors,
                source_count=source_count,
                name="source quality priors",
            )
        )
    )
    if preflight.snapshot.token_schema_fingerprint != schema.fingerprint:
        raise StageEditPipelineBridgeInputError(
            "StageEdit bridge schema differs from preflight snapshot"
        )
    if not isinstance(require_verified_source_derivative_metadata, bool):
        raise StageEditPipelineBridgeInputError(
            "require_verified_source_derivative_metadata must be a boolean"
        )
    if source_derivative_metadata is None and require_verified_source_derivative_metadata:
        raise StageEditPipelineBridgeInputError(
            "automatic GROUP StageEdit bridge requires verified source derivative metadata"
        )
    if source_derivative_metadata is not None and not isinstance(
        source_derivative_metadata, SourceDerivativeFamilyMetadata
    ):
        raise StageEditPipelineBridgeInputError(
            "source derivative metadata has an invalid type"
        )
    if (
        source_derivative_metadata is not None
        and source_derivative_metadata.source_count != source_count
    ):
        raise StageEditPipelineBridgeInputError(
            "source derivative metadata count does not match StageEdit sources"
        )
    if (
        source_derivative_metadata is not None
        and not source_derivative_metadata.production_ready
    ):
        raise StageEditPipelineBridgeInputError(
            "source derivative classification is uncertain or clock-incomplete"
        )

    heads = materialize_stage_edit_head_frames_v2(
        checkpoint_directory=preflight.checkpoint_directory,
        sequence=preflight.snapshot.sequence,
        canonical_source_ids=preflight.source_ids,
        schema=schema,
    )
    learned_section_director = build_learned_section_plan_from_checkpoint(
        section_plan,
        heads.frames,
        checkpoint_directory=preflight.checkpoint_directory,
    )
    learned_section_plan = learned_section_director.section_plan
    metadata = tuple(
        StageEditSourceMetadata(
            source_index=index,
            source_id=source_id,
            source_duration_seconds=durations[index],
            source_start_offset_seconds=offsets[index],
            internal_cut_times_seconds=cuts[index],
            maximum_occupancy_fraction=config.maximum_source_occupancy_fraction,
            quality_log_prior=quality_priors[index],
        )
        for index, source_id in enumerate(preflight.source_ids)
    )
    decoded = decode_stage_edit_semimarkov_beam(
        checkpoint_directory=preflight.checkpoint_directory,
        sources=normalized_sources,
        source_metadata=metadata,
        head_frames=heads.frames,
        section_plan=learned_section_plan,
        output=Path(output),
        schema=schema,
        config=config,
        reference_source=reference_source,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        fps=fps,
        source_derivative_metadata=source_derivative_metadata,
    )
    if not decoded.audit.request_fulfilled:
        raise StageEditPipelineBridgeError(
            "requested StageEdit candidate count was not satisfied; visual baseline "
            "was not used as a silent learned-inference fallback"
        )
    retained = tuple(
        _exact_clock_candidate(
            timeline=timeline,
            sources=normalized_sources,
            source_offsets=offsets,
            source_durations=durations,
            internal_cuts=cuts,
            timewarps=timewarps,
            section_plan=learned_section_plan,
            config=config,
        )
        for timeline in decoded.timelines
    )
    if len(retained) != config.timeline_count:
        raise StageEditPipelineBridgeError(
            "exact StageEdit clock validation did not retain every requested candidate"
        )
    return StageEditPipelineBridgeResult(
        preflight=preflight,
        head_materialization=heads,
        semimarkov=decoded,
        retained_candidates=retained,
        source_derivative_diversity=source_derivative_metadata,
        learned_section_director=learned_section_director,
        selection_policy={
            "selected_for_render": False,
            "visual_baseline_replaced": False,
            "retention": "parallel_unrendered_stage_edit_candidates",
            "promotion_gate": "complete_rendered_proxy_review_required",
            "failure_policy": "explicit_activation_fails_closed_no_silent_fallback",
            "source_derivative_policy": (
                "verified_actual_pts_families_required"
                if require_verified_source_derivative_metadata
                else "legacy_identity_default_not_automatic_group_safe"
            ),
            "section_director": (
                "validation_frozen_transformer_heads_before_semimarkov"
            ),
        },
    )


__all__ = [
    "STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT",
    "STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION",
    "RetainedStageEditCandidate",
    "StageEditPipelineBridgeError",
    "StageEditPipelineBridgeInputError",
    "StageEditPipelineBridgeNotReadyError",
    "StageEditPipelineBridgePreflight",
    "StageEditPipelineBridgeResult",
    "preflight_stage_edit_pipeline_bridge",
    "run_stage_edit_pipeline_bridge",
]
