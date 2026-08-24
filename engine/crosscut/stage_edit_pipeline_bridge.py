"""Explicit production-only StageEdit generation bridge.

The normal visual editor remains the product baseline.  This module provides
one deliberately separate path for retaining learned candidates when a caller
supplies all three identities needed to make the inference auditable:

* an immutable :class:`StageEditFeatureSnapshot`,
* an unambiguous production-ready Dynamic Crop v2 or v3 checkpoint, and
* source ids explicitly aligned with the caller's media list.

There is no implicit activation and no fallback to random/untrained weights.
The bridge runs the public head materializer and semi-Markov decoder, then
reprojects every retained shot through the pipeline's checked source clocks.
It never selects or renders a learned candidate in place of the visual
baseline; promotion remains a later rendered-proxy review decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
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
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_head_materializer import (
    StageEditHeadMaterializationResult,
    materialize_stage_edit_head_frames_v2,
    materialize_stage_edit_head_frames_v3,
)
from .stage_edit_learned_section_director import (
    LearnedSectionDirectorResult,
    build_learned_section_plan_from_checkpoint,
)
from .stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditBeamTimeline,
    StageEditDecodeAudit,
    StageEditSemiMarkovConfig,
    StageEditSemiMarkovResult,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    STAGE_EDIT_CONTINUITY_AUTHORITIES,
)
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairCheckpointError,
    StageEditSourcePairScorer,
    load_stage_edit_source_pair_checkpoint,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditReadinessReport,
    StageEditTokenSchema,
    stage_edit_readiness_report,
)
from .stage_edit_temporal_pose_motion import (
    require_scalar_stage_edit_pose_payload,
)
from .timewarp import TimeWarpMap


STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION = 4
STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT = (
    "explicit_snapshot_dynamic_crop_v2_v3_profile_chorus_anchor_"
    "continuity_authority_temporal_pose_sidecar_semimarkov_retention_v4"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_identity(value: str | None, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise StageEditPipelineBridgeInputError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class StageEditPipelineBridgePreflight:
    snapshot_path: Path
    checkpoint_directory: Path
    source_ids: tuple[str, ...]
    snapshot: StageEditFeatureSnapshot
    readiness: StageEditReadinessReport
    dynamic_crop_version: int = 2
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY
    source_pair_scorer: StageEditSourcePairScorer | None = None
    source_pair_scorer_path: Path | None = None
    source_pair_scorer_file_sha256: str | None = None
    source_pair_scorer_fingerprint: str | None = None
    source_pair_checkpoint_sha256: str | None = None
    source_pair_corpus_sha256: str | None = None

    def _validate_learned_scorer_identity(self) -> None:
        if self.continuity_authority != PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
            return
        scorer = self.source_pair_scorer
        scorer_path = self.source_pair_scorer_path
        if not isinstance(scorer, StageEditSourcePairScorer):
            raise StageEditPipelineBridgeNotReadyError(
                "learned preflight source-pair scorer is not typed"
            )
        if not isinstance(scorer_path, Path) or not scorer_path.is_absolute():
            raise StageEditPipelineBridgeNotReadyError(
                "learned preflight scorer path must be absolute"
            )
        if (
            scorer_path.is_symlink()
            or scorer_path.resolve() != scorer_path
            or scorer_path.name != STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
            or not scorer_path.is_file()
        ):
            raise StageEditPipelineBridgeNotReadyError(
                "learned preflight scorer must be one canonical regular checkpoint"
            )
        file_sha256 = _sha256_identity(
            self.source_pair_scorer_file_sha256,
            "source_pair_scorer_file_sha256",
        )
        scorer_fingerprint = _sha256_identity(
            self.source_pair_scorer_fingerprint,
            "source_pair_scorer_fingerprint",
        )
        checkpoint_sha256 = _sha256_identity(
            self.source_pair_checkpoint_sha256,
            "source_pair_checkpoint_sha256",
        )
        corpus_sha256 = _sha256_identity(
            self.source_pair_corpus_sha256,
            "source_pair_corpus_sha256",
        )
        try:
            before_sha256 = _sha256_file(scorer_path)
            loaded_scorer, loaded_checkpoint = (
                load_stage_edit_source_pair_checkpoint(
                    scorer_path.parent,
                    expected_corpus_sha256=corpus_sha256,
                    expected_checkpoint_sha256=checkpoint_sha256,
                )
            )
            after_sha256 = _sha256_file(scorer_path)
        except (OSError, StageEditSourcePairCheckpointError) as exc:
            raise StageEditPipelineBridgeNotReadyError(
                f"learned preflight source-pair scorer is invalid: {exc}"
            ) from exc
        if before_sha256 != file_sha256 or after_sha256 != file_sha256:
            raise StageEditPipelineBridgeNotReadyError(
                "learned preflight source-pair scorer file hash changed"
            )
        if (
            loaded_checkpoint.checkpoint_sha256 != checkpoint_sha256
            or loaded_checkpoint.corpus_sha256 != corpus_sha256
            or loaded_scorer.contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
            or loaded_scorer.fingerprint != scorer_fingerprint
            or loaded_scorer.to_dict() != scorer.to_dict()
        ):
            raise StageEditPipelineBridgeNotReadyError(
                "learned preflight source-pair scorer identity changed"
            )

    def __post_init__(self) -> None:
        if not self.snapshot_path.is_absolute() or not self.checkpoint_directory.is_absolute():
            raise ValueError("StageEdit preflight paths must be absolute")
        if _snapshot_source_ids(self.snapshot) != frozenset(self.source_ids):
            raise ValueError("StageEdit preflight source identity changed")
        if not (
            self.readiness.architecture_available
            and self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
            and self.readiness.checkpoint_path
            == str(self.checkpoint_directory.resolve())
            and self.readiness.training_steps > 0
        ):
            raise ValueError("StageEdit preflight requires production readiness")
        if self.dynamic_crop_version not in (2, 3):
            raise ValueError("StageEdit preflight Dynamic Crop version is invalid")
        expected_format = f"dynamic_crop_v{self.dynamic_crop_version}"
        expected_metadata = (
            DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
            if self.dynamic_crop_version == 3
            else DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
        )
        if (
            self.readiness.checkpoint_format != expected_format
            or self.readiness.dynamic_crop_version != self.dynamic_crop_version
            or self.readiness.checkpoint_metadata_filename != expected_metadata
        ):
            raise ValueError("StageEdit preflight readiness version changed")
        if self.continuity_authority not in STAGE_EDIT_CONTINUITY_AUTHORITIES:
            raise ValueError("StageEdit preflight continuity authority is invalid")
        pair_values = (
            self.source_pair_scorer,
            self.source_pair_scorer_path,
            self.source_pair_scorer_file_sha256,
            self.source_pair_scorer_fingerprint,
            self.source_pair_checkpoint_sha256,
            self.source_pair_corpus_sha256,
        )
        if self.continuity_authority == HANDCRAFTED_CONTINUITY_AUTHORITY:
            if any(value is not None for value in pair_values):
                raise ValueError(
                    "handcrafted continuity authority cannot bind a learned scorer"
                )
        else:
            if self.dynamic_crop_version != 3:
                raise ValueError(
                    "learned source-pair continuity requires Dynamic Crop v3"
                )
            if any(value is None for value in pair_values):
                raise ValueError(
                    "learned continuity authority requires a complete scorer identity"
                )
            if not isinstance(self.source_pair_scorer, StageEditSourcePairScorer):
                raise ValueError("StageEdit preflight source-pair scorer is invalid")
            if not isinstance(self.source_pair_scorer_path, Path) or not (
                self.source_pair_scorer_path.is_absolute()
            ):
                raise ValueError("StageEdit preflight scorer path must be absolute")
            if self.source_pair_scorer.contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT:
                raise ValueError("StageEdit preflight scorer contract changed")
            if self.source_pair_scorer.fingerprint != self.source_pair_scorer_fingerprint:
                raise ValueError("StageEdit preflight scorer fingerprint changed")
            actual_readiness = stage_edit_readiness_report(
                self.checkpoint_directory
            )
            if (
                not actual_readiness.learned_inference_ready
                or actual_readiness.status != "production_checkpoint_ready"
                or actual_readiness.checkpoint_path
                != self.readiness.checkpoint_path
                or actual_readiness.checkpoint_format
                != self.readiness.checkpoint_format
                or actual_readiness.dynamic_crop_version
                != self.readiness.dynamic_crop_version
                or actual_readiness.checkpoint_metadata_filename
                != self.readiness.checkpoint_metadata_filename
                or actual_readiness.training_steps != self.readiness.training_steps
            ):
                raise StageEditPipelineBridgeNotReadyError(
                    "learned preflight readiness identity changed"
                )
            self._validate_learned_scorer_identity()

    def to_dict(self) -> dict[str, Any]:
        self._validate_learned_scorer_identity()
        payload = {
            "snapshot_path": str(self.snapshot_path),
            "snapshot_sha256": self.snapshot.snapshot_sha256,
            "snapshot_sequence_id": self.snapshot.sequence.sequence_id,
            "snapshot_token_count": len(self.snapshot.sequence.tokens),
            "temporal_pose_sidecar": (
                self.snapshot.temporal_pose_sidecar_audit()
            ),
            "checkpoint_directory": str(self.checkpoint_directory),
            "dynamic_crop_version": self.dynamic_crop_version,
            "checkpoint_format": f"dynamic_crop_v{self.dynamic_crop_version}",
            "source_ids": list(self.source_ids),
            "continuity": {
                "authority": self.continuity_authority,
                "scorer_contract": (
                    self.source_pair_scorer.contract
                    if self.source_pair_scorer is not None
                    else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                ),
                "scorer_fingerprint": (
                    self.source_pair_scorer_fingerprint
                    if self.source_pair_scorer_fingerprint is not None
                    else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                ),
                "source_pair_scorer_path": (
                    str(self.source_pair_scorer_path)
                    if self.source_pair_scorer_path is not None
                    else None
                ),
                "source_pair_scorer_file_sha256": (
                    self.source_pair_scorer_file_sha256
                ),
                "source_pair_checkpoint_sha256": (
                    self.source_pair_checkpoint_sha256
                ),
                "source_pair_corpus_sha256": self.source_pair_corpus_sha256,
            },
            "readiness": self.readiness.to_dict(),
        }
        require_scalar_stage_edit_pose_payload(
            payload,
            context="StageEdit pipeline bridge preflight",
        )
        return payload


@dataclass(frozen=True, slots=True)
class RetainedStageEditCandidate:
    rank: int
    score: float
    edit: StageEdit
    exact_clock_reprojected: bool
    timewarped_segment_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rank": self.rank,
            "score": self.score,
            "exact_clock_reprojected": self.exact_clock_reprojected,
            "timewarped_segment_count": self.timewarped_segment_count,
            "selected_for_render": False,
            "stage_edit": self.edit.to_dict(),
        }
        require_scalar_stage_edit_pose_payload(
            payload,
            context="retained StageEdit candidate",
        )
        return payload


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
        if not isinstance(self.preflight, StageEditPipelineBridgePreflight):
            raise TypeError(
                "StageEdit bridge result requires a typed production preflight"
            )
        self.preflight._validate_learned_scorer_identity()
        if not isinstance(
            self.head_materialization, StageEditHeadMaterializationResult
        ):
            raise TypeError(
                "StageEdit bridge result requires typed head materialization"
            )
        if not isinstance(self.semimarkov, StageEditSemiMarkovResult):
            raise TypeError(
                "StageEdit bridge result requires a typed semi-Markov result"
            )
        if not isinstance(self.semimarkov.audit, StageEditDecodeAudit):
            raise TypeError(
                "StageEdit bridge result requires a typed semi-Markov audit"
            )
        if any(
            not isinstance(timeline, StageEditBeamTimeline)
            for timeline in self.semimarkov.timelines
        ):
            raise TypeError(
                "StageEdit bridge result requires typed semi-Markov timelines"
            )
        if self.learned_section_director is not None and not isinstance(
            self.learned_section_director, LearnedSectionDirectorResult
        ):
            raise TypeError(
                "StageEdit bridge result requires a typed learned Section Director"
            )
        if not self.retained_candidates:
            raise ValueError("StageEdit bridge must retain at least one candidate")
        if self.selection_policy.get("visual_baseline_replaced") is not False:
            raise ValueError("StageEdit bridge cannot replace the visual baseline")
        if self.selection_policy.get("selected_for_render") is not False:
            raise ValueError("StageEdit bridge cannot select an unreviewed candidate")
        if self.selection_policy.get("temporal_pose_sidecar") != (
            self.preflight.snapshot.temporal_pose_sidecar_audit()
        ):
            raise ValueError(
                "StageEdit bridge temporal-pose snapshot audit differs from preflight"
            )
        if (
            self.source_derivative_diversity is not None
            and not self.source_derivative_diversity.production_ready
        ):
            raise ValueError(
                "StageEdit bridge requires production-ready source derivative metadata"
            )
        version = self.preflight.dynamic_crop_version
        authority = self.preflight.continuity_authority
        scorer_contract = (
            self.preflight.source_pair_scorer.contract
            if self.preflight.source_pair_scorer is not None
            else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        )
        scorer_fingerprint = (
            self.preflight.source_pair_scorer_fingerprint
            if self.preflight.source_pair_scorer_fingerprint is not None
            else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        )
        if self.head_materialization.dynamic_crop_version != version:
            raise ValueError(
                "StageEdit bridge head materialization version differs"
            )
        if (
            self.head_materialization.sequence_id
            != self.preflight.snapshot.sequence.sequence_id
        ):
            raise ValueError(
                "StageEdit bridge head materialization sequence identity differs"
            )
        if self.head_materialization.canonical_source_ids != (
            self.preflight.source_ids
        ):
            raise ValueError(
                "StageEdit bridge head materialization source identity differs"
            )
        if tuple(
            frame.master_time_seconds
            for frame in self.head_materialization.frames
        ) != tuple(
            token.master_time_seconds
            for token in self.preflight.snapshot.sequence.tokens
        ):
            raise ValueError(
                "StageEdit bridge head materialization frame identity differs"
            )
        head_checkpoint = Path(
            self.head_materialization.checkpoint_directory
        ).expanduser().resolve()
        if head_checkpoint != self.preflight.checkpoint_directory.resolve():
            raise ValueError(
                "StageEdit bridge head materialization checkpoint identity differs"
            )
        if (
            self.head_materialization.readiness != self.preflight.readiness
            or self.head_materialization.training_steps
            != self.preflight.readiness.training_steps
        ):
            raise ValueError(
                "StageEdit bridge head materialization readiness identity differs"
            )
        if self.head_materialization.token_schema_fingerprint != (
            self.preflight.snapshot.token_schema_fingerprint
        ):
            raise ValueError(
                "StageEdit bridge head materialization token schema differs"
            )
        if (
            self.learned_section_director is not None
            and self.learned_section_director.dynamic_crop_version != version
        ):
            raise ValueError(
                "StageEdit bridge Section Director version differs"
            )
        expected_continuity_identity = {
            "authority": authority,
            "scorer_contract": scorer_contract,
            "scorer_fingerprint": scorer_fingerprint,
        }
        audit_config = self.semimarkov.audit.config
        if not isinstance(audit_config, Mapping) or {
            "authority": audit_config.get("continuity_authority"),
            "scorer_contract": audit_config.get(
                "continuity_scorer_contract"
            ),
            "scorer_fingerprint": audit_config.get(
                "continuity_scorer_fingerprint"
            ),
        } != expected_continuity_identity:
            raise ValueError(
                "StageEdit bridge semi-Markov audit config continuity identity "
                "differs from preflight"
            )

        audit_score_provenance = self.semimarkov.audit.score_provenance
        audit_boundary_provenance = (
            audit_score_provenance.get("boundary_continuity")
            if isinstance(audit_score_provenance, Mapping)
            else None
        )
        input_summary = self.semimarkov.audit.input_summary
        input_boundary_provenance = (
            input_summary.get("boundary_continuity_score_provenance")
            if isinstance(input_summary, Mapping)
            else None
        )
        provenance_records = (
            ("audit", audit_boundary_provenance),
            ("input summary", input_boundary_provenance),
            *(
                (
                    f"timeline rank {timeline.rank}",
                    timeline.boundary_continuity_score_provenance,
                )
                for timeline in self.semimarkov.timelines
            ),
        )
        for location, provenance in provenance_records:
            if not isinstance(provenance, Mapping) or {
                "authority": provenance.get("authority"),
                "scorer_contract": provenance.get("scorer_contract"),
                "scorer_fingerprint": provenance.get("scorer_fingerprint"),
            } != expected_continuity_identity:
                raise ValueError(
                    "StageEdit bridge semi-Markov boundary continuity "
                    f"identity in {location} differs from preflight"
                )
        if (
            self.selection_policy.get("dynamic_crop_version") != version
            or self.selection_policy.get("continuity_authority") != authority
            or self.selection_policy.get("continuity_scorer_contract")
            != scorer_contract
            or self.selection_policy.get("continuity_scorer_fingerprint")
            != scorer_fingerprint
        ):
            raise ValueError(
                "StageEdit bridge selection identity differs from preflight"
            )
        for retained in self.retained_candidates:
            history = tuple(retained.edit.candidate_history)
            explicit_versions = {
                item.get("dynamic_crop_version")
                for item in history
                if isinstance(item, Mapping)
                and "dynamic_crop_version" in item
            }
            if explicit_versions != {version}:
                raise ValueError(
                    "StageEdit bridge retained candidate version differs"
                )
            final = history[-1] if history else None
            if (
                not isinstance(final, Mapping)
                or final.get("bridge") != STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT
                or final.get("dynamic_crop_version") != version
                or final.get("continuity_authority") != authority
            ):
                raise ValueError(
                    "StageEdit bridge retained candidate identity is not bound"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        require_scalar_stage_edit_pose_payload(
            payload,
            context="StageEdit pipeline bridge result",
        )
        return payload

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
    dynamic_crop_version: int = 2,
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY,
    source_pair_scorer_path: Path | None = None,
    source_pair_scorer_file_sha256: str | None = None,
    source_pair_scorer_fingerprint: str | None = None,
    source_pair_checkpoint_sha256: str | None = None,
    source_pair_corpus_sha256: str | None = None,
) -> StageEditPipelineBridgePreflight:
    """Strict-load all explicit activation identities before expensive work."""

    snapshot_path = Path(feature_snapshot_path).expanduser().resolve()
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    canonical_source_ids = _strict_source_ids(source_ids)
    if (
        isinstance(dynamic_crop_version, bool)
        or not isinstance(dynamic_crop_version, int)
        or dynamic_crop_version not in (2, 3)
    ):
        raise StageEditPipelineBridgeInputError(
            "dynamic_crop_version must be exactly 2 or 3"
        )
    authority = str(continuity_authority).strip()
    if authority not in STAGE_EDIT_CONTINUITY_AUTHORITIES:
        raise StageEditPipelineBridgeInputError(
            "continuity_authority is unsupported"
        )
    pair_arguments = (
        source_pair_scorer_path,
        source_pair_scorer_file_sha256,
        source_pair_scorer_fingerprint,
        source_pair_checkpoint_sha256,
        source_pair_corpus_sha256,
    )
    if authority == HANDCRAFTED_CONTINUITY_AUTHORITY:
        if any(value is not None for value in pair_arguments):
            raise StageEditPipelineBridgeInputError(
                "handcrafted continuity authority rejects source-pair scorer evidence"
            )
    elif dynamic_crop_version != 3:
        raise StageEditPipelineBridgeInputError(
            "learned source-pair continuity is only valid for Dynamic Crop v3"
        )
    elif any(value is None for value in pair_arguments):
        raise StageEditPipelineBridgeInputError(
            "learned continuity authority requires the exact registered scorer "
            "path, file hash, fingerprint, checkpoint hash, and corpus hash"
        )
    if not snapshot_path.is_file():
        raise StageEditPipelineBridgeNotReadyError(
            "configured StageEdit feature snapshot does not exist"
        )
    namespace_paths = {
        "legacy": (
            checkpoint / "metadata.json",
            checkpoint / "model.safetensors",
        ),
        "dynamic_crop_v2": (
            checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
            checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
        ),
        "dynamic_crop_v3": (
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
        ),
    }
    expected_namespace = f"dynamic_crop_v{dynamic_crop_version}"
    present_namespaces = tuple(
        name
        for name, paths in namespace_paths.items()
        if any(path.exists() or path.is_symlink() for path in paths)
    )
    expected_metadata = namespace_paths[expected_namespace][0]
    if (
        not checkpoint.is_dir()
        or present_namespaces != (expected_namespace,)
        or expected_metadata.is_symlink()
        or not expected_metadata.is_file()
    ):
        raise StageEditPipelineBridgeNotReadyError(
            "configured checkpoint must contain one unambiguous "
            f"Dynamic Crop v{dynamic_crop_version} namespace"
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
    expected_metadata_filename = expected_metadata.name
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps > 0
        and readiness.checkpoint_format == expected_namespace
        and readiness.dynamic_crop_version == dynamic_crop_version
        and readiness.checkpoint_metadata_filename == expected_metadata_filename
    ):
        raise StageEditPipelineBridgeNotReadyError(
            f"configured Dynamic Crop v{dynamic_crop_version} checkpoint is not "
            "production-ready: "
            f"{readiness.status}: {readiness.reason}"
        )

    scorer: StageEditSourcePairScorer | None = None
    scorer_path: Path | None = None
    scorer_file_sha256: str | None = None
    scorer_fingerprint: str | None = None
    scorer_checkpoint_sha256: str | None = None
    scorer_corpus_sha256: str | None = None
    if authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        raw_scorer_path = Path(source_pair_scorer_path).expanduser()  # type: ignore[arg-type]
        if raw_scorer_path.is_symlink():
            raise StageEditPipelineBridgeNotReadyError(
                "registered source-pair scorer must not be a symlink"
            )
        scorer_path = raw_scorer_path.resolve()
        if (
            scorer_path.name != STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
            or not scorer_path.is_file()
        ):
            raise StageEditPipelineBridgeNotReadyError(
                "registered source-pair scorer path is missing or has the wrong filename"
            )
        scorer_file_sha256 = _sha256_identity(
            source_pair_scorer_file_sha256,
            "source_pair_scorer_file_sha256",
        )
        scorer_fingerprint = _sha256_identity(
            source_pair_scorer_fingerprint,
            "source_pair_scorer_fingerprint",
        )
        scorer_checkpoint_sha256 = _sha256_identity(
            source_pair_checkpoint_sha256,
            "source_pair_checkpoint_sha256",
        )
        scorer_corpus_sha256 = _sha256_identity(
            source_pair_corpus_sha256,
            "source_pair_corpus_sha256",
        )
        try:
            before_sha256 = _sha256_file(scorer_path)
            if before_sha256 != scorer_file_sha256:
                raise StageEditPipelineBridgeNotReadyError(
                    "registered source-pair scorer file hash changed"
                )
            scorer, loaded_checkpoint = load_stage_edit_source_pair_checkpoint(
                scorer_path.parent,
                expected_corpus_sha256=scorer_corpus_sha256,
                expected_checkpoint_sha256=scorer_checkpoint_sha256,
            )
            after_sha256 = _sha256_file(scorer_path)
        except (OSError, StageEditSourcePairCheckpointError) as exc:
            raise StageEditPipelineBridgeNotReadyError(
                f"registered source-pair scorer is invalid: {exc}"
            ) from exc
        if before_sha256 != after_sha256:
            raise StageEditPipelineBridgeNotReadyError(
                "registered source-pair scorer changed during preflight"
            )
        if (
            loaded_checkpoint.checkpoint_sha256 != scorer_checkpoint_sha256
            or loaded_checkpoint.corpus_sha256 != scorer_corpus_sha256
            or scorer.contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
            or scorer.fingerprint != scorer_fingerprint
        ):
            raise StageEditPipelineBridgeNotReadyError(
                "registered source-pair scorer identity changed"
            )
    return StageEditPipelineBridgePreflight(
        snapshot_path=snapshot_path,
        checkpoint_directory=checkpoint,
        source_ids=canonical_source_ids,
        snapshot=snapshot,
        readiness=readiness,
        dynamic_crop_version=dynamic_crop_version,
        continuity_authority=authority,
        source_pair_scorer=scorer,
        source_pair_scorer_path=scorer_path,
        source_pair_scorer_file_sha256=scorer_file_sha256,
        source_pair_scorer_fingerprint=scorer_fingerprint,
        source_pair_checkpoint_sha256=scorer_checkpoint_sha256,
        source_pair_corpus_sha256=scorer_corpus_sha256,
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
    dynamic_crop_version: int,
    continuity_authority: str,
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
            "dynamic_crop_version": dynamic_crop_version,
            "continuity_authority": continuity_authority,
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

    preflight._validate_learned_scorer_identity()
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
    if config.continuity_authority != preflight.continuity_authority:
        raise StageEditPipelineBridgeInputError(
            "semi-Markov continuity authority differs from the preflight bundle"
        )
    expected_scorer_contract = (
        preflight.source_pair_scorer.contract
        if preflight.source_pair_scorer is not None
        else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
    )
    expected_scorer_fingerprint = (
        preflight.source_pair_scorer_fingerprint
        if preflight.source_pair_scorer_fingerprint is not None
        else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    )
    if (
        config.continuity_scorer_contract is not None
        and config.continuity_scorer_contract != expected_scorer_contract
    ):
        raise StageEditPipelineBridgeInputError(
            "semi-Markov continuity scorer contract differs from preflight"
        )
    if (
        config.continuity_scorer_fingerprint is not None
        and config.continuity_scorer_fingerprint != expected_scorer_fingerprint
    ):
        raise StageEditPipelineBridgeInputError(
            "semi-Markov continuity scorer fingerprint differs from preflight"
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

    materializer = (
        materialize_stage_edit_head_frames_v3
        if preflight.dynamic_crop_version == 3
        else materialize_stage_edit_head_frames_v2
    )
    heads = materializer(
        checkpoint_directory=preflight.checkpoint_directory,
        sequence=preflight.snapshot.sequence,
        canonical_source_ids=preflight.source_ids,
        schema=schema,
    )
    if getattr(heads, "dynamic_crop_version", None) != (
        preflight.dynamic_crop_version
    ):
        raise StageEditPipelineBridgeError(
            "materialized StageEdit head version differs from preflight"
        )
    learned_section_director = build_learned_section_plan_from_checkpoint(
        section_plan,
        heads.frames,
        checkpoint_directory=preflight.checkpoint_directory,
        chorus_wide_anchor_fraction=(
            config.highlight_minimum_early_wide_anchor_fraction
        ),
    )
    if getattr(learned_section_director, "dynamic_crop_version", None) != (
        preflight.dynamic_crop_version
    ):
        raise StageEditPipelineBridgeError(
            "learned Section Director Dynamic Crop version differs from preflight"
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
        source_pair_scorer=preflight.source_pair_scorer,
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
            dynamic_crop_version=preflight.dynamic_crop_version,
            continuity_authority=preflight.continuity_authority,
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
            "dynamic_crop_version": preflight.dynamic_crop_version,
            "continuity_authority": preflight.continuity_authority,
            "continuity_scorer_contract": expected_scorer_contract,
            "continuity_scorer_fingerprint": expected_scorer_fingerprint,
            "temporal_pose_sidecar": (
                preflight.snapshot.temporal_pose_sidecar_audit()
            ),
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
