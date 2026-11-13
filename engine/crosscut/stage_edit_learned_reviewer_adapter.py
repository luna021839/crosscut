"""Strict decoded-proxy adapter for the production learned reviewer.

The adapter never decides technical or professional acceptance.  It turns the
existing schema-v3 scalar review into the learned reviewer's missing-aware
feature contract, ranks only candidates that the deterministic review already
declared hard-safe, and emits optional :class:`StageEditRepairIssue` windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .render import RenderPlan, segment_frame_counts
from .rendered_proxy_review import (
    ANALYSIS_BASIS as DECODED_PROXY_ANALYSIS_BASIS,
    BOUNDARY_REVIEW_DOMAINS,
    INTRODUCED_CROSS_SOURCE_DOMAIN,
)
from .stage_edit_learned_reviewer import (
    CANDIDATE,
    DEFAULT_LEARNED_REVIEWER_SCHEMA,
    LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256,
    LearnedReviewerBoundary,
    LearnedReviewerCheckpointMetadata,
    LearnedReviewerExample,
    LearnedReviewerNotReadyError,
    LearnedReviewerPrediction,
    LearnedReviewerReadinessReport,
    StageEditLearnedReviewer,
    load_production_learned_reviewer,
    predict_learned_reviewer,
)
from .stage_edit_repair_loop import (
    StageEditRepairIssue,
    StageEditRepairReview,
)


LEARNED_REVIEWER_ADAPTER_SCHEMA_VERSION = 1
LEARNED_REVIEWER_ADAPTER_ANALYSIS_BASIS = (
    "decoded_proxy_v3_to_stage_edit_learned_reviewer_v1"
)
LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS = (
    "deterministic_hard_safe_pool_learned_reviewer_rerank_v1"
)
LEARNED_REVIEWER_RERANK_SELECTION_BASIS = (
    "deterministic_decoded_and_professional_hard_eligibility_then_"
    "learned_quality_then_existing_deterministic_rank"
)


class LearnedReviewerAdapterError(RuntimeError):
    """Malformed decoded evidence or invalid production activation."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LearnedReviewerAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LearnedReviewerAdapterError(f"{name} must be finite")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise LearnedReviewerAdapterError(f"{name} must be normalized")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LearnedReviewerAdapterError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise LearnedReviewerAdapterError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LearnedReviewerAdapterError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _optional_unit(value: Any, name: str) -> float | None:
    return None if value is None else _unit(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LearnedReviewerAdapterError(f"{name} must be a boolean")
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_value(value: Any) -> Any:
    """Convert a render contract into stable, finite JSON evidence."""

    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _contract_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_contract_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LearnedReviewerAdapterError(
                "render plan contract contains a non-finite number"
            )
        return value
    raise LearnedReviewerAdapterError(
        f"render plan contract contains unsupported {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class LearnedReviewerProductionReviewConfig:
    checkpoint_directory: Path
    production_training_report: Path
    song_id: str
    issue_probability_threshold: float = 0.50
    cause_probability_threshold: float = 0.50
    model_quality_threshold: float = 0.50
    issue_window_radius_seconds: float = 0.20

    def __post_init__(self) -> None:
        checkpoint = Path(self.checkpoint_directory).expanduser().resolve()
        report = Path(self.production_training_report).expanduser().resolve()
        if not checkpoint.is_dir():
            raise LearnedReviewerAdapterError(
                "learned reviewer checkpoint directory does not exist"
            )
        if not report.is_file():
            raise LearnedReviewerAdapterError(
                "learned reviewer production report does not exist"
            )
        if report != (checkpoint / "training-report.json").resolve():
            raise LearnedReviewerAdapterError(
                "learned reviewer report must be the exact colocated report"
            )
        if not isinstance(self.song_id, str) or not self.song_id.strip():
            raise LearnedReviewerAdapterError("learned reviewer song_id is required")
        song = self.song_id.strip()
        object.__setattr__(self, "checkpoint_directory", checkpoint)
        object.__setattr__(self, "production_training_report", report)
        object.__setattr__(self, "song_id", song)
        for name in (
            "issue_probability_threshold",
            "cause_probability_threshold",
            "model_quality_threshold",
        ):
            value = _unit(getattr(self, name), name)
            object.__setattr__(self, name, value)
        radius = _finite(
            self.issue_window_radius_seconds,
            "issue_window_radius_seconds",
        )
        if not 0.0 <= radius <= 3.0:
            raise LearnedReviewerAdapterError(
                "issue_window_radius_seconds must be in 0...3"
            )
        object.__setattr__(self, "issue_window_radius_seconds", radius)


@dataclass(frozen=True, slots=True)
class DecodedProxyFeatureMaterialization:
    candidate_index: int
    example: LearnedReviewerExample
    boundary_indices: tuple[int, ...]
    boundary_domains: tuple[str, ...]
    canonical_start_seconds: float
    canonical_end_seconds: float
    render_plan_sha256: str
    decoded_evidence_sha256: str
    feature_snapshot_sha256: str
    technical_zero_tolerance_passed: bool

    def __post_init__(self) -> None:
        if len(self.example.boundaries) != len(self.boundary_indices) or len(
            self.example.boundaries
        ) != len(self.boundary_domains):
            raise LearnedReviewerAdapterError(
                "materialized boundary mappings have different lengths"
            )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEARNED_REVIEWER_ADAPTER_SCHEMA_VERSION,
            "analysis_basis": LEARNED_REVIEWER_ADAPTER_ANALYSIS_BASIS,
            "candidate_index": self.candidate_index,
            "example_id": self.example.example_id,
            "boundary_count": len(self.example.boundaries),
            "boundary_indices": list(self.boundary_indices),
            "boundary_domains": list(self.boundary_domains),
            "canonical_start_seconds": self.canonical_start_seconds,
            "canonical_end_seconds": self.canonical_end_seconds,
            "render_plan_sha256": self.render_plan_sha256,
            "decoded_evidence_sha256": self.decoded_evidence_sha256,
            "feature_schema_fingerprint": (
                DEFAULT_LEARNED_REVIEWER_SCHEMA.fingerprint
            ),
            "feature_materializer_contract_sha256": (
                LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256
            ),
            "feature_snapshot_sha256": self.feature_snapshot_sha256,
            "technical_zero_tolerance_passed": (
                self.technical_zero_tolerance_passed
            ),
            "raw_frames_or_embeddings_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class LearnedDecodedCandidateReview:
    candidate_index: int
    materialization: DecodedProxyFeatureMaterialization
    prediction: LearnedReviewerPrediction
    repair_issues: tuple[StageEditRepairIssue, ...]
    repair_issue_boundary_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.prediction.example_id != self.materialization.example.example_id:
            raise LearnedReviewerAdapterError(
                "learned prediction does not match materialized example"
            )
        if len(self.repair_issues) != len(self.repair_issue_boundary_indices):
            raise LearnedReviewerAdapterError(
                "learned repair issues and boundary indices differ"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "materialization": self.materialization.to_audit_dict(),
            "prediction": self.prediction.to_dict(),
            "repair_issues": [issue.to_dict() for issue in self.repair_issues],
            "repair_issue_boundary_indices": list(
                self.repair_issue_boundary_indices
            ),
        }


@dataclass(frozen=True, slots=True)
class LearnedReviewerProductionSession:
    config: LearnedReviewerProductionReviewConfig
    model: StageEditLearnedReviewer = field(repr=False)
    metadata: LearnedReviewerCheckpointMetadata
    readiness: LearnedReviewerReadinessReport
    evidence: Mapping[str, str]

    def __post_init__(self) -> None:
        if not (
            self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
            and self.readiness.checkpoint_id == self.metadata.checkpoint_id
            and self.readiness.training_steps == self.metadata.training_steps
        ):
            raise LearnedReviewerNotReadyError(
                "learned reviewer session requires a production-ready checkpoint"
            )
        if (
            self.metadata.feature_schema.fingerprint
            != DEFAULT_LEARNED_REVIEWER_SCHEMA.fingerprint
        ):
            raise LearnedReviewerNotReadyError(
                "decoded proxy adapter requires the canonical reviewer schema"
            )
        if (
            self.metadata.feature_schema.feature_materializer_contract_sha256
            != LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256
        ):
            raise LearnedReviewerNotReadyError(
                "checkpoint feature materializer contract does not match adapter"
            )
        required_evidence = {
            "metadata_sha256",
            "training_report_sha256",
            "weight_sha256",
            "bundle_sha256",
        }
        if set(self.evidence) != required_evidence or any(
            not isinstance(self.evidence[name], str)
            or len(self.evidence[name]) != 64
            for name in required_evidence
        ):
            raise LearnedReviewerNotReadyError(
                "learned reviewer session evidence is incomplete"
            )

    @classmethod
    def load(
        cls,
        config: LearnedReviewerProductionReviewConfig,
    ) -> "LearnedReviewerProductionSession":
        model, metadata, readiness, evidence = load_production_learned_reviewer(
            config.checkpoint_directory,
            config.production_training_report,
        )
        return cls(config, model, metadata, readiness, dict(evidence))

    def review_candidate(
        self,
        *,
        candidate_index: int,
        render_plan: RenderPlan,
        decoded_review: Mapping[str, Any],
        source_switches_per_minute: float | None = None,
    ) -> LearnedDecodedCandidateReview:
        materialization = materialize_decoded_proxy_reviewer_features(
            candidate_index=candidate_index,
            song_id=self.config.song_id,
            render_plan=render_plan,
            decoded_review=decoded_review,
            source_switches_per_minute=source_switches_per_minute,
        )
        prediction = predict_learned_reviewer(
            self.model,
            (materialization.example,),
            self.metadata.feature_schema,
            quality_threshold=self.config.model_quality_threshold,
        )[0]
        issues, indices = learned_prediction_to_stage_edit_repair_issues(
            materialization,
            prediction,
            issue_probability_threshold=(
                self.config.issue_probability_threshold
            ),
            cause_probability_threshold=(
                self.config.cause_probability_threshold
            ),
            window_radius_seconds=self.config.issue_window_radius_seconds,
        )
        return LearnedDecodedCandidateReview(
            candidate_index=candidate_index,
            materialization=materialization,
            prediction=prediction,
            repair_issues=issues,
            repair_issue_boundary_indices=indices,
        )

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.metadata.checkpoint_id,
            "training_steps": self.metadata.training_steps,
            "corpus_id": self.metadata.corpus_id,
            "corpus_sha256": self.metadata.corpus_sha256,
            "feature_schema_fingerprint": (
                self.metadata.feature_schema.fingerprint
            ),
            "feature_materializer_contract_sha256": (
                LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256
            ),
            "review_policy": {
                "model_quality_threshold": (
                    self.config.model_quality_threshold
                ),
                "issue_probability_threshold": (
                    self.config.issue_probability_threshold
                ),
                "cause_probability_threshold": (
                    self.config.cause_probability_threshold
                ),
                "issue_window_radius_seconds": (
                    self.config.issue_window_radius_seconds
                ),
                "model_pass_fail_is_authoritative": False,
            },
            **dict(self.evidence),
        }


def _strict_proxy_clock(
    review: Mapping[str, Any],
    plan: RenderPlan,
) -> tuple[float, int]:
    proxy = _mapping(review.get("proxy"), "decoded proxy metadata")
    fps = _finite(proxy.get("fps"), "decoded proxy fps")
    if fps <= 0.0 or not math.isclose(
        fps, plan.fps, rel_tol=0.0, abs_tol=1e-9
    ):
        raise LearnedReviewerAdapterError(
            "decoded proxy fps does not match render plan"
        )
    frame_count = _integer(
        proxy.get("decoded_frame_count"), "decoded proxy frame count"
    )
    if frame_count != plan.output_frame_count:
        raise LearnedReviewerAdapterError(
            "decoded proxy frame count does not match render plan"
        )
    duration = _finite(proxy.get("duration_seconds"), "decoded proxy duration")
    if not math.isclose(
        duration, plan.render_duration, rel_tol=0.0, abs_tol=1e-9
    ):
        raise LearnedReviewerAdapterError(
            "decoded proxy duration does not match render plan"
        )
    return fps, frame_count


def _boundary_feature_values(
    boundary: Mapping[str, Any],
) -> tuple[tuple[float | None, ...], tuple[str, ...]]:
    discontinuities = _mapping(
        boundary.get("discontinuities"), "boundary discontinuities"
    )
    score = _finite(boundary.get("weighted_score"), "boundary weighted score")
    if not 0.0 <= score <= 100.0:
        raise LearnedReviewerAdapterError(
            "boundary weighted score must be in 0...100"
        )
    face_scale = _optional_unit(
        discontinuities.get("face_scale_discontinuity"),
        "face scale discontinuity",
    )
    face_position = _optional_unit(
        discontinuities.get("face_position_discontinuity"),
        "face position discontinuity",
    )
    silhouette_values = tuple(
        value
        for value in (
            _optional_unit(
                discontinuities.get("silhouette_scale_discontinuity"),
                "silhouette scale discontinuity",
            ),
            _optional_unit(
                discontinuities.get("silhouette_position_discontinuity"),
                "silhouette position discontinuity",
            ),
        )
        if value is not None
    )
    face_match_confidence = (
        1.0 - (face_scale + face_position) / 2.0
        if face_scale is not None and face_position is not None
        else None
    )
    frame_metrics = _array(boundary.get("frame_metrics"), "boundary frame metrics")
    sharpness_values = tuple(
        _unit(_mapping(item, "boundary frame metric").get("sharpness"), "sharpness")
        for item in frame_metrics
    )
    temporal_raw = boundary.get("temporal_integrity")
    temporal = temporal_raw if isinstance(temporal_raw, Mapping) else None
    raw_technical_codes = _array(
        boundary.get("technical_integrity_issue_reasons"),
        "boundary technical issue reasons",
    )
    if any(not isinstance(code, str) or not code.strip() for code in raw_technical_codes):
        raise LearnedReviewerAdapterError(
            "boundary technical issue reasons must contain strings"
        )
    technical_codes = {str(code).strip() for code in raw_technical_codes}
    if temporal is None:
        technical_codes.add("missing_temporal_integrity")
    if temporal is None:
        black_frame_offsets: Sequence[Any] = ()
        duplicate_frame = False
    else:
        black_frame_offsets = _array(
            temporal.get("black_frame_offsets"), "black frame offsets"
        )
        for position, offset in enumerate(black_frame_offsets):
            _finite(offset, f"black frame offset {position}")
        duplicate_frame = _boolean(
            temporal.get("duplicate_or_freeze_detected"),
            "duplicate_or_freeze_detected",
        )
    black_frame = bool(black_frame_offsets)
    decode_error = temporal is None or not frame_metrics
    clock_error = any("source_clock" in code for code in technical_codes)
    if black_frame:
        technical_codes.add("decoded_black_frame")
    if duplicate_frame:
        technical_codes.add("decoded_duplicate_or_freeze")
    if decode_error:
        technical_codes.add("decoded_evidence_incomplete")
    values_by_name: dict[str, float | None] = {
        "visual_boundary_score": score / 100.0,
        "visual_discontinuity": 1.0 - score / 100.0,
        "face_match_confidence": face_match_confidence,
        "face_scale_mismatch": face_scale,
        "face_center_mismatch": face_position,
        "silhouette_mismatch": max(silhouette_values) if silhouette_values else None,
        # The decoded proxy currently has no pose provider.  A generic
        # structure-shock scalar is not semantically interchangeable with pose,
        # so preserve the missing mask instead of manufacturing supervision.
        "pose_mismatch": None,
        "motion_discontinuity": _optional_unit(
            discontinuities.get("motion_direction_discontinuity"),
            "motion discontinuity",
        ),
        "camera_motion_mismatch": _optional_unit(
            discontinuities.get("motion_direction_discontinuity"),
            "camera motion mismatch",
        ),
        "color_delta": _optional_unit(
            discontinuities.get("color_shock"), "color shock"
        ),
        "exposure_delta": _optional_unit(
            discontinuities.get("exposure_discontinuity"),
            "exposure discontinuity",
        ),
        "sharpness_normalized": (
            sum(sharpness_values) / len(sharpness_values)
            if sharpness_values
            else None
        ),
        "technical_black_frame": float(black_frame),
        "technical_duplicate_frame": float(duplicate_frame),
        "technical_decode_error": float(decode_error),
        "technical_clock_error": float(clock_error),
    }
    if set(values_by_name) != set(
        DEFAULT_LEARNED_REVIEWER_SCHEMA.boundary_feature_names
    ):
        raise LearnedReviewerAdapterError(
            "decoded boundary feature mapping is incomplete"
        )
    return (
        tuple(
            values_by_name[name]
            for name in DEFAULT_LEARNED_REVIEWER_SCHEMA.boundary_feature_names
        ),
        tuple(sorted(technical_codes)),
    )


def _timeline_feature_values(
    plan: RenderPlan,
    *,
    source_switches_per_minute: float | None,
) -> tuple[float | None, ...]:
    durations_by_source: dict[int, float] = {}
    for segment in plan.segments:
        durations_by_source[segment.source_index] = (
            durations_by_source.get(segment.source_index, 0.0)
            + float(segment.duration)
        )
    total_duration = sum(durations_by_source.values())
    if total_duration <= 0.0:
        raise LearnedReviewerAdapterError("render plan duration must be positive")
    shares = tuple(value / total_duration for value in durations_by_source.values())
    source_count = max(1, len(plan.sources))
    used_count = len(durations_by_source)
    entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
    normalized_entropy = entropy / math.log(used_count) if used_count > 1 else 0.0
    source_sequence = tuple(segment.source_index for segment in plan.segments)
    aba_count = sum(
        left == right and left != middle
        for left, middle, right in zip(
            source_sequence,
            source_sequence[1:],
            source_sequence[2:],
        )
    )
    aba_denominator = max(1, len(source_sequence) - 2)
    shot_durations = tuple(float(segment.duration) for segment in plan.segments)
    switch_count = sum(
        left != right for left, right in zip(source_sequence, source_sequence[1:])
    )
    measured_rate = 60.0 * switch_count / total_duration
    if source_switches_per_minute is not None:
        supplied_rate = _finite(
            source_switches_per_minute, "source_switches_per_minute"
        )
        if supplied_rate < 0.0 or not math.isclose(
            supplied_rate, measured_rate, rel_tol=0.0, abs_tol=0.25
        ):
            raise LearnedReviewerAdapterError(
                "source switch rate does not match render plan"
            )
        switch_rate = supplied_rate
    else:
        switch_rate = measured_rate
    values_by_name: dict[str, float | None] = {
        "cadence_score": max(0.0, 1.0 - abs(switch_rate - 15.5) / 15.5),
        "source_switch_rate_normalized": min(1.0, switch_rate / 30.0),
        "source_entropy": normalized_entropy,
        "maximum_source_share": max(shares),
        "unique_source_fraction": used_count / source_count,
        "median_shot_duration_normalized": min(
            1.0, median(shot_durations) / 8.0
        ),
        "short_shot_fraction": (
            sum(duration < 1.0 for duration in shot_durations)
            / len(shot_durations)
        ),
        "long_shot_fraction": (
            sum(duration > 7.2 for duration in shot_durations)
            / len(shot_durations)
        ),
        "section_energy_alignment": None,
        "section_cut_density_alignment": None,
        "aba_rate": aba_count / aba_denominator,
        "recent_cut_history_density": min(1.0, switch_rate / 30.0),
    }
    if set(values_by_name) != set(
        DEFAULT_LEARNED_REVIEWER_SCHEMA.timeline_feature_names
    ):
        raise LearnedReviewerAdapterError(
            "decoded timeline feature mapping is incomplete"
        )
    return tuple(
        values_by_name[name]
        for name in DEFAULT_LEARNED_REVIEWER_SCHEMA.timeline_feature_names
    )


def materialize_decoded_proxy_reviewer_features(
    *,
    candidate_index: int,
    song_id: str,
    render_plan: RenderPlan,
    decoded_review: Mapping[str, Any],
    source_switches_per_minute: float | None = None,
) -> DecodedProxyFeatureMaterialization:
    """Validate schema-v3 evidence and create one inference-only example."""

    index = _integer(candidate_index, "candidate_index")
    if not isinstance(render_plan, RenderPlan):
        raise LearnedReviewerAdapterError("render_plan must be a RenderPlan")
    review = _mapping(decoded_review, "decoded proxy review")
    if not isinstance(song_id, str) or not song_id.strip():
        raise LearnedReviewerAdapterError("song_id must be a non-empty string")
    if review.get("schema_version") != 3:
        raise LearnedReviewerAdapterError("decoded proxy review must use schema v3")
    if review.get("analysis") != DECODED_PROXY_ANALYSIS_BASIS:
        raise LearnedReviewerAdapterError("decoded proxy analysis basis is invalid")
    fps, frame_count = _strict_proxy_clock(review, render_plan)
    raw_boundaries = _array(review.get("boundaries"), "decoded proxy boundaries")
    if len(raw_boundaries) != len(render_plan.transitions):
        raise LearnedReviewerAdapterError(
            "decoded boundary count does not match render plan"
        )
    counts = segment_frame_counts(render_plan)
    expected_cut_frames: list[int] = []
    cumulative = 0
    for count in counts[:-1]:
        cumulative += count
        expected_cut_frames.append(cumulative)
    boundaries: list[LearnedReviewerBoundary] = []
    boundary_indices: list[int] = []
    boundary_domains: list[str] = []
    technical_zero = True
    for position, raw_boundary in enumerate(raw_boundaries):
        boundary = _mapping(raw_boundary, "decoded proxy boundary")
        boundary_index = _integer(boundary.get("boundary_index"), "boundary_index")
        if boundary_index != position:
            raise LearnedReviewerAdapterError(
                "decoded boundary indices must be contiguous and ordered"
            )
        cut_frame = _integer(boundary.get("cut_frame_index"), "cut_frame_index")
        if cut_frame != expected_cut_frames[position] or not 0 < cut_frame < frame_count:
            raise LearnedReviewerAdapterError(
                "decoded boundary clock does not match render plan"
            )
        left = render_plan.segments[position]
        right = render_plan.segments[position + 1]
        if (
            _integer(
                boundary.get("outgoing_source_index"),
                "outgoing_source_index",
            )
            != left.source_index
            or _integer(
                boundary.get("incoming_source_index"),
                "incoming_source_index",
            )
            != right.source_index
        ):
            raise LearnedReviewerAdapterError(
                "decoded boundary sources do not match render plan"
            )
        if boundary.get("transition_kind") != render_plan.transitions[position].kind.value:
            raise LearnedReviewerAdapterError(
                "decoded boundary transition kind does not match render plan"
            )
        domain = boundary.get("review_domain")
        if domain not in BOUNDARY_REVIEW_DOMAINS:
            raise LearnedReviewerAdapterError("decoded boundary domain is invalid")
        features, technical_codes = _boundary_feature_values(boundary)
        technical_zero = technical_zero and not technical_codes
        canonical_seconds = render_plan.master_audio_start + cut_frame / fps
        boundaries.append(
            LearnedReviewerBoundary(
                boundary_id=f"candidate-{index}-boundary-{boundary_index}",
                at_seconds=canonical_seconds,
                features=features,
                issue_target=None,
                cause_targets=None,
                technical_error_codes=technical_codes,
            )
        )
        boundary_indices.append(boundary_index)
        boundary_domains.append(str(domain))
    if not boundaries:
        raise LearnedReviewerAdapterError(
            "learned reviewer requires at least one decoded boundary"
        )
    example = LearnedReviewerExample(
        example_id=f"decoded-proxy-candidate-{index}",
        song_id=song_id.strip(),
        reference_id=f"production-decoded-proxy-{index}",
        example_kind=CANDIDATE,
        boundaries=tuple(boundaries),
        timeline_features=_timeline_feature_values(
            render_plan,
            source_switches_per_minute=source_switches_per_minute,
        ),
        quality_target=None,
    )
    snapshot_payload = {
        "schema_version": LEARNED_REVIEWER_ADAPTER_SCHEMA_VERSION,
        "analysis_basis": LEARNED_REVIEWER_ADAPTER_ANALYSIS_BASIS,
        "candidate_index": index,
        "feature_schema_fingerprint": (
            DEFAULT_LEARNED_REVIEWER_SCHEMA.fingerprint
        ),
        "feature_materializer_contract_sha256": (
            LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256
        ),
        "render_plan": _contract_value(render_plan),
        "example": example.to_dict(),
        "boundary_indices": boundary_indices,
        "boundary_domains": boundary_domains,
        "render_clock": {
            "fps": fps,
            "frame_count": frame_count,
            "canonical_start_seconds": render_plan.master_audio_start,
            "canonical_end_seconds": (
                render_plan.master_audio_start + render_plan.duration
            ),
        },
    }
    return DecodedProxyFeatureMaterialization(
        candidate_index=index,
        example=example,
        boundary_indices=tuple(boundary_indices),
        boundary_domains=tuple(boundary_domains),
        canonical_start_seconds=float(render_plan.master_audio_start),
        canonical_end_seconds=float(
            render_plan.master_audio_start + render_plan.duration
        ),
        render_plan_sha256=_canonical_sha256(_contract_value(render_plan)),
        decoded_evidence_sha256=_canonical_sha256(review),
        feature_snapshot_sha256=_canonical_sha256(snapshot_payload),
        technical_zero_tolerance_passed=technical_zero,
    )


def learned_prediction_to_stage_edit_repair_issues(
    materialization: DecodedProxyFeatureMaterialization,
    prediction: LearnedReviewerPrediction,
    *,
    issue_probability_threshold: float = 0.50,
    cause_probability_threshold: float = 0.50,
    window_radius_seconds: float = 0.20,
) -> tuple[tuple[StageEditRepairIssue, ...], tuple[int, ...]]:
    """Convert editorial predictions to repair-loop-compatible issue windows."""

    issue_threshold = _unit(
        issue_probability_threshold, "issue_probability_threshold"
    )
    cause_threshold = _unit(
        cause_probability_threshold, "cause_probability_threshold"
    )
    radius = _finite(window_radius_seconds, "window_radius_seconds")
    if not 0.0 <= radius <= 3.0:
        raise LearnedReviewerAdapterError(
            "window_radius_seconds must be in 0...3"
        )
    if prediction.example_id != materialization.example.example_id:
        raise LearnedReviewerAdapterError(
            "learned prediction does not match materialization"
        )
    if len(prediction.boundary_predictions) != len(
        materialization.example.boundaries
    ):
        raise LearnedReviewerAdapterError(
            "learned prediction boundary count does not match materialization"
        )
    issues: list[StageEditRepairIssue] = []
    indices: list[int] = []
    for boundary, boundary_prediction, boundary_index, domain in zip(
        materialization.example.boundaries,
        prediction.boundary_predictions,
        materialization.boundary_indices,
        materialization.boundary_domains,
        strict=True,
    ):
        if (
            domain != INTRODUCED_CROSS_SOURCE_DOMAIN
            or boundary_prediction.technical_hard_failure
            or boundary_prediction.issue_probability < issue_threshold
        ):
            continue
        causes = tuple(
            sorted(
                label
                for label, probability in (
                    boundary_prediction.cause_probabilities.items()
                )
                if label != "technical_integrity"
                and probability >= cause_threshold
            )
        ) or ("learned_editorial_discontinuity",)
        issues.append(
            StageEditRepairIssue(
                issue_id=(
                    f"learned-candidate-{materialization.candidate_index}-"
                    f"boundary-{boundary_index}"
                ),
                start_seconds=max(
                    materialization.canonical_start_seconds,
                    boundary.at_seconds - radius,
                ),
                end_seconds=min(
                    materialization.canonical_end_seconds,
                    boundary.at_seconds + radius,
                ),
                causes=causes,
                severity=(
                    "error"
                    if boundary_prediction.issue_probability >= 0.75
                    else "warning"
                ),
            )
        )
        indices.append(boundary_index)
    return tuple(issues), tuple(indices)


def augment_stage_edit_repair_review_with_learned_issues(
    deterministic_review: StageEditRepairReview,
    learned_candidate_review: LearnedDecodedCandidateReview,
    *,
    session_provenance: Mapping[str, Any],
) -> StageEditRepairReview:
    """Add windows without changing deterministic objectives or technical errors."""

    existing = {issue.issue_id: issue for issue in deterministic_review.issue_windows}
    added_ids: list[str] = []
    duplicate_ids: list[str] = []
    for issue in learned_candidate_review.repair_issues:
        if issue.issue_id in existing:
            duplicate_ids.append(issue.issue_id)
            continue
        existing[issue.issue_id] = issue
        added_ids.append(issue.issue_id)
    audit = dict(deterministic_review.audit)
    audit["learned_reviewer_issue_suggestions"] = {
        "analysis_basis": LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS,
        "authority": "issue_windows_only_deterministic_objectives_unchanged",
        "candidate_index": learned_candidate_review.candidate_index,
        "added_issue_ids": added_ids,
        "duplicate_issue_ids_ignored": duplicate_ids,
        "feature_snapshot_sha256": (
            learned_candidate_review.materialization.feature_snapshot_sha256
        ),
        "session_provenance": dict(session_provenance),
    }
    return replace(
        deterministic_review,
        issue_windows=tuple(existing.values()),
        audit=audit,
    )


__all__ = [
    "DecodedProxyFeatureMaterialization",
    "LEARNED_REVIEWER_ADAPTER_ANALYSIS_BASIS",
    "LEARNED_REVIEWER_ADAPTER_SCHEMA_VERSION",
    "LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS",
    "LEARNED_REVIEWER_RERANK_SELECTION_BASIS",
    "LearnedDecodedCandidateReview",
    "LearnedReviewerAdapterError",
    "LearnedReviewerProductionReviewConfig",
    "LearnedReviewerProductionSession",
    "augment_stage_edit_repair_review_with_learned_issues",
    "learned_prediction_to_stage_edit_repair_issues",
    "materialize_decoded_proxy_reviewer_features",
]
