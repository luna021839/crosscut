"""Dynamic renderer-native crop supervision from canonical reference samples.

``reference_crop_supervision`` intentionally learns one static four-vector from
three synthetic interior timestamps.  That contract is retained as crop-v1.
This module is a separate crop-v2 contract for the camera move a human editor
authored *inside* one detector-owned shot.

Every observation starts from a canonical evidence sample and its attributed
source candidate.  The reference frame is the exact evidence frame.  The
source frame is either the exact matched evidence frame or the winner of a
strict local visual search of at most two frames around it.  Each observation
is independently geometry-checked, projected through the production renderer
safety validator when necessary, and reviewed on the pixels produced by that
projected warp.  Failed and out-of-domain observations are loss-masked; their
values are never clamped, copied, or substituted.

The output retains one to N renderer-safe keyframes with explicit shot-local
times and a static/dynamic label.  It does not masquerade as the legacy static
crop head's four-vector target.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_EVIDENCE_ANALYSIS_BASIS,
)
from .reference_crop_supervision import (
    ReferenceCropSupervisionConfig,
    _canonical_output,
    _estimate_sample,
    _file_sha256,
    _media_info,
)
from .reference_evidence_builder import ReferenceSourceClock
from .reference_match import ReferenceMatchConfig, decode_target_frames_sequential
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
)


SCHEMA_VERSION = 2
ANALYSIS_BASIS = (
    "canonical_evidence_matched_dynamic_renderer_crop_supervision_v3"
)
DYNAMIC_CROP_NAMESPACE = "crop-v2"
DYNAMIC_CROP_TARGET_CONTRACT = (
    "variable_1_to_n_real_shot_start_renderer_keyframes_target_only_v2"
)
CONTENT_HASH_BASIS = (
    "sha256_canonical_json_with_provenance.artifact_content_sha256_null"
)
FINGERPRINT_BASIS = (
    "sha256_crop_v2_contract_config_input_hashes_shot_source_identity_v2"
)

SHOT_START_OBSERVATION_BASIS = (
    "detector_shot_start_bound_clocks_real_decoded_frames_v1"
)
CANONICAL_SAMPLE_OBSERVATION_BASIS = (
    "canonical_evidence_exact_reference_and_matched_source_frames_v1"
)
SHOT_START_UNAVAILABLE_REASON = "real_shot_start_transform_unavailable"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(payload))
    provenance = normalized.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("dynamic crop provenance must be an object")
    normalized_provenance = deepcopy(dict(provenance))
    normalized_provenance["artifact_content_sha256"] = None
    normalized["provenance"] = normalized_provenance
    return _canonical_json_sha256(normalized)


@dataclass(frozen=True, slots=True)
class ReferenceDynamicCropSupervisionConfig:
    """Search, motion-label, and inherited per-frame geometry gates."""

    frame_estimator: ReferenceCropSupervisionConfig = field(
        default_factory=ReferenceCropSupervisionConfig
    )
    local_visual_search_radius_frames: int = 2
    maximum_clock_quantization_error_frames: float = 0.55
    dynamic_zoom_range_threshold: float = 0.008
    dynamic_center_range_threshold: float = 0.005
    dynamic_roll_range_threshold_degrees: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.frame_estimator, ReferenceCropSupervisionConfig):
            raise TypeError("frame_estimator must be ReferenceCropSupervisionConfig")
        radius = self.local_visual_search_radius_frames
        if isinstance(radius, bool) or not isinstance(radius, int) or not 2 <= radius <= 4:
            raise ValueError("local visual search radius must be between two and four frames")
        if not 0.0 < _finite(
            self.maximum_clock_quantization_error_frames,
            "clock quantization error",
        ) <= 0.75:
            raise ValueError("clock quantization error must be in (0, 0.75] frames")
        for name in (
            "dynamic_zoom_range_threshold",
            "dynamic_center_range_threshold",
            "dynamic_roll_range_threshold_degrees",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_estimator": self.frame_estimator.to_dict(),
            "local_visual_search_radius_frames": (
                self.local_visual_search_radius_frames
            ),
            "maximum_clock_quantization_error_frames": (
                self.maximum_clock_quantization_error_frames
            ),
            "dynamic_zoom_range_threshold": self.dynamic_zoom_range_threshold,
            "dynamic_center_range_threshold": self.dynamic_center_range_threshold,
            "dynamic_roll_range_threshold_degrees": (
                self.dynamic_roll_range_threshold_degrees
            ),
        }


@dataclass(frozen=True, slots=True)
class ReferenceDynamicCropSupervisionRequest:
    """One canonical attributed shot plus the exact evidence that created it."""

    attributed_shot: Mapping[str, Any]
    attribution_path: Path
    evidence_path: Path
    reference_media_path: Path
    source_media_path: Path
    source_id: str
    expected_attribution_file_sha256: str | None = None
    expected_evidence_file_sha256: str | None = None
    expected_reference_media_sha256: str | None = None
    expected_source_media_sha256: str | None = None
    caller_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attributed_shot, Mapping):
            raise TypeError("attributed_shot must be an object")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("source_id cannot be blank")
        paths = {
            "attribution_path": Path(self.attribution_path).expanduser(),
            "evidence_path": Path(self.evidence_path).expanduser(),
            "reference_media_path": Path(self.reference_media_path).expanduser(),
            "source_media_path": Path(self.source_media_path).expanduser(),
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{name} is not a file: {path}")
            object.__setattr__(self, name, path.resolve())
        if paths["reference_media_path"].resolve() == paths[
            "source_media_path"
        ].resolve():
            raise ValueError("reference and source media must be distinct files")
        for name in (
            "expected_attribution_file_sha256",
            "expected_evidence_file_sha256",
            "expected_reference_media_sha256",
            "expected_source_media_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        try:
            caller = json.loads(
                json.dumps(
                    dict(self.caller_provenance),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("caller_provenance must be finite JSON") from exc
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "attributed_shot", deepcopy(dict(self.attributed_shot)))
        object.__setattr__(self, "caller_provenance", caller)


@dataclass(frozen=True, slots=True)
class ReferenceDynamicCropSupervisionArtifact:
    payload: dict[str, Any]

    @property
    def eligible(self) -> bool:
        return bool(self.payload["supervision"]["eligible"])

    @property
    def keyframes(self) -> tuple[Mapping[str, Any], ...]:
        value = self.payload["supervision"]["renderer_keyframes"]
        return tuple(value) if isinstance(value, list) else ()

    def write(self, output_path: Path) -> Path:
        destination = Path(output_path).expanduser()
        resolved = destination.resolve()
        protected = {
            Path(str(self.payload["provenance"][name]["path"])).resolve()
            for name in (
                "attribution",
                "evidence",
                "reference_media",
                "source_media",
            )
        }
        if resolved in protected:
            raise ValueError("dynamic crop output cannot overwrite an input artifact")
        if DYNAMIC_CROP_NAMESPACE not in destination.name:
            raise ValueError("dynamic crop v2 output filename must contain crop-v2")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("x", encoding="utf-8") as handle:
                json.dump(
                    self.payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as exc:
            raise FileExistsError(
                f"dynamic crop v2 output already exists: {destination}"
            ) from exc
        return resolved


def reference_dynamic_crop_v2_filename(
    reference_id: str,
    shot_index: int,
    source_id: str,
) -> str:
    """Return a crop-v2-only immutable artifact name."""

    if isinstance(shot_index, bool) or not isinstance(shot_index, int) or shot_index < 0:
        raise ValueError("shot_index must be a non-negative integer")

    def clean(value: str, name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip()).strip("-")
        if not normalized:
            raise ValueError(f"{name} cannot be blank")
        return normalized

    return (
        f"{clean(reference_id, 'reference_id')}-reference-dynamic-crop-"
        f"supervision-{DYNAMIC_CROP_NAMESPACE}-shot-{shot_index:04d}-"
        f"{clean(source_id, 'source_id')}.json"
    )


def _read_bound_json(
    path: Path,
    expected_hash: str | None,
    name: str,
) -> tuple[dict[str, Any], str]:
    initial_hash = _file_sha256(path)
    if expected_hash is not None and expected_hash != initial_hash:
        raise ValueError(f"{name} SHA-256 does not match the caller expectation")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != initial_hash:
        raise ValueError(f"{name} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
    return deepcopy(dict(value)), initial_hash


def _attribution_hash_identity(
    attribution: Mapping[str, Any], partition_sha256: str
) -> dict[str, Any]:
    source_path = attribution.get("source_path")
    shots = attribution.get("shots")
    boundaries = attribution.get("detected_boundaries")
    if not isinstance(source_path, list) or not isinstance(shots, list) or not isinstance(
        boundaries, list
    ):
        raise ValueError("canonical attribution path/shot/boundary arrays are missing")
    return {
        "partition_content_sha256": partition_sha256,
        "source_path_source_ids": [
            item.get("source_id") if isinstance(item, Mapping) else None
            for item in source_path
        ],
        "shot_aggregates": [
            {
                "shot_index": shot.get("shot_index"),
                "accepted": shot.get("accepted"),
                "attributed_video_id": shot.get("attributed_video_id"),
                "canonical_detected_shot": deepcopy(
                    shot.get("canonical_detected_shot")
                ),
                "dense_attribution_span": deepcopy(
                    shot.get("dense_attribution_span")
                ),
                "boundary_refinement": deepcopy(shot.get("boundary_refinement")),
            }
            for shot in shots
            if isinstance(shot, Mapping)
        ],
        "detected_boundary_labels": [
            {
                "boundary_index": boundary.get("boundary_index"),
                "incoming_shot_index": boundary.get("incoming_shot_index"),
                "old_source_id": boundary.get("old_source_id"),
                "new_source_id": boundary.get("new_source_id"),
                "supervision_classification": boundary.get(
                    "supervision_classification"
                ),
                "classification_reason": boundary.get("classification_reason"),
                "cut_target": boundary.get("cut_target"),
                "loss_masked": boundary.get("loss_masked"),
                "dense_attribution_ready": boundary.get(
                    "dense_attribution_ready"
                ),
                "boundary_refinement": deepcopy(
                    boundary.get("boundary_refinement")
                ),
            }
            for boundary in boundaries
            if isinstance(boundary, Mapping)
        ],
    }


def _validate_and_bind_partition(
    attribution: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[str, str, str]:
    if attribution.get("analysis") != REFERENCE_ATTRIBUTION_BASIS:
        raise ValueError("unsupported final attribution analysis basis")
    if evidence.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS:
        raise ValueError("unsupported reference evidence analysis basis")
    schema = attribution.get("canonical_shot_schema")
    if not isinstance(schema, Mapping) or schema.get("schema_version") != 1:
        raise ValueError("dynamic crop requires canonical detector-owned shots")
    if (
        schema.get("partition_basis") != "reference_hard_cut_detector"
        or schema.get("source_runs_are_attribution_only_not_shot_partitions")
        is not True
    ):
        raise ValueError("attribution is not detector-partition-owned")
    samples = evidence.get("samples")
    cuts = evidence.get("detected_reference_cuts")
    if not isinstance(samples, list) or not samples or not isinstance(cuts, list):
        raise ValueError("evidence samples or detected cuts are missing")
    master_start = _finite(evidence.get("master_start_seconds"), "master start")
    master_end = _finite(evidence.get("master_end_seconds"), "master end")
    if master_start < 0.0 or master_end <= master_start:
        raise ValueError("evidence master bounds are invalid")
    cut_times: list[float] = []
    previous = master_start
    for index, cut in enumerate(cuts):
        if not isinstance(cut, Mapping):
            raise ValueError(f"detected cut {index} must be an object")
        timestamp = _finite(cut.get("master_seconds"), "detected cut time")
        if not previous < timestamp < master_end:
            raise ValueError("detected cuts must be strictly chronological")
        cut_times.append(timestamp)
        previous = timestamp
    groups: list[list[int]] = [[] for _ in range(len(cut_times) + 1)]
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"evidence sample {index} must be an object")
        timestamp = _finite(sample.get("master_seconds"), "sample master time")
        if timestamp < master_start - 1e-9 or timestamp >= master_end + 1e-9:
            raise ValueError("evidence sample is outside the master clock")
        group = min(len(groups) - 1, bisect_right(cut_times, timestamp))
        groups[group].append(index)
    partition_identity = {
        "master_start_seconds": master_start,
        "master_end_seconds": master_end,
        "detected_reference_cuts": deepcopy(cuts),
        "sample_assignment": groups,
    }
    partition_sha256 = _canonical_json_sha256(partition_identity)
    if partition_sha256 != schema.get("partition_content_sha256"):
        raise ValueError("evidence does not match attribution partition hash")
    observed_identity_sha256 = _canonical_json_sha256(
        _attribution_hash_identity(attribution, partition_sha256)
    )
    # The attribution identity schema evolved while the immutable crop-v1
    # corpus was being produced.  Old final artifacts therefore carry the
    # builder-time identity hash, which cannot be recomputed with today's
    # identity projection.  Bind both values instead of falsely rewriting the
    # historical hash or rejecting an otherwise exact, file-hashed artifact.
    # The detector partition *is* version-stable and remains recomputed above.
    attribution_sha256 = schema.get("attribution_content_sha256")
    if not _is_sha256(attribution_sha256):
        raise ValueError("final attribution content hash is missing")
    provenance = attribution.get("reference_evidence_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("partition_content_sha256") != partition_sha256
        or provenance.get("attribution_content_sha256") != attribution_sha256
    ):
        raise ValueError("attribution evidence provenance is inconsistent")
    return partition_sha256, str(attribution_sha256), observed_identity_sha256


def _find_catalog_source(
    payload: Mapping[str, Any], source_id: str
) -> Mapping[str, Any]:
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("evidence source catalog is missing")
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping) and str(item.get("video_id") or "") == source_id
    ]
    if len(matches) != 1:
        raise ValueError("evidence must contain exactly one requested source")
    return matches[0]


def _validate_media_and_clocks(
    request: ReferenceDynamicCropSupervisionRequest,
    attribution: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[ReferenceSourceClock, ReferenceSourceClock, Mapping[str, Any]]:
    if Path(str(evidence.get("reference") or "")).resolve() != request.reference_media_path:
        raise ValueError("evidence reference path does not match requested media")
    reference_video = evidence.get("reference_video")
    if not isinstance(reference_video, Mapping) or Path(
        str(reference_video.get("path") or "")
    ).resolve() != request.reference_media_path:
        raise ValueError("evidence reference-video provenance is inconsistent")
    source = _find_catalog_source(evidence, request.source_id)
    if Path(str(source.get("path") or "")).resolve() != request.source_media_path:
        raise ValueError("evidence source path does not match requested media")
    attribution_source = _find_catalog_source(attribution, request.source_id)
    if _canonical_json_bytes(attribution_source) != _canonical_json_bytes(source):
        raise ValueError("attribution source catalog differs from evidence")
    clock_policy = evidence.get("clock_policy")
    if not isinstance(clock_policy, Mapping) or not isinstance(
        clock_policy.get("reference_mapping"), Mapping
    ):
        raise ValueError("evidence reference clock is missing")
    source_clock_raw = source.get("clock")
    if not isinstance(source_clock_raw, Mapping):
        raise ValueError("evidence source clock is missing")
    return (
        ReferenceSourceClock.from_mapping(clock_policy["reference_mapping"]),
        ReferenceSourceClock.from_mapping(source_clock_raw),
        source,
    )


def _validate_shot_and_samples(
    request: ReferenceDynamicCropSupervisionRequest,
    attribution: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[int, Mapping[str, Any]], ...], float, float]:
    shot = deepcopy(dict(request.attributed_shot))
    if shot.get("accepted") is not True or shot.get("attributed_video_id") != request.source_id:
        raise ValueError("dynamic crop requires an accepted shot for source_id")
    shot_index = _nonnegative_integer(shot.get("shot_index"), "shot_index")
    shots = attribution.get("shots")
    if not isinstance(shots, list) or shot_index >= len(shots) or not isinstance(
        shots[shot_index], Mapping
    ):
        raise ValueError("attribution does not contain the requested shot")
    if _canonical_json_bytes(shots[shot_index]) != _canonical_json_bytes(shot):
        raise ValueError("requested shot differs from attribution artifact")
    canonical = shot.get("canonical_detected_shot")
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("schema_version") != 1
        or canonical.get("detector_shot_index") != shot_index
        or canonical.get("source_run_merge_does_not_define_shot_partition")
        is not True
    ):
        raise ValueError("shot lacks detector-owned canonical sample provenance")
    raw_indices = canonical.get("sample_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("canonical shot has no evidence sample indices")
    indices = tuple(
        _nonnegative_integer(value, "canonical sample index") for value in raw_indices
    )
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("canonical sample indices must be unique and ordered")
    raw_samples = evidence.get("samples")
    if not isinstance(raw_samples, list) or indices[-1] >= len(raw_samples):
        raise ValueError("canonical sample index is outside evidence")
    start = _finite(shot.get("reference_start_seconds"), "shot start")
    end = _finite(shot.get("reference_end_seconds"), "shot end")
    if start < 0.0 or end <= start:
        raise ValueError("shot master span is invalid")
    selected: list[tuple[int, Mapping[str, Any]]] = []
    previous = -math.inf
    for index in indices:
        sample = raw_samples[index]
        if not isinstance(sample, Mapping):
            raise ValueError("canonical evidence sample must be an object")
        timestamp = _finite(sample.get("master_seconds"), "sample master time")
        if timestamp < start - 1e-9 or timestamp >= end + 1e-9 or timestamp <= previous:
            raise ValueError("canonical sample does not lie in its detector-owned shot")
        selected.append((index, sample))
        previous = timestamp
    return shot, tuple(selected), start, end


def _matched_candidate(sample: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    candidates = sample.get("ranked_candidates")
    if not isinstance(candidates, list):
        raise ValueError("canonical sample lacks ranked source candidates")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping) and str(item.get("video_id") or "") == source_id
    ]
    if len(matches) != 1:
        raise ValueError("canonical sample must have exactly one attributed candidate")
    candidate = matches[0]
    if candidate.get("match_status") not in (None, "matched"):
        raise ValueError("attributed source candidate was not visually matched")
    return candidate


def _renderer_keyframe(
    *, time_within_shot_seconds: float, transform: Mapping[str, Any]
) -> EditorialTransformKeyframe:
    return EditorialTransformKeyframe(
        time_within_shot_seconds,
        _finite(transform.get("zoom"), "keyframe zoom"),
        _finite(transform.get("center_x"), "keyframe center_x"),
        _finite(transform.get("center_y"), "keyframe center_y"),
        _finite(transform.get("roll_degrees"), "keyframe roll"),
    )


def _pair_interpolation_review(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, str | None]:
    left_time = _finite(left.get("time_within_shot_seconds"), "left keyframe time")
    right_time = _finite(right.get("time_within_shot_seconds"), "right keyframe time")
    if right_time <= left_time:
        return False, "non_monotonic_keyframe_time"
    try:
        left_keyframe = _renderer_keyframe(time_within_shot_seconds=0.0, transform=left)
        right_keyframe = _renderer_keyframe(
            time_within_shot_seconds=right_time - left_time,
            transform=right,
        )
        EditorialTransformTrack(
            (left_keyframe, right_keyframe),
            reason="human_reference_dynamic_crop_v2_interpolation_validation",
        )
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _motion_label(
    keyframes: Sequence[Mapping[str, Any]],
    config: ReferenceDynamicCropSupervisionConfig,
) -> tuple[str, dict[str, float]]:
    zooms = [_finite(item.get("zoom"), "keyframe zoom") for item in keyframes]
    centers = [
        (
            _finite(item.get("center_x"), "keyframe center_x"),
            _finite(item.get("center_y"), "keyframe center_y"),
        )
        for item in keyframes
    ]
    rolls = [
        _finite(item.get("roll_degrees"), "keyframe roll") for item in keyframes
    ]
    zoom_range = max(zooms) - min(zooms)
    center_range = max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )
    roll_range = max(rolls) - min(rolls)
    dynamic = (
        zoom_range >= config.dynamic_zoom_range_threshold
        or center_range >= config.dynamic_center_range_threshold
        or roll_range >= config.dynamic_roll_range_threshold_degrees
    )
    return (
        "dynamic" if dynamic else "static",
        {
            "zoom_range": zoom_range,
            "maximum_center_distance": center_range,
            "roll_range_degrees": roll_range,
        },
    )


def _sample_loss_mask(
    reason: str,
    *,
    sample_index: int,
    evidence_sample_index: int | None,
    master_seconds: float,
    time_within_shot_seconds: float,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "evidence_sample_index": evidence_sample_index,
        "master_seconds": master_seconds,
        "time_within_shot_seconds": time_within_shot_seconds,
        "accepted": False,
        "renderer_keyframe": None,
        "rejection_reason": reason,
        "loss_mask": {
            "dynamic_crop_keyframe_loss_masked": True,
            "reason": reason,
            "unsafe_or_failed_values_clamped": False,
            "substitute_keyframe_created": False,
        },
        "audit": deepcopy(dict(audit)),
    }


def _real_shot_start_observation(
    *,
    shot_start: float,
    source_id: str,
    reference_clock: ReferenceSourceClock,
    source_clock: ReferenceSourceClock,
    reference_frame_rate: float,
    source_frame_rate: float,
) -> dict[str, Any]:
    """Plan one real, clock-bound observation at detector shot time zero.

    This is not an interior keyframe moved to time zero.  Both frame indices
    are independently quantized from the immutable detector-shot start and the
    evidence-bound clocks using the evidence builder's exact ``round`` policy.
    The caller still decodes and geometry-checks the real pixels before a
    transform can become a target.
    """

    reference_seconds = reference_clock.master_to_source(shot_start)
    source_seconds = source_clock.master_to_source(shot_start)
    reference_index = int(round(reference_seconds * reference_frame_rate))
    source_index = int(round(source_seconds * source_frame_rate))
    return {
        "master_seconds": shot_start,
        "reference_frame_index": reference_index,
        "sample_reasons": ["dynamic_crop_real_detector_shot_start"],
        "dynamic_crop_observation_basis": SHOT_START_OBSERVATION_BASIS,
        "ranked_candidates": [
            {
                "video_id": source_id,
                "source_frame_index": source_index,
                "source_time_seconds": source_index / source_frame_rate,
                "expected_source_time_seconds": source_seconds,
                "requested_source_frame_indices": [source_index],
                "match_status": None,
            }
        ],
    }


def _loss_mask_track_without_real_start(
    sample_payloads: Sequence[dict[str, Any]],
) -> None:
    """Remove every otherwise-safe target when no observed t=0 target exists."""

    for sample in sample_payloads:
        if sample.get("accepted") is not True:
            continue
        sample["accepted"] = False
        sample["renderer_keyframe"] = None
        sample["rejection_reason"] = SHOT_START_UNAVAILABLE_REASON
        sample["loss_mask"] = {
            "dynamic_crop_keyframe_loss_masked": True,
            "reason": SHOT_START_UNAVAILABLE_REASON,
            "unsafe_or_failed_values_clamped": False,
            "substitute_keyframe_created": False,
        }
        audit = sample.get("audit")
        if not isinstance(audit, dict):
            raise AssertionError("accepted dynamic crop sample lost its audit")
        audit["track_rejection"] = {
            "reason": SHOT_START_UNAVAILABLE_REASON,
            "interior_keyframe_was_not_shifted_to_zero": True,
            "substitute_keyframe_created": False,
        }


def build_reference_dynamic_crop_supervision(
    request: ReferenceDynamicCropSupervisionRequest,
    *,
    config: ReferenceDynamicCropSupervisionConfig | None = None,
) -> ReferenceDynamicCropSupervisionArtifact:
    """Build a variable-keyframe crop-v2 target without mutating crop-v1."""

    if not isinstance(request, ReferenceDynamicCropSupervisionRequest):
        raise TypeError("request must be ReferenceDynamicCropSupervisionRequest")
    settings = config or ReferenceDynamicCropSupervisionConfig()
    attribution, attribution_hash = _read_bound_json(
        request.attribution_path,
        request.expected_attribution_file_sha256,
        "attribution",
    )
    evidence, evidence_hash = _read_bound_json(
        request.evidence_path,
        request.expected_evidence_file_sha256,
        "evidence",
    )
    initial_hashes = {
        "attribution": attribution_hash,
        "evidence": evidence_hash,
        "reference_media": _file_sha256(request.reference_media_path),
        "source_media": _file_sha256(request.source_media_path),
    }
    for name, expected in (
        ("reference_media", request.expected_reference_media_sha256),
        ("source_media", request.expected_source_media_sha256),
    ):
        if expected is not None and expected != initial_hashes[name]:
            raise ValueError(f"{name} SHA-256 does not match the caller expectation")
    (
        partition_sha256,
        attribution_content_sha256,
        observed_attribution_identity_sha256,
    ) = _validate_and_bind_partition(attribution, evidence)
    reference_clock, source_clock, catalog_source = _validate_media_and_clocks(
        request, attribution, evidence
    )
    shot, canonical_samples, shot_start, shot_end = _validate_shot_and_samples(
        request, attribution, evidence
    )
    reference_info = _media_info(request.reference_media_path)
    source_info = _media_info(request.source_media_path)
    reference_probe = evidence.get("reference_video")
    source_probe = catalog_source.get("media_probe")
    if not isinstance(reference_probe, Mapping) or not isinstance(source_probe, Mapping):
        raise ValueError("evidence media probes are missing")
    for name, observed, actual in (
        (
            "reference frame rate",
            reference_probe.get("frame_rate"),
            reference_info.frame_rate,
        ),
        (
            "reference frame count",
            reference_probe.get("declared_frame_count"),
            reference_info.frame_count,
        ),
        ("source frame rate", source_probe.get("frame_rate"), source_info.frame_rate),
        ("source frame count", source_probe.get("declared_frame_count"), source_info.frame_count),
    ):
        if isinstance(actual, int):
            if observed != actual:
                raise ValueError(f"{name} no longer matches evidence")
        elif not math.isclose(_finite(observed, name), actual, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"{name} no longer matches evidence")

    radius = settings.local_visual_search_radius_frames
    canonical_has_exact_start = any(
        _finite(sample.get("master_seconds"), "sample master time") == shot_start
        for _, sample in canonical_samples
    )
    observations: list[tuple[int | None, Mapping[str, Any]]] = []
    if not canonical_has_exact_start:
        observations.append(
            (
                None,
                _real_shot_start_observation(
                    shot_start=shot_start,
                    source_id=request.source_id,
                    reference_clock=reference_clock,
                    source_clock=source_clock,
                    reference_frame_rate=reference_info.frame_rate,
                    source_frame_rate=source_info.frame_rate,
                ),
            )
        )
    observations.extend(canonical_samples)
    reference_target_indices: set[int] = set()
    source_target_indices: set[int] = set()
    for _, raw_sample in observations:
        raw_reference_index = raw_sample.get("reference_frame_index")
        raw_candidate = _matched_candidate(raw_sample, request.source_id)
        raw_source_index = raw_candidate.get("source_frame_index")
        if (
            isinstance(raw_reference_index, bool)
            or not isinstance(raw_reference_index, int)
            or isinstance(raw_source_index, bool)
            or not isinstance(raw_source_index, int)
        ):
            raise ValueError("dynamic crop observation frame indices must be integers")
        reference_index = raw_reference_index
        source_index = raw_source_index
        if 0 <= reference_index < reference_info.frame_count:
            reference_target_indices.add(reference_index)
        source_target_indices.update(
            index
            for index in range(source_index - radius, source_index + radius + 1)
            if 0 <= index < source_info.frame_count
        )
    decode_config = ReferenceMatchConfig(
        working_width=settings.frame_estimator.working_width
    )
    # One forward-only union decode per media keeps frame numbers exact and
    # avoids H.264 random-seek/B-frame ambiguity.  A crop-v2 batch job can
    # further union these sets across every shot from the same source.
    reference_batch = decode_target_frames_sequential(
        request.reference_media_path,
        reference_target_indices,
        config=decode_config,
    )
    source_batch = decode_target_frames_sequential(
        request.source_media_path,
        source_target_indices,
        config=decode_config,
    )

    sample_payloads: list[dict[str, Any]] = []
    accepted_pairs: list[tuple[int, dict[str, Any]]] = []
    for sample_index, (evidence_index, raw_sample) in enumerate(observations):
        master_seconds = _finite(raw_sample.get("master_seconds"), "sample master time")
        within = master_seconds - shot_start
        raw_reference_index = raw_sample.get("reference_frame_index")
        if isinstance(raw_reference_index, bool) or not isinstance(
            raw_reference_index, int
        ):
            raise ValueError("reference observation frame index must be an integer")
        reference_index = raw_reference_index
        candidate = _matched_candidate(raw_sample, request.source_id)
        raw_source_index = candidate.get("source_frame_index")
        if isinstance(raw_source_index, bool) or not isinstance(raw_source_index, int):
            raise ValueError("source observation frame index must be an integer")
        source_index = raw_source_index
        observation_basis = str(
            raw_sample.get("dynamic_crop_observation_basis")
            or CANONICAL_SAMPLE_OBSERVATION_BASIS
        )
        is_real_shot_start_observation = master_seconds == shot_start
        source_seconds = _finite(
            candidate.get("source_time_seconds"), "matched source seconds"
        )
        expected_source_seconds = _finite(
            candidate.get("expected_source_time_seconds"),
            "candidate expected source seconds",
        )
        requested_indices = candidate.get("requested_source_frame_indices")
        if (
            not isinstance(requested_indices, list)
            or source_index not in requested_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in requested_indices
            )
        ):
            raise ValueError("matched source frame is not bound to evidence decode requests")
        reference_expected = reference_clock.master_to_source(master_seconds)
        source_expected_from_clock = source_clock.master_to_source(master_seconds)
        clock_error = settings.maximum_clock_quantization_error_frames
        if abs(reference_index / reference_info.frame_rate - reference_expected) > (
            clock_error / reference_info.frame_rate + 1e-9
        ):
            raise ValueError("reference frame does not match the evidence clock domain")
        if not math.isclose(
            expected_source_seconds,
            source_expected_from_clock,
            abs_tol=1e-6,
            rel_tol=1e-9,
        ):
            raise ValueError("source candidate expected time differs from source clock")
        if abs(source_seconds - source_index / source_info.frame_rate) > (
            clock_error / source_info.frame_rate + 1e-9
        ):
            raise ValueError("source candidate time differs from its exact frame index")
        base_audit = {
            "observation_binding": {
                "basis": observation_basis,
                "real_detector_shot_start_observation": (
                    is_real_shot_start_observation
                ),
                "evidence_sample_index": evidence_index,
                "interior_keyframe_shifted_to_shot_start": False,
                "substitute_keyframe_created": False,
            },
            "clock_binding": {
                "reference_clock": reference_clock.to_dict(),
                "source_clock": source_clock.to_dict(),
                "reference_frame_index_from_evidence": (
                    reference_index if evidence_index is not None else None
                ),
                "reference_frame_index_from_bound_clock": (
                    reference_index if evidence_index is None else None
                ),
                "matched_source_frame_index_from_evidence": (
                    source_index if evidence_index is not None else None
                ),
                "source_center_frame_index_from_bound_clock": (
                    source_index if evidence_index is None else None
                ),
                "matched_source_seconds_from_evidence": (
                    source_seconds if evidence_index is not None else None
                ),
                "expected_source_seconds_from_bound_clock": (
                    source_expected_from_clock
                ),
                "clock_values_clamped": False,
            },
            "candidate_binding": {
                "video_id": request.source_id,
                "requested_source_frame_indices": list(requested_indices),
                "match_status": candidate.get("match_status"),
                "attributed_source_from_detector_owned_shot": (
                    is_real_shot_start_observation
                ),
            },
            "local_visual_search": {
                "basis": (
                    "bound_clock_center_plus_bounded_visual_search_v1"
                    if evidence_index is None
                    else "exact_matched_frame_plus_bounded_visual_search_v1"
                ),
                "decode_policy": "one_forward_only_union_decode_per_media",
                "center_source_frame_index": source_index,
                "radius_frames": radius,
                "minimum_required_radius_frames": 2,
                "arbitrary_fractional_source_time_used": False,
            },
        }
        reference_in_domain = 0 <= reference_index < reference_info.frame_count
        source_center_in_domain = 0 <= source_index < source_info.frame_count
        if not reference_in_domain or not source_center_in_domain:
            reason = (
                "reference_frame_outside_media_domain"
                if not reference_in_domain
                else "matched_source_frame_outside_media_domain"
            )
            base_audit["decode_attempted"] = False
            sample_payloads.append(
                _sample_loss_mask(
                    reason,
                    sample_index=sample_index,
                    evidence_sample_index=evidence_index,
                    master_seconds=master_seconds,
                    time_within_shot_seconds=within,
                    audit=base_audit,
                )
            )
            continue
        try:
            reference_decoded = reference_batch.frames.get(reference_index)
            if reference_decoded is None:
                raise ValueError("reference exact frame missing from union decode")
            reference_frame = _canonical_output(
                reference_decoded, settings.frame_estimator.working_width
            )
        except (ValueError, RuntimeError) as exc:
            base_audit["decode_attempted"] = True
            base_audit["decode_error"] = str(exc)
            sample_payloads.append(
                _sample_loss_mask(
                    "reference_exact_frame_decode_failed",
                    sample_index=sample_index,
                    evidence_sample_index=evidence_index,
                    master_seconds=master_seconds,
                    time_within_shot_seconds=within,
                    audit=base_audit,
                )
            )
            continue
        reviews: list[dict[str, Any]] = []
        accepted_search: list[tuple[float, int, Any, dict[str, Any]]] = []
        for local_index in range(source_index - radius, source_index + radius + 1):
            if local_index < 0 or local_index >= source_info.frame_count:
                reviews.append(
                    {
                        "source_frame_index": local_index,
                        "frame_offset_from_evidence_match": local_index - source_index,
                        "accepted": False,
                        "rejection_reason": "local_search_frame_outside_media_domain",
                        "decode_attempted": False,
                    }
                )
                continue
            try:
                decoded = source_batch.frames.get(local_index)
                if decoded is None:
                    raise ValueError("source exact frame missing from union decode")
                source_frame = _canonical_output(
                    decoded, settings.frame_estimator.working_width
                )
                transform, rejection, diagnostics = _estimate_sample(
                    reference_frame,
                    source_frame,
                    settings.frame_estimator,
                )
            except (ValueError, RuntimeError) as exc:
                reviews.append(
                    {
                        "source_frame_index": local_index,
                        "frame_offset_from_evidence_match": local_index - source_index,
                        "accepted": False,
                        "rejection_reason": "local_exact_frame_decode_or_estimation_failed",
                        "error": str(exc),
                        "decode_attempted": True,
                    }
                )
                continue
            review = {
                "source_frame_index": local_index,
                "source_frame_seconds": local_index / source_info.frame_rate,
                "frame_offset_from_evidence_match": local_index - source_index,
                "accepted": transform is not None,
                "rejection_reason": rejection,
                "decode_attempted": True,
                "diagnostics": diagnostics,
            }
            reviews.append(review)
            if transform is not None:
                warp = diagnostics.get("actual_warp_review", {})
                score = _finite(
                    warp.get("warped_structure_ncc"), "local warped NCC"
                )
                accepted_search.append((score, local_index, transform, review))
        base_audit["local_visual_search"]["candidates"] = reviews
        if not accepted_search:
            sample_payloads.append(
                _sample_loss_mask(
                    "no_local_source_frame_passed_projected_warp_review",
                    sample_index=sample_index,
                    evidence_sample_index=evidence_index,
                    master_seconds=master_seconds,
                    time_within_shot_seconds=within,
                    audit=base_audit,
                )
            )
            continue
        accepted_search.sort(
            key=lambda item: (
                -item[0],
                abs(item[1] - source_index),
                item[1],
            )
        )
        score, selected_index, transform, selected_review = accepted_search[0]
        keyframe = {
            "time_within_shot_seconds": within,
            "master_seconds": master_seconds,
            **transform.renderer_dict(),
            "reference_frame_index": reference_index,
            "source_frame_index": selected_index,
            "source_frame_seconds": selected_index / source_info.frame_rate,
            "evidence_matched_source_frame_index": (
                source_index if evidence_index is not None else None
            ),
            "bound_clock_source_center_frame_index": (
                source_index if evidence_index is None else None
            ),
            "local_frame_offset": selected_index - source_index,
        }
        _renderer_keyframe(time_within_shot_seconds=within, transform=keyframe)
        base_audit["local_visual_search"].update(
            {
                "selected_source_frame_index": selected_index,
                "selected_frame_offset": selected_index - source_index,
                "selected_warped_structure_ncc": score,
                "selected_candidate_revalidated_after_safe_zoom_projection": True,
            }
        )
        payload = {
            "sample_index": sample_index,
            "evidence_sample_index": evidence_index,
            "master_seconds": master_seconds,
            "time_within_shot_seconds": within,
            "accepted": True,
            "renderer_keyframe": keyframe,
            "rejection_reason": None,
            "loss_mask": {
                "dynamic_crop_keyframe_loss_masked": False,
                "reason": None,
                "unsafe_or_failed_values_clamped": False,
                "substitute_keyframe_created": False,
            },
            "audit": base_audit,
        }
        sample_payloads.append(payload)
        accepted_pairs.append((len(sample_payloads) - 1, keyframe))

    interpolation_pairs: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    previous_payload_index: int | None = None
    for payload_index, keyframe in accepted_pairs:
        if not retained:
            retained.append(keyframe)
            previous_payload_index = payload_index
            continue
        safe, error = _pair_interpolation_review(retained[-1], keyframe)
        if previous_payload_index is None:
            raise AssertionError("retained dynamic crop keyframe lost its sample index")
        interpolation_pairs.append(
            {
                "left_evidence_sample_index": sample_payloads[
                    previous_payload_index
                ]["evidence_sample_index"],
                "right_evidence_sample_index": sample_payloads[payload_index][
                    "evidence_sample_index"
                ],
                "safe": safe,
                "renderer_validation_error": error,
                "validation_basis": "EditorialTransformTrack_linear_interpolation_16_steps",
            }
        )
        if safe:
            retained.append(keyframe)
            previous_payload_index = payload_index
            continue
        sample = sample_payloads[payload_index]
        reason = "renderer_interpolation_unsafe_from_previous_eligible_keyframe"
        sample["accepted"] = False
        sample["renderer_keyframe"] = None
        sample["rejection_reason"] = reason
        sample["loss_mask"] = {
            "dynamic_crop_keyframe_loss_masked": True,
            "reason": reason,
            "unsafe_or_failed_values_clamped": False,
            "substitute_keyframe_created": False,
        }
        sample["audit"]["interpolation_rejection"] = {
            "previous_retained_keyframe": deepcopy(retained[-1]),
            "renderer_validation_error": error,
        }

    real_start_retained = bool(retained) and _finite(
        retained[0].get("time_within_shot_seconds"),
        "first retained keyframe time",
    ) == 0.0
    if retained and not real_start_retained:
        _loss_mask_track_without_real_start(sample_payloads)
        retained = []

    eligible = bool(retained) and real_start_retained
    if eligible:
        motion_label, motion_statistics = _motion_label(retained, settings)
        motion_label_masked = len(retained) < 2
        motion_label_mask_reason = (
            "insufficient_keyframes_to_classify_motion"
            if motion_label_masked
            else None
        )
        rejection_reason = None
    else:
        motion_label = None
        motion_statistics = None
        motion_label_masked = True
        motion_label_mask_reason = "dynamic_crop_track_loss_masked"
        rejection_reason = (
            SHOT_START_UNAVAILABLE_REASON
            if not real_start_retained
            else "no_renderer_safe_dynamic_crop_keyframes"
        )
    final_hashes = {
        "attribution": _file_sha256(request.attribution_path),
        "evidence": _file_sha256(request.evidence_path),
        "reference_media": _file_sha256(request.reference_media_path),
        "source_media": _file_sha256(request.source_media_path),
    }
    if final_hashes != initial_hashes:
        raise ValueError("an input changed during dynamic crop supervision")
    fingerprint_identity = {
        "basis": FINGERPRINT_BASIS,
        "analysis_basis": ANALYSIS_BASIS,
        "target_contract": DYNAMIC_CROP_TARGET_CONTRACT,
        "configuration": settings.to_dict(),
        "input_hashes": initial_hashes,
        "shot_content_sha256": _canonical_json_sha256(shot),
        "source_id": request.source_id,
        "namespace": DYNAMIC_CROP_NAMESPACE,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_basis": ANALYSIS_BASIS,
        "namespace": DYNAMIC_CROP_NAMESPACE,
        "direction": "source_media_output_to_reference_edit_output",
        "target_contract": {
            "name": DYNAMIC_CROP_TARGET_CONTRACT,
            "target_only_no_reference_match_features": True,
            "legacy_static_four_vector_crop_head_compatible": False,
            "variable_keyframe_count": True,
            "interpolation": "production_renderer_piecewise_linear",
            "real_observed_shot_start_keyframe_required": True,
            "interior_keyframes_may_not_be_shifted_to_zero": True,
        },
        "shot": {
            "shot_index": shot["shot_index"],
            "source_id": request.source_id,
            "master_start_seconds": shot_start,
            "master_end_seconds": shot_end,
            "duration_seconds": shot_end - shot_start,
            "attributed_shot_content_sha256": _canonical_json_sha256(shot),
            "canonical_detected_shot": deepcopy(shot["canonical_detected_shot"]),
        },
        "configuration": settings.to_dict(),
        "samples": sample_payloads,
        "interpolation_safety_review": {
            "all_retained_pairs_safe": all(
                _pair_interpolation_review(left, right)[0]
                for left, right in zip(retained, retained[1:])
            ),
            "all_candidate_pairs_safe": all(
                item["safe"] for item in interpolation_pairs
            ),
            "candidate_pair_reviews": interpolation_pairs,
            "unsafe_samples_loss_masked_without_substitution": True,
        },
        "supervision": {
            "eligible": eligible,
            "motion_label": motion_label,
            "motion_statistics": motion_statistics,
            "motion_label_loss_mask": {
                "masked": motion_label_masked,
                "reason": motion_label_mask_reason,
                "single_keyframe_does_not_prove_static_motion": True,
            },
            "renderer_coordinate_system": (
                "EditorialTransformKeyframe_zoom_center_x_center_y_roll_degrees"
            ),
            "renderer_keyframes": retained,
            "keyframe_count": len(retained),
            "start_keyframe": deepcopy(retained[0]) if retained else None,
            "end_keyframe": deepcopy(retained[-1]) if retained else None,
            "rejection_reason": rejection_reason,
            "loss_mask": {
                "dynamic_crop_track_loss_masked": not eligible,
                "reason": rejection_reason,
                "unsafe_or_failed_values_clamped": False,
                "substitute_keyframes_created": False,
            },
        },
        "provenance": {
            "content_hash_basis": CONTENT_HASH_BASIS,
            "artifact_content_sha256": None,
            "build_fingerprint_basis": FINGERPRINT_BASIS,
            "build_fingerprint_sha256": _canonical_json_sha256(
                fingerprint_identity
            ),
            "canonical_partition_content_sha256": partition_sha256,
            "canonical_attribution_content_sha256": attribution_content_sha256,
            "current_code_attribution_identity_sha256": (
                observed_attribution_identity_sha256
            ),
            "attribution": {
                "path": str(request.attribution_path),
                "file_sha256": initial_hashes["attribution"],
            },
            "evidence": {
                "path": str(request.evidence_path),
                "file_sha256": initial_hashes["evidence"],
            },
            "reference_media": {
                **reference_info.to_dict(sha256=initial_hashes["reference_media"]),
                "clock": reference_clock.to_dict(),
            },
            "source_media": {
                **source_info.to_dict(sha256=initial_hashes["source_media"]),
                "clock": source_clock.to_dict(),
            },
            "caller": deepcopy(dict(request.caller_provenance)),
            "input_hashes_rechecked_after_decode": True,
            "storage_policy": {
                "pixels_serialized": False,
                "descriptors_serialized": False,
                "matched_keypoints_serialized": False,
                "identity_or_appearance_embeddings_computed": False,
            },
        },
    }
    payload["provenance"]["artifact_content_sha256"] = _artifact_content_sha256(
        payload
    )
    validate_reference_dynamic_crop_supervision_payload(payload)
    return ReferenceDynamicCropSupervisionArtifact(payload)


def validate_reference_dynamic_crop_supervision_payload(
    value: Mapping[str, Any],
    *,
    verify_input_files: bool = False,
) -> None:
    """Validate crop-v2 identity, masks, keyframes, and interpolation."""

    if not isinstance(value, Mapping):
        raise TypeError("dynamic crop payload must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dynamic crop schema version")
    if value.get("analysis_basis") != ANALYSIS_BASIS:
        raise ValueError("unsupported dynamic crop analysis basis")
    if value.get("namespace") != DYNAMIC_CROP_NAMESPACE:
        raise ValueError("dynamic crop namespace is invalid")
    target_contract = value.get("target_contract")
    if (
        not isinstance(target_contract, Mapping)
        or target_contract.get("name") != DYNAMIC_CROP_TARGET_CONTRACT
        or target_contract.get("legacy_static_four_vector_crop_head_compatible")
        is not False
        or target_contract.get("target_only_no_reference_match_features") is not True
        or target_contract.get("real_observed_shot_start_keyframe_required")
        is not True
        or target_contract.get("interior_keyframes_may_not_be_shifted_to_zero")
        is not True
    ):
        raise ValueError("dynamic crop target contract is invalid")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("dynamic crop provenance is missing")
    if provenance.get("content_hash_basis") != CONTENT_HASH_BASIS:
        raise ValueError("dynamic crop content-hash basis is invalid")
    content_hash = provenance.get("artifact_content_sha256")
    if not _is_sha256(content_hash) or content_hash != _artifact_content_sha256(value):
        raise ValueError("dynamic crop artifact content hash mismatch")
    if (
        provenance.get("build_fingerprint_basis") != FINGERPRINT_BASIS
        or not _is_sha256(provenance.get("build_fingerprint_sha256"))
        or not _is_sha256(provenance.get("canonical_partition_content_sha256"))
        or not _is_sha256(provenance.get("canonical_attribution_content_sha256"))
        or not _is_sha256(
            provenance.get("current_code_attribution_identity_sha256")
        )
    ):
        raise ValueError("dynamic crop fingerprint provenance is invalid")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("dynamic crop must retain canonical sample audits")
    shot = value.get("shot")
    if not isinstance(shot, Mapping):
        raise ValueError("dynamic crop shot identity is missing")
    shot_duration = _finite(shot.get("duration_seconds"), "shot duration")
    if shot_duration <= 0.0:
        raise ValueError("dynamic crop shot duration must be positive")
    accepted_from_samples: list[Mapping[str, Any]] = []
    previous_time = -math.inf
    for expected_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or sample.get("sample_index") != expected_index:
            raise ValueError("dynamic crop sample ordering is invalid")
        timestamp = _finite(
            sample.get("time_within_shot_seconds"), "sample time within shot"
        )
        if timestamp < 0.0 or timestamp <= previous_time:
            raise ValueError("dynamic crop sample times must be ordered")
        if timestamp > shot_duration + 1e-8:
            raise ValueError("dynamic crop sample lies outside its shot")
        previous_time = timestamp
        accepted = sample.get("accepted")
        mask = sample.get("loss_mask")
        if not isinstance(accepted, bool) or not isinstance(mask, Mapping):
            raise ValueError("dynamic crop sample decision is invalid")
        if (
            mask.get("unsafe_or_failed_values_clamped") is not False
            or mask.get("substitute_keyframe_created") is not False
        ):
            raise ValueError("dynamic crop sample may not clamp or substitute failures")
        keyframe = sample.get("renderer_keyframe")
        if accepted:
            if not isinstance(keyframe, Mapping) or sample.get("rejection_reason") is not None:
                raise ValueError("accepted dynamic crop sample needs one keyframe")
            if mask.get("dynamic_crop_keyframe_loss_masked") is not False:
                raise ValueError("accepted dynamic crop keyframe cannot be loss-masked")
            if not math.isclose(
                _finite(
                    keyframe.get("time_within_shot_seconds"),
                    "serialized keyframe time",
                ),
                timestamp,
                abs_tol=1e-9,
                rel_tol=0.0,
            ):
                raise ValueError("dynamic crop sample/keyframe times differ")
            if timestamp == 0.0:
                audit = sample.get("audit")
                binding = (
                    audit.get("observation_binding")
                    if isinstance(audit, Mapping)
                    else None
                )
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("real_detector_shot_start_observation") is not True
                    or binding.get("interior_keyframe_shifted_to_shot_start")
                    is not False
                    or binding.get("substitute_keyframe_created") is not False
                    or binding.get("basis")
                    not in {
                        SHOT_START_OBSERVATION_BASIS,
                        CANONICAL_SAMPLE_OBSERVATION_BASIS,
                    }
                ):
                    raise ValueError(
                        "dynamic crop shot-start keyframe lacks real observation provenance"
                    )
            _renderer_keyframe(time_within_shot_seconds=timestamp, transform=keyframe)
            accepted_from_samples.append(keyframe)
        else:
            if keyframe is not None or not isinstance(sample.get("rejection_reason"), str):
                raise ValueError("rejected dynamic crop sample must be target-free")
            if mask.get("dynamic_crop_keyframe_loss_masked") is not True:
                raise ValueError("rejected dynamic crop sample must be loss-masked")
    supervision = value.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("dynamic crop supervision decision is missing")
    eligible = supervision.get("eligible")
    keyframes = supervision.get("renderer_keyframes")
    track_mask = supervision.get("loss_mask")
    motion_mask = supervision.get("motion_label_loss_mask")
    if not isinstance(eligible, bool) or not isinstance(keyframes, list) or not isinstance(
        track_mask, Mapping
    ) or not isinstance(motion_mask, Mapping):
        raise ValueError("dynamic crop supervision schema is invalid")
    if keyframes != accepted_from_samples or supervision.get("keyframe_count") != len(
        keyframes
    ):
        raise ValueError("dynamic crop retained keyframes differ from sample masks")
    if (
        track_mask.get("unsafe_or_failed_values_clamped") is not False
        or track_mask.get("substitute_keyframes_created") is not False
    ):
        raise ValueError("dynamic crop track may not clamp or substitute failures")
    if eligible:
        if not keyframes or supervision.get("motion_label") not in {"static", "dynamic"}:
            raise ValueError("eligible dynamic crop needs labeled keyframes")
        if _finite(
            keyframes[0].get("time_within_shot_seconds"),
            "dynamic crop start keyframe time",
        ) != 0.0:
            raise ValueError(
                "eligible dynamic crop must begin with a real shot-start keyframe"
            )
        if supervision.get("rejection_reason") is not None or track_mask.get(
            "dynamic_crop_track_loss_masked"
        ) is not False:
            raise ValueError("eligible dynamic crop decision is inconsistent")
        if supervision.get("start_keyframe") != keyframes[0] or supervision.get(
            "end_keyframe"
        ) != keyframes[-1]:
            raise ValueError("dynamic crop start/end targets are inconsistent")
        configuration = value.get("configuration")
        if not isinstance(configuration, Mapping) or not isinstance(
            configuration.get("frame_estimator"), Mapping
        ):
            raise ValueError("dynamic crop configuration is missing")
        estimator_values = dict(configuration["frame_estimator"])
        if isinstance(estimator_values.get("sample_fractions"), list):
            estimator_values["sample_fractions"] = tuple(
                estimator_values["sample_fractions"]
            )
        checked_config = ReferenceDynamicCropSupervisionConfig(
            frame_estimator=ReferenceCropSupervisionConfig(**estimator_values),
            local_visual_search_radius_frames=configuration.get(
                "local_visual_search_radius_frames"
            ),
            maximum_clock_quantization_error_frames=configuration.get(
                "maximum_clock_quantization_error_frames"
            ),
            dynamic_zoom_range_threshold=configuration.get(
                "dynamic_zoom_range_threshold"
            ),
            dynamic_center_range_threshold=configuration.get(
                "dynamic_center_range_threshold"
            ),
            dynamic_roll_range_threshold_degrees=configuration.get(
                "dynamic_roll_range_threshold_degrees"
            ),
        )
        expected_label, expected_statistics = _motion_label(
            keyframes, checked_config
        )
        if (
            supervision.get("motion_label") != expected_label
            or supervision.get("motion_statistics") != expected_statistics
        ):
            raise ValueError("dynamic crop motion label/statistics are inconsistent")
        expected_motion_mask = len(keyframes) < 2
        if (
            motion_mask.get("masked") is not expected_motion_mask
            or motion_mask.get("single_keyframe_does_not_prove_static_motion")
            is not True
            or motion_mask.get("reason")
            != (
                "insufficient_keyframes_to_classify_motion"
                if expected_motion_mask
                else None
            )
        ):
            raise ValueError("dynamic crop motion-label loss mask is inconsistent")
    else:
        if keyframes or supervision.get("motion_label") is not None:
            raise ValueError("ineligible dynamic crop must be target-free")
        if not isinstance(supervision.get("rejection_reason"), str) or track_mask.get(
            "dynamic_crop_track_loss_masked"
        ) is not True:
            raise ValueError("ineligible dynamic crop must be loss-masked")
        if (
            motion_mask.get("masked") is not True
            or motion_mask.get("reason") != "dynamic_crop_track_loss_masked"
            or motion_mask.get("single_keyframe_does_not_prove_static_motion")
            is not True
        ):
            raise ValueError("ineligible dynamic crop motion label must be masked")
    for left, right in zip(keyframes, keyframes[1:]):
        safe, _ = _pair_interpolation_review(left, right)
        if not safe:
            raise ValueError("serialized dynamic crop interpolation is unsafe")
    interpolation = value.get("interpolation_safety_review")
    if (
        not isinstance(interpolation, Mapping)
        or interpolation.get("all_retained_pairs_safe") is not True
        or interpolation.get("unsafe_samples_loss_masked_without_substitution")
        is not True
        or not isinstance(interpolation.get("candidate_pair_reviews"), list)
    ):
        raise ValueError("dynamic crop interpolation review is invalid")
    for name, digest_name in (
        ("attribution", "file_sha256"),
        ("evidence", "file_sha256"),
        ("reference_media", "media_sha256"),
        ("source_media", "media_sha256"),
    ):
        item = provenance.get(name)
        if not isinstance(item, Mapping) or not _is_sha256(item.get(digest_name)):
            raise ValueError(f"dynamic crop {name} provenance is invalid")
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute():
            raise ValueError(f"dynamic crop {name} path must be absolute")
        if verify_input_files and (
            not path.is_file() or _file_sha256(path) != item[digest_name]
        ):
            raise ValueError(f"dynamic crop {name} no longer matches its input")


def load_reference_dynamic_crop_supervision(
    path: Path,
    *,
    verify_input_files: bool = True,
) -> ReferenceDynamicCropSupervisionArtifact:
    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("dynamic crop JSON root must be an object")
    validate_reference_dynamic_crop_supervision_payload(
        value, verify_input_files=verify_input_files
    )
    return ReferenceDynamicCropSupervisionArtifact(deepcopy(dict(value)))


__all__ = [
    "ANALYSIS_BASIS",
    "CONTENT_HASH_BASIS",
    "DYNAMIC_CROP_NAMESPACE",
    "DYNAMIC_CROP_TARGET_CONTRACT",
    "FINGERPRINT_BASIS",
    "SCHEMA_VERSION",
    "ReferenceDynamicCropSupervisionArtifact",
    "ReferenceDynamicCropSupervisionConfig",
    "ReferenceDynamicCropSupervisionRequest",
    "build_reference_dynamic_crop_supervision",
    "load_reference_dynamic_crop_supervision",
    "reference_dynamic_crop_v2_filename",
    "validate_reference_dynamic_crop_supervision_payload",
]
