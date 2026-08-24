"""Promote only frame-proven reference source switches into supervision.

The first attribution pass remains authoritative for UNKNOWN handling.  This
module inspects the evidence builder's exact ``-2..+2`` hard-cut samples and
adds a ``boundary_request`` only when that path contains one stable, direct,
known A->B change.  It never turns a detected scene change, an UNKNOWN span,
or a same-source broadcaster cut into editorial supervision by itself.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    UNKNOWN_SOURCE_ID,
    ReferenceAttributionConfig,
    resolve_reference_attribution_payload,
)
from .reference_evidence_builder import ReferenceSourceClock


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "unknown_aware_reference_boundary_augmentation_v1"
_REQUIRED_BOUNDARY_OFFSETS = (-2, -1, 0, 1, 2)
_TIME_ABSOLUTE_TOLERANCE_SECONDS = 1e-9


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _candidate_source_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("video_id") or candidate.get("source_id") or "").strip()


def _ranked_candidates(sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_candidates = _array(
        sample.get("ranked_candidates", sample.get("candidates")),
        "reference sample ranked_candidates",
    )
    candidates: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, f"reference candidate {index}")
        source_id = _candidate_source_id(candidate)
        if not source_id or source_id == UNKNOWN_SOURCE_ID:
            raise ValueError("reference candidate needs a known source id")
        if source_id in seen:
            raise ValueError("reference candidates cannot repeat a source")
        # Parse the exact fields consumed by the attribution gates now.  This
        # prevents malformed evidence from being silently treated as UNKNOWN.
        _finite(candidate.get("score"), "candidate score")
        _integer(
            candidate.get(
                "orb_homography_inliers",
                candidate.get("homography_inliers"),
            ),
            "candidate homography inliers",
        )
        _finite(
            candidate.get(
                "orb_inlier_ratio",
                candidate.get("homography_inlier_ratio"),
            ),
            "candidate homography inlier ratio",
        )
        seen.add(source_id)
        candidates.append(candidate)
    return tuple(candidates)


def _candidate_by_source(
    candidates: Sequence[Mapping[str, Any]],
    source_id: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            candidate
            for candidate in candidates
            if _candidate_source_id(candidate) == source_id
        ),
        None,
    )


def _candidate_passes_existing_gates(
    candidate: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    config: ReferenceAttributionConfig,
) -> bool:
    """Mirror the resolver's existing candidate gate without relaxing it."""

    if candidate is None:
        return False
    source_id = _candidate_source_id(candidate)
    score = _finite(candidate.get("score"), "candidate score")
    inliers = _integer(
        candidate.get(
            "orb_homography_inliers",
            candidate.get("homography_inliers"),
        ),
        "candidate homography inliers",
    )
    inlier_ratio = _finite(
        candidate.get(
            "orb_inlier_ratio",
            candidate.get("homography_inlier_ratio"),
        ),
        "candidate homography inlier ratio",
    )
    runner_score = max(
        (
            _finite(other.get("score"), "runner-up score")
            for other in candidates
            if _candidate_source_id(other) != source_id
        ),
        default=0.0,
    )
    margin = score - runner_score
    return bool(
        score >= config.minimum_score
        and inliers >= config.minimum_homography_inliers
        and inlier_ratio >= config.minimum_homography_inlier_ratio
        and (
            score >= config.high_confidence_score
            or margin >= config.minimum_top_two_margin
        )
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


@dataclass(frozen=True, slots=True)
class ReferenceBoundaryAugmentation:
    """An augmented evidence payload and both resolver passes."""

    payload: dict[str, Any]
    initial_attribution: dict[str, Any]
    resolved_attribution: dict[str, Any]
    added_to_run_indices: tuple[int, ...]
    skip_reason_counts: Mapping[str, int]

    @property
    def added_request_count(self) -> int:
        return len(self.added_to_run_indices)

    @property
    def resolved_hard_cut_count(self) -> int:
        return sum(
            refinement.get("status") == "resolved_hard_cut"
            for refinement in self.resolved_attribution["boundary_refinements"]
        )

    @property
    def added_resolved_hard_cut_count(self) -> int:
        added = set(self.added_to_run_indices)
        return sum(
            refinement.get("status") == "resolved_hard_cut"
            and refinement.get("to_run_index") in added
            for refinement in self.resolved_attribution["boundary_refinements"]
        )

    def write_evidence(self, output_path: Path) -> Path:
        return _atomic_write_json(output_path, self.payload)

    def write_attribution(self, output_path: Path) -> Path:
        return _atomic_write_json(output_path, self.resolved_attribution)


def _reference_clock(evidence: Mapping[str, Any]) -> ReferenceSourceClock:
    clock_policy = _mapping(evidence.get("clock_policy"), "clock_policy")
    reference_mapping = _mapping(
        clock_policy.get("reference_mapping"),
        "clock_policy.reference_mapping",
    )
    # Reuse the builder's bidirectional clock implementation so scalar and
    # piecewise source_to_master evidence retain exactly the same semantics.
    return ReferenceSourceClock.from_mapping(reference_mapping)


def _cut_sample_indices(
    samples: Sequence[Mapping[str, Any]],
    *,
    cut_frame_index: int,
) -> tuple[int, ...] | None:
    indices: list[int] = []
    for offset in _REQUIRED_BOUNDARY_OFFSETS:
        frame_index = cut_frame_index + offset
        expected_reason = f"hard_cut_boundary:{cut_frame_index}:{offset:+d}"
        matches: list[int] = []
        for sample_index, sample in enumerate(samples):
            if sample.get("reference_frame_index") != frame_index:
                continue
            reasons = sample.get("sample_reasons")
            if not isinstance(reasons, list) or expected_reason not in reasons:
                continue
            matches.append(sample_index)
        if len(matches) != 1:
            return None
        indices.append(matches[0])
    return tuple(indices)


def _exact_boundary_clock_matches(
    samples: Sequence[Mapping[str, Any]],
    sample_indices: Sequence[int],
    *,
    cut_frame_index: int,
    frame_rate: float,
    master_start_seconds: float,
    reference_clock: ReferenceSourceClock,
) -> bool:
    previous_time: float | None = None
    for offset, sample_index in zip(
        _REQUIRED_BOUNDARY_OFFSETS,
        sample_indices,
        strict=True,
    ):
        sample = samples[sample_index]
        frame_index = _integer(
            sample.get("reference_frame_index"),
            "reference frame index",
        )
        if frame_index != cut_frame_index + offset:
            return False
        actual_time = _finite(sample.get("master_seconds"), "sample master time")
        expected_time = master_start_seconds + reference_clock.source_to_master(
            frame_index / frame_rate
        )
        if expected_time < -_TIME_ABSOLUTE_TOLERANCE_SECONDS:
            return False
        expected_time = max(0.0, expected_time)
        if abs(actual_time - expected_time) > _TIME_ABSOLUTE_TOLERANCE_SECONDS:
            return False
        if previous_time is not None and actual_time <= previous_time:
            return False
        previous_time = actual_time
    return True


def _selected_known_source(
    sample: Mapping[str, Any],
    path_state: Mapping[str, Any],
    *,
    config: ReferenceAttributionConfig,
) -> str | None:
    selected = path_state.get("attributed_video_id")
    if not path_state.get("accepted") or selected is None:
        return None
    source_id = str(selected).strip()
    if not source_id or source_id == UNKNOWN_SOURCE_ID:
        return None
    candidates = _ranked_candidates(sample)
    candidate = _candidate_by_source(candidates, source_id)
    if not _candidate_passes_existing_gates(candidate, candidates, config):
        return None
    return source_id


def _single_stable_change(states: Sequence[str | None]) -> tuple[int, str, str] | None:
    if len(states) != len(_REQUIRED_BOUNDARY_OFFSETS) or any(
        state is None for state in states
    ):
        return None
    known = tuple(str(state) for state in states)
    changes = tuple(
        index
        for index in range(1, len(known))
        if known[index] != known[index - 1]
    )
    if len(changes) != 1:
        return None
    split = changes[0]
    old_source = known[0]
    new_source = known[-1]
    if (
        old_source == new_source
        or split < 2
        or len(known) - split < 2
        or any(source != old_source for source in known[:split])
        or any(source != new_source for source in known[split:])
    ):
        return None
    return (split, old_source, new_source)


def _frame_identity_evidence(
    sample: Mapping[str, Any],
    *,
    old_source_id: str,
    new_source_id: str,
    expected_identity: str,
    config: ReferenceAttributionConfig,
) -> dict[str, Any] | None:
    candidates = _ranked_candidates(sample)
    old_candidate = _candidate_by_source(candidates, old_source_id)
    new_candidate = _candidate_by_source(candidates, new_source_id)
    old_score = (
        _finite(old_candidate.get("score"), "old source score")
        if old_candidate is not None
        else 0.0
    )
    new_score = (
        _finite(new_candidate.get("score"), "new source score")
        if new_candidate is not None
        else 0.0
    )
    old_confident = _candidate_passes_existing_gates(
        old_candidate,
        candidates,
        config,
    )
    new_confident = _candidate_passes_existing_gates(
        new_candidate,
        candidates,
        config,
    )
    if old_confident and new_confident:
        difference = old_score - new_score
        if abs(difference) < config.boundary_identity_margin:
            identity = "ambiguous"
        else:
            identity = "old" if difference > 0.0 else "new"
    elif old_confident:
        identity = "old"
    elif new_confident:
        identity = "new"
    else:
        identity = "neither"
    if identity != expected_identity:
        return None
    return {
        "frame_index": _integer(
            sample.get("reference_frame_index"),
            "reference frame index",
        ),
        "master_seconds": _finite(
            sample.get("master_seconds"),
            "sample master time",
        ),
        "old_score": old_score,
        "new_score": new_score,
        "old_confident": old_confident,
        "new_confident": new_confident,
    }


def _matching_run_index(
    source_runs: Sequence[Mapping[str, Any]],
    *,
    previous_sample_index: int,
    incoming_sample_index: int,
    old_source_id: str,
    new_source_id: str,
    nominal_boundary_seconds: float,
    config: ReferenceAttributionConfig,
) -> int | None:
    matches: list[int] = []
    for to_run_index in range(1, len(source_runs)):
        previous = source_runs[to_run_index - 1]
        incoming = source_runs[to_run_index]
        if (
            previous.get("attributed_video_id") != old_source_id
            or incoming.get("attributed_video_id") != new_source_id
            or previous.get("last_sample_index") != previous_sample_index
            or incoming.get("first_sample_index") != incoming_sample_index
        ):
            continue
        start = _finite(
            incoming.get("start_master_seconds"),
            "incoming source-run start",
        )
        if (
            abs(start - nominal_boundary_seconds)
            <= config.refinement_radius_seconds + _TIME_ABSOLUTE_TOLERANCE_SECONDS
        ):
            matches.append(to_run_index)
    return matches[0] if len(matches) == 1 else None


def augment_reference_boundary_requests(
    evidence: Mapping[str, Any],
    *,
    config: ReferenceAttributionConfig | None = None,
) -> ReferenceBoundaryAugmentation:
    """Resolve, strictly augment, and resolve again without mutating input."""

    settings = config or ReferenceAttributionConfig()
    initial = resolve_reference_attribution_payload(evidence, config=settings)
    if initial.get("analysis") != REFERENCE_ATTRIBUTION_BASIS:
        raise ValueError("unexpected initial attribution analysis basis")

    raw_samples = _array(evidence.get("samples"), "reference evidence samples")
    samples = tuple(
        _mapping(sample, f"reference sample {index}")
        for index, sample in enumerate(raw_samples)
    )
    raw_cuts = _array(
        evidence.get("detected_reference_cuts"),
        "detected_reference_cuts",
    )
    cuts = tuple(
        _mapping(cut, f"detected reference cut {index}")
        for index, cut in enumerate(raw_cuts)
    )
    source_path = tuple(
        _mapping(state, f"source path state {index}")
        for index, state in enumerate(
            _array(initial.get("source_path"), "initial source_path")
        )
    )
    if len(source_path) != len(samples):
        raise ValueError("source path must contain one state per evidence sample")
    source_runs = tuple(
        _mapping(run, f"source run {index}")
        for index, run in enumerate(
            _array(initial.get("source_runs"), "initial source_runs")
        )
    )

    reference_video = _mapping(evidence.get("reference_video"), "reference_video")
    frame_rate = _finite(reference_video.get("frame_rate"), "reference frame rate")
    if frame_rate <= 0.0:
        raise ValueError("reference frame rate must be positive")
    master_start = _finite(
        evidence.get("master_start_seconds"),
        "master_start_seconds",
    )
    reference_clock = _reference_clock(evidence)

    existing_requests = deepcopy(
        _array(evidence.get("boundary_requests", []), "boundary_requests")
    )
    claimed_run_indices = {
        request.get("to_run_index")
        for request in existing_requests
        if isinstance(request, Mapping)
        and isinstance(request.get("to_run_index"), int)
        and not isinstance(request.get("to_run_index"), bool)
    }
    added_requests: list[dict[str, Any]] = []
    added_run_indices: list[int] = []
    skips: Counter[str] = Counter()

    for cut in sorted(
        cuts,
        key=lambda item: _integer(item.get("frame_index"), "cut frame index"),
    ):
        cut_frame = _integer(cut.get("frame_index"), "cut frame index")
        nominal_seconds = _finite(
            cut.get("master_seconds"),
            "cut master seconds",
        )
        sample_indices = _cut_sample_indices(
            samples,
            cut_frame_index=cut_frame,
        )
        if sample_indices is None:
            skips["missing_or_duplicate_boundary_sample"] += 1
            continue
        if not _exact_boundary_clock_matches(
            samples,
            sample_indices,
            cut_frame_index=cut_frame,
            frame_rate=frame_rate,
            master_start_seconds=master_start,
            reference_clock=reference_clock,
        ):
            skips["non_consecutive_or_clock_mismatched_frames"] += 1
            continue

        selected_sources = tuple(
            _selected_known_source(
                samples[sample_index],
                source_path[sample_index],
                config=settings,
            )
            for sample_index in sample_indices
        )
        change = _single_stable_change(selected_sources)
        if change is None:
            if any(source is None for source in selected_sources):
                skips["unknown_or_failed_candidate_gate"] += 1
            elif len(set(selected_sources)) == 1:
                skips["same_source_native_cut"] += 1
            else:
                skips["ambiguous_or_unstable_source_change"] += 1
            continue
        split, old_source, new_source = change

        frame_evidence: list[dict[str, Any]] = []
        identities_stable = True
        for local_index, sample_index in enumerate(sample_indices):
            item = _frame_identity_evidence(
                samples[sample_index],
                old_source_id=old_source,
                new_source_id=new_source,
                expected_identity="old" if local_index < split else "new",
                config=settings,
            )
            if item is None:
                identities_stable = False
                break
            frame_evidence.append(item)
        if not identities_stable:
            skips["ambiguous_or_failed_frame_identity_gate"] += 1
            continue

        to_run_index = _matching_run_index(
            source_runs,
            previous_sample_index=sample_indices[split - 1],
            incoming_sample_index=sample_indices[split],
            old_source_id=old_source,
            new_source_id=new_source,
            nominal_boundary_seconds=nominal_seconds,
            config=settings,
        )
        if to_run_index is None:
            skips["no_exact_source_run_boundary_match"] += 1
            continue
        if to_run_index in claimed_run_indices:
            skips["source_run_boundary_already_requested"] += 1
            continue

        added_requests.append(
            {
                "to_run_index": to_run_index,
                "old_source_id": old_source,
                "new_source_id": new_source,
                "nominal_boundary_seconds": nominal_seconds,
                "nominal_frame_index": cut_frame,
                "frame_evidence": frame_evidence,
            }
        )
        added_run_indices.append(to_run_index)
        claimed_run_indices.add(to_run_index)

    augmented = deepcopy(dict(evidence))
    augmented["boundary_requests"] = [*existing_requests, *added_requests]
    statistics = augmented.get("statistics", {})
    if not isinstance(statistics, Mapping):
        raise ValueError("reference evidence statistics must be an object")
    augmented_statistics = deepcopy(dict(statistics))
    augmented_statistics.update(
        {
            "boundary_request_count": len(augmented["boundary_requests"]),
            "strict_boundary_request_added_count": len(added_requests),
        }
    )
    augmented["statistics"] = augmented_statistics
    augmented["boundary_request_augmentation"] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "created_at": datetime.now(UTC).isoformat(),
        "initial_resolver_analysis": initial["analysis"],
        "policy": {
            "required_boundary_offsets_frames": list(_REQUIRED_BOUNDARY_OFFSETS),
            "minimum_consecutive_known_frames_per_side": 2,
            "single_direct_known_source_change_required": True,
            "unknown_preserved": True,
            "same_source_native_cut_excluded": True,
            "exact_builder_clock_revalidation": True,
            "existing_candidate_gates_reused_without_weakening": settings.to_dict()[
                "acceptance"
            ],
            "boundary_refinement": settings.to_dict()["boundary_refinement"],
        },
        "detected_cut_count": len(cuts),
        "existing_request_count": len(existing_requests),
        "added_request_count": len(added_requests),
        "added_to_run_indices": list(added_run_indices),
        "skip_reason_counts": dict(sorted(skips.items())),
    }
    json.dumps(augmented, ensure_ascii=False, allow_nan=False)

    resolved = resolve_reference_attribution_payload(augmented, config=settings)
    resolved_by_run = {
        refinement.get("to_run_index"): refinement
        for refinement in resolved["boundary_refinements"]
        if isinstance(refinement.get("to_run_index"), int)
    }
    for to_run_index in added_run_indices:
        refinement = resolved_by_run.get(to_run_index)
        if refinement is None or refinement.get("status") != "resolved_hard_cut":
            raise ValueError(
                "strict boundary request did not resolve as a hard cut for "
                f"to_run_index {to_run_index}"
            )

    return ReferenceBoundaryAugmentation(
        payload=augmented,
        initial_attribution=initial,
        resolved_attribution=resolved,
        added_to_run_indices=tuple(added_run_indices),
        skip_reason_counts=dict(sorted(skips.items())),
    )


def augment_reference_boundary_requests_file(
    evidence_path: Path,
    output_path: Path,
    *,
    attribution_output_path: Path | None = None,
    config: ReferenceAttributionConfig | None = None,
) -> tuple[ReferenceBoundaryAugmentation, Path, Path | None]:
    """Augment one JSON artifact while keeping the input evidence read-only."""

    source = Path(evidence_path).expanduser()
    destination = Path(output_path).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("augmented evidence output cannot overwrite input evidence")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("reference evidence JSON root must be an object")
    result = augment_reference_boundary_requests(raw, config=config)
    written_evidence = result.write_evidence(destination)
    written_attribution = (
        result.write_attribution(attribution_output_path)
        if attribution_output_path is not None
        else None
    )
    return (result, written_evidence, written_attribution)


__all__ = [
    "ANALYSIS_BASIS",
    "SCHEMA_VERSION",
    "ReferenceBoundaryAugmentation",
    "augment_reference_boundary_requests",
    "augment_reference_boundary_requests_file",
]
