"""Fail-closed human-reference cadence calibration for same-song stage mixes.

The professional edits used by this module are *timing references*, not
templates.  Their visible-cut timestamps are aligned to one canonical source
clock, clustered inside a tightly bounded beat-scale window, and exposed as a
small positive ranking prior.  The prior is read-only, never penalizes a
candidate for choosing a different cut, and cannot turn a hard-safety failure
into a pass.

Serialized profiles contain only paths, hashes, scalar audio-alignment
diagnostics, and cut timestamps.  Raw audio, frames, visual features, and
embeddings are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping, Sequence

from .audio import (
    DEFAULT_ANALYSIS_SAMPLE_RATE,
    OffsetEstimate,
    extract_mono_pcm,
    spectral_flux_alignment_offset,
)
from .timewarp import estimate_local_timewarp


PROFILE_SCHEMA_VERSION = 1
PROFILE_ANALYSIS_BASIS = "same_song_human_stage_mix_master_cadence_profile_v1"
AUDIT_ANALYSIS_BASIS = "same_song_human_stage_mix_cut_quality_calibration_v1"
PRIOR_ANALYSIS_BASIS = "bounded_professional_reference_cadence_prior_v1"
MODULE_VERSION = "1"
MAXIMUM_BEAT_SCALE_TOLERANCE_SECONDS = 0.50


class CadenceAlignmentError(RuntimeError):
    """Raised when a reference cannot be aligned with sufficient confidence."""


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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match schema; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _profile_content_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    provenance = dict(_mapping(payload.get("provenance"), "provenance"))
    provenance["profile_content_sha256"] = None
    payload["provenance"] = provenance
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(_finite(value, "distribution value") for value in values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    clean = tuple(_finite(value, "distribution value") for value in values)
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p10": _quantile(clean, 0.10),
        "p25": _quantile(clean, 0.25),
        "median": median(clean) if clean else None,
        "p75": _quantile(clean, 0.75),
        "p90": _quantile(clean, 0.90),
        "p95": _quantile(clean, 0.95),
        "maximum": max(clean) if clean else None,
    }


@dataclass(frozen=True, slots=True)
class CadenceConfig:
    """Conservative alignment, clustering, and ranking-prior policy."""

    maximum_coarse_offset_seconds: float = 30.0
    minimum_coarse_overlap_seconds: float = 60.0
    minimum_coarse_score: float = 0.75
    minimum_coarse_score_margin: float = 0.08
    minimum_coarse_overlap_ratio: float = 0.75
    require_local_timewarp: bool = True
    cluster_tolerance_seconds: float = 0.22
    minimum_consensus_support: int = 2
    maximum_prior_bonus_points: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_coarse_offset_seconds",
            "minimum_coarse_overlap_seconds",
            "minimum_coarse_score",
            "minimum_coarse_score_margin",
            "minimum_coarse_overlap_ratio",
            "cluster_tolerance_seconds",
            "maximum_prior_bonus_points",
        ):
            _finite(getattr(self, name), name)
        if not isinstance(self.require_local_timewarp, bool):
            raise TypeError("require_local_timewarp must be boolean")
        if self.maximum_coarse_offset_seconds <= 0.0:
            raise ValueError("maximum_coarse_offset_seconds must be positive")
        if self.minimum_coarse_overlap_seconds <= 0.0:
            raise ValueError("minimum_coarse_overlap_seconds must be positive")
        if not -1.0 <= self.minimum_coarse_score <= 1.0:
            raise ValueError("minimum_coarse_score must lie in [-1, 1]")
        if self.minimum_coarse_score_margin < 0.0:
            raise ValueError("minimum_coarse_score_margin cannot be negative")
        if not 0.0 < self.minimum_coarse_overlap_ratio <= 1.0:
            raise ValueError("minimum_coarse_overlap_ratio must lie in (0, 1]")
        if not 0.0 < self.cluster_tolerance_seconds <= MAXIMUM_BEAT_SCALE_TOLERANCE_SECONDS:
            raise ValueError(
                "cluster_tolerance_seconds exceeds the bounded beat-scale limit"
            )
        if (
            isinstance(self.minimum_consensus_support, bool)
            or not isinstance(self.minimum_consensus_support, int)
            or self.minimum_consensus_support < 2
        ):
            raise ValueError("minimum_consensus_support must be an integer >= 2")
        if not 0.0 <= self.maximum_prior_bonus_points <= 5.0:
            raise ValueError("maximum_prior_bonus_points must lie in [0, 5]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_coarse_offset_seconds": self.maximum_coarse_offset_seconds,
            "minimum_coarse_overlap_seconds": self.minimum_coarse_overlap_seconds,
            "minimum_coarse_score": self.minimum_coarse_score,
            "minimum_coarse_score_margin": self.minimum_coarse_score_margin,
            "minimum_coarse_overlap_ratio": self.minimum_coarse_overlap_ratio,
            "require_local_timewarp": self.require_local_timewarp,
            "cluster_tolerance_seconds": self.cluster_tolerance_seconds,
            "maximum_allowed_cluster_tolerance_seconds": (
                MAXIMUM_BEAT_SCALE_TOLERANCE_SECONDS
            ),
            "minimum_consensus_support": self.minimum_consensus_support,
            "maximum_prior_bonus_points": self.maximum_prior_bonus_points,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CadenceConfig":
        _exact_keys(
            value,
            {
                "maximum_coarse_offset_seconds",
                "minimum_coarse_overlap_seconds",
                "minimum_coarse_score",
                "minimum_coarse_score_margin",
                "minimum_coarse_overlap_ratio",
                "require_local_timewarp",
                "cluster_tolerance_seconds",
                "maximum_allowed_cluster_tolerance_seconds",
                "minimum_consensus_support",
                "maximum_prior_bonus_points",
            },
            "configuration",
        )
        maximum = _finite(
            value["maximum_allowed_cluster_tolerance_seconds"],
            "maximum_allowed_cluster_tolerance_seconds",
        )
        if maximum != MAXIMUM_BEAT_SCALE_TOLERANCE_SECONDS:
            raise ValueError("profile beat-scale tolerance limit is unsupported")
        require_local = value["require_local_timewarp"]
        if not isinstance(require_local, bool):
            raise TypeError("require_local_timewarp must be boolean")
        return cls(
            maximum_coarse_offset_seconds=_finite(
                value["maximum_coarse_offset_seconds"],
                "maximum_coarse_offset_seconds",
            ),
            minimum_coarse_overlap_seconds=_finite(
                value["minimum_coarse_overlap_seconds"],
                "minimum_coarse_overlap_seconds",
            ),
            minimum_coarse_score=_finite(
                value["minimum_coarse_score"], "minimum_coarse_score"
            ),
            minimum_coarse_score_margin=_finite(
                value["minimum_coarse_score_margin"],
                "minimum_coarse_score_margin",
            ),
            minimum_coarse_overlap_ratio=_finite(
                value["minimum_coarse_overlap_ratio"],
                "minimum_coarse_overlap_ratio",
            ),
            require_local_timewarp=require_local,
            cluster_tolerance_seconds=_finite(
                value["cluster_tolerance_seconds"], "cluster_tolerance_seconds"
            ),
            minimum_consensus_support=_integer(
                value["minimum_consensus_support"], "minimum_consensus_support"
            ),
            maximum_prior_bonus_points=_finite(
                value["maximum_prior_bonus_points"],
                "maximum_prior_bonus_points",
            ),
        )


@dataclass(frozen=True, slots=True)
class SectionSpan:
    section_id: str
    label: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.section_id.strip() or not self.label.strip():
            raise ValueError("section id and label cannot be blank")
        start = _finite(self.start_seconds, "section start_seconds")
        end = _finite(self.end_seconds, "section end_seconds")
        if start < 0.0 or end <= start:
            raise ValueError("section span must be positive")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, str | float]:
        return {
            "section_id": self.section_id,
            "label": self.label,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SectionSpan":
        _exact_keys(
            value,
            {
                "section_id",
                "label",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
            },
            "section",
        )
        section = cls(
            section_id=str(value["section_id"]),
            label=str(value["label"]),
            start_seconds=_finite(value["start_seconds"], "section start_seconds"),
            end_seconds=_finite(value["end_seconds"], "section end_seconds"),
        )
        if not math.isclose(
            _finite(value["duration_seconds"], "section duration_seconds"),
            section.duration_seconds,
            abs_tol=1e-9,
        ):
            raise ValueError("section duration does not match its endpoints")
        return section


@dataclass(frozen=True, slots=True)
class ReferenceCadenceInput:
    video_id: str
    media_path: Path
    audit_path: Path

    def __post_init__(self) -> None:
        video_id = self.video_id.strip()
        if not video_id or any(character.isspace() for character in video_id):
            raise ValueError("video_id must be non-blank and contain no whitespace")
        media_path = Path(self.media_path).expanduser().resolve()
        audit_path = Path(self.audit_path).expanduser().resolve()
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "media_path", media_path)
        object.__setattr__(self, "audit_path", audit_path)


@dataclass(frozen=True, slots=True)
class _CutPoint:
    reference_id: str
    reference_cut_index: int
    master_seconds: float


@dataclass(frozen=True, slots=True)
class CadenceCluster:
    cluster_index: int
    center_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    support_count: int
    support_fraction: float
    members: tuple[_CutPoint, ...]
    is_consensus: bool

    @property
    def spread_seconds(self) -> float:
        return self.maximum_seconds - self.minimum_seconds


def _cluster_points(
    points: Sequence[_CutPoint],
    *,
    reference_count: int,
    config: CadenceConfig,
) -> tuple[CadenceCluster, ...]:
    """Greedily form deterministic, diameter-bounded, one-cut/reference groups."""

    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    working: list[list[_CutPoint]] = []
    for point in sorted(
        points,
        key=lambda item: (
            item.master_seconds,
            item.reference_id,
            item.reference_cut_index,
        ),
    ):
        eligible: list[tuple[float, int, float, int]] = []
        for index, cluster in enumerate(working):
            if any(item.reference_id == point.reference_id for item in cluster):
                continue
            timestamps = [item.master_seconds for item in cluster]
            new_minimum = min(min(timestamps), point.master_seconds)
            new_maximum = max(max(timestamps), point.master_seconds)
            if new_maximum - new_minimum > config.cluster_tolerance_seconds + 1e-12:
                continue
            eligible.append(
                (
                    abs(point.master_seconds - median(timestamps)),
                    -len(cluster),
                    min(timestamps),
                    index,
                )
            )
        if eligible:
            working[min(eligible)[-1]].append(point)
        else:
            working.append([point])

    ordered = sorted(
        working,
        key=lambda cluster: (
            median(item.master_seconds for item in cluster),
            min(item.master_seconds for item in cluster),
        ),
    )
    result: list[CadenceCluster] = []
    for index, members in enumerate(ordered):
        normalized = tuple(
            sorted(
                members,
                key=lambda item: (item.reference_id, item.reference_cut_index),
            )
        )
        timestamps = tuple(item.master_seconds for item in normalized)
        support = len(normalized)
        result.append(
            CadenceCluster(
                cluster_index=index,
                center_seconds=float(median(timestamps)),
                minimum_seconds=min(timestamps),
                maximum_seconds=max(timestamps),
                support_count=support,
                support_fraction=support / reference_count,
                members=normalized,
                is_consensus=support >= config.minimum_consensus_support,
            )
        )
    return tuple(result)


def _validate_sections(
    sections: Sequence[SectionSpan], master_duration_seconds: float
) -> tuple[SectionSpan, ...]:
    duration = _finite(master_duration_seconds, "master_duration_seconds")
    if duration <= 0.0:
        raise ValueError("master_duration_seconds must be positive")
    normalized = tuple(sections)
    if not normalized:
        return (SectionSpan("whole", "whole_song", 0.0, duration),)
    if not math.isclose(normalized[0].start_seconds, 0.0, abs_tol=1e-6):
        raise ValueError("sections must begin at master time zero")
    if not math.isclose(normalized[-1].end_seconds, duration, abs_tol=1e-6):
        raise ValueError("sections must end at the master duration")
    if len({section.section_id for section in normalized}) != len(normalized):
        raise ValueError("section ids must be unique")
    for left, right in zip(normalized, normalized[1:]):
        if not math.isclose(left.end_seconds, right.start_seconds, abs_tol=1e-6):
            raise ValueError("sections must form a contiguous partition")
    return normalized


@dataclass(frozen=True, slots=True)
class CadencePriorScore:
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.payload))

    @property
    def raw_prior(self) -> float:
        return float(self.payload["raw_prior_0_to_1"])

    @property
    def applied_bonus_points(self) -> float:
        return float(self.payload["applied_bonus_points"])


@dataclass(frozen=True, slots=True)
class MappedCadenceModel:
    """Reusable in-memory model after all reference clocks are canonicalized."""

    reference_cut_times: Mapping[str, tuple[float, ...]]
    master_duration_seconds: float
    sections: tuple[SectionSpan, ...]
    config: CadenceConfig = CadenceConfig()

    def __post_init__(self) -> None:
        duration = _finite(self.master_duration_seconds, "master_duration_seconds")
        if duration <= 0.0:
            raise ValueError("master_duration_seconds must be positive")
        if len(self.reference_cut_times) < self.config.minimum_consensus_support:
            raise ValueError("not enough references for the configured consensus")
        normalized: dict[str, tuple[float, ...]] = {}
        for raw_reference_id, raw_times in self.reference_cut_times.items():
            reference_id = str(raw_reference_id).strip()
            if not reference_id:
                raise ValueError("reference id cannot be blank")
            times = tuple(_finite(value, "mapped cut seconds") for value in raw_times)
            if any(value < 0.0 or value > duration for value in times):
                raise ValueError("mapped cuts must lie inside the master duration")
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("mapped reference cuts must be strictly increasing")
            normalized[reference_id] = times
        sections = _validate_sections(self.sections, duration)
        object.__setattr__(self, "reference_cut_times", normalized)
        object.__setattr__(self, "master_duration_seconds", duration)
        object.__setattr__(self, "sections", sections)

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.reference_cut_times))

    def clusters(
        self, *, excluded_reference_ids: Sequence[str] = ()
    ) -> tuple[CadenceCluster, ...]:
        excluded = frozenset(str(value) for value in excluded_reference_ids)
        unknown = excluded - set(self.reference_cut_times)
        if unknown:
            raise KeyError(f"unknown excluded references: {sorted(unknown)}")
        retained = tuple(
            reference_id
            for reference_id in self.reference_ids
            if reference_id not in excluded
        )
        if len(retained) < self.config.minimum_consensus_support:
            raise ValueError("exclusion leaves too few references for consensus")
        points = tuple(
            _CutPoint(reference_id, index, seconds)
            for reference_id in retained
            for index, seconds in enumerate(self.reference_cut_times[reference_id])
        )
        return _cluster_points(
            points,
            reference_count=len(retained),
            config=self.config,
        )

    def distributions(self) -> dict[str, Any]:
        clusters = self.clusters()
        consensus = tuple(cluster for cluster in clusters if cluster.is_consensus)
        whole = SectionSpan(
            "whole_song",
            "whole_song",
            0.0,
            self.master_duration_seconds,
        )
        return {
            "definition": (
                "visible_cut_rate_is_per_reference_cuts_per_minute; "
                "shot_length_is_within_span_time_between_consecutive_visible_cuts"
            ),
            "whole_song": self._span_distribution(whole, consensus),
            "by_section": [
                self._span_distribution(section, consensus) for section in self.sections
            ],
        }

    def _span_distribution(
        self,
        span: SectionSpan,
        consensus: Sequence[CadenceCluster],
    ) -> dict[str, Any]:
        is_final = math.isclose(
            span.end_seconds, self.master_duration_seconds, abs_tol=1e-9
        )

        def included(value: float) -> bool:
            return span.start_seconds <= value and (
                value < span.end_seconds or (is_final and value <= span.end_seconds)
            )

        per_reference: list[dict[str, Any]] = []
        rates: list[float] = []
        counts: list[float] = []
        pooled_shot_lengths: list[float] = []
        for reference_id in self.reference_ids:
            times = tuple(
                value
                for value in self.reference_cut_times[reference_id]
                if included(value)
            )
            lengths = tuple(right - left for left, right in zip(times, times[1:]))
            rate = len(times) * 60.0 / span.duration_seconds
            rates.append(rate)
            counts.append(float(len(times)))
            pooled_shot_lengths.extend(lengths)
            per_reference.append(
                {
                    "reference_id": reference_id,
                    "visible_cut_count": len(times),
                    "visible_cut_rate_per_minute": rate,
                    "shot_length_seconds": _distribution(lengths),
                }
            )
        consensus_count = sum(1 for cluster in consensus if included(cluster.center_seconds))
        return {
            **span.to_dict(),
            "reference_visible_cut_count": _distribution(counts),
            "visible_cut_rate_per_minute": _distribution(rates),
            "shot_length_seconds": _distribution(pooled_shot_lengths),
            "consensus_event_count": consensus_count,
            "consensus_events_per_minute": (
                consensus_count * 60.0 / span.duration_seconds
            ),
            "per_reference": per_reference,
        }

    def score_candidate_boundaries(
        self,
        boundary_times_seconds: Sequence[float],
        *,
        hard_safety_passed: bool,
        base_score: float | None = None,
        excluded_reference_ids: Sequence[str] = (),
    ) -> CadencePriorScore:
        """Return a capped positive-only prior without changing safety outcome."""

        if not isinstance(hard_safety_passed, bool):
            raise TypeError("hard_safety_passed must be boolean")
        boundaries = tuple(
            _finite(value, "candidate boundary seconds")
            for value in boundary_times_seconds
        )
        if any(value < 0.0 or value > self.master_duration_seconds for value in boundaries):
            raise ValueError("candidate boundary lies outside the master duration")
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("candidate boundary times must be strictly increasing")
        baseline = None if base_score is None else _finite(base_score, "base_score")
        if baseline is not None and not 0.0 <= baseline <= 100.0:
            raise ValueError("base_score must lie in [0, 100]")

        excluded = tuple(sorted(set(str(value) for value in excluded_reference_ids)))
        retained_count = len(self.reference_ids) - len(excluded)
        clusters = tuple(
            cluster
            for cluster in self.clusters(excluded_reference_ids=excluded)
            if cluster.is_consensus
        )
        evidence: list[dict[str, Any]] = []
        rewards: list[float] = []
        tolerance = self.config.cluster_tolerance_seconds
        for boundary in boundaries:
            if clusters:
                nearest = min(
                    clusters,
                    key=lambda cluster: (
                        abs(boundary - cluster.center_seconds),
                        cluster.center_seconds,
                    ),
                )
                distance = abs(boundary - nearest.center_seconds)
            else:
                nearest = None
                distance = math.inf
            if nearest is not None and distance <= tolerance + 1e-12:
                neighborhood = max(0.0, 1.0 - distance / tolerance)
                support_weight = 0.5 + 0.5 * nearest.support_fraction
                reward = min(1.0, neighborhood * support_weight)
                matched = True
                cluster_index: int | None = nearest.cluster_index
                center: float | None = nearest.center_seconds
                support: int | None = nearest.support_count
            else:
                reward = 0.0
                matched = False
                cluster_index = None
                center = None
                support = None
                distance = None
            rewards.append(reward)
            evidence.append(
                {
                    "boundary_seconds": boundary,
                    "matched_consensus": matched,
                    "nearest_consensus_cluster_index": cluster_index,
                    "nearest_consensus_seconds": center,
                    "distance_seconds": distance,
                    "consensus_support_count": support,
                    "positive_reward_0_to_1": reward,
                }
            )
        raw_prior = sum(rewards) / len(rewards) if rewards else 0.0
        computed_bonus = min(
            self.config.maximum_prior_bonus_points,
            raw_prior * self.config.maximum_prior_bonus_points,
        )
        applied_bonus = computed_bonus if hard_safety_passed else 0.0
        adjusted = (
            None
            if baseline is None
            else min(100.0, baseline + applied_bonus)
        )
        match_count = sum(bool(item["matched_consensus"]) for item in evidence)
        return CadencePriorScore(
            {
                "analysis_basis": PRIOR_ANALYSIS_BASIS,
                "hard_safety_passed": hard_safety_passed,
                "hard_safety_outcome_unchanged": True,
                "eligible_for_ranking_bonus": hard_safety_passed,
                "candidate_boundary_count": len(boundaries),
                "training_reference_count": retained_count,
                "excluded_reference_ids": list(excluded),
                "consensus_event_count": len(clusters),
                "cluster_tolerance_seconds": tolerance,
                "matched_boundary_count": match_count,
                "matched_boundary_fraction": (
                    match_count / len(boundaries) if boundaries else 0.0
                ),
                "raw_prior_0_to_1": raw_prior,
                "maximum_bonus_points": self.config.maximum_prior_bonus_points,
                "computed_bonus_points": computed_bonus,
                "applied_bonus_points": applied_bonus,
                "base_score": baseline,
                "adjusted_score": adjusted,
                "boundary_evidence": evidence,
                "policy": (
                    "positive_only_neighborhood_reward; unmatched cuts are not "
                    "penalized; no cut is required; bonus is capped; hard safety "
                    "outcome cannot be overridden"
                ),
            }
        )


def build_mapped_cadence_model(
    reference_cut_times: Mapping[str, Sequence[float]],
    *,
    master_duration_seconds: float,
    sections: Sequence[SectionSpan] = (),
    config: CadenceConfig | None = None,
) -> MappedCadenceModel:
    """Build a reusable model when upstream code already owns clock mapping."""

    return MappedCadenceModel(
        reference_cut_times={
            str(reference_id): tuple(float(value) for value in values)
            for reference_id, values in reference_cut_times.items()
        },
        master_duration_seconds=master_duration_seconds,
        sections=tuple(sections),
        config=config or CadenceConfig(),
    )


def _load_strict_audit(reference: ReferenceCadenceInput) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    try:
        payload = json.loads(reference.audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read audit {reference.audit_path}: {exc}") from exc
    audit = dict(_mapping(payload, "human cut audit"))
    _exact_keys(
        audit,
        {"analysis_basis", "reference", "summary", "cuts"},
        "human cut audit",
    )
    if audit["analysis_basis"] != AUDIT_ANALYSIS_BASIS:
        raise ValueError("human cut audit analysis_basis is unsupported")
    declared = Path(str(audit["reference"])).expanduser()
    try:
        same_media = os.path.samefile(declared, reference.media_path)
    except OSError:
        same_media = declared.resolve() == reference.media_path
    if not same_media:
        raise ValueError("human cut audit reference path does not match media")
    summary = _mapping(audit["summary"], "human cut audit summary")
    cuts = _list(audit["cuts"], "human cut audit cuts")
    if _integer(summary.get("cut_count"), "audit summary cut_count") != len(cuts):
        raise ValueError("audit summary cut_count does not match cuts")
    if not cuts:
        raise ValueError("human cut audit contains no cuts")
    validated: list[dict[str, Any]] = []
    previous = -math.inf
    for index, raw_cut in enumerate(cuts):
        cut = dict(_mapping(raw_cut, f"audit cut {index}"))
        _exact_keys(
            cut,
            {"frame_index", "seconds", "score", "discontinuities", "issue_reasons"},
            f"audit cut {index}",
        )
        frame_index = _integer(cut["frame_index"], f"audit cut {index} frame_index")
        seconds = _finite(cut["seconds"], f"audit cut {index} seconds")
        _finite(cut["score"], f"audit cut {index} score")
        _mapping(cut["discontinuities"], f"audit cut {index} discontinuities")
        _list(cut["issue_reasons"], f"audit cut {index} issue_reasons")
        if frame_index < 0 or seconds < 0.0:
            raise ValueError("audit cut frame/time cannot be negative")
        if seconds <= previous:
            raise ValueError("audit cut timestamps must be strictly increasing")
        previous = seconds
        validated.append({"frame_index": frame_index, "seconds": seconds})
    return audit, tuple(validated)


def discover_reference_inputs(
    reference_directory: str | os.PathLike[str],
) -> tuple[ReferenceCadenceInput, ...]:
    """Discover one provenance-bearing audit for every MP4 in a directory.

    Matching uses the audit's declared ``reference`` path.  This intentionally
    ignores legacy or duplicate JSON files without provenance and handles video
    IDs that themselves contain hyphens.
    """

    root = Path(reference_directory).expanduser().resolve()
    evidence = root / "evidence"
    if not root.is_dir() or not evidence.is_dir():
        raise FileNotFoundError("reference directory and evidence directory are required")
    media_files = tuple(
        sorted(
            path.resolve()
            for path in root.glob("*.mp4")
            if not path.name.startswith("._")
        )
    )
    if not media_files:
        raise ValueError("reference directory contains no MP4 files")
    declared_by_media: dict[Path, list[Path]] = {path: [] for path in media_files}
    for audit_path in sorted(evidence.glob("*-human-cut-quality-audit.json")):
        if audit_path.name.startswith("._"):
            continue
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping) or "reference" not in payload:
            continue
        declared = Path(str(payload["reference"])).expanduser()
        for media_path in media_files:
            try:
                matches = os.path.samefile(declared, media_path)
            except OSError:
                matches = declared.resolve() == media_path
            if matches:
                declared_by_media[media_path].append(audit_path.resolve())
    result: list[ReferenceCadenceInput] = []
    for media_path, matches in declared_by_media.items():
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one provenance-bearing audit for {media_path}; "
                f"found {len(matches)}"
            )
        youtube_id = re.match(r"^([A-Za-z0-9_-]{11})(?:-|$)", media_path.name)
        if youtube_id is not None:
            video_id = youtube_id.group(1)
        else:
            prefix, separator, _rest = media_path.name.partition("-IVE")
            video_id = prefix if separator else media_path.stem
        result.append(ReferenceCadenceInput(video_id, media_path, matches[0]))
    if len({item.video_id for item in result}) != len(result):
        raise ValueError("discovered reference video ids are not unique")
    return tuple(sorted(result, key=lambda item: item.video_id))


def _assert_confident_alignment(
    reference_id: str,
    estimate: OffsetEstimate,
    config: CadenceConfig,
) -> None:
    failures: list[str] = []
    if estimate.score < config.minimum_coarse_score:
        failures.append("low_score")
    if estimate.score_margin < config.minimum_coarse_score_margin:
        failures.append("ambiguous_peak")
    if estimate.overlap_ratio < config.minimum_coarse_overlap_ratio:
        failures.append("insufficient_overlap_ratio")
    if estimate.overlap_seconds < config.minimum_coarse_overlap_seconds:
        failures.append("insufficient_overlap_seconds")
    if failures:
        raise CadenceAlignmentError(
            f"reference {reference_id} failed coarse alignment: {', '.join(failures)}"
        )


def _cluster_payload(
    clusters: Sequence[CadenceCluster],
    global_indices: Mapping[tuple[str, int], int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cluster in clusters:
        member_indices = [
            global_indices[(member.reference_id, member.reference_cut_index)]
            for member in cluster.members
        ]
        result.append(
            {
                "cluster_index": cluster.cluster_index,
                "center_seconds": cluster.center_seconds,
                "minimum_seconds": cluster.minimum_seconds,
                "maximum_seconds": cluster.maximum_seconds,
                "spread_seconds": cluster.spread_seconds,
                "support_count": cluster.support_count,
                "support_fraction": cluster.support_fraction,
                "reference_ids": [member.reference_id for member in cluster.members],
                "mapped_cut_indices": member_indices,
                "is_consensus": cluster.is_consensus,
            }
        )
    return result


def _consensus_summary(
    clusters: Sequence[CadenceCluster],
    *,
    mapped_cut_count: int,
    in_domain_cut_count: int,
) -> dict[str, Any]:
    consensus = tuple(cluster for cluster in clusters if cluster.is_consensus)
    histogram: dict[str, int] = {}
    for cluster in clusters:
        key = str(cluster.support_count)
        histogram[key] = histogram.get(key, 0) + 1
    consensus_cut_members = sum(cluster.support_count for cluster in consensus)
    return {
        "mapped_cut_count": mapped_cut_count,
        "in_master_domain_cut_count": in_domain_cut_count,
        "out_of_master_domain_cut_count": mapped_cut_count - in_domain_cut_count,
        "cluster_count": len(clusters),
        "consensus_cluster_count": len(consensus),
        "singleton_cluster_count": sum(cluster.support_count == 1 for cluster in clusters),
        "maximum_support_count": max(
            (cluster.support_count for cluster in clusters), default=0
        ),
        "cluster_support_histogram": histogram,
        "consensus_cut_member_count": consensus_cut_members,
        "consensus_cut_member_fraction": (
            consensus_cut_members / in_domain_cut_count
            if in_domain_cut_count
            else 0.0
        ),
    }


def build_professional_reference_cadence_profile(
    master_path: str | os.PathLike[str],
    references: Sequence[ReferenceCadenceInput],
    *,
    artist: str,
    song: str,
    sections: Sequence[SectionSpan] = (),
    section_analysis_basis: str = "caller_supplied_contiguous_master_clock_spans_v1",
    section_semantics: str = (
        "calibration_buckets_not_ground_truth_lyrical_song_structure"
    ),
    config: CadenceConfig | None = None,
    sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE,
    ffmpeg_bin: str = "ffmpeg",
) -> "ProfessionalReferenceCadenceProfile":
    """Decode, align, map, cluster, and validate a cadence profile."""

    settings = config or CadenceConfig()
    master = Path(master_path).expanduser().resolve()
    if not master.is_file():
        raise FileNotFoundError(master)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    normalized_references = tuple(sorted(references, key=lambda item: item.video_id))
    if len(normalized_references) < settings.minimum_consensus_support:
        raise ValueError("not enough human references for consensus")
    if len({item.video_id for item in normalized_references}) != len(normalized_references):
        raise ValueError("reference video ids must be unique")
    if not artist.strip() or not song.strip():
        raise ValueError("artist and song cannot be blank")
    if not section_analysis_basis.strip() or not section_semantics.strip():
        raise ValueError("section provenance cannot be blank")

    master_pcm = extract_mono_pcm(
        master,
        sample_rate=sample_rate,
        ffmpeg_bin=ffmpeg_bin,
    )
    master_duration = master_pcm.size / sample_rate
    normalized_sections = _validate_sections(sections, master_duration)
    mapped_cuts: list[dict[str, Any]] = []
    reference_provenance: list[dict[str, Any]] = []

    for reference in normalized_references:
        audit, cuts = _load_strict_audit(reference)
        source_pcm = extract_mono_pcm(
            reference.media_path,
            sample_rate=sample_rate,
            ffmpeg_bin=ffmpeg_bin,
        )
        try:
            coarse = spectral_flux_alignment_offset(
                master_pcm,
                source_pcm,
                sample_rate=sample_rate,
                max_offset_seconds=settings.maximum_coarse_offset_seconds,
                min_overlap_seconds=settings.minimum_coarse_overlap_seconds,
            )
        except (TypeError, ValueError) as exc:
            raise CadenceAlignmentError(
                f"reference {reference.video_id} coarse alignment failed: {exc}"
            ) from exc
        _assert_confident_alignment(reference.video_id, coarse, settings)
        try:
            warp = estimate_local_timewarp(
                master_pcm,
                source_pcm,
                sample_rate=sample_rate,
                global_offset_seconds=coarse.offset_seconds,
            )
        except (TypeError, ValueError) as exc:
            raise CadenceAlignmentError(
                f"reference {reference.video_id} local time-warp failed: {exc}"
            ) from exc
        if settings.require_local_timewarp and warp.scalar_fallback:
            raise CadenceAlignmentError(
                f"reference {reference.video_id} used forbidden scalar fallback: "
                f"{warp.fallback_reason}"
            )

        for cut_index, cut in enumerate(cuts):
            reference_seconds = float(cut["seconds"])
            master_seconds = warp.source_to_master(reference_seconds)
            in_domain = -1e-9 <= master_seconds <= master_duration + 1e-9
            mapped_cuts.append(
                {
                    "reference_id": reference.video_id,
                    "audit_cut_index": cut_index,
                    "audit_frame_index": int(cut["frame_index"]),
                    "reference_seconds": reference_seconds,
                    "master_seconds": master_seconds,
                    "in_master_domain": in_domain,
                }
            )
        reference_provenance.append(
            {
                "video_id": reference.video_id,
                "media_path": os.fspath(reference.media_path),
                "media_sha256": _sha256_file(reference.media_path),
                "decoded_audio_duration_seconds": source_pcm.size / sample_rate,
                "audit_path": os.fspath(reference.audit_path),
                "audit_sha256": _sha256_file(reference.audit_path),
                "audit_analysis_basis": audit["analysis_basis"],
                "audit_cut_count": len(cuts),
                "coarse_alignment": coarse.to_dict(),
                "local_timewarp": warp.to_dict(),
            }
        )

    reference_times: dict[str, tuple[float, ...]] = {}
    for reference in normalized_references:
        reference_times[reference.video_id] = tuple(
            float(item["master_seconds"])
            for item in mapped_cuts
            if item["reference_id"] == reference.video_id
            and bool(item["in_master_domain"])
        )
    model = build_mapped_cadence_model(
        reference_times,
        master_duration_seconds=master_duration,
        sections=normalized_sections,
        config=settings,
    )
    clusters = model.clusters()
    # Cluster member indices are contiguous within the *in-domain* subset.
    # Keep that local index separate from the original audit index so a
    # reference with an out-of-domain intro cut still serializes correctly.
    global_indices: dict[tuple[str, int], int] = {}
    in_domain_local_indices = {
        reference.video_id: 0 for reference in normalized_references
    }
    for global_index, item in enumerate(mapped_cuts):
        if not bool(item["in_master_domain"]):
            continue
        reference_id = str(item["reference_id"])
        local_index = in_domain_local_indices[reference_id]
        global_indices[(reference_id, local_index)] = global_index
        in_domain_local_indices[reference_id] = local_index + 1
    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "analysis_basis": PROFILE_ANALYSIS_BASIS,
        "song": {"artist": artist.strip(), "title": song.strip()},
        "configuration": settings.to_dict(),
        "provenance": {
            "content_hash_algorithm": (
                "sha256_of_canonical_json_with_profile_content_sha256_null"
            ),
            "profile_content_sha256": None,
            "generator": {
                "module": "crosscut.professional_reference_cadence",
                "module_version": MODULE_VERSION,
            },
            "master": {
                "media_path": os.fspath(master),
                "media_sha256": _sha256_file(master),
                "decoded_audio_duration_seconds": master_duration,
                "analysis_sample_rate": sample_rate,
            },
            "references": reference_provenance,
        },
        "mapped_cuts": mapped_cuts,
        "section_provenance": {
            "analysis_basis": section_analysis_basis.strip(),
            "semantics": section_semantics.strip(),
        },
        "sections": [section.to_dict() for section in normalized_sections],
        "clusters": _cluster_payload(clusters, global_indices),
        "consensus_summary": _consensus_summary(
            clusters,
            mapped_cut_count=len(mapped_cuts),
            in_domain_cut_count=sum(bool(item["in_master_domain"]) for item in mapped_cuts),
        ),
        "distributions": model.distributions(),
        "safety_policy": {
            "prior_is_read_only": True,
            "positive_reward_only": True,
            "hard_safety_can_be_overridden": False,
            "maximum_prior_bonus_points": settings.maximum_prior_bonus_points,
            "raw_audio_stored": False,
            "raw_frames_stored": False,
            "visual_features_or_embeddings_stored": False,
            "technical_integrity_rules": (
                "decode, black-frame, duplicate, freeze, timing, and other hard "
                "safety rules remain independently mandatory"
            ),
        },
    }
    profile["provenance"]["profile_content_sha256"] = _profile_content_sha256(profile)
    return ProfessionalReferenceCadenceProfile.from_mapping(profile)


def _inverse_timewarp_payload(local_timewarp: Mapping[str, Any], source_seconds: float) -> float:
    scalar = local_timewarp.get("scalar_fallback")
    if not isinstance(scalar, bool):
        raise ValueError("local_timewarp scalar_fallback must be boolean")
    offset = _finite(
        local_timewarp.get("global_offset_seconds"),
        "local_timewarp global_offset_seconds",
    )
    if scalar:
        return source_seconds - offset
    control_points = _list(
        local_timewarp.get("control_points"), "local_timewarp control_points"
    )
    if len(control_points) < 2:
        raise ValueError("local_timewarp requires at least two control points")
    normalized: list[tuple[float, float]] = []
    for raw in control_points:
        point = _mapping(raw, "local_timewarp control point")
        normalized.append(
            (
                _finite(point.get("master_seconds"), "control master_seconds"),
                _finite(point.get("source_seconds"), "control source_seconds"),
            )
        )
    if any(
        right[0] <= left[0] or right[1] <= left[1]
        for left, right in zip(normalized, normalized[1:])
    ):
        raise ValueError("local_timewarp control points are not monotonic")
    position = 0
    while position < len(normalized) and normalized[position][1] <= source_seconds:
        position += 1
    if position == 0:
        left, right = normalized[:2]
    elif position >= len(normalized):
        left, right = normalized[-2:]
    else:
        left, right = normalized[position - 1 : position + 1]
    fraction = (source_seconds - left[1]) / (right[1] - left[1])
    return left[0] + fraction * (right[0] - left[0])


def _assert_equivalent(expected: object, actual: object, name: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"{name} keys do not match recomputed evidence")
        for key in expected:
            _assert_equivalent(expected[key], actual[key], f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{name} does not match recomputed evidence")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(left, right, f"{name}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{name} numeric type mismatch")
        left = _finite(expected, name)
        right = _finite(actual, name)
        if abs(left - right) > 1e-8:
            raise ValueError(f"{name} does not match recomputed evidence")
        return
    if expected != actual:
        raise ValueError(f"{name} does not match recomputed evidence")


@dataclass(frozen=True, slots=True)
class ProfessionalReferenceCadenceProfile:
    """Strictly validated serialized cadence evidence."""

    _payload: Mapping[str, Any]
    source_path: Path | None = None
    file_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ProfessionalReferenceCadenceProfile":
        payload = deepcopy(dict(value))
        cls._validate(payload, verify_input_hashes=False)
        return cls(payload)

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        verify_input_hashes: bool = True,
    ) -> "ProfessionalReferenceCadenceProfile":
        source = Path(path).expanduser().resolve()
        try:
            raw = source.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read cadence profile {source}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("cadence profile root must be an object")
        normalized = deepcopy(dict(payload))
        cls._validate(normalized, verify_input_hashes=verify_input_hashes)
        return cls(
            normalized,
            source_path=source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def _validate(
        cls, payload: dict[str, Any], *, verify_input_hashes: bool
    ) -> None:
        _exact_keys(
            payload,
            {
                "schema_version",
                "analysis_basis",
                "song",
                "configuration",
                "provenance",
                "mapped_cuts",
                "section_provenance",
                "sections",
                "clusters",
                "consensus_summary",
                "distributions",
                "safety_policy",
            },
            "cadence profile",
        )
        if _integer(payload["schema_version"], "schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported cadence profile schema_version")
        if payload["analysis_basis"] != PROFILE_ANALYSIS_BASIS:
            raise ValueError("unsupported cadence profile analysis_basis")
        song = _mapping(payload["song"], "song")
        _exact_keys(song, {"artist", "title"}, "song")
        if not str(song["artist"]).strip() or not str(song["title"]).strip():
            raise ValueError("song artist/title cannot be blank")
        config = CadenceConfig.from_mapping(
            _mapping(payload["configuration"], "configuration")
        )
        section_provenance = _mapping(
            payload["section_provenance"], "section_provenance"
        )
        _exact_keys(
            section_provenance,
            {"analysis_basis", "semantics"},
            "section_provenance",
        )
        if not str(section_provenance["analysis_basis"]).strip() or not str(
            section_provenance["semantics"]
        ).strip():
            raise ValueError("section provenance cannot be blank")
        provenance = _mapping(payload["provenance"], "provenance")
        _exact_keys(
            provenance,
            {
                "content_hash_algorithm",
                "profile_content_sha256",
                "generator",
                "master",
                "references",
            },
            "provenance",
        )
        if provenance["content_hash_algorithm"] != (
            "sha256_of_canonical_json_with_profile_content_sha256_null"
        ):
            raise ValueError("unsupported cadence profile content hash algorithm")
        content_hash = str(provenance["profile_content_sha256"])
        if len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
            raise ValueError("profile_content_sha256 is invalid")
        if content_hash != _profile_content_sha256(payload):
            raise ValueError("profile content SHA-256 mismatch")
        generator = _mapping(provenance["generator"], "generator")
        _exact_keys(generator, {"module", "module_version"}, "generator")
        if generator["module"] != "crosscut.professional_reference_cadence":
            raise ValueError("cadence profile generator is unsupported")
        master = _mapping(provenance["master"], "master provenance")
        _exact_keys(
            master,
            {
                "media_path",
                "media_sha256",
                "decoded_audio_duration_seconds",
                "analysis_sample_rate",
            },
            "master provenance",
        )
        duration = _finite(
            master["decoded_audio_duration_seconds"], "master duration"
        )
        if duration <= 0.0:
            raise ValueError("master duration must be positive")
        _integer(master["analysis_sample_rate"], "analysis_sample_rate")

        reference_payloads = _list(provenance["references"], "references")
        if len(reference_payloads) < config.minimum_consensus_support:
            raise ValueError("profile contains too few references")
        reference_ids: list[str] = []
        local_warps: dict[str, Mapping[str, Any]] = {}
        for index, raw_reference in enumerate(reference_payloads):
            reference = _mapping(raw_reference, f"reference {index}")
            _exact_keys(
                reference,
                {
                    "video_id",
                    "media_path",
                    "media_sha256",
                    "decoded_audio_duration_seconds",
                    "audit_path",
                    "audit_sha256",
                    "audit_analysis_basis",
                    "audit_cut_count",
                    "coarse_alignment",
                    "local_timewarp",
                },
                f"reference {index}",
            )
            reference_id = str(reference["video_id"])
            if not reference_id:
                raise ValueError("reference video_id cannot be blank")
            reference_ids.append(reference_id)
            if reference["audit_analysis_basis"] != AUDIT_ANALYSIS_BASIS:
                raise ValueError("reference audit basis is unsupported")
            _integer(reference["audit_cut_count"], "reference audit_cut_count")
            coarse = _mapping(reference["coarse_alignment"], "coarse_alignment")
            estimate = OffsetEstimate(
                offset_samples=_integer(coarse.get("offset_samples"), "offset_samples"),
                offset_seconds=_finite(coarse.get("offset_seconds"), "offset_seconds"),
                score=_finite(coarse.get("score"), "coarse score"),
                overlap_samples=_integer(coarse.get("overlap_samples"), "overlap_samples"),
                overlap_seconds=_finite(coarse.get("overlap_seconds"), "overlap_seconds"),
                sample_rate=_integer(coarse.get("sample_rate"), "coarse sample_rate"),
                second_best_score=(
                    None
                    if coarse.get("second_best_score") is None
                    else _finite(coarse.get("second_best_score"), "second_best_score")
                ),
                score_margin=_finite(coarse.get("score_margin"), "score_margin"),
                overlap_ratio=_finite(coarse.get("overlap_ratio"), "overlap_ratio"),
                backend=str(coarse.get("backend")),
            )
            _assert_confident_alignment(reference_id, estimate, config)
            local_warp = _mapping(reference["local_timewarp"], "local_timewarp")
            if config.require_local_timewarp and local_warp.get("scalar_fallback") is not False:
                raise ValueError("profile contains a scalar-fallback time warp")
            local_warps[reference_id] = local_warp
            if verify_input_hashes:
                for path_key, hash_key in (
                    ("media_path", "media_sha256"),
                    ("audit_path", "audit_sha256"),
                ):
                    input_path = Path(str(reference[path_key])).expanduser().resolve()
                    if not input_path.is_file():
                        raise FileNotFoundError(input_path)
                    if _sha256_file(input_path) != reference[hash_key]:
                        raise ValueError(f"reference {reference_id} {hash_key} mismatch")
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("reference ids are not unique")
        if verify_input_hashes:
            master_path = Path(str(master["media_path"])).expanduser().resolve()
            if not master_path.is_file():
                raise FileNotFoundError(master_path)
            if _sha256_file(master_path) != master["media_sha256"]:
                raise ValueError("master media_sha256 mismatch")

        raw_sections = _list(payload["sections"], "sections")
        sections = tuple(
            SectionSpan.from_mapping(_mapping(item, "section"))
            for item in raw_sections
        )
        _validate_sections(sections, duration)
        raw_cuts = _list(payload["mapped_cuts"], "mapped_cuts")
        times: dict[str, list[float]] = {reference_id: [] for reference_id in reference_ids}
        global_indices: dict[tuple[str, int], int] = {}
        in_domain_local_indices = {reference_id: 0 for reference_id in reference_ids}
        per_reference_cut_counts = {reference_id: 0 for reference_id in reference_ids}
        for index, raw_cut in enumerate(raw_cuts):
            cut = _mapping(raw_cut, f"mapped cut {index}")
            _exact_keys(
                cut,
                {
                    "reference_id",
                    "audit_cut_index",
                    "audit_frame_index",
                    "reference_seconds",
                    "master_seconds",
                    "in_master_domain",
                },
                f"mapped cut {index}",
            )
            reference_id = str(cut["reference_id"])
            if reference_id not in times:
                raise ValueError("mapped cut references an unknown video")
            cut_index = _integer(cut["audit_cut_index"], "audit_cut_index")
            if cut_index != per_reference_cut_counts[reference_id]:
                raise ValueError("mapped audit cut indices must be contiguous")
            per_reference_cut_counts[reference_id] += 1
            _integer(cut["audit_frame_index"], "audit_frame_index")
            source_seconds = _finite(cut["reference_seconds"], "reference_seconds")
            master_seconds = _finite(cut["master_seconds"], "master_seconds")
            expected_master = _inverse_timewarp_payload(
                local_warps[reference_id], source_seconds
            )
            if abs(expected_master - master_seconds) > 1e-6:
                raise ValueError("mapped cut does not match stored time-warp")
            in_domain = cut["in_master_domain"]
            if not isinstance(in_domain, bool):
                raise TypeError("in_master_domain must be boolean")
            expected_domain = -1e-9 <= master_seconds <= duration + 1e-9
            if in_domain != expected_domain:
                raise ValueError("mapped cut master-domain flag is incorrect")
            if in_domain:
                times[reference_id].append(master_seconds)
                local_index = in_domain_local_indices[reference_id]
                global_indices[(reference_id, local_index)] = index
                in_domain_local_indices[reference_id] = local_index + 1
        for reference, count in zip(reference_payloads, per_reference_cut_counts.values(), strict=True):
            if count != _integer(reference["audit_cut_count"], "audit_cut_count"):
                raise ValueError("reference audit cut count does not match mapped cuts")

        model = build_mapped_cadence_model(
            {key: tuple(value) for key, value in times.items()},
            master_duration_seconds=duration,
            sections=sections,
            config=config,
        )
        clusters = model.clusters()
        expected_clusters = _cluster_payload(clusters, global_indices)
        _assert_equivalent(payload["clusters"], expected_clusters, "clusters")
        expected_summary = _consensus_summary(
            clusters,
            mapped_cut_count=len(raw_cuts),
            in_domain_cut_count=sum(len(value) for value in times.values()),
        )
        _assert_equivalent(
            payload["consensus_summary"], expected_summary, "consensus_summary"
        )
        _assert_equivalent(
            payload["distributions"], model.distributions(), "distributions"
        )
        safety = _mapping(payload["safety_policy"], "safety_policy")
        _exact_keys(
            safety,
            {
                "prior_is_read_only",
                "positive_reward_only",
                "hard_safety_can_be_overridden",
                "maximum_prior_bonus_points",
                "raw_audio_stored",
                "raw_frames_stored",
                "visual_features_or_embeddings_stored",
                "technical_integrity_rules",
            },
            "safety_policy",
        )
        if any(
            safety[key] is not False
            for key in (
                "hard_safety_can_be_overridden",
                "raw_audio_stored",
                "raw_frames_stored",
                "visual_features_or_embeddings_stored",
            )
        ) or safety["prior_is_read_only"] is not True or safety["positive_reward_only"] is not True:
            raise ValueError("cadence profile safety policy is not fail-closed")

    @property
    def reference_count(self) -> int:
        return len(self._payload["provenance"]["references"])

    @property
    def artist(self) -> str:
        return str(self._payload["song"]["artist"])

    @property
    def song(self) -> str:
        return str(self._payload["song"]["title"])

    @property
    def profile_content_sha256(self) -> str:
        return str(self._payload["provenance"]["profile_content_sha256"])

    @property
    def master_media_path(self) -> Path:
        return Path(
            str(self._payload["provenance"]["master"]["media_path"])
        ).expanduser().resolve()

    @property
    def master_media_sha256(self) -> str:
        return str(self._payload["provenance"]["master"]["media_sha256"])

    def reference_provenance(self) -> tuple[dict[str, str], ...]:
        """Return the strict identity fields shared with the quality profile."""

        return tuple(
            {
                "video_id": str(item["video_id"]),
                "media_path": str(
                    Path(str(item["media_path"])).expanduser().resolve()
                ),
                "media_sha256": str(item["media_sha256"]),
                "audit_path": str(
                    Path(str(item["audit_path"])).expanduser().resolve()
                ),
                "audit_sha256": str(item["audit_sha256"]),
            }
            for item in self._payload["provenance"]["references"]
        )

    def provenance_dict(self) -> dict[str, Any]:
        """Serialize hashes and identities, never the large cadence evidence."""

        return {
            "profile_path": (
                str(self.source_path) if self.source_path is not None else None
            ),
            "profile_file_sha256": self.file_sha256,
            "profile_content_sha256": self.profile_content_sha256,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "analysis_basis": PROFILE_ANALYSIS_BASIS,
            "artist": self.artist,
            "song": self.song,
            "reference_count": self.reference_count,
            "master_media_path": str(self.master_media_path),
            "master_media_sha256": self.master_media_sha256,
            "references": [dict(item) for item in self.reference_provenance()],
        }

    @property
    def model(self) -> MappedCadenceModel:
        duration = float(
            self._payload["provenance"]["master"]["decoded_audio_duration_seconds"]
        )
        times: dict[str, list[float]] = {
            str(item["video_id"]): []
            for item in self._payload["provenance"]["references"]
        }
        for item in self._payload["mapped_cuts"]:
            if item["in_master_domain"]:
                times[str(item["reference_id"])].append(float(item["master_seconds"]))
        return build_mapped_cadence_model(
            times,
            master_duration_seconds=duration,
            sections=tuple(
                SectionSpan.from_mapping(item) for item in self._payload["sections"]
            ),
            config=CadenceConfig.from_mapping(self._payload["configuration"]),
        )

    def score_candidate_boundaries(
        self,
        boundary_times_seconds: Sequence[float],
        *,
        hard_safety_passed: bool,
        base_score: float | None = None,
        excluded_reference_ids: Sequence[str] = (),
    ) -> CadencePriorScore:
        return self.model.score_candidate_boundaries(
            boundary_times_seconds,
            hard_safety_passed=hard_safety_passed,
            base_score=base_score,
            excluded_reference_ids=excluded_reference_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self._payload))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self._payload,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return destination


@dataclass(frozen=True, slots=True)
class CandidateBoundarySet:
    candidate_index: int
    proxy_fps: float
    canonical_master_start_seconds: float
    canonical_master_end_seconds: float
    boundary_times_seconds: tuple[float, ...]
    selection_report_path: Path
    selection_report_sha256: str
    clock_basis: str = "proxy_output_frame_plus_complete_candidate_master_start"


def candidate_source_switch_boundaries_from_selection_report(
    path: str | os.PathLike[str],
    *,
    candidate_indices: Sequence[int] | None = None,
) -> tuple[CandidateBoundarySet, ...]:
    """Read source-switch boundary clocks from an existing selection report."""

    report_path = Path(path).expanduser().resolve()
    raw = report_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selection report: {exc}") from exc
    report = _mapping(payload, "selection report")
    candidates = _list(report.get("candidates"), "selection report candidates")
    complete_candidates = _list(
        report.get("complete_candidates"),
        "selection report complete_candidates",
    )
    master_spans: dict[int, tuple[float, float]] = {}
    for raw_complete in complete_candidates:
        complete = _mapping(raw_complete, "complete candidate")
        complete_index = _integer(
            complete.get("candidate_index"), "complete candidate_index"
        )
        segments = _list(
            complete.get("final_render_segments"), "final_render_segments"
        )
        if not segments:
            raise ValueError("complete candidate contains no final render segments")
        first = _mapping(segments[0], "first final render segment")
        last = _mapping(segments[-1], "last final render segment")
        start = _finite(
            first.get("canonical_master_start"), "canonical_master_start"
        )
        end = _finite(last.get("canonical_master_end"), "canonical_master_end")
        if start < 0.0 or end <= start:
            raise ValueError("complete candidate canonical master span is invalid")
        if complete_index in master_spans:
            raise ValueError("duplicate complete candidate_index")
        master_spans[complete_index] = (start, end)
    requested = None if candidate_indices is None else set(candidate_indices)
    result: list[CandidateBoundarySet] = []
    seen: set[int] = set()
    report_hash = hashlib.sha256(raw).hexdigest()
    for raw_candidate in candidates:
        candidate = _mapping(raw_candidate, "selection candidate")
        index = _integer(candidate.get("candidate_index"), "candidate_index")
        if requested is not None and index not in requested:
            continue
        review = _mapping(candidate.get("review"), "candidate review")
        proxy = _mapping(review.get("proxy"), "candidate proxy")
        fps = _finite(proxy.get("fps"), "candidate proxy fps")
        if fps <= 0.0:
            raise ValueError("candidate proxy fps must be positive")
        if index not in master_spans:
            raise ValueError(
                f"candidate {index} lacks a canonical master clock span"
            )
        master_start, master_end = master_spans[index]
        proxy_duration = proxy.get("duration_seconds")
        if proxy_duration is not None:
            decoded_duration = _finite(proxy_duration, "candidate proxy duration")
            if abs(decoded_duration - (master_end - master_start)) > 1.5 / fps:
                raise ValueError(
                    "candidate proxy duration does not match canonical master span"
                )
        boundaries: list[float] = []
        for raw_boundary in _list(review.get("boundaries"), "review boundaries"):
            boundary = _mapping(raw_boundary, "review boundary")
            outgoing = _integer(
                boundary.get("outgoing_source_index"), "outgoing_source_index"
            )
            incoming = _integer(
                boundary.get("incoming_source_index"), "incoming_source_index"
            )
            if outgoing == incoming:
                continue
            frame_index = _integer(
                boundary.get("cut_frame_index"), "cut_frame_index"
            )
            master_seconds = master_start + frame_index / fps
            if master_seconds > master_end + 1.5 / fps:
                raise ValueError("candidate boundary exceeds canonical master span")
            boundaries.append(master_seconds)
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("selection source-switch boundaries are not increasing")
        result.append(
            CandidateBoundarySet(
                candidate_index=index,
                proxy_fps=fps,
                canonical_master_start_seconds=master_start,
                canonical_master_end_seconds=master_end,
                boundary_times_seconds=tuple(boundaries),
                selection_report_path=report_path,
                selection_report_sha256=report_hash,
            )
        )
        seen.add(index)
    if requested is not None and seen != requested:
        raise KeyError(f"selection report is missing candidates {sorted(requested - seen)}")
    return tuple(sorted(result, key=lambda item: item.candidate_index))


__all__ = [
    "AUDIT_ANALYSIS_BASIS",
    "CadenceAlignmentError",
    "CadenceConfig",
    "CadencePriorScore",
    "CandidateBoundarySet",
    "MappedCadenceModel",
    "PRIOR_ANALYSIS_BASIS",
    "PROFILE_ANALYSIS_BASIS",
    "PROFILE_SCHEMA_VERSION",
    "ProfessionalReferenceCadenceProfile",
    "ReferenceCadenceInput",
    "SectionSpan",
    "build_mapped_cadence_model",
    "build_professional_reference_cadence_profile",
    "candidate_source_switch_boundaries_from_selection_report",
    "discover_reference_inputs",
]
