"""Empirical professional-reference gate for visible stage-mix cuts.

Absolute pixel discontinuity warnings remain useful diagnostics, but a stage
switch naturally changes lighting, background, wardrobe colour, and camera
geometry.  This module therefore compares a candidate's visible-cut
distribution with five same-song human edits instead of raising those absolute
thresholds.  Decode and technical-integrity failures are a separate,
zero-tolerance axis and can never be excused by the human distribution.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA_VERSION = 1
PROFILE_ANALYSIS_BASIS = (
    "ive_same_song_human_stage_mix_empirical_visible_cut_profile_v1"
)
GATE_ANALYSIS_BASIS = "professional_reference_visible_cut_relative_gate_v1"
EXPECTED_REFERENCE_COUNT = 5
MINIMUM_CANDIDATE_VISIBLE_CUT_COUNT = 10
MINIMUM_COMPONENT_TAIL_CHECK_PASS_COUNT = 2

DISCONTINUITY_COMPONENTS = (
    "sharpness_discontinuity",
    "exposure_discontinuity",
    "color_shock",
    "structure_shock",
    "face_scale_discontinuity",
    "face_position_discontinuity",
    "silhouette_scale_discontinuity",
    "silhouette_position_discontinuity",
    "motion_direction_discontinuity",
)

# With only four training references in a leave-one-reference-out fold, the
# observed min/max is a noisy prediction interval.  One inter-reference range
# beyond that interval is used for scalar/quantile statistics.  A proportion
# has additional binomial sampling variance, so its outer fence uses two
# ranges.  These are fixed statistical envelope rules, not tuned pixel
# thresholds, and the tests require all five human holdouts to pass.
QUANTILE_PREDICTION_RANGE_MULTIPLIER = 1.0
PROPORTION_PREDICTION_RANGE_MULTIPLIER = 2.0

_TECHNICAL_REASON_MARKERS = (
    "black_frame",
    "duplicate_or_freeze",
    "decode",
    "drop_or_reset",
    "source_clock_discontinuity",
    "insufficient_post_transition_review_handle",
    "boundary_frame_quality",
)


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


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(_finite(value, "distribution value") for value in values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    alpha = position - low
    return clean[low] * (1.0 - alpha) + clean[high] * alpha


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equivalent(
    expected: object,
    actual: object,
    *,
    name: str,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"{name} keys do not match recomputed evidence")
        for key in expected:
            _assert_equivalent(
                expected[key],
                actual[key],
                name=f"{name}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{name} array does not match recomputed evidence")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(left, right, name=f"{name}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{name} is not numeric in recomputed evidence")
        left = _finite(expected, name)
        right = _finite(actual, name)
        if abs(left - right) > 1e-9:
            raise ValueError(f"{name} does not match recomputed evidence")
        return
    if expected != actual:
        raise ValueError(f"{name} does not match recomputed evidence")


@dataclass(frozen=True, slots=True)
class ProfessionalReferenceCut:
    score: float
    discontinuities: Mapping[str, float | None]
    absolute_issue_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        score = _finite(self.score, "professional-reference cut score")
        if not 0.0 <= score <= 100.0:
            raise ValueError("professional-reference cut score must be in [0, 100]")
        if not isinstance(self.discontinuities, Mapping):
            raise TypeError("cut discontinuities must be an object")
        normalized: dict[str, float | None] = {}
        for component in DISCONTINUITY_COMPONENTS:
            if component not in self.discontinuities:
                raise ValueError(f"cut is missing {component}")
            raw = self.discontinuities[component]
            if raw is None:
                normalized[component] = None
                continue
            value = _finite(raw, component)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{component} must be normalized")
            normalized[component] = value
        reasons = tuple(str(reason).strip() for reason in self.absolute_issue_reasons)
        if any(not reason for reason in reasons):
            raise ValueError("absolute issue reasons cannot contain empty values")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "discontinuities", normalized)
        object.__setattr__(self, "absolute_issue_reasons", reasons)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProfessionalReferenceCut":
        discontinuities = _mapping(
            value.get("discontinuities"),
            "cut discontinuities",
        )
        raw_reasons = value.get(
            "diagnostic_editorial_issue_reasons",
            value.get("issue_reasons", []),
        )
        reasons = _array(raw_reasons, "cut issue reasons")
        return cls(
            score=value.get(
                "diagnostic_editorial_weighted_score",
                value.get("weighted_score", value.get("score")),
            ),
            discontinuities=discontinuities,
            absolute_issue_reasons=tuple(str(reason) for reason in reasons),
        )


@dataclass(frozen=True, slots=True)
class ProfessionalReferenceVideo:
    video_id: str
    path: Path
    sha256: str
    audit_path: Path
    audit_sha256: str
    fps: float
    cuts: tuple[ProfessionalReferenceCut, ...]


@dataclass(frozen=True, slots=True)
class ProfessionalReferenceProfile:
    profile_path: Path
    profile_sha256: str
    artist: str
    song: str
    references: tuple[ProfessionalReferenceVideo, ...]
    payload: Mapping[str, Any]

    @property
    def reference_count(self) -> int:
        return len(self.references)

    @property
    def reference_cut_count(self) -> int:
        return sum(len(reference.cuts) for reference in self.references)

    def reference_by_id(self, video_id: str) -> ProfessionalReferenceVideo:
        matches = tuple(
            reference
            for reference in self.references
            if reference.video_id == video_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate reference id: {video_id}")
        return matches[0]

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "profile_path": str(self.profile_path),
            "profile_sha256": self.profile_sha256,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "analysis_basis": PROFILE_ANALYSIS_BASIS,
            "artist": self.artist,
            "song": self.song,
            "reference_count": self.reference_count,
            "reference_cut_count": self.reference_cut_count,
            "references": [
                {
                    "video_id": reference.video_id,
                    "path": str(reference.path),
                    "sha256": reference.sha256,
                    "audit_path": str(reference.audit_path),
                    "audit_sha256": reference.audit_sha256,
                    "cut_count": len(reference.cuts),
                }
                for reference in self.references
            ],
        }

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        verify_reference_media_sha256: bool = True,
    ) -> "ProfessionalReferenceProfile":
        profile_path = Path(path).expanduser().resolve()
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
        payload = _mapping(raw, "professional-reference profile")
        if _integer(payload.get("schema_version"), "profile schema_version") != 1:
            raise ValueError("unsupported professional-reference profile schema")
        if payload.get("analysis_basis") != PROFILE_ANALYSIS_BASIS:
            raise ValueError("unexpected professional-reference analysis basis")
        if str(payload.get("artist") or "").strip() != "IVE":
            raise ValueError("professional-reference profile must be for IVE")
        song = str(payload.get("song") or "").strip()
        if not song:
            raise ValueError("professional-reference profile song cannot be empty")
        declared_count = _integer(
            payload.get("reference_count"),
            "profile reference_count",
        )
        if declared_count != EXPECTED_REFERENCE_COUNT:
            raise ValueError(
                "professional-reference profile requires exactly five references"
            )
        raw_references = _array(
            payload.get("reference_videos"),
            "profile reference_videos",
        )
        if len(raw_references) != declared_count:
            raise ValueError("reference_count does not match reference_videos")

        references: list[ProfessionalReferenceVideo] = []
        all_cuts: list[ProfessionalReferenceCut] = []
        for index, raw_reference in enumerate(raw_references):
            reference = _mapping(raw_reference, f"reference_videos[{index}]")
            video_id = str(reference.get("video_id") or "").strip()
            if not video_id:
                raise ValueError("reference video_id cannot be empty")
            media_path = Path(str(reference.get("path") or "")).expanduser().resolve()
            if not media_path.is_file():
                raise ValueError(f"professional reference media is missing: {media_path}")
            declared_sha = str(reference.get("sha256") or "").lower()
            if len(declared_sha) != 64 or any(
                character not in "0123456789abcdef" for character in declared_sha
            ):
                raise ValueError("reference media sha256 must be 64 lowercase hex digits")
            if verify_reference_media_sha256:
                actual_sha = _sha256(media_path)
                if actual_sha != declared_sha:
                    raise ValueError(
                        f"reference media SHA-256 mismatch for {video_id}"
                    )
            fps = _finite(reference.get("fps"), "reference fps")
            if fps <= 0.0:
                raise ValueError("reference fps must be positive")
            audit_path = Path(
                str(reference.get("audit_path") or "")
            ).expanduser().resolve()
            if not audit_path.is_file():
                raise ValueError(f"reference audit is missing: {audit_path}")
            audit_raw = json.loads(audit_path.read_text(encoding="utf-8"))
            audit = _mapping(audit_raw, f"reference audit {video_id}")
            if audit.get("analysis_basis") != (
                "same_song_human_stage_mix_cut_quality_calibration_v1"
            ):
                raise ValueError(f"unexpected audit analysis basis for {video_id}")
            if Path(str(audit.get("reference") or "")).resolve() != media_path:
                raise ValueError(f"audit media path does not match {video_id}")
            cuts = tuple(
                ProfessionalReferenceCut.from_mapping(
                    _mapping(item, f"audit cut {video_id}:{cut_index}")
                )
                for cut_index, item in enumerate(
                    _array(audit.get("cuts"), f"audit cuts {video_id}")
                )
            )
            if not cuts:
                raise ValueError(f"reference audit has no cuts: {video_id}")
            expected_summary = _mapping(
                reference.get("summary"),
                f"profile reference summary {video_id}",
            )
            audit_summary = _mapping(
                audit.get("summary"),
                f"audit summary {video_id}",
            )
            _assert_equivalent(
                expected_summary,
                audit_summary,
                name=f"reference summary {video_id}",
            )
            recomputed_summary = _summarize_cuts(cuts)
            _assert_equivalent(
                expected_summary,
                recomputed_summary,
                name=f"recomputed reference summary {video_id}",
            )
            references.append(
                ProfessionalReferenceVideo(
                    video_id=video_id,
                    path=media_path,
                    sha256=declared_sha,
                    audit_path=audit_path,
                    audit_sha256=_sha256(audit_path),
                    fps=fps,
                    cuts=cuts,
                )
            )
            all_cuts.extend(cuts)

        video_ids = tuple(reference.video_id for reference in references)
        media_paths = tuple(reference.path for reference in references)
        audit_paths = tuple(reference.audit_path for reference in references)
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("professional references cannot repeat a video_id")
        if len(set(media_paths)) != len(media_paths):
            raise ValueError("professional references cannot repeat media")
        if len(set(audit_paths)) != len(audit_paths):
            raise ValueError("professional references cannot repeat an audit")
        declared_cut_count = _integer(
            payload.get("reference_cut_count"),
            "profile reference_cut_count",
        )
        if declared_cut_count != len(all_cuts):
            raise ValueError("reference_cut_count does not match loaded audits")
        aggregate = _mapping(
            payload.get("aggregate_human_visible_cut_profile"),
            "aggregate_human_visible_cut_profile",
        )
        _assert_equivalent(
            aggregate,
            _summarize_cuts(all_cuts),
            name="aggregate_human_visible_cut_profile",
        )
        return cls(
            profile_path=profile_path,
            profile_sha256=_sha256(profile_path),
            artist="IVE",
            song=song,
            references=tuple(references),
            payload=deepcopy(dict(payload)),
        )


@dataclass(frozen=True, slots=True)
class ProfessionalReferenceGateResult:
    payload: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])

    def write(self, path: Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()


def _summarize_cuts(
    cuts: Sequence[ProfessionalReferenceCut],
) -> dict[str, Any]:
    reasons: Counter[str] = Counter(
        reason
        for cut in cuts
        for reason in cut.absolute_issue_reasons
    )
    metrics = {
        component: _distribution(
            [
                float(cut.discontinuities[component])
                for cut in cuts
                if cut.discontinuities[component] is not None
            ]
        )
        for component in DISCONTINUITY_COMPONENTS
    }
    issue_count = sum(bool(cut.absolute_issue_reasons) for cut in cuts)
    return {
        "cut_count": len(cuts),
        "absolute_issue_count": issue_count,
        "absolute_issue_fraction": issue_count / len(cuts) if cuts else 0.0,
        "absolute_reason_counts": dict(reasons),
        "score": _distribution([cut.score for cut in cuts]),
        "metrics": metrics,
    }


def summarize_professional_reference_cuts(
    cuts: Sequence[ProfessionalReferenceCut],
) -> dict[str, Any]:
    """Return the canonical summary stored beside human reference cuts."""

    checked = tuple(cuts)
    if any(not isinstance(cut, ProfessionalReferenceCut) for cut in checked):
        raise TypeError("cuts must contain ProfessionalReferenceCut values")
    return _summarize_cuts(checked)


def _lower_prediction_envelope(
    values: Sequence[float],
    *,
    multiplier: float,
    domain_minimum: float,
) -> float:
    clean = tuple(_finite(value, "prediction-envelope value") for value in values)
    if not clean:
        raise ValueError("prediction envelope needs at least one reference value")
    span = max(clean) - min(clean)
    return max(domain_minimum, min(clean) - multiplier * span)


def _upper_prediction_envelope(
    values: Sequence[float],
    *,
    multiplier: float,
    domain_maximum: float,
) -> float:
    clean = tuple(_finite(value, "prediction-envelope value") for value in values)
    if not clean:
        raise ValueError("prediction envelope needs at least one reference value")
    span = max(clean) - min(clean)
    return min(domain_maximum, max(clean) + multiplier * span)


def _is_technical_reason(reason: str) -> bool:
    normalized = str(reason).strip().lower()
    return any(marker in normalized for marker in _TECHNICAL_REASON_MARKERS)


def technical_issue_reasons_from_review(
    review: Mapping[str, Any],
) -> tuple[str, ...]:
    """Collect serialized technical failures without reclassifying backgrounds."""

    reasons: set[str] = set()
    raw_boundaries = _array(review.get("boundaries"), "proxy review boundaries")
    for boundary_index, raw_boundary in enumerate(raw_boundaries):
        boundary = _mapping(raw_boundary, f"proxy boundary {boundary_index}")
        for key in ("technical_integrity_issue_reasons", "issue_reasons"):
            raw_reasons = boundary.get(key, [])
            if not isinstance(raw_reasons, list):
                raise ValueError(f"proxy boundary {boundary_index} {key} must be an array")
            for raw_reason in raw_reasons:
                reason = str(raw_reason)
                if _is_technical_reason(reason):
                    reasons.add(reason)
    raw_domains = review.get("boundary_domains")
    if isinstance(raw_domains, Mapping):
        for domain in ("tier_d_continuity", "native_source_cut"):
            summary = raw_domains.get(domain)
            if isinstance(summary, Mapping) and not bool(summary.get("passed", True)):
                reasons.add(f"{domain}_zero_tolerance_domain_failed")
    proxy = _mapping(review.get("proxy"), "proxy review proxy")
    decoded_count = _integer(
        proxy.get("decoded_frame_count"),
        "proxy decoded_frame_count",
    )
    if decoded_count <= 0:
        reasons.add("decoded_proxy_has_no_frames")
    return tuple(sorted(reasons))


def visible_cuts_from_review(
    review: Mapping[str, Any],
) -> tuple[ProfessionalReferenceCut, ...]:
    """Read introduced cross-source cuts from current or legacy review JSON."""

    selected: list[ProfessionalReferenceCut] = []
    for index, raw_boundary in enumerate(
        _array(review.get("boundaries"), "proxy review boundaries")
    ):
        boundary = _mapping(raw_boundary, f"proxy boundary {index}")
        domain = boundary.get("review_domain")
        if domain is not None:
            introduced = domain == "introduced_cross_source"
        else:
            introduced = (
                boundary.get("outgoing_source_index")
                != boundary.get("incoming_source_index")
            )
        if introduced:
            selected.append(ProfessionalReferenceCut.from_mapping(boundary))
    return tuple(selected)


def evaluate_professional_reference_gate(
    candidate_cuts: Sequence[ProfessionalReferenceCut | Mapping[str, Any]],
    profile: ProfessionalReferenceProfile,
    *,
    excluded_reference_id: str | None = None,
    technical_issue_reasons: Sequence[str] = (),
) -> ProfessionalReferenceGateResult:
    """Compare one candidate with the human distribution prediction envelope."""

    normalized_cuts = tuple(
        cut
        if isinstance(cut, ProfessionalReferenceCut)
        else ProfessionalReferenceCut.from_mapping(
            _mapping(cut, f"candidate cut {index}")
        )
        for index, cut in enumerate(candidate_cuts)
    )
    training_references = tuple(
        reference
        for reference in profile.references
        if reference.video_id != excluded_reference_id
    )
    if excluded_reference_id is not None:
        profile.reference_by_id(excluded_reference_id)
    expected_training_count = (
        EXPECTED_REFERENCE_COUNT - 1
        if excluded_reference_id is not None
        else EXPECTED_REFERENCE_COUNT
    )
    if len(training_references) != expected_training_count:
        raise ValueError("professional-reference training fold has wrong size")
    training_summaries = {
        reference.video_id: _summarize_cuts(reference.cuts)
        for reference in training_references
    }
    pooled_training_cuts = tuple(
        cut for reference in training_references for cut in reference.cuts
    )
    candidate_summary = _summarize_cuts(normalized_cuts)

    score_reference_values = tuple(
        float(summary["score"]["p10"])
        for summary in training_summaries.values()
    )
    score_floor = _lower_prediction_envelope(
        score_reference_values,
        multiplier=QUANTILE_PREDICTION_RANGE_MULTIPLIER,
        domain_minimum=0.0,
    )
    candidate_score_p10 = candidate_summary["score"]["p10"]
    scalar_passed = bool(
        candidate_score_p10 is not None
        and float(candidate_score_p10) >= score_floor - 1e-9
    )

    component_results: dict[str, Any] = {}
    for component in DISCONTINUITY_COMPONENTS:
        training_values = tuple(
            float(cut.discontinuities[component])
            for cut in pooled_training_cuts
            if cut.discontinuities[component] is not None
        )
        candidate_values = tuple(
            float(cut.discontinuities[component])
            for cut in normalized_cuts
            if cut.discontinuities[component] is not None
        )
        if not training_values:
            raise ValueError(f"professional profile has no {component} evidence")
        reference_p90 = tuple(
            float(summary["metrics"][component]["p90"])
            for summary in training_summaries.values()
            if summary["metrics"][component]["p90"] is not None
        )
        reference_p95 = tuple(
            float(summary["metrics"][component]["p95"])
            for summary in training_summaries.values()
            if summary["metrics"][component]["p95"] is not None
        )
        p90_ceiling = _upper_prediction_envelope(
            reference_p90,
            multiplier=QUANTILE_PREDICTION_RANGE_MULTIPLIER,
            domain_maximum=1.0,
        )
        p95_ceiling = _upper_prediction_envelope(
            reference_p95,
            multiplier=QUANTILE_PREDICTION_RANGE_MULTIPLIER,
            domain_maximum=1.0,
        )
        severe_threshold = float(_quantile(training_values, 0.95))
        reference_severe_fractions: list[float] = []
        for reference in training_references:
            values = tuple(
                float(cut.discontinuities[component])
                for cut in reference.cuts
                if cut.discontinuities[component] is not None
            )
            if values:
                reference_severe_fractions.append(
                    sum(value > severe_threshold for value in values) / len(values)
                )
        severe_fraction_ceiling = _upper_prediction_envelope(
            reference_severe_fractions,
            multiplier=PROPORTION_PREDICTION_RANGE_MULTIPLIER,
            domain_maximum=1.0,
        )
        candidate_p90 = _quantile(candidate_values, 0.90)
        candidate_p95 = _quantile(candidate_values, 0.95)
        candidate_severe_fraction = (
            sum(value > severe_threshold for value in candidate_values)
            / len(candidate_values)
            if candidate_values
            else None
        )
        comparable = bool(candidate_values)
        p90_passed = (
            candidate_p90 is not None
            and candidate_p90 <= p90_ceiling + 1e-9
        )
        p95_passed = (
            candidate_p95 is not None
            and candidate_p95 <= p95_ceiling + 1e-9
        )
        severe_passed = (
            candidate_severe_fraction is not None
            and candidate_severe_fraction <= severe_fraction_ceiling + 1e-9
        )
        # Preserve the reviewer's explicit not-comparable policy.  Missing
        # face/silhouette/motion evidence is not invented as maximum mismatch.
        tail_check_results = (p90_passed, p95_passed, severe_passed)
        passed_tail_check_count = sum(tail_check_results)
        # These three tail statistics are deliberately correlated views of the
        # same small reference sample.  Requiring all three rejects a real
        # human edit when only one unstable quantile falls outside the
        # four-reference prediction envelope.  A two-of-three consensus keeps
        # a single noisy estimate from becoming a false hard failure while a
        # broadly degraded component still fails closed.
        component_passed = (
            passed_tail_check_count >= MINIMUM_COMPONENT_TAIL_CHECK_PASS_COUNT
            if comparable
            else True
        )
        component_results[component] = {
            "candidate_observation_count": len(candidate_values),
            "training_observation_count": len(training_values),
            "comparable": comparable,
            "missing_candidate_policy": (
                "not_comparable_never_maximum_mismatch"
                if not comparable
                else "compared"
            ),
            "candidate_p90": candidate_p90,
            "reference_p90_values": list(reference_p90),
            "p90_ceiling": p90_ceiling,
            "p90_passed": p90_passed if comparable else None,
            "candidate_p95": candidate_p95,
            "reference_p95_values": list(reference_p95),
            "p95_ceiling": p95_ceiling,
            "p95_passed": p95_passed if comparable else None,
            "severe_outlier": {
                "threshold_basis": "pooled_training_human_p95",
                "threshold": severe_threshold,
                "candidate_fraction": candidate_severe_fraction,
                "reference_fractions": reference_severe_fractions,
                "fraction_ceiling": severe_fraction_ceiling,
                "passed": severe_passed if comparable else None,
            },
            "tail_check_consensus": {
                "passed_check_count": (
                    passed_tail_check_count if comparable else None
                ),
                "required_pass_count": (
                    MINIMUM_COMPONENT_TAIL_CHECK_PASS_COUNT
                ),
                "total_check_count": len(tail_check_results),
                "policy": "two_of_three_correlated_tail_checks",
                "passed": component_passed,
            },
            "passed": component_passed,
        }

    normalized_technical_reasons = tuple(
        sorted({str(reason).strip() for reason in technical_issue_reasons if str(reason).strip()})
    )
    technical_passed = not normalized_technical_reasons
    evidence_count_passed = (
        len(normalized_cuts) >= MINIMUM_CANDIDATE_VISIBLE_CUT_COUNT
    )
    components_passed = all(
        bool(result["passed"]) for result in component_results.values()
    )
    passed = bool(
        evidence_count_passed
        and scalar_passed
        and components_passed
        and technical_passed
    )
    return ProfessionalReferenceGateResult(
        {
            "schema_version": 1,
            "analysis": GATE_ANALYSIS_BASIS,
            "profile": profile.provenance_dict(),
            "evaluation_fold": {
                "excluded_reference_id": excluded_reference_id,
                "training_reference_ids": [
                    reference.video_id for reference in training_references
                ],
                "training_reference_count": len(training_references),
                "training_cut_count": len(pooled_training_cuts),
            },
            "policy": {
                "visible_cut_basis": "introduced_cross_source_only",
                "minimum_candidate_visible_cut_count": (
                    MINIMUM_CANDIDATE_VISIBLE_CUT_COUNT
                ),
                "scalar_gate": "candidate_p10_above_human_prediction_floor",
                "component_gates": [
                    "candidate_p90_below_human_prediction_ceiling",
                    "candidate_p95_below_human_prediction_ceiling",
                    "candidate_severe_outlier_fraction_below_human_envelope",
                ],
                "component_gate_consensus": (
                    "two_of_three_correlated_tail_checks"
                ),
                "quantile_prediction_range_multiplier": (
                    QUANTILE_PREDICTION_RANGE_MULTIPLIER
                ),
                "proportion_prediction_range_multiplier": (
                    PROPORTION_PREDICTION_RANGE_MULTIPLIER
                ),
                "absolute_editorial_issue_reasons": (
                    "diagnostic_only_not_a_relative_gate_failure"
                ),
                "technical_integrity": (
                    "zero_tolerance_never_relaxed_by_human_distribution"
                ),
            },
            "candidate_visible_cut_count": len(normalized_cuts),
            "evidence_count_passed": evidence_count_passed,
            "scalar_score": {
                "candidate_p10": candidate_score_p10,
                "reference_p10_values": list(score_reference_values),
                "prediction_floor": score_floor,
                "passed": scalar_passed,
            },
            "components": component_results,
            "absolute_editorial_diagnostics": {
                "issue_count": candidate_summary["absolute_issue_count"],
                "issue_fraction": candidate_summary["absolute_issue_fraction"],
                "reason_counts": candidate_summary["absolute_reason_counts"],
                "selection_gate": False,
            },
            "technical_integrity": {
                "issue_count": len(normalized_technical_reasons),
                "issue_reasons": list(normalized_technical_reasons),
                "zero_tolerance": True,
                "passed": technical_passed,
            },
            "components_passed": components_passed,
            "passed": passed,
        }
    )


def evaluate_professional_reference_review(
    review: Mapping[str, Any],
    profile: ProfessionalReferenceProfile,
) -> ProfessionalReferenceGateResult:
    return evaluate_professional_reference_gate(
        visible_cuts_from_review(review),
        profile,
        technical_issue_reasons=technical_issue_reasons_from_review(review),
    )


__all__ = [
    "DISCONTINUITY_COMPONENTS",
    "EXPECTED_REFERENCE_COUNT",
    "GATE_ANALYSIS_BASIS",
    "MINIMUM_CANDIDATE_VISIBLE_CUT_COUNT",
    "PROFILE_ANALYSIS_BASIS",
    "PROFILE_SCHEMA_VERSION",
    "ProfessionalReferenceCut",
    "ProfessionalReferenceGateResult",
    "ProfessionalReferenceProfile",
    "ProfessionalReferenceVideo",
    "evaluate_professional_reference_gate",
    "evaluate_professional_reference_review",
    "summarize_professional_reference_cuts",
    "technical_issue_reasons_from_review",
    "visible_cuts_from_review",
]
