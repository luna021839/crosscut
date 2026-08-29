"""Strict restart checkpoint for expensive GROUP visual candidate selection.

The checkpoint stores only JSON-native timeline geometry and scalar review
metadata.  It never stores decoded pixels or appearance embeddings.  Reuse is
allowed only when the caller supplies the exact canonical input fingerprint
that was written with the plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .failed_candidate_replay import deserialize_visual_timeline_plan
from .visual_score import VisualTimelinePlan


VISUAL_CANDIDATE_CHECKPOINT_CONTRACT = (
    "input_bound_full_visual_candidate_plan_restart_v1"
)
VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS = (
    "pre_derivation_visual_candidate_checkpoint_v2"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_CHECKPOINT_BYTES = 512 * 1024 * 1024


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointReviewIssue:
    code: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("checkpoint review issue code cannot be blank")


@dataclass(frozen=True, slots=True)
class CheckpointTimelineReview:
    hard_failures: tuple[CheckpointReviewIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.hard_failures


@dataclass(frozen=True, slots=True)
class CheckpointCandidateRecord:
    optimizer_score: float
    review: CheckpointTimelineReview

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.optimizer_score)):
            raise ValueError("checkpoint optimizer score must be finite")


@dataclass(frozen=True, slots=True)
class CheckpointMaterializedVisualCandidate:
    candidate_index: int
    review_rank: int
    cadence_profile: str
    visual_plan: VisualTimelinePlan = field(repr=False)
    pre_review_hard_failure_codes: tuple[str, ...]
    selected_for_initial_stream: bool

    def __post_init__(self) -> None:
        if self.candidate_index < 0 or self.review_rank < 0:
            raise ValueError("checkpoint candidate indices must be non-negative")
        if not self.cadence_profile.strip():
            raise ValueError("checkpoint cadence profile cannot be blank")
        if self.visual_plan.selected_candidate_index != self.candidate_index:
            raise ValueError("checkpoint visual plan selected index drifted")
        if not self.visual_plan.choices:
            raise ValueError("checkpoint visual plan cannot be empty")
        if len(set(self.pre_review_hard_failure_codes)) != len(
            self.pre_review_hard_failure_codes
        ):
            raise ValueError("checkpoint hard failure codes must be unique")

    @property
    def candidate_record(self) -> CheckpointCandidateRecord:
        return CheckpointCandidateRecord(
            optimizer_score=self.visual_plan.optimizer_score,
            review=CheckpointTimelineReview(
                tuple(
                    CheckpointReviewIssue(code)
                    for code in self.pre_review_hard_failure_codes
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedVisualCandidateCheckpoint:
    path: Path
    input_identity_sha256: str
    candidates: tuple[CheckpointMaterializedVisualCandidate, ...]

    @property
    def initial_candidates(
        self,
    ) -> tuple[CheckpointMaterializedVisualCandidate, ...]:
        selected = tuple(
            item for item in self.candidates if item.selected_for_initial_stream
        )
        if len(selected) != 8:
            raise ValueError("checkpoint must select exactly eight initial candidates")
        return selected


def materialized_candidate_checkpoint_payload(
    candidate: Any,
    *,
    selected_for_initial_stream: bool,
) -> dict[str, Any]:
    """Serialize one full non-biometric plan needed for deterministic resume."""

    visual_plan = getattr(candidate, "visual_plan", None)
    if not isinstance(visual_plan, VisualTimelinePlan):
        raise TypeError("candidate visual_plan must be a VisualTimelinePlan")
    candidate_index = int(getattr(candidate, "candidate_index"))
    review_rank = int(getattr(candidate, "review_rank"))
    cadence_profile = str(getattr(candidate, "cadence_profile"))
    record = getattr(candidate, "candidate_record", None)
    review = getattr(record, "review", None)
    hard_failure_codes = tuple(
        str(getattr(issue, "code"))
        for issue in getattr(review, "hard_failures", ())
    )
    return {
        "candidate_index": candidate_index,
        "review_rank": review_rank,
        "cadence_profile": cadence_profile,
        "selected_for_initial_stream": bool(selected_for_initial_stream),
        "pre_review_hard_failure_codes": list(hard_failure_codes),
        "visual_timeline": visual_plan.to_dict(),
        "raw_frames_persisted": False,
        "appearance_embeddings_persisted": False,
    }


def _load_candidate(
    value: Any,
    *,
    expected_source_count: int,
) -> CheckpointMaterializedVisualCandidate:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint candidate must be an object")
    candidate_index = value.get("candidate_index")
    review_rank = value.get("review_rank")
    cadence_profile = value.get("cadence_profile")
    selected = value.get("selected_for_initial_stream")
    codes = value.get("pre_review_hard_failure_codes")
    visual_payload = value.get("visual_timeline")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or isinstance(review_rank, bool)
        or not isinstance(review_rank, int)
        or not isinstance(cadence_profile, str)
        or not isinstance(selected, bool)
        or not isinstance(codes, list)
        or any(not isinstance(code, str) or not code for code in codes)
        or value.get("raw_frames_persisted") is not False
        or value.get("appearance_embeddings_persisted") is not False
    ):
        raise ValueError("checkpoint candidate scalar contract is invalid")
    visual_plan = deserialize_visual_timeline_plan(
        visual_payload if isinstance(visual_payload, Mapping) else {}
    )
    if not isinstance(visual_plan, VisualTimelinePlan):
        raise ValueError("checkpoint visual timeline reconstruction failed")
    if any(
        choice.source_index < 0 or choice.source_index >= expected_source_count
        for choice in visual_plan.choices
    ):
        raise ValueError("checkpoint source index is outside current sources")
    return CheckpointMaterializedVisualCandidate(
        candidate_index=candidate_index,
        review_rank=review_rank,
        cadence_profile=cadence_profile,
        visual_plan=visual_plan,
        pre_review_hard_failure_codes=tuple(codes),
        selected_for_initial_stream=selected,
    )


def load_visual_candidate_checkpoint(
    path: Path,
    *,
    expected_input_identity_sha256: str,
    expected_source_count: int,
) -> LoadedVisualCandidateCheckpoint:
    source = Path(path)
    if _SHA256.fullmatch(expected_input_identity_sha256) is None:
        raise ValueError("expected checkpoint input identity is invalid")
    if source.is_symlink() or not source.is_file():
        raise ValueError("visual candidate checkpoint path is invalid")
    size = source.stat().st_size
    if not 1 <= size <= _MAXIMUM_CHECKPOINT_BYTES:
        raise ValueError("visual candidate checkpoint size is invalid")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("visual candidate checkpoint must be an object")
    if payload.get("checkpoint_contract") != VISUAL_CANDIDATE_CHECKPOINT_CONTRACT:
        raise ValueError("visual candidate checkpoint contract changed")
    if payload.get("analysis_basis") != VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS:
        raise ValueError("visual candidate checkpoint analysis basis changed")
    actual_identity = payload.get("input_identity_sha256")
    if not isinstance(actual_identity, str) or not hmac.compare_digest(
        actual_identity,
        expected_input_identity_sha256,
    ):
        raise ValueError("visual candidate checkpoint input identity mismatch")
    raw_identity = payload.get("input_identity")
    if not isinstance(raw_identity, Mapping) or not hmac.compare_digest(
        canonical_json_sha256(raw_identity),
        actual_identity,
    ):
        raise ValueError("visual candidate checkpoint identity checksum mismatch")
    raw_candidates = payload.get("materialized_candidate_plans")
    if not isinstance(raw_candidates, list) or not 8 <= len(raw_candidates) <= 32:
        raise ValueError("visual candidate checkpoint needs 8 to 32 plans")
    candidates = tuple(
        _load_candidate(
            value,
            expected_source_count=expected_source_count,
        )
        for value in raw_candidates
    )
    indices = tuple(item.candidate_index for item in candidates)
    if len(indices) != len(set(indices)):
        raise ValueError("visual candidate checkpoint indices are duplicated")
    loaded = LoadedVisualCandidateCheckpoint(
        path=source.resolve(strict=True),
        input_identity_sha256=actual_identity,
        candidates=candidates,
    )
    loaded.initial_candidates
    return loaded


def find_latest_matching_visual_candidate_checkpoint(
    output_directory: Path,
    *,
    expected_input_identity_sha256: str,
    expected_source_count: int,
) -> LoadedVisualCandidateCheckpoint | None:
    root = Path(output_directory)
    if root.is_symlink() or not root.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in root.glob("*-pre-derivation-candidate-selection.json")
            if not path.name.startswith("._")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            return load_visual_candidate_checkpoint(
                path,
                expected_input_identity_sha256=expected_input_identity_sha256,
                expected_source_count=expected_source_count,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


__all__ = [
    "CheckpointMaterializedVisualCandidate",
    "LoadedVisualCandidateCheckpoint",
    "VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS",
    "VISUAL_CANDIDATE_CHECKPOINT_CONTRACT",
    "canonical_json_sha256",
    "find_latest_matching_visual_candidate_checkpoint",
    "load_visual_candidate_checkpoint",
    "materialized_candidate_checkpoint_payload",
]
