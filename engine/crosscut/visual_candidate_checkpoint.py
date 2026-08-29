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
from .visual_feature_cache import PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT
from .visual_score import VisualTimelinePlan


VISUAL_CANDIDATE_CHECKPOINT_CONTRACT = (
    "input_bound_full_visual_candidate_plan_restart_v1"
)
VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS = (
    "pre_derivation_visual_candidate_checkpoint_v2"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_CHECKPOINT_BYTES = 512 * 1024 * 1024
_CODE_FINGERPRINT_FIELDS = (
    "selector_code_sha256",
    "crop_rerank_code_sha256",
)


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
    stored_input_identity_sha256: str
    code_fingerprint_migration: str | None
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


@dataclass(frozen=True, slots=True)
class CheckpointCodeFingerprintMigration:
    """One exact, direction-specific code-hash equivalence migration.

    Candidate checkpoints bind every media, clock, policy, and selector input.
    A dependency-fingerprint implementation change can alter the two code
    digests without changing executable selector semantics.  This object
    allows only an audited old->new pair while requiring every other raw
    identity field to remain byte-for-byte canonical-JSON equivalent.
    """

    contract: str
    legacy_selector_code_sha256: str
    current_selector_code_sha256: str
    legacy_crop_rerank_code_sha256: str
    current_crop_rerank_code_sha256: str
    legacy_fingerprint_contract: str | None = None
    current_fingerprint_contract: str = PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT

    def __post_init__(self) -> None:
        if not self.contract.strip() or self.contract != self.contract.strip():
            raise ValueError("checkpoint code migration contract is invalid")
        for name in (
            "legacy_selector_code_sha256",
            "current_selector_code_sha256",
            "legacy_crop_rerank_code_sha256",
            "current_crop_rerank_code_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError("checkpoint code migration digest is invalid")
        if self.legacy_fingerprint_contract is not None and (
            not self.legacy_fingerprint_contract.strip()
            or self.legacy_fingerprint_contract
            != self.legacy_fingerprint_contract.strip()
        ):
            raise ValueError("legacy fingerprint contract is invalid")
        if (
            not self.current_fingerprint_contract.strip()
            or self.current_fingerprint_contract
            != self.current_fingerprint_contract.strip()
        ):
            raise ValueError("current fingerprint contract is invalid")

    def matches(
        self,
        legacy: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        legacy_values = dict(legacy)
        current_values = dict(current)
        if legacy_values.get("selector_code_sha256") != (
            self.legacy_selector_code_sha256
        ) or current_values.get("selector_code_sha256") != (
            self.current_selector_code_sha256
        ):
            return False
        if legacy_values.get("crop_rerank_code_sha256") != (
            self.legacy_crop_rerank_code_sha256
        ) or current_values.get("crop_rerank_code_sha256") != (
            self.current_crop_rerank_code_sha256
        ):
            return False
        legacy_has_contract = "code_fingerprint_contract" in legacy_values
        if self.legacy_fingerprint_contract is None:
            if legacy_has_contract:
                return False
        elif (
            not legacy_has_contract
            or legacy_values["code_fingerprint_contract"]
            != self.legacy_fingerprint_contract
        ):
            return False
        if (
            "code_fingerprint_contract" not in current_values
            or current_values["code_fingerprint_contract"]
            != self.current_fingerprint_contract
        ):
            return False
        ignored = {
            *_CODE_FINGERPRINT_FIELDS,
            "code_fingerprint_contract",
        }
        legacy_non_code = {
            key: value
            for key, value in legacy_values.items()
            if key not in ignored
        }
        current_non_code = {
            key: value
            for key, value in current_values.items()
            if key not in ignored
        }
        try:
            return hmac.compare_digest(
                canonical_json_sha256(legacy_non_code),
                canonical_json_sha256(current_non_code),
            )
        except (TypeError, ValueError):
            return False


VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS = (
    CheckpointCodeFingerprintMigration(
        contract=(
            "whole_import_node_ast_v1_to_alias_ast_v2_"
            "direct_cache_routing_v3_20260830"
        ),
        legacy_selector_code_sha256=(
            "363acfa826d846bfc62396f483ce1adc6d0b1ff903ffd32c39630dc2db70e857"
        ),
        current_selector_code_sha256=(
            "8004e6ff1e9813da730b07e3c031b940666cf27707cd1b231fc6af36d0d40aa2"
        ),
        legacy_crop_rerank_code_sha256=(
            "23aa118ec3a822544ca36d90ba72c821a6e9157175efe863b7c181b6e73a97ba"
        ),
        current_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
    ),
    CheckpointCodeFingerprintMigration(
        contract="alias_ast_v2_direct_cache_routing_v2_to_v3_20260830",
        legacy_selector_code_sha256=(
            "9ad134a8bac4a4946bb56d5ca4d161ee226320fe2504b826fda6ad1c5abc5982"
        ),
        current_selector_code_sha256=(
            "8004e6ff1e9813da730b07e3c031b940666cf27707cd1b231fc6af36d0d40aa2"
        ),
        legacy_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        current_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        legacy_fingerprint_contract=PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
    ),
    CheckpointCodeFingerprintMigration(
        contract="alias_ast_v2_strict_cache_liveness_v3_to_v4_20260830",
        legacy_selector_code_sha256=(
            "59aa2115e68c773db670641ecf44581006ab6a0bb4dc7d70f8f6e65d87e731e8"
        ),
        current_selector_code_sha256=(
            "8004e6ff1e9813da730b07e3c031b940666cf27707cd1b231fc6af36d0d40aa2"
        ),
        legacy_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        current_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        legacy_fingerprint_contract=PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
    ),
    CheckpointCodeFingerprintMigration(
        contract=(
            "alias_ast_v2_joint_stream_sface_equivalence_v4_to_v5_20260830"
        ),
        legacy_selector_code_sha256=(
            "d1529daef216659bf3fa9b482a1fb5ab1bf53ec8c1ac1b3f486b2d65ae349230"
        ),
        current_selector_code_sha256=(
            "8004e6ff1e9813da730b07e3c031b940666cf27707cd1b231fc6af36d0d40aa2"
        ),
        legacy_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        current_crop_rerank_code_sha256=(
            "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
        ),
        legacy_fingerprint_contract=PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
    ),
)


def _resolved_checkpoint_identity_migration(
    legacy: Mapping[str, Any],
    current: Mapping[str, Any],
    migrations: Sequence[CheckpointCodeFingerprintMigration],
) -> CheckpointCodeFingerprintMigration | None:
    matches = tuple(
        migration
        for migration in migrations
        if migration.matches(legacy, current)
    )
    if len(matches) > 1:
        raise ValueError("checkpoint code migration is ambiguous")
    return matches[0] if matches else None


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
    expected_input_identity: Mapping[str, Any] | None = None,
    code_fingerprint_migrations: Sequence[
        CheckpointCodeFingerprintMigration
    ] = (),
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
    raw_identity = payload.get("input_identity")
    if (
        not isinstance(actual_identity, str)
        or _SHA256.fullmatch(actual_identity) is None
        or not isinstance(raw_identity, Mapping)
        or not hmac.compare_digest(
            canonical_json_sha256(raw_identity),
            actual_identity,
        )
    ):
        raise ValueError("visual candidate checkpoint identity checksum mismatch")
    migration: CheckpointCodeFingerprintMigration | None = None
    if not hmac.compare_digest(
        actual_identity,
        expected_input_identity_sha256,
    ):
        if expected_input_identity is None or not hmac.compare_digest(
            canonical_json_sha256(expected_input_identity),
            expected_input_identity_sha256,
        ):
            raise ValueError("visual candidate checkpoint input identity mismatch")
        migration = _resolved_checkpoint_identity_migration(
            raw_identity,
            expected_input_identity,
            tuple(code_fingerprint_migrations),
        )
        if migration is None:
            raise ValueError("visual candidate checkpoint input identity mismatch")
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
        input_identity_sha256=expected_input_identity_sha256,
        stored_input_identity_sha256=actual_identity,
        code_fingerprint_migration=(
            migration.contract if migration is not None else None
        ),
        candidates=candidates,
    )
    loaded.initial_candidates
    return loaded


def find_latest_matching_visual_candidate_checkpoint(
    output_directory: Path,
    *,
    expected_input_identity_sha256: str,
    expected_source_count: int,
    expected_input_identity: Mapping[str, Any] | None = None,
    code_fingerprint_migrations: Sequence[
        CheckpointCodeFingerprintMigration
    ] = (),
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
                expected_input_identity=expected_input_identity,
                code_fingerprint_migrations=code_fingerprint_migrations,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


__all__ = [
    "CheckpointCodeFingerprintMigration",
    "CheckpointMaterializedVisualCandidate",
    "LoadedVisualCandidateCheckpoint",
    "VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS",
    "VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS",
    "VISUAL_CANDIDATE_CHECKPOINT_CONTRACT",
    "canonical_json_sha256",
    "find_latest_matching_visual_candidate_checkpoint",
    "load_visual_candidate_checkpoint",
    "materialized_candidate_checkpoint_payload",
]
