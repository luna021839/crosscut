"""Load a hash-bound production policy from an IVE montage profile.

The profile describes observations from human reference edits.  This module
keeps those observations separate from the small number of explicit product
editorial requirements used by the production gate.  In particular, it never
derives a non-wide quota as ``1 - wide`` and never calls weak, identity-free
person-count evidence proof of a full group formation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .professional_montage_grammar import (
    MODULE_VERSION,
    PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION,
    ROLE_NAMES,
    SHOT_SIZE_NAMES,
)


PROFILE_CONTENT_HASH_ALGORITHM = (
    "sha256_of_canonical_json_with_profile_content_sha256_null"
)
PROFILE_GENERATOR_MODULE = "crosscut.professional_montage_grammar"
FORMATION_PERSON_COUNT_SEMANTICS = (
    "identity_free_observed_counts_only_no_group_formation_threshold"
)
WEAK_FORMATION_EVIDENCE_LABEL = (
    "weak_observed_person_evidence_never_proven_full_formation"
)
PRODUCT_EDITORIAL_REQUIREMENT_BASIS = (
    "product_editorial_requirement_informed_by_observed_event_shape"
)
REFERENCE_FLOOR_BASIS = (
    "lower_quantile_of_contributing_human_reference_edits"
)
REFERENCE_FLOOR_QUANTILE = 0.10
REFERENCE_PREFERRED_TARGET_QUANTILE = 0.50
DURATION_PRODUCT_MINIMUM_QUANTILE = 0.25
DURATION_PREFERRED_TARGET_QUANTILE = 0.50
WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE = 0.75
WIDE_RUN_HARD_MAXIMUM_QUANTILE = 1.00
HIGHLIGHT_CADENCE_PRODUCT_MINIMUM_QUANTILE = 0.50
HIGHLIGHT_CADENCE_PREFERRED_TARGET_QUANTILE = 0.75
HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT = (
    "per_reference_chorus_shot_boundary_rate_per_minute_v1"
)
HIGHLIGHT_CADENCE_STATISTIC_PATH = (
    "songs[].references[].by_acoustic_role.chorus_like."
    "derived_shot_boundary_rate_per_minute"
)
CLOSE_BURST_EVENT_EVIDENCE_CONTRACT = (
    "ive_chorus_maximal_close_only_multi_source_burst_event_p10_v1"
)
CLOSE_BURST_SHOT_COUNT_STATISTIC_PATH = (
    "aggregate_by_acoustic_role.chorus_like.burst_event_distributions."
    "closeup_multi_source_shot_count"
)
CLOSE_BURST_DURATION_STATISTIC_PATH = (
    "aggregate_by_acoustic_role.chorus_like.burst_event_distributions."
    "closeup_multi_source_duration_seconds"
)
CLOSE_BURST_SOURCE_SWITCH_FRACTION_STATISTIC_PATH = (
    "aggregate_by_acoustic_role.chorus_like.burst_event_distributions."
    "closeup_multi_source_switch_fraction"
)
CLOSE_BURST_POLICY_BINDING_CONTRACT = (
    "hash_bound_professional_montage_close_burst_soft_preference_v1"
)
HIGHLIGHT_DURATION_EVIDENCE_CONTRACT = (
    "per_reference_chorus_observed_shot_scale_duration_fractions_v2"
)
HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH = (
    "songs[].references[].by_acoustic_role.chorus_like."
    "shot_size_duration_fractions.close"
)
HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH = (
    "songs[].references[].by_acoustic_role.chorus_like."
    "nonwide_duration_fraction"
)
HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT = (
    "per_reference_chorus_maximal_contiguous_known_wide_run_v2"
)
HIGHLIGHT_WIDE_RUN_STATISTIC_PATH = (
    "songs[].references[].by_acoustic_role.chorus_like."
    "maximum_consecutive_wide_run_seconds"
)
ROLE_CONDITIONED_DURATION_EVIDENCE_CONTRACT = (
    "per_reference_build_chorus_bridge_observed_duration_policy_v1"
)
ROLE_CONDITIONED_DURATION_ROLES = (
    "build",
    "chorus_like",
    "bridge_like",
)
FORMATION_EVIDENCE_THRESHOLD_NORMALIZED = 1.0 / 6.0
EXPECTED_SONG_COUNT = 4
EXPECTED_REFERENCE_COUNT = 20

_SHA256_LENGTH = 64
_TOP_LEVEL_KEYS = {
    "schema_version",
    "analysis_basis",
    "configuration",
    "distributional_policy",
    "evidence_limitations",
    "provenance",
    "scope",
    "songs",
    "aggregate_by_acoustic_role",
}
_CHORUS_KEYS = {
    "contributing_reference_count",
    "pooled_counts",
    "pooled_durations_seconds",
    "reference_distributions",
    "burst_event_distributions",
    "wide_run_event_distributions",
    "wide_shot_person_count_evidence",
}


class ProfessionalMontagePolicyError(ValueError):
    """Raised when a profile cannot safely supply a production policy."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfessionalMontagePolicyError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProfessionalMontagePolicyError(f"{name} must be an array")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfessionalMontagePolicyError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfessionalMontagePolicyError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProfessionalMontagePolicyError(f"{name} must be finite")
    return number


def _sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProfessionalMontagePolicyError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProfessionalMontagePolicyError(
            f"profile is not canonical-JSON serializable: {exc}"
        ) from exc


def _reject_constant(value: str) -> None:
    raise ProfessionalMontagePolicyError(
        f"profile contains non-finite JSON number {value}"
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfessionalMontagePolicyError(
                f"profile contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _stable_json_bytes(path: Path) -> bytes:
    source = Path(path)
    if not source.is_absolute():
        raise ProfessionalMontagePolicyError("profile path must be absolute")
    if source.suffix.lower() != ".json":
        raise ProfessionalMontagePolicyError("profile must be a JSON file")
    try:
        path_before = source.lstat()
    except OSError as exc:
        raise ProfessionalMontagePolicyError(
            f"could not stat montage profile {source}: {exc}"
        ) from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ProfessionalMontagePolicyError(
            f"profile must be a regular non-symlink file: {source}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ProfessionalMontagePolicyError(
            f"could not open montage profile {source}: {exc}"
        ) from exc
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.lstat()
    except OSError as exc:
        raise ProfessionalMontagePolicyError(
            f"profile disappeared while it was read: {source}"
        ) from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if not (
        identity(path_before)
        == identity(opened_before)
        == identity(opened_after)
        == identity(path_after)
    ):
        raise ProfessionalMontagePolicyError(
            f"profile changed while it was read: {source}"
        )
    raw = b"".join(chunks)
    if len(raw) != opened_before.st_size:
        raise ProfessionalMontagePolicyError(
            f"profile size changed while it was read: {source}"
        )
    return raw


def _decode_profile(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ProfessionalMontagePolicyError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProfessionalMontagePolicyError(
            f"could not decode montage profile: {exc}"
        ) from exc
    return dict(_mapping(value, "montage profile"))


def _at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = _mapping(current, path).get(part)
    return current


@dataclass(frozen=True, slots=True)
class ProfessionalMontageProfileIdentity:
    analysis_basis: str
    schema_version: int
    path: str
    file_sha256: str
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return the exact stable identity persisted by downstream gates."""

        return {
            "analysis_basis": self.analysis_basis,
            "schema_version": self.schema_version,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
        }


def close_burst_policy_binding_payload(
    *,
    profile_identity: ProfessionalMontageProfileIdentity | Mapping[str, Any],
    policy_fingerprint_sha256: str,
    minimum_shot_count: float,
    minimum_duration_seconds: float,
    minimum_source_switch_fraction: float,
    event_reference_count: int,
) -> dict[str, Any]:
    """Build the canonical, hashable close-burst soft-policy binding.

    The three P10 values describe the six *observed events*, not an occurrence
    quota across twenty references.  Keeping their exact paths, identity, and
    sparse count in one digest lets both pre-render decoding and final rendered
    review reject a missing or accidentally altered production binding.
    """

    identity = (
        profile_identity.to_dict()
        if isinstance(profile_identity, ProfessionalMontageProfileIdentity)
        else dict(_mapping(profile_identity, "close burst profile identity"))
    )
    if set(identity) != {
        "analysis_basis",
        "schema_version",
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise ProfessionalMontagePolicyError(
            "close burst profile identity schema mismatch"
        )
    _sha256(policy_fingerprint_sha256, "close burst policy fingerprint")
    for name in ("file_sha256", "content_sha256"):
        _sha256(identity.get(name), f"close burst profile {name}")
    if not isinstance(identity.get("analysis_basis"), str) or not str(
        identity["analysis_basis"]
    ).strip():
        raise ProfessionalMontagePolicyError(
            "close burst profile analysis_basis cannot be blank"
        )
    schema_version = _integer(
        identity.get("schema_version"), "close burst profile schema_version"
    )
    if schema_version < 1:
        raise ProfessionalMontagePolicyError(
            "close burst profile schema_version must be positive"
        )
    path = identity.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ProfessionalMontagePolicyError(
            "close burst profile path must be absolute"
        )
    shot_count = _finite(minimum_shot_count, "close burst minimum shot count")
    duration = _finite(
        minimum_duration_seconds, "close burst minimum duration seconds"
    )
    switch_fraction = _finite(
        minimum_source_switch_fraction,
        "close burst minimum source switch fraction",
    )
    if shot_count <= 0.0 or duration <= 0.0 or not 0.0 < switch_fraction <= 1.0:
        raise ProfessionalMontagePolicyError(
            "close burst event P10 thresholds are outside their valid domains"
        )
    if (
        isinstance(event_reference_count, bool)
        or not isinstance(event_reference_count, int)
        or event_reference_count < 1
    ):
        raise ProfessionalMontagePolicyError(
            "close burst event reference count must be positive"
        )
    return {
        "contract": CLOSE_BURST_POLICY_BINDING_CONTRACT,
        "profile_identity": identity,
        "professional_montage_policy_fingerprint_sha256": (
            policy_fingerprint_sha256
        ),
        "event_reference_count": event_reference_count,
        "per_reference_occurrence_p10": 0.0,
        "per_reference_occurrence_median": 0.0,
        "absence_is_hard_failure": False,
        "metrics": {
            "minimum_shot_count": {
                "professional_montage_policy_member": (
                    "minimum_close_burst_shot_count"
                ),
                "value": shot_count,
                "statistic_path": CLOSE_BURST_SHOT_COUNT_STATISTIC_PATH,
                "statistic": "observed_burst_event_distribution_p10",
                "quantile": REFERENCE_FLOOR_QUANTILE,
            },
            "minimum_duration_seconds": {
                "professional_montage_policy_member": (
                    "minimum_close_burst_duration_seconds"
                ),
                "value": duration,
                "statistic_path": CLOSE_BURST_DURATION_STATISTIC_PATH,
                "statistic": "observed_burst_event_distribution_p10",
                "quantile": REFERENCE_FLOOR_QUANTILE,
            },
            "minimum_source_switch_fraction": {
                "professional_montage_policy_member": (
                    "minimum_close_burst_source_switch_fraction"
                ),
                "value": switch_fraction,
                "statistic_path": (
                    CLOSE_BURST_SOURCE_SWITCH_FRACTION_STATISTIC_PATH
                ),
                "statistic": "observed_burst_event_distribution_p10",
                "quantile": REFERENCE_FLOOR_QUANTILE,
            },
        },
    }


def close_burst_policy_binding_sha256(**kwargs: Any) -> str:
    """Hash :func:`close_burst_policy_binding_payload` canonically."""

    return hashlib.sha256(
        _canonical_bytes(close_burst_policy_binding_payload(**kwargs))
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfileStatisticProvenance:
    profile_identity: ProfessionalMontageProfileIdentity
    statistic_path: str
    statistic: str
    quantile: float
    reference_floor_basis: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_identity": self.profile_identity.to_dict(),
            "statistic_path": self.statistic_path,
            "statistic": self.statistic,
            "quantile": self.quantile,
            "reference_floor_basis": self.reference_floor_basis,
        }


@dataclass(frozen=True, slots=True)
class ProfileDerivedMinimum:
    name: str
    value: float
    provenance: ProfileStatisticProvenance
    # Event distributions have fewer observations than the twenty-reference
    # song corpus.  Preserve that sparse count explicitly so downstream code
    # cannot mistake an event P10 for a per-reference occurrence floor.
    reference_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance.to_dict(),
        }
        if self.reference_count is not None:
            payload["reference_count"] = self.reference_count
        return payload


@dataclass(frozen=True, slots=True)
class ProfileDerivedPreferredTarget:
    """A descriptive human-reference target that is never a hard gate."""

    name: str
    value: float
    reference_count: int
    provenance: ProfileStatisticProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "reference_count": self.reference_count,
            "hard_gate": False,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProfileInformedProductMinimum:
    """A hard product requirement informed by, but not named a reference floor."""

    name: str
    value: float
    reference_count: int
    provenance: ProfileStatisticProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "reference_count": self.reference_count,
            "hard_gate": True,
            "is_reference_floor": False,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProfileInformedProductMaximum:
    """A hard upper product bound backed by observed human evidence."""

    name: str
    value: float
    reference_count: int
    provenance: ProfileStatisticProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "reference_count": self.reference_count,
            "hard_gate": True,
            "comparison": "less_than_or_equal",
            "is_reference_floor": False,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProfileDerivedPreferredMaximum:
    """A descriptive human upper target that is never itself a hard gate."""

    name: str
    value: float
    reference_count: int
    provenance: ProfileStatisticProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "reference_count": self.reference_count,
            "hard_gate": False,
            "preference": "at_or_below",
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RoleConditionedDurationPolicy:
    """Hash-bound duration policy for one positive-duration editing role.

    A zero P10 is retained as explicit evidence, but represented by ``None``
    for the local hard minimum so callers cannot silently turn absence of a
    positive human floor into an invented quota.  The role-pooled P25 and P50
    values are taken from the profile's validated empirical distributions.
    """

    role: str
    reference_count: int
    local_minimum_close_duration_fraction: ProfileDerivedMinimum | None
    local_minimum_nonwide_duration_fraction: ProfileDerivedMinimum | None
    pooled_product_minimum_close_duration_fraction: (
        ProfileInformedProductMinimum | None
    )
    pooled_product_minimum_nonwide_duration_fraction: (
        ProfileInformedProductMinimum | None
    )
    preferred_close_duration_fraction: ProfileDerivedPreferredTarget
    preferred_nonwide_duration_fraction: ProfileDerivedPreferredTarget
    maximum_consecutive_wide_run_seconds: ProfileInformedProductMaximum
    preferred_maximum_consecutive_wide_run_seconds: (
        ProfileDerivedPreferredMaximum
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reference_count": self.reference_count,
            "local_minimum_close_duration_fraction": (
                self.local_minimum_close_duration_fraction.to_dict()
                if self.local_minimum_close_duration_fraction is not None
                else None
            ),
            "local_minimum_nonwide_duration_fraction": (
                self.local_minimum_nonwide_duration_fraction.to_dict()
                if self.local_minimum_nonwide_duration_fraction is not None
                else None
            ),
            "pooled_product_minimum_close_duration_fraction": (
                self.pooled_product_minimum_close_duration_fraction.to_dict()
                if self.pooled_product_minimum_close_duration_fraction is not None
                else None
            ),
            "pooled_product_minimum_nonwide_duration_fraction": (
                self.pooled_product_minimum_nonwide_duration_fraction.to_dict()
                if self.pooled_product_minimum_nonwide_duration_fraction is not None
                else None
            ),
            "preferred_close_duration_fraction": (
                self.preferred_close_duration_fraction.to_dict()
            ),
            "preferred_nonwide_duration_fraction": (
                self.preferred_nonwide_duration_fraction.to_dict()
            ),
            "maximum_consecutive_wide_run_seconds": (
                self.maximum_consecutive_wide_run_seconds.to_dict()
            ),
            "preferred_maximum_consecutive_wide_run_seconds": (
                self.preferred_maximum_consecutive_wide_run_seconds.to_dict()
            ),
            "zero_local_p10_is_omitted_not_promoted": True,
        }


@dataclass(frozen=True, slots=True)
class UnusableReferenceFloor:
    statistic_path: str
    value: float
    reason: str
    provenance: ProfileStatisticProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic_path": self.statistic_path,
            "value": self.value,
            "usable_as_positive_production_minimum": False,
            "reason": self.reason,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProductEditorialRequirements:
    requirement_basis: str = PRODUCT_EDITORIAL_REQUIREMENT_BASIS
    # The observed non-wide multi-source burst median is 2.5 shots.  Requiring
    # two internal boundaries makes the product gate require a three-shot
    # burst instead of letting one two-shot exchange make a flat chorus pass.
    minimum_nonwide_multi_family_burst_count: int = 2
    minimum_post_burst_formation_evidenced_wide_reset_count: int = 1
    formation_evidence_threshold_normalized: float = (
        FORMATION_EVIDENCE_THRESHOLD_NORMALIZED
    )
    formation_evidence_semantics: str = FORMATION_PERSON_COUNT_SEMANTICS
    formation_evidence_label: str = WEAK_FORMATION_EVIDENCE_LABEL
    formation_evidence_proves_full_group_formation: bool = False
    missing_or_zero_formation_evidence_passes: bool = False
    is_reference_floor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_basis": self.requirement_basis,
            "minimum_nonwide_multi_family_burst_count": (
                self.minimum_nonwide_multi_family_burst_count
            ),
            "minimum_post_burst_formation_evidenced_wide_reset_count": (
                self.minimum_post_burst_formation_evidenced_wide_reset_count
            ),
            "formation_evidence_threshold_normalized": (
                self.formation_evidence_threshold_normalized
            ),
            "formation_evidence_semantics": self.formation_evidence_semantics,
            "formation_evidence_label": self.formation_evidence_label,
            "formation_evidence_proves_full_group_formation": (
                self.formation_evidence_proves_full_group_formation
            ),
            "missing_or_zero_formation_evidence_passes": (
                self.missing_or_zero_formation_evidence_passes
            ),
            "is_reference_floor": self.is_reference_floor,
        }


@dataclass(frozen=True, slots=True)
class ProfessionalMontagePolicy:
    identity: ProfessionalMontageProfileIdentity
    minimum_close_duration_fraction: ProfileDerivedMinimum
    minimum_nonwide_duration_fraction: ProfileDerivedMinimum
    highlight_pooled_product_minimum_close_duration_fraction: (
        ProfileInformedProductMinimum
    )
    highlight_pooled_product_minimum_nonwide_duration_fraction: (
        ProfileInformedProductMinimum
    )
    preferred_close_duration_fraction: ProfileDerivedPreferredTarget
    preferred_nonwide_duration_fraction: ProfileDerivedPreferredTarget
    maximum_consecutive_wide_run_seconds: ProfileInformedProductMaximum
    preferred_maximum_consecutive_wide_run_seconds: (
        ProfileDerivedPreferredMaximum
    )
    role_conditioned_duration_policies: tuple[
        RoleConditionedDurationPolicy, ...
    ]
    minimum_early_wide_anchor_fraction: ProfileDerivedMinimum
    minimum_phrase_entry_wide_anchor_fraction: ProfileDerivedMinimum
    minimum_wide_involved_source_switch_fraction: ProfileDerivedMinimum
    preferred_wide_involved_source_switch_fraction: (
        ProfileDerivedPreferredTarget
    )
    highlight_product_minimum_source_switches_per_minute: (
        ProfileInformedProductMinimum
    )
    highlight_preferred_source_switches_per_minute: (
        ProfileDerivedPreferredTarget
    )
    highlight_cadence_reference_count: int
    minimum_nonwide_burst_shot_count: ProfileDerivedMinimum
    minimum_nonwide_burst_duration_seconds: ProfileDerivedMinimum
    minimum_nonwide_burst_source_switch_fraction: ProfileDerivedMinimum
    minimum_close_burst_shot_count: ProfileDerivedMinimum
    minimum_close_burst_duration_seconds: ProfileDerivedMinimum
    minimum_close_burst_source_switch_fraction: ProfileDerivedMinimum
    nonwide_duration_fraction_policy: str
    unusable_zero_reference_floors: tuple[UnusableReferenceFloor, ...]
    product_editorial_requirements: ProductEditorialRequirements

    @property
    def profile_identity(self) -> dict[str, Any]:
        return self.identity.to_dict()

    def to_dict(self) -> dict[str, Any]:
        threshold_names = (
            "minimum_close_duration_fraction",
            "minimum_nonwide_duration_fraction",
            "minimum_early_wide_anchor_fraction",
            "minimum_phrase_entry_wide_anchor_fraction",
            "minimum_wide_involved_source_switch_fraction",
            "minimum_nonwide_burst_shot_count",
            "minimum_nonwide_burst_duration_seconds",
            "minimum_nonwide_burst_source_switch_fraction",
            "minimum_close_burst_shot_count",
            "minimum_close_burst_duration_seconds",
            "minimum_close_burst_source_switch_fraction",
        )
        return {
            "profile_identity": self.profile_identity,
            "profile_derived_minimums": {
                name: getattr(self, name).to_dict() for name in threshold_names
            },
            "profile_derived_preferred_targets": {
                "preferred_close_duration_fraction": (
                    self.preferred_close_duration_fraction.to_dict()
                ),
                "preferred_nonwide_duration_fraction": (
                    self.preferred_nonwide_duration_fraction.to_dict()
                ),
                "preferred_wide_involved_source_switch_fraction": (
                    self.preferred_wide_involved_source_switch_fraction.to_dict()
                ),
                "highlight_preferred_source_switches_per_minute": (
                    self.highlight_preferred_source_switches_per_minute.to_dict()
                ),
            },
            "profile_informed_product_minimums": {
                "highlight_pooled_product_minimum_close_duration_fraction": (
                    self.highlight_pooled_product_minimum_close_duration_fraction.to_dict()
                ),
                "highlight_pooled_product_minimum_nonwide_duration_fraction": (
                    self.highlight_pooled_product_minimum_nonwide_duration_fraction.to_dict()
                ),
                "highlight_product_minimum_source_switches_per_minute": (
                    self.highlight_product_minimum_source_switches_per_minute.to_dict()
                ),
            },
            "profile_informed_product_maximums": {
                "maximum_consecutive_wide_run_seconds": (
                    self.maximum_consecutive_wide_run_seconds.to_dict()
                ),
            },
            "profile_derived_preferred_maximums": {
                "preferred_maximum_consecutive_wide_run_seconds": (
                    self.preferred_maximum_consecutive_wide_run_seconds.to_dict()
                ),
            },
            "highlight_cadence_reference_count": (
                self.highlight_cadence_reference_count
            ),
            "highlight_cadence_evidence_contract": (
                HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT
            ),
            "highlight_duration_evidence_contract": (
                HIGHLIGHT_DURATION_EVIDENCE_CONTRACT
            ),
            "highlight_wide_run_evidence_contract": (
                HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT
            ),
            "role_conditioned_duration_evidence_contract": (
                ROLE_CONDITIONED_DURATION_EVIDENCE_CONTRACT
            ),
            "role_conditioned_duration_policies": {
                item.role: item.to_dict()
                for item in self.role_conditioned_duration_policies
            },
            "count_based_audit_minimums": {
                "minimum_nonwide_burst_shot_count": (
                    self.minimum_nonwide_burst_shot_count.to_dict()
                ),
                "minimum_close_burst_shot_count": (
                    self.minimum_close_burst_shot_count.to_dict()
                ),
                "never_used_as_duration_policy": True,
            },
            "nonwide_duration_fraction_policy": (
                self.nonwide_duration_fraction_policy
            ),
            "unusable_zero_reference_floors": [
                item.to_dict() for item in self.unusable_zero_reference_floors
            ],
            "product_editorial_requirements": (
                self.product_editorial_requirements.to_dict()
            ),
        }


def _validate_profile_contract(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(profile) != _TOP_LEVEL_KEYS:
        raise ProfessionalMontagePolicyError(
            "montage profile top-level schema mismatch"
        )
    if _integer(profile.get("schema_version"), "schema_version") != (
        PROFILE_SCHEMA_VERSION
    ):
        raise ProfessionalMontagePolicyError("unsupported profile schema version")
    if profile.get("analysis_basis") != PROFILE_ANALYSIS_BASIS:
        raise ProfessionalMontagePolicyError("unsupported profile analysis basis")

    provenance = _mapping(profile.get("provenance"), "profile provenance")
    if provenance.get("content_hash_algorithm") != PROFILE_CONTENT_HASH_ALGORITHM:
        raise ProfessionalMontagePolicyError(
            "unsupported profile content hash algorithm"
        )
    generator = _mapping(provenance.get("generator"), "profile generator")
    if set(generator) != {"module", "module_version"} or generator != {
        "module": PROFILE_GENERATOR_MODULE,
        "module_version": MODULE_VERSION,
    }:
        raise ProfessionalMontagePolicyError("unsupported profile generator")

    configuration = _mapping(profile.get("configuration"), "configuration")
    if not math.isclose(
        _finite(configuration.get("reference_floor_quantile"), "floor quantile"),
        REFERENCE_FLOOR_QUANTILE,
        abs_tol=1e-12,
    ):
        raise ProfessionalMontagePolicyError(
            "profile reference floor quantile must be p10"
        )
    distributional = _mapping(
        profile.get("distributional_policy"), "distributional policy"
    )
    if distributional.get("hand_authored_visual_quality_thresholds_present") is not False:
        raise ProfessionalMontagePolicyError(
            "profile unexpectedly contains hand-authored quality thresholds"
        )
    if distributional.get("reference_floors_are_lower_distribution_quantiles") is not True:
        raise ProfessionalMontagePolicyError("profile reference-floor semantics mismatch")
    required_duration_semantics = {
        "shot_size_count_fractions_are_count_only_audit_metrics": True,
        "shot_size_duration_fractions_use_observed_duration_only": True,
        "nonwide_duration_is_explicit_medium_plus_close_never_one_minus_wide": True,
        "consecutive_wide_runs_are_maximal_contiguous_known_wide_shots_within_role": True,
        "single_wide_shot_is_a_consecutive_wide_run": True,
        "reference_maximum_wide_run_is_zero_when_no_wide_is_observed": True,
    }
    if any(
        distributional.get(name) is not expected
        for name, expected in required_duration_semantics.items()
    ):
        raise ProfessionalMontagePolicyError(
            "profile duration and consecutive-wide-run semantics mismatch"
        )

    scope = _mapping(profile.get("scope"), "profile scope")
    if scope.get("artist") != "IVE":
        raise ProfessionalMontagePolicyError("profile is not scoped to IVE")
    if _integer(scope.get("song_count"), "scope song_count") != EXPECTED_SONG_COUNT:
        raise ProfessionalMontagePolicyError("profile must contain exactly four songs")
    if _integer(scope.get("reference_count"), "scope reference_count") != (
        EXPECTED_REFERENCE_COUNT
    ):
        raise ProfessionalMontagePolicyError(
            "profile must contain exactly twenty references"
        )
    songs = _array(profile.get("songs"), "songs")
    if len(songs) != EXPECTED_SONG_COUNT:
        raise ProfessionalMontagePolicyError("profile song payload count mismatch")
    song_ids: set[str] = set()
    reference_count = 0
    for index, raw_song in enumerate(songs):
        song = _mapping(raw_song, f"song {index}")
        song_id = song.get("song_id")
        if not isinstance(song_id, str) or not song_id or song_id in song_ids:
            raise ProfessionalMontagePolicyError("profile song ids are invalid")
        song_ids.add(song_id)
        references = _array(song.get("references"), f"song {index} references")
        if len(references) != 5:
            raise ProfessionalMontagePolicyError(
                "each of the four songs must contribute five references"
            )
        reference_count += len(references)
    if reference_count != EXPECTED_REFERENCE_COUNT:
        raise ProfessionalMontagePolicyError("profile reference payload count mismatch")

    roles = _mapping(
        profile.get("aggregate_by_acoustic_role"),
        "aggregate_by_acoustic_role",
    )
    if set(roles) != set(ROLE_NAMES):
        raise ProfessionalMontagePolicyError("acoustic-role aggregate schema mismatch")
    chorus = _mapping(roles.get("chorus_like"), "chorus_like aggregate")
    if set(chorus) != _CHORUS_KEYS:
        raise ProfessionalMontagePolicyError("chorus_like aggregate schema mismatch")
    if _integer(
        chorus.get("contributing_reference_count"),
        "chorus contributing_reference_count",
    ) != EXPECTED_REFERENCE_COUNT:
        raise ProfessionalMontagePolicyError(
            "chorus_like must have twenty contributing references"
        )
    person_evidence = _mapping(
        chorus.get("wide_shot_person_count_evidence"),
        "wide-shot person-count evidence",
    )
    if person_evidence.get("semantics") != FORMATION_PERSON_COUNT_SEMANTICS:
        raise ProfessionalMontagePolicyError(
            "formation person-count evidence semantics mismatch"
        )
    limitations = _mapping(
        profile.get("evidence_limitations"), "evidence limitations"
    )
    if (
        limitations.get("generic_wide_is_proven_group_formation") is not False
        or limitations.get("person_count_threshold_used_to_assert_group_formation")
        is not False
        or limitations.get("formation_identity_or_member_labels_present") is not False
    ):
        raise ProfessionalMontagePolicyError(
            "profile incorrectly claims proof of group formation"
        )
    return chorus


def _profile_identity(
    *,
    source: Path,
    raw: bytes,
    profile: Mapping[str, Any],
    expected_file_sha256: str | None,
) -> ProfessionalMontageProfileIdentity:
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None:
        expected = _sha256(expected_file_sha256, "expected profile file SHA-256")
        if file_sha256 != expected:
            raise ProfessionalMontagePolicyError("profile file SHA-256 mismatch")

    provenance = _mapping(profile.get("provenance"), "profile provenance")
    recorded = _sha256(
        provenance.get("profile_content_sha256"),
        "profile content SHA-256",
    )
    unsigned = deepcopy(dict(profile))
    unsigned_provenance = dict(
        _mapping(unsigned.get("provenance"), "profile provenance")
    )
    if "profile_content_sha256" not in unsigned_provenance:
        raise ProfessionalMontagePolicyError(
            "profile content SHA-256 field is missing"
        )
    unsigned_provenance["profile_content_sha256"] = None
    unsigned["provenance"] = unsigned_provenance
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise ProfessionalMontagePolicyError("profile content SHA-256 mismatch")
    return ProfessionalMontageProfileIdentity(
        analysis_basis=PROFILE_ANALYSIS_BASIS,
        schema_version=PROFILE_SCHEMA_VERSION,
        path=str(source),
        file_sha256=file_sha256,
        content_sha256=actual,
    )


def _reference_minimum(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
    *,
    name: str,
    path: str,
    fraction: bool,
) -> ProfileDerivedMinimum:
    distribution = _mapping(_at(profile, path), path)
    p10 = _finite(distribution.get("p10"), f"{path}.p10")
    floor = _mapping(distribution.get("reference_floor"), f"{path}.reference_floor")
    quantile = _finite(floor.get("quantile"), f"{path}.reference_floor.quantile")
    value = _finite(floor.get("value"), f"{path}.reference_floor.value")
    if (
        not math.isclose(quantile, REFERENCE_FLOOR_QUANTILE, abs_tol=1e-12)
        or not math.isclose(value, p10, abs_tol=1e-12)
        or floor.get("basis") != REFERENCE_FLOOR_BASIS
    ):
        raise ProfessionalMontagePolicyError(
            f"{path} is not a profile p10 reference floor"
        )
    if value <= 0.0 or (fraction and value > 1.0):
        raise ProfessionalMontagePolicyError(
            f"{path} cannot supply a positive production minimum"
        )
    return ProfileDerivedMinimum(
        name=name,
        value=value,
        provenance=ProfileStatisticProvenance(
            profile_identity=identity,
            statistic_path=path,
            statistic="profile_reference_floor_p10",
            quantile=quantile,
            reference_floor_basis=REFERENCE_FLOOR_BASIS,
        ),
    )


def _reference_preferred_target(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
    *,
    name: str,
    path: str,
    fraction: bool,
) -> ProfileDerivedPreferredTarget:
    distribution = _mapping(_at(profile, path), path)
    count = _integer(distribution.get("count"), f"{path}.count")
    p10 = _finite(distribution.get("p10"), f"{path}.p10")
    median = _finite(distribution.get("median"), f"{path}.median")
    if (
        count < 1
        or median <= 0.0
        or median + 1.0e-12 < p10
        or (fraction and median > 1.0)
    ):
        raise ProfessionalMontagePolicyError(
            f"{path} cannot supply a positive reference-median target"
        )
    return ProfileDerivedPreferredTarget(
        name=name,
        value=median,
        reference_count=count,
        provenance=ProfileStatisticProvenance(
            profile_identity=identity,
            statistic_path=path,
            statistic="profile_reference_median_soft_preferred_target",
            quantile=REFERENCE_PREFERRED_TARGET_QUANTILE,
            reference_floor_basis=None,
        ),
    )


def _nearest_rank_empirical_quantile(
    values: list[float],
    quantile: float,
) -> float:
    """Return an observed nearest-rank quantile without inventing a sample."""

    if not values:
        raise ProfessionalMontagePolicyError(
            "cadence quantile requires at least one reference"
        )
    if not 0.0 < quantile <= 1.0:
        raise ProfessionalMontagePolicyError(
            "cadence quantile must lie in (0, 1]"
        )
    ordered = sorted(_finite(value, "chorus cadence value") for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(_finite(value, "reference distribution value") for value in values)
    if not ordered:
        raise ProfessionalMontagePolicyError(
            "reference distribution requires at least one value"
        )
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_distribution_against_values(
    distribution: Mapping[str, Any],
    values: list[float],
    *,
    name: str,
    reference_floor: bool,
) -> None:
    expected_keys = {
        "count",
        "minimum",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "maximum",
    }
    if reference_floor:
        expected_keys.add("reference_floor")
    if set(distribution) != expected_keys:
        raise ProfessionalMontagePolicyError(f"{name} distribution schema mismatch")
    if _integer(distribution.get("count"), f"{name}.count") != len(values):
        raise ProfessionalMontagePolicyError(f"{name} distribution count mismatch")
    expected = {
        "minimum": min(values),
        "p10": _linear_quantile(values, 0.10),
        "p25": _linear_quantile(values, 0.25),
        "median": _linear_quantile(values, 0.50),
        "p75": _linear_quantile(values, 0.75),
        "p90": _linear_quantile(values, 0.90),
        "maximum": max(values),
    }
    for statistic, expected_value in expected.items():
        actual = _finite(distribution.get(statistic), f"{name}.{statistic}")
        if not math.isclose(actual, expected_value, abs_tol=1e-10):
            raise ProfessionalMontagePolicyError(
                f"{name} distribution {statistic} disagrees with duration evidence"
            )
    if reference_floor:
        floor = _mapping(
            distribution.get("reference_floor"), f"{name}.reference_floor"
        )
        if set(floor) != {"quantile", "value", "basis"}:
            raise ProfessionalMontagePolicyError(
                f"{name} reference-floor schema mismatch"
            )
        if (
            not math.isclose(
                _finite(floor.get("quantile"), f"{name}.reference_floor.quantile"),
                REFERENCE_FLOOR_QUANTILE,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite(floor.get("value"), f"{name}.reference_floor.value"),
                expected["p10"],
                abs_tol=1e-10,
            )
            or floor.get("basis") != REFERENCE_FLOOR_BASIS
        ):
            raise ProfessionalMontagePolicyError(
                f"{name} reference-floor provenance mismatch"
            )


def _duration_fraction_and_wide_run_evidence(
    profile: Mapping[str, Any],
    chorus: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    """Recompute typed duration evidence from all twenty reference payloads."""

    close_values: list[float] = []
    nonwide_values: list[float] = []
    maximum_wide_run_values: list[float] = []
    wide_run_durations: list[float] = []
    wide_run_shot_counts: list[float] = []
    pooled_role_duration = 0.0
    pooled_observed_duration = 0.0
    pooled_unknown_duration = 0.0
    pooled_size_durations = {name: 0.0 for name in SHOT_SIZE_NAMES}
    pooled_nonwide_duration = 0.0

    for song_index, raw_song in enumerate(_array(profile.get("songs"), "songs")):
        song = _mapping(raw_song, f"song {song_index}")
        for reference_index, raw_reference in enumerate(
            _array(song.get("references"), f"song {song_index} references")
        ):
            label = f"song {song_index} reference {reference_index} chorus_like"
            reference = _mapping(raw_reference, label)
            roles = _mapping(reference.get("by_acoustic_role"), f"{label} roles")
            role = _mapping(roles.get("chorus_like"), label)
            role_duration = _finite(
                role.get("role_duration_seconds"), f"{label} role duration"
            )
            observed_duration = _finite(
                role.get("observed_shot_scale_duration_seconds"),
                f"{label} observed shot-scale duration",
            )
            unknown_duration = _finite(
                role.get("unknown_shot_scale_duration_seconds"),
                f"{label} unknown shot-scale duration",
            )
            if (
                role_duration <= 0.0
                or observed_duration <= 0.0
                or unknown_duration < 0.0
                or not math.isclose(
                    observed_duration + unknown_duration,
                    role_duration,
                    abs_tol=1e-10,
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} has invalid duration coverage"
                )
            raw_size_durations = _mapping(
                role.get("shot_size_duration_seconds"),
                f"{label} shot_size_duration_seconds",
            )
            raw_duration_fractions = _mapping(
                role.get("shot_size_duration_fractions"),
                f"{label} shot_size_duration_fractions",
            )
            if set(raw_size_durations) != set(SHOT_SIZE_NAMES) or set(
                raw_duration_fractions
            ) != set(SHOT_SIZE_NAMES):
                raise ProfessionalMontagePolicyError(
                    f"{label} shot-size duration schema mismatch"
                )
            size_durations = {
                size: _finite(
                    raw_size_durations[size], f"{label} {size} duration"
                )
                for size in SHOT_SIZE_NAMES
            }
            if any(value < 0.0 for value in size_durations.values()) or not math.isclose(
                sum(size_durations.values()), observed_duration, abs_tol=1e-10
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} shot-size durations mismatch observed duration"
                )
            duration_fractions: dict[str, float] = {}
            for size in SHOT_SIZE_NAMES:
                actual = _finite(
                    raw_duration_fractions[size], f"{label} {size} duration fraction"
                )
                expected = size_durations[size] / observed_duration
                if not 0.0 <= actual <= 1.0 or not math.isclose(
                    actual, expected, abs_tol=1e-12
                ):
                    raise ProfessionalMontagePolicyError(
                        f"{label} {size} duration fraction mismatch"
                    )
                duration_fractions[size] = actual
            nonwide_duration = _finite(
                role.get("nonwide_duration_seconds"),
                f"{label} explicit nonwide duration",
            )
            explicit_expected_nonwide = (
                size_durations["medium"] + size_durations["close"]
            )
            nonwide_fraction = _finite(
                role.get("nonwide_duration_fraction"),
                f"{label} explicit nonwide duration fraction",
            )
            if (
                not math.isclose(
                    nonwide_duration, explicit_expected_nonwide, abs_tol=1e-10
                )
                or not 0.0 <= nonwide_fraction <= 1.0
                or not math.isclose(
                    nonwide_fraction,
                    nonwide_duration / observed_duration,
                    abs_tol=1e-12,
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} explicit nonwide duration mismatch"
                )

            run_count = _integer(
                role.get("consecutive_wide_run_count"),
                f"{label} consecutive_wide_run_count",
            )
            run_durations = [
                _finite(value, f"{label} wide-run duration")
                for value in _array(
                    role.get("consecutive_wide_run_durations_seconds"),
                    f"{label} consecutive_wide_run_durations_seconds",
                )
            ]
            run_shot_counts = [
                _integer(value, f"{label} wide-run shot count")
                for value in _array(
                    role.get("consecutive_wide_run_shot_counts"),
                    f"{label} consecutive_wide_run_shot_counts",
                )
            ]
            if (
                run_count < 0
                or run_count != len(run_durations)
                or run_count != len(run_shot_counts)
                or any(value <= 0.0 for value in run_durations)
                or any(value < 1 for value in run_shot_counts)
                or not math.isclose(
                    sum(run_durations), size_durations["wide"], abs_tol=1e-10
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} consecutive wide-run evidence mismatch"
                )
            maximum_wide_run = _finite(
                role.get("maximum_consecutive_wide_run_seconds"),
                f"{label} maximum consecutive wide run",
            )
            expected_maximum = max(run_durations) if run_durations else 0.0
            if maximum_wide_run < 0.0 or not math.isclose(
                maximum_wide_run, expected_maximum, abs_tol=1e-10
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} maximum consecutive wide-run mismatch"
                )

            close_values.append(duration_fractions["close"])
            nonwide_values.append(nonwide_fraction)
            maximum_wide_run_values.append(maximum_wide_run)
            wide_run_durations.extend(run_durations)
            wide_run_shot_counts.extend(float(value) for value in run_shot_counts)
            pooled_role_duration += role_duration
            pooled_observed_duration += observed_duration
            pooled_unknown_duration += unknown_duration
            pooled_nonwide_duration += nonwide_duration
            for size in SHOT_SIZE_NAMES:
                pooled_size_durations[size] += size_durations[size]

    if not (
        len(close_values)
        == len(nonwide_values)
        == len(maximum_wide_run_values)
        == EXPECTED_REFERENCE_COUNT
    ):
        raise ProfessionalMontagePolicyError(
            "duration policy requires exactly twenty chorus references"
        )

    reference_root = _mapping(
        chorus.get("reference_distributions"),
        "chorus reference_distributions",
    )
    duration_distributions = _mapping(
        reference_root.get("shot_size_duration_fractions"),
        "chorus shot_size_duration_fractions",
    )
    if set(duration_distributions) != set(SHOT_SIZE_NAMES):
        raise ProfessionalMontagePolicyError(
            "chorus duration distribution schema mismatch"
        )
    _validate_distribution_against_values(
        _mapping(duration_distributions.get("close"), "close duration distribution"),
        close_values,
        name="chorus close duration fraction",
        reference_floor=True,
    )
    _validate_distribution_against_values(
        _mapping(
            reference_root.get("nonwide_duration_fraction"),
            "nonwide duration distribution",
        ),
        nonwide_values,
        name="chorus nonwide duration fraction",
        reference_floor=True,
    )
    _validate_distribution_against_values(
        _mapping(
            reference_root.get("maximum_consecutive_wide_run_seconds"),
            "maximum wide-run distribution",
        ),
        maximum_wide_run_values,
        name="chorus maximum consecutive wide-run seconds",
        reference_floor=True,
    )

    pooled = _mapping(
        chorus.get("pooled_durations_seconds"), "chorus pooled durations"
    )
    if set(pooled) != {
        "role",
        "observed_shot_scale",
        "unknown_shot_scale",
        "shot_size",
        "nonwide",
    }:
        raise ProfessionalMontagePolicyError(
            "chorus pooled duration schema mismatch"
        )
    expected_pooled_scalars = {
        "role": pooled_role_duration,
        "observed_shot_scale": pooled_observed_duration,
        "unknown_shot_scale": pooled_unknown_duration,
        "nonwide": pooled_nonwide_duration,
    }
    for name, expected in expected_pooled_scalars.items():
        if not math.isclose(
            _finite(pooled.get(name), f"chorus pooled {name}"),
            expected,
            abs_tol=1e-9,
        ):
            raise ProfessionalMontagePolicyError(
                f"chorus pooled {name} duration mismatch"
            )
    pooled_sizes = _mapping(pooled.get("shot_size"), "chorus pooled shot_size")
    if set(pooled_sizes) != set(SHOT_SIZE_NAMES):
        raise ProfessionalMontagePolicyError(
            "chorus pooled shot-size duration schema mismatch"
        )
    for size in SHOT_SIZE_NAMES:
        if not math.isclose(
            _finite(pooled_sizes.get(size), f"chorus pooled {size}"),
            pooled_size_durations[size],
            abs_tol=1e-9,
        ):
            raise ProfessionalMontagePolicyError(
                f"chorus pooled {size} duration mismatch"
            )

    pooled_counts = _mapping(chorus.get("pooled_counts"), "chorus pooled_counts")
    if _integer(
        pooled_counts.get("consecutive_wide_run_count"),
        "chorus pooled consecutive_wide_run_count",
    ) != len(wide_run_durations):
        raise ProfessionalMontagePolicyError(
            "chorus pooled consecutive wide-run count mismatch"
        )
    run_events = _mapping(
        chorus.get("wide_run_event_distributions"),
        "chorus wide_run_event_distributions",
    )
    if set(run_events) != {
        "consecutive_wide_run_shot_count",
        "consecutive_wide_run_duration_seconds",
        "semantics",
    } or run_events.get("semantics") != (
        "maximal_contiguous_known_wide_shots_within_one_"
        "acoustic_role_single_wide_shot_is_a_run"
    ):
        raise ProfessionalMontagePolicyError(
            "chorus consecutive wide-run semantics mismatch"
        )
    if not wide_run_durations:
        raise ProfessionalMontagePolicyError(
            "chorus has no observed consecutive wide-run evidence"
        )
    _validate_distribution_against_values(
        _mapping(
            run_events.get("consecutive_wide_run_duration_seconds"),
            "wide-run duration event distribution",
        ),
        wide_run_durations,
        name="chorus wide-run duration event",
        reference_floor=False,
    )
    _validate_distribution_against_values(
        _mapping(
            run_events.get("consecutive_wide_run_shot_count"),
            "wide-run shot-count event distribution",
        ),
        wide_run_shot_counts,
        name="chorus wide-run shot-count event",
        reference_floor=False,
    )
    return close_values, nonwide_values, maximum_wide_run_values


def _duration_policy_targets(
    identity: ProfessionalMontageProfileIdentity,
    *,
    close_values: list[float],
    nonwide_values: list[float],
) -> tuple[
    ProfileInformedProductMinimum,
    ProfileInformedProductMinimum,
    ProfileDerivedPreferredTarget,
    ProfileDerivedPreferredTarget,
]:
    def minimum(name: str, path: str, values: list[float]) -> ProfileInformedProductMinimum:
        value = _nearest_rank_empirical_quantile(
            values, DURATION_PRODUCT_MINIMUM_QUANTILE
        )
        if not 0.0 < value <= 1.0:
            raise ProfessionalMontagePolicyError(
                f"{name} cannot supply a positive duration product minimum"
            )
        return ProfileInformedProductMinimum(
            name=name,
            value=value,
            reference_count=len(values),
            provenance=ProfileStatisticProvenance(
                profile_identity=identity,
                statistic_path=path,
                statistic=(
                    "pooled_reference_nearest_rank_p25_observed_duration_"
                    "fraction_product_minimum"
                ),
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
                reference_floor_basis=None,
            ),
        )

    def preferred(name: str, path: str, values: list[float]) -> ProfileDerivedPreferredTarget:
        value = _nearest_rank_empirical_quantile(
            values, DURATION_PREFERRED_TARGET_QUANTILE
        )
        if not 0.0 < value <= 1.0:
            raise ProfessionalMontagePolicyError(
                f"{name} cannot supply a positive duration preferred target"
            )
        return ProfileDerivedPreferredTarget(
            name=name,
            value=value,
            reference_count=len(values),
            provenance=ProfileStatisticProvenance(
                profile_identity=identity,
                statistic_path=path,
                statistic=(
                    "pooled_reference_nearest_rank_p50_observed_duration_"
                    "fraction_soft_preferred_target"
                ),
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
                reference_floor_basis=None,
            ),
        )

    return (
        minimum(
            "highlight_pooled_product_minimum_close_duration_fraction",
            HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
            close_values,
        ),
        minimum(
            "highlight_pooled_product_minimum_nonwide_duration_fraction",
            HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
            nonwide_values,
        ),
        preferred(
            "preferred_close_duration_fraction",
            HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
            close_values,
        ),
        preferred(
            "preferred_nonwide_duration_fraction",
            HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
            nonwide_values,
        ),
    )


def _wide_run_policy_targets(
    identity: ProfessionalMontageProfileIdentity,
    values: list[float],
) -> tuple[ProfileInformedProductMaximum, ProfileDerivedPreferredMaximum]:
    hard_maximum = _nearest_rank_empirical_quantile(
        values, WIDE_RUN_HARD_MAXIMUM_QUANTILE
    )
    preferred_maximum = _nearest_rank_empirical_quantile(
        values, WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE
    )
    if hard_maximum <= 0.0 or not 0.0 <= preferred_maximum <= hard_maximum:
        raise ProfessionalMontagePolicyError(
            "wide-run evidence cannot supply ordered positive upper policies"
        )
    return (
        ProfileInformedProductMaximum(
            name="maximum_consecutive_wide_run_seconds",
            value=hard_maximum,
            reference_count=len(values),
            provenance=ProfileStatisticProvenance(
                profile_identity=identity,
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                statistic="observed_human_maximum_consecutive_wide_run_hard_limit",
                quantile=WIDE_RUN_HARD_MAXIMUM_QUANTILE,
                reference_floor_basis=None,
            ),
        ),
        ProfileDerivedPreferredMaximum(
            name="preferred_maximum_consecutive_wide_run_seconds",
            value=preferred_maximum,
            reference_count=len(values),
            provenance=ProfileStatisticProvenance(
                profile_identity=identity,
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                statistic=(
                    "observed_reference_nearest_rank_p75_maximum_consecutive_"
                    "wide_run_soft_upper_target"
                ),
                quantile=WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
                reference_floor_basis=None,
            ),
        ),
    )


def _positive_role_duration_values(
    profile: Mapping[str, Any],
    *,
    role_name: str,
) -> tuple[list[float], list[float], list[float]]:
    """Recompute duration fractions for every positive instance of one role."""

    close_values: list[float] = []
    nonwide_values: list[float] = []
    maximum_wide_run_values: list[float] = []
    for song_index, raw_song in enumerate(_array(profile.get("songs"), "songs")):
        song = _mapping(raw_song, f"song {song_index}")
        for reference_index, raw_reference in enumerate(
            _array(song.get("references"), f"song {song_index} references")
        ):
            label = (
                f"song {song_index} reference {reference_index} {role_name}"
            )
            reference = _mapping(raw_reference, label)
            roles = _mapping(reference.get("by_acoustic_role"), f"{label} roles")
            role = _mapping(roles.get(role_name), label)
            if role.get("role") != role_name:
                raise ProfessionalMontagePolicyError(
                    f"{label} role label mismatch"
                )
            role_duration = _finite(
                role.get("role_duration_seconds"), f"{label} role duration"
            )
            if role_duration < 0.0:
                raise ProfessionalMontagePolicyError(
                    f"{label} role duration cannot be negative"
                )
            if role_duration == 0.0:
                continue

            observed_duration = _finite(
                role.get("observed_shot_scale_duration_seconds"),
                f"{label} observed shot-scale duration",
            )
            unknown_duration = _finite(
                role.get("unknown_shot_scale_duration_seconds"),
                f"{label} unknown shot-scale duration",
            )
            if (
                observed_duration <= 0.0
                or unknown_duration < 0.0
                or not math.isclose(
                    observed_duration + unknown_duration,
                    role_duration,
                    abs_tol=1e-10,
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} has invalid duration coverage"
                )
            raw_durations = _mapping(
                role.get("shot_size_duration_seconds"),
                f"{label} shot-size durations",
            )
            raw_fractions = _mapping(
                role.get("shot_size_duration_fractions"),
                f"{label} shot-size duration fractions",
            )
            if set(raw_durations) != set(SHOT_SIZE_NAMES) or set(
                raw_fractions
            ) != set(SHOT_SIZE_NAMES):
                raise ProfessionalMontagePolicyError(
                    f"{label} shot-size duration schema mismatch"
                )
            durations = {
                name: _finite(raw_durations.get(name), f"{label} {name} duration")
                for name in SHOT_SIZE_NAMES
            }
            if any(value < 0.0 for value in durations.values()) or not math.isclose(
                sum(durations.values()), observed_duration, abs_tol=1e-10
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} shot-size durations mismatch observed duration"
                )
            fractions = {
                name: _finite(raw_fractions.get(name), f"{label} {name} fraction")
                for name in SHOT_SIZE_NAMES
            }
            if any(
                not 0.0 <= fractions[name] <= 1.0
                or not math.isclose(
                    fractions[name],
                    durations[name] / observed_duration,
                    abs_tol=1e-12,
                )
                for name in SHOT_SIZE_NAMES
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} shot-size duration fractions mismatch"
                )
            nonwide_duration = _finite(
                role.get("nonwide_duration_seconds"),
                f"{label} explicit nonwide duration",
            )
            nonwide_fraction = _finite(
                role.get("nonwide_duration_fraction"),
                f"{label} explicit nonwide duration fraction",
            )
            expected_nonwide = durations["medium"] + durations["close"]
            if (
                not math.isclose(
                    nonwide_duration, expected_nonwide, abs_tol=1e-10
                )
                or not 0.0 <= nonwide_fraction <= 1.0
                or not math.isclose(
                    nonwide_fraction,
                    nonwide_duration / observed_duration,
                    abs_tol=1e-12,
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} explicit nonwide duration mismatch"
                )
            run_count = _integer(
                role.get("consecutive_wide_run_count"),
                f"{label} consecutive wide-run count",
            )
            run_durations = [
                _finite(value, f"{label} consecutive wide-run duration")
                for value in _array(
                    role.get("consecutive_wide_run_durations_seconds"),
                    f"{label} consecutive wide-run durations",
                )
            ]
            if (
                run_count != len(run_durations)
                or any(value <= 0.0 for value in run_durations)
                or not math.isclose(
                    sum(run_durations), durations["wide"], abs_tol=1e-10
                )
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} consecutive wide-run evidence mismatch"
                )
            maximum_wide_run = _finite(
                role.get("maximum_consecutive_wide_run_seconds"),
                f"{label} maximum consecutive wide run",
            )
            if not math.isclose(
                maximum_wide_run,
                max(run_durations) if run_durations else 0.0,
                abs_tol=1e-10,
            ):
                raise ProfessionalMontagePolicyError(
                    f"{label} maximum consecutive wide-run mismatch"
                )
            close_values.append(fractions["close"])
            nonwide_values.append(nonwide_fraction)
            maximum_wide_run_values.append(maximum_wide_run)
    return close_values, nonwide_values, maximum_wide_run_values


def _role_conditioned_duration_policy(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
    *,
    role_name: str,
) -> RoleConditionedDurationPolicy | None:
    close_values, nonwide_values, wide_run_values = (
        _positive_role_duration_values(profile, role_name=role_name)
    )
    aggregate_roles = _mapping(
        profile.get("aggregate_by_acoustic_role"),
        "aggregate_by_acoustic_role",
    )
    aggregate = _mapping(aggregate_roles.get(role_name), f"{role_name} aggregate")
    if not close_values:
        if aggregate:
            raise ProfessionalMontagePolicyError(
                f"{role_name} aggregate exists without positive-duration evidence"
            )
        return None
    if set(aggregate) != _CHORUS_KEYS:
        raise ProfessionalMontagePolicyError(
            f"{role_name} aggregate schema mismatch"
        )
    reference_count = len(close_values)
    if _integer(
        aggregate.get("contributing_reference_count"),
        f"{role_name} contributing_reference_count",
    ) != reference_count:
        raise ProfessionalMontagePolicyError(
            f"{role_name} contributing reference count mismatch"
        )
    distributions = _mapping(
        aggregate.get("reference_distributions"),
        f"{role_name} reference distributions",
    )
    shot_sizes = _mapping(
        distributions.get("shot_size_duration_fractions"),
        f"{role_name} shot-size duration distributions",
    )
    close_distribution = _mapping(
        shot_sizes.get("close"), f"{role_name} close duration distribution"
    )
    nonwide_distribution = _mapping(
        distributions.get("nonwide_duration_fraction"),
        f"{role_name} nonwide duration distribution",
    )
    wide_run_distribution = _mapping(
        distributions.get("maximum_consecutive_wide_run_seconds"),
        f"{role_name} maximum wide-run distribution",
    )
    _validate_distribution_against_values(
        close_distribution,
        close_values,
        name=f"{role_name} close duration fraction",
        reference_floor=True,
    )
    _validate_distribution_against_values(
        nonwide_distribution,
        nonwide_values,
        name=f"{role_name} nonwide duration fraction",
        reference_floor=True,
    )
    _validate_distribution_against_values(
        wide_run_distribution,
        wide_run_values,
        name=f"{role_name} maximum consecutive wide-run seconds",
        reference_floor=True,
    )

    close_path = (
        f"aggregate_by_acoustic_role.{role_name}.reference_distributions."
        "shot_size_duration_fractions.close"
    )
    nonwide_path = (
        f"aggregate_by_acoustic_role.{role_name}.reference_distributions."
        "nonwide_duration_fraction"
    )
    wide_path = (
        f"aggregate_by_acoustic_role.{role_name}.reference_distributions."
        "maximum_consecutive_wide_run_seconds"
    )

    def provenance(
        *, path: str, statistic: str, quantile: float, floor: bool = False
    ) -> ProfileStatisticProvenance:
        return ProfileStatisticProvenance(
            profile_identity=identity,
            statistic_path=path,
            statistic=statistic,
            quantile=quantile,
            reference_floor_basis=REFERENCE_FLOOR_BASIS if floor else None,
        )

    def local_minimum(
        name: str, path: str, distribution: Mapping[str, Any]
    ) -> ProfileDerivedMinimum | None:
        value = _finite(distribution.get("p10"), f"{path}.p10")
        if value == 0.0:
            return None
        return ProfileDerivedMinimum(
            name=name,
            value=value,
            provenance=provenance(
                path=path,
                statistic="profile_reference_floor_p10",
                quantile=REFERENCE_FLOOR_QUANTILE,
                floor=True,
            ),
        )

    def product_minimum(
        name: str, path: str, distribution: Mapping[str, Any]
    ) -> ProfileInformedProductMinimum | None:
        value = _finite(distribution.get("p25"), f"{path}.p25")
        if value == 0.0:
            return None
        return ProfileInformedProductMinimum(
            name=name,
            value=value,
            reference_count=reference_count,
            provenance=provenance(
                path=path,
                statistic=(
                    "profile_empirical_linear_p25_role_pooled_duration_"
                    "fraction_product_minimum"
                ),
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
            ),
        )

    def preferred(
        name: str, path: str, distribution: Mapping[str, Any]
    ) -> ProfileDerivedPreferredTarget:
        return ProfileDerivedPreferredTarget(
            name=name,
            value=_finite(distribution.get("median"), f"{path}.median"),
            reference_count=reference_count,
            provenance=provenance(
                path=path,
                statistic=(
                    "profile_empirical_linear_p50_role_pooled_duration_"
                    "fraction_soft_preferred_target"
                ),
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
            ),
        )

    return RoleConditionedDurationPolicy(
        role=role_name,
        reference_count=reference_count,
        local_minimum_close_duration_fraction=local_minimum(
            f"{role_name}_minimum_close_duration_fraction",
            close_path,
            close_distribution,
        ),
        local_minimum_nonwide_duration_fraction=local_minimum(
            f"{role_name}_minimum_nonwide_duration_fraction",
            nonwide_path,
            nonwide_distribution,
        ),
        pooled_product_minimum_close_duration_fraction=product_minimum(
            f"{role_name}_pooled_product_minimum_close_duration_fraction",
            close_path,
            close_distribution,
        ),
        pooled_product_minimum_nonwide_duration_fraction=product_minimum(
            f"{role_name}_pooled_product_minimum_nonwide_duration_fraction",
            nonwide_path,
            nonwide_distribution,
        ),
        preferred_close_duration_fraction=preferred(
            f"{role_name}_preferred_close_duration_fraction",
            close_path,
            close_distribution,
        ),
        preferred_nonwide_duration_fraction=preferred(
            f"{role_name}_preferred_nonwide_duration_fraction",
            nonwide_path,
            nonwide_distribution,
        ),
        maximum_consecutive_wide_run_seconds=ProfileInformedProductMaximum(
            name=f"{role_name}_maximum_consecutive_wide_run_seconds",
            value=_finite(wide_run_distribution.get("maximum"), f"{wide_path}.maximum"),
            reference_count=reference_count,
            provenance=provenance(
                path=wide_path,
                statistic="observed_human_role_p100_maximum_wide_run_hard_limit",
                quantile=WIDE_RUN_HARD_MAXIMUM_QUANTILE,
            ),
        ),
        preferred_maximum_consecutive_wide_run_seconds=(
            ProfileDerivedPreferredMaximum(
                name=(
                    f"{role_name}_preferred_maximum_consecutive_wide_run_seconds"
                ),
                value=_finite(wide_run_distribution.get("p75"), f"{wide_path}.p75"),
                reference_count=reference_count,
                provenance=provenance(
                    path=wide_path,
                    statistic=(
                        "profile_empirical_linear_p75_role_maximum_wide_run_"
                        "soft_upper_target"
                    ),
                    quantile=WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
                ),
            )
        ),
    )


def _role_conditioned_duration_policies(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
) -> tuple[RoleConditionedDurationPolicy, ...]:
    policies = tuple(
        policy
        for role_name in ROLE_CONDITIONED_DURATION_ROLES
        if (
            policy := _role_conditioned_duration_policy(
                profile,
                identity,
                role_name=role_name,
            )
        )
        is not None
    )
    if not any(item.role == "chorus_like" for item in policies):
        raise ProfessionalMontagePolicyError(
            "role-conditioned duration policy is missing chorus evidence"
        )
    return policies


def _derived_highlight_cadence(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
    chorus: Mapping[str, Any],
) -> tuple[
    ProfileInformedProductMinimum,
    ProfileDerivedPreferredTarget,
    int,
]:
    """Derive IVE chorus cut cadence from each immutable reference payload."""

    rates: list[float] = []
    reference_ids: set[str] = set()
    derived_shot_count = 0
    songs = _array(profile.get("songs"), "songs")
    for song_index, raw_song in enumerate(songs):
        song = _mapping(raw_song, f"song {song_index}")
        references = _array(song.get("references"), f"song {song_index} references")
        if _integer(
            song.get("reference_count"),
            f"song {song_index} reference_count",
        ) != len(references):
            raise ProfessionalMontagePolicyError(
                f"song {song_index} reference_count disagrees with references"
            )
        for reference_index, raw_reference in enumerate(references):
            label = f"song {song_index} reference {reference_index}"
            reference = _mapping(raw_reference, label)
            reference_id = reference.get("reference_id")
            if (
                not isinstance(reference_id, str)
                or not reference_id
                or reference_id in reference_ids
            ):
                raise ProfessionalMontagePolicyError(
                    "profile reference ids are invalid"
                )
            reference_ids.add(reference_id)
            roles = _mapping(reference.get("by_acoustic_role"), f"{label} roles")
            if set(roles) != set(ROLE_NAMES):
                raise ProfessionalMontagePolicyError(
                    f"{label} acoustic-role schema mismatch"
                )
            role = _mapping(roles.get("chorus_like"), f"{label} chorus_like")
            if role.get("role") != "chorus_like":
                raise ProfessionalMontagePolicyError(
                    f"{label} chorus_like role label mismatch"
                )
            shot_count = _integer(
                role.get("shot_count"), f"{label} chorus_like shot_count"
            )
            duration = _finite(
                role.get("role_duration_seconds"),
                f"{label} chorus_like role_duration_seconds",
            )
            if shot_count < 1 or duration <= 0.0:
                raise ProfessionalMontagePolicyError(
                    f"{label} cannot supply chorus cadence evidence"
                )
            rate = 60.0 * float(shot_count - 1) / duration
            if not math.isfinite(rate) or rate < 0.0:
                raise ProfessionalMontagePolicyError(
                    f"{label} has invalid chorus cadence"
                )
            rates.append(rate)
            derived_shot_count += shot_count

    if len(rates) != EXPECTED_REFERENCE_COUNT:
        raise ProfessionalMontagePolicyError(
            "chorus cadence must contain exactly twenty references"
        )
    pooled = _mapping(chorus.get("pooled_counts"), "chorus pooled_counts")
    if _integer(pooled.get("shot_count"), "chorus pooled shot_count") != (
        derived_shot_count
    ):
        raise ProfessionalMontagePolicyError(
            "chorus pooled shot_count disagrees with reference cadence evidence"
        )

    product_minimum = _nearest_rank_empirical_quantile(
        rates,
        HIGHLIGHT_CADENCE_PRODUCT_MINIMUM_QUANTILE,
    )
    preferred_target = _nearest_rank_empirical_quantile(
        rates,
        HIGHLIGHT_CADENCE_PREFERRED_TARGET_QUANTILE,
    )
    if product_minimum <= 0.0 or preferred_target < product_minimum:
        raise ProfessionalMontagePolicyError(
            "chorus cadence cannot supply ordered positive product targets"
        )

    minimum_provenance = ProfileStatisticProvenance(
        profile_identity=identity,
        statistic_path=HIGHLIGHT_CADENCE_STATISTIC_PATH,
        statistic=(
            "derived_reference_lower_median_chorus_shot_boundary_rate_"
            "product_minimum"
        ),
        quantile=HIGHLIGHT_CADENCE_PRODUCT_MINIMUM_QUANTILE,
        reference_floor_basis=None,
    )
    preferred_provenance = ProfileStatisticProvenance(
        profile_identity=identity,
        statistic_path=HIGHLIGHT_CADENCE_STATISTIC_PATH,
        statistic=(
            "derived_reference_nearest_rank_p75_chorus_shot_boundary_rate_"
            "soft_preferred_target"
        ),
        quantile=HIGHLIGHT_CADENCE_PREFERRED_TARGET_QUANTILE,
        reference_floor_basis=None,
    )
    return (
        ProfileInformedProductMinimum(
            name="highlight_product_minimum_source_switches_per_minute",
            value=product_minimum,
            reference_count=len(rates),
            provenance=minimum_provenance,
        ),
        ProfileDerivedPreferredTarget(
            name="highlight_preferred_source_switches_per_minute",
            value=preferred_target,
            reference_count=len(rates),
            provenance=preferred_provenance,
        ),
        len(rates),
    )


def _event_minimum(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
    *,
    name: str,
    path: str,
    fraction: bool = False,
) -> ProfileDerivedMinimum:
    distribution = _mapping(_at(profile, path), path)
    count = _integer(distribution.get("count"), f"{path}.count")
    value = _finite(distribution.get("p10"), f"{path}.p10")
    if count < 1 or value <= 0.0 or (fraction and value > 1.0):
        raise ProfessionalMontagePolicyError(
            f"{path} cannot supply a positive event p10"
        )
    return ProfileDerivedMinimum(
        name=name,
        value=value,
        provenance=ProfileStatisticProvenance(
            profile_identity=identity,
            statistic_path=path,
            statistic="observed_burst_event_distribution_p10",
            quantile=REFERENCE_FLOOR_QUANTILE,
            reference_floor_basis=None,
        ),
        reference_count=count,
    )


def _collect_zero_reference_floors(
    profile: Mapping[str, Any],
    identity: ProfessionalMontageProfileIdentity,
) -> tuple[UnusableReferenceFloor, ...]:
    found: list[UnusableReferenceFloor] = []

    def visit(value: Mapping[str, Any], path: str) -> None:
        if "reference_floor" in value:
            floor = _mapping(value.get("reference_floor"), f"{path}.reference_floor")
            quantile = _finite(
                floor.get("quantile"), f"{path}.reference_floor.quantile"
            )
            floor_value = _finite(
                floor.get("value"), f"{path}.reference_floor.value"
            )
            p10 = _finite(value.get("p10"), f"{path}.p10")
            if (
                not math.isclose(quantile, REFERENCE_FLOOR_QUANTILE, abs_tol=1e-12)
                or not math.isclose(floor_value, p10, abs_tol=1e-12)
                or floor.get("basis") != REFERENCE_FLOOR_BASIS
            ):
                raise ProfessionalMontagePolicyError(
                    f"{path} has invalid reference-floor provenance"
                )
            if floor_value == 0.0:
                provenance = ProfileStatisticProvenance(
                    profile_identity=identity,
                    statistic_path=path,
                    statistic="profile_reference_floor_p10",
                    quantile=quantile,
                    reference_floor_basis=REFERENCE_FLOOR_BASIS,
                )
                found.append(
                    UnusableReferenceFloor(
                        statistic_path=path,
                        value=0.0,
                        reason=(
                            "zero_reference_floor_cannot_supply_a_positive_"
                            "production_minimum"
                        ),
                        provenance=provenance,
                    )
                )
            return
        for key, child in value.items():
            if isinstance(child, Mapping):
                visit(child, f"{path}.{key}")

    for role_name in ROLE_CONDITIONED_DURATION_ROLES:
        root_path = (
            f"aggregate_by_acoustic_role.{role_name}.reference_distributions"
        )
        root_value = _at(profile, root_path)
        if root_value is None:
            continue
        root = _mapping(root_value, root_path)
        if root:
            visit(root, root_path)
    if not found:
        raise ProfessionalMontagePolicyError(
            "profile has no explicit zero reference floors to mark unusable"
        )
    return tuple(sorted(found, key=lambda item: item.statistic_path))


def load_professional_montage_policy(
    profile_path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> ProfessionalMontagePolicy:
    """Load and strictly verify a production montage policy.

    ``expected_file_sha256`` binds the raw JSON serialization when supplied.
    The profile's own canonical content hash is always verified.
    """

    source = Path(profile_path)
    raw = _stable_json_bytes(source)
    profile = _decode_profile(raw)
    identity = _profile_identity(
        source=source,
        raw=raw,
        profile=profile,
        expected_file_sha256=expected_file_sha256,
    )
    chorus = _validate_profile_contract(profile)
    (
        close_duration_values,
        nonwide_duration_values,
        maximum_wide_run_values,
    ) = _duration_fraction_and_wide_run_evidence(profile, chorus)
    role_conditioned_duration_policies = _role_conditioned_duration_policies(
        profile,
        identity,
    )

    reference_root = (
        "aggregate_by_acoustic_role.chorus_like.reference_distributions"
    )
    event_root = (
        "aggregate_by_acoustic_role.chorus_like.burst_event_distributions"
    )
    pooled = _mapping(chorus.get("pooled_counts"), "chorus pooled_counts")
    events = _mapping(
        chorus.get("burst_event_distributions"),
        "chorus burst event distributions",
    )
    for pooled_name, event_name in (
        ("nonwide_multi_source_burst_count", "nonwide_multi_source_shot_count"),
        ("closeup_multi_source_burst_count", "closeup_multi_source_shot_count"),
    ):
        pooled_count = _integer(pooled.get(pooled_name), pooled_name)
        event_count = _integer(
            _mapping(events.get(event_name), event_name).get("count"),
            f"{event_name}.count",
        )
        if pooled_count != event_count or pooled_count < 1:
            raise ProfessionalMontagePolicyError(
                f"{pooled_name} disagrees with its event distribution"
            )

    reference_specs = (
        (
            "minimum_close_duration_fraction",
            f"{reference_root}.shot_size_duration_fractions.close",
        ),
        (
            "minimum_nonwide_duration_fraction",
            f"{reference_root}.nonwide_duration_fraction",
        ),
        (
            "minimum_early_wide_anchor_fraction",
            f"{reference_root}.early_wide_anchor_fraction",
        ),
        (
            "minimum_phrase_entry_wide_anchor_fraction",
            f"{reference_root}.phrase_entry_wide_anchor_fraction",
        ),
        (
            "minimum_wide_involved_source_switch_fraction",
            f"{reference_root}.source_switch_with_wide_fraction",
        ),
    )
    reference_minimums = {
        name: _reference_minimum(
            profile,
            identity,
            name=name,
            path=path,
            fraction=True,
        )
        for name, path in reference_specs
    }
    preferred_wide_switch_target = _reference_preferred_target(
        profile,
        identity,
        name="preferred_wide_involved_source_switch_fraction",
        path=f"{reference_root}.source_switch_with_wide_fraction",
        fraction=True,
    )
    (
        product_minimum_close_duration,
        product_minimum_nonwide_duration,
        preferred_close_duration,
        preferred_nonwide_duration,
    ) = _duration_policy_targets(
        identity,
        close_values=close_duration_values,
        nonwide_values=nonwide_duration_values,
    )
    if not (
        reference_minimums["minimum_close_duration_fraction"].value
        <= product_minimum_close_duration.value
        <= preferred_close_duration.value
        and reference_minimums["minimum_nonwide_duration_fraction"].value
        <= product_minimum_nonwide_duration.value
        <= preferred_nonwide_duration.value
    ):
        raise ProfessionalMontagePolicyError(
            "duration p10, product p25, and preferred p50 policies are not ordered"
        )
    (
        maximum_consecutive_wide_run,
        preferred_maximum_consecutive_wide_run,
    ) = _wide_run_policy_targets(identity, maximum_wide_run_values)
    (
        highlight_cadence_minimum,
        highlight_cadence_preferred,
        highlight_cadence_reference_count,
    ) = _derived_highlight_cadence(profile, identity, chorus)
    event_specs = (
        (
            "minimum_nonwide_burst_shot_count",
            f"{event_root}.nonwide_multi_source_shot_count",
            False,
        ),
        (
            "minimum_nonwide_burst_duration_seconds",
            f"{event_root}.nonwide_multi_source_duration_seconds",
            False,
        ),
        (
            "minimum_nonwide_burst_source_switch_fraction",
            f"{event_root}.nonwide_multi_source_switch_fraction",
            True,
        ),
        (
            "minimum_close_burst_shot_count",
            f"{event_root}.closeup_multi_source_shot_count",
            False,
        ),
        (
            "minimum_close_burst_duration_seconds",
            f"{event_root}.closeup_multi_source_duration_seconds",
            False,
        ),
        (
            "minimum_close_burst_source_switch_fraction",
            f"{event_root}.closeup_multi_source_switch_fraction",
            True,
        ),
    )
    event_minimums = {
        name: _event_minimum(
            profile,
            identity,
            name=name,
            path=path,
            fraction=fraction,
        )
        for name, path, fraction in event_specs
    }
    return ProfessionalMontagePolicy(
        identity=identity,
        **reference_minimums,
        highlight_pooled_product_minimum_close_duration_fraction=(
            product_minimum_close_duration
        ),
        highlight_pooled_product_minimum_nonwide_duration_fraction=(
            product_minimum_nonwide_duration
        ),
        preferred_close_duration_fraction=preferred_close_duration,
        preferred_nonwide_duration_fraction=preferred_nonwide_duration,
        maximum_consecutive_wide_run_seconds=(
            maximum_consecutive_wide_run
        ),
        preferred_maximum_consecutive_wide_run_seconds=(
            preferred_maximum_consecutive_wide_run
        ),
        role_conditioned_duration_policies=(
            role_conditioned_duration_policies
        ),
        preferred_wide_involved_source_switch_fraction=(
            preferred_wide_switch_target
        ),
        highlight_product_minimum_source_switches_per_minute=(
            highlight_cadence_minimum
        ),
        highlight_preferred_source_switches_per_minute=(
            highlight_cadence_preferred
        ),
        highlight_cadence_reference_count=(
            highlight_cadence_reference_count
        ),
        **event_minimums,
        nonwide_duration_fraction_policy=(
            "explicit_observed_duration_p10_never_derived_from_one_minus_wide"
        ),
        unusable_zero_reference_floors=_collect_zero_reference_floors(
            profile, identity
        ),
        product_editorial_requirements=ProductEditorialRequirements(),
    )
