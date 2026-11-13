"""Canonical selector identity bound to exact timeline no-good evidence.

Candidate checkpoints are initially keyed by the expensive selector's base
input identity.  Once fresh boundary evidence rules out exact edges, reusing
that base identity alone would make a constrained optimizer indistinguishable
from the unconstrained run.  This module creates a small, JSON-native identity
that binds the base digest, its semantic basis, and the canonical
``TimelineNoGoodSet`` fingerprint.

The empty no-good set is intentionally represented rather than elided.  It is
therefore auditable and cannot collapse to the pre-existing base digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .timeline_no_good import TIMELINE_NO_GOOD_CONTRACT, TimelineNoGoodSet


NO_GOOD_SELECTOR_IDENTITY_CONTRACT = (
    "base_candidate_input_plus_timeline_no_good_selector_identity_v2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_basis(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("base_identity_basis must be a string")
    if not value or not value.strip():
        raise ValueError("base_identity_basis must be non-blank")
    if value != value.strip():
        raise ValueError(
            "base_identity_basis cannot have leading or trailing whitespace"
        )
    return value


def _canonical_sha256_digest(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("base_candidate_input_identity_sha256 must be a string")
    if _SHA256.fullmatch(value) is None:
        raise ValueError(
            "base_candidate_input_identity_sha256 must be canonical lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class NoGoodSelectorIdentity:
    """Effective selector identity after applying exact edge exclusions."""

    base_candidate_input_identity_sha256: str
    base_identity_basis: str
    timeline_no_goods: TimelineNoGoodSet

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_candidate_input_identity_sha256",
            _canonical_sha256_digest(
                self.base_candidate_input_identity_sha256
            ),
        )
        object.__setattr__(
            self,
            "base_identity_basis",
            _canonical_basis(self.base_identity_basis),
        )
        if not isinstance(self.timeline_no_goods, TimelineNoGoodSet):
            raise TypeError("timeline_no_goods must be a TimelineNoGoodSet")

    @property
    def effective_selector_identity(self) -> dict[str, Any]:
        """Return the canonical JSON-native payload hashed by this identity."""

        return {
            "contract": NO_GOOD_SELECTOR_IDENTITY_CONTRACT,
            "base_identity_basis": self.base_identity_basis,
            "base_candidate_input_identity_sha256": (
                self.base_candidate_input_identity_sha256
            ),
            "timeline_no_goods": {
                "contract": TIMELINE_NO_GOOD_CONTRACT,
                "edge_count": len(self.timeline_no_goods),
                "empty": len(self.timeline_no_goods) == 0,
                "fingerprint": self.timeline_no_goods.fingerprint,
            },
        }

    @property
    def effective_selector_identity_sha256(self) -> str:
        return _canonical_sha256(self.effective_selector_identity)

    def to_dict(self) -> dict[str, Any]:
        """Return the identity payload and its self-verifiable checksum."""

        return {
            "effective_selector_identity": self.effective_selector_identity,
            "timeline_no_goods": self.timeline_no_goods.to_dict(),
            "effective_selector_identity_sha256": (
                self.effective_selector_identity_sha256
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NoGoodSelectorIdentity":
        if not isinstance(payload, Mapping):
            raise TypeError("no-good selector identity payload must be a mapping")
        raw_effective = payload.get("effective_selector_identity")
        if not isinstance(raw_effective, Mapping):
            raise ValueError("effective selector identity payload is invalid")
        timeline_no_goods = TimelineNoGoodSet.from_dict(
            payload.get("timeline_no_goods")
        )
        value = cls(
            base_candidate_input_identity_sha256=raw_effective.get(
                "base_candidate_input_identity_sha256"
            ),
            base_identity_basis=raw_effective.get("base_identity_basis"),
            timeline_no_goods=timeline_no_goods,
        )
        if dict(payload) != value.to_dict():
            raise ValueError(
                "no-good selector identity payload is not canonical"
            )
        return value


def build_no_good_selector_identity(
    *,
    base_candidate_input_identity_sha256: str,
    base_identity_basis: str,
    timeline_no_goods: TimelineNoGoodSet,
) -> NoGoodSelectorIdentity:
    """Build a strictly validated effective selector identity."""

    return NoGoodSelectorIdentity(
        base_candidate_input_identity_sha256=(
            base_candidate_input_identity_sha256
        ),
        base_identity_basis=base_identity_basis,
        timeline_no_goods=timeline_no_goods,
    )
