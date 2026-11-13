"""Versioned ownership contract for StageEdit transition decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


TRANSITION_AUTHORITY_CONTRACT_VERSION = 1
LEARNED_TRANSITION_AUTHORITY = "learned_multiclass_transition_head_v1"
RENDERER_OWNED_TRANSITION_AUTHORITY = (
    "renderer_owned_hard_cut_metadata_evidence_gated_dissolve_v1"
)
RENDERER_OWNED_ALLOWED_MODEL_LABELS = ("hard_cut",)
RENDERER_OWNED_RENDER_KINDS = ("hard_cut", "face_dissolve")
RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON = (
    "renderer_owned_transition_authority_184_of_184_supervised_labels_hard_cut_"
    "zero_provable_non_hard"
)


@dataclass(frozen=True, slots=True)
class StageEditTransitionAuthority:
    mode: str = LEARNED_TRANSITION_AUTHORITY
    contract_version: int = TRANSITION_AUTHORITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != TRANSITION_AUTHORITY_CONTRACT_VERSION:
            raise ValueError("unsupported transition authority contract version")
        if self.mode not in {
            LEARNED_TRANSITION_AUTHORITY,
            RENDERER_OWNED_TRANSITION_AUTHORITY,
        }:
            raise ValueError("unsupported StageEdit transition authority")

    @property
    def renderer_owned(self) -> bool:
        return self.mode == RENDERER_OWNED_TRANSITION_AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "contract_version": self.contract_version,
            "mode": self.mode,
        }
        if not self.renderer_owned:
            return {
                **base,
                "learned_transition_head_authoritative": True,
                "renderer_group_transition_authoritative": False,
            }
        return {
            **base,
            "learned_transition_head_authoritative": False,
            "learned_transition_loss_weight_required": 0.0,
            "learned_transition_score_weight_required": 0.0,
            "model_transition_metadata_labels": list(
                RENDERER_OWNED_ALLOWED_MODEL_LABELS
            ),
            "model_transition_metadata_duration_effect": "none",
            "learned_evaluation_status": "skipped_non_authoritative",
            "learned_evaluation_reason": (
                RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON
            ),
            "renderer_group_transition_authoritative": True,
            "renderer_allowed_kinds": list(RENDERER_OWNED_RENDER_KINDS),
            "face_dissolve_requires_existing_real_evidence_gates": True,
            "zoom_match_whip_pan_render_effects_authorized": False,
            "montage_source_switch_with_wide_floor_preserved": True,
            "wide_anchor_editorial_role": (
                "source_switch_entry_then_nonwide_close_or_medium_burst_not_hold"
            ),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def transition_authority_contract(mode: str) -> dict[str, Any]:
    return StageEditTransitionAuthority(mode=mode).to_dict()


__all__ = [
    "LEARNED_TRANSITION_AUTHORITY",
    "RENDERER_OWNED_ALLOWED_MODEL_LABELS",
    "RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON",
    "RENDERER_OWNED_RENDER_KINDS",
    "RENDERER_OWNED_TRANSITION_AUTHORITY",
    "StageEditTransitionAuthority",
    "TRANSITION_AUTHORITY_CONTRACT_VERSION",
    "transition_authority_contract",
]
