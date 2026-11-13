"""Synthetic current-schema causal-sensing fixtures for unit tests only.

These helpers never claim real media/capacity evidence.  They make existing
unit fixtures structurally satisfy the same strict hash/mask/mode contract as
production so tests can exercise downstream behavior without weakening the
production loaders.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

from crosscut.ive import IVEMember
from crosscut.stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    STAGE_EDIT_CAUSAL_SENSING_BINDING_CONTRACT,
    STAGE_EDIT_CAUSAL_SENSING_MERGE_CONTRACT,
    StageEditCausalSensingBinding,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES,
    STAGE_EDIT_TARGET_MEMBER_SOURCE_FEATURE_NAMES,
    SourceCandidateToken,
    StageEditSequence,
)


def _digest(label: str) -> str:
    return hashlib.sha256(f"stageedit-unit-sensing:{label}".encode()).hexdigest()


def synthetic_production_binding(
    *,
    edit_mode: str = "group",
    target_member: IVEMember | str | None = None,
) -> StageEditCausalSensingBinding:
    normalized = str(edit_mode).strip().lower()
    if normalized == "group":
        target = None
    elif normalized == "member":
        if target_member is None:
            raise ValueError("member test binding requires target_member")
        target = (
            target_member.value
            if isinstance(target_member, IVEMember)
            else IVEMember.parse(str(target_member)).value
        )
    else:
        raise ValueError("test edit_mode must be group or member")
    return StageEditCausalSensingBinding(
        shooting_grammar_content_sha256=_digest("shooting-grammar"),
        member_state_graph_content_sha256=_digest("member-state-graph"),
        ive_performer_state_content_sha256=_digest("ive-performer-state"),
        target_member=target,
    )


def _production_candidate(
    candidate: SourceCandidateToken,
    *,
    edit_mode: str,
) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    index = {name: position for position, name in enumerate(schema.source_feature_names)}
    values = list(candidate.features)
    missing = list(candidate.missing_mask)

    for name in STAGE_EDIT_TARGET_MEMBER_SOURCE_FEATURE_NAMES:
        values[index[name]] = 0.0
        missing[index[name]] = True
    if edit_mode == "member":
        for name, value in (
            ("target_member_probability", 0.0),
            ("target_visibility_probability", 0.0),
            ("target_prominence_probability", 0.0),
            ("target_member_uncertainty_probability", 1.0),
            ("target_member_state_observed", 0.0),
        ):
            values[index[name]] = value
            missing[index[name]] = False

    center = STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES[:7]
    active = STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES[7:14]
    for names in (center, active):
        for name in names:
            values[index[name]] = 1.0 if name.endswith("unknown_probability") else 0.0
            missing[index[name]] = False
    values[index["ive_vocal_probability"]] = 0.0
    missing[index["ive_vocal_probability"]] = True
    for name in (
        "ive_center_member_known_mass",
        "ive_active_performer_known_mass",
    ):
        values[index[name]] = 0.0
        missing[index[name]] = False
    return replace(
        candidate,
        features=tuple(values),
        missing_mask=tuple(missing),
    )


def synthetic_production_snapshot(
    sequence: StageEditSequence,
    *,
    edit_mode: str = "group",
    target_member: IVEMember | str | None = None,
    provenance: StageEditFeatureProvenance | None = None,
) -> StageEditFeatureSnapshot:
    normalized = str(edit_mode).strip().lower()
    binding = synthetic_production_binding(
        edit_mode=normalized,
        target_member=target_member,
    )
    bound_sequence = replace(
        sequence,
        tokens=tuple(
            replace(
                token,
                source_candidates=tuple(
                    _production_candidate(candidate, edit_mode=normalized)
                    for candidate in token.source_candidates
                ),
            )
            for token in sequence.tokens
        ),
    )
    base = provenance or StageEditFeatureProvenance()
    upstream = dict(base.upstream_sha256)
    upstream.update(
        {
            "causal_shooting_grammar": binding.shooting_grammar_content_sha256,
            "member_state_graph": binding.member_state_graph_content_sha256,
            "ive_performer_state": binding.ive_performer_state_content_sha256,
        }
    )
    bound_provenance = replace(
        base,
        upstream_sha256=tuple(upstream.items()),
        contracts=tuple(
            sorted(
                {
                    *base.contracts,
                    STAGE_EDIT_CAUSAL_SENSING_BINDING_CONTRACT,
                    STAGE_EDIT_CAUSAL_SENSING_MERGE_CONTRACT,
                }
            )
        ),
    )
    snapshot = StageEditFeatureSnapshot.build(
        bound_sequence,
        provenance=bound_provenance,
        causal_sensing_binding=binding,
    )
    snapshot.require_production_causal_sensing(
        edit_mode=normalized,
        target_member=target_member,
    )
    return snapshot


def production_gate_statistics(snapshot: StageEditFeatureSnapshot) -> dict[str, Any]:
    mode = (
        "group"
        if snapshot.causal_sensing_binding.target_member is None
        else "member"
    )
    return snapshot.require_production_causal_sensing(
        edit_mode=mode,
        target_member=snapshot.causal_sensing_binding.target_member,
    )


def synthetic_production_gate(
    *,
    edit_mode: str = "group",
    target_member: IVEMember | str | None = None,
) -> dict[str, Any]:
    """Return a canonical unit-only gate without claiming a media snapshot."""

    binding = synthetic_production_binding(
        edit_mode=edit_mode,
        target_member=target_member,
    )
    return {
        **binding.require_production_scope(
            edit_mode=edit_mode,
            target_member=target_member,
        ),
        "feature_snapshot_schema_version": FEATURE_MATERIALIZER_SCHEMA_VERSION,
        "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        "feature_materializer_contract": FEATURE_MATERIALIZER_CONTRACT,
    }


__all__ = [
    "production_gate_statistics",
    "synthetic_production_gate",
    "synthetic_production_binding",
    "synthetic_production_snapshot",
]
