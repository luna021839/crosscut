"""Exact, reusable decoded-boundary evidence for direct-member edits.

The full member proxy remains the release gate.  This module only lets the
selector avoid re-rendering an identical pair of adjacent shots while it is
searching.  A cache identity binds the source clocks, crop, camera, look,
transition, output geometry, implementation/source binding, and transient
review authority.  Changing any of those inputs produces a different key.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .direct_member_rendered_proxy import DirectMemberRenderedProxyPreflight
from .render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    RenderPlan,
    TransitionKind,
)


BOUNDARY_PROXY_CONTRACT = "direct_member_exact_boundary_proxy_cache_v1"
BOUNDARY_PROXY_CACHE_FILENAME = "direct-member-boundary-proxy-cache-v1.json"
_MAXIMUM_CACHE_ENTRY_COUNT = 512


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validated_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def extract_direct_member_boundary_proxy_plan(
    plan: RenderPlan,
    boundary_index: int,
    output: Path,
) -> RenderPlan:
    """Return the exact two-shot render window around one selected boundary.

    Shot-local source clocks, crop, pan, editorial transform, look, and effect
    metadata stay unchanged.  Only the master timeline is rebased so the two
    shots form a standalone proxy.  The original full proxy is still required
    after every selected boundary has acceptable local evidence.
    """

    if not isinstance(plan, RenderPlan):
        raise TypeError("boundary proxy extraction needs a render plan")
    if (
        isinstance(boundary_index, bool)
        or not isinstance(boundary_index, int)
        or not 0 <= boundary_index < len(plan.segments) - 1
    ):
        raise ValueError("boundary proxy index is outside the render plan")
    destination = Path(output)
    outgoing = plan.segments[boundary_index]
    incoming = plan.segments[boundary_index + 1]
    transition = plan.transitions[boundary_index]
    member_pose_dissolve_count = int(
        transition.kind is TransitionKind.FACE_DISSOLVE
        and transition.reason == DIRECT_MEMBER_POSE_DISSOLVE_REASON
    )
    requested = plan.member_pose_dissolve_requested
    if requested is None:
        raise ValueError("boundary proxy needs complete member effect metadata")
    return RenderPlan(
        sources=plan.sources,
        segments=(
            replace(outgoing, master_start=0.0),
            replace(incoming, master_start=outgoing.duration),
        ),
        reference_source=plan.reference_source,
        aspect_ratio=plan.aspect_ratio,
        output=destination,
        master_audio_start=plan.master_audio_start + outgoing.master_start,
        width=plan.width,
        height=plan.height,
        fps=plan.fps,
        transitions=(transition,),
        member_pose_dissolve_requested=requested,
        member_pose_dissolve_effective=bool(member_pose_dissolve_count),
        member_pose_dissolve_count=member_pose_dissolve_count,
    )


@dataclass(frozen=True, slots=True)
class DirectMemberBoundaryProxySignature:
    signature_sha256: str
    dense_binding_sha256: str
    authority_binding_sha256: str
    window_plan_sha256: str
    outgoing_source_index: int
    incoming_source_index: int
    cut_master_seconds: float
    transition_kind: str
    transition_frame_count: int

    def __post_init__(self) -> None:
        _validated_sha256(self.signature_sha256, "boundary proxy signature")
        _validated_sha256(self.dense_binding_sha256, "dense binding")
        _validated_sha256(self.authority_binding_sha256, "authority binding")
        _validated_sha256(self.window_plan_sha256, "window plan")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.outgoing_source_index,
                self.incoming_source_index,
                self.transition_frame_count,
            )
        ):
            raise ValueError("boundary proxy source or transition index is invalid")
        if not math.isfinite(self.cut_master_seconds):
            raise ValueError("boundary proxy cut time must be finite")
        if not self.transition_kind:
            raise ValueError("boundary proxy transition kind cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_sha256": self.signature_sha256,
            "dense_binding_sha256": self.dense_binding_sha256,
            "authority_binding_sha256": self.authority_binding_sha256,
            "window_plan_sha256": self.window_plan_sha256,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "cut_master_seconds": self.cut_master_seconds,
            "transition_kind": self.transition_kind,
            "transition_frame_count": self.transition_frame_count,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DirectMemberBoundaryProxySignature":
        if not isinstance(payload, Mapping):
            raise TypeError("boundary proxy signature payload must be a mapping")
        return cls(
            signature_sha256=payload.get("signature_sha256"),
            dense_binding_sha256=payload.get("dense_binding_sha256"),
            authority_binding_sha256=payload.get("authority_binding_sha256"),
            window_plan_sha256=payload.get("window_plan_sha256"),
            outgoing_source_index=payload.get("outgoing_source_index"),
            incoming_source_index=payload.get("incoming_source_index"),
            cut_master_seconds=payload.get("cut_master_seconds"),
            transition_kind=payload.get("transition_kind"),
            transition_frame_count=payload.get("transition_frame_count"),
        )


def build_direct_member_boundary_proxy_signature(
    window_plan: RenderPlan,
    *,
    dense_binding_sha256: str,
    authority_binding_sha256: str,
) -> DirectMemberBoundaryProxySignature:
    """Bind one decoded boundary result to every pixel-affecting decision."""

    if not isinstance(window_plan, RenderPlan):
        raise TypeError("boundary proxy signature needs a render plan")
    if len(window_plan.segments) != 2 or len(window_plan.transitions) != 1:
        raise ValueError("boundary proxy signature needs exactly two shots")
    dense_binding = _validated_sha256(
        dense_binding_sha256,
        "dense binding",
    )
    authority_binding = _validated_sha256(
        authority_binding_sha256,
        "authority binding",
    )
    plan_payload = dict(window_plan.to_json())
    # The dense binding already owns ordered source identity.  Filesystem
    # relocation and a fresh temporary output name do not change rendered
    # pixels, so neither belongs in the reusable edge identity.
    plan_payload.pop("sources", None)
    plan_payload.pop("output", None)
    window_plan_sha256 = _sha256(plan_payload)
    identity = {
        "contract": BOUNDARY_PROXY_CONTRACT,
        "dense_binding_sha256": dense_binding,
        "authority_binding_sha256": authority_binding,
        "window_plan_sha256": window_plan_sha256,
    }
    outgoing, incoming = window_plan.segments
    transition = window_plan.transitions[0]
    return DirectMemberBoundaryProxySignature(
        signature_sha256=_sha256(identity),
        dense_binding_sha256=dense_binding,
        authority_binding_sha256=authority_binding,
        window_plan_sha256=window_plan_sha256,
        outgoing_source_index=outgoing.source_index,
        incoming_source_index=incoming.source_index,
        cut_master_seconds=(
            window_plan.master_audio_start + outgoing.duration
        ),
        transition_kind=transition.kind.value,
        transition_frame_count=transition.frame_count,
    )


def _finite_metric(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _maximum_endpoint_metric(
    boundaries: Any,
    field: str,
) -> float | None:
    values: list[float] = []
    if not isinstance(boundaries, list):
        return None
    for boundary in boundaries:
        if not isinstance(boundary, Mapping):
            continue
        steps = boundary.get("endpoint_steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            value = _finite_metric(step.get(field))
            if value is not None:
                values.append(value)
    return max(values) if values else None


@dataclass(frozen=True, slots=True)
class DirectMemberBoundaryProxyEvidence:
    signature: DirectMemberBoundaryProxySignature
    passed: bool
    continuity_passed: bool
    member_quality_passed: bool
    technical_integrity_passed: bool
    finite_solver_penalty: float
    continuity_failure_reasons: tuple[str, ...]
    member_quality_failure_reasons: tuple[str, ...]
    technical_issue_reasons: tuple[str, ...]
    metrics: tuple[tuple[str, float | None], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signature, DirectMemberBoundaryProxySignature):
            raise TypeError("boundary proxy evidence needs a signature")
        if any(
            not isinstance(value, bool)
            for value in (
                self.passed,
                self.continuity_passed,
                self.member_quality_passed,
                self.technical_integrity_passed,
            )
        ):
            raise TypeError("boundary proxy pass flags must be boolean")
        if (
            not math.isfinite(self.finite_solver_penalty)
            or self.finite_solver_penalty < 0.0
        ):
            raise ValueError("boundary proxy solver penalty must be finite")
        if len({name for name, _value in self.metrics}) != len(self.metrics):
            raise ValueError("boundary proxy metric names must be unique")
        for name, value in self.metrics:
            if not name or (
                value is not None and not math.isfinite(float(value))
            ):
                raise ValueError("boundary proxy metric is invalid")

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_dict(),
            "passed": self.passed,
            "continuity_passed": self.continuity_passed,
            "member_quality_passed": self.member_quality_passed,
            "technical_integrity_passed": self.technical_integrity_passed,
            "finite_solver_penalty": self.finite_solver_penalty,
            "continuity_failure_reasons": list(
                self.continuity_failure_reasons
            ),
            "member_quality_failure_reasons": list(
                self.member_quality_failure_reasons
            ),
            "technical_issue_reasons": list(self.technical_issue_reasons),
            "metrics": dict(self.metrics),
            "storage_policy": {
                "raw_frames_persisted": False,
                "face_embeddings_persisted": False,
                "normalized_geometry_and_scalar_quality_only": True,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._unsigned_dict()
        payload["record_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DirectMemberBoundaryProxyEvidence":
        if not isinstance(payload, Mapping):
            raise TypeError("boundary proxy evidence payload must be a mapping")
        record_sha256 = _validated_sha256(
            payload.get("record_sha256"),
            "boundary proxy evidence record",
        )
        unsigned_payload = dict(payload)
        unsigned_payload.pop("record_sha256", None)
        if _sha256(unsigned_payload) != record_sha256:
            raise ValueError("boundary proxy evidence record hash mismatched")
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("boundary proxy evidence metrics are missing")
        if any(
            not isinstance(name, str)
            or not name
            or (
                value is not None
                and _finite_metric(value) is None
            )
            for name, value in raw_metrics.items()
        ):
            raise ValueError("boundary proxy evidence metrics are malformed")

        def strings(name: str) -> tuple[str, ...]:
            values = payload.get(name)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"boundary proxy {name} is malformed")
            return tuple(values)

        return cls(
            signature=DirectMemberBoundaryProxySignature.from_dict(
                payload.get("signature")
            ),
            passed=payload.get("passed"),
            continuity_passed=payload.get("continuity_passed"),
            member_quality_passed=payload.get("member_quality_passed"),
            technical_integrity_passed=payload.get(
                "technical_integrity_passed"
            ),
            finite_solver_penalty=payload.get("finite_solver_penalty"),
            continuity_failure_reasons=strings(
                "continuity_failure_reasons"
            ),
            member_quality_failure_reasons=strings(
                "member_quality_failure_reasons"
            ),
            technical_issue_reasons=strings("technical_issue_reasons"),
            metrics=tuple(
                sorted(
                    (name, _finite_metric(value))
                    for name, value in raw_metrics.items()
                )
            ),
        )


def direct_member_boundary_proxy_evidence_from_preflight(
    signature: DirectMemberBoundaryProxySignature,
    preflight: DirectMemberRenderedProxyPreflight,
) -> DirectMemberBoundaryProxyEvidence:
    """Compact one exact boundary review into finite optimizer evidence."""

    if not isinstance(signature, DirectMemberBoundaryProxySignature):
        raise TypeError("boundary proxy evidence needs a signature")
    if not isinstance(preflight, DirectMemberRenderedProxyPreflight):
        raise TypeError("boundary proxy evidence needs a preflight")
    payload = preflight.payload
    continuity = payload.get("rendered_continuity")
    continuity_payload = continuity if isinstance(continuity, Mapping) else {}
    continuity_reasons = continuity_payload.get("failure_reasons")
    continuity_failure_reasons = tuple(
        str(reason)
        for reason in continuity_reasons
        if isinstance(reason, str) and reason
    ) if isinstance(continuity_reasons, list) else ()
    quality_reasons = payload.get("member_output_quality_failure_reasons")
    member_quality_failure_reasons = tuple(
        str(reason)
        for reason in quality_reasons
        if isinstance(reason, str) and reason
    ) if isinstance(quality_reasons, list) else ()
    generic = payload.get("rendered_proxy_review")
    generic_payload = generic if isinstance(generic, Mapping) else {}
    raw_technical = generic_payload.get(
        "introduced_cross_source_technical_issue_reasons"
    )
    technical_issue_reasons = tuple(
        str(reason)
        for reason in raw_technical
        if isinstance(reason, str) and reason
    ) if isinstance(raw_technical, list) else ()
    metrics_payload = continuity_payload.get("metrics")
    continuity_metrics = (
        metrics_payload if isinstance(metrics_payload, Mapping) else {}
    )
    boundaries = continuity_payload.get("boundaries")
    source_quality = generic_payload.get("source_quality")
    source_quality_payload = (
        source_quality if isinstance(source_quality, Mapping) else {}
    )
    admission = generic_payload.get("perceptual_source_admission")
    admission_payload = admission if isinstance(admission, Mapping) else {}
    source_medians = admission_payload.get("source_median_consistency")
    source_median_payload = (
        source_medians if isinstance(source_medians, Mapping) else {}
    )
    metrics = {
        "attention_position_mismatch_p95": _finite_metric(
            continuity_metrics.get("attention_position_mismatch_p95")
        ),
        "attention_scale_mismatch_p95": _finite_metric(
            continuity_metrics.get("attention_scale_mismatch_p95")
        ),
        "pose_mismatch_p95": _finite_metric(
            continuity_metrics.get("pose_mismatch_p95")
        ),
        "effect_endpoint_position_step_max": _maximum_endpoint_metric(
            boundaries,
            "attention_position_step",
        ),
        "effect_endpoint_scale_step_max": _maximum_endpoint_metric(
            boundaries,
            "attention_scale_step",
        ),
        "effect_endpoint_pose_step_max": _maximum_endpoint_metric(
            boundaries,
            "pose_step",
        ),
        "source_quality_median_span": _finite_metric(
            source_quality_payload.get("median_span")
        ),
        "source_sharpness_log_median_span": _finite_metric(
            source_median_payload.get("sharpness_log_median_span")
        ),
        "source_luma_median_span": _finite_metric(
            source_median_payload.get("luma_median_span")
        ),
        "source_chroma_median_span": _finite_metric(
            source_median_payload.get("chroma_median_span")
        ),
        "generic_weighted_score": _finite_metric(
            generic_payload.get("weighted_score")
        ),
    }
    continuity_passed = bool(
        payload.get("direct_member_rendered_continuity_passed")
    )
    member_quality_passed = bool(payload.get("member_output_quality_passed"))
    technical_integrity_passed = not technical_issue_reasons
    finite_penalty = (
        4.0 * len(continuity_failure_reasons)
        + 2.0 * len(member_quality_failure_reasons)
        + 8.0 * len(technical_issue_reasons)
    )
    weighted_score = metrics["generic_weighted_score"]
    if weighted_score is not None:
        finite_penalty += max(0.0, (70.0 - weighted_score) / 10.0)
    return DirectMemberBoundaryProxyEvidence(
        signature=signature,
        passed=bool(
            preflight.passed
            and continuity_passed
            and member_quality_passed
            and technical_integrity_passed
        ),
        continuity_passed=continuity_passed,
        member_quality_passed=member_quality_passed,
        technical_integrity_passed=technical_integrity_passed,
        finite_solver_penalty=finite_penalty,
        continuity_failure_reasons=continuity_failure_reasons,
        member_quality_failure_reasons=member_quality_failure_reasons,
        technical_issue_reasons=technical_issue_reasons,
        metrics=tuple(sorted(metrics.items())),
    )


def load_direct_member_boundary_proxy_cache(
    cache_directory: Path | None,
    *,
    dense_binding_sha256: str,
) -> tuple[dict[str, DirectMemberBoundaryProxyEvidence], dict[str, Any]]:
    """Load only structurally valid evidence for the current dense binding."""

    binding = _validated_sha256(dense_binding_sha256, "dense binding")
    diagnostics: dict[str, Any] = {
        "contract": BOUNDARY_PROXY_CONTRACT,
        "cache_filename": BOUNDARY_PROXY_CACHE_FILENAME,
        "requested": cache_directory is not None,
        "binding_sha256": binding,
        "raw_frames_loaded": False,
    }
    if cache_directory is None:
        diagnostics.update({"status": "cache_directory_unavailable", "loaded": 0})
        return {}, diagnostics
    path = Path(cache_directory) / BOUNDARY_PROXY_CACHE_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        diagnostics.update({"status": "cache_missing", "loaded": 0})
        return {}, diagnostics
    except (OSError, json.JSONDecodeError):
        diagnostics.update({"status": "cache_unreadable", "loaded": 0})
        return {}, diagnostics
    if not (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == 1
        and payload.get("contract") == BOUNDARY_PROXY_CONTRACT
        and isinstance(payload.get("entries"), list)
    ):
        diagnostics.update({"status": "cache_schema_mismatch", "loaded": 0})
        return {}, diagnostics
    entries: dict[str, DirectMemberBoundaryProxyEvidence] = {}
    invalid_count = 0
    foreign_binding_count = 0
    for raw_entry in payload["entries"]:
        try:
            entry = DirectMemberBoundaryProxyEvidence.from_dict(raw_entry)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        if entry.signature.dense_binding_sha256 != binding:
            foreign_binding_count += 1
            continue
        entries[entry.signature.signature_sha256] = entry
    diagnostics.update(
        {
            "status": "cache_loaded",
            "loaded": len(entries),
            "invalid_entry_count": invalid_count,
            "foreign_binding_entry_count": foreign_binding_count,
        }
    )
    return entries, diagnostics


def persist_direct_member_boundary_proxy_cache(
    cache_directory: Path | None,
    evidence: Sequence[DirectMemberBoundaryProxyEvidence],
) -> dict[str, Any]:
    """Atomically persist compact scalar evidence; never persist proxy pixels."""

    diagnostics: dict[str, Any] = {
        "contract": BOUNDARY_PROXY_CONTRACT,
        "cache_filename": BOUNDARY_PROXY_CACHE_FILENAME,
        "requested": cache_directory is not None,
        "raw_frames_persisted": False,
    }
    if cache_directory is None:
        diagnostics.update({"status": "cache_directory_unavailable", "persisted": False})
        return diagnostics
    normalized: dict[str, DirectMemberBoundaryProxyEvidence] = {}
    for item in evidence:
        if not isinstance(item, DirectMemberBoundaryProxyEvidence):
            raise TypeError("boundary proxy cache accepts evidence records only")
        normalized[item.signature.signature_sha256] = item
    retained = list(normalized.values())[-_MAXIMUM_CACHE_ENTRY_COUNT:]
    payload = {
        "schema_version": 1,
        "contract": BOUNDARY_PROXY_CONTRACT,
        "entries": [item.to_dict() for item in retained],
        "storage_policy": {
            "raw_frames_persisted": False,
            "face_embeddings_persisted": False,
            "scalar_evidence_only": True,
        },
    }
    directory = Path(cache_directory)
    destination = directory / BOUNDARY_PROXY_CACHE_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{BOUNDARY_PROXY_CACHE_FILENAME}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            finally:
                raise
    except OSError as exc:
        diagnostics.update(
            {
                "status": "cache_write_failed",
                "persisted": False,
                "error_type": type(exc).__name__,
            }
        )
        return diagnostics
    diagnostics.update(
        {
            "status": "cache_persisted",
            "persisted": True,
            "entry_count": len(retained),
            "path": str(destination),
        }
    )
    return diagnostics


__all__ = [
    "BOUNDARY_PROXY_CACHE_FILENAME",
    "BOUNDARY_PROXY_CONTRACT",
    "DirectMemberBoundaryProxyEvidence",
    "DirectMemberBoundaryProxySignature",
    "build_direct_member_boundary_proxy_signature",
    "direct_member_boundary_proxy_evidence_from_preflight",
    "extract_direct_member_boundary_proxy_plan",
    "load_direct_member_boundary_proxy_cache",
    "persist_direct_member_boundary_proxy_cache",
]
