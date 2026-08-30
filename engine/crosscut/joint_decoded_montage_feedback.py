"""Exact bridge from bound decoded reviews to joint montage soft feedback."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .joint_visual_candidate_adapter import (
    JointDecodedMontageSegmentObservation,
)
from .render import RenderPlan
from .timeline import TimelineChoice


_RENDERED_GEOMETRY_KEY = "rendered_segment_interior_geometry"


def _lowercase_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def joint_decoded_montage_observations_from_bound_review(
    *,
    timeline_choices: Sequence[TimelineChoice],
    render_plan: RenderPlan,
    review_payload: Mapping[str, Any],
    render_variant_signature: str,
) -> tuple[JointDecodedMontageSegmentObservation, ...]:
    """Extract complete per-segment shot classes after exact caller binding.

    The caller must first bind ``review_payload`` to ``render_plan`` using the
    production decoded-proxy identity contract.  This bridge independently
    checks that the montage sub-review carries that same content SHA, covers
    every render segment exactly once, and reports the renderer-domain source
    index for the corresponding unshifted joint timeline choice.
    """

    if not isinstance(render_plan, RenderPlan):
        raise TypeError("decoded montage feedback requires a RenderPlan")
    if not isinstance(review_payload, Mapping):
        raise TypeError("decoded montage feedback review must be a mapping")
    signature = _lowercase_sha256(
        render_variant_signature,
        "decoded montage render variant signature",
    )
    choices = tuple(timeline_choices)
    if any(not isinstance(choice, TimelineChoice) for choice in choices):
        raise TypeError(
            "decoded montage feedback requires unshifted TimelineChoice values"
        )
    if not choices or len(choices) != len(render_plan.segments):
        raise ValueError(
            "decoded montage choices and render segments must align exactly"
        )

    root_geometry = review_payload.get("segment_interior_geometry")
    root_signature = (
        root_geometry.get("render_timeline_binding_sha256")
        if isinstance(root_geometry, Mapping)
        else None
    )
    if root_signature != signature:
        raise ValueError("decoded montage root render binding changed")

    montage = review_payload.get("rendered_montage_grammar_review")
    if not isinstance(montage, Mapping):
        raise ValueError("decoded montage grammar review is missing")
    plan_binding = montage.get("render_plan_binding")
    if not isinstance(plan_binding, Mapping) or plan_binding.get("passed") is not True:
        raise ValueError("decoded montage render-plan binding did not pass")
    if plan_binding.get("bound_segment_count") != len(render_plan.segments):
        raise ValueError("decoded montage bound segment count changed")
    bound_fps = plan_binding.get("fps")
    if (
        isinstance(bound_fps, bool)
        or not isinstance(bound_fps, (int, float))
        or not math.isfinite(float(bound_fps))
        or not math.isclose(
            float(bound_fps),
            render_plan.fps,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("decoded montage render-plan fps binding changed")

    geometry = montage.get(_RENDERED_GEOMETRY_KEY)
    if not isinstance(geometry, Mapping) or geometry.get("available") is not True:
        raise ValueError("decoded montage segment evidence is unavailable")
    if geometry.get("source_render_timeline_binding_sha256") != signature:
        raise ValueError("decoded montage segment render binding changed")
    records = geometry.get("segment_records")
    if not isinstance(records, list) or len(records) != len(render_plan.segments):
        raise ValueError("decoded montage segment records are incomplete")

    observations: list[JointDecodedMontageSegmentObservation] = []
    for index, (choice, segment, record) in enumerate(
        zip(choices, render_plan.segments, records, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise ValueError("decoded montage segment record is invalid")
        if record.get("segment_index") != index:
            raise ValueError("decoded montage segment index binding changed")
        if record.get("source_index") != segment.source_index:
            raise ValueError("decoded montage segment source binding changed")
        if record.get("sample_count") != 3:
            raise ValueError("decoded montage segment sample count changed")
        sampled_classes = record.get("sampled_proxy_shot_classes")
        if not isinstance(sampled_classes, list):
            raise ValueError("decoded montage segment shot classes are missing")
        observations.append(
            JointDecodedMontageSegmentObservation.from_choice(
                choice,
                render_variant_signature=signature,
                sampled_proxy_shot_classes=sampled_classes,
            )
        )
    return tuple(observations)


def merge_joint_decoded_montage_observations(
    existing: Sequence[JointDecodedMontageSegmentObservation],
    incoming: Sequence[JointDecodedMontageSegmentObservation],
) -> tuple[JointDecodedMontageSegmentObservation, ...]:
    """Deterministically deduplicate exact clock/render observations."""

    merged: dict[
        tuple[object, str],
        JointDecodedMontageSegmentObservation,
    ] = {}
    for observation in (*tuple(existing), *tuple(incoming)):
        if not isinstance(observation, JointDecodedMontageSegmentObservation):
            raise TypeError("decoded montage observation is invalid")
        identity = (
            observation.candidate_clock,
            observation.render_variant_signature,
        )
        previous = merged.get(identity)
        if previous is not None and previous != observation:
            raise ValueError(
                "decoded montage observation conflicts for one bound variant"
            )
        merged[identity] = observation
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (
                item.candidate_clock.deterministic_key,
                item.render_variant_signature,
            ),
        )
    )


__all__ = [
    "joint_decoded_montage_observations_from_bound_review",
    "merge_joint_decoded_montage_observations",
]
