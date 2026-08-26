"""Rendered-frame continuity gate for automatic direct-member crosscuts.

Pre-render source descriptors are necessary but not sufficient: aspect fill,
editorial camera transforms, timing quantization, and the encoder can change
the exact frames a viewer sees.  This module decodes the final MP4 at every
hard source switch and evaluates the real outgoing-last/incoming-first window.
Only normalized scalar diagnostics are persisted; pose and silhouette vectors
remain transient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .direct_member_visual import (
    DirectMemberVisualConfig,
    _AttentionGeometry,
    _BoundarySample,
    _actual_pose_vector,
    _attention_geometry,
    _attention_mismatch,
    _cut_adjacent_motion_evidence,
    _embedding_distance,
)
from .visual_score import FrameVisualDescriptor, _VideoSampler


ANALYSIS_BASIS = "automatic_member_exact_rendered_boundary_continuity_v1"
RenderedDescriptorProvider = Callable[[float], FrameVisualDescriptor | None]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _p95(values: Sequence[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class DirectMemberRenderedContinuityConfig:
    maximum_attention_position_mismatch: float = 0.20
    maximum_attention_scale_mismatch: float = 0.18
    maximum_pose_mismatch: float = 0.30
    maximum_silhouette_mismatch: float = 0.42
    maximum_attention_position_mismatch_p95: float = 0.12
    maximum_attention_scale_mismatch_p95: float = 0.10
    maximum_pose_mismatch_p95: float = 0.24
    maximum_silhouette_mismatch_p95: float = 0.35
    minimum_actual_pose_coverage_fraction: float = 0.80
    minimum_attention_coverage_fraction: float = 0.95
    maximum_effect_endpoint_attention_position_step: float = 0.12
    maximum_effect_endpoint_attention_scale_step: float = 0.10
    maximum_effect_endpoint_pose_step: float = 0.24
    maximum_effect_endpoint_silhouette_step: float = 0.35

    def __post_init__(self) -> None:
        values = tuple(
            float(getattr(self, name)) for name in self.__dataclass_fields__
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("rendered member continuity thresholds must be normalized")

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class DirectMemberRenderedContinuityResult:
    passed: bool
    failure_reasons: tuple[str, ...]
    failure_boundary_indices: Mapping[str, tuple[int, ...]]
    metrics: Mapping[str, Any]
    boundaries: tuple[Mapping[str, Any], ...]
    config: DirectMemberRenderedContinuityConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis": ANALYSIS_BASIS,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "failure_boundary_indices": {
                name: list(indices)
                for name, indices in sorted(self.failure_boundary_indices.items())
            },
            "metrics": dict(self.metrics),
            "boundaries": [dict(item) for item in self.boundaries],
            "thresholds": self.config.to_dict(),
            "authority": {
                "input": "decoded_final_mp4_exact_output_frames",
                "hard_cut_window_frames": [-2, -1, 0, 1],
                "incoming_first_visible_frame_offset": 0,
                "overlap_effect_endpoint_windows": (
                    "[boundary-1,boundary] and "
                    "[boundary+effect_frames-1,boundary+effect_frames]"
                ),
                "aspect_fill_and_editorial_transform_already_materialized": True,
                "raw_pose_or_silhouette_vectors_persisted": False,
                "failure_effect": "retain_pending_qa_and_do_not_expose",
            },
        }


class _FinalVideoDescriptorProvider:
    def __init__(self, path: Path) -> None:
        self._sampler = _VideoSampler(path)
        self._cache: dict[object, FrameVisualDescriptor | None] = {}

    def __call__(self, seconds: float) -> FrameVisualDescriptor | None:
        identity = self._sampler.frame_identity(seconds)
        if identity in self._cache:
            return self._cache[identity]
        try:
            descriptor = self._sampler.feature_descriptor_at(
                seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
        except Exception:
            descriptor = None
        self._cache[identity] = descriptor
        return descriptor

    def close(self) -> None:
        self._sampler.close()


def _sample(
    *,
    frame_index: int,
    fps: float,
    provider: RenderedDescriptorProvider,
) -> _BoundarySample:
    seconds = frame_index / fps
    descriptor = provider(seconds)
    return _BoundarySample(
        master_seconds=seconds,
        source_local_seconds=seconds,
        descriptor=descriptor,
        attention=_attention_geometry(descriptor),
    )


def evaluate_direct_member_rendered_continuity(
    video_path: Path,
    render_plan: Mapping[str, Any],
    *,
    config: DirectMemberRenderedContinuityConfig | None = None,
    descriptor_provider: RenderedDescriptorProvider | None = None,
) -> DirectMemberRenderedContinuityResult:
    """Evaluate every real hard source switch in the encoded member video."""

    settings = config or DirectMemberRenderedContinuityConfig()
    fps = _finite(render_plan.get("fps"))
    output_frames = _integer(render_plan.get("output_frame_count"))
    segments = _sequence(render_plan.get("segments"), "render segments")
    transitions = _sequence(render_plan.get("transitions"), "render transitions")
    if (
        fps is None
        or fps <= 0.0
        or output_frames is None
        or output_frames <= 4
        or len(segments) < 2
        or len(transitions) != len(segments) - 1
    ):
        raise ValueError("render plan lacks an exact rendered continuity contract")
    path = Path(video_path)
    if descriptor_provider is None and not path.is_file():
        raise ValueError("rendered member video is unavailable")

    source_indices: list[int] = []
    boundary_frames: list[int] = []
    cumulative_seconds = 0.0
    previous_frame = 0
    for index, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment, f"render segment {index}")
        source_index = _integer(segment.get("source_index"))
        duration = _finite(segment.get("duration"))
        if source_index is None or source_index < 0 or duration is None or duration <= 0.0:
            raise ValueError("render segment timing or source is invalid")
        cumulative_seconds += duration
        end_frame = int(round(cumulative_seconds * fps))
        if end_frame - previous_frame < 2:
            raise ValueError("render segment is too short for continuity review")
        source_indices.append(source_index)
        if index < len(segments) - 1:
            boundary_frames.append(end_frame)
        previous_frame = end_frame
    if previous_frame != output_frames:
        raise ValueError("render segment frames do not match encoded frame contract")

    owned: _FinalVideoDescriptorProvider | None = None
    provider = descriptor_provider
    if provider is None:
        owned = _FinalVideoDescriptorProvider(path)
        provider = owned
    assert provider is not None
    visual_config = replace(
        DirectMemberVisualConfig(),
        sample_frame_step_seconds=1.0 / fps,
    )

    records: list[dict[str, Any]] = []
    failures_by_reason: dict[str, list[int]] = {}
    pose_values: list[float] = []
    silhouette_values: list[float] = []
    position_values: list[float] = []
    scale_values: list[float] = []
    actual_pose_count = 0
    attention_count = 0
    hard_switch_count = 0
    effect_switch_count = 0
    effect_endpoint_pair_count = 0
    effect_endpoint_attention_count = 0
    effect_endpoint_pose_count = 0

    def fail(reason: str, boundary_index: int) -> None:
        failures_by_reason.setdefault(reason, []).append(boundary_index)

    def effect_endpoint_step(
        *,
        boundary_index: int,
        label: str,
        left: _BoundarySample,
        right: _BoundarySample,
    ) -> dict[str, Any]:
        nonlocal effect_endpoint_pair_count
        nonlocal effect_endpoint_attention_count
        nonlocal effect_endpoint_pose_count
        effect_endpoint_pair_count += 1
        if left.descriptor is None or right.descriptor is None:
            fail("rendered_effect_endpoint_descriptor_missing", boundary_index)
        position, scale = _attention_mismatch(left.attention, right.attention)
        if position is None or scale is None:
            fail("rendered_effect_endpoint_attention_missing", boundary_index)
        else:
            effect_endpoint_attention_count += 1
            if (
                position
                > settings.maximum_effect_endpoint_attention_position_step
                + 1e-9
            ):
                fail(
                    "rendered_effect_endpoint_position_discontinuity",
                    boundary_index,
                )
            if (
                scale
                > settings.maximum_effect_endpoint_attention_scale_step + 1e-9
            ):
                fail(
                    "rendered_effect_endpoint_scale_discontinuity",
                    boundary_index,
                )
        left_descriptor = left.descriptor
        right_descriptor = right.descriptor
        pose_step: float | None = None
        silhouette_step: float | None = None
        if left_descriptor is not None and right_descriptor is not None:
            pose_step = _embedding_distance(
                _actual_pose_vector(left_descriptor, visual_config) or (),
                _actual_pose_vector(right_descriptor, visual_config) or (),
            )
            if pose_step is not None:
                effect_endpoint_pose_count += 1
                if pose_step > settings.maximum_effect_endpoint_pose_step + 1e-9:
                    fail(
                        "rendered_effect_endpoint_pose_discontinuity",
                        boundary_index,
                    )
            silhouette_step = _embedding_distance(
                left_descriptor.silhouette_embedding
                if left_descriptor.silhouette_confidence
                >= visual_config.minimum_silhouette_confidence
                else (),
                right_descriptor.silhouette_embedding
                if right_descriptor.silhouette_confidence
                >= visual_config.minimum_silhouette_confidence
                else (),
            )
            if (
                silhouette_step is not None
                and silhouette_step
                > settings.maximum_effect_endpoint_silhouette_step + 1e-9
            ):
                fail(
                    "rendered_effect_endpoint_silhouette_discontinuity",
                    boundary_index,
                )
        if pose_step is None and silhouette_step is None:
            fail("rendered_effect_endpoint_shape_evidence_missing", boundary_index)
        return {
            "label": label,
            "sample_frame_indices": [
                int(round(left.master_seconds * fps)),
                int(round(right.master_seconds * fps)),
            ],
            "attention_position_step": position,
            "attention_scale_step": scale,
            "pose_step": pose_step,
            "silhouette_step": silhouette_step,
        }

    try:
        for boundary_index, boundary_frame in enumerate(boundary_frames):
            outgoing_source = source_indices[boundary_index]
            incoming_source = source_indices[boundary_index + 1]
            if outgoing_source == incoming_source:
                continue
            transition = _mapping(
                transitions[boundary_index],
                f"render transition {boundary_index}",
            )
            transition_kind = str(transition.get("kind") or "")
            if transition_kind != "hard_cut":
                effect_switch_count += 1
                effect_frames = _integer(transition.get("frame_count"))
                if transition_kind not in {
                    "face_dissolve",
                    "member_zoom_bridge",
                    "member_blur_bridge",
                } or effect_frames is None or effect_frames < 3:
                    fail("rendered_effect_contract_invalid", boundary_index)
                    continue
                effect_last_frame = boundary_frame + effect_frames - 1
                effect_next_frame = boundary_frame + effect_frames
                if (
                    boundary_frame < 1
                    or effect_last_frame >= output_frames
                    or effect_next_frame >= output_frames
                ):
                    fail("rendered_effect_window_outside_video", boundary_index)
                    continue
                entry_previous = _sample(
                    frame_index=boundary_frame - 1,
                    fps=fps,
                    provider=provider,
                )
                entry_start = _sample(
                    frame_index=boundary_frame,
                    fps=fps,
                    provider=provider,
                )
                exit_endpoint = _sample(
                    frame_index=effect_last_frame,
                    fps=fps,
                    provider=provider,
                )
                exit_next = _sample(
                    frame_index=effect_next_frame,
                    fps=fps,
                    provider=provider,
                )
                records.append(
                    {
                        "boundary_index": boundary_index,
                        "output_frame_index": boundary_frame,
                        "outgoing_source_index": outgoing_source,
                        "incoming_source_index": incoming_source,
                        "transition_kind": transition_kind,
                        "effect_frame_count": effect_frames,
                        "endpoint_steps": [
                            effect_endpoint_step(
                                boundary_index=boundary_index,
                                label="effect_entry",
                                left=entry_previous,
                                right=entry_start,
                            ),
                            effect_endpoint_step(
                                boundary_index=boundary_index,
                                label="effect_exit",
                                left=exit_endpoint,
                                right=exit_next,
                            ),
                        ],
                        "raw_embeddings_persisted": False,
                    }
                )
                continue
            hard_switch_count += 1
            if boundary_frame < 2 or boundary_frame + 1 >= output_frames:
                fail("rendered_boundary_window_outside_video", boundary_index)
                continue
            outgoing_previous = _sample(
                frame_index=boundary_frame - 2,
                fps=fps,
                provider=provider,
            )
            outgoing_endpoint = _sample(
                frame_index=boundary_frame - 1,
                fps=fps,
                provider=provider,
            )
            incoming_start = _sample(
                frame_index=boundary_frame,
                fps=fps,
                provider=provider,
            )
            incoming_next = _sample(
                frame_index=boundary_frame + 1,
                fps=fps,
                provider=provider,
            )
            samples = (
                outgoing_previous,
                outgoing_endpoint,
                incoming_start,
                incoming_next,
            )
            descriptors_available = all(
                sample.descriptor is not None for sample in samples
            )
            if not descriptors_available:
                fail("rendered_boundary_descriptor_missing", boundary_index)
            position, scale = _attention_mismatch(
                outgoing_endpoint.attention,
                incoming_start.attention,
            )
            if position is None or scale is None:
                fail("rendered_boundary_attention_missing", boundary_index)
            else:
                attention_count += 1
                position_values.append(position)
                scale_values.append(scale)
                if position > settings.maximum_attention_position_mismatch + 1e-9:
                    fail("rendered_boundary_position_discontinuity", boundary_index)
                if scale > settings.maximum_attention_scale_mismatch + 1e-9:
                    fail("rendered_boundary_scale_discontinuity", boundary_index)

            left = outgoing_endpoint.descriptor
            right = incoming_start.descriptor
            pose_mismatch: float | None = None
            silhouette_mismatch: float | None = None
            if left is not None and right is not None:
                left_pose = _actual_pose_vector(left, visual_config)
                right_pose = _actual_pose_vector(right, visual_config)
                pose_mismatch = _embedding_distance(
                    left_pose or (),
                    right_pose or (),
                )
                if pose_mismatch is not None:
                    actual_pose_count += 1
                    pose_values.append(pose_mismatch)
                    if pose_mismatch > settings.maximum_pose_mismatch + 1e-9:
                        fail("rendered_boundary_pose_discontinuity", boundary_index)
                silhouette_mismatch = _embedding_distance(
                    left.silhouette_embedding
                    if left.silhouette_confidence
                    >= visual_config.minimum_silhouette_confidence
                    else (),
                    right.silhouette_embedding
                    if right.silhouette_confidence
                    >= visual_config.minimum_silhouette_confidence
                    else (),
                )
                if silhouette_mismatch is not None:
                    silhouette_values.append(silhouette_mismatch)
                    if (
                        silhouette_mismatch
                        > settings.maximum_silhouette_mismatch + 1e-9
                    ):
                        fail(
                            "rendered_boundary_silhouette_discontinuity",
                            boundary_index,
                        )
            if pose_mismatch is None and silhouette_mismatch is None:
                fail("rendered_boundary_shape_evidence_missing", boundary_index)

            motion = _cut_adjacent_motion_evidence(
                outgoing_previous=outgoing_previous,
                outgoing_endpoint=outgoing_endpoint,
                incoming_start=incoming_start,
                incoming_next=incoming_next,
                config=visual_config,
                output_space_subject_motion_used=True,
            )
            if motion.both_moving:
                if (
                    motion.direction_cosine_audit_only is None
                    or motion.direction_cosine_audit_only
                    < visual_config.minimum_motion_direction_cosine - 1e-9
                ):
                    fail("rendered_boundary_motion_direction_opposed", boundary_index)
                if (
                    motion.motion_speed_ratio_mismatch is not None
                    and motion.motion_speed_ratio_mismatch
                    > visual_config.maximum_motion_speed_ratio_mismatch + 1e-9
                ):
                    fail("rendered_boundary_motion_speed_discontinuity", boundary_index)
                if (
                    motion.pose_phase_continuation_mismatch is not None
                    and motion.pose_phase_continuation_mismatch
                    > visual_config.maximum_pose_phase_continuation_mismatch + 1e-9
                ):
                    fail("rendered_boundary_pose_phase_discontinuity", boundary_index)
            records.append(
                {
                    "boundary_index": boundary_index,
                    "output_frame_index": boundary_frame,
                    "output_seconds": boundary_frame / fps,
                    "outgoing_source_index": outgoing_source,
                    "incoming_source_index": incoming_source,
                    "transition_kind": transition_kind,
                    "sample_frame_indices": [
                        boundary_frame - 2,
                        boundary_frame - 1,
                        boundary_frame,
                        boundary_frame + 1,
                    ],
                    "descriptors_available": descriptors_available,
                    "attention_basis": {
                        "outgoing": (
                            outgoing_endpoint.attention.basis
                            if outgoing_endpoint.attention is not None
                            else None
                        ),
                        "incoming": (
                            incoming_start.attention.basis
                            if incoming_start.attention is not None
                            else None
                        ),
                    },
                    "attention_position_mismatch": position,
                    "attention_scale_mismatch": scale,
                    "pose_mismatch": pose_mismatch,
                    "silhouette_mismatch": silhouette_mismatch,
                    "motion": motion.to_dict(),
                    "raw_embeddings_persisted": False,
                }
            )
    finally:
        if owned is not None:
            owned.close()

    hard_count = hard_switch_count
    source_switch_count = hard_switch_count + effect_switch_count
    pose_coverage = actual_pose_count / hard_count if hard_count else 0.0
    attention_coverage = attention_count / hard_count if hard_count else 0.0
    if source_switch_count == 0:
        failures_by_reason.setdefault(
            "no_rendered_source_switch_boundaries",
            [],
        )
    if (
        hard_count > 0
        and pose_coverage + 1e-9
        < settings.minimum_actual_pose_coverage_fraction
    ):
        failures_by_reason.setdefault("rendered_actual_pose_coverage_below_floor", [])
    if (
        hard_count > 0
        and attention_coverage + 1e-9
        < settings.minimum_attention_coverage_fraction
    ):
        failures_by_reason.setdefault("rendered_attention_coverage_below_floor", [])

    p95_metrics = {
        "attention_position_mismatch_p95": _p95(position_values),
        "attention_scale_mismatch_p95": _p95(scale_values),
        "pose_mismatch_p95": _p95(pose_values),
        "silhouette_mismatch_p95": _p95(silhouette_values),
    }
    for name, value, maximum in (
        (
            "rendered_attention_position_p95_above_floor",
            p95_metrics["attention_position_mismatch_p95"],
            settings.maximum_attention_position_mismatch_p95,
        ),
        (
            "rendered_attention_scale_p95_above_floor",
            p95_metrics["attention_scale_mismatch_p95"],
            settings.maximum_attention_scale_mismatch_p95,
        ),
        (
            "rendered_pose_p95_above_floor",
            p95_metrics["pose_mismatch_p95"],
            settings.maximum_pose_mismatch_p95,
        ),
        (
            "rendered_silhouette_p95_above_floor",
            p95_metrics["silhouette_mismatch_p95"],
            settings.maximum_silhouette_mismatch_p95,
        ),
    ):
        if value is not None and value > maximum + 1e-9:
            failures_by_reason.setdefault(name, [])

    failure_reasons = tuple(sorted(failures_by_reason))
    failure_indices = {
        name: tuple(sorted(set(indices)))
        for name, indices in failures_by_reason.items()
        if indices
    }
    metrics = {
        "fps": fps,
        "output_frame_count": output_frames,
        "source_switch_boundary_count": source_switch_count,
        "hard_cut_source_switch_count": hard_count,
        "effect_source_switch_count": effect_switch_count,
        "effect_endpoint_pair_count": effect_endpoint_pair_count,
        "effect_endpoint_attention_count": effect_endpoint_attention_count,
        "effect_endpoint_pose_count": effect_endpoint_pose_count,
        "actual_pose_boundary_count": actual_pose_count,
        "actual_pose_coverage_fraction": pose_coverage,
        "attention_boundary_count": attention_count,
        "attention_coverage_fraction": attention_coverage,
        **p95_metrics,
        "exact_incoming_first_frame_evaluated": True,
        "final_render_transform_domain_evaluated": True,
    }
    return DirectMemberRenderedContinuityResult(
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        failure_boundary_indices=failure_indices,
        metrics=metrics,
        boundaries=tuple(records),
        config=settings,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "DirectMemberRenderedContinuityConfig",
    "DirectMemberRenderedContinuityResult",
    "evaluate_direct_member_rendered_continuity",
]
