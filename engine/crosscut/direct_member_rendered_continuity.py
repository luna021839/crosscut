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

import cv2
import numpy as np

from .direct_member_visual import (
    DirectMemberVisualConfig,
    _AttentionGeometry,
    _BoundarySample,
    _actual_pose_vector,
    _attention_geometry,
    _attention_mismatch,
    _common_attention_geometries,
    _cut_adjacent_motion_evidence,
    _embedding_distance,
    _scale_log_ratio_evidence,
)
from .direct_member_target_face import (
    FOCAL_FACE_BASIS,
    FocalFaceCandidate,
    detect_focal_face_candidates,
    select_focal_face_matching_reference,
    select_focal_face_pair,
)
from .visual_score import FrameVisualDescriptor, _VideoSampler, describe_frame
from .focal_selector_geometry import FocalSelectorGeometry, focal_pose_shape_mismatch


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
    minimum_focal_reference_guided_fraction: float = 0.75
    minimum_strong_subject_motion_magnitude: float = 0.08
    maximum_strong_subject_motion_opposition: float = 0.35
    maximum_strong_subject_motion_speed_ratio_mismatch: float = 0.75
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
                "normalized_attention_endpoint_geometry_persisted": True,
                "focal_reference_embeddings_persisted": False,
                "pose_recovery_input": (
                    "same_decoded_output_frame_with_luminance_only_"
                    "detector_normalization"
                ),
                "pose_recovery_spatial_transform_applied": False,
                "failure_effect": "retain_pending_qa_and_do_not_expose",
            },
        }


def _descriptor_has_actual_pose(
    descriptor: FrameVisualDescriptor | None,
) -> bool:
    return bool(
        descriptor is not None
        and descriptor.pose_backend != "structure_hog_fallback"
        and descriptor.pose_embedding
        and math.isfinite(float(descriptor.pose_confidence))
        and descriptor.pose_confidence > 0.0
    )


def _photometric_pose_variants(
    frame: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build bounded, geometry-preserving detector views of one output frame."""

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return ()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_luminance = clahe.apply(luminance)
    color_normalized = cv2.cvtColor(
        cv2.merge((normalized_luminance, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    neutral_normalized = cv2.cvtColor(
        clahe.apply(gray),
        cv2.COLOR_GRAY2BGR,
    )
    return (
        ("lab_luminance_clahe", color_normalized),
        ("neutral_luminance_clahe", neutral_normalized),
    )


def _recover_actual_pose_under_photometric_variants(
    frame: np.ndarray,
    descriptor: FrameVisualDescriptor,
    *,
    describer: Callable[..., FrameVisualDescriptor] = describe_frame,
) -> tuple[FrameVisualDescriptor, str | None]:
    """Recover detector evidence without changing any reviewed output geometry."""

    if _descriptor_has_actual_pose(descriptor):
        return (descriptor, None)
    recovered: tuple[str, FrameVisualDescriptor] | None = None
    for label, variant in _photometric_pose_variants(frame):
        try:
            candidate = describer(
                variant,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
        except Exception:
            continue
        if not _descriptor_has_actual_pose(candidate):
            continue
        if (
            recovered is None
            or candidate.pose_confidence > recovered[1].pose_confidence
        ):
            recovered = (label, candidate)
    if recovered is None:
        return (descriptor, None)
    label, candidate = recovered
    return (
        replace(
            descriptor,
            pose_embedding=candidate.pose_embedding,
            pose_confidence=candidate.pose_confidence,
            pose_backend=candidate.pose_backend,
        ),
        label,
    )


class _FinalVideoDescriptorProvider:
    def __init__(self, path: Path) -> None:
        self._sampler = _VideoSampler(path)
        self._cache: dict[object, FrameVisualDescriptor | None] = {}
        self._focal_face_cache: dict[
            object, tuple[FocalFaceCandidate, ...]
        ] = {}
        self.pose_recovery_attempt_frame_count = 0
        self.pose_recovery_success_frame_count = 0
        self.pose_recovery_variant_counts: dict[str, int] = {}

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
        if descriptor is not None and not _descriptor_has_actual_pose(descriptor):
            try:
                frame = self._sampler.frame_at(seconds)
                if frame is not None:
                    self.pose_recovery_attempt_frame_count += 1
                    descriptor, recovery_variant = (
                        _recover_actual_pose_under_photometric_variants(
                            frame,
                            descriptor,
                        )
                    )
                    if recovery_variant is not None:
                        self.pose_recovery_success_frame_count += 1
                        self.pose_recovery_variant_counts[recovery_variant] = (
                            self.pose_recovery_variant_counts.get(
                                recovery_variant,
                                0,
                            )
                            + 1
                        )
            except Exception:
                pass
        self._cache[identity] = descriptor
        return descriptor

    def focal_face_candidates(
        self,
        seconds: float,
    ) -> tuple[FocalFaceCandidate, ...]:
        identity = self._sampler.frame_identity(seconds)
        if identity not in self._focal_face_cache:
            try:
                frame = self._sampler.frame_at(seconds)
                candidates = (
                    detect_focal_face_candidates(frame)
                    if frame is not None
                    else ()
                )
            except Exception:
                candidates = ()
            self._focal_face_cache[identity] = candidates
        return self._focal_face_cache[identity]

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


def _samples_with_common_attention(
    samples: Sequence[_BoundarySample],
    *,
    minimum_silhouette_confidence: float,
) -> tuple[_BoundarySample, ...]:
    geometries = _common_attention_geometries(
        tuple(sample.descriptor for sample in samples),
        minimum_silhouette_confidence=minimum_silhouette_confidence,
    )
    if geometries is None:
        return tuple(replace(sample, attention=None) for sample in samples)
    return tuple(
        replace(sample, attention=geometry)
        for sample, geometry in zip(samples, geometries, strict=True)
    )


def _sample_with_focal_face(
    sample: _BoundarySample,
    candidate: FocalFaceCandidate,
    focal_geometry_provider=None,
) -> _BoundarySample:
    box = candidate.bbox_xywh
    return replace(
        sample,
        focal_geometry=(focal_geometry_provider(sample.master_seconds,candidate)
            if focal_geometry_provider is not None else sample.focal_geometry),
        attention=_AttentionGeometry(
            bbox_xywh=box,
            centroid_xy=(
                box[0] + box[2] / 2.0,
                box[1] + box[3] / 2.0,
            ),
            basis=FOCAL_FACE_BASIS,
            scale_area_proxy=(
                candidate.landmark_scale**2
                if candidate.landmark_scale is not None
                else None
            ),
        ),
    )


def _serialized_attention_geometry(
    attention: _AttentionGeometry | None,
) -> dict[str, Any] | None:
    if attention is None:
        return None
    return {
        "bbox_xywh": list(attention.bbox_xywh),
        "centroid_xy": list(attention.centroid_xy),
        "scale_area_proxy": attention.scale_area_proxy,
        "scale_measurement": (
            "landmark_configuration_rms_squared"
            if attention.scale_area_proxy is not None
            else "bbox_area"
        ),
    }


def _focal_face_window(
    provider: RenderedDescriptorProvider,
    *,
    frame_indices: Sequence[int],
    fps: float,
    outgoing_references: Sequence[FocalFaceCandidate] = (),
    incoming_references: Sequence[FocalFaceCandidate] = (),
) -> tuple[
    tuple[FocalFaceCandidate | None, ...],
    float | None,
    bool,
] | None:
    """Resolve the same focal face at a hard cut and adjacent motion frames."""

    candidate_provider = getattr(provider, "focal_face_candidates", None)
    if not callable(candidate_provider) or len(frame_indices) != 4:
        return None
    candidates = tuple(
        tuple(candidate_provider(frame_index / fps))
        for frame_index in frame_indices
    )
    reference_guided = bool(outgoing_references and incoming_references)
    selected_outgoing: FocalFaceCandidate | None = None
    selected_incoming: FocalFaceCandidate | None = None
    similarity: float | None = None
    if reference_guided:
        selected_outgoing = select_focal_face_matching_reference(
            candidates[1],
            outgoing_references,
        )
        selected_incoming = select_focal_face_matching_reference(
            candidates[2],
            incoming_references,
        )
        if selected_outgoing is not None and selected_incoming is not None:
            reference_pair = select_focal_face_pair(
                (selected_outgoing,),
                (selected_incoming,),
            )
            similarity = (
                reference_pair.cosine_similarity
                if reference_pair is not None
                else None
            )
        else:
            reference_guided = False
    if selected_outgoing is None or selected_incoming is None:
        if outgoing_references or incoming_references:
            # A source-bound target is not optional. Failure to observe that
            # face cannot authorize a different matching performer in output.
            return None
        pair = select_focal_face_pair(candidates[1], candidates[2])
        if pair is None:
            return None
        selected_outgoing = pair.outgoing
        selected_incoming = pair.incoming
        similarity = pair.cosine_similarity
    previous = select_focal_face_matching_reference(
        candidates[0],
        (selected_outgoing,),
    )
    following = select_focal_face_matching_reference(
        candidates[3],
        (selected_incoming,),
    )
    return (
        (previous, selected_outgoing, selected_incoming, following),
        similarity,
        reference_guided,
    )


def evaluate_direct_member_rendered_continuity(
    video_path: Path,
    render_plan: Mapping[str, Any],
    *,
    config: DirectMemberRenderedContinuityConfig | None = None,
    descriptor_provider: RenderedDescriptorProvider | None = None,
    focal_reference_candidates_by_boundary_index: Mapping[
        int, tuple[FocalFaceCandidate, FocalFaceCandidate]
    ] | None = None,
    focal_pose_model_path: Path | None = None,
    focal_geometry_provider=None,
    focal_body_model_path: Path | None = None,
    focal_source_cache_arguments: Mapping[str, Any] | None = None,
    body_window_provider=None,
    focal_body_execution_device: str = 'cpu',
) -> DirectMemberRenderedContinuityResult:
    """Evaluate every real hard source switch in the encoded member video."""

    settings = config or DirectMemberRenderedContinuityConfig()
    if focal_body_model_path is not None and (
        focal_pose_model_path is None or body_window_provider is not None
    ):
        raise ValueError('owned rendered body authority requires its pose model')
    if focal_pose_model_path is not None and (
        descriptor_provider is not None or focal_geometry_provider is not None
    ):
        raise ValueError('rendered focal pose model requires its owned decoded video provider')
    if focal_reference_candidates_by_boundary_index is not None:
        if not isinstance(
            focal_reference_candidates_by_boundary_index,
            Mapping,
        ) or any(
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or boundary_index < 0
            or not isinstance(reference_pair, tuple)
            or len(reference_pair) != 2
            or any(
                not isinstance(candidate, FocalFaceCandidate)
                for candidate in reference_pair
            )
            for boundary_index, reference_pair in (
                focal_reference_candidates_by_boundary_index.items()
            )
        ):
            raise ValueError("rendered focal references are malformed")
    reference_authority_active = bool(
        focal_reference_candidates_by_boundary_index
    ) or focal_body_model_path is not None or body_window_provider is not None
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
    if focal_pose_model_path is not None:
        from .rendered_focal_geometry import RenderedFocalGeometryProvider
        try:
            focal_geometry_provider=RenderedFocalGeometryProvider(owned._sampler.frame_at,focal_pose_model_path)
        except BaseException:
            owned.close()
            raise
    owned_body_provider = None
    if focal_body_model_path is not None:
        from .rendered_body_window import RenderedBodyWindowProvider
        try:
            owned_body_provider = RenderedBodyWindowProvider(render_plan,
                owned._sampler.frame_at, body_model_path=focal_body_model_path,
                pose_model_path=focal_pose_model_path,
                body_execution_device=focal_body_execution_device,
                source_cache_arguments=focal_source_cache_arguments)
            body_window_provider = owned_body_provider
        except BaseException:
            owned.close()
            raise
    target_pose_authority = focal_geometry_provider is not None or body_window_provider is not None
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
    focal_guided_position_values: list[float] = []
    focal_guided_scale_values: list[float] = []
    actual_pose_count = 0
    attention_count = 0
    hard_switch_count = 0
    effect_switch_count = 0
    effect_endpoint_pair_count = 0
    effect_endpoint_attention_count = 0
    effect_endpoint_pose_count = 0
    focal_face_correspondence_count = 0
    focal_face_motion_window_count = 0
    focal_face_reference_guided_count = 0
    effect_focal_face_correspondence_count = 0
    effect_focal_reference_guided_count = 0
    aggregate_silhouette_diagnostic_only_count = 0
    body_reference_guided_count = 0
    effect_body_reference_guided_count = 0

    def resolve_body_window(face_window, indices):
        # Keep a complete source-guided face window. Otherwise compare bodies
        # on ALL four frames, never a face area against a torso area.
        if body_window_provider is None or (face_window is not None and
                face_window[2] and all(c is not None for c in face_window[0])):
            return None, []
        from .rendered_body_reference import RenderedBodyMeasurement
        values = tuple(body_window_provider(indices))
        if len(values) != 4 or any(not isinstance(v, RenderedBodyMeasurement) for v in values):
            raise ValueError('rendered body window must contain four typed measurements')
        audit = [v.audit() for v in values]
        for item in audit:
            item.pop('observed_output_pose_xyc', None)
        observed = all(v.reason == 'observed' and v.person is not None and v.pose is not None
                       for v in values)
        return (values if observed else None), audit

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
        left, right = _samples_with_common_attention(
            (left, right),
            minimum_silhouette_confidence=(
                visual_config.minimum_silhouette_confidence
            ),
        )
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
                # Camera correction for an overlap cannot use the blended
                # entry/exit samples themselves: a face detector may jump
                # between the two performers as alpha changes. Resolve the
                # exact pure outgoing frame immediately before the effect and
                # the exact pure incoming frame immediately after it instead.
                # These are the two pictures the virtual camera must align;
                # the outgoing handle holds its final transform while the
                # incoming transform is held through the overlap.
                pure_focal_window = _focal_face_window(
                    provider,
                    frame_indices=(
                        max(0, boundary_frame - 2),
                        boundary_frame - 1,
                        effect_next_frame,
                        min(output_frames - 1, effect_next_frame + 1),
                    ),
                    fps=fps,
                    outgoing_references=(
                        (
                            focal_reference_candidates_by_boundary_index[
                                boundary_index
                            ][0],
                        )
                        if focal_reference_candidates_by_boundary_index is not None
                        and boundary_index
                        in focal_reference_candidates_by_boundary_index
                        else ()
                    ),
                    incoming_references=(
                        (
                            focal_reference_candidates_by_boundary_index[
                                boundary_index
                            ][1],
                        )
                        if focal_reference_candidates_by_boundary_index is not None
                        and boundary_index
                        in focal_reference_candidates_by_boundary_index
                        else ()
                    ),
                )
                pure_body_window, pure_body_audit = resolve_body_window(pure_focal_window,
                    (max(0, boundary_frame - 2), boundary_frame - 1,
                     effect_next_frame, min(output_frames - 1, effect_next_frame + 1)))
                pure_focal_similarity: float | None = None
                pure_focal_reference_guided = False
                if pure_body_window is not None:
                    from .rendered_body_window import body_sample
                    pure_outgoing = body_sample(entry_previous, pure_body_window[1])
                    pure_incoming = body_sample(exit_next, pure_body_window[2])
                    effect_body_reference_guided_count += 1
                elif pure_focal_window is not None:
                    (
                        pure_focal_candidates,
                        pure_focal_similarity,
                        pure_focal_reference_guided,
                    ) = pure_focal_window
                    assert pure_focal_candidates[1] is not None
                    assert pure_focal_candidates[2] is not None
                    pure_outgoing, pure_incoming = (
                        _sample_with_focal_face(
                            entry_previous,
                            pure_focal_candidates[1],
                            focal_geometry_provider,
                        ),
                        _sample_with_focal_face(
                            exit_next,
                            pure_focal_candidates[2],
                            focal_geometry_provider,
                        ),
                    )
                    effect_focal_face_correspondence_count += 1
                    effect_focal_reference_guided_count += int(
                        pure_focal_reference_guided
                    )
                else:
                    pure_outgoing, pure_incoming = (
                        _samples_with_common_attention(
                            (entry_previous, exit_next),
                            minimum_silhouette_confidence=(
                                visual_config.minimum_silhouette_confidence
                            ),
                        )
                    )
                if target_pose_authority and not (pure_focal_reference_guided or pure_body_window is not None):
                    fail("rendered_boundary_target_reference_unresolved", boundary_index)
                pure_position, pure_scale = _attention_mismatch(
                    pure_outgoing.attention,
                    pure_incoming.attention,
                )
                pure_pose = None
                if (
                    pure_outgoing.descriptor is not None
                    and pure_incoming.descriptor is not None
                ):
                    pure_pose = (focal_pose_shape_mismatch(pure_outgoing.focal_geometry,pure_incoming.focal_geometry)
                        if target_pose_authority else _embedding_distance(
                        _actual_pose_vector(
                            pure_outgoing.descriptor,
                            visual_config,
                        )
                        or (),
                        _actual_pose_vector(
                            pure_incoming.descriptor,
                            visual_config,
                        )
                        or (),
                    ))
                records.append(
                    {
                        "boundary_index": boundary_index,
                        "output_frame_index": boundary_frame,
                        "output_seconds": boundary_frame / fps,
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
                        "pure_endpoint_sample_frame_indices": [
                            boundary_frame - 1,
                            effect_next_frame,
                        ],
                        "descriptors_available": bool(
                            pure_outgoing.descriptor is not None
                            and pure_incoming.descriptor is not None
                        ),
                        "attention_basis": {
                            "outgoing": (
                                pure_outgoing.attention.basis
                                if pure_outgoing.attention is not None
                                else None
                            ),
                            "incoming": (
                                pure_incoming.attention.basis
                                if pure_incoming.attention is not None
                                else None
                            ),
                        },
                        "attention_geometry": {
                            "outgoing": _serialized_attention_geometry(
                                pure_outgoing.attention
                            ),
                            "incoming": _serialized_attention_geometry(
                                pure_incoming.attention
                            ),
                        },
                        "attention_position_mismatch": pure_position,
                        "attention_scale_mismatch": pure_scale,
                        "pose_mismatch": pure_pose,
                        "focal_face_correspondence_similarity": (
                            pure_focal_similarity
                        ),
                        "focal_face_reference_guided": (
                            pure_focal_reference_guided
                        ),
                        "body_reference_guided": pure_body_window is not None,
                        "body_reference_measurements": pure_body_audit,
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
            focal_window = _focal_face_window(
                provider,
                frame_indices=(
                    boundary_frame - 2,
                    boundary_frame - 1,
                    boundary_frame,
                    boundary_frame + 1,
                ),
                fps=fps,
                outgoing_references=(
                    (
                        focal_reference_candidates_by_boundary_index[
                            boundary_index
                        ][0],
                    )
                    if focal_reference_candidates_by_boundary_index is not None
                    and boundary_index
                    in focal_reference_candidates_by_boundary_index
                    else ()
                ),
                incoming_references=(
                    (
                        focal_reference_candidates_by_boundary_index[
                            boundary_index
                        ][1],
                    )
                    if focal_reference_candidates_by_boundary_index is not None
                    and boundary_index
                    in focal_reference_candidates_by_boundary_index
                    else ()
                ),
            )
            focal_similarity: float | None = None
            focal_reference_guided = False
            body_window, body_audit = resolve_body_window(focal_window,
                (boundary_frame - 2, boundary_frame - 1, boundary_frame, boundary_frame + 1))
            body_motion_samples = None
            if body_window is not None:
                from .rendered_body_window import body_sample
                body_motion_samples = tuple(body_sample(s, m) for s, m in zip(samples, body_window, strict=True))
                endpoint_samples = body_motion_samples[1:3]
                focal_candidates = (None, None, None, None)
                body_reference_guided_count += 1
            elif focal_window is not None:
                (
                    focal_candidates,
                    focal_similarity,
                    focal_reference_guided,
                ) = focal_window
                assert focal_candidates[1] is not None
                assert focal_candidates[2] is not None
                endpoint_samples = (
                    _sample_with_focal_face(
                        outgoing_endpoint,
                        focal_candidates[1],
                        focal_geometry_provider,
                    ),
                    _sample_with_focal_face(
                        incoming_start,
                        focal_candidates[2],
                        focal_geometry_provider,
                    ),
                )
                focal_face_correspondence_count += 1
                focal_face_reference_guided_count += int(
                    focal_reference_guided
                )
            else:
                focal_candidates = (None, None, None, None)
                endpoint_samples = _samples_with_common_attention(
                    (outgoing_endpoint, incoming_start),
                    minimum_silhouette_confidence=(
                        visual_config.minimum_silhouette_confidence
                    ),
                )
            target_reference_guided = focal_reference_guided or body_window is not None
            if target_pose_authority and not target_reference_guided:
                fail("rendered_boundary_target_reference_unresolved", boundary_index)
            endpoint_outgoing, endpoint_incoming = endpoint_samples
            position, scale = _attention_mismatch(
                endpoint_outgoing.attention,
                endpoint_incoming.attention,
            )
            bbox_scale_mismatch: float | None = None
            landmark_scale_mismatch: float | None = None
            scale_direction_agreement: bool | None = None
            if (
                endpoint_outgoing.attention is not None
                and endpoint_incoming.attention is not None
            ):
                (
                    bbox_scale_log,
                    landmark_scale_log,
                    scale_direction_agreement,
                    _consensus_scale_log,
                ) = _scale_log_ratio_evidence(
                    endpoint_outgoing.attention.bbox_xywh[2]
                    * endpoint_outgoing.attention.bbox_xywh[3],
                    endpoint_incoming.attention.bbox_xywh[2]
                    * endpoint_incoming.attention.bbox_xywh[3],
                    endpoint_outgoing.attention.scale_area_proxy,
                    endpoint_incoming.attention.scale_area_proxy,
                )
                bbox_scale_mismatch = min(
                    1.0,
                    abs(bbox_scale_log) / math.log(3.2),
                )
                landmark_scale_mismatch = (
                    min(
                        1.0,
                        abs(landmark_scale_log) / math.log(3.2),
                    )
                    if landmark_scale_log is not None
                    else None
                )
            if position is None or scale is None:
                fail("rendered_boundary_attention_missing", boundary_index)
            else:
                attention_count += 1
                position_values.append(position)
                scale_values.append(scale)
                if target_reference_guided:
                    focal_guided_position_values.append(position)
                    focal_guided_scale_values.append(scale)
                if position > settings.maximum_attention_position_mismatch + 1e-9:
                    fail("rendered_boundary_position_discontinuity", boundary_index)
                if scale > settings.maximum_attention_scale_mismatch + 1e-9:
                    fail("rendered_boundary_scale_discontinuity", boundary_index)

            left = outgoing_endpoint.descriptor
            right = incoming_start.descriptor
            pose_mismatch: float | None = None
            silhouette_mismatch: float | None = None
            aggregate_silhouette_mismatch: float | None = None
            if left is not None and right is not None:
                left_pose = _actual_pose_vector(left, visual_config)
                right_pose = _actual_pose_vector(right, visual_config)
                pose_mismatch = (focal_pose_shape_mismatch(endpoint_outgoing.focal_geometry,endpoint_incoming.focal_geometry)
                    if target_pose_authority else _embedding_distance(
                    left_pose or (),
                    right_pose or (),
                ))
                if pose_mismatch is not None:
                    actual_pose_count += 1
                    pose_values.append(pose_mismatch)
                    if pose_mismatch > settings.maximum_pose_mismatch + 1e-9:
                        fail("rendered_boundary_pose_discontinuity", boundary_index)
                aggregate_silhouette_mismatch = _embedding_distance(
                    left.silhouette_embedding
                    if left.silhouette_confidence
                    >= visual_config.minimum_silhouette_confidence
                    else (),
                    right.silhouette_embedding
                    if right.silhouette_confidence
                    >= visual_config.minimum_silhouette_confidence
                    else (),
                )
                # A full-frame segmentation contains every visible dancer.
                # With source-bound member references active, it is never a
                # target-body authority when actual pose is available -- even
                # on the occasional wide frame where the focal face vanishes.
                # A source-bound focal face also supplies direct target-member
                # identity and framing evidence when one pose frame is absent.
                # The aggregate remains a fail-closed fallback only when both
                # target-specific authorities are missing.
                if reference_authority_active:
                    silhouette_mismatch = (
                        aggregate_silhouette_mismatch
                        if pose_mismatch is None
                        and not target_reference_guided
                        else None
                    )
                else:
                    silhouette_mismatch = (
                        None
                        if focal_window is not None
                        else aggregate_silhouette_mismatch
                    )
                if (
                    aggregate_silhouette_mismatch is not None
                    and silhouette_mismatch is None
                ):
                    aggregate_silhouette_diagnostic_only_count += 1
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
            if (
                pose_mismatch is None
                and silhouette_mismatch is None
                and not target_reference_guided
            ):
                fail("rendered_boundary_shape_evidence_missing", boundary_index)

            if body_motion_samples is not None:
                motion_samples = body_motion_samples
            elif all(candidate is not None for candidate in focal_candidates):
                motion_samples = tuple(
                    _sample_with_focal_face(sample, candidate, focal_geometry_provider)
                    for sample, candidate in zip(
                        samples,
                        focal_candidates,
                        strict=True,
                    )
                    if candidate is not None
                )
                focal_face_motion_window_count += 1
            else:
                motion_samples = _samples_with_common_attention(
                    samples,
                    minimum_silhouette_confidence=(
                        visual_config.minimum_silhouette_confidence
                    ),
                )
            if target_pose_authority:
                # Missing focal frames must not reactivate whole-frame pose
                # in the four-frame motion window.
                motion_samples=tuple(replace(s,focal_geometry=s.focal_geometry or
                    FocalSelectorGeometry(0,s.master_seconds,None,pose_backend='rtmpose_body7'))
                    for s in motion_samples)
            motion = _cut_adjacent_motion_evidence(
                outgoing_previous=motion_samples[0],
                outgoing_endpoint=motion_samples[1],
                incoming_start=motion_samples[2],
                incoming_next=motion_samples[3],
                config=visual_config,
                output_space_subject_motion_used=True,
            )
            strong_output_subject_motion_gate = bool(
                target_reference_guided
                and motion.outgoing_subject_motion_magnitude is not None
                and motion.incoming_subject_motion_magnitude is not None
                and motion.outgoing_subject_motion_magnitude
                + 1e-9
                >= settings.minimum_strong_subject_motion_magnitude
                and motion.incoming_subject_motion_magnitude
                + 1e-9
                >= settings.minimum_strong_subject_motion_magnitude
            )
            if strong_output_subject_motion_gate:
                if (
                    motion.subject_direction_cosine is None
                    or motion.subject_direction_cosine
                    < -settings.maximum_strong_subject_motion_opposition
                    - 1e-9
                ):
                    fail("rendered_boundary_motion_direction_opposed", boundary_index)
                if (
                    motion.subject_motion_speed_ratio_mismatch is not None
                    and motion.subject_motion_speed_ratio_mismatch
                    > settings.maximum_strong_subject_motion_speed_ratio_mismatch
                    + 1e-9
                ):
                    fail("rendered_boundary_motion_speed_discontinuity", boundary_index)
            if motion.both_moving:
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
                            endpoint_outgoing.attention.basis
                            if endpoint_outgoing.attention is not None
                            else None
                        ),
                        "incoming": (
                            endpoint_incoming.attention.basis
                            if endpoint_incoming.attention is not None
                            else None
                        ),
                    },
                    "attention_geometry": {
                        "outgoing": (
                            {
                                "bbox_xywh": list(
                                    endpoint_outgoing.attention.bbox_xywh
                                ),
                                "centroid_xy": list(
                                    endpoint_outgoing.attention.centroid_xy
                                ),
                                "scale_area_proxy": (
                                    endpoint_outgoing.attention.scale_area_proxy
                                ),
                                "scale_measurement": (
                                    "landmark_configuration_rms_squared"
                                    if endpoint_outgoing.attention.scale_area_proxy
                                    is not None
                                    else "bbox_area"
                                ),
                            }
                            if endpoint_outgoing.attention is not None
                            else None
                        ),
                        "incoming": (
                            {
                                "bbox_xywh": list(
                                    endpoint_incoming.attention.bbox_xywh
                                ),
                                "centroid_xy": list(
                                    endpoint_incoming.attention.centroid_xy
                                ),
                                "scale_area_proxy": (
                                    endpoint_incoming.attention.scale_area_proxy
                                ),
                                "scale_measurement": (
                                    "landmark_configuration_rms_squared"
                                    if endpoint_incoming.attention.scale_area_proxy
                                    is not None
                                    else "bbox_area"
                                ),
                            }
                            if endpoint_incoming.attention is not None
                            else None
                        ),
                    },
                    "attention_position_mismatch": position,
                    "attention_scale_mismatch": scale,
                    "attention_bbox_scale_mismatch_diagnostic": (
                        bbox_scale_mismatch
                    ),
                    "attention_landmark_scale_mismatch_diagnostic": (
                        landmark_scale_mismatch
                    ),
                    "attention_scale_evidence_direction_agreement": (
                        scale_direction_agreement
                    ),
                    "pose_mismatch": pose_mismatch,
                    "silhouette_mismatch": silhouette_mismatch,
                    "aggregate_silhouette_mismatch_diagnostic_only": (
                        aggregate_silhouette_mismatch
                    ),
                    "focal_face_correspondence_similarity": focal_similarity,
                    "focal_face_reference_guided": (
                        focal_reference_guided
                    ),
                    "body_reference_guided": body_window is not None,
                    "body_reference_measurements": body_audit,
                    "strong_output_subject_motion_gate_applied": (
                        strong_output_subject_motion_gate
                    ),
                    "motion": motion.to_dict(),
                    "raw_embeddings_persisted": False,
                }
            )
    finally:
        try:
            if owned_body_provider is not None:
                owned_body_provider.close()
        finally:
            if owned is not None:
                owned.close()

    hard_count = hard_switch_count
    source_switch_count = hard_switch_count + effect_switch_count
    pose_coverage = actual_pose_count / hard_count if hard_count else 0.0
    attention_coverage = attention_count / hard_count if hard_count else 0.0
    focal_reference_coverage = (
        focal_face_reference_guided_count / hard_count
        if hard_count
        else 0.0
    )
    focal_authority_coverage = (
        focal_face_correspondence_count / hard_count
        if hard_count
        else 0.0
    )
    target_reference_coverage = (
        (focal_face_reference_guided_count + body_reference_guided_count) / hard_count
        if hard_count else 0.0
    )
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

    if (
        reference_authority_active
        and hard_count > 0
        and (target_reference_coverage if target_pose_authority
             else focal_authority_coverage) + 1e-9
        < settings.minimum_focal_reference_guided_fraction
    ):
        failures_by_reason.setdefault(
            "rendered_focal_reference_coverage_below_floor",
            [],
        )

    strict_position_values = (
        focal_guided_position_values
        if reference_authority_active
        else position_values
    )
    strict_scale_values = (
        focal_guided_scale_values
        if reference_authority_active
        else scale_values
    )
    p95_metrics = {
        "attention_position_mismatch_p95": _p95(strict_position_values),
        "attention_scale_mismatch_p95": _p95(strict_scale_values),
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
        "pose_measurement_basis": ('source_referenced_output_rtmpose_body7'
            if target_pose_authority else 'legacy_full_frame_pose'),
        "effect_endpoint_step_pose_basis": "legacy_full_frame_step_not_target_specific",
        "fps": fps,
        "output_frame_count": output_frames,
        "source_switch_boundary_count": source_switch_count,
        "hard_cut_source_switch_count": hard_count,
        "effect_source_switch_count": effect_switch_count,
        "effect_endpoint_pair_count": effect_endpoint_pair_count,
        "effect_endpoint_attention_count": effect_endpoint_attention_count,
        "effect_endpoint_pose_count": effect_endpoint_pose_count,
        "focal_face_correspondence_boundary_count": (
            focal_face_correspondence_count
        ),
        "focal_face_reference_guided_boundary_count": (
            focal_face_reference_guided_count
        ),
        "focal_face_reference_guided_fraction": focal_reference_coverage,
        "focal_face_authoritative_fraction": focal_authority_coverage,
        "body_reference_guided_boundary_count": body_reference_guided_count,
        "effect_body_reference_guided_boundary_count": effect_body_reference_guided_count,
        "target_reference_guided_fraction": target_reference_coverage,
        "body_reference_authority_active": body_window_provider is not None,
        "focal_face_authority_basis": (
            "source_bound_reference_or_verified_official_fancam_"
            "cross_output_correspondence"
        ),
        "focal_reference_authority_active": reference_authority_active,
        "focal_face_motion_window_count": focal_face_motion_window_count,
        "effect_focal_face_correspondence_boundary_count": (
            effect_focal_face_correspondence_count
        ),
        "effect_focal_face_reference_guided_boundary_count": (
            effect_focal_reference_guided_count
        ),
        "aggregate_silhouette_diagnostic_only_boundary_count": (
            aggregate_silhouette_diagnostic_only_count
        ),
        "actual_pose_boundary_count": actual_pose_count,
        "actual_pose_coverage_fraction": pose_coverage,
        "attention_boundary_count": attention_count,
        "attention_coverage_fraction": attention_coverage,
        "all_attention_position_mismatch_p95_diagnostic_only": _p95(
            position_values
        ),
        "all_attention_scale_mismatch_p95_diagnostic_only": _p95(
            scale_values
        ),
        **p95_metrics,
        "exact_incoming_first_frame_evaluated": True,
        "final_render_transform_domain_evaluated": True,
        "photometric_pose_recovery_attempt_frame_count": (
            owned.pose_recovery_attempt_frame_count if owned is not None else 0
        ),
        "photometric_pose_recovery_success_frame_count": (
            owned.pose_recovery_success_frame_count if owned is not None else 0
        ),
        "photometric_pose_recovery_variant_counts": (
            dict(sorted(owned.pose_recovery_variant_counts.items()))
            if owned is not None
            else {}
        ),
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
