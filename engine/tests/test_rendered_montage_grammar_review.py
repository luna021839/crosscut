from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.professional_montage_policy import (
    close_burst_policy_binding_sha256,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    segment_frame_counts,
)
from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from crosscut.rendered_montage_grammar_review import (
    RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
    RenderedMontageGrammarConfig,
    _proxy_geometry_shot_class,
    _symmetric_fraction_target_evidence,
    _wide_run_soft_target_evidence,
    review_rendered_montage_grammar,
)
from crosscut.rendered_proxy_review import (
    OutputSubjectGeometry,
    _segment_interior_geometry_evidence,
    _segment_interior_schedule,
    verified_file_content_sha256,
)


def _fixture_file_content_sha256(path: Path) -> str:
    return hashlib.sha256(
        f"synthetic-file:{Path(path)}".encode("utf-8")
    ).hexdigest()


def _analysis(
    shot_types: tuple[str, ...],
    sources: tuple[int, ...],
    *,
    motion_indices: tuple[int, ...] = (1,),
    role: str = "chorus_like",
    establish: bool = True,
    face_proxy: bool = True,
) -> dict[str, object]:
    if len(shot_types) != len(sources):
        raise ValueError("fixture arrays differ")
    choices = [
        {
            "segment_index": index,
            "source_index": sources[index],
            "canonical_master_start": float(index),
            "canonical_master_end": float(index + 1),
            "duration_seconds": 1.0,
            "source_local_start": float(index),
            "source_local_end": float(index + 1),
            "shot_type": shot_type,
            "entry_shot_type": shot_type,
            "exit_shot_type": shot_type,
            "shot_id": f"shot-{index}",
            "detected_person_counts": (
                [6, 6, 6]
                if shot_type in {"group_wide", "full_shot"}
                else [2, 2, 2]
                if shot_type == "medium"
                else [1, 1, 1]
            ),
            "formation_person_counts": (
                [6, 6, 6]
                if shot_type in {"group_wide", "full_shot"}
                else [0, 0, 0]
            ),
            "formation_upper_body_counts": (
                [5, 5, 5]
                if shot_type in {"group_wide", "full_shot"}
                else [0, 0, 0]
            ),
        }
        for index, shot_type in enumerate(shot_types)
    ]
    geometry = {
        "face_count": 1 if face_proxy else 0,
        "face_bbox_xywh": [0.35, 0.18, 0.30, 0.35] if face_proxy else None,
    }
    boundaries = [
        {
            "boundary_index": index,
            "cut_frame_index": (index + 1) * 30,
            "outgoing_source_index": sources[index],
            "incoming_source_index": sources[index + 1],
            "transition_kind": "hard_cut",
            "transition_frame_count": 0,
            "sample_offsets": [-2, -1, 0, 1],
            "frame_indices": [
                (index + 1) * 30 + offset for offset in (-2, -1, 0, 1)
            ],
            "frame_metrics": [
                {
                    "offset": offset,
                    "frame_index": (index + 1) * 30 + offset,
                    "subject_geometry": dict(geometry),
                }
                for offset in (-2, -1, 0, 1)
            ],
        }
        for index in range(len(choices) - 1)
    ]
    duration = float(len(choices))
    source_count = max(sources) + 1
    return {
        "stage_edit_exact_shot_clock": {
            "contract": (
                "serialized_inference_token_section_and_phrase_progress_"
                "per_selected_shot_v1"
            ),
            "source_index_domain": "eligible_source_index",
            "shots": [
                {
                    "shot_index": index,
                    "source_index": sources[index],
                    "section_role": role,
                    "section_progress": index / max(1, len(choices) - 1),
                    "phrase_progress": (index % 3) / 3.0,
                }
                for index in range(len(choices))
            ],
        },
        "source_index_mapping": {
            "request_index_domain": "request_source_index",
            "eligible_index_domain": "eligible_source_index",
            "render_plan_index_domain": "render_plan_sources_array_index",
            "domain_equivalences": {
                "render_plan_sources_array_index": "request_source_index",
            },
            "eligible_to_request": list(range(source_count)),
            "request_to_eligible": list(range(source_count)),
            "eligible_sources": [
                {
                    "eligible_source_index": index,
                    "request_source_index": index,
                    "source": f"source-{index}.mp4",
                }
                for index in range(source_count)
            ],
            "payload_index_domains": {
                "visual_timeline.choices[].source_index": (
                    "eligible_source_index"
                ),
                (
                    "rendered_proxy_review.boundaries[]"
                    ".outgoing_source_index"
                ): "request_source_index",
                (
                    "rendered_proxy_review.boundaries[]"
                    ".incoming_source_index"
                ): "request_source_index",
                "render_plan.segments[].source_index": (
                    "render_plan_sources_array_index"
                ),
            },
        },
        "source_derivative_diversity": {
            "index_domain": "eligible_source_index",
            "source_count": source_count,
            "family_ids": list(range(source_count)),
            "production_ready": True,
            "clock_contract_complete": True,
            "uncertain_pair_count": 0,
            "family_consistency_conflict_count": 0,
        },
        "section_plan": {
            "sections": [
                {
                    "section_index": 0,
                    "role": role,
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "camera_motion_budget": 0.82,
                    "establish_with_wide_shot": establish,
                }
            ]
        },
        "visual_timeline": {
            "index_domain": "eligible_source_index",
            "choices": choices,
        },
        "editorial_motion": {
            "segments": [
                {
                    "segment_index": index,
                    "applied": index in motion_indices,
                    "keyframes": (
                        [
                            {
                                "time_seconds": 0.0,
                                "zoom": 1.0,
                                "center_x": 0.5,
                                "center_y": 0.5,
                                "roll_degrees": 0.0,
                            },
                            {
                                "time_seconds": 0.75,
                                "zoom": 1.05,
                                "center_x": 0.5,
                                "center_y": 0.5,
                                "roll_degrees": 0.0,
                            },
                        ]
                        if index in motion_indices
                        else []
                    ),
                }
                for index in range(len(choices))
            ]
        },
        "rendered_proxy_review": {
            "schema_version": 3,
            "proxy": {
                "fps": 30.0,
                "decoded_frame_count": len(choices) * 30,
                "content_sha256": _fixture_file_content_sha256(
                    Path("synthetic-proxy.mp4")
                ),
            },
            "transition_count": len(boundaries),
            "reviewed_boundary_count": len(boundaries),
            "hard_cut_count": len(boundaries),
            "dissolve_count": 0,
            "boundaries": boundaries,
        },
    }


def _render_plan(
    fixture: dict[str, object],
    *,
    source_paths: tuple[Path, ...] | None = None,
) -> dict[str, object]:
    choices = fixture["visual_timeline"]["choices"]
    source_count = fixture["source_derivative_diversity"]["source_count"]
    duration = float(len(choices))
    return {
        "sources": (
            [str(path) for path in source_paths]
            if source_paths is not None
            else [f"source-{index}.mp4" for index in range(source_count)]
        ),
        "source_index_domains": {
            "sources[]": "render_plan_sources_array_index",
            "reference_source": "render_plan_sources_array_index",
            "segments[].source_index": "render_plan_sources_array_index",
        },
        "reference_source": 0,
        "master_audio_start": 0.0,
        "fps": 30.0,
        "segments": [
            {
                "source_index": choice["source_index"],
                "master_start": choice["canonical_master_start"],
                "duration": (
                    choice["canonical_master_end"]
                    - choice["canonical_master_start"]
                ),
                "source_start": choice["source_local_start"],
                "source_duration": None,
                "effective_source_duration": (
                    choice["source_local_end"] - choice["source_local_start"]
                ),
                "crop": None,
                "pan_crop": None,
                "editorial_transform": None,
                "look_adjustment": None,
                "boundary_look_track": None,
            }
            for choice in choices
        ],
        "transitions": [
            {"kind": "hard_cut", "frame_count": 0}
            for _ in range(max(0, len(choices) - 1))
        ],
        "render_duration": duration,
        "output_frame_count": int(duration * 30),
    }


def _attach_segment_interior_geometry(
    fixture: dict[str, object],
    *,
    sample_classes_by_segment: dict[int, tuple[str, ...]] | None = None,
    automatic_ive: bool = True,
    source_paths: tuple[Path, ...] | None = None,
    use_real_content: bool = False,
) -> None:
    choices = fixture["visual_timeline"]["choices"]
    source_count = fixture["source_derivative_diversity"]["source_count"]
    proxy_plan = RenderPlan(
        sources=(
            source_paths
            if source_paths is not None
            else tuple(
                Path(f"source-{index}.mp4") for index in range(source_count)
            )
        ),
        segments=tuple(
            RenderSegment(
                choice["source_index"],
                choice["source_local_start"],
                choice["duration_seconds"],
                choice["canonical_master_start"],
            )
            for choice in choices
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("synthetic-proxy.mp4"),
        width=854,
        height=480,
        fps=30.0,
        transitions=tuple(
            RenderTransition() for _ in range(max(0, len(choices) - 1))
        ),
    )
    counts = segment_frame_counts(proxy_plan)
    schedule = _segment_interior_schedule(proxy_plan, counts)
    samples_by_segment: dict[int, list[object]] = {}
    for sample in schedule:
        samples_by_segment.setdefault(sample.segment_index, []).append(sample)

    aliases = {
        "group_wide": "wide",
        "full_shot": "wide",
        "close_up": "close",
    }
    frames: dict[int, np.ndarray] = {}
    for segment_index, segment_samples in samples_by_segment.items():
        default_class = aliases.get(
            str(choices[segment_index]["shot_type"]),
            str(choices[segment_index]["shot_type"]),
        )
        classes = (
            sample_classes_by_segment.get(segment_index)
            if sample_classes_by_segment is not None
            and segment_index in sample_classes_by_segment
            else (default_class,) * len(segment_samples)
        )
        if classes is None or len(classes) != len(segment_samples):
            raise ValueError("interior fixture sample classes differ from schedule")
        for sample, shot_class in zip(segment_samples, classes, strict=True):
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            frame[0, 0, 0] = {"wide": 1, "medium": 2, "close": 3}[
                shot_class
            ]
            frames[sample.frame_index] = frame

    def geometry_provider(frame: np.ndarray) -> OutputSubjectGeometry:
        shot_class = {1: "wide", 2: "medium", 3: "close"}[
            int(frame[0, 0, 0])
        ]
        if shot_class == "wide":
            return OutputSubjectGeometry(
                face_count=0,
                detected_person_count=3,
                formation_boxes_xywh=(
                    (0.14, 0.30, 0.15, 0.50),
                    (0.42, 0.27, 0.16, 0.55),
                    (0.70, 0.31, 0.14, 0.49),
                ),
                shot_scale_basis="aggregate_formation_people",
            )
        height = 0.15 if shot_class == "medium" else 0.30
        return OutputSubjectGeometry(
            face_count=1,
            face_bbox_xywh=(0.35, 0.20, 0.20, height),
            face_backend="synthetic",
        )

    proxy = fixture["rendered_proxy_review"]
    proxy["schema_version"] = 3
    if use_real_content:
        proxy["segment_interior_geometry"] = (
            _segment_interior_geometry_evidence(
                plan=proxy_plan,
                counts=counts,
                schedule=schedule,
                frames=frames,
                geometry_provider=geometry_provider,
                proxy_identity=proxy["proxy"],
            )
        )
    else:
        with patch(
            "crosscut.rendered_proxy_review.verified_file_content_sha256",
            side_effect=_fixture_file_content_sha256,
        ):
            proxy["segment_interior_geometry"] = (
                _segment_interior_geometry_evidence(
                    plan=proxy_plan,
                    counts=counts,
                    schedule=schedule,
                    frames=frames,
                    geometry_provider=geometry_provider,
                    proxy_identity=proxy["proxy"],
                )
            )
    if automatic_ive:
        fixture["request"] = {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        }


def _review(
    fixture: dict[str, object],
    *,
    render_plan: dict[str, object] | None = None,
    config: RenderedMontageGrammarConfig | None = None,
) -> dict[str, object]:
    with patch(
        "crosscut.rendered_montage_grammar_review.verified_file_content_sha256",
        side_effect=_fixture_file_content_sha256,
    ):
        return review_rendered_montage_grammar(
            fixture,
            render_plan=(
                _render_plan(fixture) if render_plan is None else render_plan
            ),
            config=config or RenderedMontageGrammarConfig(),
        )


def _attach_v3_motion_evidence(
    fixture: dict[str, object],
    *,
    authorized_motion_indices: tuple[int, ...] = (2,),
    suppressed_motion_indices: tuple[int, ...] = (),
) -> dict[str, object]:
    choices = fixture["visual_timeline"]["choices"]
    source_count = fixture["source_derivative_diversity"]["source_count"]
    authority = DynamicCropMotionAuthorityV3.create(
        release_freeze_content_sha256="a" * 64,
        release_freeze_file_sha256="b" * 64,
        source_commit_oid="c" * 40,
        candidate_id="rendered-montage-v3-test",
    )
    decodes: list[DynamicCropTrackDecodeV3] = []
    segments: list[RenderSegment] = []
    dynamic_indices = set(authorized_motion_indices) | set(
        suppressed_motion_indices
    )
    for index, choice in enumerate(choices):
        if index in dynamic_indices:
            keyframes = (
                EditorialTransformKeyframe(0.0, 1.02, 0.5, 0.5, 0.0),
                EditorialTransformKeyframe(0.8, 1.05, 0.52, 0.5, 0.0),
            )
        else:
            # Deliberately vary static framing across cuts.  Those cut-time
            # changes remain excluded by typed v3 authority.
            keyframes = (
                EditorialTransformKeyframe(
                    0.0,
                    1.02 + index * 0.001,
                    0.5,
                    0.5,
                    0.0,
                ),
            )
        original = EditorialTransformTrack(
            keyframes,
            reason="stage_edit_dynamic_crop_head_v3",
        )
        decodes.append(
            DynamicCropTrackDecodeV3(
                source_id=f"source-{choice['source_index']}",
                source_index=choice["source_index"],
                eligible=True,
                loss_masked=False,
                track=original,
                predicted_keyframe_count=len(keyframes),
                applied_keyframe_count=len(keyframes),
                minimum_safe_zoom_projection_count=0,
                rejection_reasons=(),
                motion_logit=(
                    -20.0 if index in suppressed_motion_indices else 20.0
                ),
                shot_duration_seconds=choice["duration_seconds"],
            )
        )
        segments.append(
            RenderSegment(
                source_index=choice["source_index"],
                source_start=choice["source_local_start"],
                duration=choice["duration_seconds"],
                master_start=choice["canonical_master_start"],
                editorial_transform=EditorialTransformTrack(
                    keyframes,
                    reason=(
                        "stage_edit_dynamic_crop_head_v3_"
                        "stage_edit_preserved"
                    ),
                ),
            )
        )
    provisional = RenderPlan(
        sources=tuple(Path(f"source-{index}.mp4") for index in range(source_count)),
        segments=tuple(segments),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("synthetic-v3-final.mp4"),
        fps=30.0,
        transitions=tuple(
            RenderTransition() for _ in range(max(0, len(segments) - 1))
        ),
    )
    final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
        render_plan=provisional,
        decodes=tuple(decodes),
        authority=authority,
    )
    evidence = corroborate_dynamic_crop_motion_timeline_v3(
        render_plan=final_plan,
        decodes=tuple(decodes),
        authority=authority,
    )
    fixture["dynamic_crop_version"] = 3
    fixture["dynamic_crop_motion_v3"] = evidence.to_dict()
    return final_plan.to_json()


def _split_into_two_highlights(fixture: dict[str, object]) -> None:
    choices = fixture["visual_timeline"]["choices"]
    if len(choices) != 12:
        raise ValueError("two-highlight fixture needs twelve one-second choices")
    first = fixture["section_plan"]["sections"][0]
    fixture["section_plan"]["sections"] = [
        {**first, "section_index": 0, "start_seconds": 0.0, "end_seconds": 6.0},
        {**first, "section_index": 1, "start_seconds": 6.0, "end_seconds": 12.0},
    ]
    shots = fixture["stage_edit_exact_shot_clock"]["shots"]
    for index, shot in enumerate(shots):
        shot["section_progress"] = (index % 6) / 5.0


def _configure_dissolve(
    fixture: dict[str, object],
    plan: dict[str, object],
    *,
    boundary_index: int,
    frame_count: int = 3,
) -> dict[str, object]:
    plan["transitions"][boundary_index] = {
        "kind": "face_dissolve",
        "frame_count": frame_count,
    }
    proxy = fixture["rendered_proxy_review"]
    proxy["hard_cut_count"] -= 1
    proxy["dissolve_count"] += 1
    boundary = proxy["boundaries"][boundary_index]
    cut_frame_index = boundary["cut_frame_index"]
    offsets = (-2, -1, frame_count, frame_count + 1)
    boundary["transition_kind"] = "face_dissolve"
    boundary["transition_frame_count"] = frame_count
    boundary["sample_offsets"] = list(offsets)
    boundary["frame_indices"] = [cut_frame_index + value for value in offsets]
    boundary["frame_metrics"] = [
        {
            "offset": offset,
            "frame_index": cut_frame_index + offset,
            "subject_geometry": {
                "face_count": 1,
                "face_bbox_xywh": [0.35, 0.18, 0.30, 0.35],
            },
        }
        for offset in offsets
    ]
    return boundary


def _hash_bound_montage_config() -> RenderedMontageGrammarConfig:
    identity = {
        "analysis_basis": "ive_human_reference_visual_montage_grammar_v2",
        "schema_version": 2,
        "path": "/tmp/ive-professional-montage-v2.json",
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    binding = close_burst_policy_binding_sha256(
        profile_identity=identity,
        policy_fingerprint_sha256="c" * 64,
        minimum_shot_count=2.0,
        minimum_duration_seconds=2.1503940144767864,
        minimum_source_switch_fraction=2.0 / 3.0,
        event_reference_count=6,
    )
    return RenderedMontageGrammarConfig(
        policy_profile_analysis_basis=identity["analysis_basis"],
        policy_profile_schema_version=identity["schema_version"],
        policy_profile_path=identity["path"],
        policy_profile_file_sha256=identity["file_sha256"],
        policy_profile_content_sha256=identity["content_sha256"],
        policy_fingerprint_sha256="c" * 64,
        minimum_close_burst_shot_count=2.0,
        minimum_close_burst_duration_seconds=2.1503940144767864,
        minimum_close_burst_source_switch_fraction=2.0 / 3.0,
        close_burst_event_reference_count=6,
        close_burst_policy_binding_sha256=binding,
    )


class RenderedMontageGrammarReviewTests(unittest.TestCase):
    def test_no_face_without_body_geometry_is_unknown_not_wide(
        self,
    ) -> None:
        self.assertEqual(
            _proxy_geometry_shot_class(
                {"face_count": 0, "face_bbox_xywh": None},
                RenderedMontageGrammarConfig(),
            ),
            "unknown",
        )

    def test_no_face_proxy_uses_person_geometry_hierarchy(self) -> None:
        config = RenderedMontageGrammarConfig()
        self.assertEqual(
            _proxy_geometry_shot_class(
                OutputSubjectGeometry(
                    detected_person_count=1,
                    formation_boxes_xywh=((0.40, 0.14, 0.20, 0.72),),
                    shot_scale_basis="single_or_pair_performer_bbox",
                ).to_dict(),
                config,
            ),
            "medium",
        )
        self.assertEqual(
            _proxy_geometry_shot_class(
                OutputSubjectGeometry(
                    detected_person_count=3,
                    formation_boxes_xywh=(
                        (0.15, 0.30, 0.14, 0.48),
                        (0.43, 0.26, 0.15, 0.54),
                        (0.71, 0.31, 0.13, 0.47),
                    ),
                    shot_scale_basis="aggregate_formation_people",
                ).to_dict(),
                config,
            ),
            "wide",
        )
        self.assertEqual(
            _proxy_geometry_shot_class(
                OutputSubjectGeometry(
                    face_count=1,
                    face_bbox_xywh=(0.46, 0.22, 0.08, 0.08),
                    detected_person_count=3,
                    formation_boxes_xywh=(
                        (0.15, 0.30, 0.14, 0.48),
                        (0.43, 0.26, 0.15, 0.54),
                        (0.71, 0.31, 0.13, 0.47),
                    ),
                    shot_scale_basis="aggregate_formation_people",
                ).to_dict(),
                config,
            ),
            "wide",
        )
        self.assertEqual(
            _proxy_geometry_shot_class(
                OutputSubjectGeometry(
                    face_count=1,
                    face_bbox_xywh=(0.40, 0.18, 0.20, 0.28),
                    detected_person_count=3,
                    formation_boxes_xywh=(
                        (0.15, 0.30, 0.14, 0.48),
                        (0.43, 0.26, 0.15, 0.54),
                        (0.71, 0.31, 0.13, 0.47),
                    ),
                    shot_scale_basis="aggregate_formation_people",
                ).to_dict(),
                config,
            ),
            "close",
        )

    def test_all_wide_bridge_cannot_bypass_role_conditioned_duration_gate(
        self,
    ) -> None:
        result = _review(
            _analysis(
                ("group_wide",) * 12,
                tuple(range(12)),
                role="bridge_like",
                establish=False,
            )
        )

        section = result["sections"][0]
        aggregate = result["role_conditioned_duration_aggregates"][
            "bridge_like"
        ]
        self.assertTrue(section["strict_role_duration_gate_applied"])
        self.assertFalse(section["strict_highlight_gate_applied"])
        self.assertFalse(section["short_section_duration_exemption_applied"])
        self.assertIn(
            "bridge_nonwide_fraction_below_floor",
            section["failure_codes"],
        )
        # The immutable bridge P10 for close duration is zero: it must not be
        # promoted into a made-up positive local quota.  Pooled P25 still
        # rejects an all-wide role.
        self.assertIsNone(
            aggregate["hard_targets"][
                "local_minimum_closeup_duration_fraction"
            ]
        )
        self.assertFalse(
            aggregate["hard_targets"][
                "zero_p10_promoted_to_positive_minimum"
            ]
        )
        self.assertIn(
            "bridge_pooled_closeup_fraction_below_product_minimum",
            aggregate["failure_codes"],
        )
        self.assertIn(
            "bridge_pooled_nonwide_fraction_below_product_minimum",
            result["failure_codes"],
        )
        self.assertFalse(result["passed"])

    def test_short_chorus_no_longer_uses_five_second_gate_bypass(self) -> None:
        result = _review(
            _analysis(
                ("group_wide",) * 4,
                (0, 1, 2, 3),
                establish=False,
            )
        )

        section = result["sections"][0]
        self.assertLess(
            section["duration_seconds"],
            RenderedMontageGrammarConfig().highlight_minimum_duration_seconds,
        )
        self.assertTrue(section["strict_highlight_gate_applied"])
        self.assertFalse(section["short_section_duration_exemption_applied"])
        self.assertIn(
            "highlight_closeup_fraction_below_floor",
            section["failure_codes"],
        )
        self.assertFalse(result["passed"])

    def test_slow_highlight_source_switch_cadence_fails_final_render_gate(
        self,
    ) -> None:
        config = RenderedMontageGrammarConfig(
            minimum_wide_involved_source_switch_fraction=0.0,
            minimum_nonwide_burst_boundaries=0,
            minimum_post_burst_formation_reset_count=0,
            minimum_motion_accent_count=0,
        )
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "close_up",
                    "medium",
                    "group_wide",
                    "close_up",
                    "medium",
                    "group_wide",
                    "close_up",
                ),
                (0, 0, 1, 1, 1, 1, 1, 1),
                establish=False,
            ),
            config=config,
        )

        section = result["sections"][0]
        self.assertAlmostEqual(7.5, section["source_switches_per_minute"])
        self.assertAlmostEqual(
            18.642084765823732,
            section["source_switches_per_minute_hard_minimum"],
        )
        self.assertAlmostEqual(
            23.505554096533405,
            section["source_switches_per_minute_preferred_target"],
        )
        self.assertIn(
            "highlight_source_switches_per_minute_below_product_minimum",
            section["failure_codes"],
        )
        self.assertFalse(result["passed"])

    def test_soft_targets_are_symmetric_and_do_not_reward_over_close(self) -> None:
        target = 0.40
        under = _symmetric_fraction_target_evidence(0.30, target)
        over = _symmetric_fraction_target_evidence(0.50, target)
        farther_over = _symmetric_fraction_target_evidence(0.70, target)
        self.assertAlmostEqual(under["alignment_score"], over["alignment_score"])
        self.assertLess(farther_over["alignment_score"], over["alignment_score"])

        below_wide_target = _wide_run_soft_target_evidence(2.0, 5.0, 8.0)
        at_wide_target = _wide_run_soft_target_evidence(5.0, 5.0, 8.0)
        above_wide_target = _wide_run_soft_target_evidence(6.0, 5.0, 8.0)
        self.assertEqual(1.0, below_wide_target["alignment_score"])
        self.assertEqual(1.0, at_wide_target["alignment_score"])
        self.assertLess(above_wide_target["alignment_score"], 1.0)

    def test_highlight_aggregate_binds_exact_primitive_policy_identity(self) -> None:
        config = _hash_bound_montage_config()
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                motion_indices=(1, 4),
            ),
            config=config,
        )

        provenance = result["highlight_aggregate"]["policy_provenance"]
        self.assertTrue(provenance["profile_identity_bound"])
        self.assertEqual("c" * 64, provenance["policy_fingerprint_sha256"])
        self.assertEqual(
            {
                "analysis_basis": "ive_human_reference_visual_montage_grammar_v2",
                "schema_version": 2,
                "path": "/tmp/ive-professional-montage-v2.json",
                "file_sha256": "a" * 64,
                "content_sha256": "b" * 64,
            },
            provenance["profile_identity"],
        )
        self.assertEqual(
            config.close_burst_policy_binding_sha256,
            provenance["close_burst_soft_policy_binding"]["binding_sha256"],
        )

        with self.assertRaisesRegex(ValueError, "missing its close-burst"):
            replace(config, close_burst_policy_binding_sha256=None)
        with self.assertRaisesRegex(ValueError, "binding SHA-256 mismatch"):
            replace(config, minimum_close_burst_duration_seconds=2.2)

    def test_close_only_burst_soft_preference_is_additive_and_saturated(self) -> None:
        config = _hash_bound_montage_config()
        absent = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                motion_indices=(1, 4),
            ),
            config=config,
        )
        absent_soft = absent["highlight_aggregate"]["soft_targets"][
            "close_only_multi_source_burst_event_p10"
        ]
        self.assertTrue(absent["passed"])
        self.assertEqual(0, absent_soft["candidate_event_count"])
        self.assertEqual(0.0, absent_soft["evidence_weighted_alignment_score"])
        self.assertFalse(absent_soft["absence_is_hard_failure"])

        qualified = _review(
            _analysis(
                (
                    "group_wide",
                    "close_up",
                    "close_up",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 4, 1, 2),
                motion_indices=(1, 2, 3, 5),
            ),
            config=config,
        )
        qualified_soft = qualified["highlight_aggregate"]["soft_targets"][
            "close_only_multi_source_burst_event_p10"
        ]
        self.assertTrue(qualified["passed"])
        self.assertEqual(1, qualified_soft["qualified_event_count"])
        self.assertAlmostEqual(0.30, qualified_soft["event_occurrence_support_weight"])
        self.assertAlmostEqual(
            0.30,
            qualified_soft["evidence_weighted_alignment_score"],
        )
        self.assertEqual(3, qualified_soft["events"][0]["shot_count"])
        self.assertEqual(2, qualified_soft["events"][0]["source_switch_count"])

    def test_i_am_shaped_highlights_clear_human_parity_duration_gates(self) -> None:
        # 50% wide, 16.7% medium, 33.3% close: deliberately shaped like the
        # accepted I AM floor, while retaining formation anchors and resets.
        pattern = (
            "group_wide",
            "medium",
            "close_up",
            "full_shot",
            "close_up",
            "group_wide",
        )
        fixture = _analysis(
            pattern + pattern,
            tuple(range(12)),
            motion_indices=(1, 2, 4, 7, 8, 10),
        )
        _split_into_two_highlights(fixture)

        result = _review(fixture)

        self.assertTrue(result["passed"], result["failure_codes"])
        aggregate = result["highlight_aggregate"]
        self.assertTrue(aggregate["passed"])
        self.assertEqual(2, aggregate["strict_highlight_section_count"])
        self.assertAlmostEqual(
            1.0 / 3.0,
            aggregate["observed"]["closeup_duration_fraction"],
        )
        self.assertAlmostEqual(
            0.5,
            aggregate["observed"]["nonwide_duration_fraction"],
        )
        self.assertEqual(
            "professional_montage_policy_v2_strict_highlight_duration_gate_v1",
            aggregate["contract"],
        )
        self.assertFalse(
            aggregate["soft_targets"]["over_close_past_target_is_rewarded"]
        )
        self.assertEqual(
            0.25,
            aggregate["policy_provenance"]["metrics"][
                "pooled_closeup_duration_product_minimum"
            ]["quantile"],
        )

    def test_failed_bang_shaped_wide_occupancy_fails_duration_products(self) -> None:
        # 75% wide, 10% medium, 15% close reproduces the failed BANG shape.
        shot_types = (
            "group_wide",
            "full_shot",
            "group_wide",
            "medium",
            "close_up",
            "group_wide",
            "full_shot",
            "group_wide",
            "group_wide",
            "close_up",
            "full_shot",
            "group_wide",
            "medium",
            "group_wide",
            "full_shot",
            "close_up",
            "group_wide",
            "full_shot",
            "group_wide",
            "group_wide",
        )
        result = _review(
            _analysis(
                shot_types,
                tuple(range(20)),
                motion_indices=(4, 9, 15),
            )
        )

        self.assertFalse(result["passed"])
        aggregate = result["highlight_aggregate"]
        self.assertAlmostEqual(
            0.15,
            aggregate["observed"]["closeup_duration_fraction"],
        )
        self.assertAlmostEqual(
            0.25,
            aggregate["observed"]["nonwide_duration_fraction"],
        )
        self.assertIn(
            "highlight_pooled_closeup_fraction_below_product_minimum",
            aggregate["failure_codes"],
        )
        self.assertIn(
            "highlight_pooled_nonwide_fraction_below_product_minimum",
            aggregate["failure_codes"],
        )

    def test_strong_highlight_cannot_rescue_another_sections_local_p10_failure(
        self,
    ) -> None:
        weak = (
            "group_wide",
            "medium",
            "close_up",
            "full_shot",
            "group_wide",
            "group_wide",
        )
        strong = (
            "group_wide",
            "close_up",
            "close_up",
            "full_shot",
            "close_up",
            "close_up",
        )
        fixture = _analysis(
            weak + strong,
            tuple(range(12)),
            motion_indices=(1, 2, 7, 8, 10, 11),
        )
        _split_into_two_highlights(fixture)

        result = _review(fixture)

        aggregate = result["highlight_aggregate"]
        self.assertGreater(
            aggregate["observed"]["closeup_duration_fraction"],
            aggregate["hard_targets"][
                "pooled_product_minimum_closeup_duration_fraction"
            ],
        )
        self.assertIn(
            "highlight_closeup_fraction_below_floor",
            aggregate["local_section_failures"]["0"],
        )
        self.assertNotIn("1", aggregate["local_section_failures"])
        self.assertFalse(result["passed"])

    def test_consecutive_wide_run_above_human_maximum_is_rejected(self) -> None:
        shot_types = (
            ("group_wide", "full_shot", "group_wide") * 3
            + (
                "medium",
                "close_up",
                "medium",
                "close_up",
                "group_wide",
                "medium",
                "close_up",
                "medium",
                "close_up",
                "medium",
                "close_up",
            )
        )
        result = _review(
            _analysis(
                shot_types,
                tuple(range(20)),
                motion_indices=(9, 10, 11, 14, 15, 16, 18, 19),
            )
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            9.0,
            result["highlight_aggregate"]["observed"][
                "maximum_consecutive_wide_run_seconds"
            ],
        )
        self.assertIn(
            "section_0:highlight_maximum_consecutive_wide_run_exceeded",
            result["failure_codes"],
        )

    def test_wide_anchor_then_two_short_closeup_bursts_passes(self) -> None:
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                motion_indices=(1, 4),
            )
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
            result["analysis_basis"],
        )
        section = result["sections"][0]
        self.assertTrue(section["formation_wide_anchor_present"])
        self.assertAlmostEqual(2.0 / 6.0, section["closeup_duration_fraction"])
        self.assertAlmostEqual(4.0 / 6.0, section["nonwide_duration_fraction"])
        self.assertEqual(2, section["nonwide_burst_boundary_count"])
        self.assertEqual(
            1.0,
            result["rendered_closeup_proxy"]["face_coverage_fraction"],
        )
        self.assertEqual(
            1.0,
            result["exact_anchor_metrics"][
                "chorus_section_early_wide_anchor_fraction"
            ],
        )
        self.assertEqual(
            1.0,
            result["exact_anchor_metrics"][
                "chorus_phrase_entry_wide_anchor_fraction"
            ],
        )

    def test_exact_phrase_entry_wide_floor_rejects_nonwide_phrase_entries(self) -> None:
        fixture = _analysis(
            ("medium", "close_up", "group_wide", "medium", "close_up", "full_shot"),
            (0, 1, 2, 3, 1, 2),
            establish=False,
        )

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "highlight_phrase_entry_wide_anchor_fraction_below_reference_floor",
            result["failure_codes"],
        )

    def test_exact_shot_clock_source_mismatch_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["stage_edit_exact_shot_clock"]["shots"][2]["source_index"] = 0

        with self.assertRaisesRegex(ValueError, "source differs"):
            _review(fixture)

    def test_unknown_in_first_three_consumes_early_anchor_slot(self) -> None:
        fixture = _analysis(
            ("medium", "close_up", "unknown", "group_wide", "close_up", "full_shot"),
            (0, 1, 2, 3, 1, 2),
            establish=False,
        )

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertEqual(
            0.0,
            result["exact_anchor_metrics"][
                "chorus_section_early_wide_anchor_fraction"
            ],
        )
        self.assertIn(
            "highlight_early_wide_anchor_fraction_below_reference_floor",
            result["failure_codes"],
        )

    def test_flat_all_wide_highlight_fails_even_when_timing_is_contiguous(self) -> None:
        result = _review(
            _analysis(
                ("group_wide",) * 6,
                (0, 1, 2, 0, 1, 2),
                motion_indices=(2,),
            )
        )

        self.assertFalse(result["passed"])
        failures = set(result["failure_codes"])
        self.assertIn("section_0:highlight_closeup_fraction_below_floor", failures)
        self.assertIn("section_0:highlight_nonwide_fraction_below_floor", failures)
        self.assertIn("section_0:highlight_nonwide_burst_missing", failures)

    def test_closeup_plan_claim_without_decoded_face_geometry_fails(self) -> None:
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                face_proxy=False,
            )
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "rendered_closeup_face_coverage_below_floor",
            result["failure_codes"],
        )

    def test_same_native_shot_slices_are_collapsed_before_fraction_review(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "group_wide",
                "medium",
                "close_up",
                "group_wide",
                "medium",
            ),
            (0, 0, 1, 2, 3, 1),
            role="verse_like",
        )
        choices = fixture["visual_timeline"]["choices"]
        choices[1]["shot_id"] = choices[0]["shot_id"]
        result = _review(fixture)

        self.assertTrue(result["passed"])
        self.assertEqual(5, result["visible_run_count"])
        self.assertEqual(2.0, result["visible_runs"][0]["duration_seconds"])

    def test_proxy_boundary_count_mismatch_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["rendered_proxy_review"]["boundaries"].pop()
        with self.assertRaisesRegex(ValueError, "boundary count"):
            _review(fixture)

    def test_generic_wide_without_group_evidence_is_not_a_formation_anchor(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] in {"group_wide", "full_shot"}:
                choice["detected_person_counts"] = [0, 0, 0]
                choice["formation_person_counts"] = [0, 0, 0]
                choice["formation_upper_body_counts"] = [0, 0, 0]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:formation_wide_anchor_evidence_missing",
            result["failure_codes"],
        )

    def test_strong_multi_person_wide_survives_tiny_pose_failure(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] in {"group_wide", "full_shot"}:
                choice["detected_person_counts"] = [4, 4, 4]
                choice["formation_person_counts"] = [4, 4, 4]
                choice["formation_upper_body_counts"] = [0, 0, 0]

        result = _review(fixture)

        self.assertTrue(result["passed"], result["failure_codes"])
        self.assertTrue(
            result["sections"][0]["formation_wide_anchor_present"]
        )

    def test_declared_section_must_be_fully_covered(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["section_plan"]["sections"][0]["end_seconds"] = 7.0
        with self.assertRaisesRegex(ValueError, "fully cover|end does not match"):
            _review(fixture)

    def test_central_close_claim_cannot_escape_proxy_face_review(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
            face_proxy=False,
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] == "close_up":
                choice["entry_shot_type"] = "medium"
                choice["exit_shot_type"] = "medium"

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertGreater(
            result["rendered_closeup_proxy"]["required_closeup_endpoint_count"],
            0,
        )
        self.assertIn(
            "rendered_closeup_face_coverage_below_floor",
            result["failure_codes"],
        )

    def test_nonwide_burst_requires_later_formation_reset(self) -> None:
        result = _review(
            _analysis(
                ("group_wide", "medium", "close_up", "medium", "close_up", "medium"),
                (0, 1, 2, 3, 1, 2),
            )
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:post_burst_formation_wide_reset_missing",
            result["failure_codes"],
        )

    def test_identity_transform_does_not_count_as_motion_accent(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for segment in fixture["editorial_motion"]["segments"]:
            segment["applied"] = True
            segment["keyframes"] = [
                {
                    "time_seconds": 0.0,
                    "zoom": 1.0,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "roll_degrees": 0.0,
                }
            ]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:highlight_motion_accent_missing",
            result["failure_codes"],
        )

    def test_constant_nonidentity_framing_does_not_count_as_motion_accent(
        self,
    ) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        for segment in fixture["editorial_motion"]["segments"]:
            segment["applied"] = True
            segment["keyframes"] = [
                {
                    "time_seconds": 0.0,
                    "zoom": 1.02,
                    "center_x": 0.51,
                    "center_y": 0.49,
                    "roll_degrees": 0.2,
                },
                {
                    "time_seconds": 0.75,
                    "zoom": 1.02,
                    "center_x": 0.51,
                    "center_y": 0.49,
                    "roll_degrees": 0.2,
                },
            ]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:highlight_motion_accent_missing",
            result["failure_codes"],
        )

    def test_v3_motion_accent_uses_only_typed_authorized_segments(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
            motion_indices=(0, 1, 2, 3, 4, 5),
        )
        plan = _attach_v3_motion_evidence(
            fixture,
            authorized_motion_indices=(4,),
            suppressed_motion_indices=(2,),
        )

        result = _review(fixture, render_plan=plan)

        self.assertEqual(result["sections"][0]["motion_accent_run_count"], 1)

    def test_v3_missing_evidence_and_full_plan_hash_tamper_fail_closed(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _attach_v3_motion_evidence(fixture)
        del fixture["dynamic_crop_motion_v3"]
        with self.assertRaises(ValueError):
            _review(fixture, render_plan=plan)

        plan = _attach_v3_motion_evidence(fixture)
        plan["output"] = "another-final-output.mp4"
        with self.assertRaises(ValueError):
            _review(fixture, render_plan=plan)

    def test_v3_timeline_self_hash_tamper_fails_closed(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _attach_v3_motion_evidence(fixture)
        fixture["dynamic_crop_motion_v3"]["motion_event_applied_count"] = 0

        with self.assertRaises(ValueError):
            _review(fixture, render_plan=plan)

    def test_v2_marker_preserves_ordinary_editorial_motion_review(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
            motion_indices=(2,),
        )
        fixture["dynamic_crop_version"] = 2

        result = _review(fixture)

        self.assertEqual(result["sections"][0]["motion_accent_run_count"], 1)

    def test_same_derivative_camera_family_does_not_create_a_multicam_burst(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["source_derivative_diversity"]["family_ids"] = [0, 0, 0, 0]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:highlight_nonwide_burst_missing",
            result["failure_codes"],
        )

    def test_render_plan_is_bound_to_selected_visual_timeline(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        passing = _review(fixture, render_plan=plan)
        self.assertTrue(passing["render_plan_binding"]["passed"])

        plan["segments"][2]["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "source differs"):
            _review(fixture, render_plan=plan)

    def test_render_plan_is_required_for_strict_proxy_identity(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )

        with self.assertRaisesRegex(ValueError, "render_plan is required"):
            review_rendered_montage_grammar(fixture)

    def test_missing_explicit_source_mapping_fails_closed(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture.pop("source_index_mapping")

        with self.assertRaisesRegex(ValueError, "source_index_mapping"):
            _review(fixture, render_plan=plan)

    def test_eligible_request_gap_is_applied_to_plan_and_proxy_sources(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        mapping = fixture["source_index_mapping"]
        mapping["eligible_to_request"] = [0, 2, 3, 4]
        mapping["request_to_eligible"] = [0, None, 1, 2, 3]
        plan["sources"] = [f"source-{index}.mp4" for index in range(5)]
        translated = [0, 2, 3, 4, 2, 3]
        mapping["eligible_sources"] = [
            {
                "eligible_source_index": eligible_index,
                "request_source_index": request_index,
                "source": plan["sources"][request_index],
            }
            for eligible_index, request_index in enumerate(
                mapping["eligible_to_request"]
            )
        ]
        for segment, source_index in zip(
            plan["segments"], translated, strict=True
        ):
            segment["source_index"] = source_index
        boundaries = fixture["rendered_proxy_review"]["boundaries"]
        for index, boundary in enumerate(boundaries):
            boundary["outgoing_source_index"] = translated[index]
            boundary["incoming_source_index"] = translated[index + 1]

        result = _review(fixture, render_plan=plan)

        self.assertTrue(result["passed"])
        self.assertEqual(
            translated,
            result["render_plan_binding"]["source_index_binding"][
                "translated_choice_source_indices"
            ],
        )

    def test_dissolve_closeup_uses_completed_transition_endpoint(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        for metric in boundary["frame_metrics"]:
            if metric["offset"] != 3:
                metric["subject_geometry"]["face_bbox_xywh"] = None

        result = _review(fixture, render_plan=plan)

        self.assertTrue(result["passed"])
        close_segment = result["rendered_closeup_proxy"]["records"][0]
        incoming = close_segment["endpoints"][0]
        self.assertEqual("incoming", incoming["side"])
        self.assertEqual("face_dissolve", incoming["transition_kind"])
        self.assertEqual(3, incoming["offset"])
        self.assertEqual(boundary["cut_frame_index"] + 3, incoming["frame_index"])
        self.assertTrue(incoming["face_geometry_present"])

    def test_dissolve_forged_offset_zero_metric_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        boundary["frame_metrics"][2]["offset"] = 0
        boundary["frame_metrics"][2]["frame_index"] = boundary["cut_frame_index"]

        with self.assertRaisesRegex(ValueError, "exact sample"):
            _review(fixture, render_plan=plan)

    def test_dissolve_wrong_declared_sample_offsets_are_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        boundary["sample_offsets"] = [-2, -1, 2, 3]

        with self.assertRaisesRegex(ValueError, "sample offsets"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_source_pair_is_bound_to_render_plan(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture["rendered_proxy_review"]["boundaries"][1][
            "incoming_source_index"
        ] = 0

        with self.assertRaisesRegex(ValueError, "incoming_source_index"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_transition_kind_is_bound_to_render_plan(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture["rendered_proxy_review"]["boundaries"][1][
            "transition_kind"
        ] = "face_dissolve"

        with self.assertRaisesRegex(ValueError, "transition kind"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_frame_window_is_exactly_bound(self) -> None:
        mutations = (
            (
                "boundary_index",
                lambda boundary: boundary.__setitem__("boundary_index", 4),
                "indices",
            ),
            (
                "cut_frame_index",
                lambda boundary: boundary.__setitem__("cut_frame_index", 61),
                "cut_frame_index",
            ),
            (
                "transition_frame_count",
                lambda boundary: boundary.__setitem__(
                    "transition_frame_count", 1
                ),
                "transition_frame_count",
            ),
            (
                "frame_indices",
                lambda boundary: boundary["frame_indices"].__setitem__(2, 61),
                "frame indices",
            ),
        )
        for label, mutate, error in mutations:
            with self.subTest(field=label):
                fixture = _analysis(
                    (
                        "group_wide",
                        "medium",
                        "close_up",
                        "full_shot",
                        "medium",
                        "close_up",
                    ),
                    (0, 1, 2, 3, 1, 2),
                )
                plan = _render_plan(fixture)
                mutate(fixture["rendered_proxy_review"]["boundaries"][1])

                with self.assertRaisesRegex(ValueError, error):
                    _review(fixture, render_plan=plan)

    def test_tiny_face_box_does_not_prove_a_closeup(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for boundary in fixture["rendered_proxy_review"]["boundaries"]:
            for metric in boundary["frame_metrics"]:
                metric["subject_geometry"]["face_bbox_xywh"] = [0.4, 0.4, 0.05, 0.05]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertEqual(
            0.0,
            result["rendered_closeup_proxy"]["face_coverage_fraction"],
        )

    def test_automatic_ive_duration_and_wide_run_use_interior_geometry(
        self,
    ) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
            motion_indices=(1, 2, 4, 5),
        )
        _attach_segment_interior_geometry(fixture)

        result = _review(fixture)

        self.assertTrue(result["passed"], result["failure_codes"])
        interior = result["rendered_segment_interior_geometry"]
        self.assertTrue(interior["required"])
        self.assertTrue(interior["passed"])
        self.assertEqual(interior["unknown_sample_count"], 0)
        self.assertAlmostEqual(
            interior["whole_song_observed"]["close_duration_fraction"],
            2.0 / 6.0,
        )
        self.assertAlmostEqual(
            result["sections"][0]["closeup_duration_fraction"],
            2.0 / 6.0,
        )

    def test_claimed_close_with_wide_interior_sample_cannot_pass(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
            motion_indices=(1, 2, 4, 5),
        )
        _attach_segment_interior_geometry(
            fixture,
            sample_classes_by_segment={2: ("close", "wide", "close")},
        )

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "rendered_nonwide_claim_contradicted_by_wide_interior_geometry",
            result["failure_codes"],
        )
        self.assertGreater(
            result["rendered_segment_interior_geometry"][
                "nonwide_claim_wide_contradiction_count"
            ],
            0,
        )
        self.assertLess(
            result["sections"][0]["closeup_duration_fraction"],
            2.0 / 6.0,
        )

    def test_automatic_ive_missing_interior_geometry_fails_closed(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["request"] = {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        }

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "rendered_segment_interior_geometry_evidence_missing",
            result["failure_codes"],
        )

    def test_interior_geometry_hash_and_exact_schedule_tampering_is_rejected(
        self,
    ) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        _attach_segment_interior_geometry(fixture)
        evidence = fixture["rendered_proxy_review"][
            "segment_interior_geometry"
        ]
        evidence["segments"][2]["samples"][1]["frame_index"] += 1

        with self.assertRaisesRegex(ValueError, "evidence SHA-256 mismatch"):
            _review(fixture)

        core = dict(evidence)
        core.pop("evidence_sha256")
        evidence["evidence_sha256"] = hashlib.sha256(
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "frame_index differs from schedule"):
            _review(fixture)

    def test_same_proxy_metadata_with_different_content_is_rejected(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first-proxy.mp4"
            second = Path(directory) / "second-proxy.mp4"
            first.write_bytes(b"first-proxy-content")
            second.write_bytes(b"other-proxy-content")
            common_mtime_ns = first.stat().st_mtime_ns
            os.utime(first, ns=(common_mtime_ns, common_mtime_ns))
            os.utime(second, ns=(common_mtime_ns, common_mtime_ns))
            self.assertEqual(first.stat().st_size, second.stat().st_size)
            self.assertEqual(first.stat().st_mtime_ns, second.stat().st_mtime_ns)

            fixture["rendered_proxy_review"]["proxy"]["content_sha256"] = (
                verified_file_content_sha256(first)
            )
            _attach_segment_interior_geometry(fixture)
            fixture["rendered_proxy_review"]["proxy"]["content_sha256"] = (
                verified_file_content_sha256(second)
            )

            with self.assertRaisesRegex(ValueError, "proxy identity differs"):
                _review(fixture)

    def test_same_source_path_replacement_breaks_content_binding(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
            motion_indices=(1, 2, 4, 5),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_paths = tuple(
                root / f"source-{index}.mp4" for index in range(4)
            )
            for index, path in enumerate(source_paths):
                path.write_bytes(bytes([65 + index]) * 4096)
            for index, item in enumerate(
                fixture["source_index_mapping"]["eligible_sources"]
            ):
                item["source"] = str(source_paths[index])
            fixture["rendered_proxy_review"]["proxy"]["content_sha256"] = (
                hashlib.sha256(b"synthetic decoded proxy").hexdigest()
            )
            _attach_segment_interior_geometry(
                fixture,
                source_paths=source_paths,
                use_real_content=True,
            )
            plan = _render_plan(fixture, source_paths=source_paths)
            before = review_rendered_montage_grammar(
                fixture,
                render_plan=plan,
                config=RenderedMontageGrammarConfig(),
            )
            self.assertTrue(before["passed"], before["failure_codes"])

            target = source_paths[2]
            target_stat = target.stat()
            replacement = root / "replacement.mp4"
            replacement.write_bytes(b"Z" * target_stat.st_size)
            os.utime(
                replacement,
                ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns),
            )
            replacement.replace(target)
            self.assertEqual(target.stat().st_size, target_stat.st_size)
            self.assertEqual(target.stat().st_mtime_ns, target_stat.st_mtime_ns)

            with self.assertRaisesRegex(ValueError, "RenderPlan binding hash mismatch"):
                review_rendered_montage_grammar(
                    fixture,
                    render_plan=plan,
                    config=RenderedMontageGrammarConfig(),
                )


if __name__ == "__main__":
    unittest.main()
