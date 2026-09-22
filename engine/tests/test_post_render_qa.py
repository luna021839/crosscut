from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Callable
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.cut_local_sync import (
    ACTIVATION as CUT_LOCAL_SYNC_ACTIVATION,
    ANALYSIS_BASIS as CUT_LOCAL_SYNC_ANALYSIS_BASIS,
    SCHEMA_VERSION as CUT_LOCAL_SYNC_SCHEMA_VERSION,
    CutLocalSyncConfig,
    _SERIALIZATION_POLICY as CUT_LOCAL_SYNC_SERIALIZATION_POLICY,
    _bound_timewarp_payloads,
    _boundary_contracts,
    _canonical_sha256,
    _used_source_indices,
)
from crosscut.ive import AspectRatio
from crosscut.post_render_qa import (
    PostRenderArtifactSet,
    PostRenderQAConfig,
    PostRenderQAReport,
    PostRenderQAVerdict,
    _IssueCollector,
    _boundary_render_plan_from_serialized,
    _cadence_and_repetition_metrics,
    _selected_boundary_repair_contracts,
    _sync_quality_metrics,
    _transform_metrics,
    _validate_selection,
    _validate_plan_invariants,
    cleanup_post_render_qa_snapshot,
    discover_completed_artifact_sets,
    evaluate_post_render_qa,
    materialize_post_render_qa_snapshot,
)
from crosscut.group_boundary_grammar_adapter import (
    adapt_group_boundary_grammar_evidence,
)
from crosscut.ive_product_quality_floor import (
    APP_RESULT_EXPOSURE_CONTRACT,
)
from crosscut.pipeline import _source_index_mapping
from crosscut.professional_montage_grammar import (
    PROFILE_ANALYSIS_BASIS as MONTAGE_PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION as MONTAGE_PROFILE_SCHEMA_VERSION,
    EditorialTransformEvidence,
    IntroducedBoundaryContinuityEvidence,
    MontageSectionWindow,
    ProfessionalBoundaryGrammarEvidence,
    assess_professional_boundary_grammar,
)
from crosscut.rendered_montage_grammar_review import (
    RenderedMontageGrammarConfig,
    review_rendered_montage_grammar,
)
from crosscut.render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    segment_frame_counts,
)
from crosscut.rendered_proxy_review import (
    OutputSubjectGeometry,
    _segment_interior_geometry_evidence,
    _segment_interior_schedule,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from crosscut.stage_edit_windowed_beam_redecode import (
    STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
)


_TIMESTAMP = "20260823-123456"
_BASE = f"IVE-fixture-song-{_TIMESTAMP}"
_FPS = 30.0
_DURATION = 32.0
_FRAME_COUNT = 960
_REGULAR_SEGMENT_DURATION = 3.5
_MONTAGE_TEST_CONFIG = RenderedMontageGrammarConfig(
    burst_maximum_run_seconds=4.0,
    # The compact nine-shot fixture predates the immutable 20-reference
    # cadence profile.  Production/default-config cadence is covered by the
    # dedicated rendered-grammar tests below; this legacy fixture keeps its
    # explicit synthetic threshold instead of pretending it has extra cuts.
    highlight_product_minimum_source_switches_per_minute=15.0,
)


def _registered_qa_config() -> PostRenderQAConfig:
    return PostRenderQAConfig(
        rendered_montage_grammar=_MONTAGE_TEST_CONFIG
    )


def _segment_bounds(index: int) -> tuple[float, float]:
    start = index * _REGULAR_SEGMENT_DURATION
    end = (
        (index + 1) * _REGULAR_SEGMENT_DURATION
        if index < 8
        else _DURATION
    )
    return (start, end)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _transition(index: int) -> dict[str, object]:
    return {
        "kind": "hard_cut",
        "frame_count": 0,
        "duration_seconds": 0.0,
        "reason": f"rhythm_motion_pose_hard_cut_{index}",
        "face_mismatch": 0.08,
        "tier": "C",
        "appearance_min_cosine_similarity": None,
        "future_pose_max_mismatch": 0.12,
        "future_silhouette_max_mismatch": 0.14,
    }


def _repair(index: int, *, offset: int = 0) -> dict[str, object]:
    boundary = _REGULAR_SEGMENT_DURATION * (index + 1)
    return {
        "boundary_index": index,
        "outgoing_source_index": index % 8,
        "incoming_source_index": (index + 1) % 8,
        "result": {
            "nominal_boundary_seconds": boundary,
            "selected_frame_offset": offset,
            "selected_boundary_seconds": boundary + offset / _FPS,
            "changed": offset != 0,
            "selected": {
                "frame_offset": offset,
                "boundary_seconds": boundary + offset / _FPS,
                "beat_deviation_seconds": abs(offset) / _FPS,
                "beat_deviation_periods": abs(offset) / 15.0,
            },
        },
        "selected_visual_evidence": {"total_cost": 0.08},
    }


def _promote_boundary_repairs_to_actual_pose_v3(
    analysis: dict,
    selection: dict,
) -> None:
    selected_index = selection["selected_candidate_index"]
    selected_candidate = next(
        item
        for item in selection["complete_candidates"]
        if item.get("candidate_index") == selected_index
    )
    plans = (
        analysis["visual_boundary_repairs"],
        selected_candidate["boundary_repairs"],
    )
    for repair_plan in plans:
        repair_plan.update(
            {
                "analysis_basis": (
                    "decoded_plus_minus_three_frame_formation_layout_and_"
                    "actual_pose_boundary_repair_v3"
                ),
                "geometry_provider_attempt_count": 96,
                "geometry_provider_failure_count": 0,
            }
        )
        for repair in repair_plan["repairs"]:
            repair["selected_visual_evidence"].update(
                {
                    "pose_mismatch": None,
                    "pose_basis": "actual_pose_pair_unavailable",
                    "exact_pose_mismatch": None,
                    "nearest_temporal_pose_mismatch": None,
                    "structure_mismatch": 0.08,
                    "evidence_coverage": {
                        "exact_pose_available": [False, False],
                        "temporal_pose_available": [False, False],
                        "temporal_pose_offsets_frames": [None, None],
                        "formation_person_counts": [1, 1],
                        "formation_upper_body_counts": [0, 0],
                        "person_layout_available": [True, True],
                        "silhouette_available": [True, True],
                        "professional_evidence_basis": (
                            "formation_assignment_plus_screen_geometry"
                        ),
                        "professional_gate_passed": True,
                        "missing_mask": [
                            "outgoing_exact_actual_pose",
                            "incoming_exact_actual_pose",
                            "outgoing_temporal_actual_pose",
                            "incoming_temporal_actual_pose",
                            "outgoing_formation_upper_body_pose",
                            "incoming_formation_upper_body_pose",
                        ],
                    },
                }
            )


def _promote_boundary_repairs_to_unbound_temporal_pose_motion_v1(
    analysis: dict,
    selection: dict,
) -> None:
    _promote_boundary_repairs_to_actual_pose_v3(analysis, selection)
    selected_index = selection["selected_candidate_index"]
    selected_candidate = next(
        item
        for item in selection["complete_candidates"]
        if item.get("candidate_index") == selected_index
    )
    configuration_fingerprint = _canonical_sha256(
        {
            "contract": "visual_boundary_temporal_pose_motion_preference_v1",
            "threshold_identity_sha256": None,
            "preference_weight": 0.0,
        }
    )
    configuration = {
        "contract": "visual_boundary_temporal_pose_motion_preference_v1",
        "calibration_status": "calibration_unbound",
        "threshold_values": None,
        "threshold_identity_sha256": None,
        "preference_weight": 0.0,
        "configuration_fingerprint_sha256": configuration_fingerprint,
    }
    plans = (
        analysis["visual_boundary_repairs"],
        selected_candidate["boundary_repairs"],
    )
    for repair_plan in plans:
        repair_plan["analysis_basis"] += "+temporal_pose_motion_audit_v1"
        repair_plan["temporal_pose_motion_configuration"] = deepcopy(
            configuration
        )
        for repair in repair_plan["repairs"]:
            evidence = repair["selected_visual_evidence"]
            evidence["base_total_cost"] = evidence["total_cost"]
            evidence["temporal_pose_motion"] = {
                "contract": "scalar_temporal_pose_motion_boundary_evidence_v1",
                "calibration_status": "calibration_unbound",
                "analysis_executed": False,
                "input_valid": False,
                "qualifies": False,
                "classification": "calibration_unbound",
                "failure_reasons": [
                    "temporal_pose_motion_thresholds_unbound"
                ],
                "feature_dimension": None,
                "outgoing_sample_count": 0,
                "incoming_sample_count": 0,
                "outgoing_boundary_motion_magnitude": None,
                "incoming_boundary_motion_magnitude": None,
                "endpoint_pose_cosine": None,
                "velocity_direction_cosine": None,
                "phase_compatibility": None,
                "confidence": 0.0,
                "static_pose_match": False,
                "dynamic_pose_phase_match": False,
                "direction_evidence_available": False,
                "dynamic_direction_match_proven": False,
                "soft_preference_applied": 0.0,
                "threshold_values": None,
                "threshold_identity_sha256": None,
                "preference_weight": 0.0,
                "configuration_fingerprint_sha256": (
                    configuration_fingerprint
                ),
            }


def _bind_synthetic_temporal_pose_motion_fixture(
    analysis: dict,
    selection: dict,
) -> None:
    """Exercise the injected QA contract without claiming real calibration."""

    _promote_boundary_repairs_to_unbound_temporal_pose_motion_v1(
        analysis,
        selection,
    )
    selected_index = selection["selected_candidate_index"]
    selected_candidate = next(
        item
        for item in selection["complete_candidates"]
        if item.get("candidate_index") == selected_index
    )
    thresholds = {
        "minimum_boundary_motion_magnitude": 0.5,
        "minimum_endpoint_pose_cosine": 0.9,
        "minimum_velocity_direction_cosine": 0.8,
        "minimum_phase_compatibility": 0.85,
        "pose_norm_epsilon": 1e-8,
    }
    threshold_identity = _canonical_sha256(
        {
            "contract": "temporal_pose_motion_thresholds_v1",
            "values": thresholds,
        }
    )
    configuration_fingerprint = _canonical_sha256(
        {
            "contract": "visual_boundary_temporal_pose_motion_preference_v1",
            "threshold_identity_sha256": threshold_identity,
            "preference_weight": 1.0,
        }
    )
    configuration = {
        "contract": "visual_boundary_temporal_pose_motion_preference_v1",
        "calibration_status": "calibration_bound",
        "threshold_values": thresholds,
        "threshold_identity_sha256": threshold_identity,
        "preference_weight": 1.0,
        "configuration_fingerprint_sha256": configuration_fingerprint,
    }
    plans = (
        analysis["visual_boundary_repairs"],
        selected_candidate["boundary_repairs"],
    )
    for repair_plan in plans:
        repair_plan["temporal_pose_motion_configuration"] = deepcopy(
            configuration
        )
        for repair in repair_plan["repairs"]:
            evidence = repair["selected_visual_evidence"]
            evidence["base_total_cost"] = 0.08
            evidence["total_cost"] = 0.075
            evidence["temporal_pose_motion"] = {
                "contract": "scalar_temporal_pose_motion_boundary_evidence_v1",
                "calibration_status": "calibration_bound",
                "analysis_executed": True,
                "input_valid": True,
                "qualifies": True,
                "classification": "dynamic_pose_phase_match",
                "failure_reasons": [],
                "feature_dimension": 60,
                "outgoing_sample_count": 2,
                "incoming_sample_count": 2,
                "outgoing_boundary_motion_magnitude": 1.2,
                "incoming_boundary_motion_magnitude": 1.1,
                "endpoint_pose_cosine": 0.98,
                "velocity_direction_cosine": 0.95,
                "phase_compatibility": 0.975,
                "confidence": 0.975,
                "static_pose_match": False,
                "dynamic_pose_phase_match": True,
                "direction_evidence_available": True,
                "dynamic_direction_match_proven": True,
                "soft_preference_applied": 0.005,
                "threshold_values": thresholds,
                "threshold_identity_sha256": threshold_identity,
                "preference_weight": 1.0,
                "configuration_fingerprint_sha256": (
                    configuration_fingerprint
                ),
            }


def _proxy_boundary(index: int) -> dict[str, object]:
    cut_frame = int(round(_REGULAR_SEGMENT_DURATION * (index + 1) * _FPS))
    geometry = {
        "face_count": 1,
        "face_bbox_xywh": [0.40, 0.20, 0.20, 0.30],
        "face_backend": "fixture",
        "silhouette_bbox_xywh": [0.30, 0.10, 0.40, 0.80],
        "silhouette_centroid_xy": [0.50, 0.50],
        "silhouette_confidence": 0.9,
        "silhouette_backend": "fixture",
    }
    return {
        "boundary_index": index,
        "cut_frame_index": cut_frame,
        "outgoing_source_index": index % 8,
        "incoming_source_index": (index + 1) % 8,
        "transition_kind": "hard_cut",
        "transition_frame_count": 0,
        "transition_duration_seconds": 0.0,
        "sample_offsets": [-2, -1, 0, 1],
        "frame_indices": [cut_frame - 2, cut_frame - 1, cut_frame, cut_frame + 1],
        "frame_metrics": [
            {
                "offset": offset,
                "frame_index": cut_frame + offset,
                "quality": 0.80,
                "subject_geometry": deepcopy(geometry),
            }
            for offset in (-2, -1, 0, 1)
        ],
        "discontinuities": {
            "face_scale_discontinuity": 0.05,
            "face_position_discontinuity": 0.05,
            "silhouette_scale_discontinuity": 0.04,
            "silhouette_position_discontinuity": 0.04,
        },
        "weighted_score": 82.0,
        "passed": True,
        "issue_reasons": [],
    }


def _final_review() -> dict[str, object]:
    return {
        "schema_version": 2,
        "analysis": "decoded_output_proxy_all_transition_quality_review_v2",
        "proxy": {
            "path": None,
            "width": 854,
            "height": 480,
            "fps": _FPS,
            "decoded_frame_count": _FRAME_COUNT,
            "duration_seconds": _DURATION,
            "render_codec": "libx264",
            "content_sha256": hashlib.sha256(
                b"synthetic decoded proxy fixture"
            ).hexdigest(),
            "temporary_proxy_removed_after_review": True,
        },
        "overall_quality": {
            "count": _FRAME_COUNT,
            "minimum": 0.65,
            "p10": 0.72,
            "median": 0.80,
            "p90": 0.88,
            "maximum": 0.92,
        },
        "source_quality": {
            "per_source": [
                {
                    "source_index": index,
                    "quality": {
                        "p10": 0.65 if index == 0 else 0.75 if index == 1 else 0.70,
                        "median": (
                            0.76 if index == 0 else 0.84 if index == 1 else 0.80
                        ),
                    },
                }
                for index in range(8)
            ],
            "median_span": 0.08,
            "p10_span": 0.10,
            "maximum_allowed_median_span": 0.22,
        },
        "transition_count": 8,
        "reviewed_boundary_count": 8,
        "hard_cut_count": 8,
        "dissolve_count": 0,
        "boundary_score": {
            "distribution": {"count": 8, "p10": 82.0, "median": 82.0},
            "p10": 82.0,
            "issue_count": 0,
            "issue_fraction": 0.0,
        },
        "boundary_issue_indices": [],
        "boundaries": [_proxy_boundary(index) for index in range(8)],
        "geometry_provider_failure_count": 0,
        "weighted_score": 81.0,
        "passed": True,
    }


def _promote_review_to_domain_aware_v3(
    review: dict[str, object],
    plan: dict[str, object],
) -> None:
    from crosscut.rendered_proxy_review import ProxyReviewConfig
    review["review_policy"] = ProxyReviewConfig().to_dict()
    transitions = plan["transitions"]
    assert isinstance(transitions, list)
    boundaries = review["boundaries"]
    assert isinstance(boundaries, list)
    for index, boundary in enumerate(boundaries):
        assert isinstance(boundary, dict)
        transition = transitions[index]
        assert isinstance(transition, dict)
        boundary.update(
            {
                "review_domain": "introduced_cross_source",
                "transition_reason": transition["reason"],
                "transition_tier": transition["tier"],
                "source_clock_contiguous": False,
                "technical_integrity_issue_reasons": [],
            }
        )
    review.update(
        {
            "schema_version": 3,
            "analysis": "decoded_output_proxy_domain_aware_quality_review_v3",
            "introduced_cross_source_count": len(boundaries),
            "introduced_cross_source_boundary_issue_indices": [],
            "tier_d_continuity_count": 0,
            "native_source_cut_count": 0,
            "introduced_cross_source_technical_issue_reasons": [],
            "boundary_domains": {
                "introduced_cross_source": {
                    "domain": "introduced_cross_source",
                    "count_basis": "introduced_cross_source",
                    "boundary_count": len(boundaries),
                    "issue_count": 0,
                    "issue_fraction": 0.0,
                    "issue_indices": [],
                    "score": {
                        "distribution": {
                            "count": len(boundaries),
                            "minimum": 82.0,
                            "p10": 82.0,
                            "median": 82.0,
                            "p90": 82.0,
                            "maximum": 82.0,
                        },
                        "p10": 82.0,
                        "evidence_status": "OBSERVED",
                    },
                    "pass_policy": (
                        "editorial_boundary_p10_and_issue_fraction_gate"
                    ),
                    "passed": True,
                },
                "tier_d_continuity": {
                    "domain": "tier_d_continuity",
                    "count_basis": "tier_d_continuity",
                    "boundary_count": 0,
                    "issue_count": 0,
                    "issue_fraction": 0.0,
                    "issue_indices": [],
                    "score": {
                        "distribution": {
                            "count": 0,
                            "minimum": None,
                            "p10": None,
                            "median": None,
                            "p90": None,
                            "maximum": None,
                        },
                        "p10": None,
                        "evidence_status": "NOT_APPLICABLE",
                    },
                    "pass_policy": "zero_technical_integrity_issues",
                    "passed": True,
                },
                "native_source_cut": {
                    "domain": "native_source_cut",
                    "count_basis": "native_source_cut",
                    "boundary_count": 0,
                    "issue_count": 0,
                    "issue_fraction": 0.0,
                    "issue_indices": [],
                    "score": {
                        "distribution": {
                            "count": 0,
                            "minimum": None,
                            "p10": None,
                            "median": None,
                            "p90": None,
                            "maximum": None,
                        },
                        "p10": None,
                        "evidence_status": "NOT_APPLICABLE",
                    },
                    "pass_policy": "zero_technical_integrity_issues",
                    "passed": True,
                },
            },
        }
    )


def _timeline_review() -> dict[str, object]:
    source_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    return {
        "passed": True,
        "weighted_score": 94.0,
        "score_deductions": {},
        "segment_count": 9,
        "boundary_count": 8,
        "internal_scene_cut_count": 15,
        "comparable_visual_boundary_count": 23,
        "visual_shot_count": 24,
        "visual_shot_durations_seconds": [4.0 / 3.0] * 24,
        "visual_shot_median_seconds": 4.0 / 3.0,
        "total_duration_seconds": _DURATION,
        "source_switch_count": 8,
        "source_switch_rate": 8 / 23,
        "source_switches_per_minute": 15.0,
        "unique_source_count": 8,
        "available_source_count": 8,
        "required_unique_source_count": 8,
        "source_duration_seconds": {
            "0": 7.5,
            **{str(index): 3.5 for index in range(1, 8)},
        },
        "per_source_share": {
            "0": 7.5 / _DURATION,
            **{
                str(index): _REGULAR_SEGMENT_DURATION / _DURATION
                for index in range(1, 8)
            },
        },
        "source_runs": [
            {
                "run_index": index,
                "source_id": str(source_index),
                "first_segment_position": index,
                "last_segment_position": index,
                "start_seconds": _segment_bounds(index)[0],
                "end_seconds": _segment_bounds(index)[1],
                "duration_seconds": (
                    _segment_bounds(index)[1] - _segment_bounds(index)[0]
                ),
            }
            for index, source_index in enumerate(source_sequence)
        ],
        "source_run_median_seconds": _REGULAR_SEGMENT_DURATION,
        "longest_source_run_seconds": 4.0,
        "ping_pong_count": 0,
        "ping_pongs": [],
        "short_flash_count": 0,
        "short_flash_run_indices": [],
        "scene_cut_risk_count": 0,
        "scene_cut_risk_boundary_indices": [],
        "boundary_side_instability_count": 0,
        "outgoing_boundary_side_instability_indices": [],
        "incoming_boundary_side_instability_indices": [],
        "boundaries": [
            {
                "boundary_index": index,
                "outgoing_segment_index": index,
                "incoming_segment_index": index + 1,
                "at_seconds": _REGULAR_SEGMENT_DURATION * (index + 1),
                "outgoing_source_id": str(source_sequence[index]),
                "incoming_source_id": str(source_sequence[index + 1]),
                "source_switch": True,
                "visual_boundary": True,
                "scene_cut_risk": False,
                "available_metrics": ["scale", "focus", "layout"],
                "mismatches": {"scale": 0.05, "focus": 0.08, "layout": 0.10},
                "exceeded_metrics": [],
            }
            for index in range(8)
        ],
        "mismatch_summaries": {},
        "hard_failures": [],
        "warnings": [],
        "reference_profile": {},
    }


def _editorial_motion() -> dict[str, object]:
    segments: list[dict[str, object]] = []
    reasons_by_segment = {
        0: ["face_scale_position_matched_hard_cut"],
        1: ["face_scale_position_matched_hard_cut"],
        2: ["orb_affine_subject_matched_hard_cut"],
        3: ["orb_affine_subject_matched_hard_cut"],
        4: ["reference_calibrated_same_source_interior_zoom_pulse"],
        6: ["reference_calibrated_same_source_interior_zoom_pulse"],
    }
    for index in range(9):
        applied = index in {0, 2, 4, 6}
        zoom = 1.04 + 0.01 * (index // 2) if applied else 1.0
        segments.append(
            {
                "segment_index": index,
                "source_index": index % 8,
                "applied": applied,
                "reasons": reasons_by_segment.get(index, []),
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
                            "zoom": zoom,
                            "center_x": 0.51 if index in {0, 2} else 0.5,
                            "center_y": 0.5,
                            "roll_degrees": 0.0,
                        },
                    ]
                    if applied
                    else []
                ),
            }
        )
    return {
        "analysis_basis": "fixture_editorial_motion_v1",
        "applied_segment_count": 4,
        "boundary_match_count": 2,
        "affine_boundary_match_count": 1,
        "zoom_event_count": 4,
        "pan_event_count": 2,
        "roll_event_count": 0,
        "interior_zoom_pulse_count": 2,
        "section_directed": True,
        "blocked_dissolve_segment_count": 0,
        "skipped_boundary_reasons": [],
        "segments": segments,
    }


def _fixture_payloads(root: Path) -> tuple[PostRenderArtifactSet, dict, dict, dict]:
    artifacts = PostRenderArtifactSet.from_analysis_path(
        root / f"{_BASE}-analysis.json"
    )
    source_paths = tuple(root / f"source-{index}.mp4" for index in range(8))
    for index, source_path in enumerate(source_paths):
        source_path.write_bytes(
            f"non-empty synthetic source fixture {index}\n".encode("utf-8")
        )
    transitions = [_transition(index) for index in range(8)]
    source_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    segments = [
        {
            "source_index": source_index,
            "source_start": _segment_bounds(index)[0],
            "source_end": _segment_bounds(index)[1],
            "source_local_start": _segment_bounds(index)[0],
            "source_local_end": _segment_bounds(index)[1],
            "source_duration": None,
            "master_start": _segment_bounds(index)[0],
            "duration": _segment_bounds(index)[1] - _segment_bounds(index)[0],
            "effective_source_duration": (
                _segment_bounds(index)[1] - _segment_bounds(index)[0]
            ),
            "source_playback_rate": 1.0,
            "crop": None,
            "pan_crop": None,
            "editorial_transform": None,
            "look_adjustment": None,
        }
        for index, source_index in enumerate(source_sequence)
    ]
    plan = {
        "sources": [str(source_path) for source_path in source_paths],
        "segments": segments,
        "reference_source": 0,
        "aspect_ratio": AspectRatio.LANDSCAPE.value,
        "output": str(artifacts.video_path),
        "master_audio_start": 0.0,
        "width": 1920,
        "height": 1080,
        "fps": _FPS,
        "transitions": transitions,
        "duration": _DURATION,
        "render_duration": _DURATION,
        "output_frame_count": _FRAME_COUNT,
        "source_usage": {
            "distinct_source_count": 8,
            "used_source_indices": list(range(8)),
            "total_segment_count": 9,
            "total_duration_seconds": _DURATION,
        },
    }
    final_review = _final_review()
    repairs = [_repair(index) for index in range(8)]
    complete_candidates = [
        {
            "candidate_index": 0,
            "boundary_repairs": {"repairs": deepcopy(repairs)},
        }
    ] + [{"candidate_index": index} for index in range(1, 8)]
    selection = {
        "analysis_basis": "sequential_decoded_proxy_candidate_rerank_v1",
        "storage_policy": {
            "one_proxy_at_a_time": True,
            "temporary_proxy_removed_after_each_review": True,
            "decoded_frames_persisted": False,
        },
        "requested_review_count": 8,
        "reviewed_candidate_count": 8,
        "minimum_required_candidate_count": 8,
        "minimum_candidate_count_satisfied": True,
        "successful_review_count": 8,
        "passing_candidate_count": 8,
        "selected_candidate_index": 0,
        "complete_candidate_count": 8,
        "exact_successful_decoded_review_count_satisfied": True,
        "exact_complete_candidate_count_satisfied": True,
        "final_selected_review": deepcopy(final_review),
        "final_selected_version": "original",
        "proxy_triggered_partial_reedit": {
            "attempted": False,
            "improved": False,
            "selected_version": "original",
            "changed_boundary_indices": [],
        },
        "passed": True,
        "candidates": [
            {
                "candidate_index": index,
                "selected": index == 0,
                "passed": True,
                "review": deepcopy(final_review),
            }
            for index in range(8)
        ],
        "complete_candidates": complete_candidates,
    }
    timeline = _timeline_review()
    choices = [
        {
            "segment_index": index,
            "source_index": source_index,
            "canonical_master_start": _segment_bounds(index)[0],
            "canonical_master_end": _segment_bounds(index)[1],
            "duration_seconds": (
                _segment_bounds(index)[1] - _segment_bounds(index)[0]
            ),
            "shot_type": "close_up",
            "entry_shot_type": "close_up",
            "exit_shot_type": "close_up",
            "shot_id": f"shot-{index}",
        }
        for index, source_index in enumerate(source_sequence)
    ]
    beats = [
        {"index": index, "time_seconds": index * 0.5, "strength": 0.8}
        for index in range(65)
    ]
    phrases = [
        {
            "beat_count": 8,
            "start_beat_index": index * 8,
            "end_beat_index": (index + 1) * 8,
            "start_seconds": index * 4.0,
            "end_seconds": (index + 1) * 4.0,
            "duration_seconds": 4.0,
            "confidence": 0.9,
        }
        for index in range(8)
    ]
    group = {
        "analysis_basis": "fixture_group_transitions_v1",
        "dissolve_count": 0,
        "introduced_hard_cut_count": 8,
        "same_source_hold_count": 0,
        "hard_cut_boundary_count": 8,
        "transitions": deepcopy(transitions),
    }
    analysis = {
        "schema_version": 2,
        "created_at": "2026-08-23T12:34:56+00:00",
        "source_index_mapping": {
            "request_index_domain": "request_source_index",
            "eligible_index_domain": "eligible_source_index",
            "render_plan_index_domain": "render_plan_sources_array_index",
            "eligible_to_request": list(range(8)),
            "request_to_eligible": list(range(8)),
        },
        "local_timewarps": [
            {
                "source_index": index,
                "status": "local_monotonic_timewarp",
                "backend": "spectral_flux_local_ncc",
                "scalar_fallback": False,
                "fallback_reason": None,
                "accepted_anchor_count": 8,
                "residuals": {
                    "p50_seconds": 0.001,
                    "p95_seconds": 0.004,
                    "maximum_seconds": 0.006,
                    "sample_count": 8,
                    "definition": (
                        "absolute_anchor_residual_to_robust_affine_trend"
                    ),
                },
            }
            for index in range(8)
        ],
        "common_master_range": {
            "start_seconds": 0.0,
            "end_seconds": _DURATION,
            "duration_seconds": _DURATION,
        },
        "rhythm_analysis": {
            "duration_seconds": _DURATION,
            "bpm": 120.0,
            "tempo_confidence": 0.9,
            "beat_period_seconds": 0.5,
            "beat_times_seconds": [item["time_seconds"] for item in beats],
            "beat_grid": beats,
            "phrase_candidates": phrases,
        },
        "section_plan": {
            "role_semantics": "acoustic_editing_roles_not_ground_truth",
            "sections": [
                {
                    "section_index": 0,
                    "role": "chorus_like",
                    "start_seconds": 0.0,
                    "end_seconds": _DURATION,
                    "duration_seconds": _DURATION,
                    "target_source_switches_per_minute": 15.0,
                    "camera_motion_budget": 0.5,
                    "protect_last_beat_from_extra_cut": False,
                }
            ],
        },
        "source_look_match": {
            "target_mode": "cross_source_consensus",
            "target_source_index": None,
            "evidence": [
                {
                    "source_index": index,
                    "decoded_sample_count": 9,
                    "comparison_sample_count": 9,
                    "stable": True,
                    "reason": "stable_bounded_static_match",
                }
                for index in range(8)
            ],
        },
        "group_draft": {"segments": []},
        "visual_timeline": {"choices": choices},
        "visual_boundary_repairs": {"repairs": deepcopy(repairs)},
        "timeline_review": timeline,
        "group_transitions": group,
        "editorial_motion": _editorial_motion(),
        "rendered_proxy_review": deepcopy(final_review),
        "rendered_candidate_selection": selection,
        "render": {
            "video_path": str(artifacts.video_path),
            "plan_path": str(artifacts.render_plan_path),
            "video_codec": "libx264",
            "final_render_segments": [
                {
                    "segment_index": index,
                    "request_source_index": source_index,
                    "canonical_master_start": _segment_bounds(index)[0],
                    "canonical_master_end": _segment_bounds(index)[1],
                    "output_start_seconds": _segment_bounds(index)[0],
                    "output_end_seconds": _segment_bounds(index)[1],
                    "source_local_start": _segment_bounds(index)[0],
                    "source_local_end": _segment_bounds(index)[1],
                }
                for index, source_index in enumerate(source_sequence)
            ],
            "timeline_review": timeline,
            "rendered_proxy_review": deepcopy(final_review),
            "rendered_candidate_selection": selection,
        },
    }
    return artifacts, analysis, plan, selection


def _persist_fixture(
    root: Path,
    *,
    mutate: Callable[[dict, dict, dict], None] | None = None,
) -> tuple[PostRenderArtifactSet, dict, dict, dict]:
    artifacts, analysis, plan, selection = _fixture_payloads(root)
    if mutate is not None:
        mutate(analysis, plan, selection)
    # Keep the duplicated selection contracts synchronized after mutation.
    analysis["rendered_candidate_selection"] = selection
    analysis["render"]["rendered_candidate_selection"] = selection
    analysis["rendered_proxy_review"] = selection["final_selected_review"]
    analysis["render"]["rendered_proxy_review"] = selection["final_selected_review"]
    _write_json(artifacts.analysis_path, analysis)
    _write_json(artifacts.render_plan_path, plan)
    _write_json(artifacts.selection_path, selection)
    artifacts.video_path.write_bytes(b"synthetic MP4 fixture")
    return artifacts, analysis, plan, selection


def _install_valid_windowed_semimarkov_redecode(selection: dict) -> None:
    initial_sha = "1" * 64
    final_sha = "2" * 64
    input_sha = "3" * 64
    evidence = {
        "actual_semimarkov_redecode": True,
        "new_proxy_review_available": True,
        "new_proxy_passed": True,
        "proposal_materialization_lineage_preserved": True,
        "final_authoritative_exact_shot_clock_count": 1,
        "source_sequence_changed_by_redecode": True,
        "perceptual_targeted_source_substitution_required": True,
        "perceptual_targeted_source_substitution_satisfied": True,
        "new_proxy_perceptual_source_admission_passed": True,
    }
    lineage_event = {
        "stage": "reviewer_issue_window_actual_semimarkov_redecode",
        "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
        "repair_loop_input_sha256": input_sha,
        "initial_edit_sha256": initial_sha,
        "final_edit_sha256": final_sha,
        "committed_iteration_count": 1,
        "actual_public_decoder_invoked": True,
        "decoded_proxy_revalidated_after_splice": True,
        "heuristic_timing_repair_used_as_reviewer_repair": False,
        "proposal_materialization_lineage_event_count": 0,
        **evidence,
    }
    lineage = [
        {"stage": "windowed_semimarkov_proposal_lineage", "event_count": 0},
        lineage_event,
    ]
    selection["final_selected_version"] = "windowed_semimarkov_redecode"
    selection["proxy_triggered_partial_reedit"] = {
        "analysis_basis": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
        "attempted": True,
        "actual_local_semimarkov_redecode": True,
        "heuristic_timing_repair_used": False,
        "selected_version": "windowed_semimarkov_redecode",
        "reason": "strict_objectives_improved_and_proxy_revalidated",
        "before_review": deepcopy(selection["candidates"][0]["review"]),
        "after_review": deepcopy(selection["final_selected_review"]),
        "repair_loop_changed": True,
        "changed": True,
        "repair_loop": {
            "orchestrator": "stage_edit_repair_loop_v1",
            "input_sha256": input_sha,
            "selected_candidate_index": 0,
            "initial_edit_sha256": initial_sha,
            "final_edit_sha256": final_sha,
            "initial_review": {"decoded_objective": 1.0},
            "final_review": {"decoded_objective": 2.0},
            "iterations": [
                {
                    "iteration": 1,
                    "committed_proposal_id": "actual-windowed-semimarkov-i01-r01",
                    "committed_edit_sha256": final_sha,
                }
            ],
            "committed_iteration_count": 1,
        },
        "windowed_beam_redecode": {
            "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
            "actual_public_decoder": (
                "crosscut.stage_edit_semimarkov_beam."
                "decode_stage_edit_semimarkov_beam"
            ),
            "call_count": 1,
            "calls": [
                {
                    "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
                    "iteration": 1,
                }
            ],
        },
        "selected_redecode_evidence": evidence,
        "candidate_lineage_after": lineage,
    }
    selection["stage_edit_render_promotion"] = {
        "selected_version": "windowed_semimarkov_redecode",
        "selected_candidate_index": selection["selected_candidate_index"],
        "selected_candidate_lineage": deepcopy(lineage),
    }


def _ffprobe_payload(
    *,
    frame_count: int = _FRAME_COUNT,
    duration: float = _DURATION,
) -> dict[str, object]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "nb_read_frames": str(frame_count),
                "duration": f"{duration:.6f}",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": f"{duration:.6f}",
            },
        ],
        "format": {"duration": f"{duration:.6f}"},
    }


def _perceptual_pass_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "analysis": (
            "full_final_decode_temporal_pumping_exact_source_retention_and_"
            "structure_binding_v2"
        ),
        "verdict": "PASS",
        "passed": True,
        "issues": [],
        "policy": {},
        "metrics": {"fixture": True},
        "serialization_policy": {
            "raw_frames_serialized": False,
            "per_frame_descriptors_serialized": False,
            "sample_timestamps_serialized": False,
            "scalar_aggregate_provenance_only": True,
        },
    }


def _install_valid_registered_montage_evidence(
    analysis: dict,
    plan: dict,
    selection: dict,
) -> None:
    plan["source_index_domains"] = {
        "sources[]": "render_plan_sources_array_index",
        "reference_source": "render_plan_sources_array_index",
        "segments[].source_index": "render_plan_sources_array_index",
    }
    analysis["source_index_mapping"] = _source_index_mapping(
        tuple(Path(source) for source in plan["sources"]),
        tuple(range(len(plan["sources"]))),
    )
    shot_types = (
        "wide",
        "close",
        "medium",
        "wide",
        "close",
        "medium",
        "wide",
        "close",
        "wide",
    )
    choices = analysis["visual_timeline"]["choices"]
    analysis["visual_timeline"]["index_domain"] = "eligible_source_index"
    for index, (choice, shot_type) in enumerate(
        zip(choices, shot_types, strict=True)
    ):
        choice.update(
            {
                "shot_type": shot_type,
                "entry_shot_type": shot_type,
                "exit_shot_type": shot_type,
                "source_local_start": plan["segments"][index][
                    "source_local_start"
                ],
                "source_local_end": plan["segments"][index][
                    "source_local_end"
                ],
                "internal_scene_cut_master_seconds": [],
                "detected_person_counts": [6, 6, 6],
                "formation_person_counts": [6, 6, 6],
                "formation_upper_body_counts": [6, 6, 6],
            }
        )
    # The independent professional-boundary QA must be able to derive every
    # evidence field from its original authorities.  Keep this registered
    # fixture small, but give it one genuinely visible output-frame transform
    # rather than a hand-authored evidence count.
    transform_keyframes = [
        {
            "time_seconds": 0.0,
            "zoom": 1.0,
            "center_x": 0.5,
            "center_y": 0.5,
            "roll_degrees": 0.0,
        },
        {
            "time_seconds": 0.75,
            "zoom": 1.04,
            "center_x": 0.5,
            "center_y": 0.5,
            "roll_degrees": 0.0,
        },
    ]
    transform_reasons = {
        0: "reference_calibrated_same_source_interior_zoom_pulse",
        3: "reference_calibrated_push_in",
        6: "reference_calibrated_pull_out",
    }
    for segment_index, reason in transform_reasons.items():
        plan["segments"][segment_index]["editorial_transform"] = {
            "keyframes": deepcopy(transform_keyframes),
            "reason": reason,
        }
    motion_segments = []
    for index, segment in enumerate(plan["segments"]):
        applied = index in transform_reasons
        motion_segments.append(
            {
                "segment_index": index,
                "source_index": segment["source_index"],
                "applied": applied,
                "reasons": (
                    [transform_reasons[index]] if applied else []
                ),
                "keyframes": (
                    deepcopy(transform_keyframes) if applied else []
                ),
            }
        )
    analysis["editorial_motion"] = {
        "analysis_basis": "fixture_output_frame_visible_motion_v2",
        "applied_segment_count": 3,
        "boundary_match_count": 0,
        "affine_boundary_match_count": 0,
        "zoom_event_count": 3,
        "pan_event_count": 0,
        "roll_event_count": 0,
        "interior_zoom_pulse_count": 1,
        "section_directed": True,
        "blocked_dissolve_segment_count": 0,
        "skipped_boundary_reasons": [],
        "segments": motion_segments,
    }
    source_scene_frames: list[list[int]] = [
        [] for _source in plan["sources"]
    ]
    for index, (choice, segment) in enumerate(
        zip(choices, plan["segments"], strict=True)
    ):
        offsets = (1.0, 2.0) if index < 6 else (1.5,)
        source_frames = [
            int(round((segment["source_start"] + offset) * _FPS))
            for offset in offsets
        ]
        source_scene_frames[segment["source_index"]].extend(source_frames)
        choice["internal_scene_cut_master_seconds"] = [
            segment["master_start"]
            + frame_index / _FPS
            - segment["source_start"]
            for frame_index in source_frames
        ]
    analysis["scene_analysis"] = []
    for source_index, source in enumerate(plan["sources"]):
        frames = sorted(source_scene_frames[source_index])
        analysis["scene_analysis"].append(
            {
                "index_domain": "request_source_index",
                "source_index": source_index,
                "source": source,
                "frame_rate": _FPS,
                "changes": [
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / _FPS,
                    }
                    for frame_index in frames
                ],
            }
        )
    request_payload = analysis.get("request")
    automatic_ive_product_fixture = bool(
        isinstance(request_payload, dict)
        and str(request_payload.get("group") or "").strip().upper() == "IVE"
        and str(request_payload.get("edit_mode") or "").strip().lower()
        == "group"
        and str(request_payload.get("source_mode") or "").strip().lower()
        == "automatic"
    )
    base_section = deepcopy(analysis["section_plan"]["sections"][0])

    def fixture_section(
        section_index: int,
        role: str,
        start_seconds: float,
        end_seconds: float,
    ) -> dict[str, object]:
        item = deepcopy(base_section)
        item.update(
            {
                "section_index": section_index,
                "role": role,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "target_source_switches_per_minute": (
                    20.0 if role == "chorus_like" else 15.0
                ),
                "camera_motion_budget": 0.2,
                "establish_with_wide_shot": section_index == 0,
            }
        )
        return item

    if automatic_ive_product_fixture:
        fixture_sections = [
            fixture_section(0, "verse_like", 0.0, 2.5),
            fixture_section(1, "chorus_like", 2.5, 5.5),
            fixture_section(2, "verse_like", 5.5, _DURATION),
        ]
    else:
        fixture_sections = [
            fixture_section(0, "verse_like", 0.0, _DURATION)
        ]
    analysis["section_plan"].update(
        {
            "start_seconds": 0.0,
            "end_seconds": _DURATION,
            "section_count": len(fixture_sections),
            "sections": fixture_sections,
        }
    )
    analysis["stage_edit_exact_shot_clock"] = {
        "contract": (
            "serialized_inference_token_section_and_phrase_progress_"
            "per_selected_shot_v1"
        ),
        "source_index_domain": "eligible_source_index",
        "shots": [
            {
                "shot_index": index,
                "source_index": choice["source_index"],
                "section_role": (
                    "chorus_like"
                    if automatic_ive_product_fixture and index == 1
                    else "verse_like"
                ),
                "section_progress": (
                    (choice["canonical_master_start"] - 2.5) / 3.0
                    if automatic_ive_product_fixture and index == 1
                    else choice["canonical_master_start"] / _DURATION
                    if not automatic_ive_product_fixture or index == 0
                    else (
                        choice["canonical_master_start"] - 5.5
                    )
                    / (_DURATION - 5.5)
                ),
                # Phrase entries at shots 0, 3 and 6 are all wide in this
                # production fixture.
                "phrase_progress": (index % 3) / 3.0,
            }
            for index, choice in enumerate(choices)
        ],
    }
    analysis["source_derivative_diversity"] = {
        "index_domain": "eligible_source_index",
        "source_count": 8,
        "family_ids": list(range(8)),
        "production_ready": True,
        "clock_contract_complete": True,
        "uncertain_pair_count": 0,
        "family_consistency_conflict_count": 0,
    }
    analysis["section_plan"]["sections"][0][
        "establish_with_wide_shot"
    ] = True
    profile_identity = {
        "analysis_basis": MONTAGE_PROFILE_ANALYSIS_BASIS,
        "schema_version": MONTAGE_PROFILE_SCHEMA_VERSION,
        "path": "/fixture/checkpoint/professional-montage-profile.json",
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    analysis["stage_edit_generation"][
        "professional_montage_grammar_profile"
    ] = profile_identity
    _promote_review_to_domain_aware_v3(
        selection["final_selected_review"], plan
    )
    for candidate, complete_candidate in zip(
        selection["candidates"],
        selection["complete_candidates"],
        strict=True,
    ):
        complete_candidate["visual_timeline"] = {
            "choices": [deepcopy(segment) for segment in plan["segments"]]
        }
        complete_candidate["group_transitions"] = {
            "transitions": deepcopy(plan["transitions"])
        }
        _promote_review_to_domain_aware_v3(candidate["review"], plan)
    if automatic_ive_product_fixture:
        _attach_valid_segment_interior_geometry(analysis, plan, selection)
    analysis["rendered_proxy_review"] = deepcopy(
        selection["final_selected_review"]
    )
    review = review_rendered_montage_grammar(
        analysis,
        render_plan=plan,
        config=_MONTAGE_TEST_CONFIG,
    )
    canonical_plan_sha256 = hashlib.sha256(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    typed_plan = RenderPlan(
        sources=tuple(Path(source) for source in plan["sources"]),
        segments=tuple(
            RenderSegment(
                source_index=segment["source_index"],
                source_start=segment["source_start"],
                duration=segment["duration"],
                master_start=segment["master_start"],
                source_duration=segment["source_duration"],
                editorial_transform=(
                    EditorialTransformTrack(
                        keyframes=tuple(
                            EditorialTransformKeyframe(**keyframe)
                            for keyframe in segment[
                                "editorial_transform"
                            ]["keyframes"]
                        ),
                        reason=segment["editorial_transform"]["reason"],
                    )
                    if segment["editorial_transform"] is not None
                    else None
                ),
            )
            for segment in plan["segments"]
        ),
        reference_source=plan["reference_source"],
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path(plan["output"]),
        master_audio_start=plan["master_audio_start"],
        width=plan["width"],
        height=plan["height"],
        fps=plan["fps"],
        transitions=tuple(
            RenderTransition(
                kind=transition["kind"],
                frame_count=transition["frame_count"],
                reason=transition["reason"],
                face_mismatch=transition["face_mismatch"],
                tier=transition["tier"],
                appearance_min_cosine_similarity=transition[
                    "appearance_min_cosine_similarity"
                ],
                future_pose_max_mismatch=transition[
                    "future_pose_max_mismatch"
                ],
                future_silhouette_max_mismatch=transition[
                    "future_silhouette_max_mismatch"
                ],
            )
            for transition in plan["transitions"]
        ),
    )
    boundary_evidence = adapt_group_boundary_grammar_evidence(
        performance_id=f"ive-group-{canonical_plan_sha256[:20]}",
        render_plan=typed_plan,
        source_index_mapping=analysis["source_index_mapping"],
        selected_timeline=analysis["visual_timeline"],
        source_scene_cut_times=tuple(
            tuple(frame_index / _FPS for frame_index in frames)
            for frames in source_scene_frames
        ),
        timeline_review=analysis["timeline_review"],
        section_plan=analysis["section_plan"],
        rendered_proxy_review=analysis["rendered_proxy_review"],
        editorial_motion=analysis["editorial_motion"],
    )
    boundary_assessment = assess_professional_boundary_grammar(
        boundary_evidence
    )
    assert boundary_assessment.passed
    review["professional_boundary_grammar_evidence"] = (
        boundary_evidence.to_dict()
    )
    review["professional_boundary_grammar_assessment"] = (
        boundary_assessment.to_dict()
    )
    review["passed_before_professional_boundary_grammar"] = bool(
        review.get("passed") is True
    )
    review["passed"] = bool(
        review.get("passed") is True and boundary_assessment.passed
    )
    review["reference_profile"] = {
        **profile_identity,
        "path": "/fixture/stable/professional-montage-profile.json",
    }
    analysis["rendered_montage_grammar_review"] = review


def _attach_valid_segment_interior_geometry(
    analysis: dict,
    plan: dict,
    selection: dict,
) -> None:
    """Give the automatic-IVE fixture real hash-bound interior samples."""

    proxy_plan = RenderPlan(
        sources=tuple(Path(source) for source in plan["sources"]),
        segments=tuple(
            RenderSegment(
                source_index=segment["source_index"],
                source_start=segment["source_start"],
                duration=segment["duration"],
                master_start=segment["master_start"],
                source_duration=segment["source_duration"],
                editorial_transform=(
                    EditorialTransformTrack(
                        keyframes=tuple(
                            EditorialTransformKeyframe(**keyframe)
                            for keyframe in segment[
                                "editorial_transform"
                            ]["keyframes"]
                        ),
                        reason=segment["editorial_transform"]["reason"],
                    )
                    if segment["editorial_transform"] is not None
                    else None
                ),
            )
            for segment in plan["segments"]
        ),
        reference_source=plan["reference_source"],
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path(plan["output"]),
        master_audio_start=plan["master_audio_start"],
        width=854,
        height=480,
        fps=plan["fps"],
        transitions=tuple(
            RenderTransition(
                kind=transition["kind"],
                frame_count=transition["frame_count"],
                reason=transition["reason"],
                tier=transition["tier"],
            )
            for transition in plan["transitions"]
        ),
    )
    counts = segment_frame_counts(proxy_plan)
    schedule = _segment_interior_schedule(proxy_plan, counts)
    choices = analysis["visual_timeline"]["choices"]
    class_codes = {"wide": 1, "medium": 2, "close": 3}
    frames: dict[int, np.ndarray] = {}
    for sample in schedule:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0, 0] = class_codes[
            str(choices[sample.segment_index]["shot_type"])
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
                    (0.15, 0.30, 0.14, 0.48),
                    (0.43, 0.26, 0.15, 0.54),
                    (0.71, 0.31, 0.13, 0.47),
                ),
                shot_scale_basis="aggregate_formation_people",
            )
        height = 0.15 if shot_class == "medium" else 0.30
        return OutputSubjectGeometry(
            face_count=1,
            face_bbox_xywh=(0.35, 0.20, 0.20, height),
            face_backend="synthetic",
        )

    final_review = selection["final_selected_review"]
    evidence = _segment_interior_geometry_evidence(
        plan=proxy_plan,
        counts=counts,
        schedule=schedule,
        frames=frames,
        geometry_provider=geometry_provider,
        proxy_identity=final_review["proxy"],
    )
    final_review["segment_interior_geometry"] = evidence
    for candidate in selection["candidates"]:
        candidate["review"]["segment_interior_geometry"] = deepcopy(evidence)


def _activate_registered_cut_local_sync(
    analysis: dict,
    plan: dict,
    selection: dict,
    *,
    residual_frames: float = 0.10,
    unknown_boundary_index: int | None = None,
) -> None:
    # Install the final montage plan before hashing cut-local evidence.  The
    # production contract binds the delivered plan, never a pre-transform
    # fixture that is mutated after its proof is serialized.
    analysis["stage_edit_generation"] = {
        "activation": {
            "activation": "automatic_registered_production_bundle",
        }
    }
    _install_valid_registered_montage_evidence(analysis, plan, selection)
    plan["source_index_domains"] = {
        "sources[]": "render_plan_sources_array_index",
        "reference_source": "render_plan_sources_array_index",
        "segments[].source_index": "render_plan_sources_array_index",
    }
    analysis["source_index_mapping"]["domain_equivalences"] = {
        "render_plan_sources_array_index": "request_source_index",
    }
    for timewarp in analysis["local_timewarps"]:
        timewarp.update(
            {
                "global_offset_seconds": 0.0,
                "master_duration_seconds": _DURATION,
                "source_duration_seconds": _DURATION,
                "fallback_reason": None,
                "minimum_slope": 0.98,
                "maximum_slope": 1.02,
                "observed_minimum_slope": 1.0,
                "observed_maximum_slope": 1.0,
                "mean_slope": 1.0,
                "estimated_drift_parts_per_million": 0.0,
                "control_points": [
                    {
                        "master_seconds": 0.0,
                        "source_seconds": 0.0,
                        "offset_seconds": 0.0,
                    },
                    {
                        "master_seconds": _DURATION,
                        "source_seconds": _DURATION,
                        "offset_seconds": 0.0,
                    },
                ],
            }
        )
    boundaries = _boundary_contracts(plan)
    used_sources = _used_source_indices(plan)
    bound_timewarps = _bound_timewarp_payloads(
        analysis["local_timewarps"], used_sources
    )
    records: list[dict[str, object]] = []
    measured: list[float] = []
    unknown_reasons: dict[str, int] = {}
    selected_ordinals: dict[int, set[int]] = {
        index: set() for index in used_sources
    }
    selected_pts: dict[int, dict[int, float]] = {
        index: {} for index in used_sources
    }
    for boundary in boundaries:
        record: dict[str, object] = {
            **boundary.payload(),
            "boundary_contract_sha256": boundary.sha256,
        }
        if boundary.boundary_index == unknown_boundary_index:
            record.update(
                {
                    "status": "unknown",
                    "reason": "ambiguous_motion_lag",
                }
            )
            unknown_reasons["ambiguous_motion_lag"] = 1
        else:
            grids = list(range(-7, 8))
            requested = [
                boundary.canonical_master_seconds + grid / _FPS
                for grid in grids
            ]
            outgoing_pts = list(requested)
            incoming_pts = [
                value + residual_frames / _FPS for value in requested
            ]
            outgoing_ordinals = [int(round(value * _FPS)) for value in outgoing_pts]
            incoming_ordinals = [int(round(value * _FPS)) for value in incoming_pts]
            selected_ordinals[boundary.outgoing_source_index].update(
                outgoing_ordinals
            )
            selected_ordinals[boundary.incoming_source_index].update(
                incoming_ordinals
            )
            selected_pts[boundary.outgoing_source_index].update(
                zip(outgoing_ordinals, outgoing_pts, strict=True)
            )
            selected_pts[boundary.incoming_source_index].update(
                zip(incoming_ordinals, incoming_pts, strict=True)
            )
            outgoing_timing = [
                {
                    "grid_index": grid,
                    "actual_master_seconds": (
                        outgoing_pts[position] + outgoing_pts[position + 1]
                    )
                    / 2.0,
                    "basis": "actual_pose_motion",
                }
                for position, grid in enumerate(grids[:-1])
            ]
            incoming_timing = [
                {
                    "grid_index": grid,
                    "actual_master_seconds": (
                        incoming_pts[position] + incoming_pts[position + 1]
                    )
                    / 2.0,
                    "basis": "actual_pose_motion",
                }
                for position, grid in enumerate(grids[:-1])
            ]
            candidates: list[dict[str, object]] = []
            for lag, correlation in zip(
                range(-2, 3), (0.40, 0.60, 0.95, 0.60, 0.40), strict=True
            ):
                pair_count = 14 - abs(lag)
                candidates.append(
                    {
                        "lag_frames": lag,
                        "pair_count": pair_count,
                        "non_silhouette_pair_count": pair_count,
                        "correlation": correlation,
                        "outgoing_motion_stddev": 0.20,
                        "incoming_motion_stddev": 0.20,
                        "actual_pts_phase_frames": lag + residual_frames,
                        "motion_basis_pair_counts": {
                            "actual_pose_motion+actual_pose_motion": pair_count,
                        },
                        "eligible": True,
                    }
                )
            record.update(
                {
                    "status": "measured",
                    "reason": None,
                    "sampling_domain": "hard_cut_symmetric_canonical_context",
                    "renderer_frame_selection": {
                        "outgoing": {
                            "basis": "trim_lower_bound_setpts_first_pts_fps_round_near_tpad_clone_v1",
                            "source_index": boundary.outgoing_source_index,
                            "grid_indices": grids,
                            "frame_ordinals": outgoing_ordinals,
                            "normalized_pts_seconds": outgoing_pts,
                            "requested_master_seconds": requested,
                            "source_start_seconds": requested[0],
                            "source_duration_seconds": 0.5,
                            "timeline_duration_seconds": 0.5,
                            "frame_count": 15,
                        },
                        "incoming": {
                            "basis": "trim_lower_bound_setpts_first_pts_fps_round_near_tpad_clone_v1",
                            "source_index": boundary.incoming_source_index,
                            "grid_indices": grids,
                            "frame_ordinals": incoming_ordinals,
                            "normalized_pts_seconds": incoming_pts,
                            "requested_master_seconds": requested,
                            "source_start_seconds": requested[0],
                            "source_duration_seconds": 0.5,
                            "timeline_duration_seconds": 0.5,
                            "frame_count": 15,
                        },
                    },
                    "outgoing_motion_sample_count": 14,
                    "incoming_motion_sample_count": 14,
                    "measurement_mode": "correlated_motion_phase_v1",
                    "measurement_lag_frames": 0,
                    "best_lag_frames": 0,
                    "subframe_peak_offset_frames": 0.0,
                    "actual_pts_phase_frames": residual_frames,
                    "signed_phase_residual_frames": residual_frames,
                    "residual_frames": residual_frames,
                    "best_correlation": 0.95,
                    "runner_up_correlation": 0.60,
                    "correlation_margin": 0.35,
                    "matched_motion_pair_count": 14,
                    "matched_non_silhouette_pair_count": 14,
                    "motion_basis_pair_counts": {
                        "actual_pose_motion+actual_pose_motion": 14,
                    },
                    "motion_timing_provenance": {
                        "outgoing": outgoing_timing,
                        "incoming": incoming_timing,
                    },
                    "lag_candidates": candidates,
                }
            )
            measured.append(residual_frames)
        records.append(record)
    p95 = (
        float(np.quantile(np.asarray(measured, dtype=np.float64), 0.95))
        if measured
        else None
    )
    passed = bool(
        boundaries
        and len(measured) == len(boundaries)
        and not unknown_reasons
        and p95 is not None
        and p95 <= 0.5
    )
    analysis["stage_edit_generation"]["activation"] = {
        "activation": "automatic_registered_production_bundle",
    }
    selection["stage_edit_render_promotion"] = {
        "activation": "automatic_registered_production_bundle",
        "candidate_pool": "stage_edit_generation_only",
    }
    analysis["render"]["candidate_origin"] = "explicit_stage_edit_generation"
    analysis["render"]["stage_edit_promotion_contract"] = "fixture_stage_edit"
    payload = {
        "schema_version": CUT_LOCAL_SYNC_SCHEMA_VERSION,
        "analysis": CUT_LOCAL_SYNC_ANALYSIS_BASIS,
        "activation": CUT_LOCAL_SYNC_ACTIVATION,
        "required": True,
        "plan_sha256": _canonical_sha256(plan),
        "timewarp_contract_sha256": _canonical_sha256(bound_timewarps),
        "boundary_clock_contract_sha256": _canonical_sha256(
            [item.payload() for item in boundaries]
        ),
        "fps": _FPS,
        "used_source_indices": list(used_sources),
        "retained_cross_source_boundary_count": len(boundaries),
        "measured_boundary_count": len(measured),
        "unknown_boundary_count": sum(unknown_reasons.values()),
        "unknown_reason_counts": unknown_reasons,
        "threshold_exceeding_boundary_count": sum(
            value > 0.5 for value in measured
        ),
        "residual_p95_frames": p95,
        "maximum_allowed_residual_p95_frames": 0.5,
        "passed": passed,
        "policy": CutLocalSyncConfig().to_dict(),
        "source_decode_audit": [
            {
                "source_index": index,
                "source_name": Path(plan["sources"][index]).name,
                "status": "decoded",
                "pts_basis": "ffprobe_best_effort_timestamp_time",
                "renderer_selection_basis": "trim_lower_bound_setpts_first_pts_fps_round_near_tpad_clone_v1",
                "pts_frame_count": 2000,
                "decoded_declared_frame_count": 2000,
                "pts_decode_ordinal_count_match": True,
                "pts_origin_seconds": 0.0,
                "pts_sha256": f"{index:x}" * 64,
                "median_pts_step_seconds": 1.0 / _FPS,
                "requested_ordinal_count": len(selected_ordinals[index]),
                "selected_ordinal_pts_sha256": _canonical_sha256(
                    [
                        {
                            "frame_ordinal": ordinal,
                            "normalized_pts_seconds": pts_value,
                        }
                        for ordinal, pts_value in sorted(selected_pts[index].items())
                    ]
                ),
                "decoded_requested_ordinal_count": len(selected_ordinals[index]),
                "decoded_frame_count": len(selected_ordinals[index]),
                "described_frame_count": len(selected_ordinals[index]),
                "descriptor_failure_count": 0,
                "maximum_pts_lookup_residual_frames": residual_frames,
                "raw_frames_persisted": False,
                "motion_descriptors_persisted": False,
            }
            for index in used_sources
        ],
        "per_boundary": records,
        "serialization_policy": dict(CUT_LOCAL_SYNC_SERIALIZATION_POLICY),
    }
    payload["proof_sha256"] = _canonical_sha256(payload)
    analysis["render"]["cut_local_sync"] = payload


class _FakeRunner:
    def __init__(
        self,
        *,
        ffprobe_payload: object | None = None,
        ffprobe_returncode: int = 0,
        ffmpeg_returncode: int = 0,
    ) -> None:
        self.ffprobe_payload = (
            _ffprobe_payload() if ffprobe_payload is None else ffprobe_payload
        )
        self.ffprobe_returncode = ffprobe_returncode
        self.ffmpeg_returncode = ffmpeg_returncode
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self,
        command: list[str] | tuple[str, ...],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        values = tuple(command)
        self.calls.append((values, timeout))
        if Path(values[0]).name == "ffprobe":
            return subprocess.CompletedProcess(
                values,
                self.ffprobe_returncode,
                stdout=(
                    json.dumps(self.ffprobe_payload)
                    if self.ffprobe_returncode == 0
                    else ""
                ),
                stderr="synthetic ffprobe failure" if self.ffprobe_returncode else "",
            )
        return subprocess.CompletedProcess(
            values,
            self.ffmpeg_returncode,
            stdout="",
            stderr="synthetic decode failure" if self.ffmpeg_returncode else "",
        )


class PostRenderQATests(unittest.TestCase):
    def setUp(self) -> None:
        self._perceptual_patcher = patch(
            "crosscut.post_render_qa.evaluate_post_render_perceptual_quality",
            return_value=_perceptual_pass_payload(),
        )
        self._perceptual_patcher.start()
        self.addCleanup(self._perceptual_patcher.stop)

    def test_serialized_plan_replay_preserves_verified_transform_overlap_authority(
        self,
    ) -> None:
        incoming = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.02),
                EditorialTransformKeyframe(0.2, 1.06),
                EditorialTransformKeyframe(1.0, 1.06),
            ),
            reason="decoded_output_face_match",
        )
        plan = RenderPlan(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(
                    1,
                    1.0,
                    1.0,
                    1.0,
                    editorial_transform=incoming,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    7,
                    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                    tier="A",
                    transform_overlap_authority=(
                        MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY
                    ),
                ),
            ),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=1,
        )

        serialized = json.loads(
            json.dumps(plan.to_json(), allow_nan=False)
        )
        rebuilt = _boundary_render_plan_from_serialized(serialized)

        self.assertEqual(
            rebuilt.transitions[0].transform_overlap_authority,
            MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
        )
        self.assertEqual(rebuilt.member_pose_dissolve_count, 1)

    def test_automatic_ive_whole_song_band_accepts_profile_preferred_cadence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, _selection = _fixture_payloads(
                Path(directory)
            )
        preferred_rate = 23.505554096533405
        # Eight exact source switches at this duration yield the immutable
        # chorus-profile P75 target.  The legacy global 17.5 ceiling must not
        # mark an automatic IVE delivery conditional.
        plan["duration"] = 8.0 * 60.0 / preferred_rate
        analysis["timeline_review"]["source_switches_per_minute"] = (
            preferred_rate
        )
        analysis["request"] = {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        }
        automatic_collector = _IssueCollector()
        automatic_metrics = _cadence_and_repetition_metrics(
            analysis,
            plan,
            PostRenderQAConfig(),
            automatic_collector,
        )

        self.assertTrue(
            automatic_metrics[
                "automatic_ive_product_cadence_band_applied"
            ]
        )
        self.assertEqual(
            [13.5, 30.0],
            automatic_metrics["preferred_source_switches_per_minute_band"],
        )
        self.assertNotIn(
            "source_switch_density_outside_professional_band",
            {issue.code for issue in automatic_collector.issues},
        )

        analysis.pop("request")
        legacy_collector = _IssueCollector()
        legacy_metrics = _cadence_and_repetition_metrics(
            analysis,
            plan,
            PostRenderQAConfig(),
            legacy_collector,
        )
        self.assertFalse(
            legacy_metrics["automatic_ive_product_cadence_band_applied"]
        )
        self.assertIn(
            "source_switch_density_outside_professional_band",
            {issue.code for issue in legacy_collector.issues},
        )

    def test_discovers_only_exact_nonempty_timestamp_bound_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete, _analysis, _plan, _selection = _persist_fixture(root)
            incomplete = root / "IVE-incomplete-20260823-123457-analysis.json"
            incomplete.write_text("{}", encoding="utf-8")
            (root / f".{_BASE}-analysis.json.tmp").write_text(
                "{}", encoding="utf-8"
            )
            (root / "not-a-pipeline-analysis.json").write_text(
                "{}", encoding="utf-8"
            )

            discovered = discover_completed_artifact_sets(root)

        self.assertEqual(discovered, (complete,))

    def test_snapshot_reads_use_declared_paths_for_artifact_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory)
            )
            snapshot = materialize_post_render_qa_snapshot(artifacts)
            try:
                report = evaluate_post_render_qa(
                    snapshot.artifact_set,
                    declared_artifact_set=artifacts,
                    runner=_FakeRunner(),
                )
            finally:
                cleanup_post_render_qa_snapshot(snapshot)

        self.assertTrue(
            all(report.metrics["artifact_binding"]["checks"].values())
        )
        self.assertNotIn(
            "artifact_binding_analysis_video_path_matches",
            {issue.code for issue in report.issues},
        )

    def test_snapshot_cleanup_tolerates_vanishing_appledouble_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory)
            )
            snapshot = materialize_post_render_qa_snapshot(artifacts)
            transient = snapshot.directory / "._transient"
            transient.write_bytes(b"appledouble")
            original_lstat = Path.lstat
            vanished = False

            def racing_lstat(path: Path):
                nonlocal vanished
                if path == transient and not vanished:
                    vanished = True
                    transient.unlink(missing_ok=True)
                    raise FileNotFoundError(transient)
                return original_lstat(path)

            with patch("pathlib.Path.lstat", new=racing_lstat):
                cleanup_post_render_qa_snapshot(snapshot)

            self.assertTrue(vanished)
            self.assertFalse(snapshot.directory.exists())

    def test_relative_declaration_cannot_escape_timestamp_set_directory(self) -> None:
        def mutate(_analysis: dict, plan: dict, _selection: dict) -> None:
            plan["output"] = f"../other/{_BASE}-group-crosscut.mp4"

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "artifact_binding_render_plan_output_matches",
            {issue.code for issue in report.issues},
        )

    def test_professional_fixture_passes_and_runs_exact_probe_and_full_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(Path(directory))
            runner = _FakeRunner()

            report = evaluate_post_render_qa(artifacts, runner=runner)

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        self.assertEqual(report.issues, ())
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0][0][0], "ffprobe")
        self.assertIn("-count_frames", runner.calls[0][0])
        self.assertEqual(runner.calls[1][0][0], "ffmpeg")
        self.assertIn("-xerror", runner.calls[1][0])
        self.assertIn("0:a?", runner.calls[1][0])
        metrics = report.metrics
        self.assertTrue(metrics["final_media"]["checks"]["decoded_frame_count_matches"])
        self.assertEqual(
            metrics["beat_and_phrase"]["beat_distance_frames"][
                "within_1_frame_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["source_local_reuse"]["reused_source_local_duration_seconds"],
            0.0,
        )
        self.assertEqual(metrics["transforms"]["target_attainment"], 1.0)
        self.assertEqual(metrics["transforms"]["unique_transform_signature_count"], 4)
        self.assertTrue(metrics["production_sync"]["passed"])
        self.assertAlmostEqual(
            metrics["production_sync"]["maximum_residual_p95_frames"],
            0.12,
        )
        self.assertEqual(metrics["perceptual_quality"]["verdict"], "PASS")

    def test_perceptual_quality_fail_is_promoted_to_final_qa(self) -> None:
        payload = _perceptual_pass_payload()
        payload.update(
            {
                "verdict": "FAIL",
                "passed": False,
                "issues": [
                    {
                        "code": "perceptual_detail_retention_p10",
                        "severity": "fail",
                        "message": "Synthetic source-detail loss.",
                        "path": (
                            "perceptual_quality.exact_source_retention."
                            "detail_retention_ratio.p10"
                        ),
                        "value": 0.4,
                        "threshold": 0.55,
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory)
            )
            with patch(
                "crosscut.post_render_qa.evaluate_post_render_perceptual_quality",
                return_value=payload,
            ):
                report = evaluate_post_render_qa(
                    artifacts,
                    runner=_FakeRunner(),
                )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "perceptual_detail_retention_p10",
            {issue.code for issue in report.issues},
        )

    def test_actual_pose_v3_reports_coverage_without_requiring_every_cut_pose(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _promote_boundary_repairs_to_actual_pose_v3(analysis, selection)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        evidence = report.metrics["boundary_pose_evidence"]
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["geometry_provider_failure_count"], 0)
        self.assertEqual(evidence["exact_actual_pose_pair_count"], 0)
        self.assertEqual(evidence["formation_pair_count"], 8)
        self.assertEqual(
            evidence["professional_gate_failure_boundary_indices"], []
        )

    def test_unbound_temporal_pose_motion_is_audited_without_direction_claim(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _promote_boundary_repairs_to_unbound_temporal_pose_motion_v1(
                analysis,
                selection,
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        temporal = report.metrics["boundary_pose_evidence"][
            "temporal_pose_motion"
        ]
        self.assertTrue(temporal["available"])
        self.assertTrue(temporal["passed"])
        self.assertEqual(
            temporal["calibration_status_counts"],
            {"calibration_unbound": 8},
        )
        self.assertEqual(
            temporal["classification_counts"],
            {"calibration_unbound": 8},
        )
        self.assertEqual(temporal["direction_evidence_available_count"], 0)
        self.assertEqual(temporal["dynamic_direction_match_proven_count"], 0)
        self.assertEqual(temporal["soft_preference_applied_count"], 0)

    def test_bound_temporal_pose_motion_scalar_contract_reconciles_cost(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _bind_synthetic_temporal_pose_motion_fixture(
                analysis,
                selection,
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        temporal = report.metrics["boundary_pose_evidence"][
            "temporal_pose_motion"
        ]
        self.assertTrue(temporal["passed"])
        self.assertEqual(
            temporal["classification_counts"],
            {"dynamic_pose_phase_match": 8},
        )
        self.assertEqual(temporal["direction_evidence_available_count"], 8)
        self.assertEqual(temporal["dynamic_direction_match_proven_count"], 8)
        self.assertEqual(temporal["soft_preference_applied_count"], 8)
        self.assertAlmostEqual(
            temporal["soft_preference_applied_total"],
            0.04,
        )

    def test_temporal_pose_motion_rejects_raw_vectors_or_unbound_masquerade(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _promote_boundary_repairs_to_unbound_temporal_pose_motion_v1(
                analysis,
                selection,
            )
            selected = selection["complete_candidates"][0]
            motion = selected["boundary_repairs"]["repairs"][0][
                "selected_visual_evidence"
            ]["temporal_pose_motion"]
            motion["direction_evidence_available"] = True
            motion["outgoing_normalized_velocity"] = [1.0] * 60

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "boundary_temporal_pose_motion_contract_invalid",
            {issue.code for issue in report.issues},
        )
        invalid = report.metrics["boundary_pose_evidence"][
            "temporal_pose_motion"
        ]["invalid_records"]
        reasons = set(invalid[0]["reasons"])
        self.assertIn(
            "evidence_keys_invalid_or_nonscalar_payload_present",
            reasons,
        )
        self.assertIn(
            "unbound_evidence_masquerades_as_direction_proof",
            reasons,
        )

    def test_actual_pose_v3_rejects_hog_mislabeled_as_pose(self) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _promote_boundary_repairs_to_actual_pose_v3(analysis, selection)
            selected = selection["complete_candidates"][0]
            evidence = selected["boundary_repairs"]["repairs"][0][
                "selected_visual_evidence"
            ]
            evidence["pose_basis"] = "structure_hog_fallback"

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        codes = {issue.code for issue in report.issues}
        self.assertIn("boundary_hog_mislabeled_as_pose", codes)
        self.assertIn("boundary_pose_evidence_contract_invalid", codes)

    def test_actual_pose_v3_fails_on_provider_or_combined_gate_failure(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            _promote_boundary_repairs_to_actual_pose_v3(analysis, selection)
            selected = selection["complete_candidates"][0]["boundary_repairs"]
            selected["geometry_provider_failure_count"] = 1
            coverage = selected["repairs"][3]["selected_visual_evidence"][
                "evidence_coverage"
            ]
            coverage["professional_gate_passed"] = False
            coverage["professional_evidence_basis"] = (
                "insufficient_combined_geometry_evidence"
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        codes = {issue.code for issue in report.issues}
        self.assertIn("boundary_repair_geometry_provider_failures", codes)
        self.assertIn(
            "boundary_professional_combined_evidence_gate_failed",
            codes,
        )

    def test_used_source_scalar_sync_fallback_fails_professional_gate(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            record = analysis["local_timewarps"][3]
            record.update(
                {
                    "status": "global_scalar_offset",
                    "scalar_fallback": True,
                    "fallback_reason": "insufficient_accepted_anchors",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_sync_used_source_not_piecewise_verified",
            {issue.code for issue in report.issues},
        )
        self.assertFalse(report.metrics["production_sync"]["passed"])

    def test_used_source_sync_residual_above_half_frame_fails(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            # 0.017 s at 30 fps is 0.51 output frame.
            analysis["local_timewarps"][5]["residuals"]["p95_seconds"] = 0.017

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_sync_residual_exceeds_half_frame",
            {issue.code for issue in report.issues},
        )
        source = report.metrics["production_sync"]["per_source"][5]
        self.assertAlmostEqual(source["residual_p95_frames"], 0.51)

    def test_sync_audit_maps_compact_render_index_to_request_index(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            analysis["source_index_mapping"]["eligible_to_request"] = [
                0,
                2,
                3,
                4,
                5,
                6,
                7,
                9,
            ]
            analysis["source_index_mapping"]["request_to_eligible"] = [
                0,
                None,
                1,
                2,
                3,
                4,
                5,
                6,
                None,
                7,
            ]
            for record, request_index in zip(
                analysis["local_timewarps"],
                (0, 2, 3, 4, 5, 6, 7, 9),
                strict=True,
            ):
                record["source_index"] = request_index

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        self.assertEqual(
            [
                item["request_source_index"]
                for item in report.metrics["production_sync"]["per_source"]
            ],
            [0, 2, 3, 4, 5, 6, 7, 9],
        )

    def test_sync_audit_honors_render_request_domain_equivalence_with_gap(
        self,
    ) -> None:
        request_indices = (0, 2, 3, 4, 5, 6, 7, 9)
        analysis = {
            "source_index_mapping": {
                "eligible_to_request": list(request_indices),
                "domain_equivalences": {
                    "render_plan_sources_array_index": "request_source_index",
                },
            },
            "local_timewarps": [
                {
                    "source_index": index,
                    "status": "local_monotonic_timewarp",
                    "scalar_fallback": False,
                    "accepted_anchor_count": 8,
                    "residuals": {
                        "p95_seconds": 0.004,
                        "sample_count": 8,
                        "definition": (
                            "absolute_anchor_residual_to_robust_affine_trend"
                        ),
                    },
                }
                for index in request_indices
            ],
        }
        plan = {
            "fps": 30.0,
            "sources": [f"source-{index}.mp4" for index in range(10)],
            "source_usage": {
                "used_source_indices": list(request_indices),
            },
        }
        collector = _IssueCollector()

        metrics = _sync_quality_metrics(
            analysis,
            plan,
            PostRenderQAConfig(),
            collector,
        )

        self.assertTrue(metrics["passed"])
        self.assertTrue(metrics["render_indices_are_request_indices"])
        self.assertEqual(
            [item["request_source_index"] for item in metrics["per_source"]],
            list(request_indices),
        )
        self.assertEqual(collector.issues, [])

    def test_registered_stage_edit_requires_cut_local_sync_evidence(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            analysis["stage_edit_generation"] = {
                "activation": {
                    "activation": "automatic_registered_production_bundle",
                }
            }

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_cut_local_sync_evidence_missing",
            {issue.code for issue in report.issues},
        )
        self.assertIn(
            "production_cut_local_sync_activation_contract_invalid",
            {issue.code for issue in report.issues},
        )
        self.assertIn(
            "production_cut_local_sync_required_gate_failed",
            {issue.code for issue in report.issues},
        )
        cut_local = report.metrics["production_sync"]["cut_local_sync"]
        self.assertTrue(cut_local["required"])
        self.assertFalse(cut_local["passed"])

    def test_registered_stage_edit_accepts_hash_bound_cut_local_half_frame_proof(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        sync = report.metrics["production_sync"]
        self.assertTrue(sync["passed"])
        self.assertEqual(
            sync["basis"],
            "used_source_piecewise_pts_plus_retained_cut_pose_body_motion_v2",
        )
        self.assertTrue(sync["cut_local_sync"]["hash_bindings_valid"])
        self.assertAlmostEqual(
            sync["cut_local_sync"]["residual_p95_frames"], 0.10
        )
        montage = report.metrics["rendered_montage_grammar"]
        self.assertTrue(montage["required"])
        self.assertTrue(montage["passed"])
        self.assertTrue(montage["configuration_matches"])
        self.assertTrue(montage["reference_profile_identity_matches"])
        self.assertTrue(montage["professional_boundary_evidence_valid"])
        self.assertTrue(
            montage[
                "professional_boundary_assessment_matches_recomputation"
            ]
        )
        self.assertTrue(
            montage["professional_boundary_render_plan_binding_matches"]
        )
        self.assertTrue(
            montage["professional_boundary_legacy_pass_binding_matches"]
        )
        self.assertNotEqual(
            montage["generation_reference_profile"]["path"],
            montage["persisted_reference_profile"]["path"],
        )
        highlight = montage["recomputed_review"]["sections"][0]
        self.assertGreaterEqual(
            highlight["wide_involved_source_switch_fraction"],
            _MONTAGE_TEST_CONFIG.minimum_wide_involved_source_switch_fraction,
        )
        self.assertLess(
            highlight["wide_involved_source_switch_fraction"],
            _MONTAGE_TEST_CONFIG.preferred_wide_involved_source_switch_fraction,
        )
        self.assertEqual(
            highlight["wide_involved_source_switch_preferred_target"],
            _MONTAGE_TEST_CONFIG.preferred_wide_involved_source_switch_fraction,
        )
        self.assertEqual(
            highlight["wide_involved_source_switch_reference_count"],
            17,
        )
        self.assertFalse(
            highlight[
                "wide_involved_source_switch_preferred_target_is_hard_gate"
            ]
        )
        self.assertAlmostEqual(
            highlight[
                "wide_involved_source_switch_preferred_target_shortfall"
            ],
            0.05,
        )
        self.assertTrue(
            all(
                max(choice["formation_upper_body_counts"]) >= 1
                for choice in _analysis["visual_timeline"]["choices"]
                if choice["shot_type"] == "wide"
            )
        )

    def test_explicit_automatic_ive_group_rejects_underqualified_synthetic_highlight(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            analysis["request"] = {
                "group": "IVE",
                "edit_mode": "group",
                "source_mode": "automatic",
            }
            analysis["stage_edit_generation"] = {
                "activation": {"activation": "explicit_request_bundle"}
            }
            selection["stage_edit_render_promotion"] = {
                "activation": "explicit_stage_edit_generation_request_only",
                "candidate_pool": "stage_edit_generation_only",
            }
            analysis["render"].update(
                {
                    "candidate_origin": "explicit_stage_edit_generation",
                    "stage_edit_promotion_contract": "fixture_stage_edit",
                }
            )
            _install_valid_registered_montage_evidence(
                analysis,
                plan,
                selection,
            )
            selection["complete_candidates"][0]["visual_timeline"][
                "candidate_review_summaries"
            ] = [
                {
                    "candidate_index": 0,
                    "selected": True,
                    "passed": True,
                    "hard_failure_codes": [],
                }
            ]

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        montage = report.metrics["rendered_montage_grammar"]
        product = report.metrics["ive_product_quality_floor"]
        self.assertTrue(montage["ive_product_scope_required"])
        self.assertFalse(montage["passed"])
        self.assertTrue(product["required"])
        self.assertFalse(product["passed"])
        self.assertEqual(
            product["evaluation_mode"],
            "modern_role_conditioned_profile_gate",
        )
        self.assertFalse(
            product["metrics"]["legacy_whole_song_fallback_applied"]
        )
        self.assertIn(
            "modern_role_conditioned_montage_review_failed",
            product["failure_codes"],
        )

    def test_explicit_new_ive_generation_missing_role_evidence_fails_not_legacy(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            analysis["request"] = {
                "group": "IVE",
                "edit_mode": "group",
                "source_mode": "automatic",
            }
            analysis["stage_edit_generation"] = {
                "activation": {"activation": "explicit_request_bundle"}
            }
            selection["stage_edit_render_promotion"] = {
                "activation": "explicit_stage_edit_generation_request_only",
                "candidate_pool": "stage_edit_generation_only",
            }
            analysis["render"].update(
                {
                    "candidate_origin": "explicit_stage_edit_generation",
                    "stage_edit_promotion_contract": "fixture_stage_edit",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        product = report.metrics["ive_product_quality_floor"]
        self.assertEqual(
            product["evaluation_mode"],
            "modern_missing_role_conditioned_evidence",
        )
        self.assertFalse(
            product["metrics"]["legacy_whole_song_fallback_applied"]
        )
        self.assertIn(
            "ive_product_quality_floor_failed",
            {issue.code for issue in report.issues},
        )

    def test_explicit_app_hidden_comparison_is_not_a_product_delivery(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            analysis["request"] = {
                "group": "IVE",
                "edit_mode": "group",
                "source_mode": "automatic",
            }
            analysis["artifact_role"] = "comparison"
            analysis["app_result_exposure"] = {
                "contract": APP_RESULT_EXPOSURE_CONTRACT,
                "video_output_exposed": False,
                "delivery_completion_message_allowed": False,
            }
            analysis["render"].update(
                {
                    "artifact_role": "comparison",
                    "allow_failed_comparison_render": True,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        product = report.metrics["ive_product_quality_floor"]
        montage = report.metrics["rendered_montage_grammar"]
        self.assertFalse(product["required"])
        self.assertEqual(
            product["evaluation_mode"],
            "explicit_non_delivery_artifact",
        )
        self.assertFalse(montage["ive_product_scope_required"])

    def test_registered_stage_edit_flat_wide_montage_fails_closed(self) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            for choice in analysis["visual_timeline"]["choices"]:
                choice.update(
                    {
                        "shot_type": "wide",
                        "entry_shot_type": "wide",
                        "exit_shot_type": "wide",
                    }
                )
            analysis["section_plan"]["sections"][0]["role"] = (
                "chorus_like"
            )
            for shot in analysis["stage_edit_exact_shot_clock"]["shots"]:
                shot["section_role"] = "chorus_like"

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertFalse(montage["passed"])
        self.assertFalse(montage["recomputed_review"]["passed"])
        self.assertIn(
            "section_0:highlight_closeup_fraction_below_floor",
            montage["recomputed_review"]["failure_codes"],
        )
        self.assertIn(
            "production_rendered_montage_grammar_required_gate_failed",
            codes,
        )

    def test_registered_stage_edit_missing_montage_review_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis.pop("rendered_montage_grammar_review")

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_rendered_montage_grammar_evidence_missing_or_invalid",
            codes,
        )
        self.assertIn(
            "production_rendered_montage_grammar_required_gate_failed",
            codes,
        )

    def test_registered_stage_edit_missing_professional_boundary_evidence_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["rendered_montage_grammar_review"].pop(
                "professional_boundary_grammar_evidence"
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertFalse(montage["professional_boundary_evidence_valid"])
        self.assertIn(
            "production_professional_boundary_grammar_evidence_missing_or_invalid",
            codes,
        )

    def test_registered_stage_edit_professional_boundary_assessment_tamper_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["rendered_montage_grammar_review"][
                "professional_boundary_grammar_assessment"
            ]["passed"] = False

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertFalse(
            montage[
                "professional_boundary_assessment_matches_recomputation"
            ]
        )
        self.assertIn(
            "production_professional_boundary_grammar_assessment_mismatch",
            codes,
        )

    def test_registered_stage_edit_professional_boundary_plan_identity_tamper_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            review = analysis["rendered_montage_grammar_review"]
            original = ProfessionalBoundaryGrammarEvidence.from_mapping(
                review["professional_boundary_grammar_evidence"]
            )
            tampered = ProfessionalBoundaryGrammarEvidence(
                performance_id=f"{original.performance_id}-tampered",
                duration_seconds=original.duration_seconds,
                source_switch_times_seconds=(
                    original.source_switch_times_seconds
                ),
                native_internal_cut_times_seconds=(
                    original.native_internal_cut_times_seconds
                ),
                sections=original.sections,
                introduced_boundary_continuity=(
                    original.introduced_boundary_continuity
                ),
                editorial_transforms=original.editorial_transforms,
            )
            review["professional_boundary_grammar_evidence"] = (
                tampered.to_dict()
            )
            review["professional_boundary_grammar_assessment"] = (
                assess_professional_boundary_grammar(tampered).to_dict()
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertFalse(
            montage["professional_boundary_render_plan_binding_matches"]
        )
        self.assertIn(
            "production_professional_boundary_grammar_render_plan_binding_mismatch",
            codes,
        )

    def test_registered_stage_edit_self_hashed_transform_evidence_tamper_fails_authority_reconstruction(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            review = analysis["rendered_montage_grammar_review"]
            original = ProfessionalBoundaryGrammarEvidence.from_mapping(
                review["professional_boundary_grammar_evidence"]
            )
            transform = original.editorial_transforms
            tampered = ProfessionalBoundaryGrammarEvidence(
                performance_id=original.performance_id,
                duration_seconds=original.duration_seconds,
                source_switch_times_seconds=(
                    original.source_switch_times_seconds
                ),
                native_internal_cut_times_seconds=(
                    original.native_internal_cut_times_seconds
                ),
                sections=original.sections,
                introduced_boundary_continuity=(
                    original.introduced_boundary_continuity
                ),
                editorial_transforms=EditorialTransformEvidence(
                    segment_count=transform.segment_count,
                    applied_segment_count=2,
                    zoom_event_times_seconds=(
                        transform.zoom_event_times_seconds
                    ),
                    pan_event_times_seconds=transform.pan_event_times_seconds,
                    roll_event_times_seconds=(
                        transform.roll_event_times_seconds
                    ),
                ),
            )
            # Re-hash both evidence and assessment exactly as a self-consistent
            # producer would.  Independent QA must still reject it because the
            # final typed RenderPlan has three output-visible motion segments.
            review["professional_boundary_grammar_evidence"] = (
                tampered.to_dict()
            )
            review["professional_boundary_grammar_assessment"] = (
                assess_professional_boundary_grammar(tampered).to_dict()
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertTrue(montage["professional_boundary_evidence_valid"])
        self.assertFalse(
            montage[
                "professional_boundary_authority_reconstruction_matches"
            ]
        )
        self.assertIn(
            "production_professional_boundary_grammar_authority_reconstruction_mismatch",
            codes,
        )

    def test_registered_stage_edit_professional_boundary_legacy_pass_tamper_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["rendered_montage_grammar_review"][
                "passed_before_professional_boundary_grammar"
            ] = False

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        montage = report.metrics["rendered_montage_grammar"]
        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertFalse(
            montage["professional_boundary_legacy_pass_binding_matches"]
        )
        self.assertIn(
            "production_professional_boundary_grammar_legacy_pass_binding_mismatch",
            codes,
        )

    def test_registered_stage_edit_malformed_montage_profile_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["stage_edit_generation"][
                "professional_montage_grammar_profile"
            ]["content_sha256"] = "not-a-sha256"

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_rendered_montage_grammar_evidence_missing_or_invalid",
            codes,
        )
        self.assertIn(
            "production_rendered_montage_grammar_required_gate_failed",
            codes,
        )

    def test_registered_stage_edit_montage_profile_mismatch_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["rendered_montage_grammar_review"]["reference_profile"][
                "content_sha256"
            ] = "c" * 64

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_rendered_montage_grammar_profile_identity_mismatch",
            codes,
        )
        self.assertIn(
            "production_rendered_montage_grammar_required_gate_failed",
            codes,
        )

    def test_registered_stage_edit_montage_configuration_mismatch_fails_closed(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            analysis["rendered_montage_grammar_review"]["configuration"][
                "burst_maximum_run_seconds"
            ] = 99.0

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_rendered_montage_grammar_configuration_mismatch",
            codes,
        )
        self.assertIn(
            "production_rendered_montage_grammar_required_gate_failed",
            codes,
        )

    def test_registered_stage_edit_unknown_cut_motion_fails_with_reason_count(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(
                analysis,
                plan,
                selection,
                unknown_boundary_index=3,
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_cut_local_sync_unknown_boundaries",
            {issue.code for issue in report.issues},
        )
        self.assertEqual(
            report.metrics["production_sync"]["cut_local_sync"][
                "unknown_reason_counts"
            ],
            {"ambiguous_motion_lag": 1},
        )
        self.assertIn(
            "production_cut_local_sync_required_gate_failed",
            {issue.code for issue in report.issues},
        )

    def test_registered_stage_edit_cut_local_p95_above_half_frame_fails(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(
                analysis,
                plan,
                selection,
                residual_frames=0.60,
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_cut_local_sync_residual_exceeds_half_frame",
            {issue.code for issue in report.issues},
        )
        self.assertAlmostEqual(
            report.metrics["production_sync"]["cut_local_sync"][
                "residual_p95_frames"
            ],
            0.60,
        )
        self.assertIn(
            "production_cut_local_sync_required_gate_failed",
            {issue.code for issue in report.issues},
        )

    def test_trusted_registered_online_path_fails_when_persisted_markers_are_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory)
            )
            report = evaluate_post_render_qa(
                artifacts,
                runner=_FakeRunner(),
                trusted_registered_stage_edit=True,
            )

        codes = {issue.code for issue in report.issues}
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_cut_local_sync_activation_contract_invalid", codes
        )
        self.assertIn("production_cut_local_sync_evidence_missing", codes)
        self.assertIn("production_cut_local_sync_required_gate_failed", codes)

    def test_registered_activation_marker_disagreement_fails_even_with_valid_proof(
        self,
    ) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            _activate_registered_cut_local_sync(analysis, plan, selection)
            selection["stage_edit_render_promotion"]["activation"] = (
                "explicit_stage_edit_generation_request_only"
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                config=_registered_qa_config(),
                runner=_FakeRunner(),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "production_cut_local_sync_activation_contract_invalid",
            {issue.code for issue in report.issues},
        )
        self.assertTrue(
            report.metrics["production_sync"]["cut_local_sync"]["passed"]
        )

    def test_consistent_explicit_stage_edit_request_remains_cut_local_optional(
        self,
    ) -> None:
        def mutate(analysis: dict, _plan: dict, selection: dict) -> None:
            analysis["stage_edit_generation"] = {
                "activation": {"activation": "explicit_request_bundle"}
            }
            selection["stage_edit_render_promotion"] = {
                "activation": "explicit_stage_edit_generation_request_only",
                "candidate_pool": "stage_edit_generation_only",
            }
            analysis["render"]["candidate_origin"] = (
                "explicit_stage_edit_generation"
            )
            analysis["render"]["stage_edit_promotion_contract"] = (
                "fixture_stage_edit"
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        cut_local = report.metrics["production_sync"]["cut_local_sync"]
        self.assertFalse(cut_local["required"])
        self.assertIsNone(cut_local["passed"])

    def test_domain_aware_v3_exact_eight_reviews_are_plan_bound_and_accepted(self) -> None:
        def mutate(_analysis: dict, plan: dict, selection: dict) -> None:
            _promote_review_to_domain_aware_v3(
                selection["final_selected_review"], plan
            )
            for candidate, complete_candidate in zip(
                selection["candidates"],
                selection["complete_candidates"],
                strict=True,
            ):
                complete_candidate["visual_timeline"] = {
                    "choices": [
                        deepcopy(segment)
                        for segment in plan["segments"]
                    ]
                }
                complete_candidate["group_transitions"] = {
                    "transitions": deepcopy(plan["transitions"])
                }
                _promote_review_to_domain_aware_v3(candidate["review"], plan)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        self.assertEqual(
            report.metrics["selection"]["actual_successful_review_count"], 8
        )
        self.assertTrue(
            report.metrics["selection"]["checks"][
                "final_selected_decoded_review_contract_valid"
            ]
        )

    def test_unsupported_proxy_review_basis_fails_closed(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            for review in [
                selection["final_selected_review"],
                *(candidate["review"] for candidate in selection["candidates"]),
            ]:
                review["analysis"] = "decoded_output_proxy_domain_aware_quality_review_v99"
                review["schema_version"] = 99

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_actual_successful_review_count_matches",
            {issue.code for issue in report.issues},
        )

    def test_plan_contract_uses_canonical_offset_and_separates_native_cuts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
        canonical_offset = 4.0
        analysis["common_master_range"] = {
            "start_seconds": canonical_offset,
            "end_seconds": canonical_offset + _DURATION,
            "duration_seconds": _DURATION,
        }
        source_sequence = [0, 0, 0, 1, 2, 3, 4, 5, 6]
        for index, source_index in enumerate(source_sequence):
            plan["segments"][index]["source_index"] = source_index
            analysis["visual_timeline"]["choices"][index][
                "source_index"
            ] = source_index
            final_segment = analysis["render"]["final_render_segments"][index]
            final_segment["request_source_index"] = source_index
            final_segment["canonical_master_start"] += canonical_offset
            final_segment["canonical_master_end"] += canonical_offset
        plan["source_usage"]["distinct_source_count"] = 7
        plan["source_usage"]["used_source_indices"] = list(range(7))
        plan["transitions"][0].update(
            {"tier": "D", "reason": "same_source_hold_no_visible_switch"}
        )
        plan["transitions"][1].update(
            {"tier": "C", "reason": "same_source_internal_camera_cut"}
        )
        analysis["group_transitions"].update(
            {
                "introduced_hard_cut_count": 7,
                "same_source_hold_count": 1,
                "hard_cut_boundary_count": 8,
                "transitions": deepcopy(plan["transitions"]),
            }
        )
        timeline = analysis["timeline_review"]
        timeline["source_switch_count"] = 6
        timeline["unique_source_count"] = 7
        for index, boundary in enumerate(timeline["boundaries"]):
            boundary["at_seconds"] += canonical_offset
            boundary["outgoing_source_id"] = str(source_sequence[index])
            boundary["incoming_source_id"] = str(source_sequence[index + 1])
            boundary["source_switch"] = (
                source_sequence[index] != source_sequence[index + 1]
            )
        final_review = selection["final_selected_review"]
        for index, boundary in enumerate(final_review["boundaries"]):
            boundary["outgoing_source_index"] = source_sequence[index]
            boundary["incoming_source_index"] = source_sequence[index + 1]

        collector = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis,
            plan,
            final_review,
            collector,
        )

        self.assertEqual(collector.issues, [])
        self.assertEqual(metrics["visible_source_switch_count"], 6)
        self.assertEqual(metrics["introduced_hard_cut_count"], 7)
        self.assertEqual(metrics["derived_native_source_cut_count"], 1)
        self.assertTrue(
            metrics["checks"]["final_render_segments_match_plan"]
        )
        self.assertTrue(
            metrics["checks"]["timeline_boundary_schedule_matches_plan"]
        )

    def test_v4_group_transition_authority_tamper_fails_plan_qa(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
        group = analysis["group_transitions"]
        group["analysis_basis"] = (
            "tiered_future_face_head_sface_appearance_body_continuity_v4"
        )
        group["transition_authority"] = StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).to_dict()
        clean = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis, plan, selection["final_selected_review"], clean
        )
        self.assertTrue(
            metrics["checks"]["renderer_owned_transition_authority_exact"]
        )
        group["transition_authority"][
            "zoom_match_whip_pan_render_effects_authorized"
        ] = True
        tampered = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis, plan, selection["final_selected_review"], tampered
        )
        self.assertFalse(
            metrics["checks"]["renderer_owned_transition_authority_exact"]
        )
        self.assertTrue(
            any(
                issue.code
                == "plan_invariant_renderer_owned_transition_authority_exact"
                for issue in tampered.issues
            )
        )

    def test_member_direct_plan_requires_every_eligible_fancam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
        used = list(range(8))
        analysis["alignment_eligibility"] = [
            {
                "index_domain": "request_source_index",
                "source_index": source_index,
                "render_eligible": True,
            }
            for source_index in used
        ]
        analysis["member_direct_full_frame"] = {
            "enabled": True,
            "eligible_request_source_indices": used,
            "minimum_required_distinct_source_count": len(used),
            "required_used_request_source_indices": used,
            "final_source_usage": deepcopy(plan["source_usage"]),
        }

        clean = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis,
            plan,
            selection["final_selected_review"],
            clean,
        )

        self.assertTrue(
            metrics["checks"]["member_direct_all_eligible_sources_used"]
        )
        self.assertFalse(
            any(
                issue.code
                == "plan_invariant_member_direct_all_eligible_sources_used"
                for issue in clean.issues
            )
        )

        analysis["member_direct_full_frame"][
            "required_used_request_source_indices"
        ] = used[:-1]
        tampered = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis,
            plan,
            selection["final_selected_review"],
            tampered,
        )

        self.assertFalse(
            metrics["checks"]["member_direct_all_eligible_sources_used"]
        )
        self.assertTrue(
            any(
                issue.code
                == "plan_invariant_member_direct_all_eligible_sources_used"
                for issue in tampered.issues
            )
        )

        analysis["request"] = {
            "group": "IVE",
            "source_mode": "automatic",
            "edit_mode": "member",
        }
        analysis["render"] = {"direct_full_frame": True}
        del analysis["member_direct_full_frame"]
        missing_block = _IssueCollector()
        metrics = _validate_plan_invariants(
            analysis,
            plan,
            selection["final_selected_review"],
            missing_block,
        )
        self.assertFalse(
            metrics["checks"]["member_direct_all_eligible_sources_used"]
        )
        self.assertTrue(
            any(
                issue.code
                == "plan_invariant_member_direct_all_eligible_sources_used"
                for issue in missing_block.issues
            )
        )

    def test_generic_motion_reason_is_accepted_only_when_bound_to_render_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, _selection = _fixture_payloads(
                Path(directory)
            )
        segment_index = 4
        motion_segment = analysis["editorial_motion"]["segments"][segment_index]
        motion_segment["reasons"] = []
        plan["segments"][segment_index]["editorial_transform"] = {
            "reason": "reference_calibrated_camera_motion",
            "keyframes": deepcopy(motion_segment["keyframes"]),
        }
        collector = _IssueCollector()

        _transform_metrics(analysis, plan, PostRenderQAConfig(), collector)

        self.assertNotIn(
            "editorial_motion_track_payload_invalid",
            {issue.code for issue in collector.issues},
        )

        plan["segments"][segment_index]["editorial_transform"]["keyframes"][1][
            "zoom"
        ] = 1.17
        tampered_collector = _IssueCollector()
        _transform_metrics(
            analysis,
            plan,
            PostRenderQAConfig(),
            tampered_collector,
        )
        self.assertIn(
            "editorial_motion_track_payload_invalid",
            {issue.code for issue in tampered_collector.issues},
        )

    def test_duration_aware_dissolve_endpoints_are_reviewed_and_reported(self) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            frame_count = 6
            duration = frame_count / _FPS
            transition = plan["transitions"][0]
            transition.update(
                {
                    "kind": "face_dissolve",
                    "frame_count": frame_count,
                    "duration_seconds": duration,
                    "tier": "A",
                }
            )
            analysis["group_transitions"]["transitions"] = deepcopy(
                plan["transitions"]
            )
            analysis["group_transitions"]["dissolve_count"] = 1
            analysis["group_transitions"]["introduced_hard_cut_count"] = 7
            analysis["group_transitions"]["hard_cut_boundary_count"] = 7

            review = selection["final_selected_review"]
            review["hard_cut_count"] = 7
            review["dissolve_count"] = 1
            boundary = review["boundaries"][0]
            cut_frame = boundary["cut_frame_index"]
            offsets = [-2, -1, frame_count, frame_count + 1]
            boundary["transition_kind"] = "face_dissolve"
            boundary["transition_frame_count"] = frame_count
            boundary["transition_duration_seconds"] = duration
            boundary["sample_offsets"] = offsets
            boundary["frame_indices"] = [cut_frame + value for value in offsets]
            for offset, frame_metric in zip(
                offsets, boundary["frame_metrics"], strict=True
            ):
                frame_metric["offset"] = offset
                frame_metric["frame_index"] = cut_frame + offset
            selection["candidates"][0]["review"] = deepcopy(review)

            motion = analysis["editorial_motion"]
            motion["blocked_dissolve_segment_count"] = 2
            motion["segments"][0].update(
                {"applied": False, "reasons": [], "keyframes": []}
            )
            motion["segments"][8].update(
                {
                    "applied": True,
                    "reasons": ["face_scale_position_matched_hard_cut"],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "zoom": 1.0,
                            "center_x": 0.5,
                            "center_y": 0.5,
                            "roll_degrees": 0.0,
                        },
                        {
                            "time_seconds": 0.75,
                            "zoom": 1.08,
                            "center_x": 0.51,
                            "center_y": 0.5,
                            "roll_degrees": 0.0,
                        },
                    ],
                }
            )
            motion["segments"][7]["reasons"] = [
                "face_scale_position_matched_hard_cut"
            ]

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        self.assertTrue(
            report.metrics["proxy_quality_and_geometry"][
                "boundary_sample_contract_valid"
            ]
        )
        self.assertEqual(
            report.metrics["proxy_quality_and_geometry"][
                "subject_discontinuities_by_transition_kind"
            ]["face_dissolve"]["face_scale_discontinuity"]["count"],
            1,
        )

    def test_known_proxy_boundary_issue_is_conditional_not_silently_professional(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            review = selection["final_selected_review"]
            review["boundary_score"]["issue_count"] = 1
            review["boundary_score"]["issue_fraction"] = 0.125
            review["boundary_issue_indices"] = [3]
            review["boundaries"][3]["passed"] = False
            review["boundaries"][3]["issue_reasons"] = ["color_shock"]
            selection["candidates"][0]["review"] = deepcopy(review)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.CONDITIONAL)
        self.assertIn(
            "proxy_boundary_issues_require_manual_review",
            {issue.code for issue in report.issues},
        )

    def test_nonselected_sanitized_stub_is_not_a_successful_decoded_review(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            selection["candidates"][7]["review"] = {
                "passed": True,
                "proxy": {
                    "path": None,
                    "temporary_proxy_removed_after_review": True,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_actual_successful_review_count_matches",
            {issue.code for issue in report.issues},
        )

    def test_candidate_proxy_duration_is_bound_to_render_plan(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            selection["final_selected_review"]["proxy"][
                "duration_seconds"
            ] = 31.0
            for candidate in selection["candidates"]:
                candidate["review"]["proxy"]["duration_seconds"] = 31.0

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_actual_successful_review_count_matches",
            {issue.code for issue in report.issues},
        )

    def test_per_source_quality_stubs_cannot_satisfy_final_review(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            review = selection["final_selected_review"]
            review["source_quality"]["per_source"] = [
                {"source_index": index} for index in range(8)
            ]
            selection["candidates"][0]["review"] = deepcopy(review)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_final_selected_decoded_review_contract_valid",
            {issue.code for issue in report.issues},
        )

    def test_per_source_quality_spans_must_match_decoded_distributions(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            review = selection["final_selected_review"]
            review["source_quality"]["p10_span"] = 0.0
            selection["candidates"][0]["review"] = deepcopy(review)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_final_selected_decoded_review_contract_valid",
            {issue.code for issue in report.issues},
        )

    def test_partial_final_review_duration_is_independently_plan_bound(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            original_review = deepcopy(selection["candidates"][0]["review"])
            original_review["passed"] = False
            selection["candidates"][0]["passed"] = False
            selection["candidates"][0]["review"] = original_review
            selection["passing_candidate_count"] = 7
            selection["final_selected_version"] = "partial_reedit"
            selection["final_selected_review"]["proxy"][
                "duration_seconds"
            ] = 31.0
            selection["proxy_triggered_partial_reedit"] = {
                "attempted": True,
                "improved": True,
                "selected_version": "partial_reedit",
                "before_review": deepcopy(original_review),
                "after_review": deepcopy(selection["final_selected_review"]),
                "outside_issue_boundaries_preserved": True,
                "changed_boundary_indices": [0],
                "invariant_checks": {
                    "segment_count_preserved": True,
                    "source_sequence_preserved": True,
                    "render_source_sequence_preserved": True,
                    "canonical_first_last_range_preserved": True,
                    "outside_issue_boundary_timestamps_preserved": True,
                    "actual_changed_boundaries_subset_of_requested": True,
                    "output_frame_count_preserved": True,
                    "render_duration_preserved": True,
                },
                "repair": {"repairs": [_repair(0, offset=8)]},
            }

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "selection_final_selected_decoded_review_contract_valid",
            {issue.code for issue in report.issues},
        )

    def test_full_decode_failure_and_source_local_reuse_fail_closed(self) -> None:
        def mutate(_analysis: dict, plan: dict, _selection: dict) -> None:
            plan["segments"][8]["source_local_start"] = 2.0
            plan["segments"][8]["source_local_end"] = 6.0
            plan["segments"][8]["source_start"] = 2.0
            plan["segments"][8]["source_end"] = 6.0

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(
                artifacts,
                runner=_FakeRunner(ffmpeg_returncode=1),
            )

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        codes = {issue.code for issue in report.issues}
        self.assertIn("full_decode_failed", codes)
        self.assertIn("source_local_content_reused", codes)
        self.assertGreater(
            report.metrics["source_local_reuse"]["reused_content_ratio"], 0.0
        )

    def test_timeline_source_shares_must_match_render_plan_durations(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            review = analysis["timeline_review"]
            durations = review["source_duration_seconds"]
            durations["0"], durations["1"] = durations["1"], durations["0"]
            review["per_source_share"] = {
                key: value / _DURATION for key, value in durations.items()
            }

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "source_share_payload_mismatch",
            {issue.code for issue in report.issues},
        )

    def test_partial_reedit_allows_eight_frames_only_with_improvement_and_invariants(self) -> None:
        def mutate(analysis: dict, plan: dict, selection: dict) -> None:
            shift = 8.0 / _FPS
            boundary_seconds = _REGULAR_SEGMENT_DURATION + shift
            second_segment_end = 2.0 * _REGULAR_SEGMENT_DURATION
            original_review = deepcopy(selection["candidates"][0]["review"])
            original_review["passed"] = False
            original_review["weighted_score"] = 55.0
            original_review["boundary_score"]["issue_count"] = 1
            original_review["boundary_score"]["issue_fraction"] = 0.125
            original_review["boundary_issue_indices"] = [0]
            original_review["boundaries"][0]["passed"] = False
            original_review["boundaries"][0]["issue_reasons"] = [
                "visual_discontinuity"
            ]
            selection["candidates"][0]["passed"] = False
            selection["candidates"][0]["review"] = original_review
            selection["passing_candidate_count"] = 7
            selection["final_selected_version"] = "partial_reedit"
            selection["proxy_triggered_partial_reedit"] = {
                "attempted": True,
                "improved": True,
                "selected_version": "partial_reedit",
                "before_review": deepcopy(original_review),
                "after_review": deepcopy(selection["final_selected_review"]),
                "outside_issue_boundaries_preserved": True,
                "changed_boundary_indices": [0],
                "invariant_checks": {
                    "segment_count_preserved": True,
                    "source_sequence_preserved": True,
                    "render_source_sequence_preserved": True,
                    "canonical_first_last_range_preserved": True,
                    "outside_issue_boundary_timestamps_preserved": True,
                    "actual_changed_boundaries_subset_of_requested": True,
                    "output_frame_count_preserved": True,
                    "render_duration_preserved": True,
                },
                "repair": {"repairs": [_repair(0, offset=8)]},
            }

            left = plan["segments"][0]
            right = plan["segments"][1]
            left["source_end"] = boundary_seconds
            left["source_local_end"] = boundary_seconds
            left["duration"] = boundary_seconds
            left["effective_source_duration"] = boundary_seconds
            right["source_start"] = boundary_seconds
            right["source_local_start"] = boundary_seconds
            right["master_start"] = boundary_seconds
            right["duration"] = second_segment_end - boundary_seconds
            right["effective_source_duration"] = second_segment_end - boundary_seconds

            analysis["timeline_review"]["boundaries"][0][
                "at_seconds"
            ] = boundary_seconds
            left_run = analysis["timeline_review"]["source_runs"][0]
            right_run = analysis["timeline_review"]["source_runs"][1]
            left_run["end_seconds"] = boundary_seconds
            left_run["duration_seconds"] = boundary_seconds
            right_run["start_seconds"] = boundary_seconds
            right_run["duration_seconds"] = second_segment_end - boundary_seconds
            analysis["timeline_review"]["source_duration_seconds"]["0"] = (
                7.5 + shift
            )
            analysis["timeline_review"]["source_duration_seconds"]["1"] = (
                _REGULAR_SEGMENT_DURATION - shift
            )
            analysis["timeline_review"]["per_source_share"]["0"] = (
                7.5 + shift
            ) / _DURATION
            analysis["timeline_review"]["per_source_share"]["1"] = (
                _REGULAR_SEGMENT_DURATION - shift
            ) / _DURATION
            analysis["visual_timeline"]["choices"][0][
                "canonical_master_end"
            ] = boundary_seconds
            analysis["visual_timeline"]["choices"][1][
                "canonical_master_start"
            ] = boundary_seconds

            final_segments = analysis["render"]["final_render_segments"]
            final_segments[0]["canonical_master_end"] = boundary_seconds
            final_segments[0]["output_end_seconds"] = boundary_seconds
            final_segments[0]["source_local_end"] = boundary_seconds
            final_segments[1]["canonical_master_start"] = boundary_seconds
            final_segments[1]["output_start_seconds"] = boundary_seconds
            final_segments[1]["source_local_start"] = boundary_seconds

            review_boundary = selection["final_selected_review"]["boundaries"][0]
            cut_frame = int(round(boundary_seconds * _FPS))
            review_boundary["cut_frame_index"] = cut_frame
            review_boundary["frame_indices"] = [
                cut_frame + offset for offset in (-2, -1, 0, 1)
            ]
            for offset, frame_metric in zip(
                (-2, -1, 0, 1), review_boundary["frame_metrics"], strict=True
            ):
                frame_metric["frame_index"] = cut_frame + offset
            selection["proxy_triggered_partial_reedit"]["after_review"] = deepcopy(
                selection["final_selected_review"]
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        repair = report.metrics["beat_and_phrase"]["repair_radius_checks"][0]
        self.assertEqual(repair["selected_frame_offset"], 8)
        self.assertEqual(repair["maximum_allowed_frames"], 8)
        self.assertTrue(repair["passed"])

    def test_windowed_semimarkov_redecode_passes_only_with_changed_revalidated_lineage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
        _install_valid_windowed_semimarkov_redecode(selection)
        collector = _IssueCollector()
        metrics, _review = _validate_selection(
            analysis,
            plan,
            selection,
            _registered_qa_config(),
            collector,
        )

        self.assertFalse(collector.issues)
        contract = metrics["windowed_semimarkov_redecode"]
        self.assertTrue(contract["repair_changed"])
        self.assertTrue(contract["decoded_final_review_contract"])
        self.assertTrue(contract["candidate_lineage_contract"])
        self.assertTrue(contract["promotion_lineage_contract"])

    def test_windowed_semimarkov_redecode_tampering_fails_closed(self) -> None:
        cases = {
            "unchanged": lambda selection: selection[
                "proxy_triggered_partial_reedit"
            ].__setitem__("changed", False),
            "stale_final_review": lambda selection: selection[
                "proxy_triggered_partial_reedit"
            ].__setitem__("after_review", {"passed": True}),
            "lineage_removed": lambda selection: selection[
                "proxy_triggered_partial_reedit"
            ].__setitem__("candidate_lineage_after", []),
        }
        for name, tamper in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                _artifacts, analysis, plan, selection = _fixture_payloads(
                    Path(directory)
                )
            _install_valid_windowed_semimarkov_redecode(selection)
            tamper(selection)
            collector = _IssueCollector()
            _validate_selection(
                analysis,
                plan,
                selection,
                _registered_qa_config(),
                collector,
            )

            self.assertTrue(
                any(
                    issue.code.startswith("windowed_semimarkov_redecode_")
                    for issue in collector.issues
                )
            )

    def test_windowed_final_boundary_bundle_supersedes_stale_complete_candidate(
        self,
    ) -> None:
        final_repair = _repair(0, offset=2)
        final_repair["selected_visual_evidence"]["bundle_marker"] = "final"
        stale_repair = _repair(0, offset=-3)
        stale_repair["selected_visual_evidence"]["bundle_marker"] = "stale"
        plans, repairs = _selected_boundary_repair_contracts(
            {"visual_boundary_repairs": {"repairs": [final_repair]}},
            {
                "final_selected_version": "windowed_semimarkov_redecode",
                "selected_candidate_index": 0,
                "complete_candidates": [
                    {
                        "candidate_index": 0,
                        "boundary_repairs": {"repairs": [stale_repair]},
                    }
                ],
            },
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(repairs[0]["result"]["selected_frame_offset"], 2)
        self.assertEqual(
            repairs[0]["selected_visual_evidence"]["bundle_marker"],
            "final",
        )

    def test_stale_partial_metadata_cannot_expand_original_repair_radius(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            selection["proxy_triggered_partial_reedit"].update(
                {
                    "attempted": True,
                    "selected_version": "original",
                    "changed_boundary_indices": [0],
                    "repair": {"repairs": [_repair(0, offset=8)]},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)
        repair = report.metrics["beat_and_phrase"]["repair_radius_checks"][0]
        self.assertEqual(repair["selected_frame_offset"], 0)
        self.assertEqual(repair["maximum_allowed_frames"], 3)

    def test_partial_changed_indices_require_in_range_changed_repair_evidence(self) -> None:
        def mutate(_analysis: dict, _plan: dict, selection: dict) -> None:
            original_review = deepcopy(selection["candidates"][0]["review"])
            original_review["passed"] = False
            selection["candidates"][0]["passed"] = False
            selection["candidates"][0]["review"] = original_review
            selection["passing_candidate_count"] = 7
            selection["final_selected_version"] = "partial_reedit"
            selection["proxy_triggered_partial_reedit"] = {
                "attempted": True,
                "improved": True,
                "selected_version": "partial_reedit",
                "before_review": deepcopy(original_review),
                "after_review": deepcopy(selection["final_selected_review"]),
                "outside_issue_boundaries_preserved": True,
                "changed_boundary_indices": [999],
                "invariant_checks": {
                    "segment_count_preserved": True,
                    "source_sequence_preserved": True,
                    "render_source_sequence_preserved": True,
                    "canonical_first_last_range_preserved": True,
                    "outside_issue_boundary_timestamps_preserved": True,
                    "actual_changed_boundaries_subset_of_requested": True,
                    "output_frame_count_preserved": True,
                    "render_duration_preserved": True,
                },
                "repair": {"repairs": [_repair(0, offset=8)]},
            }

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "partial_reedit_changed_boundary_indices_valid",
            {issue.code for issue in report.issues},
        )

    def test_timeline_switch_flags_and_transform_payload_fail_closed(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            for boundary in analysis["timeline_review"]["boundaries"]:
                boundary["source_switch"] = False
            analysis["editorial_motion"]["segments"] = analysis[
                "editorial_motion"
            ]["segments"][:-1]
            analysis["editorial_motion"]["segments"][0]["keyframes"][1][
                "zoom"
            ] = 0.90

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        codes = {issue.code for issue in report.issues}
        self.assertIn("plan_invariant_timeline_boundary_schedule_matches_plan", codes)
        self.assertIn("editorial_motion_one_motion_segment_per_plan_segment", codes)
        self.assertIn("editorial_transform_safety_bound_exceeded", codes)

    def test_serialized_transform_event_count_forgery_fails_exact_reconciliation(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            motion = analysis["editorial_motion"]
            motion["pan_event_count"] = 1
            motion["boundary_match_count"] = 3
            motion["interior_zoom_pulse_count"] = 3

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        codes = {issue.code for issue in report.issues}
        self.assertIn("editorial_motion_pan_track_evidence", codes)
        self.assertIn(
            "editorial_motion_boundary_event_count_matches_reasons", codes
        )
        self.assertIn(
            "editorial_motion_interior_zoom_count_matches_reasons", codes
        )

    def test_transform_reason_detail_suffix_preserves_category_counts(self) -> None:
        def mutate(analysis: dict, _plan: dict, _selection: dict) -> None:
            for segment in analysis["editorial_motion"]["segments"]:
                segment["reasons"] = [
                    f"{reason}:fixture-detail"
                    for reason in segment["reasons"]
                ]

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(
                Path(directory), mutate=mutate
            )
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS)

    def test_missing_audio_and_probe_timeout_fail_closed_and_decode_continues(self) -> None:
        class TimeoutProbeRunner(_FakeRunner):
            def __call__(self, command, timeout):
                values = tuple(command)
                if Path(values[0]).name == "ffprobe":
                    self.calls.append((values, timeout))
                    raise subprocess.TimeoutExpired(values, timeout)
                return super().__call__(command, timeout)

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(Path(directory))
            timeout_runner = TimeoutProbeRunner()
            timeout_report = evaluate_post_render_qa(
                artifacts, runner=timeout_runner
            )
            payload = _ffprobe_payload()
            payload["streams"] = [payload["streams"][0]]
            audio_report = evaluate_post_render_qa(
                artifacts,
                runner=_FakeRunner(ffprobe_payload=payload),
            )

        self.assertEqual(timeout_report.verdict, PostRenderQAVerdict.FAIL)
        self.assertEqual(len(timeout_runner.calls), 2)
        self.assertIn(
            "ffprobe_unavailable_or_invalid",
            {issue.code for issue in timeout_report.issues},
        )
        self.assertEqual(audio_report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "final_media_exactly_one_audio_stream",
            {issue.code for issue in audio_report.issues},
        )

    def test_hidden_candidate_proxy_residue_blocks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _analysis, _plan, _selection = _persist_fixture(root)
            residue = root / ".ive-crosscut-candidate-proxy-stale.mp4"
            residue.write_bytes(b"stale proxy")

            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn(
            "temporary_proxy_residue",
            {issue.code for issue in report.issues},
        )

    def test_malformed_ffprobe_json_fails_but_still_attempts_full_decode(self) -> None:
        class MalformedRunner(_FakeRunner):
            def __call__(self, command, timeout):
                result = super().__call__(command, timeout)
                if Path(tuple(command)[0]).name == "ffprobe":
                    return subprocess.CompletedProcess(
                        result.args, 0, stdout="{not-json", stderr=""
                    )
                return result

        with tempfile.TemporaryDirectory() as directory:
            artifacts, _analysis, _plan, _selection = _persist_fixture(Path(directory))
            runner = MalformedRunner()
            report = evaluate_post_render_qa(artifacts, runner=runner)

        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertEqual(len(runner.calls), 2)
        self.assertIn(
            "ffprobe_unavailable_or_invalid",
            {issue.code for issue in report.issues},
        )

    def test_atomic_write_replaces_existing_report_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _analysis, _plan, _selection = _persist_fixture(root)
            report = evaluate_post_render_qa(artifacts, runner=_FakeRunner())
            destination = artifacts.default_report_path
            destination.write_text("old report", encoding="utf-8")

            written = report.write_atomic()

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(written, destination.resolve())
            self.assertEqual(payload["verdict"], "PASS")
            self.assertEqual(
                tuple(root.glob(f".{destination.name}.*.tmp")), ()
            )

            preserved = destination.read_bytes()
            with patch(
                "pathlib.Path.replace",
                side_effect=OSError("synthetic atomic replace failure"),
            ):
                with self.assertRaises(OSError):
                    report.write_atomic()
            self.assertEqual(destination.read_bytes(), preserved)
            self.assertEqual(
                tuple(root.glob(f".{destination.name}.*.tmp")), ()
            )

            invalid = PostRenderQAReport(
                artifact_set=artifacts,
                verdict=PostRenderQAVerdict.PASS,
                metrics={"not_finite": math.nan},
                issues=(),
                policy=PostRenderQAConfig(),
                created_at=datetime(2026, 8, 23, tzinfo=UTC),
            )
            preserved = destination.read_bytes()
            with self.assertRaises(ValueError):
                invalid.write_atomic()
            self.assertEqual(destination.read_bytes(), preserved)
            self.assertEqual(
                tuple(root.glob(f".{destination.name}.*.tmp")), ()
            )


if __name__ == "__main__":
    unittest.main()
