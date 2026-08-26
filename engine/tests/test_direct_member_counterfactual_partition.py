from __future__ import annotations

from copy import deepcopy
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    DirectMemberVisualSelectionError,
)
from crosscut.draft import DraftSegment
from crosscut.pipeline import (
    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
    _DIRECT_MEMBER_MAX_PROOF_DIRECTED_BOUNDARY_MERGE_COUNT,
    _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT,
    _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT,
    _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT,
    _DirectFullFrameBoundaryShift,
    _DirectFullFramePartitionProfile,
    _DirectFullFrameVisualFrontierSplit,
    _compact_direct_partition_failure,
    _compact_direct_severe_counterfactual,
    _direct_counterfactual_boundary_proof,
    _direct_counterfactual_secondary_signature_action,
    _direct_full_frame_proof_directed_child_profiles,
    _direct_full_frame_proof_directed_merge_profiles,
    _direct_full_frame_proof_directed_repair_profiles,
    _direct_full_frame_spans_from_partition,
    _direct_full_frame_visual_frontier_split_child_profiles,
    _direct_member_failure_report_fallback_payload,
    _direct_partition_signature_sha256,
    _direct_partition_failure_result,
    _direct_partition_aggregate_outcome,
    _direct_partition_current_shifted_boundary_indices,
    _direct_severe_counterfactual_retry_is_authorized,
)


def _diagnostics(*, span_count: int = 4) -> dict:
    boundaries = []
    for boundary_index in range(span_count - 1):
        boundaries.append(
            {
                "boundary_index": boundary_index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": [0],
                "nonsevere_cross_source_edge_count": 1,
            }
        )
    return {
        "failure": "validity_quota_path_exhausted",
        "constraint_infeasibility_proven": True,
        "hard_boundary_connectivity": {
            "analysis_basis": (
                "content_valid_nonsevere_no_repeat_reachability_v1"
            ),
            "boundaries": boundaries,
        },
        "search": {
            "exact_lifeline": {
                "invoked": True,
                "state_capacity_exhausted": False,
                "duration_state_quantization_applied": False,
            },
            "severe_transition_counterfactual": {
                "contract": (
                    "diagnostic_only_severe_transition_counterfactual_v1"
                ),
                "optimization_contract": (
                    "exact_iterative_deepening_minimum_severe_transition_count_"
                    "then_deterministic_backward_source_order_v1"
                ),
                "severe_transition_cost_policy": (
                    "one_per_adjacent_transition_with_any_original_severe_flag"
                ),
                "tie_break_contract": (
                    "final_source_order_then_reverse_predecessor_"
                    "severe_cost_mismatch_cost_source_order_v1"
                ),
                "invoked": True,
                "original_hard_constraint_sha256": "a" * 64,
                "original_failure_certificate_sha256": "b" * 64,
                "severe_disabled_hard_constraint_sha256": "c" * 64,
                "severe_transition_budget_mask_sha256": "e" * 64,
                "severe_disabled_witness_sha256": "d" * 64,
                "minimum_witness_path_sha256": "d" * 64,
                "minimum_severe_transition_count": 1,
                "maximum_diagnostic_severe_transition_budget": 3,
                "zero_budget_hard_proof": {
                    "severe_transition_budget": 0,
                    "path_found": False,
                    "explored_state_count": 20,
                    "memoized_failed_state_count": 19,
                    "state_capacity_exhausted": False,
                    "constraint_infeasibility_proven": True,
                    "duration_state_quantization_applied": False,
                    "certificate_sha256": "b" * 64,
                },
                "lower_budget_hard_proofs": [
                    {
                        "severe_transition_budget": 0,
                        "path_found": False,
                        "explored_state_count": 20,
                        "memoized_failed_state_count": 19,
                        "state_capacity_exhausted": False,
                        "constraint_infeasibility_proven": True,
                        "duration_state_quantization_applied": False,
                        "certificate_sha256": "b" * 64,
                    }
                ],
                "budget_attempts": [
                    {
                        "severe_transition_budget": 1,
                        "path_found": True,
                        "explored_state_count": 31,
                        "memoized_failed_state_count": 21,
                        "state_capacity_exhausted": False,
                        "constraint_infeasibility_proven": False,
                        "duration_state_quantization_applied": False,
                        "certificate_sha256": "f" * 64,
                    }
                ],
                "optimality_proven": True,
                "production_single_split_proposal_authorized": True,
                "witness_found": True,
                "integrity_check_passed": True,
                "blocking_boundary_indices": [1],
                "blocking_transitions": [
                    {
                        "boundary_index": 1,
                        "outgoing_eligible_source_index": 0,
                        "incoming_eligible_source_index": 1,
                        "original_severe_flags": ["pose_jump"],
                    }
                ],
                "maximum_duration_relative_deviation": 0.04,
                "state_capacity_exhausted": False,
                "state_limit_per_budget": 1_000_000,
                "constraint_infeasibility_proven": False,
                "duration_state_quantization_applied": False,
                "candidate_validity_semantics_changed": False,
                "source_quota_semantics_changed": False,
                "exact_duration_semantics_changed": False,
                "source_order_semantics_changed": False,
                "no_adjacent_repeat_semantics_changed": False,
                "severe_transition_flags_disabled": True,
                "original_matrices_mutated": False,
                "selection_eligible": False,
                "render_authorized": False,
                "visual_gates_relaxed": False,
            },
        },
    }


def _timing_segments() -> tuple[DraftSegment, ...]:
    duration = 1.25
    return tuple(
        DraftSegment(
            segment_index=index,
            source_index=0,
            canonical_master_start=index * duration,
            canonical_master_end=(index + 1) * duration,
            output_start_seconds=index * duration,
            source_local_start=index * duration,
            source_local_end=(index + 1) * duration,
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=1.0,
            start_boundary_reasons=("beat",),
            end_boundary_reasons=("beat",),
        )
        for index in range(12)
    )


def _parent_profile() -> _DirectFullFramePartitionProfile:
    segments = _timing_segments()
    unit_counts = (1, 3, 4, 4)
    existing_split = _DirectFullFrameVisualFrontierSplit(
        parent_span_index=0,
        split_unit_offset=1,
        left_unit_count=1,
        right_unit_count=3,
        canonical_master_seconds=1.25,
        boundary_reasons=("beat",),
        trigger="proven_forward_connectivity_frontier_visual_micro_split",
        parent_profile_id="prior",
        parent_partition_signature_sha256="e" * 64,
        proof_failure="validity_quota_path_exhausted",
        proof_failed_segment_index=1,
        proof_frontier_boundary_index=0,
        proof_forward_reachable_left_source_indices=(0,),
        proof_right_valid_source_indices=(0, 1),
        evidence_contract="visual_bracket_bound_existing_timing_boundary_v1",
        evidence_kind="sample_hard_safety_change_bracket",
        evidence_source_indices=(1,),
        evidence_master_seconds=(),
        evidence_bracket_master_seconds=(1.1, 1.4),
    )
    return _DirectFullFramePartitionProfile(
        profile_id="one-visual-split-parent",
        unit_counts=unit_counts,
        spans=_direct_full_frame_spans_from_partition(segments, unit_counts),
        visual_frontier_splits=(existing_split,),
    )


def _professional_timing_segments(
    count: int,
) -> tuple[DraftSegment, ...]:
    duration = 0.9
    return tuple(
        DraftSegment(
            segment_index=index,
            source_index=0,
            canonical_master_start=index * duration,
            canonical_master_end=(index + 1) * duration,
            output_start_seconds=index * duration,
            source_local_start=index * duration,
            source_local_end=(index + 1) * duration,
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=1.0,
            start_boundary_reasons=("beat",),
            end_boundary_reasons=("beat",),
        )
        for index in range(count)
    )


def _zero_boundary_diagnostics(
    *,
    span_count: int,
    boundary_index: int,
) -> dict:
    diagnostics = _diagnostics(span_count=span_count)
    connectivity = diagnostics["hard_boundary_connectivity"]
    connectivity["zero_local_nonsevere_transition_boundary_indices"] = [
        boundary_index
    ]
    connectivity["boundaries"][boundary_index][
        "nonsevere_cross_source_edge_count"
    ] = 0
    return diagnostics


def _recorded_shift(
    profile: _DirectFullFramePartitionProfile,
    *,
    boundary_index: int,
) -> _DirectFullFrameBoundaryShift:
    return _DirectFullFrameBoundaryShift(
        boundary_index=boundary_index,
        shift_units=1,
        left_unit_count=profile.unit_counts[boundary_index],
        right_unit_count=profile.unit_counts[boundary_index + 1],
        canonical_master_seconds=(
            profile.spans[boundary_index].canonical_master_end
        ),
        boundary_reasons=("beat",),
        trigger="zero_local_nonsevere_transition_edges",
        parent_profile_id="prior",
        parent_partition_signature_sha256="e" * 64,
        proof_failure="validity_quota_path_exhausted",
        proof_failed_segment_index=1,
        proof_forward_reachable_left_source_indices=(0,),
        proof_left_valid_source_indices=(0, 1),
        proof_right_valid_source_indices=(0, 1),
        proof_local_nonsevere_edge_count=0,
    )


class DirectMemberCounterfactualPartitionTests(unittest.TestCase):
    def test_terminal_audition_cannot_truncate_primary_profile_frontier(
        self,
    ) -> None:
        self.assertGreaterEqual(
            _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT,
            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
        )
        self.assertGreaterEqual(
            _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT,
            _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT,
        )

    def test_adjacent_proven_cut_can_move_after_neighbor_shift(self) -> None:
        segments = _professional_timing_segments(32)
        unit_counts = (4, 3, 3, 2, 4, 4, 4, 4, 4)
        bare_parent = _DirectFullFramePartitionProfile(
            profile_id="neighbor-shifted",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(
                segments,
                unit_counts,
            ),
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id=bare_parent.profile_id,
            unit_counts=bare_parent.unit_counts,
            spans=bare_parent.spans,
            boundary_shifts=(
                _recorded_shift(bare_parent, boundary_index=1),
            ),
            latest_operation_kind="boundary_shift",
        )
        children = _direct_full_frame_proof_directed_child_profiles(
            segments,
            parent,
            _zero_boundary_diagnostics(
                span_count=len(unit_counts),
                boundary_index=0,
            ),
        )
        self.assertEqual(
            {child.unit_counts for child in children},
            {
                (5, 2, 3, 2, 4, 4, 4, 4, 4),
                (3, 4, 3, 2, 4, 4, 4, 4, 4),
            },
        )
        self.assertTrue(
            all(len(child.boundary_shifts) == 2 for child in children)
        )

    def test_merge_after_shift_preserves_physical_cut_coordinates(self) -> None:
        segments = _professional_timing_segments(34)
        unit_counts = (4, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2)
        bare_parent = _DirectFullFramePartitionProfile(
            profile_id="shift-before-immovable-cut",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(
                segments,
                unit_counts,
            ),
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id=bare_parent.profile_id,
            unit_counts=bare_parent.unit_counts,
            spans=bare_parent.spans,
            boundary_shifts=(
                _recorded_shift(bare_parent, boundary_index=10),
            ),
            latest_operation_kind="boundary_shift",
        )
        children = _direct_full_frame_proof_directed_merge_profiles(
            segments,
            parent,
            _zero_boundary_diagnostics(
                span_count=len(unit_counts),
                boundary_index=2,
            ),
        )
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual(
            child.unit_counts,
            (4, 4, 4, 4, 2, 2, 4, 2, 2, 4, 2),
        )
        self.assertEqual(child.boundary_shifts, parent.boundary_shifts)
        self.assertEqual(len(child.boundary_merges), 1)
        self.assertEqual(
            _direct_partition_current_shifted_boundary_indices(child),
            (9,),
        )
        self.assertGreaterEqual(
            _DIRECT_MEMBER_MAX_PROOF_DIRECTED_BOUNDARY_MERGE_COUNT,
            3,
        )
        second_children = _direct_full_frame_proof_directed_merge_profiles(
            segments,
            child,
            _zero_boundary_diagnostics(
                span_count=len(child.unit_counts),
                boundary_index=4,
            ),
        )
        self.assertEqual(len(second_children), 1)
        second_child = second_children[0]
        self.assertEqual(
            second_child.unit_counts,
            (4, 4, 4, 4, 4, 4, 2, 2, 4, 2),
        )
        self.assertEqual(len(second_child.boundary_merges), 2)
        self.assertEqual(
            _direct_partition_current_shifted_boundary_indices(second_child),
            (8,),
        )
        third_children = _direct_full_frame_proof_directed_merge_profiles(
            segments,
            second_child,
            _zero_boundary_diagnostics(
                span_count=len(second_child.unit_counts),
                boundary_index=6,
            ),
        )
        self.assertEqual(len(third_children), 1)
        third_child = third_children[0]
        self.assertEqual(
            third_child.unit_counts,
            (4, 4, 4, 4, 4, 4, 4, 4, 2),
        )
        self.assertEqual(len(third_child.boundary_merges), 3)
        self.assertEqual(
            _direct_partition_current_shifted_boundary_indices(third_child),
            (7,),
        )

        shift_saturated_second_child = _DirectFullFramePartitionProfile(
            profile_id=second_child.profile_id,
            unit_counts=second_child.unit_counts,
            spans=second_child.spans,
            boundary_shifts=tuple(
                _recorded_shift(second_child, boundary_index=boundary_index)
                for boundary_index in (0, 1, 2, 3, 4, 5, 8)
            ),
            visual_frontier_splits=second_child.visual_frontier_splits,
            boundary_merges=second_child.boundary_merges,
            latest_operation_kind="boundary_shift",
        )
        self.assertEqual(
            len(shift_saturated_second_child.boundary_shifts),
            _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT,
        )
        repair_children = _direct_full_frame_proof_directed_repair_profiles(
            segments,
            shift_saturated_second_child,
            _zero_boundary_diagnostics(
                span_count=len(shift_saturated_second_child.unit_counts),
                boundary_index=6,
            ),
            config=DirectMemberVisualConfig(),
        )
        self.assertEqual(len(repair_children), 1)
        self.assertEqual(
            repair_children[0].unit_counts,
            third_child.unit_counts,
        )
        self.assertEqual(
            repair_children[0].latest_operation_kind,
            "boundary_merge",
        )
        self.assertEqual(len(repair_children[0].boundary_merges), 3)

    def test_failure_result_does_not_mislabel_capacity_as_hard(self) -> None:
        self.assertEqual(
            _direct_partition_failure_result(
                {
                    "failure": "search_capacity_exhausted",
                    "constraint_infeasibility_proven": False,
                }
            ),
            "capacity_exhausted",
        )
        self.assertEqual(
            _direct_partition_failure_result(
                {
                    "failure": "validity_quota_path_exhausted",
                    "constraint_infeasibility_proven": True,
                }
            ),
            "hard_infeasible",
        )
        self.assertEqual(
            _direct_partition_failure_result(
                {"failure": "unknown", "constraint_infeasibility_proven": False}
            ),
            "unproven_failure",
        )
        failure, result, counts = _direct_partition_aggregate_outcome(
            ("hard_infeasible", "capacity_exhausted"),
            profile_cap_reached=False,
        )
        self.assertEqual(failure, "direct_partition_selector_capacity_exhausted")
        self.assertEqual(result, "selector_capacity_exhausted")
        self.assertEqual(counts["capacity_exhausted"], 1)
        failure, result, _counts = _direct_partition_aggregate_outcome(
            ("hard_infeasible",),
            profile_cap_reached=False,
        )
        self.assertEqual(failure, "direct_partition_profiles_exhausted")
        self.assertEqual(result, "bounded_hard_infeasible")

    def test_secondary_signature_promotes_an_unevaluated_primary_queue_item(
        self,
    ) -> None:
        signature = (1, 3, 1, 3, 4)
        self.assertEqual(
            _direct_counterfactual_secondary_signature_action(
                signature,
                evaluated_signatures={(4, 4, 4)},
                queued_signatures={signature},
                reserved_signatures=set(),
                seen_signatures={(4, 4, 4), signature},
            ),
            "promote_queued",
        )
        self.assertEqual(
            _direct_counterfactual_secondary_signature_action(
                signature,
                evaluated_signatures={signature},
                queued_signatures={signature},
                reserved_signatures=set(),
                seen_signatures={signature},
            ),
            "duplicate",
        )

    def test_retry_authority_is_exact_and_fail_closed(self) -> None:
        diagnostics = _diagnostics()
        self.assertTrue(
            _direct_severe_counterfactual_retry_is_authorized(
                diagnostics,
                span_count=4,
            )
        )
        for path, value in (
            (("render_authorized",), True),
            (("duration_state_quantization_applied",), True),
            (("candidate_validity_semantics_changed",), True),
            (("severe_disabled_witness_sha256",), None),
            (("optimality_proven",), False),
            (("minimum_severe_transition_count",), 2),
            (("production_single_split_proposal_authorized",), False),
        ):
            changed = deepcopy(diagnostics)
            counterfactual = changed["search"][
                "severe_transition_counterfactual"
            ]
            counterfactual[path[0]] = value
            self.assertFalse(
                _direct_severe_counterfactual_retry_is_authorized(
                    changed,
                    span_count=4,
                )
            )

    def test_counterfactual_boundary_is_bound_to_live_connectivity(self) -> None:
        proof = _direct_counterfactual_boundary_proof(
            _diagnostics(),
            boundary_index=1,
            span_count=4,
        )
        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.boundary_index, 1)
        self.assertEqual(proof.forward_reachable_left_source_indices, (0,))
        self.assertEqual(proof.right_valid_source_indices, (0, 1))
        self.assertEqual(
            proof.trigger,
            "diagnostic_severe_transition_counterfactual_safe_bracket",
        )
        stale = _diagnostics()
        stale["hard_boundary_connectivity"]["boundaries"][1][
            "forward_reachable_left_source_indices"
        ] = [1]
        self.assertIsNone(
            _direct_counterfactual_boundary_proof(
                stale,
                boundary_index=1,
                span_count=4,
            )
        )

    def test_secondary_split_is_proposal_only_and_uses_existing_unit(self) -> None:
        diagnostics = _diagnostics()
        diagnostics["span_content_validity"] = [
            {
                "segment_index": 2,
                "canonical_master_start": 5.0,
                "canonical_master_end": 10.0,
                "sources": [
                    {
                        "eligible_source_index": 1,
                        "valid": False,
                        "native_scene_cut_master_seconds": [],
                        "quality_sample_scores": [0.95, 0.95],
                        "samples": [
                            {
                                "canonical_master_seconds": 6.10,
                                "descriptor_available": True,
                                "attention_basis": "human_pose",
                                "face_count": 1,
                                "detected_person_count": 1,
                                "formation_person_count": 1,
                                "silhouette_confidence": 0.9,
                                "foreground_obstruction_score": 0.1,
                            },
                            {
                                "canonical_master_seconds": 6.40,
                                "descriptor_available": False,
                            },
                        ],
                    }
                ],
            }
        ]
        parent = _parent_profile()
        children = _direct_full_frame_visual_frontier_split_child_profiles(
            _timing_segments(),
            parent,
            diagnostics,
            config=DirectMemberVisualConfig(),
            allow_counterfactual_secondary_split=True,
        )
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual(child.unit_counts, (1, 3, 1, 3, 4))
        self.assertEqual(len(child.visual_frontier_splits), 2)
        split = child.visual_frontier_splits[-1]
        self.assertEqual(split.canonical_master_seconds, 6.25)
        self.assertEqual(
            split.trigger,
            "diagnostic_severe_counterfactual_visual_micro_split",
        )
        self.assertEqual(
            split.evidence_contract,
            "diagnostic_severe_counterfactual_bound_existing_timing_boundary_v1",
        )
        self.assertEqual(
            split.parent_partition_signature_sha256,
            _direct_partition_signature_sha256(parent.unit_counts),
        )
        self.assertTrue(
            split.to_dict()["proposal_evidence"][
                "proposal_only_full_selector_recheck_required"
            ]
        )
        malformed_samples = deepcopy(diagnostics)
        malformed_samples["span_content_validity"][0]["sources"][0][
            "samples"
        ][0]["canonical_master_seconds"] = 10.5
        self.assertEqual(
            _direct_full_frame_visual_frontier_split_child_profiles(
                _timing_segments(),
                parent,
                malformed_samples,
                config=DirectMemberVisualConfig(),
                allow_counterfactual_secondary_split=True,
            ),
            (),
        )

    def test_compact_and_hard_cap_fallback_retain_minimum_proof(self) -> None:
        diagnostics = _diagnostics()
        compact = _compact_direct_severe_counterfactual(diagnostics)
        self.assertIsNotNone(compact)
        assert compact is not None
        self.assertEqual(compact["minimum_severe_transition_count"], 1)
        self.assertTrue(compact["optimality_proven"])
        self.assertEqual(len(compact["minimum_witness_path_sha256"]), 64)
        self.assertIn("full_counterfactual_content_fingerprint", compact)

        partition_failure = _compact_direct_partition_failure(
            DirectMemberVisualSelectionError("hard", diagnostics),
            span_count=4,
        )
        self.assertEqual(
            partition_failure["severe_transition_counterfactual"][
                "minimum_severe_transition_count"
            ],
            1,
        )

        secondary = {
            "contract": (
                "bounded_diagnostic_severe_counterfactual_secondary_partition_v1"
            ),
            "primary_evaluated_profile_count": 3851,
            "actual_evaluation_upper_bound": 3861,
            "authorization_proofs": [
                {
                    "parent_profile_id": "parent-before-latest",
                    "parent_partition_signature_sha256": "7" * 64,
                    "child_profile_id": "reserved-child",
                    "child_partition_signature_sha256": "8" * 64,
                    "authorization": compact,
                }
            ],
        }
        fallback = _direct_member_failure_report_fallback_payload(
            {
                "stage": "test",
                "diagnostics": {
                    **diagnostics,
                    "partition_search": {
                        "absolute_profile_upper_bound": 4173,
                        "severe_transition_counterfactual_secondary_partition": (
                            secondary
                        ),
                        "profiles_attempted": [{"status": "hard_infeasible"}],
                    },
                },
            },
            oversized_report_fingerprint={
                "sha256": "9" * 64,
                "serialized_size_bytes": 70_000_000,
            },
        )
        compact_diagnostics = fallback["diagnostics"]
        self.assertEqual(
            compact_diagnostics["partition_search"][
                "severe_transition_counterfactual_secondary_partition"
            ],
            secondary,
        )
        self.assertEqual(
            compact_diagnostics["partition_search"]
            ["severe_transition_counterfactual_secondary_partition"]
            ["authorization_proofs"][0]["parent_profile_id"],
            "parent-before-latest",
        )
        self.assertEqual(
            compact_diagnostics["latest_severe_transition_counterfactual"][
                "minimum_severe_transition_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
