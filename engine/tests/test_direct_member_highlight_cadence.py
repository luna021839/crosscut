from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    DirectMemberVisualSelectionError,
)
from crosscut.draft import DraftSegment
from crosscut.pipeline import (
    _DirectFullFramePartitionProfile,
    _coarsen_direct_full_frame_spans,
    _densify_direct_full_frame_highlights,
    _direct_full_frame_proof_directed_child_profiles,
    _direct_full_frame_proof_directed_merge_profiles,
    _direct_full_frame_proof_directed_repair_profiles,
    _direct_full_frame_spans_from_partition,
    _direct_full_frame_visual_frontier_split_child_profiles,
    _direct_member_highlight_partition_capacity_payload,
    _direct_member_highlight_cadence_payload,
    _direct_member_highlight_hold_repair_boundary_indices,
    _direct_member_in_run_preferred_partition_profile,
    _direct_member_maximum_hold_count_for_global_cadence,
    _direct_member_render_feedback_seed_profiles,
    _direct_member_render_feedback_strict_composition_limit,
    _direct_member_short_proof_capacity_expansion_profiles,
    _direct_strict_extra_quota_assignment_candidates,
    _direct_strict_extra_quota_assignment_search_is_authorized,
    _direct_partition_queue_priority,
    _direct_partition_structural_search_cost,
    _enforce_direct_member_highlight_cadence_selection_gate,
    _materialize_direct_member_local_cut_times,
)


def _timing_segments(count: int) -> tuple[DraftSegment, ...]:
    return tuple(
        DraftSegment(
            segment_index=index,
            source_index=index % 8,
            canonical_master_start=float(index),
            canonical_master_end=float(index + 1),
            output_start_seconds=float(index),
            source_local_start=float(index),
            source_local_end=float(index + 1),
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=2.0,
            start_boundary_reasons=("strong_beat",),
            end_boundary_reasons=("strong_beat",),
        )
        for index in range(count)
    )


class DirectMemberHighlightCadenceTests(unittest.TestCase):
    def test_short_highlight_proof_uses_last_professional_cadence_slot(
        self,
    ) -> None:
        unit_seconds = 0.952465329374325
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(28)
        )
        baseline = _coarsen_direct_full_frame_spans(timing)

        dense = _densify_direct_full_frame_highlights(
            timing,
            baseline,
            ((timing[0].canonical_master_start, timing[-1].canonical_master_end),),
            minimum_span_count=10,
        )

        duration = timing[-1].canonical_master_end
        self.assertEqual(len(dense), 10)
        self.assertLessEqual(
            (len(dense) - 1) * 60.0 / duration,
            21.0,
        )
        self.assertGreaterEqual(
            min(span.duration_seconds for span in dense),
            1.8,
        )
        ordinary = _densify_direct_full_frame_highlights(
            timing,
            baseline,
            ((timing[0].canonical_master_start, timing[-1].canonical_master_end),),
        )
        self.assertEqual(len(ordinary), 9)

    def test_short_proof_expands_cached_sparse_cadences_at_other_beats(
        self,
    ) -> None:
        unit_seconds = 0.952465329374325
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(28)
        )
        ordinary_counts = (4, 2, 2, 2, 2, 4, 4, 4, 4)
        cached_counts = (4, 2, 2, 4, 2, 2, 4, 4, 4)
        ordinary = _DirectFullFramePartitionProfile(
            profile_id="ordinary",
            unit_counts=ordinary_counts,
            spans=_direct_full_frame_spans_from_partition(
                timing,
                ordinary_counts,
            ),
        )
        cached = _DirectFullFramePartitionProfile(
            profile_id="cached",
            unit_counts=cached_counts,
            spans=_direct_full_frame_spans_from_partition(
                timing,
                cached_counts,
            ),
        )
        failed_dense_counts = (4, 2, 2, 2, 2, 2, 2, 4, 4, 4)

        profiles = _direct_member_short_proof_capacity_expansion_profiles(
            timing,
            (ordinary, cached),
            minimum_span_count=10,
            highlight_intervals=((4 * unit_seconds, 16 * unit_seconds),),
            excluded_unit_counts=(failed_dense_counts,),
        )

        self.assertEqual(len(profiles), 8)
        self.assertNotIn(
            failed_dense_counts,
            {profile.unit_counts for profile in profiles},
        )
        self.assertIn(
            (2, 2, 2, 2, 4, 2, 2, 4, 4, 4),
            {profile.unit_counts for profile in profiles},
        )
        self.assertTrue(all(len(profile.spans) == 10 for profile in profiles))
        self.assertTrue(
            all(
                min(span.duration_seconds for span in profile.spans) >= 1.8
                for profile in profiles
            )
        )

    def test_exact_quota_failure_searches_every_balanced_remainder_owner(
        self,
    ) -> None:
        diagnostics = {
            "failure": "validity_quota_path_exhausted",
            "constraint_infeasibility_proven": True,
            "valid_count_by_source": [
                {
                    "eligible_source_index": source_index,
                    "valid_span_count": 9 - int(source_index == 6),
                }
                for source_index in range(8)
            ],
            "search": {
                "exact_lifeline": {
                    "invoked": True,
                    "state_capacity_exhausted": False,
                    "duration_state_quantization_applied": False,
                }
            },
        }

        self.assertTrue(
            _direct_strict_extra_quota_assignment_search_is_authorized(
                represented_source_count=8,
                source_count=8,
                qualified_profile_count=0,
                diagnostics=diagnostics,
            )
        )
        self.assertFalse(
            _direct_strict_extra_quota_assignment_search_is_authorized(
                represented_source_count=8,
                source_count=8,
                qualified_profile_count=1,
                diagnostics=diagnostics,
                duplicate_certificate_saturated=True,
            )
        )
        assignments = _direct_strict_extra_quota_assignment_candidates(
            span_count=9,
            source_count=8,
            source_order=tuple(range(8)),
            diagnostics=diagnostics,
        )
        self.assertEqual(len(assignments), 7)
        self.assertNotIn((0,), assignments)
        self.assertEqual(set(assignments), {(index,) for index in range(1, 8)})
        ten_span_assignments = (
            _direct_strict_extra_quota_assignment_candidates(
                span_count=10,
                source_count=8,
                source_order=tuple(range(8)),
                diagnostics=diagnostics,
            )
        )
        self.assertEqual(len(ten_span_assignments), 27)
        self.assertNotIn((0, 1), ten_span_assignments)

        diagnostics["search"]["exact_lifeline"][
            "state_capacity_exhausted"
        ] = True
        self.assertFalse(
            _direct_strict_extra_quota_assignment_search_is_authorized(
                represented_source_count=8,
                source_count=8,
                qualified_profile_count=0,
                diagnostics=diagnostics,
            )
        )

    def test_restarted_render_feedback_keeps_base_partition_as_repair_root(
        self,
    ) -> None:
        timing = _timing_segments(12)
        base = _DirectFullFramePartitionProfile(
            profile_id="base",
            unit_counts=(3, 3, 3, 3),
            spans=_direct_full_frame_spans_from_partition(
                timing,
                (3, 3, 3, 3),
            ),
        )
        duplicate_base = _DirectFullFramePartitionProfile(
            profile_id="duplicate-base",
            unit_counts=base.unit_counts,
            spans=base.spans,
        )
        persisted = _DirectFullFramePartitionProfile(
            profile_id="persisted-frontier",
            unit_counts=(2, 4, 3, 3),
            spans=_direct_full_frame_spans_from_partition(
                timing,
                (2, 4, 3, 3),
            ),
        )

        seeds = _direct_member_render_feedback_seed_profiles(
            base,
            None,
            (persisted,),
        )

        self.assertEqual(
            tuple(profile.profile_id for profile in seeds),
            ("persisted-frontier", "base"),
        )
        in_run_seeds = _direct_member_render_feedback_seed_profiles(
            base,
            duplicate_base,
            (persisted,),
        )
        self.assertEqual(
            tuple(profile.profile_id for profile in in_run_seeds),
            ("duplicate-base", "persisted-frontier"),
        )

    def test_render_feedback_extends_instead_of_shrinking_strict_cohort(
        self,
    ) -> None:
        self.assertEqual(
            _direct_member_render_feedback_strict_composition_limit(0),
            24,
        )
        self.assertEqual(
            _direct_member_render_feedback_strict_composition_limit(6),
            64,
        )
        self.assertEqual(
            _direct_member_render_feedback_strict_composition_limit(100),
            64,
        )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            _direct_member_render_feedback_strict_composition_limit(-1)

    def test_in_run_feedback_retry_rebuilds_exact_partition_before_repartitioning(
        self,
    ) -> None:
        timing = _timing_segments(12)

        profile = _direct_member_in_run_preferred_partition_profile(
            timing,
            (3, 3, 3, 3),
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.unit_counts, (3, 3, 3, 3))
        self.assertEqual(len(profile.spans), 4)
        self.assertEqual(profile.spans[0].canonical_master_start, 0.0)
        self.assertEqual(profile.spans[-1].canonical_master_end, 12.0)
        self.assertTrue(profile.profile_id.startswith("in-run-path-retry-"))
        self.assertEqual(profile.structural_change_count, 0)

        with self.assertRaisesRegex(ValueError, "does not cover"):
            _direct_member_in_run_preferred_partition_profile(
                timing,
                (3, 3, 3),
            )

    def test_global_cadence_hold_budget_rejects_sparse_eight_span_path(
        self,
    ) -> None:
        duration = 26.6690292224811

        self.assertEqual(
            _direct_member_maximum_hold_count_for_global_cadence(
                boundary_count=8,
                total_duration_seconds=duration,
            ),
            1,
        )
        self.assertEqual(
            _direct_member_maximum_hold_count_for_global_cadence(
                boundary_count=7,
                total_duration_seconds=duration,
            ),
            0,
        )

    def test_visual_micro_split_rejects_sub_1_8_second_flash_shots(self) -> None:
        unit_seconds = 0.952465329374325
        unit_counts = (4, 2, 2, 2, 2, 4, 4, 2, 4)
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(sum(unit_counts))
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id="short-flash-split-parent",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(timing, unit_counts),
        )
        boundaries = []
        for index in range(len(unit_counts) - 1):
            boundaries.append(
                {
                    "boundary_index": index,
                    "left_valid_source_indices": [0, 1],
                    "right_valid_source_indices": [0, 1],
                    "forward_reachable_left_source_indices": (
                        [0] if index <= 6 else []
                    ),
                    "forward_reachable_right_source_indices": (
                        [] if index == 6 else [0]
                    ),
                    "nonsevere_cross_source_edge_count": 1,
                }
            )
        incoming = parent.spans[7]
        safe_samples = [
            {
                "canonical_master_seconds": incoming.canonical_master_start,
                "descriptor_available": True,
                "attention_basis": "aggregate_human_silhouette",
                "face_count": 1,
                "detected_person_count": 1,
                "formation_person_count": 1,
                "silhouette_confidence": 0.95,
                "foreground_obstruction_score": 0.0,
            },
            {
                "canonical_master_seconds": incoming.canonical_master_end,
                "descriptor_available": True,
                "attention_basis": "aggregate_human_silhouette",
                "face_count": 1,
                "detected_person_count": 1,
                "formation_person_count": 1,
                "silhouette_confidence": 0.95,
                "foreground_obstruction_score": 0.0,
            },
        ]
        diagnostics = {
            "failure": "boundary_connectivity_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 7,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
            "span_content_validity": [
                {
                    "segment_index": 7,
                    "canonical_master_start": incoming.canonical_master_start,
                    "canonical_master_end": incoming.canonical_master_end,
                    "sources": [
                        {
                            "eligible_source_index": 1,
                            "valid": True,
                            "native_scene_cut_master_seconds": [],
                            "samples": safe_samples,
                            "quality_sample_scores": [1.0, 1.0],
                            "portrait_fancam_lighting_quality_rescue_applied": False,
                        }
                    ],
                }
            ],
        }

        children = _direct_full_frame_visual_frontier_split_child_profiles(
            timing,
            parent,
            diagnostics,
            config=DirectMemberVisualConfig(),
        )

        self.assertEqual(children, ())

    def test_short_highlight_proof_can_move_a_safe_musical_boundary(self) -> None:
        unit_seconds = 0.952465329374325
        unit_counts = (4, 2, 2, 2, 2, 4, 4, 4, 4)
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(sum(unit_counts))
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id="short-highlight-proof",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(timing, unit_counts),
        )
        boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": list(range(8)),
                "right_valid_source_indices": list(range(8)),
                "forward_reachable_left_source_indices": list(range(8)),
                "nonsevere_cross_source_edge_count": 0 if index == 6 else 1,
            }
            for index in range(len(unit_counts) - 1)
        ]
        diagnostics = {
            "failure": "boundary_connectivity_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 3,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [6],
            },
        }

        children = _direct_full_frame_proof_directed_child_profiles(
            timing,
            parent,
            diagnostics,
        )

        self.assertEqual(
            {child.unit_counts[6:8] for child in children},
            {(5, 3), (3, 5)},
        )
        self.assertTrue(
            all(
                min(
                    child.spans[6].duration_seconds,
                    child.spans[7].duration_seconds,
                )
                >= 1.8
                for child in children
            )
        )

    def test_post_alignment_highlight_failure_moves_only_forced_hold_cut(
        self,
    ) -> None:
        unit_seconds = 0.952465329374325
        unit_counts = (5, 1, 4, 4, 2, 5, 2, 3, 2)
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(sum(unit_counts))
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id="post-alignment-parent",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(timing, unit_counts),
        )
        diagnostics = {
            "failure": "cinematic_bridge_path_infeasible",
            "constraint_infeasibility_proven": True,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": [
                    {
                        "boundary_index": index,
                        "left_valid_source_indices": list(range(8)),
                        "right_valid_source_indices": list(range(8)),
                        "forward_reachable_left_source_indices": list(range(8)),
                        "nonsevere_cross_source_edge_count": 1,
                    }
                    for index in range(len(unit_counts) - 1)
                ],
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
            "highlight_hold_repair": {
                "contract": (
                    "post_alignment_density_constraint_generation_v1"
                ),
                "attempted": True,
                "attempt_count": 1,
                "forced_cut_boundary_indices": [1],
                "result": "repair_selection_infeasible",
                "visual_gates_relaxed": False,
            },
        }

        children = _direct_full_frame_proof_directed_child_profiles(
            timing,
            parent,
            diagnostics,
        )

        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual(child.unit_counts[1:3], (2, 3))
        self.assertEqual(child.boundary_shifts[-1].boundary_index, 1)
        self.assertEqual(
            child.boundary_shifts[-1].trigger,
            "post_alignment_highlight_forced_cut_infeasible",
        )
        self.assertEqual(
            child.boundary_shifts[-1].proof_failed_segment_index,
            2,
        )

    def test_proven_unmovable_minimum_boundary_can_merge_without_flash_cut(
        self,
    ) -> None:
        unit_counts = (
            4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2,
            4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4,
            2, 2, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6,
        )
        unit_seconds = 0.952465329374325
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=1.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(sum(unit_counts))
        )
        parent = _DirectFullFramePartitionProfile(
            profile_id="real-minimum-grid-parent",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(timing, unit_counts),
        )
        boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": (
                    [0] if index <= 7 else []
                ),
                "nonsevere_cross_source_edge_count": 1,
            }
            for index in range(len(unit_counts) - 1)
        ]
        diagnostics = {
            "failure": "validity_quota_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 8,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
        }

        shifted = _direct_full_frame_proof_directed_child_profiles(
            timing,
            parent,
            diagnostics,
        )
        merged = _direct_full_frame_proof_directed_merge_profiles(
            timing,
            parent,
            diagnostics,
        )

        self.assertEqual(shifted, ())
        self.assertEqual(len(merged), 1)
        child = merged[0]
        self.assertEqual(len(child.spans), len(parent.spans) - 1)
        self.assertEqual(child.unit_counts[6:9], (2, 4, 2))
        self.assertAlmostEqual(child.spans[7].duration_seconds, 4 * unit_seconds)
        self.assertEqual(child.boundary_merges[0].boundary_index, 7)
        self.assertFalse(child.boundary_shifts)
        self.assertEqual(child.latest_operation_kind, "boundary_merge")

        merged_boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": (
                    [0] if index <= 7 else []
                ),
                "nonsevere_cross_source_edge_count": 1,
            }
            for index in range(len(child.spans) - 1)
        ]
        merged_diagnostics = {
            **diagnostics,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": merged_boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
        }
        shifted_after_merge = _direct_full_frame_proof_directed_child_profiles(
            timing,
            child,
            merged_diagnostics,
        )

        self.assertEqual(
            {profile.unit_counts[7:9] for profile in shifted_after_merge},
            {(3, 3), (2, 4)},
        )
        self.assertTrue(
            all(
                profile.boundary_merges == child.boundary_merges
                for profile in shifted_after_merge
            )
        )
        self.assertTrue(
            all(
                profile.boundary_shifts[-1].parent_profile_id
                == "boundary-7-merge-4"
                for profile in shifted_after_merge
            )
        )
        self.assertTrue(
            all(
                profile.latest_operation_kind == "boundary_shift"
                and profile.parent_profile_id == child.profile_id
                for profile in shifted_after_merge
            )
        )

        shifted_dead_end = next(
            profile
            for profile in shifted_after_merge
            if profile.unit_counts[7:9] == (2, 4)
        )
        incoming = shifted_dead_end.spans[8]
        safe_samples = [
            {
                "canonical_master_seconds": incoming.canonical_master_start
                + 0.05,
                "descriptor_available": True,
                "attention_basis": "aggregate_human_silhouette",
                "face_count": 1,
                "detected_person_count": 1,
                "formation_person_count": 1,
                "silhouette_confidence": 0.95,
                "foreground_obstruction_score": 0.0,
            },
            {
                "canonical_master_seconds": incoming.canonical_master_end
                - 0.05,
                "descriptor_available": True,
                "attention_basis": "aggregate_human_silhouette",
                "face_count": 1,
                "detected_person_count": 1,
                "formation_person_count": 1,
                "silhouette_confidence": 0.95,
                "foreground_obstruction_score": 0.0,
            },
        ]
        shifted_dead_end_diagnostics = {
            **merged_diagnostics,
            "span_content_validity": [
                {
                    "segment_index": 8,
                    "canonical_master_start": incoming.canonical_master_start,
                    "canonical_master_end": incoming.canonical_master_end,
                    "sources": [
                        {
                            "eligible_source_index": 1,
                            "valid": True,
                            "native_scene_cut_master_seconds": [],
                            "samples": safe_samples,
                            "quality_sample_scores": [1.0, 1.0],
                            "portrait_fancam_lighting_quality_rescue_applied": (
                                False
                            ),
                        }
                    ],
                }
            ],
        }

        repaired = _direct_full_frame_proof_directed_repair_profiles(
            timing,
            shifted_dead_end,
            shifted_dead_end_diagnostics,
            config=DirectMemberVisualConfig(),
        )

        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].unit_counts[7:10], (2, 2, 2))
        self.assertEqual(len(repaired[0].visual_frontier_splits), 1)
        self.assertEqual(
            repaired[0].latest_operation_kind,
            "visual_frontier_split",
        )
        self.assertEqual(
            repaired[0].visual_frontier_splits[0].parent_profile_id,
            shifted_dead_end.profile_id,
        )
        self.assertGreaterEqual(
            min(
                repaired[0].spans[8].duration_seconds,
                repaired[0].spans[9].duration_seconds,
            ),
            1.8,
        )

        advanced_boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": (
                    [0] if index <= 18 else []
                ),
                "nonsevere_cross_source_edge_count": 1,
            }
            for index in range(len(repaired[0].spans) - 1)
        ]
        advanced_diagnostics = {
            **diagnostics,
            "failed_at_segment_index": 19,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": advanced_boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
        }

        advanced = _direct_full_frame_proof_directed_child_profiles(
            timing,
            repaired[0],
            advanced_diagnostics,
        )

        self.assertTrue(advanced)
        self.assertTrue(
            all(
                profile.boundary_shifts[-1].boundary_index == 18
                for profile in advanced
            )
        )
        self.assertTrue(
            all(
                profile.visual_frontier_splits
                == repaired[0].visual_frontier_splits
                for profile in advanced
            )
        )
        self.assertTrue(
            all(
                profile.boundary_shifts[-1].parent_profile_id
                == repaired[0].profile_id
                for profile in advanced
            )
        )
        self.assertTrue(
            all(
                profile.parent_profile_id == repaired[0].profile_id
                for profile in advanced
            )
        )
        self.assertTrue(
            all(
                profile.latest_operation_kind == "boundary_shift"
                for profile in advanced
            )
        )
        self.assertLess(
            _direct_partition_queue_priority(advanced[0]),
            _direct_partition_queue_priority(repaired[0]),
        )
        self.assertGreater(
            _direct_partition_structural_search_cost(advanced[0]),
            _direct_partition_structural_search_cost(repaired[0]),
        )

    def test_proven_shift_uses_real_seconds_not_a_two_unit_proxy(self) -> None:
        unit_counts = (4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4)
        durations: list[float] = []
        for span_index, count in enumerate(unit_counts):
            if span_index == 4:
                durations.extend((1.9, 1.9))
            elif span_index == 5:
                durations.extend((0.475,) * 4)
            else:
                # Keep the complete fixture inside the production 15/min
                # cadence gate while retaining the 1.9s one-unit proof at the
                # shifted boundary under test.
                durations.extend((0.95,) * count)
        timing: list[DraftSegment] = []
        cursor = 0.0
        for index, duration in enumerate(durations):
            timing.append(
                DraftSegment(
                    segment_index=index,
                    source_index=index % 8,
                    canonical_master_start=cursor,
                    canonical_master_end=cursor + duration,
                    output_start_seconds=cursor,
                    source_local_start=cursor,
                    source_local_end=cursor + duration,
                    audio_offset_seconds=0.0,
                    sync_score=1.0,
                    estimated_beat_count=1.0,
                    start_boundary_reasons=("strong_beat",),
                    end_boundary_reasons=("strong_beat",),
                )
            )
            cursor += duration
        parent = _DirectFullFramePartitionProfile(
            profile_id="two-plus-four-parent",
            unit_counts=unit_counts,
            spans=_direct_full_frame_spans_from_partition(timing, unit_counts),
        )
        boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": (
                    [0] if index <= 4 else []
                ),
                "nonsevere_cross_source_edge_count": 1,
            }
            for index in range(len(unit_counts) - 1)
        ]
        diagnostics = {
            "failure": "validity_quota_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 5,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [],
            },
        }

        children = _direct_full_frame_proof_directed_child_profiles(
            timing,
            parent,
            diagnostics,
        )

        one_plus_five = next(
            child
            for child in children
            if child.unit_counts[4:6] == (1, 5)
        )
        self.assertAlmostEqual(
            one_plus_five.spans[4].duration_seconds,
            1.9,
        )
        self.assertAlmostEqual(
            one_plus_five.spans[5].duration_seconds,
            3.8,
        )

    def test_real_time_floor_allows_a_complete_one_unit_highlight_shot(
        self,
    ) -> None:
        unit_seconds = 1.9
        timing = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 8,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=2.0,
                start_boundary_reasons=("strong_beat",),
                end_boundary_reasons=("strong_beat",),
            )
            for index in range(8)
        )
        base = _coarsen_direct_full_frame_spans(timing)

        dense = _densify_direct_full_frame_highlights(
            timing,
            base,
            ((0.0, 15.2),),
        )

        self.assertEqual(len(base), 4)
        self.assertEqual(len(dense), 5)
        self.assertIn(1, [span.timing_segment_count for span in dense])
        self.assertGreaterEqual(
            min(span.duration_seconds for span in dense),
            1.8,
        )

    def test_highlight_density_adds_only_existing_musical_boundaries(self) -> None:
        timing = _timing_segments(64)
        base = _coarsen_direct_full_frame_spans(timing)

        dense = _densify_direct_full_frame_highlights(
            timing,
            base,
            ((16.0, 48.0),),
        )

        self.assertEqual(len(base), 16)
        self.assertEqual(len(dense), 21)
        base_boundaries = {
            span.canonical_master_end for span in base[:-1]
        }
        dense_boundaries = {
            span.canonical_master_end for span in dense[:-1]
        }
        added = dense_boundaries - base_boundaries
        self.assertEqual(len(added), 5)
        self.assertTrue(all(16.0 <= value <= 48.0 for value in added))
        self.assertTrue(all(value.is_integer() for value in dense_boundaries))
        self.assertTrue(all(span.duration_seconds <= 7.2 for span in dense))

    def test_exact_selected_highlight_cuts_must_be_denser_than_verse(self) -> None:
        timing = _timing_segments(64)
        dense = _densify_direct_full_frame_highlights(
            timing,
            _coarsen_direct_full_frame_spans(timing),
            ((16.0, 48.0),),
        )
        selection = SimpleNamespace(
            source_indices=tuple(index % 8 for index in range(len(dense))),
            boundaries=tuple(
                SimpleNamespace(
                    cut_applied=True,
                    mismatch=SimpleNamespace(
                        motion_connection=SimpleNamespace(both_moving=True)
                    ),
                )
                for _ in range(len(dense) - 1)
            ),
        )

        payload = _direct_member_highlight_cadence_payload(
            spans=dense,
            visual_selection=selection,
            highlight_intervals=((16.0, 48.0),),
        )

        self.assertTrue(payload["audit_required"])
        self.assertTrue(payload["passed"])
        self.assertGreaterEqual(
            payload["highlight_source_switches_per_minute"],
            18.0,
        )
        self.assertGreaterEqual(payload["density_lift_per_minute"], 2.0)
        self.assertEqual(payload["highlight_no_cut_hold_count"], 0)

    def test_partition_capacity_rejects_too_few_possible_highlight_cuts(
        self,
    ) -> None:
        timing = _timing_segments(12)
        spans = _coarsen_direct_full_frame_spans(timing)

        payload = _direct_member_highlight_partition_capacity_payload(
            spans=spans,
            highlight_intervals=((0.0, 12.0),),
        )

        self.assertTrue(payload["audit_required"])
        self.assertFalse(payload["passed"])
        self.assertEqual(
            payload["failure"],
            "highlight_partition_cut_capacity_exhausted",
        )
        self.assertLess(
            payload["planned_highlight_boundary_count"],
            payload["minimum_required_highlight_cut_count"],
        )

    def test_selected_local_cut_shift_is_used_by_highlight_gate(self) -> None:
        timing = _timing_segments(40)
        spans = _direct_full_frame_spans_from_partition(
            timing,
            (4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
        )
        selected_boundaries = [
            span.canonical_master_end for span in spans[:-1]
        ]
        selected_boundaries[1] = 7.9
        selection = SimpleNamespace(
            source_indices=tuple(index % 8 for index in range(len(spans))),
            selected_boundary_master_seconds=tuple(selected_boundaries),
            config=DirectMemberVisualConfig(boundary_search_radius_frames=6),
            optimizer_search_diagnostics={},
            boundaries=tuple(
                SimpleNamespace(
                    cut_applied=True,
                    mismatch=SimpleNamespace(
                        motion_connection=SimpleNamespace(both_moving=True)
                    ),
                )
                for _ in range(len(spans) - 1)
            ),
        )

        canonical_payload = _direct_member_highlight_cadence_payload(
            spans=spans,
            visual_selection=selection,
            highlight_intervals=((8.0, 28.0),),
        )
        materialized = _materialize_direct_member_local_cut_times(
            spans,
            selection,
        )

        self.assertTrue(canonical_payload["passed"])
        with self.assertRaises(DirectMemberVisualSelectionError) as caught:
            _enforce_direct_member_highlight_cadence_selection_gate(
                spans=materialized,
                visual_selection=selection,
                highlight_intervals=((8.0, 28.0),),
            )
        self.assertIn(
            "highlight_switch_density_below_minimum",
            caught.exception.diagnostics["highlight_cut_density"][
                "failure_reasons"
            ],
        )

    def test_density_failure_feeds_observed_highlight_holds_back_as_cuts(
        self,
    ) -> None:
        payload = {
            "failure_reasons": [
                "highlight_switch_density_below_minimum",
                "highlight_density_lift_below_minimum",
            ],
            "highlight_no_cut_hold_boundary_indices": [3, 1, 3],
        }

        self.assertEqual(
            _direct_member_highlight_hold_repair_boundary_indices(payload),
            (1, 3),
        )

    def test_motion_only_failure_does_not_misuse_hold_density_repair(
        self,
    ) -> None:
        payload = {
            "failure_reasons": [
                "highlight_moving_cut_fraction_below_minimum"
            ],
            "highlight_no_cut_hold_boundary_indices": [2],
        }

        self.assertEqual(
            _direct_member_highlight_hold_repair_boundary_indices(payload),
            (),
        )

    def test_compensated_highlight_holds_retain_the_section_audit(self) -> None:
        timing = _timing_segments(64)
        dense = _densify_direct_full_frame_highlights(
            timing,
            _coarsen_direct_full_frame_spans(timing),
            ((16.0, 48.0),),
        )
        cut_flags = [True] * (len(dense) - 1)
        highlight_boundary_index = next(
            index
            for index, span in enumerate(dense[:-1])
            if 16.0 <= span.canonical_master_end <= 48.0
        )
        cut_flags[highlight_boundary_index] = False
        selection = SimpleNamespace(
            source_indices=tuple(index % 8 for index in range(len(dense))),
            optimizer_search_diagnostics={},
            boundaries=tuple(
                SimpleNamespace(
                    cut_applied=value,
                    mismatch=SimpleNamespace(
                        motion_connection=(
                            SimpleNamespace(both_moving=True)
                            if value
                            else None
                        )
                    ),
                )
                for value in cut_flags
            ),
        )

        payload = _direct_member_highlight_cadence_payload(
            spans=dense,
            visual_selection=selection,
            highlight_intervals=((16.0, 48.0),),
        )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["highlight_no_cut_hold_count"], 1)
        self.assertGreaterEqual(
            payload["highlight_source_switches_per_minute"],
            18.0,
        )

    def test_too_many_highlight_holds_fail_the_section_audit(self) -> None:
        timing = _timing_segments(64)
        dense = _densify_direct_full_frame_highlights(
            timing,
            _coarsen_direct_full_frame_spans(timing),
            ((16.0, 48.0),),
        )
        cut_flags = [True] * (len(dense) - 1)
        highlight_boundary_indices = [
            index
            for index, span in enumerate(dense[:-1])
            if 16.0 <= span.canonical_master_end <= 48.0
        ]
        for boundary_index in highlight_boundary_indices[:5]:
            cut_flags[boundary_index] = False
        selection = SimpleNamespace(
            source_indices=tuple(index % 8 for index in range(len(dense))),
            optimizer_search_diagnostics={},
            boundaries=tuple(
                SimpleNamespace(
                    cut_applied=value,
                    mismatch=SimpleNamespace(
                        motion_connection=(
                            SimpleNamespace(both_moving=True)
                            if value
                            else None
                        )
                    ),
                )
                for value in cut_flags
            ),
        )

        payload = _direct_member_highlight_cadence_payload(
            spans=dense,
            visual_selection=selection,
            highlight_intervals=((16.0, 48.0),),
        )

        self.assertFalse(payload["passed"])
        self.assertIn(
            "compensated_highlight_hold_budget_exceeded",
            payload["failure_reasons"],
        )
        with self.assertRaises(DirectMemberVisualSelectionError) as caught:
            _enforce_direct_member_highlight_cadence_selection_gate(
                spans=dense,
                visual_selection=selection,
                highlight_intervals=((16.0, 48.0),),
            )
        self.assertEqual(
            caught.exception.diagnostics["failure"],
            "highlight_cadence_gate_failed",
        )
        self.assertFalse(
            caught.exception.diagnostics["constraint_infeasibility_proven"]
        )
        self.assertEqual(
            selection.optimizer_search_diagnostics["highlight_cut_density"],
            payload,
        )


if __name__ == "__main__":
    unittest.main()
