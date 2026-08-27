from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.pipeline import (
    _DirectFullFramePartitionProfile,
    _direct_member_dense_render_feedback_binding,
    _direct_member_render_feedback_partition_profiles,
    _direct_partition_queue_priority,
    _direct_partition_terminal_audition_due,
    _direct_partition_terminal_hold_failure_is_eligible,
    _direct_full_frame_proof_directed_child_profiles,
    _direct_full_frame_spans_from_partition,
    _load_direct_member_dense_render_feedback_edges,
    _load_direct_member_partition_warm_start_profiles,
    _normalize_direct_member_dense_render_feedback_edges,
    _persist_direct_member_dense_render_feedback_edges,
    _persist_direct_member_partition_failure_frontier_profiles,
)
from crosscut.direct_member_visual import DirectMemberVisualConfig


class DirectMemberDenseRenderFeedbackTests(unittest.TestCase):
    @staticmethod
    def _timeline() -> GroupDraftTimeline:
        return GroupDraftTimeline(
            source_count=2,
            common_master_start=0.0,
            common_master_end=2.0,
            beat_period_seconds=0.5,
            audio_offsets=(0.0, 0.0),
            sync_scores=(0.9, 0.9),
            segments=(
                DraftSegment(
                    segment_index=0,
                    source_index=0,
                    canonical_master_start=0.0,
                    canonical_master_end=1.0,
                    output_start_seconds=0.0,
                    source_local_start=0.0,
                    source_local_end=1.0,
                    audio_offset_seconds=0.0,
                    sync_score=0.9,
                    estimated_beat_count=2.0,
                    start_boundary_reasons=("performance_start",),
                    end_boundary_reasons=("beat",),
                ),
                DraftSegment(
                    segment_index=1,
                    source_index=1,
                    canonical_master_start=1.0,
                    canonical_master_end=2.0,
                    output_start_seconds=1.0,
                    source_local_start=1.0,
                    source_local_end=2.0,
                    audio_offset_seconds=0.0,
                    sync_score=0.9,
                    estimated_beat_count=2.0,
                    start_boundary_reasons=("beat",),
                    end_boundary_reasons=("performance_end",),
                ),
            ),
        )

    @staticmethod
    def _write_acquisition_manifest(directory: Path) -> None:
        selected = []
        for source_index in range(2):
            selected.append(
                {
                    "source_index": source_index,
                    "video_id": f"video0{source_index}",
                    "broadcast_date": f"2026-01-0{source_index + 1}",
                    "broadcast_program": f"program-{source_index}",
                    "content_integrity": {
                        "algorithm": "sha256",
                        "sha256": str(source_index) * 64,
                        "size_bytes": 1_000 + source_index,
                    },
                }
            )
        (directory / "acquisition-manifest.json").write_text(
            json.dumps(
                {
                    "group": "IVE",
                    "song": "TEST SONG",
                    "edit_mode": "member",
                    "target_member": "gaeul",
                    "selected": selected,
                }
            ),
            encoding="utf-8",
        )

    def _binding(
        self,
        directory: Path,
        *,
        dissolve_enabled: bool = True,
    ) -> dict[str, object]:
        binding, diagnostics = _direct_member_dense_render_feedback_binding(
            directory,
            timing_draft=self._timeline(),
            eligible_indices=(0, 1),
            source_timewarps=None,
            scene_cut_times=((), ()),
            eligible_source_frame_sizes=((2160, 3840), (2160, 3840)),
            output_frame_size=(1080, 1920),
            output_fps=29.97,
            highlight_intervals=((0.5, 1.5),),
            member_pose_dissolve_enabled=dissolve_enabled,
        )
        self.assertIsNotNone(binding)
        self.assertEqual(diagnostics["status"], "binding_ready")
        assert binding is not None
        return binding

    def test_checkpoint_round_trip_is_bound_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_acquisition_manifest(directory)
            binding = self._binding(directory)
            edge = (1.0, 0, 1)

            persisted = _persist_direct_member_dense_render_feedback_edges(
                directory,
                binding,
                (edge, edge),
            )
            loaded, diagnostics = (
                _load_direct_member_dense_render_feedback_edges(
                    directory,
                    binding,
                )
            )

            self.assertTrue(persisted["persisted"])
            self.assertEqual(persisted["certificate_count"], 1)
            self.assertEqual(loaded, (edge,))
            self.assertEqual(diagnostics["status"], "certificates_loaded")
            self.assertFalse(diagnostics["visual_thresholds_relaxed"])

            changed_binding = self._binding(
                directory,
                dissolve_enabled=False,
            )
            self.assertNotEqual(
                changed_binding["binding_sha256"],
                binding["binding_sha256"],
            )
            changed_loaded, changed_diagnostics = (
                _load_direct_member_dense_render_feedback_edges(
                    directory,
                    changed_binding,
                )
            )
            self.assertEqual(changed_loaded, ())
            self.assertEqual(
                changed_diagnostics["status"],
                "no_matching_binding",
            )

    def test_invalid_feedback_edges_fail_closed(self) -> None:
        for edge in (
            (-0.1, 0, 1),
            (2.1, 0, 1),
            (1.0, 0, 0),
            (1.0, 0, 2),
        ):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                _normalize_direct_member_dense_render_feedback_edges(
                    (edge,),
                    source_count=2,
                    range_start=0.0,
                    range_end=2.0,
                )

    def test_failure_frontier_checkpoint_restores_exact_shift_lineage(
        self,
    ) -> None:
        unit_seconds = 0.875
        timing_segments = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 2,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=0.9,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("beat",),
            )
            for index in range(32)
        )
        counts = (4,) * 8
        base = _DirectFullFramePartitionProfile(
            profile_id="base",
            unit_counts=counts,
            spans=_direct_full_frame_spans_from_partition(
                timing_segments,
                counts,
            ),
        )
        shifted = _direct_member_render_feedback_partition_profiles(
            timing_segments,
            base,
            ((7.0, 0, 1),),
            source_count=2,
            config=DirectMemberVisualConfig(
                boundary_search_radius_frames=6,
                boundary_search_step_frames=2,
            ),
        )[0]

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_acquisition_manifest(directory)
            persisted = (
                _persist_direct_member_partition_failure_frontier_profiles(
                    directory,
                    base,
                    (shifted,),
                    source_count=2,
                )
            )
            loaded, diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    directory,
                    timing_segments,
                    base,
                    source_count=2,
                )
            )

        self.assertTrue(persisted["persisted"])
        self.assertEqual(persisted["persisted_candidate_count"], 1)
        self.assertEqual(diagnostics["status"], "candidates_loaded")
        self.assertEqual(len(loaded), 1)
        restored = loaded[0]
        self.assertTrue(
            restored.profile_id.startswith("input-bound-failure-frontier-")
        )
        self.assertEqual(restored.unit_counts, shifted.unit_counts)
        self.assertEqual(restored.boundary_shifts, shifted.boundary_shifts)
        self.assertEqual(restored.latest_operation_kind, "boundary_shift")
        self.assertEqual(restored.proof_directed_boundary_motion_count, 1)
        self.assertEqual(
            restored.boundary_shifts[0].trigger,
            "dense_render_camera_infeasible_pair_cluster",
        )
        self.assertFalse(diagnostics["visual_gates_relaxed"])

    def test_reproved_shifted_boundary_refines_monotonically(self) -> None:
        unit_seconds = 0.9
        timing_segments = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 2,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=0.9,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("beat",),
            )
            for index in range(44)
        )
        counts = (4,) * 11
        base = _DirectFullFramePartitionProfile(
            profile_id="base",
            unit_counts=counts,
            spans=_direct_full_frame_spans_from_partition(
                timing_segments,
                counts,
            ),
        )
        first_candidates = _direct_member_render_feedback_partition_profiles(
            timing_segments,
            base,
            ((7.2, 0, 1),),
            source_count=2,
            config=DirectMemberVisualConfig(
                boundary_search_radius_frames=6,
                boundary_search_step_frames=2,
            ),
        )
        first = next(
            profile
            for profile in first_candidates
            if profile.boundary_shifts[-1].shift_units == 1
        )
        boundary_index = first.boundary_shifts[-1].boundary_index
        boundaries = [
            {
                "boundary_index": index,
                "left_valid_source_indices": [0, 1],
                "right_valid_source_indices": [0, 1],
                "forward_reachable_left_source_indices": (
                    [] if index == boundary_index else [0]
                ),
                "nonsevere_cross_source_edge_count": (
                    0 if index == boundary_index else 1
                ),
            }
            for index in range(len(first.spans) - 1)
        ]
        diagnostics = {
            "failure": "boundary_connectivity_path_exhausted",
            "failed_at_segment_index": boundary_index,
            "constraint_infeasibility_proven": True,
            "hard_boundary_connectivity": {
                "analysis_basis": (
                    "content_valid_nonsevere_no_repeat_reachability_v1"
                ),
                "boundaries": boundaries,
                "zero_local_nonsevere_transition_boundary_indices": [
                    boundary_index
                ],
            },
        }

        refined = _direct_full_frame_proof_directed_child_profiles(
            timing_segments,
            first,
            diagnostics,
        )

        self.assertEqual(len(refined), 1)
        child = refined[0]
        self.assertIn("__refine-", child.profile_id)
        self.assertEqual(len(child.boundary_shifts), 2)
        self.assertEqual(
            [shift.shift_units for shift in child.boundary_shifts],
            [1, 1],
        )
        self.assertEqual(
            child.unit_counts[boundary_index : boundary_index + 2],
            (6, 2),
        )
        self.assertNotEqual(child.unit_counts, base.unit_counts)

    def test_feedback_cluster_moves_its_own_cut_before_other_boundaries(
        self,
    ) -> None:
        unit_seconds = 0.875
        timing_segments = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 3,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=0.9,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("beat",),
            )
            for index in range(32)
        )
        counts = (4,) * 8
        parent = _DirectFullFramePartitionProfile(
            profile_id="warm",
            unit_counts=counts,
            spans=_direct_full_frame_spans_from_partition(
                timing_segments,
                counts,
            ),
            warm_start_binding_sha256="a" * 64,
            warm_start_cache_file_sha256="b" * 64,
        )
        config = DirectMemberVisualConfig(
            boundary_search_radius_frames=6,
            boundary_search_step_frames=2,
        )

        profiles = _direct_member_render_feedback_partition_profiles(
            timing_segments,
            parent,
            (
                (21.05, 0, 1),
                (21.02, 0, 2),
                (21.00, 1, 2),
                (7.00, 2, 0),
            ),
            source_count=3,
            config=config,
        )

        self.assertEqual(len(profiles), 4)
        self.assertEqual(
            [profile.boundary_shifts[-1].boundary_index for profile in profiles],
            [5, 5, 1, 1],
        )
        self.assertEqual(
            [profile.unit_counts[5:7] for profile in profiles[:2]],
            [(5, 3), (3, 5)],
        )
        for profile in profiles:
            shift = profile.boundary_shifts[-1]
            self.assertEqual(
                shift.trigger,
                "dense_render_camera_infeasible_pair_cluster",
            )
            self.assertEqual(
                shift.proof_failure,
                "dense_render_camera_infeasible",
            )
            self.assertEqual(profile.warm_start_binding_sha256, "a" * 64)
            self.assertEqual(profile.warm_start_cache_file_sha256, "b" * 64)

    def test_feedback_lineage_precedes_warm_and_generic_search_branches(
        self,
    ) -> None:
        unit_seconds = 0.875
        timing_segments = tuple(
            DraftSegment(
                segment_index=index,
                source_index=index % 3,
                canonical_master_start=index * unit_seconds,
                canonical_master_end=(index + 1) * unit_seconds,
                output_start_seconds=index * unit_seconds,
                source_local_start=index * unit_seconds,
                source_local_end=(index + 1) * unit_seconds,
                audio_offset_seconds=0.0,
                sync_score=0.9,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("beat",),
            )
            for index in range(32)
        )
        counts = (4,) * 8
        spans = _direct_full_frame_spans_from_partition(
            timing_segments,
            counts,
        )
        generic = _DirectFullFramePartitionProfile(
            profile_id="generic",
            unit_counts=counts,
            spans=spans,
        )
        warm = _DirectFullFramePartitionProfile(
            profile_id="warm",
            unit_counts=counts,
            spans=spans,
            warm_start_binding_sha256="a" * 64,
            warm_start_cache_file_sha256="b" * 64,
        )
        feedback = _direct_member_render_feedback_partition_profiles(
            timing_segments,
            warm,
            ((7.0, 0, 1),),
            source_count=3,
            config=DirectMemberVisualConfig(
                boundary_search_radius_frames=6,
                boundary_search_step_frames=2,
            ),
        )[0]

        self.assertLess(
            _direct_partition_queue_priority(feedback),
            _direct_partition_queue_priority(warm),
        )
        self.assertLess(
            _direct_partition_queue_priority(warm),
            _direct_partition_queue_priority(generic),
        )

    def test_render_feedback_composition_can_use_bounded_extra_cohort(
        self,
    ) -> None:
        self.assertFalse(
            _direct_partition_terminal_audition_due(
                evaluated_profile_count=24,
                feasible_profile_count=0,
                queued_profile_count=16,
                maximum_evaluation_count=40,
            )
        )
        self.assertTrue(
            _direct_partition_terminal_audition_due(
                evaluated_profile_count=40,
                feasible_profile_count=0,
                queued_profile_count=1,
                maximum_evaluation_count=40,
            )
        )

    def test_terminal_hold_cohort_excludes_content_invalid_profiles(
        self,
    ) -> None:
        self.assertFalse(
            _direct_partition_terminal_hold_failure_is_eligible(
                {
                    "failure": "content_validity_quota_precheck_failed",
                    "quota_shortfalls": [],
                    "empty_segment_indices": [7],
                }
            )
        )
        self.assertTrue(
            _direct_partition_terminal_hold_failure_is_eligible(
                {
                    "failure": "boundary_connectivity_path_exhausted",
                    "constraint_infeasibility_proven": True,
                }
            )
        )
        self.assertTrue(
            _direct_partition_terminal_hold_failure_is_eligible(
                {
                    "failure": "validity_quota_path_exhausted",
                    "constraint_infeasibility_proven": True,
                }
            )
        )
        self.assertFalse(
            _direct_partition_terminal_hold_failure_is_eligible(
                {
                    "failure": "search_capacity_exhausted",
                    "constraint_infeasibility_proven": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
