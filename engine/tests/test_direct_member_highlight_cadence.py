from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.draft import DraftSegment
from crosscut.pipeline import (
    _DirectFullFramePartitionProfile,
    _coarsen_direct_full_frame_spans,
    _densify_direct_full_frame_highlights,
    _direct_full_frame_proof_directed_child_profiles,
    _direct_full_frame_spans_from_partition,
    _direct_member_highlight_cadence_payload,
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
    def test_proven_shift_uses_real_seconds_not_a_two_unit_proxy(self) -> None:
        unit_counts = (4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4)
        durations: list[float] = []
        for span_index, count in enumerate(unit_counts):
            if span_index == 4:
                durations.extend((1.9, 1.9))
            elif span_index == 5:
                durations.extend((0.475,) * 4)
            else:
                durations.extend((1.0,) * count)
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
                SimpleNamespace(cut_applied=True)
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
            boundaries=tuple(
                SimpleNamespace(cut_applied=value) for value in cut_flags
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
        for boundary_index in highlight_boundary_indices[:3]:
            cut_flags[boundary_index] = False
        selection = SimpleNamespace(
            source_indices=tuple(index % 8 for index in range(len(dense))),
            boundaries=tuple(
                SimpleNamespace(cut_applied=value) for value in cut_flags
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


if __name__ == "__main__":
    unittest.main()
