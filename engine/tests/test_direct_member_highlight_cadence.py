from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.draft import DraftSegment
from crosscut.pipeline import (
    _coarsen_direct_full_frame_spans,
    _densify_direct_full_frame_highlights,
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

    def test_highlight_no_cut_hold_fails_the_section_audit(self) -> None:
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

        self.assertFalse(payload["passed"])
        self.assertIn(
            "highlight_boundary_consumed_by_no_cut_hold",
            payload["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
