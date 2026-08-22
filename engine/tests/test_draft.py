from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from crosscut.draft import build_group_draft


class GroupDraftTests(unittest.TestCase):
    def build(self, **overrides: object):
        inputs: dict[str, object] = {
            "source_count": 3,
            "common_master_start": 2.0,
            "common_master_end": 14.0,
            "audio_offsets": (-0.5, 0.0, 0.75),
            "sync_scores": (0.91, 0.97, 0.84),
            "beat_times": tuple(float(value) for value in range(0, 17)),
            "phrase_times": (4.0, 8.0, 12.0),
            "scene_cut_times": (
                (3.5, 7.5, 11.5),       # canonical 4, 8, 12
                (4.0, 8.0, 12.0),       # canonical 4, 8, 12
                (4.75, 8.75, 12.75),    # canonical 4, 8, 12
            ),
        }
        inputs.update(overrides)
        return build_group_draft(**inputs)

    def test_contiguous_cover_one_to_four_beats_and_source_rotation(self) -> None:
        timeline = self.build()

        self.assertEqual(timeline.segments[0].canonical_master_start, 2.0)
        self.assertEqual(timeline.segments[-1].canonical_master_end, 14.0)
        for previous, current in zip(timeline.segments, timeline.segments[1:]):
            self.assertAlmostEqual(previous.canonical_master_end, current.canonical_master_start)
            self.assertAlmostEqual(
                previous.output_start_seconds + previous.duration_seconds,
                current.output_start_seconds,
            )
            self.assertNotEqual(previous.source_index, current.source_index)
        self.assertTrue(
            all(1.0 <= segment.estimated_beat_count <= 4.0 for segment in timeline.segments)
        )
        self.assertEqual(timeline.used_source_indices, (0, 1, 2))
        # Each complete three-shot rotation contains every source exactly once.
        for start in range(0, len(timeline.segments) - 2, 3):
            self.assertEqual(
                {segment.source_index for segment in timeline.segments[start : start + 3]},
                {0, 1, 2},
            )

    def test_positive_and_negative_offsets_use_master_plus_offset(self) -> None:
        timeline = self.build(audio_offsets=(-1.25, 0.0, 1.5))

        seen_negative = False
        seen_positive = False
        for segment in timeline.segments:
            expected = segment.canonical_master_start + segment.audio_offset_seconds
            self.assertAlmostEqual(segment.source_local_start, expected)
            self.assertAlmostEqual(
                segment.source_local_end,
                segment.canonical_master_end + segment.audio_offset_seconds,
            )
            seen_negative |= segment.audio_offset_seconds < 0
            seen_positive |= segment.audio_offset_seconds > 0
        self.assertTrue(seen_negative)
        self.assertTrue(seen_positive)

    def test_source_local_scene_cuts_are_converted_before_boundary_scoring(self) -> None:
        # All three source-local timestamps map to canonical master 6.0 even
        # though their raw timestamp values differ by the alignment offsets.
        timeline = self.build(
            common_master_start=2.0,
            common_master_end=10.0,
            phrase_times=(),
            scene_cut_times=((5.5,), (6.0,), (6.75,)),
        )

        boundaries = {segment.canonical_master_end: segment for segment in timeline.segments}
        self.assertIn(6.0, boundaries)
        self.assertIn("multi_source_scene_cut", boundaries[6.0].end_boundary_reasons)

    def test_deterministic_and_explicitly_metadata_only(self) -> None:
        first = self.build()
        second = self.build()

        self.assertEqual(first, second)
        self.assertEqual(first.selection_basis, "rhythm_scene_metadata_baseline")
        self.assertFalse(first.visual_quality_scored)
        self.assertTrue(any("pixel-level" in note for note in first.baseline_limitations))
        self.assertIn("segments", first.to_dict())
        with self.assertRaises(FrozenInstanceError):
            first.segments[0].source_index = 1  # type: ignore[misc]

    def test_common_range_too_short_for_all_sources_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            self.build(
                source_count=5,
                common_master_start=2.0,
                common_master_end=5.0,
                audio_offsets=(0.0,) * 5,
                sync_scores=(0.9,) * 5,
                scene_cut_times=((),) * 5,
            )

    def test_sub_two_frame_edge_beat_is_merged_into_neighboring_shot(self) -> None:
        timeline = self.build(
            common_master_start=0.0,
            common_master_end=6.0,
            audio_offsets=(0.0, 0.0, 0.0),
            beat_times=(0.15, 0.675, 1.20, 1.725, 2.25, 2.775, 3.30, 3.825,
                        4.35, 4.875, 5.40, 5.936222),
            phrase_times=(),
            scene_cut_times=((), (), ()),
        )

        self.assertEqual(timeline.segments[-1].canonical_master_end, 6.0)
        self.assertNotEqual(timeline.segments[-1].canonical_master_start, 5.936222)
        self.assertTrue(
            all(segment.duration_seconds >= 2.0 / 29.97 for segment in timeline.segments)
        )

    def test_rejects_invalid_sync_and_out_of_range_source_clock(self) -> None:
        with self.assertRaisesRegex(ValueError, "sync_scores"):
            self.build(sync_scores=(0.9, 1.1, 0.8))
        with self.assertRaisesRegex(ValueError, "local timeline"):
            self.build(audio_offsets=(-2.5, 0.0, 0.75))


if __name__ == "__main__":
    unittest.main()
