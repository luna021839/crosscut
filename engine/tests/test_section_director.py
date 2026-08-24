from __future__ import annotations

import unittest

from crosscut.rhythm import (
    Beat,
    OnsetEnvelope,
    PhraseCandidate,
    RhythmAnalysis,
    TempoEstimate,
)
from crosscut.section_director import SectionRole, build_section_plan


def _rhythm() -> RhythmAnalysis:
    # Six non-overlapping eight-beat phrases with a deliberate low -> build ->
    # high -> low -> high energy arc.  The first/last are position-labelled.
    phrase_energies = (0.12, 0.24, 0.48, 1.00, 0.18, 0.82)
    values: list[float] = []
    for energy in phrase_energies:
        values.extend([energy] * 80)
    onset = OnsetEnvelope(tuple(values), sample_rate=100, frame_length=20, hop_length=10)
    beats = tuple(
        Beat(index=index, time_seconds=float(index), strength=phrase_energies[min(5, index // 8)])
        for index in range(49)
    )
    phrases = tuple(
        PhraseCandidate(
            beat_count=8,
            start_beat_index=index * 8,
            end_beat_index=(index + 1) * 8,
            start_seconds=float(index * 8),
            end_seconds=float((index + 1) * 8),
            confidence=0.90,
        )
        for index in range(6)
    )
    return RhythmAnalysis(
        sample_rate=100,
        duration_seconds=48.0,
        onset_envelope=onset,
        tempo=TempoEstimate(60.0, 1.0, 0.0, 0.9),
        beats=beats,
        phrase_candidates=phrases,
    )


class SectionDirectorTests(unittest.TestCase):
    def test_builds_contiguous_deterministic_acoustic_roles(self) -> None:
        first = build_section_plan(_rhythm(), start_seconds=0.0, end_seconds=48.0)
        second = build_section_plan(_rhythm(), start_seconds=0.0, end_seconds=48.0)
        self.assertEqual(first, second)
        self.assertEqual(first.start_seconds, 0.0)
        self.assertEqual(first.end_seconds, 48.0)
        self.assertEqual(first.sections[0].role, SectionRole.INTRO)
        self.assertEqual(first.sections[-1].role, SectionRole.OUTRO)
        self.assertTrue(any(item.role is SectionRole.CHORUS_LIKE for item in first.sections))
        for left, right in zip(first.sections, first.sections[1:]):
            self.assertAlmostEqual(left.end_seconds, right.start_seconds)

    def test_high_energy_role_has_denser_cut_and_motion_budget(self) -> None:
        plan = build_section_plan(_rhythm(), start_seconds=0.0, end_seconds=48.0)
        chorus = next(item for item in plan.sections if item.role is SectionRole.CHORUS_LIKE)
        verse = next(item for item in plan.sections if item.role is SectionRole.VERSE_LIKE)
        self.assertGreater(
            chorus.target_source_switches_per_minute,
            verse.target_source_switches_per_minute,
        )
        self.assertGreater(chorus.camera_motion_budget, verse.camera_motion_budget)
        self.assertLess(
            chorus.preferred_source_hold_beats,
            verse.preferred_source_hold_beats,
        )

    def test_reference_profile_is_clamped(self) -> None:
        plan = build_section_plan(
            _rhythm(),
            start_seconds=0.0,
            end_seconds=48.0,
            reference_profile={
                "target_source_switches_per_minute_by_role": {
                    "chorus_like": 99.0,
                },
                "camera_motion_budget_by_role": {"chorus_like": 5.0},
                "target_roll_events_per_minute_by_role": {"chorus_like": 8.0},
            },
        )
        chorus = next(item for item in plan.sections if item.role is SectionRole.CHORUS_LIKE)
        self.assertEqual(chorus.target_source_switches_per_minute, 22.0)
        self.assertEqual(chorus.camera_motion_budget, 1.0)
        self.assertEqual(chorus.target_roll_events_per_minute, 0.75)

    def test_no_phrase_fallback_is_honest_and_serializable(self) -> None:
        rhythm = _rhythm()
        empty = RhythmAnalysis(
            rhythm.sample_rate,
            rhythm.duration_seconds,
            rhythm.onset_envelope,
            rhythm.tempo,
            rhythm.beats,
            (),
        )
        plan = build_section_plan(empty, start_seconds=2.0, end_seconds=12.0)
        self.assertEqual(len(plan.sections), 1)
        self.assertEqual(plan.sections[0].role, SectionRole.VERSE_LIKE)
        self.assertIn("not_ground_truth", plan.to_dict()["role_semantics"])


if __name__ == "__main__":
    unittest.main()
