from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product
import math
from random import Random
import unittest

from crosscut.ive import AspectRatio, IVEMember
from crosscut.member_draft import (
    MemberDraftConfig,
    MemberSegmentKind,
    SegmentCropSample,
    _Candidate,
    _exact_diversity_path,
    _matching_anchored_balanced_beam_path,
    _maximum_source_segment_matching,
    build_member_draft,
)
from crosscut.tracking import (
    MemberTrackObservation,
    MemberTrackingResult,
    TrackingState,
)


def tracking_result(
    source_id: str,
    *,
    start: float,
    end: float,
    fps: float = 10.0,
    confidence: float = 0.95,
    member: IVEMember = IVEMember.GAEUL,
    movement_per_frame: float = 0.0,
    omitted_frames: frozenset[int] = frozenset(),
) -> MemberTrackingResult:
    first = round(start * fps)
    last = round(end * fps)
    observations: list[MemberTrackObservation] = []
    for frame_index in range(first, last + 1):
        if frame_index in omitted_frames:
            continue
        center_x = 0.30 + movement_per_frame * (frame_index - first)
        observations.append(
            MemberTrackObservation(
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                bbox_xyxy=(center_x - 0.07, 0.20, center_x + 0.07, 0.75),
                confidence=confidence,
                state=TrackingState.TRACKED,
                member=member,
                source_id=source_id,
            )
        )
    return MemberTrackingResult(
        member=member,
        source_id=source_id,
        fps=fps,
        backend_name="synthetic",
        observations=tuple(observations),
        terminal_failure=None,
    )


def selection_candidate(
    source_index: int,
    base_score: float,
    framing: tuple[float, float, float],
) -> _Candidate:
    return _Candidate(
        source_index=source_index,
        local_start=0.0,
        local_end=1.0,
        coverage=1.0,
        mean_confidence=base_score,
        stability=1.0,
        base_score=base_score,
        crop_samples=(SegmentCropSample(0.0, 0.0, 0.0, 0.31640625, 1.0),),
        entry_framing=framing,
        exit_framing=framing,
    )


def brute_force_selection(
    rows: tuple[tuple[_Candidate, ...], ...],
    settings: MemberDraftConfig,
) -> tuple[int, ...]:
    """Independent small-case oracle for the exact path objective."""

    best_metrics: tuple[int, float, float, float, int, float, float] | None = None
    best_sequence: tuple[int, ...] | None = None
    source_count = max(
        candidate.source_index for row in rows for candidate in row
    ) + 1
    for candidates in product(*rows):
        counts = [0] * source_count
        last_positions = [-1] * source_count
        utility = 0.0
        base_sum = 0.0
        continuity_sum = 0.0
        repeat_count = 0
        accumulated_sum = 0.0
        recurrence_sum = 0.0
        previous: _Candidate | None = None
        sequence: list[int] = []
        for segment_index, candidate in enumerate(candidates):
            mismatch = 0.0
            if previous is not None:
                left_x, left_y, left_height = previous.exit_framing
                right_x, right_y, right_height = candidate.entry_framing
                mismatch = min(
                    3.0,
                    math.hypot(right_x - left_x, right_y - left_y)
                    + 0.35
                    * abs(
                        math.log(
                            max(right_height, 1e-6)
                            / max(left_height, 1e-6)
                        )
                    ),
                )
            repeated = (
                previous is not None
                and previous.source_index == candidate.source_index
            )
            prior_count = counts[candidate.source_index]
            accumulated = settings.accumulated_use_penalty * prior_count
            last_position = last_positions[candidate.source_index]
            recurrence = (
                0.0
                if last_position < 0
                else 1.0 / (segment_index - last_position)
            )
            utility += (
                candidate.base_score
                - settings.continuity_weight * mismatch
                - (settings.repeat_source_penalty if repeated else 0.0)
                - accumulated
            )
            base_sum += candidate.base_score
            continuity_sum += mismatch
            repeat_count += int(repeated)
            accumulated_sum += accumulated
            recurrence_sum += recurrence
            counts[candidate.source_index] += 1
            last_positions[candidate.source_index] = segment_index
            sequence.append(candidate.source_index)
            previous = candidate
        metrics = (
            len(set(sequence)),
            utility,
            base_sum,
            -continuity_sum,
            -repeat_count,
            -accumulated_sum,
            -recurrence_sum,
        )
        sequence_tuple = tuple(sequence)
        if (
            best_metrics is None
            or metrics > best_metrics
            or (metrics == best_metrics and sequence_tuple < best_sequence)
        ):
            best_metrics = metrics
            best_sequence = sequence_tuple
    assert best_sequence is not None
    return best_sequence


class MemberDraftTests(unittest.TestCase):
    def test_positive_and_negative_offsets_keep_three_clocks_separate(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((2.0, 3.0), (3.0, 4.0)),
            audio_offsets=(-1.0, 1.0),
            tracking_by_source=(
                tracking_result("negative", start=1.0, end=3.0),
                tracking_result("positive", start=3.0, end=5.0),
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        # Equal sources rotate deterministically, exercising both offset signs.
        self.assertEqual([item.source_index for item in timeline.segments], [0, 1])
        self.assertAlmostEqual(timeline.segments[0].source_local_start, 1.0)
        self.assertAlmostEqual(timeline.segments[0].output_start_seconds, 0.0)
        self.assertAlmostEqual(timeline.segments[1].source_local_start, 4.0)
        self.assertAlmostEqual(timeline.segments[1].output_start_seconds, 1.0)
        for segment in timeline.segments:
            assert segment.source_index is not None
            self.assertAlmostEqual(
                segment.source_local_start,
                segment.canonical_master_start
                + timeline.audio_offsets[segment.source_index],
            )

    def test_low_confidence_wrong_member_and_long_gap_never_become_crop(self) -> None:
        long_gap = frozenset(range(4, 8))
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0),),
            audio_offsets=(0.0, 0.0, 0.0, 0.0),
            tracking_by_source=(
                tracking_result("low", start=0.0, end=1.0, confidence=0.70),
                tracking_result(
                    "wrong",
                    start=0.0,
                    end=1.0,
                    member=IVEMember.REI,
                ),
                tracking_result(
                    "gapped",
                    start=0.0,
                    end=1.0,
                    omitted_frames=long_gap,
                ),
                None,
            ),
            target_member=IVEMember.GAEUL,
            output_aspect=AspectRatio.PORTRAIT,
            group_wide_fallback_source=3,
        )

        segment = timeline.segments[0]
        self.assertIs(segment.kind, MemberSegmentKind.GROUP_WIDE_FALLBACK)
        self.assertEqual(segment.source_index, 3)
        self.assertEqual(segment.crop_samples, ())
        self.assertEqual(segment.interval_coverage, 0.0)

    def test_no_safe_source_without_explicit_fallback_is_unavailable(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0),),
            audio_offsets=(0.0,),
            tracking_by_source=(
                tracking_result("lost", start=0.0, end=1.0, confidence=0.2),
            ),
            target_member="가을",
            output_aspect="16:9",
        )

        segment = timeline.segments[0]
        self.assertIs(segment.kind, MemberSegmentKind.UNAVAILABLE)
        self.assertIsNone(segment.source_index)
        self.assertEqual(segment.crop_samples, ())

    def test_repeat_penalty_switches_equally_safe_sources_deterministically(self) -> None:
        first = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0),
            tracking_by_source=(
                tracking_result("a", start=0.0, end=3.0),
                tracking_result("b", start=0.0, end=3.0),
            ),
            target_member="가을",
            output_aspect="16:9",
        )
        second = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0),
            tracking_by_source=(
                tracking_result("a", start=0.0, end=3.0),
                tracking_result("b", start=0.0, end=3.0),
            ),
            target_member="가을",
            output_aspect="16:9",
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [segment.source_index for segment in first.segments], [0, 1, 0]
        )
        self.assertTrue(
            all(
                segment.kind is MemberSegmentKind.TARGET_CROP
                for segment in first.segments
            )
        )

    def test_full_length_draft_distributes_equal_candidates_across_nine_sources(self) -> None:
        timeline = build_member_draft(
            canonical_spans=tuple(
                (float(index), float(index + 1)) for index in range(18)
            ),
            audio_offsets=(0.0,) * 9,
            tracking_by_source=tuple(
                tracking_result(str(index), start=0.0, end=18.0)
                for index in range(9)
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        sequence = [segment.source_index for segment in timeline.segments]
        counts = [sequence.count(source_index) for source_index in range(9)]
        self.assertEqual(set(sequence), set(range(9)))
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(set(sequence[:9]), set(range(9)))
        self.assertEqual(set(sequence[9:]), set(range(9)))
        self.assertTrue(
            all(left != right for left, right in zip(sequence, sequence[1:]))
        )
        for first_use, second_use in zip(
            timeline.segments[:9], timeline.segments[9:], strict=True
        ):
            assert first_use.selection_score is not None
            assert second_use.selection_score is not None
            self.assertAlmostEqual(
                first_use.selection_score - second_use.selection_score,
                0.015,
            )
        self.assertEqual(
            timeline.selection_diagnostics.maximum_distinct_quality_source_count,
            9,
        )
        self.assertEqual(
            timeline.to_dict()["selection_diagnostics"][
                "selected_distinct_source_count"
            ],
            9,
        )

    def test_global_diversity_uses_quality_guarded_source_greedy_would_starve(self) -> None:
        first = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0, 0.0),
            tracking_by_source=(
                tracking_result("best", start=0.0, end=3.0, confidence=0.99),
                tracking_result("second", start=0.0, end=3.0, confidence=0.97),
                tracking_result("third", start=0.0, end=3.0, confidence=0.96),
            ),
            target_member="가을",
            output_aspect="9:16",
        )
        second = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0, 0.0),
            tracking_by_source=(
                tracking_result("best", start=0.0, end=3.0, confidence=0.99),
                tracking_result("second", start=0.0, end=3.0, confidence=0.97),
                tracking_result("third", start=0.0, end=3.0, confidence=0.96),
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        sequence = [segment.source_index for segment in first.segments]
        self.assertEqual(first, second)
        self.assertEqual(sequence, [0, 1, 2])
        self.assertEqual(first.selection_diagnostics.optimizer, "exact_distinct_mask_dp")

    def test_exact_state_keeps_usage_and_last_positions_for_future_costs(self) -> None:
        rows = (
            (
                selection_candidate(0, 0.897, (0.64, 0.92, 0.74)),
                selection_candidate(1, 0.809, (0.05, 0.41, 0.28)),
                selection_candidate(2, 0.895, (0.40, 0.70, 0.23)),
            ),
            (
                selection_candidate(1, 0.860, (0.24, 0.88, 0.23)),
                selection_candidate(2, 0.880, (0.77, 0.98, 0.61)),
            ),
            (
                selection_candidate(1, 0.926, (0.95, 0.04, 0.24)),
                selection_candidate(2, 0.948, (0.08, 0.37, 0.79)),
            ),
            (
                selection_candidate(0, 0.876, (0.36, 0.02, 0.29)),
                selection_candidate(2, 0.868, (0.92, 0.73, 0.35)),
            ),
            (
                selection_candidate(0, 0.931, (0.32, 0.72, 0.65)),
                selection_candidate(1, 0.892, (0.16, 0.67, 0.53)),
                selection_candidate(2, 0.930, (0.81, 0.80, 0.27)),
            ),
        )
        settings = MemberDraftConfig()

        selected = _exact_diversity_path(
            rows,
            quality_source_indices=(0, 1, 2),
            settings=settings,
        )
        sequence = tuple(candidate.source_index for candidate in selected if candidate)

        self.assertEqual(sequence, brute_force_selection(rows, settings))
        self.assertEqual(sequence, (0, 2, 1, 0, 2))

    def test_exact_path_matches_seeded_random_brute_force_oracle(self) -> None:
        random = Random(20260823)
        settings = MemberDraftConfig()
        for case_index in range(80):
            rows: list[tuple[_Candidate, ...]] = []
            for _ in range(5):
                choices = tuple(
                    selection_candidate(
                        source_index,
                        random.uniform(0.84, 0.98),
                        (
                            random.uniform(0.05, 0.95),
                            random.uniform(0.05, 0.95),
                            random.uniform(0.20, 0.85),
                        ),
                    )
                    for source_index in range(3)
                    if random.random() < 0.78
                )
                if choices:
                    rows.append(choices)
                else:
                    source_index = random.randrange(3)
                    rows.append(
                        (
                            selection_candidate(
                                source_index,
                                random.uniform(0.84, 0.98),
                                (0.5, 0.5, 0.5),
                            ),
                        )
                    )
            frozen_rows = tuple(rows)
            quality_sources = tuple(
                sorted(
                    {
                        candidate.source_index
                        for row in frozen_rows
                        for candidate in row
                    }
                )
            )

            selected = _exact_diversity_path(
                frozen_rows,
                quality_source_indices=quality_sources,
                settings=settings,
            )
            sequence = tuple(
                candidate.source_index for candidate in selected if candidate
            )
            self.assertEqual(
                sequence,
                brute_force_selection(frozen_rows, settings),
                msg=f"random oracle mismatch at case {case_index}",
            )

    def test_balanced_beam_matches_small_anchored_brute_force_oracle(self) -> None:
        random = Random(20260824)
        settings = MemberDraftConfig()
        for case_index in range(60):
            rows: list[tuple[_Candidate, ...]] = []
            for _ in range(5):
                choices = tuple(
                    selection_candidate(
                        source_index,
                        random.uniform(0.84, 0.98),
                        (
                            random.uniform(0.05, 0.95),
                            random.uniform(0.05, 0.95),
                            random.uniform(0.20, 0.85),
                        ),
                    )
                    for source_index in range(3)
                    if random.random() < 0.78
                )
                rows.append(
                    choices
                    or (
                        selection_candidate(
                            random.randrange(3),
                            random.uniform(0.84, 0.98),
                            (0.5, 0.5, 0.5),
                        ),
                    )
                )
            frozen_rows = tuple(rows)
            matching = _maximum_source_segment_matching(frozen_rows)
            anchored_rows = tuple(
                tuple(
                    candidate
                    for candidate in row
                    if matching.get(segment_index) is None
                    or candidate.source_index == matching[segment_index]
                )
                for segment_index, row in enumerate(frozen_rows)
            )

            selected = _matching_anchored_balanced_beam_path(
                frozen_rows,
                matching=matching,
                settings=settings,
            )
            sequence = tuple(
                candidate.source_index for candidate in selected if candidate
            )
            self.assertEqual(
                sequence,
                brute_force_selection(anchored_rows, settings),
                msg=f"balanced beam oracle mismatch at case {case_index}",
            )

    def test_quality_guard_never_forces_visibly_lower_safe_source(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0, 0.0),
            tracking_by_source=(
                tracking_result("best", start=0.0, end=3.0, confidence=0.99),
                tracking_result("near", start=0.0, end=3.0, confidence=0.97),
                tracking_result("safe-but-low", start=0.0, end=3.0, confidence=0.90),
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        payload = timeline.to_dict()["selection_diagnostics"]
        self.assertEqual(set(segment.source_index for segment in timeline.segments), {0, 1})
        self.assertEqual(payload["quality_guard_excluded_source_indices"], [2])
        self.assertEqual(payload["sources"][2]["safe_candidate_count"], 3)
        self.assertEqual(payload["sources"][2]["outcome"], "quality_guard_excluded")

    def test_global_path_reserves_source_that_is_safe_only_in_first_span(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
            audio_offsets=(0.0, 0.0, 0.0),
            tracking_by_source=(
                tracking_result("a", start=0.0, end=3.0, confidence=0.99),
                tracking_result("b", start=0.0, end=3.0, confidence=0.98),
                tracking_result("early", start=0.0, end=1.0, confidence=0.97),
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        sequence = [segment.source_index for segment in timeline.segments]
        self.assertEqual(sequence[0], 2)
        self.assertEqual(set(sequence), {0, 1, 2})

    def test_large_source_fallback_preserves_maximum_matching_diversity(self) -> None:
        kwargs = dict(
            canonical_spans=tuple(
                (float(index), float(index + 1)) for index in range(18)
            ),
            audio_offsets=(0.0,) * 9,
            tracking_by_source=tuple(
                tracking_result(str(index), start=0.0, end=18.0)
                for index in range(9)
            ),
            target_member="가을",
            output_aspect="9:16",
            config=MemberDraftConfig(exact_diversity_source_limit=8),
        )
        timeline = build_member_draft(**kwargs)
        repeated = build_member_draft(**kwargs)

        self.assertEqual(timeline, repeated)
        self.assertEqual(
            timeline.selection_diagnostics.optimizer,
            "maximum_matching_anchored_balanced_beam",
        )
        sequence = [segment.source_index for segment in timeline.segments]
        counts = [sequence.count(source_index) for source_index in range(9)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(set(sequence[:9]), set(range(9)))
        self.assertEqual(set(sequence[9:]), set(range(9)))
        self.assertTrue(
            all(left != right for left, right in zip(sequence, sequence[1:]))
        )

    def test_virtual_camera_emits_portrait_dynamic_pan_samples(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0),),
            audio_offsets=(0.0,),
            tracking_by_source=(
                tracking_result(
                    "moving",
                    start=0.0,
                    end=1.0,
                    movement_per_frame=0.012,
                ),
            ),
            target_member="가을",
            output_aspect="9:16",
        )

        samples = timeline.segments[0].crop_samples
        self.assertGreater(len(samples), 2)
        self.assertEqual(samples[0].time_seconds, 0.0)
        self.assertGreater(samples[-1].x, samples[0].x)
        self.assertAlmostEqual(timeline.normalized_crop_aspect, 81.0 / 256.0)
        for sample in samples:
            self.assertAlmostEqual(
                sample.width / sample.height,
                (9.0 / 16.0) / (16.0 / 9.0),
                places=7,
            )
        pan_track = timeline.segments[0].to_pan_crop_track()
        self.assertIsNotNone(pan_track)
        assert pan_track is not None
        self.assertEqual(len(pan_track.keyframes), len(samples))
        self.assertAlmostEqual(pan_track.keyframes[-1].crop.x, samples[-1].x)

    def test_raw_observations_are_accepted_and_results_are_immutable_serializable(self) -> None:
        raw = tracking_result("raw", start=0.0, end=1.0).observations
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0),),
            audio_offsets=(0.0,),
            tracking_by_source=(raw,),
            target_member="가을",
            output_aspect="9:16",
        )

        payload = timeline.to_dict()
        self.assertEqual(payload["target_member"], "가을")
        self.assertEqual(payload["segments"][0]["kind"], "target_crop")
        self.assertTrue(payload["segments"][0]["crop_samples"])
        with self.assertRaises(FrozenInstanceError):
            timeline.segments[0].source_index = 99  # type: ignore[misc]

    def test_config_can_demand_complete_interval_coverage(self) -> None:
        timeline = build_member_draft(
            canonical_spans=((0.0, 1.0),),
            audio_offsets=(0.0, 0.0),
            tracking_by_source=(
                tracking_result(
                    "missing",
                    start=0.0,
                    end=1.0,
                    omitted_frames=frozenset({5}),
                ),
                None,
            ),
            target_member="가을",
            output_aspect="9:16",
            group_wide_fallback_source=1,
            config=MemberDraftConfig(
                minimum_interval_coverage=0.99,
                maximum_untracked_gap_seconds=0.05,
            ),
        )

        self.assertIs(
            timeline.segments[0].kind, MemberSegmentKind.GROUP_WIDE_FALLBACK
        )


if __name__ == "__main__":
    unittest.main()
