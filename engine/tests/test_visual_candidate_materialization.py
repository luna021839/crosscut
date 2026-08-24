from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.timeline import (
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.timeline_review import review_timeline
from crosscut.visual_candidate_materialization import (
    materialize_visual_timeline_candidates,
    materialize_visual_timeline_render_plans,
)
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelinePlan,
    _DissolveEvidence,
    _review_ranked_candidate_indices,
    build_visual_timeline,
    build_visual_timeline_candidate_set,
)


class VisualCandidateMaterializationTests(unittest.TestCase):
    RAW_SENTINEL = 0.9876543212345

    @classmethod
    def timeline_result(
        cls,
        source_ids: tuple[str, ...],
        *,
        total_score: float,
    ) -> TimelineResult:
        choices = tuple(
            TimelineChoice(
                candidate=TimelineCandidate(
                    segment_index=index,
                    source_id=source_id,
                    start_seconds=2.0 + index,
                    end_seconds=3.0 + index,
                    candidate_id=f"{index}:{source_id}",
                    shot_id=f"{source_id}:shot-{index}",
                    face_identity_embedding=(cls.RAW_SENTINEL,),
                ),
                emission_score=1.0,
                transition_score=0.0,
                cumulative_score=float(index + 1),
                used_group_wide_fallback=False,
            )
            for index, source_id in enumerate(source_ids)
        )
        return TimelineResult(
            mode=TimelineMode.GENERAL,
            target_member=None,
            choices=choices,
            total_score=total_score,
        )

    @classmethod
    def candidate_set(cls) -> VisualTimelineCandidateSet:
        sequences = (
            ("0", "0", "0", "0"),
            ("1", "1", "1", "1"),
            ("0", "1", "0", "1"),
            ("0", "0", "1", "1"),
            ("0", "1", "1", "0"),
            ("1", "0", "0", "1"),
            ("1", "1", "0", "0"),
            ("1", "0", "1", "0"),
        )
        results = tuple(
            cls.timeline_result(sequence, total_score=10.0 - index)
            for index, sequence in enumerate(sequences)
        )
        reviews = tuple(
            review_timeline(result, available_source_count=2) for result in results
        )
        ranked = _review_ranked_candidate_indices(reviews)
        rank_by_index = {
            candidate_index: rank
            for rank, candidate_index in enumerate(ranked)
        }
        records = tuple(
            VisualTimelineCandidateRecord(
                candidate_index=index,
                cadence_profile=("balanced", "reference", "dense_safe")[index % 3],
                timeline_result=result,
                review=reviews[index],
                review_rank=rank_by_index[index],
                selected=index == ranked[0],
            )
            for index, result in enumerate(results)
        )
        return VisualTimelineCandidateSet(
            candidates=records,
            selected_candidate_index=ranked[0],
            sources=(Path("source-0.mp4"), Path("source-1.mp4")),
            audio_offsets=(0.25, 0.50),
            source_timewarps=(None, None),
            source_clock_basis="global_scalar_offset",
            requested_maximum_results=8,
        )

    def test_candidate_set_preserves_typed_results_and_privacy_safe_review_data(
        self,
    ) -> None:
        candidate_set = self.candidate_set()
        first = candidate_set.candidates[0]

        self.assertIs(
            first.timeline_result.choices[0].candidate,
            first.timeline_result.candidates[0],
        )
        self.assertIsInstance(first.review.weighted_score, float)
        self.assertEqual(
            [item.review_rank for item in candidate_set.ranked_candidates],
            list(range(8)),
        )
        self.assertIs(
            candidate_set.selected,
            candidate_set.ranked_candidates[0],
        )
        self.assertTrue(candidate_set.minimum_materialization_target_satisfied)
        payload = candidate_set.to_dict()
        self.assertFalse(payload["raw_candidate_embeddings_persisted"])
        self.assertIn("review", payload["candidates"][0])
        self.assertNotIn(str(self.RAW_SENTINEL), repr(candidate_set))
        self.assertNotIn(str(self.RAW_SENTINEL), repr(payload))

    def test_materializes_ranked_top_n_with_exact_source_clock_and_render_plans(
        self,
    ) -> None:
        candidate_set = self.candidate_set()
        materialized = materialize_visual_timeline_candidates(
            candidate_set,
            maximum_candidates=8,
        )

        self.assertEqual(len(materialized), 8)
        self.assertEqual(
            [item.candidate_index for item in materialized],
            [item.candidate_index for item in candidate_set.ranked_candidates],
        )
        for item in materialized:
            self.assertEqual(item.common_master_start, 2.0)
            self.assertEqual(
                [segment.master_start for segment in item.segments],
                [0.0, 1.0, 2.0, 3.0],
            )
            for source_candidate, segment in zip(
                item.candidate_record.timeline_result.candidates,
                item.segments,
                strict=True,
            ):
                source_index = int(source_candidate.source_id)
                self.assertAlmostEqual(
                    segment.source_start,
                    source_candidate.start_seconds
                    + candidate_set.audio_offsets[source_index],
                )
            self.assertNotIn(
                str(self.RAW_SENTINEL), repr(item.candidate_history_entry)
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = materialize_visual_timeline_render_plans(
                candidate_set,
                output_directory=Path(temporary_directory),
                maximum_candidates=8,
                aspect_ratio=AspectRatio.PORTRAIT,
            )
            repeated = materialize_visual_timeline_render_plans(
                candidate_set,
                output_directory=Path(temporary_directory),
                maximum_candidates=8,
                aspect_ratio=AspectRatio.PORTRAIT,
            )

        self.assertEqual(plans, repeated)
        self.assertEqual(len({plan.output for plan in plans}), 8)
        self.assertEqual((plans[0].width, plans[0].height), (1080, 1920))
        self.assertEqual(plans[0].master_audio_start, 2.0)
        self.assertTrue(
            all(transition.frame_count == 0 for transition in plans[0].transitions)
        )
        self.assertIn("rank-01", plans[0].output.name)

    def test_candidate_set_rejects_duplicate_paths(self) -> None:
        candidate_set = self.candidate_set()
        duplicate_result = candidate_set.candidates[0].timeline_result
        duplicate_review = review_timeline(
            duplicate_result,
            available_source_count=2,
        )
        records = (
            VisualTimelineCandidateRecord(
                candidate_index=0,
                cadence_profile="balanced",
                timeline_result=duplicate_result,
                review=duplicate_review,
                review_rank=0,
                selected=True,
            ),
            VisualTimelineCandidateRecord(
                candidate_index=1,
                cadence_profile="reference",
                timeline_result=duplicate_result,
                review=duplicate_review,
                review_rank=1,
            ),
        )

        with self.assertRaisesRegex(ValueError, "duplicate timeline paths"):
            VisualTimelineCandidateSet(
                candidates=records,
                selected_candidate_index=0,
                sources=(Path("source-0.mp4"), Path("source-1.mp4")),
                audio_offsets=(0.0, 0.0),
                source_timewarps=(None, None),
                source_clock_basis="global_scalar_offset",
                requested_maximum_results=8,
            )

    def test_builder_retains_eight_actual_alternatives_and_legacy_plan_contract(
        self,
    ) -> None:
        descriptor = FrameVisualDescriptor(
            quality_score=0.8,
            composition_score=0.8,
            shot_scale=0.4,
            shot_type=ShotType.MEDIUM,
            face_count=0,
            structure_embedding=(0.0,) * 96,
            person_layout_embedding=(0.2, 0.4, 0.3),
        )

        class Sampler:
            frame_rate = 30.0

            def __init__(self, _path: Path) -> None:
                pass

            def frame_at(self, _seconds: float) -> np.ndarray:
                return np.full((48, 80, 3), 118, dtype=np.uint8)

            def descriptor_at(self, _seconds: float) -> FrameVisualDescriptor:
                return descriptor

            def sface_embedding_at(
                self,
                _seconds: float,
                _descriptor: FrameVisualDescriptor,
            ) -> tuple[float, ...]:
                return (VisualCandidateMaterializationTests.RAW_SENTINEL,)

            def close(self) -> None:
                pass

        captured_results: list[tuple[TimelineResult, ...]] = []
        maximum_requests: list[int] = []

        def alternatives(
            candidates_by_segment: list[list[TimelineCandidate]],
            **kwargs: object,
        ) -> tuple[tuple[TimelineResult, ...], tuple[str, ...]]:
            maximum_requests.append(int(kwargs["maximum_results"]))
            patterns = tuple(
                tuple((pattern >> index) & 1 for index in range(4))
                for pattern in range(8)
            )
            results: list[TimelineResult] = []
            for pattern_index, pattern in enumerate(patterns):
                choices = tuple(
                    TimelineChoice(
                        candidate=candidates_by_segment[index][source_index],
                        emission_score=1.0,
                        transition_score=0.0,
                        cumulative_score=float(index + 1),
                        used_group_wide_fallback=False,
                    )
                    for index, source_index in enumerate(pattern)
                )
                results.append(
                    TimelineResult(
                        mode=TimelineMode.GENERAL,
                        target_member=None,
                        choices=choices,
                        total_score=20.0 - pattern_index,
                    )
                )
            captured_results.append(tuple(results))
            return (
                tuple(results),
                tuple(
                    ("balanced", "reference", "dense_safe")[index % 3]
                    for index in range(8)
                ),
            )

        common_arguments = {
            "sources": (Path("source-0.mp4"), Path("source-1.mp4")),
            "canonical_spans": tuple(
                (float(index + 1), float(index + 2)) for index in range(4)
            ),
            "audio_offsets": (0.0, 0.0),
            "sync_scores": (0.9, 0.8),
            "scene_cut_times": ((), ()),
        }
        evidence = _DissolveEvidence(
            face_identity_embeddings=((self.RAW_SENTINEL,),) * 3,
            pose_embeddings=((0.1,) * 12,) * 3,
            pose_backends=("test_pose",) * 3,
            pose_confidences=(0.9,) * 3,
        )

        def dissolve_for_path(**kwargs: object):  # type: ignore[no-untyped-def]
            optimized = kwargs["optimized"]
            assert isinstance(optimized, TimelineResult)
            return (
                {
                    item.candidate_id: evidence
                    for item in optimized.candidates
                },
                {
                    item.candidate_id: evidence
                    for item in optimized.candidates
                },
            )

        with patch("crosscut.visual_score._VideoSampler", Sampler), patch(
            "crosscut.visual_score.describe_frame",
            return_value=descriptor,
        ), patch(
            "crosscut.visual_score._select_safe_cadence_candidates",
            side_effect=alternatives,
        ), patch(
            "crosscut.visual_score._selected_dissolve_evidence",
            side_effect=dissolve_for_path,
        ) as dissolve:
            candidate_set = build_visual_timeline_candidate_set(
                **common_arguments,
                maximum_results=8,
            )
            dissolve.assert_not_called()
            runner_up_record = candidate_set.ranked_candidates[1]
            runner_up_plan = candidate_set.materialize(runner_up_record)
            selected_from_set = candidate_set.materialize()
            legacy_plan = build_visual_timeline(**common_arguments)

        self.assertEqual(maximum_requests, [8, 32])
        self.assertEqual(len(candidate_set.candidates), 8)
        for record, original in zip(
            candidate_set.candidates,
            captured_results[0],
            strict=True,
        ):
            self.assertIs(record.timeline_result, original)
        self.assertIsInstance(legacy_plan, VisualTimelinePlan)
        self.assertEqual(dissolve.call_count, 3)
        self.assertEqual(
            runner_up_plan.selected_candidate_index,
            runner_up_record.candidate_index,
        )
        self.assertEqual(
            [choice.source_index for choice in runner_up_plan.choices],
            [
                int(item.source_id)
                for item in runner_up_record.timeline_result.candidates
            ],
        )
        self.assertEqual(
            runner_up_plan.choices[0].face_match_candidate_failure,
            "face_not_detected",
        )
        self.assertEqual(
            runner_up_plan.choices[0].dissolve_entry_face_identity_window,
            ((self.RAW_SENTINEL,),) * 3,
        )
        self.assertEqual(
            sum(
                summary["selected"]
                for summary in runner_up_plan.candidate_review_summaries
            ),
            1,
        )
        self.assertEqual(
            legacy_plan.selected_candidate_index,
            candidate_set.selected_candidate_index,
        )
        self.assertEqual(selected_from_set, legacy_plan)
        self.assertEqual(
            legacy_plan.candidate_review_summaries,
            candidate_set.candidate_review_summaries,
        )
        self.assertNotIn(str(self.RAW_SENTINEL), repr(candidate_set.to_dict()))

    def test_candidate_builder_enforces_configured_eight_to_thirty_two_range(
        self,
    ) -> None:
        arguments = {
            "sources": (Path("source-0.mp4"), Path("source-1.mp4")),
            "canonical_spans": ((1.0, 2.0),),
            "audio_offsets": (0.0, 0.0),
            "sync_scores": (0.9, 0.8),
            "scene_cut_times": ((), ()),
        }
        for maximum in (7, 33, True):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(
                ValueError, "between 8 and 32"
            ):
                build_visual_timeline_candidate_set(
                    **arguments,
                    maximum_results=maximum,
                )


if __name__ == "__main__":
    unittest.main()
