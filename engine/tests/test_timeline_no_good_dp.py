from __future__ import annotations

import unittest
from unittest.mock import patch

from crosscut import timeline as timeline_module
from crosscut.timeline import (
    NoCompleteTimelinePath,
    TimelineCandidate,
    TimelineConfig,
    select_timeline,
    select_timeline_candidates,
)
from crosscut.timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
)


def _candidate(
    segment_index: int,
    source_id: str,
    shot_id: str,
    *,
    quality_score: float = 1.0,
) -> TimelineCandidate:
    return TimelineCandidate(
        segment_index=segment_index,
        source_id=source_id,
        start_seconds=float(segment_index),
        end_seconds=float(segment_index + 1),
        quality_score=quality_score,
        sync_score=1.0,
        composition_score=1.0,
        boundary_score=1.0,
        shot_id=shot_id,
    )


def _edge(
    *,
    boundary_index: int = 0,
    outgoing_source_id: str = "0",
    outgoing_shot_id: str = "0:shot-a",
    incoming_source_id: str = "1",
    incoming_shot_id: str = "1:shot-b",
) -> TimelineNoGoodEdgeKey:
    return TimelineNoGoodEdgeKey(
        boundary_index=boundary_index,
        outgoing_source_id=outgoing_source_id,
        outgoing_shot_id=outgoing_shot_id,
        incoming_source_id=incoming_source_id,
        incoming_shot_id=incoming_shot_id,
    )


class TimelineNoGoodDynamicProgrammingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outgoing = _candidate(0, "0", "0:shot-a")
        self.preferred = _candidate(1, "1", "1:shot-b")
        self.alternate = _candidate(
            1,
            "2",
            "2:shot-c",
            quality_score=0.75,
        )
        self.segments = (
            (self.outgoing,),
            (self.preferred, self.alternate),
        )
        self.config = TimelineConfig(
            continuity_bonus=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            source_switch_bonus=0.0,
        )

    def test_exact_forbidden_edge_is_removed_before_transition_scoring(self) -> None:
        baseline = select_timeline(self.segments, config=self.config)
        constrained = select_timeline(
            self.segments,
            config=self.config,
            forbidden_boundary_edges=TimelineNoGoodSet((_edge(),)),
        )

        self.assertEqual(baseline.candidates[-1].shot_id, "1:shot-b")
        self.assertEqual(constrained.candidates[-1].shot_id, "2:shot-c")

    def test_boundary_source_and_shot_specificity_do_not_overblock(self) -> None:
        graph_with_later_boundary = (
            *self.segments,
            (_candidate(2, "3", "3:shot-d"),),
        )
        variants = (
            _edge(boundary_index=1),
            _edge(outgoing_source_id="9"),
            _edge(outgoing_shot_id="0:shot-other"),
            _edge(incoming_source_id="9"),
            _edge(incoming_shot_id="1:shot-other"),
        )

        for forbidden in variants:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    select_timeline(
                        graph_with_later_boundary,
                        config=self.config,
                        forbidden_boundary_edges=(forbidden,),
                    )

        valid_but_different = TimelineNoGoodEdgeKey(
            boundary_index=1,
            outgoing_source_id="1",
            outgoing_shot_id="1:shot-b",
            incoming_source_id="3",
            incoming_shot_id="3:shot-d",
        )
        result = select_timeline(
            graph_with_later_boundary,
            config=self.config,
            forbidden_boundary_edges=(valid_but_different,),
        )
        self.assertEqual(result.candidates[1].shot_id, "2:shot-c")

    def test_all_top_n_paths_respect_the_same_no_good_set_deterministically(self) -> None:
        third = _candidate(1, "3", "3:shot-e", quality_score=0.65)
        segments = (self.segments[0], (*self.segments[1], third))
        no_goods = TimelineNoGoodSet((_edge(),))

        first = select_timeline_candidates(
            segments,
            config=self.config,
            maximum_results=3,
            minimum_hamming_fraction=0.0,
            forbidden_boundary_edges=no_goods,
        )
        second = select_timeline_candidates(
            segments,
            config=self.config,
            maximum_results=3,
            minimum_hamming_fraction=0.0,
            forbidden_boundary_edges=no_goods,
        )

        first_paths = tuple(
            tuple(item.shot_id for item in result.candidates)
            for result in first
        )
        second_paths = tuple(
            tuple(item.shot_id for item in result.candidates)
            for result in second
        )
        self.assertEqual(first_paths, second_paths)
        self.assertTrue(first_paths)
        self.assertTrue(
            all(path[1] != "1:shot-b" for path in first_paths)
        )

    def test_later_diversity_exhaustion_preserves_earlier_valid_path(self) -> None:
        first = _candidate(0, "a", "a:0")
        middle_a = _candidate(1, "a", "a:1")
        middle_b = _candidate(
            1,
            "b",
            "b:1",
            quality_score=0.6,
        )
        final = _candidate(2, "a", "a:2")
        dead_alternate = TimelineNoGoodEdgeKey(
            boundary_index=1,
            outgoing_source_id="b",
            outgoing_shot_id="b:1",
            incoming_source_id="a",
            incoming_shot_id="a:2",
        )
        config = TimelineConfig(
            continuity_bonus=0.0,
            cut_penalty=0.0,
            early_cut_penalty=0.0,
            source_switch_bonus=0.0,
            minimum_hold_segments=1,
            preferred_max_hold_segments=1,
            maximum_same_source_hold_segments=1,
            maximum_switch_score_sacrifice=0.2,
        )

        results = select_timeline_candidates(
            ((first,), (middle_a, middle_b), (final,)),
            config=config,
            maximum_results=4,
            diversity_penalty=1.0,
            minimum_hamming_fraction=0.0,
            forbidden_boundary_edges=(dead_alternate,),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            tuple(candidate.source_id for candidate in results[0].candidates),
            ("a", "a", "a"),
        )

    def test_no_complete_path_fails_closed_with_no_good_diagnostic(self) -> None:
        with self.assertRaisesRegex(
            NoCompleteTimelinePath,
            "exact_boundary_no_good:1",
        ) as caught:
            select_timeline(
                ((self.outgoing,), (self.preferred,)),
                config=self.config,
                forbidden_boundary_edges=(_edge(),),
            )
        self.assertEqual(
            caught.exception.reason_code,
            "cadence_state_graph_exhausted",
        )
        self.assertEqual(
            caught.exception.rejection_counts,
            {"exact_boundary_no_good": 1},
        )

    def test_top_n_compiles_no_good_graph_membership_once(self) -> None:
        with patch.object(
            timeline_module,
            "compile_timeline_no_good_constraint_index",
            wraps=timeline_module.compile_timeline_no_good_constraint_index,
        ) as compiler:
            select_timeline_candidates(
                self.segments,
                config=self.config,
                maximum_results=3,
                minimum_hamming_fraction=0.0,
                forbidden_boundary_edges=(_edge(),),
            )

        self.assertEqual(compiler.call_count, 1)

    def test_unknown_edge_is_rejected_instead_of_silently_ignored(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "does not belong to the current timeline graph",
        ):
            select_timeline(
                self.segments,
                config=self.config,
                forbidden_boundary_edges=(
                    _edge(incoming_shot_id="1:shot-stale"),
                ),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
