from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Barrier, Lock
import unittest

from crosscut.lazy_splice_feasibility_graph import (
    LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT,
    LazySpliceFeasibilityGraph,
    LazySpliceFeasibilityKey,
)
from crosscut.timeline_no_good import TimelineNoGoodEdgeKey, TimelineNoGoodSet
from crosscut.visual_splice_feasibility import (
    SpliceFeasibilityAssessment,
    SpliceFeasibilityStatus,
    SpliceOffsetAssessment,
    build_splice_feasibility_assessment,
)


def _edge(
    boundary_index: int = 4,
    outgoing_source_id: str = "camera-a",
    outgoing_shot_id: str = "camera-a:shot-2",
    incoming_source_id: str = "camera-b",
    incoming_shot_id: str = "camera-b:shot-7",
) -> TimelineNoGoodEdgeKey:
    return TimelineNoGoodEdgeKey(
        boundary_index=boundary_index,
        outgoing_source_id=outgoing_source_id,
        outgoing_shot_id=outgoing_shot_id,
        incoming_source_id=incoming_source_id,
        incoming_shot_id=incoming_shot_id,
    )


def _context(label: str = "context-a") -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _key(
    edge: TimelineNoGoodEdgeKey,
    label: str = "context-a",
) -> LazySpliceFeasibilityKey:
    return LazySpliceFeasibilityKey(edge, _context(label))


def _assessment(
    status: SpliceFeasibilityStatus,
) -> SpliceFeasibilityAssessment:
    if status is SpliceFeasibilityStatus.RAW_PASS:
        offsets = (SpliceOffsetAssessment(0, True, (), True),)
    elif status is SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED:
        offsets = (SpliceOffsetAssessment(0, False, ("focus",), True),)
    elif status is SpliceFeasibilityStatus.HARD_INCOMPATIBLE:
        offsets = (
            SpliceOffsetAssessment(0, False, ("silhouette",), True),
        )
    else:
        offsets = (
            SpliceOffsetAssessment(
                0,
                False,
                ("unexplained_production_gate_rejection",),
                False,
            ),
        )
    assessment = build_splice_feasibility_assessment(
        expected_frame_offsets=(0,),
        offsets=offsets,
    )
    if assessment.status is not status:
        raise AssertionError("test assessment fixture status mismatch")
    return assessment


def _partial_hard_assessment(
    offsets: range,
) -> SpliceFeasibilityAssessment:
    return build_splice_feasibility_assessment(
        expected_frame_offsets=tuple(range(-8, 9)),
        offsets=tuple(
            SpliceOffsetAssessment(offset, False, ("silhouette",), True)
            for offset in offsets
        ),
    )


class LazySpliceFeasibilityGraphTests(unittest.TestCase):
    def test_exact_edge_identity_keeps_direction_boundary_and_shots_specific(
        self,
    ) -> None:
        base = _edge()
        variants = (
            _edge(boundary_index=5),
            _edge(outgoing_shot_id="camera-a:shot-3"),
            _edge(incoming_shot_id="camera-b:shot-8"),
            _edge(
                outgoing_source_id="camera-b",
                outgoing_shot_id="camera-b:shot-7",
                incoming_source_id="camera-a",
                incoming_shot_id="camera-a:shot-2",
            ),
        )
        graph = LazySpliceFeasibilityGraph("basis-v1")
        result = _assessment(SpliceFeasibilityStatus.RAW_PASS)

        for edge in (base, *variants):
            graph.record_assessment(edge, _context(), result)

        self.assertEqual(len(graph), 5)
        self.assertEqual(
            tuple(entry.edge for entry in graph.entries),
            tuple(sorted((base, *variants))),
        )

    def test_get_or_assess_memoizes_and_calls_assessor_once(self) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        edge = _edge()
        expected = _assessment(SpliceFeasibilityStatus.RAW_PASS)
        calls: list[LazySpliceFeasibilityKey] = []

        def assessor(
            value: LazySpliceFeasibilityKey,
        ) -> SpliceFeasibilityAssessment:
            calls.append(value)
            return expected

        first = graph.get_or_assess(edge, _context(), assessor)
        second = graph.get_or_assess(edge, _context(), assessor)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        self.assertEqual(calls, [_key(edge)])
        self.assertIs(graph.assessment_for(edge, _context()), expected)

    def test_same_raw_edge_with_changed_span_context_is_reassessed(self) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        edge = _edge()
        passed = _assessment(SpliceFeasibilityStatus.RAW_PASS)
        calls: list[str] = []

        graph.get_or_assess(
            edge,
            _context("before-adjacent-repair"),
            lambda _edge_value: calls.append("before") or passed,
        )
        graph.get_or_assess(
            edge,
            _context("after-adjacent-repair"),
            lambda _edge_value: calls.append("after") or passed,
        )

        self.assertEqual(calls, ["before", "after"])
        self.assertEqual(len(graph), 2)

    def test_distinct_edge_assessors_overlap_but_same_edge_is_singleflight(
        self,
    ) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        first_edge = _edge(boundary_index=1)
        second_edge = _edge(boundary_index=2)
        rendezvous = Barrier(2, timeout=2.0)
        calls: list[LazySpliceFeasibilityKey] = []
        calls_lock = Lock()

        def distinct_assessor(
            key: LazySpliceFeasibilityKey,
        ) -> SpliceFeasibilityAssessment:
            with calls_lock:
                calls.append(key)
            rendezvous.wait()
            return _assessment(SpliceFeasibilityStatus.RAW_PASS)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(
                future.result(timeout=3.0)
                for future in (
                    executor.submit(
                        graph.get_or_assess,
                        first_edge,
                        _context("first"),
                        distinct_assessor,
                    ),
                    executor.submit(
                        graph.get_or_assess,
                        second_edge,
                        _context("second"),
                        distinct_assessor,
                    ),
                )
            )
        self.assertEqual(len(results), 2)
        self.assertCountEqual(
            calls,
            (
                _key(first_edge, "first"),
                _key(second_edge, "second"),
            ),
        )

        same_edge_graph = LazySpliceFeasibilityGraph("basis-v1")
        same_edge_calls = 0
        start = Barrier(2, timeout=2.0)

        def same_edge_assessor(
            key: LazySpliceFeasibilityKey,
        ) -> SpliceFeasibilityAssessment:
            nonlocal same_edge_calls
            same_edge_calls += 1
            return _assessment(SpliceFeasibilityStatus.RAW_PASS)

        def request_same_edge() -> SpliceFeasibilityAssessment:
            start.wait()
            return same_edge_graph.get_or_assess(
                first_edge,
                _context(),
                same_edge_assessor,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            same_results = tuple(
                future.result(timeout=3.0)
                for future in (
                    executor.submit(request_same_edge),
                    executor.submit(request_same_edge),
                )
            )
        self.assertEqual(same_edge_calls, 1)
        self.assertEqual(same_results[0], same_results[1])

    def test_recursive_same_edge_assessor_fails_instead_of_deadlocking(self) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        edge = _edge()

        def recursive_assessor(
            value: LazySpliceFeasibilityKey,
        ) -> SpliceFeasibilityAssessment:
            return graph.get_or_assess(
                value.edge,
                value.edge_context_identity,
                recursive_assessor,
            )

        with self.assertRaisesRegex(RuntimeError, "recursive same-edge"):
            graph.get_or_assess(edge, _context(), recursive_assessor)

    def test_conflicting_write_is_rejected_but_identical_write_is_idempotent(
        self,
    ) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        edge = _edge()
        passed = _assessment(SpliceFeasibilityStatus.RAW_PASS)
        hard = _assessment(SpliceFeasibilityStatus.HARD_INCOMPATIBLE)

        self.assertIs(graph.record_assessment(edge, _context(), passed), passed)
        self.assertIs(graph.record_assessment(edge, _context(), passed), passed)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            graph.record_assessment(edge, _context(), hard)
        self.assertIs(graph.assessment_for(edge, _context()), passed)

    def test_analysis_basis_change_invalidates_reuse_and_fingerprint(self) -> None:
        edge = _edge()
        result = _assessment(SpliceFeasibilityStatus.RAW_PASS)
        first = LazySpliceFeasibilityGraph(
            "basis-v1",
            ((_key(edge), result),),
        )

        same = first.for_analysis_basis("basis-v1")
        changed = first.for_analysis_basis("basis-v2")
        calls: list[LazySpliceFeasibilityKey] = []

        self.assertIs(same, first)
        self.assertIsNot(changed, first)
        self.assertEqual(changed.analysis_basis_identity, "basis-v2")
        self.assertEqual(len(changed), 0)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)

        changed.get_or_assess(
            edge,
            _context(),
            lambda value: calls.append(value) or result,
        )
        self.assertEqual(calls, [_key(edge)])
        self.assertEqual(first.entries, changed.entries)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_new_run_reopens_indeterminate_contexts(self) -> None:
        raw_edge = _edge(boundary_index=1)
        missing_edge = _edge(boundary_index=2)
        raw_key = _key(raw_edge, "raw-pass")
        missing_key = _key(missing_edge, "temporary-descriptor-miss")
        prior = LazySpliceFeasibilityGraph(
            "basis-v1",
            (
                (
                    raw_key,
                    _assessment(SpliceFeasibilityStatus.RAW_PASS),
                ),
                (
                    missing_key,
                    _assessment(SpliceFeasibilityStatus.INDETERMINATE),
                ),
            ),
        )

        resumed = prior.for_new_run()

        self.assertIsNot(resumed, prior)
        self.assertFalse(
            resumed.needs_assessment(
                raw_edge,
                raw_key.edge_context_identity,
            )
        )
        self.assertTrue(
            resumed.needs_assessment(
                missing_edge,
                missing_key.edge_context_identity,
            )
        )
        self.assertEqual(len(resumed), 1)

    def test_only_hard_incompatible_edges_become_optimizer_no_goods(self) -> None:
        raw_edge = _edge(boundary_index=0)
        geometry_edge = _edge(boundary_index=1)
        hard_edge = _edge(boundary_index=2)
        indeterminate_edge = _edge(boundary_index=3)
        graph = LazySpliceFeasibilityGraph(
            "basis-v1",
            (
                (
                    _key(raw_edge, "raw"),
                    _assessment(SpliceFeasibilityStatus.RAW_PASS),
                ),
                (
                    _key(geometry_edge, "geometry"),
                    _assessment(
                        SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED
                    ),
                ),
                (
                    _key(hard_edge, "hard"),
                    _assessment(SpliceFeasibilityStatus.HARD_INCOMPATIBLE),
                ),
                (
                    _key(indeterminate_edge, "indeterminate"),
                    _assessment(SpliceFeasibilityStatus.INDETERMINATE),
                ),
            ),
        )

        self.assertEqual(
            graph.hard_no_goods_for_contexts((_key(hard_edge, "hard"),)),
            TimelineNoGoodSet((hard_edge,)),
        )
        self.assertEqual(
            graph.blocked_set_for_contexts(
                (
                    _key(hard_edge, "hard"),
                    _key(indeterminate_edge, "indeterminate"),
                )
            ),
            TimelineNoGoodSet((hard_edge, indeterminate_edge)),
        )
        self.assertEqual(graph.hard_incompatible_edges, (hard_edge,))
        self.assertEqual(graph.geometry_proof_required_edges, (geometry_edge,))
        self.assertEqual(graph.indeterminate_edges, (indeterminate_edge,))
        self.assertEqual(graph.raw_pass_edges, (raw_edge,))

    def test_transient_hard_context_cannot_block_unknown_raw_context(self) -> None:
        edge = _edge(boundary_index=6)
        transient_key = _key(edge, "after-adjacent-repair")
        raw_key = _key(edge, "raw-optimizer-context")
        graph = LazySpliceFeasibilityGraph(
            "basis-v1",
            (
                (
                    transient_key,
                    _assessment(SpliceFeasibilityStatus.HARD_INCOMPATIBLE),
                ),
            ),
        )

        self.assertEqual(
            graph.hard_no_goods_for_contexts((raw_key,)),
            TimelineNoGoodSet(),
        )
        self.assertEqual(
            graph.blocked_set_for_contexts((raw_key,)),
            TimelineNoGoodSet(),
        )
        self.assertEqual(
            graph.hard_no_goods_for_contexts((transient_key,)),
            TimelineNoGoodSet((edge,)),
        )

    def test_partial_indeterminate_refines_once_to_wide_pass_or_hard(self) -> None:
        edge = _edge()
        partial = _partial_hard_assessment(range(-3, 4))
        graph = LazySpliceFeasibilityGraph(
            "basis-v1",
            ((_key(edge), partial),),
        )
        self.assertTrue(graph.needs_assessment(edge, _context()))
        self.assertNotIn(
            edge,
            graph.blocked_set_for_contexts((_key(edge),)),
        )

        wide_offsets = tuple(
            SpliceOffsetAssessment(
                offset,
                offset == 7,
                () if offset == 7 else ("silhouette",),
                True,
            )
            for offset in range(-8, 9)
        )
        wide_pass = build_splice_feasibility_assessment(
            expected_frame_offsets=tuple(range(-8, 9)),
            offsets=wide_offsets,
        )
        calls = 0

        def pass_assessor(
            _key_value: LazySpliceFeasibilityKey,
        ) -> SpliceFeasibilityAssessment:
            nonlocal calls
            calls += 1
            return wide_pass

        refined = graph.get_or_assess(edge, _context(), pass_assessor)
        self.assertIs(refined.status, SpliceFeasibilityStatus.RAW_PASS)
        self.assertFalse(graph.needs_assessment(edge, _context()))
        self.assertNotIn(
            edge,
            graph.blocked_set_for_contexts((_key(edge),)),
        )
        graph.get_or_assess(edge, _context(), pass_assessor)
        self.assertEqual(calls, 1)

        hard_edge = _edge(boundary_index=8)
        hard_graph = LazySpliceFeasibilityGraph(
            "basis-v1",
            ((_key(hard_edge), partial),),
        )
        wide_hard = _partial_hard_assessment(range(-8, 9))
        hard_graph.get_or_assess(
            hard_edge,
            _context(),
            lambda _edge_value: wide_hard,
        )
        self.assertIn(
            hard_edge,
            hard_graph.hard_no_goods_for_contexts((_key(hard_edge),)),
        )
        self.assertFalse(hard_graph.needs_assessment(hard_edge, _context()))

    def test_serialization_and_fingerprint_ignore_insertion_order(self) -> None:
        entries = (
            (
                _key(_edge(boundary_index=9), "nine"),
                _assessment(SpliceFeasibilityStatus.INDETERMINATE),
            ),
            (
                _key(_edge(boundary_index=2), "two"),
                _assessment(SpliceFeasibilityStatus.HARD_INCOMPATIBLE),
            ),
            (
                _key(_edge(boundary_index=5), "five"),
                _assessment(
                    SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED
                ),
            ),
        )
        forward = LazySpliceFeasibilityGraph("basis-v1", entries)
        reverse = LazySpliceFeasibilityGraph(
            "basis-v1",
            tuple(reversed(entries)),
        )

        self.assertEqual(forward.fingerprint, reverse.fingerprint)
        self.assertEqual(forward.to_dict(), reverse.to_dict())
        self.assertEqual(
            json.dumps(
                forward.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            json.dumps(
                reverse.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        self.assertEqual(
            forward.to_dict()["contract"],
            LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT,
        )
        self.assertEqual(forward.to_dict()["assessment_count"], 3)
        self.assertEqual(
            LazySpliceFeasibilityGraph.from_dict(forward.to_dict()).to_dict(),
            forward.to_dict(),
        )
        tampered = forward.to_dict()
        tampered["fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            LazySpliceFeasibilityGraph.from_dict(tampered)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
