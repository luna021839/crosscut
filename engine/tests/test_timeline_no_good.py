from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest

from crosscut.timeline_no_good import (
    TIMELINE_NO_GOOD_CONTRACT,
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
    extract_timeline_no_good_edge_keys,
)


def _edge(
    boundary_index: int = 4,
    outgoing_source_id: str = "0",
    outgoing_shot_id: str = "0:shot-2",
    incoming_source_id: str = "5",
    incoming_shot_id: str = "5:shot-7",
) -> TimelineNoGoodEdgeKey:
    return TimelineNoGoodEdgeKey(
        boundary_index=boundary_index,
        outgoing_source_id=outgoing_source_id,
        outgoing_shot_id=outgoing_shot_id,
        incoming_source_id=incoming_source_id,
        incoming_shot_id=incoming_shot_id,
    )


class TimelineNoGoodEdgeKeyTests(unittest.TestCase):
    def test_key_is_frozen_orderable_and_exactly_specific(self) -> None:
        base = _edge()
        variants = (
            _edge(boundary_index=5),
            _edge(outgoing_source_id="1"),
            _edge(outgoing_shot_id="0:shot-3"),
            _edge(incoming_source_id="6"),
            _edge(incoming_shot_id="5:shot-8"),
        )

        self.assertEqual(len({base, *variants}), 6)
        self.assertEqual(tuple(sorted((variants[0], base)))[0], base)
        with self.assertRaises(FrozenInstanceError):
            base.boundary_index = 99  # type: ignore[misc]

    def test_invalid_boundary_and_identifiers_fail_strictly(self) -> None:
        for invalid_boundary in (True, -1, 1.0, "1", None):
            with self.subTest(boundary=invalid_boundary):
                with self.assertRaises((TypeError, ValueError)):
                    _edge(boundary_index=invalid_boundary)  # type: ignore[arg-type]

        identifier_fields = (
            "outgoing_source_id",
            "outgoing_shot_id",
            "incoming_source_id",
            "incoming_shot_id",
        )
        for field_name in identifier_fields:
            for invalid_value in ("", "   ", " padded", "padded ", 3, None):
                values = {field_name: invalid_value}
                with self.subTest(field=field_name, value=invalid_value):
                    with self.assertRaises((TypeError, ValueError)):
                        TimelineNoGoodEdgeKey(
                            boundary_index=0,
                            outgoing_source_id=values.get(
                                "outgoing_source_id",
                                "0",
                            ),
                            outgoing_shot_id=values.get(
                                "outgoing_shot_id",
                                "0:shot-0",
                            ),
                            incoming_source_id=values.get(
                                "incoming_source_id",
                                "1",
                            ),
                            incoming_shot_id=values.get(
                                "incoming_shot_id",
                                "1:shot-0",
                            ),
                        )

    def test_builder_supports_candidate_visual_and_wrapper_shapes(self) -> None:
        timeline_candidate = SimpleNamespace(
            source_id="camera-a",
            shot_id="camera-a:shot-2",
        )
        visual_choice = SimpleNamespace(
            source_index=7,
            shot_id="7:shot-9",
        )
        wrapped_choice = SimpleNamespace(candidate=timeline_candidate)

        built = build_timeline_no_good_edge_key(
            11,
            wrapped_choice,
            visual_choice,
        )

        self.assertEqual(
            built,
            TimelineNoGoodEdgeKey(
                boundary_index=11,
                outgoing_source_id="camera-a",
                outgoing_shot_id="camera-a:shot-2",
                incoming_source_id="7",
                incoming_shot_id="7:shot-9",
            ),
        )
        mapping_built = build_timeline_no_good_edge_key(
            2,
            {"source_index": 1, "shot_id": "1:shot-1"},
            {
                "candidate": {
                    "source_id": "broadcast-b",
                    "shot_id": "broadcast-b:shot-4",
                }
            },
        )
        self.assertEqual(mapping_built.outgoing_source_id, "1")
        self.assertEqual(mapping_built.incoming_source_id, "broadcast-b")

    def test_builder_rejects_missing_or_disagreeing_identity(self) -> None:
        valid = SimpleNamespace(source_index=0, shot_id="0:shot-0")
        invalid_values = (
            SimpleNamespace(shot_id="missing-source"),
            SimpleNamespace(source_index=1),
            SimpleNamespace(source_index=True, shot_id="bad-index"),
            SimpleNamespace(
                source_id="2",
                source_index=3,
                shot_id="contradiction",
            ),
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    build_timeline_no_good_edge_key(0, invalid, valid)


class TimelineNoGoodSetTests(unittest.TestCase):
    def test_collection_is_sorted_unique_and_fingerprint_is_canonical(self) -> None:
        edges = (_edge(boundary_index=8), _edge(boundary_index=2))
        forward = TimelineNoGoodSet((edges[0], edges[1], edges[0]))
        reverse = TimelineNoGoodSet(tuple(reversed(edges)))

        self.assertEqual(forward, reverse)
        self.assertEqual(
            tuple(edge.boundary_index for edge in forward.edges),
            (2, 8),
        )
        self.assertEqual(forward.fingerprint, reverse.fingerprint)
        self.assertEqual(len(forward.fingerprint), 64)
        self.assertEqual(len(forward), 2)
        self.assertEqual(tuple(forward), forward.edges)
        self.assertIn(edges[0], forward)
        self.assertNotIn(_edge(boundary_index=3), forward)
        self.assertNotIn(object(), forward)
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
            TIMELINE_NO_GOOD_CONTRACT,
        )
        self.assertEqual(forward.to_dict()["edge_count"], 2)
        self.assertEqual(TimelineNoGoodSet.from_dict(forward.to_dict()), forward)

        tampered = forward.to_dict()
        tampered["fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            TimelineNoGoodSet.from_dict(tampered)

    def test_collection_rejects_non_key_values(self) -> None:
        for invalid in ("not-edges", (object(),), None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    TimelineNoGoodSet(invalid)  # type: ignore[arg-type]

    def test_plan_boundary_extraction_is_exact_and_deterministic(self) -> None:
        choices = tuple(
            SimpleNamespace(
                source_index=source_index,
                shot_id=f"{source_index}:shot-{shot_number}",
            )
            for source_index, shot_number in ((0, 1), (5, 7), (5, 8), (2, 4))
        )
        plan = SimpleNamespace(choices=choices)

        all_edges = extract_timeline_no_good_edge_keys(plan)
        selected = extract_timeline_no_good_edge_keys(
            {"choices": choices},
            boundary_indices=(2, 0, 2),
        )
        no_goods = TimelineNoGoodSet.from_plan_boundaries(
            plan,
            boundary_indices=(2, 0),
        )

        self.assertEqual(
            tuple(edge.boundary_index for edge in all_edges),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(edge.boundary_index for edge in selected),
            (0, 2),
        )
        self.assertEqual(no_goods.edges, selected)
        self.assertEqual(selected[0].outgoing_shot_id, "0:shot-1")
        self.assertEqual(selected[0].incoming_shot_id, "5:shot-7")
        self.assertEqual(selected[1].outgoing_shot_id, "5:shot-8")
        self.assertEqual(selected[1].incoming_shot_id, "2:shot-4")

    def test_plan_boundary_extraction_rejects_invalid_indices(self) -> None:
        choices = (
            SimpleNamespace(source_index=0, shot_id="0:shot-0"),
            SimpleNamespace(source_index=1, shot_id="1:shot-0"),
        )
        for invalid in ((1,), (-1,), (True,), (0.0,), "0", 0):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    extract_timeline_no_good_edge_keys(
                        choices,
                        boundary_indices=invalid,  # type: ignore[arg-type]
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
