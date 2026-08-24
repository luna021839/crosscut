from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    ReferenceAttributionConfig,
)
from crosscut.reference_dataset import (
    SCHEMA_VERSION,
    build_reference_dataset,
    build_reference_dataset_file,
)


def _shot(index: int, source: str | None, *, duration: float = 1.25) -> dict:
    start = index * duration
    accepted = source is not None
    return {
        "shot_index": index,
        "reference_start_seconds": start,
        "reference_end_seconds": start + duration,
        "accepted": accepted,
        "attributed_video_id": source,
        "decision_reason": "match" if accepted else "low_confidence",
        "top_two_margin": 0.4 if accepted else 0.01,
        "best": {
            "video_id": source or "guess",
            "score": 0.9 if accepted else 0.2,
            "source_time_seconds": start + 2.0,
            "orb_homography_inliers": 40,
        },
        "runner_up": {
            "video_id": "other",
            "score": 0.3,
            "source_time_seconds": start + 1.0,
        },
    }


def _resolved_boundary(old_source: str = "A", new_source: str = "B") -> dict:
    return {
        "old_source_id": old_source,
        "new_source_id": new_source,
        "nominal_boundary_seconds": 1.25,
        "nominal_frame_index": 25,
        "refinement_radius_seconds": 0.5,
        "status": "resolved_hard_cut",
        "selected_boundary_seconds": 1.35,
        "selected_frame_index": 27,
        "frame_offset": 2,
        "objective_score": 5.8,
        "old_identity_frames": 4,
        "new_identity_frames": 3,
        "uncertain_middle_frames": 0,
        "frame_identities": ["old", "old", "old", "old", "new", "new", "new"],
        "evidence_frame_indices": [23, 24, 25, 26, 27, 28, 29],
        "evidence_master_seconds": [1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45],
        "to_run_index": 1,
    }


def _trusted_attribution(shots: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "analysis": REFERENCE_ATTRIBUTION_BASIS,
        "decision_policy": ReferenceAttributionConfig().to_dict(),
        "shots": shots,
    }


class ReferenceDatasetTests(unittest.TestCase):
    def test_first_known_shot_after_leading_unknown_is_observed_reference_start(self) -> None:
        attribution = {
            "analysis": "test",
            "reference": "/tmp/reference.mp4",
            "shots": [
                _shot(0, None),
                _shot(1, None),
                _shot(2, "A"),
                _shot(3, "A"),
            ],
        }

        dataset = build_reference_dataset(attribution, style_id="fan_reference")

        self.assertEqual([0, 1], dataset.payload["unresolved_shot_indices"])
        self.assertEqual(2, dataset.record_count)
        first = dataset.payload["records"][0]
        self.assertEqual(2, first["shot_index"])
        self.assertEqual("reference_start", first["boundary_label"])
        self.assertEqual(
            "reference_start", first["boundary_event"]["event_type"]
        )
        self.assertIsNone(first["previous_known_source"])

    def test_unknown_shots_are_not_promoted_to_ground_truth(self) -> None:
        attribution = {
            "analysis": "test",
            "reference": "/tmp/reference.mp4",
            "shots": [
                _shot(0, "A"),
                _shot(1, "A"),
                _shot(2, None),
                _shot(3, "B"),
                _shot(4, "C"),
            ],
        }
        dataset = build_reference_dataset(attribution, style_id="fan_reference")

        self.assertEqual(dataset.record_count, 4)
        self.assertEqual(dataset.payload["unresolved_shot_indices"], [2])
        profile = dataset.payload["style_profile"]
        self.assertEqual(profile["comparable_adjacent_edges"], 2)
        self.assertEqual(profile["observed_cross_source_switches"], 1)
        self.assertEqual(profile["cross_source_hard_cuts"], 0)
        self.assertEqual(profile["unresolved_cross_source_transitions"], 1)
        self.assertEqual(profile["same_source_internal_camera_cuts"], 1)
        self.assertEqual(profile["unique_attributed_sources"], 3)
        self.assertEqual(
            dataset.payload["records"][1]["boundary_label"],
            "same_source_internal_camera_cut",
        )
        self.assertEqual(
            dataset.payload["records"][2]["boundary_label"],
            "known_after_unknown",
        )
        unresolved = dataset.payload["records"][3]
        self.assertEqual(
            unresolved["boundary_label"],
            "cross_source_transition_unresolved",
        )
        self.assertTrue(unresolved["is_source_switch"])
        self.assertIsNone(unresolved["transition"])
        self.assertFalse(unresolved["transition_supervision_eligible"])
        self.assertEqual(
            dataset.payload["unresolved_transition_edges"][0]["status"],
            "not_refined",
        )

    def test_only_resolved_frame_boundary_becomes_a_hard_cut_label(self) -> None:
        first = _shot(0, "A")
        second = _shot(1, "B")
        second["boundary_refinement"] = _resolved_boundary()

        dataset = build_reference_dataset(
            _trusted_attribution([first, second]),
            style_id="verified_editor",
        )

        self.assertEqual(dataset.payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 4)
        profile = dataset.payload["style_profile"]
        self.assertEqual(profile["observed_cross_source_switches"], 1)
        self.assertEqual(profile["cross_source_hard_cuts"], 1)
        self.assertEqual(profile["unresolved_cross_source_transitions"], 0)
        record = dataset.payload["records"][1]
        self.assertEqual(record["boundary_label"], "cross_source_hard_cut")
        self.assertEqual(record["transition"], "hard_cut")
        self.assertTrue(record["transition_supervision_eligible"])
        self.assertTrue(record["source_switch_hard_cut_confirmed"])
        self.assertEqual(
            record["boundary_event"]["event_type"],
            "editorial_source_switch_hard_cut",
        )
        self.assertTrue(
            record["boundary_event"][
                "editorial_timing_supervision_eligible"
            ]
        )
        self.assertEqual(record["nominal_cut_master_time"], 1.25)
        self.assertEqual(record["cut_master_time"], 1.35)
        self.assertEqual(dataset.payload["records"][0]["end_master_time"], 1.35)

    def test_native_camera_cut_is_explicitly_not_editorial_supervision(self) -> None:
        dataset = build_reference_dataset(
            {"shots": [_shot(0, "A"), _shot(1, "A")]},
            style_id="native-cut",
        )
        record = dataset.payload["records"][1]
        event = record["boundary_event"]
        self.assertEqual(event["event_type"], "native_source_camera_cut")
        self.assertTrue(event["is_native_source_camera_cut"])
        self.assertFalse(event["is_editorial_source_switch"])
        self.assertFalse(event["editorial_timing_supervision_eligible"])
        self.assertFalse(record["transition_supervision_eligible"])
        self.assertIsNone(record["transition"])

    def test_unresolved_refinement_preserves_nominal_span_and_label(self) -> None:
        first = _shot(0, "A")
        second = _shot(1, "B")
        second["boundary_refinement"] = {
            "old_source_id": "A",
            "new_source_id": "B",
            "status": "unresolved_blend_or_unknown",
            "selected_boundary_seconds": None,
        }

        dataset = build_reference_dataset(
            {"shots": [first, second]},
            style_id="unresolved_editor",
        )
        record = dataset.payload["records"][1]
        self.assertEqual(record["cut_master_time"], 1.25)
        self.assertEqual(
            record["boundary_label"],
            "cross_source_transition_unresolved",
        )
        self.assertEqual(
            dataset.payload["unresolved_transition_edges"][0]["status"],
            "unresolved_blend_or_unknown",
        )

    def test_builds_pairwise_ranking_examples(self) -> None:
        dataset = build_reference_dataset(
            {"shots": [_shot(0, "A")]},
            style_id="editor_a",
            reference_id="ive_i_am",
        )
        record = dataset.payload["records"][0]
        self.assertEqual(record["selected_source"], "A")
        self.assertEqual(record["negative_examples"][0]["source_id"], "other")
        self.assertAlmostEqual(
            record["ranking_pairs"][0]["observed_score_margin"],
            0.6,
        )

    def test_file_builder_writes_reusable_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "attribution.json"
            output = root / "dataset.json"
            source.write_text(
                json.dumps({"shots": [_shot(0, "A")]}),
                encoding="utf-8",
            )
            resolved = build_reference_dataset_file(
                source,
                output,
                style_id="private_reference",
            )
            self.assertEqual(resolved, output.resolve())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(
                saved["label_policy"]["unknown_shots_excluded_from_supervised_records"]
            )

    def test_rejects_empty_style_and_invalid_spans(self) -> None:
        with self.assertRaisesRegex(ValueError, "style_id"):
            build_reference_dataset({"shots": [_shot(0, "A")]}, style_id=" ")
        broken = _shot(0, "A")
        broken["reference_end_seconds"] = broken["reference_start_seconds"]
        with self.assertRaisesRegex(ValueError, "positive duration"):
            build_reference_dataset({"shots": [broken]}, style_id="style")

    def test_rejects_mismatched_resolved_boundary_sources(self) -> None:
        first = _shot(0, "A")
        second = _shot(1, "B")
        second["boundary_refinement"] = _resolved_boundary(old_source="wrong")
        with self.assertRaisesRegex(ValueError, "old_source_id"):
            build_reference_dataset(
                _trusted_attribution([first, second]),
                style_id="style",
            )

    def test_rejects_status_only_hard_cut_without_resolver_provenance(self) -> None:
        first = _shot(0, "A")
        second = _shot(1, "B")
        second["boundary_refinement"] = _resolved_boundary()
        with self.assertRaisesRegex(ValueError, "resolver schema"):
            build_reference_dataset(
                {"shots": [first, second]},
                style_id="untrusted",
            )

    def test_rejects_tampered_resolved_boundary_proof(self) -> None:
        first = _shot(0, "A")
        second = _shot(1, "B")
        proof = _resolved_boundary()
        proof["old_identity_frames"] = 3
        second["boundary_refinement"] = proof
        with self.assertRaisesRegex(ValueError, "counts"):
            build_reference_dataset(
                _trusted_attribution([first, second]),
                style_id="tampered",
            )


if __name__ == "__main__":
    unittest.main()
