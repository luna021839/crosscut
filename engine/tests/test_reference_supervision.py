from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_supervision import (
    SCHEMA_VERSION,
    build_reference_supervision,
    build_reference_supervision_file,
)


def _candidate(source_id: str, score: float, **extra: object) -> dict:
    return {
        "video_id": source_id,
        "score": score,
        "orb_good_matches": 40,
        "orb_homography_inliers": 28,
        "orb_inlier_ratio": 0.7,
        "structure_ncc": 0.8,
        "color_similarity": 0.6,
        "source_time_seconds": 12.0,
        **extra,
    }


def _shot(
    index: int,
    selected: str | None,
    candidates: list[dict],
) -> dict:
    start = index * 1.5
    return {
        "shot_index": index,
        "reference_start_seconds": start,
        "reference_end_seconds": start + 1.5,
        "accepted": selected is not None,
        "attributed_video_id": selected,
        "decision_reason": "temporal_path" if selected else "low_confidence",
        "top_two_margin": 0.2 if selected else 0.01,
        "best": candidates[0] if candidates else None,
        "runner_up": candidates[1] if len(candidates) > 1 else None,
        "ranked_candidates": candidates,
    }


class ReferenceSupervisionTests(unittest.TestCase):
    def test_builds_listwise_group_and_all_selected_vs_rejected_pairs(self) -> None:
        # Temporal continuity can legitimately select A even if local visual
        # score B is higher.  The training label must follow the recovered edit.
        shot = _shot(
            0,
            "A",
            [
                _candidate("B", 0.70),
                _candidate("A", 0.61),
                _candidate("C", 0.40),
            ],
        )

        dataset = build_reference_supervision(
            {"shots": [shot]},
            style_id="ive_reference",
            reference_id="stage_mix_1",
        )

        self.assertEqual(dataset.payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 3)
        self.assertEqual(dataset.listwise_count, 1)
        self.assertEqual(dataset.pairwise_count, 2)
        group = dataset.payload["listwise_records"][0]
        self.assertEqual(group["chosen_source_id"], "A")
        self.assertEqual(group["candidate_count"], 3)
        self.assertEqual(
            [candidate["source_id"] for candidate in group["candidates"]],
            ["B", "A", "C"],
        )
        chosen = next(
            candidate for candidate in group["candidates"] if candidate["selected"]
        )
        self.assertEqual(chosen["observed_match_rank"], 1)
        pairs = dataset.payload["pairwise_records"]
        self.assertEqual(
            {pair["rejected"]["source_id"] for pair in pairs},
            {"B", "C"},
        )
        margins = {
            pair["rejected"]["source_id"]: pair["observed_score_margin"]
            for pair in pairs
        }
        self.assertAlmostEqual(margins["B"], -0.09)
        self.assertAlmostEqual(margins["C"], 0.21)

    def test_unknown_shots_never_become_source_ranking_labels(self) -> None:
        known = _shot(
            0,
            "A",
            [_candidate("A", 0.9), _candidate("B", 0.2)],
        )
        unknown = _shot(
            1,
            None,
            [_candidate("B", 0.3), _candidate("A", 0.29)],
        )

        dataset = build_reference_supervision(
            {"shots": [known, unknown]},
            style_id="ive_reference",
        )

        self.assertEqual(len(dataset.payload["shot_labels"]), 1)
        self.assertEqual(dataset.listwise_count, 1)
        self.assertTrue(
            dataset.payload["label_policy"][
                "unknown_shots_excluded_from_source_supervision"
            ]
        )

    def test_keeps_label_record_when_no_observed_negative_exists(self) -> None:
        shot = _shot(0, "A", [_candidate("A", 0.9)])
        dataset = build_reference_supervision(
            {"shots": [shot]},
            style_id="ive_reference",
        )

        self.assertEqual(len(dataset.payload["shot_labels"]), 1)
        self.assertEqual(dataset.listwise_count, 0)
        self.assertEqual(dataset.pairwise_count, 0)
        self.assertEqual(
            dataset.payload["statistics"][
                "insufficient_candidate_shot_indices"
            ],
            [0],
        )

    def test_emits_explicit_crop_target_and_separate_homography_evidence(self) -> None:
        crop = {"scale": 1.08, "center_x": 0.48, "center_y": 0.44}
        homography = [
            [1.0, 0.0, 8.0],
            [0.0, 1.0, -3.0],
            [0.0, 0.0, 1.0],
        ]
        shot = _shot(
            0,
            "A",
            [
                _candidate(
                    "A",
                    0.9,
                    crop_transform=crop,
                    homography_reference_to_source=homography,
                ),
                _candidate("B", 0.3),
            ],
        )

        dataset = build_reference_supervision(
            {"shots": [shot]},
            style_id="ive_reference",
        )
        labels = dataset.payload["shot_labels"][0]["edit_labels"]

        self.assertTrue(labels["crop"]["supervision_eligible"])
        self.assertEqual(labels["crop"]["transform"], crop)
        self.assertTrue(labels["geometry_evidence"]["available"])
        self.assertFalse(labels["geometry_evidence"]["is_direct_crop_target"])
        self.assertEqual(
            dataset.payload["statistics"]["crop_supervision_count"],
            1,
        )
        self.assertEqual(
            dataset.payload["statistics"]["geometry_evidence_count"],
            1,
        )

    def test_unresolved_cross_source_transition_stays_masked(self) -> None:
        first = _shot(
            0,
            "A",
            [_candidate("A", 0.9), _candidate("B", 0.2)],
        )
        second = _shot(
            1,
            "B",
            [_candidate("B", 0.85), _candidate("A", 0.3)],
        )

        dataset = build_reference_supervision(
            {"shots": [first, second]},
            style_id="ive_reference",
        )
        incoming = dataset.payload["shot_labels"][1]["edit_labels"]

        self.assertEqual(
            incoming["cut"]["boundary_label"],
            "cross_source_transition_unresolved",
        )
        self.assertIsNone(incoming["source_transition"]["transition"])
        self.assertFalse(
            incoming["source_transition"]["supervision_eligible"]
        )
        self.assertEqual(
            dataset.payload["statistics"]["transition_supervision_count"],
            0,
        )

    def test_native_source_cut_event_cannot_be_editorial_switch_label(self) -> None:
        first = _shot(
            0,
            "A",
            [_candidate("A", 0.9), _candidate("B", 0.2)],
        )
        second = _shot(
            1,
            "A",
            [_candidate("A", 0.88), _candidate("B", 0.25)],
        )
        dataset = build_reference_supervision(
            {"shots": [first, second]}, style_id="ive_reference"
        )
        incoming = dataset.payload["shot_labels"][1]["edit_labels"]
        event = incoming["boundary_event"]
        self.assertEqual(event["event_type"], "native_source_camera_cut")
        self.assertFalse(event["is_editorial_source_switch"])
        self.assertFalse(event["editorial_timing_supervision_eligible"])
        self.assertEqual(
            dataset.payload["statistics"]["native_source_camera_cut_count"],
            1,
        )
        self.assertEqual(
            dataset.payload["statistics"][
                "editorial_source_switch_supervision_count"
            ],
            0,
        )

    def test_file_builder_writes_reusable_json(self) -> None:
        shot = _shot(
            0,
            "A",
            [_candidate("A", 0.9), _candidate("B", 0.2)],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "attribution.json"
            output = root / "supervision.json"
            source.write_text(json.dumps({"shots": [shot]}), encoding="utf-8")

            resolved = build_reference_supervision_file(
                source,
                output,
                style_id="ive_reference",
            )

            self.assertEqual(resolved, output.resolve())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["statistics"]["pairwise_record_count"], 1)


if __name__ == "__main__":
    unittest.main()
