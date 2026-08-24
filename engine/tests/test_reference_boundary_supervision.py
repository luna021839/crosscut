from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crosscut_cli
from crosscut.reference_boundary_supervision import (
    ANALYSIS_BASIS,
    augment_reference_boundary_requests,
    augment_reference_boundary_requests_file,
)
from crosscut.reference_dataset import build_reference_dataset
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.reference_supervision import build_reference_supervision


def _candidate(
    source_id: str,
    score: float,
    *,
    inliers: int = 32,
    inlier_ratio: float = 0.70,
) -> dict:
    return {
        "video_id": source_id,
        "score": score,
        "orb_good_matches": 44,
        "orb_homography_inliers": inliers,
        "orb_inlier_ratio": inlier_ratio,
        "structure_ncc": 0.8,
        "color_similarity": 0.7,
        "source_frame_index": 100,
        "source_time_seconds": 1.0,
    }


def _sample(frame_index: int, selected: str) -> dict:
    dominant = selected
    other = "B" if selected == "A" else "A"
    ranked = [
        _candidate(dominant, 0.91),
        _candidate(other, 0.22),
    ]
    return {
        "master_seconds": frame_index / 10.0,
        "reference_frame_index": frame_index,
        "sample_reasons": [
            f"hard_cut_boundary:10:{frame_index - 10:+d}"
        ],
        "ranked_candidates": ranked,
        "best": deepcopy(ranked[0]),
        "runner_up": deepcopy(ranked[1]),
    }


def _evidence(states: tuple[str, ...] = ("A", "A", "B", "B", "B")) -> dict:
    return {
        "schema_version": 1,
        "analysis": "reference_hard_cut_reverse_match_evidence_v1",
        "reference": "/tmp/reference.mp4",
        "master_start_seconds": 0.0,
        "master_end_seconds": 1.3,
        "sources": [
            {"video_id": "A", "path": "/tmp/a.mp4"},
            {"video_id": "B", "path": "/tmp/b.mp4"},
        ],
        "clock_policy": {
            "reference_mapping": ReferenceSourceClock.scalar(
                0.0,
                basis="synthetic_reference_identity_v1",
            ).to_dict(),
        },
        "reference_video": {
            "path": "/tmp/reference.mp4",
            "frame_rate": 10.0,
            "declared_frame_count": 13,
            "width": 320,
            "height": 180,
        },
        "detected_reference_cuts": [
            {
                "frame_index": 10,
                "master_seconds": 1.0,
                "score": 0.9,
            }
        ],
        "samples": [
            _sample(frame_index, state)
            for frame_index, state in zip(range(8, 13), states, strict=True)
        ],
        "boundary_requests": [],
        "statistics": {"hard_cut_count": 1, "sample_count": 5},
    }


class ReferenceBoundarySupervisionTests(unittest.TestCase):
    def test_strict_switch_resolves_and_becomes_real_edit_supervision(self) -> None:
        evidence = _evidence()
        original = deepcopy(evidence)

        augmentation = augment_reference_boundary_requests(evidence)

        self.assertEqual(evidence, original)
        self.assertEqual(augmentation.added_request_count, 1)
        self.assertEqual(augmentation.added_resolved_hard_cut_count, 1)
        self.assertEqual(augmentation.added_to_run_indices, (1,))
        self.assertEqual(
            augmentation.payload["boundary_request_augmentation"]["analysis"],
            ANALYSIS_BASIS,
        )
        request = augmentation.payload["boundary_requests"][0]
        self.assertEqual(request["old_source_id"], "A")
        self.assertEqual(request["new_source_id"], "B")
        self.assertEqual(request["to_run_index"], 1)
        self.assertEqual(
            [item["frame_index"] for item in request["frame_evidence"]],
            [8, 9, 10, 11, 12],
        )
        refinement = augmentation.resolved_attribution["shots"][1][
            "boundary_refinement"
        ]
        self.assertEqual(refinement["status"], "resolved_hard_cut")
        self.assertEqual(refinement["selected_frame_index"], 10)

        dataset = build_reference_dataset(
            augmentation.resolved_attribution,
            style_id="synthetic_professional_reference",
        )
        self.assertEqual(
            dataset.payload["style_profile"]["cross_source_hard_cuts"],
            1,
        )
        incoming = dataset.payload["records"][1]
        self.assertEqual(incoming["boundary_label"], "cross_source_hard_cut")
        self.assertTrue(incoming["transition_supervision_eligible"])

        supervision = build_reference_supervision(
            augmentation.resolved_attribution,
            style_id="synthetic_professional_reference",
        )
        self.assertEqual(
            supervision.payload["statistics"]["transition_supervision_count"],
            1,
        )
        self.assertEqual(
            supervision.payload["statistics"][
                "editorial_source_switch_supervision_count"
            ],
            1,
        )
        edit_labels = supervision.payload["shot_labels"][1]["edit_labels"]
        self.assertTrue(edit_labels["source_transition"]["supervision_eligible"])
        self.assertEqual(
            edit_labels["boundary_event"]["event_type"],
            "editorial_source_switch_hard_cut",
        )

    def test_same_source_unknown_and_missing_frames_never_promote(self) -> None:
        same_source = _evidence(("A", "A", "A", "A", "A"))

        unknown = _evidence()
        # The resolver must keep this exact frame UNKNOWN: neither candidate
        # passes the existing score/inlier gates.
        unknown["samples"][1]["ranked_candidates"] = [
            _candidate("A", 0.30, inliers=2),
            _candidate("B", 0.29, inliers=2),
        ]

        missing = _evidence()
        del missing["samples"][1]

        for name, payload, reason in (
            ("same_source", same_source, "same_source_native_cut"),
            ("unknown", unknown, "unknown_or_failed_candidate_gate"),
            ("missing", missing, "missing_or_duplicate_boundary_sample"),
        ):
            with self.subTest(case=name):
                result = augment_reference_boundary_requests(payload)
                self.assertEqual(result.added_request_count, 0)
                self.assertEqual(result.added_resolved_hard_cut_count, 0)
                self.assertEqual(result.skip_reason_counts.get(reason), 1)
                dataset = build_reference_dataset(
                    result.resolved_attribution,
                    style_id=f"strict_{name}",
                )
                self.assertEqual(
                    dataset.payload["style_profile"]["cross_source_hard_cuts"],
                    0,
                )

    def test_score_inlier_and_ratio_gates_are_not_weakened(self) -> None:
        mutations = {
            "score": {"score": 0.39},
            "inliers": {"orb_homography_inliers": 9},
            "ratio": {"orb_inlier_ratio": 0.24},
        }
        for name, fields in mutations.items():
            with self.subTest(gate=name):
                payload = _evidence()
                selected = payload["samples"][1]["ranked_candidates"][0]
                selected.update(fields)
                result = augment_reference_boundary_requests(payload)
                self.assertEqual(result.added_request_count, 0)
                self.assertEqual(
                    result.skip_reason_counts.get(
                        "unknown_or_failed_candidate_gate"
                    ),
                    1,
                )

    def test_exact_builder_clock_is_revalidated_with_source_to_master(self) -> None:
        payload = _evidence()
        payload["samples"][2]["master_seconds"] += 0.000001

        result = augment_reference_boundary_requests(payload)

        self.assertEqual(result.added_request_count, 0)
        self.assertEqual(
            result.skip_reason_counts["non_consecutive_or_clock_mismatched_frames"],
            1,
        )

    def test_ambiguous_frame_identity_and_unstable_path_are_skipped(self) -> None:
        ambiguous = _evidence()
        ambiguous["samples"][1]["ranked_candidates"] = [
            _candidate("A", 0.72),
            _candidate("B", 0.69),
        ]
        unstable = _evidence(("A", "B", "A", "B", "B"))

        ambiguous_result = augment_reference_boundary_requests(ambiguous)
        unstable_result = augment_reference_boundary_requests(unstable)

        self.assertEqual(ambiguous_result.added_request_count, 0)
        self.assertEqual(unstable_result.added_request_count, 0)
        self.assertEqual(
            ambiguous_result.skip_reason_counts.get(
                "ambiguous_or_failed_frame_identity_gate"
            )
            + ambiguous_result.skip_reason_counts.get(
                "unknown_or_failed_candidate_gate",
                0,
            ),
            1,
        )
        self.assertEqual(
            unstable_result.skip_reason_counts[
                "ambiguous_or_unstable_source_change"
            ],
            1,
        )

    def test_file_augmentation_is_read_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.json"
            output = root / "evidence-with-boundaries.json"
            attribution = root / "attribution.json"
            source.write_text(json.dumps(_evidence()), encoding="utf-8")
            source_bytes = source.read_bytes()

            result, written, written_attribution = (
                augment_reference_boundary_requests_file(
                    source,
                    output,
                    attribution_output_path=attribution,
                )
            )

            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(written, output.resolve())
            self.assertEqual(written_attribution, attribution.resolve())
            self.assertEqual(result.added_request_count, 1)
            saved = json.loads(output.read_text(encoding="utf-8"))
            second = augment_reference_boundary_requests(saved)
            self.assertEqual(second.added_request_count, 0)
            self.assertEqual(len(second.payload["boundary_requests"]), 1)
            with self.assertRaisesRegex(ValueError, "overwrite input evidence"):
                augment_reference_boundary_requests_file(source, source)

    def test_cli_writes_augmented_evidence_and_resolved_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.json"
            output = root / "augmented.json"
            attribution = root / "attribution.json"
            source.write_text(json.dumps(_evidence()), encoding="utf-8")
            stdout = io.StringIO()

            with patch(
                "crosscut_cli._prepare_analysis_process_group",
                return_value=False,
            ), redirect_stdout(stdout):
                exit_code = crosscut_cli.main(
                    [
                        "augment-reference-boundaries",
                        "--evidence",
                        str(source),
                        "--output",
                        str(output),
                        "--attribution-output",
                        str(attribution),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["added_request_count"], 1)
            self.assertEqual(summary["added_resolved_hard_cut_count"], 1)
            self.assertEqual(
                json.loads(attribution.read_text(encoding="utf-8"))[
                    "shots"
                ][1]["boundary_refinement"]["status"],
                "resolved_hard_cut",
            )


if __name__ == "__main__":
    unittest.main()
