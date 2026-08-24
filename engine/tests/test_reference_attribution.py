from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import (
    BoundaryFrameEvidence,
    ReferenceAttributionConfig,
    ReferenceCandidateScore,
    ReferencePathState,
    ReferenceSample,
    collapse_reference_source_runs,
    refine_reference_source_boundary,
    resolve_reference_attribution_file,
    resolve_reference_attribution_payload,
    solve_reference_source_path,
)
from crosscut.reference_dataset import build_reference_dataset


def _candidate(
    source_id: str,
    score: float,
    *,
    inliers: int = 30,
    inlier_ratio: float = 0.60,
) -> ReferenceCandidateScore:
    return ReferenceCandidateScore(
        source_id=source_id,
        score=score,
        homography_inliers=inliers,
        homography_inlier_ratio=inlier_ratio,
        good_matches=max(inliers, 40),
    )


def _sample(time_seconds: float, *candidates: ReferenceCandidateScore) -> ReferenceSample:
    return ReferenceSample(time_seconds, tuple(candidates))


def _evidence(
    frame_index: int,
    identity: str,
    *,
    frame_rate: float = 10.0,
) -> BoundaryFrameEvidence:
    if identity == "old":
        old_score, new_score = 0.90, 0.10
        old_confident, new_confident = True, False
    elif identity == "new":
        old_score, new_score = 0.10, 0.90
        old_confident, new_confident = False, True
    elif identity == "ambiguous":
        old_score = new_score = 0.50
        old_confident = new_confident = True
    elif identity == "neither":
        old_score = new_score = 0.15
        old_confident = new_confident = False
    else:
        raise ValueError(identity)
    return BoundaryFrameEvidence(
        frame_index=frame_index,
        master_seconds=frame_index / frame_rate,
        old_score=old_score,
        new_score=new_score,
        old_confident=old_confident,
        new_confident=new_confident,
    )


def _canonical_detected_evidence(*, failed_offset: int | None = None) -> dict:
    sample_specs = [
        (0.40, ["shot_midpoint:0"], 0.92),
        (0.80, ["hard_cut_boundary:10:-2"], 0.91),
        (0.90, ["hard_cut_boundary:10:-1"], 0.91),
        (1.00, ["hard_cut_boundary:10:+0"], 0.91),
        (1.10, ["hard_cut_boundary:10:+1"], 0.91),
        (1.20, ["hard_cut_boundary:10:+2"], 0.91),
        (1.40, ["shot_midpoint:1"], 0.92),
    ]
    samples = []
    for master_seconds, reasons, score in sample_specs:
        offset = None
        if reasons[0].startswith("hard_cut_boundary"):
            offset = int(reasons[0].rsplit(":", 1)[-1])
        candidate = (
            _candidate("A", 0.30, inliers=2, inlier_ratio=0.10)
            if failed_offset is not None and offset == failed_offset
            else _candidate("A", score)
        )
        samples.append(
            {
                "master_seconds": master_seconds,
                "reference_frame_index": int(round(master_seconds * 10.0)),
                "sample_reasons": reasons,
                "ranked_candidates": [candidate.to_dict()],
            }
        )
    return {
        "schema_version": 1,
        "analysis": "reference_hard_cut_reverse_match_evidence_v1",
        "reference": "/tmp/canonical-reference.mp4",
        "master_start_seconds": 0.0,
        "master_end_seconds": 1.6,
        "samples": samples,
        "detected_reference_cuts": [
            {"frame_index": 10, "master_seconds": 1.0, "score": 0.9}
        ],
        "boundary_requests": [],
        "reference_video": {
            "path": "/tmp/canonical-reference.mp4",
            "frame_rate": 10.0,
            "declared_frame_count": 16,
        },
        "provenance": {"fixture": "canonical_detected_shot_v1"},
    }


class ReferenceAttributionTests(unittest.TestCase):
    def test_unknown_is_a_real_state_when_no_candidate_passes_gates(self) -> None:
        path = solve_reference_source_path(
            (
                _sample(0.00, _candidate("A", 0.91)),
                _sample(0.25, _candidate("A", 0.88)),
                _sample(0.50, _candidate("B", 0.32)),
                _sample(0.75, _candidate("B", 0.90)),
                _sample(1.00, _candidate("B", 0.92)),
            )
        )

        self.assertEqual(
            [state.source_id for state in path],
            ["A", "A", None, "B", "B"],
        )
        self.assertEqual(
            path[2].decision_reason,
            "no_source_passed_conservative_gates",
        )
        self.assertFalse(path[2].to_dict()["accepted"])

    def test_geometrically_weak_argmax_remains_unknown(self) -> None:
        path = solve_reference_source_path(
            (
                _sample(
                    0.0,
                    _candidate("A", 0.85, inliers=4, inlier_ratio=0.60),
                ),
            )
        )
        self.assertIsNone(path[0].source_id)

    def test_ping_pong_penalty_can_reject_a_weak_single_sample_excursion(self) -> None:
        samples = (
            _sample(0.00, _candidate("A", 0.90)),
            _sample(0.25, _candidate("B", 0.70), _candidate("C", 0.60)),
            _sample(0.50, _candidate("A", 0.90)),
        )
        without_penalty = solve_reference_source_path(
            samples,
            config=ReferenceAttributionConfig(ping_pong_penalty=0.0),
        )
        with_penalty = solve_reference_source_path(
            samples,
            config=ReferenceAttributionConfig(ping_pong_penalty=0.60),
        )

        self.assertEqual(
            [state.source_id for state in without_penalty],
            ["A", "B", "A"],
        )
        self.assertEqual(
            [state.source_id for state in with_penalty],
            ["A", None, "A"],
        )

    def test_strong_single_sample_excursion_is_not_ping_pong_suppressed(self) -> None:
        path = solve_reference_source_path(
            (
                _sample(0.00, _candidate("A", 0.90)),
                _sample(0.25, _candidate("B", 0.95)),
                _sample(0.50, _candidate("A", 0.90)),
            ),
            config=ReferenceAttributionConfig(ping_pong_penalty=0.60),
        )
        self.assertEqual([state.source_id for state in path], ["A", "B", "A"])

    def test_collapse_runs_uses_midpoints_and_preserves_unknown(self) -> None:
        states = (
            ReferencePathState(0, 1.0, "A", _candidate("A", 0.9), 0.5, 0.0, 0.5, "match"),
            ReferencePathState(1, 2.0, "A", _candidate("A", 0.8), 0.4, 0.04, 0.94, "match"),
            ReferencePathState(2, 3.0, None, None, 0.0, -0.02, 0.92, "unknown"),
        )
        runs = collapse_reference_source_runs(
            states,
            master_start_seconds=0.5,
            master_end_seconds=3.5,
        )

        self.assertEqual([run.source_id for run in runs], ["A", None])
        self.assertEqual(
            [(run.start_master_seconds, run.end_master_seconds) for run in runs],
            [(0.5, 2.5), (2.5, 3.5)],
        )

    def test_single_sample_run_gets_nonzero_default_extent(self) -> None:
        state = ReferencePathState(
            0,
            1.0,
            None,
            None,
            0.0,
            0.0,
            0.0,
            "unknown",
        )
        run = collapse_reference_source_runs((state,))[0]
        self.assertEqual((run.start_master_seconds, run.end_master_seconds), (0.5, 1.5))

    def test_refines_to_first_new_frame_inside_half_second_window(self) -> None:
        result = refine_reference_source_boundary(
            old_source_id="A",
            new_source_id="B",
            nominal_boundary_seconds=10.3,
            nominal_frame_index=103,
            frame_evidence=tuple(
                _evidence(index, "old" if index < 100 else "new")
                for index in range(96, 109)
            ),
        )

        self.assertTrue(result.resolved_hard_cut)
        self.assertEqual(result.selected_frame_index, 100)
        self.assertAlmostEqual(result.selected_boundary_seconds or 0.0, 10.0)
        self.assertEqual(result.frame_offset, -3)
        # Frames 96 and 97 are outside the configured +/-0.5 second window.
        self.assertEqual(len(result.frame_identities), 11)

    def test_blended_middle_is_not_promoted_to_hard_cut(self) -> None:
        identities = ("old", "old", "ambiguous", "ambiguous", "new", "new")
        result = refine_reference_source_boundary(
            old_source_id="A",
            new_source_id="B",
            nominal_boundary_seconds=10.0,
            nominal_frame_index=100,
            frame_evidence=tuple(
                _evidence(index, identity)
                for index, identity in zip(range(98, 104), identities, strict=True)
            ),
        )

        self.assertEqual(result.status, "unresolved_blend_or_unknown")
        self.assertIsNone(result.selected_boundary_seconds)
        self.assertGreater(result.uncertain_middle_frames, 1)

    def test_non_monotonic_source_identity_is_not_promoted_to_hard_cut(self) -> None:
        identities = ("old", "old", "new", "new", "old", "new", "new")
        result = refine_reference_source_boundary(
            old_source_id="A",
            new_source_id="B",
            nominal_boundary_seconds=10.0,
            nominal_frame_index=100,
            frame_evidence=tuple(
                _evidence(index, identity)
                for index, identity in zip(range(97, 104), identities, strict=True)
            ),
        )

        self.assertEqual(result.status, "unresolved_non_monotonic_identity")
        self.assertIsNone(result.selected_frame_index)

    def test_duplicate_frame_evidence_is_rejected_before_resolution(self) -> None:
        duplicate = _evidence(100, "old")
        with self.assertRaisesRegex(ValueError, "unique"):
            refine_reference_source_boundary(
                old_source_id="A",
                new_source_id="B",
                nominal_boundary_seconds=10.0,
                nominal_frame_index=100,
                frame_evidence=(duplicate, duplicate),
            )

    def test_sparse_frame_evidence_cannot_bridge_an_unobserved_gap(self) -> None:
        frames = (
            BoundaryFrameEvidence(1, 9.50, 0.9, 0.1, True, False),
            BoundaryFrameEvidence(2, 9.51, 0.9, 0.1, True, False),
            BoundaryFrameEvidence(100, 10.49, 0.1, 0.9, False, True),
            BoundaryFrameEvidence(101, 10.50, 0.1, 0.9, False, True),
        )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            refine_reference_source_boundary(
                old_source_id="A",
                new_source_id="B",
                nominal_boundary_seconds=10.0,
                nominal_frame_index=50,
                frame_evidence=frames,
            )

    def test_payload_resolver_attaches_only_evidence_resolved_boundaries(self) -> None:
        evidence = {
            "reference": "/tmp/fan-edit.mp4",
            "master_start_seconds": 0.0,
            "master_end_seconds": 0.875,
            "samples": [
                {
                    "master_seconds": time_seconds,
                    "ranked_candidates": [candidate.to_dict()],
                }
                for time_seconds, candidate in (
                    (0.00, _candidate("A", 0.92)),
                    (0.25, _candidate("A", 0.90)),
                    (0.50, _candidate("B", 0.91)),
                    (0.75, _candidate("B", 0.93)),
                )
            ],
            "boundary_requests": [
                {
                    "to_run_index": 1,
                    "old_source_id": "A",
                    "new_source_id": "B",
                    "nominal_boundary_seconds": 0.4,
                    "nominal_frame_index": 4,
                    "frame_evidence": [
                        {
                            "frame_index": frame.frame_index,
                            "master_seconds": frame.master_seconds,
                            "old_score": frame.old_score,
                            "new_score": frame.new_score,
                            "old_confident": frame.old_confident,
                            "new_confident": frame.new_confident,
                        }
                        for frame in (
                            _evidence(2, "old"),
                            _evidence(3, "old"),
                            _evidence(4, "new"),
                            _evidence(5, "new"),
                        )
                    ],
                }
            ],
        }

        resolved = resolve_reference_attribution_payload(evidence)

        self.assertEqual(
            [run["attributed_video_id"] for run in resolved["source_runs"]],
            ["A", "B"],
        )
        self.assertTrue(
            resolved["dense_attribution_schema"][
                "sample_level_source_path_available"
            ]
        )
        first_span = resolved["shots"][0]["dense_attribution_span"]
        self.assertEqual(first_span["sample_count"], 2)
        self.assertTrue(first_span["source_path_samples_are_contiguous"])
        refinement = resolved["shots"][1]["boundary_refinement"]
        self.assertEqual(refinement["status"], "resolved_hard_cut")
        self.assertEqual(refinement["selected_frame_index"], 4)
        dataset = build_reference_dataset(
            resolved,
            style_id="resolved_reference",
        )
        self.assertEqual(
            dataset.payload["style_profile"]["cross_source_hard_cuts"],
            1,
        )
        switch_event = dataset.payload["records"][1]["boundary_event"]
        self.assertTrue(switch_event["dense_attribution_ready"])

    def test_payload_resolver_preserves_an_explicit_unknown_run(self) -> None:
        resolved = resolve_reference_attribution_payload(
            {
                "samples": [
                    {
                        "master_seconds": 0.0,
                        "ranked_candidates": [
                            _candidate(
                                "A",
                                0.80,
                                inliers=3,
                                inlier_ratio=0.50,
                            ).to_dict()
                        ],
                    }
                ]
            }
        )

        self.assertEqual(len(resolved["shots"]), 1)
        self.assertFalse(resolved["shots"][0]["accepted"])
        self.assertIsNone(resolved["shots"][0]["attributed_video_id"])

    def test_detector_partition_preserves_same_source_native_cut(self) -> None:
        resolved = resolve_reference_attribution_payload(
            _canonical_detected_evidence()
        )

        self.assertEqual(len(resolved["source_runs"]), 1)
        self.assertEqual(len(resolved["shots"]), 2)
        self.assertEqual(
            [shot["attributed_video_id"] for shot in resolved["shots"]],
            ["A", "A"],
        )
        self.assertTrue(
            resolved["canonical_shot_schema"][
                "same_source_adjacent_detector_shots_are_never_merged"
            ]
        )
        boundary = resolved["detected_boundaries"][0]
        self.assertEqual(
            boundary["supervision_classification"],
            "native_source_camera_cut_hard_negative",
        )
        self.assertEqual(boundary["cut_target"], 0.0)
        dataset = build_reference_dataset(resolved, style_id="canonical_native")
        self.assertEqual(dataset.record_count, 2)
        self.assertEqual(
            dataset.payload["records"][1]["boundary_label"],
            "same_source_internal_camera_cut",
        )
        self.assertEqual(
            dataset.payload["detected_boundary_records"][0]["cut_target"],
            0.0,
        )

    def test_boundary_gate_failure_is_retained_but_loss_masked(self) -> None:
        resolved = resolve_reference_attribution_payload(
            _canonical_detected_evidence(failed_offset=-1)
        )

        self.assertEqual(len(resolved["shots"]), 2)
        self.assertEqual(
            [shot["attributed_video_id"] for shot in resolved["shots"]],
            ["A", "A"],
        )
        boundary = resolved["detected_boundaries"][0]
        self.assertEqual(boundary["supervision_classification"], "loss_masked")
        self.assertEqual(
            boundary["classification_reason"],
            "unknown_or_failed_candidate_gate",
        )
        self.assertIsNone(boundary["cut_target"])
        dataset = build_reference_dataset(resolved, style_id="canonical_masked")
        self.assertEqual(
            dataset.payload["records"][1]["boundary_label"],
            "detected_boundary_loss_masked",
        )
        detected = dataset.payload["detected_boundary_records"][0]
        self.assertTrue(detected["loss_masked"])
        self.assertFalse(detected["target_masks"]["cut"])

    def test_canonical_partition_hash_tamper_fails_closed(self) -> None:
        resolved = resolve_reference_attribution_payload(
            _canonical_detected_evidence()
        )
        resolved["canonical_shot_schema"]["partition_content_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "partition hash mismatch"):
            build_reference_dataset(resolved, style_id="tampered_partition")

    def test_boundary_request_cannot_attach_to_a_different_repeated_pair(self) -> None:
        samples = [
            {
                "master_seconds": time_seconds,
                "ranked_candidates": [candidate.to_dict()],
            }
            for time_seconds, candidate in (
                (0.00, _candidate("A", 0.92)),
                (0.25, _candidate("A", 0.90)),
                (0.50, _candidate("B", 0.91)),
                (0.75, _candidate("B", 0.93)),
            )
        ]
        request = {
            "to_run_index": 1,
            "old_source_id": "A",
            "new_source_id": "B",
            "nominal_boundary_seconds": 10.0,
            "nominal_frame_index": 100,
            "frame_evidence": [
                {
                    "frame_index": frame.frame_index,
                    "master_seconds": frame.master_seconds,
                    "old_score": frame.old_score,
                    "new_score": frame.new_score,
                    "old_confident": frame.old_confident,
                    "new_confident": frame.new_confident,
                }
                for frame in (
                    _evidence(98, "old"),
                    _evidence(99, "old"),
                    _evidence(100, "new"),
                    _evidence(101, "new"),
                )
            ],
        }
        with self.assertRaisesRegex(ValueError, "incoming run"):
            resolve_reference_attribution_payload(
                {
                    "master_end_seconds": 0.875,
                    "samples": samples,
                    "boundary_requests": [request],
                }
            )

    def test_payload_file_resolver_writes_dataset_ready_json(self) -> None:
        payload = {
            "samples": [
                {
                    "master_seconds": 1.0,
                    "ranked_candidates": [_candidate("A", 0.90).to_dict()],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.json"
            output = root / "attribution.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            result = resolve_reference_attribution_file(source, output)

            self.assertEqual(result, output.resolve())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["shots"][0]["attributed_video_id"], "A")
            self.assertIn("boundary_refinement", saved["decision_policy"])


if __name__ == "__main__":
    unittest.main()
