from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.reference_attribution import (
    resolve_reference_attribution_payload,
)
from crosscut.reference_evidence_builder import (
    ClockControlPoint,
    ReferenceEvidenceConfig,
    ReferenceSourceCatalogEntry,
    ReferenceSourceClock,
    WatermarkMaskPolicy,
    _plan_samples,
    _project_detected_reference_cuts,
    build_reference_evidence,
)
from crosscut.reference_match import (
    ReferenceMatchConfig,
    decode_target_frames_sequential,
    match_aligned_target_frames,
)
from crosscut.reference_supervision import build_reference_supervision
from crosscut.vision import SceneChange, SceneDetectionConfig


def _textured_frame(
    seed: int,
    label: str,
    base_bgr: tuple[int, int, int],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((180, 320, 3), base_bgr, dtype=np.uint8)
    for _ in range(130):
        center = (
            int(rng.integers(5, frame.shape[1] - 5)),
            int(rng.integers(5, frame.shape[0] - 5)),
        )
        color = tuple(int(value) for value in rng.integers(20, 256, size=3))
        cv2.circle(frame, center, int(rng.integers(1, 6)), color, -1)
    cv2.putText(
        frame,
        label,
        (54, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return frame


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float = 12.0) -> None:
    if not frames:
        raise ValueError("test video needs frames")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test MP4 writer is unavailable")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _builder_config() -> ReferenceEvidenceConfig:
    return ReferenceEvidenceConfig(
        scene_detection=SceneDetectionConfig(
            threshold=0.18,
            minimum_frame_difference=0.12,
            minimum_histogram_distance=0.02,
            min_gap_frames=4,
        ),
        reference_match=ReferenceMatchConfig(
            working_width=320,
            temporal_jitter_frames=(-1, 0, 1),
        ),
        watermark_mask=WatermarkMaskPolicy(
            policy_id="synthetic_broadcast_corner_mask_v1",
            rectangles=((0.0, 0.0, 0.12, 0.12),),
            rationale="exercise identical normalized masking in the test fixture",
        ),
    )


def _all_mapping_keys(value: object) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        output.update(str(key) for key in value)
        for child in value.values():
            output.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_all_mapping_keys(child))
    return output


class ReferenceEvidenceBuilderIntegrationTests(unittest.TestCase):
    def test_builds_cut_samples_with_one_sequential_decode_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a_path = root / "source-a.mp4"
            source_b_path = root / "source-b.mp4"
            reference_path = root / "finished-stage-mix.mp4"
            frame_a = _textured_frame(101, "SOURCE A", (12, 25, 125))
            frame_b = _textured_frame(909, "SOURCE B", (20, 145, 25))
            _write_video(source_a_path, [frame_a.copy() for _ in range(24)])
            _write_video(source_b_path, [frame_b.copy() for _ in range(24)])
            _write_video(
                reference_path,
                [frame_a.copy() for _ in range(12)]
                + [frame_b.copy() for _ in range(12)],
            )
            catalog = [
                {
                    "video_id": "source_a",
                    "path": source_a_path,
                    "clock": {
                        "basis": "synthetic_master_to_source_zero_offset_v1",
                        "offset_seconds": 0.0,
                        "provenance": {"alignment_fixture": "identity"},
                    },
                    "provenance": {"fixture_role": "first_shot"},
                },
                ReferenceSourceCatalogEntry(
                    video_id="source_b",
                    path=source_b_path,
                    clock=ReferenceSourceClock.scalar(
                        0.0,
                        basis="synthetic_master_to_source_zero_offset_v1",
                        provenance={"alignment_fixture": "identity"},
                    ),
                    provenance={"fixture_role": "second_shot"},
                ),
            ]
            files_before = set(root.rglob("*"))

            with patch(
                "crosscut.reference_evidence_builder.match_aligned_target_frames",
                wraps=match_aligned_target_frames,
            ) as aligned_match, patch(
                "crosscut.reference_match.decode_target_frames_sequential",
                wraps=decode_target_frames_sequential,
            ) as source_decode:
                evidence = build_reference_evidence(
                    reference_path,
                    catalog,
                    reference_clock=ReferenceSourceClock.scalar(
                        0.25,
                        basis="synthetic_reference_audio_alignment_v1",
                    ),
                    config=_builder_config(),
                    provenance={
                        "reference_video_id": "synthetic_stage_mix",
                        "purpose": "integration_test",
                    },
                )

            self.assertEqual(set(root.rglob("*")), files_before)
            self.assertEqual(aligned_match.call_count, 2)
            self.assertEqual(source_decode.call_count, 2)
            decoded_paths = {
                Path(call.args[0]).resolve()
                for call in source_decode.call_args_list
            }
            self.assertEqual(
                decoded_paths,
                {source_a_path.resolve(), source_b_path.resolve()},
            )

            payload = evidence.payload
            cut_indices = [
                item["frame_index"]
                for item in payload["detected_reference_cuts"]
            ]
            self.assertEqual(cut_indices, [12])
            self.assertEqual(payload["statistics"]["shot_midpoint_count"], 2)
            self.assertEqual(payload["statistics"]["boundary_sample_count"], 5)
            self.assertEqual(payload["statistics"]["sample_count"], 7)
            self.assertAlmostEqual(
                payload["detected_reference_cuts"][0]["master_seconds"],
                0.75,
            )
            self.assertEqual(
                payload["clock_policy"]["reference_mapping"]["basis"],
                "synthetic_reference_audio_alignment_v1",
            )
            self.assertEqual(
                payload["clock_policy"]["decode_policy"],
                "one_forward_only_union_decode_per_source",
            )
            self.assertEqual(
                payload["watermark_mask_policy"]["policy_id"],
                "synthetic_broadcast_corner_mask_v1",
            )
            self.assertEqual(
                payload["sources"][0]["clock"]["basis"],
                "synthetic_master_to_source_zero_offset_v1",
            )
            self.assertEqual(
                payload["sources"][1]["provenance"]["fixture_role"],
                "second_shot",
            )

            for sample in payload["samples"]:
                ranked = sample["ranked_candidates"]
                self.assertEqual(
                    {candidate["video_id"] for candidate in ranked},
                    {"source_a", "source_b"},
                )
                self.assertEqual(sample["top_two"], ranked[:2])
                self.assertEqual(sample["best"], ranked[0])
                self.assertEqual(sample["runner_up"], ranked[1])
                expected_best = (
                    "source_a"
                    if sample["reference_frame_index"] < 12
                    else "source_b"
                )
                self.assertEqual(ranked[0]["video_id"], expected_best)

            feature_policy = payload["feature_serialization_policy"]
            self.assertFalse(feature_policy["raw_frames_serialized"])
            self.assertFalse(feature_policy["orb_descriptors_serialized"])
            self.assertFalse(feature_policy["keypoints_serialized"])
            self.assertTrue(feature_policy["compact_scores_and_homography_only"])
            self.assertTrue(
                {
                    "raw_frames",
                    "descriptors",
                    "keypoints",
                    "structure",
                    "hsv_histogram",
                }.isdisjoint(_all_mapping_keys(payload))
            )
            json.dumps(payload, ensure_ascii=False, allow_nan=False)

            resolved = resolve_reference_attribution_payload(payload)
            self.assertEqual(
                [run["attributed_video_id"] for run in resolved["source_runs"]],
                ["source_a", "source_b"],
            )
            supervision = build_reference_supervision(
                resolved,
                style_id="synthetic_stage_mix",
                reference_id="synthetic_stage_mix",
            )
            self.assertEqual(supervision.listwise_count, 2)
            self.assertEqual(supervision.pairwise_count, 2)
            self.assertEqual(len(supervision.payload["source_catalog"]), 2)
            self.assertEqual(
                supervision.payload["statistics"]["transition_supervision_count"],
                0,
            )

            output_path = root / "caller-owned-external" / "evidence.json"
            written = evidence.write(output_path)
            self.assertEqual(written, output_path.resolve())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                payload,
            )
            with self.assertRaisesRegex(ValueError, "overwrite input media"):
                evidence.write(reference_path)

    def test_low_feature_matches_remain_unknown_through_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "flat-reference.mp4"
            source_a_path = root / "flat-a.mp4"
            source_b_path = root / "flat-b.mp4"
            reference_frame = np.full((90, 160, 3), (20, 70, 160), np.uint8)
            other_frame = np.full((90, 160, 3), (160, 30, 20), np.uint8)
            _write_video(reference_path, [reference_frame.copy() for _ in range(12)])
            _write_video(source_a_path, [reference_frame.copy() for _ in range(12)])
            _write_video(source_b_path, [other_frame.copy() for _ in range(12)])

            evidence = build_reference_evidence(
                reference_path,
                (
                    ReferenceSourceCatalogEntry(
                        "flat_a",
                        source_a_path,
                        ReferenceSourceClock.scalar(0.0),
                    ),
                    ReferenceSourceCatalogEntry(
                        "flat_b",
                        source_b_path,
                        ReferenceSourceClock.scalar(0.0),
                    ),
                ),
                config=ReferenceEvidenceConfig(
                    reference_match=ReferenceMatchConfig(working_width=160),
                    watermark_mask=WatermarkMaskPolicy.explicit_none(),
                ),
            )

            self.assertEqual(evidence.sample_count, 1)
            sample = evidence.payload["samples"][0]
            self.assertEqual(
                {item["video_id"] for item in sample["ranked_candidates"]},
                {"flat_a", "flat_b"},
            )
            self.assertFalse(sample["conservative_decision"]["accepted"])
            self.assertIsNone(
                sample["conservative_decision"]["accepted_video_id"]
            )
            resolved = resolve_reference_attribution_payload(evidence.payload)
            self.assertEqual(len(resolved["shots"]), 1)
            self.assertFalse(resolved["shots"][0]["accepted"])
            self.assertIsNone(resolved["shots"][0]["attributed_video_id"])
            supervision = build_reference_supervision(
                resolved,
                style_id="synthetic_unknown",
            )
            self.assertEqual(supervision.listwise_count, 0)
            self.assertEqual(supervision.pairwise_count, 0)
            self.assertEqual(
                supervision.payload["statistics"]["known_shot_label_count"],
                0,
            )


class ReferenceEvidencePolicyTests(unittest.TestCase):
    def test_piecewise_clock_interpolates_and_extrapolates(self) -> None:
        clock = ReferenceSourceClock(
            basis="synthetic_piecewise_clock_v1",
            global_offset_seconds=1.0,
            control_points=(
                ClockControlPoint(0.0, 1.0),
                ClockControlPoint(10.0, 12.0),
            ),
            provenance={"derived_from": "fixture_control_points"},
        )

        self.assertAlmostEqual(clock.master_to_source(5.0), 6.5)
        self.assertAlmostEqual(clock.master_to_source(15.0), 17.5)
        self.assertAlmostEqual(clock.source_to_master(6.5), 5.0)
        self.assertAlmostEqual(clock.source_to_master(17.5), 15.0)
        self.assertEqual(clock.to_dict()["mapping_mode"], "piecewise_linear")
        self.assertEqual(
            clock.to_dict()["global_offset_application"],
            "diagnostic_scalar_baseline_control_points_are_absolute",
        )
        manifest_clock = ReferenceSourceClock.from_mapping(
            {
                "backend": "spectral_flux_local_ncc",
                "global_offset_seconds": 1.0,
                "control_points": [
                    {"master_seconds": 0.0, "source_seconds": 1.0},
                    {"master_seconds": 10.0, "source_seconds": 12.0},
                ],
                "provenance": {"status": "local_monotonic_timewarp"},
            }
        )
        self.assertAlmostEqual(manifest_clock.master_to_source(5.0), 6.5)
        self.assertAlmostEqual(manifest_clock.source_to_master(6.5), 5.0)
        self.assertEqual(manifest_clock.basis, "spectral_flux_local_ncc")

    def test_reference_clock_maps_decoded_reference_time_to_master(self) -> None:
        reference_clock = ReferenceSourceClock.scalar(
            0.375,
            basis="synthetic_reference_audio_alignment_v1",
        )

        self.assertAlmostEqual(reference_clock.source_to_master(4.375), 4.0)
        self.assertAlmostEqual(reference_clock.master_to_source(4.0), 4.375)

    def test_reference_preroll_samples_before_master_zero_are_excluded(self) -> None:
        samples = _plan_samples(
            (SceneChange(2, 1.0, 1.0, 1.0),),
            frame_count=12,
            frame_rate=10.0,
            master_start_seconds=0.0,
            reference_clock=ReferenceSourceClock.scalar(0.25),
            boundary_offsets_frames=(-2, -1, 0, 1, 2),
        )

        self.assertTrue(samples)
        self.assertTrue(all(item.master_seconds >= 0.0 for item in samples))
        self.assertNotIn(0, {item.frame_index for item in samples})

    def test_reference_preroll_cut_is_audited_but_not_trainable(self) -> None:
        retained, excluded = _project_detected_reference_cuts(
            (
                SceneChange(2, 1.0, 1.0, 1.0),
                SceneChange(8, 1.0, 1.0, 1.0),
            ),
            frame_rate=10.0,
            master_start_seconds=0.0,
            master_end_seconds=1.2,
            reference_clock=ReferenceSourceClock.scalar(0.25),
        )

        self.assertEqual([cut["frame_index"] for cut in retained], [8])
        self.assertAlmostEqual(retained[0]["master_seconds"], 0.55)
        self.assertEqual([cut["frame_index"] for cut in excluded], [2])
        self.assertAlmostEqual(excluded[0]["master_seconds"], -0.05)
        self.assertEqual(
            excluded[0]["exclusion_reason"],
            "at_or_before_performance_master_start",
        )

    def test_reference_outro_samples_after_caller_master_end_are_excluded(self) -> None:
        samples = _plan_samples(
            (SceneChange(5, 1.0, 1.0, 1.0),),
            frame_count=20,
            frame_rate=10.0,
            master_start_seconds=0.0,
            master_end_seconds=0.75,
            reference_clock=ReferenceSourceClock.scalar(0.0),
            boundary_offsets_frames=(-2, -1, 0, 1, 2),
        )

        self.assertTrue(samples)
        self.assertTrue(all(item.master_seconds <= 0.75 for item in samples))
        self.assertNotIn(12, {item.frame_index for item in samples})

    def test_watermark_mask_is_normalized_and_does_not_mutate_input(self) -> None:
        policy = WatermarkMaskPolicy(
            policy_id="upper_left_quarter_v1",
            rectangles=((0.0, 0.0, 0.5, 0.5),),
            fill_bgr=(1, 2, 3),
        )
        frame = np.full((10, 20, 3), 255, np.uint8)

        masked = policy.apply(frame)

        self.assertEqual(tuple(masked[0, 0]), (1, 2, 3))
        self.assertEqual(tuple(masked[-1, -1]), (255, 255, 255))
        self.assertEqual(tuple(frame[0, 0]), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
