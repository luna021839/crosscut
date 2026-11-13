from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_EVIDENCE_ANALYSIS_BASIS,
)
from crosscut.dynamic_crop_input_verification import (
    verify_dynamic_crop_input_files,
)
from crosscut.reference_dynamic_crop_supervision import (
    DYNAMIC_CROP_NAMESPACE,
    ReferenceDynamicCropSupervisionConfig,
    ReferenceDynamicCropSupervisionRequest,
    _pair_interpolation_review,
    build_reference_dynamic_crop_supervision,
    load_reference_dynamic_crop_supervision,
    reference_dynamic_crop_v2_filename,
    validate_reference_dynamic_crop_supervision_payload,
)
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.stage_edit_dynamic_crop_adapter import (
    build_stage_edit_dynamic_crop_target,
)
from tests.test_reference_crop_supervision import (
    _fixture_frame,
    _renderer_affine,
    _test_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_dynamic_pair(
    root: Path,
    transforms: list[np.ndarray],
    *,
    reference_source_frame_offset: int = 0,
    blank_reference_frames: tuple[int, ...] = (),
) -> tuple[Path, Path]:
    width, height = 480, 270
    source_path = root / "source.avi"
    reference_path = root / "reference.avi"
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    source_writer = cv2.VideoWriter(str(source_path), codec, 12.0, (width, height))
    reference_writer = cv2.VideoWriter(
        str(reference_path), codec, 12.0, (width, height)
    )
    if not source_writer.isOpened() or not reference_writer.isOpened():
        raise RuntimeError("MJPG fixture writer is unavailable")
    frames = [_fixture_frame(index, width, height) for index in range(len(transforms))]
    try:
        for frame in frames:
            source_writer.write(frame)
        for index, transform in enumerate(transforms):
            source_index = min(
                len(frames) - 1, max(0, index + reference_source_frame_offset)
            )
            if index in blank_reference_frames:
                reference_writer.write(np.zeros_like(frames[source_index]))
            else:
                reference_writer.write(
                    cv2.warpAffine(
                        frames[source_index],
                        transform,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                )
    finally:
        source_writer.release()
        reference_writer.release()
    return reference_path, source_path


def _attribution_identity(
    partition_sha: str,
    source_path: list[dict],
    shots: list[dict],
    boundaries: list[dict],
) -> dict:
    return {
        "partition_content_sha256": partition_sha,
        "source_path_source_ids": [item.get("source_id") for item in source_path],
        "shot_aggregates": [
            {
                "shot_index": shot.get("shot_index"),
                "accepted": shot.get("accepted"),
                "attributed_video_id": shot.get("attributed_video_id"),
                "canonical_detected_shot": deepcopy(
                    shot.get("canonical_detected_shot")
                ),
                "dense_attribution_span": deepcopy(shot.get("dense_attribution_span")),
                "boundary_refinement": deepcopy(shot.get("boundary_refinement")),
            }
            for shot in shots
        ],
        "detected_boundary_labels": [
            {
                "boundary_index": item.get("boundary_index"),
                "incoming_shot_index": item.get("incoming_shot_index"),
                "old_source_id": item.get("old_source_id"),
                "new_source_id": item.get("new_source_id"),
                "supervision_classification": item.get(
                    "supervision_classification"
                ),
                "classification_reason": item.get("classification_reason"),
                "cut_target": item.get("cut_target"),
                "loss_masked": item.get("loss_masked"),
                "dense_attribution_ready": item.get("dense_attribution_ready"),
                "boundary_refinement": deepcopy(item.get("boundary_refinement")),
            }
            for item in boundaries
        ],
    }


def _write_canonical_inputs(
    root: Path,
    reference: Path,
    source: Path,
    *,
    sample_frames: tuple[int, ...] = (3, 9, 15),
    evidence_source_frame_offset: int = 0,
    master_end: float = 1.40,
) -> tuple[Path, Path, dict]:
    reference_clock = ReferenceSourceClock.scalar(
        0.0, basis="fixture_reference_identity_clock_v2"
    )
    source_clock = ReferenceSourceClock.scalar(
        0.0, basis="fixture_source_identity_clock_v2"
    )
    source_catalog = {
        "video_id": "SOURCE_A",
        "path": str(source.resolve()),
        "clock": source_clock.to_dict(),
        "media_probe": {
            "frame_rate": 12.0,
            "declared_frame_count": 18,
            "width": 480,
            "height": 270,
        },
    }
    samples = []
    for frame in sample_frames:
        master = frame / 12.0
        source_frame = frame + evidence_source_frame_offset
        samples.append(
            {
                "master_seconds": master,
                "reference_frame_index": frame,
                "sample_reasons": ["shot_midpoint:0"],
                "ranked_candidates": [
                    {
                        "video_id": "SOURCE_A",
                        "score": 0.95,
                        "source_frame_index": source_frame,
                        "source_time_seconds": source_frame / 12.0,
                        "expected_source_time_seconds": master,
                        "requested_source_frame_indices": [source_frame],
                        "match_status": "matched",
                    }
                ],
            }
        )
    evidence = {
        "schema_version": 1,
        "analysis": REFERENCE_EVIDENCE_ANALYSIS_BASIS,
        "reference": str(reference.resolve()),
        "master_start_seconds": 0.20,
        "master_end_seconds": master_end,
        "samples": samples,
        "detected_reference_cuts": [],
        "clock_policy": {"reference_mapping": reference_clock.to_dict()},
        "reference_video": {
            "path": str(reference.resolve()),
            "frame_rate": 12.0,
            "declared_frame_count": 18,
            "width": 480,
            "height": 270,
        },
        "sources": [source_catalog],
        "provenance": {"fixture": "dynamic-crop-v2"},
    }
    sample_indices = list(range(len(samples)))
    partition_identity = {
        "master_start_seconds": 0.20,
        "master_end_seconds": master_end,
        "detected_reference_cuts": [],
        "sample_assignment": [sample_indices],
    }
    partition_sha = _canonical_sha(partition_identity)
    shot = {
        "shot_index": 0,
        "reference_start_seconds": 0.20,
        "reference_end_seconds": master_end,
        "accepted": True,
        "attributed_video_id": "SOURCE_A",
        "decision_reason": "canonical_fixture",
        "canonical_detected_shot": {
            "schema_version": 1,
            "detector_shot_index": 0,
            "sample_indices": sample_indices,
            "source_run_merge_does_not_define_shot_partition": True,
        },
        "dense_attribution_span": {
            "schema_version": 1,
            "first_sample_index": 0,
            "last_sample_index": len(samples) - 1,
            "sample_count": len(samples),
            "selected_source_id": "SOURCE_A",
            "state": "known",
            "source_path_samples_are_contiguous": True,
            "ready": True,
            "reason": "validated_synchronized_sample_path",
        },
    }
    path = [{"source_id": "SOURCE_A"} for _ in samples]
    boundaries: list[dict] = []
    attribution_sha = _canonical_sha(
        _attribution_identity(partition_sha, path, [shot], boundaries)
    )
    attribution = {
        "schema_version": 1,
        "analysis": REFERENCE_ATTRIBUTION_BASIS,
        "reference": str(reference.resolve()),
        "sources": [source_catalog],
        "source_path": path,
        "shots": [shot],
        "detected_boundaries": boundaries,
        "canonical_shot_schema": {
            "schema_version": 1,
            "partition_basis": "reference_hard_cut_detector",
            "source_runs_are_attribution_only_not_shot_partitions": True,
            "partition_content_sha256": partition_sha,
            "attribution_content_sha256": attribution_sha,
        },
        "reference_evidence_provenance": {
            "partition_content_sha256": partition_sha,
            "attribution_content_sha256": attribution_sha,
        },
    }
    evidence_path = root / "reference-evidence-canonical-v2.json"
    attribution_path = root / "reference-attribution-canonical-v2-final.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    attribution_path.write_text(json.dumps(attribution), encoding="utf-8")
    return attribution_path, evidence_path, shot


def _request(
    attribution: Path,
    evidence: Path,
    shot: dict,
    reference: Path,
    source: Path,
) -> ReferenceDynamicCropSupervisionRequest:
    return ReferenceDynamicCropSupervisionRequest(
        attributed_shot=shot,
        attribution_path=attribution,
        evidence_path=evidence,
        reference_media_path=reference,
        source_media_path=source,
        source_id="SOURCE_A",
        expected_attribution_file_sha256=_sha256(attribution),
        expected_evidence_file_sha256=_sha256(evidence),
        expected_reference_media_sha256=_sha256(reference),
        expected_source_media_sha256=_sha256(source),
        caller_provenance={"fixture": "unit-test"},
    )


def _config() -> ReferenceDynamicCropSupervisionConfig:
    return ReferenceDynamicCropSupervisionConfig(frame_estimator=_test_config())


class ReferenceDynamicCropSupervisionTests(unittest.TestCase):
    def test_moving_zoom_and_pan_remain_variable_dynamic_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = [
                _renderer_affine(
                    480,
                    270,
                    zoom=1.045 + 0.005 * index,
                    center_x=0.500 + 0.0008 * index,
                    center_y=0.500,
                    roll_degrees=0.0,
                )
                for index in range(18)
            ]
            reference, source = _write_dynamic_pair(root, transforms)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )

            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            self.assertTrue(artifact.eligible)
            self.assertEqual(artifact.payload["namespace"], DYNAMIC_CROP_NAMESPACE)
            self.assertEqual(artifact.payload["supervision"]["motion_label"], "dynamic")
            self.assertEqual(len(artifact.keyframes), 4)
            self.assertEqual(
                artifact.keyframes[0]["time_within_shot_seconds"], 0.0
            )
            shot_start_sample = artifact.payload["samples"][0]
            self.assertTrue(shot_start_sample["accepted"])
            self.assertTrue(
                shot_start_sample["audit"]["observation_binding"][
                    "real_detector_shot_start_observation"
                ]
            )
            self.assertFalse(
                shot_start_sample["audit"]["observation_binding"][
                    "interior_keyframe_shifted_to_shot_start"
                ]
            )
            self.assertLess(
                artifact.keyframes[0]["zoom"], artifact.keyframes[-1]["zoom"]
            )
            self.assertNotEqual(
                artifact.keyframes[0]["center_x"],
                artifact.keyframes[-1]["center_x"],
            )
            validate_reference_dynamic_crop_supervision_payload(
                artifact.payload, verify_input_files=True
            )

    def test_unavailable_real_shot_start_loss_masks_interior_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.502,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(
                root,
                [transform] * 18,
                # The bound clocks quantize the 0.20s detector start to frame 2.
                # Its pixels are genuinely unavailable for geometry, while all
                # later canonical evidence frames remain valid.
                blank_reference_frames=(2,),
            )
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )

            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            self.assertFalse(artifact.eligible)
            self.assertEqual(artifact.keyframes, ())
            self.assertEqual(
                artifact.payload["supervision"]["rejection_reason"],
                "real_shot_start_transform_unavailable",
            )
            start = artifact.payload["samples"][0]
            self.assertEqual(start["time_within_shot_seconds"], 0.0)
            self.assertFalse(start["accepted"])
            self.assertTrue(
                start["loss_mask"]["dynamic_crop_keyframe_loss_masked"]
            )
            later = artifact.payload["samples"][1:]
            self.assertTrue(any(item["master_seconds"] > 0.20 for item in later))
            self.assertTrue(all(not item["accepted"] for item in later))
            self.assertTrue(
                all(
                    item["renderer_keyframe"] is None
                    and item["loss_mask"]["substitute_keyframe_created"] is False
                    for item in later
                )
            )
            validate_reference_dynamic_crop_supervision_payload(artifact.payload)

    def test_strict_local_search_recovers_one_frame_evidence_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.504,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(
                root,
                [transform] * 18,
                reference_source_frame_offset=1,
            )
            attribution, evidence, shot = _write_canonical_inputs(
                root,
                reference,
                source,
                evidence_source_frame_offset=0,
            )

            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            self.assertTrue(artifact.eligible)
            self.assertEqual(
                [item["local_frame_offset"] for item in artifact.keyframes],
                [1, 1, 1, 1],
            )
            self.assertTrue(
                all(
                    sample["audit"]["local_visual_search"][
                        "arbitrary_fractional_source_time_used"
                    ]
                    is False
                    for sample in artifact.payload["samples"]
                )
            )

    def test_renderer_safe_zoom_projection_is_revalidated_per_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.008,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.35,
            )
            reference, source = _write_dynamic_pair(root, [transform] * 18)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )

            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            self.assertTrue(artifact.eligible)
            for sample in artifact.payload["samples"]:
                selected = sample["audit"]["local_visual_search"][
                    "selected_source_frame_index"
                ]
                candidate = next(
                    item
                    for item in sample["audit"]["local_visual_search"]["candidates"]
                    if item["source_frame_index"] == selected
                )
                projection = candidate["diagnostics"][
                    "renderer_safe_zoom_projection"
                ]
                self.assertTrue(projection["applied"])
                self.assertTrue(projection["pixel_warp_revalidation_required"])
                self.assertIn("actual_warp_review", candidate["diagnostics"])

    def test_out_of_domain_canonical_sample_is_masked_not_job_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(root, [transform] * 18)
            attribution, evidence, shot = _write_canonical_inputs(
                root,
                reference,
                source,
                sample_frames=(3, 24),
                master_end=2.50,
            )

            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            self.assertTrue(artifact.eligible)
            self.assertEqual(len(artifact.keyframes), 2)
            self.assertEqual(
                artifact.keyframes[0]["time_within_shot_seconds"], 0.0
            )
            rejected = artifact.payload["samples"][-1]
            self.assertFalse(rejected["accepted"])
            self.assertEqual(
                rejected["rejection_reason"], "reference_frame_outside_media_domain"
            )
            self.assertTrue(
                rejected["loss_mask"]["dynamic_crop_keyframe_loss_masked"]
            )
            self.assertFalse(rejected["loss_mask"]["unsafe_or_failed_values_clamped"])

    def test_clock_domain_mismatch_fails_before_pixels_become_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(root, [transform] * 18)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["samples"][1]["ranked_candidates"][0][
                "expected_source_time_seconds"
            ] += 0.2
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            request = _request(attribution, evidence, shot, reference, source)

            with self.assertRaisesRegex(ValueError, "differs from source clock"):
                build_reference_dynamic_crop_supervision(request, config=_config())

    def test_hash_binding_and_crop_v2_writer_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(root, [transform] * 18)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )
            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )
            destination = root / reference_dynamic_crop_v2_filename(
                "REFERENCE_A", 0, "SOURCE_A"
            )

            artifact.write(destination)
            loaded = load_reference_dynamic_crop_supervision(
                destination, verify_input_files=True
            )
            self.assertEqual(
                loaded.payload["provenance"]["build_fingerprint_sha256"],
                artifact.payload["provenance"]["build_fingerprint_sha256"],
            )
            with self.assertRaises(FileExistsError):
                artifact.write(destination)
            with self.assertRaises(ValueError):
                artifact.write(root / "legacy-crop-v1.json")
            tampered = deepcopy(artifact.payload)
            tampered["supervision"]["renderer_keyframes"][0]["zoom"] += 0.001
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                validate_reference_dynamic_crop_supervision_payload(tampered)

    def test_shared_input_digest_cache_hashes_each_large_input_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = _renderer_affine(
                480,
                270,
                zoom=1.09,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_dynamic_pair(root, [transform] * 18)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )
            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )
            destination = root / reference_dynamic_crop_v2_filename(
                "REFERENCE_A", 0, "SOURCE_A"
            )
            artifact.write(destination)
            digest_cache: dict[str, str] = {}
            identity_cache: dict[str, tuple[int, int, int, int, int]] = {}

            loaded = load_reference_dynamic_crop_supervision(
                destination,
                verify_input_files=False,
            )
            with patch(
                "crosscut.dynamic_crop_input_verification._stable_file_sha256",
                wraps=_sha256,
            ) as hash_file:
                verify_dynamic_crop_input_files(
                    loaded.payload,
                    verified_sha256_cache=digest_cache,
                    verified_identity_cache=identity_cache,
                )
                verify_dynamic_crop_input_files(
                    loaded.payload,
                    verified_sha256_cache=digest_cache,
                    verified_identity_cache=identity_cache,
                )

            self.assertEqual(4, hash_file.call_count)
            self.assertEqual(4, len(digest_cache))
            self.assertEqual(4, len(identity_cache))

            source.write_bytes(b"x" * source.stat().st_size)
            with self.assertRaisesRegex(ValueError, "identity changed after hashing"):
                verify_dynamic_crop_input_files(
                    loaded.payload,
                    verified_sha256_cache=digest_cache,
                    verified_identity_cache=identity_cache,
                )

    def test_input_drift_error_identifies_path_and_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed-source.mp4"
            path.write_bytes(b"current source bytes")
            observed = _sha256(path)
            expected = hashlib.sha256(b"original source bytes").hexdigest()
            payload = {
                "provenance": {
                    "attribution": {
                        "path": str(path.resolve()),
                        "file_sha256": observed,
                    },
                    "evidence": {
                        "path": str(path.resolve()),
                        "file_sha256": observed,
                    },
                    "reference_media": {
                        "path": str(path.resolve()),
                        "media_sha256": observed,
                    },
                    "source_media": {
                        "path": str(path.resolve()),
                        "media_sha256": expected,
                    },
                }
            }

            with self.assertRaises(ValueError) as raised:
                verify_dynamic_crop_input_files(payload)

            message = str(raised.exception)
            self.assertIn("name=source_media", message)
            self.assertIn(f"path={path.resolve()}", message)
            self.assertIn(f"expected_sha256={expected}", message)
            self.assertIn(f"observed_sha256={observed}", message)

    def test_interpolation_safety_rejects_individually_safe_endpoints(self) -> None:
        left = {
            "time_within_shot_seconds": 0.2,
            "zoom": 1.0325846172744377,
            "center_x": 0.49955211673394156,
            "center_y": 0.5002950952466916,
            "roll_degrees": 0.5404180985226001,
        }
        right = {
            "time_within_shot_seconds": 1.0,
            "zoom": 1.0872770599014896,
            "center_x": 0.5099961525870863,
            "center_y": 0.49354861234563674,
            "roll_degrees": 1.568702272916933,
        }

        safe, error = _pair_interpolation_review(left, right)

        self.assertFalse(safe)
        self.assertIn("interpolation", str(error))

    def test_training_adapter_exposes_variable_target_without_legacy_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = [
                _renderer_affine(
                    480,
                    270,
                    zoom=1.05 + index * 0.004,
                    center_x=0.5 + index * 0.0006,
                    center_y=0.5,
                    roll_degrees=0.0,
                )
                for index in range(18)
            ]
            reference, source = _write_dynamic_pair(root, transforms)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )
            artifact = build_reference_dynamic_crop_supervision(
                _request(attribution, evidence, shot, reference, source),
                config=_config(),
            )

            target = build_stage_edit_dynamic_crop_target(artifact.payload)

            self.assertIsNotNone(target)
            assert target is not None
            serialized = target.to_target_dict()
            self.assertIsNone(target.legacy_static_crop_head_target)
            self.assertFalse(serialized["legacy_static_crop_head_compatible"])
            self.assertFalse(serialized["motion_label_loss_masked"])
            self.assertEqual(len(serialized["keyframes"]), 4)
            forbidden = (
                "score",
                "rank",
                "homography",
                "ncc",
                "clock",
                "source_frame_index",
            )
            binding = serialized.pop("source_frame_binding")
            self.assertFalse(binding["model_input"])
            self.assertEqual(len(binding["observations"]), 4)
            flattened = json.dumps(serialized).lower()
            self.assertTrue(all(name not in flattened for name in forbidden))


if __name__ == "__main__":
    unittest.main()
