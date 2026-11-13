from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.reference_crop_supervision import (
    ANALYSIS_BASIS,
    ReferenceCropSupervisionConfig,
    ReferenceCropSupervisionRequest,
    _minimum_renderer_safe_target,
    build_reference_crop_supervision,
    load_reference_crop_supervision,
    validate_reference_crop_supervision_payload,
)
from crosscut.reference_evidence_builder import ReferenceSourceClock


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer_affine(
    width: int,
    height: int,
    *,
    zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
) -> np.ndarray:
    radians = math.radians(roll_degrees)
    linear = zoom * np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )
    source_center = np.asarray([width / 2.0, height / 2.0])
    reference_center = np.asarray(
        [
            width * (zoom * (0.5 - center_x) + 0.5),
            height * (zoom * (0.5 - center_y) + 0.5),
        ]
    )
    translation = reference_center - linear @ source_center
    return np.column_stack((linear, translation))


def _fixture_frame(index: int, width: int, height: int) -> np.ndarray:
    random = np.random.default_rng(1900 + index)
    frame = random.integers(10, 70, (height, width, 3), dtype=np.uint8)
    for row in range(18, height - 12, 30):
        for column in range(18, width - 12, 34):
            color = (
                (column * 5 + index * 3) % 220 + 25,
                (row * 7 + index * 5) % 220 + 25,
                (column + row * 3 + index * 9) % 220 + 25,
            )
            cv2.circle(frame, (column, row), 7, color, 2, cv2.LINE_AA)
            cv2.line(
                frame,
                (column - 8, row + 10),
                (column + 9, row - 11),
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(frame, (45, 38), (width - 58, height - 44), (230, 230, 230), 2)
    cv2.putText(
        frame,
        f"IVE CROP {index:02d}",
        (64, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (250, 245, 235),
        2,
        cv2.LINE_AA,
    )
    return frame


def _write_pair(
    root: Path,
    transforms: list[np.ndarray],
    *,
    perspective: np.ndarray | None = None,
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
    try:
        for index in range(18):
            source = _fixture_frame(index, width, height)
            transform = transforms[index]
            if perspective is None:
                reference = cv2.warpAffine(
                    source,
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            else:
                affine_h = np.vstack((transform, np.asarray([0.0, 0.0, 1.0])))
                reference = cv2.warpPerspective(
                    source,
                    perspective @ affine_h,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            source_writer.write(source)
            reference_writer.write(reference)
    finally:
        source_writer.release()
        reference_writer.release()
    return (reference_path, source_path)


def _attribution(root: Path) -> tuple[Path, dict]:
    shot = {
        "shot_index": 0,
        "reference_start_seconds": 0.20,
        "reference_end_seconds": 1.30,
        "accepted": True,
        "attributed_video_id": "SOURCE_A",
        "decision_reason": "detected_shot_midpoint_with_no_conflicting_known_source",
        "best": {
            "video_id": "SOURCE_A",
            "score": 0.94,
            "source_time_seconds": 0.75,
        },
        "canonical_detected_shot": {
            "schema_version": 1,
            "detector_shot_index": 0,
            "sample_indices": [0, 1, 2],
            "source_run_merge_does_not_define_shot_partition": True,
        },
    }
    payload = {
        "schema_version": 1,
        "analysis": "reference_hard_cut_reverse_match_evidence_v1",
        "shots": [shot],
        "provenance": {"fixture": "small_synthetic_crop_supervision_v1"},
    }
    path = root / "attribution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return (path, shot)


def _request(
    attribution_path: Path,
    shot: dict,
    reference_path: Path,
    source_path: Path,
) -> ReferenceCropSupervisionRequest:
    return ReferenceCropSupervisionRequest(
        attributed_shot=shot,
        attribution_path=attribution_path,
        reference_media_path=reference_path,
        source_media_path=source_path,
        reference_clock=ReferenceSourceClock.scalar(
            0.0, basis="fixture_reference_identity_clock_v1"
        ),
        source_clock=ReferenceSourceClock.scalar(
            0.0, basis="fixture_source_identity_clock_v1"
        ),
        source_id="SOURCE_A",
        expected_attribution_file_sha256=_sha256(attribution_path),
        expected_reference_media_sha256=_sha256(reference_path),
        expected_source_media_sha256=_sha256(source_path),
        caller_provenance={"fixture": "unit-test"},
    )


def _test_config(**changes: object) -> ReferenceCropSupervisionConfig:
    values = {
        "working_width": 480,
        "minimum_ratio_matches": 18,
        "minimum_mutual_matches": 12,
        "minimum_affine_inliers": 10,
        "minimum_affine_inlier_ratio": 0.42,
        "minimum_homography_inlier_ratio": 0.40,
        "maximum_median_reprojection_error_pixels": 3.0,
        "maximum_p90_reprojection_error_pixels": 6.0,
        "maximum_homography_median_reprojection_error_pixels": 3.2,
        "maximum_homography_p90_reprojection_error_pixels": 6.5,
        "minimum_warp_structure_ncc": 0.28,
        "minimum_warp_structure_ncc_improvement": 0.015,
    }
    values.update(changes)
    return ReferenceCropSupervisionConfig(**values)


class ReferenceCropSupervisionTests(unittest.TestCase):
    def test_bounded_rotation_is_projected_to_minimum_renderer_safe_zoom(self) -> None:
        settings = _test_config()

        target, diagnostics = _minimum_renderer_safe_target(
            zoom=1.001,
            center_x=0.5,
            center_y=0.5,
            roll_degrees=0.35,
            settings=settings,
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(diagnostics["applied"])
        self.assertGreater(target["zoom"], 1.001)
        self.assertLessEqual(target["zoom"], settings.maximum_zoom)
        self.assertEqual(target["center_x"], 0.5)
        self.assertEqual(target["center_y"], 0.5)
        self.assertEqual(target["roll_degrees"], 0.35)
        from crosscut.render import EditorialTransformKeyframe

        EditorialTransformKeyframe(0.0, **target)

    def test_out_of_domain_clock_is_loss_masked_without_decode_or_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = _renderer_affine(
                480,
                270,
                zoom=1.08,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_pair(root, [affine] * 18)
            attribution, shot = _attribution(root)
            request = _request(attribution, shot, reference, source)
            request = ReferenceCropSupervisionRequest(
                attributed_shot=request.attributed_shot,
                attribution_path=request.attribution_path,
                reference_media_path=request.reference_media_path,
                source_media_path=request.source_media_path,
                reference_clock=request.reference_clock,
                source_clock=ReferenceSourceClock.scalar(
                    10.0, basis="fixture_out_of_domain_clock_v1"
                ),
                source_id=request.source_id,
                expected_attribution_file_sha256=request.expected_attribution_file_sha256,
                expected_reference_media_sha256=request.expected_reference_media_sha256,
                expected_source_media_sha256=request.expected_source_media_sha256,
                caller_provenance=request.caller_provenance,
            )

            artifact = build_reference_crop_supervision(
                request, config=_test_config()
            )

            self.assertFalse(artifact.eligible)
            self.assertEqual(
                artifact.payload["supervision"]["rejection_reason"],
                "one_or_more_interior_sample_times_outside_media_domain",
            )
            self.assertEqual(
                {
                    item["rejection_reason"]
                    for item in artifact.payload["samples"]
                },
                {"source_clock_outside_media_domain"},
            )
            self.assertTrue(
                all(
                    item["diagnostics"]["decode_attempted"] is False
                    and item["diagnostics"]["clock_values_clamped"] is False
                    for item in artifact.payload["samples"]
                )
            )

    def test_recovers_stable_renderer_transform_from_three_redecoded_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                "zoom": 1.085,
                "center_x": 0.507,
                "center_y": 0.503,
                "roll_degrees": 0.45,
            }
            affine = _renderer_affine(480, 270, **expected)
            reference, source = _write_pair(root, [affine] * 18)
            attribution, shot = _attribution(root)

            artifact = build_reference_crop_supervision(
                _request(attribution, shot, reference, source),
                config=_test_config(),
            )

            self.assertTrue(artifact.eligible)
            self.assertEqual(artifact.payload["analysis_basis"], ANALYSIS_BASIS)
            self.assertEqual(len(artifact.payload["samples"]), 3)
            self.assertEqual(
                len(
                    {
                        item["reference_frame_index"]
                        for item in artifact.payload["samples"]
                    }
                ),
                3,
            )
            target = artifact.target
            assert target is not None
            self.assertAlmostEqual(target["zoom"], expected["zoom"], delta=0.012)
            self.assertAlmostEqual(
                target["center_x"], expected["center_x"], delta=0.006
            )
            self.assertAlmostEqual(
                target["center_y"], expected["center_y"], delta=0.006
            )
            self.assertAlmostEqual(
                target["roll_degrees"], expected["roll_degrees"], delta=0.20
            )
            supervision = artifact.payload["supervision"]
            self.assertFalse(supervision["loss_mask"]["crop_transform_loss_masked"])
            self.assertFalse(
                supervision["loss_mask"]["unsafe_or_failed_values_clamped"]
            )
            aggregate = artifact.payload["aggregate_actual_warp_review"]
            self.assertGreater(aggregate["minimum_structure_ncc_improvement"], 0.015)
            validate_reference_crop_supervision_payload(
                artifact.payload, verify_input_files=True
            )

    def test_unstable_temporal_transform_is_loss_masked_without_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = []
            for index in range(18):
                if index < 7:
                    zoom = 1.035
                elif index < 12:
                    zoom = 1.085
                else:
                    zoom = 1.145
                transforms.append(
                    _renderer_affine(
                        480,
                        270,
                        zoom=zoom,
                        center_x=0.5,
                        center_y=0.5,
                        roll_degrees=0.0,
                    )
                )
            reference, source = _write_pair(root, transforms)
            attribution, shot = _attribution(root)

            artifact = build_reference_crop_supervision(
                _request(attribution, shot, reference, source),
                config=_test_config(maximum_temporal_zoom_deviation=0.015),
            )

            self.assertFalse(artifact.eligible)
            supervision = artifact.payload["supervision"]
            self.assertEqual(supervision["rejection_reason"], "temporal_zoom_instability")
            self.assertIsNone(supervision["renderer_target"])
            self.assertTrue(supervision["loss_mask"]["crop_transform_loss_masked"])
            self.assertFalse(
                supervision["loss_mask"]["unsafe_or_failed_values_clamped"]
            )

    def test_perspective_scene_change_is_rejected_before_becoming_a_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = _renderer_affine(
                480,
                270,
                zoom=1.08,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            perspective = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.00045, 0.0, 1.0]],
                dtype=np.float64,
            )
            reference, source = _write_pair(
                root, [affine] * 18, perspective=perspective
            )
            attribution, shot = _attribution(root)

            artifact = build_reference_crop_supervision(
                _request(attribution, shot, reference, source),
                config=_test_config(
                    maximum_perspective_distortion=0.025,
                    maximum_median_reprojection_error_pixels=20.0,
                    maximum_p90_reprojection_error_pixels=35.0,
                ),
            )

            self.assertFalse(artifact.eligible)
            reasons = {
                item["rejection_reason"] for item in artifact.payload["samples"]
            }
            self.assertIn("perspective_distortion_too_high", reasons)
            self.assertIsNone(artifact.payload["supervision"]["renderer_target"])

    def test_identity_warp_must_not_be_promoted_without_actual_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _renderer_affine(
                480,
                270,
                zoom=1.0,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_pair(root, [identity] * 18)
            attribution, shot = _attribution(root)

            artifact = build_reference_crop_supervision(
                _request(attribution, shot, reference, source),
                config=_test_config(minimum_warp_structure_ncc_improvement=0.010),
            )

            self.assertFalse(artifact.eligible)
            self.assertEqual(
                {item["rejection_reason"] for item in artifact.payload["samples"]},
                {"actual_warp_improvement_too_small"},
            )
            self.assertEqual(
                artifact.payload["supervision"]["rejection_reason"],
                "one_or_more_interior_sample_gates_failed",
            )

    def test_writer_refuses_overwrite_and_loader_verifies_content_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = _renderer_affine(
                480,
                270,
                zoom=1.08,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_pair(root, [affine] * 18)
            attribution, shot = _attribution(root)
            artifact = build_reference_crop_supervision(
                _request(attribution, shot, reference, source),
                config=_test_config(),
            )
            destination = root / "crop-supervision.json"

            artifact.write(destination)
            loaded = load_reference_crop_supervision(
                destination, verify_input_files=True
            )
            self.assertEqual(
                loaded.payload["provenance"]["artifact_content_sha256"],
                artifact.payload["provenance"]["artifact_content_sha256"],
            )
            with self.assertRaises(FileExistsError):
                artifact.write(destination)
            with self.assertRaises(ValueError):
                artifact.write(reference)

            tampered = deepcopy(artifact.payload)
            tampered["supervision"]["renderer_target"]["zoom"] += 0.001
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                validate_reference_crop_supervision_payload(tampered)

    def test_expected_media_hash_mismatch_fails_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = _renderer_affine(
                480,
                270,
                zoom=1.08,
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
            )
            reference, source = _write_pair(root, [affine] * 18)
            attribution, shot = _attribution(root)
            request = ReferenceCropSupervisionRequest(
                attributed_shot=shot,
                attribution_path=attribution,
                reference_media_path=reference,
                source_media_path=source,
                reference_clock=ReferenceSourceClock.scalar(0.0),
                source_clock=ReferenceSourceClock.scalar(0.0),
                source_id="SOURCE_A",
                expected_source_media_sha256="0" * 64,
            )

            with self.assertRaisesRegex(ValueError, "source_media SHA-256"):
                build_reference_crop_supervision(request, config=_test_config())


if __name__ == "__main__":
    unittest.main()
