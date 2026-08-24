from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import resolve_reference_attribution_payload
from crosscut.reference_crop_supervision import (
    ReferenceCropSupervisionRequest,
    build_reference_crop_supervision,
)
from crosscut.reference_dataset import (
    build_reference_dataset,
    build_reference_dataset_file,
)
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.reference_supervision import (
    build_reference_supervision,
    build_reference_supervision_file,
)
from crosscut.stage_edit_reference_adapter import build_reference_training_example
from tests.test_reference_crop_supervision import (
    _renderer_affine,
    _test_config,
    _write_pair,
)
from tests.test_stage_edit_reference_adapter import _canonical_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_attribution(
    root: Path,
    reference_path: Path,
    source_path: Path,
    *,
    catalog_source_path: Path | None = None,
) -> tuple[Path, dict]:
    other_path = root / "other-source.avi"
    other_path.write_bytes(b"unused-catalog-fixture")
    evidence = {
        "schema_version": 1,
        "analysis": "reference_hard_cut_reverse_match_evidence_v1",
        "reference": str(reference_path.resolve()),
        "master_start_seconds": 0.20,
        "master_end_seconds": 1.30,
        "samples": [
            {
                "master_seconds": 0.75,
                "reference_frame_index": 9,
                "sample_reasons": ["shot_midpoint:0"],
                "ranked_candidates": [
                    {
                        "video_id": "SOURCE_A",
                        "score": 0.94,
                        "orb_good_matches": 60,
                        "orb_homography_inliers": 45,
                        "orb_inlier_ratio": 0.75,
                        "source_time_seconds": 0.75,
                    },
                    {
                        "video_id": "SOURCE_B",
                        "score": 0.30,
                        "orb_good_matches": 30,
                        "orb_homography_inliers": 20,
                        "orb_inlier_ratio": 0.50,
                        "source_time_seconds": 0.75,
                    },
                ],
            }
        ],
        "detected_reference_cuts": [],
        "boundary_requests": [],
        "reference_video": {
            "path": str(reference_path.resolve()),
            "frame_rate": 12.0,
            "declared_frame_count": 18,
        },
        "sources": [
            {
                "video_id": "SOURCE_A",
                "path": str((catalog_source_path or source_path).resolve()),
            },
            {"video_id": "SOURCE_B", "path": str(other_path.resolve())},
        ],
        "provenance": {"fixture": "canonical-crop-integration-v1"},
    }
    attribution = resolve_reference_attribution_payload(evidence)
    path = root / "canonical-attribution.json"
    path.write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (path, attribution)


def _build_fixture(
    root: Path,
    *,
    zoom: float,
    catalog_source_path: Path | None = None,
) -> tuple[dict, object]:
    affine = _renderer_affine(
        480,
        270,
        zoom=zoom,
        center_x=0.5,
        center_y=0.5,
        roll_degrees=0.0,
    )
    reference_path, source_path = _write_pair(root, [affine] * 18)
    attribution_path, attribution = _canonical_attribution(
        root,
        reference_path,
        source_path,
        catalog_source_path=catalog_source_path,
    )
    request = ReferenceCropSupervisionRequest(
        attributed_shot=attribution["shots"][0],
        attribution_path=attribution_path,
        reference_media_path=reference_path,
        source_media_path=source_path,
        reference_clock=ReferenceSourceClock.scalar(
            0.0,
            basis="fixture-reference-master-to-media-v1",
        ),
        source_clock=ReferenceSourceClock.scalar(
            0.0,
            basis="fixture-source-master-to-media-v1",
        ),
        source_id="SOURCE_A",
        expected_attribution_file_sha256=_sha256(attribution_path),
        expected_reference_media_sha256=_sha256(reference_path),
        expected_source_media_sha256=_sha256(source_path),
    )
    artifact = build_reference_crop_supervision(request, config=_test_config())
    return (attribution, artifact)


class ReferenceCropDatasetIntegrationTests(unittest.TestCase):
    def test_canonical_adapter_attaches_crop_only_to_editorial_positive(self) -> None:
        baseline_payload = _canonical_payload("crop-canonical")
        crop_payload = deepcopy(baseline_payload)
        editorial_transform = {
            "zoom": 1.08,
            "center_x": 0.5,
            "center_y": 0.5,
            "roll_degrees": 0.0,
        }
        native_transform = {
            "zoom": 1.12,
            "center_x": 0.5,
            "center_y": 0.5,
            "roll_degrees": 0.0,
        }
        crop_payload["shot_labels"][1]["edit_labels"]["crop"] = {
            "supervision_eligible": True,
            "transform": native_transform,
            "label_basis": "validated_fixture",
        }
        crop_payload["shot_labels"][2]["edit_labels"]["crop"] = {
            "supervision_eligible": True,
            "transform": editorial_transform,
            "label_basis": "validated_fixture",
        }

        baseline = build_reference_training_example(
            baseline_payload,
            negative_samples_per_hold=0,
        )
        example = build_reference_training_example(
            crop_payload,
            negative_samples_per_hold=0,
        )
        # Crop is target-only: token values, source availability, and missing
        # masks remain byte-for-byte equivalent to the no-crop sequence.
        self.assertEqual(example.sequence, baseline.sequence)
        targets = {
            token.master_time_seconds: target
            for token, target in zip(
                example.sequence.tokens,
                example.targets.targets,
                strict=True,
            )
        }
        self.assertIsNone(targets[1.0].crop)
        self.assertEqual(
            targets[2.5].crop,
            (1.08, 0.5, 0.5, 0.0),
        )
        self.assertEqual(
            example.statistics["crop_coordinate_supervision_count"],
            4,
        )

    def test_canonical_masked_boundary_cannot_leak_accepted_shot_crop(self) -> None:
        payload = _canonical_payload("crop-masked")
        payload["shot_labels"][2]["edit_labels"]["crop"] = {
            "supervision_eligible": True,
            "transform": {
                "zoom": 1.08,
                "center_x": 0.5,
                "center_y": 0.5,
                "roll_degrees": 0.0,
            },
            "label_basis": "validated_fixture",
        }
        masked = payload["detected_boundary_records"][1]
        masked.update(
            {
                "classification": "loss_masked",
                "classification_reason": "fixture_gate_failure",
                "cut_target": None,
                "loss_masked": True,
                "target_masks": {
                    "cut": False,
                    "source_selection": False,
                    "duration": False,
                    "transition": False,
                    "crop": False,
                },
                "is_editorial_source_switch_positive": False,
                "dense_attribution_ready": False,
                "boundary_refinement": None,
            }
        )
        payload["detected_boundary_statistics"].update(
            {
                "editorial_switch_positive_count": 0,
                "loss_masked_count": 1,
            }
        )

        example = build_reference_training_example(
            payload,
            negative_samples_per_hold=0,
        )
        masked_target = next(
            target
            for token, target in zip(
                example.sequence.tokens,
                example.targets.targets,
                strict=True,
            )
            if token.master_time_seconds == 2.5
        )
        self.assertEqual(masked_target, type(masked_target)())
        self.assertEqual(
            example.statistics["crop_coordinate_supervision_count"],
            0,
        )

    def test_accepted_artifact_reaches_existing_stage_edit_crop_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attribution, artifact = _build_fixture(Path(directory), zoom=1.085)
            self.assertTrue(artifact.eligible)

            dataset = build_reference_dataset(
                attribution,
                style_id="crop-integration",
                reference_id="fixture-reference",
                crop_supervision_artifacts=(artifact,),
            )
            record = dataset.payload["records"][0]
            target = artifact.payload["supervision"]["renderer_target"]
            self.assertEqual(record["crop_transform"], target)
            self.assertEqual(record["positive_features"]["crop_transform"], target)
            self.assertEqual(
                record["crop_supervision"]["status"],
                "accepted_renderer_target",
            )
            self.assertTrue(record["crop_supervision"]["supervision_eligible"])
            self.assertEqual(
                dataset.payload["crop_supervision_schema"][
                    "accepted_shot_indices"
                ],
                [0],
            )

            supervision = build_reference_supervision(
                attribution,
                style_id="crop-integration",
                reference_id="fixture-reference",
                crop_supervision_artifacts=(artifact,),
            )
            crop_label = supervision.payload["shot_labels"][0]["edit_labels"][
                "crop"
            ]
            self.assertTrue(crop_label["supervision_eligible"])
            self.assertEqual(crop_label["transform"], target)
            self.assertEqual(
                crop_label["provenance"]["artifact_content_sha256"],
                artifact.payload["provenance"]["artifact_content_sha256"],
            )

            training = build_reference_training_example(
                supervision.payload,
                song_id="fixture-song",
                negative_samples_per_hold=0,
            )
            expected_tuple = (
                target["zoom"],
                target["center_x"],
                target["center_y"],
                target["roll_degrees"],
            )
            self.assertIn(expected_tuple, [item.crop for item in training.targets.targets])
            self.assertEqual(
                training.statistics["crop_coordinate_supervision_count"],
                4,
            )

    def test_rejected_artifact_remains_loss_masked_through_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attribution, artifact = _build_fixture(Path(directory), zoom=1.0)
            self.assertFalse(artifact.eligible)

            supervision = build_reference_supervision(
                attribution,
                style_id="crop-rejected",
                reference_id="fixture-reference",
                crop_supervision_artifacts=(artifact,),
            )
            crop_label = supervision.payload["shot_labels"][0]["edit_labels"][
                "crop"
            ]
            self.assertFalse(crop_label["supervision_eligible"])
            self.assertIsNone(crop_label["transform"])
            self.assertTrue(
                crop_label["loss_mask"]["crop_transform_loss_masked"]
            )
            self.assertEqual(
                supervision.payload["crop_supervision_schema"][
                    "rejected_shot_indices"
                ],
                [0],
            )

            training = build_reference_training_example(
                supervision.payload,
                song_id="fixture-song",
                negative_samples_per_hold=0,
            )
            self.assertTrue(all(item.crop is None for item in training.targets.targets))
            self.assertEqual(
                training.statistics["crop_coordinate_supervision_count"],
                0,
            )

    def test_enabled_but_missing_artifact_masks_legacy_crop_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attribution, _artifact = _build_fixture(Path(directory), zoom=1.08)
            legacy_crop = {"scale": 1.05, "center_x": 0.49, "center_y": 0.48}
            attribution["shots"][0]["best"]["crop_transform"] = legacy_crop

            legacy = build_reference_dataset(
                attribution,
                style_id="legacy",
            )
            self.assertEqual(legacy.payload["records"][0]["crop_transform"], legacy_crop)
            self.assertNotIn("crop_supervision", legacy.payload["records"][0])
            self.assertNotIn("crop_supervision_schema", legacy.payload)

            strict = build_reference_dataset(
                attribution,
                style_id="strict",
                crop_supervision_artifacts=(),
            )
            record = strict.payload["records"][0]
            self.assertIsNone(record["crop_transform"])
            self.assertNotIn("crop_transform", record["positive_features"])
            self.assertEqual(
                record["crop_supervision"]["status"],
                "loss_masked_missing",
            )
            self.assertEqual(
                strict.payload["crop_supervision_schema"]["missing_shot_indices"],
                [0],
            )
            strict_supervision = build_reference_supervision(
                attribution,
                style_id="strict",
                crop_supervision_artifacts=(),
            )
            strict_crop = strict_supervision.payload["shot_labels"][0][
                "edit_labels"
            ]["crop"]
            self.assertFalse(strict_crop["supervision_eligible"])
            self.assertTrue(
                strict_crop["loss_mask"]["crop_transform_loss_masked"]
            )
            self.assertTrue(
                all(
                    "crop_transform" not in candidate["evidence"]
                    for candidate in strict_supervision.payload[
                        "listwise_records"
                    ][0]["candidates"]
                )
            )

    def test_artifact_is_bound_to_exact_attribution_and_source_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attribution, artifact = _build_fixture(root, zoom=1.08)
            changed = deepcopy(attribution)
            changed["unrelated_but_content_bearing_field"] = "tamper"
            with self.assertRaisesRegex(
                ValueError,
                "attribution canonical content hash mismatch",
            ):
                build_reference_dataset(
                    changed,
                    style_id="tampered",
                    crop_supervision_artifacts=(artifact,),
                )

            mismatch_root = root / "catalog-mismatch"
            mismatch_root.mkdir()
            wrong_source = mismatch_root / "wrong-source.avi"
            wrong_source.write_bytes(b"wrong-catalog-source")
            wrong_catalog, wrong_artifact = _build_fixture(
                mismatch_root,
                zoom=1.08,
                catalog_source_path=wrong_source,
            )
            with self.assertRaisesRegex(ValueError, "source media path mismatch"):
                build_reference_dataset(
                    wrong_catalog,
                    style_id="wrong-source",
                    crop_supervision_artifacts=(wrong_artifact,),
                )

    def test_tampered_artifact_self_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attribution, artifact = _build_fixture(Path(directory), zoom=1.08)
            tampered = deepcopy(artifact.payload)
            tampered["shot"]["source_id"] = "SOURCE_B"
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                build_reference_supervision(
                    attribution,
                    style_id="tampered-artifact",
                    crop_supervision_artifacts=(tampered,),
                )

    def test_file_builders_accept_explicit_crop_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _attribution, artifact = _build_fixture(root, zoom=1.08)
            artifact_path = root / "crop-supervision.json"
            artifact.write(artifact_path)
            dataset_path = root / "reference-dataset.json"
            supervision_path = root / "reference-supervision.json"

            build_reference_dataset_file(
                root / "canonical-attribution.json",
                dataset_path,
                style_id="file-integration",
                crop_supervision_paths=(artifact_path,),
            )
            build_reference_supervision_file(
                root / "canonical-attribution.json",
                supervision_path,
                style_id="file-integration",
                crop_supervision_paths=(artifact_path,),
            )

            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            supervision = json.loads(
                supervision_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                dataset["crop_supervision_schema"]["accepted_shot_indices"],
                [0],
            )
            self.assertTrue(
                supervision["shot_labels"][0]["edit_labels"]["crop"][
                    "supervision_eligible"
                ]
            )


if __name__ == "__main__":
    unittest.main()
