from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

from crosscut.stage_edit_production_train_cli import (
    _dynamic_artifact_map,
    _training_result_exit_code,
    build_parser,
    main,
)


class StageEditProductionTrainCliTests(unittest.TestCase):
    def test_evaluation_only_exit_code_tracks_the_heldout_gate(self) -> None:
        self.assertEqual(
            0,
            _training_result_exit_code(
                activation_authorized=False,
                production_ready=False,
                quality_gate_passed=True,
            ),
        )
        self.assertEqual(
            2,
            _training_result_exit_code(
                activation_authorized=False,
                production_ready=False,
                quality_gate_passed=False,
            ),
        )

    def test_authorized_exit_code_requires_production_ready(self) -> None:
        self.assertEqual(
            2,
            _training_result_exit_code(
                activation_authorized=True,
                production_ready=False,
                quality_gate_passed=True,
            ),
        )
        self.assertEqual(
            0,
            _training_result_exit_code(
                activation_authorized=True,
                production_ready=True,
                quality_gate_passed=True,
            ),
        )

    def test_complete_dynamic_crop_manifest_expands_to_reference_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "shot-0.json"
            second = root / "shot-1.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "reference_id": "reference-a",
                                "artifact": {
                                    "path": str(first),
                                    "file_sha256": "a" * 64,
                                    "content_sha256": "b" * 64,
                                },
                            },
                            {
                                "reference_id": "reference-b",
                                "artifact": {
                                    "path": str(second),
                                    "file_sha256": "a" * 64,
                                    "content_sha256": "b" * 64,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "validate_reference_dynamic_crop_manifest"
                ) as validate,
                patch(
                    "crosscut.stage_edit_production_train_cli._file_sha256",
                    return_value="a" * 64,
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_dynamic_crop_supervision",
                    return_value=SimpleNamespace(
                        payload={
                            "provenance": {
                                "artifact_content_sha256": "b" * 64
                            }
                        }
                    ),
                ),
            ):
                result = _dynamic_artifact_map((), (manifest,))

            validate.assert_called_once_with(
                ANY,
                verify_artifact_files=False,
            )
            self.assertEqual((first.resolve(),), result["reference-a"])
            self.assertEqual((second.resolve(),), result["reference-b"])

    def test_manifest_and_explicit_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "shot.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "reference_id": "reference-a",
                                "artifact": {"path": str(artifact)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "validate_reference_dynamic_crop_manifest"
                ),
                self.assertRaisesRegex(ValueError, "duplicate dynamic crop"),
            ):
                _dynamic_artifact_map(
                    (f"reference-a={artifact}",),
                    (manifest,),
                )

    def test_parser_accepts_manifest_without_per_shot_arguments(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--input",
                "/tmp/input.json",
                "--song-id",
                "reference-a=IVE-XOXZ",
                "--held-out-song",
                "IVE-XOXZ",
                "--feature-snapshot",
                "reference-a=/tmp/features.json",
                "--dynamic-crop-manifest",
                "/tmp/crop-manifest.json",
                "--checkpoint-dir",
                "/tmp/checkpoint",
            ]
        )

        self.assertEqual([], parsed.dynamic_crop_artifact)
        self.assertEqual(
            [Path("/tmp/crop-manifest.json")],
            parsed.dynamic_crop_manifest,
        )

    def test_existing_checkpoint_directory_fails_before_loading_data(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "existing-checkpoint"
            checkpoint.mkdir()
            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_training_corpus"
                ) as load,
                self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
            ):
                main(
                    [
                        "--input",
                        "/tmp/input.json",
                        "--song-id",
                        "reference-a=IVE-XOXZ",
                        "--held-out-song",
                        "IVE-XOXZ",
                        "--feature-snapshot",
                        "reference-a=/tmp/features.json",
                        "--dynamic-crop-manifest",
                        "/tmp/crop-manifest.json",
                        "--checkpoint-dir",
                        str(checkpoint),
                    ]
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
