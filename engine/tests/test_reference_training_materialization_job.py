from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_training_materialization_job import (
    materialize_reference_training_artifacts,
)
from tests.test_reference_crop_dataset_integration import _build_fixture


class ReferenceTrainingMaterializationJobTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        _attribution, artifact = _build_fixture(root, zoom=1.08)
        attribution_path = root / "canonical-attribution.json"
        crop_path = root / (
            "REF-reference-crop-supervision-crop-v1-shot-0000-SOURCE_A.json"
        )
        artifact.write(crop_path)
        return attribution_path, crop_path

    def test_materializes_current_schemas_and_resumes_immutable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            attribution_path, _crop_path = self._fixture(root)
            output = root / "materialized"
            first = materialize_reference_training_artifacts(
                {"REF": attribution_path},
                crop_directories=(root,),
                style_id="ive-fixture",
                output_directory=output,
                expected_reference_count=1,
                expected_source_count=2,
            )
            dataset = json.loads(first.dataset_paths["REF"].read_text())
            supervision = json.loads(first.supervision_paths["REF"].read_text())
            self.assertEqual(dataset["schema_version"], 4)
            self.assertEqual(supervision["schema_version"], 3)
            self.assertEqual(supervision["base_reference_dataset_schema_version"], 4)
            self.assertEqual(
                supervision["statistics"]["crop_supervision_count"], 1
            )
            original_manifest = first.manifest_path.read_bytes()

            second = materialize_reference_training_artifacts(
                {"REF": attribution_path},
                crop_directories=(root,),
                style_id="ive-fixture",
                output_directory=output,
                expected_reference_count=1,
                expected_source_count=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(second.manifest_path.read_bytes(), original_manifest)

    def test_refuses_incomplete_crop_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            attribution_path, crop_path = self._fixture(root)
            crop_path.unlink()
            with self.assertRaisesRegex(ValueError, "no crop artifacts"):
                materialize_reference_training_artifacts(
                    {"REF": attribution_path},
                    crop_directories=(root,),
                    style_id="ive-fixture",
                    output_directory=root / "materialized",
                    expected_reference_count=1,
                    expected_source_count=2,
                )


if __name__ == "__main__":
    unittest.main()
