from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import REFERENCE_EVIDENCE_ANALYSIS_BASIS
from crosscut.stage_edit_dynamic_crop_training import (
    _reference_id_from_attribution,
)


class StageEditDynamicCropTrainingTests(unittest.TestCase):
    @staticmethod
    def _payload(
        root: Path,
        *,
        canonical_reference_id: str = "reference-a",
        artifact_reference_id: str = "reference-a",
        legacy_reference_id: str | None = None,
    ) -> dict[str, object]:
        attribution: dict[str, object] = {
            "reference_evidence_provenance": {
                "schema_version": 1,
                "analysis": REFERENCE_EVIDENCE_ANALYSIS_BASIS,
                "provenance": {
                    "caller": {
                        "reference_video_id": canonical_reference_id,
                    }
                },
            }
        }
        if legacy_reference_id is not None:
            attribution["reference_id"] = legacy_reference_id
        path = root / "canonical-final-attribution.json"
        content = json.dumps(attribution).encode("utf-8")
        path.write_bytes(content)
        return {
            "provenance": {
                "attribution": {
                    "path": str(path),
                    "file_sha256": hashlib.sha256(content).hexdigest(),
                },
                "caller": {"reference_id": artifact_reference_id},
            }
        }

    def test_current_canonical_attribution_identity_is_derived_from_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = self._payload(Path(raw))

            self.assertEqual(
                "reference-a",
                _reference_id_from_attribution(payload),
            )

    def test_canonical_youtube_id_may_begin_with_underscore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = self._payload(
                Path(raw),
                canonical_reference_id="_IuHLlXCQpk",
                artifact_reference_id="_IuHLlXCQpk",
            )

            self.assertEqual(
                "_IuHLlXCQpk",
                _reference_id_from_attribution(payload),
            )

    def test_conflicting_canonical_legacy_or_artifact_identity_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(ValueError, "reference ids conflict"):
                _reference_id_from_attribution(
                    self._payload(root, legacy_reference_id="reference-b")
                )

        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(
                ValueError,
                "artifact and attribution reference ids differ",
            ):
                _reference_id_from_attribution(
                    self._payload(
                        Path(raw),
                        artifact_reference_id="reference-b",
                    )
                )


if __name__ == "__main__":
    unittest.main()
