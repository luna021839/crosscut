from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.reference_training_source_pool import (
    build_reference_training_source_manifest,
    build_synchronized_training_source_catalog_from_manifest,
    validate_reference_training_source_manifest,
)


def _entry(path: Path, video_id: str, *, eligible: bool) -> dict:
    return {
        "video_id": video_id,
        "local_path": str(path),
        "download_status": (
            "downloaded" if eligible else "downloaded_quality_excluded"
        ),
        "rejection_reasons": [],
        "download_hard_rejection_reasons": [],
        "media_validation": {"passed": True},
        "aspect_validation": {"passed": True},
        "content_integrity": {
            "contract_version": 1,
            "algorithm": "sha256",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        },
        "group_quality_eligible": eligible,
        "group_quality_exclusion_reasons": (
            [] if eligible else ["relative_quality_below_top_eight"]
        ),
    }


class ReferenceTrainingSourcePoolTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        selected = []
        candidates = []
        for index in range(10):
            path = root / f"S{index}.mp4"
            path.write_bytes(f"media-{index}".encode())
            entry = _entry(path, f"S{index}", eligible=index < 8)
            candidates.append(deepcopy(entry))
            if index < 8:
                selected.append(deepcopy(entry))
        manifest = {
            "schema_version": 1,
            "group": "IVE",
            "song": "BANG BANG",
            "desired_source_count": 10,
            "selected": selected,
            "candidates": candidates,
            "summary": {"eligible_source_count": 8},
        }
        path = root / "acquisition.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _exact_ten_fixture(self, root: Path) -> Path:
        selected = []
        for index in range(10):
            path = root / f"T{index}.mp4"
            path.write_bytes(f"exact-media-{index}".encode())
            selected.append(_entry(path, f"T{index}", eligible=True))
        manifest = {
            "schema_version": 1,
            "group": "IVE",
            "song": "BLACKHOLE",
            "desired_source_count": 10,
            "selected": deepcopy(selected),
            "candidates": deepcopy(selected),
            "summary": {"eligible_source_count": 10},
        }
        path = root / "acquisition-exact-ten.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_builds_exact_ten_without_changing_production_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self._fixture(root)

            result = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=("S8", "S9"),
                expected_artist="IVE",
                expected_song="BANG BANG",
                expected_source_count=10,
            )

            self.assertEqual(len(result["selected"]), 10)
            self.assertEqual(
                [item["group_quality_eligible"] for item in result["selected"]],
                [True] * 8 + [False, False],
            )
            self.assertTrue(
                all(item["training_attribution_eligible"] for item in result["selected"])
            )
            self.assertEqual(
                validate_reference_training_source_manifest(
                    result,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    verify_acquisition_binding=True,
                ),
                tuple(f"S{index}" for index in range(10)),
            )

    def test_tampering_or_hard_rejection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self._fixture(root)
            result = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=("S8", "S9"),
                expected_artist="IVE",
                expected_song="BANG BANG",
                expected_source_count=10,
            )
            result["selected"][8]["group_quality_eligible"] = True

            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                validate_reference_training_source_manifest(
                    result,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    verify_acquisition_binding=True,
                )

            acquisition_payload = json.loads(acquisition.read_text())
            acquisition_payload["candidates"][8]["download_hard_rejection_reasons"] = [
                "broken"
            ]
            acquisition.write_text(json.dumps(acquisition_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hard-rejected"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=("S8", "S9"),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=10,
                )

    def test_builds_from_scratch_catalog_for_exact_ten_acquisition(self) -> None:
        class _Capture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return True

            def get(self, key: int) -> float:
                import cv2

                return {
                    cv2.CAP_PROP_FPS: 30.0,
                    cv2.CAP_PROP_FRAME_COUNT: 6000.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
                }[key]

            def release(self) -> None:
                pass

        class _Coarse:
            offset_seconds = 1.25
            score = 0.91
            score_margin = 0.22

            def to_dict(self) -> dict:
                return {
                    "offset_samples": 10_000,
                    "offset_seconds": self.offset_seconds,
                    "score": self.score,
                    "second_best_score": self.score - self.score_margin,
                    "score_margin": self.score_margin,
                    "sample_rate": 8_000,
                    "backend": "test_spectral_flux",
                }

        class _Warp:
            scalar_fallback = False
            fallback_reason = None

            def to_dict(self) -> dict:
                return {
                    "backend": "test_local_timewarp",
                    "global_offset_seconds": 1.25,
                    "scalar_fallback": False,
                    "fallback_reason": None,
                    "control_points": [
                        {"master_seconds": 0.0, "source_seconds": 1.25},
                        {"master_seconds": 100.0, "source_seconds": 101.25},
                    ],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self._exact_ten_fixture(root)
            manifest = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=(),
                expected_artist="IVE",
                expected_song="BLACKHOLE",
                expected_source_count=10,
            )
            with (
                patch(
                    "crosscut.audio.extract_mono_pcm",
                    return_value=np.ones(160_000, dtype=np.float32),
                ),
                patch(
                    "crosscut.audio.spectral_flux_alignment_offset",
                    return_value=_Coarse(),
                ),
                patch(
                    "crosscut.timewarp.estimate_local_timewarp",
                    return_value=_Warp(),
                ),
                patch("cv2.VideoCapture", _Capture),
            ):
                catalog = build_synchronized_training_source_catalog_from_manifest(
                    manifest,
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                )

            self.assertEqual(tuple(item["video_id"] for item in catalog), tuple(
                f"T{index}" for index in range(10)
            ))
            self.assertEqual(catalog[0]["clock"]["mapping_mode"], "scalar_offset")
            self.assertTrue(catalog[0]["provenance"]["identity_master"])
            self.assertTrue(all(
                item["clock"]["mapping_mode"] == "piecewise_linear"
                for item in catalog[1:]
            ))
            self.assertTrue(all(
                item["provenance"]["production_group_quality_eligible"]
                for item in catalog
            ))

    def test_legacy_selected_membership_preserves_implicit_production_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self._exact_ten_fixture(root)
            payload = json.loads(acquisition.read_text(encoding="utf-8"))
            for collection in (payload["selected"], payload["candidates"]):
                for entry in collection:
                    entry.pop("group_quality_eligible")
                    entry.pop("group_quality_exclusion_reasons")
            acquisition.write_text(json.dumps(payload), encoding="utf-8")

            manifest = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=(),
                expected_artist="IVE",
                expected_song="BLACKHOLE",
                expected_source_count=10,
            )

            self.assertTrue(all(
                item["training_admission"]["production_group_quality_eligible"]
                for item in manifest["selected"]
            ))
            self.assertTrue(all(
                item["training_admission"][
                    "production_group_quality_eligibility_explicit"
                ]
                is False
                for item in manifest["selected"]
            ))
            self.assertEqual(
                validate_reference_training_source_manifest(
                    manifest,
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                    verify_acquisition_binding=True,
                ),
                tuple(f"T{index}" for index in range(10)),
            )


if __name__ == "__main__":
    unittest.main()
