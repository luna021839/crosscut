from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosscut.reference_training_source_pool import (
    build_reference_training_source_manifest,
    build_synchronized_training_source_catalog,
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

    def _supplemental_fixture(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, Path]]:
        media: dict[str, Path] = {}
        entries: dict[str, dict] = {}
        for index in range(10):
            path = root / f"S{index}.mp4"
            path.write_bytes(f"supplemental-media-{index}".encode())
            media[f"S{index}"] = path
            entries[f"S{index}"] = _entry(path, f"S{index}", eligible=True)

        acquisition = root / "changing-acquisition.json"
        historical_ids = (*[f"S{index}" for index in range(6)], "S8", "S9")
        historical = {
            "schema_version": 1,
            "group": "IVE",
            "song": "BANG BANG",
            "selected": [deepcopy(entries[value]) for value in historical_ids],
            "candidates": [deepcopy(entries[f"S{index}"]) for index in range(10)],
        }
        acquisition.write_text(json.dumps(historical), encoding="utf-8")
        prior_training = build_reference_training_source_manifest(
            acquisition,
            additional_source_ids=(),
            expected_artist="IVE",
            expected_song="BANG BANG",
            expected_source_count=8,
        )
        prior_path = root / "historical-training.json"
        prior_path.write_text(json.dumps(prior_training), encoding="utf-8")

        # Model a canonical-cache promotion after the historical manifest was
        # sealed. S5 exists in both selections, but only the current acquisition
        # may describe it; treating every old selected entry as supplemental
        # would now fail its stale size/SHA contract, exactly like repaired QFB.
        media["S5"].write_bytes(b"current-promoted-overlapping-source-five")
        entries["S5"] = _entry(media["S5"], "S5", eligible=True)
        current_ids = tuple(f"S{index}" for index in range(8))
        current = {
            "schema_version": 1,
            "group": "IVE",
            "song": "BANG BANG",
            "selected": [deepcopy(entries[value]) for value in current_ids],
            # The current acquisition intentionally has no admissible metadata
            # for S9; its only trusted source is the sealed historical manifest.
            "candidates": [
                *[deepcopy(entries[value]) for value in current_ids],
                {"video_id": "S8"},
                {"video_id": "S9"},
            ],
        }
        acquisition.write_text(json.dumps(current), encoding="utf-8")
        # S8 is selected only by the old manifest.  Its now-stale live path must
        # not block the explicitly requested, independently revalidated S9.
        media["S8"].unlink()
        return acquisition, prior_path, media

    @staticmethod
    def _reseal_training_manifest(payload: dict) -> None:
        contract = payload["reference_training_source_pool"]
        contract["manifest_content_sha256"] = None
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        contract["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()

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
            self.assertNotIn(
                "supplemental_training_source_manifest",
                result["reference_training_source_pool"],
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

    def test_supplements_one_prior_production_source_with_explicit_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            acquisition_before = acquisition.read_bytes()
            prior_bytes = prior_path.read_bytes()
            prior = json.loads(prior_bytes)

            result = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=(),
                expected_artist="IVE",
                expected_song="BANG BANG",
                expected_source_count=9,
                supplemental_training_manifest_path=prior_path,
                supplemental_source_ids=("S9",),
            )

            self.assertEqual(acquisition.read_bytes(), acquisition_before)
            contract = result["reference_training_source_pool"]
            lineage = contract["supplemental_training_source_manifest"]
            current = json.loads(acquisition_before)
            current_s5 = next(
                item for item in current["selected"] if item["video_id"] == "S5"
            )
            prior_s5 = next(
                item for item in prior["selected"] if item["video_id"] == "S5"
            )
            result_s5 = next(
                item for item in result["selected"] if item["video_id"] == "S5"
            )
            self.assertNotEqual(
                prior_s5["content_integrity"], current_s5["content_integrity"]
            )
            self.assertEqual(
                result_s5["content_integrity"], current_s5["content_integrity"]
            )
            self.assertEqual(contract["source_ids"], [
                *[f"S{index}" for index in range(8)],
                "S9",
            ])
            self.assertEqual(
                lineage["file_sha256"], hashlib.sha256(prior_bytes).hexdigest()
            )
            self.assertEqual(
                lineage["manifest_content_sha256"],
                prior["reference_training_source_pool"]["manifest_content_sha256"],
            )
            self.assertEqual(lineage["source_ids"], ["S9"])
            self.assertTrue(
                lineage["historical_acquisition_binding_drifted_from_current"]
            )
            self.assertFalse(
                lineage["unrequested_historical_sources_live_revalidated"]
            )
            self.assertIn(
                "paths_may_have_drifted",
                lineage[
                    "unrequested_historical_sources_live_revalidation_reason"
                ],
            )
            admission = result["selected"][-1]["training_admission"]
            self.assertTrue(result["selected"][-1]["group_quality_eligible"])
            self.assertTrue(admission["prior_manifest_production_selected"])
            self.assertTrue(
                admission["prior_manifest_production_group_quality_eligible"]
            )
            self.assertFalse(admission["current_production_render_admission"])
            self.assertEqual(
                validate_reference_training_source_manifest(
                    result,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    verify_acquisition_binding=True,
                ),
                (*[f"S{index}" for index in range(8)], "S9"),
            )

    def test_supplemental_manifest_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            payload = json.loads(prior_path.read_text())
            payload["selected"][-1]["title"] = "tampered"
            prior_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            result = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=(),
                expected_artist="IVE",
                expected_song="BANG BANG",
                expected_source_count=9,
                supplemental_training_manifest_path=prior_path,
                supplemental_source_ids=("S9",),
            )
            prior_path.write_bytes(prior_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "lineage changed"):
                validate_reference_training_source_manifest(
                    result,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    verify_acquisition_binding=True,
                )

    def test_supplemental_ids_are_explicit_unique_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            cases = (
                ({"supplemental_source_ids": ("MISSING",), "expected_source_count": 9}, "absent"),
                ({"supplemental_source_ids": ("S9", "S9"), "expected_source_count": 10}, "unique"),
                ({"supplemental_source_ids": ("S0",), "expected_source_count": 9}, "overlap"),
                ({
                    "additional_source_ids": ("S9",),
                    "supplemental_source_ids": ("S9",),
                    "expected_source_count": 10,
                }, "overlap"),
            )
            for overrides, message in cases:
                arguments = {
                    "additional_source_ids": (),
                    "expected_artist": "IVE",
                    "expected_song": "BANG BANG",
                    "expected_source_count": 9,
                    "supplemental_training_manifest_path": prior_path,
                }
                arguments.update(overrides)
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(ValueError, message):
                        build_reference_training_source_manifest(
                            acquisition, **arguments
                        )

            with self.assertRaisesRegex(ValueError, "paired"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=8,
                    supplemental_training_manifest_path=prior_path,
                )
            with self.assertRaisesRegex(ValueError, "paired"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_source_ids=("S9",),
                )

    def test_supplemental_live_media_and_hard_gates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, media = self._supplemental_fixture(root)
            original = media["S9"].read_bytes()
            media["S9"].write_bytes(b"x" * len(original))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, media = self._supplemental_fixture(root)
            media["S9"].unlink()
            with self.assertRaises(FileNotFoundError):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, media = self._supplemental_fixture(root)
            original = media["S9"].read_bytes()
            replacement = root / "replacement.mp4"
            replacement.write_bytes(original)
            media["S9"].unlink()
            media["S9"].symlink_to(replacement)
            with self.assertRaisesRegex(FileNotFoundError, "non-symlink"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            payload = json.loads(prior_path.read_text())
            next(
                item for item in payload["selected"] if item["video_id"] == "S9"
            )["aspect_validation"]["passed"] = False
            self._reseal_training_manifest(payload)
            prior_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aspect contract"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, _media = self._supplemental_fixture(root)
            payload = json.loads(prior_path.read_text())
            next(
                item for item in payload["selected"] if item["video_id"] == "S9"
            )["media_validation"]["passed"] = False
            self._reseal_training_manifest(payload)
            prior_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hard media contract"):
                build_reference_training_source_manifest(
                    acquisition,
                    additional_source_ids=(),
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=9,
                    supplemental_training_manifest_path=prior_path,
                    supplemental_source_ids=("S9",),
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

    def test_supplemental_catalog_provenance_is_consistent_for_both_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition, prior_path, media = self._supplemental_fixture(root)
            media["S8"].write_bytes(b"current-training-extra-eight")
            payload = json.loads(acquisition.read_text())
            current_extra = _entry(media["S8"], "S8", eligible=False)
            payload["candidates"] = [
                current_extra if item["video_id"] == "S8" else item
                for item in payload["candidates"]
            ]
            acquisition.write_text(json.dumps(payload), encoding="utf-8")
            manifest = build_reference_training_source_manifest(
                acquisition,
                additional_source_ids=("S8",),
                expected_artist="IVE",
                expected_song="BANG BANG",
                expected_source_count=10,
                supplemental_training_manifest_path=prior_path,
                supplemental_source_ids=("S9",),
            )
            selected = {item["video_id"]: item for item in manifest["selected"]}
            production_ids = manifest["reference_training_source_pool"][
                "production_selected_source_ids"
            ]
            seed = {
                "sources": [
                    {
                        "video_id": video_id,
                        "path": selected[video_id]["local_path"],
                        "clock": {"mapping_mode": "seed"},
                        "provenance": {
                            "seed_marker": video_id,
                            "production_group_quality_eligible": False,
                            "source_lineage": "stale_seed_value",
                            "prior_manifest_production_selected": False,
                        },
                    }
                    for video_id in production_ids
                ]
            }

            coarse = Mock()
            coarse.offset_seconds = 1.25
            coarse.score = 0.91
            coarse.score_margin = 0.22
            coarse.to_dict.return_value = {
                "offset_samples": 10_000,
                "offset_seconds": 1.25,
                "score": 0.91,
                "second_best_score": 0.69,
                "score_margin": 0.22,
                "sample_rate": 8_000,
                "backend": "test_spectral_flux",
            }
            warp = Mock()
            warp.scalar_fallback = False
            warp.fallback_reason = None
            warp.to_dict.return_value = {
                "backend": "test_local_timewarp",
                "global_offset_seconds": 1.25,
                "scalar_fallback": False,
                "fallback_reason": None,
                "control_points": [
                    {"master_seconds": 0.0, "source_seconds": 1.25},
                    {"master_seconds": 100.0, "source_seconds": 101.25},
                ],
            }
            capture = Mock()
            capture.isOpened.return_value = True

            def capture_value(key: int) -> float:
                import cv2

                return {
                    cv2.CAP_PROP_FPS: 30.0,
                    cv2.CAP_PROP_FRAME_COUNT: 6000.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
                }[key]

            capture.get.side_effect = capture_value
            patches = (
                patch(
                    "crosscut.audio.extract_mono_pcm",
                    return_value=np.ones(160_000, dtype=np.float32),
                ),
                patch(
                    "crosscut.audio.spectral_flux_alignment_offset",
                    return_value=coarse,
                ),
                patch(
                    "crosscut.timewarp.estimate_local_timewarp",
                    return_value=warp,
                ),
                patch("cv2.VideoCapture", return_value=capture),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                extended = build_synchronized_training_source_catalog(
                    seed,
                    manifest,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                )
                from_manifest = build_synchronized_training_source_catalog_from_manifest(
                    manifest,
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                )

            for catalog in (extended, from_manifest):
                provenance = {
                    item["video_id"]: item["provenance"] for item in catalog
                }
                for video_id in production_ids:
                    self.assertTrue(
                        provenance[video_id]["production_group_quality_eligible"]
                    )
                    self.assertTrue(
                        provenance[video_id]["current_production_render_admission"]
                    )
                    self.assertTrue(
                        provenance[video_id]["training_attribution_eligible"]
                    )
                    self.assertNotIn("source_lineage", provenance[video_id])
                    self.assertNotIn(
                        "prior_manifest_production_selected", provenance[video_id]
                    )
                self.assertFalse(
                    provenance["S8"]["production_group_quality_eligible"]
                )
                self.assertFalse(
                    provenance["S8"]["current_production_render_admission"]
                )
                self.assertTrue(provenance["S8"]["training_attribution_eligible"])
                self.assertTrue(
                    provenance["S9"]["production_group_quality_eligible"]
                )
                self.assertFalse(
                    provenance["S9"]["current_production_render_admission"]
                )
                self.assertTrue(provenance["S9"]["training_attribution_eligible"])
                self.assertEqual(
                    provenance["S9"]["source_lineage"],
                    "verified_historical_derived_training_source_manifest",
                )
                self.assertTrue(
                    provenance["S9"]["prior_manifest_production_selected"]
                )
            self.assertEqual(extended[0]["provenance"]["seed_marker"], "S0")

    def test_legacy_seed_catalog_shape_is_unchanged(self) -> None:
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
            sources = [
                {
                    "video_id": entry["video_id"],
                    "path": entry["local_path"],
                    "clock": {"mapping_mode": "legacy_seed"},
                    "provenance": {"legacy_key": entry["video_id"]},
                    "legacy_extra": [entry["video_id"]],
                }
                for entry in manifest["selected"]
            ]
            expected = tuple(deepcopy(sources))
            with patch(
                "crosscut.audio.extract_mono_pcm",
                return_value=np.ones(160_000, dtype=np.float32),
            ):
                catalog = build_synchronized_training_source_catalog(
                    {"sources": sources},
                    manifest,
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                )
            self.assertEqual(catalog, expected)

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
            for item in catalog:
                self.assertNotIn(
                    "current_production_render_admission", item["provenance"]
                )
                self.assertNotIn(
                    "training_attribution_eligible", item["provenance"]
                )
                self.assertNotIn("source_lineage", item["provenance"])
                self.assertNotIn(
                    "prior_manifest_production_selected", item["provenance"]
                )

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
