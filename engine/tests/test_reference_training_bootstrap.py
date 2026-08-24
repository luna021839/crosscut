from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.audio import OffsetEstimate
from crosscut.professional_reference_cadence import (
    AUDIT_ANALYSIS_BASIS,
    CadenceConfig,
    ReferenceCadenceInput,
    build_professional_reference_cadence_profile,
)
from crosscut.reference_training_bootstrap import run_reference_training_bootstrap


class _IdentityWarp:
    scalar_fallback = False
    fallback_reason = None

    def source_to_master(self, source_seconds: float) -> float:
        return source_seconds

    def to_dict(self) -> dict:
        return {
            "backend": "synthetic_local_timewarp",
            "global_offset_seconds": 0.0,
            "master_duration_seconds": 30.0,
            "source_duration_seconds": 30.0,
            "scalar_fallback": False,
            "fallback_reason": None,
            "control_points": [
                {"master_seconds": 0.0, "source_seconds": 0.0},
                {"master_seconds": 30.0, "source_seconds": 30.0},
            ],
        }


def _offset() -> OffsetEstimate:
    return OffsetEstimate(
        offset_samples=0,
        offset_seconds=0.0,
        score=0.95,
        overlap_samples=3_000,
        overlap_seconds=30.0,
        sample_rate=100,
        second_best_score=0.60,
        score_margin=0.35,
        overlap_ratio=1.0,
        backend="spectral_flux_ncc",
    )


def _audit(path: Path, media: Path) -> None:
    cuts = [
        {
            "frame_index": int(seconds * 30),
            "seconds": seconds,
            "score": 70.0,
            "discontinuities": {},
            "issue_reasons": [],
        }
        for seconds in (5.0, 10.0, 15.0)
    ]
    path.write_text(
        json.dumps(
            {
                "analysis_basis": AUDIT_ANALYSIS_BASIS,
                "reference": str(media.resolve()),
                "summary": {"cut_count": len(cuts)},
                "cuts": cuts,
            }
        ),
        encoding="utf-8",
    )


def _entry(path: Path, video_id: str) -> dict:
    return {
        "video_id": video_id,
        "local_path": str(path.resolve()),
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
        "group_quality_eligible": True,
        "group_quality_exclusion_reasons": [],
    }


class ReferenceTrainingBootstrapTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        sources = []
        for index in range(10):
            media = root / f"S{index}.mp4"
            media.write_bytes(f"source-{index}".encode())
            sources.append(_entry(media, f"S{index}"))
        acquisition = root / "acquisition.json"
        acquisition.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "group": "IVE",
                    "song": "BLACKHOLE",
                    "selected": deepcopy(sources),
                    "candidates": deepcopy(sources),
                }
            ),
            encoding="utf-8",
        )
        references = []
        for index in range(5):
            media = root / f"R{index}.mp4"
            media.write_bytes(f"reference-{index}".encode())
            audit = root / f"R{index}-audit.json"
            _audit(audit, media)
            references.append(ReferenceCadenceInput(f"R{index}", media, audit))
        with (
            patch(
                "crosscut.professional_reference_cadence.extract_mono_pcm",
                return_value=np.zeros(3_000, dtype=np.float32),
            ),
            patch(
                "crosscut.professional_reference_cadence.spectral_flux_alignment_offset",
                return_value=_offset(),
            ),
            patch(
                "crosscut.professional_reference_cadence.estimate_local_timewarp",
                return_value=_IdentityWarp(),
            ),
        ):
            profile = build_professional_reference_cadence_profile(
                Path(sources[0]["local_path"]),
                references,
                artist="IVE",
                song="BLACKHOLE",
                config=CadenceConfig(minimum_coarse_overlap_seconds=10.0),
                sample_rate=100,
            )
        cadence = profile.write_json(root / "cadence.json")
        return cadence, acquisition

    def test_dry_run_plans_exact_ten_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cadence, acquisition = self._inputs(root)
            output = root / "output"

            result = run_reference_training_bootstrap(
                cadence_profile_path=cadence,
                acquisition_manifest_path=acquisition,
                output_directory=output,
                style_id="blackhole-human-reference-10source-v2",
                expected_artist="IVE",
                expected_song="BLACKHOLE",
                dry_run=True,
            )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["source_count"], 10)
            self.assertEqual(result["seed_reference_id"], "R0")
            self.assertFalse(result["writes_performed"])
            self.assertFalse(output.exists())

    def test_real_run_is_immutable_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cadence, acquisition = self._inputs(root)
            output = root / "output"
            sources = []
            acquisition_payload = json.loads(acquisition.read_text())
            for index, item in enumerate(acquisition_payload["selected"]):
                sources.append(
                    {
                        "video_id": item["video_id"],
                        "path": item["local_path"],
                        "clock": {
                            "basis": "synthetic_identity",
                            "mapping_mode": "scalar_offset",
                            "global_offset_seconds": 0.0,
                            "global_offset_application": "added_to_master_seconds",
                            "control_points": [],
                            "provenance": {},
                        },
                        "provenance": {"identity_master": index == 0},
                        "media_probe": {
                            "frame_rate": 30.0,
                            "declared_frame_count": 900,
                            "width": 1920,
                            "height": 1080,
                            "size_bytes": Path(item["local_path"]).stat().st_size,
                            "mtime_ns": Path(item["local_path"]).stat().st_mtime_ns,
                        },
                    }
                )
            evidence = {"synthetic": "seed"}
            initial = {"synthetic": "initial"}
            boundary = {"synthetic": "boundary"}
            final = {"synthetic": "final"}
            dataset = {"synthetic": "dataset"}
            supervision = {"synthetic": "supervision"}

            with (
                patch(
                    "crosscut.reference_training_bootstrap."
                    "build_synchronized_training_source_catalog_from_manifest",
                    return_value=tuple(sources),
                ) as catalog_builder,
                patch(
                    "crosscut.reference_training_bootstrap.build_reference_evidence",
                    return_value=SimpleNamespace(payload=evidence),
                ) as evidence_builder,
                patch(
                    "crosscut.reference_training_bootstrap._validate_seed_evidence"
                ),
                patch(
                    "crosscut.reference_training_bootstrap."
                    "resolve_reference_attribution_payload",
                    side_effect=[initial, final, initial, final],
                ),
                patch(
                    "crosscut.reference_training_bootstrap."
                    "augment_reference_boundary_requests",
                    return_value=SimpleNamespace(payload=boundary),
                ),
                patch(
                    "crosscut.reference_training_bootstrap."
                    "_expansion_validate_boundary_evidence"
                ),
                patch(
                    "crosscut.reference_training_bootstrap."
                    "_expansion_validate_dataset_payload"
                ),
                patch(
                    "crosscut.reference_training_bootstrap.build_reference_dataset",
                    return_value=SimpleNamespace(payload=dataset),
                ),
                patch(
                    "crosscut.reference_training_bootstrap.build_reference_supervision",
                    return_value=SimpleNamespace(payload=supervision),
                ),
            ):
                first = run_reference_training_bootstrap(
                    cadence_profile_path=cadence,
                    acquisition_manifest_path=acquisition,
                    output_directory=output,
                    style_id="blackhole-human-reference-10source-v2",
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                )
                second = run_reference_training_bootstrap(
                    cadence_profile_path=cadence,
                    acquisition_manifest_path=acquisition,
                    output_directory=output,
                    style_id="blackhole-human-reference-10source-v2",
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                )

            self.assertTrue(first["writes_performed"])
            self.assertFalse(second["writes_performed"])
            catalog_builder.assert_called_once()
            evidence_builder.assert_called_once()
            self.assertTrue(all(
                item["action"] == "validated_existing"
                for item in second["artifacts"].values()
            ))


if __name__ == "__main__":
    unittest.main()
