from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut import reference_training_bootstrap as bootstrap
from crosscut.audio import OffsetEstimate
from crosscut.professional_reference_cadence import (
    AUDIT_ANALYSIS_BASIS,
    CadenceConfig,
    ReferenceCadenceInput,
    build_professional_reference_cadence_profile,
)
from crosscut.reference_training_bootstrap import run_reference_training_bootstrap
from crosscut.reference_training_source_pool import (
    build_reference_training_source_manifest,
)


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

    def _catalog_sources(
        self, acquisition_payload: dict, *, count: int | None = None
    ) -> tuple[dict, ...]:
        selected = acquisition_payload["selected"]
        if count is not None:
            selected = selected[:count]
        return tuple(
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
                "provenance": {
                    "identity_master": index == 0,
                    "current_acquisition_selected": True,
                    "content_sha256": item["content_integrity"]["sha256"],
                },
                "media_probe": {
                    "frame_rate": 30.0,
                    "declared_frame_count": 900,
                    "width": 1920,
                    "height": 1080,
                    "size_bytes": Path(item["local_path"]).stat().st_size,
                    "mtime_ns": Path(item["local_path"]).stat().st_mtime_ns,
                },
            }
            for index, item in enumerate(selected)
        )

    def _supplemental_inputs(
        self, root: Path
    ) -> tuple[Path, Path, Path, Path, tuple[dict, ...]]:
        cadence, acquisition = self._inputs(root)
        complete_acquisition = json.loads(acquisition.read_text(encoding="utf-8"))
        prior_manifest = build_reference_training_source_manifest(
            acquisition,
            additional_source_ids=(),
            expected_artist="IVE",
            expected_song="BLACKHOLE",
            expected_source_count=10,
        )
        prior_path = root / "prior-training-manifest.json"
        prior_path.write_text(json.dumps(prior_manifest), encoding="utf-8")

        current_acquisition = deepcopy(complete_acquisition)
        current_acquisition["selected"] = current_acquisition["selected"][:8]
        current_acquisition["candidates"] = current_acquisition["candidates"][:8]
        acquisition.write_text(json.dumps(current_acquisition), encoding="utf-8")
        production_sources = self._catalog_sources(
            current_acquisition, count=8
        )
        profile_payload = json.loads(cadence.read_text(encoding="utf-8"))
        profile_provenance = profile_payload["provenance"]
        reference = profile_provenance["references"][0]
        reference_clock_model = bootstrap.ReferenceSourceClock.from_mapping(
            reference["local_timewarp"]
        )
        reference_clock = reference_clock_model.to_dict()
        master_duration = profile_provenance["master"][
            "decoded_audio_duration_seconds"
        ]
        reference_end = reference_clock_model.source_to_master(
            reference["decoded_audio_duration_seconds"]
        )
        acquisition_sha256 = hashlib.sha256(acquisition.read_bytes()).hexdigest()
        production_ids = [
            item["video_id"] for item in current_acquisition["selected"]
        ]
        cut_count = reference["audit_cut_count"]
        production_seed = root / "production-seed.json"
        production_seed.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis": "reference_hard_cut_reverse_match_evidence_v1",
                    "reference": reference["media_path"],
                    "master_start_seconds": 0.0,
                    "master_end_seconds": reference_end,
                    "sources": production_sources,
                    "clock_policy": {
                        "master_start_seconds": 0.0,
                        "master_end_seconds": reference_end,
                        "attribution_sample_master_end_seconds": master_duration,
                        "reference_mapping": reference_clock,
                    },
                    "reference_video": {"path": reference["media_path"]},
                    "detected_reference_cuts": [
                        {"cut_index": index} for index in range(cut_count)
                    ],
                    "excluded_detected_reference_cuts": [],
                    "boundary_requests": [],
                    "statistics": {
                        "source_count": len(production_ids),
                        "hard_cut_count": cut_count,
                        "detector_hard_cut_count": cut_count,
                        "excluded_detected_hard_cut_count": 0,
                    },
                    "provenance": {
                        "caller": {
                            "artist": "IVE",
                            "song": "BLACKHOLE",
                            "reference_video_id": reference["video_id"],
                            "purpose": (
                                "same_song_professional_reference_learning_"
                                "current_canonical_production_pool"
                            ),
                            "acquisition_manifest": {
                                "path": str(acquisition.resolve()),
                                "file_sha256": acquisition_sha256,
                            },
                            "production_source_ids": production_ids,
                            "production_render_admission_modified": False,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return (
            cadence,
            acquisition,
            prior_path,
            production_seed,
            self._catalog_sources(complete_acquisition),
        )

    @contextmanager
    def _mock_downstream(
        self,
        catalog_sources: tuple[dict, ...],
    ):
        evidence = {"synthetic": "seed"}
        initial = {"synthetic": "initial"}
        boundary = {"synthetic": "boundary"}
        final = {"synthetic": "final"}
        dataset = {"synthetic": "dataset"}
        supervision = {"synthetic": "supervision"}
        with ExitStack() as stack:
            seeded_catalog_builder = stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "build_synchronized_training_source_catalog",
                    return_value=catalog_sources,
                )
            )
            legacy_catalog_builder = stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "build_synchronized_training_source_catalog_from_manifest",
                    return_value=catalog_sources,
                )
            )
            evidence_builder = stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap.build_reference_evidence",
                    return_value=SimpleNamespace(payload=evidence),
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap._validate_seed_evidence"
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "resolve_reference_attribution_payload",
                    side_effect=[initial, final] * 8,
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "augment_reference_boundary_requests",
                    return_value=SimpleNamespace(payload=boundary),
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "_expansion_validate_boundary_evidence"
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "_expansion_validate_dataset_payload"
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap.build_reference_dataset",
                    return_value=SimpleNamespace(payload=dataset),
                )
            )
            stack.enter_context(
                patch(
                    "crosscut.reference_training_bootstrap."
                    "build_reference_supervision",
                    return_value=SimpleNamespace(payload=supervision),
                )
            )
            yield SimpleNamespace(
                seeded_catalog_builder=seeded_catalog_builder,
                legacy_catalog_builder=legacy_catalog_builder,
                evidence_builder=evidence_builder,
            )

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
            self.assertNotIn("production_seed_evidence", first)
            self.assertNotIn("production_seed_evidence", second)
            catalog = json.loads(
                Path(first["artifacts"]["catalog"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn(
                "production_seed_evidence", catalog["provenance"]
            )

    def test_supplemental_production_seed_routes_and_binds_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                cadence,
                acquisition,
                prior_manifest,
                production_seed,
                catalog_sources,
            ) = self._supplemental_inputs(root)
            output = root / "output"
            kwargs = {
                "cadence_profile_path": cadence,
                "acquisition_manifest_path": acquisition,
                "output_directory": output,
                "style_id": "blackhole-human-reference-10source-v3",
                "expected_artist": "IVE",
                "expected_song": "BLACKHOLE",
                "supplemental_training_manifest_path": prior_manifest,
                "supplemental_source_ids": ("S8", "S9"),
                "production_seed_evidence_path": production_seed,
                "production_seed_evidence_sha256": hashlib.sha256(
                    production_seed.read_bytes()
                ).hexdigest(),
            }

            with self._mock_downstream(catalog_sources) as mocks:
                first = run_reference_training_bootstrap(**kwargs)
                second = run_reference_training_bootstrap(**kwargs)

            mocks.seeded_catalog_builder.assert_called_once()
            mocks.legacy_catalog_builder.assert_not_called()
            mocks.evidence_builder.assert_called_once()
            seed_payload = json.loads(production_seed.read_text(encoding="utf-8"))
            self.assertEqual(
                seed_payload,
                mocks.seeded_catalog_builder.call_args.args[0],
            )
            self.assertEqual(
                {
                    "expected_artist": "IVE",
                    "expected_song": "BLACKHOLE",
                },
                mocks.seeded_catalog_builder.call_args.kwargs,
            )
            routed_manifest = mocks.seeded_catalog_builder.call_args.args[1]
            self.assertEqual(
                [f"S{index}" for index in range(10)],
                [item["video_id"] for item in routed_manifest["selected"]],
            )
            self.assertEqual(
                ("S8", "S9"),
                tuple(
                    routed_manifest["reference_training_source_pool"][
                        "supplemental_source_ids"
                    ]
                ),
            )
            expected_binding = {
                "path": str(production_seed.resolve()),
                "file_sha256": hashlib.sha256(
                    production_seed.read_bytes()
                ).hexdigest(),
            }
            self.assertEqual(expected_binding, first["production_seed_evidence"])
            self.assertEqual(expected_binding, second["production_seed_evidence"])
            catalog = json.loads(
                Path(first["artifacts"]["catalog"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                expected_binding,
                catalog["provenance"]["production_seed_evidence"],
            )
            self.assertFalse(second["writes_performed"])
            self.assertTrue(
                all(
                    item["action"] == "validated_existing"
                    for item in second["artifacts"].values()
                )
            )

    def test_production_seed_and_catalog_binding_drift_fail_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                cadence,
                acquisition,
                prior_manifest,
                production_seed,
                catalog_sources,
            ) = self._supplemental_inputs(root)
            output = root / "output"
            kwargs = {
                "cadence_profile_path": cadence,
                "acquisition_manifest_path": acquisition,
                "output_directory": output,
                "style_id": "blackhole-human-reference-10source-v3",
                "expected_artist": "IVE",
                "expected_song": "BLACKHOLE",
                "supplemental_training_manifest_path": prior_manifest,
                "supplemental_source_ids": ("S8", "S9"),
                "production_seed_evidence_path": production_seed,
                "production_seed_evidence_sha256": hashlib.sha256(
                    production_seed.read_bytes()
                ).hexdigest(),
            }
            original_seed = production_seed.read_bytes()

            with self._mock_downstream(catalog_sources):
                first = run_reference_training_bootstrap(**kwargs)
                tampered_seed = json.loads(original_seed)
                tampered_seed["tampered"] = True
                production_seed.write_text(
                    json.dumps(tampered_seed), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, "production seed evidence SHA-256 mismatch"
                ):
                    run_reference_training_bootstrap(**kwargs)

                production_seed.write_bytes(original_seed)
                replacement_seed = root / "replacement-production-seed.json"
                replacement_seed.write_bytes(original_seed)
                with self.assertRaisesRegex(
                    ValueError, "production seed evidence binding mismatch"
                ):
                    run_reference_training_bootstrap(
                        **{
                            **kwargs,
                            "production_seed_evidence_path": replacement_seed,
                        }
                    )

                catalog_path = Path(first["artifacts"]["catalog"]["path"])
                catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                catalog["provenance"]["production_seed_evidence"][
                    "file_sha256"
                ] = "0" * 64
                catalog["provenance"]["content_sha256"] = None
                catalog["provenance"][
                    "content_sha256"
                ] = bootstrap._catalog_content_sha256(catalog)
                catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "production seed evidence binding mismatch"
                ):
                    run_reference_training_bootstrap(**kwargs)

    def test_cli_routes_supplemental_and_production_seed_options(self) -> None:
        with patch.object(
            bootstrap,
            "run_reference_training_bootstrap",
            return_value={"status": "ready"},
        ) as run, patch("builtins.print"):
            result = bootstrap.main(
                [
                    "--cadence-profile",
                    "/tmp/cadence.json",
                    "--acquisition-manifest",
                    "/tmp/acquisition.json",
                    "--output-directory",
                    "/tmp/output",
                    "--style-id",
                    "style",
                    "--expected-artist",
                    "IVE",
                    "--expected-song",
                    "BLACKHOLE",
                    "--supplemental-training-manifest",
                    "/tmp/prior.json",
                    "--supplemental-source-id",
                    "S8",
                    "--supplemental-source-id",
                    "S9",
                    "--production-seed-evidence",
                    "/tmp/production-seed.json",
                    "--production-seed-evidence-sha256",
                    "a" * 64,
                    "--dry-run",
                ]
            )

        self.assertEqual(0, result)
        routed = run.call_args.kwargs
        self.assertEqual(Path("/tmp/prior.json"), routed["supplemental_training_manifest_path"])
        self.assertEqual(("S8", "S9"), routed["supplemental_source_ids"])
        self.assertEqual(
            Path("/tmp/production-seed.json"),
            routed["production_seed_evidence_path"],
        )
        self.assertEqual("a" * 64, routed["production_seed_evidence_sha256"])

    def test_production_seed_expected_sha_is_external_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cadence, acquisition = self._inputs(root)
            seed = root / "seed.json"
            seed.write_text(json.dumps({"sources": []}), encoding="utf-8")
            seed_sha256 = hashlib.sha256(seed.read_bytes()).hexdigest()
            common = {
                "cadence_profile_path": cadence,
                "acquisition_manifest_path": acquisition,
                "output_directory": root / "output",
                "style_id": "blackhole-human-reference-10source-v3",
                "expected_artist": "IVE",
                "expected_song": "BLACKHOLE",
                "dry_run": True,
            }

            with self.assertRaisesRegex(ValueError, "must be paired"):
                run_reference_training_bootstrap(
                    **common,
                    production_seed_evidence_path=seed,
                )
            with self.assertRaisesRegex(ValueError, "must be paired"):
                run_reference_training_bootstrap(
                    **common,
                    production_seed_evidence_sha256=seed_sha256,
                )
            with self.assertRaisesRegex(ValueError, "64 lowercase hex"):
                run_reference_training_bootstrap(
                    **common,
                    production_seed_evidence_path=seed,
                    production_seed_evidence_sha256="A" * 64,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_reference_training_bootstrap(
                    **common,
                    production_seed_evidence_path=seed,
                    production_seed_evidence_sha256="0" * 64,
                )

    def test_production_seed_semantic_identity_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                cadence,
                acquisition,
                prior_manifest,
                production_seed,
                _catalog_sources,
            ) = self._supplemental_inputs(root)
            original = json.loads(production_seed.read_text(encoding="utf-8"))
            common = {
                "cadence_profile_path": cadence,
                "acquisition_manifest_path": acquisition,
                "output_directory": root / "output",
                "style_id": "blackhole-human-reference-10source-v3",
                "expected_artist": "IVE",
                "expected_song": "BLACKHOLE",
                "supplemental_training_manifest_path": prior_manifest,
                "supplemental_source_ids": ("S8", "S9"),
                "production_seed_evidence_path": production_seed,
                "dry_run": True,
            }

            def assert_rejected(payload: dict, pattern: str) -> None:
                production_seed.write_text(json.dumps(payload), encoding="utf-8")
                expected_sha256 = hashlib.sha256(
                    production_seed.read_bytes()
                ).hexdigest()
                with self.assertRaisesRegex(ValueError, pattern):
                    run_reference_training_bootstrap(
                        **common,
                        production_seed_evidence_sha256=expected_sha256,
                    )

            caller_mismatch = deepcopy(original)
            caller_mismatch["provenance"]["caller"]["artist"] = "OTHER"
            assert_rejected(caller_mismatch, "caller provenance mismatch")

            acquisition_mismatch = deepcopy(original)
            acquisition_mismatch["provenance"]["caller"][
                "acquisition_manifest"
            ]["file_sha256"] = "0" * 64
            assert_rejected(acquisition_mismatch, "caller provenance mismatch")

            reference_mismatch = deepcopy(original)
            reference_mismatch["reference"] = str(root / "unrelated.mp4")
            assert_rejected(reference_mismatch, "reference path mismatch")

            reference_clock_mismatch = deepcopy(original)
            reference_clock_mismatch["clock_policy"]["reference_mapping"][
                "global_offset_seconds"
            ] = 0.25
            assert_rejected(reference_clock_mismatch, "reference clock mismatch")

            order_mismatch = deepcopy(original)
            order_mismatch["sources"][0], order_mismatch["sources"][1] = (
                order_mismatch["sources"][1],
                order_mismatch["sources"][0],
            )
            assert_rejected(order_mismatch, "source order differs")

            path_mismatch = deepcopy(original)
            path_mismatch["sources"][1]["path"] = path_mismatch["sources"][0][
                "path"
            ]
            assert_rejected(path_mismatch, "source path mismatch")

            size_mismatch = deepcopy(original)
            size_mismatch["sources"][1]["media_probe"]["size_bytes"] += 1
            assert_rejected(size_mismatch, "media size mismatch")

            cut_count_mismatch = deepcopy(original)
            cut_count_mismatch["detected_reference_cuts"].pop()
            assert_rejected(cut_count_mismatch, "detected-cut count differs")

    def test_production_seed_canonical_999_second_clock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                cadence,
                acquisition,
                prior_manifest,
                production_seed,
                _catalog_sources,
            ) = self._supplemental_inputs(root)
            payload = json.loads(production_seed.read_text(encoding="utf-8"))
            payload["sources"][1]["clock"] = {
                "basis": "synthetic_but_unrelated",
                "mapping_mode": "scalar_offset",
                "global_offset_seconds": 999.0,
                "global_offset_application": "added_to_master_seconds",
                "control_points": [],
                "provenance": {},
            }
            production_seed.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "insufficient media overlap"):
                run_reference_training_bootstrap(
                    cadence_profile_path=cadence,
                    acquisition_manifest_path=acquisition,
                    output_directory=root / "output",
                    style_id="blackhole-human-reference-10source-v3",
                    expected_artist="IVE",
                    expected_song="BLACKHOLE",
                    supplemental_training_manifest_path=prior_manifest,
                    supplemental_source_ids=("S8", "S9"),
                    production_seed_evidence_path=production_seed,
                    production_seed_evidence_sha256=hashlib.sha256(
                        production_seed.read_bytes()
                    ).hexdigest(),
                    dry_run=True,
                )

    def test_supplemental_pairing_and_seed_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cadence, acquisition = self._inputs(root)
            common = {
                "cadence_profile_path": cadence,
                "acquisition_manifest_path": acquisition,
                "output_directory": root / "output",
                "style_id": "blackhole-human-reference-10source-v3",
                "expected_artist": "IVE",
                "expected_song": "BLACKHOLE",
                "dry_run": True,
            }
            with self.assertRaisesRegex(
                ValueError, "manifest path and source ids must be paired"
            ):
                run_reference_training_bootstrap(
                    **common,
                    supplemental_source_ids=("S9",),
                )

            seed = root / "seed.json"
            seed.write_text(json.dumps({"sources": []}), encoding="utf-8")
            seed_link = root / "seed-link.json"
            seed_link.symlink_to(seed)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                run_reference_training_bootstrap(
                    **common,
                    production_seed_evidence_path=seed_link,
                    production_seed_evidence_sha256=hashlib.sha256(
                        seed.read_bytes()
                    ).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
