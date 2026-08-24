from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from crosscut.reference_expansion_job import (
    ANALYSIS_BASIS,
    PIPELINE_REVISION,
    ActiveAnalysisError,
    ArtifactPaths,
    ExistingArtifactValidationError,
    ExpansionInputs,
    MediaIdentity,
    MINIMUM_EXPECTED_SOURCE_COUNT,
    ReferenceSpec,
    SourceSpec,
    _artifact_paths,
    _atomic_update_json,
    _cleanup_stale_temps,
    _load_existing_artifact,
    _materialize_or_skip,
    _normalize_boundary_timestamp,
    _publish_json_no_replace,
    _recover_reserved_artifact,
    _reservation_document,
    _run_one_reference,
    _validate_boundary_evidence,
    _validate_detector_cut_accounting,
    _write_json_temporary,
    run_reference_expansion_job,
)
from crosscut.reference_boundary_supervision import (
    ANALYSIS_BASIS as BOUNDARY_ANALYSIS_BASIS,
)
from engine.tests.test_reference_boundary_supervision import _evidence


def _identity(path: Path, digest: str = "a" * 64) -> MediaIdentity:
    info = path.stat()
    return MediaIdentity(
        path=path.resolve(),
        sha256=digest,
        size_bytes=info.st_size,
        stat_identity=(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
    )


class ImmutableArtifactTests(unittest.TestCase):
    def test_detector_cut_accounting_accepts_audited_preroll_exclusion(self) -> None:
        payload = {
            "master_start_seconds": 0.0,
            "master_end_seconds": 2.0,
            "detected_reference_cuts": [
                {"frame_index": 20, "master_seconds": 1.0}
            ],
            "excluded_detected_reference_cuts": [
                {
                    "frame_index": 2,
                    "master_seconds": -0.1,
                    "exclusion_reason": "at_or_before_performance_master_start",
                }
            ],
            "statistics": {
                "hard_cut_count": 1,
                "detector_hard_cut_count": 2,
                "excluded_detected_hard_cut_count": 1,
            },
        }

        cuts, excluded = _validate_detector_cut_accounting(
            payload,
            expected_detector_count=2,
            name="fixture",
        )

        self.assertEqual(len(cuts), 1)
        self.assertEqual(len(excluded), 1)

    def test_job_owned_reservation_recovers_valid_crash_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            payload = {"canonical": True}
            temporary = _write_json_temporary(destination, payload)
            reservation = _reservation_document(temporary, destination)
            destination.write_text(
                json.dumps(reservation),
                encoding="utf-8",
            )

            recovered = _recover_reserved_artifact(
                destination,
                validator=lambda value: (
                    None
                    if value == payload
                    else (_ for _ in ()).throw(ValueError("not canonical"))
                ),
            )

            self.assertIsNotNone(recovered)
            assert recovered is not None
            observed, result = recovered
            self.assertEqual(observed, payload)
            self.assertEqual(result["action"], "recovered_valid_job_temporary")
            self.assertFalse(temporary.exists())

    def test_atomic_publish_creates_new_artifact_without_temp_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"

            written = _publish_json_no_replace(destination, {"canonical": True})

            self.assertEqual(written, destination.resolve())
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"canonical": True},
            )
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")),
                [],
            )

    def test_atomic_publish_refuses_to_replace_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            destination.write_text('{"owner":"existing"}\n', encoding="utf-8")
            original = destination.read_bytes()

            with self.assertRaises(FileExistsError):
                _publish_json_no_replace(destination, {"owner": "new"})

            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")),
                [],
            )

    def test_existing_artifact_is_skipped_only_after_validator_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            destination.write_text('{"canonical":true}\n', encoding="utf-8")
            producer = Mock(side_effect=AssertionError("producer must not run"))

            payload, result = _materialize_or_skip(
                destination,
                producer=producer,
                validator=lambda value: (
                    None
                    if value == {"canonical": True}
                    else (_ for _ in ()).throw(ValueError("not canonical"))
                ),
            )

            self.assertEqual(payload, {"canonical": True})
            self.assertEqual(result["action"], "skipped_valid_existing")
            self.assertEqual(len(result["sha256"]), 64)
            producer.assert_not_called()

            original = destination.read_bytes()
            with self.assertRaises(ExistingArtifactValidationError):
                _materialize_or_skip(
                    destination,
                    producer=producer,
                    validator=lambda _value: (_ for _ in ()).throw(
                        ValueError("tampered")
                    ),
                )
            self.assertEqual(destination.read_bytes(), original)

    def test_stale_cleanup_is_scoped_to_exact_destination_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "wanted.json"
            matching = root / (
                f".{destination.name}.crosscut-reference-expansion-old.tmp"
            )
            unrelated = root / ".other.json.crosscut-reference-expansion-old.tmp"
            matching.write_text("partial", encoding="utf-8")
            unrelated.write_text("user-owned", encoding="utf-8")

            self.assertEqual(_cleanup_stale_temps(destination), 1)

            self.assertFalse(matching.exists())
            self.assertTrue(unrelated.exists())

    def test_mutable_status_update_is_atomic_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "status.json"

            _atomic_update_json(destination, {"state": "running"})
            _atomic_update_json(destination, {"state": "complete"})

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"state": "complete"},
            )
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")),
                [],
            )


class BoundaryResumeValidationTests(unittest.TestCase):
    def _payload(self, created_at: str) -> dict:
        return {
            "schema_version": 1,
            "analysis": "reference_hard_cut_reverse_match_evidence_v1",
            "samples": [],
            "boundary_requests": [],
            "boundary_request_augmentation": {
                "schema_version": 1,
                "analysis": BOUNDARY_ANALYSIS_BASIS,
                "created_at": created_at,
                "added_request_count": 0,
            },
        }

    def test_resume_ignores_only_valid_augmentation_timestamp(self) -> None:
        expected = self._payload("2026-08-24T00:00:00+00:00")
        observed = self._payload("2026-08-24T01:00:00+00:00")

        _validate_boundary_evidence(
            observed,
            base_evidence={"analysis": observed["analysis"]},
            expected=expected,
        )
        self.assertEqual(
            _normalize_boundary_timestamp(observed)[
                "boundary_request_augmentation"
            ]["created_at"],
            "<validated-utc-timestamp>",
        )

        tampered = deepcopy(observed)
        tampered["boundary_request_augmentation"]["added_request_count"] = 1
        with self.assertRaises(ExistingArtifactValidationError):
            _validate_boundary_evidence(
                tampered,
                base_evidence={"analysis": observed["analysis"]},
                expected=expected,
            )

    def test_naive_boundary_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            _normalize_boundary_timestamp(
                self._payload("2026-08-24T00:00:00")
            )


class JobControlTests(unittest.TestCase):
    def test_reference_pipeline_resumes_and_refuses_tampered_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "reference.mp4"
            audit_path = root / "audit.json"
            source_a_path = root / "a.mp4"
            source_b_path = root / "b.mp4"
            for path, content in (
                (media_path, b"reference"),
                (audit_path, b"audit"),
                (source_a_path, b"source-a"),
                (source_b_path, b"source-b"),
            ):
                path.write_bytes(content)
            reference = ReferenceSpec(
                video_id="target",
                media=_identity(media_path),
                audit=_identity(audit_path, "b" * 64),
                audit_cut_count=1,
                reference_clock={"basis": "test", "control_points": []},
            )
            sources = tuple(
                SourceSpec(
                    video_id=video_id,
                    media=_identity(path, digest),
                    clock={"basis": "test", "control_points": []},
                    provenance={},
                    media_probe={"size_bytes": path.stat().st_size},
                )
                for video_id, path, digest in (
                    ("A", source_a_path, "c" * 64),
                    ("B", source_b_path, "d" * 64),
                )
            )
            inputs = ExpansionInputs(
                profile=SimpleNamespace(
                    artist="IVE",
                    song="BANG BANG",
                    profile_content_sha256="e" * 64,
                ),
                profile_payload={},
                profile_path=root / "profile.json",
                profile_file_sha256="f" * 64,
                seed_evidence={},
                seed_evidence_path=root / "seed.json",
                seed_evidence_file_sha256="1" * 64,
                source_manifest_path=root / "manifest.json",
                source_manifest_file_sha256="2" * 64,
                references=(reference,),
                sources=sources,
                seed_reference_id="seed",
                target_reference_ids=("target",),
                style_id="bang-bang-human-reference",
                output_directory=root,
                pipeline_code_sha256={"module.py": "3" * 64},
                job_fingerprint="4" * 64,
                media_sha256_verified=True,
            )
            synthetic = _evidence()
            synthetic["reference"] = str(media_path)
            built = SimpleNamespace(payload=synthetic)

            with patch(
                "crosscut.reference_expansion_job._active_analysis_processes",
                return_value=(),
            ), patch(
                "crosscut.reference_expansion_job._validate_generated_evidence"
            ), patch(
                "crosscut.reference_expansion_job.build_reference_evidence",
                return_value=built,
            ) as evidence_builder:
                first = _run_one_reference(inputs, reference, emit=lambda _event: None)
                second = _run_one_reference(inputs, reference, emit=lambda _event: None)

                self.assertEqual(first["status"], "complete")
                self.assertTrue(
                    all(
                        item["action"] == "created"
                        for item in first["stages"].values()
                    )
                )
                self.assertTrue(
                    all(
                        item["action"] == "skipped_valid_existing"
                        for item in second["stages"].values()
                    )
                )
                evidence_builder.assert_called_once()

                paths = _artifact_paths(root, "target")
                tampered = json.loads(paths.dataset.read_text(encoding="utf-8"))
                tampered["style_id"] = "tampered"
                paths.dataset.write_text(json.dumps(tampered), encoding="utf-8")
                tampered_bytes = paths.dataset.read_bytes()

                with self.assertRaises(ExistingArtifactValidationError):
                    _run_one_reference(inputs, reference, emit=lambda _event: None)
                self.assertEqual(paths.dataset.read_bytes(), tampered_bytes)

            status = json.loads(
                _artifact_paths(root, "target").status_manifest.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "failed")
            self.assertEqual(
                status["failure"]["type"], "ExistingArtifactValidationError"
            )

    def test_real_run_is_blocked_before_input_hashes_or_media_decode(self) -> None:
        with patch(
            "crosscut.reference_expansion_job._active_analysis_processes",
            return_value=({"pid": 4490, "command": "crosscut_cli.py analyze"},),
        ), patch(
            "crosscut.reference_expansion_job._load_inputs"
        ) as load_inputs:
            with self.assertRaises(ActiveAnalysisError):
                run_reference_expansion_job(
                    cadence_profile_path=Path("profile.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=Path("evidence"),
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                )

        load_inputs.assert_not_called()

    def test_dry_run_defers_media_hashes_and_performs_no_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "reference.mp4"
            audit_path = root / "audit.json"
            source_path = root / "source.mp4"
            for path, content in (
                (media_path, b"reference"),
                (audit_path, b"audit"),
                (source_path, b"source"),
            ):
                path.write_bytes(content)
            reference = ReferenceSpec(
                video_id="target",
                media=_identity(media_path),
                audit=_identity(audit_path, "b" * 64),
                audit_cut_count=3,
                reference_clock={"basis": "test", "control_points": []},
            )
            seed_reference = ReferenceSpec(
                video_id="seed",
                media=_identity(media_path),
                audit=_identity(audit_path, "b" * 64),
                audit_cut_count=3,
                reference_clock={"basis": "test", "control_points": []},
            )
            source = SourceSpec(
                video_id="source",
                media=_identity(source_path, "c" * 64),
                clock={"basis": "test", "control_points": []},
                provenance={},
                media_probe={"size_bytes": source_path.stat().st_size},
            )
            profile = SimpleNamespace(
                artist="IVE",
                song="BANG BANG",
                profile_content_sha256="d" * 64,
            )
            inputs = ExpansionInputs(
                profile=profile,
                profile_payload={},
                profile_path=(root / "profile.json"),
                profile_file_sha256="e" * 64,
                seed_evidence={},
                seed_evidence_path=(root / "seed.json"),
                seed_evidence_file_sha256="f" * 64,
                source_manifest_path=(root / "manifest.json"),
                source_manifest_file_sha256="1" * 64,
                references=(seed_reference, reference),
                sources=(source,),
                seed_reference_id="seed",
                target_reference_ids=("target",),
                style_id="bang-bang-human-reference",
                output_directory=root,
                pipeline_code_sha256={"module.py": "2" * 64},
                job_fingerprint="3" * 64,
                media_sha256_verified=False,
            )
            planned = {
                "reference_id": "target",
                "status": "planned",
                "stages": {},
            }
            with patch(
                "crosscut.reference_expansion_job._active_analysis_processes",
                return_value=({"pid": 4490, "command": "crosscut_cli.py analyze"},),
            ), patch(
                "crosscut.reference_expansion_job._load_inputs",
                return_value=inputs,
            ) as load_inputs, patch(
                "crosscut.reference_expansion_job._plan_one_reference",
                return_value=planned,
            ):
                before = set(root.iterdir())
                result = run_reference_expansion_job(
                    cadence_profile_path=Path("profile.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=root,
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_reference_count=2,
                    expected_source_count=MINIMUM_EXPECTED_SOURCE_COUNT,
                    dry_run=True,
                )
                after = set(root.iterdir())

            self.assertEqual(before, after)
            self.assertTrue(result["execution_blocked_until_active_analysis_exits"])
            self.assertFalse(result["writes_performed"])
            self.assertEqual(
                result["input_validation"]["media_sha256"],
                "deferred_until_real_run",
            )
            self.assertEqual(result["references"], [planned])
            self.assertFalse(load_inputs.call_args.kwargs["verify_media_hashes"])

    def test_public_entrypoint_rejects_fewer_than_eight_sources(self) -> None:
        with patch(
            "crosscut.reference_expansion_job._active_analysis_processes",
            return_value=(),
        ), patch("crosscut.reference_expansion_job._load_inputs") as loader:
            with self.assertRaisesRegex(ValueError, "at least 8"):
                run_reference_expansion_job(
                    cadence_profile_path=Path("profile.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=Path("evidence"),
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=7,
                    dry_run=True,
                )
        loader.assert_not_called()

    def test_public_entrypoint_preserves_caller_exact_ten_source_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = Mock()
            inputs.profile = Mock(profile_content_sha256="a" * 64)
            inputs.profile_file_sha256 = "b" * 64
            inputs.seed_evidence_file_sha256 = "c" * 64
            inputs.source_manifest_file_sha256 = "d" * 64
            inputs.references = tuple(range(5))
            inputs.sources = tuple(range(10))
            inputs.seed_reference_id = "seed"
            inputs.media_sha256_verified = False
            inputs.pipeline_code_sha256 = {}
            inputs.target_reference_ids = ()
            inputs.job_fingerprint = "e" * 64
            with patch(
                "crosscut.reference_expansion_job._active_analysis_processes",
                return_value=(),
            ), patch(
                "crosscut.reference_expansion_job._load_inputs",
                return_value=inputs,
            ) as loader:
                result = run_reference_expansion_job(
                    cadence_profile_path=Path("profile.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=root,
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=10,
                    dry_run=True,
                )

            self.assertEqual(loader.call_args.kwargs["expected_source_count"], 10)
            self.assertEqual(result["input_validation"]["source_count"], 10)

    def test_artifact_names_are_reference_scoped_and_canonical_v2(self) -> None:
        root = Path("/tmp/evidence")

        paths = _artifact_paths(root, "CzAbmSPZIMs")

        self.assertIsInstance(paths, ArtifactPaths)
        self.assertEqual(
            paths.evidence.name,
            "CzAbmSPZIMs-reference-evidence-canonical-v2.json",
        )
        self.assertEqual(
            paths.supervision.name,
            "CzAbmSPZIMs-reference-supervision-canonical-v2-final.json",
        )
        self.assertTrue(
            all(Path(path).parent == root for path in paths.to_dict().values())
        )


if __name__ == "__main__":
    unittest.main()
