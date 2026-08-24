from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from crosscut.reference_crop_job import (
    ANALYSIS_BASIS,
    PIPELINE_REVISION,
    CropJobInputs,
    ValidatedAttribution,
    _crop_caller_provenance,
    _crop_output_paths,
    _load_or_plan_crop_artifacts,
    _plan_one_reference,
    _run_one_reference,
    _validate_crop_artifact,
    _validate_attribution_caller,
    run_reference_crop_job,
)
from crosscut.reference_crop_supervision import (
    ANALYSIS_BASIS as CROP_ANALYSIS_BASIS,
    CONTENT_HASH_BASIS,
    ReferenceCropSupervisionConfig,
    _artifact_content_sha256,
)
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.reference_expansion_job import (
    ActiveAnalysisError,
    ExistingArtifactValidationError,
    ExpansionInputs,
    MediaIdentity,
    ReferenceSpec,
    SourceSpec,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> MediaIdentity:
    info = path.stat()
    return MediaIdentity(
        path=path.resolve(),
        sha256=_sha256(path),
        size_bytes=info.st_size,
        stat_identity=(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
    )


def _fixture(root: Path, *, shot_count: int = 1) -> tuple[
    CropJobInputs,
    ReferenceSpec,
    ValidatedAttribution,
]:
    reference_path = root / "reference.mp4"
    audit_path = root / "audit.json"
    source_path = root / "source.mp4"
    attribution_path = root / "REF-reference-attribution-canonical-v2-final.json"
    for path, value in (
        (reference_path, b"reference-media"),
        (audit_path, b"audit"),
        (source_path, b"source-media"),
    ):
        path.write_bytes(value)
    reference_clock = ReferenceSourceClock.scalar(
        0.0, basis="fixture-reference-clock-v1"
    ).to_dict()
    source_clock = ReferenceSourceClock.scalar(
        0.25, basis="fixture-source-clock-v1"
    ).to_dict()
    reference = ReferenceSpec(
        video_id="REF",
        media=_identity(reference_path),
        audit=_identity(audit_path),
        audit_cut_count=shot_count - 1,
        reference_clock=reference_clock,
    )
    source = SourceSpec(
        video_id="SOURCE_A",
        media=_identity(source_path),
        clock=source_clock,
        provenance={"fixture": True},
        media_probe={"size_bytes": source_path.stat().st_size},
    )
    expansion = ExpansionInputs(
        profile=SimpleNamespace(
            artist="IVE",
            song="BANG BANG",
            profile_content_sha256="a" * 64,
        ),
        profile_payload={},
        profile_path=root / "cadence.json",
        profile_file_sha256="b" * 64,
        seed_evidence={},
        seed_evidence_path=root / "seed.json",
        seed_evidence_file_sha256="c" * 64,
        source_manifest_path=root / "manifest.json",
        source_manifest_file_sha256="d" * 64,
        references=(reference,),
        sources=(source,),
        seed_reference_id="REF",
        target_reference_ids=(),
        style_id="bang-bang-human-reference",
        output_directory=root,
        pipeline_code_sha256={"expansion.py": "e" * 64},
        job_fingerprint="f" * 64,
        media_sha256_verified=True,
    )
    config = ReferenceCropSupervisionConfig()
    inputs = CropJobInputs(
        expansion=expansion,
        target_reference_ids=("REF",),
        config=config,
        pipeline_code_sha256={"reference_crop_job.py": "1" * 64},
        job_fingerprint="2" * 64,
    )
    shots = []
    for index in range(shot_count):
        shots.append(
            {
                "shot_index": index,
                "reference_start_seconds": float(index),
                "reference_end_seconds": float(index + 1),
                "accepted": True,
                "attributed_video_id": "SOURCE_A",
                "canonical_detected_shot": {
                    "schema_version": 1,
                    "detector_shot_index": index,
                    "sample_indices": [index],
                    "source_run_merge_does_not_define_shot_partition": True,
                },
            }
        )
    attribution_payload = {"shots": shots, "fixture": True}
    attribution_path.write_text(json.dumps(attribution_payload), encoding="utf-8")
    attribution = ValidatedAttribution(
        path=attribution_path.resolve(),
        payload=attribution_payload,
        file_sha256=_sha256(attribution_path),
        canonical_json_sha256=hashlib.sha256(
            json.dumps(
                attribution_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        accepted_shots=tuple(shots),
        stat_identity=(
            attribution_path.stat().st_dev,
            attribution_path.stat().st_ino,
            attribution_path.stat().st_size,
            attribution_path.stat().st_mtime_ns,
        ),
    )
    return inputs, reference, attribution


def _crop_payload(
    inputs: CropJobInputs,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    shot: dict,
    *,
    eligible: bool = True,
) -> dict:
    source = inputs.source_by_id["SOURCE_A"]
    start = float(shot["reference_start_seconds"])
    end = float(shot["reference_end_seconds"])
    samples = []
    for index, fraction in enumerate(inputs.config.sample_fractions):
        samples.append(
            {
                "sample_index": index,
                "shot_fraction": fraction,
                "master_seconds": start + (end - start) * fraction,
                "accepted": eligible,
                "rejection_reason": None if eligible else "fixture_rejection",
                "diagnostics": {},
            }
        )
    target = (
        {
            "zoom": 1.08,
            "center_x": 0.5,
            "center_y": 0.5,
            "roll_degrees": 0.0,
        }
        if eligible
        else None
    )
    rejection = None if eligible else "fixture_rejection"
    payload = {
        "schema_version": 1,
        "analysis_basis": CROP_ANALYSIS_BASIS,
        "direction": "source_media_output_to_reference_edit_output",
        "shot": {
            "shot_index": shot["shot_index"],
            "source_id": "SOURCE_A",
            "master_start_seconds": start,
            "master_end_seconds": end,
            "attributed_shot_content_sha256": hashlib.sha256(
                json.dumps(
                    shot,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "canonical_detected_shot": deepcopy(shot["canonical_detected_shot"]),
        },
        "configuration": inputs.config.to_dict(),
        "time_mapping": {
            "master_clock_semantics": "shared_performance_clock",
            "reference_master_to_media": deepcopy(dict(reference.reference_clock)),
            "source_master_to_media": deepcopy(dict(source.clock)),
        },
        "samples": samples,
        "temporal_stability_review": {},
        "aggregate_actual_warp_review": None,
        "supervision": {
            "eligible": eligible,
            "renderer_coordinate_system": (
                "EditorialTransformKeyframe_zoom_center_x_center_y_roll_degrees"
            ),
            "renderer_target": target,
            "rejection_reason": rejection,
            "loss_mask": {
                "crop_transform_loss_masked": not eligible,
                "reason": rejection,
                "unsafe_or_failed_values_clamped": False,
            },
        },
        "provenance": {
            "content_hash_basis": CONTENT_HASH_BASIS,
            "artifact_content_sha256": None,
            "attribution": {
                "path": str(attribution.path),
                "file_sha256": attribution.file_sha256,
                "canonical_json_sha256": attribution.canonical_json_sha256,
            },
            "reference_media": {
                "path": str(reference.media.path),
                "media_sha256": reference.media.sha256,
                "byte_size": reference.media.size_bytes,
            },
            "source_media": {
                "path": str(source.media.path),
                "media_sha256": source.media.sha256,
                "byte_size": source.media.size_bytes,
            },
            "caller": _crop_caller_provenance(
                inputs,
                reference=reference,
                attribution=attribution,
                shot=shot,
                source_id="SOURCE_A",
            ),
            "input_hashes_rechecked_after_decode": True,
            "storage_policy": {},
        },
    }
    payload["provenance"]["artifact_content_sha256"] = (
        _artifact_content_sha256(payload)
    )
    return payload


class CropArtifactResumeTests(unittest.TestCase):
    def test_valid_existing_crop_is_skipped_and_tamper_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, attribution = _fixture(root)
            shot = dict(attribution.accepted_shots[0])
            paths = _crop_output_paths(root, "REF")
            destination = paths.crop_artifact(0, "SOURCE_A")
            valid = _crop_payload(inputs, reference, attribution, shot)
            destination.write_text(json.dumps(valid), encoding="utf-8")

            payloads, results = _load_or_plan_crop_artifacts(
                inputs,
                reference=reference,
                attribution=attribution,
                dry_run=True,
            )

            self.assertEqual(len(payloads), 1)
            self.assertEqual(results[0]["validation"], "passed_without_live_media_rehash")
            original = destination.read_bytes()
            tampered = deepcopy(valid)
            tampered["supervision"]["renderer_target"]["zoom"] = 1.09
            destination.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_bytes = destination.read_bytes()

            with self.assertRaises(ExistingArtifactValidationError):
                _load_or_plan_crop_artifacts(
                    inputs,
                    reference=reference,
                    attribution=attribution,
                    dry_run=True,
                )
            self.assertNotEqual(tampered_bytes, original)
            self.assertEqual(destination.read_bytes(), tampered_bytes)

    def test_exact_clock_and_provenance_are_part_of_crop_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, reference, attribution = _fixture(Path(directory))
            shot = dict(attribution.accepted_shots[0])
            payload = _crop_payload(inputs, reference, attribution, shot)
            _validate_crop_artifact(
                payload,
                inputs=inputs,
                reference=reference,
                attribution=attribution,
                shot=shot,
            )

            tampered = deepcopy(payload)
            tampered["time_mapping"]["source_master_to_media"][
                "global_offset_seconds"
            ] += 0.01
            tampered["provenance"]["artifact_content_sha256"] = (
                _artifact_content_sha256(tampered)
            )
            with self.assertRaisesRegex(ValueError, "clock mismatch"):
                _validate_crop_artifact(
                    tampered,
                    inputs=inputs,
                    reference=reference,
                    attribution=attribution,
                    shot=shot,
                )

    def test_unexpected_crop_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, attribution = _fixture(root)
            unexpected = root / (
                "REF-reference-crop-supervision-crop-v1-shot-9999-SOURCE_A.json"
            )
            unexpected.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                ExistingArtifactValidationError, "do not belong"
            ):
                _load_or_plan_crop_artifacts(
                    inputs,
                    reference=reference,
                    attribution=attribution,
                    dry_run=True,
                )


class CropJobControlTests(unittest.TestCase):
    def test_expanded_training_seed_caller_requires_exact_manifest_and_catalog_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, _attribution = _fixture(root)
            manifest = root / "training-manifest.json"
            catalog = root / "training-catalog.json"
            manifest.write_text("{}", encoding="utf-8")
            catalog.write_text("{}", encoding="utf-8")
            expansion = replace(
                inputs.expansion,
                sources=inputs.expansion.sources * 10,
                source_manifest_path=manifest.resolve(),
                source_manifest_file_sha256=_sha256(manifest),
            )
            inputs = replace(inputs, expansion=expansion)
            caller = {
                "artist": "IVE",
                "song": "BANG BANG",
                "reference_video_id": "REF",
                "purpose": (
                    "same_song_professional_reference_learning_10source_training_pool"
                ),
                "training_source_manifest": {
                    "path": str(manifest.resolve()),
                    "file_sha256": _sha256(manifest),
                },
                "synchronized_source_catalog": {
                    "path": str(catalog.resolve()),
                    "file_sha256": _sha256(catalog),
                },
                "production_render_admission_modified": False,
            }

            _validate_attribution_caller(
                caller, inputs=inputs, reference=reference
            )
            caller["production_render_admission_modified"] = True
            with self.assertRaisesRegex(ValueError, "modified production"):
                _validate_attribution_caller(
                    caller, inputs=inputs, reference=reference
                )

    def test_real_run_blocks_before_loading_or_hashing_inputs(self) -> None:
        with patch(
            "crosscut.reference_crop_job._active_analysis_processes",
            return_value=({"pid": 4490, "command": "crosscut_cli.py analyze"},),
        ), patch("crosscut.reference_crop_job._load_crop_inputs") as loader:
            with self.assertRaises(ActiveAnalysisError):
                run_reference_crop_job(
                    cadence_profile_path=Path("cadence.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=Path("evidence"),
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                )
        loader.assert_not_called()

    def test_dry_run_defers_media_hashes_and_performs_no_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, _attribution = _fixture(root)
            inputs = replace(
                inputs,
                expansion=replace(
                    inputs.expansion,
                    media_sha256_verified=False,
                ),
            )
            planned = {"reference_id": "REF", "status": "planned"}
            before = {item.name: item.read_bytes() for item in root.iterdir()}
            with patch(
                "crosscut.reference_crop_job._active_analysis_processes",
                return_value=({"pid": 4490, "command": "crosscut_cli.py analyze"},),
            ), patch(
                "crosscut.reference_crop_job._load_crop_inputs",
                return_value=inputs,
            ) as loader, patch(
                "crosscut.reference_crop_job._plan_one_reference",
                return_value=planned,
            ):
                result = run_reference_crop_job(
                    cadence_profile_path=Path("cadence.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=root,
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    dry_run=True,
                )
            after = {item.name: item.read_bytes() for item in root.iterdir()}

            self.assertEqual(before, after)
            self.assertFalse(result["writes_performed"])
            self.assertTrue(result["execution_blocked_until_active_analysis_exits"])
            self.assertEqual(
                result["input_validation"]["media_sha256"],
                "deferred_until_real_run",
            )
            self.assertFalse(loader.call_args.kwargs["verify_media_hashes"])

    def test_missing_upstream_is_waiting_and_does_not_create_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, _attribution = _fixture(root)
            upstream = root / "REF-reference-attribution-canonical-v2-final.json"
            upstream.unlink()

            plan = _plan_one_reference(inputs, reference)

            self.assertEqual(plan["status"], "waiting_for_expansion_final_attribution")
            self.assertFalse(_crop_output_paths(root, "REF").status_manifest.exists())

    def test_only_eligible_crop_artifacts_rebuild_outputs_and_status_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, reference, attribution = _fixture(root, shot_count=2)
            eligible = _crop_payload(
                inputs,
                reference,
                attribution,
                dict(attribution.accepted_shots[0]),
                eligible=True,
            )
            rejected = _crop_payload(
                inputs,
                reference,
                attribution,
                dict(attribution.accepted_shots[1]),
                eligible=False,
            )
            dataset_builder = Mock(
                return_value=SimpleNamespace(payload={"records": [{"shot_index": 0}]})
            )
            supervision_builder = Mock(
                return_value=SimpleNamespace(payload={"shot_labels": [{"shot_index": 0}]})
            )

            def materialize(path: Path, *, producer, validator):
                payload = dict(producer())
                return payload, {
                    "action": "created",
                    "path": str(path),
                    "sha256": "9" * 64,
                }

            with patch(
                "crosscut.reference_crop_job._active_analysis_processes",
                return_value=(),
            ), patch(
                "crosscut.reference_crop_job._assert_stat_unchanged"
            ), patch(
                "crosscut.reference_crop_job._load_or_plan_crop_artifacts",
                return_value=(
                    [eligible, rejected],
                    [
                        {"eligible": True, "shot_index": 0},
                        {"eligible": False, "shot_index": 1},
                    ],
                ),
            ), patch(
                "crosscut.reference_crop_job.build_reference_dataset",
                dataset_builder,
            ), patch(
                "crosscut.reference_crop_job.build_reference_supervision",
                supervision_builder,
            ), patch(
                "crosscut.reference_crop_job._materialize_or_skip",
                side_effect=materialize,
            ):
                result = _run_one_reference(
                    inputs,
                    reference=reference,
                    attribution=attribution,
                    emit=lambda _event: None,
                )

            self.assertEqual(result["crop_eligible_count"], 1)
            self.assertEqual(result["crop_rejected_count"], 1)
            for builder in (dataset_builder, supervision_builder):
                crop_inputs = builder.call_args.kwargs["crop_supervision_artifacts"]
                self.assertEqual(len(crop_inputs), 1)
                self.assertIs(crop_inputs[0], eligible)
            status_path = _crop_output_paths(root, "REF").status_manifest
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["analysis"], ANALYSIS_BASIS)
            self.assertEqual(status["pipeline_revision"], PIPELINE_REVISION)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(
                list(root.glob(f".{status_path.name}.crosscut-reference-expansion-*.tmp")),
                [],
            )

    def test_crop_v1_output_names_are_immutable_reference_scoped(self) -> None:
        root = Path("/tmp/evidence")
        paths = _crop_output_paths(root, "RkaI7WsNOGE")

        self.assertTrue(paths.dataset.name.endswith("-crop-v1.json"))
        self.assertTrue(paths.supervision.name.endswith("-crop-v1.json"))
        self.assertEqual(
            paths.crop_artifact(12, "SOURCE_A").name,
            "RkaI7WsNOGE-reference-crop-supervision-crop-v1-shot-0012-SOURCE_A.json",
        )

    def test_public_entrypoint_keeps_five_references_and_minimum_eight_sources(self) -> None:
        with patch("crosscut.reference_crop_job._active_analysis_processes", return_value=()):
            with self.assertRaisesRegex(ValueError, "exactly 5"):
                run_reference_crop_job(
                    cadence_profile_path=Path("cadence.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=Path("evidence"),
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_reference_count=1,
                )
            with self.assertRaisesRegex(ValueError, "at least 8"):
                run_reference_crop_job(
                    cadence_profile_path=Path("cadence.json"),
                    seed_evidence_path=Path("seed.json"),
                    source_manifest_path=Path("manifest.json"),
                    output_directory=Path("evidence"),
                    style_id="bang-bang-human-reference",
                    expected_artist="IVE",
                    expected_song="BANG BANG",
                    expected_source_count=1,
                )

    def test_public_entrypoint_preserves_caller_exact_ten_source_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _reference, _attribution = _fixture(root)
            planned = {"reference_id": "REF", "status": "planned"}
            with patch(
                "crosscut.reference_crop_job._active_analysis_processes",
                return_value=(),
            ), patch(
                "crosscut.reference_crop_job._load_crop_inputs",
                return_value=inputs,
            ) as loader, patch(
                "crosscut.reference_crop_job._plan_one_reference",
                return_value=planned,
            ):
                result = run_reference_crop_job(
                    cadence_profile_path=Path("cadence.json"),
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
            self.assertEqual(result["references"], [planned])


if __name__ == "__main__":
    unittest.main()
