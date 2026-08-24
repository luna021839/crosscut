from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.reference_dynamic_crop_job import (
    DynamicCropReferencePair,
    ExistingDynamicCropArtifactError,
    ReferenceDynamicCropJobError,
    discover_reference_dynamic_crop_pairs,
    run_reference_dynamic_crop_job,
    validate_reference_dynamic_crop_manifest,
)
from tests.test_reference_dynamic_crop_supervision import (
    _config,
    _renderer_affine,
    _write_canonical_inputs,
    _write_dynamic_pair,
)


def _bound_pair(root: Path) -> DynamicCropReferencePair:
    transforms = [
        _renderer_affine(
            480,
            270,
            zoom=1.045 + 0.004 * index,
            center_x=0.500 + 0.0006 * index,
            center_y=0.500,
            roll_degrees=0.0,
        )
        for index in range(18)
    ]
    reference, source = _write_dynamic_pair(root, transforms)
    attribution_path, evidence_path, _shot = _write_canonical_inputs(
        root, reference, source
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    caller = {
        "reference_video_id": "REFERENCE_A",
        "artist": "IVE",
        "song": "SYNTHETIC",
        "purpose": "focused_dynamic_crop_job_test",
    }
    evidence["provenance"] = {"caller": caller}
    evidence["reference_video"]["size_bytes"] = reference.stat().st_size
    evidence["sources"][0]["media_probe"]["size_bytes"] = source.stat().st_size
    attribution["sources"] = deepcopy(evidence["sources"])
    attribution["reference_evidence_provenance"]["provenance"] = {
        "caller": caller
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    attribution_path.write_text(json.dumps(attribution), encoding="utf-8")
    canonical_attribution = (
        root / "REFERENCE_A-reference-attribution-canonical-v2-final.json"
    )
    canonical_evidence = (
        root / "REFERENCE_A-reference-evidence-canonical-v2-boundaries.json"
    )
    attribution_path.replace(canonical_attribution)
    evidence_path.replace(canonical_evidence)
    return DynamicCropReferencePair(
        "REFERENCE_A", canonical_attribution, canonical_evidence
    )


class ReferenceDynamicCropJobTests(unittest.TestCase):
    def test_rejected_invalid_perspective_infinity_is_explicitly_missing(self) -> None:
        from crosscut import reference_dynamic_crop_job as job

        original = {
            "perspective_distortion": math.inf,
            "homography_inlier_ratio": 0.75,
        }
        transform, rejection, normalized = job._normalize_rejected_estimator_audit(
            None,
            "invalid_perspective_geometry",
            original,
        )

        self.assertIsNone(transform)
        self.assertEqual(rejection, "invalid_perspective_geometry")
        self.assertIsNone(normalized["perspective_distortion"])
        self.assertEqual(
            normalized["perspective_distortion_missing"],
            {
                "reason": "undefined_invalid_projective_denominator",
                "candidate_rejected": True,
                "target_value_used": False,
                "numeric_value_clamped_or_substituted": False,
            },
        )
        self.assertTrue(math.isinf(original["perspective_distortion"]))
        json.dumps(normalized, allow_nan=False)

    def test_nonfinite_estimator_audit_outside_exact_rejection_fails_closed(self) -> None:
        from crosscut import reference_dynamic_crop_job as job

        cases = (
            (
                object(),
                "invalid_perspective_geometry",
                {"perspective_distortion": math.inf},
            ),
            (None, "different_rejection", {"perspective_distortion": math.inf}),
            (
                None,
                "invalid_perspective_geometry",
                {"perspective_distortion": math.nan},
            ),
            (
                None,
                "invalid_perspective_geometry",
                {"perspective_distortion": math.inf, "other": -math.inf},
            ),
        )
        for transform, rejection, diagnostics in cases:
            with self.subTest(rejection=rejection, diagnostics=diagnostics):
                with self.assertRaisesRegex(
                    ReferenceDynamicCropJobError,
                    "unsafe non-finite estimator diagnostic",
                ):
                    job._normalize_rejected_estimator_audit(
                        transform, rejection, diagnostics
                    )

    def test_job_safe_builder_scopes_and_restores_estimator_adapter(self) -> None:
        from crosscut import reference_dynamic_crop_job as job
        from crosscut import reference_dynamic_crop_supervision as dynamic

        def unsafe_estimator(*_args: object, **_kwargs: object):
            return (
                None,
                "invalid_perspective_geometry",
                {"perspective_distortion": math.inf},
            )

        observed: dict[str, object] = {}

        def fake_builder(_request: object, *, config: object) -> object:
            observed["config"] = config
            observed["result"] = dynamic._estimate_sample()
            return observed

        sentinel_config = object()
        with patch.object(dynamic, "_estimate_sample", unsafe_estimator), patch.object(
            job, "build_reference_dynamic_crop_supervision", side_effect=fake_builder
        ):
            result = job._build_reference_dynamic_crop_supervision_job_safe(
                object(), config=sentinel_config
            )
            self.assertIs(dynamic._estimate_sample, unsafe_estimator)

        self.assertIs(result, observed)
        self.assertIs(observed["config"], sentinel_config)
        transform, rejection, diagnostics = observed["result"]
        self.assertIsNone(transform)
        self.assertEqual(rejection, "invalid_perspective_geometry")
        self.assertIsNone(diagnostics["perspective_distortion"])

    def test_discovery_requires_complete_exact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = _bound_pair(root)
            discovered = discover_reference_dynamic_crop_pairs(
                root, expected_reference_count=1
            )
            self.assertEqual(discovered, (pair,))
            with self.assertRaisesRegex(
                ReferenceDynamicCropJobError, "pair count mismatch"
            ):
                discover_reference_dynamic_crop_pairs(
                    root, expected_reference_count=5
                )

    def test_dry_run_never_hashes_media_decodes_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = _bound_pair(root)
            output = root / "crop-v3-output"

            from crosscut import reference_dynamic_crop_job as job

            real_file_sha256 = job._file_sha256

            def reject_media_hash(path: Path) -> str:
                if Path(path).suffix == ".avi":
                    raise AssertionError("dry run hashed media")
                return real_file_sha256(path)

            with patch.object(job, "_file_sha256", side_effect=reject_media_hash), patch.object(
                job,
                "build_reference_dynamic_crop_supervision",
                side_effect=AssertionError("dry run invoked the pixel generator"),
            ):
                result = run_reference_dynamic_crop_job(
                    (pair,),
                    output_directory=output,
                    config=_config(),
                    dry_run=True,
                )

            self.assertTrue(result.dry_run)
            self.assertEqual(result.built_count, 0)
            self.assertEqual(result.resumed_count, 0)
            self.assertEqual(len(result.plan), 1)
            self.assertEqual(result.plan[0]["status"], "would_build")
            self.assertFalse(output.exists())

    def test_real_run_publishes_content_addressed_manifest_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = _bound_pair(root)
            output = root / "crop-v3-output"
            first = run_reference_dynamic_crop_job(
                (pair,), output_directory=output, config=_config()
            )

            self.assertEqual(first.built_count, 1)
            self.assertEqual(first.resumed_count, 0)
            self.assertIsNotNone(first.manifest_path)
            manifest_path = first.manifest_path
            assert manifest_path is not None
            self.assertIn(first.manifest_content_sha256 or "", manifest_path.name)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_reference_dynamic_crop_manifest(
                manifest, verify_artifact_files=True
            )
            statistics = manifest["statistics"]
            self.assertEqual(statistics["artifact_count"], 1)
            self.assertEqual(statistics["silent_skip_count"], 0)
            self.assertEqual(statistics["copied_target_count"], 0)
            self.assertEqual(statistics["synthetic_time_zero_keyframe_count"], 0)
            self.assertEqual(
                statistics["eligible_count"] + statistics["ineligible_count"], 1
            )
            artifact_path = Path(manifest["artifacts"][0]["artifact"]["path"])
            artifact_identity = artifact_path.stat().st_mtime_ns

            with patch(
                "crosscut.reference_dynamic_crop_job.build_reference_dynamic_crop_supervision",
                side_effect=AssertionError("identical resume rebuilt an artifact"),
            ):
                second = run_reference_dynamic_crop_job(
                    (pair,), output_directory=output, config=_config()
                )
            self.assertEqual(second.built_count, 0)
            self.assertEqual(second.resumed_count, 1)
            self.assertEqual(second.manifest_path, first.manifest_path)
            self.assertEqual(second.manifest_content_sha256, first.manifest_content_sha256)
            self.assertEqual(artifact_path.stat().st_mtime_ns, artifact_identity)

    def test_existing_nonidentical_artifact_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = _bound_pair(root)
            output = root / "crop-v3-output"
            first = run_reference_dynamic_crop_job(
                (pair,), output_directory=output, config=_config()
            )
            assert first.manifest_path is not None
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            artifact_path = Path(manifest["artifacts"][0]["artifact"]["path"])
            tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
            tampered["provenance"]["caller"]["reference_id"] = "WRONG"
            artifact_path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaises(ExistingDynamicCropArtifactError):
                run_reference_dynamic_crop_job(
                    (pair,), output_directory=output, config=_config()
                )
            self.assertEqual(
                json.loads(artifact_path.read_text(encoding="utf-8"))["provenance"][
                    "caller"
                ]["reference_id"],
                "WRONG",
            )

    def test_shot_filter_cannot_silently_select_an_unretained_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = _bound_pair(root)
            with self.assertRaisesRegex(ValueError, "not retained canonical shots"):
                run_reference_dynamic_crop_job(
                    (pair,),
                    output_directory=root / "out",
                    shot_indices={"REFERENCE_A": (9,)},
                    config=_config(),
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
