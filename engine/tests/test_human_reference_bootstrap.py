from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.human_reference_bootstrap import (
    AUDIT_PROVENANCE_ANALYSIS_BASIS,
    EXPECTED_REFERENCE_COUNT,
    EXPECTED_SOURCE_COUNT,
    HumanReferenceBootstrapConfig,
    _content_hash,
    _validate_audit_payload,
    _validate_provenance,
    _write_bytes_no_replace,
    build_human_cut_quality_audit,
    discover_reference_media,
    run_human_reference_bootstrap,
)
from crosscut.professional_reference_cadence import AUDIT_ANALYSIS_BASIS
from crosscut.professional_reference_gate import (
    ProfessionalReferenceCut,
    summarize_professional_reference_cuts,
)
from crosscut.reference_match import DecodedTargetFrames
from crosscut.reference_training_source_pool import (
    build_reference_training_source_manifest,
)
from crosscut.rendered_proxy_review import (
    BOUNDARY_FRAME_OFFSETS,
    HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS,
    OutputSubjectGeometry,
    review_hard_cut_frame_window,
)
from crosscut.vision import SceneChange


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(seed: int, base: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 80, size=(180, 320, 3), dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + base, 0, 255).astype(np.uint8)


def _geometry(_frame: np.ndarray) -> OutputSubjectGeometry:
    return OutputSubjectGeometry(
        face_count=1,
        face_bbox_xywh=(0.4, 0.2, 0.2, 0.25),
        face_backend="synthetic",
        silhouette_bbox_xywh=(0.25, 0.1, 0.5, 0.8),
        silhouette_centroid_xy=(0.5, 0.5),
        silhouette_confidence=1.0,
        silhouette_backend="synthetic",
    )


def _all_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(str(key) for key in value)
        for child in value.values():
            result.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_keys(child))
    return result


def _manifest(root: Path, *, source_count: int = EXPECTED_SOURCE_COUNT) -> Path:
    selected = []
    for index in range(source_count):
        video_id = f"source{index:05d}"  # exactly eleven ASCII characters
        media = root / f"{video_id}.mp4"
        media.write_bytes((f"source-media-{index}" * 10).encode())
        selected.append(
            {
                "source_index": index,
                "video_id": video_id,
                "local_path": str(media.resolve()),
                "media_validation": {
                    "passed": True,
                    "width": 1920,
                    "height": 1080,
                    "frame_rate": 29.97002997002997,
                },
                "content_integrity": {
                    "contract_version": 1,
                    "algorithm": "sha256",
                    "size_bytes": media.stat().st_size,
                    "sha256": _sha256(media),
                },
            }
        )
    path = root / "acquisition-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "group": "IVE",
                "song": "SYNTHETIC",
                "desired_source_count": source_count,
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    return path


def _training_pool_manifest(root: Path) -> Path:
    entries = []
    for index in range(EXPECTED_SOURCE_COUNT):
        video_id = f"source{index:05d}"
        media = root / f"{video_id}.mp4"
        media.write_bytes((f"source-media-{index}" * 10).encode())
        production_eligible = index < 8
        entries.append(
            {
                "video_id": video_id,
                "local_path": str(media.resolve()),
                "selected": production_eligible,
                "group_quality_eligible": production_eligible,
                "group_quality_exclusion_reasons": (
                    [] if production_eligible else ["relative_quality_below_top_eight"]
                ),
                "rejection_reasons": [],
                "download_hard_rejection_reasons": [],
                "media_validation": {
                    "passed": True,
                    "width": 1920,
                    "height": 1080,
                    "frame_rate": 29.97002997002997,
                },
                "aspect_validation": {"passed": True},
                "content_integrity": {
                    "contract_version": 1,
                    "algorithm": "sha256",
                    "size_bytes": media.stat().st_size,
                    "sha256": _sha256(media),
                },
            }
        )
    acquisition = root / "production-acquisition-manifest.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "group": "IVE",
                "song": "SYNTHETIC",
                "desired_source_count": EXPECTED_SOURCE_COUNT,
                "selected": entries[:8],
                "candidates": entries,
            }
        ),
        encoding="utf-8",
    )
    payload = build_reference_training_source_manifest(
        acquisition,
        additional_source_ids=(entries[8]["video_id"], entries[9]["video_id"]),
        expected_artist="IVE",
        expected_song="SYNTHETIC",
        expected_source_count=EXPECTED_SOURCE_COUNT,
    )
    path = root / "reference-training-source-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _references(root: Path, *, count: int = EXPECTED_REFERENCE_COUNT) -> Path:
    references = root / "human-references"
    references.mkdir()
    for index in range(count):
        video_id = f"human{index:06d}"  # exactly eleven ASCII characters
        (references / f"{video_id}.mp4").write_bytes(
            (f"human-stage-mix-{index}" * 10).encode()
        )
    return references


class HumanBoundaryReviewTests(unittest.TestCase):
    def test_public_hard_cut_review_uses_exact_four_frame_contract(self) -> None:
        frames = {
            8: _frame(1, 40),
            9: _frame(2, 42),
            10: _frame(3, 120),
            11: _frame(4, 122),
        }

        review = review_hard_cut_frame_window(
            frames,
            cut_frame_index=10,
            geometry_provider=_geometry,
        )

        self.assertEqual(
            review["analysis_basis"],
            HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS,
        )
        self.assertEqual(review["sample_offsets"], [-2, -1, 0, 1])
        self.assertEqual(review["frame_indices"], [8, 9, 10, 11])
        self.assertEqual(len(review["frame_metrics"]), 4)
        self.assertEqual(
            set(review["discontinuities"]),
            {
                "sharpness_discontinuity",
                "exposure_discontinuity",
                "color_shock",
                "structure_shock",
                "face_scale_discontinuity",
                "face_position_discontinuity",
                "silhouette_scale_discontinuity",
                "silhouette_position_discontinuity",
                "motion_direction_discontinuity",
            },
        )
        self.assertFalse(
            {
                "raw_frame",
                "pixels",
                "descriptor",
                "keypoints",
                "embedding",
            }
            & _all_keys(review)
        )

        with self.assertRaisesRegex(ValueError, "exactly"):
            review_hard_cut_frame_window(
                {key: value for key, value in frames.items() if key != 8},
                cut_frame_index=10,
                geometry_provider=_geometry,
            )

    def test_intentional_flash_quality_is_diagnostic_not_technical(self) -> None:
        rng = np.random.default_rng(90210)
        bright_texture = rng.integers(
            235,
            242,
            size=(180, 320, 3),
            dtype=np.uint8,
        )
        dark_texture = rng.integers(
            20,
            85,
            size=(180, 320, 3),
            dtype=np.uint8,
        )
        review = review_hard_cut_frame_window(
            {
                18: bright_texture.copy(),
                19: np.roll(bright_texture, 1, axis=1),
                20: dark_texture.copy(),
                21: np.roll(dark_texture, 1, axis=1),
            },
            cut_frame_index=20,
            geometry_provider=_geometry,
        )

        self.assertFalse(review["temporal_integrity"]["boundary_quality_passed"])
        self.assertNotIn(
            "human_reference_boundary_frame_quality",
            review["technical_integrity_issue_reasons"],
        )
        self.assertNotIn(
            "human_reference_black_frame",
            review["technical_integrity_issue_reasons"],
        )


class HumanCutAuditTests(unittest.TestCase):
    def test_builder_emits_exact_bang_schema_and_boundary_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "abcdefghijk.mp4"
            media.write_bytes(b"fixture")
            frames = {
                2: _frame(11, 40),
                3: _frame(12, 42),
                4: _frame(13, 120),
                5: _frame(14, 122),
            }
            decoded = DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=12,
                requested_indices=(2, 3, 4, 5),
                frames=frames,
            )
            change = SceneChange(
                frame_index=4,
                score=0.88,
                frame_difference=0.90,
                histogram_distance=0.72,
            )
            with (
                patch(
                    "crosscut.human_reference_bootstrap."
                    "detect_scene_changes_in_video",
                    return_value=(change,),
                ),
                patch(
                    "crosscut.human_reference_bootstrap."
                    "decode_target_frames_sequential",
                    return_value=decoded,
                ),
            ):
                result = build_human_cut_quality_audit(
                    media,
                    geometry_provider=_geometry,
                )

            self.assertEqual(
                set(result.audit),
                {"analysis_basis", "reference", "summary", "cuts"},
            )
            self.assertEqual(result.audit["analysis_basis"], AUDIT_ANALYSIS_BASIS)
            self.assertEqual(result.audit["summary"]["cut_count"], 1)
            cut = result.audit["cuts"][0]
            self.assertEqual(
                set(cut),
                {
                    "frame_index",
                    "seconds",
                    "score",
                    "discontinuities",
                    "issue_reasons",
                },
            )
            self.assertEqual(cut["frame_index"], 4)
            self.assertAlmostEqual(cut["seconds"], 4.0 / 30.0)
            evidence = result.boundary_evidence[0]
            self.assertEqual(evidence["detector"]["frame_index"], 4)
            self.assertEqual(
                evidence["boundary_review"]["sample_offsets"],
                list(BOUNDARY_FRAME_OFFSETS),
            )
            self.assertFalse(
                {
                    "raw_frame",
                    "pixels",
                    "descriptor",
                    "keypoints",
                    "embedding",
                }
                & _all_keys(evidence)
            )

    def test_canonical_summary_public_api_rejects_wrong_types(self) -> None:
        cut = ProfessionalReferenceCut(
            score=70.0,
            discontinuities={
                "sharpness_discontinuity": 0.1,
                "exposure_discontinuity": 0.1,
                "color_shock": 0.2,
                "structure_shock": 0.3,
                "face_scale_discontinuity": None,
                "face_position_discontinuity": None,
                "silhouette_scale_discontinuity": 0.1,
                "silhouette_position_discontinuity": 0.1,
                "motion_direction_discontinuity": None,
            },
        )
        self.assertEqual(summarize_professional_reference_cuts((cut,))["cut_count"], 1)
        with self.assertRaises(TypeError):
            summarize_professional_reference_cuts((object(),))  # type: ignore[arg-type]


class BootstrapContractTests(unittest.TestCase):
    def test_dry_run_hashes_and_binds_exact_five_by_ten_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = _references(root)
            manifest = _manifest(root)
            output = root / "evidence-stageedit-bootstrap-v1"

            result = run_human_reference_bootstrap(
                reference_directory=references,
                source_manifest_path=manifest,
                output_directory=output,
                artist="IVE",
                song="SYNTHETIC",
                dry_run=True,
            )

            self.assertEqual(result["status"], "planned")
            self.assertEqual(result["reference_count"], 5)
            self.assertEqual(
                result["source_manifest"]["selected_source_count"],
                10,
            )
            self.assertTrue(
                result["source_manifest"]["all_selected_media_sha256_verified"]
            )
            self.assertEqual(len(result["job_fingerprint"]), 64)
            self.assertEqual(len(result["configuration_sha256"]), 64)
            self.assertEqual(len(result["references"]), 5)
            self.assertEqual(list(output.iterdir()), [])

    def test_dry_run_accepts_explicit_training_only_relative_quality_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_human_reference_bootstrap(
                reference_directory=_references(root),
                source_manifest_path=_training_pool_manifest(root),
                output_directory=root / "evidence-stageedit-bootstrap-v1",
                artist="IVE",
                song="SYNTHETIC",
                dry_run=True,
            )

            self.assertEqual(result["status"], "planned")
            self.assertEqual(
                result["source_manifest"]["selected_source_count"],
                EXPECTED_SOURCE_COUNT,
            )

    def test_plain_manifest_still_rejects_quality_ineligible_selected_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = _references(root)
            manifest = _manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["selected"][8]["group_quality_eligible"] = False
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "quality-ineligible"):
                run_human_reference_bootstrap(
                    reference_directory=references,
                    source_manifest_path=manifest,
                    output_directory=root / "output",
                    artist="IVE",
                    song="SYNTHETIC",
                    dry_run=True,
                )

    def test_contract_rejects_reference_or_source_count_downgrade(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            HumanReferenceBootstrapConfig(expected_reference_count=4)
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            HumanReferenceBootstrapConfig(expected_source_count=9)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "exactly 5"):
                discover_reference_media(_references(root, count=4))

    def test_contract_rejects_tampered_source_before_any_video_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = _references(root)
            manifest = _manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            Path(payload["selected"][3]["local_path"]).write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_human_reference_bootstrap(
                    reference_directory=references,
                    source_manifest_path=manifest,
                    output_directory=root / "output",
                    artist="IVE",
                    song="SYNTHETIC",
                    dry_run=True,
                )

    def test_immutable_writer_refuses_to_replace_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.json"
            _write_bytes_no_replace(path, b"first\n")
            with self.assertRaises(FileExistsError):
                _write_bytes_no_replace(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_audit_and_sidecar_validation_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "abcdefghijk.mp4"
            media.write_bytes(b"reference")
            from crosscut.human_reference_bootstrap import (
                MediaIdentity,
                ReferenceIdentity,
            )

            reference = ReferenceIdentity(
                "abcdefghijk",
                MediaIdentity(
                    media.resolve(),
                    _sha256(media),
                    media.stat().st_size,
                    media.stat().st_mtime_ns,
                ),
            )
            discontinuities = {
                "sharpness_discontinuity": 0.1,
                "exposure_discontinuity": 0.1,
                "color_shock": 0.2,
                "structure_shock": 0.3,
                "face_scale_discontinuity": None,
                "face_position_discontinuity": None,
                "silhouette_scale_discontinuity": 0.1,
                "silhouette_position_discontinuity": 0.1,
                "motion_direction_discontinuity": None,
            }
            cut_model = ProfessionalReferenceCut(70.0, discontinuities)
            audit = {
                "analysis_basis": AUDIT_ANALYSIS_BASIS,
                "reference": str(media.resolve()),
                "summary": summarize_professional_reference_cuts((cut_model,)),
                "cuts": [
                    {
                        "frame_index": 10,
                        "seconds": 1.0,
                        "score": 70.0,
                        "discontinuities": discontinuities,
                        "issue_reasons": [],
                    }
                ],
            }
            _validate_audit_payload(audit, reference=reference)
            audit_path = root / "abcdefghijk-human-cut-quality-audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_sha = _sha256(audit_path)
            sidecar = {
                "schema_version": 1,
                "analysis": AUDIT_PROVENANCE_ANALYSIS_BASIS,
                "provenance_content_sha256": None,
                "job_fingerprint": "a" * 64,
                "configuration_sha256": "b" * 64,
                "pipeline_code_sha256": {"module": "c" * 64},
                "source_contract": {"manifest_file_sha256": "d" * 64},
                "reference_media": reference.media.to_dict(),
                "audit_artifact": {
                    "path": str(audit_path.resolve()),
                    "sha256": audit_sha,
                },
                "boundary_frame_offsets": [-2, -1, 0, 1],
            }
            sidecar["provenance_content_sha256"] = _content_hash(
                sidecar,
                "provenance_content_sha256",
            )
            _validate_provenance(
                sidecar,
                job_fingerprint="a" * 64,
                config_sha256="b" * 64,
                code_sha256={"module": "c" * 64},
                source_manifest_sha256="d" * 64,
                reference=reference,
                audit_path=audit_path.resolve(),
                audit_sha256=audit_sha,
            )
            sidecar["audit_artifact"]["sha256"] = "e" * 64
            sidecar["provenance_content_sha256"] = _content_hash(
                sidecar,
                "provenance_content_sha256",
            )
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                _validate_provenance(
                    sidecar,
                    job_fingerprint="a" * 64,
                    config_sha256="b" * 64,
                    code_sha256={"module": "c" * 64},
                    source_manifest_sha256="d" * 64,
                    reference=reference,
                    audit_path=audit_path.resolve(),
                    audit_sha256=audit_sha,
                )


if __name__ == "__main__":
    unittest.main()
