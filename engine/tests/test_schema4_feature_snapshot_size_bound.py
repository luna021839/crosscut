from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.ive_stage_edit_training_corpus_job import (
    IVEStageEditTrainingCorpusError,
    _MAXIMUM_SCHEMA4_FEATURE_SNAPSHOT_BYTES,
    _MAXIMUM_UPSTREAM_JSON_BYTES,
    _canonical_sha256,
    main,
    _schema4_feature_snapshot_maximum_bytes,
    _stable_regular_file_bytes,
    _strictly_reload_existing_feature_snapshot_job,
)


_BLACKHOLE_LARGE_SNAPSHOT = Path(
    "/Volumes/CrossCutMedia/CrossCut-Training/causal-schema4-v1/blackhole/"
    "feature-snapshots/"
    "_IuHLlXCQpk-stageedit-feature-snapshot-5609d1a1c5ea375d.json"
)
_BLACKHOLE_FEATURE_MANIFEST = _BLACKHOLE_LARGE_SNAPSHOT.parent / (
    "stageedit-feature-snapshot-job-manifest-83ca2fd0a8464461.json"
)


class Schema4FeatureSnapshotSizeBoundTests(unittest.TestCase):
    @staticmethod
    def _write_resume_fixture(
        output_directory: Path,
        *,
        snapshot_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        snapshot_directory = snapshot_directory or output_directory
        snapshot_sha256 = "a" * 64
        snapshot_path = snapshot_directory / (
            "reference-stageedit-feature-snapshot-"
            f"{snapshot_sha256[:16]}.json"
        )
        snapshot_path.write_text("{}", encoding="utf-8")
        unsigned = {
            "schema_version": 4,
            "source_count": 1,
            "statistics": {
                "reference_count": 1,
                "source_count": 1,
            },
            "snapshots": [
                {
                    "reference_id": "reference",
                    "path": str(snapshot_path),
                    "snapshot_sha256": snapshot_sha256,
                    "source_count": 1,
                    "token_count": 1,
                }
            ],
        }
        manifest_sha256 = _canonical_sha256(unsigned)
        manifest_path = output_directory / (
            "stageedit-feature-snapshot-job-manifest-"
            f"{manifest_sha256[:16]}.json"
        )
        manifest_path.write_text(
            json.dumps(
                {**unsigned, "manifest_sha256": manifest_sha256},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return manifest_path, snapshot_path

    def test_bound_scales_with_hash_bound_token_source_cells(self) -> None:
        compact = _schema4_feature_snapshot_maximum_bytes(
            token_count=61,
            source_count=10,
        )
        large = _schema4_feature_snapshot_maximum_bytes(
            token_count=369,
            source_count=10,
        )
        self.assertEqual(compact, _MAXIMUM_UPSTREAM_JSON_BYTES)
        self.assertGreater(large, 68_276_348)
        self.assertLessEqual(large, _MAXIMUM_SCHEMA4_FEATURE_SNAPSHOT_BYTES)
        self.assertGreater(large, compact)

    def test_bound_rejects_invalid_or_over_cell_limit_dimensions(self) -> None:
        for token_count, source_count in ((0, 10), (1, 0), (True, 10)):
            with self.assertRaises(IVEStageEditTrainingCorpusError):
                _schema4_feature_snapshot_maximum_bytes(
                    token_count=token_count,
                    source_count=source_count,
                )
        with self.assertRaisesRegex(
            IVEStageEditTrainingCorpusError,
            "source-time cell limit",
        ):
            _schema4_feature_snapshot_maximum_bytes(
                token_count=5_001,
                source_count=10,
            )

    def test_strict_feature_manifest_resume_reconstructs_only_signed_entries(
        self,
    ) -> None:
        with TemporaryDirectory() as raw_directory:
            output_directory = Path(raw_directory)
            manifest_path, snapshot_path = self._write_resume_fixture(
                output_directory
            )
            result = _strictly_reload_existing_feature_snapshot_job(
                manifest_path,
                expected_output_directory=output_directory,
                expected_reference_count=1,
                expected_source_count=1,
            )
            self.assertEqual(result.manifest_path, manifest_path.resolve())
            self.assertEqual(
                result.snapshots,
                {"reference": snapshot_path},
            )
            self.assertEqual(result.snapshot_sha256, {"reference": "a" * 64})

    def test_strict_feature_manifest_resume_rejects_output_escape(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            output_directory = root / "output"
            outside_directory = root / "outside"
            output_directory.mkdir()
            outside_directory.mkdir()
            manifest_path, _snapshot_path = self._write_resume_fixture(
                output_directory,
                snapshot_directory=outside_directory,
            )
            with self.assertRaisesRegex(
                IVEStageEditTrainingCorpusError,
                "left its immutable output directory",
            ):
                _strictly_reload_existing_feature_snapshot_job(
                    manifest_path,
                    expected_output_directory=output_directory,
                    expected_reference_count=1,
                    expected_source_count=1,
                )

    def test_strict_feature_manifest_resume_rejects_leaf_symlink(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            output_directory = root / "output"
            output_directory.mkdir()
            manifest_path, snapshot_path = self._write_resume_fixture(
                output_directory
            )
            external_target = root / "external.json"
            external_target.write_text("{}", encoding="utf-8")
            snapshot_path.unlink()
            snapshot_path.symlink_to(external_target)
            with self.assertRaisesRegex(
                IVEStageEditTrainingCorpusError,
                "left its immutable output directory",
            ):
                _strictly_reload_existing_feature_snapshot_job(
                    manifest_path,
                    expected_output_directory=output_directory,
                    expected_reference_count=1,
                    expected_source_count=1,
                )

    @patch(
        "crosscut.ive_stage_edit_training_corpus_job."
        "launch_ive_stage_edit_training_corpus_request"
    )
    def test_launch_cli_forwards_all_resume_manifests(self, launch) -> None:
        launch.return_value = SimpleNamespace(to_dict=lambda: {})
        with patch("builtins.print"):
            self.assertEqual(
                main(
                    [
                        "launch",
                        "--song-id",
                        "BLACKHOLE",
                        "--request",
                        "/tmp/request.json",
                        "--reuse-native-cut-manifest",
                        "/tmp/native.json",
                        "--reuse-feature-manifest",
                        "/tmp/features.json",
                    ]
                ),
                0,
            )
        keyword = launch.call_args.kwargs
        self.assertEqual(
            keyword["existing_native_cut_manifest_path"],
            Path("/tmp/native.json"),
        )
        self.assertEqual(
            keyword["existing_feature_snapshot_manifest_path"],
            Path("/tmp/features.json"),
        )

    @patch(
        "crosscut.ive_stage_edit_training_corpus_job."
        "preflight_ive_stage_edit_training_corpus_request"
    )
    def test_preflight_cli_does_not_read_launch_only_resume_args(
        self,
        preflight,
    ) -> None:
        preflight.return_value = SimpleNamespace(to_dict=lambda: {})
        with patch("builtins.print"):
            self.assertEqual(
                main(
                    [
                        "preflight",
                        "--song-id",
                        "BLACKHOLE",
                        "--request",
                        "/tmp/request.json",
                    ]
                ),
                0,
            )
        preflight.assert_called_once_with(
            song_id="BLACKHOLE",
            request_path=Path("/tmp/request.json"),
        )

    @unittest.skipUnless(
        _BLACKHOLE_LARGE_SNAPSHOT.is_file(),
        "current BLACKHOLE schema4 snapshot is not mounted",
    )
    def test_current_blackhole_large_snapshot_fits_derived_stable_read_bound(
        self,
    ) -> None:
        maximum_bytes = _schema4_feature_snapshot_maximum_bytes(
            token_count=369,
            source_count=10,
        )
        resolved, encoded = _stable_regular_file_bytes(
            _BLACKHOLE_LARGE_SNAPSHOT,
            maximum_bytes=maximum_bytes,
            name="BLACKHOLE schema4 large feature snapshot",
        )
        self.assertEqual(resolved, _BLACKHOLE_LARGE_SNAPSHOT.resolve(strict=True))
        self.assertEqual(len(encoded), 68_276_348)
        self.assertLess(len(encoded), maximum_bytes)

    @unittest.skipUnless(
        _BLACKHOLE_FEATURE_MANIFEST.is_file(),
        "current BLACKHOLE schema4 manifest is not mounted",
    )
    def test_current_blackhole_feature_job_strictly_resumes_without_rebuild(
        self,
    ) -> None:
        result = _strictly_reload_existing_feature_snapshot_job(
            _BLACKHOLE_FEATURE_MANIFEST,
            expected_output_directory=_BLACKHOLE_FEATURE_MANIFEST.parent,
            expected_reference_count=5,
            expected_source_count=10,
        )
        self.assertEqual(len(result.snapshots), 5)
        self.assertEqual(
            result.manifest_path,
            _BLACKHOLE_FEATURE_MANIFEST.resolve(strict=True),
        )


if __name__ == "__main__":
    unittest.main()
