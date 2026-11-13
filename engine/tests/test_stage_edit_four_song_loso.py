from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract
from crosscut.stage_edit_four_song_loso import (
    BANG_BANG,
    BLACKHOLE,
    FEATURE_SONG_ID_BY_TRAINING_SONG,
    FOUR_SONG_LOSO_REQUEST_CONTRACT,
    FourSongLOSOError,
    LOSO_SONG_IDS,
    REBEL_HEART,
    SOURCE_RELATIVE_PATHS,
    TRAINING_SCHEDULE,
    XOXZ,
    _build_train_cli_args,
    _canonical_sha256,
    _external_directory,
    _strict_schema4_receipt,
    _validate_song_artifacts,
    build_parser,
)
from crosscut.stage_edit_transformer import (
    StageEditLossContract,
    production_stage_edit_config,
)


def _song_record(song_id: str, offset: int) -> dict[str, object]:
    references = [f"reference-{offset + index}" for index in range(5)]
    return {
        "training_song_id": song_id,
        "feature_song_id": FEATURE_SONG_ID_BY_TRAINING_SONG[song_id],
        "reference_ids": references,
        "materialization_manifest": {
            "path": f"/Volumes/CrossCutMedia/materialization-{offset}.json",
            "file_sha256": "1" * 64,
        },
        "feature_manifest": {
            "path": f"/Volumes/CrossCutMedia/features-{offset}.json",
            "file_sha256": "2" * 64,
        },
        "dynamic_crop_manifest": {
            "path": f"/Volumes/CrossCutMedia/dynamic-{offset}.json",
            "file_sha256": "3" * 64,
        },
        "schema4_receipt": {
            "path": f"/Volumes/CrossCutMedia/receipt-{offset}.json",
            "file_sha256": "4" * 64,
        },
        "supervision_paths_by_reference": {
            reference: f"/Volumes/CrossCutMedia/{reference}-supervision.json"
            for reference in references
        },
        "feature_snapshot_paths_by_reference": {
            reference: f"/Volumes/CrossCutMedia/{reference}-snapshot.json"
            for reference in references
        },
    }


class FourSongLOSOTests(unittest.TestCase):
    def test_fixed_song_and_feature_alias_contract_is_exact(self) -> None:
        self.assertEqual(
            LOSO_SONG_IDS,
            (BANG_BANG, BLACKHOLE, REBEL_HEART, XOXZ),
        )
        self.assertEqual(
            tuple(FEATURE_SONG_ID_BY_TRAINING_SONG),
            LOSO_SONG_IDS,
        )
        self.assertEqual(len(set(FEATURE_SONG_ID_BY_TRAINING_SONG.values())), 4)

    def test_training_schedule_enables_all_three_pretraining_phases(self) -> None:
        self.assertEqual(TRAINING_SCHEDULE["cut_pretrain_steps"], 500)
        self.assertEqual(TRAINING_SCHEDULE["source_pretrain_steps"], 500)
        self.assertEqual(
            TRAINING_SCHEDULE["dynamic_crop_v3_prior_pretrain_steps"],
            500,
        )
        self.assertEqual(TRAINING_SCHEDULE["device"], "gpu")
        self.assertGreaterEqual(TRAINING_SCHEDULE["epochs"], 30)

    def test_train_args_cover_twenty_references_and_three_training_songs(self) -> None:
        songs = tuple(
            _song_record(song_id, index * 5)
            for index, song_id in enumerate(LOSO_SONG_IDS)
        )
        base = StageEditLossContract(source_objective="eligible_all_hybrid")
        args = _build_train_cli_args(
            songs,
            held_out_song_id=REBEL_HEART,
            checkpoint_directory=Path("/Volumes/CrossCutMedia/checkpoint"),
            model_config=production_stage_edit_config().to_dict(),
            base_loss_contract=asdict(base),
            dynamic_crop_loss_contract=asdict(
                DynamicCropV3LossContract(base=base)
            ),
        )
        self.assertEqual(args.count("--input"), 20)
        self.assertEqual(args.count("--song-id"), 20)
        self.assertEqual(args.count("--feature-snapshot"), 20)
        self.assertEqual(args.count("--dynamic-crop-manifest"), 4)
        self.assertEqual(args.count("--required-training-song"), 3)
        self.assertEqual(args.count("--held-out-song"), 1)
        self.assertNotIn("--authorize-production-activation", args)
        self.assertNotIn("--production-registry", args)

    def test_train_args_pin_v3_and_eligible_all_source_objective(self) -> None:
        songs = tuple(
            _song_record(song_id, index * 5)
            for index, song_id in enumerate(LOSO_SONG_IDS)
        )
        base = StageEditLossContract(source_objective="eligible_all_hybrid")
        args = _build_train_cli_args(
            songs,
            held_out_song_id=XOXZ,
            checkpoint_directory=Path("/Volumes/CrossCutMedia/checkpoint"),
            model_config=production_stage_edit_config().to_dict(),
            base_loss_contract=asdict(base),
            dynamic_crop_loss_contract=asdict(
                DynamicCropV3LossContract(base=base)
            ),
        )
        version_index = args.index("--dynamic-crop-version")
        self.assertEqual(args[version_index + 1], "3")
        loss_index = args.index("--base-loss-contract-json")
        loss = json.loads(args[loss_index + 1])
        self.assertEqual(loss["source_objective"], "eligible_all_hybrid")
        prior_index = args.index("--dynamic-crop-v3-prior-pretrain-steps")
        self.assertEqual(args[prior_index + 1], "500")

    def test_schema4_strict_receipt_requires_reused_artifacts(self) -> None:
        references = tuple(f"reference-{index}" for index in range(5))
        feature_path = Path("/Volumes/CrossCutMedia/features.json")
        payload = {
            "schema_version": 1,
            "contract": "ive_stage_edit_schema4_training_corpus_launch_receipt_v1",
            "scope": {
                "artist": "IVE",
                "edit_mode": "group",
                "song_id": "REBEL HEART",
                "target_member": None,
            },
            "source_count": 10,
            "reference_count": 5,
            "decision_time_count": 100,
            "native_cut_execution": {
                "contract": "strict_native_cut_manifest_resume_v1",
                "existing_manifest_strictly_reloaded": True,
                "detector_invoked": False,
                "all_source_records_strictly_validated": True,
                "persisted_manifest_strictly_replayed": True,
            },
            "causal_sensing_execution": {
                "contract": "strict_causal_sensing_triple_resume_v1",
                "existing_triple_strictly_reloaded": True,
                "media_materializer_invoked": False,
                "all_artifacts_bound_to_exact_request": True,
            },
            "feature_job_execution": {
                "contract": "strict_schema4_feature_manifest_resume_v1",
                "existing_manifest_strictly_reloaded": True,
                "feature_builder_invoked": False,
                "all_snapshots_strictly_replayed": True,
            },
            "feature_job": {
                "manifest_path": str(feature_path),
                "manifest_file_sha256": "a" * 64,
                "snapshots": [
                    {"reference_id": reference_id}
                    for reference_id in references
                ],
            },
            "same_exact_triple_bound_to_all_five_snapshots": True,
            "raw_frames_persisted_by_coordinator": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = (Path(temporary) / "receipt.json").resolve()
            receipt_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = _strict_schema4_receipt(
                receipt_path,
                training_song_id=REBEL_HEART,
                feature_song_id="REBEL HEART",
                feature_manifest_path=feature_path,
                feature_manifest_file_sha256="a" * 64,
                reference_ids=references,
            )
            self.assertEqual(loaded["decision_time_count"], 100)
            payload["feature_job_execution"]["feature_builder_invoked"] = True
            receipt_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(FourSongLOSOError):
                _strict_schema4_receipt(
                    receipt_path,
                    training_song_id=REBEL_HEART,
                    feature_song_id="REBEL HEART",
                    feature_manifest_path=feature_path,
                    feature_manifest_file_sha256="a" * 64,
                    reference_ids=references,
                )

    def test_request_record_rejects_inexact_keys_before_file_access(self) -> None:
        with self.assertRaisesRegex(FourSongLOSOError, "keys are not exact"):
            _validate_song_artifacts(
                {
                    "training_song_id": BANG_BANG,
                    "feature_song_id": "BANG BANG",
                }
            )

    def test_non_external_output_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(FourSongLOSOError, "must be under"):
            _external_directory(Path("/private/tmp/loso-output"), "output")

    def test_source_provenance_paths_are_unique_and_repository_relative(self) -> None:
        self.assertEqual(len(SOURCE_RELATIVE_PATHS), len(set(SOURCE_RELATIVE_PATHS)))
        self.assertTrue(
            all(
                not Path(path).is_absolute() and ".." not in Path(path).parts
                for path in SOURCE_RELATIVE_PATHS
            )
        )

    def test_canonical_hash_is_order_independent(self) -> None:
        left = _canonical_sha256({"a": 1, "b": [2, 3]})
        right = _canonical_sha256({"b": [2, 3], "a": 1})
        self.assertEqual(left, right)
        self.assertEqual(len(left), hashlib.sha256().digest_size * 2)

    def test_parser_exposes_only_inactive_fold_workflow(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("four-song", help_text)
        self.assertNotIn("production-registry", help_text)
        self.assertEqual(
            FOUR_SONG_LOSO_REQUEST_CONTRACT,
            "ive_stage_edit_four_song_loso_request_v1",
        )


if __name__ == "__main__":
    unittest.main()
