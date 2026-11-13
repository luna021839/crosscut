from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut import ive_stage_edit_training_corpus_job as corpus_job
from crosscut.ive_causal_sensing_media_job import IVECausalSensingMediaJobError
from crosscut.ive_stage_edit_training_corpus_job import (
    create_ive_stage_edit_training_corpus_request,
    launch_ive_stage_edit_training_corpus_request,
    preflight_ive_stage_edit_training_corpus_request,
)
from crosscut.stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    StageEditCausalSensingBinding,
)
from crosscut.stage_edit_feature_snapshot_job import (
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
    StageEditFeatureSnapshotJobResult,
)
from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA


_BANG_ROOT = Path(
    "/Volumes/CrossCutMedia/CrossCut-Scratch/reference-edits/bang-bang/"
    "evidence-10source-current-canonical-v3"
)
_BANG_SUPERVISION_ROOT = _BANG_ROOT / "downstream-v4/stageedit-training"
_BANG_REFERENCE_IDS = (
    "0O-bHFSeYxQ",
    "CzAbmSPZIMs",
    "EzSUAomKFWY",
    "M35eywYQLD0",
    "RkaI7WsNOGE",
)
_BANG_SUPERVISIONS = tuple(
    _BANG_SUPERVISION_ROOT
    / f"{reference_id}-reference-supervision-stageedit-schema3-447806e937c2b40e.json"
    for reference_id in _BANG_REFERENCE_IDS
)
_INSTALLATION = Path(
    "/Volumes/CrossCutMedia/CrossCut/Models/CausalSensing/active-private-gallery.json"
)
_TRAINING_ROOT = Path("/Volumes/CrossCutMedia/CrossCut-Training")


class IVEStageEditTrainingCorpusJobTests(unittest.TestCase):
    def test_feature_verifier_requires_same_complete_gate_on_all_five_snapshots(
        self,
    ) -> None:
        binding = StageEditCausalSensingBinding(
            shooting_grammar_content_sha256="a" * 64,
            member_state_graph_content_sha256="b" * 64,
            ive_performer_state_content_sha256="c" * 64,
        )
        media_job = SimpleNamespace(
            stage_edit_artifacts=SimpleNamespace(binding=binding)
        )
        preflight = SimpleNamespace(
            song_id="BANG BANG",
            source_count=1,
            reference_count=5,
            union_decision_time_count=1,
            edit_mode="group",
            target_member=None,
            request_payload={
                "decision_grid": {"master_times": [0.5]},
                "sources": [
                    {
                        "source_id": "source-0",
                        "unavailable_master_times": [],
                        "feature_hard_sync_gate_passed": True,
                    }
                ],
            },
        )
        full_gate = {
            **binding.require_production_scope(edit_mode="group"),
            "feature_snapshot_schema_version": FEATURE_MATERIALIZER_SCHEMA_VERSION,
            "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            "feature_materializer_contract": FEATURE_MATERIALIZER_CONTRACT,
        }
        require_gate = Mock(return_value=full_gate)
        fake_snapshot = SimpleNamespace(
            schema_version=FEATURE_MATERIALIZER_SCHEMA_VERSION,
            causal_sensing_binding=binding,
            require_production_causal_sensing=require_gate,
            snapshot_sha256="d" * 64,
            sequence=SimpleNamespace(
                tokens=(
                    SimpleNamespace(
                        master_time_seconds=0.5,
                        source_candidates=(
                            SimpleNamespace(
                                source_id="source-0",
                                available=True,
                                hard_gate_passed=True,
                            ),
                        ),
                    ),
                )
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = {}
            for index in range(5):
                path = root / f"reference-{index}.json"
                path.write_text("{}", encoding="utf-8")
                snapshots[f"reference-{index}"] = path
            unsigned = {
                "schema_version": FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
                "song_id": "BANG BANG",
                "source_count": 1,
                "statistics": {
                    "reference_count": 5,
                    "union_decision_time_count": 1,
                },
                "causal_sensing_binding": binding.to_dict(),
                "production_causal_sensing_gate": full_gate,
            }
            manifest = {
                **unsigned,
                "manifest_sha256": corpus_job._canonical_sha256(unsigned),
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            feature_job = StageEditFeatureSnapshotJobResult(
                output_directory=root,
                manifest_path=manifest_path,
                snapshots=snapshots,
                snapshot_sha256={key: "d" * 64 for key in snapshots},
                statistics={},
            )
            with patch.object(
                corpus_job.StageEditFeatureSnapshot,
                "from_dict",
                return_value=fake_snapshot,
            ) as strict_snapshot_loader:
                receipt = corpus_job._verify_feature_job(
                    preflight=preflight,
                    media_job=media_job,
                    feature_job=feature_job,
                )

        self.assertEqual(strict_snapshot_loader.call_count, 5)
        self.assertEqual(require_gate.call_count, 5)
        self.assertEqual(len(receipt["snapshots"]), 5)
        self.assertTrue(
            all(
                call.kwargs
                == {"edit_mode": "group", "target_member": None}
                for call in require_gate.call_args_list
            )
        )

    def test_public_entrypoints_quarantine_attitude_before_filesystem_access(
        self,
    ) -> None:
        forbidden = Path("/definitely-missing/ATTITUDE/request.json")
        with patch(
            "crosscut.ive_stage_edit_training_corpus_job._derive_inputs",
            side_effect=AssertionError("filesystem derivation must not run"),
        ):
            with self.assertRaisesRegex(
                IVECausalSensingMediaJobError,
                "ATTITUDE is strictly quarantined",
            ):
                create_ive_stage_edit_training_corpus_request(
                    song_id="ATTITUDE",
                    edit_mode="group",
                    target_member=None,
                    supervision_paths=(forbidden,) * 5,
                    expected_source_count=10,
                    installation_pointer_path=forbidden,
                    request_directory=forbidden.parent,
                    native_cut_artifact_directory=forbidden.parent,
                    causal_sensing_artifact_directory=forbidden.parent,
                    feature_snapshot_output_directory=forbidden.parent,
                    receipt_directory=forbidden.parent,
                    created_at_utc="2026-08-25T00:00:00Z",
                )
        with patch(
            "crosscut.ive_stage_edit_training_corpus_job."
            "_load_request_after_quarantine",
            side_effect=AssertionError("request must not be opened"),
        ):
            with self.assertRaisesRegex(
                IVECausalSensingMediaJobError,
                "ATTITUDE is strictly quarantined",
            ):
                preflight_ive_stage_edit_training_corpus_request(
                    song_id="ATTITUDE",
                    request_path=forbidden,
                )
        with patch(
            "crosscut.ive_stage_edit_training_corpus_job."
            "preflight_ive_stage_edit_training_corpus_request",
            side_effect=AssertionError("preflight must not run"),
        ):
            with self.assertRaisesRegex(
                IVECausalSensingMediaJobError,
                "ATTITUDE is strictly quarantined",
            ):
                launch_ive_stage_edit_training_corpus_request(
                    song_id="ATTITUDE",
                    request_path=forbidden,
                )

    @unittest.skipUnless(
        all(path.is_file() for path in (*_BANG_SUPERVISIONS, _INSTALLATION))
        and _TRAINING_ROOT.is_dir(),
        "current BANG 10-source corpus/private gallery is not mounted",
    )
    def test_current_bang_union_request_and_preflight_replay(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="bang-schema4-preflight-",
            dir=_TRAINING_ROOT,
        ) as temporary:
            root = Path(temporary)
            request = create_ive_stage_edit_training_corpus_request(
                song_id="BANG BANG",
                edit_mode="group",
                target_member=None,
                supervision_paths=_BANG_SUPERVISIONS,
                expected_source_count=10,
                installation_pointer_path=_INSTALLATION,
                request_directory=root / "requests",
                native_cut_artifact_directory=root / "native-cuts",
                causal_sensing_artifact_directory=root / "causal-sensing",
                feature_snapshot_output_directory=root / "schema4",
                receipt_directory=root / "receipts",
                created_at_utc="2026-08-25T00:00:00Z",
            )
            preflight = preflight_ive_stage_edit_training_corpus_request(
                song_id="BANG BANG",
                request_path=request.path,
            )

            self.assertEqual(preflight.reference_count, 5)
            self.assertEqual(preflight.source_count, 10)
            self.assertEqual(preflight.union_decision_time_count, 633)
            self.assertEqual(preflight.source_time_cell_count, 6_330)
            self.assertGreater(preflight.total_source_media_bytes, 1_000_000_000)
            self.assertGreater(
                preflight.minimum_observed_free_storage_bytes,
                preflight.required_free_storage_bytes,
            )
            payload = json.loads(request.path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["decision_grid"]["decision_grid_sha256"],
                preflight.decision_grid_sha256,
            )
            self.assertEqual(len(payload["supervisions"]), 5)
            self.assertEqual(len(payload["sources"]), 10)
            self.assertTrue(
                all(source["feature_hard_sync_gate_passed"] for source in payload["sources"])
            )
            self.assertTrue(
                any(source["unavailable_master_times"] for source in payload["sources"])
            )
            for source in payload["sources"]:
                self.assertEqual(
                    source["live_media_probe"]["declared_frame_count"],
                    source["declared_frame_count"],
                )
                self.assertEqual(
                    source["live_media_probe"]["frame_rate"],
                    source["frame_rate"],
                )
                self.assertRegex(
                    source["unavailable_master_time_evidence_sha256"],
                    r"^[0-9a-f]{64}$",
                )
                self.assertRegex(
                    source["media_probe_evidence_sha256"],
                    r"^[0-9a-f]{64}$",
                )


if __name__ == "__main__":
    unittest.main()
