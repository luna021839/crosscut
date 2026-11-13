from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.stage_edit_candidate_rerank import (
    StageEditCandidateRerankNotReadyError,
    materialize_stage_edit_rerank_snapshots,
    rerank_visual_timeline_candidates_with_stage_edit,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_RUN_KIND,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
    StageEditLossContract,
    StageEditTransformer,
    StageEditTransformerConfig,
    mlx_runtime_version,
    new_stage_edit_checkpoint_metadata,
    save_stage_edit_checkpoint,
    stage_edit_cut_imbalance_contract,
    stage_edit_cut_scoring_contract,
    stage_edit_recent_history_mask_contract,
    stage_edit_source_scoring_contract,
)
from crosscut.stage_edit_transition_authority import (
    StageEditTransitionAuthority,
)
from crosscut.timeline import (
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.timeline_review import review_timeline
from crosscut.visual_score import (
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    _review_ranked_candidate_indices,
)
from tests.stage_edit_causal_sensing_test_support import synthetic_production_gate

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional on non-Apple CI.
    mx = None


@unittest.skipIf(mlx_runtime_version() is None, "MLX is not installed")
class StageEditCandidateRerankTests(unittest.TestCase):
    @classmethod
    def _timeline_result(
        cls,
        source_ids: tuple[str, ...],
        *,
        total_score: float,
    ) -> TimelineResult:
        choices = tuple(
            TimelineChoice(
                candidate=TimelineCandidate(
                    segment_index=index,
                    source_id=source_id,
                    start_seconds=2.0 + index,
                    end_seconds=3.0 + index,
                    sync_score=0.90 - 0.02 * int(source_id),
                    composition_score=0.72 + 0.03 * int(source_id),
                    boundary_score=0.76,
                    shot_scale=0.45 + 0.05 * int(source_id),
                    entry_shot_scale=0.45 + 0.05 * int(source_id),
                    exit_shot_scale=0.47 + 0.05 * int(source_id),
                    entry_focus_bbox_xywh=(
                        0.32 + 0.04 * int(source_id),
                        0.18,
                        0.24,
                        0.40 - 0.02 * int(source_id),
                    ),
                    entry_subject_centroid_xy=(0.50, 0.55),
                    entry_motion=(0.1, -0.1),
                    candidate_id=f"{index}:{source_id}",
                    shot_id=f"{source_id}:shot-{index}",
                ),
                emission_score=1.0,
                transition_score=0.0,
                cumulative_score=float(index + 1),
                used_group_wide_fallback=False,
            )
            for index, source_id in enumerate(source_ids)
        )
        return TimelineResult(
            mode=TimelineMode.GENERAL,
            target_member=None,
            choices=choices,
            total_score=total_score,
        )

    @classmethod
    def _candidate_set(cls) -> VisualTimelineCandidateSet:
        sequences = (
            ("0", "0", "0", "0"),
            ("1", "1", "1", "1"),
            ("0", "1", "0", "1"),
            ("0", "0", "1", "1"),
            ("0", "1", "1", "0"),
            ("1", "0", "0", "1"),
            ("1", "1", "0", "0"),
            ("1", "0", "1", "0"),
        )
        results = tuple(
            cls._timeline_result(sequence, total_score=10.0 - index)
            for index, sequence in enumerate(sequences)
        )
        reviews = tuple(
            review_timeline(result, available_source_count=2) for result in results
        )
        ranked = _review_ranked_candidate_indices(reviews)
        rank_by_index = {
            candidate_index: rank
            for rank, candidate_index in enumerate(ranked)
        }
        records = tuple(
            VisualTimelineCandidateRecord(
                candidate_index=index,
                cadence_profile=("balanced", "reference", "dense_safe")[index % 3],
                timeline_result=result,
                review=reviews[index],
                review_rank=rank_by_index[index],
                selected=index == ranked[0],
            )
            for index, result in enumerate(results)
        )
        return VisualTimelineCandidateSet(
            candidates=records,
            selected_candidate_index=ranked[0],
            sources=(Path("source-0.mp4"), Path("source-1.mp4")),
            audio_offsets=(0.0, 0.0),
            source_timewarps=(None, None),
            source_clock_basis="global_scalar_offset",
            requested_maximum_results=8,
        )

    @staticmethod
    def _write_production_report(root: Path, saved: object) -> Path:
        threshold_freeze = {
            "contract": STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
            "selected_probability_threshold": 0.5,
            "heldout_labels_consumed": False,
            "heldout_probabilities_consumed": False,
            "selected_before_heldout_forward": True,
            "selected_checkpoint_id": saved.checkpoint_id,
            "selected_checkpoint_weight_sha256": saved.weight_sha256,
        }
        threshold_freeze["freeze_sha256"] = hashlib.sha256(
            json.dumps(
                threshold_freeze,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        sensing_gate = synthetic_production_gate()
        reference_ids = (
            "a-train",
            "b-train",
            "a-validation",
            "b-validation",
            "c-heldout-1",
            "c-heldout-2",
        )
        report = {
            "schema_version": PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
            "run_kind": PRODUCTION_TRAINING_RUN_KIND,
            "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
            "production_ready": True,
            "decision_history_input_contract": (
                stage_edit_recent_history_mask_contract(
                    DEFAULT_STAGE_EDIT_SCHEMA
                )
            ),
            "source_scoring_contract": stage_edit_source_scoring_contract(
                saved.model_config
            ),
            "cut_scoring_contract": stage_edit_cut_scoring_contract(
                saved.model_config
            ),
            "transition_authority_contract": StageEditTransitionAuthority(
                mode=saved.model_config.transition_authority
            ).to_dict(),
            "transition_authority_contract_sha256": (
                StageEditTransitionAuthority(
                    mode=saved.model_config.transition_authority
                ).fingerprint
            ),
            "cut_imbalance_contract": stage_edit_cut_imbalance_contract(
                positive_count=20,
                negative_count=80,
                maximum_positive_weight=8.0,
            ),
            "cut_threshold_freeze": threshold_freeze,
            "structured_head_coverage_gate": {
                "contract": (
                    "duration_learned_transition_or_renderer_metadata_coverage_gate_v2"
                ),
                "passed": True,
                "heldout_used_for_model_selection": False,
            },
            "production_supervision_gate": {
                "passed": True,
                "scope": "training_songs_only_including_reference_validation",
                "held_out_references_included": False,
                "observed": {
                    "dense_editorial_switch_count": 120,
                    "reference_count_with_dense_editorial_switches": 6,
                    "song_count_with_dense_editorial_switches": 3,
                    "native_source_camera_cut_editorial_positive_count": 0,
                },
            },
            "production_feature_snapshot_gate": {
                "passed": True,
                "contract": (
                    PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT
                ),
                "reference_count": 6,
                "feature_snapshot_sha256_by_reference": {
                    "a-train": "a" * 64,
                    "b-train": "b" * 64,
                    "a-validation": "c" * 64,
                    "b-validation": "d" * 64,
                    "c-heldout-1": "e" * 64,
                    "c-heldout-2": "f" * 64,
                },
                "production_causal_sensing_gate_by_reference": {
                    reference_id: sensing_gate for reference_id in reference_ids
                },
                "production_causal_sensing_scope": {
                    key: sensing_gate[key]
                    for key in (
                        "edit_mode",
                        "target_member",
                        "member_feature_mode",
                        "feature_snapshot_schema_version",
                        "token_schema_fingerprint",
                    )
                },
                "reference_match_or_current_choice_features_present": False,
                "missing_reference_ids": [],
            },
            "checkpoint": {
                "checkpoint_id": saved.checkpoint_id,
                "training_steps": saved.training_steps,
                "weight_sha256": saved.weight_sha256,
                "production_activation_authorized": True,
                "production_supervision_gate_passed": True,
                "dynamic_crop_v2_checkpoint": False,
            },
            "model": {
                "architecture": "causal_long_context_multi_head_transformer_legacy_crop",
                "dynamic_crop_v2_enabled": False,
            },
            "split": {
                "split_level": "song",
                "training_song_ids": ["IVE-song-a", "IVE-song-b"],
                "held_out_song_ids": ["IVE-song-c"],
                "training_reference_ids": ["a-train", "b-train"],
                "model_selection_validation_level": (
                    "whole_reference_within_training_songs"
                ),
                "validation_reference_ids": ["a-validation", "b-validation"],
                "held_out_reference_ids": ["c-heldout-1", "c-heldout-2"],
                "held_out_used_for_model_selection": False,
            },
            "optimization": {
                "effective_loss_contract": asdict(StageEditLossContract()),
                "model_selection": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
                "best_validation_selection_evidence": {
                    "contract": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
                    "selectable": True,
                    "heldout_metrics_consumed": False,
                },
                "unseen_song_used_for_model_selection": False,
            },
            "held_out_evaluation": {
                "passed": True,
                "split_level": "song",
                "training_song_ids": ["IVE-song-a", "IVE-song-b"],
                "held_out_song_ids": ["IVE-song-c"],
                "held_out_reference_count": 2,
                "checkpoint_id": saved.checkpoint_id,
                "weight_sha256": saved.weight_sha256,
                "quality_gate": {"passed": True},
            },
        }
        path = root / PRODUCTION_TRAINING_REPORT_FILENAME
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    @classmethod
    def _ready_checkpoint(cls, root: Path) -> Path:
        assert mx is not None
        config = StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=16,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=32,
            max_sequence_length=8,
            max_sources=2,
            dropout=0.0,
        )
        mx.random.seed(71)
        model = StageEditTransformer(config)
        model.eval()
        metadata = new_stage_edit_checkpoint_metadata(
            "synthetic-production-rerank",
            DEFAULT_STAGE_EDIT_SCHEMA,
            config,
            training_steps=100,
            weights_trained=True,
            training_data_id="synthetic-dense-disjoint-fixture",
        )
        saved = save_stage_edit_checkpoint(root, model, metadata)
        return cls._write_production_report(root, saved)

    def test_ready_checkpoint_runs_real_forward_and_audits_all_components(
        self,
    ) -> None:
        candidate_set = self._candidate_set()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = self._ready_checkpoint(root)
            result = rerank_visual_timeline_candidates_with_stage_edit(
                candidate_set,
                checkpoint_directory=root,
                production_training_report=report,
            )
            repeated = rerank_visual_timeline_candidates_with_stage_edit(
                candidate_set,
                checkpoint_directory=root,
                production_training_report=report,
            )

        self.assertEqual(result.ranked_candidate_indices, repeated.ranked_candidate_indices)
        self.assertEqual(
            result.feature_snapshot_set_sha256,
            repeated.feature_snapshot_set_sha256,
        )
        self.assertEqual(len(result.ranked_candidate_indices), 8)
        self.assertEqual(
            [record.candidate_index for record in result.ordered_records(candidate_set)],
            list(result.ranked_candidate_indices),
        )
        self.assertTrue(result.readiness.learned_inference_ready)
        self.assertEqual(result.readiness.status, "production_checkpoint_ready")
        self.assertEqual(len(result.checkpoint_bundle_sha256), 64)
        self.assertEqual(len(result.production_training_report_sha256), 64)
        self.assertEqual(len(result.weights_sha256), 64)
        self.assertEqual(len(result.component_scores), 8)
        for score in result.component_scores:
            self.assertEqual(score.token_count, 4)
            self.assertEqual(score.boundary_count, 3)
            self.assertGreaterEqual(score.source_event_count, 1)
            self.assertEqual(score.duration_event_count, score.source_event_count)
            self.assertEqual(score.crop_event_count, score.source_event_count)
            self.assertEqual(
                score.transition_event_count,
                score.source_event_count - 1,
            )
        payload = result.to_dict()
        self.assertFalse(payload["creates_new_timeline_paths"])
        self.assertIn("component_scores", payload)
        self.assertIn("weights_sha256", payload["checkpoint"])

    def test_sparse_checkpoint_and_non_colocated_report_are_rejected(self) -> None:
        candidate_set = self._candidate_set()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            production_report = self._ready_checkpoint(root)
            sparse_report = {
                "schema_version": 3,
                "run_kind": "stage_edit_sparse_offline_calibration",
                "checkpoint_scope": "offline_sparse_calibration_only",
                "production_ready": False,
            }
            production_report.write_text(json.dumps(sparse_report), encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditCandidateRerankNotReadyError,
                "production_training_report_rejected",
            ):
                rerank_visual_timeline_candidates_with_stage_edit(
                    candidate_set,
                    checkpoint_directory=root,
                    production_training_report=production_report,
                )

            outside = root / "outside-report.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditCandidateRerankNotReadyError,
                "exact colocated",
            ):
                rerank_visual_timeline_candidates_with_stage_edit(
                    candidate_set,
                    checkpoint_directory=root,
                    production_training_report=outside,
                )

    def test_feature_snapshots_ignore_reviewer_rank_and_selection_poison(self) -> None:
        candidate_set = self._candidate_set()
        before = materialize_stage_edit_rerank_snapshots(
            candidate_set,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        # Simulate hostile post-choice metadata on the already validated in-
        # process object.  The shared materializer must never inspect it.
        for record in candidate_set.candidates:
            object.__setattr__(record, "review_rank", 1000 + record.candidate_index)
            object.__setattr__(record, "selected", True)
        after = materialize_stage_edit_rerank_snapshots(
            candidate_set,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )

        self.assertEqual(
            [snapshot.snapshot_sha256 for snapshot in before],
            [snapshot.snapshot_sha256 for snapshot in after],
        )
        for snapshot in after:
            payload = snapshot.to_dict()
            self.assertFalse(
                payload["provenance"]["reference_attribution_inputs_present"]
            )
            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn("review_rank", serialized)
            self.assertFalse(payload["input_policy"]["selected_labels_present"])


if __name__ == "__main__":
    unittest.main()
