from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import REFERENCE_ATTRIBUTION_BASIS
from crosscut.stage_edit_reference_adapter import load_reference_training_corpus
from crosscut.stage_edit_training import (
    select_stage_edit_cut_threshold,
    stage_edit_source_evaluation_contract,
    stage_edit_validation_selection_score,
    train_sparse_stage_edit,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditTransformerConfig,
    load_stage_edit_checkpoint,
    mlx_runtime_version,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


def _payload(reference_id: str, selected_pattern: tuple[str, ...]) -> dict:
    starts = (0.0, 1.0, 2.0, 3.0)
    shot_labels = []
    rankings = []
    for index, (start, selected) in enumerate(zip(starts, selected_pattern, strict=True)):
        rejected = "B" if selected == "A" else "A"
        previous = selected_pattern[index - 1] if index else None
        is_editorial = previous is not None and previous != selected
        boundary = (
            "reference_start"
            if index == 0
            else (
                "cross_source_hard_cut"
                if is_editorial
                else "same_source_internal_camera_cut"
            )
        )
        event_type = (
            "reference_start"
            if index == 0
            else (
                "editorial_source_switch_hard_cut"
                if is_editorial
                else "native_source_camera_cut"
            )
        )
        labels = {
            "cut": {
                "boundary_label": boundary,
                "cut_master_time": start,
                "end_master_time": start + 1.0,
                "duration_seconds": 1.0,
            },
            "source_transition": {
                "previous_source_id": previous,
                "selected_source_id": selected,
                "is_source_switch": is_editorial,
                "supervision_eligible": is_editorial,
                "transition": "hard_cut" if is_editorial else None,
            },
            "boundary_event": {
                "schema_version": 1,
                "event_type": event_type,
                "event_scope": "editorial" if is_editorial else "reference",
                "previous_source_id": previous,
                "selected_source_id": selected,
                "is_native_source_camera_cut": (
                    event_type == "native_source_camera_cut"
                ),
                "is_editorial_source_switch": is_editorial,
                "editorial_timing_supervision_eligible": is_editorial,
                "editorial_source_selection_supervision_eligible": is_editorial,
                "editorial_duration_supervision_eligible": is_editorial,
                "editorial_transition_supervision_eligible": is_editorial,
                "dense_attribution_ready": True,
                "label_basis": "test_fixture",
            },
            "dense_attribution": {
                "schema_version": 1,
                "ready": True,
                "reason": "validated_synchronized_sample_path",
                "first_sample_index": index,
                "last_sample_index": index,
                "sample_count": 1,
            },
            "crop": {"supervision_eligible": False, "transform": None},
        }
        shot_labels.append(
            {
                "shot_index": index,
                "selected_source_id": selected,
                "edit_labels": labels,
            }
        )
        rankings.append(
            {
                "shot_index": index,
                "chosen_source_id": selected,
                "candidates": [
                    {"source_id": selected, "selected": True, "evidence": {"score": 0.9}},
                    {"source_id": rejected, "selected": False, "evidence": {"score": 0.1}},
                ],
            }
        )
    return {
        "schema_version": 2,
        "dataset_kind": "reference_stage_mix_pairwise_listwise_supervision",
        "style_id": "training-test",
        "reference_id": reference_id,
        "attribution_analysis": REFERENCE_ATTRIBUTION_BASIS,
        "dense_attribution_schema": {
            "schema_version": 1,
            "temporal_granularity": "synchronized_sample_path",
            "sample_level_source_path_available": True,
            "shot_spans_reference_contiguous_source_path_samples": True,
            "frame_dense_boundary_proof_required_for_hard_cut": True,
            "native_source_camera_cut_is_not_editorial_source_switch": True,
        },
        "source_catalog": [
            {"video_id": "A", "alignment_score": 0.85},
            {"video_id": "B", "alignment_score": 0.65},
        ],
        "shot_labels": shot_labels,
        "listwise_records": rankings,
    }


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditSparseTrainingTests(unittest.TestCase):
    def test_validation_composite_and_threshold_are_predeclared_and_finite(
        self,
    ) -> None:
        selection = stage_edit_validation_selection_score(
            {
                "total_loss": 3.0,
                "cut": {"auroc": 0.75},
                "source": {"mean_reciprocal_rank": 0.60},
            }
        )
        self.assertTrue(selection["selectable"])
        self.assertAlmostEqual(0.5975, selection["score"])
        self.assertFalse(selection["heldout_metrics_consumed"])

        threshold = select_stage_edit_cut_threshold(
            (0, 0, 1, 1),
            (0.1, 0.4, 0.6, 0.9),
        )
        self.assertAlmostEqual(0.5, threshold["selected_probability_threshold"])
        self.assertEqual(1.0, threshold["validation_balanced_accuracy"])
        self.assertFalse(threshold["heldout_labels_consumed"])
        source_contract = stage_edit_source_evaluation_contract()
        self.assertEqual(
            0.35,
            source_contract["end_to_end_decoder_metrics"][
                "fixed_pair_continuity_weight"
            ],
        )
        self.assertNotEqual(
            source_contract["head_metrics"]["scope"],
            source_contract["end_to_end_decoder_metrics"]["scope"],
        )

    def test_training_saves_only_validated_checkpoint_and_reports_untrained_crop(self) -> None:
        schema = DEFAULT_STAGE_EDIT_SCHEMA
        config = StageEditTransformerConfig.from_schema(
            schema,
            model_dimension=16,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=32,
            max_sequence_length=16,
            max_sources=4,
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            specifications = (
                ("ref-a", ("A", "B", "A", "B"), "song-a"),
                ("ref-b", ("B", "A", "B", "A"), "song-b"),
            )
            song_map = {}
            for reference, pattern, song in specifications:
                path = root / f"{reference}.json"
                path.write_text(json.dumps(_payload(reference, pattern)), encoding="utf-8")
                inputs.append(path)
                song_map[reference] = song
            corpus = load_reference_training_corpus(
                inputs,
                song_id_by_reference=song_map,
                max_sources=4,
            )
            checkpoint = root / "checkpoint"
            report_path = root / "reports" / "training.json"
            result = train_sparse_stage_edit(
                corpus,
                checkpoint_directory=checkpoint,
                report_path=report_path,
                model_config=config,
                epochs=4,
                early_stopping_patience=2,
                learning_rate=1.0e-3,
                gradient_clip_norm=0.5,
                seed=31,
                device=mx.cpu,
            )
            self.assertGreater(result.best_training_step, 0)
            self.assertTrue((checkpoint / "model.safetensors").is_file())
            self.assertTrue((checkpoint / "metadata.json").is_file())
            self.assertTrue(report_path.is_file())
            model, metadata = load_stage_edit_checkpoint(
                checkpoint, expected_schema=schema
            )
            self.assertIsNotNone(model)
            self.assertTrue(metadata.weights_trained)
            self.assertGreater(metadata.training_steps, 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["production_ready"])
            self.assertTrue(report["production_supervision_gate"]["passed"])
            self.assertEqual(
                report["head_training_status"]["crop"],
                "masked_untrained_no_labels",
            )
            self.assertEqual(
                report["head_training_status"]["transition"],
                "trained_sparse",
            )
            self.assertTrue(
                report["checkpoint"]["saved_only_after_validation_improvement"]
            )
            self.assertGreater(
                report["optimization"]["validation_improvement_count"], 0
            )
            self.assertTrue(
                report["optimization"]["gradient_clipping_supported"]
            )
            metrics = report["best_validation_metrics"]
            self.assertGreater(metrics["cut"]["positive_count"], 0)
            self.assertGreater(metrics["cut"]["negative_count"], 0)
            self.assertIsNotNone(metrics["cut"]["auroc"])
            self.assertGreater(metrics["source"]["listwise_group_count"], 0)
            self.assertIsNotNone(metrics["source"]["top1_accuracy"])
            self.assertGreater(metrics["duration"]["label_count"], 0)
            self.assertGreater(metrics["transition"]["label_count"], 0)
            self.assertEqual(metrics["crop"]["coordinate_label_count"], 0)


if __name__ == "__main__":
    unittest.main()
