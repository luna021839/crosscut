from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import crosscut.stage_edit_learned_reviewer as reviewer_module
from crosscut.stage_edit_learned_reviewer import (
    CANDIDATE,
    DEFAULT_LEARNED_REVIEWER_SCHEMA,
    HUMAN_REFERENCE,
    LEARNED_REVIEWER_CHECKPOINT_SCOPE,
    LEARNED_REVIEWER_REPORT_FILENAME,
    LEARNED_REVIEWER_REPORT_VERSION,
    LEARNED_REVIEWER_RUN_KIND,
    SYNTHETIC_SHIFTED,
    SYNTHETIC_TECHNICAL,
    SYNTHETIC_WORSENED,
    LearnedReviewerBoundary,
    LearnedReviewerCheckpointMetadata,
    LearnedReviewerConfig,
    LearnedReviewerCorpus,
    LearnedReviewerExample,
    LearnedReviewerInputError,
    LearnedReviewerListwiseGroup,
    LearnedReviewerNotReadyError,
    LearnedReviewerPairwisePreference,
    StageEditLearnedReviewer,
    learned_reviewer_loss,
    learned_reviewer_production_data_gate,
    learned_reviewer_readiness_report,
    load_learned_reviewer_checkpoint,
    load_production_learned_reviewer,
    mlx_runtime_version,
    new_learned_reviewer_checkpoint_metadata,
    predict_learned_reviewer,
    save_learned_reviewer_checkpoint,
    split_learned_reviewer_corpus,
    tensorize_learned_reviewer_corpus,
    train_learned_reviewer,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except ImportError:  # pragma: no cover - optional on non-Apple CI.
    mx = None
    nn = None
    optim = None
    tree_flatten = None


SCHEMA = DEFAULT_LEARNED_REVIEWER_SCHEMA


def _features(
    severity: float,
    *,
    technical: bool = False,
    missing_first: bool = False,
) -> tuple[float | None, ...]:
    values: list[float | None] = [severity] * SCHEMA.boundary_dimension
    for index in SCHEMA.technical_feature_indices:
        values[index] = 0.0
    if technical:
        values[SCHEMA.technical_feature_indices[0]] = 1.0
    if missing_first:
        values[0] = None
    return tuple(values)


def _timeline(severity: float, *, missing_first: bool = False) -> tuple[float | None, ...]:
    values: list[float | None] = [severity] * SCHEMA.timeline_dimension
    if missing_first:
        values[0] = None
    return tuple(values)


def _example(
    song: str,
    kind: str,
    suffix: str,
    quality: float,
    severity: float,
    *,
    parent: str | None = None,
    technical: bool = False,
    missing: bool = False,
) -> LearnedReviewerExample:
    issue = min(1.0, max(0.0, severity))
    causes = [issue] * len(SCHEMA.cause_labels)
    causes[-1] = 1.0 if technical else 0.0
    return LearnedReviewerExample(
        example_id=f"{song}-{suffix}",
        song_id=song,
        reference_id=f"reference-{song}",
        example_kind=kind,
        boundaries=tuple(
            LearnedReviewerBoundary(
                boundary_id=f"{song}-{suffix}-b{index}",
                at_seconds=float(index + 1),
                features=_features(
                    severity,
                    technical=technical,
                    missing_first=missing and index == 0,
                ),
                issue_target=issue,
                cause_targets=tuple(causes),
                technical_error_codes=("black_frame",) if technical else (),
            )
            for index in range(2)
        ),
        timeline_features=_timeline(severity, missing_first=missing),
        quality_target=quality,
        parent_human_example_id=parent,
    )


def _corpus(song_ids: tuple[str, ...] = ("song-a", "song-b", "song-c")) -> LearnedReviewerCorpus:
    examples: list[LearnedReviewerExample] = []
    preferences: list[LearnedReviewerPairwisePreference] = []
    groups: list[LearnedReviewerListwiseGroup] = []
    for song in song_ids:
        human_id = f"{song}-human"
        examples.extend(
            (
                _example(song, HUMAN_REFERENCE, "human", 0.95, 0.05),
                _example(
                    song,
                    SYNTHETIC_SHIFTED,
                    "shifted",
                    0.62,
                    0.40,
                    parent=human_id,
                ),
                _example(
                    song,
                    SYNTHETIC_WORSENED,
                    "worsened",
                    0.15,
                    0.85,
                    parent=human_id,
                ),
                _example(
                    song,
                    SYNTHETIC_TECHNICAL,
                    "technical",
                    0.0,
                    1.0,
                    parent=human_id,
                    technical=True,
                ),
                _example(
                    song,
                    CANDIDATE,
                    "candidate-good",
                    0.82,
                    0.18,
                    missing=True,
                ),
                _example(
                    song,
                    CANDIDATE,
                    "candidate-bad",
                    0.28,
                    0.72,
                ),
            )
        )
        ordered_pairs = (
            ("human-over-worsened", human_id, f"{song}-worsened"),
            ("human-over-shifted", human_id, f"{song}-shifted"),
            (
                "candidate-good-over-bad",
                f"{song}-candidate-good",
                f"{song}-candidate-bad",
            ),
            ("human-over-technical", human_id, f"{song}-technical"),
        )
        preferences.extend(
            LearnedReviewerPairwisePreference(
                preference_id=f"{song}-{name}",
                group_id=f"{song}-pairwise",
                preferred_example_id=preferred,
                rejected_example_id=rejected,
            )
            for name, preferred, rejected in ordered_pairs
        )
        groups.append(
            LearnedReviewerListwiseGroup(
                group_id=f"{song}-listwise",
                ordered_example_ids=(
                    human_id,
                    f"{song}-candidate-good",
                    f"{song}-shifted",
                    f"{song}-candidate-bad",
                    f"{song}-worsened",
                    f"{song}-technical",
                ),
            )
        )
    return LearnedReviewerCorpus(
        corpus_version=1,
        corpus_id="synthetic-learned-reviewer-corpus",
        feature_schema=SCHEMA,
        examples=tuple(examples),
        pairwise_preferences=tuple(preferences),
        listwise_groups=tuple(groups),
    )


def _small_config() -> LearnedReviewerConfig:
    return LearnedReviewerConfig(
        feature_schema_fingerprint=SCHEMA.fingerprint,
        boundary_input_dimension=SCHEMA.encoded_boundary_dimension,
        timeline_input_dimension=SCHEMA.encoded_timeline_dimension,
        cause_count=len(SCHEMA.cause_labels),
        max_boundaries=8,
        model_dimension=8,
        num_layers=1,
        num_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
    )


def _ready_report(
    directory: Path,
    corpus: LearnedReviewerCorpus,
    metadata: LearnedReviewerCheckpointMetadata,
) -> dict[str, object]:
    split = split_learned_reviewer_corpus(corpus)
    gate = learned_reviewer_production_data_gate(corpus, split)
    metadata_sha = reviewer_module._sha256_file(directory / "metadata.json")
    weight_sha = reviewer_module._sha256_file(
        directory / metadata.weight_filename
    )
    return {
        "schema_version": LEARNED_REVIEWER_REPORT_VERSION,
        "run_kind": LEARNED_REVIEWER_RUN_KIND,
        "checkpoint_scope": LEARNED_REVIEWER_CHECKPOINT_SCOPE,
        "production_ready": True,
        "production_activation_authorized": True,
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "corpus_sha256": corpus.sha256,
            "feature_schema_fingerprint": corpus.feature_schema.fingerprint,
        },
        "checkpoint": {
            "checkpoint_id": metadata.checkpoint_id,
            "training_steps": metadata.training_steps,
            "weight_sha256": weight_sha,
            "metadata_sha256": metadata_sha,
        },
        "production_data_gate": gate.to_dict(),
        "split": {
            "split_level": "song",
            "training_song_ids": list(split.training_song_ids),
            "heldout_song_ids": list(split.heldout_song_ids),
        },
        "held_out_evaluation": {
            "passed": True,
            "checkpoint_id": metadata.checkpoint_id,
            "weight_sha256": weight_sha,
            "corpus_sha256": corpus.sha256,
            "split_level": "song",
            "training_song_ids": list(split.training_song_ids),
            "heldout_song_ids": list(split.heldout_song_ids),
            "human_count": 1,
            "worsened_count": 1,
            "rank_pair_count": 4,
            "quality_label_count": 6,
            "human_vs_worsened_auc": 0.95,
            "mean_rank_margin": 0.25,
            "brier_calibration_error": 0.08,
            "technical_zero_tolerance_policy_verified": True,
            "thresholds": {
                "minimum_human_vs_worsened_auc": 0.80,
                "minimum_mean_rank_margin": 0.10,
                "maximum_brier_calibration_error": 0.20,
            },
            "loss_components": {"total": 0.2, "cause_bce": 0.1},
        },
        "training": {
            "epochs_completed": 2,
            "best_epoch": 2,
            "best_training_step": metadata.training_steps,
            "seed": 20260824,
            "loss_contract": {"pairwise_margin_weight": 1.0},
            "history": [{"epoch": 1}, {"epoch": 2}],
        },
    }


class LearnedReviewerCorpusTests(unittest.TestCase):
    def test_strict_corpus_round_trip_missing_masks_and_song_split(self) -> None:
        corpus = _corpus()
        restored = LearnedReviewerCorpus.from_dict(corpus.to_dict())
        self.assertEqual(restored, corpus)
        self.assertEqual(restored.sha256, corpus.sha256)
        split = split_learned_reviewer_corpus(corpus, seed=19)
        self.assertEqual(split.split_level, "song")
        self.assertTrue(set(split.training_song_ids).isdisjoint(split.heldout_song_ids))
        self.assertEqual(len(split.training_song_ids), 2)
        self.assertEqual(len(split.heldout_song_ids), 1)

    def test_strict_corpus_rejects_unsupervised_candidate(self) -> None:
        corpus = _corpus(("song-a",))
        preferences = tuple(
            value
            for value in corpus.pairwise_preferences
            if "candidate" not in value.preference_id
        )
        groups: tuple[LearnedReviewerListwiseGroup, ...] = ()
        with self.assertRaisesRegex(LearnedReviewerInputError, "candidate"):
            LearnedReviewerCorpus(
                corpus_version=1,
                corpus_id=corpus.corpus_id,
                feature_schema=SCHEMA,
                examples=corpus.examples,
                pairwise_preferences=preferences,
                listwise_groups=groups,
            )

    def test_one_song_data_gate_is_fail_closed(self) -> None:
        corpus = _corpus(("only-human-song",))
        split = split_learned_reviewer_corpus(corpus)
        gate = learned_reviewer_production_data_gate(corpus, split)
        self.assertFalse(gate.passed)
        self.assertEqual(gate.observed["song_count"], 1)
        self.assertEqual(gate.observed["heldout_song_count"], 0)

    def test_corpus_version_boolean_is_not_coerced_to_one(self) -> None:
        payload = _corpus().to_dict()
        payload["corpus_version"] = True
        with self.assertRaisesRegex(LearnedReviewerInputError, "integer"):
            LearnedReviewerCorpus.from_dict(payload)


@unittest.skipIf(mlx_runtime_version() is None, "MLX is not installed")
class LearnedReviewerMLXTests(unittest.TestCase):
    def test_real_mlx_forward_loss_gradient_and_missing_mask(self) -> None:
        corpus = _corpus()
        config = _small_config()
        batch, supervision = tensorize_learned_reviewer_corpus(
            corpus,
            tuple(example.example_id for example in corpus.examples),
            config,
        )
        candidate_index = batch.example_ids.index("song-a-candidate-good")
        self.assertFalse(bool(np.asarray(batch.boundary_observed_mask)[candidate_index, 0, 0]))
        model = StageEditLearnedReviewer(config)
        output = model(batch)
        mx.eval(
            output.global_quality_logits,
            output.boundary_issue_logits,
            output.boundary_cause_logits,
        )
        self.assertEqual(output.global_quality_logits.shape, (18,))
        self.assertEqual(output.boundary_issue_logits.shape, (18, 2))
        self.assertEqual(output.boundary_cause_logits.shape, (18, 2, 7))
        loss, components = learned_reviewer_loss(model, batch, supervision)
        mx.eval(loss, components)
        self.assertTrue(np.isfinite(float(np.asarray(loss))))
        self.assertEqual(
            set(components),
            {
                "total",
                "pairwise_ranking",
                "listwise_ranking",
                "boundary_issue_bce",
                "cause_bce",
                "calibration_brier",
            },
        )

        def loss_function(trained_model: StageEditLearnedReviewer):
            return learned_reviewer_loss(trained_model, batch, supervision)[0]

        optimizer = optim.AdamW(learning_rate=1.0e-3)
        loss_and_grad = nn.value_and_grad(model, loss_function)
        first_before = np.array(tree_flatten(model.parameters())[0][1])
        _, gradients = loss_and_grad(model)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        first_after = np.array(tree_flatten(model.parameters())[0][1])
        self.assertFalse(np.array_equal(first_before, first_after))

    def test_technical_zero_tolerance_cannot_be_overridden_by_high_score(self) -> None:
        corpus = _corpus(("song-a",))
        technical = next(
            example
            for example in corpus.examples
            if example.example_kind == SYNTHETIC_TECHNICAL
        )
        model = StageEditLearnedReviewer(_small_config())
        prediction = predict_learned_reviewer(
            model,
            (technical,),
            quality_threshold=0.0,
        )[0]
        self.assertTrue(prediction.model_recommendation_passed)
        self.assertFalse(prediction.technical_zero_tolerance_passed)
        self.assertFalse(prediction.review_passed)
        self.assertTrue(
            all(item.technical_hard_failure for item in prediction.boundary_predictions)
        )

    def test_missing_technical_evidence_is_also_a_hard_failure(self) -> None:
        human = next(
            example
            for example in _corpus(("song-a",)).examples
            if example.example_kind == HUMAN_REFERENCE
        )
        boundary = human.boundaries[0]
        values = list(boundary.features)
        technical_name = SCHEMA.boundary_feature_names[
            SCHEMA.technical_feature_indices[0]
        ]
        values[SCHEMA.technical_feature_indices[0]] = None
        missing_boundary = replace(boundary, features=tuple(values))
        example = replace(
            human,
            boundaries=(missing_boundary, *human.boundaries[1:]),
        )
        prediction = predict_learned_reviewer(
            StageEditLearnedReviewer(_small_config()),
            (example,),
            quality_threshold=0.0,
        )[0]
        self.assertFalse(prediction.review_passed)
        self.assertIn(
            f"missing_{technical_name}",
            prediction.boundary_predictions[0].technical_error_codes,
        )

    def test_checkpoint_round_trip_and_weight_tamper_rejected(self) -> None:
        corpus = _corpus()
        model = StageEditLearnedReviewer(_small_config())
        metadata = new_learned_reviewer_checkpoint_metadata(
            "reviewer-round-trip",
            corpus,
            model.config,
            training_steps=2,
            weights_trained=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = save_learned_reviewer_checkpoint(root, model, metadata)
            loaded_model, loaded = load_learned_reviewer_checkpoint(root)
            self.assertEqual(loaded, saved)
            self.assertIsInstance(loaded_model, StageEditLearnedReviewer)
            with (root / saved.weight_filename).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(LearnedReviewerNotReadyError, "digest"):
                load_learned_reviewer_checkpoint(root)

    def test_strict_ready_loader_and_report_toctou(self) -> None:
        corpus = _corpus()
        model = StageEditLearnedReviewer(_small_config())
        metadata = new_learned_reviewer_checkpoint_metadata(
            "reviewer-production",
            corpus,
            model.config,
            training_steps=2,
            weights_trained=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = save_learned_reviewer_checkpoint(root, model, metadata)
            report_path = root / LEARNED_REVIEWER_REPORT_FILENAME
            report_path.write_text(
                json.dumps(_ready_report(root, corpus, saved), allow_nan=False),
                encoding="utf-8",
            )
            readiness = learned_reviewer_readiness_report(root)
            self.assertTrue(readiness.learned_inference_ready, readiness.reason)
            _, loaded, strict_readiness, evidence = load_production_learned_reviewer(
                root, report_path
            )
            self.assertEqual(loaded, saved)
            self.assertTrue(strict_readiness.learned_inference_ready)
            self.assertEqual(len(evidence["bundle_sha256"]), 64)
            outside = root / "copied-report.json"
            outside.write_bytes(report_path.read_bytes())
            with self.assertRaisesRegex(LearnedReviewerNotReadyError, "colocated"):
                load_production_learned_reviewer(root, outside)

            original_loader = reviewer_module.load_learned_reviewer_checkpoint

            def mutate_report(*args, **kwargs):
                result = original_loader(*args, **kwargs)
                report_path.write_text(
                    report_path.read_text(encoding="utf-8") + " ",
                    encoding="utf-8",
                )
                return result

            with patch.object(
                reviewer_module,
                "load_learned_reviewer_checkpoint",
                side_effect=mutate_report,
            ):
                with self.assertRaisesRegex(
                    LearnedReviewerNotReadyError, "changed|rejected"
                ):
                    load_production_learned_reviewer(root, report_path)

    def test_report_authorization_and_metric_thresholds_fail_closed(self) -> None:
        corpus = _corpus()
        model = StageEditLearnedReviewer(_small_config())
        metadata = new_learned_reviewer_checkpoint_metadata(
            "reviewer-unauthorized",
            corpus,
            model.config,
            training_steps=2,
            weights_trained=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = save_learned_reviewer_checkpoint(root, model, metadata)
            report = _ready_report(root, corpus, saved)
            report["production_activation_authorized"] = False
            report["held_out_evaluation"]["human_vs_worsened_auc"] = 0.79
            (root / LEARNED_REVIEWER_REPORT_FILENAME).write_text(
                json.dumps(report, allow_nan=False), encoding="utf-8"
            )
            readiness = learned_reviewer_readiness_report(root)
            self.assertFalse(readiness.learned_inference_ready)
            self.assertIn("authorized", readiness.reason)
            self.assertIn("AUROC", readiness.reason)
            report["split"]["training_song_ids"] = [[]]
            (root / LEARNED_REVIEWER_REPORT_FILENAME).write_text(
                json.dumps(report, allow_nan=False), encoding="utf-8"
            )
            malformed = learned_reviewer_readiness_report(root)
            self.assertFalse(malformed.learned_inference_ready)
            self.assertEqual(malformed.status, "production_evidence_invalid")

    def test_current_one_song_sparse_training_checkpoint_never_ready(self) -> None:
        corpus = _corpus(("current-human-reference-song",))
        with tempfile.TemporaryDirectory() as temporary:
            result = train_learned_reviewer(
                corpus,
                checkpoint_directory=Path(temporary),
                model_config=_small_config(),
                epochs=1,
                early_stopping_patience=1,
                authorize_production=True,
                seed=7,
            )
            self.assertFalse(result.report["production_ready"])
            self.assertFalse(result.report["production_data_gate"]["passed"])
            self.assertEqual(
                result.report["production_data_gate"]["observed"]["song_count"],
                1,
            )
            self.assertFalse(result.readiness.learned_inference_ready)
            with self.assertRaises(LearnedReviewerNotReadyError):
                load_production_learned_reviewer(
                    Path(temporary), result.report_path
                )

    def test_metadata_rejects_string_boolean(self) -> None:
        corpus = _corpus()
        metadata = new_learned_reviewer_checkpoint_metadata(
            "strict-types",
            corpus,
            _small_config(),
            training_steps=1,
            weights_trained=True,
        ).to_dict()
        metadata["weights_trained"] = "false"
        with self.assertRaisesRegex(LearnedReviewerInputError, "boolean"):
            LearnedReviewerCheckpointMetadata.from_dict(metadata)


if __name__ == "__main__":
    unittest.main()
