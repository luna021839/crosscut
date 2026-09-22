from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crosscut.stage_edit_source_pair_production as production
from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from crosscut.stage_edit_source_pair_production import (
    STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME,
    StageEditSourcePairCrossValidationError,
    StageEditSourcePairProductionGateError,
    cross_validate_stage_edit_source_pair_materialized_corpus,
    cross_validate_stage_edit_source_pair_reference_corpus,
    produce_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    load_stage_edit_source_pair_checkpoint,
    train_stage_edit_source_pair_scorer,
)
from crosscut.stage_edit_source_pair_training import (
    evaluate_stage_edit_source_pair_ranking,
    materialize_stage_edit_source_pair_training_corpus,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
)


def _candidate(source_id: str, face_scale: float) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    values = [0.0] * schema.raw_source_dimension
    values[schema.source_feature_names.index("face_scale")] = face_scale
    causal_sensing_names = set(STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES)
    missing_mask = tuple(
        name in causal_sensing_names for name in schema.source_feature_names
    )
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(values),
        missing_mask=missing_mask,
        available=True,
        hard_gate_passed=True,
    )


def _token(
    time_seconds: float,
    candidates: tuple[SourceCandidateToken, ...],
) -> StageEditToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return StageEditToken(
        master_time_seconds=time_seconds,
        music_features=(0.0,) * len(schema.music_feature_names),
        section_features=(0.0,) * len(schema.section_feature_names),
        source_candidates=candidates,
        recent_edit_features=(0.0,) * len(schema.recent_edit_feature_names),
        style_features=(0.0,) * len(schema.style_feature_names),
        missing_mask=(False,) * schema.raw_context_dimension,
    )


def _example(song_id: str, reference_id: str) -> ReferenceTrainingExample:
    candidate_rows = (
        (
            _candidate("A", 0.20),
            _candidate("B", 0.80),
            _candidate("C", 0.50),
        ),
        (
            _candidate("A", 0.20),
            _candidate("B", 0.25),
            _candidate("C", 0.90),
        ),
        (
            _candidate("A", 0.70),
            _candidate("B", 0.25),
            _candidate("C", 0.26),
        ),
        (
            _candidate("A", 0.60),
            _candidate("B", 0.55),
            _candidate("C", 0.95),
        ),
    )
    tokens = tuple(
        _token(float(index), candidates)
        for index, candidates in enumerate(candidate_rows)
    )
    targets = (
        StageEditTarget(cut=None, selected_source_id="A"),
        StageEditTarget(cut=1.0, selected_source_id="B"),
        StageEditTarget(cut=None, selected_source_id="C"),
        StageEditTarget(cut=1.0, selected_source_id="A"),
    )
    return ReferenceTrainingExample(
        reference_id=reference_id,
        song_id=song_id,
        style_id="human-stage-mix",
        sequence=StageEditSequence(reference_id, tokens),
        targets=StageEditTargetSequence(reference_id, targets),
        statistics={},
    )


def _corpus(
    examples: tuple[ReferenceTrainingExample, ...] | None = None,
) -> ReferenceTrainingCorpus:
    values = examples or (
        _example("SONG-C", "reference-c"),
        _example("SONG-A", "reference-a"),
        _example("SONG-B", "reference-b"),
    )
    return ReferenceTrainingCorpus(values, "pair-production-data", (), {})


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageEditSourcePairProductionTests(unittest.TestCase):
    def test_three_song_leave_one_out_is_disjoint_and_deterministic(self) -> None:
        first = cross_validate_stage_edit_source_pair_reference_corpus(
            _corpus(), steps=60, learning_rate=0.30
        )
        second = cross_validate_stage_edit_source_pair_reference_corpus(
            _corpus(
                (
                    _example("SONG-B", "reference-b"),
                    _example("SONG-C", "reference-c"),
                    _example("SONG-A", "reference-a"),
                )
            ),
            steps=60,
            learning_rate=0.30,
        )

        self.assertTrue(first.authorized, first.reasons)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(3, len(first.folds))
        self.assertEqual(3, first.learned_metrics.reference_count)
        self.assertEqual(6, first.learned_metrics.group_count)
        self.assertEqual(64, len(first.validation_sha256))
        all_heldout_references: list[str] = []
        for fold in first.folds:
            self.assertNotIn(fold.heldout_song_id, fold.training_song_ids)
            self.assertTrue(
                set(fold.training_reference_ids).isdisjoint(
                    fold.heldout_reference_ids
                )
            )
            self.assertTrue(
                set(fold.training_group_ids).isdisjoint(fold.heldout_group_ids)
            )
            self.assertTrue(fold.activation_gate.authorized)
            all_heldout_references.extend(fold.heldout_reference_ids)
        self.assertEqual(
            list(first.corpus.reference_ids), sorted(all_heldout_references)
        )

    def test_production_materializes_once_saves_deterministically_and_never_overwrites(
        self,
    ) -> None:
        created_at = "2026-08-24T12:00:00+09:00"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_directory = root / "first"
            second_directory = root / "second"
            with patch.object(
                production,
                "materialize_stage_edit_source_pair_training_corpus",
                wraps=materialize_stage_edit_source_pair_training_corpus,
            ) as materialize:
                first = produce_stage_edit_source_pair_checkpoint(
                    _corpus(),
                    first_directory,
                    checkpoint_id="pair-production-v1",
                    created_at=created_at,
                    steps=60,
                    learning_rate=0.30,
                )
            self.assertEqual(1, materialize.call_count)
            self.assertEqual(
                (
                    first_directory / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
                ).resolve(),
                first.checkpoint_path,
            )
            self.assertEqual(
                (
                    first_directory
                    / STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME
                ).resolve(),
                first.report_path,
            )

            checkpoint_bytes = first.checkpoint_path.read_bytes()
            report_bytes = first.report_path.read_bytes()
            report_payload = json.loads(report_bytes)
            report_digest = report_payload.pop("report_sha256")
            self.assertEqual(report_digest, _canonical_sha256(report_payload))
            authorization = report_payload["production_authorization"]
            self.assertTrue(authorization["all_fold_gates_authorized"])
            self.assertTrue(authorization["overall_gate_authorized"])
            self.assertFalse(
                authorization["in_sample_fit_gate_used_for_authorization"]
            )
            self.assertEqual(1, authorization["materialization_pass_count"])
            _scorer, loaded = load_stage_edit_source_pair_checkpoint(
                first_directory,
                expected_checkpoint_sha256=(
                    first.report.checkpoint.checkpoint_sha256
                ),
            )
            self.assertEqual(first.report.checkpoint, loaded)

            second = produce_stage_edit_source_pair_checkpoint(
                _corpus(),
                second_directory,
                checkpoint_id="pair-production-v1",
                created_at=created_at,
                steps=60,
                learning_rate=0.30,
            )
            self.assertEqual(checkpoint_bytes, second.checkpoint_path.read_bytes())
            self.assertEqual(report_bytes, second.report_path.read_bytes())

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                produce_stage_edit_source_pair_checkpoint(
                    _corpus(),
                    first_directory,
                    checkpoint_id="pair-production-v1",
                    created_at=created_at,
                    steps=60,
                    learning_rate=0.30,
                )
            self.assertEqual(checkpoint_bytes, first.checkpoint_path.read_bytes())
            self.assertEqual(report_bytes, first.report_path.read_bytes())

    def test_regressed_heldout_fold_denies_before_final_fit_or_write(self) -> None:
        real_evaluate = evaluate_stage_edit_source_pair_ranking

        def regress_one_fold(
            groups, *, scorer, scorer_name, handcrafted_baseline=False
        ):
            metrics = real_evaluate(
                groups,
                scorer=scorer,
                scorer_name=scorer_name,
                handcrafted_baseline=handcrafted_baseline,
            )
            if scorer_name == "learned:heldout:SONG-C":
                return replace(
                    metrics,
                    reference_balanced_top1_accuracy=0.0,
                    reference_balanced_pairwise_accuracy=0.0,
                    reference_balanced_mean_reciprocal_rank=0.0,
                    reference_balanced_mean_chosen_margin=-1.0,
                )
            return metrics

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "denied"
            with patch.object(
                production,
                "evaluate_stage_edit_source_pair_ranking",
                side_effect=regress_one_fold,
            ), patch.object(
                production,
                "train_stage_edit_source_pair_scorer",
                wraps=train_stage_edit_source_pair_scorer,
            ) as train:
                with self.assertRaisesRegex(
                    StageEditSourcePairProductionGateError,
                    "heldout_song=SONG-C",
                ):
                    produce_stage_edit_source_pair_checkpoint(
                        _corpus(),
                        output,
                        checkpoint_id="must-not-exist",
                        created_at="2026-08-24T12:00:00+09:00",
                        steps=40,
                        learning_rate=0.30,
                    )
            # Three fold fits occurred. The post-gate full-corpus fit did not.
            self.assertEqual(3, train.call_count)
            self.assertFalse(output.exists())

    def test_fails_closed_for_too_few_songs_and_cross_song_reference_leakage(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            StageEditSourcePairCrossValidationError, "at least two songs"
        ):
            cross_validate_stage_edit_source_pair_reference_corpus(
                _corpus((_example("SONG-A", "reference-a"),)), steps=20
            )

        materialized = materialize_stage_edit_source_pair_training_corpus(_corpus())
        leaked_groups = tuple(
            replace(group, reference_id="reference-a")
            if dict(materialized.group_song_ids)[group.group_id] == "SONG-B"
            else group
            for group in materialized.groups
        )
        contaminated = replace(
            materialized,
            groups=leaked_groups,
            reference_ids=("reference-a", "reference-c"),
            per_reference_group_counts=(
                ("reference-a", 4),
                ("reference-c", 2),
            ),
        )
        with self.assertRaisesRegex(
            StageEditSourcePairCrossValidationError,
            "references span multiple songs",
        ):
            cross_validate_stage_edit_source_pair_materialized_corpus(
                contaminated, steps=20
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
