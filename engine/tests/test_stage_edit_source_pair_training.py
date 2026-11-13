from __future__ import annotations

from dataclasses import replace
import unittest

from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from crosscut.stage_edit_source_pair_training import (
    StageEditSourcePairRankingMetrics,
    StageEditSourcePairTrainingError,
    fit_stage_edit_source_pair_scorer_from_reference_corpus,
    materialize_stage_edit_source_pair_training_corpus,
    stage_edit_source_pair_no_worse_activation_gate,
    stage_edit_source_pair_observation_from_candidate,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
)


def _candidate(
    source_id: str,
    face_scale: float,
    *,
    available: bool = True,
    hard_gate: bool = True,
    face_scale_missing: bool = False,
) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    values = [0.0] * schema.raw_source_dimension
    missing = [False] * schema.raw_source_dimension
    face_index = schema.source_feature_names.index("face_scale")
    values[face_index] = face_scale
    missing[face_index] = face_scale_missing
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(values),
        missing_mask=tuple(missing),
        available=available,
        hard_gate_passed=hard_gate,
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


def _example(
    song_id: str,
    reference_id: str,
    *,
    reverse_candidates: bool = False,
) -> ReferenceTrainingExample:
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
    if reverse_candidates:
        candidate_rows = tuple(tuple(reversed(row)) for row in candidate_rows)
    tokens = tuple(
        _token(float(index), candidates)
        for index, candidates in enumerate(candidate_rows)
    )
    targets = (
        StageEditTarget(cut=None, selected_source_id="A"),
        StageEditTarget(cut=1.0, selected_source_id="B"),
        # Unconfirmed labels must neither form a group nor change the prior.
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
    examples: tuple[ReferenceTrainingExample, ...],
) -> ReferenceTrainingCorpus:
    return ReferenceTrainingCorpus(examples, "pair-training-data", (), {})


class StageEditSourcePairReferenceTrainingTests(unittest.TestCase):
    def test_builds_deterministic_same_token_confirmed_switch_groups(self) -> None:
        first = _example("SONG-A", "reference-a")
        second = _example("SONG-B", "reference-b")
        materialized = materialize_stage_edit_source_pair_training_corpus(
            _corpus((second, first))
        )
        permuted = materialize_stage_edit_source_pair_training_corpus(
            _corpus(
                (
                    _example("SONG-A", "reference-a", reverse_candidates=True),
                    _example("SONG-B", "reference-b", reverse_candidates=True),
                )
            )
        )

        self.assertEqual(2, materialized.song_count)
        self.assertEqual(2, materialized.reference_count)
        self.assertEqual(4, materialized.group_count)
        self.assertEqual(2, materialized.opening_selection_excluded_count)
        self.assertEqual(2, materialized.unconfirmed_selection_excluded_count)
        self.assertEqual(0, materialized.same_source_selection_excluded_count)
        self.assertEqual(0, materialized.priorless_selection_excluded_count)
        self.assertEqual(
            (("SONG-A", 2), ("SONG-B", 2)),
            materialized.per_song_group_counts,
        )
        self.assertEqual(materialized.content_sha256, permuted.content_sha256)

        groups = {
            group.group_id: group for group in materialized.groups
        }
        first_switch = groups["reference-a:source-pair:000001"]
        self.assertEqual("A", first_switch.outgoing_source_id)
        self.assertEqual("B", first_switch.chosen_incoming_source_id)
        self.assertAlmostEqual(0.20, first_switch.outgoing.face_scale)
        self.assertEqual(
            ("B", "C"),
            tuple(candidate.source_id for candidate in first_switch.candidates),
        )
        second_switch = groups["reference-a:source-pair:000003"]
        # C was an unconfirmed label at token two. Prior remains confirmed B,
        # and the outgoing observation is B at this exact switch token (0.55).
        self.assertEqual("B", second_switch.outgoing_source_id)
        self.assertAlmostEqual(0.55, second_switch.outgoing.face_scale)

    def test_first_confirmed_source_bootstraps_unknown_prior_without_pair_label(
        self,
    ) -> None:
        base = _example("SONG-A", "reference-a")
        no_prior = replace(
            base,
            targets=replace(
                base.targets,
                targets=(StageEditTarget(), *base.targets.targets[1:]),
            ),
        )

        materialized = materialize_stage_edit_source_pair_training_corpus(
            _corpus((no_prior,))
        )

        self.assertEqual(1, materialized.priorless_selection_excluded_count)
        self.assertEqual(1, materialized.group_count)
        self.assertEqual(
            "reference-a:source-pair:000003",
            materialized.groups[0].group_id,
        )
        self.assertEqual("B", materialized.groups[0].outgoing_source_id)

    def test_confirmed_same_source_boundary_is_excluded_without_moving_prior(
        self,
    ) -> None:
        example = _example("SONG-A", "reference-a")
        targets = list(example.targets.targets)
        targets[1] = StageEditTarget(cut=1.0, selected_source_id="A")
        targets[2] = StageEditTarget(cut=1.0, selected_source_id="B")
        same_source = replace(
            example,
            targets=StageEditTargetSequence(
                example.reference_id,
                tuple(targets),
            ),
        )

        materialized = materialize_stage_edit_source_pair_training_corpus(
            _corpus((same_source,))
        )

        self.assertEqual(1, materialized.same_source_selection_excluded_count)
        self.assertEqual(2, materialized.group_count)
        first_switch = materialized.groups[0]
        self.assertEqual("reference-a:source-pair:000002", first_switch.group_id)
        self.assertEqual("A", first_switch.outgoing_source_id)
        self.assertEqual("B", first_switch.chosen_incoming_source_id)

    def test_schema_names_and_missing_mask_drive_observation(self) -> None:
        candidate = _candidate("A", 0.90, face_scale_missing=True)
        observation = stage_edit_source_pair_observation_from_candidate(candidate)
        self.assertIsNone(observation.face_scale)
        self.assertEqual(0.0, observation.camera_roll_normalized)

        malformed = replace(candidate, features=candidate.features[:-1])
        with self.assertRaisesRegex(
            StageEditSourcePairTrainingError, "feature length differs"
        ):
            stage_edit_source_pair_observation_from_candidate(malformed)

    def test_corpus_contract_fails_closed_on_identity_and_eligibility_errors(self) -> None:
        base = _example("SONG-A", "reference-a")

        cases: list[tuple[str, ReferenceTrainingExample, str]] = []
        cases.append(
            (
                "length",
                replace(
                    base,
                    targets=replace(
                        base.targets, targets=base.targets.targets[:-1]
                    ),
                ),
                "length mismatch",
            )
        )
        duplicate_tokens = list(base.sequence.tokens)
        duplicate_tokens[1] = replace(
            duplicate_tokens[1],
            source_candidates=(
                duplicate_tokens[1].source_candidates[0],
                duplicate_tokens[1].source_candidates[0],
                duplicate_tokens[1].source_candidates[1],
            ),
        )
        cases.append(
            (
                "duplicate",
                replace(
                    base,
                    sequence=replace(base.sequence, tokens=tuple(duplicate_tokens)),
                ),
                "duplicate source ids",
            )
        )
        missing_outgoing_tokens = list(base.sequence.tokens)
        missing_outgoing_tokens[1] = replace(
            missing_outgoing_tokens[1],
            source_candidates=(
                _candidate("B", 0.25),
                _candidate("C", 0.90),
            ),
        )
        cases.append(
            (
                "outgoing",
                replace(
                    base,
                    sequence=replace(
                        base.sequence, tokens=tuple(missing_outgoing_tokens)
                    ),
                ),
                "prior/outgoing",
            )
        )
        gated_tokens = list(base.sequence.tokens)
        gated_tokens[1] = replace(
            gated_tokens[1],
            source_candidates=(
                _candidate("A", 0.20),
                _candidate("B", 0.25, hard_gate=False),
                _candidate("C", 0.90),
            ),
        )
        cases.append(
            (
                "chosen",
                replace(
                    base,
                    sequence=replace(base.sequence, tokens=tuple(gated_tokens)),
                ),
                "chosen source is not eligible",
            )
        )
        no_alternative_tokens = list(base.sequence.tokens)
        no_alternative_tokens[1] = replace(
            no_alternative_tokens[1],
            source_candidates=(
                _candidate("A", 0.20),
                _candidate("B", 0.25),
            ),
        )
        cases.append(
            (
                "no_viable_alternative",
                replace(
                    base,
                    sequence=replace(
                        base.sequence, tokens=tuple(no_alternative_tokens)
                    ),
                ),
                "fewer than two viable incoming candidates",
            )
        )

        for name, example, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                StageEditSourcePairTrainingError, message
            ):
                materialize_stage_edit_source_pair_training_corpus(
                    _corpus((example,))
                )

    def test_fit_compares_reference_balanced_baselines_and_gates_activation(self) -> None:
        result = fit_stage_edit_source_pair_scorer_from_reference_corpus(
            _corpus(
                (
                    _example("SONG-A", "reference-a"),
                    _example("SONG-B", "reference-b"),
                )
            ),
            steps=120,
            learning_rate=0.30,
        )

        self.assertEqual(2, result.training.reference_count)
        self.assertEqual(4, result.training.group_count)
        self.assertLessEqual(result.training.final_loss, result.training.initial_loss)
        self.assertTrue(result.activation_authorized, result.activation_gate.reasons)
        for metric in (
            "reference_balanced_top1_accuracy",
            "reference_balanced_pairwise_accuracy",
            "reference_balanced_mean_reciprocal_rank",
        ):
            learned = getattr(result.learned_metrics, metric)
            self.assertGreaterEqual(learned, getattr(result.uniform_metrics, metric))
            self.assertGreaterEqual(
                learned, getattr(result.handcrafted_metrics, metric)
            )
        self.assertEqual(64, len(result.corpus.content_sha256))
        self.assertEqual(64, len(result.training.corpus_sha256))

    def test_no_worse_gate_rejects_a_regressed_learned_ranker(self) -> None:
        def metrics(name: str, value: float) -> StageEditSourcePairRankingMetrics:
            return StageEditSourcePairRankingMetrics(
                scorer_name=name,
                reference_count=2,
                group_count=4,
                reference_balanced_top1_accuracy=value,
                reference_balanced_pairwise_accuracy=value,
                reference_balanced_mean_reciprocal_rank=value,
                reference_balanced_mean_chosen_margin=0.0,
            )

        gate = stage_edit_source_pair_no_worse_activation_gate(
            metrics("learned", 0.49),
            metrics("uniform", 0.50),
            metrics("handcrafted", 0.60),
        )
        self.assertFalse(gate.authorized)
        self.assertEqual(3, len(gate.reasons))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
