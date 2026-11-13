from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from crosscut.stage_edit_source_pair_contextual_diagnostic import (
    StageEditSourcePairContextualDiagnosticConfig,
    StageEditSourcePairContextualDiagnosticError,
    cross_validate_stage_edit_source_pair_contextual_diagnostic,
    materialize_stage_edit_source_pair_contextual_diagnostic_corpus,
    train_stage_edit_source_pair_contextual_residual,
)
from crosscut.stage_edit_source_pair_training import (
    StageEditSourcePairActivationGate,
    stage_edit_source_pair_no_worse_activation_gate,
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


def _candidate(
    source_id: str,
    *,
    shot_scale: float,
    face_scale: float,
    camera_motion: float,
    exposure: float,
) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    values = [0.0] * schema.raw_source_dimension
    for name, value in (
        ("sync_confidence", 0.9),
        ("shot_scale", shot_scale),
        ("face_scale", face_scale),
        ("face_center_x", 0.5),
        ("face_center_y", 0.5),
        ("body_center_x", 0.5),
        ("body_center_y", 0.5),
        ("camera_motion_x", camera_motion),
        ("camera_motion_y", 0.0),
        ("exposure", exposure),
        ("visual_match_confidence", 0.8),
        ("pose_model_confidence", 0.85),
        ("detected_person_count_normalized", shot_scale),
        ("formation_person_count_normalized", shot_scale),
        ("formation_pose_coverage", 0.75),
    ):
        values[schema.source_feature_names.index(name)] = value
    values[schema.source_feature_names.index("pose_model_embedding_00")] = (
        face_scale
    )
    values[schema.source_feature_names.index("formation_layout_embedding_000")] = (
        shot_scale
    )
    values[
        schema.source_feature_names.index("formation_pose_person_0_joint_0_x")
    ] = 0.5
    causal_sensing_names = set(STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES)
    missing_mask = tuple(
        name in causal_sensing_names for name in schema.source_feature_names
    )
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(values),
        missing_mask=missing_mask,
    )


def _token(
    index: int,
    candidates: tuple[SourceCandidateToken, ...],
    *,
    recent_value: float,
) -> StageEditToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    music = [0.0] * len(schema.music_feature_names)
    music[schema.music_feature_names.index("downbeat_probability")] = (
        1.0 if index % 2 == 0 else 0.0
    )
    music[schema.music_feature_names.index("onset_strength")] = index / 4.0
    section = [0.0] * len(schema.section_feature_names)
    section[schema.section_feature_names.index("chorus_probability")] = (
        1.0 if index >= 2 else 0.0
    )
    section[schema.section_feature_names.index("verse_probability")] = (
        1.0 if index < 2 else 0.0
    )
    style = [0.0] * len(schema.style_feature_names)
    style[schema.style_feature_names.index("matchcut_strength")] = 0.7
    style[schema.style_feature_names.index("closeup_preference")] = 0.6
    return StageEditToken(
        master_time_seconds=float(index),
        music_features=tuple(music),
        section_features=tuple(section),
        source_candidates=candidates,
        recent_edit_features=(recent_value,) * len(schema.recent_edit_feature_names),
        style_features=tuple(style),
        missing_mask=(False,) * schema.raw_context_dimension,
    )


def _example(
    song_id: str,
    reference_id: str,
    *,
    recent_value: float = 0.0,
    reverse_preference: bool = False,
) -> ReferenceTrainingExample:
    rows = (
        (
            _candidate("A", shot_scale=0.2, face_scale=0.2, camera_motion=0.1, exposure=0.3),
            _candidate("B", shot_scale=0.8, face_scale=0.8, camera_motion=0.7, exposure=0.7),
            _candidate("C", shot_scale=0.5, face_scale=0.5, camera_motion=0.4, exposure=0.5),
        ),
        (
            _candidate("A", shot_scale=0.2, face_scale=0.2, camera_motion=0.2, exposure=0.3),
            _candidate("B", shot_scale=0.3, face_scale=0.3, camera_motion=0.3, exposure=0.4),
            _candidate("C", shot_scale=0.9, face_scale=0.9, camera_motion=0.8, exposure=0.8),
        ),
        (
            _candidate("A", shot_scale=0.7, face_scale=0.7, camera_motion=0.7, exposure=0.7),
            _candidate("B", shot_scale=0.3, face_scale=0.3, camera_motion=0.2, exposure=0.3),
            _candidate("C", shot_scale=0.4, face_scale=0.4, camera_motion=0.3, exposure=0.4),
        ),
        (
            _candidate("A", shot_scale=0.6, face_scale=0.6, camera_motion=0.6, exposure=0.6),
            _candidate("B", shot_scale=0.5, face_scale=0.5, camera_motion=0.5, exposure=0.5),
            _candidate("C", shot_scale=0.9, face_scale=0.9, camera_motion=0.9, exposure=0.9),
        ),
    )
    selected = ("A", "B", "C", "A")
    if reverse_preference:
        selected = ("A", "C", "B", "A")
    return ReferenceTrainingExample(
        reference_id=reference_id,
        song_id=song_id,
        style_id="human-stage-mix",
        sequence=StageEditSequence(
            reference_id,
            tuple(
                _token(index, row, recent_value=recent_value)
                for index, row in enumerate(rows)
            ),
        ),
        targets=StageEditTargetSequence(
            reference_id,
            (
                StageEditTarget(selected_source_id=selected[0]),
                StageEditTarget(cut=1.0, selected_source_id=selected[1]),
                StageEditTarget(cut=1.0, selected_source_id=selected[2]),
                StageEditTarget(cut=1.0, selected_source_id=selected[3]),
            ),
        ),
        statistics={},
    )


def _corpus(*examples: ReferenceTrainingExample) -> ReferenceTrainingCorpus:
    return ReferenceTrainingCorpus(
        examples=tuple(examples),
        data_id="contextual-diagnostic-data",
        source_paths=(),
        statistics={},
    )


class StageEditSourcePairContextualDiagnosticTests(unittest.TestCase):
    def test_materializes_real_causal_evidence_but_never_recent_history_or_logits(
        self,
    ) -> None:
        corpus = materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
            _corpus(_example("SONG-A", "reference-a"))
        )

        self.assertFalse(corpus.production_activation_permitted)
        self.assertIn("pair.face_scale_log_delta", corpus.feature_names)
        self.assertIn("incoming.shot_scale", corpus.feature_names)
        self.assertIn("outgoing.shot_scale", corpus.feature_names)
        self.assertIn("shot_size.incoming.close", corpus.feature_names)
        self.assertIn(
            "embedding.pose_model_embedding_cosine",
            corpus.feature_names,
        )
        self.assertTrue(
            any(
                name.startswith("interaction.section.chorus_probability*")
                for name in corpus.feature_names
            )
        )
        self.assertTrue(
            any(
                name.startswith("interaction.style.closeup_preference*")
                for name in corpus.feature_names
            )
        )
        self.assertFalse(any("recent" in name for name in corpus.feature_names))
        self.assertFalse(any("logit" in name for name in corpus.feature_names))

    def test_reference_recent_history_cannot_change_contextual_features_or_hash(
        self,
    ) -> None:
        cold = materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
            _corpus(_example("SONG-A", "reference-a", recent_value=0.0))
        )
        contaminated = materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
            _corpus(_example("SONG-A", "reference-a", recent_value=0.95))
        )

        self.assertEqual(cold.content_sha256, contaminated.content_sha256)
        self.assertEqual(
            tuple(
                candidate.features
                for group in cold.groups
                for candidate in group.candidates
            ),
            tuple(
                candidate.features
                for group in contaminated.groups
                for candidate in group.candidates
            ),
        )

    def test_signed_residual_is_l1_bounded_and_song_balanced(self) -> None:
        corpus = materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
            _corpus(
                _example("SONG-A", "reference-a"),
                _example("SONG-B", "reference-b", reverse_preference=True),
            )
        )
        config = StageEditSourcePairContextualDiagnosticConfig(
            steps=12,
            l1_radius=0.30,
        )
        original = train_stage_edit_source_pair_contextual_residual(
            corpus.groups,
            config=config,
        )
        duplicates = tuple(
            replace(
                group,
                pair_group=replace(
                    group.pair_group,
                    group_id=group.group_id + ":copy",
                ),
            )
            for group in corpus.groups
            if group.reference_id == "reference-a"
        )
        duplicated = train_stage_edit_source_pair_contextual_residual(
            (*corpus.groups, *duplicates),
            config=config,
        )

        self.assertLessEqual(
            sum(abs(value) for value in original.model.weights),
            config.l1_radius + 1.0e-8,
        )
        self.assertLessEqual(original.final_loss, original.initial_loss)
        # Duplicating one exact event within its reference does not change
        # song/reference-balanced influence.
        np.testing.assert_allclose(
            original.model.weights,
            duplicated.model.weights,
            atol=1.0e-10,
            rtol=0.0,
        )

    def test_loso_keeps_songs_references_and_groups_disjoint_and_reuses_gate(
        self,
    ) -> None:
        corpus = materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
            _corpus(
                _example("SONG-A", "reference-a"),
                _example("SONG-B", "reference-b", reverse_preference=True),
                _example("SONG-C", "reference-c"),
            )
        )
        result = cross_validate_stage_edit_source_pair_contextual_diagnostic(
            corpus,
            config=StageEditSourcePairContextualDiagnosticConfig(steps=8),
        )

        self.assertFalse(result.production_activation_permitted)
        self.assertEqual(corpus.song_ids, tuple(fold.heldout_song_id for fold in result.folds))
        for fold in result.folds:
            self.assertNotIn(
                fold.heldout_song_id,
                fold.training.training_song_ids,
            )
            self.assertFalse(
                set(fold.heldout_reference_ids).intersection(
                    fold.training.training_reference_ids
                )
            )
            self.assertFalse(
                set(fold.heldout_group_ids).intersection(
                    fold.training.training_group_ids
                )
            )
            self.assertEqual(
                stage_edit_source_pair_no_worse_activation_gate(
                    fold.learned_metrics,
                    fold.uniform_metrics,
                    fold.handcrafted_metrics,
                ),
                fold.activation_gate,
            )
        self.assertEqual(
            stage_edit_source_pair_no_worse_activation_gate(
                result.learned_metrics,
                result.uniform_metrics,
                result.handcrafted_metrics,
            ),
            result.aggregate_gate,
        )
        payload = result.to_dict()
        self.assertFalse(payload["production_activation_permitted"])
        self.assertFalse(payload["causal_evidence"]["recent_edit_features_consumed"])
        self.assertFalse(payload["causal_evidence"]["model_logits_consumed"])
        with self.assertRaisesRegex(
            StageEditSourcePairContextualDiagnosticError,
            "aggregate gate",
        ):
            replace(
                result,
                aggregate_gate=StageEditSourcePairActivationGate(
                    authorized=False,
                    reasons=("tampered",),
                ),
            )

    def test_rejects_malformed_source_and_context_masks(self) -> None:
        example = _example("SONG-A", "reference-a")
        tokens = list(example.sequence.tokens)
        candidates = list(tokens[1].source_candidates)
        candidates[0] = replace(candidates[0], missing_mask=(False,))
        tokens[1] = replace(tokens[1], source_candidates=tuple(candidates))
        malformed_source = replace(
            example,
            sequence=replace(example.sequence, tokens=tuple(tokens)),
        )
        with self.assertRaisesRegex(
            StageEditSourcePairContextualDiagnosticError,
            "missing-mask",
        ):
            materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
                _corpus(malformed_source)
            )

        tokens = list(example.sequence.tokens)
        tokens[1] = replace(tokens[1], missing_mask=(False,))
        malformed_context = replace(
            example,
            sequence=replace(example.sequence, tokens=tuple(tokens)),
        )
        with self.assertRaisesRegex(
            StageEditSourcePairContextualDiagnosticError,
            "token missing-mask dimension",
        ):
            materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
                _corpus(malformed_context)
            )


if __name__ == "__main__":
    unittest.main()
