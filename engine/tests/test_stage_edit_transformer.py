from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from crosscut.stage_edit_dynamic_crop_training_v3 import (
    StageEditDynamicCropTrainingCorpusV3,
    build_stage_edit_dynamic_crop_training_corpus_v3,
)
from crosscut.stage_edit_head_materializer import (
    materialize_stage_edit_head_frames_v3,
)
from crosscut.stage_edit_production_training import (
    ProductionQualityThresholds,
    train_production_stage_edit,
)
from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)

from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    MASKED_SOURCE_LOGIT,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_RUN_KIND,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
    SourceCandidateToken,
    StageEditLossContract,
    StageEditModelOutput,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
    StageEditTokenSchema,
    StageEditTransformer,
    StageEditTransformerConfig,
    load_stage_edit_checkpoint,
    mask_stage_edit_recent_history_for_semimarkov,
    mlx_runtime_version,
    new_stage_edit_checkpoint_metadata,
    production_stage_edit_config,
    save_stage_edit_checkpoint,
    smoke_stage_edit_config,
    stage_edit_loss,
    stage_edit_cut_imbalance_contract,
    stage_edit_cut_scoring_contract,
    stage_edit_parameter_count,
    stage_edit_recent_history_mask_contract,
    stage_edit_readiness_report,
    stage_edit_source_scoring_contract,
    targets_from_reference_supervision,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
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


SCHEMA = DEFAULT_STAGE_EDIT_SCHEMA


def _candidate(
    source_id: str,
    value: float,
    *,
    gate: bool = True,
    missing_first: bool = False,
) -> SourceCandidateToken:
    values = [value] * SCHEMA.raw_source_dimension
    missing = [False] * SCHEMA.raw_source_dimension
    if missing_first:
        values[0] = float("nan")
        missing[0] = True
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(values),
        missing_mask=tuple(missing),
        hard_gate_passed=gate,
    )


def _token(
    timestamp: float,
    *,
    music_value: float = 0.1,
    candidates: tuple[SourceCandidateToken, ...] | None = None,
    missing_first: bool = False,
) -> StageEditToken:
    music = [music_value] * len(SCHEMA.music_feature_names)
    missing = [False] * SCHEMA.raw_context_dimension
    if missing_first:
        music[0] = float("nan")
        missing[0] = True
    return StageEditToken(
        master_time_seconds=timestamp,
        music_features=tuple(music),
        section_features=tuple([0.2] * len(SCHEMA.section_feature_names)),
        source_candidates=candidates
        or (
            _candidate("A", 0.2),
            _candidate("B", 0.4, gate=False),
            _candidate("C", 0.6),
        ),
        recent_edit_features=tuple(
            [0.3] * len(SCHEMA.recent_edit_feature_names)
        ),
        style_features=tuple([0.4] * len(SCHEMA.style_feature_names)),
        missing_mask=tuple(missing),
    )


def _sequence(
    sequence_id: str = "ive:test",
    *,
    future_music_value: float = 0.1,
) -> StageEditSequence:
    return StageEditSequence(
        sequence_id,
        (
            _token(0.0, missing_first=True),
            _token(0.5),
            _token(1.0, music_value=future_music_value),
        ),
    )


def _targets(sequence_id: str = "ive:test") -> StageEditTargetSequence:
    return StageEditTargetSequence(
        sequence_id,
        (
            StageEditTarget(
                cut=1.0,
                selected_source_id="A",
                rejected_source_ids=("C",),
                duration_bucket=2,
                transition_label="hard_cut",
                crop=(1.08, 0.48, 0.45, 0.0),
            ),
            StageEditTarget(
                cut=0.0,
                selected_source_id="C",
                rejected_source_ids=("A", "B"),
                duration_bucket=3,
                crop=(1.0, None, None, None),
            ),
            StageEditTarget(
                cut=1.0,
                selected_source_id="A",
                rejected_source_ids=("C",),
                duration_bucket=1,
                transition_label="zoom_match",
            ),
        ),
    )


def _v3_production_corpora() -> tuple[
    ReferenceTrainingCorpus,
    StageEditDynamicCropTrainingCorpusV3,
    tuple[ReferenceTrainingExample, ...],
]:
    examples: list[ReferenceTrainingExample] = []
    dynamic_sequences = {}
    for song in ("BANG", "BLACKHOLE", "XOXZ"):
        for index in range(2):
            reference_id = f"{song}-v3-readiness-{index}"
            sequence = _sequence(reference_id)
            targets = _targets(reference_id)
            examples.append(
                ReferenceTrainingExample(
                    reference_id=reference_id,
                    song_id=song,
                    style_id="human-stage-mix",
                    sequence=sequence,
                    targets=targets,
                    statistics={
                        "dense_editorial_source_switch_positive_count": 2,
                        "editorial_source_switch_positive_count": 2,
                        "cut_positive_count": 2,
                        "native_source_camera_cut_count": 0,
                        "native_source_camera_cut_editorial_positive_count": 0,
                        "materialized_feature_snapshot_used": True,
                        "shared_feature_materializer_used": True,
                    },
                    feature_snapshot_sha256=hashlib.sha256(
                        reference_id.encode("utf-8")
                    ).hexdigest(),
                )
            )
            dynamic_targets = [None] * len(sequence.tokens)
            for token_index in (0, 2):
                dynamic_targets[token_index] = DynamicCropTrajectoryTargetV2(
                    shot_index=token_index,
                    source_id="A",
                    shot_duration_seconds=1.0,
                    keyframes=(
                        EditorialTransformKeyframe(
                            0.0, 1.04, 0.50, 0.50, 0.0
                        ),
                        EditorialTransformKeyframe(
                            0.5, 1.07, 0.51, 0.50, 0.0
                        ),
                    ),
                    motion_label="dynamic",
                    motion_label_loss_masked=False,
                    artifact_content_sha256="a" * 64,
                    build_fingerprint_sha256="b" * 64,
                )
            dynamic_sequences[reference_id] = (
                StageEditDynamicCropTargetSequenceV2(
                    reference_id,
                    tuple(dynamic_targets),
                )
            )
    example_tuple = tuple(examples)
    corpus = ReferenceTrainingCorpus(
        examples=example_tuple,
        data_id="7" * 64,
        source_paths=(),
        statistics={
            "crop_coordinate_supervision_count": 24,
            "transition_supervision_count": 12,
        },
    )
    dynamic_v2 = StageEditDynamicCropTrainingCorpusV2(
        sequences=dynamic_sequences,
        data_id="dynamic-v2-readiness-test",
        artifact_paths=(),
        statistics={
            "track_label_count": 2 * len(example_tuple),
            "keyframe_label_count": 4 * len(example_tuple),
            "motion_label_count": 2 * len(example_tuple),
        },
    )
    return (
        corpus,
        build_stage_edit_dynamic_crop_training_corpus_v3(dynamic_v2),
        example_tuple,
    )


def _metal_available() -> bool:
    if mx is None:
        return False
    try:
        with mx.stream(mx.gpu):
            value = mx.sum(mx.ones((2,), dtype=mx.float32))
            mx.eval(value)
            return float(value.item()) == 2.0
    except Exception:
        return False


class StageEditSchemaTests(unittest.TestCase):
    def test_renderer_owned_loss_requires_zero_transition_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "transition_weight=0"):
            StageEditLossContract(
                transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
            )
        contract = StageEditLossContract(
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
            transition_weight=0.0,
        )
        self.assertEqual(0.0, contract.transition_weight)

    def test_production_history_preprocessing_removes_human_slice_and_masks_it(self) -> None:
        original = _sequence()
        masked = mask_stage_edit_recent_history_for_semimarkov(original, SCHEMA)
        start = len(SCHEMA.music_feature_names) + len(
            SCHEMA.section_feature_names
        )
        stop = start + len(SCHEMA.recent_edit_feature_names)
        self.assertTrue(
            any(any(value != 0.0 for value in token.recent_edit_features) for token in original.tokens)
        )
        for before, after in zip(original.tokens, masked.tokens, strict=True):
            self.assertEqual(
                (0.0,) * len(SCHEMA.recent_edit_feature_names),
                after.recent_edit_features,
            )
            self.assertEqual(
                (True,) * len(SCHEMA.recent_edit_feature_names),
                after.missing_mask[start:stop],
            )
            self.assertEqual(before.music_features, after.music_features)
            self.assertEqual(before.section_features, after.section_features)
            self.assertEqual(before.style_features, after.style_features)
            self.assertEqual(before.source_candidates, after.source_candidates)
        contract = stage_edit_recent_history_mask_contract(SCHEMA)
        self.assertEqual(start, contract["raw_context_slice"]["start_inclusive"])
        self.assertEqual(stop, contract["raw_context_slice"]["stop_exclusive"])
        self.assertEqual(64, len(contract["contract_sha256"]))

    def test_schema_round_trip_and_production_parameter_budget(self) -> None:
        restored = StageEditTokenSchema.from_dict(SCHEMA.to_dict())
        self.assertEqual(restored, SCHEMA)
        self.assertEqual(restored.fingerprint, SCHEMA.fingerprint)
        legacy_schema = SCHEMA.to_dict()
        legacy_schema["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported StageEdit token schema"):
            StageEditTokenSchema.from_dict(legacy_schema)
        parameter_count = stage_edit_parameter_count(
            production_stage_edit_config(SCHEMA)
        )
        self.assertGreater(parameter_count, 95_000_000)
        self.assertLess(parameter_count, 110_000_000)
        config = production_stage_edit_config(SCHEMA)
        self.assertEqual(3, config.config_version)
        source_contract = stage_edit_source_scoring_contract(config)
        self.assertFalse(source_contract["absolute_source_index_parameters"])
        self.assertEqual(
            4 * config.source_input_dimension,
            source_contract["relational_source_input_dimension"],
        )
        legacy = config.to_dict()
        for legacy_version in (1, 2):
            legacy["config_version"] = legacy_version
            with self.assertRaisesRegex(ValueError, "unsupported model config"):
                StageEditTransformerConfig.from_dict(legacy)

    def test_cut_imbalance_contract_is_training_only_and_bounded(self) -> None:
        contract = stage_edit_cut_imbalance_contract(
            positive_count=10,
            negative_count=200,
            maximum_positive_weight=8.0,
        )
        self.assertEqual(20.0, contract["raw_negative_to_positive_ratio"])
        self.assertEqual(8.0, contract["applied_positive_weight"])
        self.assertFalse(contract["reference_validation_labels_consumed"])
        self.assertFalse(contract["heldout_labels_consumed"])
        self.assertEqual(64, len(contract["contract_sha256"]))

    def test_readiness_never_calls_architecture_learned_without_weights(self) -> None:
        report = stage_edit_readiness_report()
        self.assertTrue(report.architecture_available)
        self.assertFalse(report.learned_inference_ready)
        self.assertEqual(report.training_steps, 0)
        self.assertIn("no_checkpoint", report.status)


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditMLXTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = smoke_stage_edit_config(SCHEMA)

    def test_tensorization_preserves_missing_variable_candidates_and_masks(self) -> None:
        short = StageEditSequence(
            "short",
            (
                _token(
                    0.0,
                    candidates=(
                        _candidate("A", 0.2, missing_first=True),
                    ),
                ),
            ),
        )
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                (_sequence(), short), SCHEMA, self.config, device=mx.cpu
            )
            mx.eval(
                batch.context_features,
                batch.source_features,
                batch.sequence_mask,
                batch.source_candidate_mask,
                batch.hard_source_gate_mask,
            )
        self.assertEqual(batch.context_features.shape, (2, 3, 76))
        self.assertEqual(
            batch.source_features.shape,
            (2, 3, 8, DEFAULT_STAGE_EDIT_SCHEMA.encoded_source_dimension),
        )
        context = np.asarray(batch.context_features)
        self.assertEqual(float(context[0, 0, 0]), 0.0)
        self.assertEqual(float(context[0, 0, SCHEMA.raw_context_dimension]), 1.0)
        candidate_mask = np.asarray(batch.source_candidate_mask)
        gate_mask = np.asarray(batch.hard_source_gate_mask)
        self.assertEqual(candidate_mask[0, 0].tolist()[:4], [True, True, True, False])
        self.assertEqual(gate_mask[0, 0].tolist()[:4], [True, False, True, False])
        self.assertEqual(candidate_mask[1, 0].tolist()[:3], [True, False, False])
        self.assertEqual(np.asarray(batch.sequence_mask)[1].tolist(), [True, False, False])

    def test_reference_supervision_adapter_keeps_only_observed_labels(self) -> None:
        sequence = _sequence()
        payload = {
            "dataset_kind": "reference_stage_mix_pairwise_listwise_supervision",
            "listwise_records": [
                {
                    "shot_index": 0,
                    "chosen_source_id": "C",
                    "candidates": [
                        {"source_id": "C", "selected": True},
                        {"source_id": "A", "selected": False},
                    ],
                },
                {
                    "shot_index": 1,
                    "chosen_source_id": "A",
                    "candidates": [
                        {"source_id": "A", "selected": True},
                        {"source_id": "C", "selected": False},
                    ],
                }
            ],
            "shot_labels": [
                {
                    "shot_index": 0,
                    "edit_labels": {
                        "cut": {
                            "boundary_label": "same_source_internal_camera_cut",
                            "duration_seconds": 1.0,
                        },
                        "boundary_event": {
                            "schema_version": 1,
                            "event_type": "native_source_camera_cut",
                            "is_native_source_camera_cut": True,
                            "is_editorial_source_switch": False,
                        },
                        "source_transition": {
                            "supervision_eligible": False,
                            "transition": None,
                        },
                        "crop": {"supervision_eligible": False},
                    },
                },
                {
                    "shot_index": 1,
                    "edit_labels": {
                        "cut": {
                            "boundary_label": "cross_source_hard_cut",
                            "duration_seconds": 1.2,
                        },
                        "source_transition": {
                            "supervision_eligible": True,
                            "transition": "hard_cut",
                        },
                        "crop": {
                            "supervision_eligible": True,
                            "transform": {
                                "scale": 1.08,
                                "center_x": 0.48,
                                "center_y": 0.44,
                            },
                        },
                    },
                },
                {
                    "shot_index": 2,
                    "edit_labels": {
                        "cut": {"boundary_label": "reference_start"},
                        "source_transition": {
                            "supervision_eligible": False,
                            "transition": None,
                        },
                        "crop": {"supervision_eligible": False},
                    },
                },
            ],
        }
        targets = targets_from_reference_supervision(payload, sequence, SCHEMA)
        native = targets.targets[0]
        self.assertEqual(native.cut, 0.0)
        self.assertIsNone(native.selected_source_id)
        self.assertEqual(native.rejected_source_ids, ())
        self.assertIsNone(native.duration_bucket)
        self.assertIsNone(native.transition_label)
        learned = targets.targets[1]
        self.assertEqual(learned.cut, 1.0)
        self.assertEqual(learned.selected_source_id, "A")
        self.assertEqual(learned.rejected_source_ids, ("C",))
        self.assertEqual(learned.duration_bucket, SCHEMA.duration_bucket(1.2))
        self.assertEqual(learned.transition_label, "hard_cut")
        self.assertEqual(learned.crop, (1.08, 0.48, 0.44, None))
        unresolved = targets.targets[2]
        self.assertIsNone(unresolved.cut)
        self.assertIsNone(unresolved.transition_label)
        self.assertIsNone(unresolved.crop)

    def test_all_heads_shape_hard_gate_and_exact_parameter_count(self) -> None:
        with mx.stream(mx.cpu):
            mx.random.seed(7)
            model = StageEditTransformer(self.config)
            model.eval()
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            output = model(batch)
            mx.eval(model.parameters(), output.source_selection_logits)
        self.assertEqual(output.cut_timing_logits.shape, (1, 3))
        self.assertEqual(output.source_selection_logits.shape, (1, 3, 8))
        self.assertEqual(output.duration_bucket_logits.shape, (1, 3, 7))
        self.assertEqual(output.transition_logits.shape, (1, 3, 4))
        self.assertEqual(output.crop.shape, (1, 3, 4))
        logits = np.asarray(output.source_selection_logits)
        self.assertEqual(float(logits[0, 0, 1]), MASKED_SOURCE_LOGIT)
        self.assertEqual(float(logits[0, 0, 7]), MASKED_SOURCE_LOGIT)
        actual = sum(value.size for _, value in tree_flatten(model.parameters()))
        self.assertEqual(actual, stage_edit_parameter_count(self.config))

    def test_causal_outputs_do_not_change_when_future_features_change(self) -> None:
        with mx.stream(mx.cpu):
            mx.random.seed(11)
            model = StageEditTransformer(self.config)
            model.eval()
            original = tensorize_stage_edit_batch(
                (_sequence(future_music_value=0.1),),
                SCHEMA,
                self.config,
                device=mx.cpu,
            )
            changed = tensorize_stage_edit_batch(
                (_sequence(future_music_value=0.95),),
                SCHEMA,
                self.config,
                device=mx.cpu,
            )
            first = model(original)
            second = model(changed)
            mx.eval(first.cut_timing_logits, second.cut_timing_logits)
        np.testing.assert_allclose(
            np.asarray(first.cut_timing_logits)[:, :2],
            np.asarray(second.cut_timing_logits)[:, :2],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(first.source_selection_logits)[:, :2, :],
            np.asarray(second.source_selection_logits)[:, :2, :],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_hard_gated_source_is_invisible_to_every_head(self) -> None:
        def sequence(gated_value: float) -> StageEditSequence:
            candidates = (
                _candidate("A", 0.2),
                _candidate("B", gated_value, gate=False),
                _candidate("C", 0.6),
            )
            return StageEditSequence(
                "ive:hard-gate-invariance",
                (
                    _token(0.0, candidates=candidates),
                    _token(0.5, candidates=candidates),
                ),
            )

        with mx.stream(mx.cpu):
            mx.random.seed(13)
            model = StageEditTransformer(self.config)
            model.eval()
            baseline = model(
                tensorize_stage_edit_batch(
                    (sequence(0.4),), SCHEMA, self.config, device=mx.cpu
                )
            )
            changed = model(
                tensorize_stage_edit_batch(
                    (sequence(50.0),), SCHEMA, self.config, device=mx.cpu
                )
            )
            mx.eval(
                baseline.cut_timing_logits,
                changed.cut_timing_logits,
                baseline.source_selection_logits,
                changed.source_selection_logits,
                baseline.duration_bucket_logits,
                changed.duration_bucket_logits,
                baseline.transition_logits,
                changed.transition_logits,
                baseline.crop,
                changed.crop,
            )
        for first, second in (
            (baseline.cut_timing_logits, changed.cut_timing_logits),
            (baseline.duration_bucket_logits, changed.duration_bucket_logits),
            (baseline.transition_logits, changed.transition_logits),
            (baseline.crop, changed.crop),
        ):
            np.testing.assert_allclose(
                np.asarray(first), np.asarray(second), rtol=0.0, atol=0.0
            )
        np.testing.assert_allclose(
            np.asarray(baseline.source_selection_logits)[..., (0, 2)],
            np.asarray(changed.source_selection_logits)[..., (0, 2)],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            np.all(
                np.asarray(baseline.source_selection_logits)[..., 1]
                == MASKED_SOURCE_LOGIT
            )
        )

    def test_relational_source_encoder_is_candidate_permutation_equivariant(
        self,
    ) -> None:
        original_candidates = (
            _candidate("A", 0.2),
            _candidate("B", 0.4, gate=False),
            _candidate("C", 0.7),
        )
        permuted_candidates = (
            original_candidates[2],
            original_candidates[0],
            original_candidates[1],
        )

        def sequence(
            sequence_id: str,
            candidates: tuple[SourceCandidateToken, ...],
        ) -> StageEditSequence:
            return StageEditSequence(
                sequence_id,
                tuple(
                    _token(index * 0.5, candidates=candidates)
                    for index in range(3)
                ),
            )

        with mx.stream(mx.cpu):
            mx.random.seed(17)
            model = StageEditTransformer(self.config)
            model.eval()
            first_batch = tensorize_stage_edit_batch(
                (sequence("order-a", original_candidates),),
                SCHEMA,
                self.config,
                device=mx.cpu,
            )
            second_batch = tensorize_stage_edit_batch(
                (sequence("order-b", permuted_candidates),),
                SCHEMA,
                self.config,
                device=mx.cpu,
            )
            first = model(first_batch)
            second = model(second_batch)
            mx.eval(
                first.cut_timing_logits,
                second.cut_timing_logits,
                first.source_selection_logits,
                second.source_selection_logits,
                first.duration_bucket_logits,
                second.duration_bucket_logits,
                first.transition_logits,
                second.transition_logits,
                first.crop,
                second.crop,
            )
        for before, after in (
            (first.cut_timing_logits, second.cut_timing_logits),
            (first.duration_bucket_logits, second.duration_bucket_logits),
            (first.transition_logits, second.transition_logits),
            (first.crop, second.crop),
        ):
            np.testing.assert_allclose(
                np.asarray(before), np.asarray(after), rtol=1e-5, atol=1e-5
            )
        first_logits = np.asarray(first.source_selection_logits)[0]
        second_logits = np.asarray(second.source_selection_logits)[0]
        first_ids = first_batch.source_ids[0]
        second_ids = second_batch.source_ids[0]
        for time_index in range(3):
            by_source_first = {
                source_id: first_logits[time_index, slot]
                for slot, source_id in enumerate(first_ids[time_index])
                if source_id is not None
            }
            by_source_second = {
                source_id: second_logits[time_index, slot]
                for slot, source_id in enumerate(second_ids[time_index])
                if source_id is not None
            }
            self.assertEqual(set(by_source_first), set(by_source_second))
            for source_id in by_source_first:
                self.assertAlmostEqual(
                    float(by_source_first[source_id]),
                    float(by_source_second[source_id]),
                    places=5,
                )

    @unittest.skipUnless(_metal_available(), "Metal MLX device unavailable")
    def test_cpu_and_metal_shapes_are_deterministic(self) -> None:
        observed_shapes: list[tuple[tuple[int, ...], ...]] = []
        for device in (mx.cpu, mx.gpu):
            with mx.stream(device):
                mx.random.seed(19)
                model = StageEditTransformer(self.config)
                model.eval()
                batch = tensorize_stage_edit_batch(
                    (_sequence(),), SCHEMA, self.config, device=device
                )
                first = model(batch)
                second = model(batch)
                mx.eval(
                    first.cut_timing_logits,
                    second.cut_timing_logits,
                    first.source_selection_logits,
                    second.source_selection_logits,
                )
                self.assertTrue(
                    bool(
                        mx.allclose(
                            first.cut_timing_logits,
                            second.cut_timing_logits,
                            rtol=0.0,
                            atol=0.0,
                        ).item()
                    )
                )
                observed_shapes.append(
                    (
                        first.cut_timing_logits.shape,
                        first.source_selection_logits.shape,
                        first.duration_bucket_logits.shape,
                        first.transition_logits.shape,
                        first.crop.shape,
                    )
                )
        self.assertEqual(observed_shapes[0], observed_shapes[1])

    @unittest.skipUnless(_metal_available(), "Metal MLX device unavailable")
    def test_metal_forward_backward_optimizer_step_is_finite(self) -> None:
        with mx.stream(mx.gpu):
            mx.random.seed(23)
            model = StageEditTransformer(self.config)
            model.train()
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.gpu
            )
            supervision = tensorize_stage_edit_supervision(
                (_targets(),), batch, SCHEMA, device=mx.gpu
            )
            loss_and_grad = nn.value_and_grad(
                model,
                lambda trained_model: stage_edit_loss(
                    trained_model,
                    batch,
                    supervision,
                    StageEditLossContract(source_objective="hybrid"),
                )[0],
            )
            loss, gradients = loss_and_grad(model)
            optimizer = optim.AdamW(learning_rate=1e-3)
            first_parameter = tree_flatten(model.parameters())[0][1]
            before = mx.stop_gradient(first_parameter + 0.0)
            mx.eval(before)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            after = tree_flatten(model.parameters())[0][1]
            update_size = mx.sum(mx.abs(after - before))
            mx.eval(update_size)
        self.assertTrue(bool(mx.isfinite(loss).item()))
        self.assertGreater(float(update_size.item()), 0.0)

    def test_cut_positive_weight_changes_only_the_bounded_cut_objective(self) -> None:
        with mx.stream(mx.cpu):
            mx.random.seed(24)
            model = StageEditTransformer(self.config)
            model.eval()
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            supervision = tensorize_stage_edit_supervision(
                (_targets(),), batch, SCHEMA, device=mx.cpu
            )
            _, unweighted = stage_edit_loss(
                model,
                batch,
                supervision,
                StageEditLossContract(cut_positive_weight=1.0),
            )
            _, weighted = stage_edit_loss(
                model,
                batch,
                supervision,
                StageEditLossContract(cut_positive_weight=3.0),
            )
            mx.eval(*unweighted.values(), *weighted.values())
        self.assertGreater(
            float(weighted["cut"].item()), float(unweighted["cut"].item())
        )
        for name in (
            "source_listwise",
            "source_pairwise",
            "duration",
            "transition",
            "crop",
        ):
            self.assertAlmostEqual(
                float(weighted[name].item()),
                float(unweighted[name].item()),
                places=6,
            )

    def test_hybrid_source_losses_are_reference_balanced(self) -> None:
        def example(
            sequence_id: str,
            *,
            token_count: int,
            include_third_source: bool,
        ) -> tuple[StageEditSequence, StageEditTargetSequence]:
            candidates = (
                _candidate("A", 0.2),
                _candidate("B", 0.4),
            ) + ((_candidate("C", 0.6),) if include_third_source else ())
            rejected = tuple(candidate.source_id for candidate in candidates[1:])
            return (
                StageEditSequence(
                    sequence_id,
                    tuple(
                        _token(index * 0.5, candidates=candidates)
                        for index in range(token_count)
                    ),
                ),
                StageEditTargetSequence(
                    sequence_id,
                    tuple(
                        StageEditTarget(
                            selected_source_id="A",
                            rejected_source_ids=rejected,
                        )
                        for _ in range(token_count)
                    ),
                ),
            )

        class FixedSourceLogitModel:
            def __call__(self, batch):
                batch_size, length, max_sources = batch.source_candidate_mask.shape
                logits = np.full(
                    (batch_size, length, max_sources),
                    MASKED_SOURCE_LOGIT,
                    dtype=np.float32,
                )
                for batch_index, sequence_id in enumerate(batch.sequence_ids):
                    scores = (
                        {"A": 2.0, "B": 0.0, "C": -1.0}
                        if sequence_id == "dominant"
                        else {"A": 0.0, "B": 2.0, "C": -2.0}
                    )
                    for time_index, source_ids in enumerate(
                        batch.source_ids[batch_index]
                    ):
                        for source_index, source_id in enumerate(source_ids):
                            if source_id is not None:
                                logits[batch_index, time_index, source_index] = scores[
                                    source_id
                                ]
                effective_mask = mx.logical_and(
                    batch.source_candidate_mask, batch.hard_source_gate_mask
                )
                return StageEditModelOutput(
                    cut_timing_logits=mx.zeros((batch_size, length)),
                    source_selection_logits=mx.array(logits),
                    duration_bucket_logits=mx.zeros(
                        (batch_size, length, self_config.duration_bucket_count)
                    ),
                    transition_logits=mx.zeros(
                        (batch_size, length, self_config.transition_count)
                    ),
                    crop=mx.zeros((batch_size, length, 4)),
                    effective_source_mask=effective_mask,
                )

        self_config = self.config
        dominant_once = example(
            "dominant", token_count=1, include_third_source=True
        )
        dominant_repeated = example(
            "dominant", token_count=7, include_third_source=True
        )
        other = example("other", token_count=1, include_third_source=False)
        contract = StageEditLossContract(source_objective="hybrid")

        def source_components(examples):
            sequences, targets = zip(*examples, strict=True)
            batch = tensorize_stage_edit_batch(
                sequences, SCHEMA, self.config, device=mx.cpu
            )
            supervision = tensorize_stage_edit_supervision(
                targets, batch, SCHEMA, device=mx.cpu
            )
            _, components = stage_edit_loss(
                FixedSourceLogitModel(), batch, supervision, contract
            )
            mx.eval(components["source_listwise"], components["source_pairwise"])
            return batch, supervision, components

        with mx.stream(mx.cpu):
            _, _, once = source_components((dominant_once, other))
            repeated_batch, repeated_supervision, repeated = source_components(
                (dominant_repeated, other)
            )
            _, eligible = stage_edit_loss(
                FixedSourceLogitModel(),
                repeated_batch,
                repeated_supervision,
                StageEditLossContract(source_objective="eligible_all_pairwise"),
            )
            mx.eval(eligible["source_pairwise"])

        for component in ("source_listwise", "source_pairwise"):
            self.assertAlmostEqual(
                float(once[component].item()),
                float(repeated[component].item()),
                places=6,
            )
        # Every effective negative was explicitly rejected.  Explicit pairwise
        # and eligible-all therefore have the same group-then-reference mean,
        # even though references and groups have unequal negative counts.
        self.assertAlmostEqual(
            float(repeated["source_pairwise"].item()),
            float(eligible["source_pairwise"].item()),
            places=6,
        )

    def test_source_target_slots_follow_candidate_ids_after_shuffle(self) -> None:
        candidates = (
            _candidate("A", 0.2),
            _candidate("B", 0.4),
            _candidate("C", 0.6),
        )
        sequences = (
            StageEditSequence(
                "ordered", (_token(0.0, candidates=candidates),)
            ),
            StageEditSequence(
                "shuffled",
                (
                    _token(
                        0.0,
                        candidates=(candidates[2], candidates[0], candidates[1]),
                    ),
                ),
            ),
        )
        targets = tuple(
            StageEditTargetSequence(
                sequence.sequence_id,
                (
                    StageEditTarget(
                        selected_source_id="C",
                        rejected_source_ids=("A", "B"),
                    ),
                ),
            )
            for sequence in sequences
        )
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                sequences, SCHEMA, self.config, device=mx.cpu
            )
            supervision = tensorize_stage_edit_supervision(
                targets, batch, SCHEMA, device=mx.cpu
            )
            mx.eval(
                supervision.source_targets,
                supervision.rejected_source_targets,
                supervision.pairwise_source_mask,
            )
        self.assertEqual(("A", "B", "C"), batch.source_ids[0][0][:3])
        self.assertEqual(("C", "A", "B"), batch.source_ids[1][0][:3])
        np.testing.assert_array_equal(
            np.asarray(supervision.source_targets)[:, 0], np.asarray([2, 0])
        )
        np.testing.assert_array_equal(
            np.asarray(supervision.rejected_source_targets)[:, 0, :2],
            np.asarray([[0, 1], [1, 2]]),
        )
        self.assertTrue(
            np.asarray(supervision.pairwise_source_mask)[:, 0, :2].all()
        )

    def test_tiny_source_rule_receives_gradient_and_learns_shuffled_targets(
        self,
    ) -> None:
        candidates_by_token = (
            (
                _candidate("B", 0.1),
                _candidate("A", 0.9),
                _candidate("C", 0.4),
            ),
            (
                _candidate("C", 0.8),
                _candidate("A", 0.2),
                _candidate("B", 0.5),
            ),
            (
                _candidate("A", 0.3),
                _candidate("B", 0.7),
                _candidate("C", 0.1),
            ),
        )
        selected_ids = ("A", "C", "B")
        sequence = StageEditSequence(
            "tiny-rule",
            tuple(
                _token(index * 0.5, candidates=candidates)
                for index, candidates in enumerate(candidates_by_token)
            ),
        )
        targets = StageEditTargetSequence(
            sequence.sequence_id,
            tuple(
                StageEditTarget(selected_source_id=selected_source_id)
                for selected_source_id in selected_ids
            ),
        )
        self_config = self.config

        class TinySourceRule(nn.Module):
            def __init__(self):
                super().__init__()
                self.scorer = nn.Linear(1, 1)
                self.scorer.weight = -mx.ones_like(self.scorer.weight)
                self.scorer.bias = mx.zeros_like(self.scorer.bias)

            def __call__(self, batch):
                batch_size, length, _ = batch.source_candidate_mask.shape
                effective_mask = mx.logical_and(
                    batch.source_candidate_mask, batch.hard_source_gate_mask
                )
                source_logits = self.scorer(
                    batch.source_features[..., :1]
                ).squeeze(-1)
                source_logits = mx.where(
                    effective_mask,
                    source_logits,
                    mx.array(MASKED_SOURCE_LOGIT, dtype=source_logits.dtype),
                )
                return StageEditModelOutput(
                    cut_timing_logits=mx.zeros((batch_size, length)),
                    source_selection_logits=source_logits,
                    duration_bucket_logits=mx.zeros(
                        (batch_size, length, self_config.duration_bucket_count)
                    ),
                    transition_logits=mx.zeros(
                        (batch_size, length, self_config.transition_count)
                    ),
                    crop=mx.zeros((batch_size, length, 4)),
                    effective_source_mask=effective_mask,
                )

        contract = StageEditLossContract(
            source_objective="listwise",
            cut_weight=0.0,
            source_pairwise_weight=0.0,
            duration_weight=0.0,
            transition_weight=0.0,
            crop_weight=0.0,
        )
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                (sequence,), SCHEMA, self.config, device=mx.cpu
            )
            supervision = tensorize_stage_edit_supervision(
                (targets,), batch, SCHEMA, device=mx.cpu
            )
            model = TinySourceRule()
            loss_and_grad = nn.value_and_grad(
                model,
                lambda trained_model: stage_edit_loss(
                    trained_model, batch, supervision, contract
                )[0],
            )
            initial_loss, gradients = loss_and_grad(model)
            first_gradient = tree_flatten(gradients)[0][1]
            gradient_size = mx.sum(mx.abs(first_gradient))
            mx.eval(initial_loss, gradient_size)
            optimizer = optim.AdamW(learning_rate=0.1, weight_decay=0.0)
            for _ in range(40):
                loss, gradients = loss_and_grad(model)
                optimizer.update(model, gradients)
                mx.eval(loss, model.parameters(), optimizer.state)
            output = model(batch)
            final_loss = stage_edit_loss(model, batch, supervision, contract)[0]
            mx.eval(output.source_selection_logits, final_loss)

        self.assertGreater(float(gradient_size.item()), 0.0)
        self.assertLess(float(final_loss.item()), float(initial_loss.item()))
        predicted = np.argmax(
            np.asarray(output.source_selection_logits)[0, : len(selected_ids)],
            axis=-1,
        )
        expected = np.asarray(supervision.source_targets)[0, : len(selected_ids)]
        np.testing.assert_array_equal(predicted, expected)

    def test_eligible_all_pairwise_uses_every_hard_gated_negative(self) -> None:
        targets = StageEditTargetSequence(
            "ive:test",
            (
                StageEditTarget(selected_source_id="A"),
                StageEditTarget(),
                StageEditTarget(),
            ),
        )
        with mx.stream(mx.cpu):
            mx.random.seed(26)
            model = StageEditTransformer(self.config)
            model.eval()
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            supervision = tensorize_stage_edit_supervision(
                (targets,), batch, SCHEMA, device=mx.cpu
            )
            _, explicit = stage_edit_loss(
                model,
                batch,
                supervision,
                StageEditLossContract(source_objective="pairwise"),
            )
            _, eligible_all = stage_edit_loss(
                model,
                batch,
                supervision,
                StageEditLossContract(
                    source_objective="eligible_all_pairwise"
                ),
            )
            mx.eval(explicit["source_pairwise"], eligible_all["source_pairwise"])
        self.assertEqual(float(explicit["source_pairwise"].item()), 0.0)
        self.assertGreater(float(eligible_all["source_pairwise"].item()), 0.0)

    def test_schema_v3_cut_transformer_residual_is_exactly_zero_initialized(
        self,
    ) -> None:
        with mx.stream(mx.cpu):
            mx.random.seed(25)
            model = StageEditTransformer(self.config)
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            (
                hidden,
                _relational,
                _source_embeddings,
                effective_mask,
            ) = model._hidden_and_sources(batch)
            summary = model._cut_candidate_summary(batch, effective_mask)
            baseline = model.cut_linear_baseline(
                mx.concatenate((batch.context_features, summary), axis=-1)
            ).squeeze(-1)
            output = model(batch)
            mx.eval(
                model.cut_timing_head.weight,
                model.cut_timing_head.bias,
                baseline,
                output.cut_timing_logits,
            )
        self.assertEqual(
            0.0, float(mx.max(mx.abs(model.cut_timing_head.weight)).item())
        )
        self.assertEqual(
            0.0, float(mx.max(mx.abs(model.cut_timing_head.bias)).item())
        )
        np.testing.assert_allclose(
            np.asarray(output.cut_timing_logits),
            np.asarray(baseline),
            rtol=0.0,
            atol=0.0,
        )

    def test_checkpoint_round_trip_and_conservative_readiness(self) -> None:
        with mx.stream(mx.cpu):
            mx.random.seed(29)
            model = StageEditTransformer(self.config)
            model.eval()
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            expected = model(batch).cut_timing_logits
            mx.eval(expected)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                untrained = new_stage_edit_checkpoint_metadata(
                    "smoke-untrained",
                    SCHEMA,
                    self.config,
                    training_steps=0,
                    weights_trained=False,
                )
                save_stage_edit_checkpoint(root, model, untrained)
                report = stage_edit_readiness_report(root)
                self.assertFalse(report.learned_inference_ready)
                self.assertEqual(report.status, "untrained_checkpoint_only")
                with self.assertRaisesRegex(ValueError, "trained weights"):
                    load_stage_edit_checkpoint(root, expected_schema=SCHEMA)

                trained = replace(
                    untrained,
                    checkpoint_id="smoke-trained",
                    training_steps=1,
                    weights_trained=True,
                )
                saved = save_stage_edit_checkpoint(root, model, trained)
                self.assertIsNotNone(saved.weight_sha256)
                report = stage_edit_readiness_report(root)
                self.assertFalse(report.learned_inference_ready)
                self.assertEqual(
                    report.status, "production_training_report_missing"
                )
                loaded, loaded_metadata = load_stage_edit_checkpoint(
                    root, expected_schema=SCHEMA
                )
                loaded.eval()
                actual = loaded(batch).cut_timing_logits
                mx.eval(actual)
                self.assertEqual(loaded_metadata.training_steps, 1)
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0
                )

                sparse_report = {
                    "schema_version": 2,
                    "run_kind": "stage_edit_sparse_offline_calibration",
                    "checkpoint_scope": "offline_sparse_calibration_only",
                    "production_ready": False,
                }
                (root / PRODUCTION_TRAINING_REPORT_FILENAME).write_text(
                    json.dumps(sparse_report), encoding="utf-8"
                )
                report = stage_edit_readiness_report(root)
                self.assertFalse(report.learned_inference_ready)
                self.assertEqual(
                    report.status, "production_training_report_rejected"
                )
                self.assertIn("run_kind", report.reason)

                production_metadata = replace(
                    trained,
                    checkpoint_id="smoke-production-evidence",
                    training_steps=100,
                )
                saved = save_stage_edit_checkpoint(
                    root, model, production_metadata
                )
                threshold_freeze = {
                    "contract": STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
                    "selected_probability_threshold": 0.42,
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
                production_report = {
                    "schema_version": PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
                    "run_kind": PRODUCTION_TRAINING_RUN_KIND,
                    "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
                    "production_ready": True,
                    "decision_history_input_contract": (
                        stage_edit_recent_history_mask_contract(SCHEMA)
                    ),
                    "source_scoring_contract": (
                        stage_edit_source_scoring_contract(self.config)
                    ),
                    "cut_scoring_contract": (
                        stage_edit_cut_scoring_contract(self.config)
                    ),
                    "transition_authority_contract": (
                        StageEditTransitionAuthority(
                            mode=self.config.transition_authority
                        ).to_dict()
                    ),
                    "transition_authority_contract_sha256": (
                        StageEditTransitionAuthority(
                            mode=self.config.transition_authority
                        ).fingerprint
                    ),
                    "cut_imbalance_contract": (
                        stage_edit_cut_imbalance_contract(
                            positive_count=20,
                            negative_count=80,
                            maximum_positive_weight=8.0,
                        )
                    ),
                    "structured_head_coverage_gate": {
                        "contract": (
                            "duration_learned_transition_or_renderer_metadata_coverage_gate_v2"
                        ),
                        "passed": True,
                        "heldout_used_for_model_selection": False,
                    },
                    "cut_threshold_freeze": threshold_freeze,
                    "production_supervision_gate": {
                        "passed": True,
                        "scope": (
                            "training_songs_only_including_reference_validation"
                        ),
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
                        "architecture": (
                            "causal_long_context_multi_head_transformer_legacy_crop"
                        ),
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
                        "validation_reference_ids": [
                            "a-validation",
                            "b-validation",
                        ],
                        "held_out_reference_ids": [
                            "c-heldout-1",
                            "c-heldout-2",
                        ],
                        "held_out_used_for_model_selection": False,
                    },
                    "optimization": {
                        "effective_loss_contract": asdict(
                            StageEditLossContract()
                        ),
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
                (root / PRODUCTION_TRAINING_REPORT_FILENAME).write_text(
                    json.dumps(production_report), encoding="utf-8"
                )
                report = stage_edit_readiness_report(root)
                self.assertTrue(report.learned_inference_ready)
                self.assertEqual(report.status, "production_checkpoint_ready")

                missing_history_contract = dict(production_report)
                missing_history_contract.pop("decision_history_input_contract")
                (root / PRODUCTION_TRAINING_REPORT_FILENAME).write_text(
                    json.dumps(missing_history_contract), encoding="utf-8"
                )
                rejected_history = stage_edit_readiness_report(root)
                self.assertFalse(rejected_history.learned_inference_ready)
                self.assertIn("decision-history input contract", rejected_history.reason)

                leaked_report = dict(production_report)
                leaked_report["optimization"] = {
                    **production_report["optimization"],
                    "unseen_song_used_for_model_selection": True,
                }
                (root / PRODUCTION_TRAINING_REPORT_FILENAME).write_text(
                    json.dumps(leaked_report), encoding="utf-8"
                )
                rejected = stage_edit_readiness_report(root)
                self.assertFalse(rejected.learned_inference_ready)
                self.assertEqual(
                    rejected.status, "production_training_report_rejected"
                )
                self.assertIn(
                    "unseen song was not excluded", rejected.reason
                )

    def test_v3_readiness_report_and_materializer_are_version_exact(self) -> None:
        corpus, dynamic_corpus, examples = _v3_production_corpora()
        thresholds = ProductionQualityThresholds(
            minimum_cut_labels=2,
            minimum_cut_positives=1,
            minimum_cut_negatives=1,
            minimum_cut_auroc=0.0,
            minimum_cut_balanced_accuracy=0.0,
            minimum_source_groups=1,
            minimum_source_top1_accuracy=0.0,
            minimum_source_mrr=0.0,
            minimum_duration_labels=1,
            minimum_duration_accuracy=0.0,
            minimum_transition_labels=1,
            minimum_transition_accuracy=0.0,
            minimum_crop_labels_per_coordinate=1,
            maximum_zoom_mae=100.0,
            maximum_center_mae=100.0,
            maximum_roll_mae_degrees=100.0,
            minimum_dynamic_crop_tracks=1,
            minimum_dynamic_crop_keyframes=1,
            minimum_dynamic_crop_motion_labels=1,
            maximum_dynamic_crop_zoom_mae=100.0,
            maximum_dynamic_crop_center_mae=100.0,
            maximum_dynamic_crop_roll_mae_degrees=100.0,
            maximum_dynamic_crop_normalized_time_mae=100.0,
            minimum_dynamic_crop_presence_f1=0.0,
            minimum_dynamic_crop_keyframe_count_accuracy=0.0,
            minimum_dynamic_crop_motion_accuracy=0.0,
            minimum_dynamic_crop_renderer_decode_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3-production"
            trained = train_production_stage_edit(
                corpus,
                checkpoint_directory=checkpoint,
                held_out_song_ids=("XOXZ",),
                model_config=self.config,
                epochs=8,
                early_stopping_patience=16,
                batch_size=2,
                learning_rate=3.0e-3,
                cut_pretrain_steps=0,
                source_pretrain_steps=0,
                finetune_warmup_steps=0,
                minimum_finetune_optimizer_steps=1,
                minimum_checkpoint_selection_finetune_steps=1,
                seed=37,
                device=mx.cpu,
                dynamic_crop_corpus=dynamic_corpus,
                quality_thresholds=thresholds,
                authorize_production_activation=True,
            )
            self.assertTrue(trained.production_ready)
            readiness = stage_edit_readiness_report(checkpoint)
            self.assertTrue(readiness.learned_inference_ready, readiness.reason)
            self.assertEqual("production_checkpoint_ready", readiness.status)
            self.assertEqual("dynamic_crop_v3", readiness.checkpoint_format)
            self.assertEqual(3, readiness.dynamic_crop_version)
            self.assertEqual(
                "dynamic-crop-v3-metadata.json",
                readiness.checkpoint_metadata_filename,
            )

            materialized = materialize_stage_edit_head_frames_v3(
                checkpoint_directory=checkpoint,
                sequence=examples[0].sequence,
                canonical_source_ids=("A", "B", "C"),
                schema=SCHEMA,
            )
            self.assertEqual(3, materialized.dynamic_crop_version)
            self.assertEqual(len(examples[0].sequence.tokens), len(materialized.frames))
            self.assertTrue(
                all(frame.dynamic_crop_v3_by_source for frame in materialized.frames)
            )

            report_path = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
            original_report = json.loads(report_path.read_text(encoding="utf-8"))
            tampered_model = copy.deepcopy(original_report)
            tampered_model["model"]["dynamic_crop_version"] = 2
            report_path.write_text(
                json.dumps(tampered_model), encoding="utf-8"
            )
            rejected_model = stage_edit_readiness_report(checkpoint)
            self.assertFalse(rejected_model.learned_inference_ready)
            self.assertIn("does not declare v3", rejected_model.reason)

            tampered_activation_checkpointing = copy.deepcopy(original_report)
            tampered_activation_checkpointing["model"][
                "activation_checkpointing"
            ]["enabled"] = False
            report_path.write_text(
                json.dumps(tampered_activation_checkpointing), encoding="utf-8"
            )
            rejected_activation_checkpointing = stage_edit_readiness_report(
                checkpoint
            )
            self.assertFalse(
                rejected_activation_checkpointing.learned_inference_ready
            )
            self.assertIn(
                "activation checkpointing evidence differs",
                rejected_activation_checkpointing.reason,
            )

            tampered_activation_types = copy.deepcopy(original_report)
            tampered_activation_types["model"]["activation_checkpointing"][
                "enabled"
            ] = 1
            tampered_activation_types["model"]["activation_checkpointing"][
                "previous_value"
            ] = 0
            report_path.write_text(
                json.dumps(tampered_activation_types), encoding="utf-8"
            )
            rejected_activation_types = stage_edit_readiness_report(checkpoint)
            self.assertFalse(rejected_activation_types.learned_inference_ready)
            self.assertIn(
                "activation checkpointing evidence differs",
                rejected_activation_types.reason,
            )

            tampered_gate = copy.deepcopy(original_report)
            tampered_gate["optimization"][
                "best_validation_selection_evidence"
            ]["dynamic_crop_v3_selection_gate"][
                "minimum_safe_zoom_projection_count"
            ] = 1
            report_path.write_text(
                json.dumps(tampered_gate), encoding="utf-8"
            )
            rejected_gate = stage_edit_readiness_report(checkpoint)
            self.assertFalse(rejected_gate.learned_inference_ready)
            self.assertIn("projection count is not zero", rejected_gate.reason)

            report_path.write_text(
                json.dumps(original_report), encoding="utf-8"
            )
            mixed_path = checkpoint / "dynamic-crop-v2-metadata.json"
            mixed_path.write_text("{}", encoding="utf-8")
            mixed = stage_edit_readiness_report(checkpoint)
            self.assertFalse(mixed.learned_inference_ready)
            self.assertEqual("checkpoint_invalid", mixed.status)
            self.assertIn("mixed model namespaces", mixed.reason)
            mixed_path.unlink()

            weight_path = checkpoint / "dynamic-crop-v3-model.safetensors"
            original_weights = weight_path.read_bytes()
            weight_path.write_bytes(original_weights + b"tamper")
            tampered_weights = stage_edit_readiness_report(checkpoint)
            self.assertFalse(tampered_weights.learned_inference_ready)
            self.assertEqual("untrained_checkpoint_only", tampered_weights.status)
            weight_path.write_bytes(original_weights)


if __name__ == "__main__":
    unittest.main()
