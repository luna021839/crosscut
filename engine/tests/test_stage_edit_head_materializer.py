from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.stage_edit_dynamic_crop_head_v2 import (
    StageEditDynamicCropModelOutputV2,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    StageEditDynamicCropModelOutputV3,
    StageEditTransformerDynamicCropV3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
)
from crosscut.stage_edit_head_materializer import (
    STAGE_EDIT_HEAD_MATERIALIZER_SCHEMA_VERSION,
    StageEditHeadMaterializerInputError,
    StageEditHeadMaterializerNotReadyError,
    materialize_stage_edit_head_frames_v3,
    stage_edit_head_frames_from_dynamic_crop_output_v2,
    stage_edit_head_frames_from_dynamic_crop_output_v3,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
    SourceCandidateToken,
    StageEditSequence,
    StageEditReadinessReport,
    StageEditTemporalPoseSidecar,
    StageEditToken,
    StageEditTransformerConfig,
    mlx_runtime_version,
    tensorize_stage_edit_batch,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


def _candidate(source_id: str, value: float, *, hard_gate: bool = True):
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return SourceCandidateToken(
        source_id=source_id,
        features=(value,) * schema.raw_source_dimension,
        missing_mask=(False,) * schema.raw_source_dimension,
        available=True,
        hard_gate_passed=hard_gate,
    )


def _candidate_with_pose(
    source_id: str,
    value: float,
    pose_embedding: tuple[float, ...] | None,
    *,
    partial_missing_index: int | None = None,
) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    features = [value] * schema.raw_source_dimension
    missing = [False] * schema.raw_source_dimension
    pose_indices = tuple(
        schema.source_feature_names.index(name)
        for name in STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES
    )
    if pose_embedding is None:
        for index in pose_indices:
            missing[index] = True
    else:
        if len(pose_embedding) != len(pose_indices):
            raise ValueError("test pose embedding must contain 60 values")
        for index, coordinate in zip(
            pose_indices, pose_embedding, strict=True
        ):
            features[index] = coordinate
    if partial_missing_index is not None:
        missing[pose_indices[partial_missing_index]] = True
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(features),
        missing_mask=tuple(missing),
        available=True,
        hard_gate_passed=True,
    )


def _sequence() -> StageEditSequence:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    tokens = (
        StageEditToken(
            0.0,
            (0.1,) * len(schema.music_feature_names),
            (0.2,) * len(schema.section_feature_names),
            (_candidate("B", 0.2), _candidate("A", 0.8)),
            (0.3,) * len(schema.recent_edit_feature_names),
            (0.4,) * len(schema.style_feature_names),
            (),
        ),
        StageEditToken(
            0.5,
            (0.2,) * len(schema.music_feature_names),
            (0.3,) * len(schema.section_feature_names),
            (_candidate("A", 0.7), _candidate("B", 0.1, hard_gate=False)),
            (0.4,) * len(schema.recent_edit_feature_names),
            (0.5,) * len(schema.style_feature_names),
            (),
        ),
    )
    return StageEditSequence("ive:materializer", tokens)


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditHeadMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        schema = DEFAULT_STAGE_EDIT_SCHEMA
        self.config = StageEditTransformerConfig.from_schema(
            schema,
            model_dimension=16,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=32,
            max_sequence_length=8,
            max_sources=3,
            dropout=0.0,
        )
        self.metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
            "head-materializer-test",
            schema,
            self.config,
            training_steps=1,
            weights_trained=True,
            training_data_id="synthetic",
        )
        self.metadata_v3 = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
            "head-materializer-v3-test",
            schema,
            self.config,
            training_steps=1,
            weights_trained=True,
            training_data_id="synthetic-v3",
        )
        self.sequence = _sequence()
        with mx.stream(mx.cpu):
            self.batch = tensorize_stage_edit_batch(
                (self.sequence,), schema, self.config, device=mx.cpu
            )

    def _output(self) -> StageEditDynamicCropModelOutputV2:
        source_logits = np.zeros((1, 2, 3), dtype=np.float32)
        source_logits[0, 0] = (11.0, 22.0, -1.0)
        source_logits[0, 1] = (33.0, 44.0, -1.0)
        values = np.zeros((1, 2, 8, 4), dtype=np.float32)
        values[:, :, 0] = (1.04, 0.500, 0.500, 0.0)
        values[:, :, 1] = (1.08, 0.505, 0.500, 0.0)
        times = np.zeros((1, 2, 8), dtype=np.float32)
        times[:, :, 1] = 0.5
        presence = np.full((1, 2, 8), -20.0, dtype=np.float32)
        presence[:, :, :2] = 20.0
        candidate = np.asarray(self.batch.source_candidate_mask, dtype=bool)
        gates = np.asarray(self.batch.hard_source_gate_mask, dtype=bool)
        return StageEditDynamicCropModelOutputV2(
            cut_timing_logits=np.array([[0.1, 0.2]], dtype=np.float32),
            source_selection_logits=source_logits,
            duration_bucket_logits=np.zeros(
                (1, 2, DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket_count),
                dtype=np.float32,
            ),
            transition_logits=np.zeros(
                (1, 2, len(DEFAULT_STAGE_EDIT_SCHEMA.transition_labels)),
                dtype=np.float32,
            ),
            crop=np.full((1, 2, 4), 999.0, dtype=np.float32),
            effective_source_mask=np.logical_and(candidate, gates),
            dynamic_crop_values=values,
            dynamic_crop_normalized_times=times,
            dynamic_crop_presence_logits=presence,
            dynamic_crop_motion_logits=np.ones((1, 2), dtype=np.float32),
        )

    def _output_v3(self) -> StageEditDynamicCropModelOutputV3:
        source_logits = np.zeros((1, 2, 3), dtype=np.float32)
        source_logits[0, 0] = (11.0, 22.0, -1.0)
        source_logits[0, 1] = (33.0, 44.0, -1.0)
        values = np.zeros((1, 2, 3, 8, 4), dtype=np.float32)
        values[..., 0] = 1.02
        values[..., 1] = 0.5
        values[..., 2] = 0.5
        # Token zero is padded B,A but the expected materialized order is A,B.
        values[0, 0, 0, :, 0] = 1.04
        values[0, 0, 1, :, 0] = 1.14
        values[0, 1, 0, :, 0] = 1.12
        values[0, 1, 1, :, 0] = 1.05
        count_logits = np.full((1, 2, 3, 8), -20.0, dtype=np.float32)
        count_logits[..., 0] = 20.0
        candidate = np.asarray(self.batch.source_candidate_mask, dtype=bool)
        gates = np.asarray(self.batch.hard_source_gate_mask, dtype=bool)
        return StageEditDynamicCropModelOutputV3(
            cut_timing_logits=np.array([[0.1, 0.2]], dtype=np.float32),
            source_selection_logits=source_logits,
            duration_bucket_logits=np.zeros(
                (1, 2, DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket_count),
                dtype=np.float32,
            ),
            transition_logits=np.zeros(
                (1, 2, len(DEFAULT_STAGE_EDIT_SCHEMA.transition_labels)),
                dtype=np.float32,
            ),
            crop=np.full((1, 2, 4), 999.0, dtype=np.float32),
            effective_source_mask=np.logical_and(candidate, gates),
            dynamic_crop_values=values,
            dynamic_crop_count_logits=count_logits,
            dynamic_crop_time_interval_logits=np.zeros(
                (1, 2, 3, 35), dtype=np.float32
            ),
            dynamic_crop_motion_logits=np.ones((1, 2, 3), dtype=np.float32),
        )

    def test_remaps_padded_source_slots_and_preserves_raw_k8_prediction(self) -> None:
        frames = stage_edit_head_frames_from_dynamic_crop_output_v2(
            sequence=self.sequence,
            batch=self.batch,
            output=self._output(),
            canonical_source_ids=("A", "B"),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            metadata=self.metadata,
            source_shot_sizes_by_token=(
                {"A": "closeup", "B": "wide"},
                {"A": "medium", "B": "full"},
            ),
        )

        self.assertEqual(2, len(frames))
        self.assertEqual((22.0, 11.0), frames[0].source_logits)
        self.assertEqual((33.0, 44.0), frames[1].source_logits)
        self.assertEqual((True, True), frames[0].effective_source_mask)
        self.assertEqual((True, False), frames[1].effective_source_mask)
        self.assertEqual(("closeup", "wide"), frames[0].source_shot_sizes)
        self.assertEqual(2, len(frames[0].source_continuity))
        self.assertAlmostEqual(0.8, frames[0].source_continuity[0].face_scale)
        self.assertAlmostEqual(0.2, frames[0].source_continuity[1].face_scale)
        self.assertAlmostEqual(
            0.8,
            frames[0].source_continuity[0].camera_motion_x,
        )
        self.assertEqual(
            (0.8,) * len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES),
            frames[0].source_continuity[0].pose_model_embedding,
        )
        serialized_continuity = frames[0].to_dict()["source_continuity"][0]
        pose_evidence = serialized_continuity["same_time_pose_evidence"]
        self.assertTrue(pose_evidence["observed"])
        self.assertEqual(60, pose_evidence["feature_dimension"])
        self.assertEqual(64, len(pose_evidence["content_sha256"]))
        self.assertNotIn("pose_model_embedding", serialized_continuity)
        self.assertEqual((999.0, 999.0, 999.0, 999.0), frames[0].crop)
        self.assertAlmostEqual(0.2, frames[0].section_progress)
        self.assertAlmostEqual(0.1, frames[0].phrase_progress)
        assert frames[0].dynamic_crop_v2 is not None
        self.assertEqual(8, len(frames[0].dynamic_crop_v2.values))
        decoded = frames[0].dynamic_crop_v2.decode(2.0)
        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        assert decoded.track is not None
        self.assertEqual(1.0, decoded.track.keyframes[1].time_seconds)

    def test_source_exact_temporal_sidecar_reaches_head_continuity(self) -> None:
        sidecar = StageEditTemporalPoseSidecar(
            embeddings=((0.1,) * 60, (0.2,) * 60),
            interval_seconds=0.04,
            sampling_contract=(
                STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT
            ),
        )
        first_token = self.sequence.tokens[0]
        # The supplied first token is padded B,A. Canonical head order is A,B.
        modified_sequence = replace(
            self.sequence,
            tokens=(
                replace(
                    first_token,
                    source_candidates=(
                        first_token.source_candidates[0],
                        replace(
                            first_token.source_candidates[1],
                            temporal_pose_sidecar=sidecar,
                        ),
                    ),
                ),
                self.sequence.tokens[1],
            ),
        )
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                (modified_sequence,),
                DEFAULT_STAGE_EDIT_SCHEMA,
                self.config,
                device=mx.cpu,
            )
            mx.eval(
                batch.source_features,
                batch.source_candidate_mask,
                batch.hard_source_gate_mask,
            )
        frames = stage_edit_head_frames_from_dynamic_crop_output_v2(
            sequence=modified_sequence,
            batch=batch,
            output=self._output(),
            canonical_source_ids=("A", "B"),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            metadata=self.metadata,
        )
        self.assertEqual(frames[0].source_continuity[0].temporal_pose_sidecar, sidecar)
        self.assertIsNone(frames[0].source_continuity[1].temporal_pose_sidecar)
        serialized = frames[0].to_dict()["source_continuity"][0]
        temporal_evidence = serialized["temporal_pose_evidence"]
        self.assertTrue(temporal_evidence["observed"])
        self.assertEqual(2, temporal_evidence["sample_count"])
        self.assertEqual(60, temporal_evidence["feature_dimension"])
        self.assertEqual(
            sidecar.producer_backend,
            temporal_evidence["trusted_producer_contract"],
        )
        self.assertEqual(64, len(temporal_evidence["content_sha256"]))
        self.assertNotIn("temporal_pose_sidecar", serialized)

        alternate_sidecar = replace(
            sidecar,
            embeddings=((0.1,) * 60, (0.21,) * 60),
        )
        alternate_frame = replace(
            frames[0],
            source_continuity=(
                replace(
                    frames[0].source_continuity[0],
                    temporal_pose_sidecar=alternate_sidecar,
                ),
                *frames[0].source_continuity[1:],
            ),
        )

        def frame_sha256(frame: object) -> str:
            return hashlib.sha256(
                json.dumps(
                    frame.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        alternate_evidence = alternate_frame.to_dict()["source_continuity"][
            0
        ]["temporal_pose_evidence"]
        self.assertNotEqual(
            temporal_evidence["content_sha256"],
            alternate_evidence["content_sha256"],
        )
        self.assertNotEqual(frame_sha256(frames[0]), frame_sha256(alternate_frame))

    def test_output_mask_mismatch_fails_before_beam(self) -> None:
        output = self._output()
        output.effective_source_mask[0, 0, 0] = False
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError, "effective-source mask"
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v2(
                sequence=self.sequence,
                batch=self.batch,
                output=output,
                canonical_source_ids=("A", "B"),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata,
            )

    def test_recovers_shot_family_and_formation_evidence_from_model_input(self) -> None:
        frames = stage_edit_head_frames_from_dynamic_crop_output_v2(
            sequence=self.sequence,
            batch=self.batch,
            output=self._output(),
            canonical_source_ids=("A", "B"),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            metadata=self.metadata,
        )

        self.assertEqual(("closeup", "wide"), frames[0].source_shot_sizes)
        self.assertEqual(("closeup", "wide"), frames[1].source_shot_sizes)
        self.assertAlmostEqual(
            0.8,
            frames[0].source_continuity[0].formation_person_count_normalized,
        )
        self.assertAlmostEqual(
            0.2,
            frames[0].source_continuity[1].formation_pose_coverage,
        )

    def test_v3_remaps_each_padded_prediction_to_exact_canonical_source(self) -> None:
        frames = stage_edit_head_frames_from_dynamic_crop_output_v3(
            sequence=self.sequence,
            batch=self.batch,
            output=self._output_v3(),
            canonical_source_ids=("A", "B"),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            metadata=self.metadata_v3,
        )

        self.assertEqual((22.0, 11.0), frames[0].source_logits)
        predictions = frames[0].dynamic_crop_v3_by_source
        self.assertEqual(2, len(predictions))
        assert predictions[0] is not None
        assert predictions[1] is not None
        self.assertEqual(("A", 0), (predictions[0].source_id, predictions[0].source_index))
        self.assertEqual(("B", 1), (predictions[1].source_id, predictions[1].source_index))
        self.assertAlmostEqual(1.14, predictions[0].values[0][0], places=5)
        self.assertAlmostEqual(1.04, predictions[1].values[0][0], places=5)
        self.assertIsNone(frames[1].dynamic_crop_v3_by_source[1])
        self.assertEqual((True, False), frames[1].effective_source_mask)
        self.assertIsNone(frames[0].dynamic_crop_v2)
        self.assertEqual(
            (0.8,) * len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES),
            frames[0].source_continuity[0].pose_model_embedding,
        )

    def test_pose_embedding_missingness_is_explicit_and_partial_fails_closed(
        self,
    ) -> None:
        missing_candidate = _candidate_with_pose("A", 0.8, None)
        missing_sequence = replace(
            self.sequence,
            tokens=(
                replace(
                    self.sequence.tokens[0],
                    source_candidates=(
                        self.sequence.tokens[0].source_candidates[0],
                        missing_candidate,
                    ),
                ),
                self.sequence.tokens[1],
            ),
        )
        with mx.stream(mx.cpu):
            missing_batch = tensorize_stage_edit_batch(
                (missing_sequence,),
                DEFAULT_STAGE_EDIT_SCHEMA,
                self.config,
                device=mx.cpu,
            )
        missing_frames = stage_edit_head_frames_from_dynamic_crop_output_v2(
            sequence=missing_sequence,
            batch=missing_batch,
            output=self._output(),
            canonical_source_ids=("A", "B"),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            metadata=self.metadata,
        )
        self.assertIsNone(
            missing_frames[0].source_continuity[0].pose_model_embedding
        )
        missing_pose_evidence = missing_frames[0].to_dict()[
            "source_continuity"
        ][0]["same_time_pose_evidence"]
        self.assertFalse(missing_pose_evidence["observed"])
        self.assertIsNone(missing_pose_evidence["content_sha256"])

        partial_candidate = _candidate_with_pose(
            "A",
            0.8,
            (0.25,) * len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES),
            partial_missing_index=17,
        )
        partial_sequence = replace(
            missing_sequence,
            tokens=(
                replace(
                    missing_sequence.tokens[0],
                    source_candidates=(
                        missing_sequence.tokens[0].source_candidates[0],
                        partial_candidate,
                    ),
                ),
                missing_sequence.tokens[1],
            ),
        )
        with mx.stream(mx.cpu):
            partial_batch = tensorize_stage_edit_batch(
                (partial_sequence,),
                DEFAULT_STAGE_EDIT_SCHEMA,
                self.config,
                device=mx.cpu,
            )
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError,
            "all 60 coordinates observed or all 60 missing",
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v2(
                sequence=partial_sequence,
                batch=partial_batch,
                output=self._output(),
                canonical_source_ids=("A", "B"),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata,
            )

    def test_observed_pose_embedding_must_be_finite_and_bounded(self) -> None:
        invalid_candidate = _candidate_with_pose(
            "A",
            0.8,
            (1.01,) * len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES),
        )
        invalid_sequence = replace(
            self.sequence,
            tokens=(
                replace(
                    self.sequence.tokens[0],
                    source_candidates=(
                        self.sequence.tokens[0].source_candidates[0],
                        invalid_candidate,
                    ),
                ),
                self.sequence.tokens[1],
            ),
        )
        with mx.stream(mx.cpu):
            invalid_batch = tensorize_stage_edit_batch(
                (invalid_sequence,),
                DEFAULT_STAGE_EDIT_SCHEMA,
                self.config,
                device=mx.cpu,
            )
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError,
            r"finite and lie in \[-1, 1\]",
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v3(
                sequence=invalid_sequence,
                batch=invalid_batch,
                output=self._output_v3(),
                canonical_source_ids=("A", "B"),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata_v3,
            )

    def test_v3_duplicate_padded_source_identity_fails_closed(self) -> None:
        bad_batch = replace(
            self.batch,
            source_ids=((('B', 'B', None), ('A', 'B', None)),),
        )
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError, "padded source ids.*unique"
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v3(
                sequence=self.sequence,
                batch=bad_batch,
                output=self._output_v3(),
                canonical_source_ids=("A", "B"),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata_v3,
            )

    def test_v3_source_order_and_effective_gates_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError,
            "outside canonical order",
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v3(
                sequence=self.sequence,
                batch=self.batch,
                output=self._output_v3(),
                canonical_source_ids=("A",),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata_v3,
            )

        mismatched = self._output_v3()
        mismatched.effective_source_mask[0, 0, 0] = False
        with self.assertRaisesRegex(
            StageEditHeadMaterializerInputError,
            "effective-source mask",
        ):
            stage_edit_head_frames_from_dynamic_crop_output_v3(
                sequence=self.sequence,
                batch=self.batch,
                output=mismatched,
                canonical_source_ids=("A", "B"),
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
                metadata=self.metadata_v3,
            )

    def test_v3_checkpoint_forward_reports_versioned_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            checkpoint = Path(directory)
            model = StageEditTransformerDynamicCropV3(self.config)
            saved = save_stage_edit_dynamic_crop_checkpoint_v3(
                checkpoint,
                model,
                self.metadata_v3,
            )
            readiness = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="test production evidence",
                mlx_version=mlx_runtime_version(),
                checkpoint_path=str(checkpoint.resolve()),
                training_steps=saved.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename="dynamic-crop-v3-metadata.json",
            )
            with patch(
                "crosscut.stage_edit_head_materializer.stage_edit_readiness_report",
                return_value=readiness,
            ):
                result = materialize_stage_edit_head_frames_v3(
                    checkpoint_directory=checkpoint,
                    sequence=self.sequence,
                    canonical_source_ids=("A", "B"),
                    schema=DEFAULT_STAGE_EDIT_SCHEMA,
                )

            self.assertEqual(
                STAGE_EDIT_HEAD_MATERIALIZER_SCHEMA_VERSION,
                result.schema_version,
            )
            self.assertEqual("dynamic_crop_v3", result.checkpoint_format)
            self.assertEqual(3, result.dynamic_crop_version)
            self.assertEqual(
                "dynamic-crop-v3-metadata.json",
                result.checkpoint_metadata_filename,
            )
            self.assertEqual(
                DYNAMIC_CROP_V3_MODEL_CONTRACT,
                result.checkpoint_contracts["model_contract"],
            )
            self.assertEqual(
                saved.contract_fingerprint_sha256,
                result.checkpoint_contracts["contract_fingerprint_sha256"],
            )
            result_payload = result.to_dict()
            self.assertEqual(3, result_payload["dynamic_crop_version"])
            self.assertEqual(
                DYNAMIC_CROP_V3_MODEL_CONTRACT,
                result_payload["checkpoint_contracts"]["model_contract"],
            )
            self.assertTrue(
                all(
                    frame.dynamic_crop_v3_by_source
                    for frame in result.frames
                )
            )
            self.assertTrue(
                all(frame.dynamic_crop_v2 is None for frame in result.frames)
            )

    def test_v3_checkpoint_namespace_and_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            checkpoint = Path(directory)
            model = StageEditTransformerDynamicCropV3(self.config)
            saved = save_stage_edit_dynamic_crop_checkpoint_v3(
                checkpoint,
                model,
                self.metadata_v3,
            )
            readiness = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="test production evidence",
                mlx_version=mlx_runtime_version(),
                checkpoint_path=str(checkpoint.resolve()),
                training_steps=saved.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename="dynamic-crop-v3-metadata.json",
            )
            (checkpoint / "dynamic-crop-v2-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                StageEditHeadMaterializerNotReadyError,
                "unambiguous Dynamic Crop v3",
            ):
                materialize_stage_edit_head_frames_v3(
                    checkpoint_directory=checkpoint,
                    sequence=self.sequence,
                    canonical_source_ids=("A", "B"),
                )
            (checkpoint / "dynamic-crop-v2-metadata.json").unlink()

            incompatible_schema = replace(
                DEFAULT_STAGE_EDIT_SCHEMA,
                style_feature_names=(
                    *DEFAULT_STAGE_EDIT_SCHEMA.style_feature_names,
                    "test_only_extra_style",
                ),
            )
            with patch(
                "crosscut.stage_edit_head_materializer.stage_edit_readiness_report",
                return_value=readiness,
            ), self.assertRaisesRegex(
                StageEditHeadMaterializerInputError,
                "token schema differs",
            ):
                materialize_stage_edit_head_frames_v3(
                    checkpoint_directory=checkpoint,
                    sequence=self.sequence,
                    canonical_source_ids=("A", "B"),
                    schema=incompatible_schema,
                )


if __name__ == "__main__":
    unittest.main()
