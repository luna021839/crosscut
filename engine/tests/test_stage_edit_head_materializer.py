from __future__ import annotations

import unittest

import numpy as np

from crosscut.stage_edit_dynamic_crop_head_v2 import (
    StageEditDynamicCropModelOutputV2,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
)
from crosscut.stage_edit_head_materializer import (
    StageEditHeadMaterializerInputError,
    stage_edit_head_frames_from_dynamic_crop_output_v2,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
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
            (False,) * schema.raw_context_dimension,
        ),
        StageEditToken(
            0.5,
            (0.2,) * len(schema.music_feature_names),
            (0.3,) * len(schema.section_feature_names),
            (_candidate("A", 0.7), _candidate("B", 0.1, hard_gate=False)),
            (0.4,) * len(schema.recent_edit_feature_names),
            (0.5,) * len(schema.style_feature_names),
            (False,) * schema.raw_context_dimension,
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
        self.assertEqual((999.0, 999.0, 999.0, 999.0), frames[0].crop)
        assert frames[0].dynamic_crop_v2 is not None
        self.assertEqual(8, len(frames[0].dynamic_crop_v2.values))
        decoded = frames[0].dynamic_crop_v2.decode(2.0)
        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        assert decoded.track is not None
        self.assertEqual(1.0, decoded.track.keyframes[1].time_seconds)

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


if __name__ == "__main__":
    unittest.main()
