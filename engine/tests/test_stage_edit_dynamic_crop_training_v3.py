from __future__ import annotations

import unittest

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from crosscut.stage_edit_dynamic_crop_training_v3 import (
    build_stage_edit_dynamic_crop_training_corpus_v3,
)


def _trajectory(*, shot: int, motion: str) -> DynamicCropTrajectoryTargetV2:
    frames = tuple(
        EditorialTransformKeyframe(
            time_seconds=index * 0.2,
            zoom=(1.02 if motion == "static" else 1.01 + 0.006 * index),
            center_x=(0.5 if motion == "static" else 0.498 + 0.001 * index),
            center_y=0.5,
            roll_degrees=(0.0 if motion == "static" else -0.04 + 0.02 * index),
        )
        for index in range(6)
    )
    return DynamicCropTrajectoryTargetV2(
        shot_index=shot,
        source_id=f"source-{shot}",
        shot_duration_seconds=2.0,
        keyframes=frames,
        motion_label=motion,
        motion_label_loss_masked=False,
        artifact_content_sha256=f"{shot + 1:x}" * 64,
        build_fingerprint_sha256=f"{shot + 3:x}" * 64,
    )


class DynamicCropTrainingCorpusV3Tests(unittest.TestCase):
    def test_v2_evidence_is_promoted_without_new_keyframes(self) -> None:
        source = StageEditDynamicCropTrainingCorpusV2(
            sequences={
                "sequence-A": StageEditDynamicCropTargetSequenceV2(
                    "sequence-A",
                    (
                        _trajectory(shot=0, motion="static"),
                        None,
                        _trajectory(shot=1, motion="dynamic"),
                    ),
                )
            },
            data_id="v2-data-id",
            artifact_paths=("/immutable/a.json", "/immutable/b.json"),
            statistics={
                "reference_count": 1,
                "song_count": 1,
                "artifact_count": 2,
            },
        )

        first = build_stage_edit_dynamic_crop_training_corpus_v3(source)
        second = build_stage_edit_dynamic_crop_training_corpus_v3(source)

        self.assertEqual(first.data_id, second.data_id)
        self.assertEqual(first.source_v2_data_id, source.data_id)
        self.assertEqual(first.artifact_paths, source.artifact_paths)
        self.assertEqual(first.statistics["track_label_count"], 2)
        self.assertEqual(first.statistics["original_keyframe_label_count"], 12)
        self.assertEqual(first.statistics["simplified_keyframe_label_count"], 3)
        self.assertEqual(first.statistics["new_keyframe_count"], 0)
        self.assertEqual(
            first.statistics["simplified_keyframe_count_histogram"],
            {"1": 1, "2": 1},
        )
        sequence = first.sequence_for("sequence-A")
        self.assertEqual(len(sequence.targets[0].keyframes), 1)
        self.assertEqual(len(sequence.targets[2].keyframes), 2)
        self.assertEqual(
            sequence.targets[0].keyframes[0],
            source.sequence_for("sequence-A").targets[0].keyframes[0],
        )
        manifest = first.manifest()
        self.assertTrue(manifest["input_policy"]["v2_artifacts_remain_immutable"])
        self.assertFalse(manifest["input_policy"]["new_keyframes_synthesized"])


if __name__ == "__main__":
    unittest.main()
