from __future__ import annotations

import unittest

from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack
from crosscut.stage_edit_dynamic_crop_head_v2 import DynamicCropTrajectoryTargetV2
from crosscut.stage_edit_dynamic_crop_target_v3 import (
    DynamicCropV3SimplificationConfig,
    simplify_dynamic_crop_trajectory_v3,
)


def _target(
    keyframes: tuple[EditorialTransformKeyframe, ...],
    *,
    motion: str,
) -> DynamicCropTrajectoryTargetV2:
    return DynamicCropTrajectoryTargetV2(
        shot_index=3,
        source_id="source-A",
        shot_duration_seconds=2.0,
        keyframes=keyframes,
        motion_label=motion,
        motion_label_loss_masked=False,
        artifact_content_sha256="a" * 64,
        build_fingerprint_sha256="b" * 64,
    )


class DynamicCropTargetV3Tests(unittest.TestCase):
    def test_static_redundant_observations_collapse_to_real_time_zero(self) -> None:
        keyframes = tuple(
            EditorialTransformKeyframe(
                time_seconds=index * 0.25,
                zoom=1.020 + index * 0.0002,
                center_x=0.500 + index * 0.00005,
                center_y=0.500 - index * 0.00004,
                roll_degrees=index * 0.003,
            )
            for index in range(6)
        )
        simplified = simplify_dynamic_crop_trajectory_v3(
            _target(keyframes, motion="static")
        )

        self.assertEqual(simplified.original_keyframe_count, 6)
        self.assertEqual(simplified.retained_original_indices, (0,))
        self.assertEqual(simplified.keyframes, (keyframes[0],))
        self.assertEqual(simplified.removed_original_indices, (1, 2, 3, 4, 5))
        self.assertLessEqual(
            simplified.maximum_normalized_reconstruction_error,
            DynamicCropV3SimplificationConfig().static_maximum_normalized_error,
        )
        EditorialTransformTrack(simplified.keyframes)

    def test_dynamic_curve_keeps_original_endpoints_and_removes_linear_knots(self) -> None:
        keyframes = tuple(
            EditorialTransformKeyframe(
                time_seconds=index * 0.25,
                zoom=1.010 + index * 0.006,
                center_x=0.498 + index * 0.001,
                center_y=0.501 + index * 0.0004,
                roll_degrees=-0.08 + index * 0.04,
            )
            for index in range(6)
        )
        simplified = simplify_dynamic_crop_trajectory_v3(
            _target(keyframes, motion="dynamic")
        )

        self.assertEqual(simplified.retained_original_indices, (0, 5))
        self.assertEqual(simplified.keyframes, (keyframes[0], keyframes[-1]))
        self.assertEqual(simplified.original_keyframe_count, 6)
        self.assertLessEqual(
            simplified.maximum_normalized_reconstruction_error,
            DynamicCropV3SimplificationConfig().dynamic_maximum_normalized_error,
        )
        EditorialTransformTrack(simplified.keyframes)

    def test_non_linear_dynamic_knot_is_retained(self) -> None:
        keyframes = (
            EditorialTransformKeyframe(0.0, 1.02, 0.500, 0.500, 0.0),
            EditorialTransformKeyframe(0.5, 1.08, 0.515, 0.500, 0.2),
            EditorialTransformKeyframe(1.0, 1.02, 0.500, 0.500, 0.0),
        )
        simplified = simplify_dynamic_crop_trajectory_v3(
            _target(keyframes, motion="dynamic")
        )

        self.assertEqual(simplified.retained_original_indices, (0, 1, 2))
        self.assertEqual(simplified.keyframes, keyframes)

    def test_simplification_never_synthesizes_keyframe_values(self) -> None:
        keyframes = (
            EditorialTransformKeyframe(0.0, 1.01, 0.500, 0.500, 0.0),
            EditorialTransformKeyframe(0.3, 1.02, 0.501, 0.500, 0.02),
            EditorialTransformKeyframe(0.6, 1.03, 0.502, 0.500, 0.04),
            EditorialTransformKeyframe(0.9, 1.04, 0.503, 0.500, 0.06),
        )
        simplified = simplify_dynamic_crop_trajectory_v3(
            _target(keyframes, motion="dynamic")
        )

        original_values = set(keyframes)
        self.assertTrue(all(item in original_values for item in simplified.keyframes))
        self.assertEqual(simplified.keyframes[0], keyframes[0])
        self.assertEqual(simplified.keyframes[-1], keyframes[-1])


if __name__ == "__main__":
    unittest.main()
