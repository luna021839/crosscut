from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from crosscut.auto_anchor import infer_member_anchor
from crosscut.pose import PersonDetection


class AutoAnchorTests(unittest.TestCase):
    def test_accepts_stable_centered_dominant_person(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = infer_member_anchor(
            Path("fancam.mp4"),
            source_index=2,
            duration_seconds=180,
            canonical_start_seconds=5,
            audio_offset_seconds=0.25,
            frame_reader=lambda source, time: frame,
            person_detector=lambda image: (
                PersonDetection((0.30, 0.12, 0.68, 0.96), 0.93),
            ),
        )
        self.assertTrue(result.accepted)
        assert result.anchor is not None
        self.assertEqual(result.anchor.source_index, 2)
        self.assertAlmostEqual(result.anchor.time_seconds, 6.05)
        self.assertGreater(result.confidence, 0.7)
        self.assertEqual(len(result.sampled_times_seconds), 7)
        self.assertAlmostEqual(
            result.sampled_times_seconds[-1]
            - result.sampled_times_seconds[0],
            1.5,
        )
        self.assertEqual(
            result.to_dict()["identity_claim"],
            "not_established_no_face_reference_geometry_only",
        )

    def test_rejects_comparable_people_in_every_sample(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = infer_member_anchor(
            Path("ambiguous.mp4"),
            source_index=0,
            duration_seconds=30,
            canonical_start_seconds=0,
            audio_offset_seconds=0,
            frame_reader=lambda source, time: frame,
            person_detector=lambda image: (
                PersonDetection((0.18, 0.12, 0.48, 0.94), 0.92),
                PersonDetection((0.52, 0.12, 0.82, 0.94), 0.91),
            ),
        )
        self.assertFalse(result.accepted)
        self.assertIsNone(result.anchor)
        self.assertIn("ambiguous_comparable_people", result.rejection_reasons)

    def test_accepts_clearly_subordinate_background_person(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detector_calls = 0

        def detect(image: np.ndarray) -> tuple[PersonDetection, ...]:
            nonlocal detector_calls
            del image
            checkpoint = detector_calls % 7
            detector_calls += 1
            primary = PersonDetection((0.30, 0.10, 0.70, 0.95), 0.94)
            if checkpoint == 3:
                return (
                    primary,
                    PersonDetection((0.73, 0.24, 0.91, 0.80), 0.58),
                )
            return (primary,)

        result = infer_member_anchor(
            Path("temporary-multiple.mp4"),
            source_index=0,
            duration_seconds=30,
            canonical_start_seconds=0,
            audio_offset_seconds=0,
            frame_reader=lambda source, time: frame,
            person_detector=detect,
        )

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.anchor)

    def test_accepts_one_soft_checkpoint_during_stable_choreography_pan(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detector_calls = 0
        boxes = (
            (0.30, 0.10, 0.66, 0.82),
            (0.34, 0.10, 0.70, 0.82),
            (0.40, 0.10, 0.76, 0.82),
            (0.57, 0.10, 0.93, 0.82),
            (0.52, 0.10, 0.88, 0.82),
            (0.47, 0.10, 0.83, 0.82),
            (0.43, 0.10, 0.79, 0.82),
        )

        def detect(image: np.ndarray) -> tuple[PersonDetection, ...]:
            nonlocal detector_calls
            del image
            checkpoint = detector_calls % 7
            detector_calls += 1
            confidence = 0.49 if checkpoint == 5 else 0.90
            return (PersonDetection(boxes[checkpoint], confidence),)

        result = infer_member_anchor(
            Path("choreography-pan.mp4"),
            source_index=0,
            duration_seconds=30,
            canonical_start_seconds=0,
            audio_offset_seconds=0,
            frame_reader=lambda source, time: frame,
            person_detector=detect,
        )

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.anchor)

    def test_rejects_when_only_first_three_checkpoints_match(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detector_calls = 0

        def detect(image: np.ndarray) -> tuple[PersonDetection, ...]:
            nonlocal detector_calls
            del image
            checkpoint = detector_calls % 7
            detector_calls += 1
            if checkpoint < 3:
                return (PersonDetection((0.28, 0.10, 0.68, 0.95), 0.94),)
            return (PersonDetection((0.58, 0.10, 0.92, 0.95), 0.94),)

        result = infer_member_anchor(
            Path("three-checkpoints-only.mp4"),
            source_index=0,
            duration_seconds=30,
            canonical_start_seconds=0,
            audio_offset_seconds=0,
            frame_reader=lambda source, time: frame,
            person_detector=detect,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.anchor)
        self.assertIn(
            "single_subject_not_continuous_across_window",
            result.rejection_reasons,
        )

    def test_skips_samples_near_scene_change(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = infer_member_anchor(
            Path("fancam.mp4"),
            source_index=0,
            duration_seconds=20,
            canonical_start_seconds=0,
            audio_offset_seconds=0,
            scene_change_seconds=(0.8,),
            frame_reader=lambda source, time: frame,
            person_detector=lambda image: (
                PersonDetection((0.30, 0.10, 0.70, 0.95), 0.95),
            ),
        )
        self.assertTrue(result.accepted)
        assert result.anchor is not None
        self.assertAlmostEqual(result.anchor.time_seconds, 2.0)
        self.assertIn("near_scene_change", result.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
