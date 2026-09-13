from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crosscut.reference_match import (
    ANALYSIS_BASIS,
    DecodedTargetFrames,
    FrameMatch,
    OfflineReferenceMatcher,
    ReferenceMatchConfig,
    decode_target_frames_sequential,
    decide_reference_match,
    match_aligned_target_frames,
)


def textured_frame(seed: int, label: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((216, 384, 3), dtype=np.uint8)
    for _ in range(80):
        center = (
            int(rng.integers(8, frame.shape[1] - 8)),
            int(rng.integers(8, frame.shape[0] - 8)),
        )
        color = tuple(int(value) for value in rng.integers(40, 256, size=3))
        cv2.circle(
            frame,
            center,
            int(rng.integers(2, 8)),
            color,
            -1,
        )
    cv2.putText(
        frame,
        label,
        (55, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return frame


class ReferenceMatchFeatureTests(unittest.TestCase):
    def test_production_defaults_preserve_calibrated_scratch_policy(self) -> None:
        config = ReferenceMatchConfig()
        payload = config.to_dict()

        self.assertEqual(payload["analysis_basis"], ANALYSIS_BASIS)
        self.assertEqual(config.working_width, 384)
        self.assertEqual(config.orb_nfeatures, 900)
        self.assertEqual(config.orb_fast_threshold, 12)
        self.assertEqual(config.ratio_test_threshold, 0.76)
        self.assertEqual(config.ransac_reprojection_threshold, 4.0)
        self.assertEqual((config.structure_width, config.structure_height), (64, 36))
        self.assertEqual((config.hsv_hue_bins, config.hsv_saturation_bins), (24, 16))
        self.assertEqual(
            payload["score_weights"],
            {
                "homography_inliers": 0.50,
                "ratio_matches": 0.12,
                "homography_inlier_ratio": 0.10,
                "structure_ncc": 0.18,
                "hsv_color_similarity": 0.10,
            },
        )
        self.assertEqual(config.temporal_jitter_frames, (-2, 0, 2))
        json.dumps(payload)

    def test_identical_and_translated_stage_frames_beat_unrelated_image(self) -> None:
        reference_frame = textured_frame(42, "IVE I AM")
        translated = cv2.warpAffine(
            reference_frame,
            np.float32(((1.0, 0.0, 8.0), (0.0, 1.0, 4.0))),
            (384, 216),
        )
        unrelated = textured_frame(777, "OTHER")
        matcher = OfflineReferenceMatcher()
        reference = matcher.extract(reference_frame)

        identical_match = matcher.compare(reference, matcher.extract(reference_frame))
        translated_match = matcher.compare(reference, matcher.extract(translated))
        unrelated_match = matcher.compare(reference, matcher.extract(unrelated))

        self.assertGreater(identical_match.score, 0.95)
        self.assertGreater(identical_match.orb_homography_inliers, 100)
        self.assertAlmostEqual(identical_match.orb_inlier_ratio, 1.0)
        self.assertIsNotNone(identical_match.homography_reference_to_source)
        self.assertGreater(translated_match.score, 0.72)
        self.assertGreater(translated_match.orb_homography_inliers, 40)
        self.assertGreater(translated_match.orb_inlier_ratio, 0.70)
        self.assertIsNotNone(translated_match.homography_reference_to_source)
        self.assertGreater(translated_match.score, unrelated_match.score + 0.45)

    def test_textureless_frames_stay_finite_without_inventing_orb_support(self) -> None:
        matcher = OfflineReferenceMatcher()
        flat = np.full((108, 192, 3), (20, 80, 160), dtype=np.uint8)

        match = matcher.compare(matcher.extract(flat), matcher.extract(flat))

        self.assertEqual(match.orb_good_matches, 0)
        self.assertEqual(match.orb_homography_inliers, 0)
        self.assertEqual(match.orb_inlier_ratio, 0.0)
        self.assertEqual(match.structure_ncc, 0.0)
        self.assertTrue(0.0 <= match.score <= 0.10)
        self.assertIsNone(match.homography_reference_to_source)

    def test_match_payload_uses_reference_dataset_compatible_field_names(self) -> None:
        matcher = OfflineReferenceMatcher()
        frame = textured_frame(9, "GAEUL")
        match = matcher.compare(matcher.extract(frame), matcher.extract(frame))
        payload = match.to_dict()

        self.assertEqual(
            set(payload),
            {
                "score",
                "orb_good_matches",
                "orb_homography_inliers",
                "orb_inlier_ratio",
                "structure_ncc",
                "color_similarity",
                "homography_reference_to_source",
            },
        )
        self.assertEqual(len(payload["homography_reference_to_source"]), 3)
        json.dumps(payload)


class SequentialTargetDecoderTests(unittest.TestCase):
    def test_requested_indices_are_deduplicated_and_read_in_one_forward_pass(self) -> None:
        frames = [
            np.full((20, 30, 3), index, dtype=np.uint8)
            for index in range(6)
        ]

        class FakeCapture:
            def __init__(self) -> None:
                self.position = 0
                self.read_count = 0
                self.grab_count = 0
                self.retrieve_count = 0
                self.released = False

            @staticmethod
            def isOpened() -> bool:
                return True

            @staticmethod
            def get(property_id: int) -> float:
                if property_id == cv2.CAP_PROP_FPS:
                    return 30.0
                if property_id == cv2.CAP_PROP_FRAME_COUNT:
                    return 6.0
                return 0.0

            def read(self):  # type: ignore[no-untyped-def]
                self.read_count += 1
                raise AssertionError("unrequested BGR retrieval")

            def grab(self):  # type: ignore[no-untyped-def]
                self.grab_count += 1
                self.position += 1
                return self.position <= len(frames)

            def retrieve(self):  # type: ignore[no-untyped-def]
                self.retrieve_count += 1
                return True, frames[self.position - 1].copy()

            def release(self) -> None:
                self.released = True

        capture = FakeCapture()
        with patch("crosscut.reference_match.cv2.VideoCapture", return_value=capture):
            batch = decode_target_frames_sequential(
                Path("offline-source.mp4"),
                (4, 1, 4, -2),
            )

        self.assertEqual(batch.frame_rate, 30.0)
        self.assertEqual(batch.declared_frame_count, 6)
        self.assertEqual(batch.requested_indices, (1, 4))
        self.assertEqual(batch.decoded_indices, (1, 4))
        self.assertEqual(batch.missing_indices, ())
        self.assertEqual(set(batch.frames), {1, 4})
        self.assertEqual(batch.frames[1].shape[1], 384)
        self.assertEqual(int(batch.frames[4][0, 0, 0]), 4)
        self.assertEqual(capture.read_count, 0)
        self.assertEqual(capture.grab_count, 5)
        self.assertEqual(capture.retrieve_count, 2)
        self.assertTrue(capture.released)

    def test_eof_and_retrieve_failure_keep_exact_missing_indices(self) -> None:
        for fail_retrieve in (False, True):
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [30.0, 2.0]
            capture.grab.side_effect = [True, True, False]
            capture.retrieve.return_value = (
                (False, None) if fail_retrieve else
                (True, np.zeros((20, 30, 3), dtype=np.uint8))
            )
            with patch("crosscut.reference_match.cv2.VideoCapture", return_value=capture):
                batch = decode_target_frames_sequential(Path("source.mp4"), (0, 3))
            self.assertEqual(batch.decoded_indices, () if fail_retrieve else (0,))
            self.assertEqual(batch.missing_indices, (0, 3) if fail_retrieve else (3,))
            capture.release.assert_called_once()
            capture.read.assert_not_called()
            capture.set.assert_not_called()

    def test_retrieve_exception_releases_capture(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [30.0, 6.0]
        capture.grab.return_value = True
        capture.retrieve.side_effect = RuntimeError("backend failed")
        with patch("crosscut.reference_match.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                decode_target_frames_sequential(Path("source.mp4"), (0,))
        capture.release.assert_called_once()

    def test_invalid_index_and_invalid_video_metadata_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "indices"):
            decode_target_frames_sequential(Path("x"), (1, 2.5))  # type: ignore[arg-type]

        class BadRateCapture:
            @staticmethod
            def isOpened() -> bool:
                return True

            @staticmethod
            def get(_property_id: int) -> float:
                return 0.0

            @staticmethod
            def release() -> None:
                return None

        with patch(
            "crosscut.reference_match.cv2.VideoCapture",
            return_value=BadRateCapture(),
        ), self.assertRaisesRegex(ValueError, "frame rate"):
            decode_target_frames_sequential(Path("x"), (1,))


class AlignedReferenceMatchTests(unittest.TestCase):
    def test_equal_scores_prefer_expected_time_then_group_center(self) -> None:
        matcher = OfflineReferenceMatcher()
        held_frame = textured_frame(71, "HELD")
        reference = matcher.extract(held_frame)
        decoded = DecodedTargetFrames(
            frame_rate=20.0,
            declared_frame_count=20,
            requested_indices=(8, 10, 12),
            frames={
                8: held_frame.copy(),
                10: held_frame.copy(),
                12: held_frame.copy(),
            },
        )
        with patch(
            "crosscut.reference_match.decode_target_frames_sequential",
            return_value=decoded,
        ):
            expected_time_match = match_aligned_target_frames(
                Path("source.mp4"),
                ((8, 10, 12),),
                (reference,),
                expected_source_times=(0.60,),
                matcher=matcher,
            )[0]
            centered_match = match_aligned_target_frames(
                Path("source.mp4"),
                ((8, 10, 12),),
                (reference,),
                matcher=matcher,
            )[0]

        self.assertIsNotNone(expected_time_match)
        self.assertIsNotNone(centered_match)
        assert expected_time_match is not None and centered_match is not None
        self.assertEqual(expected_time_match.source_frame_index, 12)
        self.assertEqual(centered_match.source_frame_index, 10)

    def test_union_is_decoded_once_and_best_jitter_frame_is_selected(self) -> None:
        matcher = OfflineReferenceMatcher()
        first = textured_frame(10, "FIRST")
        second = textured_frame(20, "SECOND")
        unrelated = textured_frame(30, "NONE")
        references = (matcher.extract(first), matcher.extract(second))
        decoded = DecodedTargetFrames(
            frame_rate=20.0,
            declared_frame_count=8,
            requested_indices=(2, 4, 6),
            frames={2: first, 4: unrelated, 6: second},
        )

        with patch(
            "crosscut.reference_match.decode_target_frames_sequential",
            return_value=decoded,
        ) as decode:
            matches = match_aligned_target_frames(
                Path("source.mp4"),
                ((2, 4), (4, 6)),
                references,
                expected_source_times=(0.10, 0.20),
                matcher=matcher,
            )

        decode.assert_called_once()
        self.assertIsNotNone(matches[0])
        self.assertIsNotNone(matches[1])
        assert matches[0] is not None and matches[1] is not None
        self.assertEqual(matches[0].source_frame_index, 2)
        self.assertAlmostEqual(matches[0].source_time_seconds, 0.10)
        self.assertAlmostEqual(matches[0].time_adjustment_seconds or 0.0, 0.0)
        self.assertEqual(matches[1].source_frame_index, 6)
        self.assertAlmostEqual(matches[1].source_time_seconds, 0.30)
        self.assertAlmostEqual(matches[1].time_adjustment_seconds or 0.0, 0.10)
        self.assertGreater(matches[0].score, 0.90)
        self.assertGreater(matches[1].score, 0.90)
        json.dumps(matches[1].to_dict())

    def test_missing_decodes_remain_unknown_instead_of_forcing_a_match(self) -> None:
        matcher = OfflineReferenceMatcher()
        reference = matcher.extract(textured_frame(8, "IVE"))
        decoded = DecodedTargetFrames(
            frame_rate=30.0,
            declared_frame_count=10,
            requested_indices=(7,),
            frames={},
        )
        with patch(
            "crosscut.reference_match.decode_target_frames_sequential",
            return_value=decoded,
        ):
            result = match_aligned_target_frames(
                Path("source.mp4"),
                ((7,),),
                (reference,),
                matcher=matcher,
            )

        self.assertEqual(result, (None,))

    def test_conservative_acceptance_policy_preserves_unknown_reasons(self) -> None:
        def match(score: float, inliers: int, ratio: float) -> FrameMatch:
            return FrameMatch(
                score=score,
                orb_good_matches=max(inliers, 20),
                orb_homography_inliers=inliers,
                orb_inlier_ratio=ratio,
                structure_ncc=0.8,
                color_similarity=0.8,
            )

        self.assertEqual(decide_reference_match(None).reason, "no_candidate")
        self.assertEqual(
            decide_reference_match(match(0.39, 30, 0.8)).reason,
            "combined_score_below_0_40",
        )
        self.assertEqual(
            decide_reference_match(match(0.60, 9, 0.8)).reason,
            "fewer_than_10_homography_inliers",
        )
        self.assertEqual(
            decide_reference_match(match(0.60, 20, 0.24)).reason,
            "homography_inlier_ratio_below_0_25",
        )
        ambiguous = decide_reference_match(
            match(0.60, 20, 0.8),
            runner_up_score=0.57,
        )
        self.assertFalse(ambiguous.accepted)
        self.assertEqual(ambiguous.reason, "top_two_margin_below_0_055")
        accepted = decide_reference_match(
            match(0.72, 20, 0.8),
            runner_up_score=0.70,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.reason, "conservative_orb_homography_match")

    def test_lengths_and_conflicting_matcher_config_are_validated(self) -> None:
        matcher = OfflineReferenceMatcher()
        reference = matcher.extract(textured_frame(3, "IVE"))
        with self.assertRaisesRegex(ValueError, "feature count"):
            match_aligned_target_frames(Path("x"), (), (reference,))
        with self.assertRaisesRegex(ValueError, "source times"):
            match_aligned_target_frames(
                Path("x"),
                ((1,),),
                (reference,),
                expected_source_times=(),
            )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            match_aligned_target_frames(
                Path("x"),
                ((1,),),
                (reference,),
                expected_source_times=(-0.1,),
            )
        with self.assertRaisesRegex(ValueError, "different policies"):
            match_aligned_target_frames(
                Path("x"),
                ((1,),),
                (reference,),
                config=ReferenceMatchConfig(working_width=320),
                matcher=matcher,
            )


class ReferenceMatchConfigValidationTests(unittest.TestCase):
    def test_invalid_calibration_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceMatchConfig(ratio_test_threshold=1.0)
        with self.assertRaises(ValueError):
            ReferenceMatchConfig(ransac_reprojection_threshold=0.0)
        with self.assertRaises(ValueError):
            ReferenceMatchConfig(inlier_weight=0.40)
        with self.assertRaises(ValueError):
            ReferenceMatchConfig(temporal_jitter_frames=(-2, 0, 0, 2))


if __name__ == "__main__":
    unittest.main()
