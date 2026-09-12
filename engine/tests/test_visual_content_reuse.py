from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor, _VideoSampler


def descriptor():
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.7,
        shot_scale=0.6,
        shot_type=ShotType.CLOSE_UP,
        face_count=1,
        structure_embedding=(0.2, 0.3),
    )


class VisualContentReuseTests(unittest.TestCase):
    def make_sampler(self):
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 29.97
        with mock.patch('crosscut.visual_score.cv2.VideoCapture', return_value=capture):
            sampler = _VideoSampler(Path('synthetic-source.mp4'))
        self.addCleanup(sampler.close)
        return sampler

    def test_distinct_exact_times_with_identical_pixels_share_inference(self):
        sampler = self.make_sampler()
        frame = np.full((48, 64, 3), 96, dtype=np.uint8)
        with mock.patch.object(sampler, 'frame_at', return_value=frame), mock.patch(
            'crosscut.visual_score.describe_frame', return_value=descriptor()
        ) as describe:
            first = sampler.feature_descriptor_at(1.0, include_formation_pose=True)
            second = sampler.feature_descriptor_at(1.001, include_formation_pose=True)
        self.assertIs(first, second)
        self.assertEqual(describe.call_count, 1)
        self.assertEqual(len(sampler.descriptor_locator_cache), 2)
        self.assertEqual(sampler.descriptor_content_cache_hit_count, 1)

    def test_changed_pixels_do_not_share_inference(self):
        sampler = self.make_sampler()
        frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (96, 97)]
        with mock.patch.object(sampler, 'frame_at', side_effect=frames), mock.patch(
            'crosscut.visual_score.describe_frame', side_effect=[descriptor(), replace(descriptor(), quality_score=0.9)]
        ) as describe:
            first = sampler.feature_descriptor_at(1.0)
            second = sampler.feature_descriptor_at(1.001)
        self.assertNotEqual(first, second)
        self.assertEqual(describe.call_count, 2)

    def test_following_pixels_and_absence_remain_distinct(self):
        sampler = self.make_sampler()
        primary = np.full((48, 64, 3), 96, dtype=np.uint8)
        following = np.full((48, 64, 3), 97, dtype=np.uint8)
        changed = np.full((48, 64, 3), 98, dtype=np.uint8)
        with mock.patch.object(sampler, 'frame_at', side_effect=[primary, primary, following, primary, changed, primary, following]), mock.patch(
            'crosscut.visual_score.describe_frame', return_value=descriptor()
        ) as describe:
            sampler.feature_descriptor_at(1.0)
            sampler.feature_descriptor_at(1.001, following_seconds=2.0)
            sampler.feature_descriptor_at(1.002, following_seconds=3.0)
            sampler.feature_descriptor_at(1.003, following_seconds=2.001)
        self.assertEqual(describe.call_count, 3)

    def test_analysis_options_do_not_alias(self):
        sampler = self.make_sampler()
        frame = np.full((48, 64, 3), 96, dtype=np.uint8)
        with mock.patch.object(sampler, 'frame_at', return_value=frame), mock.patch(
            'crosscut.visual_score.describe_frame', return_value=descriptor()
        ) as describe:
            for index, (subject, formation) in enumerate(((False, False), (True, False), (True, True))):
                sampler.feature_descriptor_at(1.0 + index / 1000, include_subject_geometry=subject, include_formation_pose=formation)
        self.assertEqual(describe.call_count, 3)

    def test_missing_following_frame_cannot_use_primary_content_hit(self):
        sampler = self.make_sampler()
        frame = np.full((48, 64, 3), 96, dtype=np.uint8)
        with mock.patch.object(sampler, 'frame_at', side_effect=[frame, frame, None]), mock.patch(
            'crosscut.visual_score.describe_frame', return_value=descriptor()
        ) as describe:
            self.assertIsNotNone(sampler.feature_descriptor_at(1.0))
            self.assertIsNone(sampler.feature_descriptor_at(1.001, following_seconds=2.0))
        self.assertEqual(describe.call_count, 1)

    def test_content_lru_is_bounded_and_close_clears_it(self):
        sampler = self.make_sampler()
        frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(130)]
        with mock.patch.object(sampler, 'frame_at', side_effect=frames), mock.patch(
            'crosscut.visual_score.describe_frame', return_value=descriptor()
        ):
            for index in range(130):
                sampler.feature_descriptor_at(float(index))
        self.assertEqual(len(sampler.descriptor_content_cache), 128)
        sampler.close()
        self.assertFalse(sampler.descriptor_content_cache)


if __name__ == '__main__':
    unittest.main()
