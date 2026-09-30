from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.render import RenderSegment, EditorialTransformTrack, EditorialTransformKeyframe
from crosscut.visual_feature_cache import CanonicalFrameIdentity
from crosscut.visual_score import build_visual_timeline_optimizer_seed
from engine.tests.test_sparse_source_lattice import DESCRIPTOR, Sampler


class PixelSampler(Sampler):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.path = path

    def frame_at(self, seconds):
        return np.zeros((90, 160, 3), np.uint8)

    def frame_identity(self, seconds):
        return CanonicalFrameIdentity.from_seconds(seconds, frame_rate=30.)


class GroupIncumbentGeometryTests(unittest.TestCase):
    def setUp(self):
        PixelSampler.instances = []
        self.segment = RenderSegment(0, 11., 1., 0., editorial_transform=EditorialTransformTrack((
            EditorialTransformKeyframe(0., 1.1), EditorialTransformKeyframe(1., 1.1))))
        self.arguments = dict(sources=(Path('0.mp4'), Path('1.mp4')),
            canonical_spans=((1., 2.),), audio_offsets=(10., 20.),
            sync_scores=(.9, .9), scene_cut_times=((), ()))

    def observe(self, segment=None):
        with patch('crosscut.visual_score._VideoSampler', PixelSampler), patch(
                'crosscut.visual_score.describe_frame', return_value=replace(DESCRIPTOR, quality_score=.95)):
            return build_visual_timeline_optimizer_seed(**self.arguments,
                group_incumbent_render_segments=(segment or self.segment,))

    def test_only_incumbent_source_uses_its_camera_and_alternative_is_not_borrowed(self):
        seed = self.observe()
        cells = seed.optimizer_workspace.candidates_by_segment
        for candidate, score in zip(cells[0], (.95, .8), strict=True):
            self.assertAlmostEqual(candidate.quality_score, score)
        # Output observations never overwrite source features or classify an
        # alternative through a camera authored for another broadcaster.
        self.assertEqual([len(s.times) for s in PixelSampler.instances], [0, 5])
        self.assertTrue(all(s.closed for s in PixelSampler.instances))
        with patch('crosscut.visual_score._VideoSampler', PixelSampler):
            raw = build_visual_timeline_optimizer_seed(**self.arguments)
        for candidate in raw.optimizer_workspace.candidates_by_segment[0]:
            self.assertAlmostEqual(candidate.quality_score, .8)

    def test_camera_identity_changes_even_if_candidate_features_happen_to_match(self):
        a = self.observe()
        b = self.observe(replace(self.segment, editorial_transform=EditorialTransformTrack((
            EditorialTransformKeyframe(0., 1.2), EditorialTransformKeyframe(1., 1.2)))))
        self.assertEqual(a.optimizer_workspace.candidates_by_segment, b.optimizer_workspace.candidates_by_segment)
        self.assertNotEqual(a.optimizer_input_identity, b.optimizer_input_identity)

    def test_stale_source_or_master_clock_fails_before_sensing(self):
        with patch('crosscut.visual_score._create_video_samplers') as opened:
            for segment in (replace(self.segment, source_start=12.),
                    replace(self.segment, master_start=.1), replace(self.segment, duration=.9, editorial_transform=None),
                    replace(self.segment, source_index=2)):
                with self.subTest(segment=segment), self.assertRaises(ValueError):
                    build_visual_timeline_optimizer_seed(**self.arguments, group_incumbent_render_segments=(segment,))
        opened.assert_not_called()

    def test_neutral_incumbent_retains_source_feature_cache(self):
        seed = self.observe(replace(self.segment, editorial_transform=None))
        for candidate in seed.optimizer_workspace.candidates_by_segment[0]:
            self.assertAlmostEqual(candidate.quality_score, .8)
        self.assertEqual([len(s.times) for s in PixelSampler.instances], [5, 5])

    def test_projected_sampler_keeps_native_identity_but_binds_arm_observation_geometry(self):
        from crosscut.prescribed_render_geometry import PrescribedGeometrySampler
        base = PixelSampler(Path('0.mp4'))
        a = PrescribedGeometrySampler(base, self.segment, master_start=1.)
        b = PrescribedGeometrySampler(base, replace(self.segment, editorial_transform=None), master_start=1.)
        self.assertEqual(a.path, base.path)
        self.assertEqual(a.frame_identity(11.2), base.frame_identity(11.2))
        self.assertNotEqual(a.observation_geometry_identity, b.observation_geometry_identity)


if __name__ == '__main__':
    unittest.main()
