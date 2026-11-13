from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.timeline import ShotType, TimelineChoice, TimelineMode, TimelineResult
from crosscut.visual_score import (
    FrameVisualDescriptor, VisualTimelinePlan,
    build_visual_timeline, build_visual_timeline_candidate_set,
    build_visual_timeline_optimizer_seed, build_prescribed_visual_timeline_candidate_set,
)


DESCRIPTOR = FrameVisualDescriptor(
    quality_score=.8, composition_score=.7, shot_scale=.4,
    shot_type=ShotType.MEDIUM, face_count=0, structure_embedding=(0.,)*96,
)


class Sampler:
    instances = []
    unreadable_source = None

    def __init__(self, path, **_kwargs):
        self.source = int(path.stem)
        self.times = []
        self.identity_times = []
        self.closed = False
        self.instances.append(self)

    def feature_descriptor_at(self, seconds, **_kwargs):
        self.times.append(seconds)
        return None if self.source == self.unreadable_source else DESCRIPTOR

    def sface_embedding_at(self, seconds, _descriptor):
        self.identity_times.append(seconds)
        return ()

    def close(self):
        self.closed = True


def select_first(workspace, **_kwargs):
    # Builder/sensing fixture only, not a simulated visual-safety pass.
    choices = tuple(TimelineChoice(cell[0], 1., 0., float(i+1), False)
        for i, cell in enumerate(workspace.candidates_by_segment))
    return (TimelineResult(TimelineMode.GENERAL, None, choices, float(len(choices))),), ('balanced',), []


class SparseSourceLatticeTests(unittest.TestCase):
    def setUp(self):
        Sampler.instances = []
        Sampler.unreadable_source = None
        self.arguments = dict(
            sources=tuple(Path(f'{i}.mp4') for i in range(3)),
            canonical_spans=((1., 1.1), (2., 2.1), (3., 3.1)),
            audio_offsets=(10., 20., 30.), sync_scores=(.9, .8, .7),
            scene_cut_times=((), (), ()),
        )
        self.options = ((0,), (0, 2), (2,))

    def test_restriction_is_before_sensing_and_preserves_allowed_candidates(self):
        with patch('crosscut.visual_score._VideoSampler', Sampler):
            full = build_visual_timeline_optimizer_seed(**self.arguments)
            first_count = len(Sampler.instances)
            sparse = build_visual_timeline_optimizer_seed(**self.arguments,
                candidate_source_indices_by_segment=self.options)
        self.assertEqual(tuple(tuple(int(c.source_id) for c in cell)
            for cell in sparse.optimizer_workspace.candidates_by_segment), self.options)
        for cell, options, source_cell in zip(sparse.optimizer_workspace.candidates_by_segment,
                self.options, full.optimizer_workspace.candidates_by_segment, strict=True):
            self.assertEqual(cell, tuple(c for c in source_cell if int(c.source_id) in options))
        measured = Sampler.instances[first_count:]
        self.assertEqual([len(s.times) for s in measured], [10, 0, 10])
        self.assertTrue(all(s.closed for s in Sampler.instances))
        self.assertNotEqual(sparse.optimizer_input_identity, full.optimizer_input_identity)

    def test_all_public_modes_forward_restriction_before_descriptor_and_identity(self):
        for builder in (build_visual_timeline, build_visual_timeline_candidate_set,
                build_visual_timeline_optimizer_seed):
            Sampler.instances = []
            kwargs = {'maximum_results': 1} if builder is build_visual_timeline else {}
            with self.subTest(builder=builder.__name__), patch('crosscut.visual_score._VideoSampler', Sampler), patch(
                'crosscut.visual_score._select_optimizer_workspace_candidates', side_effect=select_first), patch(
                'crosscut.visual_score.materialize_visual_timeline_candidate', return_value=VisualTimelinePlan((), 0.)):
                builder(**self.arguments, candidate_source_indices_by_segment=self.options, **kwargs)
            self.assertEqual([len(s.times) for s in Sampler.instances], [10, 0, 10])
            self.assertFalse(Sampler.instances[1].identity_times)
            self.assertTrue(all(s.closed for s in Sampler.instances))

    def test_prescribed_path_also_avoids_discarded_source_analysis(self):
        with patch('crosscut.visual_score._VideoSampler', Sampler), patch(
                'crosscut.visual_score._select_optimizer_workspace_candidates', side_effect=select_first):
            result = build_prescribed_visual_timeline_candidate_set(**self.arguments,
                prescribed_source_indices=(0, 2, 2))
        self.assertEqual([len(s.times) for s in Sampler.instances], [5, 0, 10])
        self.assertEqual(tuple(tuple(c.source_id for c in cell)
            for cell in result.optimizer_workspace.candidates_by_segment), (('0',), ('2',), ('2',)))

    def test_invalid_sparse_input_fails_before_opening_media(self):
        bad_values = ((), ((0,),), ((0,), (), (1,)), ((0,), (0, 0), (1,)),
            ((0,), (True,), (1,)), ((0,), (3,), (1,)), ((0,), (-1,), (1,)),
            ((0,), ('1',), (1,)))
        with patch('crosscut.visual_score._create_video_samplers') as samplers:
            for value in bad_values:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    build_visual_timeline_optimizer_seed(**self.arguments,
                        candidate_source_indices_by_segment=value)
        samplers.assert_not_called()

    def test_unreadable_fixed_cell_fails_without_expansion_or_fabrication(self):
        Sampler.unreadable_source = 0
        with patch('crosscut.visual_score._VideoSampler', Sampler), self.assertRaisesRegex(
                ValueError, 'no source frame is readable for segment 0'):
            build_visual_timeline_optimizer_seed(**self.arguments,
                candidate_source_indices_by_segment=self.options)
        self.assertEqual([len(s.times) for s in Sampler.instances], [5, 0, 0])
        self.assertTrue(all(s.closed for s in Sampler.instances))

    def test_source_options_are_canonical_and_explicit_identity_context(self):
        with patch('crosscut.visual_score._VideoSampler', Sampler):
            a = build_visual_timeline_optimizer_seed(**self.arguments,
                candidate_source_indices_by_segment=self.options)
            b = build_visual_timeline_optimizer_seed(**self.arguments,
                candidate_source_indices_by_segment=((0,), (2, 0), (2,)))
            all_explicit = build_visual_timeline_optimizer_seed(**self.arguments,
                candidate_source_indices_by_segment=((0, 1, 2),)*3)
            all_implicit = build_visual_timeline_optimizer_seed(**self.arguments)
        self.assertEqual(a.optimizer_input_identity, b.optimizer_input_identity)
        self.assertEqual(all_explicit.optimizer_workspace.candidates_by_segment,
            all_implicit.optimizer_workspace.candidates_by_segment)
        self.assertNotEqual(all_explicit.optimizer_input_identity, all_implicit.optimizer_input_identity)


if __name__ == '__main__':
    unittest.main()
