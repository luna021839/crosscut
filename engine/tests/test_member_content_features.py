from dataclasses import fields, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _BoundarySample, _VideoDescriptorProvider,
    _attention_geometry, _span_source_validity,
    select_balanced_direct_member_sources,
)
from crosscut.pose import SubjectGeometryAnalyzer
from crosscut.visual_feature_cache import DescriptorRequestIdentity, DescriptorRequestLocator
from crosscut.visual_score import _VideoSampler
from engine.tests.test_visual_feature_cache import _FakeCapture, _analyzer, _descriptor, _request
from engine.tests.test_direct_member_visual import _descriptor as selector_descriptor
from engine.tests.test_source_first_focal_geometry import geometry


def without_joints(value):
    return replace(value, pose_embedding=value.structure_embedding, pose_confidence=0.,
        pose_backend='structure_hog_fallback',
        formation_people=tuple(replace(p, upper_body_keypoints_xyc=()) for p in value.formation_people))


class ContentFeatureTests(unittest.TestCase):
    def test_content_keeps_person_geometry_without_running_joint_model(self):
        analyzer = SubjectGeometryAnalyzer()
        analyzer._pose_load_attempted = analyzer._formation_load_attempted = analyzer._segmenter_load_attempted = True
        person = np.asarray([[20, 10, 90, 180, 55, 60, 55, 140, 30, 80, 80, 80, .95]])
        analyzer._pose_detector = Mock(infer=Mock(return_value=person))
        analyzer._formation_detector = Mock(infer=Mock(return_value=np.asarray([[20, 10, 90, 180, .95]])))
        analyzer._pose_estimator = Mock(infer=Mock(side_effect=AssertionError('unused joints')))
        result = analyzer.describe(np.zeros((200, 120, 3), dtype=np.uint8), include_pose=False)
        self.assertEqual(result.detected_person_count, 1)
        self.assertEqual(len(result.formation_people), 1)
        self.assertEqual(result.formation_people[0].upper_body_keypoints_xyc, ())
        self.assertEqual(result.pose_embedding, ())
        analyzer._pose_estimator.infer.assert_not_called()
        with self.assertRaisesRegex(AssertionError, 'unused joints'):
            analyzer.describe(np.zeros((200, 120, 3), dtype=np.uint8))

    def test_pose_option_is_typed_and_cannot_disable_requested_formation_pose(self):
        analyzer = SubjectGeometryAnalyzer()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for bad in (0, 1, None, 'false'):
            with self.assertRaises(TypeError):
                analyzer.describe(frame, include_pose=bad)
        with self.assertRaises(ValueError):
            analyzer.describe(frame, include_pose=False, include_formation_pose=True)

    def test_legacy_request_identity_is_stable_and_content_is_disjoint(self):
        full = _request()
        old_payload = full.to_dict()
        self.assertNotIn('include_pose', old_payload)
        self.assertEqual(DescriptorRequestIdentity.from_dict(old_payload), full)
        content = replace(full, include_pose=False)
        self.assertIs(content.to_dict()['include_pose'], False)
        self.assertEqual(DescriptorRequestIdentity.from_dict(content.to_dict()), content)
        self.assertNotEqual(full.locator, content.locator)
        self.assertNotEqual(full.to_dict(), content.to_dict())
        with self.assertRaises(ValueError):
            replace(content, include_formation_pose=True)
        for bad in (0, 1, None, 'false'):
            with self.assertRaises(TypeError):
                replace(full, include_pose=bad)
            with self.assertRaises(TypeError):
                replace(full.locator, include_pose=bad)

    def test_sampler_never_promotes_content_pixels_to_full_pose(self):
        full = _descriptor()
        content = without_joints(full)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.write_bytes(b'stable source')
            with patch('crosscut.visual_score.cv2.VideoCapture', side_effect=_FakeCapture), \
                 patch('crosscut.visual_score._visual_analyzer_fingerprint', return_value=_analyzer()), \
                 patch('crosscut.visual_score.probe_constant_frame_clock', return_value=None), \
                 patch('crosscut.visual_score.describe_frame', side_effect=[content, full]) as describe:
                sampler = _VideoSampler(source, feature_cache_directory=root/'cache')
                self.assertIsNone(sampler.feature_descriptor_at(1., cache_only=True))
                self.assertEqual(sampler.capture.read_count, 0)
                self.assertEqual(sampler.descriptor_locator_cache, {})
                self.assertEqual(sampler.feature_descriptor_at(1., include_pose=False), content)
                self.assertIsNone(sampler.feature_descriptor_at(1., cache_only=True))
                self.assertEqual(sampler.feature_descriptor_at(1.), full)
                self.assertEqual(describe.call_count, 2)
                self.assertIs(describe.call_args_list[0].kwargs['include_pose'], False)
                self.assertNotIn('include_pose', describe.call_args_list[1].kwargs)
                sampler.close()
            with patch('crosscut.visual_score.cv2.VideoCapture', side_effect=_FakeCapture), \
                 patch('crosscut.visual_score._visual_analyzer_fingerprint', return_value=_analyzer()), \
                 patch('crosscut.visual_score.describe_frame', side_effect=AssertionError('must be warm')):
                warm = _VideoSampler(source, feature_cache_directory=root/'cache')
                self.assertEqual(warm.feature_descriptor_at(1., include_pose=False), content)
                self.assertEqual(warm.feature_descriptor_at(1.), full)
                self.assertEqual(warm.capture.read_count, 0)
                warm.close()

    def test_provider_full_and_content_memory_caches_are_separate(self):
        full, content = _descriptor(), without_joints(_descriptor())
        sampler = SimpleNamespace(frame_identity=lambda value: value, close=Mock(),
            feature_descriptor_at=Mock(side_effect=[None, content, full]))
        with patch('crosscut.direct_member_visual._VideoSampler', return_value=sampler):
            provider = _VideoDescriptorProvider((Path('source'),))
            self.assertIs(provider.content_at(0, Path('source'), 1.), content)
            self.assertIs(provider(0, Path('source'), 1.), full)
            self.assertIs(provider.content_at(0, Path('source'), 1.), content)
            self.assertIs(provider(0, Path('source'), 1.), full)
            self.assertEqual(sampler.feature_descriptor_at.call_count, 3)
            self.assertTrue(sampler.feature_descriptor_at.call_args_list[0].kwargs['cache_only'])
            self.assertFalse(sampler.feature_descriptor_at.call_args_list[1].kwargs['include_pose'])
            self.assertTrue(sampler.feature_descriptor_at.call_args_list[2].kwargs['include_formation_pose'])
            provider.close()

    def test_content_can_reuse_full_cache_without_reinferring_or_relabeling(self):
        full = _descriptor()
        sampler = SimpleNamespace(frame_identity=lambda value: value, close=Mock(),
            feature_descriptor_at=Mock(return_value=full))
        with patch('crosscut.direct_member_visual._VideoSampler', return_value=sampler):
            provider = _VideoDescriptorProvider((Path('source'),))
            self.assertIs(provider.content_at(0, Path('source'), 1.), full)
            sampler.feature_descriptor_at.assert_called_once_with(1.,
                include_subject_geometry=True, include_formation_pose=True, cache_only=True)
            self.assertEqual(provider._content_full_cache_hit_count, 1)
            provider.close()

    def test_content_joints_cannot_affect_content_admission(self):
        for changes in ({}, {'foreground_obstruction_score': .95}, {'quality_score': 0.},
                        {'face_count': 0, 'face_detector_backend': 'none'},
                        {'color_embedding': (0.,)*8, 'face_count': 0, 'detected_person_count': 0,
                         'silhouette_bbox_xywh': None, 'formation_people': (), 'primary_face_bbox': None}):
            full = replace(_descriptor(), **changes)
            content = without_joints(full)
            decisions = []
            for value in (full, content):
                samples = tuple(_BoundarySample(master_seconds=float(i), source_local_seconds=float(i),
                    descriptor=value, attention=_attention_geometry(value)) for i in range(6))
                result = _span_source_validity(source_index=0, samples=samples,
                    native_scene_cuts=(), suppressed_portrait_fancam_scene_cuts=(),
                    continuous_official_portrait_fancam_contract=True, config=DirectMemberVisualConfig())
                decisions.append(tuple(getattr(result, f.name) for f in fields(result) if f.name != 'samples'))
            self.assertEqual(*decisions)

    def test_actual_selector_keeps_full_exact_cut_samples_and_path(self):
        def execute(split):
            content_calls, full_calls, focal_calls = [], [], []
            class Provider:
                def __call__(self, source, path, seconds):
                    full_calls.append((source, seconds))
                    return selector_descriptor(.2 + .02*source)
                def focal_geometry_at(self, source, seconds, raw):
                    focal_calls.append((source, seconds))
                    self_outer.assertEqual(raw.pose_confidence, 1.)
                    return replace(geometry(source), source_seconds=seconds)
            self_outer = self
            provider = Provider()
            if split:
                def content_at(source, path, seconds):
                    content_calls.append((source, seconds))
                    return without_joints(selector_descriptor(.2+.02*source))
                provider.content_at = content_at
            result = select_balanced_direct_member_sources(
                sources=tuple(Path(f'{i}.mp4') for i in range(3)),
                canonical_spans=tuple((float(i),float(i+1)) for i in range(6)),
                audio_offsets=(0.,)*3, descriptor_provider=provider,
                config=DirectMemberVisualConfig(beam_width=1000))
            return result, content_calls, full_calls, focal_calls
        prior, _, previous_full, previous_focal = execute(False)
        result, content, full, focal = execute(True)
        self.assertEqual(len(content), 6*3*6)
        self.assertEqual(len(previous_full)-len(full), len(content))
        self.assertEqual(focal, previous_focal)
        self.assertEqual(result.source_indices, prior.source_indices)
        self.assertEqual(result.selected_boundary_master_seconds, prior.selected_boundary_master_seconds)
        self.assertEqual(result.optimizer_cost, prior.optimizer_cost)
        self.assertEqual(result.boundaries, prior.boundaries)


if __name__ == '__main__':
    unittest.main()
