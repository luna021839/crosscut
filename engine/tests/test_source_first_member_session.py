from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from crosscut.source_first_member_session import open_source_first_member_session
from crosscut.direct_member_visual import select_balanced_direct_member_sources,DirectMemberVisualConfig,_VideoDescriptorProvider
from engine.tests.test_focal_selector_geometry import descriptor


class SharedMemberSessionTests(unittest.TestCase):
    def test_descriptor_progress_is_visible_before_any_focal_geometry_query(self):
        reader = SimpleNamespace(decode_seconds=20., process_seconds=1., read_count=100,
            materialized_frame_count=10, skipped_materialization_count=90)
        raw = descriptor()
        sampler = SimpleNamespace(_seek_locality=reader, describe_frame_call_count=1,
            frame_identity=lambda value: value, feature_descriptor_at=Mock(return_value=raw),
            close=Mock())
        progress = Mock()
        focal = Mock(side_effect=AssertionError('content progress must not trigger focal inference'))
        with patch('crosscut.direct_member_visual._VideoSampler', return_value=sampler), \
             patch('crosscut.direct_member_visual.monotonic', side_effect=(0., 1., 32., 33., 34.)):
            provider = _VideoDescriptorProvider((Path('source'),), focal_geometry_provider=focal,
                progress_callback=progress)
            self.assertIs(provider(0, Path('source'), 1.), raw)
            self.assertIs(provider(0, Path('source'), 1.), raw)
            self.assertEqual(provider._cache_hit_count, 1)
            progress.assert_called_once()
            report = progress.call_args.args[0]
            self.assertEqual(report['phase'], 'visual_descriptors')
            self.assertEqual(report['descriptor_seconds'], 31.)
            self.assertEqual(report['decode_seconds'], 20.)
            self.assertEqual(report['materialized_frame_count'], 10)
            focal.assert_not_called()
            provider.close()

    def test_selector_receives_same_authority_returned_to_dense_camera(self):
        with patch('crosscut.source_first_member_session.ensure_source_first_model_assets') as assets, \
             patch('crosscut.source_first_member_session.SourceFirstFocalGeometryProvider') as focal, \
             patch('crosscut.source_first_member_session._VideoDescriptorProvider') as visual:
            assets.return_value=SimpleNamespace(body_model_path=Path('body'),pose_model_path=Path('pose'))
            dense,selector=open_source_first_member_session((Path('source'),),feature_cache_directory=Path('external'))
            self.assertIs(dense,focal.return_value)
            self.assertIs(selector,visual.return_value)
            self.assertIs(visual.call_args.kwargs['focal_geometry_provider'],dense.selector_geometry)
            assets.assert_called_once_with(Path('external'))

    def test_visual_constructor_failure_closes_focal_capture_ownership(self):
        with patch('crosscut.source_first_member_session.ensure_source_first_model_assets') as assets, \
             patch('crosscut.source_first_member_session.SourceFirstFocalGeometryProvider') as focal, \
             patch('crosscut.source_first_member_session._VideoDescriptorProvider',side_effect=OSError('SSD')):
            assets.return_value=SimpleNamespace(body_model_path=Path('body'),pose_model_path=Path('pose'))
            with self.assertRaises(OSError):open_source_first_member_session((),feature_cache_directory=Path('external'))
            focal.return_value._source_first_unresolved_memo.original_close.assert_called_once()

    def test_missing_explicit_cache_never_creates_home_fallback(self):
        with patch('crosscut.source_first_member_session.ensure_source_first_model_assets') as assets:
            with self.assertRaises(ValueError):open_source_first_member_session(())
            assets.assert_not_called()

    def test_content_screening_never_spends_target_tracking_but_exact_cut_does(self):
        calls=[];focal_calls=[]
        class ReachedExactBoundary(Exception):pass
        class Provider:
            def __call__(self,source,path,seconds):
                calls.append((source,seconds));return descriptor()
            def focal_geometry_at(self,source,seconds,raw):
                focal_calls.append((source,seconds));raise ReachedExactBoundary()
        config=DirectMemberVisualConfig()
        with self.assertRaises(ReachedExactBoundary):
            select_balanced_direct_member_sources(sources=(Path('a'),Path('b')),
                canonical_spans=((0.,1.),(1.,2.)),audio_offsets=(0.,0.),
                config=config,descriptor_provider=Provider())
        self.assertEqual(len(calls),2*2*6+5)
        self.assertEqual(len(focal_calls),1)
        self.assertAlmostEqual(focal_calls[0][1],1.-2*config.sample_frame_step_seconds)


if __name__=='__main__':unittest.main()
