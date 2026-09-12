from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from crosscut.source_first_member_session import open_source_first_member_session
from crosscut.direct_member_visual import select_balanced_direct_member_sources,DirectMemberVisualConfig
from engine.tests.test_focal_selector_geometry import descriptor


class SharedMemberSessionTests(unittest.TestCase):
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
            focal.return_value.close.assert_called_once()

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
