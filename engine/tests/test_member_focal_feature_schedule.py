from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.direct_member_visual import (
    DirectMemberFeatureHydrationError, DirectMemberVisualConfig, DirectMemberVisualSelectionError, _VideoDescriptorProvider,
    _hydrate_focal_boundary_sample, _same_nonjoint_features, select_balanced_direct_member_sources,
)
from engine.tests.test_member_content_features import without_joints
from engine.tests.test_direct_member_visual import _descriptor
from engine.tests.test_source_first_focal_geometry import geometry
from engine.tests.test_focal_selector_geometry import sample
from engine.tests.test_direct_member_transition import _plan


class FocalFeatureScheduleTests(unittest.TestCase):
    def test_proposal_endpoint_requires_focal_authority_and_keeps_full_endpoint(self):
        full = _descriptor(.2)
        sampler = SimpleNamespace(frame_identity=lambda time: time, close=Mock(),
            feature_descriptor_at=Mock(side_effect=[None, without_joints(full), full]))
        with patch('crosscut.direct_member_visual._VideoSampler', return_value=sampler):
            provider = _VideoDescriptorProvider((Path('source'),))
            with self.assertRaises(ValueError):
                provider.focal_proposal_at(0, Path('source'), 1.)
            provider.focal_geometry_at = lambda *args: geometry()
            self.assertEqual(provider.focal_proposal_at(0, Path('source'), 1.), without_joints(full))
            self.assertEqual(provider(0, Path('source'), 1.), full)
            self.assertEqual(provider._focal_proposal_request_count, 1)
            self.assertFalse(sampler.feature_descriptor_at.call_args_list[1].kwargs['include_pose'])
            self.assertTrue(sampler.feature_descriptor_at.call_args_list[2].kwargs['include_formation_pose'])
            provider.close()

    def selection(self, deferred, *, full_failure=False, missing_target=False, prepared=None):
        full_calls, proposal_calls, focal_calls = [], [], []
        class Provider:
            def __call__(self, source, path, seconds):
                full_calls.append((source, seconds))
                return None if full_failure else _descriptor(.2+.02*source)
            def content_at(self, source, path, seconds):
                return without_joints(_descriptor(.2+.02*source))
            def focal_geometry_at(self, source, seconds, raw):
                focal_calls.append((source, seconds))
                value = replace(geometry(source), source_seconds=seconds)
                return replace(value, face_bbox_xywh=None, face_scale=None, person=None) if missing_target else value
        provider = Provider()
        if prepared is not None:
            provider.prepare_focal_observations = lambda queries: prepared.extend(queries)
        if deferred:
            def proposal(source, path, seconds):
                proposal_calls.append((source, seconds))
                return without_joints(_descriptor(.2+.02*source))
            provider.focal_proposal_at = proposal
        result = select_balanced_direct_member_sources(
            sources=tuple(Path(f'{i}.mp4') for i in range(3)),
            canonical_spans=tuple((float(4*i),float(4*i+4)) for i in range(6)),
            audio_offsets=(0.,)*3, descriptor_provider=provider,
            config=DirectMemberVisualConfig(beam_width=1000, sample_frame_step_seconds=1/30,
                boundary_search_radius_frames=6, boundary_search_step_frames=2))
        return result, full_calls, proposal_calls, focal_calls

    def test_batch_plans_exactly_the_same_cut_queries_without_changing_selection(self):
        expected, full, proposals, focal = self.selection(True)
        prepared = []
        actual, batched_full, batched_proposals, batched_focal = self.selection(True, prepared=prepared)
        self.assertEqual(prepared, focal)
        self.assertEqual(actual, expected)
        self.assertEqual((batched_full, batched_proposals, batched_focal), (full, proposals, focal))

    def test_complete_selector_preserves_path_cost_boundaries_and_effect_decisions(self):
        old, old_full, _, old_focal = self.selection(False)
        new, new_full, proposals, new_focal = self.selection(True)
        self.assertEqual(new.source_indices, old.source_indices)
        self.assertEqual(new.selected_boundary_master_seconds, old.selected_boundary_master_seconds)
        self.assertEqual(new.optimizer_cost, old.optimizer_cost)
        self.assertEqual(new.span_validity, old.span_validity)
        self.assertEqual(new.boundaries, old.boundaries)
        self.assertEqual(new_focal, old_focal)
        self.assertEqual(proposals, old_full)
        self.assertLess(len(new_full), len(old_full)/10)
        self.assertLessEqual(len(new_full), 6*len(new.boundaries))
        for enabled in (False, True):
            self.assertEqual(_plan(new.source_indices, selection=new, enabled=enabled),
                             _plan(old.source_indices, selection=old, enabled=enabled))
        self.assertEqual(new.optimizer_search_diagnostics['focal_feature_schedule']
            ['retained_full_descriptor_query_count'], len(new_full))

    def test_missing_selected_full_features_cannot_escape_to_effect_planner(self):
        with self.assertRaises(DirectMemberFeatureHydrationError) as caught:
            self.selection(True, full_failure=True)
        self.assertEqual(caught.exception.diagnostics['failure'], 'selected_focal_descriptor_hydration_failed')
        self.assertTrue(caught.exception.diagnostics['full_descriptor_missing'])
        self.assertNotIsInstance(caught.exception, DirectMemberVisualSelectionError)
        self.assertFalse(caught.exception.diagnostics['retry_as_another_partition'])

    def test_missing_target_stays_missing_in_both_schedules(self):
        for deferred in (False, True):
            with self.assertRaises(DirectMemberVisualSelectionError):
                self.selection(deferred, missing_target=True)

    def test_hydration_is_exact_bound_cached_and_checks_nonjoint_drift(self):
        original = sample()
        partial = replace(original, descriptor=without_joints(original.descriptor))
        provider = Mock(return_value=original.descriptor)
        cache = {}
        for _ in range(2):
            result = _hydrate_focal_boundary_sample(partial, source_index=0,
                source=Path('source'), provider=provider, cache=cache)
            self.assertEqual(result, original)
        provider.assert_called_once_with(0, Path('source'), 1.)
        with self.assertRaises(ValueError):
            _hydrate_focal_boundary_sample(partial, source_index=1,
                source=Path('source'), provider=provider, cache={})
        with self.assertRaises(ValueError):
            _hydrate_focal_boundary_sample(replace(partial, source_local_seconds=2.), source_index=0,
                source=Path('source'), provider=provider, cache={})
        self.assertTrue(_same_nonjoint_features(partial.descriptor, original.descriptor))
        for change in ({'quality_score': .01}, {'primary_face_bbox': (.1,.1,.2,.2)},
                       {'silhouette_embedding': (.123,)*6}):
            with self.assertRaises(DirectMemberFeatureHydrationError):
                _hydrate_focal_boundary_sample(partial, source_index=0, source=Path('source'),
                    provider=Mock(return_value=replace(original.descriptor, **change)), cache={})


if __name__ == '__main__':
    unittest.main()
