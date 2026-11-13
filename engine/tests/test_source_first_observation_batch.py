from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.focal_body_track_provider import SourceLocalFocalBodyProvider
from crosscut.focal_body_track import FocalBodyFrame
from crosscut.source_first_observation_batch import SourceFirstObservationBatch, track_dependency_times
from engine.tests.test_focal_body_track_provider import FACE, observation


def sampler(fps=30., count=3000):
    return SimpleNamespace(frame_rate=fps, capture=SimpleNamespace(get=lambda _: count),
        frame_identity=lambda t: SimpleNamespace(frame_index=int(t*fps+.5)))


class SourceFirstObservationBatchTests(unittest.TestCase):
    def provider(self, *, references=(1, 2, 3)):
        samples = (sampler(), sampler())
        key = lambda source, sample, t: (source, sample.frame_identity(t).frame_index, sample.frame_rate)
        faces = SimpleNamespace(_samplers=samples, _identity_key=key,
            _ensure_focal_consensus=Mock(return_value=None), _focal_candidates=Mock(return_value=()))
        bodies = SimpleNamespace(_ensure_references=Mock(return_value=references),
            frame_observations=Mock(side_effect=lambda s, t: FocalBodyFrame(s, t, ())))
        return SimpleNamespace(sources=(Path('a'), Path('b')), _identity=lambda _: 'fixed',
            _identities=('fixed', 'fixed'), faces=faces, bodies=bodies,
            _cache=OrderedDict(), _persistent=None, _persistent_key=lambda source, key: str(key))

    def test_lattice_is_identical_to_actual_tracker_for_edges_and_fractional_rates(self):
        for fps in (23.976, 29.97, 30., 59.94, 60.):
            for time in (0., .03, 3.556645, 49.995):
                sample = sampler(fps, 1500)
                body = SourceLocalFocalBodyProvider.__new__(SourceLocalFocalBodyProvider)
                body.face_provider = SimpleNamespace(_sources=(Path('a'),), _samplers=(sample,),
                    _identity_key=lambda source, sample, t: (source, sample.frame_identity(t).frame_index))
                body._track_cache = OrderedDict()
                body.frame_observations = Mock(side_effect=lambda s, t: t)
                with patch('crosscut.focal_body_track_provider.resolve_focal_body_track', return_value=object()) as resolve:
                    body.track_at(0, time)
                self.assertEqual(track_dependency_times(sample, time), resolve.call_args.args[0])

    def test_source_sorted_unique_exact_dependencies_without_resolving_a_target(self):
        provider = self.provider()
        provider.selector_geometry = Mock(side_effect=AssertionError('batch cannot select a target'))
        batch = SourceFirstObservationBatch(provider)
        queries = ((1, 3.55), (0, 3.6), (1, 3.53), (0, 3.5), (0, 3.5))
        batch(queries)
        calls = [call.args for call in provider.bodies.frame_observations.call_args_list]
        # Each source's exact queries precede its sorted recovery lattice.
        self.assertEqual([s for s, _ in calls], sorted(s for s, _ in calls))
        for source in (0, 1):
            actual = [t for s, t in calls if s == source]
            exact = sorted({t for s, t in queries if s == source})
            self.assertEqual(actual[:len(exact)], exact)
            self.assertEqual(actual[len(exact):], sorted(actual[len(exact):]))
        keys = [provider.faces._identity_key(s, provider.faces._samplers[s], t) for s, t in calls]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(provider.faces._focal_candidates.call_args_list,
                         provider.bodies.frame_observations.call_args_list)
        for source, time in queries:
            for dependency in track_dependency_times(provider.faces._samplers[source], time):
                self.assertIn(provider.faces._identity_key(source, provider.faces._samplers[source], dependency), keys)
        provider.selector_geometry.assert_not_called()
        self.assertFalse(batch.diagnostics()['query_offsets_removed'])

    def test_exact_face_pose_query_skips_only_unused_track_dependencies(self):
        provider = self.provider()
        provider.faces._ensure_focal_consensus.return_value = SimpleNamespace(select=lambda _: FACE)
        provider.faces._focal_candidates.return_value = (FACE,)
        provider.bodies.frame_observations.side_effect = lambda s, t: FocalBodyFrame(s, t,
            (replace(observation(), source_index=s, source_seconds=t),))
        batch = SourceFirstObservationBatch(provider)
        batch(((0, 3.5), (0, 3.6), (0, 3.5)))
        self.assertEqual([call.args for call in provider.bodies.frame_observations.call_args_list],
                         [(0, 3.5), (0, 3.6)])
        self.assertEqual(batch.direct_resolved_query_count, 2)
        self.assertEqual(batch.recovery_query_count, 0)
        self.assertEqual(batch.dependency_frame_count, 2)

    def test_unproved_pose_still_prefetches_unchanged_recovery_lattice(self):
        provider = self.provider()
        provider.faces._ensure_focal_consensus.return_value = SimpleNamespace(select=lambda _: FACE)
        provider.faces._focal_candidates.return_value = (FACE,)
        provider.bodies.frame_observations.side_effect = lambda s, t: FocalBodyFrame(s, t,
            (replace(observation(consistent=None), source_index=s, source_seconds=t),))
        batch = SourceFirstObservationBatch(provider)
        batch(((0, 3.5),))
        actual = {provider.faces._identity_key(s, provider.faces._samplers[s], t)
                  for s, t in (call.args for call in provider.bodies.frame_observations.call_args_list)}
        expected = {provider.faces._identity_key(0, provider.faces._samplers[0], t)
                    for t in track_dependency_times(provider.faces._samplers[0], 3.5)}
        self.assertEqual(actual, expected)
        self.assertEqual(batch.direct_resolved_query_count, 0)
        self.assertEqual(batch.recovery_query_count, 1)

    def test_scalar_cache_hit_does_not_create_unused_model_work(self):
        provider = self.provider()
        key = provider.faces._identity_key(0, provider.faces._samplers[0], 3.5)
        provider._cache[key] = object()
        batch = SourceFirstObservationBatch(provider)
        batch(((0, 3.5),))
        provider.faces._ensure_focal_consensus.assert_not_called()
        provider.bodies.frame_observations.assert_not_called()
        self.assertEqual(batch.cached_geometry_count, 1)

    def test_published_scalar_is_reused_without_new_inference_or_write(self):
        provider = self.provider()
        value = object()
        provider._persistent = SimpleNamespace(get=Mock(return_value=value), put=Mock())
        SourceFirstObservationBatch(provider)(((0, 3.5),))
        self.assertEqual(tuple(provider._cache.values()), (value,))
        provider.bodies.frame_observations.assert_not_called()
        provider._persistent.put.assert_not_called()

    def test_progress_is_observation_work_not_geometry_or_render_approval(self):
        provider = self.provider()
        callback = Mock()
        batch = SourceFirstObservationBatch(provider, progress_callback=callback)
        batch(((0, 3.5),))
        payload = callback.call_args.args[0]
        self.assertEqual(payload['phase'], 'focal_observation_batch')
        self.assertEqual(payload['source_index'], 0)
        self.assertGreater(payload['dependency_frame_count'], 0)
        self.assertNotIn('quality_approved', payload)

    def test_absent_body_references_do_not_force_recovery_dependencies(self):
        provider = self.provider(references=(1, 2))
        SourceFirstObservationBatch(provider)(((0, 3.5), (0, 3.6)))
        self.assertEqual([call.args for call in provider.bodies.frame_observations.call_args_list],
                         [(0, 3.5), (0, 3.6)])

    def test_invalid_inputs_disconnection_and_failures_propagate(self):
        for query in ((True, 1.), (0, float('nan')), (0, -1.), (2, 1.), (0, True)):
            provider = self.provider()
            with self.assertRaises(ValueError): SourceFirstObservationBatch(provider)((query,))
            provider.bodies.frame_observations.assert_not_called()
        provider = self.provider()
        provider._identity = lambda _: 'changed'
        with self.assertRaises(OSError): SourceFirstObservationBatch(provider)(((0, 3.),))
        provider.bodies.frame_observations.assert_not_called()
        provider = self.provider()
        provider.bodies.frame_observations.side_effect = OSError('SSD')
        with self.assertRaises(OSError): SourceFirstObservationBatch(provider)(((0, 3.),))

    def test_large_or_nonlocal_batch_retains_scalar_fallback_without_observations(self):
        for queries in (((0, 1.),)*1025, ((0, 1.), (0, 4.))):
            provider = self.provider()
            batch = SourceFirstObservationBatch(provider)
            batch(queries)
            self.assertEqual(batch.capacity_fallback_count, 1)
            provider.bodies.frame_observations.assert_not_called()


if __name__ == '__main__': unittest.main()
