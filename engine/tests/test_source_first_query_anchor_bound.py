from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.source_first_anchor_prefilter import install_source_first_anchor_prefilter
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.source_first_observation_batch import SourceFirstObservationBatch
from crosscut.source_first_unresolved_memo import install_source_first_unresolved_memo
from tests import test_source_first_anchor_prefilter as anchor_fixtures


def fixture(*, optimized, face_times=(), reference_score=.8):
    provider = anchor_fixtures.SourceFirstAnchorPrefilterTests().provider(
        face_times=face_times, reference_score=reference_score)
    provider._reasons = {}
    provider._persistent_key = lambda source, key: str(key)
    provider.close = Mock()
    provider.diagnostics = lambda: {}
    provider.faces._samplers[0].frame_at = Mock(return_value=None)
    provider.selector_geometry = Mock(side_effect=lambda *args:
        SourceFirstFocalGeometryProvider.selector_geometry(provider, *args))
    if optimized:
        completed = dict(provider._source_first_unresolved_memo.completed_face_frames)
        install_source_first_unresolved_memo(provider)
        provider._source_first_unresolved_memo.completed_face_frames.update(completed)
        install_source_first_anchor_prefilter(provider)
    return provider


class SourceFirstQueryAnchorBoundTests(unittest.TestCase):
    def test_complete_zero_or_one_nonquery_anchor_matches_actual_scalar_resolver(self):
        for anchors in ((), (9.6,)):
            baseline = fixture(optimized=False, face_times=anchors)
            expected = baseline.selector_geometry(0, 10.)
            provider = fixture(optimized=True, face_times=anchors)
            provider.bodies._frame_cache.clear()  # Body work has not happened.
            original = provider.selector_geometry.original
            batch = SourceFirstObservationBatch(provider)
            batch(((0, 10.),))
            self.assertEqual(provider.selector_geometry(0, 10.), expected)
            self.assertEqual(batch.proven_missing_query_body_skip_count, 1)
            provider.bodies.frame_observations.assert_not_called()
            original.assert_not_called()
            memo = provider._source_first_unresolved_memo
            self.assertEqual(memo.anchor_bound_records, 1)
            provider.faces._focal_candidate_cache.clear()
            self.assertEqual(provider.selector_geometry(0, 10.), expected)
            self.assertEqual(memo.hits, 1)
            self.assertFalse(memo.diagnostics()['negative_observations_persisted'])

    def test_one_exact_query_face_keeps_direct_body_pose_authority(self):
        baseline = fixture(optimized=False, face_times=(10.,))
        expected = baseline.selector_geometry(0, 10.)
        self.assertIsNotNone(expected.face_bbox_xywh)
        provider = fixture(optimized=True, face_times=(10.,))
        self.assertFalse(provider._source_first_anchor_prefilter.query_cannot_resolve(0, 10.))
        SourceFirstObservationBatch(provider)(((0, 10.),))
        self.assertEqual(provider.selector_geometry(0, 10.), expected)
        self.assertGreater(provider.bodies.frame_observations.call_count, 0)
        self.assertEqual(provider._source_first_unresolved_memo.anchor_bound_records, 0)

    def test_two_anchors_keep_body_recovery_authority(self):
        provider = fixture(optimized=True, face_times=(9.6, 10.4))
        self.assertFalse(provider._source_first_anchor_prefilter.query_cannot_resolve(0, 10.))
        actual = provider.selector_geometry(0, 10.)
        self.assertIsNotNone(actual.person)
        self.assertEqual(provider._source_first_unresolved_memo.anchor_bound_records, 0)

    def test_body_only_disproof_cannot_be_memoized_as_native_face_absence(self):
        provider = fixture(optimized=True, face_times=(9.6, 10.4), reference_score=.3)
        for seconds in (9.6, 10.4):
            provider.bodies.frame_observations(0, seconds)
        prefilter = provider._source_first_anchor_prefilter
        self.assertTrue(prefilter.cannot_resolve(0, 10.))
        self.assertFalse(prefilter.query_cannot_resolve(0, 10.))

    def test_partial_nonempty_native_candidate_set_is_not_complete_absence_proof(self):
        provider = fixture(optimized=True, face_times=(9.6,))
        key = provider.faces._identity_key(0, provider.faces._samplers[0], 9.6)
        provider._source_first_unresolved_memo.completed_face_frames.pop(key)
        self.assertFalse(provider._source_first_anchor_prefilter.query_cannot_resolve(0, 10.))

    def test_unobserved_native_query_and_unverified_empty_decode_remain_unknown(self):
        for missing in ('faces', 'empty_decode', 'bank', 'window'):
            provider = fixture(optimized=True)
            memo = provider._source_first_unresolved_memo
            key = provider.faces._identity_key(0, provider.faces._samplers[0], 10.)
            if missing == 'faces':
                provider.faces._focal_candidate_cache.pop(key)
            elif missing == 'empty_decode':
                memo.completed_face_frames.pop(key)
            elif missing == 'bank':
                provider.faces._frozen_focal_banks.clear()
            else:
                key = provider.faces._identity_key(0, provider.faces._samplers[0], 9.6)
                provider.faces._focal_candidate_cache.pop(key)
            with self.subTest(missing=missing):
                self.assertFalse(provider._source_first_anchor_prefilter.query_cannot_resolve(0, 10.))

    def test_existing_and_concurrently_published_positive_outrank_bound_negative(self):
        positive = FocalSelectorGeometry(0, 10., (.1, .1, .1, .1))
        for channel in ('memory', 'persistent', 'persistent_after_memo'):
            provider = fixture(optimized=True)
            key = provider.faces._identity_key(0, provider.faces._samplers[0], 10.)
            if channel == 'memory':
                provider._cache[key] = positive
            else:
                if channel == 'persistent_after_memo':
                    self.assertIsNone(provider.selector_geometry(0, 10.).person)
                provider._persistent = SimpleNamespace(get=Mock(return_value=positive), put=Mock())
            with self.subTest(channel=channel):
                self.assertEqual(provider.selector_geometry(0, 10.), positive)
                if channel != 'memory':
                    provider._persistent.put.assert_not_called()

    def test_namespace_change_disconnection_and_capacity_do_not_become_proof(self):
        provider = fixture(optimized=True)
        provider.observation_namespace = 'different-model'
        self.assertFalse(provider._source_first_anchor_prefilter.query_cannot_resolve(0, 10.))
        provider = fixture(optimized=True)
        provider.shared_source_frames[0].maximum_working_frames = 1
        # Existing completed scalar face evidence does not need cached pixels.
        SourceFirstObservationBatch(provider)(((0, 10.),))
        provider.bodies.frame_observations.assert_not_called()
        provider._identity = lambda _: 'disconnected'
        with self.assertRaises(OSError):
            provider.selector_geometry(0, 10.)

    def test_published_positive_does_not_resurrect_an_old_scalar_negative_next_call(self):
        provider = fixture(optimized=True)
        missing = provider.selector_geometry(0, 10.)
        key = provider.faces._identity_key(0, provider.faces._samplers[0], 10.)
        provider._cache[key] = missing  # A prior ordinary scalar miss.
        positive = FocalSelectorGeometry(0, 10., (.1, .1, .1, .1))
        provider._persistent = SimpleNamespace(get=Mock(return_value=positive), put=Mock())
        self.assertEqual(provider.selector_geometry(0, 10.), positive)
        self.assertNotIn(key, provider._source_first_unresolved_memo.entries)
        self.assertEqual(provider.selector_geometry(0, 10.), positive)
        self.assertEqual(provider.selector_geometry(0, 10.), positive)


if __name__ == '__main__':
    unittest.main()
