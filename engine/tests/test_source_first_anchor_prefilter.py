from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.focal_body_track import FocalBodyFrame, resolve_focal_body_track
from crosscut.source_first_anchor_prefilter import (
    SourceFirstAnchorPrefilter, install_source_first_anchor_prefilter)
from crosscut.source_first_observation_batch import SourceFirstObservationBatch, track_dependency_times
from engine.tests.test_focal_body_track_provider import FACE, observation
from engine.tests.test_source_first_observation_batch import sampler


class SourceFirstAnchorPrefilterTests(unittest.TestCase):
    def provider(self, *, face_times=(), query_empty=False, reference_score=.8):
        sample = sampler()
        key = lambda s, r, t: (s, r.frame_identity(t).frame_index, r.frame_rate)
        bank = SimpleNamespace(select=lambda faces: next(iter(faces), None))
        faces = SimpleNamespace(_samplers=(sample,), _identity_key=key,
            _frozen_focal_banks={0: bank}, _focal_candidate_cache={},
            _ensure_focal_consensus=Mock(return_value=bank))
        bodies = SimpleNamespace(_frame_cache=OrderedDict(), _track_cache=OrderedDict(),
            _ensure_references=Mock(return_value=(1, 2, 3)))
        memo = SimpleNamespace(namespace='measured-v1', completed_face_frames={})
        provider = SimpleNamespace(sources=(Path('source'),), faces=faces, bodies=bodies,
            _identity=lambda _: 'fixed', _identities=('fixed',), observation_namespace='measured-v1',
            _source_first_unresolved_memo=memo, _cache=OrderedDict(), _persistent=None,
            shared_source_frames=(SimpleNamespace(maximum_working_frames=128),))
        times = track_dependency_times(sample, 10.)
        actual = {}
        for t in times:
            frame_key = key(0, sample, t)
            is_anchor = t in face_times
            faces._focal_candidate_cache[frame_key] = (FACE,) if is_anchor else ()
            memo.completed_face_frames[frame_key] = True
            values = () if query_empty and t == 10. else (replace(observation(score=reference_score),
                source_seconds=t, frozen_face_match=is_anchor,
                face=FACE if is_anchor else None, face_pose_consistent=None),)
            actual[frame_key] = FocalBodyFrame(0, t, values)
        query_key = key(0, sample, 10.)
        bodies._frame_cache[query_key] = actual[query_key]
        def frame(s, t):
            k = key(s, sample, t)
            measured = bodies._frame_cache.setdefault(k, actual[k])
            return FocalBodyFrame(s, t, tuple(replace(o, source_seconds=t) for o in measured.observations))
        bodies.frame_observations = Mock(side_effect=frame)
        def full_track(s, t):
            if isinstance(s, bool) or isinstance(t, bool) or s != 0 or not 0 <= t < 100:
                raise ValueError('invalid track query')
            k = key(s, sample, t)
            if k in bodies._track_cache:
                return bodies._track_cache[k]
            return resolve_focal_body_track(tuple(frame(s, time) for time in times),
                source_index=s, query_seconds=t, maximum_window_seconds=10*4/30+.000001)
        bodies.track_at = Mock(side_effect=full_track)
        faces._focal_candidates = Mock(side_effect=lambda s, t: faces._focal_candidate_cache[key(s, sample, t)])
        return provider

    def test_zero_and_one_face_anchor_are_identical_to_complete_tracker(self):
        for anchor_times in ((), (9.6,), (10.,)):
            for empty in (False, True):
                with self.subTest(anchors=anchor_times, empty=empty):
                    baseline = self.provider(face_times=anchor_times, query_empty=empty)
                    expected = baseline.bodies.track_at(0, 10.)
                    provider = self.provider(face_times=anchor_times, query_empty=empty)
                    original = provider.bodies.track_at
                    prefilter = SourceFirstAnchorPrefilter(provider)
                    actual = prefilter.track_at(0, 10.)
                    self.assertEqual(actual, expected)
                    self.assertIsNone(actual.observation)
                    original.assert_not_called()
                    provider.bodies.frame_observations.assert_called_once_with(0, 10.)
                    self.assertEqual(prefilter.skipped_tracks, 1)

    def test_two_possible_anchors_never_skip_the_original_tracker(self):
        provider = self.provider(face_times=(9.6, 10.4))
        original = provider.bodies.track_at
        prefilter = SourceFirstAnchorPrefilter(provider)
        self.assertFalse(prefilter.cannot_resolve(0, 10.))
        result = prefilter.track_at(0, 10.)
        original.assert_called_once_with(0, 10.)
        self.assertEqual(prefilter.skipped_tracks, 0)
        self.assertIsNotNone(result.observation)

    def test_missing_empty_decode_proof_or_face_cache_remains_unknown(self):
        for mode in ('unverified_empty', 'missing_faces', 'bank', 'namespace', 'sampler'):
            provider = self.provider()
            k = provider.faces._identity_key(0, provider.faces._samplers[0], 9.6)
            if mode == 'unverified_empty':
                provider._source_first_unresolved_memo.completed_face_frames.pop(k)
            elif mode == 'missing_faces':
                provider.faces._focal_candidate_cache.pop(k)
            elif mode == 'bank':
                provider.faces._frozen_focal_banks.clear()
            elif mode == 'namespace':
                provider.observation_namespace = 'different-v2'
            else:
                provider.faces._samplers = (None,)
            with self.subTest(mode=mode):
                self.assertFalse(SourceFirstAnchorPrefilter(provider).cannot_resolve(0, 10.))

    def test_existing_body_anchors_survive_face_cache_eviction(self):
        provider = self.provider(face_times=(9.6, 10.4))
        for t in (9.6, 10.4):
            provider.bodies.frame_observations(0, t)
            key = provider.faces._identity_key(0, provider.faces._samplers[0], t)
            provider.faces._focal_candidate_cache.pop(key)
        self.assertFalse(SourceFirstAnchorPrefilter(provider).cannot_resolve(0, 10.))

    def test_cached_clothing_and_competitor_proof_matches_full_resolver(self):
        for score in (.3, .585, .60, .67, .679999, .68, .8):
            provider = self.provider(face_times=(9.6, 10.4), reference_score=score)
            for t in (9.6, 10.4):
                provider.bodies.frame_observations(0, t)
            prefilter = SourceFirstAnchorPrefilter(provider)
            self.assertEqual(prefilter.cannot_resolve(0, 10.), score < .68)
            full = provider.bodies.track_at(0, 10.)
            result = prefilter.track_at(0, 10.)
            self.assertEqual(result, full)
        for gap in (.0, .03, .059, .061, .15):
            provider = self.provider(face_times=(9.6, 10.4))
            for t in (9.6, 10.4):
                current = provider.bodies.frame_observations(0, t)
                target = current.observations[0]
                rival = replace(target, face=None, frozen_face_match=False,
                    reference_similarity=.8-gap, pose_screen_xyc=(), face_pose_consistent=None,
                    person=replace(target.person, bbox_xywh=(.7, .1, .2, .8), centroid_xy=(.8, .5)))
                key = provider.faces._identity_key(0, provider.faces._samplers[0], t)
                provider.bodies._frame_cache[key] = replace(current, observations=(target, rival))
            prefilter = SourceFirstAnchorPrefilter(provider)
            if prefilter.cannot_resolve(0, 10.):
                self.assertIsNone(provider.bodies.track_at(0, 10.).observation)

    def test_sparse_anchor_body_pass_skips_nonanchor_trajectory_only_after_disproof(self):
        for score in (.585, .8):
            provider = self.provider(face_times=(9.6, 10.4), reference_score=score)
            install_source_first_anchor_prefilter(provider)
            batch = SourceFirstObservationBatch(provider)
            batch(((0, 10.),))
            if score < .68:
                self.assertEqual(provider.bodies.frame_observations.call_count, 3)
                self.assertGreater(batch.skipped_body_dependency_count, 10)
                self.assertIsNone(provider.bodies.track_at(0, 10.).observation)
            else:
                self.assertEqual(batch.skipped_body_dependency_count, 0)
                self.assertIsNotNone(provider.bodies.track_at(0, 10.).observation)

    def test_existing_track_and_unobserved_query_keep_original_authority(self):
        for cached in (False, True):
            provider = self.provider()
            key = provider.faces._identity_key(0, provider.faces._samplers[0], 10.)
            if cached:
                provider.bodies._track_cache[key] = object()
            else:
                provider.bodies._frame_cache.clear()
            original = provider.bodies.track_at
            SourceFirstAnchorPrefilter(provider).track_at(0, 10.)
            original.assert_called_once_with(0, 10.)

    def test_invalid_inputs_and_changed_source_are_not_negative_proof(self):
        provider = self.provider()
        prefilter = SourceFirstAnchorPrefilter(provider)
        for source, t in ((True, 10.), (1, 10.), (0, True), (0, float('nan')), (0, -1.)):
            self.assertFalse(prefilter.cannot_resolve(source, t))
            with self.assertRaises(ValueError):
                prefilter.track_at(source, t)
        provider._identity = lambda _: 'changed'
        with self.assertRaises(OSError):
            prefilter.cannot_resolve(0, 10.)
        self.assertEqual(prefilter.skipped_tracks, 0)

    def test_source_ordered_batch_skips_only_proved_impossible_body_dependencies(self):
        for anchors in ((), (9.6, 10.4)):
            provider = self.provider(face_times=anchors)
            install_source_first_anchor_prefilter(provider)
            batch = SourceFirstObservationBatch(provider)
            batch(((0, 10.),))
            calls = provider.bodies.frame_observations.call_count
            if anchors:
                self.assertGreater(calls, 10)
                self.assertEqual(batch.skipped_body_dependency_count, 0)
            else:
                self.assertEqual(calls, 1)
                self.assertGreater(batch.skipped_body_dependency_count, 10)
                self.assertEqual(batch.anchor_impossible_query_count, 1)
            self.assertEqual(provider.faces._focal_candidates.call_count,
                len(track_dependency_times(provider.faces._samplers[0], 10.)))
            self.assertFalse(batch.diagnostics()['query_offsets_removed'])

    def test_incomplete_faces_and_small_working_cache_keep_all_body_work(self):
        for mode in ('incomplete', 'capacity'):
            provider = self.provider()
            if mode == 'incomplete':
                provider._source_first_unresolved_memo.completed_face_frames.clear()
            else:
                provider.shared_source_frames[0].maximum_working_frames = 2
            install_source_first_anchor_prefilter(provider)
            batch = SourceFirstObservationBatch(provider)
            batch(((0, 10.),))
            self.assertEqual(batch.skipped_body_dependency_count, 0)
            self.assertGreater(provider.bodies.frame_observations.call_count, 10)
            self.assertEqual(batch.face_first_capacity_fallback_count, int(mode == 'capacity'))

    def test_install_is_idempotent_and_records_no_persistent_negative(self):
        provider = self.provider()
        install_source_first_anchor_prefilter(provider)
        first = provider._source_first_anchor_prefilter
        install_source_first_anchor_prefilter(provider)
        self.assertIs(first, provider._source_first_anchor_prefilter)
        self.assertFalse(first.diagnostics()['negative_observations_persisted'])
        self.assertFalse(first.diagnostics()['positive_observation_namespace_changed'])


if __name__ == '__main__':
    unittest.main()
