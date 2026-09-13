from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.joint_observation_reuse import (
    JointObservationBank, ReusingDecodedBoundaryOptionProvider,
)
from crosscut.joint_timeline import BoundaryTransformState
from crosscut.cut_anchored_containment import ContainmentSample
from crosscut.visual_boundary_repair import BoundaryDescriptorCache
from engine.tests.test_joint_boundary_provider import _descriptor, _key


class JointObservationReuseTests(unittest.TestCase):
    def containment_arguments(self, provider, **overrides):
        values = dict(
            source_index=0, start_master_seconds=10.,
            sample_times=(10., 10.1), required_master_sample_times=(10.,),
            entry_transform=BoundaryTransformState("entry", subject_basis="dominant_face"),
            exit_transform=BoundaryTransformState("exit"),
            editorial_track=SimpleNamespace(keyframes=((0., 1.2), (.1, 1.))),
            descriptor_resolver=provider._sample_descriptor,
        )
        return {**values, **overrides}

    def test_successful_containment_observations_reuse_without_proof_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.touch()
            options = dict(sources=(source, source), descriptor_provider=None,
                           transformed_descriptor_provider=None, face_identity_provider=None)
            samples = {0: ContainmentSample(0, (.4,.2,.2,.2), 'observed_cut_anchor', 0)}
            with patch.object(DecodedBoundaryOptionProvider, '_cut_handle_containment',
                              side_effect=lambda **_: (dict(samples), None)) as observe:
                with self.provider(**options) as first:
                    actual, failure = first._cut_handle_containment(**self.containment_arguments(first))
                    self.assertIsNone(failure)
                    actual.clear()
                with self.provider(**options) as second:
                    args = self.containment_arguments(second)
                    args['entry_transform'] = replace(args['entry_transform'], transform_identity='other-proof')
                    actual, failure = second._cut_handle_containment(**args)
                    self.assertEqual(actual, samples)
                    self.assertIsNone(failure)
                    self.assertEqual(observe.call_count, 1)
                    self.assertFalse(second._materialization_by_proof)
                    # Full proof/custom resolver callbacks must still execute.
                    args['descriptor_resolver'] = lambda source, seconds: None
                    second._cut_handle_containment(**args)
                    self.assertEqual(observe.call_count, 2)

    def test_containment_time_geometry_and_storage_are_bound_on_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.touch()
            options = dict(sources=(source, source), descriptor_provider=None,
                           transformed_descriptor_provider=None, face_identity_provider=None)
            sample = ContainmentSample(0, (.4,.2,.2,.2), 'observed_cut_anchor', 0)
            with patch.object(DecodedBoundaryOptionProvider, '_cut_handle_containment',
                              return_value=({0: sample}, None)) as observe:
                with self.provider(**options) as provider:
                    base = self.containment_arguments(provider)
                    provider._cut_handle_containment(**base)
                    for changes in (
                        dict(sample_times=(10.,10.10000001)),
                        dict(required_master_sample_times=(10.1,)),
                        dict(start_master_seconds=10.00000001),
                        dict(source_index=1),
                        dict(entry_transform=BoundaryTransformState('entry')),
                        dict(editorial_track=SimpleNamespace(keyframes=((0.,1.3),(.1,1.)))),
                    ):
                        provider._cut_handle_containment(**{**base,**changes})
                    self.assertEqual(observe.call_count, 7)
                    with patch.object(provider, '_assert_required_persistent_cache_available',
                                      side_effect=OSError('SSD unavailable')):
                        with self.assertRaisesRegex(OSError, 'SSD unavailable'):
                            provider._cut_handle_containment(**base)
                    source.write_bytes(b'changed')
                    with self.assertRaisesRegex(RuntimeError, 'source changed'):
                        provider._cut_handle_containment(**base)

    def test_failed_and_aborted_containment_are_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.touch()
            options = dict(sources=(source, source), descriptor_provider=None,
                           transformed_descriptor_provider=None, face_identity_provider=None)
            sample = ContainmentSample(0, (.4,.2,.2,.2), 'observed_cut_anchor', 0)
            with self.provider(**options) as provider:
                args = self.containment_arguments(provider)
                with patch.object(DecodedBoundaryOptionProvider, '_cut_handle_containment',
                                  return_value=({0: sample}, 'pixel_flow_failed')):
                    provider._cut_handle_containment(**args)
                    self.assertEqual(self.bank.diagnostics()['entry_count'], 0)
                def aborted(_key):
                    provider._cut_handle_containment(**args)
                    raise RuntimeError('aborted containment transaction')
                with patch.object(DecodedBoundaryOptionProvider, '_cut_handle_containment',
                                  return_value=({0: sample}, None)), patch.object(provider, '_measure', side_effect=aborted):
                    with self.assertRaisesRegex(RuntimeError, 'aborted containment transaction'):
                        provider(_key())
                self.assertEqual(self.bank.diagnostics()['entry_count'], 0)
                self.assertEqual(provider._pending_observations, {})

    def setUp(self):
        self.calls = [0, 0, 0]
        self.bank = JointObservationBank()

        def raw(source, seconds):
            self.calls[0] += 1
            return _descriptor(.5, face_bbox=(.4, .2, .2, .2))

        def transformed(source, seconds, descriptor, transform):
            self.calls[1] += 1
            return descriptor

        def face(source, seconds, descriptor):
            self.calls[2] += 1
            return (1.0, 0.0)

        self.arguments = dict(
            sources=(Path('source-a.mp4'), Path('source-b.mp4')),
            audio_offsets=(0.0, 0.0), scene_cut_times=((), ()),
            source_timewarps=None,
            output_fps=30.0, total_segment_count=2,
            descriptor_provider=raw, transformed_descriptor_provider=transformed,
            face_identity_provider=face,
        )

    def provider(self, **overrides):
        return ReusingDecodedBoundaryOptionProvider(
            observation_bank=self.bank, descriptor_cache=BoundaryDescriptorCache(),
            **{**self.arguments, **overrides})

    def observe(self, provider):
        raw = provider._sample_descriptor(0, 10.0)
        transformed = provider._sample_transformed_descriptor(
            0, 10.0, raw, BoundaryTransformState('identity'))
        face = provider._sample_face_identity(0, 10.0, raw)
        return raw, transformed, face

    def test_fresh_provider_reuses_only_observations(self):
        first = self.provider()
        observed = self.observe(first)
        first.close()
        second = self.provider()
        self.assertEqual(self.observe(second), observed)
        self.assertEqual(self.calls, [1, 1, 1])
        self.assertFalse(second._memo)
        self.assertFalse(second._materialization_by_proof)
        self.assertFalse(second._bridge_memo)
        self.assertEqual(second._configuration_identity, first._configuration_identity)
        second.close()

    def test_production_raster_reuse_ignores_proof_labels_not_clock_or_geometry(self):
        with tempfile.TemporaryDirectory(prefix="joint-raster-observation-") as directory:
            sources = (Path(directory) / "a.mp4", Path(directory) / "b.mp4")
            for source in sources:
                source.touch()
            overrides = dict(sources=sources, descriptor_provider=None,
                             transformed_descriptor_provider=None,
                             face_identity_provider=None)
            raw = _descriptor(.5, face_bbox=(.4, .2, .2, .2))
            transform = BoundaryTransformState(
                "proof-a", crop_xywh=(.1, .1, .8, .8), track_id="track-a",
                subject_basis="dominant_face")
            sampler = Mock()
            sampler.frame_at.return_value = np.full((48, 64, 3), 96, np.uint8)
            sampler.frame_digest_at.return_value = "b" * 64
            with patch("crosscut.joint_boundary_provider.describe_frame", return_value=raw) as describe:
                with self.provider(**overrides) as first:
                    with patch.object(first, "_sampler", return_value=sampler):
                        self.assertEqual(first._sample_transformed_descriptor(0, 10., raw, transform), raw)
                with self.provider(**overrides) as second:
                    with patch.object(second, "_sampler", return_value=sampler):
                        relabeled = replace(transform, transform_identity="proof-b", track_id="track-b")
                        self.assertEqual(second._sample_transformed_descriptor(0, 10., raw, relabeled), raw)
                        self.assertEqual(describe.call_count, 1)
                        self.assertFalse(second._materialization_by_proof)
                        for index, seconds, geometry in (
                            (0, 10.001, relabeled),
                            (1, 10., relabeled),
                            (0, 10., replace(relabeled, crop_xywh=(.11, .1, .8, .8))),
                            (0, 10., replace(relabeled, rotation_degrees=.1)),
                        ):
                            second._sample_transformed_descriptor(index, seconds, raw, geometry)
                        self.assertEqual(describe.call_count, 5)

    def test_injected_transform_provider_retains_full_proof_identity(self):
        raw = _descriptor(.5)
        with self.provider() as first:
            first._sample_transformed_descriptor(0, 10., raw, BoundaryTransformState("a"))
        with self.provider() as second:
            second._sample_transformed_descriptor(0, 10., raw, BoundaryTransformState("b"))
        self.assertEqual(self.calls[1], 2)

    def test_complete_boundary_options_equal_uncached_provider(self):
        with DecodedBoundaryOptionProvider(
            descriptor_cache=BoundaryDescriptorCache(), **self.arguments
        ) as baseline:
            expected = baseline(_key())
        with self.provider() as first:
            self.assertEqual(first(_key()), expected)
        calls = tuple(self.calls)
        with self.provider() as second:
            self.assertEqual(second(_key()), expected)
        self.assertEqual(tuple(self.calls), calls)

    def test_failed_measurement_does_not_publish(self):
        provider = self.provider()

        def fail(_key):
            self.observe(provider)
            raise RuntimeError('abort measurement')

        with patch.object(provider, '_measure', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'abort measurement'):
                provider(_key())
        self.assertEqual(self.bank.diagnostics()['entry_count'], 0)
        self.assertEqual(provider._pending_observations, {})
        provider.close()
        with self.provider() as retry:
            self.observe(retry)
        self.assertEqual(self.calls, [2, 2, 2])

    def test_clock_backend_and_geometry_are_not_aliased(self):
        with self.provider() as first:
            raw, _, _ = self.observe(first)
        with self.provider() as next_provider:
            next_provider._sample_descriptor(0, 10.0 + 1e-8)
            next_provider._sample_transformed_descriptor(
                0, 10.0, replace(raw, shot_scale=.8), BoundaryTransformState('identity'))
        self.assertEqual(self.calls[:2], [2, 2])
        with self.provider(descriptor_provider=lambda s, t: _descriptor(.7)) as other:
            self.assertEqual(other._sample_descriptor(0, 10.0), _descriptor(.7))

    def test_source_change_rejected_even_on_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.write_bytes(b'fixture')
            with self.provider(sources=(source, source)) as provider:
                provider._sample_descriptor(0, 10.0)
                source.write_bytes(b'changed fixture')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    provider._sample_descriptor(0, 10.0)

    def test_storage_check_not_bypassed_by_hit(self):
        with self.provider() as provider:
            self.observe(provider)
            with patch.object(provider, '_assert_required_persistent_cache_available',
                              side_effect=OSError('SSD unavailable')):
                with self.assertRaisesRegex(OSError, 'SSD unavailable'):
                    self.observe(provider)

    def test_missing_observation_is_not_published(self):
        with self.provider(descriptor_provider=lambda s, t: None) as provider:
            self.assertIsNone(provider._sample_descriptor(0, 10.0))
        self.assertEqual(self.bank.diagnostics()['entry_count'], 0)

    def test_bank_has_byte_and_count_limits(self):
        bank = JointObservationBank(maximum_bytes=1200, maximum_entries=2)
        for index in range(100):
            bank.publish((index,), (float(index),))
            self.assertLessEqual(bank.accounted_bytes, 1200)
            self.assertLessEqual(bank.diagnostics()['entry_count'], 2)
        self.assertGreater(bank.evictions, 0)
        bank.publish(('oversized',), tuple(range(1000)))
        self.assertFalse(bank.lookup(('oversized',))[0])

    def test_replay_larger_than_base_transformed_lru_avoids_model_repeats(self):
        transform = BoundaryTransformState('identity')
        for batch in range(2):
            with self.provider() as provider:
                for index in range(4200):
                    seconds = 10. + index / 30.
                    raw = provider._sample_descriptor(0, seconds)
                    provider._sample_transformed_descriptor(0, seconds, raw, transform)
            if batch == 0:
                first_calls = tuple(self.calls)
        self.assertEqual(tuple(self.calls), first_calls)
        self.assertEqual(self.calls[:2], [4200, 4200])
        self.assertEqual(self.bank.evictions, 0)
        self.assertLess(self.bank.accounted_bytes, self.bank.maximum_bytes)

    def test_closed_provider_cannot_read_bank(self):
        provider = self.provider()
        self.observe(provider)
        provider.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.observe(provider)


if __name__ == '__main__':
    unittest.main()
