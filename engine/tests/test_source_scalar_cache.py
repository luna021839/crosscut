from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosscut.audio import AlignmentResult, OffsetEstimate, SourceAlignment
from crosscut.pipeline import (
    _align_sequentially,
    _audio_alignment_from_scalar_payload,
    _audio_alignment_scalar_payload,
    _estimate_source_timewarps,
    _timewarp_from_scalar_payload,
    _timewarp_scalar_payload,
)
from crosscut.source_scalar_cache import (
    PersistentSourceScalarCache,
    PersistentSourceScalarCacheStorageUnavailableError,
)
from crosscut.timewarp import (
    TimeWarpAnchor,
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.vision import (
    SceneChange,
    SceneDetectionConfig,
    detect_scene_changes_in_video_cached,
)


class SourceScalarCacheTests(unittest.TestCase):
    @staticmethod
    def _timewarp(global_offset_seconds: float = 0.2) -> TimeWarpMap:
        return TimeWarpMap(
            global_offset_seconds=global_offset_seconds,
            master_duration_seconds=12.0,
            source_duration_seconds=13.0,
            control_points=(
                TimeWarpControlPoint(
                    master_seconds=0.0,
                    source_seconds=global_offset_seconds,
                ),
                TimeWarpControlPoint(
                    master_seconds=10.0,
                    source_seconds=10.0 + global_offset_seconds + 0.001,
                ),
            ),
            anchors=tuple(
                TimeWarpAnchor(
                    window_index=index,
                    window_start_seconds=float(index),
                    window_end_seconds=float(index + 8),
                    master_seconds=float(index + 4),
                    source_seconds=float(index + 4) + global_offset_seconds,
                    offset_seconds=global_offset_seconds,
                    score=0.8,
                    second_best_score=0.4,
                    score_margin=0.4,
                    accepted=True,
                    trend_residual_seconds=0.01,
                )
                for index in range(3)
            ),
            residuals=TimeWarpResidualDiagnostics(
                p50_seconds=0.01,
                p95_seconds=0.01,
                maximum_seconds=0.01,
                sample_count=3,
            ),
            config=TimeWarpConfig(),
            scalar_fallback=False,
            fallback_reason=None,
        )

    def test_timewarp_scalar_round_trip_reconstructs_checked_map(self) -> None:
        expected = self._timewarp()
        payload = _timewarp_scalar_payload(expected)
        restored = _timewarp_from_scalar_payload(payload)
        self.assertEqual(restored, expected)
        payload["control_points"][1]["offset_seconds"] = 999.0
        with self.assertRaises(ValueError):
            _timewarp_from_scalar_payload(payload)

    def test_local_timewarp_skips_warm_pcm_decode_and_estimation(self) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

            def log(self, _message: str) -> None:
                return None

            def emit(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "cache",
                visual_feature_derived_cache_directory=None,
            )
            alignments = (
                SourceAlignment(
                    source=str(reference),
                    is_reference=True,
                    offset_samples=0,
                    offset_seconds=0.0,
                    score=1.0,
                    overlap_samples=12_000,
                    overlap_seconds=3.0,
                    backend="reference",
                ),
                SourceAlignment(
                    source=str(source),
                    is_reference=False,
                    offset_samples=800,
                    offset_seconds=0.2,
                    score=0.8,
                    overlap_samples=8_000,
                    overlap_seconds=2.0,
                    second_best_score=0.5,
                    score_margin=0.3,
                    overlap_ratio=0.9,
                    backend="spectral_flux_ncc",
                ),
            )
            alignment = AlignmentResult(
                reference=str(reference),
                sample_rate=4_000,
                alignments=alignments,
            )
            reference_pcm = np.ones(12_000, dtype=np.float32)

            def estimate(
                _reference_pcm,  # type: ignore[no-untyped-def]
                _source_pcm,  # type: ignore[no-untyped-def]
                *,
                global_offset_seconds: float,
                **_kwargs,
            ) -> TimeWarpMap:
                return self._timewarp(global_offset_seconds)

            with patch(
                "crosscut.pipeline.extract_mono_pcm",
                return_value=reference_pcm,
            ) as decode, patch(
                "crosscut.pipeline.estimate_local_timewarp",
                side_effect=estimate,
            ) as estimate_timewarp:
                cold, _cold_payloads = _estimate_source_timewarps(
                    request,
                    alignment,
                    reference_pcm,
                    eligible_indices=(0, 1),
                    reporter=Reporter(),
                )
                warm, _warm_payloads = _estimate_source_timewarps(
                    request,
                    alignment,
                    reference_pcm,
                    eligible_indices=(0, 1),
                    reporter=Reporter(),
                )

            self.assertEqual(cold, warm)
            self.assertEqual(estimate_timewarp.call_count, 2)
            self.assertEqual(decode.call_count, 1)

    def test_audio_alignment_scalar_round_trip_is_strict(self) -> None:
        alignment = SourceAlignment(
            source="ignored.mp4",
            is_reference=False,
            offset_samples=800,
            offset_seconds=0.2,
            score=0.8,
            overlap_samples=8_000,
            overlap_seconds=2.0,
            second_best_score=0.5,
            score_margin=0.3,
            overlap_ratio=0.9,
            backend="spectral_flux_ncc",
        )
        payload = _audio_alignment_scalar_payload(alignment)
        restored = _audio_alignment_from_scalar_payload(
            payload,
            source=Path("current.mp4"),
        )
        self.assertEqual(restored.source, "current.mp4")
        self.assertEqual(
            _audio_alignment_scalar_payload(restored),
            payload,
        )
        payload["score"] = True
        with self.assertRaises(ValueError):
            _audio_alignment_from_scalar_payload(
                payload,
                source=Path("current.mp4"),
            )

    def test_pairwise_audio_alignment_skips_warm_source_decode(self) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

            def log(self, _message: str) -> None:
                return None

            def progress(self, _value: float, _message: str) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "cache",
                visual_feature_derived_cache_directory=None,
            )
            estimate = OffsetEstimate(
                offset_samples=800,
                offset_seconds=0.2,
                score=0.8,
                overlap_samples=8_000,
                overlap_seconds=2.0,
                sample_rate=4_000,
                second_best_score=0.5,
                score_margin=0.3,
                overlap_ratio=0.9,
                backend="spectral_flux_ncc",
            )
            with patch(
                "crosscut.pipeline.extract_mono_pcm",
                return_value=np.ones(12_000, dtype=np.float32),
            ) as decode, patch(
                "crosscut.pipeline.spectral_flux_alignment_offset",
                return_value=estimate,
            ) as align:
                cold, _cold_reference = _align_sequentially(
                    request,
                    Reporter(),
                )
                warm, _warm_reference = _align_sequentially(
                    request,
                    Reporter(),
                )

            self.assertEqual(cold, warm)
            self.assertEqual(decode.call_count, 3)
            self.assertEqual(align.call_count, 1)

    def test_direct_audio_semantic_cache_miss_recomputes_without_storage_error(
        self,
    ) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

            def log(self, _message: str) -> None:
                return None

            def progress(self, _value: float, _message: str) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "legacy-cache",
                visual_feature_derived_cache_directory=direct_root,
            )
            estimate = OffsetEstimate(
                offset_samples=800,
                offset_seconds=0.2,
                score=0.8,
                overlap_samples=8_000,
                overlap_seconds=2.0,
                sample_rate=4_000,
                second_best_score=0.5,
                score_margin=0.3,
                overlap_ratio=0.9,
                backend="spectral_flux_ncc",
            )
            cache = Mock()
            cache.get.return_value = {"score": True}
            cache.put.return_value = True
            with patch(
                "crosscut.pipeline.PersistentSourceScalarCache",
                return_value=cache,
            ), patch(
                "crosscut.pipeline.extract_mono_pcm",
                return_value=np.ones(12_000, dtype=np.float32),
            ) as decode, patch(
                "crosscut.pipeline.spectral_flux_alignment_offset",
                return_value=estimate,
            ) as align:
                result, _reference_pcm = _align_sequentially(
                    request,
                    Reporter(),
                )

            self.assertEqual(result.alignments[1].offset_seconds, 0.2)
            self.assertEqual(align.call_count, 1)
            self.assertEqual(decode.call_count, 2)
            cache.put.assert_called_once()

    def test_direct_audio_reference_hash_failure_never_starts_live_alignment(
        self,
    ) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

            def log(self, _message: str) -> None:
                return None

            def progress(self, _value: float, _message: str) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "legacy-cache",
                visual_feature_derived_cache_directory=direct_root,
            )
            with patch(
                "crosscut.pipeline.extract_mono_pcm",
                return_value=np.ones(12_000, dtype=np.float32),
            ) as decode, patch(
                "crosscut.pipeline.source_content_sha256",
                side_effect=OSError("source hash failed"),
            ), patch(
                "crosscut.pipeline.PersistentSourceScalarCache"
            ) as cache_type, patch(
                "crosscut.pipeline.spectral_flux_alignment_offset"
            ) as live_alignment:
                with self.assertRaisesRegex(OSError, "source hash failed"):
                    _align_sequentially(request, Reporter())

            self.assertEqual(decode.call_count, 1)
            cache_type.assert_not_called()
            live_alignment.assert_not_called()

    def test_direct_timewarp_semantic_cache_miss_recomputes_without_storage_error(
        self,
    ) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

            def log(self, _message: str) -> None:
                return None

            def emit(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "legacy-cache",
                visual_feature_derived_cache_directory=direct_root,
            )
            alignment = AlignmentResult(
                reference=str(reference),
                sample_rate=4_000,
                alignments=(
                    SourceAlignment(
                        source=str(reference),
                        is_reference=True,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0,
                        overlap_samples=12_000,
                        overlap_seconds=3.0,
                        backend="reference",
                    ),
                    SourceAlignment(
                        source=str(source),
                        is_reference=False,
                        offset_samples=800,
                        offset_seconds=0.2,
                        score=0.8,
                        overlap_samples=8_000,
                        overlap_seconds=2.0,
                        second_best_score=0.5,
                        score_margin=0.3,
                        overlap_ratio=0.9,
                        backend="spectral_flux_ncc",
                    ),
                ),
            )
            reference_pcm = np.ones(12_000, dtype=np.float32)
            cache = Mock()
            cache.get.return_value = {"scalar_fallback": "invalid"}
            cache.put.return_value = True
            cache.diagnostics.return_value = {
                "cache_contract": "ive_source_scalar_analysis_v1",
                "namespace": "local-timewarp-v1",
                "raw_media_persisted": False,
            }

            def estimate(
                _reference_pcm,  # type: ignore[no-untyped-def]
                _source_pcm,  # type: ignore[no-untyped-def]
                *,
                global_offset_seconds: float,
                **_kwargs,
            ) -> TimeWarpMap:
                return self._timewarp(global_offset_seconds)

            with patch(
                "crosscut.pipeline.PersistentSourceScalarCache",
                return_value=cache,
            ), patch(
                "crosscut.pipeline.extract_mono_pcm",
                return_value=reference_pcm,
            ) as decode, patch(
                "crosscut.pipeline.estimate_local_timewarp",
                side_effect=estimate,
            ) as estimate_timewarp:
                maps, _payloads = _estimate_source_timewarps(
                    request,
                    alignment,
                    reference_pcm,
                    eligible_indices=(0, 1),
                    reporter=Reporter(),
                )

            self.assertIsNotNone(maps[0])
            self.assertIsNotNone(maps[1])
            self.assertEqual(estimate_timewarp.call_count, 2)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(cache.put.call_count, 2)

    def test_direct_timewarp_reference_hash_failure_never_starts_estimator(
        self,
    ) -> None:
        class Reporter:
            def status(self, _message: str) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mp4"
            source = root / "source.mp4"
            reference.write_bytes(b"reference-video")
            source.write_bytes(b"source-video")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            request = SimpleNamespace(
                sources=(reference, source),
                reference_source=0,
                source_cache_directory=root / "legacy-cache",
                visual_feature_derived_cache_directory=direct_root,
            )
            alignment = AlignmentResult(
                reference=str(reference),
                sample_rate=4_000,
                alignments=(
                    SourceAlignment(
                        source=str(reference),
                        is_reference=True,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0,
                        overlap_samples=12_000,
                        overlap_seconds=3.0,
                        backend="reference",
                    ),
                    SourceAlignment(
                        source=str(source),
                        is_reference=False,
                        offset_samples=800,
                        offset_seconds=0.2,
                        score=0.8,
                        overlap_samples=8_000,
                        overlap_seconds=2.0,
                        backend="spectral_flux_ncc",
                    ),
                ),
            )
            with patch(
                "crosscut.pipeline.source_content_sha256",
                side_effect=OSError("source hash failed"),
            ), patch(
                "crosscut.pipeline.PersistentSourceScalarCache"
            ) as cache_type, patch(
                "crosscut.pipeline.estimate_local_timewarp"
            ) as estimator:
                with self.assertRaisesRegex(OSError, "source hash failed"):
                    _estimate_source_timewarps(
                        request,
                        alignment,
                        np.ones(12_000, dtype=np.float32),
                        eligible_indices=(0, 1),
                        reporter=Reporter(),
                    )

            cache_type.assert_not_called()
            estimator.assert_not_called()

    def test_atomic_record_is_content_and_analyzer_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            cache = PersistentSourceScalarCache(
                source_cache_directory=root / "cache",
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1", "threshold": 0.5},
            )
            payload = {"values": [0.1, 0.2], "count": 2}
            self.assertTrue(cache.put(payload))
            self.assertEqual(cache.get(), payload)

            changed_analyzer = PersistentSourceScalarCache(
                source_cache_directory=root / "cache",
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1", "threshold": 0.6},
            )
            self.assertIsNone(changed_analyzer.get())

            records = tuple((root / "cache" / "derived").rglob("*.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            record["scalar_analysis"]["count"] = 1
            records[0].write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(cache.get())
            self.assertEqual(cache.diagnostics()["invalid_record_count"], 1)

    def test_direct_derived_root_ignores_legacy_derived_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            source_cache = root / "legacy-source-cache"
            source_cache.mkdir()
            legacy_target = root / "legacy-derived-target"
            legacy_target.mkdir()
            (source_cache / "derived").symlink_to(
                legacy_target,
                target_is_directory=True,
            )
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            payload = {"values": [0.1, 0.2], "count": 2}

            cache = PersistentSourceScalarCache(
                source_cache_directory=source_cache,
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1"},
                derived_cache_directory=direct_root,
            )

            self.assertTrue(cache.put(payload))
            self.assertEqual(cache.get(), payload)
            self.assertEqual(
                len(tuple((direct_root / "scalar-v1").rglob("*.json"))),
                1,
            )
            self.assertEqual(tuple(legacy_target.iterdir()), ())
            self.assertTrue((source_cache / "derived").is_symlink())
            diagnostics = cache.diagnostics()
            self.assertTrue(diagnostics["required"])
            self.assertTrue(
                diagnostics["direct_derived_cache_configured"]
            )
            self.assertFalse(diagnostics["local_paths_exposed"])
            self.assertNotIn(str(root), json.dumps(diagnostics))

    def test_missing_direct_root_is_typed_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            missing_direct_root = root / "missing-derived-cache"

            with self.assertRaises(
                PersistentSourceScalarCacheStorageUnavailableError
            ) as raised:
                PersistentSourceScalarCache(
                    source_cache_directory=root / "unused-source-cache",
                    source_path=source,
                    namespace="test-analysis",
                    analyzer={"contract": "test-v1"},
                    derived_cache_directory=missing_direct_root,
                )

            self.assertNotIn(str(root), str(raised.exception))

    def test_direct_root_rename_fails_before_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            cache = PersistentSourceScalarCache(
                source_cache_directory=root / "unused-source-cache",
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1"},
                derived_cache_directory=direct_root,
            )
            self.assertTrue(cache.put({"value": 1.0}))
            direct_root.rename(root / "detached-derived-cache")

            with self.assertRaises(
                PersistentSourceScalarCacheStorageUnavailableError
            ) as raised:
                cache.get()

            self.assertNotIn(str(root), str(raised.exception))
            diagnostics = cache.diagnostics()
            self.assertEqual(diagnostics["hit_count"], 0)
            self.assertEqual(diagnostics["miss_count"], 0)
            self.assertEqual(diagnostics["storage_io_failure_count"], 1)

    def test_direct_root_identity_replacement_fails_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            direct_root = root / "direct-derived-cache"
            direct_root.mkdir()
            cache = PersistentSourceScalarCache(
                source_cache_directory=root / "unused-source-cache",
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1"},
                derived_cache_directory=direct_root,
            )
            direct_root.rename(root / "detached-derived-cache")
            direct_root.mkdir()

            with self.assertRaises(
                PersistentSourceScalarCacheStorageUnavailableError
            ) as raised:
                cache.put({"value": 1.0})

            self.assertNotIn(str(root), str(raised.exception))
            self.assertFalse((direct_root / "scalar-v1").exists())
            diagnostics = cache.diagnostics()
            self.assertEqual(diagnostics["write_count"], 0)
            self.assertEqual(diagnostics["storage_io_failure_count"], 1)

    def test_legacy_derived_root_remains_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            source_cache = root / "source-cache"
            cache = PersistentSourceScalarCache(
                source_cache_directory=source_cache,
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1"},
            )

            self.assertTrue(cache.put({"value": 1.0}))
            self.assertEqual(
                len(
                    tuple(
                        (source_cache / "derived" / "scalar-v1").rglob(
                            "*.json"
                        )
                    )
                ),
                1,
            )
            diagnostics = cache.diagnostics()
            self.assertFalse(diagnostics["required"])
            self.assertFalse(
                diagnostics["direct_derived_cache_configured"]
            )
            self.assertEqual(diagnostics["storage_io_failure_count"], 0)

    def test_source_change_disables_existing_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"AAAA")
            cache = PersistentSourceScalarCache(
                source_cache_directory=root / "cache",
                source_path=source,
                namespace="test-analysis",
                analyzer={"contract": "test-v1"},
            )
            self.assertTrue(cache.put({"value": 1.0}))
            original_mtime = source.stat().st_mtime_ns
            source.write_bytes(b"BBBB")
            os.utime(source, ns=(original_mtime, original_mtime))
            self.assertIsNone(cache.get())
            self.assertTrue(cache.diagnostics()["disabled_by_source_change"])

    def test_scene_scan_is_reused_and_config_change_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"fake-video")
            expected = (
                SceneChange(
                    frame_index=30,
                    score=0.8,
                    frame_difference=0.9,
                    histogram_distance=0.7,
                ),
            )
            with patch(
                "crosscut.vision.detect_scene_changes_in_video",
                return_value=expected,
            ) as detect:
                cold, cold_diagnostics = detect_scene_changes_in_video_cached(
                    source,
                    config=SceneDetectionConfig(min_gap_frames=8),
                    sample_every=2,
                    source_cache_directory=root / "cache",
                )
                warm, warm_diagnostics = detect_scene_changes_in_video_cached(
                    source,
                    config=SceneDetectionConfig(min_gap_frames=8),
                    sample_every=2,
                    source_cache_directory=root / "cache",
                )
                changed, changed_diagnostics = (
                    detect_scene_changes_in_video_cached(
                        source,
                        config=SceneDetectionConfig(min_gap_frames=9),
                        sample_every=2,
                        source_cache_directory=root / "cache",
                    )
                )

            self.assertEqual((cold, warm, changed), (expected, expected, expected))
            self.assertEqual(detect.call_count, 2)
            self.assertFalse(cold_diagnostics["analysis_reused"])
            self.assertTrue(warm_diagnostics["analysis_reused"])
            self.assertFalse(changed_diagnostics["analysis_reused"])


if __name__ == "__main__":
    unittest.main()
