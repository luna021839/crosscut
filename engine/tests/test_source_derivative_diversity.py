from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.reference_match import DecodedTargetFrames
from crosscut.source_derivative_diversity import (
    DERIVATIVE_CONFIRMED,
    DERIVATIVE_DISTINCT,
    DERIVATIVE_UNCERTAIN,
    DecodedSourceFrame,
    PersistentSourceDerivativeDiversityCache,
    analyze_source_derivative_diversity,
    classify_source_derivative_frames,
    source_derivative_diversity_from_dict,
)
from crosscut.source_scalar_cache import (
    PersistentSourceScalarCacheStorageUnavailableError,
)
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)


def _textured_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    frame[:] = rng.integers(0, 75, (1, 1, 3), dtype=np.uint8)
    for _ in range(115):
        x = int(rng.integers(0, frame.shape[1]))
        y = int(rng.integers(0, frame.shape[0]))
        color = tuple(int(value) for value in rng.integers(45, 256, 3))
        cv2.circle(frame, (x, y), int(rng.integers(2, 10)), color, -1)
    cv2.putText(
        frame,
        f"IVE {seed}",
        (90, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        4,
    )
    return frame


def _cropped_derivative(frame: np.ndarray) -> np.ndarray:
    cropped = frame[6:-6, 13:-13]
    resized = cv2.resize(
        cropped,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    # Mild re-encode-like filtering/color error must not defeat structure.
    filtered = cv2.GaussianBlur(resized, (3, 3), 0.25)
    return np.clip(filtered.astype(np.int16) + 2, 0, 255).astype(np.uint8)


def _observation(frame: np.ndarray, index: int) -> DecodedSourceFrame:
    return DecodedSourceFrame(
        frame=frame,
        frame_index=index,
        requested_source_time_seconds=float(index),
        actual_pts_seconds=float(index),
        normalized_pts_seconds=float(index),
        pts_residual_seconds=0.0,
    )


def _identity_timewarp(duration: float = 10.0) -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=0.0,
        master_duration_seconds=duration,
        source_duration_seconds=duration,
        control_points=(
            TimeWarpControlPoint(0.0, 0.0),
            TimeWarpControlPoint(duration, duration),
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(0.0, 0.0, 0.0, 3),
        config=TimeWarpConfig(),
        scalar_fallback=False,
        fallback_reason=None,
    )


class SourceDerivativeDiversityTests(unittest.TestCase):
    @staticmethod
    def _uncertain_two_source_result():
        blank = np.full((270, 480, 3), 127, dtype=np.uint8)
        rows = tuple(
            tuple(_observation(blank.copy(), index) for index in range(9))
            for _ in range(2)
        )
        return classify_source_derivative_frames(
            rows,
            canonical_sample_times_seconds=tuple(
                float(index + 1) for index in range(9)
            ),
            source_decode_audit=(
                {"source_index": 0, "decoded_sample_count": 9},
                {"source_index": 1, "decoded_sample_count": 9},
            ),
            clock_contract_complete=False,
        )

    def test_crop_transcode_forms_one_family_while_distinct_camera_stays_separate(self):
        rows: list[list[DecodedSourceFrame | None]] = [[], [], []]
        for index in range(9):
            base = _textured_frame(100 + index)
            derivative = _cropped_derivative(base)
            distinct = _textured_frame(500 + index)
            for row, frame in zip(rows, (base, derivative, distinct), strict=True):
                row.append(_observation(frame, index))

        result = classify_source_derivative_frames(
            rows,
            canonical_sample_times_seconds=tuple(float(index + 1) for index in range(9)),
            source_decode_audit=tuple(
                {"source_index": index, "decoded_sample_count": 9}
                for index in range(3)
            ),
            clock_contract_complete=True,
        )

        statuses = {
            (pair.left_source_index, pair.right_source_index): pair.status
            for pair in result.pairs
        }
        self.assertEqual(DERIVATIVE_CONFIRMED, statuses[(0, 1)])
        self.assertEqual(DERIVATIVE_DISTINCT, statuses[(0, 2)])
        self.assertEqual(DERIVATIVE_DISTINCT, statuses[(1, 2)])
        self.assertEqual((0, 0, 1), result.family_ids)
        self.assertTrue(result.production_ready)
        self.assertEqual(2, result.decoder_metadata().family_count)

        payload = result.to_dict()
        # The full payload must be JSON-safe and contain scalar audit only.
        json.dumps(payload, sort_keys=True)
        self.assertEqual(payload["source_count"], 3)
        self.assertFalse(payload["persistence_policy"]["raw_frames_persisted"])
        self.assertFalse(payload["persistence_policy"]["descriptors_persisted"])
        self.assertFalse(payload["persistence_policy"]["embeddings_persisted"])
        self.assertNotIn("homography_reference_to_source", json.dumps(payload))

    def test_low_structure_or_missing_clock_is_uncertain_not_false_positive(self):
        result = self._uncertain_two_source_result()
        self.assertEqual(DERIVATIVE_UNCERTAIN, result.pairs[0].status)
        self.assertEqual((0, 1), result.family_ids)
        self.assertFalse(result.production_ready)
        self.assertEqual(1, result.uncertain_pair_count)

    def test_media_path_uses_timewarp_then_nearest_actual_pts_and_forward_decode(self):
        paths = tuple(Path(f"/tmp/derivative-{index}.mp4") for index in range(3))
        points = tuple(float(index) for index in range(11))

        def fake_decode(path: Path, indices, *, config):
            requested = tuple(sorted(set(indices)))
            source_index = int(path.stem.rsplit("-", 1)[1])
            frames = {}
            for index in requested:
                base = _textured_frame(900 + index)
                if source_index == 1:
                    frame = _cropped_derivative(base)
                elif source_index == 2:
                    frame = _textured_frame(1900 + index)
                else:
                    frame = base
                frames[index] = frame
            return DecodedTargetFrames(
                frame_rate=1.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames=frames,
            )

        with (
            patch(
                "crosscut.source_derivative_diversity._probe_actual_video_pts",
                return_value=points,
            ),
            patch(
                "crosscut.source_derivative_diversity.decode_target_frames_sequential",
                side_effect=fake_decode,
            ) as decode,
        ):
            result = analyze_source_derivative_diversity(
                paths,
                source_offsets_seconds=(0.0, 0.0, 0.0),
                source_timewarps=(
                    _identity_timewarp(),
                    _identity_timewarp(),
                    _identity_timewarp(),
                ),
                common_master_start=0.0,
                common_master_end=10.0,
            )

        self.assertEqual(3, decode.call_count)
        self.assertTrue(result.clock_contract_complete)
        self.assertTrue(result.production_ready)
        self.assertEqual((0, 0, 1), result.family_ids)
        self.assertTrue(
            all(
                item["pts_basis"] == "ffprobe_best_effort_timestamp_time"
                and item["clock_basis"] == "checked_local_monotonic_timewarp"
                and item["decoded_sample_count"] == 9
                for item in result.source_decode_audit
            )
        )

    def test_scalar_payload_round_trip_recomputes_audit_sha(self):
        result = self._uncertain_two_source_result()
        decoded = source_derivative_diversity_from_dict(result.to_dict())
        self.assertEqual(result.to_dict(), decoded.to_dict())

        tampered = json.loads(json.dumps(result.to_dict()))
        tampered["pairs"][0]["matching_sample_count"] += 1
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            source_derivative_diversity_from_dict(tampered)

    def test_persistent_cache_binds_all_sources_clocks_and_common_range(self):
        result = self._uncertain_two_source_result()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_directory = root / "cache"
            derived_cache_directory = root / "mounted-derived-cache"
            derived_cache_directory.mkdir()
            sources = (root / "one.mp4", root / "two.mp4")
            sources[0].write_bytes(b"first-source")
            sources[1].write_bytes(b"second-source")
            clocks = (_identity_timewarp(), _identity_timewarp())
            constructor = dict(
                source_cache_directory=cache_directory,
                derived_cache_directory=derived_cache_directory,
                sources=sources,
                source_offsets_seconds=(0.0, 0.0),
                source_timewarps=clocks,
                common_master_start=0.0,
                common_master_end=10.0,
            )
            with patch(
                "crosscut.source_derivative_diversity._ffprobe_version_identity",
                return_value="ffprobe-test",
            ):
                cache = PersistentSourceDerivativeDiversityCache(**constructor)
                self.assertIsNone(cache.get())
                self.assertTrue(cache.put(result))
                restored = cache.get()
                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertEqual(result.to_dict(), restored.to_dict())
                self.assertEqual(1, cache.diagnostics()["hit_count"])
                self.assertFalse((cache_directory / "derived").exists())
                self.assertEqual(
                    1,
                    len(tuple(derived_cache_directory.rglob("*.json"))),
                )

                changed_range = PersistentSourceDerivativeDiversityCache(
                    **{**constructor, "common_master_end": 9.5}
                )
                self.assertIsNone(changed_range.get())

                sources[1].write_bytes(b"changed-second-source")
                changed_source = PersistentSourceDerivativeDiversityCache(
                    **constructor
                )
                self.assertIsNone(changed_source.get())

    def test_direct_cache_loss_precedes_non_anchor_source_change_on_get_and_put(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived_cache_directory = root / "mounted-derived-cache"
            derived_cache_directory.mkdir()
            sources = (root / "one.mp4", root / "two.mp4")
            sources[0].write_bytes(b"first-source")
            sources[1].write_bytes(b"second-source")
            with patch(
                "crosscut.source_derivative_diversity._ffprobe_version_identity",
                return_value="ffprobe-test",
            ):
                cache = PersistentSourceDerivativeDiversityCache(
                    source_cache_directory=root / "legacy-cache",
                    derived_cache_directory=derived_cache_directory,
                    sources=sources,
                    source_offsets_seconds=(0.0, 0.0),
                    source_timewarps=(
                        _identity_timewarp(),
                        _identity_timewarp(),
                    ),
                    common_master_start=0.0,
                    common_master_end=10.0,
                )
            derived_cache_directory.rename(root / "detached-derived-cache")
            sources[1].write_bytes(b"changed-second-source")

            operations = (
                cache.get,
                lambda: cache.put(self._uncertain_two_source_result()),
            )
            for operation in operations:
                with self.subTest(operation=operation):
                    with self.assertRaises(
                        PersistentSourceScalarCacheStorageUnavailableError
                    ):
                        operation()


if __name__ == "__main__":
    unittest.main()
