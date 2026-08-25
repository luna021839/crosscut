from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.direct_member_visual import _VideoDescriptorProvider
from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import (
    ShotType,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.visual_feature_cache import (
    AnalyzerFingerprint,
    CanonicalFrameIdentity,
    DescriptorRequestIdentity,
    PersistentVisualFeatureCache,
    build_analyzer_fingerprint,
)
from crosscut.visual_score import (
    FrameVisualDescriptor,
    _VideoSampler,
    build_visual_timeline,
)


def _analyzer(salt: str = "a") -> AnalyzerFingerprint:
    return AnalyzerFingerprint(
        digest=salt * 64,
        components={
            "descriptor_schema": "test_numeric_descriptor_v1",
            "model_asset_sha256_or_state": {"test": "present"},
            "runtime": {"opencv": cv2.__version__},
            "code_sha256": {"test.py": salt * 64},
        },
    )


def _request(
    *,
    frame_index: int = 30,
    frame_digest: str = "1" * 64,
    following_index: int | None = None,
    following_digest: str | None = None,
    subject: bool = True,
    formation: bool = False,
) -> DescriptorRequestIdentity:
    return DescriptorRequestIdentity(
        primary=CanonicalFrameIdentity(frame_index, 30, 1),
        following=(
            CanonicalFrameIdentity(following_index, 30, 1)
            if following_index is not None
            else None
        ),
        primary_working_frame_sha256=frame_digest,
        following_working_frame_sha256=following_digest,
        include_subject_geometry=subject,
        include_formation_pose=formation,
    )


def _descriptor() -> FrameVisualDescriptor:
    formation = FormationPersonDescriptor(
        bbox_xywh=(0.40, 0.15, 0.20, 0.70),
        centroid_xy=(0.50, 0.50),
        detection_confidence=0.91,
        upper_body_keypoints_xyc=(
            (-0.2, -0.3, 0.9),
            (0.2, -0.3, 0.9),
            (-0.3, -0.1, 0.9),
            (0.3, -0.1, 0.9),
            (-0.4, 0.1, 0.9),
            (0.4, 0.1, 0.9),
            (-0.1, 0.4, 0.9),
            (0.1, 0.4, 0.9),
        ),
    )
    return FrameVisualDescriptor(
        quality_score=0.81,
        composition_score=0.72,
        shot_scale=0.44,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        structure_embedding=(0.1,) * 96,
        motion=(0.02, -0.03),
        primary_face_bbox=(0.43, 0.20, 0.14, 0.20),
        color_embedding=(0.2,) * 8,
        face_detector_backend="yunet",
        primary_face_landmarks_xy=(
            (0.46, 0.26),
            (0.54, 0.26),
            (0.50, 0.30),
            (0.47, 0.35),
            (0.53, 0.35),
        ),
        primary_face_head_roll_radians=0.01,
        primary_face_eye_scale=0.08,
        primary_face_geometry=(
            0.46,
            0.26,
            0.54,
            0.26,
            0.08,
            0.01,
            0.0,
            0.3,
            0.7,
            0.0,
        ),
        pose_embedding=(0.1,) * 60,
        pose_confidence=0.88,
        detected_person_count=1,
        person_layout_embedding=(0.2,) * 134,
        formation_people=(formation,),
        pose_backend="opencv_zoo_mediapipe_pose_2023mar",
        silhouette_embedding=(0.2,) * 121,
        silhouette_bbox_xywh=(0.35, 0.10, 0.30, 0.82),
        silhouette_centroid_xy=(0.50, 0.51),
        silhouette_area_ratio=0.22,
        silhouette_confidence=0.89,
        silhouette_backend="opencv_zoo_pphumanseg_2023mar",
        foreground_obstruction_score=0.0,
        highlight_clipping_ratio=0.0,
    )


def _group_descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.75,
        shot_scale=0.18,
        shot_type=ShotType.GROUP_WIDE,
        face_count=0,
        structure_embedding=(0.1,) * 96,
        motion=(0.0, 0.0),
        color_embedding=(0.2,) * 8,
        face_detector_backend="none",
        pose_embedding=(0.1,) * 96,
        pose_confidence=0.0,
        detected_person_count=0,
        person_layout_embedding=(),
        pose_backend="structure_hog_fallback",
        silhouette_embedding=(),
        silhouette_area_ratio=0.0,
        silhouette_confidence=0.0,
        silhouette_backend="none",
    )


class _FakeCapture:
    def __init__(self, _path: str, *, frame_rate: float = 30.0) -> None:
        self.frame_rate = frame_rate
        self.position = 0
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.frame_rate
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        elif property_id == cv2.CAP_PROP_POS_MSEC:
            self.position = int(round(value / 1000.0 * self.frame_rate))
        return True

    def read(self):  # type: ignore[no-untyped-def]
        value = self.position % 251
        return (True, np.full((12, 20, 3), value, dtype=np.uint8))

    def release(self) -> None:
        self.opened = False


class VisualFeatureCacheTests(unittest.TestCase):
    def test_canonical_frame_identity_removes_rounded_millisecond_collision(
        self,
    ) -> None:
        first = CanonicalFrameIdentity.from_seconds(1.0001, frame_rate=2000.0)
        second = CanonicalFrameIdentity.from_seconds(1.0004, frame_rate=2000.0)
        self.assertEqual(round(1.0001 * 1000), round(1.0004 * 1000))
        self.assertNotEqual(first, second)
        self.assertEqual((first.frame_index, second.frame_index), (2000, 2001))

    def test_analyzer_fingerprint_changes_with_runtime_and_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            code = root / "analyzer.py"
            code.write_text("version = 1\n", encoding="utf-8")
            first = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "1"},
            )
            second = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "2"},
            )
            code.write_text("version = 2\n", encoding="utf-8")
            third = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "1"},
            )
            self.assertEqual(len({first.digest, second.digest, third.digest}), 3)

    def test_atomic_round_trip_binds_source_frame_options_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            request = _request(
                following_index=31,
                following_digest="2" * 64,
                formation=True,
            )
            payload = {
                "quality": 0.8,
                "vector": [0.1, 0.2, 0.3],
                "backend": "test",
            }
            self.assertTrue(cache.put(request, payload))
            self.assertEqual(cache.get(request), payload)
            stale_analyzer = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer("b"),
            )
            self.assertIsNone(stale_analyzer.get(request))
            self.assertIsNone(
                cache.get(
                    _request(
                        following_index=31,
                        following_digest="3" * 64,
                        formation=True,
                    )
                )
            )
            self.assertIsNone(
                cache.get(
                    _request(
                        following_index=31,
                        following_digest="2" * 64,
                        formation=False,
                    )
                )
            )
            records = tuple((root / "cache" / "derived").rglob("*.json"))
            self.assertEqual(len(records), 1)
            raw_record = records[0].read_text(encoding="utf-8")
            self.assertNotIn(str(source), raw_record)
            self.assertNotIn("source-content-v1", raw_record)
            record = json.loads(raw_record)
            self.assertFalse(record["privacy"]["raw_frames_persisted"])
            self.assertFalse(
                record["privacy"]["face_identity_embeddings_persisted"]
            )

            # A valid JSON edit without a matching checksum is corruption, not
            # evidence that can flow into timeline scoring.
            record["numeric_features"]["quality"] = 0.1
            records[0].write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(cache.get(request))
            self.assertEqual(cache.diagnostics()["invalid_record_count"], 1)

    def test_changed_source_never_reuses_old_content_address(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"AAAA")
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            request = _request()
            self.assertTrue(cache.put(request, {"value": 1.0}))
            original_mtime = source.stat().st_mtime_ns
            source.write_bytes(b"BBBB")
            os.utime(source, ns=(original_mtime, original_mtime))
            self.assertIsNone(cache.get(request))
            self.assertTrue(cache.diagnostics()["disabled_by_source_change"])
            replacement = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            self.assertNotEqual(cache.source_sha256, replacement.source_sha256)
            self.assertIsNone(replacement.get(request))

    def test_sampler_reuses_full_descriptor_without_persisting_sface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"fake-video-source")
            cache_root = root / "cache"
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path),
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_descriptor(),
            ) as describe:
                first = _VideoSampler(
                    source,
                    feature_cache_directory=cache_root,
                )
                cold = first.feature_descriptor_at(
                    1.0,
                    following_seconds=1.08,
                    include_formation_pose=True,
                )
                first.close()
                second = _VideoSampler(
                    source,
                    feature_cache_directory=cache_root,
                )
                warm = second.feature_descriptor_at(
                    1.0,
                    following_seconds=1.08,
                    include_formation_pose=True,
                )
                diagnostics = second.feature_cache_diagnostics()
                second.close()
            self.assertEqual(cold, warm)
            self.assertEqual(describe.call_count, 1)
            assert diagnostics is not None
            self.assertEqual(diagnostics["hit_count"], 1)
            records = tuple((cache_root / "derived").rglob("*.json"))
            self.assertTrue(records)
            joined = b"".join(path.read_bytes() for path in records)
            self.assertNotIn(b"SFACE_PRIVATE_SENTINEL", joined)
            self.assertNotIn(source.read_bytes(), joined)

    def test_direct_member_provider_reuses_interrupted_run_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"fake-video-source")
            cache_root = root / "cache"
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path),
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_descriptor(),
            ) as describe:
                cold_provider = _VideoDescriptorProvider(
                    (source,),
                    feature_cache_directory=cache_root,
                )
                cold = cold_provider(0, source, 1.0)
                cold_diagnostics = cold_provider.cache_diagnostics()
                cold_provider.close()

                warm_provider = _VideoDescriptorProvider(
                    (source,),
                    feature_cache_directory=cache_root,
                )
                warm = warm_provider(0, source, 1.0)
                warm_diagnostics = warm_provider.cache_diagnostics()
                warm_provider.close()

            self.assertEqual(cold, warm)
            self.assertEqual(describe.call_count, 1)
            cold_persistent = cold_diagnostics[
                "persistent_visual_feature_cache"
            ]
            warm_persistent = warm_diagnostics[
                "persistent_visual_feature_cache"
            ]
            self.assertEqual(cold_persistent["write_count"], 1)
            self.assertEqual(warm_persistent["hit_count"], 1)
            self.assertTrue(
                warm_persistent["interrupted_run_records_reusable"]
            )

    def test_sampler_frame_lru_uses_canonical_frames_not_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.mp4"
            source.write_bytes(b"fake-video-source")
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path, frame_rate=2000.0),
            ):
                sampler = _VideoSampler(source)
                first = sampler.frame_at(1.0001)
                second = sampler.frame_at(1.0004)
                sampler.close()
            assert first is not None and second is not None
            self.assertFalse(np.array_equal(first, second))

    def test_warm_features_rebuild_identical_timeline_instead_of_caching_dp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = (root / "a.mp4", root / "b.mp4")
            sources[0].write_bytes(b"fake-source-a")
            sources[1].write_bytes(b"fake-source-b")
            arguments = {
                "sources": sources,
                "canonical_spans": tuple(
                    (float(index + 1), float(index + 2)) for index in range(8)
                ),
                "audio_offsets": (0.0, 0.0),
                "sync_scores": (0.9, 0.8),
                "scene_cut_times": ((), ()),
                "preferred_source_indices": (0, 1, 0, 1, 0, 1, 0, 1),
                "feature_cache_directory": root / "cache",
            }

            def select_again(candidates_by_segment, **_kwargs):  # type: ignore[no-untyped-def]
                choices = tuple(
                    TimelineChoice(
                        candidate=candidates[index % len(candidates)],
                        emission_score=1.0,
                        transition_score=0.0,
                        cumulative_score=float(index + 1),
                        used_group_wide_fallback=False,
                    )
                    for index, candidates in enumerate(candidates_by_segment)
                )
                return (
                    (
                        TimelineResult(
                            mode=TimelineMode.GENERAL,
                            target_member=None,
                            choices=choices,
                            total_score=float(len(choices)),
                        ),
                    ),
                    ("test",),
                )

            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path),
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_group_descriptor(),
            ) as describe, patch(
                "crosscut.visual_score._select_safe_cadence_candidates",
                side_effect=select_again,
            ) as select_timeline:
                cold = build_visual_timeline(**arguments)
                cold_call_count = describe.call_count
                describe.reset_mock()
                warm = build_visual_timeline(**arguments)
                warm_call_count = describe.call_count
            self.assertGreater(cold_call_count, 0)
            self.assertEqual(warm_call_count, 0)
            self.assertEqual(select_timeline.call_count, 2)
            self.assertEqual(cold.to_dict(), warm.to_dict())


if __name__ == "__main__":
    unittest.main()
