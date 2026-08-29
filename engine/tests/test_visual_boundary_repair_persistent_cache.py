from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.timeline import ShotType
from crosscut.visual_boundary_repair import (
    BoundaryDescriptorCache,
    repair_visual_timeline_boundaries,
)
from crosscut.visual_feature_cache import AnalyzerFingerprint
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualTimelineChoice,
    VisualTimelinePlan,
)


def _analyzer() -> AnalyzerFingerprint:
    return AnalyzerFingerprint(
        digest="a" * 64,
        components={
            "descriptor_schema": "boundary-repair-cache-test-v1",
            "model_asset_sha256_or_state": {"test": "present"},
            "runtime": {"opencv": cv2.__version__},
            "code_sha256": {"test.py": "b" * 64},
        },
    )


def _descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
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
        foreground_obstruction_score=0.0,
        highlight_clipping_ratio=0.0,
    )


def _choice(index: int, source_index: int) -> VisualTimelineChoice:
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source_index,
        canonical_master_start=float(index),
        canonical_master_end=float(index + 1),
        source_local_start=float(index),
        source_local_end=float(index + 1),
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.5,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source_index}:shot",
    )


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self.opened = True
        self.position = 0
        self.read_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            self.position = int(round(value / 1000.0 * 30.0))
        return True

    def read(self):  # type: ignore[no-untyped-def]
        self.read_count += 1
        value = self.position % 251
        return True, np.full((12, 20, 3), value, dtype=np.uint8)

    def release(self) -> None:
        self.opened = False


class VisualBoundaryRepairPersistentCacheTests(unittest.TestCase):
    def test_second_repair_run_reuses_persisted_full_descriptors_decode_free(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = (root / "source-a.mp4", root / "source-b.mp4")
            sources[0].write_bytes(b"source-a-content")
            sources[1].write_bytes(b"source-b-content")
            cache_root = root / "cache"
            timeline = VisualTimelinePlan(
                choices=(_choice(0, 0), _choice(1, 1)),
                optimizer_score=2.0,
            )
            captures: list[_FakeCapture] = []

            def capture(path: str) -> _FakeCapture:
                value = _FakeCapture(path)
                captures.append(value)
                return value

            arguments = {
                "visual_timeline": timeline,
                "sources": sources,
                "audio_offsets": (0.0, 0.0),
                "scene_cut_times": ((), ()),
                "feature_cache_directory": cache_root,
            }
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=capture,
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_descriptor(),
            ) as describe:
                repair_visual_timeline_boundaries(
                    **arguments,
                    descriptor_cache=BoundaryDescriptorCache(),
                )
                cold_describe_count = describe.call_count
                repair_visual_timeline_boundaries(
                    **arguments,
                    descriptor_cache=BoundaryDescriptorCache(),
                )

            self.assertGreater(cold_describe_count, 0)
            self.assertEqual(describe.call_count, cold_describe_count)
            self.assertEqual(len(captures), 4)
            self.assertGreater(
                sum(item.read_count for item in captures[:2]),
                0,
            )
            self.assertEqual(
                sum(item.read_count for item in captures[2:]),
                0,
            )
            records = tuple((cache_root / "derived").rglob("*.json"))
            self.assertTrue(records)
            persisted = b"".join(path.read_bytes() for path in records)
            self.assertNotIn(b"sface", persisted.lower())
            self.assertNotIn(sources[0].read_bytes(), persisted)
            self.assertNotIn(sources[1].read_bytes(), persisted)

    def test_explicit_provider_rejects_persistent_cache_directory(self) -> None:
        plan = VisualTimelinePlan(
            choices=(_choice(0, 0), _choice(1, 1)),
            optimizer_score=2.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "feature_cache_directory cannot wrap an explicit descriptor_provider",
        ):
            repair_visual_timeline_boundaries(
                visual_timeline=plan,
                sources=(Path("a"), Path("b")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                descriptor_provider=lambda _source, _seconds: _descriptor(),
                feature_cache_directory=Path("cache"),
            )


if __name__ == "__main__":
    unittest.main()
