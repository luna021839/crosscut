from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.source_scalar_cache import (
    PersistentSourceScalarCacheStorageUnavailableError,
)
from crosscut.vision import detect_scene_changes_in_video_cached


class CachedSceneDetectionTests(unittest.TestCase):
    def test_direct_derived_cache_directory_is_forwarded(self) -> None:
        derived_cache_directory = Path("mounted-derived-cache")
        with patch("crosscut.vision.PersistentSourceScalarCache") as cache_type:
            cache_type.return_value.get.return_value = {"changes": []}
            cache_type.return_value.diagnostics.return_value = {
                "cache_contract": "test"
            }
            changes, diagnostics = detect_scene_changes_in_video_cached(
                Path("source.mp4"),
                derived_cache_directory=derived_cache_directory,
            )

        self.assertEqual((), changes)
        self.assertTrue(diagnostics["analysis_reused"])
        self.assertEqual(
            derived_cache_directory,
            cache_type.call_args.kwargs["derived_cache_directory"],
        )
        self.assertEqual(
            derived_cache_directory,
            cache_type.call_args.kwargs["source_cache_directory"],
        )

    def test_direct_storage_error_does_not_trigger_live_scene_scan(self) -> None:
        error = PersistentSourceScalarCacheStorageUnavailableError(
            "persistent scalar cache storage is unavailable"
        )
        with patch(
            "crosscut.vision.PersistentSourceScalarCache",
            side_effect=error,
        ), patch(
            "crosscut.vision.detect_scene_changes_in_video"
        ) as live_scan:
            with self.assertRaises(
                PersistentSourceScalarCacheStorageUnavailableError
            ):
                detect_scene_changes_in_video_cached(
                    Path("source.mp4"),
                    source_cache_directory=Path("legacy-cache"),
                    derived_cache_directory=Path("mounted-derived-cache"),
                )

        live_scan.assert_not_called()

    def test_direct_non_storage_initialization_error_is_not_misclassified(
        self,
    ) -> None:
        with patch(
            "crosscut.vision.PersistentSourceScalarCache",
            side_effect=ValueError("invalid analyzer configuration"),
        ), patch(
            "crosscut.vision.detect_scene_changes_in_video"
        ) as live_scan:
            with self.assertRaisesRegex(
                ValueError,
                "invalid analyzer configuration",
            ):
                detect_scene_changes_in_video_cached(
                    Path("source.mp4"),
                    source_cache_directory=Path("legacy-cache"),
                    derived_cache_directory=Path("mounted-derived-cache"),
                )

        live_scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
