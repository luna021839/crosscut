from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.derived_cache_location import (
    LOCATION_CONTRACT, LOCATION_FILE, DerivedCacheLocationError,
    resolve_migrated_derived_cache,
)
from crosscut.request import AnalysisRequest


class DerivedCacheLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.mount = self.root / "mounted"
        self.mount.mkdir()
        (self.mount / "visual-v1").mkdir()
        self.image = self.root / "cache.sparsebundle"
        self.image.mkdir()
        self.uuid = "A730FA58-F435-4B7F-9D76-7B70EBC3AB14"
        self.marker = self.source / LOCATION_FILE
        self.value = dict(contract=LOCATION_CONTRACT, image_path=str(self.image),
                          mount_path=str(self.mount), volume_uuid=self.uuid)
        self.marker.write_text(json.dumps(self.value))
        self.volume = dict(FilesystemType="apfs", VolumeUUID=self.uuid, MountPoint=str(self.mount))
        self.images = {"images": [{"image-path": str(self.image),
                                  "system-entities": [{"mount-point": str(self.mount)}]}]}

    def mock_storage(self, volume=None, images=None):
        return patch("crosscut.derived_cache_location._plist_command", side_effect=[
            self.volume if volume is None else volume,
            self.images if images is None else images,
        ])

    def test_legacy_request_is_unchanged_and_creates_nothing(self):
        absent = self.root / "not-created"
        with patch("crosscut.derived_cache_location._plist_command") as command:
            self.assertIsNone(resolve_migrated_derived_cache(absent))
            self.assertEqual(resolve_migrated_derived_cache(absent, self.mount), self.mount)
            command.assert_not_called()
        self.assertFalse(absent.exists())

    def test_resolves_exact_volume_and_image(self):
        with self.mock_storage():
            self.assertEqual(resolve_migrated_derived_cache(self.source), self.mount)

    def test_request_uses_marker_without_swift_json_change(self):
        with self.mock_storage():
            request = AnalysisRequest.from_dict(dict(song="BANG BANG", source_mode="automatic",
                edit_mode="group", aspect_ratio="16:9", sources=[],
                source_cache_directory=str(self.source), output_directory=str(self.root / "out")))
        self.assertEqual(request.visual_feature_derived_cache_directory, self.mount)
        self.assertEqual(request.as_dict()["visual_feature_derived_cache_directory"], str(self.mount))
        with self.mock_storage():
            self.assertEqual(AnalysisRequest.from_dict(request.as_dict()), request)
        self.assertFalse((self.source / "derived").exists())

    def test_disconnected_mount_does_not_fall_back(self):
        self.value["mount_path"] = str(self.root / "disconnected")
        self.marker.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(DerivedCacheLocationError, "not mounted"):
            resolve_migrated_derived_cache(self.source)
        self.assertFalse((self.source / "derived").exists())
        self.assertFalse((self.root / "disconnected").exists())

    def test_wrong_uuid_filesystem_or_mount_rejected(self):
        for key, value in [("VolumeUUID", "other"), ("FilesystemType", "exfat"), ("MountPoint", "/wrong")]:
            with self.subTest(key=key), self.mock_storage(volume={**self.volume, key: value}):
                with self.assertRaisesRegex(DerivedCacheLocationError, "identity mismatch"):
                    resolve_migrated_derived_cache(self.source)

    def test_wrong_backing_image_rejected(self):
        with self.mock_storage(images={"images": []}):
            with self.assertRaisesRegex(DerivedCacheLocationError, "backing image"):
                resolve_migrated_derived_cache(self.source)

    def test_explicit_conflict_rejected(self):
        with self.assertRaisesRegex(DerivedCacheLocationError, "conflicts"):
            resolve_migrated_derived_cache(self.source, self.root / "other")

    def test_matching_explicit_location_accepted(self):
        with self.mock_storage():
            self.assertEqual(resolve_migrated_derived_cache(self.source, self.mount), self.mount)

    def test_malformed_marker_rejected(self):
        for value in ["null", "[]", "{}", "bad json", "x" * 16385]:
            with self.subTest(value=value[:10]):
                self.marker.write_text(value)
                with self.assertRaises(DerivedCacheLocationError):
                    resolve_migrated_derived_cache(self.source)

    def test_marker_symlink_rejected(self):
        original = self.source / "original.json"
        self.marker.rename(original)
        self.marker.symlink_to(original)
        with self.assertRaisesRegex(DerivedCacheLocationError, "symlink"):
            resolve_migrated_derived_cache(self.source)

    def test_missing_cache_tree_rejected(self):
        (self.mount / "visual-v1").rename(self.mount / "moved")
        with self.mock_storage():
            with self.assertRaisesRegex(DerivedCacheLocationError, "data is missing"):
                resolve_migrated_derived_cache(self.source)


if __name__ == "__main__":
    unittest.main()
