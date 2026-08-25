from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive_causal_sensing_installation import (
    IVE_CAUSAL_SENSING_INSTALLATION_ENV,
    IVECausalSensingInstallationError,
    IVECausalSensingInstallationNotFoundError,
    discover_ive_causal_sensing_installation,
    load_ive_causal_sensing_installation,
    register_ive_causal_sensing_installation,
)
from tests.test_ive_causal_sensing_media_job import _gallery_artifact


class IVECausalSensingInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gallery = _gallery_artifact(self.root / "gallery")
        self.pointer = self.root / "installation" / "active-private-gallery.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _register(self):
        return register_ive_causal_sensing_installation(
            gallery_artifact_path=self.gallery.path,
            gallery_content_sha256=self.gallery.content_sha256,
            pointer_path=self.pointer,
        )

    def test_register_load_and_idempotent_replay_bind_exact_gallery(self) -> None:
        first = self._register()
        second = self._register()
        restored = load_ive_causal_sensing_installation(
            self.pointer,
            expected_pointer_file_sha256=first.pointer_file_sha256,
        )

        self.assertEqual(first, second)
        self.assertEqual(restored, first)
        self.assertEqual(
            restored.gallery_artifact_path,
            self.gallery.path.resolve(strict=True),
        )
        self.assertEqual(
            restored.gallery_content_sha256,
            self.gallery.content_sha256,
        )
        payload = json.loads(self.pointer.read_text(encoding="utf-8"))
        self.assertFalse(payload["privacy"]["face_vectors_embedded"])
        encoded = self.pointer.read_text(encoding="utf-8")
        self.assertNotIn("prototypes", encoded)
        self.assertNotIn('"embedding": [', encoded)

    def test_pointer_file_hash_tamper_and_gallery_rebinding_fail_closed(self) -> None:
        installed = self._register()
        with self.assertRaisesRegex(
            IVECausalSensingInstallationError,
            "pointer file SHA-256 mismatch",
        ):
            load_ive_causal_sensing_installation(
                self.pointer,
                expected_pointer_file_sha256="0" * 64,
            )

        payload = json.loads(self.pointer.read_text(encoding="utf-8"))
        payload["gallery"]["content_sha256"] = "0" * 64
        unsigned = dict(payload)
        unsigned.pop("installation_sha256")
        payload["installation_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.pointer.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            IVECausalSensingInstallationError,
            "gallery failed strict replay",
        ):
            load_ive_causal_sensing_installation(self.pointer)
        self.assertRegex(installed.installation_sha256, r"^[0-9a-f]{64}$")

    def test_symlink_pointer_and_different_immutable_install_are_rejected(self) -> None:
        self._register()
        alias = self.root / "pointer-alias.json"
        alias.symlink_to(self.pointer)
        with self.assertRaisesRegex(
            IVECausalSensingInstallationError,
            "non-symlink regular file",
        ):
            load_ive_causal_sensing_installation(alias)

        second_gallery = _gallery_artifact(self.root / "gallery-copy")
        self.assertEqual(second_gallery.content_sha256, self.gallery.content_sha256)
        # Same canonical gallery remains idempotent even if referred to by an
        # independently materialized content-addressed copy.
        with self.assertRaisesRegex(
            IVECausalSensingInstallationError,
            "different IVE causal-sensing installation",
        ):
            register_ive_causal_sensing_installation(
                gallery_artifact_path=second_gallery.path,
                gallery_content_sha256=second_gallery.content_sha256,
                pointer_path=self.pointer,
            )

    def test_explicit_environment_discovery_is_mandatory_and_exact(self) -> None:
        installed = self._register()
        with patch.dict(
            os.environ,
            {IVE_CAUSAL_SENSING_INSTALLATION_ENV: str(self.pointer)},
        ):
            discovered = discover_ive_causal_sensing_installation()
        self.assertEqual(discovered.installation_sha256, installed.installation_sha256)

        missing = self.root / "missing.json"
        with patch.dict(
            os.environ,
            {IVE_CAUSAL_SENSING_INSTALLATION_ENV: str(missing)},
        ):
            with self.assertRaises(IVECausalSensingInstallationNotFoundError):
                discover_ive_causal_sensing_installation()
        with patch.dict(
            os.environ,
            {IVE_CAUSAL_SENSING_INSTALLATION_ENV: "   "},
        ):
            with self.assertRaisesRegex(
                IVECausalSensingInstallationError,
                "cannot be blank",
            ):
                discover_ive_causal_sensing_installation()

    def test_default_discovery_rejects_absence_and_ambiguity(self) -> None:
        first = self.pointer
        second = self.root / "other" / "active-private-gallery.json"
        self._register()
        second.parent.mkdir(parents=True)
        second.write_bytes(first.read_bytes())
        with patch.dict(os.environ, {}, clear=True), patch(
            "crosscut.ive_causal_sensing_installation."
            "default_ive_causal_sensing_installation_paths",
            return_value=(first, second),
        ):
            with self.assertRaisesRegex(
                IVECausalSensingInstallationError,
                "multiple",
            ):
                discover_ive_causal_sensing_installation()
        first.unlink()
        second.unlink()
        with patch.dict(os.environ, {}, clear=True), patch(
            "crosscut.ive_causal_sensing_installation."
            "default_ive_causal_sensing_installation_paths",
            return_value=(first, second),
        ):
            with self.assertRaises(IVECausalSensingInstallationNotFoundError):
                discover_ive_causal_sensing_installation()


if __name__ == "__main__":
    unittest.main()
