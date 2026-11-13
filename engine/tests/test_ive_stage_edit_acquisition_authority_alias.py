from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crosscut.ive_stage_edit_training_corpus_job as corpus_job
from crosscut.ive_stage_edit_training_corpus_job import (
    IVE_STAGE_EDIT_ACQUISITION_AUTHORITY_ALIAS_CONTRACT,
    IVEStageEditTrainingCorpusError,
    _verify_byte_identical_acquisition_authority_alias,
)


class AcquisitionAuthorityAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _encoded(payload: object) -> bytes:
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def _write(self, name: str, encoded: bytes) -> Path:
        path = self.root / name
        path.write_bytes(encoded)
        return path

    @staticmethod
    def _pointers(first: Path, second: Path, digest: str) -> tuple[dict, ...]:
        first_pointer = {"path": str(first), "file_sha256": digest}
        second_pointer = {"path": str(second), "file_sha256": digest}
        return (
            first_pointer,
            first_pointer,
            second_pointer,
            first_pointer,
            second_pointer,
        )

    def test_byte_identical_regular_paths_emit_explicit_provenance(self) -> None:
        encoded = self._encoded({"schema_version": 1, "selected": ["a", "b"]})
        first = self._write("authority-a.json", encoded)
        second = self._write("authority-b.json", encoded)
        digest = hashlib.sha256(encoded).hexdigest()

        observation, provenance = (
            _verify_byte_identical_acquisition_authority_alias(
                self._pointers(first, second, digest)
            )
        )

        self.assertEqual(observation.encoded, encoded)
        self.assertEqual(
            provenance["contract"],
            IVE_STAGE_EDIT_ACQUISITION_AUTHORITY_ALIAS_CONTRACT,
        )
        self.assertEqual(provenance["authority_count"], 2)
        self.assertEqual(
            [item["path"] for item in provenance["authorities"]],
            sorted((str(first), str(second))),
        )
        self.assertEqual(
            {item["file_sha256"] for item in provenance["authorities"]},
            {digest},
        )
        self.assertTrue(provenance["verification"]["byte_identical"])
        self.assertTrue(
            provenance["verification"]["stable_pre_post_stat_replay"]
        )

    def test_different_bytes_are_rejected_with_one_declared_digest(self) -> None:
        first_bytes = self._encoded({"schema_version": 1, "selected": ["a"]})
        second_bytes = self._encoded({"schema_version": 1, "selected": ["b"]})
        first = self._write("authority-a.json", first_bytes)
        second = self._write("authority-b.json", second_bytes)
        digest = hashlib.sha256(first_bytes).hexdigest()

        with self.assertRaisesRegex(
            IVEStageEditTrainingCorpusError,
            "not byte-identical JSON",
        ):
            _verify_byte_identical_acquisition_authority_alias(
                self._pointers(first, second, digest)
            )

    def test_symlink_alias_is_rejected(self) -> None:
        encoded = self._encoded({"schema_version": 1, "selected": ["a"]})
        first = self._write("authority-a.json", encoded)
        second = self.root / "authority-link.json"
        second.symlink_to(first)
        digest = hashlib.sha256(encoded).hexdigest()

        with self.assertRaisesRegex(
            IVEStageEditTrainingCorpusError,
            "non-symlink regular file",
        ):
            _verify_byte_identical_acquisition_authority_alias(
                self._pointers(first, second, digest)
            )

    def test_tamper_after_success_is_rejected_on_strict_replay(self) -> None:
        encoded = self._encoded({"schema_version": 1, "selected": ["a"]})
        first = self._write("authority-a.json", encoded)
        second = self._write("authority-b.json", encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        pointers = self._pointers(first, second, digest)
        _verify_byte_identical_acquisition_authority_alias(pointers)

        second.write_bytes(
            self._encoded({"schema_version": 1, "selected": ["changed"]})
        )
        with self.assertRaisesRegex(
            IVEStageEditTrainingCorpusError,
            "not byte-identical JSON",
        ):
            _verify_byte_identical_acquisition_authority_alias(pointers)

    def test_identity_race_between_read_passes_is_rejected(self) -> None:
        encoded = self._encoded({"schema_version": 1, "selected": ["a"]})
        first = self._write("authority-a.json", encoded)
        second = self._write("authority-b.json", encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        real_observer = corpus_job._stable_json_observation
        call_count = 0

        def racing_observer(*args, **kwargs):
            nonlocal call_count
            observation = real_observer(*args, **kwargs)
            call_count += 1
            if call_count == 2:
                current = first.stat()
                os.utime(
                    first,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
                )
            return observation

        with patch.object(
            corpus_job,
            "_stable_json_observation",
            side_effect=racing_observer,
        ):
            with self.assertRaisesRegex(
                IVEStageEditTrainingCorpusError,
                "changed between verification passes",
            ):
                _verify_byte_identical_acquisition_authority_alias(
                    self._pointers(first, second, digest)
                )

    def test_more_than_two_distinct_authority_paths_remains_fail_closed(self) -> None:
        encoded = self._encoded({"schema_version": 1, "selected": ["a"]})
        paths = tuple(
            self._write(f"authority-{index}.json", encoded) for index in range(3)
        )
        digest = hashlib.sha256(encoded).hexdigest()
        pointers = tuple(
            {"path": str(path), "file_sha256": digest} for path in paths
        )

        with self.assertRaisesRegex(
            IVEStageEditTrainingCorpusError,
            "exactly two distinct paths",
        ):
            _verify_byte_identical_acquisition_authority_alias(pointers)


if __name__ == "__main__":
    unittest.main()
