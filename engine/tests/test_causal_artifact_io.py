from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from crosscut import causal_artifact_io as artifact_io


class CausalArtifactIOTests(unittest.TestCase):
    def test_canonical_stream_hash_is_schema_compatible(self) -> None:
        payload = {"IVE": ["가을", 0.25, {"unknown": False}]}
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest, size = artifact_io.streaming_canonical_json_identity(payload)
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
        self.assertEqual(len(encoded), size)

    def test_stream_publish_load_and_no_clobber_collision(self) -> None:
        payload = {"contract": "scalar_v1", "records": [1, 2, 3]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / f"{digest}.json"
            identity = artifact_io.publish_json_content_addressed(
                destination,
                payload,
                content_sha256=digest,
            )
            self.assertEqual(payload, artifact_io.load_bounded_json_artifact(destination))
            self.assertEqual(destination.stat().st_size, identity.size_bytes)
            self.assertEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(),
                identity.file_sha256,
            )
            before = destination.read_bytes()
            replacement = {"contract": "scalar_v1", "records": [9]}
            with self.assertRaisesRegex(ValueError, "different bytes"):
                artifact_io.publish_json_content_addressed(
                    destination,
                    replacement,
                    content_sha256=digest,
                )
            self.assertEqual(before, destination.read_bytes())
            self.assertFalse(tuple(destination.parent.glob("*.tmp")))
            self.assertFalse(tuple(destination.parent.glob("*.publish-lock")))

    def test_over_cap_write_and_load_fail_closed(self) -> None:
        payload = {"records": ["x" * 4096]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / f"{digest}.json"
            with self.assertRaisesRegex(ValueError, "finite byte cap"):
                artifact_io.publish_json_content_addressed(
                    destination,
                    payload,
                    content_sha256=digest,
                    maximum_bytes=1024,
                )
            self.assertFalse(destination.exists())
            self.assertFalse(tuple(root.glob("*.tmp")))
            destination.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the finite cap"):
                artifact_io.load_bounded_json_artifact(
                    destination,
                    maximum_bytes=1024,
                )

    def test_existing_fast_path_rejects_swap_after_exact_compare(self) -> None:
        payload = {"contract": "scalar_v1", "records": [1, 2, 3]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / f"{digest}.json"
            artifact_io.publish_json_content_addressed(
                destination,
                payload,
                content_sha256=digest,
            )
            replacement = root / "replacement.json"
            replacement.write_text('{"swapped":true}', encoding="utf-8")
            real_equal = artifact_io._stream_files_equal
            swapped = False

            def compare_then_swap(*args: object, **kwargs: object) -> bool:
                nonlocal swapped
                equal = real_equal(*args, **kwargs)
                if not swapped:
                    os.replace(replacement, destination)
                    swapped = True
                return equal

            with patch.object(
                artifact_io,
                "_stream_files_equal",
                side_effect=compare_then_swap,
            ):
                with self.assertRaisesRegex(ValueError, "readback differs"):
                    artifact_io.publish_json_content_addressed(
                        destination,
                        payload,
                        content_sha256=digest,
                    )

    def test_loader_rejects_pathname_replacement_during_parse(self) -> None:
        payload = {"contract": "scalar_v1", "records": [1, 2, 3]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / f"{digest}.json"
            artifact_io.publish_json_content_addressed(
                destination,
                payload,
                content_sha256=digest,
            )
            replacement = root / "replacement.json"
            replacement.write_text('{"swapped":true}', encoding="utf-8")
            real_load = json.load

            def load_then_swap(*args: object, **kwargs: object) -> object:
                loaded = real_load(*args, **kwargs)
                os.replace(replacement, destination)
                return loaded

            with patch.object(artifact_io.json, "load", side_effect=load_then_swap):
                with self.assertRaisesRegex(ValueError, "pathname changed"):
                    artifact_io.load_bounded_json_artifact(destination)

    def test_publication_uses_same_filesystem_replace_path(self) -> None:
        payload = {"contract": "scalar_v1", "records": [1]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / f"{digest}.json"
            real_replace = os.replace
            with patch.object(
                artifact_io.os,
                "replace",
                wraps=real_replace,
            ) as replace:
                artifact_io.publish_json_content_addressed(
                    destination,
                    payload,
                    content_sha256=digest,
                )
            replace.assert_called_once()
            source, target = replace.call_args.args
            self.assertEqual(Path(source).parent, Path(target).parent)

    def test_stale_lock_is_never_stolen_or_silently_removed(self) -> None:
        payload = {"contract": "scalar_v1", "records": [1]}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / f"{digest}.json"
            lock = root / f".{digest}.publish-lock"
            marker = b"stale-owner\n"
            lock.write_bytes(marker)
            with self.assertRaisesRegex(ValueError, "lock is already held"):
                artifact_io.publish_json_content_addressed(
                    destination,
                    payload,
                    content_sha256=digest,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(marker, lock.read_bytes())
            self.assertFalse(tuple(root.glob("*.tmp")))

    def test_concurrent_identical_publish_is_retry_safe_and_exact(self) -> None:
        payload = {"contract": "scalar_v1", "records": list(range(1000))}
        digest, _ = artifact_io.streaming_canonical_json_identity(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / f"{digest}.json"
            barrier = threading.Barrier(2)
            successes: list[str] = []
            errors: list[BaseException] = []

            def publish() -> None:
                try:
                    barrier.wait()
                    identity = artifact_io.publish_json_content_addressed(
                        destination,
                        payload,
                        content_sha256=digest,
                    )
                    successes.append(identity.file_sha256)
                except BaseException as exc:  # capture the concurrent worker
                    errors.append(exc)

            workers = tuple(threading.Thread(target=publish) for _ in range(2))
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self.assertTrue(successes)
            self.assertTrue(
                all("lock is already held" in str(error) for error in errors)
            )
            # A loser never stole or truncated the winner. Retrying after the
            # lock owner exits deterministically reuses the exact bytes.
            retried = artifact_io.publish_json_content_addressed(
                destination,
                payload,
                content_sha256=digest,
            )
            self.assertEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(),
                retried.file_sha256,
            )
            self.assertEqual(payload, artifact_io.load_bounded_json_artifact(destination))
            self.assertFalse(tuple(root.glob("*.tmp")))
            self.assertFalse(tuple(root.glob("*.publish-lock")))


if __name__ == "__main__":
    unittest.main()
