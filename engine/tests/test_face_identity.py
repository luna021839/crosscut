from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crosscut.face_identity import (
    SFACE_EMBEDDING_SIZE,
    SFACE_SHA256,
    SFaceEmbedder,
    cosine_similarity,
    sface_model_asset_status,
)


class SFaceIdentitySafetyTests(unittest.TestCase):
    @staticmethod
    def synthetic_face() -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        tuple[tuple[float, float], ...],
    ]:
        y, x = np.mgrid[0:160, 0:160]
        frame = np.stack(
            (
                (40 + x * 0.7).clip(0, 255),
                (35 + y * 0.8).clip(0, 255),
                (70 + (x + y) * 0.45).clip(0, 255),
            ),
            axis=2,
        ).astype(np.uint8)
        return (
            frame,
            (0.20, 0.12, 0.60, 0.76),
            (
                (0.38, 0.39),
                (0.62, 0.39),
                (0.50, 0.53),
                (0.40, 0.67),
                (0.60, 0.67),
            ),
        )

    def test_bundled_sface_has_exact_digest_and_runs_real_inference(self) -> None:
        status = sface_model_asset_status(verify_load=True)

        self.assertTrue(status.present)
        self.assertTrue(status.digest_valid)
        self.assertTrue(status.loadable)
        self.assertEqual(status.actual_sha256, SFACE_SHA256)
        self.assertEqual(status.embedding_size, SFACE_EMBEDDING_SIZE)

    def test_missing_and_corrupt_assets_never_attempt_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.onnx"
            missing_status = sface_model_asset_status(
                model_path=missing,
                verify_load=True,
            )
            corrupt = Path(directory) / "corrupt.onnx"
            corrupt.write_bytes(b"not an onnx model")
            corrupt_status = sface_model_asset_status(
                model_path=corrupt,
                verify_load=True,
            )

        self.assertFalse(missing_status.present)
        self.assertFalse(missing_status.digest_valid)
        self.assertIsNone(missing_status.loadable)
        self.assertTrue(corrupt_status.present)
        self.assertFalse(corrupt_status.digest_valid)
        self.assertIsNone(corrupt_status.loadable)

    def test_same_aligned_frame_produces_normalized_stable_embedding(self) -> None:
        frame, bbox, landmarks = self.synthetic_face()
        embedder = SFaceEmbedder()

        first = embedder.embedding(
            frame,
            bbox_xywh=bbox,
            landmarks_xy=landmarks,
        )
        second = embedder.embedding(
            frame.copy(),
            bbox_xywh=bbox,
            landmarks_xy=landmarks,
        )

        self.assertEqual(len(first), SFACE_EMBEDDING_SIZE)
        self.assertTrue(all(math.isfinite(value) for value in first))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)
        self.assertAlmostEqual(cosine_similarity(first, second) or 0.0, 1.0)

    def test_cosine_similarity_rejects_invalid_vectors(self) -> None:
        self.assertIsNone(cosine_similarity((), ()))
        self.assertIsNone(cosine_similarity((1.0,), (1.0, 2.0)))
        self.assertIsNone(cosine_similarity((0.0, 0.0), (1.0, 0.0)))
        self.assertIsNone(cosine_similarity((float("nan"),), (1.0,)))
        self.assertAlmostEqual(
            cosine_similarity((1.0, 0.0), (0.0, 1.0)) or 0.0,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
