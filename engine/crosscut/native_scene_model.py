"""Digest-pinned pretrained shot detection on source frames, never edit quality.

Uses the official TransNetV2 SavedModel (MIT, soCzech/TransNetV2). Runtime and
model storage are explicit. Missing dependencies/storage never silently select
the old histogram detector. Numeric predictions are source-local observations,
not pose alignment, identity, dissolve authorization or Transformer training.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .source_first_model_assets import _download, _publish_bytes, _sha256

UPSTREAM_COMMIT = "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed"
MODEL_FILES = {
    "saved_model.pb": (5933260, "8ac2a52c5719690d512805b6eaf5ce12097c1d8860b3d9de245dcbbc3100f554"),
    "variables/variables.data-00000-of-00001": (30516656, "b8c9dc3eb807583e6215cabee9ca61737b3eb1bceff68418b43bf71459669367"),
    "variables/variables.index": (5526, "8b99e28b4ad11372d9a1ad9703298c2e370df14859da4245fdbe818e92dd403f"),
}
CONTRACT = "transnetv2_source_native_first_incoming_frame_v1"


def ensure_native_scene_model(cache_directory: Path) -> Path:
    root = Path(cache_directory)
    if not root.is_dir():
        raise OSError("native scene model storage is unavailable; no internal fallback")
    directory = root / "models" / "transnetv2-official"
    directory.mkdir(parents=True, exist_ok=True)
    for relative, (size, digest) in MODEL_FILES.items():
        target = directory / relative
        if target.exists():
            if target.stat().st_size != size or _sha256(target) != digest:
                raise ValueError("existing native scene model differs; preserved")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = ("https://media.githubusercontent.com/media/soCzech/TransNetV2/"
                   + UPSTREAM_COMMIT + "/inference/transnetv2-weights/" + relative)
            _publish_bytes(target, _download(url, size), digest)
    license_data = (Path(__file__).parent / "licenses" / "TransNetV2-MIT.txt").read_bytes()
    _publish_bytes(directory / "LICENSE.txt", license_data, hashlib.sha256(license_data).hexdigest())
    return directory


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained native camera-cut detection requires TensorFlow. "
            "Install engine/requirements-native-cuts.txt in the engine runtime."
        ) from exc
    return tf


def native_scene_analyzer_identity() -> dict:
    tf = _tensorflow()
    return {
        "contract": CONTRACT,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_commit": UPSTREAM_COMMIT,
        "models": {name: digest for name, (_, digest) in MODEL_FILES.items()},
        "tensorflow": tf.__version__, "opencv": cv2.__version__,
        "input": "native_sequential_rgb_uint8_48x27_opencv_linear",
        "threshold": 0.5, "context_frames": 25, "central_window_frames": 50,
        "frame_domain": "first_incoming_frame_after_positive_transition_interval",
    }


@lru_cache(maxsize=1)
def _load_model(directory: str):
    return _tensorflow().saved_model.load(directory)


def prediction_intervals(predictions: Sequence[float], threshold: float = .5) -> tuple[tuple[int, int], ...]:
    """Inclusive transition-frame intervals; no max-gap suppression of cuts."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("native scene probabilities must be finite and between zero and one")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("native scene probability threshold must be between zero and one")
    intervals = []
    start = None
    for index, positive in enumerate(values > threshold):
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(values) - 1))
    return tuple(intervals)


def predict_native_frames(frames: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (27, 48, 3) or not len(frames):
        raise ValueError("native scene input must be nonempty RGB uint8 N x 27 x 48 x 3")
    tf = _tensorflow()
    tail = 25 + (-len(frames) % 50)
    padded = np.concatenate((np.repeat(frames[:1], 25, axis=0), frames,
                             np.repeat(frames[-1:], tail, axis=0)))
    single, many = [], []
    for start in range(0, len(padded) - 99, 50):
        logits, heads = model(tf.cast(padded[None, start:start + 100], tf.float32))
        single.append(tf.sigmoid(logits).numpy()[0, 25:75, 0])
        many.append(tf.sigmoid(heads["many_hot"]).numpy()[0, 25:75, 0])
    result = np.concatenate(single)[:len(frames)], np.concatenate(many)[:len(frames)]
    for values in result:
        prediction_intervals(values)
    return result


def detect_native_scene_changes(path: Path, model_directory: Path):
    # Imports are delayed because vision's cached wrapper calls this module.
    from .vision import SceneChange, SceneDetectionConfig, _scene_features
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open source for native scene detection: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(frame, (48, 27), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError("source has no decoded frames for native scene detection")
    single, many = predict_native_frames(np.stack(frames), _load_model(str(model_directory)))
    changes = []
    transition_intervals = []
    settings = replace(SceneDetectionConfig(), analysis_size=(48, 27))
    for first, last in prediction_intervals(single):
        incoming = last + 1
        if first == 0 or incoming >= len(frames):
            continue  # No two-sided source shot evidence at a video edge.
        previous_gray, previous_hist = _scene_features(cv2.cvtColor(frames[incoming-1], cv2.COLOR_RGB2BGR), settings)
        gray, hist = _scene_features(cv2.cvtColor(frames[incoming], cv2.COLOR_RGB2BGR), settings)
        changes.append(SceneChange(
            frame_index=incoming, score=float(np.max(single[first:last+1])),
            frame_difference=min(1., float(np.mean(np.abs(gray-previous_gray))) * settings.difference_gain),
            histogram_distance=float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)),
        ))
        transition_intervals.append({
            "first_transition_frame": first, "last_transition_frame": last,
            "first_incoming_frame": incoming,
            "maximum_many_frame_probability": float(np.max(many[first:last+1])),
        })
    return tuple(changes), {"decoded_frame_count": len(frames),
                            "transition_intervals": transition_intervals,
                            "raw_frames_persisted": False}


def detect_native_scene_changes_cached(path: Path, *, source_cache_directory: Path | None,
                                      derived_cache_directory: Path | None):
    from .source_scalar_cache import PersistentSourceScalarCache
    from .vision import _scene_changes_from_scalar_payload
    root = derived_cache_directory or source_cache_directory
    if root is None or not Path(root).is_dir():
        raise OSError("native camera-cut detection requires available selected cache storage")
    identity = native_scene_analyzer_identity()
    cache = PersistentSourceScalarCache(
        source_cache_directory=source_cache_directory or root,
        derived_cache_directory=derived_cache_directory,
        source_path=Path(path), namespace="native-transnet-scene-changes-v1", analyzer=identity,
    )
    cached = cache.get()
    invalid = 0
    if cached is not None:
        try:
            changes = _scene_changes_from_scalar_payload(cached)
        except (TypeError, ValueError):
            invalid = 1
        else:
            return changes, {**cache.diagnostics(), "enabled": True, "analysis_reused": True,
                             "detection_policy": identity, "semantic_invalid_record_count": 0}
    model_directory = ensure_native_scene_model(Path(root))
    changes, observations = detect_native_scene_changes(Path(path), model_directory)
    cache.put({"changes": [{"frame_index": c.frame_index, "score": c.score,
                            "frame_difference": c.frame_difference,
                            "histogram_distance": c.histogram_distance} for c in changes]})
    return changes, {**cache.diagnostics(), "enabled": True, "analysis_reused": False,
                     "detection_policy": identity, "semantic_invalid_record_count": invalid,
                     "native_observations": observations}
