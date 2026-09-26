"""Compare actual pinned TransNetV2 outputs; does not render or populate caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import cv2
import numpy as np

from crosscut.native_scene_model import MODEL_FILES, _tensorflow, prediction_intervals, predict_native_frames
from crosscut.native_scene_mlx import CalibratedMLXSceneModel, MLXScenePort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--start-frame', type=int, action='append')
    parser.add_argument('--frame-count', type=int, default=250)
    args = parser.parse_args()
    if args.frame_count < 100:
        parser.error('use at least 100 native frames')
    for name, (size, digest) in MODEL_FILES.items():
        path = args.model_directory / name
        with path.open('rb') as handle:
            if path.stat().st_size != size or hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                raise ValueError(f'not the pinned TransNetV2 weights: {name}')
    reference = _tensorflow().saved_model.load(str(args.model_directory))
    port = MLXScenePort(reference)
    accelerated = CalibratedMLXSceneModel(reference, port=port)
    reference_seconds = accelerated_seconds = 0.0
    maximum_error = 0.0
    total_frames = 0
    all_passed = True
    for source in args.source:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f'cannot open source: {source}')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        starts = args.start_frame or [0, max(0, count // 2 - args.frame_count // 2)]
        try:
            for start in starts:
                if start < 0 or start + args.frame_count > count:
                    raise ValueError('sample outside source frame range')
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                frames = []
                started = monotonic()
                for _ in range(args.frame_count):
                    ok, frame = capture.read()
                    if not ok: raise RuntimeError('incomplete source sample')
                    frames.append(cv2.cvtColor(cv2.resize(frame, (48, 27), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
                frames = np.stack(frames)
                decode_seconds = monotonic() - started
                started = monotonic(); expected = predict_native_frames(frames, reference)
                slow = monotonic() - started
                started = monotonic(); actual = predict_native_frames(frames, accelerated)
                fast = monotonic() - started
                error = max(float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True))
                intervals = [prediction_intervals(values) for values in expected]
                decisions_equal = all(np.array_equal(a > .5, b > .5) for a, b in zip(expected, actual, strict=True))
                passed = error <= 1e-5 and decisions_equal and accelerated.disabled_reason is None
                maximum_error = max(maximum_error, error)
                all_passed &= passed
                total_frames += len(frames)
                reference_seconds += slow; accelerated_seconds += fast
                print(json.dumps(dict(source=str(source), start_frame=start, frame_count=len(frames),
                    decode_seconds=decode_seconds, reference_seconds=slow, accelerated_seconds=fast,
                    maximum_probability_error=error, all_frame_decisions_equal=decisions_equal,
                    reference_transition_intervals=intervals, passed=passed), ensure_ascii=False), flush=True)
        finally:
            capture.release()
    print(json.dumps(dict(passed=all_passed, total_native_frames=total_frames,
        reference_seconds=reference_seconds, accelerated_seconds=accelerated_seconds,
        inference_speedup=reference_seconds / accelerated_seconds,
        maximum_probability_error=maximum_error, executor=accelerated.diagnostics())), flush=True)
    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
