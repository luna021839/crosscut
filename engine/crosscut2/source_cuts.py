"""Bounded streaming camera-cut census; no neural pass or proxy movie files.

Every source is decoded serially at its native frame clock. A 320x180 stream
keeps at most one raw frame in Python and produces a small source-bound JSON.
Detection is a heuristic; full frame coverage is not semantic cut certification.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
from time import monotonic

import cv2
import numpy as np
from scenedetect import FrameTimecode
from scenedetect.detectors import AdaptiveDetector

from .critic import looks_like_flash, signature
from .grid_perception import file_identity
from .observation_cache_contract import function_text


def read_frames(stream, *, check, width=320, height=180):
    """Read complete packed frames without an uninterruptible pipe read."""
    size = width * height * 3
    pending = bytearray()
    while True:
        check()
        ready, _, _ = select.select([stream], [], [], .25)
        if not ready:
            continue
        chunk = os.read(stream.fileno(), size - len(pending))
        if not chunk:
            if pending:
                raise OSError('source decoder emitted an incomplete packed frame')
            return
        pending.extend(chunk)
        if len(pending) == size:
            yield np.frombuffer(pending, dtype=np.uint8).reshape(height, width, 3)
            pending = bytearray()


def cache_implementation():
    """Output face-QA edits cannot invalidate unchanged source-cut evidence.

    Keep the historical key only for the exact canonical cut consumer, pipe
    reader, image signature and flash rule. Changes to those still get a new
    implementation hash. No Git invocation or arbitrary old-cache adoption.
    """
    here=Path(__file__)
    consumer=function_text(here,'census').replace('implementation = cache_implementation()',
        "implementation = hashlib.sha256(Path(__file__).read_bytes()\n        + Path(__file__).with_name('critic.py').read_bytes()).hexdigest()")
    semantic=hashlib.sha256((function_text(here,'read_frames')+consumer+function_text(here.with_name('critic.py'),'signature')
        +function_text(here.with_name('critic.py'),'looks_like_flash')).encode()).hexdigest()
    if semantic=='5a62e2698d75acc1240ee76c97306f10c66318e93dc1dcf031594cefbdb2f0af':
        return 'ae6a48d0d9abad25ab659487877fff6988a4bd615e26e2e3c1f5e2c24cfb9a9e'
    return semantic


def census(sources, *, cache, check, maximum_seconds=600.):
    if not math.isfinite(maximum_seconds) or not 1 <= maximum_seconds <= 1200:
        raise ValueError('one finite camera-census time allowance required')
    started = monotonic()
    def live():
        check()
        if monotonic() - started >= maximum_seconds:
            raise TimeoutError('bounded camera census exhausted; complete caches retained')
    implementation = cache_implementation()
    decoder = shutil.which('ffmpeg')
    if decoder is None:
        raise OSError('streaming camera census requires the existing FFmpeg decoder')
    result = []
    for source, path in enumerate(sources):
        live()
        binding = dict(source_file_identity=list(file_identity(path)), implementation=implementation,
            detector='adaptive-3-window-2-min-content-15-min-scene-5-structure-flash-7')
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        target = Path(cache) / (key + '.json')
        if target.is_file():
            saved = json.loads(target.read_text())
            if saved['binding'] != binding:
                raise ValueError('source camera census binding differs')
            frames = saved['frames']
            if (not math.isfinite(saved['fps']) or saved['fps'] <= 0 or saved['frame_count'] < 2
                    or saved['all_encoded_frames_decoded'] is not True
                    or any(type(frame) is not int or not 0 < frame < saved['frame_count'] for frame in frames)
                    or frames != sorted(set(frames))):
                raise ValueError('source camera census has an invalid native domain')
            result.append(saved)
            print(json.dumps(dict(phase='source_camera_census', source=source, reused=True,
                cuts=len(saved['frames']), elapsed_seconds=monotonic() - started)), flush=True)
            continue
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise OSError('native source clock unavailable')
            fps = cap.get(cv2.CAP_PROP_FPS)
            expected = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if not math.isfinite(fps) or fps <= 0 or expected < 2:
            raise ValueError('source frame domain unavailable')
        detector = AdaptiveDetector(adaptive_threshold=3., min_scene_len=5,
            window_width=2, min_content_val=15.)
        recent, candidates, count = deque(maxlen=7), [], 0
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([decoder, '-nostdin', '-v', 'error', '-threads', '2',
                '-i', str(path), '-map', '0:v:0', '-an', '-filter_threads', '1',
                '-vf', 'scale=320:180:flags=area', '-fps_mode', 'passthrough',
                '-pix_fmt', 'bgr24', '-threads', '1', '-f', 'rawvideo', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=errors)
            try:
                for pixels in read_frames(process.stdout, check=live):
                    sig = signature(pixels)
                    candidates.extend([cut.frame_num, True] for cut in detector.process_frame(
                        FrameTimecode(count, fps=fps), pixels))
                    recent.append((count, sig))
                    if len(recent) == 7 and looks_like_flash(recent[0][1], recent[-1][1]):
                        for row in candidates[-6:]:
                            if recent[0][0] < row[0] <= recent[-1][0]:
                                row[1] = False
                    count += 1
                    if count % 1200 == 0:
                        print(json.dumps(dict(phase='source_camera_decode', source=source,
                            pid=process.pid, frames=count, expected_frames=expected,
                            elapsed_seconds=monotonic() - started)), flush=True)
                code = process.wait(timeout=10.)
                if code != 0:
                    errors.seek(0)
                    raise OSError('camera-census decoder failed: ' + errors.read(4096).decode(errors='replace'))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.)
                process.stdout.close()
        if count != expected or tuple(binding['source_file_identity']) != file_identity(path):
            raise OSError('source camera census did not cover the unchanged native frame domain')
        saved = dict(binding=binding, fps=fps, frame_count=count,
            frames=sorted({frame for frame, keep in candidates if keep}),
            all_encoded_frames_decoded=True, cut_detection_recall_validated=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            json.dump(saved, stream, allow_nan=False)
        result.append(saved)
        print(json.dumps(dict(phase='source_camera_census', source=source, reused=False,
            cuts=len(saved['frames']), elapsed_seconds=monotonic() - started)), flush=True)
    return tuple(result)
