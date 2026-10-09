"""One reusable, exact-ordinal face pass on a declared editing grid.

No gallery or private photographs. Local anonymous SFace vectors remain in the
source cache only; reports and rendered plans never expose them. This sparse
grid does NOT certify all-frame actor paths, pose or hand motion.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import monotonic

import cv2
import numpy as np


def file_identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def measure_grid(*, sources, warps, master_start, fps, nodes, cache, models, check,
                 maximum_seconds=900.):
    """Nodes have distinct left (-1) / right (0) output-native frame inputs."""
    started = monotonic()
    cache, models = Path(cache), Path(models)
    model_paths = (models / 'face_detection_yunet_2023mar.onnx', models / 'face_recognition_sface_2021dec.onnx')
    model_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths]
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result, identities = [], []
    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create(str(model_paths[0]), '', (960, 540), .75, .3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(model_paths[1]), '')
    for source, path in enumerate(sources):
        check()
        identity = file_identity(path)
        reader = cv2.VideoCapture(str(path))
        if not reader.isOpened():
            raise OSError('source ordinal decoder unavailable')
        try:
            native_fps = reader.get(cv2.CAP_PROP_FPS)
            native_count = round(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(native_fps) or native_fps <= 0 or native_count < 2:
                raise ValueError('source native clock unavailable')
            indices = np.asarray([[int(round(warps[source].master_to_source(
                master_start + max(0, min(nodes[-1] - 1, frame + side)) / fps) * native_fps))
                for side in (-1, 0)] for frame in nodes], dtype=np.int64)
            if indices.min() < 0 or indices.max() >= native_count:
                raise ValueError('declared source grid exceeds physical coverage')
            binding = dict(implementation=implementation, models=model_hashes,
                source_file_identity=identity, native_fps=native_fps, native_count=native_count,
                native_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
            key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            target = cache / (key + '.npz')
            if target.is_file():
                with np.load(target, allow_pickle=False) as saved:
                    if str(saved['binding']) != json.dumps(binding, sort_keys=True):
                        raise ValueError('source grid cache binding mismatch')
                    heads, vectors = saved['heads'], saved['vectors']
                    widths, heights = int(saved['width']), int(saved['height'])
                if heads.shape != (len(nodes), 2, 8, 4) or vectors.shape != (len(nodes), 2, 3, 128):
                    raise ValueError('source grid cache shape mismatch')
                reused = True
            else:
                heads = np.zeros((len(nodes), 2, 8, 4), np.float32)
                vectors = np.zeros((len(nodes), 2, 3, 128), np.float32)
                widths, heights = round(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                physical = {}
                for row, sides in enumerate(indices):
                    for side, native in enumerate(sides):
                        check()
                        if monotonic() - started >= maximum_seconds:
                            raise TimeoutError('single declared perception allowance exhausted; completed source caches retained')
                        if int(native) in physical:
                            heads[row, side], vectors[row, side] = physical[int(native)]
                            continue
                        if not reader.set(cv2.CAP_PROP_POS_FRAMES, int(native)):
                            raise OSError('native ordinal seek rejected')
                        ok, pixels = reader.read()
                        if not ok or round(reader.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != int(native):
                            raise OSError('decoder did not return declared native ordinal')
                        image = cv2.resize(pixels, (960, 540), interpolation=cv2.INTER_AREA)
                        _, found = detector.detect(image)
                        if found is not None:
                            found = sorted((face for face in found if face[3] / 540 >= .02),
                                key=lambda face: float(face[3]), reverse=True)[:8]
                            for i, face in enumerate(found):
                                heads[row, side, i] = ((face[0] + face[2] / 2) / 960,
                                    (face[1] + face[3] / 2) / 540, face[3] / 540, face[-1])
                                if i < 3 and face[3] * heights / 540 >= 60:
                                    scaled = face.copy()
                                    scaled[:14:2] *= widths / 960
                                    scaled[1:14:2] *= heights / 540
                                    vector = recognizer.feature(recognizer.alignCrop(pixels, scaled)).ravel()
                                    norm = float(np.linalg.norm(vector))
                                    if vector.shape != (128,) or not np.isfinite(vector).all():
                                        raise ValueError('SFace appearance family or numeric shape changed')
                                    if norm > 1e-8:
                                        vectors[row, side, i] = vector / norm
                        physical[int(native)] = (heads[row, side].copy(), vectors[row, side].copy())
                    if row % 40 == 0:
                        print(json.dumps(dict(phase='grid_perception', source=source,
                            completed_nodes=row + 1, total_nodes=len(nodes), elapsed_seconds=monotonic() - started)), flush=True)
                check()
                if identity != file_identity(path):
                    raise OSError('source changed during declared observation pass')
                cache.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    np.savez_compressed(stream, binding=json.dumps(binding, sort_keys=True),
                        heads=heads, vectors=vectors, width=widths, height=heights)
                reused = False
            if not np.isfinite(heads).all() or not np.isfinite(vectors).all() or identity != file_identity(path):
                raise ValueError('source cache is not finite or no longer bound to source')
            result.append(dict(heads=heads, vectors=vectors, native_indices=indices,
                zoom_cap=2. if widths >= 3840 else 1.35))
            identities.append(dict(source=source, reused=reused, native_inputs=len(set(indices.ravel())),
                source_file_identity=identity, width=widths, height=heights))
            print(json.dumps(dict(phase='source_grid_complete', **identities[-1],
                elapsed_seconds=monotonic() - started)), flush=True)
        finally:
            reader.release()
    return result, identities
