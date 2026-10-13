"""Reusable source-native conditional body windows on the existing face grid.

Decode each source once through a small raw pipe. Never seek thousands of 4K
frames independently or invalidate the cached face/audio observations. The
face-associated eight joints are model evidence, not named actor certification.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic

import cv2
import numpy as np

from crosscut.body_pose_2d import BodyPose2DAnalyzer, RTMPOSE_BODY7_MODEL_SHA256
from crosscut.focal_pose import face_associated_person
from crosscut.pose import (FormationPersonDescriptor, _NanoDetFormationDetector,
    _EXPECTED_SHA256, FORMATION_PERSON_DETECTOR_MODEL)
from .grid_perception import file_identity
from .source_cuts import read_frames

OFFSETS = (-7, -1, 0, 6)


class PoseSensor:
    def __init__(self, models, body_model):
        people_model = Path(models) / FORMATION_PERSON_DETECTOR_MODEL
        for path, expected in ((people_model, _EXPECTED_SHA256[FORMATION_PERSON_DETECTOR_MODEL]),
                (Path(body_model), RTMPOSE_BODY7_MODEL_SHA256)):
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('conditional action model differs from the pinned asset')
        self.detector = _NanoDetFormationDetector(people_model)
        self.body = BodyPose2DAnalyzer(body_model)
        self.calls = 0

    def observe(self, image, heads):
        height, width = image.shape[:2]
        people = []
        for x0, y0, x1, y1, score in self.detector.infer(image):
            if score < .35 or x1 <= x0 or y1 <= y0:
                continue
            box = tuple(map(float, (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)))
            people.append(FormationPersonDescriptor(box, (box[0] + box[2] / 2, box[1] + box[3] / 2), float(score)))
        result, states = np.zeros((3, 8, 3), np.float32), np.zeros(3, np.uint8)
        for i, (x, y, h, score) in enumerate(heads[:3]):
            if h < .055 or score < .75:
                continue
            # The older compact face bank did not retain face width. This
            # approximate association ROI is explicit, not a whole-head mask.
            half_width = h * height / width * .45
            x0, x1 = max(0., x - half_width), min(1., x + half_width)
            y0, y1 = max(0., y - h / 2), min(1., y + h / 2)
            if x1 <= x0 or y1 <= y0:
                continue
            face = tuple(map(float, (x0, y0, x1 - x0, y1 - y0)))
            person = face_associated_person(face, people)
            if person is None:
                states[i] = 2
                continue
            self.calls += 1
            evidence = self.body.describe(image, bbox_xywh=person.bbox_xywh, face_bbox_xywh=face)
            if evidence.reason == 'observed' and evidence.face_anchor_consistent is not False:
                measured = np.asarray(evidence.upper_body_xyc, np.float32)
                if (measured.shape != (8, 3) or not np.isfinite(measured).all()
                        or np.any(measured[:, 2] < 0) or np.any(measured[:, 2] > 1)):
                    raise ValueError('conditional body sensor returned malformed observations')
                result[i] = measured
                states[i] = 1
            else:
                states[i] = 3
        return result, states


def pose_indices(warp, *, nodes, master_start, fps, native_fps, native_count):
    if not math.isfinite(native_fps) or native_fps <= 0 or native_count < 2:
        raise ValueError('finite native pose clock required')
    result = np.asarray([[round(warp.master_to_source(master_start
        + max(0, min(nodes[-1] - 1, int(tick) + offset)) / fps) * native_fps)
        for offset in OFFSETS] for tick in nodes], dtype=np.int64)
    if result.min() < 0 or result.max() >= native_count:
        raise ValueError('conditional pose grid escapes its physical source')
    return result


def measure_pose_grid(*, sources, warps, nodes, master_start, fps, banks,
        models, body_model, cache, check, camera_census, maximum_seconds=900.):
    if not math.isfinite(maximum_seconds) or not 1 <= maximum_seconds <= 1800:
        raise ValueError('one finite conditional pose allowance required')
    started = monotonic()
    last_mount_check = [-math.inf]
    def live():
        now = monotonic()
        # A raw pipe returns many small chunks per frame. Avoid millions of
        # filesystem mount stats while still checking disconnection every .1s
        # and the absolute deadline on every chunk (select waits at most .25s).
        if now - last_mount_check[0] >= .1:
            check()
            last_mount_check[0] = now
        if now - started >= maximum_seconds:
            raise TimeoutError('bounded pose census exhausted; complete source caches retained')
    # Bind the sensor semantics, not unrelated reporting/loader edits. A
    # changed raster/physical index/source/model still changes the cache key.
    implementation = hashlib.sha256(inspect.getsource(PoseSensor).encode()
        + Path(__file__).parents[1].joinpath('crosscut/body_pose_2d.py').read_bytes()
        + Path(__file__).parents[1].joinpath('crosscut/pose.py').read_bytes()
        + Path(__file__).parents[1].joinpath('crosscut/focal_pose.py').read_bytes()).hexdigest()
    sensor, result, receipts = None, [], []
    decoder = shutil.which('ffmpeg')
    if decoder is None:
        raise OSError('pose census requires the existing FFmpeg decoder')
    decoder_identity = list(file_identity(Path(decoder).resolve()))
    for s, path in enumerate(sources):
        live()
        clock = camera_census[s]
        indices = pose_indices(warps[s], nodes=nodes, master_start=master_start,
            fps=fps, native_fps=clock['fps'], native_count=clock['frame_count'])
        binding = dict(implementation=implementation, source_file_identity=list(file_identity(path)),
            indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            heads_sha256=hashlib.sha256(banks[s]['heads'][:, :, :3].tobytes()).hexdigest(),
            body_model_sha256=RTMPOSE_BODY7_MODEL_SHA256,
            detector_sha256=_EXPECTED_SHA256[FORMATION_PERSON_DETECTOR_MODEL],
            raster=[960, 540], scale_filter='area', pixel_format='bgr24',
            decoder_file_identity=decoder_identity, opencv_version=cv2.__version__,
            numpy_version=np.__version__, offsets=list(OFFSETS), native_fps=clock['fps'],
            native_frame_count=clock['frame_count'])
        target = Path(cache) / (hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() + '.npz')
        reused = target.is_file()
        calls = 0
        if reused:
            with np.load(target, allow_pickle=False) as saved:
                if str(saved['binding']) != json.dumps(binding, sort_keys=True):
                    raise ValueError('conditional body cache binding differs')
                poses, states = saved['poses'], saved['states']
        else:
            if sensor is None:
                cv2.setNumThreads(2)
                sensor = PoseSensor(models, body_model)
            before = sensor.calls
            poses = np.zeros((len(nodes), 4, 3, 8, 3), np.float32)
            states = np.zeros((len(nodes), 4, 3), np.uint8)
            needed = {}
            for n, row in enumerate(indices):
                for phase, native in enumerate(row):
                    needed.setdefault(int(native), []).append((n, phase))
            count = 0
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen([decoder, '-nostdin', '-v', 'error', '-threads', '2',
                    '-i', str(path), '-map', '0:v:0', '-an', '-filter_threads', '1',
                    '-vf', 'scale=960:540:flags=area', '-fps_mode', 'passthrough',
                    '-pix_fmt', 'bgr24', '-threads', '1', '-f', 'rawvideo', 'pipe:1'],
                    stdout=subprocess.PIPE, stderr=errors)
                try:
                    for pixels in read_frames(process.stdout, check=live, width=960, height=540):
                        observed = {}
                        for n, phase in needed.get(count, ()):
                            side = 0 if phase < 2 else 1
                            if np.any(banks[s]['heads'][n, side, :3, 2] >= .055):
                                heads = banks[s]['heads'][n, side, :3]
                                key = heads.tobytes()
                                if key not in observed:
                                    observed[key] = sensor.observe(pixels, heads)
                                poses[n, phase], states[n, phase] = observed[key]
                        count += 1
                        if count % 1200 == 0:
                            print(json.dumps(dict(phase='source_body_decode', source=s, frames=count,
                                neural_pose_calls=sensor.calls - before,
                                elapsed_seconds=monotonic() - started)), flush=True)
                    if process.wait(timeout=10.) != 0:
                        errors.seek(0)
                        raise OSError('conditional body decoder failed: ' + errors.read(4096).decode(errors='replace'))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=5.)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=5.)
                    process.stdout.close()
            if count != clock['frame_count']:
                raise OSError('body census did not cover the complete native frame domain')
            calls = sensor.calls - before
            check()
            if tuple(binding['source_file_identity']) != file_identity(path):
                raise OSError('source changed during conditional body observation')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                np.savez_compressed(stream, binding=json.dumps(binding, sort_keys=True), poses=poses, states=states)
        if (poses.shape != (len(nodes), 4, 3, 8, 3) or states.shape != (len(nodes), 4, 3)
                or states.dtype != np.uint8 or not np.isfinite(poses).all() or np.any(states > 3)
                or np.any(poses[..., 2] < 0) or np.any(poses[..., 2] > 1)
                or np.any(poses[states != 1] != 0)
                or tuple(binding['source_file_identity']) != file_identity(path)):
            raise ValueError('conditional body cache shape/numeric domain differs')
        result.append(poses)
        receipt = dict(source=s, reused=reused, observed_heads=int((states == 1).sum()),
            uncertain_heads=int((states >= 2).sum()), neural_pose_calls=calls,
            sparse_conditional_body_is_not_all_frame_actor_certification=True)
        receipts.append(receipt)
        print(json.dumps(dict(phase='source_body_complete', **receipt,
            elapsed_seconds=monotonic() - started)), flush=True)
    return result, receipts
