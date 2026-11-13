"""Additive exact-native flank observations for unmeasured physical cut events.

Only registry rows that are unobserved, carry exactly ONE physical native
first-incoming key and expose exactly [f-1, f] are eligible. Pixels come only
from the verified sparse PTS reader (no whole-source fallback) at the original
raster and use the native_ports YuNet/SFace conventions. A pair without a
detected face on BOTH sides stays unknown (observed=False). Geometry is not
member identity, pose or action certification; old caches are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from crosscut import render_frame_schedule
from . import sparse_native_frames as sparse
from .grid_perception import file_identity
from .observation_cache_contract import function_text

CONTRACT = 'exact_native_flank_supplement_v1'
FILES = ('face_detection_yunet_2023mar.onnx', 'face_recognition_sface_2021dec.onnx')
FIELDS = {'binding', 'heads', 'vectors', 'raster', 'measurement'}


def implementation_seal():
    """This module, the sparse reader, its raw-frame parser and the clock proof."""
    here = Path(__file__)
    parts = [here.read_bytes(), here.with_name('sparse_native_frames.py').read_bytes(),
             function_text(here.with_name('source_cuts.py'), 'read_frames').encode(),
             function_text(render_frame_schedule.__file__, 'render_frame_clock_from_probe').encode()]
    return hashlib.sha256(json.dumps([hashlib.sha256(p).hexdigest() for p in parts]).encode()).hexdigest()


def decoder_identity():
    """Resolved FFmpeg executable identity; no subprocess on the warm path."""
    found = shutil.which('ffmpeg')
    return None if not found else [str(Path(found).resolve()), *file_identity(Path(found).resolve())]


def _face_models(assets):
    return (cv2.FaceDetectorYN.create(str(assets[0]), '', (960, 540), .75, .3, 5000),
            cv2.FaceRecognizerSF.create(str(assets[1]), ''))


def _raster(path):
    """Container metadata only: original width, height and frame count."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError('source raster metadata unavailable')
        return tuple(round(capture.get(p)) for p in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _measure(pixels, width, height, detector, recognizer, receipt):
    """Same per-raster arithmetic as native_ports._measure_flanks; counts real calls."""
    if pixels.shape != (height, width, 3) or pixels.dtype != np.uint8:
        raise OSError('sparse raster is not the declared original BGR raster')
    image = cv2.resize(pixels, (960, 540), interpolation=cv2.INTER_AREA)
    receipt['new_face_input_frames'] += 1
    receipt['detector_calls'] += 1
    _, found = detector.detect(image)
    heads, vectors = np.zeros((8, 4), np.float32), np.zeros((3, 128), np.float32)
    if found is not None:
        found = sorted((f for f in found if f[3] / 540 >= .02), key=lambda f: float(f[3]), reverse=True)[:8]
        for i, face in enumerate(found):
            heads[i] = ((face[0] + face[2] / 2) / 960, (face[1] + face[3] / 2) / 540, face[3] / 540, face[-1])
            if i < 3 and face[3] * height / 540 >= 60:
                scaled = face.copy()
                scaled[:14:2] *= width / 960
                scaled[1:14:2] *= height / 540
                receipt['recognizer_calls'] += 1
                value = recognizer.feature(recognizer.alignCrop(pixels, scaled)).ravel()
                if value.shape != (128,) or not np.isfinite(value).all():
                    raise ValueError('invalid native SFace descriptor')
                norm = float(np.linalg.norm(value))
                if norm > 1e-8:
                    vectors[i] = value / norm
    return heads, vectors


def _validate(heads, vectors, raster):
    ok = (heads.shape == (2, 8, 4) and vectors.shape == (2, 3, 128) and raster.shape == (2,)
          and heads.dtype == vectors.dtype == np.float32 and raster.dtype.kind in 'iu'
          and np.isfinite(heads).all() and np.isfinite(vectors).all() and (raster >= 1).all())
    if ok:
        norms = np.linalg.norm(vectors, axis=-1)
        ok = not (heads[:, :, 2] < 0).any() and not ((norms > 1e-6) & (abs(norms - 1) > 1e-3)).any()
    if not ok:
        raise ValueError('native flank supplement record is malformed')


def _load(target, binding):
    with np.load(target, allow_pickle=False) as saved:
        if set(saved.files) != FIELDS or str(saved['binding']) != binding:
            raise ValueError('native flank supplement cache binding differs')
        heads, vectors, raster = saved['heads'], saved['vectors'], saved['raster']
        json.loads(str(saved['measurement']))
    _validate(heads, vectors, raster)
    return heads, vectors


def _save(target, *, check, **arrays):
    """Atomic replace of one complete physical pair; no partial file survives."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix='.partial-', suffix='.npz')
    try:
        with os.fdopen(handle, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _apply(new, receipt, i, heads, vectors, outcome):
    receipt[outcome] += 1
    if not (heads[:, :, 2] > .019).any(axis=1).all():
        receipt['kept_unknown_without_faces_on_both_sides'] += 1
        return
    new['heads'][i], new['vectors'][i], new['observed'][i] = heads, vectors, True
    receipt['supplemented_output_frames'].append(int(new['frames'][i]))


def _chunks(jobs, rate):
    chunk = []
    for job in sorted(jobs, key=lambda job: job[1]):
        if chunk and (len(chunk) >= 16 or (job[1] - chunk[0][1]) / float(rate) > 1.5):
            yield chunk
            chunk = []
        chunk.append(job)
    if chunk:
        yield chunk


def augment_missing_native_flanks(*, sources, bound, registry, cache, models, check,
                                  maximum_new_face_frames=256, maximum_seconds=120.):
    started = monotonic()
    if (type(maximum_new_face_frames) is not int or not 0 <= maximum_new_face_frames <= 4096
            or isinstance(maximum_seconds, bool) or not isinstance(maximum_seconds, (int, float))
            or not math.isfinite(maximum_seconds) or not 1 <= maximum_seconds <= 600
            or not len(sources) == len(bound) == len(registry)):
        raise ValueError('bounded source-matched native flank supplement domain required')

    def live():
        check()
        if monotonic() - started >= maximum_seconds:
            raise TimeoutError('native flank supplement time allowance exhausted')

    cache, assets = Path(cache), [Path(models) / name for name in FILES]
    runtime = dict(implementation=implementation_seal(), opencv=cv2.__version__, numpy=np.__version__,
                   decoder=decoder_identity(), models=[hashlib.sha256(p.read_bytes()).hexdigest() for p in assets])
    states = []
    for s, (path, record, entry) in enumerate(zip(sources, bound, registry)):
        live()
        n = len(entry['frames'])
        new = dict(entry, frames=np.array(entry['frames']), observed=np.array(entry['observed'], bool),
                   heads=np.array(entry['heads'], np.float32), vectors=np.array(entry['vectors'], np.float32),
                   native_event_keys=copy.deepcopy(entry['native_event_keys']))
        keys, observed = new['native_event_keys'], new['observed']
        if (record['source'] != s or new['frames'].ndim != 1 or observed.shape != (n,) or len(keys) != n
                or new['heads'].shape != (n, 2, 8, 4) or new['vectors'].shape != (n, 2, 3, 128)):
            raise ValueError('native event registry and bound source cuts differ')
        rows, by_tick = {}, {}
        for event in record['native_event_rows']:
            tick, f = int(event['output_frame']), int(event['source_native_first_incoming'])
            rows.setdefault((tick, f), []).append([int(v) for v in event['exposed_native_ordinals']])
            by_tick.setdefault(tick, set()).add(f)
        receipt = dict(source=s, contract=CONTRACT, unobserved_events=int((~observed).sum()),
            ineligible_multiple_or_missing_keys=0, ineligible_not_exact_adjacent_pair=0, reused_pairs=0,
            measured_pairs=0, failed_pairs=0, deferred_face_frame_budget=0, deferred_time_budget=0,
            kept_unknown_without_faces_on_both_sides=0, new_face_input_frames=0, detector_calls=0,
            recognizer_calls=0, probe_calls=0, sparse_decode_calls=0, decoded_native_rasters=0,
            native_raster_executions=[], supplemented_output_frames=[],
            errors=[], full_source_scan_fallback=False, actor_binding_certified=False, quality_certified=False)
        identity = file_identity(path)
        stale = list(identity) != list(record['source_file_identity'])
        if stale:
            raise OSError('bound source changed before native flank supplementation')
        misses = []
        for i in map(int, np.flatnonzero(~observed)):
            live()
            if len(keys[i]) != 1:
                receipt['ineligible_multiple_or_missing_keys'] += 1
                continue
            tick, f = int(new['frames'][i]), int(keys[i][0])
            exposed = rows.get((tick, f), [])
            if (by_tick.get(tick) != {f} or not exposed or any(e != [f - 1, f] for e in exposed)
                    or not 1 <= f < record['frame_count']):
                receipt['ineligible_not_exact_adjacent_pair'] += 1
                continue
            if stale:
                receipt['failed_pairs'] += 1
                continue
            binding = json.dumps(dict(contract=CONTRACT, runtime=runtime, source=list(identity),
                source_sha256=record['source_sha256'], fps=record['fps'], frame_count=record['frame_count'],
                ordinals=[f - 1, f], face_input=[960, 540], raster='original_bgr24_area_sparse_pts'),
                sort_keys=True)
            target = cache / (hashlib.sha256(binding.encode()).hexdigest() + '.npz')
            if target.is_file():
                _apply(new, receipt, i, *_load(target, binding), 'reused_pairs')
            else:
                misses.append((i, f, binding, target))
        states.append((path, record, new, receipt, identity, misses))

    budget, face_models, out_of_time = maximum_new_face_frames, None, False
    for path, record, new, receipt, identity, misses in states:
        accepted, planned, done = [], set(), set()
        for job in misses:
            extra = {job[1] - 1, job[1]} - planned
            if out_of_time:
                receipt['deferred_time_budget'] += 1
            elif len(extra) > budget:
                receipt['deferred_face_frame_budget'] += 1
            else:
                planned |= extra
                budget -= len(extra)
                accepted.append(job)
        if not accepted:
            continue
        try:
            def stable():
                live()
                if file_identity(path)!=identity:raise OSError('source changed during native flank supplementation')
            stable()
            receipt['probe_calls'] += 1
            clock = sparse.probe(path, check=live)
            if (clock.identity != identity or clock.count != record['frame_count']
                    or not math.isclose(float(clock.rate), float(record['fps']), rel_tol=1e-6)):
                raise sparse.UnsupportedSeek('verified sparse clock differs from the bound native clock')
            width, height, count = _raster(path)
            if count != record['frame_count'] or min(width, height) < 1 or width * height * 3 > 64 * 1024 ** 2:
                raise sparse.UnsupportedSeek('original raster metadata differs or exceeds the sparse bound')
            face_models = face_models or _face_models(assets)
            raster = np.array([width, height], np.int64)
            # A physical raster shared across adjacent chunks is measured
            # once. Otherwise the16-pair boundary can exceed the unique-frame
            # NN allowance even though planning stayed within that allowance.
            measured={}
            for chunk in _chunks(accepted, clock.rate):
                wanted=sorted({n for _, f, _, _ in chunk for n in (f - 1, f)}-measured.keys())
                receipt['sparse_decode_calls'] += 1
                execution={};receipt['native_raster_executions'].append(execution)
                for tick, pixels in sparse.frames(path, wanted, clock=clock, check=live, width=width,
                        height=height, receipt=execution,maximum_seconds=max(.001, maximum_seconds - (monotonic() - started))):
                    stable();receipt['decoded_native_rasters']+=1
                    measured[tick] = _measure(pixels, width, height, *face_models, receipt)
                stable()
                if any(tick not in measured for tick in wanted):
                    raise OSError('sparse flank window incomplete or source changed')
                for i, f, binding, target in chunk:
                    heads = np.stack([measured[f - 1][0], measured[f][0]])
                    vectors = np.stack([measured[f - 1][1], measured[f][1]])
                    _validate(heads, vectors, raster)
                    _save(target, check=stable,binding=binding, heads=heads, vectors=vectors, raster=raster,
                          measurement=json.dumps(dict(packet_grid_sha256=clock.packet_digest, width=width,
                              height=height, reader='verified_original_PTS_bounded_native_seek_v1')))
                    _apply(new, receipt, i, heads, vectors, 'measured_pairs')
                    done.add(i)
        except TimeoutError as error:
            out_of_time = True
            receipt['deferred_time_budget'] += sum(job[0] not in done for job in accepted)
            receipt['errors'].append(f'{type(error).__name__}: {error}'[:300])
        except (sparse.UnsupportedSeek, OSError) as error:
            receipt['failed_pairs'] += sum(job[0] not in done for job in accepted)
            receipt['errors'].append(f'{type(error).__name__}: {error}'[:300])
    result, receipts = [], []
    for path, _, new, receipt, identity, _ in states:
        if file_identity(path)!=identity:raise OSError('source changed after native flank cache consumption')
        receipt['still_unknown_events'] = int((~new['observed']).sum())
        receipt['status'] = ('partial' if receipt['errors'] or receipt['failed_pairs'] or receipt['deferred_time_budget']
                             or receipt['deferred_face_frame_budget'] else 'complete')
        result.append(new)
        receipts.append(receipt)
    return result, receipts
