"""Bounded additive ORIGINAL-frame person rectangles, not editorial decisions.

The pinned existing YOLO adapter is explicit. No download, face/pose cache
rewrite, whole-source fallback, actor identity or absence certification. An
empty detection is a cached attempt, not evidence that no performer exists.
All rectangles use normalized original-raster XYXY coordinates plus score;
cropping, focal association and any framing cost belong to the consumer.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic

import cv2
import numpy as np

from crosscut import native_person_onnx, onnx_cpu_inference, render_frame_schedule
from . import sparse_native_frames as sparse
from .grid_perception import file_identity
from .observation_cache_contract import function_text

CONTRACT = 'exact_native_person_rectangle_bank_v1'
FIELDS = {'binding', 'rectangles', 'raster', 'measurement'}
# The initial successful291-raster pass used this exact orchestration file.
# Its pixel arithmetic is qualified below, not silently relabelled as new.
LEGACY_RAW_BANK_IMPLEMENTATION_SHA256 = 'fd3cf41c601467977beaad1aaa918a73c7498858eb89763c2817b51948f74efa'
QUALIFIED_RAW_RECTANGLE_SEMANTICS_SHA256 = 'd18bd1337fe47311e336a4eebd87832a271d6fd3812b1270c40487409bc9752d'


def requested_unknown_endpoints(banks, bound, registry):
    """Native exposed ordinals first, then raw face-absent grid endpoints.

    This is not pair-coverage proof. A known opposite endpoint can still need
    an additional person observation before a consumer compares two extents.
    A skipped native flank uses the ACTUALLY EXPOSED pair, not [f-1,f].
    """
    if not len(banks) == len(bound) == len(registry):
        raise ValueError('source-matched person request domains required')
    requests = []
    for source, (bank, record, native) in enumerate(zip(banks, bound, registry)):
        if record['source'] != source:
            raise ValueError('bound person source order differs')
        heads, indices = np.asarray(bank['heads']), np.asarray(bank['native_indices'])
        if heads.shape != indices.shape + (8, 4) or indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError('physical grid endpoint heads and ordinals required')
        seen = ((heads[..., 2] > .019) & (heads[..., 3] >= .75)).any(-1)
        request = {int(f): 1 for f in indices[~seen]}
        for i, tick in enumerate(native['frames']):
            both = bool(native['observed'][i]) and bool(((native['heads'][i, ..., 2] > .019)
                & (native['heads'][i, ..., 3] >= .75)).any(-1).all())
            if both:
                continue
            rows = [r for r in record['native_event_rows'] if r['output_frame'] == int(tick)]
            for row in rows:
                for f in row['exposed_native_ordinals']:
                    request[int(f)] = 0
        requests.append(request)
    return requests


def _runtime(model):
    """Warm fingerprint without creating an ORT session or probing media."""
    found = shutil.which('ffmpeg')
    decoder = None if found is None else [str(Path(found).resolve()), *file_identity(Path(found).resolve())]
    files = [Path(__file__), Path(native_person_onnx.__file__), Path(onnx_cpu_inference.__file__),
             Path(sparse.__file__)]
    seals = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    # Failure handling and queue order do not change a physical observation.
    # Seal all pixel arithmetic, model construction and raster metadata code;
    # keep imported decoder/model/clock implementations independently sealed.
    semantics = '\n'.join(function_text(__file__, f) for f in ('_measure', '_validate', '_detector', '_raster'))
    seals[0] = hashlib.sha256(semantics.encode()).hexdigest()
    seals += [hashlib.sha256(function_text(Path(sparse.__file__).with_name('source_cuts.py'),
                                          'read_frames').encode()).hexdigest(),
              hashlib.sha256(function_text(render_frame_schedule.__file__,
                             'render_frame_clock_from_probe').encode()).hexdigest()]
    identity = file_identity(model)
    digest = hashlib.sha256()
    with Path(model).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    if (digest.hexdigest() != native_person_onnx.PINNED_YOLO11L_ONNX_SHA256
            or file_identity(model) != identity):
        raise ValueError('person bank requires the existing immutable pinned YOLO asset')
    return dict(implementation=seals, decoder=decoder, numpy=np.__version__, opencv=cv2.__version__,
        onnxruntime=version('onnxruntime'), model_sha256=digest.hexdigest(), maximum_side=1280,
        confidence=.35, nms_iou=.55, backend='onnxruntime_CPU_4_intra_1_inter',
        coordinates='original_raster_normalized_xyxy_score'), identity


def _detector(model):
    return native_person_onnx.NativePersonOnnxDetector(model,
        expected_sha256=native_person_onnx.PINNED_YOLO11L_ONNX_SHA256,
        config=native_person_onnx.PersonDetectorConfig(maximum_side=1280))


def _raster(path):
    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise OSError('original person raster metadata unavailable')
        return tuple(round(reader.get(p)) for p in
            (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))
    finally:
        reader.release()


def _validate(rectangles, raster):
    if (rectangles.dtype != np.float32 or rectangles.ndim != 2 or rectangles.shape[1] != 5
            or not np.isfinite(rectangles).all() or (rectangles < 0).any() or (rectangles > 1).any()
            or np.any(rectangles[:, 2:4] <= rectangles[:, :2])
            or raster.shape != (2,) or raster.dtype.kind not in 'iu' or (raster < 1).any()):
        raise ValueError('malformed normalized native person rectangles')


def _measure(pixels, detector, raster):
    width, height = map(int, raster)
    rectangles = np.asarray(detector.infer(pixels), np.float32).copy()
    rectangles[:, :4] /= np.array([width, height, width, height], np.float32)
    _validate(rectangles, raster)
    return rectangles


def _qualified_legacy_runtime(runtime):
    seals = runtime.get('implementation')
    if (not isinstance(seals, list) or len(seals) != 6
            or seals[0] != QUALIFIED_RAW_RECTANGLE_SEMANTICS_SHA256):
        return None
    # Other five reader/model/clock seals and ALL runtime/source fields must
    # still match. Reading an old binding never rewrites its original record.
    return dict(runtime, implementation=[LEGACY_RAW_BANK_IMPLEMENTATION_SHA256, *seals[1:]])


def _load(target, binding):
    with np.load(target, allow_pickle=False) as saved:
        if set(saved.files) != FIELDS or str(saved['binding']) != binding:
            raise ValueError('native person rectangle cache binding differs')
        rectangles, raster = saved['rectangles'], saved['raster']
        measurement = json.loads(str(saved['measurement']))
    _validate(rectangles, raster)
    if measurement.get('verified_original_pts_raster') is not True:
        raise ValueError('native person rectangle lacks original raster provenance')
    return rectangles


def _save(target, *, stable, **arrays):
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix='.partial-', suffix='.npz')
    try:
        with os.fdopen(handle, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        stable()
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _supported_batches(natives, clock, width, height):
    """Partition metadata BEFORE decode; one unsupported seek is not a source veto.

    Splitting a planned window runs no decoder/model and changes no raster.
    Even a combined window exceeding preroll may contain valid individual
    windows. An unsupported singleton stays UNKNOWN, never a scan fallback.
    """
    limit=min(32,(64*1024**2)//(width*height*3));batches=[];unsupported=[]
    def partition(group):
        try:
            sparse.windows(group,clock,maximum_selected_frames=limit)
            batches.append(group)
        except sparse.UnsupportedSeek:
            if len(group)==1:unsupported.extend(group)
            else:
                mid=len(group)//2;partition(group[:mid]);partition(group[mid:])
    partition(sorted(natives))
    return batches,set(unsupported)


def observe_person_rectangles(*, sources, bound, requests, cache, model, check,
                              maximum_new_frames=256, maximum_seconds=180.):
    """Return per-source {physical ordinal: rectangles} and honest receipts.

    Request values are nonnegative priorities (native 0, grid 1). Global unique
    NN and elapsed limits defer work; completed verified raster windows remain
    reusable. No silent scan fallback or automatic retry. Warm runs make no
    model session, media probe or decode. Missing keys remain UNMEASURED.
    """
    started = monotonic()
    if (type(maximum_new_frames) is not int or not 0 <= maximum_new_frames <= 4096
            or type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
            or not 1 <= maximum_seconds <= 600 or not len(sources) == len(bound) == len(requests)):
        raise ValueError('bounded source-matched native person requests required')
    check()
    runtime, model_identity = _runtime(model)
    legacy_runtime = _qualified_legacy_runtime(runtime)
    check()
    output, receipts, identities, jobs = [], [], [], []
    cache = Path(cache)

    def stable():
        check()
        if file_identity(model) != model_identity or any(file_identity(p) != i for p, i in zip(sources, identities)):
            raise OSError('source or person model changed during rectangle pass')

    def live():
        stable()
        if monotonic() - started >= maximum_seconds:
            raise TimeoutError('person rectangle time allowance exhausted')

    for s, (path, record, request) in enumerate(zip(sources, bound, requests)):
        identity = file_identity(path)
        if (record['source'] != s or list(identity) != list(record['source_file_identity'])
                or type(record['frame_count']) is not int or record['frame_count'] < 1
                or not math.isfinite(float(record['fps'])) or float(record['fps']) <= 0):
            raise OSError('bound source changed before rectangle pass')
        identities.append(identity)
        result = {}; output.append(result)
        receipt = dict(source=s, contract=CONTRACT, requested_frames=len(request), reused_frames=0,
            qualified_legacy_raw_frames_reused=0,
            new_person_forwards=0, measured_frames=0, zero_box_attempts=0, deferred_frame_budget=0,
            deferred_time_budget=0, failed_frames=0, probe_calls=0, sparse_decode_calls=0,
            decoded_native_rasters=0, errors=[], raster_executions=[],
            whole_source_scan_fallback=False, actor_identity_or_absence_certified=False,
            editorial_risk_or_utility_activated=False, model_or_face_pose_cache_replaced=False)
        receipts.append(receipt)
        for f, priority in request.items():
            if type(f) is not int or not 0 <= f < record['frame_count'] or type(priority) is not int or priority < 0:
                raise ValueError('exact physical ordinal and nonnegative integer priority required')
            domain = dict(contract=CONTRACT, runtime=runtime, source=list(identity),
                source_sha256=record['source_sha256'], fps=record['fps'], frame_count=record['frame_count'],
                native_ordinal=f)
            binding = json.dumps(domain, sort_keys=True)
            target = cache / (hashlib.sha256(binding.encode()).hexdigest() + '.npz')
            old_binding = None if legacy_runtime is None else json.dumps(dict(domain, runtime=legacy_runtime), sort_keys=True)
            old_target = None if old_binding is None else cache / (hashlib.sha256(old_binding.encode()).hexdigest() + '.npz')
            if target.is_file():
                result[f] = _load(target, binding)
                receipt['reused_frames'] += 1
            elif old_target is not None and old_target.is_file():
                result[f] = _load(old_target, old_binding)
                receipt['reused_frames'] += 1
                receipt['qualified_legacy_raw_frames_reused'] += 1
            else:
                jobs.append((priority, s, f, binding, target))
    stable()
    jobs.sort(key=lambda x: x[:3])
    for _, s, *_ in jobs[maximum_new_frames:]:
        receipts[s]['deferred_frame_budget'] += 1
    jobs = jobs[:maximum_new_frames]
    groups = {}
    for priority, s, f, binding, target in jobs:
        groups.setdefault((priority, s), {})[f] = (binding, target)
    detector = None; clocks = {}; rasters = {}; out_of_time = False
    for (_, s), pending in sorted(groups.items()):
        receipt, record, path = receipts[s], bound[s], sources[s]
        unsupported=set()
        if out_of_time or monotonic() - started >= maximum_seconds:
            out_of_time = True
            receipt['deferred_time_budget'] += len(pending)
            continue
        try:
            live()
            if s not in clocks:
                receipt['probe_calls'] += 1
                clock = sparse.probe(path, check=live, maximum_seconds=min(10., maximum_seconds - (monotonic() - started)))
                if (clock.identity != identities[s] or clock.count != record['frame_count']
                        or not math.isclose(float(clock.rate), float(record['fps']), rel_tol=1e-6)):
                    raise sparse.UnsupportedSeek('person rectangle clock differs from bound source')
                width, height, count = _raster(path)
                if count != clock.count or min(width, height) < 1 or width * height * 3 > 64 * 1024 ** 2:
                    raise sparse.UnsupportedSeek('person original raster exceeds sparse bound')
                clocks[s], rasters[s] = clock, np.array([width, height], np.int64)
            width, height = map(int, rasters[s])
            batches,unsupported=_supported_batches(pending,clocks[s],width,height)
            if unsupported:
                receipt['failed_frames']+=len(unsupported)
                receipt['errors'].append('UnsupportedSeek: individual native ordinals exceed verified preroll bound: '+str(sorted(unsupported)))
            if batches and detector is None:detector=_detector(model)
            for batch in batches:
                execution = {}; receipt['raster_executions'].append(execution)
                receipt['sparse_decode_calls'] += 1
                for f, pixels in sparse.frames(path, batch, clock=clocks[s], check=live,
                        width=width, height=height, receipt=execution,
                        maximum_seconds=max(.001, maximum_seconds - (monotonic() - started))):
                    live()
                    if f not in pending or f in output[s] or pixels.shape != (height, width, 3) or pixels.dtype != np.uint8:
                        raise OSError('person reader returned an undeclared or different original raster')
                    receipt['decoded_native_rasters'] += 1
                    receipt['new_person_forwards'] += 1
                    rectangles = _measure(pixels, detector, rasters[s])
                    binding, target = pending[f]
                    measurement = json.dumps(dict(verified_original_pts_raster=True,
                        packet_grid_sha256=clocks[s].packet_digest, person_model_forwards=1,
                        cast_or_anatomy_certified=False, empty_means_no_detection_not_absence=True))
                    _save(target, stable=live, binding=binding, rectangles=rectangles,
                        raster=rasters[s], measurement=measurement)
                    output[s][f] = rectangles
                    receipt['measured_frames'] += 1
            if any(f not in output[s] and f not in unsupported for f in pending):
                raise OSError('person sparse reader omitted declared ordinals')
        except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            missing = sum(f not in output[s] and f not in unsupported for f in pending)
            receipt['errors'].append(f'{type(error).__name__}: {error}')
            # A single reader-window deadline is a local failure. Only an
            # actually spent GLOBAL allowance defers later native sources.
            if monotonic() - started >= maximum_seconds:
                out_of_time = True; receipt['deferred_time_budget'] += missing
            else:
                receipt['failed_frames'] += missing
    stable()
    for result, receipt in zip(output, receipts):
        receipt['zero_box_attempts'] = sum(not len(v) for v in result.values())
        receipt['unmeasured_frames'] = receipt['requested_frames'] - len(result)
        receipt['status'] = 'partial' if receipt['unmeasured_frames'] else 'complete'
        receipt['elapsed_seconds_shared_global_pass'] = monotonic() - started
    return output, receipts
