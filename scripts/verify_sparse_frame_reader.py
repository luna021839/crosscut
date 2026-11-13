"""Read-only real-source parity against the pre-sparse decoder at 065e299.

No analysis times, sources, model policies or output pixels are changed. Each
source gets the same 54 sparse/dense/backward queries. A supervised wall-clock
limit covers the entire comparison, including native decoder calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine'))

import cv2
import numpy as np

from crosscut.seek_locality import SeekLocalityReader, probe_constant_frame_clock
from crosscut.visual_score import _working_frame, _descriptor_cache_payload, describe_frame
from crosscut.workflow_supervisor import run_supervised_workflow


def reference_reader():
    code = subprocess.run(['git', 'show', '065e299:engine/crosscut/seek_locality.py'],
        cwd=ROOT, check=True, capture_output=True, timeout=10).stdout
    module = types.ModuleType('_pre_sparse_decoder_reference')
    sys.modules[module.__name__] = module
    exec(compile(code, '065e299:seek_locality.py', 'exec'), module.__dict__)
    return module.SeekLocalityReader, hashlib.sha256(code).hexdigest()


def measure(source, clock, reader_type, queries):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError('source decoder unavailable')
    processed = 0
    def process(frame):
        nonlocal processed
        processed += 1
        return _working_frame(frame)
    reader = reader_type(capture, clock=clock, process_frame=process, maximum_cached_frames=128)
    started = monotonic()
    try:
        frames = [reader.frame_at(seconds) for seconds in queries]
        elapsed = monotonic() - started
        if any(frame is None for frame in frames):
            raise RuntimeError('incomplete source sample')
        return frames, dict(seconds=elapsed, processed_frames=processed,
            read_count=reader.read_count, seek_count=reader.seek_count,
            pixel_cache_hits=reader.pixel_cache_hit_count, enabled=reader.enabled,
            fallback_count=reader.fallback_count)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--budget-seconds', type=float, default=180.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 600:
        parser.error('budget must be between 1 and 600 seconds')
    if not args.worker:
        try:
            return run_supervised_workflow([sys.executable, '-B', '-u', __file__, *sys.argv[1:], '--worker'],
                absolute_deadline_monotonic_seconds=monotonic()+args.budget_seconds)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(passed=False, reason='comparison_deadline_exceeded')), flush=True)
            return 1
    previous, digest = reference_reader()
    print(json.dumps(dict(reference_revision='065e299', reference_decoder_sha256=digest)), flush=True)
    queries = [80.+2*i+offset for i in range(8) for offset in (1/29.97, .12, .5, 1., 1.5, 1.8)]
    queries += [83.+n/29.97 for n in (0, 1, 2, -1, 3, 0)]
    slow = fast = 0.
    for source_index, source in enumerate(args.source):
        clock = probe_constant_frame_clock(source)
        # Alternate timing order to avoid systematically favoring the second decoder.
        order = (previous, SeekLocalityReader) if source_index % 2 == 0 else (SeekLocalityReader, previous)
        results = {reader: measure(source, clock, reader, queries) for reader in order}
        old_frames, old = results[previous]
        new_frames, new = results[SeekLocalityReader]
        equal = all(np.array_equal(a, b) for a, b in zip(old_frames, new_frames, strict=True))
        if not equal:
            raise AssertionError(f'decoded pixels changed: {source.name}')
        descriptor_equal = all(
            _descriptor_cache_payload(describe_frame(old_frames[index], include_formation_pose=True))
            == _descriptor_cache_payload(describe_frame(new_frames[index], include_formation_pose=True))
            for index in (0, 20))
        if not descriptor_equal:
            raise AssertionError('numeric features changed')
        slow += old['seconds']; fast += new['seconds']
        print(json.dumps(dict(source=source.name, request_count=len(queries), pixels_equal=equal,
            descriptors_equal=descriptor_equal, reference=old, current=new)), flush=True)
    print(json.dumps(dict(passed=True, sources=len(args.source), requests=len(queries)*len(args.source),
        reference_seconds=slow, current_seconds=fast, speedup=slow/fast,
        media_quality_approval=False)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
