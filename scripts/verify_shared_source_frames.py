"""Compare independent and shared decoders on source-bank and motion queries.

Uses the committed pre-sharing implementation, three actual consumers, full
native/working pixel hashes and paired numeric descriptors. No acquisition,
media output or persistent feature-cache write is performed.
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
sys.path.insert(0, str(ROOT/'engine'))

import cv2
from crosscut.native_focal_faces import NativeFocalFaceProvider
from crosscut.shared_source_frames import SharedSourceFrames
from crosscut.visual_score import _VideoSampler, _descriptor_cache_payload, describe_frame
from crosscut.workflow_supervisor import run_supervised_workflow

REFERENCE = '90a86ebccd25bb430f12a3b4e4025454324adc3d'


def reference_types():
    modules = []
    for filename in ('visual_score.py', 'native_focal_faces.py'):
        code = subprocess.run(['git', 'show', f'{REFERENCE}:engine/crosscut/{filename}'],
            cwd=ROOT, check=True, capture_output=True, timeout=10).stdout
        name = 'crosscut._independent_reference_'+filename[:-3]
        module = types.ModuleType(name)
        module.__file__ = str(ROOT/'engine/crosscut'/filename)
        module.__package__ = 'crosscut'; sys.modules[name] = module
        exec(compile(code, f'{REFERENCE}:{filename}', 'exec'), module.__dict__)
        modules.append(module)
    return modules[0]._VideoSampler, modules[1].NativeFocalFaceProvider


def measure(source, queries, shared, reference):
    owner = SharedSourceFrames(source) if shared else None
    sampler_type, native_type = (_VideoSampler, NativeFocalFaceProvider) if shared else reference
    body = visual = native = None
    hashes, descriptors = [], []
    seconds = 0.
    try:
        options = {'shared_source_frames': owner} if shared else {}
        body, visual = sampler_type(source, **options), sampler_type(source, **options)
        native = native_type((source,), **({'shared_source_frames': (owner,)} if shared else {}))
        for index, (consumer, timestamp) in enumerate(queries):
            started = monotonic()
            frame = native.frame_at(0, timestamp) if consumer == 'native' else (
                body if consumer == 'body' else visual).frame_at(timestamp)
            seconds += monotonic()-started
            if frame is None:
                raise RuntimeError('missing exact source raster')
            hashes.append((frame.shape, str(frame.dtype), hashlib.sha256(frame.tobytes()).hexdigest()))
            if index in (24, len(queries)-2):
                descriptors.append(_descriptor_cache_payload(describe_frame(frame, include_formation_pose=True)))
        return dict(hashes=hashes, descriptors=descriptors, seconds=seconds,
            shared=owner.diagnostics() if owner else None)
    finally:
        for value in (body, visual, native, owner):
            if value is not None: value.close()


def run(sources):
    reference = reference_types()
    total_old = total_new = 0.
    for index, source in enumerate(sources):
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        finally:
            capture.release()
        bank = tuple(duration*(.03+.94*i/20) for i in range(21))
        queries = [(consumer, t) for consumer in ('native', 'body', 'visual') for t in bank]
        queries += [(consumer, duration*.45+i/fps) for i in range(8)
                    for consumer in ('body', 'native', 'visual')]
        order = (False, True) if index % 2 == 0 else (True, False)
        measured = {shared: measure(source, queries, shared, reference) for shared in order}
        old, new = measured[False], measured[True]
        if old['hashes'] != new['hashes'] or old['descriptors'] != new['descriptors']:
            raise AssertionError(f'shared pixels or numeric descriptors changed: {source.name}')
        total_old += old['seconds']; total_new += new['seconds']
        print(json.dumps(dict(source=source.name, requests=len(queries), pixels_equal=True,
            numeric_equal=True, independent_seconds=old['seconds'], shared_seconds=new['seconds'],
            shared_diagnostics=new['shared'])), flush=True)
    print(json.dumps(dict(passed=True, reference_revision=REFERENCE, sources=len(sources),
        requests=len(sources)*87, independent_seconds=total_old, shared_seconds=total_new,
        speedup=total_old/total_new, media_quality_approval=False)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--budget-seconds', type=float, default=180.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 600:
        parser.error('bounded budget required')
    if not args.worker:
        try:
            return run_supervised_workflow([sys.executable, '-B', '-u', __file__, *sys.argv[1:], '--worker'],
                absolute_deadline_monotonic_seconds=monotonic()+args.budget_seconds)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(passed=False, reason='comparison_deadline_exceeded')), flush=True)
            return 1
    return run(args.source)


if __name__ == '__main__': raise SystemExit(main())
