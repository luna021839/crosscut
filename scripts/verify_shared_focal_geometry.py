"""Bounded real-model comparison of independent versus shared source pixels.

No acquisition, persistent observations or rendered output are authorized by
this diagnostic. Source reference bootstrapping and later queries are timed
separately; no recognition embeddings or raw pixels are written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))

import cv2
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.workflow_supervisor import run_supervised_workflow


def emit(value):
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def numeric_geometry(value):
    # to_dict deliberately excludes transient identity embeddings. Include
    # all measured joints as well, which the compact geometry report omits.
    result = value.to_dict()
    result['person'] = asdict(value.person) if value.person else None
    result['pose_screen_xyc'] = value.pose_screen_xyc
    return result


def measure(source, queries, shared, args):
    started = monotonic()
    provider = SourceFirstFocalGeometryProvider((source,),
        body_model_path=args.body_model, pose_model_path=args.pose_model,
        body_execution_device=args.body_device, share_source_frames=shared)
    construction_seconds = monotonic()-started
    values, times = [], []
    try:
        for index, seconds in enumerate(queries):
            started = monotonic()
            value = provider.selector_geometry(0, seconds)
            elapsed = monotonic()-started
            values.append(numeric_geometry(value)); times.append(elapsed)
            emit(dict(event='query_complete', source=source.name, shared=shared,
                query_index=index, source_seconds=seconds, elapsed_seconds=elapsed,
                observed_face=value.face_bbox_xywh is not None,
                observed_body=value.person is not None,
                observed_pose=bool(value.pose_screen_xyc),
                native_inferences=provider.native_faces.inference_frame_count,
                body_inferences=provider.bodies.frame_inference_count))
        return dict(values=values, times=times, construction_seconds=construction_seconds,
            diagnostics=provider.diagnostics())
    finally:
        provider.close()


def run(args):
    results = []
    for index, source in enumerate(args.source):
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        finally:
            capture.release()
        if duration <= 83.:
            raise ValueError('diagnostic expects a full source containing the fixed window')
        queries = (3.55, 82., 82.+1/fps, 82.+2/fps)
        order = (False, True) if index % 2 == 0 else (True, False)
        measured = {shared: measure(source, queries, shared, args) for shared in order}
        old, new = measured[False], measured[True]
        if old['values'] != new['values']:
            raise AssertionError(f'shared decoding changed actual focal geometry: {source.name}')
        for key in ('reasons', 'body_frame_inference_count', 'native_face_inference_frame_count'):
            if old['diagnostics'][key] != new['diagnostics'][key]:
                raise AssertionError(f'shared decoding changed inference coverage: {key}')
        result = dict(event='source_complete', source=source.name, queries=len(queries),
            numeric_geometry_equal=True, inference_coverage_equal=True,
            geometry_sha256=hashlib.sha256(json.dumps(old['values'], sort_keys=True,
                allow_nan=False, separators=(',', ':')).encode()).hexdigest(),
            independent_seconds=sum(old['times']), shared_seconds=sum(new['times']),
            independent_query_seconds=old['times'], shared_query_seconds=new['times'],
            independent_construction_seconds=old['construction_seconds'],
            shared_construction_seconds=new['construction_seconds'],
            shared_diagnostics=new['diagnostics'])
        results.append(result); emit(result)
    emit(dict(passed=True, sources=len(results), queries=len(results)*4,
        independent_seconds=sum(r['independent_seconds'] for r in results),
        shared_seconds=sum(r['shared_seconds'] for r in results),
        persistent_cache_used=False, media_quality_approval=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--body-model', type=Path, required=True)
    parser.add_argument('--pose-model', type=Path, required=True)
    parser.add_argument('--body-device', choices=('cpu', 'mps', 'auto'), default='auto')
    parser.add_argument('--budget-seconds', type=float, default=180.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 600:
        parser.error('bounded budget required')
    for path in (*args.source, args.body_model, args.pose_model):
        if not path.is_file():
            parser.error(f'existing input required: {path}')
    if not args.worker:
        try:
            return run_supervised_workflow([sys.executable, '-B', '-u', __file__, *sys.argv[1:], '--worker'],
                absolute_deadline_monotonic_seconds=monotonic()+args.budget_seconds)
        except subprocess.TimeoutExpired:
            emit(dict(passed=False, reason='comparison_deadline_exceeded'))
            return 1
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
