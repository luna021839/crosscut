"""Cold, real-model scalar-vs-batched observation equivalence and timing.

No persistent observations, source acquisition or rendering. This compares
requested face/body/joint geometry and track certificates, not editing quality.
The dependency batch is included in elapsed time, including unused support.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))
sys.path.insert(0, str(ROOT/'scripts'))
from verify_shared_focal_geometry import numeric_geometry
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.source_first_observation_batch import SourceFirstObservationBatch
from crosscut.source_first_unresolved_memo import install_source_first_unresolved_memo
from crosscut.workflow_supervisor import run_supervised_workflow


def emit(value): print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def queries(cut):
    return tuple(cut+offset/29.97+sample for offset in (0, -2, 2, -4, 4, -6, 6)
        for sample in (-2/29.97, -1/29.97, 0., 1/29.97, .20))


def measure(source, batched, args):
    provider = SourceFirstFocalGeometryProvider((source,), body_model_path=args.body_model,
        pose_model_path=args.pose_model, body_execution_device='auto')
    install_source_first_unresolved_memo(provider)
    batch = SourceFirstObservationBatch(provider)
    values, windows = [], []
    try:
        for cut in (3.556645199033104, 82.):
            times = queries(cut)
            started = monotonic()
            if batched: batch(tuple((0, t) for t in times))
            values.extend(numeric_geometry(provider.selector_geometry(0, t)) for t in times)
            elapsed = monotonic()-started
            windows.append(elapsed)
            emit(dict(event='window', source=source.name, batched=batched,
                cut=cut, seconds=elapsed, queries=len(times)))
        return dict(values=values, seconds=sum(windows), window_seconds=windows,
            diagnostics=provider.diagnostics(), batch=batch.diagnostics())
    finally:
        provider.close()


def run(args):
    results = []
    for index, source in enumerate(args.source):
        order = (False, True) if index%2 == 0 else (True, False)
        paths = {batched: measure(source, batched, args) for batched in order}
        old, new = paths[False], paths[True]
        if old['values'] != new['values']:
            differences = [i for i, (a, b) in enumerate(zip(old['values'], new['values'])) if a != b]
            emit(dict(passed=False, source=source.name, differing_queries=differences,
                first_difference={key: [old['values'][differences[0]][key], new['values'][differences[0]][key]]
                    for key in old['values'][differences[0]]
                    if old['values'][differences[0]][key] != new['values'][differences[0]][key]}))
            return 1
        result = dict(source=source.name, queries=len(old['values']), geometry_equal=True,
            geometry_sha256=hashlib.sha256(json.dumps(old['values'], sort_keys=True).encode()).hexdigest(),
            scalar_seconds=old['seconds'], batched_seconds=new['seconds'],
            scalar_windows=old['window_seconds'], batched_windows=new['window_seconds'],
            scalar_diagnostics=old['diagnostics'], batched_diagnostics=new['diagnostics'], batch=new['batch'])
        emit(dict(event='source_verified', **result)); results.append(result)
    emit(dict(passed=True, queries=sum(r['queries'] for r in results), sources=len(results),
        scalar_seconds=sum(r['scalar_seconds'] for r in results),
        batched_seconds=sum(r['batched_seconds'] for r in results),
        persistent_cache_used=False, media_quality_approval=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--body-model', type=Path, required=True)
    parser.add_argument('--pose-model', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=180.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 600: parser.error('finite bounded budget required')
    for path in (*args.source, args.body_model, args.pose_model):
        if not path.is_file(): parser.error(f'existing input required: {path}')
    if args.worker: return run(args)
    try:
        return run_supervised_workflow([sys.executable, '-B', '-u', __file__, *sys.argv[1:], '--worker'],
            absolute_deadline_monotonic_seconds=monotonic()+args.budget_seconds)
    except subprocess.TimeoutExpired:
        emit(dict(passed=False, reason='comparison_deadline_exceeded')); return 1


if __name__ == '__main__': raise SystemExit(main())
