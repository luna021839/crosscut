"""Compare deferred generic joints with full features on real focal cut windows.

This checks scalar decisions and retained effect samples, not synchronization,
an optimized full edit or rendered quality. No acquisition/persistent cache is
used and no frames or identity embeddings are written.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _VideoDescriptorProvider, _boundary_mismatch,
    _boundary_sample_with_focal_geometry, _hydrate_focal_boundary_sample,
    _same_nonjoint_features,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.workflow_supervisor import run_supervised_workflow


def emit(value):
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def run(args):
    if len(args.source) != 2:
        raise ValueError('comparison requires two distinct source views')
    focal = old = current = None
    seconds_full = seconds_partial = seconds_hydrate = 0.
    windows = pairs = observed = requests = 0
    full_cache = {}
    try:
        focal = SourceFirstFocalGeometryProvider(args.source,
            body_model_path=args.body_model, pose_model_path=args.pose_model,
            body_execution_device='auto')
        options = dict(focal_geometry_provider=focal.selector_geometry,
                       shared_source_frames=focal.shared_source_frames)
        old = _VideoDescriptorProvider(args.source, **options)
        current = _VideoDescriptorProvider(args.source, **options)
        import cv2
        sizes = [direct_member_source_frame_geometry((
            int(s.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(s.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))) for s in old._samplers]
        config = DirectMemberVisualConfig(sample_frame_step_seconds=1/30)
        for nominal in (3.55, 82.):
            for offset in (-2, 0, 2):
                cut = nominal+offset/30
                times = (cut-2/30, cut-1/30, cut, cut+1/30, cut+.2)
                previous, proposed = [], []
                for source, path in enumerate(args.source):
                    full_samples, partial_samples = [], []
                    for time in times:
                        order = ('partial', 'full') if requests%2 else ('full', 'partial')
                        values = {}
                        for mode in order:
                            started = monotonic()
                            values[mode] = (old(source,path,time) if mode == 'full'
                                else current.focal_proposal_at(source,path,time))
                            elapsed = monotonic()-started
                            if mode == 'full': seconds_full += elapsed
                            else: seconds_partial += elapsed
                        if values['full'] is None or values['partial'] is None:
                            raise AssertionError('real feature observation missing')
                        if not _same_nonjoint_features(values['full'], values['partial']):
                            raise AssertionError('nonjoint numeric features changed')
                        full_sample, partial_sample = [
                            _boundary_sample_with_focal_geometry(master_seconds=time,
                                source_local_seconds=time,source_index=source,descriptor=values[mode],
                                provider=old if mode == 'full' else current)
                            for mode in ('full','partial')]
                        if full_sample.focal_geometry != partial_sample.focal_geometry:
                            raise AssertionError('exact target geometry changed')
                        observed += full_sample.focal_geometry.person is not None
                        full_samples.append(full_sample); partial_samples.append(partial_sample)
                        requests += 1
                    previous.append(full_samples); proposed.append(partial_samples)
                for left, right in ((0,1),(1,0)):
                    measured = [_boundary_mismatch(samples[left][1],samples[right][2],config,
                        outgoing_previous=samples[left][0],incoming_next=samples[right][3],
                        outgoing_source_index=left,incoming_source_index=right,
                        outgoing_source_geometry=sizes[left],incoming_source_geometry=sizes[right],
                        continuous_official_portrait_fancam_contract=True)
                        for samples in (previous,proposed)]
                    if measured[0] != measured[1]:
                        raise AssertionError('actual cut, transform, motion or bridge decision changed')
                    pairs += 1
                # Audit one explicitly chosen diagnostic pair's full handles;
                # it is not represented as an optimized/approved edit.
                if offset == 0:
                    for source, indices in ((0,(1,3,4)),(1,(2,3,4))):
                        for index in indices:
                            started = monotonic()
                            restored = _hydrate_focal_boundary_sample(proposed[source][index],
                                source_index=source,source=args.source[source],provider=current,cache=full_cache)
                            seconds_hydrate += monotonic()-started
                            if restored != previous[source][index]:
                                raise AssertionError('full retained effect sample changed')
                windows += 1
                emit(dict(event='window_verified',cut_seconds=cut,comparisons=pairs,
                    requests=requests,observed_body_queries=observed))
        if not observed:
            raise AssertionError('comparison has no observed target bodies')
        emit(dict(passed=True,sources=2,windows=windows,feature_pairs=requests,
            actual_boundary_pairs=pairs,observed_body_queries=observed,
            full_support_seconds=seconds_full,partial_support_seconds=seconds_partial,
            retained_full_hydration_seconds=seconds_hydrate,retained_full_queries=len(full_cache),
            full_feature_inferences=sum(s.describe_frame_call_count for s in old._samplers),
            deferred_feature_inferences=sum(s.describe_frame_call_count for s in current._samplers),
            media_quality_approval=False))
        return 0
    finally:
        for provider in (old,current,focal):
            if provider is not None: provider.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--body-model',type=Path,required=True)
    parser.add_argument('--pose-model',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=180.)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1 <= args.budget_seconds <= 600: parser.error('bounded budget required')
    for path in (*args.source,args.body_model,args.pose_model):
        if not path.is_file(): parser.error(f'existing input required: {path}')
    if args.worker: return run(args)
    try:
        return run_supervised_workflow([sys.executable,'-B','-u',__file__,*sys.argv[1:],'--worker'],
            absolute_deadline_monotonic_seconds=monotonic()+args.budget_seconds)
    except subprocess.TimeoutExpired:
        emit(dict(passed=False,reason='comparison_deadline_exceeded')); return 1


if __name__ == '__main__': raise SystemExit(main())
