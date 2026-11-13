"""Compare full and content-only evidence on real frames without rendering.

The committed reference is the last full-only analyzer. Full descriptors must
remain identical. Content descriptors may omit only pose/joints; every other
field, attention geometry and actual content-admission decision must match.
Timing excludes decoding and model warmup; it is not whole-app latency.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine'))

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _BoundarySample, _attention_geometry,
    _continuous_portrait_fancam_scene_cut_is_transient_artifact,
    _sample_has_human_evidence, _sample_is_blank_or_flat, _span_source_validity,
)
from crosscut.visual_score import _VideoSampler, describe_frame, _descriptor_cache_payload
from crosscut.workflow_supervisor import run_supervised_workflow

REFERENCE = 'a06df96ec0b3177fbb6e8fdb55887b3c5c76625a'


def reference_describer():
    modules = []
    for filename in ('pose.py', 'visual_score.py'):
        code = subprocess.run(['git', 'show', f'{REFERENCE}:engine/crosscut/{filename}'],
            cwd=ROOT, check=True, capture_output=True, timeout=10).stdout
        name = 'crosscut._content_reference_' + filename[:-3]
        module = types.ModuleType(name)
        module.__file__ = str(ROOT/'engine/crosscut'/filename)
        module.__package__ = 'crosscut'
        sys.modules[name] = module
        exec(compile(code, f'{REFERENCE}:{filename}', 'exec'), module.__dict__)
        modules.append(module)
    modules[1].describe_subject_geometry = modules[0].describe_subject_geometry
    return modules[1].describe_frame


def content_fields(value):
    payload = _descriptor_cache_payload(value)
    for name in ('pose_embedding', 'pose_backend', 'pose_confidence'):
        payload.pop(name)
    for person in payload['formation_people']:
        person.pop('upper_body_keypoints_xyc')
    return payload


def content_decision(values, start):
    config = DirectMemberVisualConfig()
    times = (start+1/29.97, start+.12, start+.5, start+1., start+1.5, start+1.8)
    samples = tuple(_BoundarySample(master_seconds=t, source_local_seconds=t,
        descriptor=d, attention=_attention_geometry(d)) for t, d in zip(times, values, strict=True))
    result = _span_source_validity(source_index=0, samples=samples,
        native_scene_cuts=(), suppressed_portrait_fancam_scene_cuts=(),
        continuous_official_portrait_fancam_contract=True, config=config)
    # Samples legitimately carry fewer joints; all decision fields must agree.
    fields_used = tuple(getattr(result, f.name) for f in fields(result) if f.name != 'samples')
    return (fields_used, tuple(_attention_geometry(d) for d in values),
        tuple(_sample_has_human_evidence(s, config) for s in samples),
        tuple(_sample_is_blank_or_flat(s, config) for s in samples),
        _continuous_portrait_fancam_scene_cut_is_transient_artifact(tuple(values[:5]), config))


def run(sources):
    old = reference_describer()
    describe = {
        'reference_full': lambda f: old(f, include_formation_pose=True),
        'current_full': lambda f: describe_frame(f, include_formation_pose=True),
        'content_only': lambda f: describe_frame(f, include_pose=False),
    }
    totals = dict.fromkeys(describe, 0.)
    frame_count = 0
    for source_index, source in enumerate(sources):
        sampler = _VideoSampler(source)
        elapsed = dict.fromkeys(describe, 0.)
        try:
            for start in (82., 120., 150.):
                queries = (start+1/29.97, start+.12, start+.5, start+1., start+1.5, start+1.8)
                values = {name: [] for name in describe}
                for seconds in queries:
                    frame = sampler.frame_at(seconds)
                    if frame is None:
                        raise RuntimeError('missing source frame')
                    if frame_count == 0:
                        for function in describe.values():
                            function(frame)
                    names = list(describe)
                    rotation = frame_count % len(names)
                    for name in names[rotation:]+names[:rotation]:
                        started = monotonic()
                        value = describe[name](frame)
                        elapsed[name] += monotonic()-started
                        values[name].append(value)
                    reference, full, content = (values[name][-1] for name in describe)
                    if _descriptor_cache_payload(reference) != _descriptor_cache_payload(full):
                        raise AssertionError(f'full descriptor changed: {source.name}@{seconds}')
                    if content_fields(full) != content_fields(content):
                        raise AssertionError(f'content evidence changed: {source.name}@{seconds}')
                    if content.pose_confidence or any(p.upper_body_keypoints_xyc for p in content.formation_people):
                        raise AssertionError('content descriptor carries unexpected pose evidence')
                    frame_count += 1
                if content_decision(values['current_full'], start) != content_decision(values['content_only'], start):
                    raise AssertionError(f'actual content decision changed: {source.name}@{start}')
        finally:
            sampler.close()
        for name in totals:
            totals[name] += elapsed[name]
        print(json.dumps(dict(source=source.name, frames=18, full_numeric_equal=True,
            content_fields_equal=True, content_decisions_equal=True, feature_seconds=elapsed)), flush=True)
    print(json.dumps(dict(passed=True, reference_revision=REFERENCE, sources=len(sources),
        frames=frame_count, feature_seconds=totals,
        feature_speedup=totals['current_full']/totals['content_only'],
        decoding_timed=False, media_quality_approval=False)), flush=True)
    return 0


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
    return run(args.source)


if __name__ == '__main__':
    raise SystemExit(main())
