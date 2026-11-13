"""Replay an explicit saved cut-window cohort without decoding or inference.

Usage: PYTHONPATH=engine:. python scripts/audit_member_cached_articulation.py
       /path/to/member/project /path/to/new-report.json
The project must contain the two named source-window reports. This does not
search media folders or authorize a render from a four-frame proposal.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time

from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _boundary_mismatch, _boundary_sample_with_focal_geometry,
)
from crosscut.source_first_member_session import open_source_first_member_session
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.visual_feature_cache import DescriptorRequestLocator
from crosscut.visual_score import _descriptor_from_cache_payload


def audit(root):
    manifest = json.loads((root / 'cache/acquisition-manifest.json').read_text())
    previous = json.loads((root / 'required-source-measured-edges-20260913.json').read_text())['rows']
    sources = tuple(Path(s['local_path']) for s in manifest['selected'])
    geometries = tuple(direct_member_source_frame_geometry(
        (s['media_validation']['width'], s['media_validation']['height']),
        output_size=(1080, 1920)) for s in manifest['selected'])
    focal, visual = open_source_first_member_session(sources,
        feature_cache_directory=root / 'cache', body_execution_device='cpu')
    config = DirectMemberVisualConfig(boundary_search_radius_frames=6, boundary_search_step_frames=2)
    rows = []
    started = time.monotonic()

    def sample(query):
        source, local = query['source'], query['source_seconds']
        identity = focal.faces._identity_key(source, focal.faces._samplers[source], local)
        value = focal._persistent.get(focal._persistent_key(source, identity), source, local)
        sampler = visual._samplers[source]
        payload = sampler.persistent_feature_cache.get_by_locator(DescriptorRequestLocator(
            primary=sampler.frame_identity(local), following=None,
            include_subject_geometry=True, include_formation_pose=True))
        if value is None or payload is None:
            raise RuntimeError('Explicit replay window no longer has its cached observations')
        return _boundary_sample_with_focal_geometry(master_seconds=query['master_seconds'],
            source_local_seconds=local, source_index=source,
            descriptor=_descriptor_from_cache_payload(payload),
            provider=SimpleNamespace(focal_geometry_at=lambda *args: value))

    try:
        with patch('crosscut.visual_score._VideoSampler.frame_at',
                side_effect=AssertionError('Cache-only replay must not decode frames')):
            for old in previous:
                w = old['window']
                a, b, c, d = tuple(sample(q) for q in w['frames'])
                current = _boundary_mismatch(b, c, config, outgoing_previous=a, incoming_next=d,
                    outgoing_source_index=w['outgoing'], incoming_source_index=w['incoming'],
                    outgoing_source_geometry=geometries[w['outgoing']],
                    incoming_source_geometry=geometries[w['incoming']],
                    continuous_official_portrait_fancam_contract=True)
                rows.append(dict(window=w, previous=old['mismatch'], current=asdict(current)))
        return dict(rows=rows, seconds=time.monotonic() - started,
            source_observation_fingerprint=SourceFirstFocalGeometryProvider.observation_fingerprint(),
            fresh_body_inferences=focal.bodies.frame_inference_count,
            fresh_face_inferences=focal.native_faces.inference_frame_count,
            decoded_frames=0, quality_approval=False, render_authorized=False,
            limitation='Four-frame selection proposals only; overlap, camera containment and decoded QA remain required')
    finally:
        visual.close()
        focal.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = audit(args.project)
    with args.report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    summary = {k: v for k, v in result.items() if k != 'rows'}
    summary['count'] = len(result['rows'])
    for label in ('previous', 'current'):
        summary[label] = dict(
            hard=sum(not r[label]['severe_flags'] for r in result['rows']),
            effect_proposals=sum(r[label]['cinematic_bridge'] is not None for r in result['rows']))
    print(json.dumps(summary))
