"""Reuse retrieved subject tracks and choose an actual common-music cut instant.

Only missing pose observations in the existing native windows are acquired.
No detector/recognition forwards, new source search, phase fitting, whole-film
reanalysis, or automatic user-quality approval is performed.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from crosscut2 import sparse_native_frames as sparse
from crosscut2 import subject_action_match as matcher
from crosscut2 import subject_window_bank as bank
from crosscut2.grid_perception import file_identity
from crosscut2.partial_pose import face_proposals, face_guided_hypothesis
from crosscut2.rtmw_pose import MODEL_SHA256, WholeBodySensor
from crosscut2.subject_cut_moments import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ssd-device', type=int, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    previous = json.loads(args.windows.read_text())
    root = Path('/Volumes/CrossCutMedia/CrossCut-Validation').resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('new observations must remain in the SSD project')
    if args.output.exists():
        raise FileExistsError('recover the existing run instead of repeating it')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    identities = {s['path']: file_identity(Path(s['path'])) for s in spec['sources']}

    def check():
        if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev != args.ssd_device:
            raise OSError('authorized SSD changed or disconnected')
        if any(file_identity(Path(p)) != ident for p, ident in identities.items()):
            raise OSError('source changed during cut-moment refinement')

    budget = bank.ProbeBudget(face_limit=0, recognition_limit=0, pose_limit=160,
                              raster_limit=200, seconds=120., check=check)
    sensor = WholeBodySensor(spec['pose_model'])
    cv2.setNumThreads(2)
    receipts = []

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            binding = str(z['binding'])
        return bank.load(path, binding=binding)

    def enrich(path):
        original = load(path)
        target = args.output.parent / (original.track + '-dense-seam.npz')
        if target.exists():
            return bank.load(target, binding=original.binding), str(target)
        # An existing supported raw hypothesis is reused. Missing hypotheses
        # are sampled once at their actual ordinal, never filled by a neighbour.
        requested = [int(n) for i, n in enumerate(original.ordinals)
                     if original.associated[i] and not original.support[i].any()]
        body, support = original.body.copy(), original.support.copy()
        raw, natives, rois, hashes = [], [], [], []
        if requested:
            source = Path(spec['sources'][original.source]['path'])
            budget.reserve('raster', len(requested))
            budget.arrays((2 * len(requested) + 8) * 960 * 540 * 3)
            receipt = {}
            clock = sparse.probe(source, check=budget.live)
            for native, pixels in sparse.frames(source, requested, clock=clock,
                                                check=budget.live, receipt=receipt, maximum_seconds=20.):
                i = native - int(original.ordinals[0])
                boxes = face_proposals(original.heads[i])
                if not boxes:
                    continue
                budget.reserve('pose')
                value = sensor.describe(pixels, bbox_xywh=boxes[0])
                partial = face_guided_hypothesis(value, original.heads[i], original.competitors[i])
                body[i], support[i] = partial.points, partial.supported
                raw.append(value.wholebody_xyc)
                natives.append(native); rois.append(boxes[0])
                hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            receipts.append(dict(source=original.source, **receipt))
        value = replace(original, body=body, support=support)
        bank.save(value, target, provenance=dict(original=str(path), model=MODEL_SHA256,
                  new_pose_inputs=len(raw), original_snapshot_unchanged=True))
        if raw:
            with target.with_name(target.stem + '-raw133.npz').open('xb') as stream:
                np.savez_compressed(stream, native=np.asarray(natives), raw133=np.asarray(raw),
                                    roi=np.asarray(rois), raster_sha256=hashes)
        return value, str(target)

    def clock(window, start, rate):
        s = spec['sources'][window.source]
        return matcher.clock(source=window.source, binding=window.binding,
                             relative_frames=np.arange(-14, 15), source_start=start,
                             playback_rate=rate, output_fps=spec['fps'], native_fps=s['fps'],
                             native_count=s['frame_count'])

    cases = []
    for old in previous['cases']:
        context = next(c for c in spec['cases'] if c['id'] == old['id'])
        left, left_path = enrich(Path(old['window_A']))
        right, right_path = enrich(Path(old['window_B']))
        incoming = context['incoming'][right.source]
        result = search(left, right, clock(left, context['source_start_A'], context['rate_A']),
                        clock(right, incoming['start'], incoming['rate']),
                        candidate_frames=list(range(-8, 9)),
                        outgoing_cap=spec['sources'][left.source]['zoom_cap'],
                        incoming_cap=spec['sources'][right.source]['zoom_cap'])
        cases.append(dict(id=old['id'], window_A=left_path, window_B=right_path, **result))
        best = result['supported'][0] if result['supported'] else None
        print(json.dumps(dict(case=old['id'], best_offset=None if best is None else best['cut_offset_frames'],
                              best_score=None if best is None else best['cut_moment_score'],
                              new_pose=sensor.calls, elapsed=monotonic() - started)), flush=True)
    check()
    with args.output.open('x') as stream:
        stream.write(json.dumps(dict(cases=cases, used=budget.used, new_pose_forwards=sensor.calls,
                                    elapsed_seconds=monotonic() - started, decode_receipts=receipts,
                                    old_caches_changed=False, source_retimed=False,
                                    whole_film_reanalysis=False, professional_quality_approved=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
