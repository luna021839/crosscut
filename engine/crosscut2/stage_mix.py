"""Compact experimental whole-song semi-Markov editor over observed sources.

This path deliberately does not patch a retained montage or copy manual cuts.
Sparse anonymous face evidence, motion ownership and professional quality are
different things: only the first is measured here. Not yet the normal app path.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np

from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack, RenderSegment, render
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut.timewarp import TimeWarpMap, TimeWarpConfig, TimeWarpAnchor, TimeWarpControlPoint, TimeWarpResidualDiagnostics
from .grid_perception import measure_grid
from .paired_geometry import Camera, Match, match_heads


def unpack(cls, raw):
    return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


def load_warps(manifest):
    return tuple(unpack(TimeWarpMap, dict(raw,
        config=unpack(TimeWarpConfig, raw['configuration']),
        control_points=tuple(unpack(TimeWarpControlPoint, p) for p in raw['control_points']),
        anchors=tuple(unpack(TimeWarpAnchor, p) for p in raw['accepted_anchors'] + raw['rejected_anchors']),
        residuals=unpack(TimeWarpResidualDiagnostics, raw['residuals'])))
        for raw in manifest['local_timewarps'])


def editing_grid(*, duration_frames, fps, master_start, beats, warps, source_cuts):
    # Native source cuts contribute candidates; near-duplicates are quantized
    # once, with beat locations taking priority. Never import old/manual edits.
    musical = {round((t - master_start) * fps) for t in beats}
    musical = {t for t in musical if 0 < t < duration_frames}
    native = {round((warps[s].source_to_master(t) - master_start) * fps)
        for s, cuts in enumerate(source_cuts) for t in cuts}
    nodes = sorted({0, duration_frames, *musical})
    for t in sorted(native):
        if 0 < t < duration_frames and min(abs(t - b) for b in nodes) > 5:
            nodes.append(t)
    return np.asarray(sorted(nodes), dtype=np.int64)


def node_quality(heads):
    valid = heads[heads[:, 2] > .019]
    if not len(valid):
        return 1.2
    dominant = float(valid[:, 2].max())
    # Soft preference, not a blanket ban on wide/group shots.
    if dominant < .055:
        return .30 + .7 * max(0., .04 - dominant) / .04
    return -.30 * min(1., dominant / .12)


def semi_markov(nodes, *, fps, unary, seam_cost, minimum_hold=1.15,
                maximum_hold=6., target_hold=2.8, return_penalty=.7):
    """Exact DP with previous-source state; no source-count veto or k-best loop.

    Costs are additive interval + duration + boundary, not quality certificates.
    Same-source continuations are not extra editorial cuts. Their camera changes
    remain in the observed interval cost instead of inflating the cut count.
    """
    nodes = np.asarray(nodes)
    if not math.isfinite(fps) or fps <= 0 or not 0 < minimum_hold <= maximum_hold:
        raise ValueError('finite clock and ordered positive duration preferences required')
    source_count = unary.shape[0]
    if (len(nodes) < 2 or nodes[0] != 0 or np.any(np.diff(nodes) <= 0)
            or unary.shape != (source_count, len(nodes) - 1)
            or seam_cost.shape != (len(nodes), source_count, source_count)
            or not np.isfinite(unary).all() or not np.isfinite(seam_cost).all()):
        raise ValueError('finite, ordered complete DP inputs required')
    if source_count == 1:
        # A sole available source cannot become a stage mix, but does not need
        # impossible cross-source cuts to cover the complete programme.
        return ((0, len(nodes) - 1, 0),), float(np.sum(unary[0] * np.diff(nodes) / fps))
    sentinel = source_count
    cost = np.full((len(nodes), source_count, source_count + 1), np.inf)
    previous = {}
    lengths = np.diff(nodes) / fps
    prefix = np.pad(np.cumsum(unary * lengths, axis=1), ((0, 0), (1, 0)))
    for end in range(1, len(nodes)):
        for start in range(end - 1, -1, -1):
            duration = (nodes[end] - nodes[start]) / fps
            if duration > maximum_hold and start != 0:
                break
            if duration < minimum_hold and end != len(nodes) - 1:
                continue
            # A short closing remainder is possible but explicitly expensive.
            grammar = .23 * (duration - target_hold) ** 2 + (max(0., minimum_hold - duration) * 2.)
            for source in range(source_count):
                hold = prefix[source, end] - prefix[source, start] + grammar
                if start == 0:
                    if hold < cost[end, source, sentinel]:
                        cost[end, source, sentinel] = hold
                        previous[end, source, sentinel] = (0, -1, -1)
                    continue
                for old in range(source_count):
                    if old == source:
                        continue
                    values = cost[start, old].copy()
                    values[source] += return_penalty if duration < 3. else 0.
                    prior = int(np.argmin(values))
                    candidate = values[prior] + hold + seam_cost[start, old, source] + .35
                    if candidate < cost[end, source, old]:
                        cost[end, source, old] = candidate
                        previous[end, source, old] = (start, old, prior)
    source, prior = np.unravel_index(np.argmin(cost[-1]), cost[-1].shape)
    if not math.isfinite(cost[-1, source, prior]):
        raise ValueError('no physically covering programme under declared duration range')
    optimum = float(cost[-1, source, prior])
    runs, end = [], len(nodes) - 1
    while end:
        start, old, before = previous[end, source, prior]
        runs.append((start, end, int(source)))
        end, source, prior = start, old, before
    return tuple(reversed(runs)), optimum


def choose_programme(nodes, banks, *, fps, check, broadcast_slots=None):
    count = len(banks)
    quality = np.asarray([[node_quality(h) for h in bank['heads'][:, 1]] for bank in banks])
    unary = (quality[:, :-1] + quality[:, 1:]) / 2
    # A source's own camera jump is not free merely because we do not switch
    # broadcast. Sparse observed geometry is a cost, never an integrity waiver.
    for s, bank in enumerate(banks):
        h = bank['heads'][:, 1, 0, 2]
        for i in range(len(nodes) - 1):
            if min(h[i], h[i + 1]) > .019:
                jump = abs(math.log(h[i] / h[i + 1]))
                if jump > .5:
                    unary[s, i] += .8 * (jump - .5) / max(.1, (nodes[i + 1] - nodes[i]) / fps)
    costs = np.zeros((len(nodes), count, count))
    matches = {}
    started = monotonic()
    for n in range(1, len(nodes) - 1):
        check()
        for left in range(count):
            for right in range(count):
                if left == right:
                    continue
                a, b = banks[left], banks[right]
                m = match_heads(a['heads'][n, 0], b['heads'][n, 1],
                    a['vectors'][n, 0], b['vectors'][n, 1], caps=(a['zoom_cap'], b['zoom_cap']))
                matches[n, left, right] = m
                same_show = (broadcast_slots is not None and broadcast_slots[left] is not None
                    and broadcast_slots[left] == broadcast_slots[right])
                costs[n, left, right] = m.cost + (.9 if same_show else 0.)
        if n % 40 == 0:
            print(json.dumps(dict(phase='paired_boundary_costs', completed_nodes=n,
                total_nodes=len(nodes), elapsed_seconds=monotonic() - started)), flush=True)
    runs, score = semi_markov(nodes, fps=fps, unary=unary, seam_cost=costs)
    return runs, matches, score


def camera_track(entry, exit, duration):
    # Interpolate over the *whole* hold; no half-second neutral bounce/pulse.
    # A held crop can still cut a hand: full body support remains unvalidated.
    if entry == exit == Camera():
        return None
    return EditorialTransformTrack((
        EditorialTransformKeyframe(0., entry.zoom, entry.x, entry.y),
        EditorialTransformKeyframe(duration, exit.zoom, exit.x, exit.y)),
        reason='experimental_whole_hold_two_sided_face_or_formation_alignment')


def realize_programme(original, warps, nodes, runs, matches, *, output):
    segments, seam_report = [], []
    for i, (start, end, source) in enumerate(runs):
        first, final = nodes[start] / original.fps, nodes[end] / original.fps
        duration = final - first
        source_first = warps[source].master_to_source(original.master_audio_start + first)
        source_final = warps[source].master_to_source(original.master_audio_start + final)
        before = None if i == 0 else matches[start, runs[i - 1][2], source]
        after = None if i == len(runs) - 1 else matches[end, source, runs[i + 1][2]]
        entry, exit = (Camera() if before is None else before.right), (Camera() if after is None else after.left)
        segments.append(RenderSegment(source, source_first, duration, first,
            editorial_transform=camera_track(entry, exit, max(1 / original.fps, duration - 1 / original.fps)),
            source_duration=source_final - source_first))
        if before is not None:
            seam_report.append(dict(output_frame=int(nodes[start]), seconds=first,
                left_source=runs[i - 1][2], right_source=source, role=before.role,
                geometry_loss=before.geometry_loss, soft_cost=before.cost,
                appearance_cosine=before.appearance_cosine,
                anonymous_appearance_is_not_member_probability=True))
    plan = replace(original, segments=tuple(segments), output=Path(output),
        transitions=(), authorized_shot_video_clocks=(), member_pose_dissolve_requested=None,
        member_pose_dissolve_effective=None, member_pose_dissolve_count=None)
    return plan, seam_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--preview-height', type=int, choices=(540, 1080), default=540)
    parser.add_argument('--maximum-perception-seconds', type=float, default=900.)
    args = parser.parse_args()
    if not math.isfinite(args.maximum_perception_seconds) or not 1 <= args.maximum_perception_seconds <= 1800:
        raise ValueError('one finite declared perception allowance required')
    started = monotonic()
    volume = Path('/Volumes/CrossCutMedia')
    expected = int(os.environ['IVECROSSCUT_AUTHORIZED_SSD_DEVICE'])
    def check():
        if not os.path.ismount(volume) or volume.stat().st_dev != expected:
            raise OSError('authorized external SSD disconnected or changed')
    check()
    for path in (args.cache, args.output):
        if not path.is_relative_to(volume):
            raise ValueError('new media/cache must remain on authorized external SSD')
    if args.output.exists() or args.output.with_suffix('.plan.json').exists():
        raise FileExistsError('do not overwrite any existing experiment')
    if os.statvfs(volume).f_bavail * os.statvfs(volume).f_frsize < 2 * 1024 ** 3:
        raise OSError('less than finite preview storage allowance remains')
    original = render_plan_from_json(json.loads(args.source_plan.read_text()))
    manifest = json.loads(args.analysis.read_text())
    if tuple(map(str, original.sources)) != tuple(manifest['request']['sources']):
        raise ValueError('retained synchronization and source domain differ')
    warps = load_warps(manifest)
    original = replace(original, height=args.preview_height, width=args.preview_height * 16 // 9)
    nodes = editing_grid(duration_frames=original.output_frame_count, fps=original.fps,
        master_start=original.master_audio_start, beats=manifest['rhythm_analysis']['beat_times_seconds'],
        warps=warps, source_cuts=tuple(tuple(c['timestamp_seconds'] for c in source['changes'])
            for source in manifest['scene_analysis']))
    banks, receipts = measure_grid(sources=original.sources, warps=warps, master_start=original.master_audio_start,
        fps=original.fps, nodes=nodes, cache=args.cache, models=args.models, check=check,
        maximum_seconds=args.maximum_perception_seconds)
    slots = manifest.get('group_broadcast_diversity', {}).get('source_broadcast_slots')
    if slots is not None and len(slots) != len(banks):
        raise ValueError('broadcast slot domain differs from observed sources')
    runs, matches, score = choose_programme(nodes, banks, fps=original.fps, check=check,
        broadcast_slots=slots)
    plan, seams = realize_programme(original, warps, nodes, runs, matches, output=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.plan.json').open('x') as stream:
        json.dump(dict(render_plan=plan.to_json(), sparse_perception=receipts,
            grid_nodes=len(nodes), objective_cost=score, seams=seams,
            complete_actor_motion_validated=False, professional_quality_approved=False,
            normal_app_path_executed=False, old_or_manual_editorial_choices_used=False,
            elapsed_before_render=monotonic() - started), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(phase='whole_programme_ready', runs=len(runs),
        formation_seams=sum(s['role'] == 'formation' for s in seams),
        focal_seams=sum(s['role'] == 'focal' for s in seams), elapsed_seconds=monotonic() - started)), flush=True)
    if args.render:
        check()
        render(plan, maximum_runtime_seconds=900., video_bitrate='8M' if args.preview_height == 540 else '24M',
            progress_callback=lambda event: print(json.dumps(dict(phase='render', **event)), flush=True))
        print(json.dumps(dict(phase='movie_ready', path=str(args.output),
            professional_quality_approved=False, elapsed_seconds=monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
