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
from .grid_perception import file_identity, measure_grid
from .paired_geometry import Camera, Match, match_heads
from .source_cuts import census
from .pose_grid import measure_pose_grid


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


def temporal_focal_salience(bank, nodes, fps):
    """Soft source-local persistence, not an all-frame actor certification.

    A very large head seen at just this flank can be an occluder. Conversely,
    selecting a tiny background head because its cosine fits is not a good
    focal cut. Neighboring observed heads supply evidence, never imputed poses.
    """
    heads, vectors = bank['heads'], bank['vectors']
    result = np.zeros(heads.shape[:2] + (3,))
    for n in range(len(nodes)):
        near = [k for k in range(max(0, n - 2), min(len(nodes), n + 3))
            if k != n and abs(nodes[k] - nodes[n]) / fps <= 2.]
        for side in (0, 1):
            maximum = max(.001, float(heads[n, side, :, 2].max()))
            for i in range(3):
                relative = float(heads[n, side, i, 2]) / maximum
                if relative <= 0:
                    continue
                v = vectors[n, side, i]
                observed = 0
                if np.linalg.norm(v) > .5:
                    for k in near:
                        if float(np.max(vectors[k, 1] @ v)) >= .55:
                            observed += 1
                    persistence = observed / max(1, len(near))
                else:
                    persistence = .5
                result[n, side, i] = relative * (.30 + .70 * persistence)
    return result


def native_event_cost(first, final, events, fps, *, edge_seconds=.7):
    """Charge actual retained-source cuts, especially tiny boundary fragments.

    An original broadcast cut exactly at an authored boundary is one cut.
    A different cut just inside the hold remains visible in the final pixels;
    it cannot disappear from the objective because the source is unchanged.
    """
    inside = events[(events > first) & (events < final)]
    if not len(inside):
        return 0.
    # The short piece may be between two retained native cuts far from BOTH
    # authored ends. Measure every actual fragment, not only edge distances.
    fragments = np.diff(np.r_[first, inside, final]) / fps
    return float(.25 * len(inside) + 4. * np.maximum(0., 1. - fragments / edge_seconds).sum())


def semi_markov(nodes, *, fps, unary, seam_cost, minimum_hold=1.15,
                maximum_hold=6., target_hold=2.8, return_penalty=.7, source_events=None):
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
    if source_events is not None:
        source_events = tuple(np.asarray(events, float) for events in source_events)
        if (len(source_events) != source_count or any(events.ndim != 1
                or not np.isfinite(events).all() or np.any(np.diff(events) <= 0)
                or np.any(events < 0) or np.any(events > nodes[-1]) for events in source_events)):
            raise ValueError('source cuts require ordered finite output-clock bindings')
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
                if source_events is not None:
                    hold += native_event_cost(nodes[start], nodes[end], source_events[source], fps)
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


def choose_programme(nodes, banks, *, fps, check, broadcast_slots=None, source_events=None, poses=None):
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
    salience = [temporal_focal_salience(bank, nodes, fps) for bank in banks]
    started = monotonic()
    for n in range(1, len(nodes) - 1):
        check()
        for left in range(count):
            for right in range(count):
                if left == right:
                    continue
                a, b = banks[left], banks[right]
                m = match_heads(a['heads'][n, 0], b['heads'][n, 1],
                    a['vectors'][n, 0], b['vectors'][n, 1], caps=(a['zoom_cap'], b['zoom_cap']),
                    left_salience=salience[left][n, 0], right_salience=salience[right][n, 1],
                    left_pose=None if poses is None else poses[left][n, :2],
                    right_pose=None if poses is None else poses[right][n, 2:], motion_seconds=6 / fps)
                matches[n, left, right] = m
                same_show = (broadcast_slots is not None and broadcast_slots[left] is not None
                    and broadcast_slots[left] == broadcast_slots[right])
                costs[n, left, right] = m.cost + (.9 if same_show else 0.)
        if n % 40 == 0:
            print(json.dumps(dict(phase='paired_boundary_costs', completed_nodes=n,
                total_nodes=len(nodes), elapsed_seconds=monotonic() - started)), flush=True)
    runs, score = semi_markov(nodes, fps=fps, unary=unary, seam_cost=costs, source_events=source_events)
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
                left_head_index=before.left_head_index, right_head_index=before.right_head_index,
                conditional_pose_loss=before.pose_loss, conditional_velocity_loss=before.velocity_loss,
                comparable_action_joints=before.action_joints, framing_loss=before.framing_loss,
                anonymous_appearance_is_not_member_probability=True))
    plan = replace(original, segments=tuple(segments), output=Path(output),
        transitions=(), authorized_shot_video_clocks=(), member_pose_dissolve_requested=None,
        member_pose_dissolve_effective=None, member_pose_dissolve_count=None)
    return plan, seam_report


def resume_plan(record, original, output):
    plan = render_plan_from_json(record['render_plan'])
    if (plan.output != output or plan.sources != original.sources
            or plan.output_frame_count != original.output_frame_count
            or plan.master_audio_start != original.master_audio_start
            or (plan.width, plan.height, plan.fps) != (original.width, original.height, original.fps)):
        raise ValueError('saved render plan differs from declared source/clock/output domain')
    receipts = record['sparse_perception']
    if len(receipts) != len(plan.sources):
        raise ValueError('saved source binding census is incomplete')
    for source, (path, receipt) in enumerate(zip(plan.sources, receipts, strict=True)):
        if receipt['source'] != source or tuple(receipt['source_file_identity']) != file_identity(path):
            raise OSError('saved plan source changed; do not silently recompute or overwrite it')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--body-pose-model', type=Path,
        help='Existing pinned RTMPose asset; optional conditional action costs, not named actor authority')
    parser.add_argument('--fixed-actor-camera-holds', action='store_true',
        help='Keep anonymous actor and fixed full-hold camera states inside programme selection')
    parser.add_argument('--native-source-readability',action='store_true',
        help='Reuse source-bound full-resolution scalar face-quality observations in fixed holds')
    parser.add_argument('--boundary-native-readability',action='store_true',
        help='Experimental boundary event cost; not a validated whole-film quality improvement')
    parser.add_argument('--joint-musical-opportunities',action='store_true',
        help='Experimental observed music/action C-W objective; not a validated app editor')
    parser.add_argument('--native-camera-ports',action='store_true',
        help='measure exact native-cut flanks without invalidating the song/body grids')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume-render', action='store_true',
        help='Render the existing exact source-bound plan without re-observation or selection')
    parser.add_argument('--preview-height', type=int, choices=(540, 1080), default=540)
    parser.add_argument('--maximum-perception-seconds', type=float, default=900.)
    args = parser.parse_args()
    if args.native_source_readability and not args.fixed_actor_camera_holds:
        raise ValueError('native readability requires the fixed hold consumer')
    if args.boundary_native_readability and not args.native_source_readability:
        raise ValueError('experimental boundary readability requires native source observations')
    if args.joint_musical_opportunities and (not args.fixed_actor_camera_holds or args.body_pose_model is None
                                            or args.boundary_native_readability):
        raise ValueError('joint opportunity experiment requires fixed body-supported holds and boundary readability OFF')
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
    if args.output.exists() or (args.output.with_suffix('.plan.json').exists() and not args.resume_render):
        raise FileExistsError('do not overwrite any existing experiment')
    if os.statvfs(volume).f_bavail * os.statvfs(volume).f_frsize < 2 * 1024 ** 3:
        raise OSError('less than finite preview storage allowance remains')
    original = render_plan_from_json(json.loads(args.source_plan.read_text()))
    manifest = json.loads(args.analysis.read_text())
    if tuple(map(str, original.sources)) != tuple(manifest['request']['sources']):
        raise ValueError('retained synchronization and source domain differ')
    warps = load_warps(manifest)
    original = replace(original, height=args.preview_height, width=args.preview_height * 16 // 9)
    if args.resume_render:
        record = json.loads(args.output.with_suffix('.plan.json').read_text())
        plan = resume_plan(record, original, args.output)
        check()
        render(plan, maximum_runtime_seconds=900., video_bitrate='8M' if args.preview_height == 540 else '24M',
            progress_callback=lambda event: print(json.dumps(dict(phase='render', render_event=event)), flush=True))
        print(json.dumps(dict(phase='movie_ready', path=str(args.output),
            plan_resumed=True, new_neural_inferences=0, professional_quality_approved=False,
            elapsed_seconds=monotonic() - started)), flush=True)
        return
    source_cuts = tuple(tuple(c['timestamp_seconds'] for c in source['changes'])
        for source in manifest['scene_analysis'])
    nodes = editing_grid(duration_frames=original.output_frame_count, fps=original.fps,
        master_start=original.master_audio_start, beats=manifest['rhythm_analysis']['beat_times_seconds'],
        warps=warps, source_cuts=source_cuts)
    banks, receipts = measure_grid(sources=original.sources, warps=warps, master_start=original.master_audio_start,
        fps=original.fps, nodes=nodes, cache=args.cache, models=args.models, check=check,
        maximum_seconds=args.maximum_perception_seconds)
    camera_census = census(original.sources, cache=args.cache.parent / 'crosscut2-camera-census-v1', check=check)
    native_ports,native_port_receipts=None,[]
    if args.native_camera_ports:
        if not args.fixed_actor_camera_holds:raise ValueError('native camera ports require fixed actor/camera holds')
        from .native_ports import measure_native_ports
        native_ports,native_port_receipts=measure_native_ports(sources=original.sources,census=camera_census,
            warps=warps,master_start=original.master_audio_start,fps=original.fps,
            final_frame=original.output_frame_count,cache=args.cache.parent/'crosscut2-native-camera-ports-v1',
            models=args.models,check=check)
    readability_receipts=[]
    if args.native_source_readability:
        from .native_readability import measure_native_readability
        banks,readability_receipts=measure_native_readability(sources=original.sources,banks=banks,
            cache=args.cache.parent/'crosscut2-native-readability-v1',check=check,maximum_seconds=300.)
    poses, body_receipts, body_states = (None, [], None) if args.body_pose_model is None else measure_pose_grid(
        sources=original.sources, warps=warps, nodes=nodes, master_start=original.master_audio_start,
        fps=original.fps, banks=banks, models=args.models, body_model=args.body_pose_model,
        cache=args.cache.parent / 'crosscut2-body-grid-v1', check=check, camera_census=camera_census,
        maximum_seconds=args.maximum_perception_seconds,include_states=True)
    # Retain the observed face grid. Newly measured native cuts change interval
    # costs without invalidating face caches or inventing unobserved heads.
    source_events = tuple(np.asarray(sorted({round((warps[s].source_to_master(frame / record['fps'])
        - original.master_audio_start) * original.fps) for frame in record['frames']
        if 0 < (warps[s].source_to_master(frame / record['fps']) - original.master_audio_start) * original.fps
        < original.output_frame_count}), dtype=np.int64) for s, record in enumerate(camera_census))
    slots = manifest.get('group_broadcast_diversity', {}).get('source_broadcast_slots')
    if slots is not None and len(slots) != len(banks):
        raise ValueError('broadcast slot domain differs from observed sources')
    fixed_receipt = None
    if args.fixed_actor_camera_holds:
        from .fixed_holds import select_fixed_holds, realize_fixed_holds
        music_strengths=None
        if args.joint_musical_opportunities:
            from .opportunity import music_on_nodes
            from .pose_grid import pose_indices
            music_strengths=music_on_nodes(nodes,fps=original.fps,master_start=original.master_audio_start,
                                          rhythm=manifest['rhythm_analysis'])
            for source,bank in enumerate(banks):
                clock=camera_census[source]
                bank['pose_native_indices']=pose_indices(warps[source],nodes=nodes,
                    master_start=original.master_audio_start,fps=original.fps,
                    native_fps=clock['fps'],native_count=clock['frame_count'])
        runs, states, data, score, fixed_receipt = select_fixed_holds(nodes, banks, poses,
            fps=original.fps, source_events=source_events, broadcast_slots=slots, check=check,body_states=body_states,
            native_ports=native_ports,boundary_readability=args.boundary_native_readability,
            music_strengths=music_strengths)
        plan, seams = realize_fixed_holds(original, warps, nodes, runs, states, data, output=args.output)
        from .native_timing import repair_native_timing
        def build(boundaries):
            return realize_fixed_holds(original,warps,nodes,runs,states,data,output=args.output,
                boundary_frames=boundaries)[0]
        boundaries,timing_receipt=repair_native_timing(plan,build,camera_census,check=check)
        plan,seams=realize_fixed_holds(original,warps,nodes,runs,states,data,output=args.output,
            boundary_frames=boundaries)
        fixed_receipt['exact_native_timing_repair']=timing_receipt
    else:
        runs, matches, score = choose_programme(nodes, banks, fps=original.fps, check=check,
            broadcast_slots=slots, source_events=source_events, poses=poses)
        plan, seams = realize_programme(original, warps, nodes, runs, matches, output=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the entire finite JSON before creating the exact resume plan.
    # Serialization failures must not leave a truncated file masquerading as
    # a source-bound saved recipe and block an otherwise warm-cache retry.
    encoded = json.dumps(dict(render_plan=plan.to_json(), sparse_perception=receipts,
            native_readability_observations=readability_receipts,
            native_camera_port_observations=native_port_receipts,
            stage_edit_transformer_used=False,
            conditional_body_perception=body_receipts,
            conditional_body_action_costs_enabled=poses is not None,
            fixed_actor_camera_hold_receipt=fixed_receipt,
            grid_nodes=len(nodes), objective_cost=score, seams=seams,
            retained_source_cut_counts=[len(events) for events in source_events],
            camera_census_all_sources_decoded=all(record['all_encoded_frames_decoded'] for record in camera_census),
            retained_source_cut_detection_completeness_validated=False,
            complete_actor_motion_validated=False, professional_quality_approved=False,
            normal_app_path_executed=False, old_or_manual_editorial_choices_used=False,
            elapsed_before_render=monotonic() - started), indent=2, allow_nan=False)
    with args.output.with_suffix('.plan.json').open('x') as stream:
        stream.write(encoded)
    print(json.dumps(dict(phase='whole_programme_ready', runs=len(runs),
        formation_seams=sum(s['role'] == 'formation' for s in seams),
        focal_seams=sum(s['role'] == 'focal' for s in seams), elapsed_seconds=monotonic() - started)), flush=True)
    if args.render:
        check()
        render(plan, maximum_runtime_seconds=900., video_bitrate='8M' if args.preview_height == 540 else '24M',
            progress_callback=lambda event: print(json.dumps(dict(phase='render', render_event=event)), flush=True))
        print(json.dumps(dict(phase='movie_ready', path=str(args.output),
            professional_quality_approved=False, elapsed_seconds=monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
