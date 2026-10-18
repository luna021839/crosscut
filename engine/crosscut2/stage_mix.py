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


def selection_runtime_limit(*,paired_cameras,played_intervals,explicit_seconds=None):
    """One finite numeric solve, separate from perception and render budgets.

    Completed394-node eight-source paired solves take504..563s including
    preparation. The old fixed300s selection ceiling could abort useful
    larger domains. A900s ceiling permits this work; it is not a retry,
    minimum wait, speed improvement, or a change to the editing objective.
    """
    if type(paired_cameras) is not bool or type(played_intervals) is not bool:
        raise ValueError('explicit paired-camera and played-interval modes required')
    if explicit_seconds is not None:
        if (type(explicit_seconds) not in (int,float) or not math.isfinite(explicit_seconds)
                or not 1<=explicit_seconds<=1200):
            raise ValueError('finite selection allowance in1..1200seconds required')
        return float(explicit_seconds)
    return 900. if paired_cameras and played_intervals else 300. if paired_cameras else 180.


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
    parser.add_argument('--supplementary-wholebody-model',type=Path,
        help='Explicit optional RTMW133 failed-closeup supplement, separate source cache; not body ownership certification')
    parser.add_argument('--fixed-actor-camera-holds', action='store_true',
        help='Keep anonymous actor and fixed full-hold camera states inside programme selection')
    parser.add_argument('--native-source-readability',action='store_true',
        help='Reuse source-bound full-resolution scalar face-quality observations in fixed holds')
    parser.add_argument('--boundary-native-readability',action='store_true',
        help='Experimental boundary event cost; not a validated whole-film quality improvement')
    parser.add_argument('--joint-musical-opportunities',action='store_true',
        help='Experimental observed music/action C-W objective; not a validated app editor')
    parser.add_argument('--joint-motion-event-seconds',action='store_true',
        help='Opt-in coherent same-clock motion/style events with source-shot-constant paired cameras')
    parser.add_argument('--local-ive-photo-tool',type=Path,
        help='Explicit existing local IVE ArcFace/photo-bank tool directory; references remain private and are not copied')
    parser.add_argument('--individual-member-scores',action='store_true',
        help='Retain individual cached gallery likeness scores as soft appearance cues; no forced member labels')
    parser.add_argument('--direct-native-face-observations',type=Path,
        help='Reuse an exact source/grid-bound ArcFace query manifest for current-face pairs; cache-only, no inference')
    parser.add_argument('--native-part-pixel-support',action='store_true',
        help='Opt-in additive original-pixel corroboration of close-up body parts, not anatomical identity certification')
    parser.add_argument('--experimental-preference-model',type=Path,
        help='Explicit fitted same-event experimental model; not normal app or professional-quality approval')
    parser.add_argument('--experimental-sequence-preference-model',type=Path,
        help='Explicit fitted fixed-time four-second context model; E4 only, no same-event model or default weights')
    parser.add_argument('--experimental-sequence-kernel',choices=('numpy','numba'),default='numpy',
        help='Opt-in strict compiled context statistics; requires sequence model and installed sequence-context-numba extra')
    parser.add_argument('--fixed-connection-prior',action='store_true',
        help='One declared untrained monotone face/action B prior; no weak imitation checkpoint')
    parser.add_argument('--common-evidence-connection-prior',action='store_true',
        help='Explicit untrained C: symmetric evidence admission, signed K, unified content and played-event support')
    parser.add_argument('--played-face-connection-prior',action='store_true',
        help='Explicit untrained GROUP D: actual played focal geometry and authored/native fragment integrity')
    parser.add_argument('--played-pose-port-camera-prior',action='store_true',
        help='Explicit untrained GROUP E: physical focal played partial pose and immutable two-ended camera port DAG')
    parser.add_argument('--reuse-subject-observations',type=Path,
        help='Existing source-bound dense subject-observation plan; cache-only E matching, no new inference')
    parser.add_argument('--extra-subject-observations',type=Path,action='append',default=[],
        help='Additional immutable subject-observation plans merged with the primary source-bound reference')
    parser.add_argument('--original-person-observations',type=Path,
        help='Explicit all-source original-frame COCO17 reference; cache-only E6, no per-chosen-face inference')
    parser.add_argument('--original-person-attention',action='store_true',
        help='Use those same full-image instances for observed target/foreground competition; no new NN')
    parser.add_argument('--face-endpoint-observations',type=Path,action='append',
        help='Explicit source-bound observations completing missing face endpoints; never triggers new inference')
    parser.add_argument('--face-endpoint-record-limit',type=int,default=128,
        help='Explicit cached face-record allowance (0..8192); default128, no implicit source work')
    parser.add_argument('--editorial-opportunity-exposure',action='store_true',
        help='Explicit E4 trial: once-per-musical-episode actual subject exposure and readable framing deficit')
    parser.add_argument('--played-interval-utility',action='store_true',
        help='Explicit E6: score actual pair-bound action delivery and composed image motion over every candidate camera')
    parser.add_argument('--matched-event-utility',action='store_true',
        help='Explicit comparison: fixed musical-event value times capped actual pair/camera action coverage')
    parser.add_argument('--matched-action-preference',type=float,default=1.,
        help='Editorial weight for observed matched-action events, not measurement confidence; default unchanged')
    parser.add_argument('--visible-shot-duration-model',type=Path,
        help='Explicit TRAIN-only soft visible-shot duration distribution; requires E6, not app/default activation')
    parser.add_argument('--retain-pair-cameras',action='store_true',
        help='Explicit finite endpoint expansion: retain up to four existing cameras per primary actor pair')
    parser.add_argument('--maximum-selection-seconds',type=float,
        help='One numeric solve allowance1..1200s, no retries; default900s for E6 paired intervals, otherwise unchanged')
    parser.add_argument('--body-camera-proposals',action='store_true',
        help='Add an actual face/arm similarity-fit camera option before DP, retaining old cameras; no post-fit')
    parser.add_argument('--focal-image-motion',action='store_true',
        help='Explicit E6 cue: compare same-music focal image movement composed with actual terminal digital view rates')
    parser.add_argument('--focal-geometry-weight',type=float,default=1.,
        help='Explicit1..8 importance of played face size/position continuity; same native/authored cost, no admission change')
    parser.add_argument('--focal-appearance-weight',type=float,default=1.,
        help='Explicit1..8 importance of subject appearance continuity; same native/authored cost, no identity/source gate')
    parser.add_argument('--boundary-readability',action='store_true',
        help='Score original face detail at the actual cut separately from interval-average quality; no motion/source gate')
    parser.add_argument('--sparse-arm-contradictions',action='store_true',
        help='Keep visible original-image arm opposition from being diluted by other joints; soft cost, no cut gate')
    parser.add_argument('--editorial-reference-source',type=int,action='append',default=None,
        help='Physical source index for original framing/featured-member votes; repeatable, does not exclude render sources')
    parser.add_argument('--detailed-pose',action='store_true',
        help='Explicit E6: reuse existing raw133 head/hand observations; missing sidecars stay unknown')
    parser.add_argument('--recycle-static-camera-capacity',action='store_true',
        help='Use redundant static port slots for distinct paired crops without enlarging the total E6 port bound')
    parser.add_argument('--opportunity-subject-spec',type=Path,
        help='Explicit local source/model JSON for bounded raw observations BEFORE E4 final selection')
    parser.add_argument('--explore-missing-action',action='store_true',
        help='Spend the same bounded preselection probe allowance on unmeasured face-supported action candidates')
    parser.add_argument('--opportunity-observation-centres',type=int,default=12,
        help='Explicit whole-programme observation allowance0..64; not a final cut quota')
    parser.add_argument('--provisional-subject-observation-plan',type=Path,
        help='Explicit source-bound preliminary plan whose missing endpoints are measured once before a new final solve')
    parser.add_argument('--provisional-member-observation-plan',type=Path,
        help='One explicit automatic draft: measure original focal faces across source alternatives before final selection')
    parser.add_argument('--maximum-current-member-queries',type=int,default=1024,
        help='Explicit current-face query allowance0..2048;0 reuses cached queries only')
    parser.add_argument('--maximum-opportunity-perception-seconds',type=float,default=180.,
        help='One shared opportunity observation deadline, at most900s; requires explicit source/model spec')
    parser.add_argument('--opportunity-basis',choices=('exposed','paired_clock'),default='exposed',
        help='Experimental comparison clock; paired_clock does not penalize natural one-frame motion')
    parser.add_argument('--native-camera-ports',action='store_true',
        help='measure exact native-cut flanks without invalidating the song/body grids')
    parser.add_argument('--supplement-native-cut-flanks',action='store_true',
        help='Bounded additive exact flanks missing from the native event union; no full-source fallback')
    parser.add_argument('--person-rectangle-model',type=Path,
        help='Explicit pinned existing YOLO11l asset for experimental visible-height interval fallback; not identity')
    parser.add_argument('--maximum-new-person-frames',type=int,default=256,
        help='Pair-closed raw pre-selection forward limit0..256;0 means cache-only')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume-render', action='store_true',
        help='Render the existing exact source-bound plan without re-observation or selection')
    parser.add_argument('--preview-height', type=int, choices=(540, 1080), default=540)
    parser.add_argument('--maximum-perception-seconds', type=float, default=900.)
    args = parser.parse_args()
    selection_seconds=selection_runtime_limit(paired_cameras=args.retain_pair_cameras,
        played_intervals=args.played_interval_utility,explicit_seconds=args.maximum_selection_seconds)
    if args.maximum_selection_seconds is not None and not args.joint_motion_event_seconds:
        raise ValueError('Explicit selection allowance requires the joint motion-event programme consumer')
    from .subject_opportunity_probe import observation_limits
    observation_limits(args.opportunity_observation_centres,args.maximum_opportunity_perception_seconds)
    if ((args.opportunity_observation_centres!=12 or args.maximum_opportunity_perception_seconds!=180.)
            and args.opportunity_subject_spec is None):
        raise ValueError('Changed opportunity observation allowance requires an explicit bounded subject opportunity probe')
    if args.provisional_subject_observation_plan is not None and args.opportunity_subject_spec is None:
        raise ValueError('Provisional endpoint completion requires an explicit bounded subject opportunity probe')
    if (not 0<=args.maximum_current_member_queries<=2048
            or (args.maximum_current_member_queries!=1024 and args.provisional_member_observation_plan is None)):
        raise ValueError('Changed current-member allowance requires an explicit provisional member plan')
    if args.provisional_member_observation_plan is not None and not args.individual_member_scores:
        raise ValueError('Provisional member observations require explicit individual member scores and local photo assets')
    if (not 0<=args.face_endpoint_record_limit<=8192
            or (args.face_endpoint_record_limit!=128 and args.face_endpoint_observations is None)):
        raise ValueError('A changed face-record allowance requires explicit face endpoint observations')
    if args.reuse_subject_observations is not None and not args.played_pose_port_camera_prior:
        raise ValueError('dense subject observations require the explicit played port-camera consumer')
    if args.native_source_readability and not args.fixed_actor_camera_holds:
        raise ValueError('native readability requires the fixed hold consumer')
    if args.boundary_native_readability and not args.native_source_readability:
        raise ValueError('experimental boundary readability requires native source observations')
    if args.joint_musical_opportunities and (not args.fixed_actor_camera_holds or args.body_pose_model is None
                                            or args.boundary_native_readability):
        raise ValueError('joint opportunity experiment requires fixed body-supported holds and boundary readability OFF')
    if args.opportunity_basis!='exposed' and not args.joint_musical_opportunities:
        raise ValueError('paired-clock matching requires the explicit joint opportunity experiment')
    if args.supplementary_wholebody_model is not None and (args.body_pose_model is None or not args.fixed_actor_camera_holds):
        raise ValueError('wholebody supplement requires explicit base-body fixed holds')
    if args.joint_motion_event_seconds and (not args.fixed_actor_camera_holds or args.body_pose_model is None
            or not args.native_source_readability or not args.native_camera_ports
            or args.joint_musical_opportunities or args.boundary_native_readability):
        raise ValueError('coherent event experiment requires fixed body/native-readability/native-ports and legacy experiments OFF')
    if args.local_ive_photo_tool is not None and not args.joint_motion_event_seconds:
        raise ValueError('local recurrent IVE hints require the explicit joint motion event consumer')
    if args.individual_member_scores and (args.local_ive_photo_tool is None or not args.played_pose_port_camera_prior):
        raise ValueError('individual member scores require explicit local photo assets and played port consumer')
    if args.direct_native_face_observations is not None and not (
            args.played_pose_port_camera_prior and args.played_interval_utility):
        raise ValueError('Direct face queries require the actual played-interval port consumer')
    from .editorial_roles import geometry_importance,appearance_importance
    geometry_importance(args.focal_geometry_weight)
    appearance_importance(args.focal_appearance_weight)
    if args.focal_geometry_weight!=1. and not(args.played_pose_port_camera_prior and args.played_interval_utility):
        raise ValueError('focal geometry importance requires the explicit played-interval port consumer')
    if args.focal_appearance_weight!=1. and not(args.played_pose_port_camera_prior and args.played_interval_utility):
        raise ValueError('focal appearance importance requires the explicit played-interval port consumer')
    if args.boundary_readability and not(args.played_pose_port_camera_prior and args.played_interval_utility):
        raise ValueError('endpoint readability requires the explicit played-interval port consumer')
    if args.sparse_arm_contradictions and not(args.played_pose_port_camera_prior and args.played_interval_utility
            and args.original_person_attention):
        raise ValueError('sparse arm contradictions require original full-image people and the played-interval consumer')
    if args.editorial_reference_source is not None and not(args.played_pose_port_camera_prior and args.editorial_opportunity_exposure):
        raise ValueError('editorial reference sources require the explicit editorial port consumer')
    if args.native_part_pixel_support and not args.joint_motion_event_seconds:
        raise ValueError('native part-pixel support requires the explicit joint motion event consumer')
    if args.supplement_native_cut_flanks and not args.joint_motion_event_seconds:
        raise ValueError('missing native flanks require the explicit physical native event registry')
    if args.person_rectangle_model is not None and not args.played_pose_port_camera_prior:
        raise ValueError('person extent experiment requires the actual played port-camera prior')
    if not 0<=args.maximum_new_person_frames<=256:
        raise ValueError('new person observation allowance must stay in0..256')
    if args.experimental_preference_model is not None and not args.native_part_pixel_support:
        raise ValueError('experimental preference requires the explicit partial-evidence event consumer')
    if args.fixed_connection_prior and (not args.native_part_pixel_support or args.experimental_preference_model is not None):
        raise ValueError('fixed connection prior requires partial event consumer and no preference checkpoint')
    if args.common_evidence_connection_prior and (not args.native_part_pixel_support or args.fixed_connection_prior
            or args.experimental_preference_model is not None):
        raise ValueError('common-evidence prior requires partial event consumer and no other prior/checkpoint')
    if args.played_face_connection_prior and (not args.native_part_pixel_support or args.fixed_connection_prior
            or args.common_evidence_connection_prior or args.experimental_preference_model is not None):
        raise ValueError('played-face prior requires partial event consumer and no other prior/checkpoint')
    if args.played_pose_port_camera_prior and (not args.native_part_pixel_support or args.fixed_connection_prior
            or args.common_evidence_connection_prior or args.played_face_connection_prior or args.experimental_preference_model is not None):
        raise ValueError('played-pose port prior requires partial event consumer and no other prior/checkpoint')
    if not math.isfinite(args.maximum_perception_seconds) or not 1 <= args.maximum_perception_seconds <= 1800:
        raise ValueError('one finite declared perception allowance required')
    if args.editorial_opportunity_exposure and not args.played_pose_port_camera_prior:
        raise ValueError('editorial opportunity exposure requires the actual played port-camera consumer')
    if args.played_interval_utility and (not args.editorial_opportunity_exposure
            or args.experimental_sequence_preference_model is not None):
        raise ValueError('actual interval utility requires E4 and cannot silently replace a sequence model')
    if args.original_person_observations is not None and not args.played_interval_utility:
        raise ValueError('original person observations require the actual E6 played-interval consumer')
    if args.original_person_attention and args.original_person_observations is None:
        raise ValueError('original person competition requires the already measured original person reference')
    if args.matched_event_utility and not args.played_interval_utility:
        raise ValueError('matched event utility requires actual played interval utility')
    if (not math.isfinite(args.matched_action_preference) or args.matched_action_preference<0
            or (args.matched_action_preference!=1. and not args.matched_event_utility)):
        raise ValueError('finite nonnegative matched-action preference requires fixed musical-event utility')
    if args.visible_shot_duration_model is not None and not args.played_interval_utility:
        raise ValueError('visible-shot pacing requires the actual E6 interval consumer')
    if args.retain_pair_cameras and not args.played_interval_utility:
        raise ValueError('paired camera options require actual played interval utility')
    if args.body_camera_proposals and not (args.retain_pair_cameras and args.played_interval_utility):
        raise ValueError('Articulated cameras require retained actual played-interval camera options')
    if args.focal_image_motion and not args.played_interval_utility:
        raise ValueError('focal image motion requires actual played interval utility')
    if args.detailed_pose and not args.played_interval_utility:
        raise ValueError('detailed head/hand pose requires actual played interval utility')
    if args.recycle_static_camera_capacity and not args.played_interval_utility:
        raise ValueError('recovered static capacity requires the actual E6 interval consumer')
    if args.experimental_sequence_preference_model is not None and (
            not args.editorial_opportunity_exposure or not args.played_pose_port_camera_prior
            or args.experimental_preference_model is not None):
        raise ValueError('new sequence preference requires E4 played ports and no same-event checkpoint')
    if args.experimental_sequence_kernel!='numpy' and args.experimental_sequence_preference_model is None:
        raise ValueError('compiled sequence kernel requires an explicit sequence model')
    if args.opportunity_subject_spec is not None and (not args.editorial_opportunity_exposure or args.reuse_subject_observations is None):
        raise ValueError('bounded subject opportunity probe requires explicit E4 and bound existing subject observations')
    if args.explore_missing_action and args.opportunity_subject_spec is None:
        raise ValueError('missing action exploration requires an explicit bounded subject opportunity probe')
    if args.extra_subject_observations and args.reuse_subject_observations is None:
        raise ValueError('extra subject observations require the primary bound subject reference')
    if args.face_endpoint_observations is not None and not args.played_pose_port_camera_prior:
        raise ValueError('face endpoint completion requires the actual played-pose programme path')
    started = monotonic()
    preference_model=None
    sequence_preference_model=None
    connection_prior=None
    if args.experimental_preference_model is not None:
        from .preference_model import Model
        preference_model=Model(json.loads(args.experimental_preference_model.read_text()))
        if preference_model.artifact['supervision_kinds']==['reference_editor_choice_weak_imitation']:
            raise ValueError('weak source-choice checkpoints are diagnostic, not primary connection quality')
    if args.experimental_sequence_preference_model is not None:
        from .ordered_preference_model import load_model
        sequence_preference_model=load_model(json.loads(args.experimental_sequence_preference_model.read_text()))
    if args.fixed_connection_prior:
        from .connection_prior import ConnectionPrior
        connection_prior=ConnectionPrior()
    if args.common_evidence_connection_prior:
        from .common_evidence_prior import CommonEvidencePrior
        connection_prior=CommonEvidencePrior()
    if args.played_face_connection_prior:
        from .played_face_prior import PlayedFacePrior
        connection_prior=PlayedFacePrior()
    if args.played_pose_port_camera_prior:
        from .played_pose_port_programme import PlayedPosePortPrior
        duration_model=None
        if args.visible_shot_duration_model is not None:
            from .visible_shot_duration import Model as DurationModel
            duration_model=DurationModel(json.loads(args.visible_shot_duration_model.read_text()))
        connection_prior=PlayedPosePortPrior(person_extents=args.person_rectangle_model is not None,
            editorial_opportunities=args.editorial_opportunity_exposure,
            played_interval_utility=args.played_interval_utility,shot_duration_model=duration_model,
            recycle_static_capacity=args.recycle_static_camera_capacity,matched_event_utility=args.matched_event_utility,
            matched_action_preference=args.matched_action_preference,retain_pair_cameras=args.retain_pair_cameras,
            focal_motion=args.focal_image_motion,detailed_pose=args.detailed_pose,
            person_attention=args.original_person_attention,original_person_attention=args.original_person_attention,
            output_height=args.preview_height if args.played_interval_utility else None,
            geometry_weight=args.focal_geometry_weight,editorial_reference_sources=args.editorial_reference_source,
            body_camera_proposals=args.body_camera_proposals,
            appearance_weight=args.focal_appearance_weight,
            boundary_readability=args.boundary_readability,
            sparse_arm_contradictions=args.sparse_arm_contradictions,
            direct_face_appearance=args.direct_native_face_observations is not None)
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
    supplementary_receipts=[]
    if args.supplementary_wholebody_model is not None:
        from .supplementary_pose_bank import augment_missing_grid
        poses,supplementary_receipts=augment_missing_grid(sources=original.sources,banks=banks,poses=poses,
            body_states=body_states,warps=warps,nodes=nodes,master_start=original.master_audio_start,fps=original.fps,
            camera_census=camera_census,model_path=args.supplementary_wholebody_model,
            cache=args.cache.parent/'crosscut2-rtmw-supplement-v1',check=check,
            maximum_seconds=min(600.,args.maximum_perception_seconds),maximum_calls=6000)
    # Retain the observed face grid. Newly measured native cuts change interval
    # costs without invalidating face caches or inventing unobserved heads.
    source_events = tuple(np.asarray(sorted({round((warps[s].source_to_master(frame / record['fps'])
        - original.master_audio_start) * original.fps) for frame in record['frames']
        if 0 < (warps[s].source_to_master(frame / record['fps']) - original.master_audio_start) * original.fps
        < original.output_frame_count}), dtype=np.int64) for s, record in enumerate(camera_census))
    slots = manifest.get('group_broadcast_diversity', {}).get('source_broadcast_slots')
    if slots is not None and len(slots) != len(banks):
        raise ValueError('broadcast slot domain differs from observed sources')
    if slots is not None:
        for bank,slot in zip(banks,slots):bank['consensus_family']=str(slot)
    fixed_receipt = None
    bound_source_cut_events=None
    if args.fixed_actor_camera_holds:
        from .fixed_holds import select_fixed_holds, realize_fixed_holds
        music_strengths=None
        if args.joint_musical_opportunities or args.joint_motion_event_seconds:
            from .opportunity import music_on_nodes
            from .pose_grid import pose_indices
            music_strengths=music_on_nodes(nodes,fps=original.fps,master_start=original.master_audio_start,
                                          rhythm=manifest['rhythm_analysis'])
            for source,bank in enumerate(banks):
                clock=camera_census[source]
                bank['pose_native_indices']=pose_indices(warps[source],nodes=nodes,
                    master_start=original.master_audio_start,fps=original.fps,
                    native_fps=clock['fps'],native_count=clock['frame_count'])
        if args.joint_motion_event_seconds:
            from .source_style import observe
            from .event_holds import select,realize,check_selected_clock
            from .source_event_union import combine,attach_existing_ports
            source_events,bound_source_cut_events,union_receipts=combine(sources=original.sources,manifest=manifest,
                census=camera_census,warps=warps,master_start=original.master_audio_start,fps=original.fps,
                final_frame=original.output_frame_count,check=check,
                supplemental_cut_cache=args.cache.parent/'crosscut2-confirmed-native-cuts-v1')
            for bank,record in zip(banks,bound_source_cut_events):
                bank['native_cut_frames']=np.asarray(record['frames'],int)
            if args.played_pose_port_camera_prior:
                from .programme_source_clock import attach as attach_programme_clock
                attach_programme_clock(banks,warps,camera_census,nodes,
                    master_start=original.master_audio_start,fps=original.fps)
                from .terminal_stencil import observe as observe_terminal_stencils
                endings=observe_terminal_stencils(original.sources,banks,camera_census,check=check)
                for bank,ending in zip(banks,endings):bank['terminal_stencil_observation']=ending
            native_ports,registry_receipts=attach_existing_ports(bound=bound_source_cut_events,census=camera_census,
                ports=native_ports,warps=warps,master_start=original.master_audio_start,fps=original.fps,
                final_frame=original.output_frame_count)
            supplementary_flank_receipts=[]
            if args.supplement_native_cut_flanks:
                from .native_flank_supplement import augment_missing_native_flanks
                native_ports,supplementary_flank_receipts=augment_missing_native_flanks(sources=original.sources,
                    bound=bound_source_cut_events,registry=native_ports,cache=args.cache.parent/'crosscut2-native-flank-supplement-v1',
                    models=args.models,check=check,maximum_new_face_frames=256,maximum_seconds=120.)
            style,style_receipts=observe(sources=original.sources,banks=banks,
                cache=args.cache.parent/'crosscut2-native-style-v1',check=check,maximum_seconds=180.)
            identity_hints=None;identity_receipts=[]
            if args.local_ive_photo_tool is not None and args.provisional_member_observation_plan is None:
                from .recurrent_member_hints import measure
                identity_hints,identity_receipts=measure(sources=original.sources,banks=banks,census=camera_census,
                    photo_tool=args.local_ive_photo_tool,models=args.models,
                    cache=args.cache.parent/'crosscut2-local-member-query-v1',check=check,
                    maximum_seconds=min(240.,args.maximum_perception_seconds),maximum_calls=288,
                    preserve_member_scores=args.individual_member_scores)
            face_completion=None
            if args.face_endpoint_observations is not None:
                from .sparse_face_completion import load as load_faces,apply as apply_faces
                from .fine_subject_grid import rebind_hints
                hashes=[v['source_sha256'] for v in bound_source_cut_events]
                completed,face_completion=load_faces(args.face_endpoint_observations,sources=original.sources,
                    source_hashes=hashes,master_start=original.master_audio_start,fps=original.fps,
                    authorized_root=args.cache.parent.parent,maximum_records=args.face_endpoint_record_limit)
                previous_banks=banks
                banks,poses,style,body_states,merged=apply_faces(banks,poses,style,body_states,completed,
                    source_hashes=hashes,maximum_records=args.face_endpoint_record_limit)
                if identity_hints is not None:
                    identity_hints=rebind_hints(previous_banks,banks,identity_hints,allow_previously_unobserved=True)
                face_completion['merge']=merged
            if args.provisional_member_observation_plan is not None:
                from .source_member_scores import load_programme_endpoint_requests
                from .recurrent_member_hints import measure
                requests,request_receipt=load_programme_endpoint_requests(args.provisional_member_observation_plan,
                    nodes=nodes,banks=banks,sources=original.sources,
                    source_hashes=[v['source_sha256'] for v in bound_source_cut_events],
                    master_start=original.master_audio_start,fps=original.fps,authorized_root=args.cache.parent.parent)
                identity_hints,identity_receipts=measure(sources=original.sources,banks=banks,census=camera_census,
                    photo_tool=args.local_ive_photo_tool,models=args.models,
                    cache=args.cache.parent/'crosscut2-local-member-query-v1',check=check,
                    maximum_seconds=min(600.,args.maximum_perception_seconds),maximum_calls=args.maximum_current_member_queries,
                    cache_only_partial=args.maximum_current_member_queries==0,preserve_member_scores=True,
                    additional_requests=requests)
                for row in identity_receipts:row['automatic_endpoint_request_policy']=request_receipt
            part_observer=None
            person_observer=None
            if args.person_rectangle_model is not None:
                from .person_pair_observer import prepare_pair_observer
                person_observer=prepare_pair_observer(sources=original.sources,bound=bound_source_cut_events,
                    cache=args.cache.parent/'crosscut2-person-rectangles-v1',model=args.person_rectangle_model,
                    maximum_new_frames=args.maximum_new_person_frames,maximum_seconds=120.)
            if args.native_part_pixel_support:
                from .part_support_bank import corroborate
                from .part_hypotheses import retain_model_hypotheses
                def part_observer(local_states,local_data,*,check=check):
                    rows,receipts=corroborate(sources=original.sources,states=local_states,data=local_data,census=bound_source_cut_events,
                        cache=args.cache.parent/'crosscut2-native-part-support-v1',check=check,
                        maximum_seconds=180.,maximum_new_intervals=6000,allow_sequential_fallback=False)
                    return retain_model_hypotheses(rows=rows,states=local_states,sources=original.sources,
                        cache=args.cache.parent/'crosscut2-native-part-support-v1'),receipts
            subjects=None
            if args.reuse_subject_observations is not None:
                from .dense_subject_catalogue import load_reused
                subjects=load_reused([args.reuse_subject_observations,*args.extra_subject_observations],sources=original.sources,
                    source_hashes=[v['source_sha256'] for v in bound_source_cut_events],banks=banks,nodes=nodes,
                    warps=warps,clocks=camera_census,master_start=original.master_audio_start,fps=original.fps,
                    authorized_root=args.cache.parent.parent,check=check,detailed_pose=args.detailed_pose)
            if args.original_person_observations is not None:
                from .original_people_bank import load_reference
                observed=load_reference(args.original_person_observations,sources=original.sources,
                    source_hashes=[v['source_sha256'] for v in bound_source_cut_events],banks=banks,
                    authorized_root=args.cache.parent.parent,check=check)
                for bank,people in zip(banks,observed,strict=True):bank['original_people_observations']=people
            opportunity_observer=None
            if args.opportunity_subject_spec is not None:
                from .subject_opportunity_probe import observer
                opportunity_observer=observer(args.opportunity_subject_spec,sources=original.sources,
                    source_hashes=[v['source_sha256'] for v in bound_source_cut_events],
                    identities=[file_identity(p) for p in original.sources],
                    cache=args.cache.parent/'crosscut2-subject-opportunities-v1',
                    authorized_root=args.cache.parent.parent,device=expected,image_changes=args.played_interval_utility,
                    explore_missing_action=args.explore_missing_action,
                    maximum_centres=args.opportunity_observation_centres,
                    maximum_seconds=args.maximum_opportunity_perception_seconds,
                    provisional_plan=args.provisional_subject_observation_plan,master_start=original.master_audio_start)
            direct_face_receipt=None
            if args.direct_native_face_observations is not None:
                from .direct_face_queries import bind_manifest,refresh_member_scores
                bindings,direct_face_receipt=bind_manifest(args.direct_native_face_observations,
                    banks=banks,nodes=nodes,sources=original.sources,
                    source_hashes=[v['source_sha256'] for v in bound_source_cut_events],
                    master_start=original.master_audio_start,fps=original.fps,
                    authorized_root=args.cache.parent.parent,
                    legacy_cache=args.cache.parent/'crosscut2-local-member-query-v1',
                    cache=args.cache.parent/'crosscut2-direct-native-arcface-v2',native_ports=native_ports,
                    bound_native_events=bound_source_cut_events)
                if args.individual_member_scores:
                    from crosscut.ive_local_arcface_gallery import LocalIVEArcFaceGallery
                    gallery=LocalIVEArcFaceGallery.from_existing_photo_banks(
                        args.local_ive_photo_tool/'face_models/user_member_arcface_embeddings.json',
                        args.local_ive_photo_tool/'face_models/yujin_arcface_embeddings.json')
                    identity_hints,scoring=refresh_member_scores(banks,bindings,identity_hints,gallery=gallery,check=check)
                    direct_face_receipt['current_member_scores']=scoring
            runs,states,data,score,fixed_receipt,event_ports=select(nodes,banks,poses,style,fps=original.fps,
                source_events=source_events,body_states=body_states,native_ports=native_ports,music=music_strengths,check=check,
                identity_hints=identity_hints,part_observer=part_observer,preference_model=preference_model,
                maximum_seconds=selection_seconds,
                maximum_preparation_seconds=240.+(args.maximum_opportunity_perception_seconds
                    if args.opportunity_subject_spec is not None else 0.),
                preference_workspace=args.cache.parent/'crosscut2-transient-preference',connection_prior=connection_prior,
                person_observer=person_observer,subjects=subjects,opportunity_observer=opportunity_observer,
                sequence_preference_model=sequence_preference_model,
                sequence_context_kernel=args.experimental_sequence_kernel,
                sequence_source_bindings=None if sequence_preference_model is None else
                    [v['source_sha256'] for v in bound_source_cut_events])
            plan,seams=realize(original,warps,nodes,runs,states,data,output=args.output,measured=event_ports)
            fixed_receipt['native_style_observations']=style_receipts
            fixed_receipt['source_event_union']=union_receipts
            fixed_receipt['native_event_registry']=registry_receipts
            fixed_receipt['supplementary_native_flanks']=supplementary_flank_receipts
            fixed_receipt['recurrent_local_member_hints']=identity_receipts
            fixed_receipt['direct_native_face_observations']=direct_face_receipt
            fixed_receipt['sparse_face_endpoint_completion']=face_completion
            fixed_receipt['selection_runtime_limit_seconds']=selection_seconds
            fixed_receipt['selected_clock_check']=check_selected_clock(plan,nodes,runs,states,banks,data=data,
                native_ports=native_ports,check=check)
            if args.common_evidence_connection_prior or args.played_face_connection_prior or args.played_pose_port_camera_prior:
                verified=fixed_receipt['selected_clock_check']
                if not (verified['all_source_clocks_verified'] and verified['source_observation_ordinals_exact']
                        and verified['maximum_native_ordinal_difference']==0):
                    raise ValueError('common-evidence trial refuses to render an unverified or differing source clock')
                if args.played_pose_port_camera_prior and not verified['camera_transform_replay_exact_within_roundoff']:
                    raise ValueError('E refuses to render cameras different from its actual scored strips/path')
                if args.played_pose_port_camera_prior and not verified['retained_native_clock_verified']:
                    raise ValueError('E refuses differing retained native first-incoming clocks or measured flanks')
        else:
            runs, states, data, score, fixed_receipt = select_fixed_holds(nodes, banks, poses,
                fps=original.fps, source_events=source_events, broadcast_slots=slots, check=check,body_states=body_states,
                native_ports=native_ports,boundary_readability=args.boundary_native_readability,
                music_strengths=music_strengths,opportunity_basis=args.opportunity_basis)
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
            bound_source_cut_events=bound_source_cut_events,
            native_readability_observations=readability_receipts,
            native_camera_port_observations=native_port_receipts,
            stage_edit_transformer_used=False,
            conditional_body_perception=body_receipts,
            supplementary_wholebody_perception=supplementary_receipts,
            body_observations_loaded=poses is not None,
            played_pose_cost_used=args.played_pose_port_camera_prior,
            person_visible_extent_fallback_used=args.person_rectangle_model is not None,
            ownership_certified=False,
            conditional_body_action_costs_enabled=poses is not None and not args.played_face_connection_prior,
            fixed_actor_camera_hold_receipt=fixed_receipt,
            grid_nodes=len(nodes), objective_cost=score, seams=seams,
            retained_source_cut_counts=[len(events) for events in source_events],
            camera_census_all_sources_decoded=all(record['all_encoded_frames_decoded'] for record in camera_census),
            retained_source_cut_detection_completeness_validated=False,
            complete_actor_motion_validated=False, professional_quality_approved=False,
            normal_app_path_executed=False,
            old_or_manual_editorial_choices_used=bool(preference_model is not None and
                'reference_editor_choice_weak_imitation' in preference_model.artifact['supervision_kinds']),
            old_or_manual_editorial_timeline_directly_copied=False,
            preference_model_supervision=[] if preference_model is None else preference_model.artifact['supervision_kinds'],
            fixed_time_sequence_preference_used=sequence_preference_model is not None,
            sequence_preference_model_sha256=None if sequence_preference_model is None else
                sequence_preference_model.artifact['content_sha256'],
            fixed_connection_prior_used=connection_prior is not None,
            common_evidence_connection_prior_used=args.common_evidence_connection_prior,
            played_face_connection_prior_used=args.played_face_connection_prior,
            played_pose_port_camera_prior_used=args.played_pose_port_camera_prior,
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
