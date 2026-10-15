"""Choose a musical cut moment, not a different source/audio alignment.

Both clocks move together through immutable subject observations.  Whole-window
pose averages are context; they cannot conceal a wrist mismatch at the actual
seam.  Results are candidate proposals, never visual-quality certification.
"""
from dataclasses import replace
import hashlib

import numpy as np

from .subject_action_match import SubjectClock, _sample, compare


def _at_moment(clock, center, radius):
    selected = (clock.output_frames >= center - radius) & (clock.output_frames <= center + radius)
    ticks = clock.output_frames[selected] - center
    if not np.array_equal(ticks, np.arange(-radius, radius + 1)):
        raise ValueError('cut moment needs a complete common-clock neighbourhood')
    native = clock.native_indices[selected].copy()
    ticks = ticks.copy()
    signature = hashlib.sha256(
        clock.schedule_sha256.encode() + ticks.tobytes() + native.tobytes()
    ).hexdigest()
    ticks.setflags(write=False)
    native.setflags(write=False)
    return replace(clock, output_frames=ticks, native_indices=native, schedule_sha256=signature)


def _normalized_body(window, indices):
    xy = window.body[indices, :, :2] * [16 / 9, 1.]
    heads = window.heads[indices]
    shoulders = window.support[indices, :2].all(axis=1)
    root = np.where(shoulders[:, None], xy[:, :2].mean(axis=1), heads[:, :2] * [16 / 9, 1.])
    return (xy - root[:, None]) / np.maximum(.02, heads[:, None, 2, None])


def _seam_action(left, right, lc, rc, geometry):
    """Measure arms at -1/0 and their contemporary motion, not just shoulders.

    A shoulder-only estimate cannot establish an articulated arm connection.
    Missing observations stay missing; no interpolation or neutral cost is used.
    """
    ai, bi = _sample(left, lc), _sample(right, rc)
    common = left.support[ai] & right.support[bi]
    common &= left.associated[ai, None] & right.associated[bi, None]
    common &= np.asarray(geometry['common_source_shot_sample_mask'])[:,None]
    seam = np.flatnonzero(np.isin(lc.output_frames, [-1, 0]))
    if len(seam) != 2:
        raise ValueError('physical outgoing last and incoming first frames required')
    complete = (left.support[ai[seam[0]]]&right.support[bi[seam[1]]]
                &left.associated[ai[seam[0]]]&right.associated[bi[seam[1]]])
    # Body8 order: shoulders, elbows, wrists, hips. Require at least one
    # complete measured arm on BOTH sides of the cut, including its wrist.
    arms = [(0, 2, 4), (1, 3, 5)]
    measured_arms = [arm for arm in arms if complete[list(arm)].all()]
    if not measured_arms:
        return dict(seam_action_supported=False, reason='no_complete_measured_arm_at_seam')
    joints = sorted({joint for arm in measured_arms for joint in arm[1:]})
    na = _normalized_body(left, ai)
    nb = _normalized_body(right, bi)
    measured=common[seam][:,joints]
    if not measured.any():
        return dict(seam_action_supported=False, reason='no_common_music_sample_next_to_played_seam')
    delta = na[seam][:, joints] - nb[seam][:, joints]
    # RMS over arm joints makes a badly displaced wrist visible in the cost;
    # adding several correctly matched shoulders cannot dilute it.
    pose = float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1)[measured])))
    directions = []
    xy_a = left.body[ai[seam[:1]], :, :2] * [16 / 9, 1.]
    xy_b = right.body[bi[seam[1:]], :, :2] * [16 / 9, 1.]
    for shoulder, elbow, wrist in measured_arms:
        for first, last in ((shoulder, elbow), (elbow, wrist)):
            da = xy_a[:, last] - xy_a[:, first]
            db = xy_b[:, last] - xy_b[:, first]
            la, lb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
            valid = (la > .005) & (lb > .005)
            cosine = np.sum(da * db, axis=1) / np.maximum(1e-12, la * lb)
            directions.extend((np.arccos(np.clip(cosine, -1., 1.)) / np.pi)[valid])
    if not directions:
        return dict(seam_action_supported=False, reason='no_measured_arm_directions_at_seam')
    ca = np.asarray(geometry['outgoing_camera'])
    cb = np.asarray(geometry['incoming_camera'])
    pa = .5 + ca[0] * (left.body[ai][:, joints, :2] - ca[1:])
    pb = .5 + cb[0] * (right.body[bi][:, joints, :2] - cb[1:])
    scale = np.maximum(.02, (left.heads[ai, 2] * ca[0] + right.heads[bi, 2] * cb[0]) / 2)
    projected = (pa[seam] - pb[seam]) * [16 / 9, 1.]
    screen = float(np.sqrt(np.mean((np.sum(projected * projected, axis=-1) / scale[seam, None] ** 2)[measured])))
    motions = []
    activities = []
    # Match both films over the SAME music intervals. Comparing unrelated
    # outgoing/incoming slopes would penalize real choreography reversals.
    for first_tick, last_tick in [(-4, -1), (-1, 0), (0, 3)]:
        first = np.flatnonzero(lc.output_frames == first_tick)
        last = np.flatnonzero(lc.output_frames == last_tick)
        if not len(first) or not len(last):
            continue
        first, last = int(first[0]), int(last[0])
        for joint in joints:
            if not common[[first, last], joint].all():
                continue
            if (not left.associated[ai[first]:ai[last] + 1].all()
                    or not right.associated[bi[first]:bi[last] + 1].all()):
                continue
            da = na[last, joint] - na[first, joint]
            db = nb[last, joint] - nb[first, joint]
            motions.append(float(np.linalg.norm(da - db)))
            activities.append(float(max(np.linalg.norm(da), np.linalg.norm(db))))
    if not motions:
        return dict(seam_action_supported=False, reason='no_measured_motion_near_seam')
    return dict(seam_action_supported=True, reason='measured_articulated_seam',
                seam_arm_count=len(measured_arms), seam_arm_joints=joints,
                seam_arm_direction_residual=float(np.mean(directions)),
                seam_pose_residual=pose, seam_screen_residual=screen,
                seam_motion_residual=float(np.mean(motions)),
                seam_activity=float(np.mean(activities)))


def search(left, right, left_clock, right_clock, *, candidate_frames,
           outgoing_cap, incoming_cap, radius=6):
    """Rank alternative edit instants with no independent source retiming.

    Candidate frames are output/music offsets from the supplied anchor.  This
    is a pre-selection camera proposal and changes both views explicitly.
    It is not an H/K matched-prefix comparison or a patch to a frozen edit.
    """
    if (not isinstance(left_clock, SubjectClock) or not isinstance(right_clock, SubjectClock)
            or left_clock.phase_frames or right_clock.phase_frames
            or left_clock.fps != right_clock.fps
            or not np.array_equal(left_clock.output_frames, right_clock.output_frames)):
        raise ValueError('two zero-phase clocks on the same music timeline required')
    candidates = list(candidate_frames)
    if (not candidates or any(type(t) is not int for t in candidates)
            or len(set(candidates)) != len(candidates) or type(radius) is not int or radius < 4):
        raise ValueError('distinct integer cut moments and a measured context required')
    results = []
    for center in candidates:
        lc = _at_moment(left_clock, center, radius)
        rc = _at_moment(right_clock, center, radius)
        geometry = compare(left, right, lc, rc,
                           left_cameras=np.tile([1., .5, .5], (len(lc.output_frames), 1)),
                           incoming_cap=incoming_cap, outgoing_locked=False,
                           outgoing_cap=outgoing_cap)
        seam = _seam_action(left, right, lc, rc, geometry)
        score = None
        if geometry['status'] == 'SUPPORTED_CANDIDATE' and seam['seam_action_supported']:
            score = (seam['seam_pose_residual'] + seam['seam_screen_residual']
                     + seam['seam_motion_residual'] + geometry['face_residual']
                     + 1. - geometry['appearance_cosine'])
        results.append(dict(**geometry, **seam, cut_offset_frames=center,
                            cut_offset_seconds=center / lc.fps, cut_moment_score=score,
                            both_music_clocks_moved_together=True, source_retimed=False))
    # First match the articulated configuration, then screen placement and
    # motion. A small positional/velocity gain must not trade an overhead
    # crossed-arm pose for a differently directed, blurred low-arm swing.
    # This is an explicit engineering ordering, not a fitted human preference.
    supported = sorted((r for r in results if r['cut_moment_score'] is not None),
                       key=lambda r: (r['seam_arm_direction_residual'], r['cut_moment_score'],
                                      abs(r['cut_offset_frames']), r['cut_offset_frames']))
    return dict(candidates=results, supported=supported,
                status='SUPPORTED_CUT_MOMENTS' if supported else 'NO_SUPPORTED_CUT_MOMENT',
                rank_contract='articulated_configuration_then_screen_motion_v1',
                phase_search_performed=False, weights_fitted=False,
                professional_quality_approved=False)
