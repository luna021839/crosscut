"""Bounded, cache-only focal editing construction for real-media review.

This is not raw-parent admission, named identity or professional-quality QA.
The caller freezes roles, observation inventory and native clocks before cost;
no detector, network call, completion retry or camera refit happens here.
"""
from bisect import bisect_right
from dataclasses import dataclass
import math
import time

import numpy as np

from .group_review_two_shot_camera import (
    measure_review_focal_incident, shared_focal_extent_domain,
)


@dataclass(frozen=True)
class ObservedProgrammeBudget:
    beam_width: int = 16
    maximum_incidents: int = 20000
    maximum_seconds: float = 20.
    minimum_shot_ticks: int = 14
    maximum_zoom: float = 2.

    def __post_init__(self):
        if (type(self.beam_width) is not int or not 1 <= self.beam_width <= 32
                or type(self.maximum_incidents) is not int or not 1 <= self.maximum_incidents <= 20000
                or type(self.minimum_shot_ticks) is not int or self.minimum_shot_ticks < 6
                or type(self.maximum_seconds) not in (int, float)
                or not math.isfinite(self.maximum_seconds) or not 0 < self.maximum_seconds <= 60
                or type(self.maximum_zoom) not in (int, float)
                or not math.isfinite(self.maximum_zoom) or not 1 <= self.maximum_zoom <= 2.4):
            raise ValueError('observed programme needs a finite predeclared review budget')


class ObservedProgrammeUnresolved(ValueError):
    """Persistable unknown outcome, not proof of bad geometry or a render."""
    def __init__(self, layer, *, attempted, measured, unknown, elapsed_seconds):
        super().__init__(f'observed programme layer {layer} has no known legal path; no completion retry')
        self.report = dict(contract='observed_focal_programme_unresolved_v1', layer=layer,
            attempted_incidents=attempted, measured_incidents=measured, unknown_alternatives=unknown,
            elapsed_seconds=elapsed_seconds, observed_geometry_rejected=False,
            new_model_calls=0, rendered_movie=False, professional_quality_approved=False)


def focal_extent(face):
    xy = np.asarray(face['landmarks_xy'], float)
    if xy.shape != (5, 2) or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
        raise ValueError('focal programme needs actual finite five-point landmarks')
    center = xy.mean(0)
    size = float(np.sqrt(np.mean(np.sum(((xy-center)*[1., 9/16])**2, axis=1))))
    if size <= 1e-6:
        raise ValueError('focal programme has degenerate facial extent')
    return center, size


def coupled_focal_cameras(a, b, *, maximum_zoom=2.):
    ac, sa = focal_extent(a['face'])
    bc, sb = focal_extent(b['face'])
    common = shared_focal_extent_domain(centers=(tuple(ac), tuple(bc)),
        extents=(sa, sb), maximum_zoom=maximum_zoom, minimum_extent=1.12*max(sa, sb))
    if common is None:
        return None
    target = np.clip(np.array((.5, .42)), common['lower'], common['upper'])
    values = []
    for center, size in ((ac, sa), (bc, sb)):
        zoom = common['extent']/size
        c = center + (np.array((.5, .5))-target)/zoom
        values.append([zoom, float(c[0]), float(c[1]), 0.])
    return values


def single_focal_camera(row, maximum_zoom):
    center, _ = focal_extent(row['face'])
    height = row['face']['box_xywh'][3]
    if not math.isfinite(height) or height <= 0:
        raise ValueError('observed focal head has no finite extent')
    zoom = min(maximum_zoom, max(1., .28/height))
    c = np.clip(center + (np.array((.5, .5))-np.array((.5, .42)))/zoom,
        .5/zoom, 1-.5/zoom)
    return [zoom, float(c[0]), float(c[1]), 0.]


def project_focal_observation(row, camera):
    zoom, x, y, roll = camera
    if roll != 0:
        raise ValueError('observed programme currently supports only actual zero-roll crops')
    def point(p):
        return (float((p[0]-x)*zoom+.5), float((p[1]-y)*zoom+.5))
    face = dict(row['face'], landmarks_xy=[point(p) for p in row['face']['landmarks_xy']])
    body, points = row.get('body'), {}
    if body and body['reason'] == 'observed' and body['face_anchor_consistent'] is True:
        for j, (bx, by, confidence) in enumerate(body['upper_body_xyc'][:6]):
            if confidence >= .3 and 0 <= bx <= 1 and 0 <= by <= 1:
                p = point((bx, by))
                if all(0 <= v <= 1 for v in p):
                    points[j] = p
    return face, points


def select_observed_focal_programme(*, first, end, layer_sources, cut_centers, roles,
        lookup, native_at, source_native_cuts, pose_model_identity, fps,
        cut_radius=2, phases=(-2, 0, 2), budget=ObservedProgrammeBudget()):
    """Select one finite observed programme, leaving unknown terms unknown.

    ``lookup(source,tick,phase,role)`` must be cache-only, source/role frozen.
    Native ordinals must come from the declared clock, not nearby-frame repair.
    Missing anatomy keeps the fixed six-joint upper cost in every comparison.
    Entry/return are NOT declared pose matches; the caller owns their review.
    """
    if (type(first) is not int or type(end) is not int or not 0 <= first < end
            or end-first > 1200 or type(cut_radius) is not int or not 0 <= cut_radius <= 3
            or not isinstance(budget, ObservedProgrammeBudget)
            or not isinstance(phases, tuple) or not phases or len(phases) > 5
            or any(type(p) is not int or abs(p) > 2 for p in phases)
            or len(set(phases)) != len(phases)
            or not 1 <= len(cut_centers) <= 7 or len(layer_sources) != len(cut_centers)+1
            or len(roles) != len(layer_sources) or any(not isinstance(r, str) or not r for r in roles)
            or tuple(sorted(set(cut_centers))) != tuple(cut_centers)
            or any(type(t) is not int or not first+2 < t < end-2 for t in cut_centers)
            or any(not isinstance(s, tuple) or not s or len(s) > 8 or len(set(s)) != len(s)
                or any(type(i) is not int or i < 0 or i not in source_native_cuts for i in s) for s in layer_sources)
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or not isinstance(pose_model_identity, str) or len(pose_model_identity) != 64):
        raise ValueError('observed programme needs a frozen bounded source/role/cut/phase inventory')
    for cuts in source_native_cuts.values():
        if tuple(cuts) != tuple(sorted(set(cuts))) or any(type(n) is not int or n < 0 for n in cuts):
            raise ValueError('observed programme native cuts must be exact ordered ordinals')
    started = time.monotonic()
    deadline = started+budget.maximum_seconds
    measured = unknown = attempted = 0
    cache = {}

    def row(source, tick, phase, role):
        key = source, tick, phase, role
        if key not in cache:
            value = lookup(*key)
            if value is not None:
                if (value['source_index'] != source
                        or value['native_ordinal'] != native_at(source, tick)+phase
                        or not isinstance(value.get('native_pixel_sha256'), str)
                        or len(value['native_pixel_sha256']) != 64):
                    raise ValueError('observed focal row lost its exact declared native parent')
                focal_extent(value['face'])
            cache[key] = value
        return cache[key]

    def valid(source, start, stop, phase):
        cuts = source_native_cuts[source]
        a, b = native_at(source, start)+phase, native_at(source, stop-1)+phase
        return stop-start >= budget.minimum_shot_ticks and 0 <= a <= b and bisect_right(cuts, a) == bisect_right(cuts, b)

    beam = []
    for source in layer_sources[0]:
        for phase in phases:
            start = row(source, first, phase, roles[0])
            if start is not None:
                beam.append(dict(score=.05*(budget.maximum_zoom-single_focal_camera(start, budget.maximum_zoom)[0]),
                    source=source, phase=phase, first=first, start=start, complete=[], seams=[]))
    for layer, center in enumerate(cut_centers):
        next_beam = []
        for state in beam:
            for source in layer_sources[layer+1]:
                if source == state['source'] and roles[layer] == roles[layer+1]:
                    continue
                for phase in phases:
                    for cut in range(center-cut_radius, center+cut_radius+1):
                        attempted += 1
                        if attempted > budget.maximum_incidents or time.monotonic() > deadline:
                            raise TimeoutError('observed programme exhausted its original finite budget; no retry')
                        if not valid(state['source'], state['first'], cut, state['phase']):
                            continue
                        a = [row(state['source'], t, state['phase'], roles[layer]) for t in (cut-2, cut-1)]
                        b = [row(source, t, phase, roles[layer+1]) for t in (cut, cut+1)]
                        if any(r is None for r in (*a, *b)):
                            unknown += 1
                            continue
                        cameras = coupled_focal_cameras(a[-1], b[0], maximum_zoom=budget.maximum_zoom)
                        if cameras is None:
                            continue
                        ap = [project_focal_observation(r, cameras[0]) for r in a]
                        bp = [project_focal_observation(r, cameras[1]) for r in b]
                        def signatures(rows):
                            return tuple((str(r['source_index']), pose_model_identity, r['source_index'],
                                bisect_right(source_native_cuts[r['source_index']], r['native_ordinal']),
                                r['native_ordinal']) for r in rows)
                        proof = measure_review_focal_incident(left_faces=tuple(p[0] for p in ap),
                            right_faces=tuple(p[0] for p in bp), left_points=tuple(p[1] for p in ap),
                            right_points=tuple(p[1] for p in bp), left_ticks=(cut-2, cut-1),
                            right_ticks=(cut, cut+1), fps=fps, left_native=signatures(a), right_native=signatures(b))
                        handoff = roles[layer] != roles[layer+1]
                        proof['singing_part_handoff_not_same_member_pose_match'] = handoff
                        similarity = float(np.dot(a[-1]['sface_vector'], b[0]['sface_vector']))
                        amount = proof['exploration_score']+(0. if handoff else .12*(1-float(np.clip(similarity,-1.,1.))))
                        measured += 1
                        clip = dict(first=state['first'], end=cut, source_index=state['source'],
                            phase_native_frames=state['phase'], start_observation=state['start'], end_observation=a[-1],
                            start_camera=state.get('start_camera') or single_focal_camera(state['start'], budget.maximum_zoom),
                            end_camera=cameras[0])
                        seam = dict(output_tick=cut, seconds=cut/fps, left=a[-1], right=b[0],
                            left_source=state['source'], right_source=source, actual_fixed_inventory_incident=proof,
                            face_scale_basis='actual_five_landmark_rms', named_identity_certified=False,
                            body_ownership_certified=False, quality_approved=False)
                        next_beam.append(dict(score=state['score']+amount+.001*abs(cut-center)+.001*abs(phase),
                            source=source, phase=phase, first=cut, start=b[0], start_camera=cameras[1],
                            complete=[*state['complete'], clip], seams=[*state['seams'], seam]))
        beam = sorted(next_beam, key=lambda r:(r['score'], r['source'], r['phase'], r['first']))[:budget.beam_width]
        if not beam:
            raise ObservedProgrammeUnresolved(layer, attempted=attempted, measured=measured,
                unknown=unknown, elapsed_seconds=time.monotonic()-started)
    completed = []
    for state in beam:
        if not valid(state['source'], state['first'], end, state['phase']):
            continue
        last = row(state['source'], end-1, state['phase'], roles[-1])
        if last is None:
            continue
        clip = dict(first=state['first'], end=end, source_index=state['source'], phase_native_frames=state['phase'],
            start_observation=state['start'], end_observation=last, start_camera=state['start_camera'],
            end_camera=single_focal_camera(last, budget.maximum_zoom))
        clips = [*state['complete'], clip]
        cameras = {}
        legal = True
        for c in clips:
            for tick in range(c['first'], c['end']):
                t = (tick-c['first'])/max(1, c['end']-c['first']-1)
                t = t*t*(3-2*t)
                camera = ((1-t)*np.asarray(c['start_camera'])+t*np.asarray(c['end_camera'])).tolist()
                z, x, y, roll = camera
                if not (1-1e-9 <= z <= budget.maximum_zoom+1e-9 and roll == 0 and
                        all(-1e-9 <= v <= 1+1e-9 for v in (x-.5/z,x+.5/z,y-.5/z,y+.5/z))):
                    legal = False
                    break
                cameras[tick] = dict(tick=tick, source_index=c['source_index'],
                    native_ordinal=native_at(c['source_index'],tick)+c['phase_native_frames'],
                    phase_native_frames=c['phase_native_frames'], camera=camera)
            if not legal:
                break
        if legal:
            completed.append((state['score'], clips, state['seams'], cameras))
    if not completed:
        raise ValueError('observed programme has no fully raster-covered native sequence')
    score, clips, seams, cameras = min(completed, key=lambda r:r[0])
    return dict(contract='bounded_observed_focal_programme_review_v1', clips=clips, seams=seams,
        cameras=cameras, score=score, measured_incidents=measured, unknown_alternatives=unknown,
        attempted_incidents=attempted, beam_width=budget.beam_width, completion_retries=0,
        new_model_calls=0, camera_refit_after_selection=False, normal_app_admitted=False,
        full_hold_action_certified=False, entry_return_pose_certified=False, professional_quality_approved=False)
