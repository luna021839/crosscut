"""MEMBER dissolves on a fixed source/camera programme, never reselection.

Requests cover the actual outgoing physical handle and incoming frame clock.
Only missing native pictures need observing. Nominal cuts and camera paths
remain untouched; the user can turn the effect off without another solve.
"""
from dataclasses import replace
import numpy as np

from crosscut.render import (CACHED_MEMBER_SHORT_DISSOLVE_REASON,
    MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY, EditorialTransformKeyframe,
    EditorialTransformTrack, NormalizedCrop, RenderTransition, TransitionKind, _base_crop_values,
    FACE_DISSOLVE_ALPHA_CURVE)
from crosscut.render_frame_schedule import sampler_frame_schedule
from .bounded_plan_render import outgoing_handle
from .member_short_dissolve import assess
from .member_viewport_observations import compose_camera, project_box, project_points


def view(segment, seconds):
    base = NormalizedCrop(*_base_crop_values(segment, seconds))
    if segment.editorial_transform is None:
        return base
    value = segment.editorial_transform.evaluate(seconds)
    if value.roll_degrees:
        raise ValueError('this measured MEMBER dissolve requires an unrotated physical view')
    return compose_camera(base, [value.zoom, value.center_x, value.center_y])


def windows(plan, sources, *, frame_count=6, current_faces=None):
    """Return all declared real overlaps and deduplicated missing native reads."""
    if (type(frame_count) is not int or not 3 <= frame_count <= 60 or len(sources) != len(plan.sources)
            or tuple(s.path for s in sources) != plan.sources):
        raise ValueError('fixed programme sources and a bounded physical overlap required')
    current_faces = [{} for _ in sources] if current_faces is None else current_faces
    if len(current_faces) != len(sources):
        raise ValueError('one current native face bank per source required')
    requested = [set() for _ in sources]; reports = []; reused = 0
    ticks = (0, frame_count//2, frame_count-1)
    for index, (left, right) in enumerate(zip(plan.segments, plan.segments[1:])):
        if round(right.duration*plan.fps) <= frame_count:
            reports.append(dict(boundary=index, seconds=right.master_start, physical_overlap_available=False, samples=[]))
            continue
        handle = outgoing_handle(left, frame_count=frame_count, fps=plan.fps)
        pair = (handle, right); natives = []
        try:
            for segment in pair:
                source = sources[segment.source_index]
                natives.append(sampler_frame_schedule(source_start=segment.source_start,
                    source_playback_rate=segment.source_playback_rate, native_fps=source.fps,
                    output_fps=plan.fps, frame_count=frame_count, source_frame_count=source.frame_count).source_frame_indices)
        except ValueError:
            reports.append(dict(boundary=index, seconds=right.master_start, physical_overlap_available=False, samples=[]))
            continue
        samples = []
        for tick in ticks:
            for side, segment in enumerate(pair):
                source = segment.source_index; native = int(natives[side][tick])
                known = native in current_faces[source]; reused += known
                if not known: requested[source].add(native)
                crop = view(segment, tick/plan.fps)
                samples.append(dict(side=side, source=source, native=native, relative_frame=tick,
                    crop=[crop.x, crop.y, crop.width, crop.height], existing_observation_reused=known))
        reports.append(dict(boundary=index, seconds=right.master_start, physical_overlap_available=True, samples=samples))
    return dict(frame_count=frame_count, ticks=list(ticks), boundaries=reports,
        source_requests=[dict(source=s, natives=sorted(values)) for s, values in enumerate(requested)],
        requested_native_rasters=sum(map(len, requested)), existing_query_reuses=reused,
        source_or_cut_sequence_changed=False, whole_source_reanalysis=False)


def evaluate(window, *, current_faces, original_geometry, sources, member, output_size):
    """Use observed target faces; missing samples decline only this effect."""
    if not window['physical_overlap_available']:
        return dict(boundary=window['boundary'], seconds=window['seconds'], eligible=False,
                    reason='outgoing_handle_outside_physical_source')
    boxes = np.zeros((3, 2, 4)); known = np.zeros((3, 2), bool); poses = np.zeros((3, 2, 8, 3))
    ticks = sorted({v['relative_frame'] for v in window['samples']}); bindings = []
    if len(ticks) != 3 or len(window['samples']) != 6:
        raise ValueError('three actual paired physical overlap instants required')
    for value in window['samples']:
        t, side = ticks.index(value['relative_frame']), value['side']
        s, n = value['source'], value['native']; row = current_faces[s].get(n)
        crop = NormalizedCrop(*value['crop']); candidates = []
        for face in ([] if row is None else row['faces']):
            if face['member'] != member:
                continue
            box = np.array(project_box(face['bbox_xywh'], crop)); xy, wh = box[:2], box[2:]
            fraction = np.maximum(0., np.minimum(1., xy+wh)-np.maximum(0., xy)).prod()/wh.prod()
            if fraction >= .9:
                candidates.append((face['member_scores'][member], face, box))
        bound = False
        if candidates:
            _, face, box = max(candidates, key=lambda v: v[0]); boxes[t, side] = box; known[t, side] = True
            old = original_geometry[s].get(n)
            if old is not None and old['face_bbox_xywh'] is not None and old.get('pose_screen_xyc'):
                a, b = np.array(old['face_bbox_xywh']), np.array(face['bbox_xywh'])
                distance = np.linalg.norm((a[:2]+a[2:]/2-b[:2]-b[2:]/2)*[sources[s].width/sources[s].height, 1.])/max(.01, b[3])
                if distance <= .5 and abs(np.log(a[3]/b[3])) <= .4:
                    poses[t, side] = old['pose_screen_xyc']
                    poses[t, side, :, :2] = project_points(poses[t, side, :, :2], crop); bound = True
        bindings.append(dict(source=s, native=n, relative_frame=value['relative_frame'],
            current_target_face_observed=bool(known[t, side]), original_pose_geometrically_bound_to_current_face=bound))
    return dict(boundary=window['boundary'], seconds=window['seconds'],
                **assess(boxes, known, poses, output_size=output_size), source_samples=bindings)


def apply(plan, reports, *, frame_count=6, enabled=True, minimum_spacing_seconds=3.5,
          alpha_curve=FACE_DISSOLVE_ALPHA_CURVE):
    """Retain all nominal pictures/cameras; apply only the explicit preference."""
    if type(enabled) is not bool or type(frame_count) is not int or not 3 <= frame_count <= 60:
        raise ValueError('explicit MEMBER effect preference and bounded overlap required')
    if not np.isfinite(minimum_spacing_seconds) or minimum_spacing_seconds < 0:
        raise ValueError('finite nonnegative effect spacing required')
    chosen = []
    if enabled:
        for candidate in sorted((r for r in reports if r['eligible']), key=lambda r: (r['score'], r['seconds'])):
            index = candidate['boundary']
            if (type(index) is not int or not 0 <= index < len(plan.segments)-1
                    or candidate['seconds'] != plan.segments[index+1].master_start):
                raise ValueError('effect report differs from the fixed programme boundary')
            if all(abs(candidate['seconds']-r['seconds']) >= minimum_spacing_seconds for r in chosen):
                chosen.append(candidate)
    selected = {r['boundary']: r for r in chosen}; segments = list(plan.segments); transitions = []
    for index in range(len(segments)-1):
        if index not in selected:
            segments[index] = replace(segments[index], outgoing_overlap_transform=None)
            transitions.append(RenderTransition()); continue
        segment = segments[index]
        if segment.editorial_transform is not None:
            final = segment.editorial_transform.final_keyframe
            future = EditorialTransformTrack(tuple(EditorialTransformKeyframe(t, final.zoom, final.center_x,
                final.center_y, final.roll_degrees) for t in (0., (frame_count-1)/plan.fps)),
                reason='current_source_face_and_arm_short_overlap_hold',
                sampling_canvas_zoom=segment.editorial_transform.sampling_canvas_zoom)
            segments[index] = replace(segment, outgoing_overlap_transform=future)
        result = selected[index]
        transitions.append(RenderTransition(TransitionKind.FACE_DISSOLVE, frame_count,
            reason=CACHED_MEMBER_SHORT_DISSOLVE_REASON,
            transform_overlap_authority=MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY,
            face_mismatch=result['maximum_face_position_in_face_heights'],
            future_pose_max_mismatch=result['maximum_visible_arm_direction_mismatch'],
            face_dissolve_curve=alpha_curve))
    new = replace(plan, segments=tuple(segments), transitions=tuple(transitions),
        member_pose_dissolve_requested=enabled, member_pose_dissolve_effective=bool(chosen), member_pose_dissolve_count=len(chosen))
    return new, dict(actual_effect_count=len(chosen), effect_seconds=sorted(r['seconds'] for r in chosen),
        source_or_cut_sequence_changed=False, effect_duration_seconds=frame_count/plan.fps,
        minimum_effect_spacing_seconds=minimum_spacing_seconds, alpha_curve=alpha_curve)
