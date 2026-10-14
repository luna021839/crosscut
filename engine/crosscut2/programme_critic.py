"""Plan-aware actual-pixel diagnostics, not a professional quality certificate.

Inspect every authored seam, retained native cut and extra output detection.
Source mapping shares the renderer's verified native-frame schedule. Conditional
primary upper-body measurements do not establish all-frame actor ownership.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from crosscut.render_frame_schedule import probe_render_frame_clock, sampler_frame_schedule
from crosscut.render import TransitionKind
from crosscut.render_plan_serialization import render_plan_from_json
from .action_geometry import compare_action, compare_screen_motion
from .critic import PixelFaces, pair_features, scan_cuts
from .grid_perception import file_identity
from .pose_grid import PoseSensor
from .source_cuts import census
from .focal_output_correspondence import predictions_by_tick


def boundary_union(authored, native, detected, *, frame_count):
    """Exact output ticks; equal counts or nearby ticks do not prove coverage."""
    events = {}
    groups = ((authored, 'authored'), (native, 'retained_native'), (detected, 'detected'))
    for values, origin in groups:
        for tick in values:
            if isinstance(tick, bool) or not isinstance(tick, int) or not 0 < tick < frame_count:
                raise ValueError('boundary requires an interior integer output tick')
            events.setdefault(tick, set()).add(origin)
    return tuple(dict(frame=tick, origins=sorted(origins)) for tick, origins in sorted(events.items()))


def coverage(authored, native, detected):
    a, n, d = set(authored), set(native), set(detected)
    return dict(authored_count=len(a), retained_native_count=len(n), detected_count=len(d),
        missed_authored_by_detector=sorted(a - d), missed_native_by_detector=sorted(n - d),
        unexplained_detections=sorted(d - a - n),
        detector_authored_set_exact=a <= d)


def programme_schedules(plan, source_clocks, source_cut_frames):
    schedules, native, authored = [], [], []
    expected = 0
    for i, segment in enumerate(plan.segments):
        start, size = round(segment.master_start * plan.fps), round(segment.duration * plan.fps)
        if start != expected or size <= 0:
            raise ValueError('critic requires a contiguous finite hard-cut programme')
        clock = source_clocks[segment.source_index]
        if clock is None:
            raise ValueError('exact retained-cut mapping requires the renderer verified native clock')
        schedule = sampler_frame_schedule(source_start=float(segment.source_start),
            source_playback_rate=float(segment.source_playback_rate), native_fps=float(clock.frame_rate),
            output_fps=float(plan.fps), frame_count=size, source_frame_count=clock.frame_count,
            maximum_terminal_hold_frames=math.ceil(plan.fps))
        schedules.append((start, segment, schedule))
        if i:
            authored.append(start)
        for cut in source_cut_frames[segment.source_index]:
            # A cut before the first retained source frame cannot be visible
            # inside this hold. An exact authored overlap remains one event.
            if schedule.source_frame_indices[0] < cut <= schedule.source_frame_indices[-1]:
                offset = schedule.first_output_tick(cut)
                if 0 < offset < size:
                    native.append(start + offset)
        expected += size
    # RenderPlan expands implicit hard cuts into explicit HARD_CUT records.
    # Nonempty transition metadata is not evidence of a dissolve/overlap.
    if expected != plan.output_frame_count or any(t.kind is not TransitionKind.HARD_CUT for t in plan.transitions):
        raise ValueError('critic schedule must share the complete hard-cut renderer domain')
    return tuple(schedules), tuple(authored), tuple(sorted(set(native)))


def rendered_source_frame(schedules, tick):
    for start, segment, schedule in reversed(schedules):
        if start <= tick < start + len(schedule.source_frame_indices):
            offset = tick - start
            return dict(source=segment.source_index, native_frame=schedule.source_frame_indices[offset],
                output_frame=tick, segment_local_frame=offset,
                terminal_hold=offset in schedule.terminal_hold_output_indices)
    raise ValueError('output tick is outside the shared rendered source schedule')


def bound_source_cuts(records,union,source_bindings):
    if union is None:return records
    if len(union)!=len(records):raise ValueError('critic bound source-cut union domain differs')
    merged=[]
    for s,(base,value) in enumerate(zip(records,union)):
        frames=value.get('frames')
        if (value.get('contract') not in ('bound_transnet_adaptive_first_incoming_union_v1',
                'bound_source_cut_union_with_confirmed_output_observations_v2')
                or value.get('source')!=s or tuple(value.get('source_file_identity',()))!=source_bindings[s]
                or value.get('frame_count')!=base['frame_count'] or value.get('fps')!=base['fps']
                or not isinstance(frames,list) or any(type(f) is not int or not 0<f<base['frame_count'] for f in frames)
                or frames!=sorted(set(frames))):
            raise ValueError('critic native union is not bound to current source/cut clocks')
        merged.append(dict(base,frames=frames))
    return tuple(merged)


def evaluate_programme(movie, *, record, models, body_model, cut_cache, check, maximum_seconds=240.):
    started = monotonic()
    if not math.isfinite(maximum_seconds) or not 1 <= maximum_seconds <= 600:
        raise ValueError('one finite complete critic time budget required')
    def live():
        check()
        if monotonic() - started >= maximum_seconds:
            raise TimeoutError('one complete programme-critic deadline reached')
    movie = Path(movie)
    binding = file_identity(movie)
    plan = render_plan_from_json(record['render_plan'])
    if movie != plan.output:
        raise ValueError('critic movie differs from the selected exact plan')
    source_bindings = tuple(file_identity(path) for path in plan.sources)
    if source_bindings != tuple(tuple(r['source_file_identity']) for r in record['sparse_perception']):
        raise OSError('selected source version changed before programme evaluation')
    clock_started = monotonic()
    clocks = []
    for source in plan.sources:
        live()
        clocks.append(probe_render_frame_clock(source))
    records = census(plan.sources, cache=cut_cache, check=live)
    records=bound_source_cuts(records,record.get('bound_source_cut_events'),source_bindings)
    schedules, authored, native = programme_schedules(plan, clocks, [r['frames'] for r in records])
    mapping_seconds = monotonic() - clock_started
    fps, count, detected = scan_cuts(movie, check=live)
    if count != plan.output_frame_count or not math.isclose(fps, plan.fps, rel_tol=1e-8):
        raise OSError('encoded output clock differs from the selected renderer clock')
    events = boundary_union(authored, native, detected, frame_count=count)
    if len(events) > 512:
        raise ValueError('declared finite output observation bound exceeded')
    face = PixelFaces(models / 'face_detection_yunet_2023mar.onnx', models / 'face_recognition_sface_2021dec.onnx')
    pose = None if body_model is None else PoseSensor(models, body_model)
    cap = cv2.VideoCapture(str(movie))
    observed, rows = {}, []
    seam_intents = {row['output_frame']: row['role'] for row in record['seams']}
    predicted_focal=predictions_by_tick(record['seams'],count)
    def at(tick):
        live()
        tick = max(0, min(count - 1, tick))
        if tick not in observed:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, tick):
                raise OSError('output ordinal seek unavailable')
            ok, pixels = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != tick:
                raise OSError('critic did not receive the requested actual output frame')
            expected=predicted_focal.get(tick)
            faces, vector = face.read(pixels) if expected is None else face.read(pixels,expected_focal=expected)
            joints, state = np.zeros((8, 3), np.float32), 0
            if pose is not None:
                heads = np.zeros((3, 4), np.float32)
                for i, person in enumerate(faces[:3]):
                    heads[i] = (*person['center'], person['height'], person['confidence'])
                result, statuses = pose.observe(pixels, heads)
                joints, state = result[0], int(statuses[0])
            observed[tick] = faces, vector, joints, state,dict(face.last_focal_receipt)
        return observed[tick]
    try:
        for event in events:
            frame = event['frame']
            ticks = tuple(max(0, min(count - 1, frame + offset)) for offset in (-7, -1, 0, 6))
            windows = [at(tick) for tick in ticks]
            left,lv=windows[1][:2]
            right,rv=windows[2][:2]
            similarity = None if lv is None or rv is None else float(np.clip(lv @ rv, -1., 1.))
            a, b = np.asarray([w[2] for w in windows[:2]]), np.asarray([w[2] for w in windows[2:]])
            action = compare_action(a, b, seconds=6 / fps)
            motion = compare_screen_motion(a, b, seconds=6 / fps, frame_seconds=1 / fps)
            rows.append(dict(event, seconds=frame / fps, planned_intent=seam_intents.get(frame),
                actual_output_ticks=ticks, rendered_source_frames=[rendered_source_frame(schedules, t) for t in ticks],
                conditional_primary_body_states=[w[3] for w in windows],
                conditional_primary_upper_body_xyc=[w[2].tolist() for w in windows],
                conditional_primary_body_phase=action.phase_loss,
                conditional_primary_body_velocity=action.velocity_loss,
                conditional_primary_action_basis=action.measurement_basis,
                conditional_primary_comparable_joints=action.common_joints,
                conditional_primary_action_measured=action.phase_loss is not None,
                focal_correspondence=[w[4] for w in windows],
                unassociated_output_geometry=pair_features(windows[1][4]['candidates'],windows[2][4]['candidates']),
                realized_screen_motion=motion,
                **pair_features(left, right, appearance_cosine=similarity)))
            if len(rows) % 8 == 0:
                print(json.dumps(dict(phase='programme_pixel_critic', boundaries=len(rows),
                    total_boundaries=len(events), neural_pose_calls=0 if pose is None else pose.calls,
                    elapsed_seconds=monotonic() - started)), flush=True)
    finally:
        cap.release()
    if binding != file_identity(movie) or source_bindings != tuple(file_identity(p) for p in plan.sources):
        raise OSError('movie/source version changed during complete programme critic')
    evaluated = {r['frame'] for r in rows}
    if not set(authored) <= evaluated or not set(native) <= evaluated:
        raise AssertionError('an expected authored/native boundary escaped actual-pixel evaluation')
    return dict(contract='selected_programme_pixel_critic_v1', movie=movie.name,
        output_frame_count=count, fps=fps, source_file_identity=binding,
        boundary_coverage=coverage(authored, native, detected), evaluated_boundary_count=len(rows),
        all_authored_boundaries_evaluated=set(authored) <= evaluated,
        all_exactly_mapped_retained_native_boundaries_evaluated=set(native) <= evaluated,
        source_mapping_seconds=mapping_seconds, observed_output_frames=len(observed),
        neural_pose_calls=0 if pose is None else pose.calls,
        conditional_primary_body_measured_cuts=sum(r['conditional_primary_action_measured'] for r in rows),
        body_target_definition=('planned physical focal when geometrically matched; UNKNOWN on miss; output-only largest at unpredicted frames'
            if predicted_focal else 'largest detected visible head per actual frame; not source-local actor certification'),
        predicted_focal_output_frames=len(predicted_focal),
        matched_predicted_focal_output_frames=sum(v[4].get('matched',False) for v in observed.values()),
        body_action_complete=False, learned_quality_critic=False, held_out_quality_validated=False,
        private_gallery_accessed=False, private_vectors_serialized=False,
        professional_quality_approved=False, normal_speed_audio_playback_performed=False,
        cuts=rows, elapsed_seconds=monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('movie', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--body-pose-model', type=Path)
    parser.add_argument('--cut-cache', type=Path, required=True)
    parser.add_argument('--maximum-seconds', type=float, default=240.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('do not overwrite an actual-pixel critic record')
    cv2.setNumThreads(2)
    record = json.loads(args.movie.with_suffix('.plan.json').read_text())
    result = evaluate_programme(args.movie, record=record, models=args.models, body_model=args.body_pose_model,
        cut_cache=args.cut_cache, check=lambda: None, maximum_seconds=args.maximum_seconds)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x') as stream:
        stream.write(encoded)
    print(json.dumps({k: v for k, v in result.items() if k != 'cuts'}), flush=True)


if __name__ == '__main__':
    main()
