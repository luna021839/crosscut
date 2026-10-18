"""Reusable cached MEMBER selection, independent of a song or audit script.

This is the source/camera-state selection used by the accepted comparison,
not a replay of that film's cuts. Callers bind their original measurements
and provide alignment and a TRAIN-only duration model. No files, decoders or
models are opened here. Pair-camera fitting and effects remain separate.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.render import NormalizedCrop, PanCropKeyframe, PanCropTrack, RenderPlan, RenderSegment
from crosscut.render_frame_schedule import sampler_frame_schedule
from .member_cached_programme import follow_crops, pair_cost, projected, solve
from .visible_shot_duration import Model


@dataclass(frozen=True)
class Source:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    def __post_init__(self):
        if (not np.isfinite(self.fps) or self.fps <= 0 or any(type(v) is not int or v < 1
                for v in (self.width, self.height, self.frame_count))):
            raise ValueError('finite physical source clock and dimensions required')
        object.__setattr__(self, 'path', Path(self.path))


def _observations(rows, natives, source):
    """Nearest content quadrature, separately labelled exact cut evidence."""
    if not rows:
        return (np.zeros((len(natives), 4)), np.zeros((len(natives), 8, 3)),
                np.zeros(len(natives), bool), np.zeros(len(natives), bool))
    keys = np.array(sorted(rows), int)
    if any(type(k) is not int or k < 0 or k >= source.frame_count for k in rows):
        raise ValueError('observations require original in-range integer native indices')
    raw = [rows[int(k)] for k in keys]
    boxes = np.array([r['face_bbox_xywh'] if r['face_bbox_xywh'] is not None else [0.] * 4 for r in raw], float)
    poses = np.array([r.get('pose_screen_xyc') or [[0.] * 3] * 8 for r in raw], float)
    seen = np.array([r['face_bbox_xywh'] is not None for r in raw])
    if (boxes.shape != (len(keys), 4) or poses.shape != (len(keys), 8, 3)
            or not np.isfinite(boxes).all() or not np.isfinite(poses).all()
            or np.any(boxes[seen, 2:] <= 0) or np.any((poses[..., 2] < 0) | (poses[..., 2] > 1))):
        raise ValueError('original finite measured face/pose geometry required')
    at = np.searchsorted(keys, natives)
    left, right = np.maximum(0, at - 1), np.minimum(len(keys) - 1, at)
    choose = np.where(natives - keys[left] < keys[right] - natives, left, right)
    delta = np.abs(natives - keys[choose])
    return boxes[choose], poses[choose], seen[choose] & (delta <= .15 * source.fps), seen[choose] & (delta == 0)


def select(sources: Sequence[Source], observations: Sequence[Mapping], *, warps,
           master_start: float, duration_frames: int, fps: float, beat_times,
           output: Path, output_size=(1080, 1920), duration_model: Model | None = None,
           content_loss: Callable | None = None, unknown_geometry=4.,
           articulated_configuration=False, check=lambda: None, progress=None):
    """Return a fresh whole-film RenderPlan and numeric selection receipt.

    `observations[s][native]` is measured source-normalized face/COCO8 geometry.
    A sparse content sample never becomes an exact face at a cut. An optional
    body-context callback changes only finite content cost, not source access.
    Source identity/producer validation belongs to the input adapter.
    """
    from .aspect_viewport import _dimensions
    _dimensions(output_size)
    if (not sources or not all(isinstance(s, Source) for s in sources)
            or len(sources) != len(observations) or len(sources) != len(warps)
            or type(duration_frames) is not int or duration_frames < 2
            or not np.isfinite([fps, master_start]).all() or fps <= 0 or master_start < 0
            or output_size[0] >= output_size[1]):
        raise ValueError('complete bound MEMBER sources, portrait output and aligned clock required')
    beats = np.asarray(beat_times, float)
    if beats.ndim != 1 or not len(beats) or not np.isfinite(beats).all():
        raise ValueError('finite musical candidate times required')
    F, size = duration_frames, tuple(output_size)
    boxes, poses, content, exact, native, seconds = [], [], [], [], [], []
    for s, (source, rows, warp) in enumerate(zip(sources, observations, warps, strict=True)):
        check()
        t = np.array([warp.master_to_source(master_start + f / fps) for f in range(F)])
        if not np.isfinite(t).all() or np.any(t < 0):
            raise ValueError('aligned member source time must be finite and nonnegative')
        n = np.floor((t * 1000 / 1000) * source.fps + .5).astype(int)
        if np.any(n >= source.frame_count):
            raise ValueError('aligned member sequence exceeds its physical source')
        b, p, c, e = _observations(rows, n, source)
        boxes.append(b); poses.append(p); content.append(c); exact.append(e)
        native.append(n); seconds.append(t)
        if progress is not None:
            progress(dict(phase='member_cached_source', source=s, exact_face_frames=int(e.sum()),
                          quadrature_face_frames=int(c.sum()), source_excluded=False))
    native, exact, content = np.array(native), np.array(exact), np.array(content)
    beat_frames = np.unique(np.clip(np.rint((beats - master_start) * fps).astype(int), 0, F))
    can = (exact[:, :-1].sum(0) * exact[:, 1:].sum(0) - (exact[:, :-1] & exact[:, 1:]).sum(0)) > 0
    seams = np.flatnonzero(can) + 1
    distance = np.min(np.abs(seams[:, None] - beat_frames[None, :]), axis=1)
    nodes = np.unique(np.r_[0, F, beat_frames, seams[distance <= 4]])
    nodes = nodes[(nodes == 0) | (nodes == F) | ((nodes >= round(.5 * fps)) & (nodes < F - round(.5 * fps)))]
    state_sources, cameras, faces, bodies, loss = [], [], [], [], []
    for s, source in enumerate(sources):
        for zoom in (1., 1.25, 1.5, 1.8):
            check()
            crop = follow_crops(boxes[s], content[s], source_size=(source.width, source.height),
                                output_size=size, zoom=zoom, fps=fps)
            grid = np.unique(np.r_[np.arange(0, F, 10), F - 1]); curve = crop.copy()
            for axis in (0, 1):
                curve[:, axis] = np.interp(np.arange(F), grid, crop[grid, axis])
            face, pose = projected(boxes[s], poses[s], curve, output_size=size)
            confidence = pose[:, :, 2] >= .3
            outside = (pose[:, :, :2] < 0).any(-1) | (pose[:, :, :2] > 1).any(-1)
            arm_loss = (confidence[:, :6] & outside[:, :6]).sum(1) / np.maximum(1, confidence[:, :6].sum(1))
            h = face[:, 3]; native_pixels = boxes[s][:, 3] * source.height
            sampling = np.maximum(0., np.log2(np.maximum(.01, h * size[1] / np.maximum(1., native_pixels))))
            unary = .35 * np.maximum(0., 1. - h / .16) + 6 * np.maximum(0., h - .28) + .35 * arm_loss + .15 * sampling
            unary = np.where(content[s], unary, .9) + .05 * np.log2(zoom)
            if content_loss is not None:
                unary = np.asarray(content_loss(s, curve, unary, native[s]), float)
                if unary.shape != (F,) or not np.isfinite(unary).all():
                    raise ValueError('member context callback must return finite per-frame content loss')
            state_sources.append(s); cameras.append(curve); faces.append(face); bodies.append(pose); loss.append(unary)
    state_sources = np.array(state_sources)
    cameras, faces, bodies = np.array(cameras), np.array(faces), np.array(bodies)
    state_known = exact[state_sources]

    def transition(f):
        check()
        a, b = bodies[:, f - 1].copy(), bodies[:, f].copy()
        a[~state_known[:, f - 1], :, 2] = 0.; b[~state_known[:, f], :, 2] = 0.
        return pair_cost(faces[:, f - 1], faces[:, f], a, b, state_known[:, f - 1], state_known[:, f],
                         output_size=size, unknown_geometry=unknown_geometry,
                         articulated_configuration=articulated_configuration)

    kwargs = {}
    if duration_model is not None:
        horizon = F / fps
        kwargs.update(minimum_seconds=1 / fps, maximum_seconds=horizon,
                      preferred_seconds=min(2.6, (horizon + 1 / fps) / 2), duration_model=duration_model)
    music = np.min(np.abs(nodes[:, None] - beat_frames[None, :]), axis=1)
    holds, objective = solve(nodes, np.array(loss), state_sources, transition, fps=fps,
                             cut_music_penalty=.2 * (music / 4.) ** 2, **kwargs)
    segments, schedules, seams = [], [], []
    middle_differences = 0
    for start, end, state in holds:
        check()
        s = int(state_sources[state]); source = sources[s]
        duration = (end - start) / fps; first = float(seconds[s][start])
        rate = (warps[s].master_to_source(master_start + end / fps) - first) / duration
        wanted = int(native[s, end - 1]); span = (end - start - 1) / fps
        if span > 0:
            low = ((wanted - .5) / source.fps - first) / span
            high = ((wanted + .5) / source.fps - first) / span
            rate = float(np.clip(rate, np.nextafter(low, np.inf), np.nextafter(high, -np.inf)))
        if abs(rate - 1.) > .02:
            raise ValueError('selected member hold exceeds the source rate envelope')
        schedule = sampler_frame_schedule(source_start=first, source_playback_rate=rate, native_fps=source.fps,
                    output_fps=fps, frame_count=end-start, source_frame_count=source.frame_count)
        if schedule.source_frame_indices[0] != native[s, start] or schedule.source_frame_indices[-1] != wanted:
            raise ValueError('member endpoint differs from the scored native geometry')
        middle_differences += int(np.sum(np.array(schedule.source_frame_indices) != native[s, start:end]))
        ticks = np.unique(np.r_[start, np.arange(((start + 9) // 10) * 10, end, 10), end - 1])
        keys = tuple(PanCropKeyframe((int(f) - start) / fps, NormalizedCrop(*map(float, cameras[state, f]))) for f in ticks)
        segments.append(RenderSegment(s, first, duration, start / fps, pan_crop=PanCropTrack(keys), source_duration=rate * duration))
        schedules.append(schedule)
        if len(segments) > 1:
            previous = holds[len(segments) - 2][2]
            seams.append(dict(output_frame=start, seconds=start/fps, outgoing_source=int(state_sources[previous]),
                incoming_source=s, cost=float(transition(start)[previous, state]),
                exact_face_flanks=bool(state_known[previous, start-1] and state_known[state, start]),
                source_native_indices=[int(schedules[-2].source_frame_indices[-1]), int(schedule.source_frame_indices[0])]))
    plan = RenderPlan(tuple(s.path for s in sources), tuple(segments), 0, AspectRatio.PORTRAIT, Path(output),
                      master_audio_start=master_start, width=size[0], height=size[1], fps=fps)
    check()
    return plan, dict(objective=objective, nodes=len(nodes), states=len(state_sources), seams=seams,
        affine_interior_native_differences_from_piecewise_warp=middle_differences,
        reference_duration_model=None if duration_model is None else duration_model.artifact,
        articulated_configuration=articulated_configuration, source_usage=plan.source_usage_payload(),
        unknown_face_or_pose_not_a_source_veto=True, camera_interpolation_not_a_sensor_observation=True,
        exact_native_endpoint_comparisons=True, final_member_identity_all_frames_certified=False,
        professional_quality_approved=False)
