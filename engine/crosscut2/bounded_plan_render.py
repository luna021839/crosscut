"""Sequential source decoding for a saved hard-cut/face-dissolve RenderPlan.

The production native sampler and camera filters are reused. Overlap pictures
are small lossless source handles, blended at delivery resolution. Pure shot
tails are encoded once and the finished film is stream-copied, not re-encoded.
No selection, media observation, NN call, codec retry or quality certification.
"""
from bisect import bisect_right
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic

from crosscut.render import (
    EditorialTransformKeyframe, EditorialTransformTrack, NormalizedCrop,
    TransitionKind, _base_crop_values, _video_post_clock_filters,
    _FACE_DISSOLVE_LOGISTIC_ENDPOINT, _FACE_DISSOLVE_LOGISTIC_SPAN,
    segment_frame_counts, _video_render_units,
)
from crosscut.render_frame_schedule import sampler_frame_schedule, probe_render_frame_clock, NativeFrameSchedule, FRAME_SCHEDULE_CONTRACT
from crosscut.render_clock_artifact import RENDER_CLOCK_ARTIFACT_CONTRACT, render_clock_path
from crosscut.render_process import run_observed_render
from . import sparse_native_frames as sparse


def verified_source_clocks(sources,clocks,*,ffprobe,check=lambda:None):
    """Bind richer QA metadata without changing any raw raster cache producer.

    In particular, a VP9 nearest-index grid must not be labelled exact CFR.
    The metadata probe must reproduce the very packet grid used for seeking.
    """
    result={}
    for s,clock in clocks.items():
        check();value=probe_render_frame_clock(sources[s],ffprobe=ffprobe)
        if (value is None or (value.frame_rate,value.frame_count,value.packet_grid_sha256)
                !=(clock.rate,clock.count,clock.packet_digest)):
            raise ValueError('QA frame clock differs from the actual bounded seek clock')
        result[s]=value
    return result


def nominal_schedules(plan,clocks):
    """Keep the production shared native origin across same-source splits.

    A fractional same-source split must not restart the sampler at each
    segment's rounded local boundary. Only pixels/cameras get local clocks.
    """
    result={}
    for unit in _video_render_units(plan,segment_frame_counts(plan)):
        first,last=unit.segments[0],unit.segments[-1];clock=clocks[first.source_index]
        rate=(unit.coalesced_segment.source_playback_rate if unit.coalesced_segment is not None else
            (last.source_end-first.source_start)/sum(s.duration for s in unit.segments))
        shared=sampler_frame_schedule(source_start=first.source_start,source_playback_rate=rate,
            native_fps=float(clock.rate),output_fps=plan.fps,frame_count=unit.frame_count,
            source_frame_count=clock.count,maximum_terminal_hold_frames=math.ceil(plan.fps))
        offset=0
        for index,n in enumerate(unit.segment_frame_counts,unit.first_segment_index):
            result[index]=NativeFrameSchedule(shared.source_frame_indices[offset:offset+n]);offset+=n
    return result


def outgoing_handle(segment, *, frame_count, fps):
    """The same outgoing-only physical handle as the production renderer."""
    if type(frame_count) is not int or frame_count < 3 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('a physical overlap needs at least three output frames')
    base = NormalizedCrop(*_base_crop_values(segment, segment.duration))
    transform = segment.outgoing_overlap_transform
    if transform is None and segment.editorial_transform is not None:
        k = segment.editorial_transform.final_keyframe
        transform = EditorialTransformTrack((EditorialTransformKeyframe(
            0., k.zoom, k.center_x, k.center_y, k.roll_degrees),),
            reason=segment.editorial_transform.reason + '_dissolve_handle_hold',
            sampling_canvas_zoom=segment.editorial_transform.sampling_canvas_zoom)
    duration = frame_count / fps
    return replace(segment, source_start=segment.source_end, duration=duration,
        source_duration=duration * segment.source_playback_rate, master_start=0.,
        crop=base, pan_crop=None, editorial_transform=transform,
        outgoing_overlap_transform=None, boundary_look_track=None)


def blend_graph(fps):
    """Duration-neutral production logistic curve; first sample is pure A."""
    rate = Fraction(fps).limit_denominator(1_001_000)
    # T is the canonical elapsed time, not blend's version-dependent N origin.
    # H is supplied as the frame budget by the caller through a substitution.
    p = f'(1-T*{rate.numerator}/{rate.denominator}/__H__)'
    weight = (f'((1/(1+exp(-12*({p}-0.5)))-'
        f'{_FACE_DISSOLVE_LOGISTIC_ENDPOINT:.19f})/{_FACE_DISSOLVE_LOGISTIC_SPAN:.16f})')
    return (f'[0:v]settb={rate.denominator}/{rate.numerator},setpts=N[a];'
        f'[1:v]settb={rate.denominator}/{rate.numerator},setpts=N[b];'
        f"[a][b]blend=all_expr='A*{weight}+B*(1-{weight})',format=yuv420p[v]")


def render(plan, *, ffmpeg='ffmpeg', maximum_seconds=660., check=lambda: None,
           progress_callback=lambda _: None, reserve_bytes=2 * 2**30,
           video_codec='h264_videotoolbox'):
    """One finite render; atomic movie publication only after complete AV decode.

    Caller supplies its saved plan/source authority and cancellation check.
    The plan's existing hard-cut or explicit face-dissolve choices are retained.
    Unsupported effects fail before decoding, never silently become hard cuts.
    """
    if (type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= 1800 or type(reserve_bytes) is not int or reserve_bytes < 256 * 2**20):
        raise ValueError('finite render deadline and explicit storage reserve required')
    if any(t.kind not in (TransitionKind.HARD_CUT, TransitionKind.FACE_DISSOLVE) for t in plan.transitions):
        raise ValueError('bounded renderer supports only explicit hard cuts and face dissolves')
    if video_codec not in ('h264_videotoolbox', 'libx264'):
        raise ValueError('explicit hardware or software H264 encoder required; no automatic retry')
    movie = Path(plan.output)
    clock_path=render_clock_path(movie)
    if movie.exists() or clock_path.exists():
        raise FileExistsError('bounded renderer never overwrites an existing movie')
    movie.parent.mkdir(parents=True, exist_ok=True)
    identities = tuple(sparse.file_identity(p) for p in plan.sources)
    started = monotonic(); peak = 0; details = []; blocks = []
    decoder = str(ffmpeg); probe = str(Path(ffmpeg).with_name('ffprobe'))
    rate = Fraction(plan.fps).limit_denominator(1_001_000)
    counts = segment_frame_counts(plan)

    def live():
        check()
        if tuple(sparse.file_identity(p) for p in plan.sources) != identities:
            raise OSError('bounded render source changed')
        if monotonic() - started > maximum_seconds:
            raise TimeoutError('single bounded programme render exhausted')
        if shutil.disk_usage(movie.parent).free < reserve_bytes:
            raise OSError('bounded programme render storage reserve exhausted')

    def progress(event):
        nonlocal peak
        live()
        if event.get('pid'):
            value = subprocess.run(['/bin/ps', '-p', str(event['pid']), '-o', 'rss='],
                capture_output=True, text=True, timeout=3.)
            if value.stdout.strip():
                rss = int(value.stdout.strip()) * 1024; peak = max(peak, rss)
                if rss > 2 * 2**30:
                    raise MemoryError('single bounded encoder exceeded2GiB')
        progress_callback(dict(event, bounded_encoder_peak_RSS_bytes=peak))

    live()
    work = Path(tempfile.mkdtemp(prefix='.crosscut-bounded-render-', dir=movie.parent))
    clocks = {}; output = work / 'complete.mp4'; success = False

    def encode(command, name, budget=45.):
        live()
        result = run_observed_render(command, stderr_path=work / (name + '.stderr.log'),
            progress_callback=progress, timeout_seconds=min(budget, maximum_seconds-(monotonic()-started)))
        if result.returncode:
            raise RuntimeError('bounded render block failed: ' + name)

    def video_options(path, n, *, lossless=False):
        codec = ['-c:v', 'ffv1', '-level', '3', '-g', '1', '-threads', '2'] if lossless else [
            '-c:v', video_codec,
            *(['-preset', 'fast', '-crf', '16'] if video_codec == 'libx264' else ['-b:v', '24M']),
            # Independent tiny blocks can have different B-frame reorder
            # delays. Zero reorder keeps concat DTS and presentation ticks
            # identical even for a two-frame pure tail beside a long shot.
            '-bf', '0', '-maxrate', '36M', '-bufsize', '48M', '-profile:v', 'high', '-threads', '2']
        return ['-frames:v', str(n), *codec, '-pix_fmt', 'yuv420p', '-r', str(rate),
            '-fps_mode:v', 'cfr', '-enc_time_base:v', str(1/rate),
            *([] if lossless else ['-video_track_timescale', str(rate.numerator), '-movflags', '+faststart']),
            '-n', str(path)]

    def verify(path, n):
        live()
        result = subprocess.run([probe, '-v', 'error', '-count_packets', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_packets,width,height', '-of', 'json', str(path)],
            check=True, capture_output=True, timeout=10.)
        actual = json.loads(result.stdout)['streams'][0]
        if int(actual['nb_read_packets']) != n or (actual['width'], actual['height']) != (plan.width, plan.height):
            raise ValueError('bounded render block changed output geometry or frame budget')

    def source_block(segment, nominal_count, begin, end, path, *, lossless=False, schedule=None):
        clock = clocks[segment.source_index]
        if schedule is None:
            schedule = sampler_frame_schedule(source_start=segment.source_start,
                source_playback_rate=segment.source_playback_rate, native_fps=float(clock.rate),
                output_fps=plan.fps, frame_count=nominal_count, source_frame_count=clock.count)
        first, last = schedule.source_frame_indices[0], schedule.source_frame_indices[end-1]
        key = clock.keyframes[bisect_right(clock.keyframes, first)-1]
        chain = [f"select='between(round(t*{float(clock.rate):.17g}),{key},{min(clock.count-1,last+2)})'"]
        filters = list(schedule.ffmpeg_filters(plan.fps))
        filters[2] = "setpts='" + schedule.ffmpeg_tick_expression().replace('N', f'(N+{key})') + "'"
        chain.extend(filters)
        chain.extend(_video_post_clock_filters(crop=segment.crop, pan_crop=segment.pan_crop,
            editorial_transform=segment.editorial_transform, look_adjustment=segment.look_adjustment,
            boundary_look_track=segment.boundary_look_track, width=plan.width, height=plan.height,
            fps=plan.fps, frame_count=nominal_count, input_clock_is_canonical=True))
        chain.extend([f'trim=start_frame={begin}:end_frame={end}',
            f'settb={rate.denominator}/{rate.numerator}', 'setpts=N'])
        script = work / (path.stem + '.ffscript'); script.write_text(','.join(chain))
        command = [decoder, '-nostdin', '-v', 'error', '-noaccurate_seek', '-copyts',
            '-ss', format(float(Fraction(key, 1)/clock.rate), '.17g'), '-threads', '2',
            '-i', str(plan.sources[segment.source_index]), '-an', '-filter_threads', '2',
            '-filter_script:v', str(script), *video_options(path, end-begin, lossless=lossless)]
        encode(command, path.stem); verify(path, end-begin)
        return dict(source=segment.source_index, first_native=schedule.source_frame_indices[begin],
            last_native=schedule.source_frame_indices[end-1], packet_clock_sha256=clock.packet_digest)

    try:
        for s in sorted({v.source_index for v in plan.segments}):
            clocks[s] = sparse.probe(plan.sources[s], check=live)
        from crosscut.shot_video_clock import validate_shot_video_clock_bindings
        verified=verified_source_clocks(plan.sources,clocks,ffprobe=probe,check=live)
        validate_shot_video_clock_bindings(plan, frame_clocks=verified)
        schedules=nominal_schedules(plan,clocks)
        for i, (segment, n) in enumerate(zip(plan.segments, counts, strict=True)):
            overlap = plan.transitions[i-1].frame_count if i else 0
            if overlap:
                a = work / f'{i:03d}-outgoing.mkv'; b = work / f'{i:03d}-incoming.mkv'
                handle = outgoing_handle(plan.segments[i-1], frame_count=overlap, fps=plan.fps)
                left = source_block(handle, overlap, 0, overlap, a, lossless=True)
                right = source_block(segment, n, 0, overlap, b, lossless=True,schedule=schedules[i])
                effect = work / f'{i:03d}-effect.mp4'
                graph = blend_graph(plan.fps).replace('__H__', str(overlap))
                encode([decoder, '-nostdin', '-v', 'error', '-threads', '2', '-i', str(a),
                    '-threads', '2', '-i', str(b), '-an', '-filter_complex_threads', '2',
                    '-filter_complex', graph, '-map', '[v]', *video_options(effect, overlap)], effect.stem)
                verify(effect, overlap); blocks.append(effect)
                details.append(dict(kind='face_dissolve', segment=i, output_frames=overlap,
                    outgoing=left, incoming=right, alpha_curve=plan.transitions[i-1].to_dict(plan.fps)['alpha_curve']))
                a.unlink(); b.unlink()
            if overlap < n:
                pure = work / f'{i:03d}-pure.mp4'
                binding = source_block(segment, n, overlap, n, pure,schedule=schedules[i])
                blocks.append(pure); details.append(dict(kind='pure', segment=i, output_frames=n-overlap, **binding))
            progress_callback(dict(phase='bounded_segment_complete', completed=i+1, total=len(counts),
                elapsed_seconds=monotonic()-started, bounded_encoder_peak_RSS_bytes=peak))
        listing = work / 'blocks.ffconcat'
        listing.write_text('ffconcat version 1.0\n' + ''.join(f"file '{p.name}'\n" for p in blocks))
        encode([decoder, '-nostdin', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
            '-threads', '2', '-ss', format(plan.master_audio_start, '.17g'),
            '-i', str(plan.sources[plan.reference_source]), '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k',
            '-af', f'atrim=duration={plan.render_duration:.17g},asetpts=PTS-STARTPTS',
            '-t', format(plan.render_duration, '.17g'), '-video_track_timescale', str(rate.numerator),
            '-movflags', '+faststart', '-n', str(output)], 'mux')
        final = sparse.probe(output, check=live)
        if final.count != plan.output_frame_count or final.rate != rate:
            raise ValueError('bounded film output packet clock changed')
        validation = subprocess.run([decoder, '-nostdin', '-v', 'error', '-threads', '2', '-i', str(output),
            '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], capture_output=True,
            timeout=min(60., maximum_seconds-(monotonic()-started)))
        if validation.returncode or validation.stderr:
            raise ValueError('bounded complete film failed video/audio decode')
        live()
        if movie.exists() or clock_path.exists():raise FileExistsError('bounded output appeared during encoding')
        with output.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        size = output.stat().st_size
        clock_payload=dict(artifact_contract=RENDER_CLOCK_ARTIFACT_CONTRACT,contract=FRAME_SCHEDULE_CONTRACT,
            enabled=True,source_clocks={str(s):c.to_dict() for s,c in verified.items()},
            timestamp_fallback_source_indices=[],plan=plan.to_json(),quality_approval=False,
            video_content_sha256=digest,backend='sequential_native_clock_blocks_v1',
            resource_policy=dict(decoder_concurrency=1,automatic_codec_retry=False),blocks=details)
        clock_output=work/'frame-clock.json'
        with clock_output.open('x') as stream:json.dump(clock_payload,stream,indent=2,allow_nan=False)
        live()
        os.replace(clock_output,clock_path)
        os.replace(output, movie); success = True
        return dict(movie=str(movie), movie_sha256=digest, movie_bytes=size, blocks=details,
            elapsed_seconds=monotonic()-started, encoder_peak_RSS_bytes=peak,
            source_NN=0, output_frames=final.count, full_AV_decode_exit_code=0,
            frame_clock_artifact=str(clock_path),frame_clock_contract=RENDER_CLOCK_ARTIFACT_CONTRACT,
            source_file_identities=identities, exact_sampler_used=True,
            canonical_source_handle_time_preserved=True, normal_speed_audio_playback_performed=False,
            professional_quality_approved=False, own_transient_blocks_removed=True)
    finally:
        # Never remove someone else's output or a failed render's evidence.
        if success:
            def ignore_companion(function, path, error):
                if not isinstance(error[1], FileNotFoundError):raise error[1]
            shutil.rmtree(work, onerror=ignore_companion)
