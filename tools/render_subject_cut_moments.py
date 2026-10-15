"""Render frozen cut-moment proposals through the production video renderer."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import numpy as np
from crosscut.render import (EditorialTransformKeyframe, EditorialTransformTrack,
                             RenderSegment, render)
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut.render_plan_serialization import render_plan_from_json


def review_union(folder, spec, clips):
    """Include every retained broadcaster cut and comparison-context reset."""
    union = {}
    base = 0

    def add(tick, origin):
        union.setdefault(int(tick), set()).add(origin)

    for clip in clips:
        plan = render_plan_from_json(json.loads((folder / clip['movie']).with_suffix('.plan.json').read_text()))
        if base:
            add(base, 'context_reset_NOT_EDITORIAL')
        add(base + clip['boundary_frame'], 'authored_' + clip['case'] + '_' + clip['variant'])
        for segment in plan.segments:
            source = spec['sources'][segment.source_index]
            native = np.asarray(sampler_frame_schedule(source_start=segment.source_start,
                source_playback_rate=segment.source_playback_rate, native_fps=source['fps'],
                output_fps=spec['fps'], frame_count=round(segment.duration * spec['fps']),
                source_frame_count=source['frame_count']).source_frame_indices)
            for cut in source['native_cuts']:
                crossings = np.flatnonzero((native[:-1] < cut) & (native[1:] >= cut))
                if len(crossings):
                    add(base + round(segment.master_start * spec['fps']) + int(crossings[0]) + 1,
                        'retained_native')
        base += clip['frames']
    return dict(output_frame_count=base, fps=spec['fps'], evaluated_boundary_count=len(union),
                cuts=[dict(frame=tick, origins=sorted(origins)) for tick, origins in sorted(union.items())],
                normal_speed_audio_playback_performed=False, professional_quality_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--ssd-device', type=int, required=True)
    parser.add_argument('--include-anchor-control', action='store_true')
    args = parser.parse_args()
    root = Path('/Volumes/CrossCutMedia/CrossCut-Validation').resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('render output must stay inside the SSD project')
    if args.output.exists():
        raise FileExistsError('recover existing render batch; do not repeat it')
    spec = json.loads(args.input.read_text())
    proposal = json.loads(args.report.read_text())
    template_json = json.loads(args.template.read_text())
    template = render_plan_from_json(template_json['render_plan'])
    if template.fps != spec['fps']:
        raise ValueError('music/output clock changed')
    started = monotonic()
    args.output.mkdir()

    def check():
        if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev != args.ssd_device:
            raise OSError('authorized SSD changed or disconnected')
        if monotonic() - started > 240:
            raise TimeoutError('single render batch exceeded its allowance')

    def transform(camera, seconds):
        return EditorialTransformTrack(tuple(EditorialTransformKeyframe(t, *map(float, camera))
                                       for t in (0., seconds)), reason='preselected_subject_cut_moment')

    def natives(segment, count):
        source = spec['sources'][segment.source_index]
        return sampler_frame_schedule(source_start=segment.source_start,
            source_playback_rate=segment.source_playback_rate, native_fps=source['fps'],
            output_fps=spec['fps'], frame_count=count,
            source_frame_count=source['frame_count']).source_frame_indices

    movies = []
    receipts = []
    for case in proposal['cases']:
        if case['id'] not in args.cases or not case['supported']:
            continue
        context = next(c for c in spec['cases'] if c['id'] == case['id'])
        selected = [('SELECTED', case['supported'][0])]
        if args.include_anchor_control:
            anchor = next(c for c in case['candidates'] if c['cut_offset_frames'] == 0)
            selected.insert(0, ('ANCHOR-CONTROL', anchor))
        for variant, value in selected:
            check()
            offset = value['cut_offset_frames']
            before, after, fps = 45, 75, spec['fps']
            incoming = context['incoming'][value['source_B']]
            a = RenderSegment(value['source_A'],
                context['source_start_A'] + (offset - before) / fps * context['rate_A'],
                before / fps, 0., source_duration=before / fps * context['rate_A'],
                editorial_transform=transform(value['outgoing_camera'], before / fps))
            b = RenderSegment(value['source_B'], incoming['start'] + offset / fps * incoming['rate'],
                after / fps, before / fps, source_duration=after / fps * incoming['rate'],
                editorial_transform=transform(value['incoming_camera'], after / fps))
            radius = (len(value['native_A']) - 1) // 2
            if int(natives(a, before)[-1]) != value['native_A'][radius - 1]:
                raise ValueError('played outgoing endpoint differs from the selected observation')
            if int(natives(b, after)[0]) != value['native_B'][radius]:
                raise ValueError('played incoming endpoint differs from the selected observation')
            output = args.output / f'{case["id"]}-{variant}.mp4'
            global_time = template.master_audio_start + (context['original_failed_K']['global_frame'] + offset) / fps
            plan = replace(template, segments=(a, b), output=output, width=1280, height=720,
                master_audio_start=global_time - before / fps, transitions=(),
                authorized_shot_video_clocks=(), member_pose_dissolve_requested=None,
                member_pose_dissolve_effective=None, member_pose_dissolve_count=None)
            with output.with_suffix('.plan.json').open('x') as stream:
                stream.write(json.dumps(plan.to_json(), indent=2) + '\n')
            render(plan, maximum_runtime_seconds=50., video_bitrate='10M')
            check()
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(output),
                            '-f', 'null', '-'], check=True, capture_output=True, timeout=15.)
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_frames', '-of', 'json', str(output)], timeout=5.))
            if int(probe['streams'][0]['nb_frames']) != before + after:
                raise ValueError('rendered output frame count changed')
            movies.append(output)
            receipts.append(dict(case=case['id'], variant=variant, movie=output.name,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(), frames=before + after,
                boundary_frame=before, cut_offset_frames=offset,
                music_and_source_clocks_moved_together=True, source_retimed=False,
                exact_native_endpoints_checked=True, metrics=value))
            print(json.dumps(dict(case=case['id'], variant=variant,
                                  elapsed_seconds=monotonic() - started)), flush=True)
    if not movies:
        raise ValueError('no supported requested cut moments')
    listing = args.output / 'concat.txt'
    with listing.open('x') as stream:
        stream.write(''.join("file '" + str(p) + "'\n" for p in movies))
    reel = args.output / 'cut-moment-comparison-internal.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', str(reel)], check=True, timeout=15.)
    with (args.output / 'actual-union-critic.json').open('x') as stream:
        stream.write(json.dumps(review_union(args.output, spec, receipts), indent=2) + '\n')
    with (args.output / 'receipt.json').open('x') as stream:
        stream.write(json.dumps(dict(clips=receipts, reel=reel.name,
            reel_sha256=hashlib.sha256(reel.read_bytes()).hexdigest(),
            elapsed_seconds=monotonic() - started, professional_quality_approved=False,
            normal_speed_audio_review_performed=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
