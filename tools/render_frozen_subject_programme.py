"""Render a saved subject-moment programme without repeating its selection."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import monotonic

import numpy as np
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from crosscut.render import render,segment_frame_counts,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut2.grid_perception import file_identity
from crosscut2.subject_moment_sequence import curve_digest


def remaining_render_space(duration_seconds,written_bytes=0,*,reserve_bytes=2**30):
    """One atomic MP4 and an explicit reserve(default1GiB,min256MiB).

    The explicit36Mbps video maxrate,320kbps audio,48Mbit VBV and16MiB
    container allowance estimate outstanding bytes. Hardware and software
    encoders share the same rate/VBV bound. No existing output or
    user data is deleted to make the budget fit.
    """
    if (type(duration_seconds) not in (int,float) or not math.isfinite(duration_seconds)
            or duration_seconds<=0 or type(written_bytes) is not int or written_bytes<0):
        raise ValueError('positive render duration and observed nonnegative byte count required')
    if type(reserve_bytes) is not int or reserve_bytes<256*2**20:
        raise ValueError('an explicit workspace reserve of at least256MiB is required')
    payload=math.ceil(duration_seconds*(36_000_000+320_000)/8+48_000_000/8)+16*2**20
    return reserve_bytes+max(0,payload-written_bytes)


def camera_values(segment,count,fps):
    if segment.editorial_transform is None:
        if segment.crop is not None or segment.pan_crop is not None:raise ValueError('a different crop needs its own replay proof')
        return np.tile([1.,.5,.5],(count,1))
    keys=segment.editorial_transform.keyframes;t=np.arange(count)/fps
    if any(k.roll_degrees!=0 for k in keys):raise ValueError('unscored rotation in subject programme')
    return np.column_stack([np.interp(t,[k.time_seconds for k in keys],[getattr(k,a) for k in keys])
        for a in ('zoom','center_x','center_y')])


def materialize_scored_cameras(plan,record):
    """Recover an exact frozen arc, never accept a merely similar camera hash.

    Compact linear keyframes can differ by an ULP from the scored sampled
    curve. A declared connector is recoverable from its hashed strips and
    unchanged constant interior. Accept ONLY the original full-frame hash;
    the compact plan must also differ by no more than floating roundoff.
    No media observation, source/cut change, fitting or selection occurs.
    """
    frozen=record.get('fixed_actor_camera_hold_receipt',{}).get('frozen_selected_camera_arcs')
    if frozen is None:return plan,[]
    from crosscut2.boundary_arc_costs import cameras,whole_viewports
    result=[];receipts=[];seams=record.get('seams',[])
    for i,(segment,n,row) in enumerate(zip(plan.segments,segment_frame_counts(plan),frozen,strict=True)):
        values=camera_values(segment,n,plan.fps);expected=row['camera_path_sha256']
        if curve_digest(values)==expected:result.append(segment);continue
        code=row.get('camera_arc_code')
        if code not in (0,1,2,3,5) or n<38:
            raise ValueError('cannot restore exact frozen camera family')
        entry=np.asarray(seams[i-1]['actual_camera_strips'][1] if i else values[:7],float)
        leave=np.asarray(seams[i]['actual_camera_strips'][0] if i<len(plan.segments)-1 else values[-7:],float)
        if (hashlib.sha256(entry.tobytes()).hexdigest()!=row.get('entry_geometry_hash')
                or hashlib.sha256(leave.tobytes()).hexdigest()!=row.get('exit_geometry_hash')):
            raise ValueError('frozen endpoint geometry changed')
        if code==5:paths,valid=whole_viewports(entry[None],leave[None],length=n,cap=2.)
        else:
            base=np.tile(values[n//2],(n,1))
            paths,valid=cameras(entry[None],leave[None],base,cap=2.)
        exact=paths[0]
        if not valid[0] or curve_digest(exact)!=expected:
            raise ValueError('cannot restore exact frozen camera hash')
        difference=float(np.abs(exact-values).max())
        if difference>8*np.finfo(float).eps*max(1.,float(np.abs(values).max())):
            raise ValueError('camera differs by more than replay roundoff')
        track=EditorialTransformTrack(tuple(EditorialTransformKeyframe(k/plan.fps,*map(float,v))
            for k,v in enumerate(exact)),reason='exact_hash_verified_scored_arc_materialization')
        result.append(replace(segment,editorial_transform=track))
        receipts.append(dict(segment=i,camera_arc_code=code,frames=n,maximum_difference=difference,
            exact_original_scored_sha256=expected,endpoint_geometry_hashes_exact=True,
            source_or_cut_or_camera_choice_changed=False))
    return replace(plan,segments=tuple(result)),receipts


def validate_camera_clock(plan,proof,sources):
    counts=segment_frame_counts(plan)
    if len(proof)!=len(plan.segments):raise ValueError('one frozen camera/native proof per hold required')
    offset=0;cuts={}
    for s,n,row in zip(plan.segments,counts,proof,strict=True):
        if (row['source'],row['first_frame'],row['final_frame'])!=(s.source_index,offset,offset+n):
            raise ValueError('frozen source/output interval changed')
        values=camera_values(s,n,plan.fps)
        if 'scored_camera_sha256' in row:
            if 'camera_path_sha256' in row or not all(k in row for k in ('native_first','native_last')):
                raise ValueError('ambiguous or incomplete subject camera/native proof')
            expected=row['scored_camera_sha256']
        else:
            if row.get('per_frame_camera_count')!=n:raise ValueError('incomplete global camera proof')
            expected=row.get('camera_path_sha256')
        if expected is None or curve_digest(values)!=expected:raise ValueError('scored camera changed')
        source=sources[s.source_index]
        native=np.asarray(sampler_frame_schedule(source_start=s.source_start,source_playback_rate=s.source_playback_rate,
            native_fps=source['fps'],output_fps=plan.fps,frame_count=n,
            source_frame_count=source['frame_count']).source_frame_indices)
        if 'native_first' in row and (int(native[0]),int(native[-1]))!=(row['native_first'],row['native_last']):
            raise ValueError('observed endpoint native clock changed')
        if offset:cuts.setdefault(offset,set()).add('authored')
        for tick in source['native_cuts']:
            if native[0]<tick<=native[-1]:cuts.setdefault(offset+int(np.searchsorted(native,tick)),set()).add('retained_native')
        offset+=n
    if offset!=plan.output_frame_count:raise ValueError('incomplete output schedule')
    return dict(output_frame_count=offset,fps=plan.fps,evaluated_boundary_count=len(cuts),
        cuts=[dict(frame=k,origins=sorted(v)) for k,v in sorted(cuts.items())])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','source-spec','authorized-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ssd-device',type=int,required=True)
    p.add_argument('--expected-plan-sha256',help='Digest retained by the actual global selection, not a new post-fit plan.')
    p.add_argument('--output',type=Path,
        help='Explicit new output filename only; source, timeline, camera and selected plan digest stay unchanged.')
    p.add_argument('--maximum-render-seconds',type=float,
        help='Explicit finite encoder deadline; does not repeat selection or relax the4.5GiB process limit.')
    p.add_argument('--workspace-reserve-mib',type=int,default=1024,
        help='Explicit workspace reserve,minimum256MiB; source/camera/quality settings are unchanged.')
    args=p.parse_args()
    root=args.authorized_root.resolve(strict=True);raw=args.plan.read_bytes();record=json.loads(raw)
    plan=render_plan_from_json(record['render_plan']);spec=json.loads(args.source_spec.read_text())
    original_output=plan.output
    if args.output is not None:plan=replace(plan,output=args.output)
    if args.maximum_render_seconds is not None and (not math.isfinite(args.maximum_render_seconds)
            or not 1<=args.maximum_render_seconds<=1800):raise ValueError('bounded explicit render deadline required')
    reserve=args.workspace_reserve_mib*2**20
    remaining_render_space(plan.output_frame_count/plan.fps,reserve_bytes=reserve)
    if args.expected_plan_sha256 is not None and hashlib.sha256(raw).hexdigest()!=args.expected_plan_sha256:
        raise ValueError('saved plan differs from the selected immutable plan identity')
    for path in (args.plan,plan.output,*plan.sources):
        if not path.resolve().is_relative_to(root):raise ValueError('explicit media root only')
    if [str(v) for v in plan.sources]!=[v['path'] for v in spec['sources']] or plan.fps!=spec['fps']:
        raise ValueError('source/clock specification changed')
    report=plan.output.with_suffix('.render.json');claim=plan.output.with_suffix('.render.claim.json')
    if any(v.exists() for v in (plan.output,report,claim)):raise FileExistsError('recover the saved render; no overwrite/retry')
    identities=record['source_file_identities'];written=0;minimum_free=None
    def check():
        nonlocal minimum_free
        if root.stat().st_dev!=args.ssd_device or not Path('/Volumes/CrossCutMedia').is_mount():raise OSError('authorized SSD changed')
        if [list(file_identity(v)) for v in plan.sources]!=identities:raise OSError('frozen source changed')
        free=shutil.disk_usage(root).free
        minimum_free=free if minimum_free is None else min(minimum_free,free)
        required=remaining_render_space(plan.output_frame_count/plan.fps,written,reserve_bytes=reserve)
        if free<required:raise OSError(f'insufficient authorized SSD working space: {free} available, {required} still required')
    check()
    if 'subject_moment_refinement_receipt' in record:
        replay=record['subject_moment_refinement_receipt']['camera_native_replay']
    else:
        if args.expected_plan_sha256 is None:
            raise ValueError('global selection requires its independently retained expected plan digest')
        clock=record.get('selected_clock_check',{})
        if (not all(clock.get(k) is True for k in ('all_source_clocks_verified','source_observation_ordinals_exact',
                'retained_native_clock_verified','camera_transform_replay_exact_within_roundoff'))
                or clock.get('selected_scored_camera_frames_checked')!=plan.output_frame_count):
            raise ValueError('global programme requires its complete preselected source/camera/native replay')
        replay=record['fixed_actor_camera_hold_receipt']['frozen_selected_camera_arcs']
    plan,materialization=materialize_scored_cameras(plan,record)
    boundary=validate_camera_clock(plan,replay,spec['sources'])
    render_input=None
    if materialization:
        path=plan.output.with_suffix('.render-input-plan.json')
        content=json.dumps(plan.to_json(),indent=2,allow_nan=False)+'\n'
        if path.exists():
            if path.read_text()!=content:raise ValueError('saved exact render input differs')
        else:
            with path.open('x') as f:f.write(content)
        render_input=dict(path=str(path),sha256=hashlib.sha256(content.encode()).hexdigest())
    limit=plan.output_frame_count/plan.fps+35. if args.maximum_render_seconds is None else args.maximum_render_seconds
    started=monotonic();peak=0;last=-10.;failure=None;codec=None
    with claim.open('x') as f:json.dump(dict(plan_sha256=hashlib.sha256(raw).hexdigest(),maximum_seconds=limit,automatic_retry=False),f)
    def progress(event):
        nonlocal peak,last,written
        size=event.get('total_size')
        if type(size) is int and size>=0:written=size
        check()
        try:peak=max(peak,psutil.Process(event['pid']).memory_info().rss)
        except psutil.NoSuchProcess:pass
        if peak>int(4.5*2**30):raise MemoryError('one renderer4.5GiB RSS limit')
        elapsed=monotonic()-started
        if elapsed-last>=10. or event['phase']=='end':
            print(json.dumps(dict(phase='frozen_subject_render',seconds=elapsed,event=event)),flush=True);last=elapsed
    try:
        codec=render(plan,ffmpeg='/opt/homebrew/bin/ffmpeg',maximum_runtime_seconds=limit,
            video_bitrate='24M',video_maxrate='36M',video_buffer_size='48M',audio_bitrate='320k',
            progress_callback=progress,decoder_threads=2,filter_complex_threads=2)
        check()
        if args.plan.read_bytes()!=raw:raise OSError('frozen plan changed while encoding')
    except Exception as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    result=dict(movie=str(plan.output),failure=failure,elapsed_seconds=monotonic()-started,
        encoder_peak_RSS_bytes=peak,codec=codec,output_frames=plan.output_frame_count,
        plan_sha256=hashlib.sha256(raw).hexdigest(),selection_repeated=False,new_source_NN=0,
        original_selected_output=str(original_output),output_filename_only_override=plan.output!=original_output,
        resource_policy=dict(decoder_threads_per_input=2,filter_complex_threads=2,maximum_seconds=limit,
            maximum_RSS_bytes=int(4.5*2**30),codec_or_bitrate_or_resolution_or_camera_changed=False),
        exact_camera_materialization=materialization,render_input_plan=render_input,
        disk_space_budget=dict(reserve_bytes=reserve,observed_output_bytes=written,
            initial_required_bytes=remaining_render_space(plan.output_frame_count/plan.fps,reserve_bytes=reserve),
            minimum_observed_free_bytes=minimum_free,fixed_two_GiB_completion_cliff_removed=True),
        normal_speed_audio_playback_performed=False,professional_quality_approved=False)
    if plan.output.exists():
        result['movie_bytes']=plan.output.stat().st_size
        with plan.output.open('rb') as f:result['movie_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    with report.open('x') as f:json.dump(result,f,indent=2)
    if failure is None:
        with plan.output.with_suffix('.boundaries.json').open('x') as f:json.dump(boundary,f,indent=2)
    print(json.dumps(result),flush=True)
    if failure is not None:raise SystemExit(1)


if __name__=='__main__':main()
