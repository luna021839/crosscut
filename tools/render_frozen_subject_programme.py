"""Render a saved subject-moment programme without repeating its selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import monotonic

import numpy as np
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from crosscut.render import render,segment_frame_counts
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut2.grid_perception import file_identity
from crosscut2.subject_moment_sequence import curve_digest


def validate_camera_clock(plan,proof,sources):
    counts=segment_frame_counts(plan)
    if len(proof)!=len(plan.segments):raise ValueError('one frozen camera/native proof per hold required')
    offset=0;cuts={}
    for s,n,row in zip(plan.segments,counts,proof,strict=True):
        if (row['source'],row['first_frame'],row['final_frame'])!=(s.source_index,offset,offset+n):
            raise ValueError('frozen source/output interval changed')
        if s.editorial_transform is None:
            if s.crop is not None or s.pan_crop is not None:raise ValueError('a different crop needs its own replay proof')
            values=np.tile([1.,.5,.5],(n,1))
        else:
            keys=s.editorial_transform.keyframes;t=np.arange(n)/plan.fps
            values=np.column_stack([np.interp(t,[k.time_seconds for k in keys],[getattr(k,a) for k in keys])
                for a in ('zoom','center_x','center_y')])
            if any(k.roll_degrees!=0 for k in keys):raise ValueError('unscored rotation in subject programme')
        if curve_digest(values)!=row['scored_camera_sha256']:raise ValueError('scored camera changed')
        source=sources[s.source_index]
        native=np.asarray(sampler_frame_schedule(source_start=s.source_start,source_playback_rate=s.source_playback_rate,
            native_fps=source['fps'],output_fps=plan.fps,frame_count=n,
            source_frame_count=source['frame_count']).source_frame_indices)
        if (int(native[0]),int(native[-1]))!=(row['native_first'],row['native_last']):
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
    p.add_argument('--ssd-device',type=int,required=True);args=p.parse_args()
    root=args.authorized_root.resolve(strict=True);raw=args.plan.read_bytes();record=json.loads(raw)
    plan=render_plan_from_json(record['render_plan']);spec=json.loads(args.source_spec.read_text())
    for path in (args.plan,plan.output,*plan.sources):
        if not path.resolve().is_relative_to(root):raise ValueError('explicit media root only')
    if [str(v) for v in plan.sources]!=[v['path'] for v in spec['sources']] or plan.fps!=spec['fps']:
        raise ValueError('source/clock specification changed')
    report=plan.output.with_suffix('.render.json');claim=plan.output.with_suffix('.render.claim.json')
    if any(v.exists() for v in (plan.output,report,claim)):raise FileExistsError('recover the saved render; no overwrite/retry')
    identities=record['source_file_identities']
    def check():
        if root.stat().st_dev!=args.ssd_device or not Path('/Volumes/CrossCutMedia').is_mount():raise OSError('authorized SSD changed')
        if [list(file_identity(v)) for v in plan.sources]!=identities:raise OSError('frozen source changed')
        if shutil.disk_usage(root).free<2*2**30:raise OSError('insufficient authorized SSD working space')
    check()
    boundary=validate_camera_clock(plan,record['subject_moment_refinement_receipt']['camera_native_replay'],spec['sources'])
    limit=plan.output_frame_count/plan.fps+35.;started=monotonic();peak=0;last=-10.;failure=None;codec=None
    with claim.open('x') as f:json.dump(dict(plan_sha256=hashlib.sha256(raw).hexdigest(),maximum_seconds=limit,automatic_retry=False),f)
    def progress(event):
        nonlocal peak,last
        check()
        try:peak=max(peak,psutil.Process(event['pid']).memory_info().rss)
        except psutil.NoSuchProcess:pass
        if peak>int(4.5*2**30):raise MemoryError('one renderer4.5GiB RSS limit')
        elapsed=monotonic()-started
        if elapsed-last>=10. or event['phase']=='end':
            print(json.dumps(dict(phase='frozen_subject_render',seconds=elapsed,event=event)),flush=True);last=elapsed
    try:
        codec=render(plan,ffmpeg='/opt/homebrew/bin/ffmpeg',maximum_runtime_seconds=limit,
            video_bitrate='24M',progress_callback=progress)
        check()
        if args.plan.read_bytes()!=raw:raise OSError('frozen plan changed while encoding')
    except Exception as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    result=dict(movie=str(plan.output),failure=failure,elapsed_seconds=monotonic()-started,
        encoder_peak_RSS_bytes=peak,codec=codec,output_frames=plan.output_frame_count,
        plan_sha256=hashlib.sha256(raw).hexdigest(),selection_repeated=False,new_source_NN=0,
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
