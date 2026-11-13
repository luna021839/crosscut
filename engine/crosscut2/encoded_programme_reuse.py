"""Reuse verified independent H264 shot packets for a fixed programme.

Only effect choices may differ. No source, nominal camera, cut, frame or audio
clock is changed. Exact IDR boundaries and zero B-frames permit packet copying;
every final packet is checked against its selected original encoded packet.
"""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from time import monotonic

from crosscut.render import segment_frame_counts
from crosscut.render_clock_artifact import render_clock_path,video_content_sha256
from crosscut.render_reference_clock import load_render_reference_clock
from . import sparse_native_frames as sparse


def nominal_signature(plan):
    raw=plan.to_json()
    for key in ('output','transitions','member_pose_dissolve_requested','member_pose_dissolve_effective','member_pose_dissolve_count'):
        raw.pop(key,None)
    for shot in raw['segments']:
        shot.pop('outgoing_overlap_transform',None)
    return raw


def shot_choices(plan,candidates):
    if not candidates or any(nominal_signature(p)!=nominal_signature(plan) for _,p in candidates):
        raise ValueError('encoded reuse requires the exact same nominal source/camera/clock programme')
    selected=[]
    for index in range(len(plan.segments)):
        wanted=None if index==0 else plan.transitions[index-1]
        options=[k for k,(_,other) in enumerate(candidates) if wanted is None or (
            other.transitions[index-1]==wanted and (not wanted.kind.uses_overlap or
            other.segments[index-1].outgoing_overlap_transform==plan.segments[index-1].outgoing_overlap_transform))]
        if not options:raise ValueError('no existing encoded shot has the requested effect and exact future camera')
        selected.append(options[0])
    return selected


def _packets(path,ffprobe,timeout):
    command=[ffprobe,'-v','error','-select_streams','v:0','-show_packets','-show_streams',
        '-show_data_hash','sha256','-show_entries',
        'stream=codec_name,profile,level,width,height,pix_fmt,has_b_frames,time_base,avg_frame_rate,extradata_hash:packet=pts,dts,duration,flags,data_hash',
        '-of','json',str(path)]
    result=subprocess.run(command,check=True,capture_output=True,timeout=timeout)
    data=json.loads(result.stdout);stream=data['streams'][0]
    if stream['codec_name']!='h264' or stream['has_b_frames']!=0:
        raise ValueError('encoded reuse requires independent zero-B H264 frames')
    if len(data['streams'])!=1 or not data['packets'] or any('data_hash' not in p for p in data['packets']):
        raise ValueError('complete encoded packet fingerprints required')
    return stream,data['packets']


def _audio_packets(path,ffprobe,timeout):
    result=subprocess.run([ffprobe,'-v','error','-select_streams','a:0','-show_packets','-show_streams',
        '-show_data_hash','sha256','-show_entries',
        'stream=codec_name,sample_rate,channels,time_base:packet=pts,dts,duration,data_hash',
        '-of','json',str(path)],check=True,capture_output=True,timeout=timeout)
    data=json.loads(result.stdout)
    if len(data['streams'])!=1 or not data['packets'] or any('data_hash' not in p for p in data['packets']):
        raise ValueError('complete original audio packet fingerprints required')
    return data


def render(plan,candidates,*,ffmpeg='ffmpeg',maximum_seconds=180.,check=lambda:None,progress_callback=lambda _:None):
    """Publish a packet-identical assembly; never retry or encode a video."""
    selected=shot_choices(plan,candidates);movie=Path(plan.output);sidecar=render_clock_path(movie)
    if movie.exists() or sidecar.exists():raise FileExistsError('encoded reuse never overwrites a movie')
    if not 0<maximum_seconds<=600:raise ValueError('finite encoded-reuse allowance required')
    started=monotonic();identities=[sparse.file_identity(p) for p,_ in candidates]
    ffprobe=str(Path(ffmpeg).with_name('ffprobe'));rate=Fraction(plan.fps).limit_denominator(1_001_000)
    def live():
        check()
        if monotonic()-started>maximum_seconds:raise TimeoutError('single encoded reuse allowance exhausted')
        if [sparse.file_identity(p) for p,_ in candidates]!=identities:raise OSError('encoded input movies changed')
    def run(command):
        live()
        return subprocess.run(command,check=True,capture_output=True,timeout=max(1.,maximum_seconds-(monotonic()-started)))
    fingerprints=[];clocks=[];payloads=[];codec=None
    for path,original in candidates:
        # The strict sidecar API consumes serialized JSON arrays. Pan-crop
        # dataclasses contain tuples until JSON encoding; normalize only that
        # Python representation, without changing any numeric camera value.
        live();load_render_reference_clock(path,json.loads(json.dumps(original.to_json(),allow_nan=False)),())
        clock=sparse.probe(path,check=live);stream,packets=_packets(path,ffprobe,maximum_seconds-(monotonic()-started))
        if clock.count!=plan.output_frame_count or clock.rate!=rate or len(packets)!=clock.count:
            raise ValueError('encoded candidate differs from the fixed output clock')
        if (stream['width'],stream['height'])!=(plan.width,plan.height):
            raise ValueError('encoded candidate raster differs from the declared nominal cameras')
        if codec is not None and stream!=codec:raise ValueError('encoded candidate codec headers differ')
        codec=stream;clocks.append(clock);fingerprints.append([p['data_hash'] for p in packets])
        payloads.append(json.loads(render_clock_path(path).read_bytes()))
    if any(p['source_clocks']!=payloads[0]['source_clocks'] for p in payloads):
        raise ValueError('encoded candidates disagree on original source frame clocks')
    original_audio=_audio_packets(candidates[0][0],ffprobe,maximum_seconds-(monotonic()-started))
    counts=segment_frame_counts(plan);ranges=[];cursor=0
    for source,count in zip(selected,counts,strict=True):
        if ranges and ranges[-1][2]==source:ranges[-1]=(ranges[-1][0],cursor+count,source)
        else:ranges.append((cursor,cursor+count,source))
        cursor+=count
    if any(first not in clocks[source].keyframes for first,_,source in ranges):
        raise ValueError('encoded range must start at a verified independent IDR')
    movie.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(movie.parent).free<2*max(Path(p).stat().st_size for p,_ in candidates)+256*2**20:
        raise OSError('encoded reuse needs room for temporary ranges and the complete film')
    work=Path(tempfile.mkdtemp(prefix='.crosscut-encoded-reuse-',dir=movie.parent));success=False
    try:
        blocks=[];expected=[];details=[]
        for i,(first,final,source) in enumerate(ranges):
            path=candidates[source][0];block=work/f'{i:03d}.mp4';count=final-first
            # Seek a quarter-frame after the verified IDR, retain that IDR,
            # then copy exactly count packets. No rounded timestamp trimming.
            run([ffmpeg,'-nostdin','-v','error','-noaccurate_seek','-ss',format((first+.25)/plan.fps,'.17g'),
                '-i',str(path),'-map','0:v:0','-an','-c:v','copy','-frames:v',str(count),
                '-avoid_negative_ts','make_zero','-video_track_timescale',str(rate.numerator),'-n',str(block)])
            clock=sparse.probe(block,check=live);_,packets=_packets(block,ffprobe,maximum_seconds-(monotonic()-started))
            wanted=fingerprints[source][first:final]
            if clock.count!=count or clock.rate!=rate or [p['data_hash'] for p in packets]!=wanted:
                raise ValueError('copied encoded range changed a frame or its packet payload')
            expected.extend(wanted);blocks.append(block)
            details.append(dict(first_output_frame=first,final_output_frame_exclusive=final,
                encoded_input_movie_sha256=payloads[source]['video_content_sha256'],video_reencoded=False))
            progress_callback(dict(phase='encoded_range_copy_complete',completed=i+1,total=len(ranges),
                covered_output_frames=final,elapsed_seconds=monotonic()-started,video_encoder_calls=0))
        listing=work/'blocks.ffconcat'
        listing.write_text('ffconcat version 1.0\n'+''.join(f"file '{p.name}'\n" for p in blocks))
        output=work/'complete.mp4'
        run([ffmpeg,'-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),
            '-i',str(candidates[0][0]),'-map','0:v:0','-map','1:a:0','-c','copy',
            '-video_track_timescale',str(rate.numerator),'-movflags','+faststart','-n',str(output)])
        final_clock=sparse.probe(output,check=live);_,packets=_packets(output,ffprobe,maximum_seconds-(monotonic()-started))
        if final_clock.count!=plan.output_frame_count or final_clock.rate!=rate or [p['data_hash'] for p in packets]!=expected:
            raise ValueError('assembled output changed an encoded frame or its clock')
        if _audio_packets(output,ffprobe,maximum_seconds-(monotonic()-started))!=original_audio:
            raise ValueError('assembled output changed an audio packet, sample clock or timestamp')
        progress_callback(dict(phase='whole_AV_decode',elapsed_seconds=monotonic()-started,video_encoder_calls=0))
        decoded=run([ffmpeg,'-nostdin','-v','error','-threads','2','-i',str(output),
            '-map','0:v:0','-map','0:a:0','-f','null','-'])
        if decoded.stderr:raise ValueError('reused whole film failed complete video/audio decode')
        digest=video_content_sha256(output);payload=dict(payloads[0],plan=plan.to_json(),
            video_content_sha256=digest,backend='verified_encoded_fixed_programme_reuse_v1',blocks=details,
            resource_policy=dict(decoder_concurrency=1,automatic_codec_retry=False,video_encoder_calls=0))
        clock_output=work/'frame-clock.json';clock_output.write_text(json.dumps(payload,indent=2,allow_nan=False))
        live()
        if movie.exists() or sidecar.exists():raise FileExistsError('encoded reuse output appeared')
        os.replace(clock_output,sidecar);os.replace(output,movie);success=True
        return dict(movie=str(movie),movie_sha256=digest,output_frames=plan.output_frame_count,
            elapsed_seconds=monotonic()-started,source_NN=0,source_rasters=0,video_encoder_calls=0,
            full_AV_decode_exit_code=0,encoded_ranges=details,all_video_packet_payloads_preserved=True,
            audio_packet_payloads_and_timestamps_preserved=True,
            frame_clock_artifact=str(sidecar),professional_quality_approved=False)
    finally:
        if success:
            def companion(function,path,error):
                if not isinstance(error[1],FileNotFoundError):raise error[1]
            shutil.rmtree(work,onerror=companion)
