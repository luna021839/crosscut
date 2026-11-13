"""Duration-neutral delivery overlays for separately measured wardrobe windows.

The input film remains intact. A new movie is encoded once; source selection,
camera solving, source NN and training are not repeated. This is an explicit
post-effect manifest, not a claim that the old RenderPlan describes new pixels.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import monotonic

from crosscut.render_process import run_observed_render
from . import sparse_native_frames as sparse


def overlay_graph(effects, rate):
    rate=Fraction(rate)
    graph=[f'[0:v]settb={rate.denominator}/{rate.numerator},setpts=N[base0]']
    for i,effect in enumerate(effects,1):
        first,n=effect['first_frame'],effect['frame_count']
        edge=effect.get('edge_blend_frames',0)
        feather=('format=yuva420p,'+f'fade=t=in:start_frame=0:nb_frames={edge}:alpha=1,'
            +f'fade=t=out:start_frame={n-1-edge}:nb_frames={edge}:alpha=1,') if edge else ''
        graph.extend((f'[{i}:v]{feather}settb={rate.denominator}/{rate.numerator},setpts=N+{first}[patch{i}]',
            f"[base{i-1}][patch{i}]overlay=x=0:y=0:eof_action=pass:repeatlast=0:enable='between(n,{first},{first+n-1})'[base{i}]"))
    return ';'.join(graph)


def render(base_movie, effects, output, *, ffmpeg='ffmpeg', maximum_seconds=240.,
        video_codec='h264_videotoolbox', reserve_bytes=768*2**20, video_bitrate=35_000_000,
        check=lambda:None, progress=lambda _:None):
    """Overlay whole-canvas exact-frame clips and copy the original audio."""
    base_movie,output=Path(base_movie),Path(output)
    if output.exists() or output.with_suffix('.wardrobe-effects.json').exists():
        raise FileExistsError('never overwrite a viewing reference or prior wardrobe film')
    if (video_codec not in ('h264_videotoolbox','libx264') or not 1<=maximum_seconds<=600
            or type(reserve_bytes) is not int or reserve_bytes<256*2**20
            or type(video_bitrate) is not int or not 16_000_000<=video_bitrate<=80_000_000):
        raise ValueError('one explicit bounded encoder required')
    started=monotonic();check()
    clock=sparse.probe(base_movie,check=check)
    probe=str(Path(ffmpeg).with_name('ffprobe')) if Path(ffmpeg).is_absolute() else 'ffprobe'
    def raster(path):
        value=subprocess.run([probe,'-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height','-of','json',str(path)],capture_output=True,check=True,timeout=10.)
        row=json.loads(value.stdout)['streams'][0]
        return row['width'],row['height']
    def audio_packets(path):
        value=subprocess.run([probe,'-v','error','-select_streams','a:0','-show_packets','-show_streams',
            '-show_data_hash','sha256','-show_entries',
            'stream=codec_name,sample_rate,channels,time_base:packet=pts,dts,duration,data_hash',
            '-of','json',str(path)],capture_output=True,check=True,timeout=15.)
        return json.loads(value.stdout)
    dimensions=raster(base_movie)
    original_audio=audio_packets(base_movie)
    paths=[base_movie,*[Path(e['movie']) for e in effects]]
    identities=[sparse.file_identity(p) for p in paths]
    hashes=[]
    for path in paths:
        with path.open('rb') as f:hashes.append(hashlib.file_digest(f,'sha256').hexdigest())
    if not effects:raise ValueError('no wardrobe windows supplied')
    previous=-1
    for effect in effects:
        first,n=effect['first_frame'],effect['frame_count']
        edge=effect.get('edge_blend_frames',0)
        if type(first) is not int or type(n) is not int or first<0 or n<3 or first<=previous or first+n>clock.count:
            raise ValueError('nonoverlapping integer windows must fit the input film')
        if type(edge) is not int or not 0<=edge<n/2:
            raise ValueError('edge blends must leave a fully visible garment interval')
        previous=first+n-1
        patch=sparse.probe(Path(effect['movie']),check=check)
        if patch.count!=n or patch.rate!=clock.rate or raster(effect['movie'])!=dimensions:
            raise ValueError('wardrobe patch clock differs from the exact output grid')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name('.'+output.stem+'.encoding.mp4')
    if temporary.exists():raise FileExistsError('recover the existing bounded wardrobe encode')
    rate=clock.rate
    command=[str(ffmpeg),'-nostdin','-v','error','-threads','2','-i',str(base_movie)]
    for path in paths[1:]:command+=['-threads','2','-i',str(path)]
    command+=['-filter_complex_threads','2','-filter_complex',overlay_graph(effects,rate),
        '-map',f'[base{len(effects)}]','-map','0:a:0?','-c:v',video_codec]
    command+=(['-b:v',str(video_bitrate)] if video_codec=='h264_videotoolbox' else ['-crf','0','-preset','ultrafast','-threads','2'])
    command+=['-bf','0','-g','30','-pix_fmt','yuv420p','-r',str(rate),
        '-c:a','copy','-video_track_timescale',str(rate.numerator),'-movflags','+faststart','-n',str(temporary)]
    def live(event=None):
        check()
        if monotonic()-started>maximum_seconds:
            raise TimeoutError('single wardrobe delivery allowance exhausted')
        if any(sparse.file_identity(p)!=i for p,i in zip(paths,identities,strict=True)):
            raise OSError('bound wardrobe inputs changed')
        if shutil.disk_usage(output.parent).free<reserve_bytes:
            raise OSError('wardrobe delivery storage reserve exhausted')
        if event is not None:progress(event)
    live()
    result=run_observed_render(command,stderr_path=output.with_suffix('.wardrobe-encode.log'),
        timeout_seconds=max(1.,maximum_seconds-(monotonic()-started)),progress_callback=live)
    if result.returncode:raise RuntimeError('wardrobe delivery encoder failed; no auto retry')
    actual=sparse.probe(temporary,check=live)
    if actual.count!=clock.count or actual.rate!=clock.rate:
        raise ValueError('wardrobe delivery changed the complete film clock')
    if audio_packets(temporary)!=original_audio:
        raise ValueError('wardrobe delivery changed an original audio packet or timestamp')
    decoded=subprocess.run([str(ffmpeg),'-nostdin','-v','error','-threads','2','-i',str(temporary),
        '-map','0:v','-map','0:a?','-f','null','-'],capture_output=True,
        timeout=min(60.,max(1.,maximum_seconds-(monotonic()-started))))
    if decoded.returncode or decoded.stderr:raise ValueError('wardrobe delivery did not fully decode')
    live()
    with temporary.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    receipt=dict(contract='member_spatial_garment_first_post_effect_v1',base_movie=str(base_movie),
        base_movie_sha256=hashes[0],effects=[dict(e,sha256=h) for e,h in zip(effects,hashes[1:],strict=True)],
        output=str(output),output_sha256=sha,output_frames=actual.count,fps=str(actual.rate),
        elapsed_seconds=monotonic()-started,source_NN=0,source_selection_repeated=False,
        camera_programme_reselected=False,audio_stream_copied=True,
        audio_packet_payloads_and_timestamps_preserved=True,full_AV_decode_exit_code=0,
        unaffected_video_reencoded_once=True,normal_speed_audio_playback_performed=False,
        professional_quality_approved=False,storage_reserve_bytes=reserve_bytes,
        video_bitrate=video_bitrate if video_codec=='h264_videotoolbox' else None)
    if output.exists():raise FileExistsError('a viewing result appeared during wardrobe encoding')
    with output.with_suffix('.wardrobe-effects.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    temporary.replace(output)
    return receipt
