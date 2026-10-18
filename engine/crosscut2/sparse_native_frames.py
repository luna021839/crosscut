"""Bounded exact native rasters from a VERIFIED zero-origin packet/PTS grid.

Input seeking retains original PTS. Selection uses the verified presentation
index, not a guessed decoder-relative frame number. Every window has a known
keyframe/preroll bound. Unproven media are explicit, never silently retimed.
The canonical raster is the existing sequential select(n)+area-scale decoder.
"""
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import json,math,shutil,subprocess,tempfile
from pathlib import Path
from time import monotonic
from crosscut.render_frame_schedule import render_frame_clock_from_probe
from .grid_perception import file_identity
from .source_cuts import read_frames


class UnsupportedSeek(ValueError):pass


def avc_idr_packet(data,length_size):
    """MP4 K flags include open-GOP I pictures: only an actual IDR resets AVC refs."""
    if length_size not in (1,2,4):raise UnsupportedSeek('unproven AVC NAL length format')
    offset=0;idr=False
    while offset<len(data):
        if offset+length_size>len(data):raise UnsupportedSeek('truncated AVC length prefix')
        size=int.from_bytes(data[offset:offset+length_size],'big');offset+=length_size
        if size<1 or offset+size>len(data):raise UnsupportedSeek('truncated AVC packet')
        idr|=(data[offset]&31)==5;offset+=size
    return idr


def avc_idr_headers(source,position,size,length_size):
    """Inspect NAL length/type bytes only; large I-frame payloads need not be read."""
    if length_size not in (1,2,4):raise UnsupportedSeek('unproven AVC NAL length format')
    offset=0;idr=False;read=0
    while offset<size:
        if offset+length_size>=size:raise UnsupportedSeek('truncated AVC packet header')
        source.seek(position+offset);header=source.read(length_size+1);read+=len(header)
        if len(header)!=length_size+1:raise UnsupportedSeek('AVC header read incomplete')
        count=int.from_bytes(header[:length_size],'big');offset+=length_size
        if count<1 or offset+count>size:raise UnsupportedSeek('truncated AVC packet')
        idr|=(header[-1]&31)==5;offset+=count
    return idr,read


@dataclass(frozen=True)
class SeekClock:
    rate: Fraction
    count: int
    keyframes: tuple[int,...]
    identity: tuple
    packet_digest: str
    inspected_packet_header_bytes: int=0
    grid_basis: str='exact_cfr_presentation_grid'


def probe(path,*,check=lambda:None,maximum_seconds=10.):
    check();identity=file_identity(path)
    try:
        result=subprocess.run([shutil.which('ffprobe') or 'ffprobe','-v','error','-select_streams','v:0',
            '-show_streams','-show_packets','-show_entries',
            'stream=codec_name,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames,is_avc,nal_length_size:packet=pts,duration,flags,pos,size',
            '-of','json',str(path)],capture_output=True,check=True,timeout=maximum_seconds)
        payload=json.loads(result.stdout)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        raise UnsupportedSeek('native packet clock is not verified') from error
    check()
    if file_identity(path)!=identity:raise OSError('source changed during sparse clock probe')
    clock=render_frame_clock_from_probe(payload)
    if clock is None:raise UnsupportedSeek('sparse seek requires the exact verified zero-origin presentation grid')
    stream=payload['streams'][0];base=Fraction(stream['time_base']);keyframes=[];bytes_read=0
    with Path(path).open('rb') as source:
        for packet in payload['packets']:
            if 'K' not in packet.get('flags',''):continue
            if stream['codec_name']=='h264':
                if stream.get('is_avc')!='true':raise UnsupportedSeek('sparse AVC requires verified MP4 NAL lengths')
                size,position=int(packet['size']),int(packet['pos'])
                if not 0<size<=4*1024**2 or position<0 or position+size>identity[2]:
                    raise UnsupportedSeek('bounded random-access packet inspection exhausted')
                idr,read=avc_idr_headers(source,position,size,int(stream['nal_length_size']));bytes_read+=read
                if bytes_read>32*1024**2:raise UnsupportedSeek('bounded NAL-header inspection exhausted')
                if not idr:continue
            keyframes.append(int(Fraction(packet['pts'])*base*clock.frame_rate+Fraction(1,2)))
    check()
    if file_identity(path)!=identity:raise OSError('source changed during random-access packet inspection')
    keyframes=tuple(sorted(set(keyframes)))
    if not keyframes or keyframes[0]!=0:raise UnsupportedSeek('verified native keyframe/preroll mapping required')
    return SeekClock(clock.frame_rate,clock.frame_count,keyframes,identity,clock.packet_grid_sha256,bytes_read,clock.grid_basis)


def windows(natives,clock,*,maximum_window_seconds=2.,maximum_decode_seconds=8.,maximum_selected_frames=32):
    ticks=list(natives)
    if (not ticks or any(type(t) is not int or not 0<=t<clock.count for t in ticks)
            or ticks!=sorted(set(ticks)) or type(maximum_selected_frames) is not int or maximum_selected_frames<1
            or not all(math.isfinite(v) and v>0 for v in (maximum_window_seconds,maximum_decode_seconds))):
        raise ValueError('sorted unique actual native requests and finite seek bounds required')
    groups=[]
    for tick in ticks:
        if not groups or len(groups[-1])>=maximum_selected_frames or (tick-groups[-1][0])/float(clock.rate)>maximum_window_seconds:groups.append([tick])
        else:groups[-1].append(tick)
    result=[]
    for group in groups:
        first=max(0,group[0]-2);last=min(clock.count-1,group[-1]+2)
        key=clock.keyframes[bisect_right(clock.keyframes,first)-1]
        if (last-key+1)/float(clock.rate)>maximum_decode_seconds:
            raise UnsupportedSeek('requested exact seek exceeds declared keyframe/preroll decode bound')
        result.append((first,last,key,tuple(group)))
    return tuple(result)


def frames(path,natives,*,clock,check=lambda:None,width=960,height=540,receipt=None,maximum_seconds=120.,
           maximum_decode_seconds=8.):
    if any(type(v) is not int or v<1 for v in (width,height)) or width*height*3>64*1024**2:
        raise ValueError('bounded positive raster dimensions required')
    path=Path(path);started=monotonic();consumer_pause=0.;window_started=None
    plan=windows(natives,clock,maximum_decode_seconds=maximum_decode_seconds,
        maximum_selected_frames=min(32,(64*1024**2)//(width*height*3)))
    def live():
        check()
        now=monotonic()
        if now-started-consumer_pause>maximum_seconds:raise TimeoutError('bounded active sparse decode allowance exhausted')
        if window_started is not None and now-window_started>min(30.,maximum_seconds):
            raise TimeoutError('single sparse decode window exhausted')
        if file_identity(path)!=clock.identity:raise OSError('source changed during sparse raster pass')
    live();decoder=shutil.which('ffmpeg')
    if not decoder:raise OSError('sparse native raster requires existing FFmpeg')
    if receipt is not None:receipt.pop('full_source_decode_used',None)
    if receipt is not None:receipt.update(strategy='verified_original_PTS_bounded_native_seek_v1',
        packet_grid_sha256=clock.packet_digest,windows=len(plan),requested_frames=len(natives),
        maximum_window_presentation_span_frames=max(last-key+1 for _,last,key,_ in plan),
        maximum_buffered_raster_bytes=max(len(group)*width*height*3 for _,_,_,group in plan),
        total_window_presentation_span_frames=sum(last-key+1 for _,last,key,_ in plan),
        actual_internal_decoder_work_not_instrumented=True,
        whole_source_sequential_strategy_requested=False,
        planned_whole_source_presentation_coverage=any(key==0 and last==clock.count-1 for _,last,key,_ in plan),
        inspected_AVC_packet_header_bytes=clock.inspected_packet_header_bytes,
        maximum_active_decoder_seconds=maximum_seconds,per_window_deadline_seconds=min(30.,maximum_seconds),
        maximum_declared_preroll_seconds=maximum_decode_seconds,
        AVC_random_access_requires_IDR_not_just_K_flag=True,
        source_NN=0,canonical_raster_reference='sequential_select_n_area_bgr24')
    rate=clock.rate
    for first,last,key,group in plan:
        live();selection='+'.join(f'eq(floor(t*{rate.numerator}/{rate.denominator}+.5)\\,{n})' for n in group)
        window_started=monotonic()
        with tempfile.TemporaryFile() as errors:
            process=subprocess.Popen([decoder,'-nostdin','-v','error','-threads','2','-copyts',
                '-noaccurate_seek','-ss',f'{float((Fraction(key)+Fraction(1,4))/rate):.12f}',
                '-t',f'{float(Fraction(last-key+1,1)/rate):.12f}',
                '-i',str(path),'-map','0:v:0','-an','-filter_threads','1',
                '-vf','select='+selection+f',scale={width}:{height}:flags=area',
                '-frames:v',str(len(group)),'-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1'],
                stdout=subprocess.PIPE,stderr=errors)
            try:
                count=0;validated_window=[]
                for pixels in read_frames(process.stdout,check=live,width=width,height=height):
                    if count>=len(group):raise OSError('sparse seek emitted an undeclared native frame')
                    validated_window.append((group[count],pixels.copy()));count+=1
                code=process.wait(timeout=10.);errors.seek(0);diagnostic=errors.read(4096)
                if code or diagnostic or count!=len(group):
                    raise OSError('sparse exact-native decode incomplete: '+diagnostic.decode(errors='replace'))
                live()
                window_started=None
                # Never let a decoder-error window reach model inference or
                # durable raw-cache checkpoints before its exit is verified.
                for row in validated_window:
                    paused=monotonic()
                    try:yield row
                    finally:consumer_pause+=monotonic()-paused
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
                process.stdout.close()
    live()
    if receipt is not None:receipt['decoder_active_seconds']=monotonic()-started-consumer_pause
