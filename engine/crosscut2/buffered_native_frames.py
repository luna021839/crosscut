"""Larger explicit raw windows for finishing, separate from frozen sensors.

The existing sparse observation reader remains byte-identical: changing a
finishing buffer must not invalidate source detector/pose/appearance caches.
Physical clocks, PTS selection and error-before-publication use the same helpers.
"""
from pathlib import Path
from fractions import Fraction
import shutil,subprocess,tempfile
from time import monotonic
from .sparse_native_frames import windows,file_identity,read_frames


def frames(path,natives,*,clock,check=lambda:None,width=960,height=540,receipt=None,maximum_seconds=120.,
           maximum_decode_seconds=8.,maximum_buffered_raster_bytes=64*1024**2):
    if any(type(v) is not int or v<1 for v in (width,height)) or width*height*3>64*1024**2:
        raise ValueError('bounded positive raster dimensions required')
    if (type(maximum_buffered_raster_bytes) is not int
            or not width*height*3<=maximum_buffered_raster_bytes<=256*1024**2):
        raise ValueError('explicit native window buffer must fit a frame and stay within256MiB')
    path=Path(path);started=monotonic();consumer_pause=0.;window_started=None
    plan=windows(natives,clock,maximum_decode_seconds=maximum_decode_seconds,
        maximum_selected_frames=min(32,maximum_buffered_raster_bytes//(width*height*3)))
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
