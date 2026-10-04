"""Bounded retained-raster delivery without a multi-GiB persisted FFV1 master.

Exactness applies to the declared PRE-ENCODE raster recipe. The delivered
H.264 file is lossy; neither its pixels nor its bits are called exact copies.
Two independent input decodes bind that recipe before and after encoding.
"""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import math
import shutil
import subprocess
import cv2
import numpy as np

from .group_retained_raster_review import content_sha,_spans
from .group_output_media_clock import ordinal_video_encoder_arguments,verify_final_output_media_clock

CONTRACT='hash_bound_retained_preencode_raster_stream_v1'


def bounded_review_encoder_bytes(*,frame_count,fps,maximum_video_bitrate,
        video_buffer_bits,audio_payload_bytes,container_reserve_bytes=16*1024**2):
    """VBV-stream byte bound, not a guess at CRF average filesize.

    The caller must use this maxrate/bufsize in the ACTUAL encoder and bound
    copied/encoded audio separately. Runtime byte/free-space checks remain in
    force; this arithmetic is not a perceptual quality guarantee.
    """
    if (type(frame_count) is not int or not 1<=frame_count<=36000
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or any(type(v) is not int or v<0 for v in
                (maximum_video_bitrate,video_buffer_bits,audio_payload_bytes,container_reserve_bytes))
            or not 1<=maximum_video_bitrate<=100000000 or video_buffer_bits<1):
        raise ValueError('encoder byte bound requires exact frames, finite clock and explicit nonnegative payloads')
    return math.ceil((frame_count/fps*maximum_video_bitrate+video_buffer_bits)/8)+audio_payload_bytes+container_reserve_bytes


def _recipe_frames(reference,proposed,*,count,width,height,changed,deadline):
    readers=[cv2.VideoCapture(str(p)) for p in (reference,proposed)]
    try:
        for tick in range(count):
            if monotonic()>=deadline:raise TimeoutError('retained stream input decode deadline exceeded')
            pair=[]
            for reader in readers:
                ok,frame=reader.read()
                if not ok or frame.shape!=(height,width,3) or frame.dtype!=np.uint8:
                    raise ValueError('retained stream input missing actual raster at '+str(tick))
                pair.append(frame)
            yield tick,pair[1] if tick in changed else pair[0]
        if any(r.read()[0] for r in readers):raise ValueError('retained stream input has undeclared trailing frames')
    finally:
        for reader in readers:reader.release()


def construct_retained_stream_movie(*,reference_movie,proposed_movie,output,
        frame_count,fps,width,height,changed_intervals,protected_intervals,
        expected_reference_sha256,expected_proposed_sha256,maximum_seconds=600.,
        maximum_output_bytes=1536*1024**2,minimum_free_bytes=2*1024**3,progress=None,
        maximum_video_bitrate=None,video_buffer_bits=None,audio_payload_bound=None):
    changed_spans=_spans(changed_intervals,frame_count);protected_spans=_spans(protected_intervals,frame_count)
    if any(max(a,c)<min(b,d) for a,b in changed_spans for c,d in protected_spans):
        raise ValueError('retained stream patch overlaps a protected decoded frame')
    reference,proposed,output=(Path(p) for p in (reference_movie,proposed_movie,output))
    if (reference==proposed or output in (reference,proposed) or output.exists() or output.suffix!='.mp4'
            or type(frame_count) is not int or frame_count<1 or width<2 or height<2 or fps<=0):
        raise ValueError('retained stream requires distinct complete inputs and a new compatible MP4')
    if (content_sha(reference)!=expected_reference_sha256 or content_sha(proposed)!=expected_proposed_sha256):
        raise ValueError('retained stream input content changed before decode')
    vbv=[];encoder_bound=None
    if any(v is not None for v in (maximum_video_bitrate,video_buffer_bits,audio_payload_bound)):
        encoder_bound=bounded_review_encoder_bytes(frame_count=frame_count,fps=fps,
            maximum_video_bitrate=maximum_video_bitrate,video_buffer_bits=video_buffer_bits,
            audio_payload_bytes=audio_payload_bound)
        if encoder_bound>maximum_output_bytes:
            raise ValueError('declared encoder/audio byte bound exceeds the output cap')
        vbv=['-maxrate',str(maximum_video_bitrate),'-bufsize',str(video_buffer_bits)]
    if shutil.disk_usage(output.parent).free<minimum_free_bytes+maximum_output_bytes:
        raise OSError('retained stream destination lacks its declared encoder cap plus free-space floor')
    temporary=output.with_name(output.stem+'.partial.mp4')
    if temporary.exists():raise FileExistsError(temporary)
    changed={t for a,b in changed_spans for t in range(a,b)}
    protected={t for a,b in protected_spans for t in range(a,b)}
    full_hash,outside_hash,protected_hash=sha256(),sha256(),sha256()
    started=monotonic();deadline=started+maximum_seconds
    stderr_path=output.with_name(output.stem+'-stream-encoder.log')
    command=['ffmpeg','-nostdin','-v','warning','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0','-i',str(reference),
        '-map','0:v:0','-map','1:a:0',*ordinal_video_encoder_arguments(fps),
        '-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p','-threads','4',
        *vbv,
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
        '-c:a','copy','-movflags','+faststart',str(temporary)]
    with stderr_path.open('xb') as error_stream:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error_stream)
        try:
            for tick,pixels in _recipe_frames(reference,proposed,count=frame_count,width=width,height=height,
                    changed=changed,deadline=deadline):
                full_hash.update(memoryview(pixels))
                if tick not in changed:outside_hash.update(memoryview(pixels))
                if tick in protected:protected_hash.update(memoryview(pixels))
                data=memoryview(pixels).cast('B')
                while data:
                    amount=process.stdin.write(data)
                    if amount is None or amount<=0:raise OSError('retained stream encoder pipe stopped')
                    data=data[amount:]
                if tick%60==0:
                    if temporary.exists() and temporary.stat().st_size>maximum_output_bytes:
                        raise OSError('retained stream MP4 exceeded its declared output-byte cap')
                    if shutil.disk_usage(output.parent).free<minimum_free_bytes:
                        raise OSError('retained stream reached its declared free-space floor')
                    if progress:progress(dict(phase='retained_preencode_raster_stream',frame=tick,
                        total_frames=frame_count,elapsed_seconds=monotonic()-started))
            process.stdin.close();process.stdin=None
            if process.wait(timeout=max(.1,deadline-monotonic())):
                raise RuntimeError('retained stream compatible encoder failed; inspect '+str(stderr_path))
        except BaseException:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
            raise
        finally:
            if process.stdin is not None:process.stdin.close()
    if temporary.stat().st_size>maximum_output_bytes:raise OSError('final retained stream exceeds its byte cap')
    if (content_sha(reference)!=expected_reference_sha256 or content_sha(proposed)!=expected_proposed_sha256):
        raise ValueError('retained stream input changed during encode')
    media_clock=verify_final_output_media_clock(movie=temporary,original_audio_movie=reference,
        frame_count=frame_count,fps=fps,timeout_seconds=max(.1,deadline-monotonic()))
    temporary.replace(output)
    return dict(contract=CONTRACT,reference_movie=str(reference),proposed_movie=str(proposed),movie=str(output),
        reference_content_sha256=expected_reference_sha256,proposed_content_sha256=expected_proposed_sha256,
        movie_content_sha256=content_sha(output),frame_count=frame_count,fps=fps,frame_size_wh=[width,height],
        changed_intervals=[list(s) for s in changed_spans],protected_intervals=[list(s) for s in protected_spans],
        decoded_format='opencv_bgr24',selected_full_raster_sha256=full_hash.hexdigest(),
        outside_patch_raster_sha256=outside_hash.hexdigest(),protected_raster_sha256=protected_hash.hexdigest(),
        outside_patch_frames=frame_count-len(changed),protected_frames=len(protected),
        persisted_lossless_master=False,compatible_movie_is_lossy=True,
        compatible_mp4_decoded_identity_claimed=False,preencode_recipe_independently_verified=False,
        original_audio_stream_copied=True,maximum_output_bytes=maximum_output_bytes,
        maximum_video_bitrate=maximum_video_bitrate,video_buffer_bits=video_buffer_bits,
        audio_payload_bound=audio_payload_bound,encoder_byte_bound=encoder_bound,
        final_output_media_clock=dict(media_clock,movie=str(output)),
        quality_approved=False,construction_seconds=monotonic()-started)


def verify_retained_stream_recipe(report,*,maximum_seconds=180.,progress=None):
    if report.get('contract')!=CONTRACT or report.get('persisted_lossless_master') is not False:
        raise ValueError('retained stream verifier requires its honest preencode recipe receipt')
    reference,proposed,movie=(Path(report[k]) for k in ('reference_movie','proposed_movie','movie'))
    if any(content_sha(p)!=report[k] for p,k in (
            (reference,'reference_content_sha256'),(proposed,'proposed_content_sha256'),(movie,'movie_content_sha256'))):
        raise ValueError('retained stream verification input content changed')
    count=report['frame_count'];width,height=report['frame_size_wh']
    changed={t for a,b in report['changed_intervals'] for t in range(a,b)}
    protected={t for a,b in report['protected_intervals'] for t in range(a,b)}
    full_hash,outside_hash,protected_hash=sha256(),sha256(),sha256()
    started=monotonic()
    for tick,pixels in _recipe_frames(reference,proposed,count=count,width=width,height=height,changed=changed,
            deadline=started+maximum_seconds):
        full_hash.update(memoryview(pixels))
        if tick not in changed:outside_hash.update(memoryview(pixels))
        if tick in protected:protected_hash.update(memoryview(pixels))
        if progress and tick%120==0:progress(dict(phase='independent_retained_stream_recipe_decode',
            frame=tick,total_frames=count))
    checks=dict(full_preencode_raster_equal=full_hash.hexdigest()==report['selected_full_raster_sha256'],
        outside_preencode_raster_equal=outside_hash.hexdigest()==report['outside_patch_raster_sha256'],
        protected_preencode_raster_equal=protected_hash.hexdigest()==report['protected_raster_sha256'])
    if not all(checks.values()):raise ValueError('independent retained stream raster recipe verification failed')
    return dict(contract=CONTRACT,checks=checks,decoded_recipe_frames=count,
        preencode_recipe_independently_verified=True,persisted_lossless_master=False,
        compatible_mp4_decoded_identity_claimed=False,compatible_movie_decode_verified=False,
        quality_approved=False,verification_seconds=monotonic()-started)
