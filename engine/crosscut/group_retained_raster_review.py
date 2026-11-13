"""Content-bound full-song review raster, with exact retained outside pixels.

The lossless master, not a plan equality assertion, proves protected decoded
pixel identity. A separate compatible MP4 is lossy and never labeled exact.
This is review artifact construction, not semantic/professional admission.
"""
from hashlib import sha256,file_digest
from pathlib import Path
from time import monotonic
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

CONTRACT='exact_retained_raster_with_declared_review_patch_union_v1'


def content_sha(path):
    with Path(path).open('rb') as stream:return file_digest(stream,'sha256').hexdigest()


def _spans(spans,count):
    previous=0
    if not isinstance(spans,tuple):raise ValueError('raster patch spans must be immutable')
    for first,end in spans:
        if (type(first) is not int or type(end) is not int or not previous<=first<end<=count):
            raise ValueError('raster patch spans must be ordered disjoint in the complete output clock')
        previous=end
    return spans


def construct_retained_raster_master(*,reference_movie,proposed_movie,output,
        frame_count,fps,width,height,changed_intervals,protected_intervals,
        expected_reference_sha256,expected_proposed_sha256,maximum_seconds=600.,
        maximum_output_bytes=10*1024**3,minimum_free_bytes=2*1024**3,progress=None):
    """Decode two actual complete movies; replace only explicit owned ticks."""
    changed=_spans(changed_intervals,frame_count);protected=_spans(protected_intervals,frame_count)
    if any(max(a,c)<min(b,d) for a,b in changed for c,d in protected):
        raise ValueError('review patch tried to write a protected decoded frame')
    paths=tuple(Path(p) for p in (reference_movie,proposed_movie,output))
    reference_movie,proposed_movie,output=paths
    if (reference_movie==proposed_movie or output in paths[:2] or output.exists()
            or output.suffix!='.mkv' or frame_count<1 or width<1 or height<1 or fps<=0):
        raise ValueError('raster review requires distinct complete inputs and a new lossless MKV')
    if (content_sha(reference_movie)!=expected_reference_sha256
            or content_sha(proposed_movie)!=expected_proposed_sha256):
        raise ValueError('review raster input content changed before decode')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.partial.mkv')
    if temporary.exists():raise FileExistsError(temporary)
    if shutil.disk_usage(output.parent).free<minimum_free_bytes+512*1024**2:
        raise OSError('insufficient verified destination space for a bounded raster master')
    deadline=monotonic()+maximum_seconds;started=monotonic()
    changed_ticks={tick for a,b in changed for tick in range(a,b)}
    protected_ticks={tick for a,b in protected for tick in range(a,b)}
    outside_sha,protected_sha,selected_sha=sha256(),sha256(),sha256()
    readers=[cv2.VideoCapture(str(path)) for path in paths[:2]]
    stderr_path=output.with_name(output.stem+'-raster-encoder.log')
    command=['ffmpeg','-nostdin','-v','warning','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0','-i',str(reference_movie),
        '-map','0:v:0','-map','1:a:0','-c:v','ffv1','-level','3','-g','1','-coder','1',
        '-context','1','-slicecrc','1','-pix_fmt','bgr0','-threads','4','-c:a','copy',str(temporary)]
    with stderr_path.open('xb') as error_stream:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error_stream)
        try:
            for tick in range(frame_count):
                if monotonic()>=deadline:raise TimeoutError('retained raster construction deadline exceeded')
                pair=[]
                for reader in readers:
                    ok,pixels=reader.read()
                    if not ok or pixels.shape!=(height,width,3) or pixels.dtype!=np.uint8:
                        raise ValueError('actual complete movie raster missing at tick '+str(tick))
                    pair.append(pixels)
                chosen=pair[1] if tick in changed_ticks else pair[0]
                if tick not in changed_ticks:outside_sha.update(memoryview(chosen))
                if tick in protected_ticks:protected_sha.update(memoryview(chosen))
                selected_sha.update(memoryview(chosen))
                data=memoryview(chosen).cast('B')
                while data:
                    amount=process.stdin.write(data)
                    if amount is None or amount<=0:raise OSError('lossless raster encoder pipe stopped')
                    data=data[amount:]
                if tick%60==0:
                    if temporary.exists() and temporary.stat().st_size>maximum_output_bytes:
                        raise OSError('lossless review master exceeds its declared space cap')
                    if shutil.disk_usage(output.parent).free<minimum_free_bytes:
                        raise OSError('external destination reached retained free-space floor')
                    if progress:progress(dict(phase='lossless_raster_master',frame=tick,total_frames=frame_count,
                        elapsed_seconds=monotonic()-started))
            if any(reader.read()[0] for reader in readers):
                raise ValueError('complete input contains undeclared trailing decoded frames')
            process.stdin.close();process.stdin=None
            code=process.wait(timeout=max(.1,deadline-monotonic()))
            if code:raise RuntimeError('lossless raster encoder failed; inspect '+str(stderr_path))
        except BaseException:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
            raise
        finally:
            if process.stdin is not None:process.stdin.close()
            for reader in readers:reader.release()
    if (content_sha(reference_movie)!=expected_reference_sha256
            or content_sha(proposed_movie)!=expected_proposed_sha256):
        raise ValueError('review raster input changed while constructing the full master')
    temporary.replace(output)
    report=dict(contract=CONTRACT,reference_movie=str(reference_movie),proposed_movie=str(proposed_movie),
        reference_content_sha256=expected_reference_sha256,proposed_content_sha256=expected_proposed_sha256,
        master=str(output),master_content_sha256=content_sha(output),frame_count=frame_count,
        fps=fps,frame_size_wh=[width,height],changed_intervals=[list(s) for s in changed],
        protected_intervals=[list(s) for s in protected],decoded_format='opencv_bgr24',
        selected_full_raster_sha256=selected_sha.hexdigest(),outside_patch_raster_sha256=outside_sha.hexdigest(),
        protected_raster_sha256=protected_sha.hexdigest(),outside_patch_frames=frame_count-len(changed_ticks),
        protected_frames=len(protected_ticks),master_decoded_identity_verified=False,
        compatible_mp4_decoded_identity_claimed=False,semantic_qa_approved=False,quality_approved=False,
        construction_seconds=monotonic()-started)
    return report


def verify_retained_raster_master(report,*,maximum_seconds=180.,progress=None):
    """Independently decode all master frames and exact outside/protected pixels."""
    if report.get('contract')!=CONTRACT or report.get('master_decoded_identity_verified') is not False:
        raise ValueError('raster master verifier needs the original unapproved construction receipt')
    master=Path(report['master']);original=Path(report['reference_movie'])
    if (content_sha(master)!=report['master_content_sha256']
            or content_sha(original)!=report['reference_content_sha256']):
        raise ValueError('full raster verification input hash changed')
    count=report['frame_count'];width,height=report['frame_size_wh']
    changed={n for a,b in report['changed_intervals'] for n in range(a,b)}
    protected={n for a,b in report['protected_intervals'] for n in range(a,b)}
    readers=[cv2.VideoCapture(str(p)) for p in (original,master)]
    outside_hash,protected_hash,full_hash=sha256(),sha256(),sha256()
    failures=[];started=monotonic()
    try:
        for tick in range(count):
            if monotonic()-started>=maximum_seconds:raise TimeoutError('full raster verification deadline exceeded')
            pair=[]
            for reader in readers:
                ok,pixels=reader.read()
                if not ok or pixels.shape!=(height,width,3):raise ValueError('full master decode missing at '+str(tick))
                pair.append(pixels)
            if tick not in changed:
                if not np.array_equal(pair[0],pair[1]):failures.append(tick)
                outside_hash.update(memoryview(pair[1]))
            if tick in protected:protected_hash.update(memoryview(pair[1]))
            full_hash.update(memoryview(pair[1]))
            if tick%120==0 and progress:progress(dict(phase='independent_full_master_decode',frame=tick,total_frames=count))
        if any(reader.read()[0] for reader in readers):raise ValueError('master has undeclared trailing frames')
    finally:
        for reader in readers:reader.release()
    checks=dict(outside_patch_hash_equal=outside_hash.hexdigest()==report['outside_patch_raster_sha256'],
        protected_hash_equal=protected_hash.hexdigest()==report['protected_raster_sha256'],
        complete_master_raster_hash_equal=full_hash.hexdigest()==report['selected_full_raster_sha256'],
        every_outside_frame_equal=not failures,decoded_frames=count)
    if not all(checks[k] for k in checks if k!='decoded_frames'):
        raise ValueError('decoded master raster identity failed: '+json.dumps(dict(checks,failures=failures[:12])))
    return dict(contract=CONTRACT,master=str(master),checks=checks,
        master_decoded_identity_verified=True,protected_frames=len(protected),
        compatible_mp4_decoded_identity_claimed=False,semantic_qa_approved=False,quality_approved=False,
        verification_seconds=monotonic()-started)
