"""Bounded native-face image observations on the retained grid; no neural work.

These are scalar readability proxies, not identity, pose or occlusion labels.
The source/input/face binding is independent of editing costs. No pixels,
appearance vectors, model weights or private gallery data are persisted.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from time import monotonic

import cv2
import numpy as np

from .grid_perception import file_identity
from .source_cuts import read_frames


def native_frames(path,ticks,*,width,height,check):
    """One full-resolution FFmpeg decode, selecting exact native ordinals.

    Repeated GOP seeks are both slower and can break decoder references. Never
    downscale the measurement to make a 4K input look artificially softer.
    """
    decoder=shutil.which('ffmpeg')
    if decoder is None:raise OSError('native readability requires FFmpeg')
    selection='select='+ '+'.join(f'eq(n\\,{tick})' for tick in ticks)
    with tempfile.TemporaryFile() as errors:
        process=subprocess.Popen([decoder,'-nostdin','-v','error','-threads','2',
            '-i',str(path),'-map','0:v:0','-an','-filter_threads','1','-vf',selection,
            '-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1'],
            stdout=subprocess.PIPE,stderr=errors)
        try:
            received=0
            for pixels in read_frames(process.stdout,check=check,width=width,height=height):
                if received>=len(ticks):raise OSError('native decoder emitted an undeclared frame')
                yield ticks[received],pixels
                received+=1
            code=process.wait(timeout=10.)
            errors.seek(0);diagnostic=errors.read(4096)
            if code!=0 or diagnostic or received!=len(ticks):
                raise OSError('native readability decode incomplete or corrupt: '+diagnostic.decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=5.)
            process.stdout.close()


def face_readability(pixels, head):
    """Contrast, scale-normalized edge energy, clipped-luma fraction, native h."""
    h,w=pixels.shape[:2]
    x,y,size,confidence=map(float,head)
    if size <= .019 or confidence < .75:
        return np.zeros(4,np.float32),False
    ph=size*h;pw=.8*ph
    x0,x1=max(0,math.floor(x*w-pw/2)),min(w,math.ceil(x*w+pw/2))
    y0,y1=max(0,math.floor(y*h-ph/2)),min(h,math.ceil(y*h+ph/2))
    if x1-x0<8 or y1-y0<8:
        return np.zeros(4,np.float32),False
    gray=cv2.cvtColor(pixels[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(52,64),interpolation=cv2.INTER_AREA if ph>=64 else cv2.INTER_LINEAR)
    # Exclude the rectangular border: hair/background edges can otherwise hide
    # a smeared facial interior. This is an anonymous center region, not eyes.
    center=gray[10:54,8:44].astype(np.float32)
    contrast=float(center.std())
    lap=cv2.Laplacian(gray.astype(np.float32),cv2.CV_32F)[10:54,8:44]
    energy=float(np.sqrt(np.mean(lap*lap)))/max(5.,contrast)
    clipped=float(np.mean((center<=4)|(center>=251)))
    return np.array([contrast/255.,energy,clipped,ph],np.float32),True


def measure_native_readability(*,sources,banks,cache,check,maximum_seconds=240.):
    """Stream unique observed face ordinals once; keep completed source caches."""
    if len(sources)!=len(banks) or not math.isfinite(maximum_seconds) or not 1<=maximum_seconds<=600:
        raise ValueError('one bounded source-bound image measurement pass required')
    cache=Path(cache);started=monotonic();result=[];receipts=[]
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    next_volume_check=started
    def live():
        nonlocal next_volume_check
        now=monotonic()
        # Packed 4K frames need many pipe reads. Check the time on every read,
        # but don't make millions of redundant external-volume stat calls.
        if now>=next_volume_check:
            check();next_volume_check=now+.25
        if now-started>=maximum_seconds:
            raise TimeoutError('native readability allowance exhausted; completed source caches retained')
    for source,(path,bank) in enumerate(zip(sources,banks)):
        live();identity=file_identity(path)
        heads=np.asarray(bank['heads']);indices=np.asarray(bank['native_indices'])
        if heads.ndim!=4 or heads.shape[1:]!=(2,8,4) or indices.shape!=heads.shape[:2] or not np.isfinite(heads).all():
            raise ValueError('retained exact-native face grid required')
        binding=json.dumps(dict(implementation=implementation,source_identity=identity,
            heads_sha256=hashlib.sha256(heads.tobytes()).hexdigest(),
            native_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()),sort_keys=True)
        target=cache/(hashlib.sha256(binding.encode()).hexdigest()+'.npz')
        measured_inputs=0
        if target.is_file():
            with np.load(target,allow_pickle=False) as saved:
                if str(saved['binding'])!=binding:
                    raise ValueError('native readability cache binding differs')
                metrics,valid=saved['metrics'],saved['valid']
            reused=True
        else:
            metrics=np.zeros(heads.shape,np.float32);valid=np.zeros(heads.shape[:3],bool)
            physical={}
            for n in range(len(heads)):
                for side in (0,1):
                    if np.any((heads[n,side,:,2]>.019)&(heads[n,side,:,3]>=.75)):
                        physical.setdefault(int(indices[n,side]),[]).append((n,side))
            cap=cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():raise OSError('native readability source decoder unavailable')
                count=round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if any(tick<0 or tick>=count for tick in physical):
                    raise ValueError('native readability input exceeds source')
            finally:cap.release()
            if physical:
                for tick,pixels in native_frames(path,sorted(physical),width=width,height=height,check=live):
                    locations=physical[tick]
                    # Repeated physical frames may bind different head ordering;
                    # decode once, but preserve each declared face rectangle.
                    for n,side in locations:
                        for head in range(8):
                            metrics[n,side,head],valid[n,side,head]=face_readability(pixels,heads[n,side,head])
                    measured_inputs+=1
                    if measured_inputs%100==0:
                        print(json.dumps(dict(phase='native_readability',source=source,
                            decoded_native_inputs=measured_inputs,total_native_inputs=len(physical),
                            elapsed_seconds=monotonic()-started)),flush=True)
            live();check()
            if file_identity(path)!=identity:raise OSError('source changed during native readability')
            cache.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                np.savez_compressed(stream,binding=binding,metrics=metrics,valid=valid)
            reused=False
        if (metrics.shape!=heads.shape or valid.shape!=heads.shape[:3] or valid.dtype!=np.bool_
                or not np.isfinite(metrics).all() or (metrics<0).any() or file_identity(path)!=identity):
            raise ValueError('native readability cache domain or source changed')
        result.append(dict(bank,native_readability=metrics,native_readability_valid=valid))
        receipt=dict(source=source,reused=reused,decoded_native_inputs=measured_inputs,
            measured_face_inputs=int(valid.sum()),cache_bytes=target.stat().st_size,
            new_neural_calls=0,proxy_is_not_occlusion_or_identity=True,
            decoder='full_resolution_single_pass_ffmpeg_native_ordinals')
        receipts.append(receipt)
        print(json.dumps(dict(phase='source_readability_complete',**receipt,
            elapsed_seconds=monotonic()-started)),flush=True)
    return result,receipts
