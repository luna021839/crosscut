"""Small native-pixel style observations, independent of camera/edit choices.

Colour distance is a visual-change proxy, NOT clothing segmentation, identity,
ownership or aesthetic quality. No frames or neural features are persisted.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import cv2
import numpy as np
from .grid_perception import file_identity
from .source_cuts import read_frames


def histogram(pixels):
    hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    hs=cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).ravel()
    value=cv2.calcHist([hsv],[2],None,[8],[0,256]).ravel()
    return np.r_[.5*hs/max(1.,hs.sum()),.5*value/max(1.,value.sum())].astype(np.float32)


def describe(pixels,heads):
    """Face-relative chest rectangles plus actual global colour; not masks."""
    height,width=pixels.shape[:2];global_hist=histogram(pixels)
    values=np.zeros((len(heads),2,40),np.float32);valid=np.zeros(len(heads),bool)
    for h,(x,y,size,confidence) in enumerate(heads):
        if size<=.019 or confidence<.75:continue
        # Isotropic face-height units in real pixels, not stretched x units.
        half=1.3*size*height
        x0,x1=max(0,int(x*width-half)),min(width,int(np.ceil(x*width+half)))
        y0,y1=max(0,int((y+.5*size)*height)),min(height,int(np.ceil((y+2.5*size)*height)))
        if min(x1-x0,y1-y0)<5:continue
        values[h]=np.array([histogram(pixels[y0:y1,x0:x1]),global_hist]);valid[h]=True
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    # A missing face is not a black-frame label. These are actual pixels.
    presence=np.array([float((gray<=4).mean()),float(gray.std()/255.)],np.float32)
    return values,valid,presence


def frames(path,natives,*,check):
    decoder=shutil.which('ffmpeg')
    if decoder is None:raise OSError('native colour observation requires FFmpeg')
    select='select='+'+'.join(f'eq(n\\,{v})' for v in natives)
    with tempfile.TemporaryFile() as errors:
        process=subprocess.Popen([decoder,'-nostdin','-v','error','-threads','2','-i',str(path),
            '-map','0:v:0','-an','-filter_threads','1','-vf',select+',scale=320:180:flags=area',
            '-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1'],
            stdout=subprocess.PIPE,stderr=errors)
        try:
            count=0
            for pixels in read_frames(process.stdout,check=check,width=320,height=180):
                if count>=len(natives):raise OSError('colour decoder emitted undeclared native ordinal')
                yield natives[count],pixels;count+=1
            code=process.wait(timeout=10.);errors.seek(0);diagnostic=errors.read(4096)
            if code or diagnostic or count!=len(natives):
                raise OSError('exact native colour decode failed: '+diagnostic.decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
            process.stdout.close()


def observe(*,sources,banks,cache,check,maximum_seconds=180.):
    """One bounded sequential decode per missing source; completed NPZs reused."""
    if len(sources)!=len(banks) or not 1<=maximum_seconds<=600:
        raise ValueError('finite source-matched colour observation allowance required')
    started=monotonic();cache=Path(cache);result=[];receipts=[];next_check=started
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def live():
        nonlocal next_check
        now=monotonic()
        if now>=next_check:check();next_check=now+.2
        if now-started>=maximum_seconds:raise TimeoutError('colour allowance exhausted; completed source caches retained')
    for source,(path,bank) in enumerate(zip(sources,banks)):
        live();identity=file_identity(path);heads=np.asarray(bank['heads']);indices=np.asarray(bank['native_indices'])
        if heads.shape[1:]!=(2,8,4) or indices.shape!=heads.shape[:2] or not np.isfinite(heads).all():
            raise ValueError('exact native face inputs required for colour observations')
        binding=json.dumps(dict(implementation=implementation,source=identity,
            heads=hashlib.sha256(heads.tobytes()).hexdigest(),
            native_indices=hashlib.sha256(indices.tobytes()).hexdigest(),raster=[320,180],
            decoder=file_identity(Path(shutil.which('ffmpeg')).resolve()),opencv=cv2.__version__),sort_keys=True)
        target=cache/(hashlib.sha256(binding.encode()).hexdigest()+'.npz');decoded=0
        if target.is_file():
            with np.load(target,allow_pickle=False) as saved:
                if str(saved['binding'])!=binding:raise ValueError('native colour binding differs')
                colour,valid,presence=(saved[k] for k in ('colour','valid','presence'))
            reused=True
        else:
            colour=np.zeros(heads.shape[:3]+(2,40),np.float32)
            valid=np.zeros(heads.shape[:3],bool);presence=np.zeros(heads.shape[:2]+(2,),np.float32)
            physical={}
            for n in range(len(heads)):
                for side in (0,1):physical.setdefault(int(indices[n,side]),[]).append((n,side))
            if min(physical)<0:raise ValueError('negative source-native colour ordinal')
            for native,pixels in frames(path,sorted(physical),check=live):
                for n,side in physical[native]:
                    colour[n,side],valid[n,side],presence[n,side]=describe(pixels,heads[n,side])
                decoded+=1
            live();check()
            if file_identity(path)!=identity:raise OSError('source changed during colour observation')
            cache.mkdir(parents=True,exist_ok=True)
            descriptor,temporary=tempfile.mkstemp(prefix='.colour-',suffix='.npz',dir=cache)
            try:
                with os.fdopen(descriptor,'wb') as stream:
                    np.savez_compressed(stream,binding=binding,colour=colour,valid=valid,presence=presence)
                    stream.flush();os.fsync(stream.fileno())
                check();os.replace(temporary,target)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            reused=False
        if (colour.shape!=heads.shape[:3]+(2,40) or valid.shape!=heads.shape[:3] or valid.dtype!=bool
                or presence.shape!=heads.shape[:2]+(2,) or not np.isfinite(colour).all()
                or (colour<0).any() or not np.isfinite(presence).all() or file_identity(path)!=identity):
            raise ValueError('native colour source/domain changed')
        result.append(dict(colour=colour,colour_valid=valid,presence=presence))
        receipt=dict(source=source,reused=reused,decoded_native_inputs=decoded,new_neural_calls=0,
            cache_bytes=target.stat().st_size,source_file_identity=identity,
            colour_is_not_clothing_segmentation=True,elapsed_seconds=monotonic()-started)
        receipts.append(receipt);print(json.dumps(dict(phase='source_colour_complete',**receipt)),flush=True)
    return result,receipts
