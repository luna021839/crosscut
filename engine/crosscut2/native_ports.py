"""Small exact-native cut observation bank; independent of the song grid.

Only detected broadcaster-cut flanks are added, with the existing face-model
conventions and a separate cache, without invalidating the song/body grids.
Geometry is evidence-limited, not member identity or an action certificate.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from .grid_perception import file_identity
from .native_readability import native_frames


def _measure_flanks(path,frames,record,*,cache,models,check):
    """One sequential ordinal decode; no GOP-seek reference corruption."""
    identity=file_identity(path);models=Path(models);cache=Path(cache)
    assets=[models/'face_detection_yunet_2023mar.onnx',models/'face_recognition_sface_2021dec.onnx']
    indices=frames[:,None]+[-1,0]
    implementation=hashlib.sha256(Path(__file__).read_bytes()
        +Path(__file__).with_name('native_readability.py').read_bytes()
        +Path(__file__).with_name('source_cuts.py').read_bytes()).hexdigest()
    binding=json.dumps(dict(implementation=implementation,source=identity,
        model_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in assets],
        fps=record['fps'],count=record['frame_count'],ordinals=indices.tolist()),sort_keys=True)
    target=cache/(hashlib.sha256(binding.encode()).hexdigest()+'.npz')
    if target.is_file():
        with np.load(target,allow_pickle=False) as saved:
            if str(saved['binding'])!=binding:raise ValueError('native port cache binding differs')
            heads,vectors=saved['heads'],saved['vectors']
        reused=True
    else:
        cap=cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=record['frame_count']:
                raise OSError('native port frame domain differs from census')
            width=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:cap.release()
        detector=cv2.FaceDetectorYN.create(str(assets[0]),'',(960,540),.75,.3,5000)
        recognizer=cv2.FaceRecognizerSF.create(str(assets[1]),'')
        heads=np.zeros((len(frames),2,8,4),np.float32)
        vectors=np.zeros((len(frames),2,3,128),np.float32)
        locations={}
        for n in range(len(frames)):
            for side in (0,1):locations.setdefault(int(indices[n,side]),[]).append((n,side))
        for tick,pixels in native_frames(path,sorted(locations),width=width,height=height,check=check):
            image=cv2.resize(pixels,(960,540),interpolation=cv2.INTER_AREA)
            _,found=detector.detect(image)
            measured=np.zeros((8,4),np.float32);appearance=np.zeros((3,128),np.float32)
            if found is not None:
                found=sorted((face for face in found if face[3]/540>=.02),key=lambda f:float(f[3]),reverse=True)[:8]
                for i,face in enumerate(found):
                    measured[i]=((face[0]+face[2]/2)/960,(face[1]+face[3]/2)/540,face[3]/540,face[-1])
                    if i<3 and face[3]*height/540>=60:
                        scaled=face.copy();scaled[:14:2]*=width/960;scaled[1:14:2]*=height/540
                        value=recognizer.feature(recognizer.alignCrop(pixels,scaled)).ravel()
                        if value.shape!=(128,) or not np.isfinite(value).all():
                            raise ValueError('invalid native SFace descriptor')
                        norm=float(np.linalg.norm(value))
                        if norm>1e-8:appearance[i]=value/norm
            for n,side in locations[tick]:heads[n,side],vectors[n,side]=measured,appearance
        check()
        if file_identity(path)!=identity:raise OSError('source changed during native port observation')
        if not np.isfinite(heads).all() or not np.isfinite(vectors).all():
            raise ValueError('native port observations must be finite before caching')
        cache.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:np.savez_compressed(stream,binding=binding,heads=heads,vectors=vectors)
        reused=False
    if (heads.shape!=(len(frames),2,8,4) or vectors.shape!=(len(frames),2,3,128)
            or not np.isfinite(heads).all() or not np.isfinite(vectors).all() or file_identity(path)!=identity):
        raise ValueError('native port cache numeric or source domain differs')
    return dict(heads=heads,vectors=vectors),dict(reused=reused,source_file_identity=identity,
        native_inputs=len(np.unique(indices)),decoder='sequential_ffmpeg_exact_native_flanks')


def measure_native_ports(*, sources, census, warps, master_start, fps, final_frame,
                         cache, models, check, maximum_seconds=300.):
    started=monotonic()
    if (not 1<=maximum_seconds<=600 or not math.isfinite(maximum_seconds)
            or not len(sources)==len(census)==len(warps)):
        raise ValueError('bounded source-matched native-port domain required')
    next_check=started
    def live():
        nonlocal next_check
        now=monotonic()
        if now>=next_check:
            check();next_check=now+.25
        if now-started>=maximum_seconds:
            raise TimeoutError('bounded native-cut flank observation allowance exhausted')
    result,receipts=[],[]
    for source,(path,record) in enumerate(zip(sources,census)):
        live()
        frames=np.asarray(record['frames'],np.int64)
        output=np.array([round((warps[source].source_to_master(f/record['fps'])-master_start)*fps)
            for f in frames],np.int64)
        keep=(output>0)&(output<final_frame)
        frames,output=frames[keep],output[keep]
        if not len(frames):
            result.append(dict(frames=output,heads=np.zeros((0,2,8,4),np.float32),
                vectors=np.zeros((0,2,3,128),np.float32)))
            receipts.append(dict(source=source,native_cuts=0,reused=True))
            continue
        bank,bound=_measure_flanks(path,frames,record,cache=cache,models=models,check=live)
        result.append(dict(frames=output,**bank))
        receipts.append(dict(source=source,native_cuts=len(frames),reused=bound['reused'],
            source_file_identity=bound['source_file_identity'],
            native_input_frames=bound['native_inputs'],decoder=bound['decoder'],
            geometry_is_not_action_or_identity_certification=True))
        print(json.dumps(dict(phase='source_native_ports',**receipts[-1],elapsed_seconds=monotonic()-started)),flush=True)
    return result,receipts


def native_geometry_cost(heads, vectors, cameras):
    """Soft screen geometry/appearance cost of one retained native cut.

    Missing faces supply no invented mismatch. Native broadcaster cuts are not
    exempt from observed scale/layout changes, and no source is forbidden.
    """
    if (np.shape(heads)!=(2,8,4) or np.shape(vectors)!=(2,3,128) or np.shape(cameras)!=(2,3)
            or not all(np.isfinite(value).all() for value in (heads,vectors,cameras))
            or np.any(np.asarray(cameras)[:,0]<=0)):
        raise ValueError('finite bound native-cut geometry required')
    chosen=[];appearance=[]
    for side in (0,1):
        visible=heads[side][heads[side,:,2]>.019]
        if not len(visible):return 0.
        formation=len(visible)>=3 and np.max(visible[:,2])<1.6*np.sort(visible[:,2])[-2]
        first=int(np.flatnonzero(heads[side,:,2]>.019)[0])
        head=np.r_[visible[:,:2].mean(0),np.median(visible[:,2])] if formation else heads[side,first,:3]
        zoom,x,y=cameras[side]
        layout=.5+zoom*(visible[np.argsort(visible[:,0]),:2]-[x,y])
        chosen.append((.5+zoom*(head[:2]-[x,y]),float(head[2])*zoom,formation,layout))
        appearance.append(vectors[side,first] if first<3 else np.zeros(128))
    a,b=chosen
    cost=float(np.linalg.norm((a[0]-b[0])*[1.,9/16]))/.06+abs(math.log(a[1]/b[1]))/.25
    if a[2]!=b[2]:cost+=3.
    elif a[2]:
        distances=np.linalg.norm((a[3][:,None]-b[3][None])*[1.,9/16],axis=2)
        cost+=(float((distances.min(0).mean()+distances.min(1).mean())/2)/.08
            +.45*abs(len(a[3])-len(b[3])))
    elif not a[2]:
        va,vb=appearance
        if min(np.linalg.norm(va),np.linalg.norm(vb))>=.5:
            cost+=5.*max(0.,.65-float(np.clip(va@vb,-1.,1.)))
    return float(cost)


def native_geometry_events(nodes, bank, ticks, curve):
    """Assign every measured cut once: interval rate or interior-node event."""
    frames=np.asarray(bank['frames'])
    if (frames.ndim!=1 or frames.dtype.kind not in 'iu'
            or np.shape(bank['heads'])!=(len(frames),2,8,4)
            or np.shape(bank['vectors'])!=(len(frames),2,3,128)
            or np.any((frames<=0)|(frames>=nodes[-1]))):
        raise ValueError('native camera ports must be inside the bound programme')
    interval=np.zeros(len(nodes)-1)
    through=np.zeros(len(nodes))
    for i,frame in enumerate(frames):
        cameras=np.array([[np.interp(t,ticks,curve[:,j]) for j in range(3)] for t in (frame-1,frame)])
        cost=native_geometry_cost(bank['heads'][i],bank['vectors'][i],cameras)
        at=int(np.searchsorted(nodes,frame))
        if at<len(nodes) and nodes[at]==frame:through[at]+=cost
        elif 0<at<len(nodes):interval[at-1]+=cost
    return interval,through
