"""Additive, exact-native RTMW observations without invalidating base caches.

One portable NPZ snapshot per source/model/raster implementation, not one file
per frame or QA attempt. Completed checkpoints survive an interruption without
restarting a long source analysis. Changing a requested
editing grid can reuse the same physical frame/crop. No actor/visibility or
editorial authority is inferred from the presence of a cached model output.
"""
import hashlib
import inspect
from importlib.metadata import version
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
from .partial_pose import face_proposals,face_rectangle
from . import rtmw_pose


def _key(native,box):
    return hashlib.sha256(np.array([native,*box],np.float64).tobytes()).hexdigest()


def _frames(path,natives,*,check):
    decoder=shutil.which('ffmpeg')
    select='select='+'+'.join(f'eq(n\\,{v})' for v in natives)
    with tempfile.TemporaryFile() as errors:
        process=subprocess.Popen([decoder,'-nostdin','-v','error','-threads','2','-i',str(path),
            '-map','0:v:0','-an','-filter_threads','1','-vf',select+',scale=960:540:flags=area',
            '-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1'],
            stdout=subprocess.PIPE,stderr=errors)
        try:
            count=0
            for pixels in read_frames(process.stdout,check=check,width=960,height=540):
                if count>=len(natives):raise OSError('supplementary decoder emitted an undeclared ordinal')
                yield natives[count],pixels;count+=1
            code=process.wait(timeout=10.);errors.seek(0);diagnostic=errors.read(4096)
            if code or diagnostic or count!=len(natives):
                raise OSError('supplementary exact-native decode failed: '+diagnostic.decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
            process.stdout.close()


_DECODER_IMPLEMENTATION=hashlib.sha256((inspect.getsource(_frames)+inspect.getsource(read_frames)).encode()).hexdigest()


def observe_requests(path,requests,*,sensor,cache,source_clock,check,maximum_seconds=600.,maximum_calls=6000):
    """Return raw133 in request order; one predetermined ROI per actual face.

    Requests are (native ordinal, actual cached head row). This function does
    not choose a body/performer or replace a missing joint. The caller must
    retain all original observation states and resolve support independently.
    """
    if not 1<=maximum_seconds<=1200 or not 0<=maximum_calls<=12000:
        raise ValueError('one finite supplementary observation allowance required')
    started=monotonic();identity=file_identity(path);decoder=shutil.which('ffmpeg')
    if decoder is None:raise OSError('supplementary pose requires existing FFmpeg')
    binding=json.dumps(dict(source=identity,clock={k:source_clock[k] for k in ('fps','frame_count')},model=rtmw_pose.MODEL_SHA256,
        inference=hashlib.sha256(Path(rtmw_pose.__file__).read_bytes()).hexdigest(),
        geometry=hashlib.sha256(inspect.getsource(rtmw_pose.pose_input_geometry).encode()).hexdigest(),
        inference_runtime=dict(name='onnxruntime',version=version('onnxruntime'),provider='CPUExecutionProvider',threads=2),
        roi=hashlib.sha256((inspect.getsource(face_proposals)+inspect.getsource(face_rectangle)).encode()).hexdigest(),
        decoder=file_identity(Path(decoder).resolve()),raster=[960,540],filter='area',
        decode_implementation=_DECODER_IMPLEMENTATION,
        opencv=cv2.__version__,numpy=np.__version__),sort_keys=True)
    key=hashlib.sha256(binding.encode()).hexdigest();cache=Path(cache);check()
    cache.mkdir(parents=True,exist_ok=True);target=cache/(key+'.npz')
    calls=0;next_check=started;written=0;dirty=False;values={}
    def live():
        nonlocal next_check
        now=monotonic()
        if now>=next_check:
            check();next_check=now+.1
            if file_identity(path)!=identity:raise OSError('source changed during supplementary observation')
        if now-started>=maximum_seconds:raise TimeoutError('supplementary observation budget exhausted; completed inputs retained')
    if target.is_file():
        with np.load(target,allow_pickle=False) as saved:
            if str(saved['binding'])!=binding:raise ValueError('supplementary source/cache binding changed')
            keys,raw=saved['keys'],saved['raw133']
        if (keys.ndim!=1 or keys.dtype.kind!='U' or len(set(keys))!=len(keys)
                or raw.shape!=(len(keys),133,3) or raw.dtype!=np.float32 or not np.isfinite(raw).all()):
            raise ValueError('supplementary raw133 snapshot shape or numeric domain differs')
        values={str(k):v.copy() for k,v in zip(keys,raw)}
    def checkpoint():
        nonlocal dirty,written
        if not dirty:return
        check()
        if file_identity(path)!=identity:raise OSError('source changed before supplementary checkpoint')
        descriptor,temporary=tempfile.mkstemp(prefix='.rtmw-snapshot-',suffix='.npz',dir=cache)
        try:
            ordered=sorted(values)
            with os.fdopen(descriptor,'wb') as stream:
                np.savez_compressed(stream,binding=binding,keys=np.asarray(ordered,dtype='U64'),
                    raw133=np.asarray([values[k] for k in ordered],np.float32))
                stream.flush();os.fsync(stream.fileno())
            check()
            if file_identity(path)!=identity:raise OSError('source changed while checkpointing supplementary poses')
            os.replace(temporary,target);dirty=False;written+=1
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    try:
        keys=[];needed={}
        for native,head in requests:
            live()
            if type(native) is not int or not 0<=native<source_clock['frame_count']:
                raise ValueError('supplementary request escapes actual native source')
            boxes=face_proposals(head)
            if not boxes:raise ValueError('supplementary request has no observed focal face')
            box=boxes[0];cell=_key(native,box);keys.append(cell)
            if cell not in values:needed.setdefault(native,{})[cell]=box
        if sum(map(len,needed.values()))>maximum_calls:
            raise ValueError('supplementary request exceeds declared neural call allowance')
        if needed:
            for native,pixels in _frames(path,sorted(needed),check=live):
                for cell,box in needed[native].items():
                    live();observation=sensor.describe(pixels,bbox_xywh=box);calls+=1
                    raw=np.asarray(observation.wholebody_xyc,np.float32)
                    if raw.shape!=(133,3) or not np.isfinite(raw).all():raise ValueError('supplementary raw133 contract differs')
                    live()
                    if file_identity(path)!=identity:raise OSError('source changed before supplementary cache commit')
                    values[cell]=raw.copy();dirty=True
                    if calls%16==0:checkpoint()
        live()
        if file_identity(path)!=identity:raise OSError('source changed before supplementary result')
        checkpoint()
        for value in values.values():value.setflags(write=False)
        return [values[k] for k in keys],dict(source_file_identity=identity,source_requests=len(requests),
            unique_requested_inputs=len(set(keys)),new_neural_calls=calls,reused_neural_inputs=len(set(keys))-calls,
            completed_checkpointed_inputs_survive_interrupt=True,checkpoint_every_inputs=16,
            written_checkpoints=written,cache_format='atomic_portable_npz',base_body_cache_invalidated=False,
            model=rtmw_pose.MODEL_SHA256,raw_peaks_are_not_probabilities=True,
            elapsed_seconds=monotonic()-started,cache_file=str(target))
    finally:checkpoint()


def augment_missing_grid(*,sources,banks,poses,body_states,warps,nodes,master_start,fps,
        camera_census,model_path,cache,check,maximum_seconds=600.,maximum_calls=6000,node_scope=None):
    """Supplement a failed/degenerate close-up window, never mix model flanks.

    Base pose/state arrays and caches remain unchanged. A targeted window uses
    RTMW for ALL four samples, including explicit missing samples, not one
    model before the cut and another afterwards. Original base states remain
    uncertainty inputs; this adds geometry, not body ownership or visibility.
    """
    from types import SimpleNamespace
    from .pose_grid import pose_indices
    from .partial_pose import face_guided_hypothesis
    started=monotonic();sensor=rtmw_pose.WholeBodySensor(model_path)
    if not 1<=maximum_seconds<=1200 or not 0<=maximum_calls<=12000:raise ValueError('finite whole-grid supplementary allowance required')
    if not len(sources)==len(banks)==len(poses)==len(body_states)==len(warps)==len(camera_census):
        raise ValueError('supplementary grid lost its source domains')
    selected=set(range(len(nodes))) if node_scope is None else set(map(int,node_scope))
    if any(n<0 or n>=len(nodes) for n in selected):raise ValueError('supplementary node scope escapes grid')
    def live():
        check()
        if monotonic()-started>=maximum_seconds:raise TimeoutError('whole-grid supplementary allowance exhausted; raw checkpoints retained')
    result=[p.copy() for p in poses];receipts=[];spent=0
    for s,path in enumerate(sources):
        live();base=np.asarray(poses[s]);states=np.asarray(body_states[s]);heads=np.asarray(banks[s]['heads'])
        if base.shape!=(len(nodes),4,3,8,3) or states.shape!=(len(nodes),4,3):raise ValueError('supplementary grid shape differs from base')
        shoulder=np.linalg.norm((base[:,:,:,0,:2]-base[:,:,:,1,:2])*[16/9,1.],axis=-1)
        heights=np.repeat(heads[:,:,:3,2],2,axis=1).reshape(len(nodes),4,3)
        targets=(((states!=1)|(shoulder<.025))&(heights>=.12)).any(1)
        targets[[n for n in range(len(nodes)) if n not in selected]]=False
        clock=camera_census[s];indices=pose_indices(warps[s],nodes=nodes,master_start=master_start,
            fps=fps,native_fps=clock['fps'],native_count=clock['frame_count'])
        requests=[];links=[]
        for n,h in zip(*np.nonzero(targets)):
            result[s][n,:,h]=0.
            for phase in range(4):
                head=heads[n,0 if phase<2 else 1,h]
                if head[2]<.055 or head[3]<.75:continue
                requests.append((int(indices[n,phase]),head));links.append((n,phase,h))
        if not requests:
            receipts.append(dict(source=s,target_windows=0,new_neural_calls=0,reused_neural_inputs=0));continue
        raw,receipt=observe_requests(path,requests,sensor=sensor,cache=cache,source_clock=clock,check=live,
            maximum_seconds=max(1.,maximum_seconds-(monotonic()-started)),maximum_calls=maximum_calls-spent)
        supported=0
        for values,(n,phase,h) in zip(raw,links):
            side=0 if phase<2 else 1
            others=np.delete(heads[n,side],h,axis=0)
            evidence=SimpleNamespace(reason='observed',raw_keypoints_xyc=values[:17])
            partial=face_guided_hypothesis(evidence,heads[n,side,h],others)
            result[s][n,phase,h]=partial.points;supported+=int(partial.supported.any())
        spent+=receipt['new_neural_calls']
        receipt.update(source=s,target_windows=int(targets.sum()),supported_partial_samples=supported,
            same_model_for_all_four_temporal_samples=True,base_observation_states_unchanged=True,
            points_are_model_hypotheses_not_verified_anatomical_ownership=True)
        receipts.append(receipt)
        print(json.dumps(dict(phase='source_wholebody_supplement_complete',source=s,new_neural_calls=receipt['new_neural_calls'],
            reused_neural_inputs=receipt['reused_neural_inputs'],elapsed_seconds=monotonic()-started)),flush=True)
    return result,receipts
