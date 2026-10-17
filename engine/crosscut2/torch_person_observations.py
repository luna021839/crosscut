"""Explicit original-frame person observations from the retained YOLO weights.

This optional producer is separate from the earlier ONNX bank. Its Torch/MPS
records are never relabelled as ONNX inference. Consumers can read the small
source-bound rectangles without importing Torch, a model, or source pixels.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import monotonic

import cv2
import numpy as np
from crosscut import native_person_onnx as adapter
from . import sparse_native_frames as sparse
from .grid_perception import file_identity
from .person_rectangle_bank import _validate, _raster
from .observation_cache_contract import function_text

CONTRACT='exact_native_torch_person_rectangles_v1'
WEIGHTS_SHA256='9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f'


def _implementation():
    pixels='\n'.join(function_text(__file__,name) for name in ('_detector','_predict'))
    return hashlib.sha256(pixels.encode()+Path(adapter.__file__).read_bytes()
        +Path(sparse.__file__).read_bytes()).hexdigest()


def _detector(weights):
    import torch
    if not torch.backends.mps.is_available():raise OSError('explicit MPS producer unavailable; no silent CPU fallback')
    torch.set_num_threads(4)
    return torch.load(weights,map_location='cpu')['model'].float().eval().to('mps'),torch


def _predict(pixels,model,torch,config):
    height,width=pixels.shape[:2]
    tensor,transform=adapter.native_letterbox(pixels,maximum_side=1280)
    output=model(torch.from_numpy(tensor).to('mps'))
    prediction=output[0] if isinstance(output,tuple) else output
    boxes=adapter.person_rows(prediction.cpu().numpy(),transform,config=config)
    boxes[:,:4]/=np.array([width,height,width,height],np.float32)
    _validate(boxes,np.array([width,height],np.int64))
    return boxes


def _check_record(record, *, source, bound):
    original_identity=record.get('source_file_identity',[]);current_identity=list(file_identity(source))
    if (record['contract']!=CONTRACT or record['implementation_sha256']!=_implementation()
            or record['weights_sha256']!=WEIGHTS_SHA256
            or record['source_sha256']!=bound['source_sha256']
            or len(original_identity)!=4 or original_identity[1:]!=current_identity[1:]
            or record['fps']!=bound['fps'] or record['frame_count']!=bound['frame_count']
            or record['raster_semantics']!='original_normalized_xyxy_score'
            or record['backend']!='torch_MPS_float32' or record['maximum_side']!=1280
            or record.get('confidence')!=.35 or record.get('nms_iou')!=.55):
        raise ValueError('source-bound original Torch person observation changed')
    if original_identity[0]!=current_identity[0]:
        # A USB remount changes the device number, not the physical pixels.
        # Preserve the old producer/binding and verify actual bytes instead
        # of forcing another model pass or merely trusting the old hash.
        with Path(source).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=bound['source_sha256']:
                raise ValueError('remounted person source content changed')
    raster=np.asarray(record['raster'],np.int64);result={}
    for key,raw in record['rectangles'].items():
        f=int(key);boxes=np.asarray(raw,np.float32).reshape((-1,5))
        if str(f)!=key or not 0<=f<bound['frame_count']:
            raise ValueError('invalid physical person ordinal')
        _validate(boxes,raster);result[f]=boxes
    return result


def load(*,sources,bound,cache,check=lambda:None):
    """Read qualified original records only; missing inputs remain missing."""
    if len(sources)!=len(bound):raise ValueError('source-matched person bank required')
    paths=sorted(p for p in Path(cache).glob('*.json') if not p.name.startswith('._'))
    if len(paths)>32:raise ValueError('bounded person snapshot inventory exceeded')
    result=[{} for _ in sources];lookup={r['source_sha256']:s for s,r in enumerate(bound)}
    for path in paths:
        check()
        if path.is_symlink() or path.stat().st_size>8*1024**2:
            raise ValueError('bounded original person snapshot required')
        record=json.loads(path.read_bytes());s=lookup.get(record.get('source_sha256'))
        if s is None:continue
        values=_check_record(record,source=sources[s],bound=bound[s])
        for f,boxes in values.items():
            if f in result[s] and not np.array_equal(result[s][f],boxes):
                raise ValueError('conflicting actual person snapshots')
            result[s][f]=boxes
    return result


def observe(*,sources,bound,requests,cache,weights,check=lambda:None,
            maximum_new_frames=1536,maximum_seconds=600.):
    """One finite sparse pass. No acquisition, training, or scan fallback."""
    if (not len(sources)==len(bound)==len(requests)
            or type(maximum_new_frames) is not int or not 0<=maximum_new_frames<=2048
            or not 1<=maximum_seconds<=600):raise ValueError('bounded physical person requests required')
    started=monotonic();identities=[file_identity(p) for p in sources]
    def live():
        check()
        if monotonic()-started>=maximum_seconds:raise TimeoutError('one person observation allowance exhausted')
        if any(file_identity(p)!=v for p,v in zip(sources,identities)):
            raise OSError('person observation source changed')
    for path,r,identity,wanted in zip(sources,bound,identities,requests):
        live()
        if (list(identity)!=r['source_file_identity'] or len(wanted)!=len(set(wanted))
                or any(type(f) is not int or not 0<=f<r['frame_count'] for f in wanted)):
            raise ValueError('exact source/native person requests required')
        with Path(path).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=r['source_sha256']:
                raise ValueError('person source content changed')
    values=load(sources=sources,bound=bound,cache=cache,check=live)
    missing=[sorted(set(r)-set(v)) for r,v in zip(requests,values)]
    total=sum(map(len,missing))
    if total>maximum_new_frames:
        raise ValueError(f'Complete person request domain needs {total} new frames; allowance {maximum_new_frames}')
    # A completely warm pass needs no model asset or producer dependencies.
    if not total:return values,dict(new_person_forwards=0,source_rasters=0,
        reused_frames=sum(map(len,requests)),elapsed_seconds=monotonic()-started,complete=True)
    weights=Path(weights);model_identity=file_identity(weights)
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=WEIGHTS_SHA256:
        raise ValueError('explicit retained person weights differ')
    model,torch=_detector(weights)
    config=adapter.PersonDetectorConfig(maximum_side=1280)
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);calls=0;decodes=[]
    for s,wanted in enumerate(missing):
        if not wanted:continue
        live();path=sources[s];record=bound[s];width,height,count=_raster(path)
        clock=sparse.probe(path,check=live)
        if (count!=record['frame_count'] or clock.count!=count
                or abs(float(clock.rate)-record['fps'])>1e-8):raise ValueError('person packet/raster clock differs')
        raw=dict(contract=CONTRACT,implementation_sha256=_implementation(),weights_sha256=WEIGHTS_SHA256,
            source_sha256=record['source_sha256'],source_file_identity=list(identities[s]),fps=record['fps'],
            frame_count=count,raster=[width,height],raster_semantics='original_normalized_xyxy_score',
            backend='torch_MPS_float32',torch_version=torch.__version__,numpy_version=np.__version__,
            opencv_version=cv2.__version__,maximum_side=1280,confidence=.35,nms_iou=.55,
            packet_grid_sha256=clock.packet_digest,rectangles={str(f):v.tolist() for f,v in values[s].items()})
        target=cache/(record['source_sha256']+'.json')
        execution={};decodes.append(dict(source=s,execution=execution))
        def checkpoint():
            # Flushing completed observations after a deadline authorizes
            # no further pixel read or model call.
            check()
            if any(file_identity(p)!=v for p,v in zip(sources,identities)):
                raise OSError('person observation source changed before flush')
            if file_identity(weights)!=model_identity:raise OSError('person weights changed')
            handle,tmp=tempfile.mkstemp(prefix='.person-',suffix='.tmp',dir=cache)
            try:
                with os.fdopen(handle,'w') as stream:
                    json.dump(raw,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
                os.replace(tmp,target)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        try:
            with torch.inference_mode():
                for f,pixels in sparse.frames(path,wanted,clock=clock,check=live,width=width,height=height,
                        receipt=execution,maximum_decode_seconds=12.,maximum_seconds=120.):
                    live();boxes=_predict(pixels,model,torch,config)
                    values[s][f]=boxes;raw['rectangles'][str(f)]=boxes.tolist();calls+=1
                    if calls%16==0:checkpoint()
                    if calls%64==0:print(json.dumps(dict(phase='original_people',source=s,
                        new_person_forwards=calls,planned_new_frames=total,elapsed_seconds=monotonic()-started)),flush=True)
        finally:
            checkpoint()
        print(json.dumps(dict(phase='source_people_complete',source=s,new_person_forwards=calls,
            measured_native_frames=len(values[s]),elapsed_seconds=monotonic()-started)),flush=True)
    return values,dict(new_person_forwards=calls,source_rasters=calls,
        reused_frames=sum(map(len,requests))-total,elapsed_seconds=monotonic()-started,
        complete=all(set(r)<=set(v) for r,v in zip(requests,values)),decodes=decodes,
        person_masks_occlusion_ownership_identity_or_quality_certified=False)
