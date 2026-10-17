"""Exact current-face ArcFace observations, separate from six-member scores.

Only the original source/native/face rectangle can retrieve a query. Runtime
provenance stays with each snapshot; raw vectors are local numeric inputs, not
named identities, transferable labels, or material for review reports.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from crosscut.native_face_descriptor import ARCFACE_R50_SHA256

MAXIMUM_QUERIES=4096
MAXIMUM_SNAPSHOT_BYTES=16*1024**2


def query_key(native,head):
    head=np.asarray(head)
    if (type(native) is not int or native<0 or head.shape!=(4,) or not np.isfinite(head).all()):
        raise ValueError('actual source native ordinal and finite original face required')
    return hashlib.sha256(np.array([native,*head],np.float64).tobytes()).hexdigest()


def descriptor_family(binding):
    if (binding.get('model')!=ARCFACE_R50_SHA256 or binding.get('raster')!=[960,540]
            or any(not isinstance(binding.get(k),str) or len(binding[k])!=64
                   or any(c not in '0123456789abcdef' for c in binding[k])
                   for k in ('source_sha256','adapter','detector','association','decoder_implementation'))):
        raise ValueError('pinned original-source ArcFace observation binding required')
    return hashlib.sha256(json.dumps({k:binding[k] for k in
        ('model','adapter','detector','association','raster','decoder_implementation')},sort_keys=True).encode()).hexdigest()


def validate_values(keys,vectors,valid):
    keys=np.asarray(keys);vectors=np.asarray(vectors);valid=np.asarray(valid)
    if (keys.ndim!=1 or keys.dtype.kind!='U' or len(keys)>MAXIMUM_QUERIES
            or len(set(map(str,keys)))!=len(keys) or any(len(k)!=64 for k in keys)
            or vectors.shape!=(len(keys),512) or vectors.dtype.kind!='f' or not np.isfinite(vectors).all()
            or valid.shape!=(len(keys),) or valid.dtype!=bool
            or np.any(np.abs(np.linalg.norm(vectors[valid],axis=-1)-1.)>.001)
            or np.any(vectors[~valid]!=0.)):
        raise ValueError('bounded actual512D unit queries or explicit failed observations required')
    result={}
    for key,vector,ok in zip(map(str,keys),vectors,valid):
        vector=np.asarray(vector,np.float32).copy();vector.setflags(write=False)
        result[key]=(vector,bool(ok))
    return result


def read_snapshot(path,*,binding):
    path=Path(path);descriptor_family(binding)
    if path.is_symlink() or path.stat().st_size>MAXIMUM_SNAPSHOT_BYTES:
        raise ValueError('bounded local query snapshot required')
    with np.load(path,allow_pickle=False) as saved:
        if str(saved['binding'])!=json.dumps(binding,sort_keys=True):
            raise ValueError('actual query snapshot producer differs')
        return validate_values(saved['keys'],saved['vectors'],saved['valid'])


def publish_snapshot(path,values,*,binding,check=lambda:None):
    """Checkpoint only queries produced by THIS runtime, never relabel parents."""
    path=Path(path);descriptor_family(binding);check()
    keys=np.array(sorted(values),dtype='U64')
    vectors=np.array([values[k][0] for k in keys],np.float32).reshape(-1,512)
    valid=np.array([values[k][1] for k in keys],bool)
    validate_values(keys,vectors,valid)
    descriptor,temporary=tempfile.mkstemp(prefix='.direct-face-',suffix='.npz',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            np.savez_compressed(stream,binding=json.dumps(binding,sort_keys=True),keys=keys,vectors=vectors,valid=valid)
            stream.flush();os.fsync(stream.fileno())
        if Path(temporary).stat().st_size>MAXIMUM_SNAPSHOT_BYTES:
            raise ValueError('direct face snapshot exceeded its explicit16MiB budget')
        check();os.replace(temporary,path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()


def fields(bank,values):
    """No nearest-face/other-frame transfer, fabricated query, or source veto."""
    heads=np.asarray(bank['heads']);natives=np.asarray(bank['native_indices'])
    if (heads.ndim!=4 or heads.shape[-1]!=4 or natives.shape!=heads.shape[:2]
            or natives.dtype.kind not in 'iu' or np.any(natives<0) or not np.isfinite(heads).all()):
        raise ValueError('original source-bound endpoint face bank required')
    vectors=np.zeros(heads.shape[:-1]+(512,),np.float32);known=np.zeros(heads.shape[:-1],bool)
    for address in np.ndindex(natives.shape):
        for h,head in enumerate(heads[address]):
            item=values.get(query_key(int(natives[address]),head))
            if item is None or not item[1]:continue
            vector=np.asarray(item[0])
            if vector.shape!=(512,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1.)>.001:
                raise ValueError('current face observation is not a512D unit query')
            vectors[address+(h,)]=vector;known[address+(h,)]=True
    vectors.setflags(write=False);known.setflags(write=False)
    return vectors,known
