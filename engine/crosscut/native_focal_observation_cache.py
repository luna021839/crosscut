"""Transparent private cache of native YuNet/SFace observations.

The source observation producer is unchanged. Its query-time basis, model and
decoder dependencies bind this separate cache; edit/camera/body policy does
not. Complete negatives may be reused, but failed/partial decodes never become
cached absence. No private reference photographs or named-member labels enter
this store. Query embeddings remain local to the user's explicit media cache.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3
import zlib

import cv2
import numpy as np

from .direct_member_target_face import FocalFaceCandidate
from .visual_score import YUNET_EXPECTED_SHA256

MAX_RECORD_BYTES=512*1024


def _identity(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns


def producer_components(producer):
    # visual_score also owns thousands of lines of editorial policy. Hash
    # only its actual detector/normalization kernels, not unrelated edit costs.
    from . import face_identity as identity, visual_score as visual, direct_member_target_face as target
    kernels={
        'identity.'+name:getattr(identity,name) for name in ('_default_model_path','_sha256',
            '_validate_normalized_face','SFaceEmbedder','sface_model_asset_status','extract_sface_embedding')}
    kernels.update({'visual.'+name:getattr(visual,name) for name in ('_working_frame','_detect_faces',
        '_normalized_face_metadata','_unit','_yunet_detector','yunet_model_asset_status','_DetectedFace')})
    kernels.update({'target.FocalFaceCandidate':target.FocalFaceCandidate,
        'target._landmark_rms_scale':target._landmark_rms_scale})
    dependencies=('native_focal_faces.py','native_frame_reader.py','seek_locality.py','shared_source_frames.py')
    return dict(producer=producer.fingerprint(),yunet_sha256=YUNET_EXPECTED_SHA256,
        opencv=cv2.__version__,numpy=np.__version__,
        opencv_build=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        decoder_dependencies={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in dependencies},
        kernels={name:hashlib.sha256(inspect.getsource(value).encode()).hexdigest() for name,value in kernels.items()},
        cache_schema=1,exact_requested_seconds_not_a_VFR_native_index=True)


def producer_namespace(producer):
    value=producer_components(producer)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _faces(payload):
    if (not isinstance(payload,dict) or set(payload)!= {'schema','complete','faces'}
            or payload['schema']!=1 or payload['complete'] is not True
            or not isinstance(payload['faces'],list) or len(payload['faces'])>64):
        raise ValueError('invalid complete native face cache record')
    result=[]
    for row in payload['faces']:
        if not isinstance(row,dict) or set(row)!= {'box','vector','eye_scale','landmark_scale'}:
            raise ValueError('invalid native face cache fields')
        vector=row['vector']
        if (not isinstance(vector,list) or len(vector)!=128 or
                any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1.001 for v in vector)
                or abs(math.fsum(v*v for v in vector)-1.)>.002):
            raise ValueError('native SFace128 cache cannot accept another descriptor family')
        result.append(FocalFaceCandidate(tuple(row['box']),tuple(vector),row['eye_scale'],row['landmark_scale']))
    return tuple(result)


class PersistentNativeFocalFaces:
    """Delegate native inference unchanged; memoize complete source results."""
    def __init__(self,producer,root):
        self.producer=producer;self.root=Path(root).resolve(strict=True)
        if not self.root.is_dir():raise OSError('explicit native face cache root unavailable')
        self.sources=tuple(producer.sources);self.identities=tuple(_identity(p) for p in self.sources)
        self.namespace=producer_namespace(producer);self.path=self.root/'native-focal-faces-v1.sqlite3'
        if self.path.is_symlink():raise OSError('native face cache must be a regular local artifact')
        # Finish ExFAT initialization before reopening for publication, as in
        # the scalar source cache. Never hold a write transaction during NN.
        bootstrap=sqlite3.connect(self.path,timeout=5.)
        try:
            bootstrap.execute('PRAGMA journal_mode=DELETE')
            bootstrap.execute('CREATE TABLE IF NOT EXISTS observations(namespace TEXT NOT NULL, query_key TEXT NOT NULL, digest TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(namespace,query_key)) WITHOUT ROWID')
            bootstrap.commit()
        finally:bootstrap.close()
        self.connection=sqlite3.connect(self.path,timeout=5.)
        self.connection.execute('PRAGMA synchronous=FULL');self.connection.execute('PRAGMA cache_size=-4096')
        self.cache_identity=_identity(self.path)[:2];self.root_identity=_identity(self.root)[:2]
        self.hits=self.misses=self.writes=self.invalid=0;self.closed=False

    def __getattr__(self,name):return getattr(self.producer,name)

    def _check(self,source):
        if self.closed:raise OSError('native face cache is closed')
        if (_identity(self.root)[:2]!=self.root_identity or _identity(self.path)[:2]!=self.cache_identity
                or _identity(self.sources[source])!=self.identities[source]):
            raise OSError('native face cache or physical source changed/disconnected')

    def candidates_at(self,source,seconds):
        if (type(source) is not int or not 0<=source<len(self.sources) or type(seconds) not in (int,float)
                or not math.isfinite(seconds) or seconds<0):
            raise ValueError('exact source and finite native-face query seconds required')
        self._check(source)
        memory=(source,seconds)
        if memory in self.producer._cache:
            return self.producer.candidates_at(source,seconds)
        query=hashlib.sha256(json.dumps((str(self.sources[source].resolve()),self.identities[source],
            float(seconds).hex()),separators=(',',':')).encode()).hexdigest()
        row=self.connection.execute('SELECT digest,payload FROM observations WHERE namespace=? AND query_key=?',
            (self.namespace,query)).fetchone()
        if row is not None:
            try:
                if len(row[1])>MAX_RECORD_BYTES:raise ValueError('oversize native face compressed record')
                decoder=zlib.decompressobj();raw=decoder.decompress(row[1],MAX_RECORD_BYTES+1)
                if (len(raw)>MAX_RECORD_BYTES or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                        or hashlib.sha256(raw).hexdigest()!=row[0]):raise ValueError('native face cache checksum differs')
                result=_faces(json.loads(raw))
            except (ValueError,TypeError,KeyError,OverflowError,zlib.error):self.invalid+=1
            else:
                self._check(source);self.hits+=1
                self.producer._cache[memory]=result;self.producer._candidate_completeness[memory]=True
                if len(self.producer._cache)>4096:
                    evicted,_=self.producer._cache.popitem(last=False);self.producer._candidate_completeness.pop(evicted,None)
                return result
        self.misses+=1
        result=self.producer.candidates_at(source,seconds);self._check(source)
        if self.producer.candidate_observation_complete(source,seconds):
            payload=dict(schema=1,complete=True,faces=[dict(box=list(f.bbox_xywh),vector=list(f.embedding),
                eye_scale=f.eye_scale,landmark_scale=f.landmark_scale) for f in result])
            try:
                _faces(payload);raw=json.dumps(payload,separators=(',',':'),allow_nan=False).encode()
                if len(raw)>MAX_RECORD_BYTES:raise ValueError('native face cache record budget exceeded')
            except (ValueError,TypeError):return result  # Unsupported producer data is not rewritten.
            self.connection.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?)',
                (self.namespace,query,hashlib.sha256(raw).hexdigest(),zlib.compress(raw)))
            self.connection.commit();self.writes+=1
        return result

    def diagnostics(self):
        return dict(hits=self.hits,misses=self.misses,writes=self.writes,invalid_records=self.invalid,
            complete_negative_reuse=True,private_query_embeddings_persisted=True,reference_photos_copied=False,
            original_sensor_producer_unchanged=True,body_or_editor_policy_in_namespace=False)

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.connection.close()
        finally:self.producer.close()
