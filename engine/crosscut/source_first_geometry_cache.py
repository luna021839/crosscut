"""Small scalar focal-observation cache, separate from edit-path decisions.

Contains face/body boxes, observed nonfacial joints and track certificates.
No raw pixels, face embeddings or body appearance embeddings are written.
SQLite's rollback journal keeps publication bounded; a disconnected external
cache is an error, never a reason to create an internal substitute.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from .focal_body_track import FocalBodyGeometryCertificate
from .focal_selector_geometry import FocalSelectorGeometry
from .pose import FormationPersonDescriptor

MAXIMUM_RECORD_BYTES=32*1024


def serialize_focal_geometry(value):
    person=value.person
    certificate=value.body_track
    if certificate is not None and not isinstance(certificate,FocalBodyGeometryCertificate):
        certificate=FocalBodyGeometryCertificate.from_observed_track(certificate)
    return {'schema':1,'source_seconds':value.source_seconds,'face_bbox_xywh':value.face_bbox_xywh,
        'face_scale':value.face_scale,'pose_backend':value.pose_backend,
        'pose_screen_xyc':value.pose_screen_xyc,
        'person':None if person is None else {'bbox_xywh':person.bbox_xywh,
            'centroid_xy':person.centroid_xy,'detection_confidence':person.detection_confidence,
            'upper_body_keypoints_xyc':person.upper_body_keypoints_xyc},
        'body_certificate':certificate.to_dict() if certificate else None}


def deserialize_focal_geometry(payload,source_index,seconds):
    if not isinstance(payload,dict) or set(payload)!={'schema','source_seconds','face_bbox_xywh',
            'face_scale','pose_backend','pose_screen_xyc','person','body_certificate'} or payload['schema']!=1:
        raise ValueError('source-first geometry cache schema differs')
    person=payload['person']
    if person is not None:
        if not isinstance(person,dict) or set(person)!={'bbox_xywh','centroid_xy','detection_confidence','upper_body_keypoints_xyc'}:
            raise ValueError('source-first cached body fields differ')
        person=FormationPersonDescriptor(tuple(person['bbox_xywh']),tuple(person['centroid_xy']),
            person['detection_confidence'],tuple(tuple(p) for p in person['upper_body_keypoints_xyc']))
    certificate=payload['body_certificate']
    if certificate is not None:
        if certificate.get('contract')!='source_local_focal_body_geometry_certificate_v1':
            raise ValueError('source-first cached body certificate differs')
        certificate=FocalBodyGeometryCertificate(source_index,seconds,tuple(certificate['body_bbox_xywh']),
            tuple(certificate['anchor_seconds']),certificate['path_margin'],certificate['reason'])
    box=payload['face_bbox_xywh']
    return FocalSelectorGeometry(source_index,seconds,tuple(box) if box is not None else None,
        payload['face_scale'],person,certificate,payload['pose_backend'],tuple(tuple(p) for p in payload['pose_screen_xyc']))


class SourceFirstGeometryCache:
    def __init__(self,root:Path,namespace:str,*,compatible_namespaces=()):
        if (not isinstance(compatible_namespaces,tuple)
                or any(not isinstance(item,str) or not item for item in compatible_namespaces)
                or namespace in compatible_namespaces
                or len(set(compatible_namespaces))!=len(compatible_namespaces)):
            raise ValueError('source-first compatible namespaces must be unique predecessors')
        self.root=Path(root)
        if not self.root.is_dir():raise OSError('selected source-first cache is unavailable')
        self.path=self.root/'source-first-geometry-v1.sqlite3'
        self.namespace=namespace
        self._read_namespaces=(namespace,*compatible_namespaces)
        # ExFAT can assign a different internal file identity while a new
        # SQLite file is initialized. Its creating handle then reports
        # SQLITE_READONLY_DBMOVED on the first INSERT, even though the path
        # is writable. Reopen the completed schema, BEFORE any observations;
        # never disable SQLite's moved-file checks or replay a failed write.
        bootstrap=sqlite3.connect(self.path,timeout=5.)
        try:
            bootstrap.execute('PRAGMA journal_mode=DELETE')
            bootstrap.execute('CREATE TABLE IF NOT EXISTS observations (namespace TEXT NOT NULL, frame_key TEXT NOT NULL, digest TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(namespace,frame_key)) WITHOUT ROWID')
            bootstrap.commit()
        finally:bootstrap.close()
        self.connection=sqlite3.connect(self.path,timeout=5.)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA temp_store=MEMORY')
        self.connection.execute('PRAGMA cache_size=-8192')
        self._file_identity=self._identity()
        self.hits=self.misses=self.invalid_records=self.writes=0
        self.legacy_hits=0
        self._pending=0

    def _identity(self):
        info=self.path.stat()
        return info.st_dev,info.st_ino

    def _assert_available(self):
        if not self.root.is_dir() or self._identity()!=self._file_identity:
            raise OSError('external source-first cache disconnected or replaced')

    def get(self,key,source_index,seconds):
        self._assert_available()
        row=None
        for namespace in self._read_namespaces:
            row=self.connection.execute('SELECT digest,payload FROM observations WHERE namespace=? AND frame_key=?',
                (namespace,key)).fetchone()
            if row is not None:break
        if row is None:
            self.misses+=1;return None
        try:
            decoder=zlib.decompressobj()
            raw=decoder.decompress(row[1],MAXIMUM_RECORD_BYTES+1)
            if len(raw)>MAXIMUM_RECORD_BYTES or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
                raise ValueError('source-first cached geometry exceeds its record bound')
            if hashlib.sha256(raw).hexdigest()!=row[0]:raise ValueError('cached geometry checksum differs')
            value=deserialize_focal_geometry(json.loads(raw),source_index,seconds)
        except (ValueError,TypeError,KeyError,OverflowError,zlib.error):
            self.invalid_records+=1;self.misses+=1;return None
        self.hits+=1
        self.legacy_hits+=int(namespace!=self.namespace)
        return value

    def put(self,key,value):
        self._assert_available()
        raw=json.dumps(serialize_focal_geometry(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAXIMUM_RECORD_BYTES:raise ValueError('source-first geometry record exceeds bound')
        self.connection.execute('INSERT OR REPLACE INTO observations(namespace,frame_key,digest,payload) VALUES(?,?,?,?)',
            (self.namespace,key,hashlib.sha256(raw).hexdigest(),zlib.compress(raw)))
        self._pending+=1;self.writes+=1
        if self._pending>=16:
            self.connection.commit();self._pending=0

    def diagnostics(self):
        return {'contract':'source_first_scalar_geometry_cache_v1','hits':self.hits,'misses':self.misses,
            'decode_equivalent_legacy_hits':self.legacy_hits,
            'invalid_records':self.invalid_records,'writes':self.writes,
            'appearance_embeddings_persisted':False,'raw_pixels_persisted':False}

    def close(self):
        try:self.connection.commit()
        finally:self.connection.close()
