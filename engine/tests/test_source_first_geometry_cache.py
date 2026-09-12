from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from crosscut.source_first_geometry_cache import (SourceFirstGeometryCache,
    serialize_focal_geometry,deserialize_focal_geometry)
from crosscut.focal_body_track import FocalBodyGeometryCertificate
from engine.tests.test_source_first_focal_geometry import geometry


class SourceFirstGeometryCacheTests(unittest.TestCase):
    def test_track_round_trip_without_dummy_appearance_embedding(self):
        original=geometry(body_only=True)
        payload=serialize_focal_geometry(original)
        loaded=deserialize_focal_geometry(payload,0,1.)
        self.assertIsInstance(loaded.body_track,FocalBodyGeometryCertificate)
        self.assertFalse(hasattr(loaded.body_track,'observation'))
        self.assertEqual(loaded.person,original.person)
        self.assertEqual(loaded.body_track.support_seconds,original.body_track.support_seconds)
        self.assertIsNone(loaded.face_bbox_xywh)

    def test_reordered_source_pool_rebinds_only_runtime_source_index(self):
        loaded=deserialize_focal_geometry(serialize_focal_geometry(geometry(body_only=True)),3,1.)
        self.assertEqual(loaded.source_index,3)
        self.assertEqual(loaded.body_track.source_index,3)

    def test_restart_hits_published_external_scalar_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=SourceFirstGeometryCache(Path(directory),'model-A')
            cache.put('source-frame',geometry(body_only=True));cache.close()
            cache=SourceFirstGeometryCache(Path(directory),'model-A')
            result=cache.get('source-frame',0,1.)
            self.assertIsNotNone(result)
            self.assertEqual(cache.hits,1)
            self.assertEqual(cache.writes,0)
            self.assertFalse(cache.diagnostics()['appearance_embeddings_persisted'])
            cache.close()

    def test_model_binding_change_cannot_reuse_old_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=SourceFirstGeometryCache(Path(directory),'old-model')
            cache.put('source-frame',geometry());cache.close()
            cache=SourceFirstGeometryCache(Path(directory),'new-model')
            self.assertIsNone(cache.get('source-frame',0,1.));cache.close()

    def test_corrupt_record_is_not_treated_as_quality_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=SourceFirstGeometryCache(Path(directory),'model')
            cache.put('source-frame',geometry())
            cache.connection.execute("UPDATE observations SET payload=?",(b'broken-zlib',))
            self.assertIsNone(cache.get('source-frame',0,1.))
            self.assertEqual(cache.invalid_records,1);cache.close()

    def test_unavailable_root_does_not_create_internal_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing=Path(directory)/'not-mounted'
            with self.assertRaises(OSError):SourceFirstGeometryCache(missing,'model')
            self.assertFalse(missing.exists())

    def test_cached_proof_cannot_relabel_an_unrelated_body(self):
        payload=serialize_focal_geometry(geometry(body_only=True))
        payload['body_certificate']['body_bbox_xywh']=(.01,.01,.1,.1)
        with self.assertRaises(ValueError):deserialize_focal_geometry(payload,0,1.)


if __name__=='__main__':unittest.main()
