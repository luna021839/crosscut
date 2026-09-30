from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from crosscut.source_first_published_cache import (
    SourceFirstPublishedGeometryCache, decode_equivalent_focal_fingerprints,
    published_focal_geometry_predecessors,
)
from crosscut.source_first_execution import body_observation_namespace
from engine.tests.test_source_first_focal_geometry import geometry


class PublishedGeometryCacheTests(unittest.TestCase):
    def test_exact_forward_aliases_keep_device_model_and_unknown_code_bound(self):
        current = '50bc2a90b6bc5fb0582316f88e3feb7de12410890dd2496e7492ebcb4b4876bd'
        previous = decode_equivalent_focal_fingerprints(current)
        self.assertEqual(len(previous), 4)
        self.assertEqual(decode_equivalent_focal_fingerprints(previous[0]), previous[1:])
        self.assertEqual(decode_equivalent_focal_fingerprints(previous[1]), previous[2:])
        for unknown in (*previous[2:], '0'*64):
            self.assertEqual(decode_equivalent_focal_fingerprints(unknown), ())
        encoder = Mock()
        encoder.fingerprint.return_value = {'model': 'exact', 'adapter': 'mps-v1'}
        legacy_mps = body_observation_namespace(previous[-1], encoder, 'mps')
        self.assertNotEqual(legacy_mps, body_observation_namespace(previous[-1], encoder, 'cpu'))
        encoder.fingerprint.return_value = {'model': 'changed', 'adapter': 'mps-v1'}
        self.assertNotEqual(legacy_mps, body_observation_namespace(previous[-1], encoder, 'mps'))

    def test_pose_anchor_expansion_reuses_positive_incumbents_not_decode_equivalence(self):
        from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
        current=SourceFirstFocalGeometryProvider.observation_fingerprint()
        previous='50bc2a90b6bc5fb0582316f88e3feb7de12410890dd2496e7492ebcb4b4876bd'
        self.assertNotEqual(current,previous)
        self.assertEqual(decode_equivalent_focal_fingerprints(current),())
        self.assertEqual(published_focal_geometry_predecessors(current),
                         (previous,*decode_equivalent_focal_fingerprints(previous)))
        self.assertEqual(published_focal_geometry_predecessors('0'*64),())

    def test_predecessor_lookup_preserves_original_rows_and_new_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = SourceFirstPublishedGeometryCache(root, 'old')
            current = SourceFirstPublishedGeometryCache(root, 'new', compatible_namespaces=('old',))
            unrelated = SourceFirstPublishedGeometryCache(root, 'unknown')
            try:
                old.put('same-source-frame', geometry())
                self.assertEqual(current.get('same-source-frame', 0, 1.), old.get('same-source-frame', 0, 1.))
                self.assertIsNone(current.get('different-source-frame', 0, 1.))
                self.assertIsNone(unrelated.get('same-source-frame', 0, 1.))
                self.assertEqual(current.diagnostics()['compatible_published_geometry_hits'], 1)
                self.assertEqual(current.writes, 0)
                current.put('new-frame', geometry(body_only=True))
                self.assertIsNone(old.get('new-frame', 0, 1.))
                self.assertIsNotNone(current.get('new-frame', 0, 1.))
                self.assertEqual(current.namespace, 'new')
                self.assertEqual(current.connection.execute(
                    'SELECT namespace,frame_key FROM observations ORDER BY namespace').fetchall(),
                    [('new', 'new-frame'), ('old', 'same-source-frame')])
                # Current records take precedence, even if corrupt. A stale
                # compatible record must not conceal current corruption.
                current.put('same-source-frame', geometry(body_only=True))
                current.connection.execute("UPDATE observations SET digest='invalid' WHERE namespace='new' AND frame_key='same-source-frame'")
                current.connection.commit()
                self.assertIsNone(current.get('same-source-frame', 0, 1.))
                self.assertEqual(current.invalid_records, 1)
            finally:
                old.close(); current.close(); unrelated.close()

    def test_compatible_namespace_list_rejects_duplicates_and_current(self):
        with tempfile.TemporaryDirectory() as directory:
            for invalid in (('current',), ('old', 'old'), ['old'], ('',)):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    SourceFirstPublishedGeometryCache(Path(directory), 'current', compatible_namespaces=invalid)

    def test_two_live_sessions_can_publish_without_waiting_for_sixteen_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = SourceFirstPublishedGeometryCache(root, 'cpu')
            second = SourceFirstPublishedGeometryCache(root, 'mps')
            try:
                first.put('one', geometry())
                self.assertFalse(first.connection.in_transaction)
                second.put('two', geometry(body_only=True))
                self.assertFalse(second.connection.in_transaction)
                self.assertEqual(first.connection.execute('SELECT count(*) FROM observations').fetchone()[0], 2)
                self.assertIsNone(second.get('one', 0, 1.))
                self.assertIsNotNone(first.get('one', 0, 1.))
                self.assertIsNotNone(second.get('two', 0, 1.))
            finally:
                first.close()
                second.close()

    def test_publication_failure_rolls_back_without_claiming_a_cached_write(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SourceFirstPublishedGeometryCache(Path(directory), 'cpu')
            cache.connection.execute('PRAGMA busy_timeout=20')
            competitor = sqlite3.connect(cache.path)
            try:
                competitor.execute('BEGIN IMMEDIATE')
                with self.assertRaises(sqlite3.OperationalError):
                    cache.put('one', geometry())
                self.assertFalse(cache.connection.in_transaction)
                self.assertEqual(cache.writes, 0)
                competitor.rollback()
                cache.put('one', geometry())
                self.assertEqual(cache.writes, 1)
            finally:
                competitor.close()
                cache.close()


if __name__ == '__main__':
    unittest.main()
