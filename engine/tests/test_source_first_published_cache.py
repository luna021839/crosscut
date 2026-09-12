from pathlib import Path
import sqlite3
import tempfile
import unittest

from crosscut.source_first_published_cache import SourceFirstPublishedGeometryCache
from engine.tests.test_source_first_focal_geometry import geometry


class PublishedGeometryCacheTests(unittest.TestCase):
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
