"""Publish each completed observation without holding a lock during inference.

The base cache's record format, validation and namespace are unchanged. Its
16-record transaction batching held a SQLite write lock across potentially
minutes of model work, preventing another app/QA session from publishing.
"""
import sqlite3
from .source_first_geometry_cache import SourceFirstGeometryCache


class SourceFirstPublishedGeometryCache(SourceFirstGeometryCache):
    def put(self, key, value):
        writes_before = self.writes
        try:
            super().put(key, value)
            self.connection.commit()
            self._pending = 0
        except BaseException:
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass  # Preserve the original IO/publication failure.
            self.writes = writes_before
            self._pending = 0
            raise
