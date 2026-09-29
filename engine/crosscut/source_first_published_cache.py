"""Publish each completed observation without holding a lock during inference.

The base cache's record format, validation and namespace are unchanged. Its
16-record transaction batching held a SQLite write lock across potentially
minutes of model work, preventing another app/QA session from publishing.
"""
import sqlite3
from .source_first_geometry_cache import SourceFirstGeometryCache


def decode_equivalent_focal_fingerprints(current):
    """Exact forward-only aliases; active observations keep their new identity.

    The changes are pixel-identical short-gap RGB optimization, read-only
    predecessor lookup, and scalar native-candidate completion receipts. The
    receipts and exact CFR working-frame cache aliases do not change returned
    face candidates or positive geometry.
    The full dependency
    hash still invalidates unknown code/runtime changes. Device/model adapters
    are bound separately by the provider, never aliased across devices.
    """
    predecessors = (
        '54c9ce4dec339b1ef0a08c9ff7b349900990a1c40ef7b0a4360bfa4d4f533efa',
        '7256813d75301484a1245a811ce6b2e2449ca78b366b9d5b8e18269964f3f34b',
    )
    receipt_free = '4df743005101022ea8fff0496a251e86b5e7206ba3be181fcef7fc6a0133150a'
    receipt_bound = 'e6de0986993634829207a5c6565d3e032a995ea037e6a4d4be0be1153b5758b6'
    if current == '50bc2a90b6bc5fb0582316f88e3feb7de12410890dd2496e7492ebcb4b4876bd':
        return (receipt_bound, receipt_free, *predecessors)
    if current == receipt_bound:
        return (receipt_free, *predecessors)
    if current == receipt_free:
        return predecessors
    return ()


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
