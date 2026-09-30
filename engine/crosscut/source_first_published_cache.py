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


def published_focal_geometry_predecessors(current):
    """Reuse only proven positive geometry across an incumbent-preserving expansion.

    The exact-pose anchor fallback first executes the unchanged legacy graph
    and returns its resolved result verbatim. Only unresolved queries reach the
    new family, and the persistent store never contains unresolved geometry.
    This is positive-result compatibility, NOT decode equivalence or permission
    to reuse negative results. All other code/device/model changes remain bound
    to their own namespace, and no existing database row is rewritten.
    """
    if current == '1389f32b65bc50b7226217d638c36b844ddb0164c53482983922b6d6e22d5e4f':
        previous='50bc2a90b6bc5fb0582316f88e3feb7de12410890dd2496e7492ebcb4b4876bd'
        return (previous,*decode_equivalent_focal_fingerprints(previous))
    return decode_equivalent_focal_fingerprints(current)


class SourceFirstPublishedGeometryCache(SourceFirstGeometryCache):
    def diagnostics(self):
        payload=super().diagnostics()
        payload['compatible_published_geometry_hits']=payload.pop('decode_equivalent_legacy_hits')
        payload['predecessor_lookup_scope']='positive_geometry_only_not_unresolved_or_decode_equivalence'
        return payload

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
