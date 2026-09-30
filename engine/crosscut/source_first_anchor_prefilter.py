"""Avoid body inference when completed faces prove recovery has no anchors.

The existing tracker requires at least two distinct frozen-face/body anchors.
This necessary-condition check never identifies a person or admits a cut. It
only reads the same exact native face observations, frozen bank and completed
body observations the tracker would consume. Missing decodes/models and
incomplete windows stay unknown.
Positive geometry decisions are unchanged. The extracted query-edge floor
changes the dependency identity; only its exact verified positive predecessor
is reusable, through the ordinary published-geometry cache contract.
"""
import math

from .focal_body_track import (FocalBodyFrame, FocalBodyTrackResult, deduplicate_body_observations,
    focal_body_anchor_indices, query_body_recovery_impossible, resolve_focal_body_track)


class SourceFirstAnchorPrefilter:
    def __init__(self, provider):
        self.provider = provider
        self.original_track_at = provider.bodies.track_at
        self.checked_windows = self.impossible_windows = self.skipped_tracks = 0
        self.disconnected_query_tracks = 0

    def query_body_cannot_recover(self, source, seconds):
        """Read only the exact cached query; never reject direct face admission.

        This can avoid an unused recovery lattice even when its face-anchor
        windows are still unobserved. Existing track results keep priority;
        unknown/mismatched query frames supply no disproof. No negative is
        persisted, and the normal resolver still decides the query geometry.
        """
        provider = self.provider
        if (type(source) is not int or not 0 <= source < len(provider.sources)
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            return False
        if provider._identity(provider.sources[source]) != provider._identities[source]:
            raise OSError('source-first focal source changed during analysis')
        sampler = provider.faces._samplers[source]
        memo = getattr(provider, '_source_first_unresolved_memo', None)
        if sampler is None or memo is None or memo.namespace != provider.observation_namespace:
            return False
        key = provider.faces._identity_key(source, sampler, seconds)
        query = provider.bodies._frame_cache.get(key)
        if key in provider.bodies._track_cache or not isinstance(query, FocalBodyFrame):
            return False
        if (query.source_index != source
                or provider.faces._identity_key(source, sampler, query.source_seconds) != key):
            return False
        # Do not turn the ordinary tracker's unavailable-clock error into a
        # negative target observation merely because one old frame is cached.
        from .source_first_observation_batch import track_dependency_times
        try:
            track_dependency_times(sampler, seconds)
        except ValueError:
            return False
        return query_body_recovery_impossible(query)

    def anchor_state(self, source, seconds):
        """True means possible, False disproved, None missing observation."""
        provider = self.provider
        sampler = provider.faces._samplers[source]
        key = provider.faces._identity_key(source, sampler, seconds)
        body = provider.bodies._frame_cache.get(key)
        if body is not None and body.observations:
            # These are the same cached observations the full resolver uses,
            # after the same duplicate-body removal. Exact pose-owned anchors
            # must not be erased by the old clothing-only upper bound.
            observed = deduplicate_body_observations(body.observations)
            return bool(focal_body_anchor_indices(observed))
        return self.native_face_anchor_state(source, seconds)

    def native_face_anchor_state(self, source, seconds, *, require_complete=False):
        provider = self.provider
        sampler = provider.faces._samplers[source]
        key = provider.faces._identity_key(source, sampler, seconds)
        faces = provider.faces._focal_candidate_cache.get(key)
        memo = provider._source_first_unresolved_memo
        if (faces is None or (require_complete or not faces)
                and key not in memo.completed_face_frames):
            return None
        bank = provider.faces._frozen_focal_banks.get(source)
        if bank is None or memo.namespace != provider.observation_namespace:
            return None
        return bank.select(faces) is not None

    def cannot_resolve(self, source, seconds, *, native_faces_only=False):
        """A complete upper bound of fewer than two anchors, never a guess."""
        from .source_first_observation_batch import track_dependency_times
        provider = self.provider
        if (isinstance(source, bool) or not isinstance(source, int)
                or not 0 <= source < len(provider.sources)
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            return False
        if provider._identity(provider.sources[source]) != provider._identities[source]:
            raise OSError('source-first focal source changed during analysis')
        sampler = provider.faces._samplers[source]
        bank = provider.faces._frozen_focal_banks.get(source)
        memo = getattr(provider, '_source_first_unresolved_memo', None)
        if (sampler is None or bank is None or memo is None
                or memo.namespace != provider.observation_namespace):
            return False
        self.checked_windows += 1
        possible_times = set()
        for time in track_dependency_times(sampler, seconds):
            state = (self.native_face_anchor_state(source, time, require_complete=True)
                     if native_faces_only else self.anchor_state(source, time))
            if state is None:
                return False
            if state:
                possible_times.add(time)
            if len(possible_times) >= 2:
                return False
        self.impossible_windows += 1
        return True

    def query_cannot_resolve(self, source, seconds):
        """Prove both exact-face admission and body recovery are impossible.

        This is stricter than the track-only test: even one visible frozen
        face can directly admit an exact body/pose without two track anchors.
        No model work is done here. An unobserved/failed query remains unknown.
        """
        # Do not memoize disproof from possibly incomplete body/pose inference.
        # The complete native face upper bound alone must be sufficient.
        if not self.cannot_resolve(source, seconds, native_faces_only=True):
            return False
        provider = self.provider
        sampler = provider.faces._samplers[source]
        key = provider.faces._identity_key(source, sampler, seconds)
        faces = provider.faces._focal_candidate_cache.get(key)
        memo = provider._source_first_unresolved_memo
        bank = provider.faces._frozen_focal_banks[source]
        if (faces is None or not faces and key not in memo.completed_face_frames
                or bank.select(faces) is not None):
            return False
        existing = provider.bodies._track_cache.get(key)
        # An already resolved trajectory retains its original authority.
        return existing is None or existing.observation is None

    def track_at(self, source, seconds):
        provider = self.provider
        # Keep validation and existing positive/negative track cache semantics
        # with the original provider. Only a fully observed new query can skip.
        if (isinstance(source, bool) or not isinstance(source, int)
                or not 0 <= source < len(provider.sources)
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            return self.original_track_at(source, seconds)
        sampler = provider.faces._samplers[source]
        if sampler is None:
            return self.original_track_at(source, seconds)
        key = provider.faces._identity_key(source, sampler, seconds)
        query = provider.bodies._frame_cache.get(key)
        if self.query_body_cannot_recover(source, seconds):
            # No synthetic anchors: the query-only necessary condition proves
            # that every possible route in the full graph is disconnected.
            result = FocalBodyTrackResult(source, seconds, None,
                'query_body_edges_unreachable' if query.observations else 'query_body_unobserved')
            self.disconnected_query_tracks += 1
            provider.bodies._track_cache[key] = result
            provider.bodies._track_cache.move_to_end(key)
            if len(provider.bodies._track_cache) > 4096:
                provider.bodies._track_cache.popitem(last=False)
            return result
        if (key in provider.bodies._track_cache or query is None
                or not self.cannot_resolve(source, seconds)):
            return self.original_track_at(source, seconds)
        # Resolve the same observed query with a proved anchor upper bound.
        # This preserves query_body_unobserved vs anchors_missing, and cannot
        # return a body observation from an incomplete recovery window.
        query = provider.bodies.frame_observations(source, seconds)
        result = resolve_focal_body_track((query,), source_index=source, query_seconds=seconds)
        assert result.observation is None
        self.skipped_tracks += 1
        provider.bodies._track_cache[key] = result
        provider.bodies._track_cache.move_to_end(key)
        if len(provider.bodies._track_cache) > 4096:
            provider.bodies._track_cache.popitem(last=False)
        return result

    def diagnostics(self):
        return dict(contract='complete_native_face_anchor_upper_bound_v1',
            checked_windows=self.checked_windows, impossible_windows=self.impossible_windows,
            skipped_tracks=self.skipped_tracks, new_identity_decisions=False,
            disconnected_query_tracks=self.disconnected_query_tracks,
            positive_geometry_decisions_changed=False, negative_observations_persisted=False)


def install_source_first_anchor_prefilter(provider):
    if isinstance(getattr(provider, '_source_first_anchor_prefilter', None), SourceFirstAnchorPrefilter):
        return
    prefilter = SourceFirstAnchorPrefilter(provider)
    provider._source_first_anchor_prefilter = prefilter
    provider.bodies.track_at = prefilter.track_at
