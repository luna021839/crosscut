"""Avoid body inference when completed faces prove recovery has no anchors.

The existing tracker requires at least two distinct frozen-face/body anchors.
This necessary-condition check never identifies a person or admits a cut. It
only reads the same exact native face observations, frozen bank and completed
body observations the tracker would consume. Missing decodes/models and
incomplete windows stay unknown.
Positive geometry and its persistent measurement namespace are unchanged.
"""
import math

from .focal_body_track import deduplicate_body_observations, resolve_focal_body_track


class SourceFirstAnchorPrefilter:
    def __init__(self, provider):
        self.provider = provider
        self.original_track_at = provider.bodies.track_at
        self.checked_windows = self.impossible_windows = self.skipped_tracks = 0

    def anchor_state(self, source, seconds):
        """True means possible, False disproved, None missing observation."""
        provider = self.provider
        sampler = provider.faces._samplers[source]
        key = provider.faces._identity_key(source, sampler, seconds)
        body = provider.bodies._frame_cache.get(key)
        if body is not None and body.observations:
            # These are the same cached observations the full resolver uses,
            # after the same duplicate-body removal. A face alone is not an
            # anchor when body appearance/competition already disproves it.
            observed = deduplicate_body_observations(body.observations)
            return any(item.frozen_face_match and item.reference_similarity >= .68
                and item.reference_similarity - max(
                    (other.reference_similarity for k, other in enumerate(observed) if k != i),
                    default=-1.) >= .06 for i, item in enumerate(observed))
        faces = provider.faces._focal_candidate_cache.get(key)
        memo = provider._source_first_unresolved_memo
        if faces is None or not faces and key not in memo.completed_face_frames:
            return None
        bank = provider.faces._frozen_focal_banks.get(source)
        if bank is None or memo.namespace != provider.observation_namespace:
            return None
        return bank.select(faces) is not None

    def cannot_resolve(self, source, seconds):
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
            state = self.anchor_state(source, time)
            if state is None:
                return False
            if state:
                possible_times.add(time)
            if len(possible_times) >= 2:
                return False
        self.impossible_windows += 1
        return True

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
            positive_observation_namespace_changed=False, negative_observations_persisted=False)


def install_source_first_anchor_prefilter(provider):
    if isinstance(getattr(provider, '_source_first_anchor_prefilter', None), SourceFirstAnchorPrefilter):
        return
    prefilter = SourceFirstAnchorPrefilter(provider)
    provider._source_first_anchor_prefilter = prefilter
    provider.bodies.track_at = prefilter.track_at
