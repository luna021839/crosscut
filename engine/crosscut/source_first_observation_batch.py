"""Schedule one cut-search window's existing observations in source order.

This does not choose a target, invent observations or replace the tracker.
The ordinary resolver still runs at every requested source/time and alone
publishes geometry. Only its native faces and body-frame dependencies are
materialized early, on the same frame lattice as ``track_at``. Raw pixels and
appearance vectors remain inside the existing bounded, session-local caches.
"""
from collections import defaultdict
import math
from time import monotonic

import cv2

from .focal_body_track_provider import select_direct_focal_observation


def track_dependency_times(sampler, seconds):
    """The tracker's exact native-frame lattice; tested against track_at."""
    fps = float(sampler.frame_rate)
    count = sampler.capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
        raise ValueError('source-local focal track frame clock unavailable')
    stride = max(1, round(.12 * fps))
    frame = sampler.frame_identity(seconds).frame_index
    center = frame // stride * stride
    times = [index / fps for step in range(-10, 11)
             if 0 <= (index := center + step * stride) < count - 1 and index != frame]
    return tuple(sorted((*times, seconds)))


class SourceFirstObservationBatch:
    """Bounded prefetch, never a new geometry authority or retry controller."""

    def __init__(self, provider, *, progress_callback=None):
        self.provider = provider
        self.progress_callback = progress_callback
        self.batches = self.request_count = self.dependency_frame_count = 0
        self.cached_geometry_count = self.capacity_fallback_count = 0
        self.direct_resolved_query_count = self.recovery_query_count = 0
        self.anchor_impossible_query_count = self.skipped_body_dependency_count = 0
        self.proven_missing_query_body_skip_count = 0
        self.disconnected_query_dependency_skip_count = 0
        self.face_first_capacity_fallback_count = 0
        self.elapsed_seconds = 0.

    def __call__(self, queries):
        queries = tuple(queries)
        provider = self.provider
        grouped = defaultdict(list)
        # Validate the whole request before observing anything. A large or
        # nonlocal request keeps the ordinary on-demand path, not an unbounded
        # pre-analysis of the entire song or a reduced set of editing choices.
        for source, seconds in queries:
            if (isinstance(source, bool) or not isinstance(source, int)
                    or not 0 <= source < len(provider.sources)
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError('observation batch requires exact source/time queries')
            grouped[source].append(seconds)
        for source in grouped:
            if provider._identity(provider.sources[source]) != provider._identities[source]:
                raise OSError('source-first focal source changed during analysis')
        if len(queries) > 1024 or any(max(times)-min(times) > 2. for times in grouped.values()):
            self.capacity_fallback_count += 1
            return
        started = monotonic()
        self.batches += 1
        self.request_count += len(queries)
        try:
            for source, times in sorted(grouped.items()):
                sampler = provider.faces._samplers[source]
                if sampler is None:
                    continue
                pending = {}
                for seconds in times:
                    key = provider.faces._identity_key(source, sampler, seconds)
                    if key in provider._cache:
                        self.cached_geometry_count += 1
                        continue
                    if provider._persistent is not None:
                        value = provider._persistent.get(provider._persistent_key(source, key), source, seconds)
                        if value is not None:
                            provider._cache[key] = value
                            provider._cache.move_to_end(key)
                            if len(provider._cache) > 4096:
                                provider._cache.popitem(last=False)
                            self.cached_geometry_count += 1
                            continue
                    pending.setdefault(key, seconds)
                if not pending:
                    continue
                # Frozen face/body reference banks are the same as the scalar
                # path, and are established before sorted query observations.
                bank = provider.faces._ensure_focal_consensus(source)
                references = provider.bodies._ensure_references(source)
                dependencies = {}
                recovery_times = []
                prefilter = getattr(provider, '_source_first_anchor_prefilter', None)
                # First observe the exact requested frames, in source order.
                # Only unresolved queries need the expensive track lattice.
                # Use the resolver's same pure rule; do not publish geometry
                # here or substitute adjacent evidence for an exact frame.
                for _key, seconds in sorted(pending.items(), key=lambda item: item[1]):
                    faces = provider.faces._focal_candidates(source, seconds)
                    self.dependency_frame_count += 1
                    face = bank.select(faces) if bank else None
                    # Adjacent offsets often share an already completed native
                    # face window. A missing exact target face plus fewer than
                    # two possible recovery anchors cannot produce geometry,
                    # irrespective of this query's unmeasured body/pose.
                    if (face is None and prefilter is not None
                            and prefilter.query_cannot_resolve(source, seconds)):
                        self.proven_missing_query_body_skip_count += 1
                        continue
                    frame = provider.bodies.frame_observations(source, seconds)
                    selected, reason = select_direct_focal_observation(face, faces, frame)
                    if selected is not None or reason == 'frozen_solo_face_body_unobserved':
                        self.direct_resolved_query_count += 1
                    elif prefilter is not None and prefilter.query_body_cannot_recover(source, seconds):
                        self.disconnected_query_dependency_skip_count += 1
                    elif len(references) >= 3:
                        self.recovery_query_count += 1
                        recovery_times.append(seconds)
                        for time in track_dependency_times(sampler, seconds):
                            key = provider.faces._identity_key(source, sampler, time)
                            if key not in pending:
                                dependencies.setdefault(key, time)
                owners = getattr(provider, 'shared_source_frames', None)
                capacity = owners[source].maximum_working_frames if owners else 0
                if prefilter is not None and len(dependencies) + len(pending) <= capacity:
                    # Faces establish an upper bound on possible identity
                    # anchors. Keep the corresponding working pixels inside
                    # the existing bounded cache for any body work still needed.
                    for _key, seconds in sorted(dependencies.items(), key=lambda item: item[1]):
                        provider.faces._focal_candidates(source, seconds)
                        self.dependency_frame_count += 1
                    # Materialize the body/pose only where the frozen face
                    # could support an anchor (or face evidence is unknown).
                    # Then keep the full trajectory only for windows that
                    # can still supply two independently observed anchors.
                    anchor_body_keys = set()
                    for key, seconds in sorted(dependencies.items(), key=lambda item: item[1]):
                        if prefilter.anchor_state(source, seconds) is not False:
                            provider.bodies.frame_observations(source, seconds)
                            anchor_body_keys.add(key)
                    required = set()
                    for seconds in recovery_times:
                        if prefilter.cannot_resolve(source, seconds):
                            self.anchor_impossible_query_count += 1
                        else:
                            required.update(provider.faces._identity_key(source, sampler, time)
                                for time in track_dependency_times(sampler, seconds))
                    for key, seconds in sorted(dependencies.items(), key=lambda item: item[1]):
                        if key in required and key not in anchor_body_keys:
                            provider.bodies.frame_observations(source, seconds)
                        elif key not in required and key not in anchor_body_keys:
                            self.skipped_body_dependency_count += 1
                else:
                    if prefilter is not None:
                        self.face_first_capacity_fallback_count += 1
                    for _key, seconds in sorted(dependencies.items(), key=lambda item: item[1]):
                        provider.faces._focal_candidates(source, seconds)
                        provider.bodies.frame_observations(source, seconds)
                        self.dependency_frame_count += 1
                if self.progress_callback is not None:
                    self.progress_callback(dict(phase='focal_observation_batch',
                        source_index=source, batch_seconds=monotonic()-started,
                        **self.diagnostics()))
        finally:
            self.elapsed_seconds += monotonic()-started

    def diagnostics(self):
        return dict(contract='source_ordered_exact_focal_dependency_batch_v1',
            batches=self.batches, request_count=self.request_count,
            dependency_frame_count=self.dependency_frame_count,
            cached_geometry_count=self.cached_geometry_count,
            direct_resolved_query_count=self.direct_resolved_query_count,
            recovery_query_count=self.recovery_query_count,
            anchor_impossible_query_count=self.anchor_impossible_query_count,
            skipped_body_dependency_count=self.skipped_body_dependency_count,
            proven_missing_query_body_skip_count=self.proven_missing_query_body_skip_count,
            disconnected_query_dependency_skip_count=self.disconnected_query_dependency_skip_count,
            face_first_capacity_fallback_count=self.face_first_capacity_fallback_count,
            capacity_fallback_count=self.capacity_fallback_count,
            elapsed_seconds=self.elapsed_seconds,
            target_selection_changed=False, query_offsets_removed=False,
            pixels_or_embeddings_persisted=False)
