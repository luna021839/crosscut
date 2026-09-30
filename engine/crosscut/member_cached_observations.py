"""Warm exact member proposals from positive observations, without inference.

This is a read-only observation path, not a saved edit decision. Every pair is
scored by the current policy, every selected partial pair retains its charged
refinement, and full camera/render/QA authority stays downstream.
"""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

from .focal_body_track import FocalBodyGeometryCertificate
from .focal_selector_geometry import FocalSelectorGeometry
from .visual_score import FrameVisualDescriptor


@dataclass(frozen=True, slots=True)
class CachedMemberObservation:
    descriptor: FrameVisualDescriptor
    geometry: FocalSelectorGeometry

    def __post_init__(self):
        if (not isinstance(self.descriptor, FrameVisualDescriptor)
                or not isinstance(self.geometry, FocalSelectorGeometry)
                or self.geometry.face_bbox_xywh is None and self.geometry.person is None):
            raise ValueError('cached member evidence needs actual positive geometry and context')


def attach_cached_member_observations(focal, visual):
    """Use the same source/frame/model-bound stores; never call a resolver."""
    def cached(source, seconds):
        if (type(source) is not int or not 0 <= source < len(focal.sources)
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('cached member request must identify a source and finite time')
        if focal._identity(focal.sources[source]) != focal._identities[source]:
            raise OSError('cached member source changed during analysis')
        sampler = visual._samplers[source]
        focal_sampler = focal.faces._samplers[source]
        if sampler is None or focal_sampler is None:
            return None
        if visual._require_persistent_feature_cache:
            sampler.assert_required_persistent_cache_available()
        key = focal.faces._identity_key(source, focal_sampler, seconds)
        geometry = focal._cache.get(key)
        if geometry is None and focal._persistent is not None:
            geometry = focal._persistent.get(focal._persistent_key(source, key), source, seconds)
        if geometry is None:
            return None
        if not isinstance(geometry, FocalSelectorGeometry) or geometry.source_index != source:
            raise ValueError('cached focal observation has a different source authority')
        if geometry.face_bbox_xywh is None and geometry.person is None:
            return None
        # One exact native frame can be requested at several equivalent times;
        # retime the query only, never its independently measured anchor times.
        track = geometry.body_track
        if track is not None:
            track = (replace(track, source_seconds=seconds)
                if isinstance(track, FocalBodyGeometryCertificate) else
                replace(track, source_seconds=seconds,
                    observation=replace(track.observation, source_seconds=seconds)))
        geometry = replace(geometry, source_seconds=seconds, body_track=track)
        identity = sampler.frame_identity(seconds)
        raw = visual._descriptor_cache.get((source, identity, False))
        if raw is None:
            raw = visual._descriptor_cache.get((source, identity, True))
        if raw is None:
            raw = sampler.feature_descriptor_at(seconds, include_subject_geometry=True,
                include_formation_pose=True, cache_only=True)
        if raw is None:
            raw = sampler.feature_descriptor_at(seconds, include_subject_geometry=True,
                include_formation_pose=False, include_pose=False, cache_only=True)
        return CachedMemberObservation(raw, geometry) if raw is not None else None
    visual.cached_focal_observation_at = cached


def seed_cached_boundary_samples(store, provider):
    """Populate only present, content-eligible exact slots in an empty store."""
    from .direct_member_visual import _boundary_sample_with_focal_geometry
    callback = getattr(provider, 'cached_focal_observation_at', None)
    if not callable(callback):
        return 0
    count = 0
    for plan in store.plans:
        for source in range(store.source_count):
            samples = {}
            for slot, eligible in enumerate(plan.eligibility_by_source[source]):
                if not eligible:
                    continue
                seconds = plan.local_times_by_source[source][slot]
                value = callback(source, seconds)
                if value is None:
                    continue
                if not isinstance(value, CachedMemberObservation):
                    raise TypeError('cached member provider must return typed observed evidence')
                if (value.geometry.source_index != source
                        or not math.isclose(value.geometry.source_seconds, seconds, abs_tol=1e-9, rel_tol=0)):
                    raise ValueError('cached member observation has a different source or clock')
                samples[slot] = _boundary_sample_with_focal_geometry(
                    master_seconds=plan.master_times[slot], source_local_seconds=seconds,
                    source_index=source, descriptor=value.descriptor,
                    provider=SimpleNamespace(focal_geometry_at=lambda *a, geometry=value.geometry:geometry))
            store.publish_cached_samples(plan.offset_frames, source, samples)
            count += len(samples)
    return count
