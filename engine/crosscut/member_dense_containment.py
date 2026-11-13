"""Interior-only crop support, never focal identity, pose or cut authority.

A short detector/identity gap may still contain an actual detected body which
two independently resolved source-local neighbors uniquely select. Reuse the
existing bracket certificate for camera containment; never feed it back into
the selector, the frozen target bank or cross-source endpoint matching.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .focal_bracket_containment import (
    FocalBracketContainmentSupport, measure_bracketed_body_containment,
)

CONTAINMENT_BASIS = 'bracketed_observed_body_containment_only'


@dataclass(frozen=True, slots=True)
class MemberDenseContainmentSample:
    support: FocalBracketContainmentSupport
    canonical_master_seconds: float
    subject_confidence: float

    def __post_init__(self):
        if (not isinstance(self.support, FocalBracketContainmentSupport)
                or not math.isfinite(self.canonical_master_seconds) or self.canonical_master_seconds < 0
                or not math.isfinite(self.subject_confidence) or not .30 <= self.subject_confidence <= 1.):
            raise ValueError('dense interior containment needs an observed source-bound certificate')

    @property
    def bbox_xywh(self):
        return self.support.bbox_xywh

    @property
    def subject_bbox_xywh(self):
        return self.support.bbox_xywh

    @property
    def face_count(self):
        return 0

    @property
    def detector_backend(self):
        return 'none'

    @property
    def subject_geometry_basis(self):
        return CONTAINMENT_BASIS

    @property
    def alignment_geometry_basis(self):
        return CONTAINMENT_BASIS


def dense_camera_sample(provider, source_index, master_seconds, *, interior):
    """Strict evidence wins; containment cannot fill either segment endpoint."""
    sample = provider(source_index, master_seconds)
    if sample is not None or not interior:
        return sample
    containment = getattr(provider, 'interior_containment_at', None)
    if not callable(containment):
        return None
    supported = containment(source_index, master_seconds)
    if supported is None:
        return None
    if (not isinstance(supported, MemberDenseContainmentSample)
            or supported.support.source_index != source_index
            or not math.isclose(supported.canonical_master_seconds, master_seconds, abs_tol=1e-9, rel_tol=0)):
        raise ValueError('dense containment sample disagrees with requested source or clock')
    return supported


def selected_source_native_cuts(selection, source_count):
    """Unknown scene evidence disables support rather than assuming no cuts."""
    if not getattr(selection, 'native_scene_cut_evidence_available', False):
        return None
    cuts = [set() for _ in range(source_count)]
    observed = set()
    for span in getattr(selection, 'span_validity', ()):
        for source in span.sources:
            index = source.eligible_source_index
            if type(index) is not int or not 0 <= index < source_count:
                raise ValueError('dense containment scene evidence source is invalid')
            observed.add(index)
            # Even an upstream-suppressed transient cut cannot provide an
            # independent temporal identity bridge across its timestamp.
            values = (*source.native_scene_cut_source_seconds,
                      *source.suppressed_portrait_fancam_scene_cut_source_seconds)
            if any(not math.isfinite(t) or t < 0 for t in values):
                raise ValueError('dense containment scene evidence clock is invalid')
            cuts[index].update(values)
    if not set(selection.source_indices) <= observed:
        return None
    return tuple(tuple(sorted(values)) for values in cuts)


def attach_member_dense_containment(provider, *, focal_provider, source_local_time,
                                    native_scene_cut_times, fps):
    """Add one bounded interior capability without replacing strict sample()."""
    if native_scene_cut_times is None or not hasattr(focal_provider, 'bodies'):
        return
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('dense containment needs an output frame clock')
    native = tuple(tuple(float(t) for t in values) for values in native_scene_cut_times)
    if any(not math.isfinite(t) or t < 0 for values in native for t in values):
        raise ValueError('dense containment source cuts must be finite')
    cache, audit = {}, {}

    def containment_at(source, master):
        if type(source) is not int or not 0 <= source < len(native):
            raise ValueError('dense containment source is outside the scene evidence')
        key = (source, master)
        if key in cache:
            return cache[key]
        cache[key] = None
        seconds = source_local_time(source, master)
        # The failed strict query must already have measured this exact frame.
        # Do not launch an anonymous person detector or synthesize a query box.
        sampler = focal_provider.faces._samplers[source]
        frame_key = focal_provider.faces._identity_key(source, sampler, seconds)
        observed = focal_provider.bodies._frame_cache.get(frame_key)
        if observed is None or observed.source_index != source or not observed.observations:
            return None
        anchors = [None, None]
        for distance in (1, 2, 3, 4):
            for side, direction in enumerate((-1, 1)):
                if anchors[side] is not None:
                    continue
                anchor_seconds = source_local_time(source, master+direction*distance/fps)
                if anchor_seconds < 0 or abs(anchor_seconds-seconds) > .20:
                    continue
                anchor = focal_provider.sample(source, anchor_seconds)
                if anchor is not None and anchor.subject_bbox_xywh is not None:
                    anchors[side] = (source, anchor_seconds, anchor)
            if all(anchors):
                break
        if not all(anchors):
            return None
        support = measure_bracketed_body_containment(source_index=source, source_seconds=seconds,
            observed_query_boxes=tuple(o.person.bbox_xywh for o in observed.observations),
            anchors=tuple(anchors), native_scene_cut_seconds=native[source])
        if support is None:
            return None
        confidence = observed.observations[support.selected_index].person.detection_confidence
        if confidence < .30:
            return None
        value = MemberDenseContainmentSample(support, master, confidence)
        cache[key] = value
        audit[key] = dict(canonical_master_seconds=master, **asdict(support),
            evidence_basis=CONTAINMENT_BASIS, face_detection_claimed=False,
            focal_identity_or_pose_authority=False, cross_source_endpoint_authority=False,
            scope='camera_trial_interior_only_not_final_quality_approval')
        return value

    provider.interior_containment_at = containment_at
    provider.interior_containment_audit = lambda: tuple(audit[key] for key in sorted(audit))
