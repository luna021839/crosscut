"""One-sided cut exclusions from COMPLETE exact focal endpoint observations.

An explicitly unresolved focal subject at every allowed offset makes that
source unusable on that side of this cross-source cut, regardless of the other
endpoint. This is not a paired mismatch measurement or a whole-source ban.
Missing observations, one bad offset, missing pose alone and valid body tracks
cannot supply this certificate. Same-source holds are outside its scope.
"""
from dataclasses import dataclass
import math

from .focal_selector_geometry import FocalSelectorGeometry


@dataclass(frozen=True, slots=True)
class MemberEndpointExclusion:
    boundary_index: int
    source_index: int
    role: str
    offsets: tuple[int, ...]
    master_seconds: tuple[float, ...]
    source_seconds: tuple[float, ...]

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (self.boundary_index, self.source_index)):
            raise ValueError('endpoint exclusion requires nonnegative source/boundary indices')
        if self.role not in ('outgoing', 'incoming'):
            raise ValueError('endpoint exclusion has no cut side')
        if (not self.offsets or 0 not in self.offsets or len(set(self.offsets)) != len(self.offsets)
                or any(isinstance(v, bool) or not isinstance(v, int) for v in self.offsets)
                or len(self.master_seconds) != len(self.offsets) or len(self.source_seconds) != len(self.offsets)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (*self.master_seconds, *self.source_seconds))):
            raise ValueError('endpoint exclusion must bind every exact offset and clock')

    def excludes(self, demand):
        return (demand.boundary_index == self.boundary_index
            and demand.outgoing_source_index != demand.incoming_source_index
            and self.source_index == (demand.outgoing_source_index
                if self.role == 'outgoing' else demand.incoming_source_index))


def observed_unresolved_endpoints(store, boundary_index):
    """Read only already-published exact samples; never decode/prepare/infer."""
    available = store.observed_sample_demands
    certificates = []
    for source in sorted({source for _offset, source, _slot in available}):
        for role, slot in (('outgoing', 1), ('incoming', 2)):
            if not all((plan.offset_frames, source, slot) in available for plan in store.plans):
                continue
            samples = tuple(store.observed_sample(plan.offset_frames, source, slot) for plan in store.plans)
            # None means unobserved/legacy authority, not an explicit negative.
            if not all(isinstance(sample.focal_geometry, FocalSelectorGeometry) for sample in samples):
                continue
            for sample, plan in zip(samples, store.plans, strict=True):
                geometry = sample.focal_geometry
                if (geometry.source_index != source
                        or not math.isclose(geometry.source_seconds, sample.source_local_seconds, rel_tol=0., abs_tol=1e-9)
                        or sample.source_local_seconds != plan.local_times_by_source[source][slot]
                        or sample.master_seconds != plan.master_times[slot]):
                    raise ValueError('endpoint exclusion source/clock evidence changed')
            if all(sample.focal_geometry.face_bbox_xywh is None
                   and sample.focal_geometry.body_track is None for sample in samples):
                certificates.append(MemberEndpointExclusion(boundary_index, source, role,
                    tuple(plan.offset_frames for plan in store.plans),
                    tuple(sample.master_seconds for sample in samples),
                    tuple(sample.source_local_seconds for sample in samples)))
    return tuple(certificates)
