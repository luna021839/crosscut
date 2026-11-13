"""Explicit eligible-local -> request-source clock binding for GROUP proposals.

This is caller-supplied index/clock data, not audio alignment or file-identity
proof. Excluded request sources receive no invented zero offsets or clocks.
Media identity and decoded validation remain the owning worker's obligations.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
from pathlib import Path

from .render_frame_schedule import VerifiedRenderFrameClock
from .timewarp import TimeWarpMap


@dataclass(frozen=True, slots=True)
class GroupSourceDomain:
    request_sources: tuple[Path, ...]
    eligible_indices: tuple[int, ...]
    audio_offsets: tuple[float, ...]
    source_timewarps: tuple[TimeWarpMap | None, ...]
    scene_cut_times: tuple[tuple[float, ...], ...]
    render_frame_clocks: tuple[VerifiedRenderFrameClock, ...]

    def __post_init__(self):
        if (not isinstance(self.request_sources,tuple) or not self.request_sources
                or any(not isinstance(path,Path) for path in self.request_sources)
                or not isinstance(self.eligible_indices,tuple) or not self.eligible_indices
                or any(type(i) is not int or not 0 <= i < len(self.request_sources)
                    for i in self.eligible_indices)
                or len(set(self.eligible_indices)) != len(self.eligible_indices)):
            raise ValueError('GROUP source domain needs unique valid eligible request indices')
        arrays=(self.audio_offsets,self.source_timewarps,self.scene_cut_times,self.render_frame_clocks)
        if any(not isinstance(values,tuple) or len(values)!=len(self.eligible_indices) for values in arrays):
            raise ValueError('GROUP source domain clock arrays must share the eligible-local order')
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in self.audio_offsets)
                or any(w is not None and not isinstance(w,TimeWarpMap) for w in self.source_timewarps)
                or any(not isinstance(cuts,tuple) or any(type(t) not in (int,float)
                    or not math.isfinite(t) or t<0 for t in cuts) for cuts in self.scene_cut_times)
                or any(not isinstance(clock,VerifiedRenderFrameClock) for clock in self.render_frame_clocks)):
            raise ValueError('GROUP source domain needs finite clocks/cuts and verified native frame clocks')

    @property
    def is_identity(self):
        return self.eligible_indices==tuple(range(len(self.request_sources)))

    @property
    def eligible_sources(self):
        return tuple(self.request_sources[i] for i in self.eligible_indices)

    @property
    def clocks_by_request(self):
        return dict(zip(self.eligible_indices,self.render_frame_clocks,strict=True))

    def request_index(self, eligible_index):
        if type(eligible_index) is not int or not 0 <= eligible_index < len(self.eligible_indices):
            raise ValueError('source index is outside the eligible-local GROUP domain')
        return self.eligible_indices[eligible_index]

    def eligible_index(self, request_index):
        if type(request_index) is not int or request_index not in self.eligible_indices:
            raise ValueError('request source is excluded from the GROUP source domain')
        return self.eligible_indices.index(request_index)

    def cuts_for_request(self, request_index):
        return self.scene_cut_times[self.eligible_index(request_index)]

    def to_dict(self):
        return dict(contract='eligible_local_group_source_clock_domain_v1',
            eligible_request_indices=list(self.eligible_indices),
            eligible_sources=[str(path) for path in self.eligible_sources],
            audio_offsets=list(self.audio_offsets),
            source_timewarps=[warp.to_dict() if warp is not None else None for warp in self.source_timewarps],
            scene_cut_times=[list(cuts) for cuts in self.scene_cut_times],
            render_frame_clocks=[clock.to_dict() for clock in self.render_frame_clocks],
            source_reference_alignment_authority='caller_supplied_not_established_by_index_mapping',
            file_identity_verified=False)


def bind_group_source_domain(*,plan,source_domain=None,audio_offsets=None,
        source_timewarps=None,scene_cut_times=None,source_clocks=None):
    """Preserve the legacy identity API, or validate one explicit complete DTO.

    Redundant arrays are accepted only if exactly equal, never silently ignored.
    Native clock keys are request-domain keys, not eligible-local keys.
    """
    if source_domain is None:
        if any(value is None for value in (audio_offsets,source_timewarps,scene_cut_times,source_clocks)):
            raise ValueError('GROUP source proposal requires one complete source clock domain')
        if not isinstance(source_clocks,Mapping):
            raise ValueError('native frame clocks must be keyed by request source index')
        count=len(plan.sources)
        source_domain=GroupSourceDomain(tuple(plan.sources),tuple(range(count)),
            tuple(audio_offsets),tuple(source_timewarps),tuple(tuple(cuts) for cuts in scene_cut_times),
            tuple(source_clocks.get(i) for i in range(count)))
    if not isinstance(source_domain,GroupSourceDomain) or tuple(plan.sources)!=source_domain.request_sources:
        raise ValueError('GROUP source domain does not bind the RenderPlan source files/order')
    for supplied,expected in ((audio_offsets,source_domain.audio_offsets),
            (source_timewarps,source_domain.source_timewarps),(scene_cut_times,source_domain.scene_cut_times)):
        if supplied is not None and tuple(supplied)!=expected:
            raise ValueError('redundant GROUP clock arrays differ from the source domain')
    if source_clocks is not None and (not isinstance(source_clocks,Mapping)
            or dict(source_clocks)!=source_domain.clocks_by_request):
        raise ValueError('native frame clock keys differ from the request-source domain')
    return source_domain
