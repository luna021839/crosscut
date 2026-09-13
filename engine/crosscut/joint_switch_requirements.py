"""Exact section switch counts carried by the joint selector, not a QA waiver."""
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .timeline import NoCompleteTimelinePath


@dataclass(frozen=True, slots=True)
class SourceSwitchRequirement:
    start_seconds: float
    end_seconds: float
    minimum_count: int

    def __post_init__(self):
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in (self.start_seconds,self.end_seconds))
                or not 0 <= self.start_seconds < self.end_seconds
                or type(self.minimum_count) is not int or self.minimum_count < 1):
            raise ValueError('source-switch requirement needs a finite window and positive count')

    def contains(self, seconds):
        # A cut exactly on a section's start has no preceding run overlapping
        # that section, so the decoded montage reviewer does not count it.
        return self.start_seconds + 1e-9 < seconds < self.end_seconds - 1e-9


class SourceSwitchCounter:
    def __init__(self, requirements: Sequence[SourceSwitchRequirement], lattice,
                 *, fps: float, maximum_offset_frames: int, domain=None,
                 source_families: Mapping[str,str] | None = None):
        self.requirements=tuple(requirements)
        if len(self.requirements)>32 or any(not isinstance(w,SourceSwitchRequirement) for w in self.requirements):
            raise ValueError('source-switch requirements must be at most 32 typed windows')
        if any(a.end_seconds>b.start_seconds for a,b in zip(self.requirements,self.requirements[1:])):
            raise ValueError('source-switch windows must be ordered and non-overlapping')
        self.fps=fps;self.origin=lattice[0][0].start_seconds
        self.families=dict(source_families or {})
        ids={c.source_id for row in lattice for c in row}
        if self.families and (not ids.issubset(self.families) or any(not isinstance(v,str) or not v for v in self.families.values())):
            raise ValueError('source families must bind every current source exactly')
        self.initial_counts=(0,)*len(self.requirements)
        # Count potential boundary positions, not source pairs. This is only
        # an optimistic reachability bound; every actual edge still needs proof.
        allowed_positions=(None if domain is None else {k.boundary_index for k in domain
            if self.different(k.outgoing.source_id,k.incoming.source_id)})
        potential=[]
        for boundary,(left,right) in enumerate(zip(lattice,lattice[1:])):
            has_pair=(boundary in allowed_positions if allowed_positions is not None else
                      any(self.different(a.source_id,b.source_id) for a in left for b in right))
            time=right[0].start_seconds
            radius=(maximum_offset_frames+.5)/fps
            potential.append(tuple(int(has_pair and time+radius>w.start_seconds+1e-9
                                       and time-radius<w.end_seconds-1e-9) for w in self.requirements))
        remaining=[self.initial_counts]*(len(potential)+1)
        for i in range(len(potential)-1,-1,-1):
            remaining[i]=tuple(a+b for a,b in zip(remaining[i+1],potential[i]))
        self.remaining=tuple(remaining)
        if not self.feasible(self.initial_counts,0):
            raise NoCompleteTimelinePath('search domain cannot supply the required section source switches',
                reason_code='section_source_switch_capacity_insufficient',
                rejection_counts={'section_source_switch_capacity_insufficient':1,
                                  'proof_materialization_call_count':0})

    def different(self,left,right):
        return self.families.get(left,left)!=self.families.get(right,right)

    def advance(self,counts,*,left,right,nominal_boundary_seconds,frame_offset):
        if not self.different(left,right):return counts
        tick=round((nominal_boundary_seconds-self.origin)*self.fps+frame_offset)
        seconds=self.origin+tick/self.fps
        return tuple(min(w.minimum_count,count+int(w.contains(seconds)))
                     for w,count in zip(self.requirements,counts))

    def feasible(self,counts,next_boundary):
        return all(count+remaining>=w.minimum_count for w,count,remaining in
                   zip(self.requirements,counts,self.remaining[next_boundary]))
