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
        self.maximum_offset_frames=maximum_offset_frames
        self.lattice=tuple(tuple(row) for row in lattice)
        if not self.lattice or any(not row for row in self.lattice):
            raise ValueError('source-switch lattice must contain nonempty rows')
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
        connected, blocking_segment_index = self._connected_capacity(domain)
        if not self.feasible(self.initial_counts,0) or not connected:
            raise NoCompleteTimelinePath('search domain cannot supply the required section source switches',
                reason_code='section_source_switch_capacity_insufficient',
                segment_index=blocking_segment_index,
                rejection_counts={'section_source_switch_capacity_insufficient':1,
                                  'connected_source_path_required':1,
                                  'proof_materialization_call_count':0})

    def _potential_increment(self, boundary, left, right):
        if not self.different(left.source_id,right.source_id):
            return self.initial_counts
        time=right.start_seconds
        radius=(self.maximum_offset_frames+.5)/self.fps
        return tuple(int(time+radius>w.start_seconds+1e-9
                         and time-radius<w.end_seconds-1e-9)
                     for w in self.requirements)

    def _connected_capacity(self,domain):
        """Prove one connected source/candidate path can meet every count.

        This remains an optimistic descriptor-domain check: a potential switch
        may use any allowed frame offset.  Unlike the previous position count,
        it cannot combine mutually disconnected A-to-B edges from unrelated
        paths into a fictional capacity proof.
        """
        if not self.requirements:
            return True,None
        allowed=(None if domain is None else {
            (key.boundary_index,key.outgoing,key.incoming) for key in domain
        })
        states={(candidate,self.initial_counts) for candidate in self.lattice[0]}
        potential_boundaries=[[] for _ in self.requirements]
        for boundary,(left_row,right_row) in enumerate(zip(self.lattice,self.lattice[1:])):
            next_states=set()
            for left,counts in states:
                for right in right_row:
                    if allowed is not None and (boundary,left,right) not in allowed:
                        continue
                    increments=self._potential_increment(boundary,left,right)
                    next_counts=tuple(min(requirement.minimum_count,count+increment)
                        for requirement,count,increment in zip(self.requirements,counts,increments))
                    next_states.add((right,next_counts))
            for index,requirement in enumerate(self.requirements):
                if any(self._potential_increment(boundary,left,right)[index]
                       for left in left_row for right in right_row
                       if allowed is None or (boundary,left,right) in allowed):
                    potential_boundaries[index].append(boundary)
            if not next_states:
                return False,right_row[0].segment_index
            states=next_states
        if any(all(count>=requirement.minimum_count
                   for count,requirement in zip(counts,self.requirements))
               for _candidate,counts in states):
            return True,None
        best=max((counts for _candidate,counts in states),default=self.initial_counts,
                 key=lambda counts:(sum(counts),counts))
        for index,(count,requirement) in enumerate(zip(best,self.requirements)):
            if count>=requirement.minimum_count:
                continue
            boundaries=potential_boundaries[index]
            target=min(len(boundaries)-1,requirement.minimum_count-1)
            if boundaries:
                return False,self.lattice[boundaries[target]+1][0].segment_index
        return False,self.lattice[-1][0].segment_index

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
