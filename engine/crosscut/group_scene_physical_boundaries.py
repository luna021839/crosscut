"""Original physical events, derivative relations and diversity are separate.

A preserved native camera cut in one file may support CAMERA-only editing.
It is never a distinct-source proof. This numerical policy neither detects a
camera cut nor certifies pixels/motion; the externally bound native cut times,
fresh native sensors and existing editorial gates remain required.
"""
from collections import defaultdict
from hashlib import sha256
import json

from .group_actual_shots import visible_group_cut_inventory
from .group_source_domain import GroupSourceDomain
from .render import RenderPlan
from .render_boundary_clock import native_frame_clocks_for_output
from .source_derivative_diversity import DERIVATIVE_DISTINCT

CONTRACT = 'external_original_physical_boundary_policy_v1'


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _sampler(row):
    # Camera/canonical cells are not clock origins. Preserve the renderer's
    # actual request seconds as well as native IDs (rounding IDs alone is weak).
    return [row.source_index,row.native_frame_index,row.source_request_seconds,row.terminal_clone]


class PreparedPhysicalBoundaryPolicy:
    """Prepare both full numeric schedules once, not once per incident edge."""
    def __init__(self, *, original, observed, domain, derivative_binding):
        if (not isinstance(original,RenderPlan) or not isinstance(observed,RenderPlan)
                or not isinstance(domain,GroupSourceDomain)
                or original.sources != domain.request_sources or observed.sources != original.sources
                or original.output_frame_count != observed.output_frame_count
                or original.fps != observed.fps or not 1 <= original.output_frame_count <= 36000):
            raise ValueError('physical boundary policy requires one complete external output/source clock')
        from .group_scene_source_owner import _validated_derivative_source_count
        if derivative_binding is not None and (
                _validated_derivative_source_count(derivative_binding) != len(domain.eligible_indices)
                or _digest(derivative_binding.get('eligible_request_indices')) != _digest(list(domain.eligible_indices))):
            raise ValueError('physical boundary derivative binding uses another eligible source domain')
        self.domain=domain
        self.binding=derivative_binding
        self.events=[]
        self.samples=[]
        for plan in (original,observed):
            events=defaultdict(list)
            for event in visible_group_cut_inventory(plan,domain):
                events[event.first_incoming_output_frame].append(event.to_dict())
            self.events.append(events)
            self.samples.append(native_frame_clocks_for_output(plan,domain.clocks_by_request,
                range(plan.output_frame_count)))

    def classify(self, left, right):
        tick=right.first_output_frame
        if (type(tick) is not int or left.end_output_frame != tick
                or not 0 < tick < len(self.samples[1])):
            raise ValueError('physical boundary cannot manufacture a cut across a gap or output end')
        pair=[]
        for state in (left,right):
            first,end=state.first_output_frame,state.end_output_frame
            local=state.eligible_source_index
            if (type(first) is not int or type(end) is not int or type(local) is not int
                    or not 0 <= first < end <= len(self.samples[1])
                    or not 0 <= local < len(self.domain.eligible_indices)):
                raise ValueError('physical boundary lost the exact observed source/span')
            sampled=[self.samples[1][t] for t in range(first,end)]
            if (tuple(r.native_frame_index for r in sampled) != state.native_frame_indices
                    or any(r.source_index != self.domain.request_index(local) or r.terminal_clone for r in sampled)):
                raise ValueError('physical boundary state differs from its actual full renderer sampler')
            pair.append(local)
        from .group_scene_source_owner import relation
        related=relation(self.binding,*pair)
        original_events=self.events[0].get(tick,[])
        observed_events=self.events[1].get(tick,[])
        first,end=left.first_output_frame,right.end_output_frame
        original_sampler=[_sampler(self.samples[0][t]) for t in range(first,end)]
        observed_sampler=[_sampler(self.samples[1][t]) for t in range(first,end)]
        unchanged=original_sampler == observed_sampler
        native=bool(observed_events) and all(e['kind']=='native_camera_cut' for e in observed_events)
        preserved_native=native and original_events == observed_events and unchanged
        switch=(len(observed_events)==1 and observed_events[0]['kind']=='source_switch'
            and pair[0] != pair[1])
        admitted=preserved_native or switch and related == DERIVATIVE_DISTINCT
        original_pair=([self.samples[0][tick-1].source_index,self.samples[0][tick].source_index])
        observed_pair=[self.samples[1][tick-1].source_index,self.samples[1][tick].source_index]
        return dict(contract=CONTRACT,output_tick=tick,eligible_source_pair=pair,
            original_events=original_events,observed_events=observed_events,
            source_relation=related,preserved_original_native_cut=preserved_native,
            incident_source_sampler_preserved=unchanged,
            original_incident_sampler_sha256=_digest(original_sampler),
            observed_incident_sampler_sha256=_digest(observed_sampler),
            distinct_source_pair_confirmed=switch and related == DERIVATIVE_DISTINCT,
            new_distinct_source_credit=switch and related == DERIVATIVE_DISTINCT and original_pair != observed_pair,
            admitted=bool(admitted),native_motion_observed=False,quality_approved=False)

    def attach(self, row, left, right):
        policy=self.classify(left,right)
        row['source_relation']=policy['source_relation']
        row['physical_boundary_policy']=policy
        if not policy['admitted']:
            # Do not erase already measured geometry or replace UNKNOWN with 0.
            row['available']=False
            row['physical_boundary_rejection']='not a preserved native CAMERA boundary or verified distinct source-switch'
        return row

    def require(self, pairs):
        pairs=tuple(pairs)
        if not 1 <= len(pairs) <= 12:
            raise ValueError('physical boundary policy needs the complete bounded incident inventory')
        rows=[self.classify(a,b) for a,b in pairs]
        if any(not row['admitted'] for row in rows):
            raise ValueError('global physical cut is not preserved native CAMERA-only or a verified distinct source-switch')
        return rows


def validate_dense_physical_boundaries(dense):
    """Freshly derive receipt policy; a self-rehashed flag cannot admit a cut."""
    from .group_scene_execution_scope import dense_execution_path,incident_pairs
    from .group_scene_edit_operations import numeric_replay_domain
    path=dense_execution_path(dense)
    domain=numeric_replay_domain(path.original_plan,dense['source_domain'])
    policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
        domain=domain,derivative_binding=dense.get('source_derivative_binding'))
    rows=policy.require(incident_pairs(path))
    if (_digest(dense.get('physical_boundary_policies')) != _digest(rows)
            or len(dense.get('incident_projected_observations',())) != len(rows)
            or _digest(dense.get('derivative_window_source_indices')) != _digest([s.eligible_source_index for s in path])
            or any(_digest(edge.get('physical_boundary_policy')) != _digest(row) or edge.get('source_relation') != row['source_relation']
                for edge,row in zip(dense['incident_projected_observations'],rows,strict=True))):
        raise ValueError('global dense physical boundary/event/sampler/diversity policy changed')
    return rows
