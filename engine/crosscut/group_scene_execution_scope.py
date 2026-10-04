"""Explicit multi-interval scene execution, not positional L/B/C/R ownership.

The legacy joint state is a numeric camera/observation record only: its slot
does not decide global ownership. Stable original unit IDs decide mutable
shots and actual incident pairs. Uniform whole-shot roles and the unchanged
ordinary GROUP grid are the first supported execution slice; mixed epochs,
subcell mutation, learned authority and request-wide leases remain separate.
"""
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json

from .group_joint_native_preparation import _plan_identity
from .group_joint_source_camera import GroupOutputRole, joint_state_from_render_plan
from .group_scene_epoch_program import validate_original_role_epochs
from .group_scene_owned_program import (OwnedSceneProgram, OwnedSceneUnit,
    OwnedSceneIntervals, NumericOwnedSceneUnion, own_scene_intervals,
    validate_owned_scene_program, camera_unit_from_plan, assemble_owned_scene_union)

CONTRACT = 'explicit_global_scene_execution_scope_v1'
MAX_MUTABLE_UNITS = 6
MAX_MUTABLE_FRAMES = 540
MAX_REFRESHED_CELLS = 18


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _unit_indices(program, allocation):
    mutable = {i for i,u in enumerate(program.units) if u.unit_id in allocation.mutable_unit_ids}
    return tuple(sorted(mutable | {j for i in mutable for j in (i-1,i+1) if 0 <= j < len(program.units)}))


def _uniform_role(unit):
    epochs = unit.role_program['epochs']
    roles = {epoch['role'] for epoch in epochs}
    if len(roles) != 1 or None in roles:
        raise ValueError('global execution needs mixed-role observation-epoch support; duties are not weakened')
    return GroupOutputRole(next(iter(roles)))


@dataclass(frozen=True, slots=True)
class GlobalSceneExecutionPath(Sequence):
    program: OwnedSceneProgram
    allocation: OwnedSceneIntervals
    original_plan: object
    numeric_plan: object
    states: tuple
    source_domain_json: str

    def __post_init__(self):
        if (not isinstance(self.program, OwnedSceneProgram)
                or not isinstance(self.allocation, OwnedSceneIntervals)
                or self.allocation.program != self.program
                or self.allocation.maximum_mutable_units > MAX_MUTABLE_UNITS
                or self.allocation.maximum_mutable_frames > MAX_MUTABLE_FRAMES
                or _plan_identity(self.original_plan) != self.program.original_plan_identity
                or not isinstance(self.states, tuple) or not isinstance(self.source_domain_json,str)):
            raise ValueError('global execution needs one external original and a finite explicit allocation')
        from .group_performer_family_owner import digest
        if digest(json.loads(self.source_domain_json))!=self.program.source_domain_sha256:
            raise ValueError('global execution changed its original complete alignment/domain preimage')
        canonical = own_scene_intervals(self.program, mutable_unit_ids=self.allocation.mutable_unit_ids,
            maximum_mutable_units=self.allocation.maximum_mutable_units,
            maximum_mutable_frames=self.allocation.maximum_mutable_frames)
        if canonical != self.allocation or len(self.states) != len(self.scoped_units):
            raise ValueError('global execution changed its original canonical interval inventory')
        for unit,state in zip(self.scoped_units,self.states,strict=True):
            validate_original_role_epochs(unit.role_program,original_plan=self.original_plan)
            if ((state.first_output_frame,state.end_output_frame)!=(unit.first_output_frame,unit.end_output_frame)
                    or state.role != _uniform_role(unit) or state.slot != 1):
                raise ValueError('global observation record changed its original unit/role; slots do not own edits')

    def __len__(self): return len(self.states)
    def __getitem__(self,index): return self.states[index]

    @property
    def scoped_units(self):
        return tuple(self.program.units[i] for i in _unit_indices(self.program,self.allocation))

    @property
    def mutable_indices(self):
        return tuple(i for i,u in enumerate(self.scoped_units) if u.unit_id in self.allocation.mutable_unit_ids)

    @property
    def mutable_intervals(self): return self.allocation.mutable_output_intervals

    @property
    def incident_pair_indices(self):
        mutable=set(self.mutable_indices)
        return tuple((i,i+1) for i,(a,b) in enumerate(zip(self.scoped_units,self.scoped_units[1:]))
            if a.end_output_frame==b.first_output_frame and (i in mutable or i+1 in mutable))

    def to_dict(self):
        return dict(contract=CONTRACT,original_owned_program=self.program.to_dict(),
            source_domain=json.loads(self.source_domain_json),
            allocation=self.allocation.to_dict(),scoped_unit_ids=[u.unit_id for u in self.scoped_units],
            unit_intervals=[[u.first_output_frame,u.end_output_frame] for u in self.scoped_units],
            mutable_state_indices=list(self.mutable_indices),
            incident_state_pair_indices=[list(pair) for pair in self.incident_pair_indices],
            endpoint_scopes=['whole']*len(self.states),
            maximum_refreshed_cells=MAX_REFRESHED_CELLS,
            positional_fixed_endpoints_assumed=False,
            whole_camera_restarts_at_role_epochs=False,
            original_native_duties_frozen_by_this_numeric_scope=False,
            whole_request_phase_reserves_implemented=False,quality_approved=False)

    def with_states(self,states): return replace(self,states=tuple(states))

    def with_plan(self,plan,domain):
        return replace(self,numeric_plan=plan,states=tuple(joint_state_from_render_plan(
            plan=plan,domain=domain,slot=1,first_output_frame=unit.first_output_frame,
            end_output_frame=unit.end_output_frame,role=_uniform_role(unit),
            editorial_contract_sha256=state.editorial_contract_sha256,raw_shot_family=state.raw_shot_family)
            for unit,state in zip(self.scoped_units,self.states,strict=True)))

    def validate_external(self,context,domain):
        if context.incumbent.render_plan != self.original_plan:
            raise ValueError('global execution differs from the external incumbent')
        if json.loads(self.source_domain_json)!=domain.to_dict():
            raise ValueError('global execution differs from the external original alignment domain')
        validate_owned_scene_program(self.program.to_dict(),context,domain)
        cameras=tuple(camera_unit_from_plan(program=self.program,unit_id=unit_id,
            plan=self.numeric_plan,domain=domain) for unit_id in self.allocation.mutable_unit_ids)
        assembled=assemble_owned_scene_union(context=context,domain=domain,
            allocation=self.allocation,cameras=cameras)
        if assembled.plan != self.numeric_plan:
            raise ValueError('global execution differs from the complete external numeric camera union')

    def assemble(self,*,plan,domain,path):
        if path is not self or plan != self.original_plan:
            raise ValueError('global numeric materialization lost the external original')
        from .group_scene_edit_operations import derive_scene_edit_operations
        actual=domain.to_dict();retained=json.loads(self.source_domain_json)
        if any(actual[key]!=retained[key] for key in ('eligible_request_indices','eligible_sources',
                'scene_cut_times','render_frame_clocks')):
            raise ValueError('global numeric materialization changed its materialized sampler domain')
        derive_scene_edit_operations(original=plan,proposed=self.numeric_plan,domain=domain,
            mutable_intervals=self.mutable_intervals)
        rebound=self.with_plan(self.numeric_plan,domain)
        if rebound.states != self.states:
            raise ValueError('global numeric states differ from the actual whole renderer/source clock')
        return self.numeric_plan,tuple(i for i,(a,b) in enumerate(zip(plan.segments,
            self.numeric_plan.segments,strict=True)) if a != b)


def execution_path_from_union(context,domain,union):
    if not isinstance(union,NumericOwnedSceneUnion):
        raise ValueError('global execution requires an atomic numeric union, not local completed bundles')
    states=tuple(joint_state_from_render_plan(plan=union.plan,domain=domain,slot=1,
        first_output_frame=union.allocation.program.units[i].first_output_frame,
        end_output_frame=union.allocation.program.units[i].end_output_frame,
        role=_uniform_role(union.allocation.program.units[i]))
        for i in _unit_indices(union.allocation.program,union.allocation))
    result=GlobalSceneExecutionPath(union.allocation.program,union.allocation,
        context.incumbent.render_plan,union.plan,states,_json(domain.to_dict()))
    result.validate_external(context,domain)
    return result


def execution_path_from_payload(*,original,proposed,domain,payload,rows=None):
    """Reconstruct numeric scope; parent must also rederive external role intent."""
    try:
        raw=payload['original_owned_program']
        program=OwnedSceneProgram(raw['original_plan_identity'],_json(raw['original_plan_header']),
            raw['source_domain_sha256'],tuple(raw['global_cell_frame_counts']),
            tuple(OwnedSceneUnit(u['unit_id'],u['first_output_frame'],u['end_output_frame'],
                _json(u['original_role_program'])) for u in raw['units']))
        allocation=own_scene_intervals(program,
            mutable_unit_ids=tuple(payload['allocation']['mutable_unit_ids']),
            maximum_mutable_units=payload['allocation']['maximum_mutable_units'],
            maximum_mutable_frames=payload['allocation']['maximum_mutable_frames'])
        units=tuple(program.units[i] for i in _unit_indices(program,allocation))
        if rows is not None and (not isinstance(rows,list) or len(rows)!=len(units)):
            raise ValueError('global selected state inventory has another original scope')
        states=tuple(joint_state_from_render_plan(plan=proposed,domain=domain,slot=1,
            first_output_frame=u.first_output_frame,end_output_frame=u.end_output_frame,role=_uniform_role(u),
            editorial_contract_sha256=None if rows is None else rows[i]['key_preimage'].get('editorial_contract_sha256'),
            raw_shot_family=None if rows is None else rows[i].get('raw_shot_family')) for i,u in enumerate(units))
        result=GlobalSceneExecutionPath(program,allocation,original,proposed,states,_json(payload['source_domain']))
        if _json(result.to_dict())!=_json(payload) or _json(program.to_dict())!=_json(raw):
            raise ValueError('global execution changed its canonical original scope or authority')
        if rows is not None and _json([s.to_dict() for s in states])!=_json(rows):
            raise ValueError('global selected numeric states lost their exact renderer preimages')
        result.assemble(plan=original,domain=domain,path=result)
        return result
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('global execution lost its explicit typed original scope') from exc


def mutable_states(path):
    return tuple(path[i] for i in path.mutable_indices) if isinstance(path,GlobalSceneExecutionPath) else path[1:-1]


def incident_pairs(path):
    return tuple((path[a],path[b]) for a,b in path.incident_pair_indices) if isinstance(path,GlobalSceneExecutionPath) else tuple(zip(path,path[1:]))


def endpoint_scope(path,index):
    return 'whole' if isinstance(path,GlobalSceneExecutionPath) else 'exit' if index==0 else 'entry' if index==len(path)-1 else 'whole'


def dense_layout(dense):
    """Do not infer mutable ownership from first/last positions in global JSON."""
    scope=dense.get('global_execution_scope')
    if scope is None:
        return tuple(range(1,len(dense['state_keys'])-1)),tuple((i,i+1) for i in range(len(dense['state_keys'])-1))
    path=dense_execution_path(dense)
    return path.mutable_indices,path.incident_pair_indices


def dense_execution_path(dense):
    """Reconstruct global numeric sampling; not external alignment authority."""
    scope=dense.get('global_execution_scope')
    if scope is None:
        raise ValueError('global physical boundary validation requires its explicit numeric scope')
    from .render_plan_serialization import render_plan_from_json
    from .group_scene_edit_operations import numeric_replay_domain
    program=dense.get('global_execution_original_numeric_plan')
    numeric=dense.get('global_execution_selected_numeric_plan')
    original=render_plan_from_json(program);proposed=render_plan_from_json(numeric)
    path=execution_path_from_payload(original=original,proposed=proposed,
        domain=numeric_replay_domain(original,dense['source_domain']),payload=scope)
    # This replays a materialized clock, not original alignment offsets. The
    # external alignment domain is independently verified by the parent.
    return path
