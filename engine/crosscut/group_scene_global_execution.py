"""One preselected atomic global programme through native completion.

This executes a bounded externally selected numeric union, not another local
repair search, automatic whole-song director or learned authority adapter.
Original duties/reference are frozen before donor observations; all selected
mutable intervals and actual incident edges reach completion together. This
does not provide controller-wide reserves or professional media approval.
"""
from itertools import islice, product

from . import group_performer_family_owner as family
from . import group_source_changing_owner as source
from .group_joint_observations import GroupJointObservationSession
from .group_joint_source_camera import JointGroupWorkBudget,GroupOutputRole
from .group_scene_execution_scope import execution_path_from_union,incident_pairs
from .group_scene_observations import (SceneSensors,compare_scene_edge,coherent_baseline,
    preserves_known,has_editorial_gain,scene_anchor_ticks)
from .optional_native_worker import isolated_optional

MAX_CONDITIONAL_PATHS=32


def global_original_reference(sensors,baseline,groups,read):
    """One canonical ORIGINAL hypothesis per unit, before donor measurement.

    No edge can silently pick another middle hypothesis. Ambiguous focal
    assignment remains UNKNOWN; all original conditional duty branches remain
    in their independent original duty programme. This is not actor identity.
    """
    if len(groups)!=len(baseline) or any(not group for group in groups):
        raise ValueError('global original reference requires its complete original conditional inventory')
    original=baseline.with_states(tuple(group[0] for group in groups))
    known={};rows=[]
    for index,(a,b) in enumerate(incident_pairs(original)):
        row=read(a,b);unique=True
        for state in (a,b):
            if state.role in (GroupOutputRole.FORMATION,GroupOutputRole.COMPOSITION):continue
            try:
                for tick in scene_anchor_ticks(state.first_output_frame,state.end_output_frame,'whole'):
                    source._dominant(sensors.raw(state,tick,None),state.role)
            except ValueError:unique=False
        rows.append(row)
        known[str(index)]=coherent_baseline([row],focal_assignment_unique=unique)
    return dict(contract='coherent_original_global_scene_reference_v1',
        original_state_keys=[s.key for s in original],edges=rows,
        incident_state_pair_indices=[list(p) for p in baseline.incident_pair_indices],
        original_selection_frozen_before_donor_measurement=True,
        original_conditional_duties_redefined=False,cross_camera_identity_certified=False,
        quality_approved=False),known


def global_editorial_decision(known,measured):
    preserved=all(preserves_known(known[str(i)],row) for i,row in enumerate(measured))
    gained=any(has_editorial_gain(known[str(i)],row) for i,row in enumerate(measured))
    return dict(contract='global_scene_known_boundary_preservation_and_gain_v1',
        admitted=preserved and gained,known_geometry_preserved=preserved,
        observed_editorial_gain=gained,coverage_only_is_not_gain=True,
        quality_approved=False)


@isolated_optional()
def complete_global_scene_program(context,union):
    """Execute ONE numeric union; do not concatenate local completed bundles."""
    from .group_scene_intent import declared_shot_intent
    from .group_scene_original_duties import freeze_original_duty_program
    from .group_scene_source_owner import relation,preserve_incumbent_required_parts
    from .source_derivative_diversity import DERIVATIVE_DISTINCT
    from .group_scene_selected_replay import GLOBAL_CONTRACT
    family._fresh(context)
    path=execution_path_from_union(context,context.domain,union)
    baseline=path.with_plan(context.incumbent.render_plan,context.domain)
    assets=source.installed_native_assets(context)
    attempts=[];session=None;sensors=None
    try:
        with GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
                absolute_deadline=context.deadline,**source.WORK_LIMITS),maximum_raw_cache_bytes=32*1024*1024) as session:
            sensors=SceneSensors(session,assets)
            shots={(s.first_output_frame,s.end_output_frame):s
                for s in source.visible_group_shots(context.incumbent.render_plan,context.domain)}
            role_refs=[]
            for unit,state in zip(baseline.scoped_units,baseline,strict=True):
                role_refs.append(sensors.record_original_role_program(unit.role_program))
                intent=declared_shot_intent(context,shots[state.first_output_frame,state.end_output_frame],endpoint_scope='whole')
                sensors.shot_intents[state.first_output_frame,state.end_output_frame,'whole']=intent
            groups=[tuple(sensors.bind(state,h) for h in sensors.discover(state,rolls=(None,),endpoint_scope='whole'))
                for state in baseline]
            original_duties=freeze_original_duty_program(sensors,groups,
                original_plan_identity=family.digest(family.plan_payload(context.incumbent.render_plan)),
                global_execution_scope=path.to_dict())
            sensors.install_original_duty_program(original_duties)
            edges={}
            def read(a,b):
                key=a.key,b.key
                if key not in edges:
                    session.budget.spend('edges')
                    row=compare_scene_edge(sensors,a,b)
                    row['source_relation']=relation(context.source_derivative_binding,a.eligible_source_index,b.eligible_source_index)
                    if row['source_relation']!=DERIVATIVE_DISTINCT:row['available']=False
                    edges[key]=row
                return edges[key]
            reference,known=global_original_reference(sensors,baseline,groups,read)
            # No selected/unselected donor has been sensed before original
            # duties and the coherent original boundary reference are frozen.
            selected_groups=[tuple(sensors.bind(state,h) for h in sensors.discover(state,
                rolls=(None,),endpoint_scope='whole')) for state in path]
            for values in islice(product(*selected_groups),MAX_CONDITIONAL_PATHS):
                candidate_path=path.with_states(values);session.budget.spend('states')
                measured=[read(a,b) for a,b in incident_pairs(candidate_path)]
                decision=global_editorial_decision(known,measured)
                if not decision['admitted']:
                    attempts.append(dict(state_keys=[s.key for s in candidate_path],status='NO_GAIN_OR_OBSERVED_REJECT',
                        editorial_decision=decision));continue
                session.budget.spend('dense_paths')
                dense=sensors.dense(candidate_path)
                preserved=preserve_incumbent_required_parts(sensors,baseline,candidate_path,dense)
                program=dict(contract=GLOBAL_CONTRACT,quality_approved=False,
                    global_execution_scope=candidate_path.to_dict(),
                    selected_states=[s.to_dict() for s in candidate_path],
                    baseline_states=[s.to_dict() for s in baseline],
                    baseline_hypothesis_keys=[[s.editorial_contract_sha256 for s in group] for group in groups],
                    original_role_programme_sha256=role_refs,original_duty_program=original_duties,
                    analysis_rolls=[None],composition_window=any(s.role==GroupOutputRole.COMPOSITION for s in path))
                search=dict(scene_replay_program=program,composition_reference=reference,
                    baseline_components=known,editorial_decision=decision,incumbent_known_parts=preserved,
                    original_duty_program=original_duties,prior_complete_rejections=attempts,
                    atomic_numeric_programmes_completed=1,local_complete_bundles_concatenated=False,
                    automatic_whole_song_discovery_implemented=False,quality_approved=False)
                return source.complete_source_candidate(context,candidate_path,sensors,session,assets,dense,measured,search)
        return family.PerformerFamilyRejection('preselected global programme has no observed preserved gain',
            dict(attempts=attempts,normal_global_execution_performed=True,quality_approved=False))
    except (ValueError,RuntimeError,OSError) as exc:
        return family.PerformerFamilyRejection(str(exc),dict(attempts=attempts,
            scene_observations=None if sensors is None else sensors.archive(),
            native_work=None if session is None else session.budget.to_dict(),
            normal_global_execution_performed=True,fallback_work_restart_allowed=False,quality_approved=False))
