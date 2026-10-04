"""One preselected atomic global programme through native completion.

This executes a bounded externally selected numeric union, not another local
repair search, automatic whole-song director or learned authority adapter.
Original duties/reference are frozen before donor observations; all selected
mutable intervals and actual incident edges reach completion together. This
does not provide controller-wide reserves or professional media approval.
"""
from . import group_performer_family_owner as family
from . import group_source_changing_owner as source
from .group_joint_observations import GroupJointObservationSession
from .group_joint_source_camera import JointGroupWorkBudget,GroupOutputRole,JointGroupBudgetExhausted
from .group_scene_conditional_frontier import ConditionalSceneFrontier,MAX_CONDITIONAL_PATHS
from .group_scene_hypothesis_outcomes import SceneHypothesisRejection
from .group_scene_execution_scope import execution_path_from_union,incident_pairs,MAX_REFRESHED_CELLS
from .group_scene_observations import (SceneSensors,compare_scene_edge,coherent_baseline,
    preserves_known,has_editorial_gain,scene_anchor_ticks)
from .optional_native_worker import isolated_optional

def preflight_global_numeric_union(context,union,path):
    """Re-derive operations and dependency support BEFORE any native work.

    Mutable ownership alone is not evidence of an edit. The numeric producer's
    claimed support is independently checked against the external incumbent;
    completion still performs its own defensive re-derivation afterwards.
    """
    from .group_scene_dependency_support import derive_support
    from .group_scene_edit_operations import derive_scene_edit_operations
    numeric,changed=path.assemble(plan=context.incumbent.render_plan,domain=context.domain,path=path)
    operations=derive_scene_edit_operations(original=context.incumbent.render_plan,
        proposed=numeric,domain=context.domain,mutable_intervals=path.mutable_intervals)
    look_edges=family.retained_applied_look_edges(context.incumbent)
    support=derive_support(context.incumbent.render_plan,numeric,retained_applied_look_edges=look_edges)
    if union.changed_segment_indices!=changed or union.dependency_support!=support:
        raise ValueError('global numeric union differs from independently derived operations/dependency support')
    if not operations['has_numeric_edit']:
        raise ValueError('global programme requires an actual source/camera edit before native work')
    if len(support['descriptor_refresh_cells'])>MAX_REFRESHED_CELLS:
        raise ValueError('global refresh cell budget exceeded before native work')
    return dict(contract='global_scene_numeric_preflight_v1',actual_edit_operations=operations,
        dependency_support=support,maximum_refreshed_cells=MAX_REFRESHED_CELLS,
        native_work_performed=False,quality_approved=False)


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
            except JointGroupBudgetExhausted:raise
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
    from .group_scene_source_owner import preserve_incumbent_required_parts
    from .group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from .group_scene_selected_replay import GLOBAL_CONTRACT
    attempts=[];session=None;sensors=None;frontier=None;stage='numeric_preflight'
    try:
        family._fresh(context)
        path=execution_path_from_union(context,context.domain,union)
        numeric_preflight=preflight_global_numeric_union(context,union,path)
        baseline=path.with_plan(context.incumbent.render_plan,context.domain)
        original_policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.original_plan,
            domain=context.domain,derivative_binding=context.source_derivative_binding)
        selected_policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
            domain=context.domain,derivative_binding=context.source_derivative_binding)
        stage='native_assets'
        assets=source.installed_native_assets(context)
        stage='native_observations'
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
            def read(a,b,policy):
                key=a.key,b.key,id(policy)
                if key not in edges:
                    session.budget.spend('edges')
                    row=compare_scene_edge(sensors,a,b)
                    policy.attach(row,a,b)
                    edges[key]=row
                return edges[key]
            reference,known=global_original_reference(sensors,baseline,groups,
                lambda a,b:read(a,b,original_policy))
            # No selected/unselected donor has been sensed before original
            # duties and the coherent original boundary reference are frozen.
            selected_groups=tuple(tuple(sensors.bind(state,h) for h in sensors.discover(state,
                rolls=(None,),endpoint_scope='whole')) for state in path)
            frontier=ConditionalSceneFrontier(selected_groups,path.incident_pair_indices,
                maximum_paths=MAX_CONDITIONAL_PATHS)
            while frontier.peek() is not None:
                values=frontier.peek()
                blocked=frontier.blocked_pair(values)
                if blocked is not None:
                    frontier.begin();frontier.finish('CACHED_OBSERVED_PAIR_REJECT')
                    attempts.append(dict(state_keys=[s.key for s in values],
                        status='CACHED_OBSERVED_PAIR_REJECT',infeasible_pair=blocked));continue
                blocked_state=sensors.cached_state_dense_rejection(path.with_states(values))
                if blocked_state is not None:
                    frontier.begin();frontier.finish('CACHED_OBSERVED_STATE_DENSE_REJECT')
                    attempts.append(dict(state_keys=[s.key for s in values],
                        status='CACHED_OBSERVED_STATE_DENSE_REJECT',
                        state_local_dense_rejection=blocked_state));continue
                # Spend BEFORE advancing: budget at B preserves B as pending,
                # never records it as a negative observation or restarts work.
                stage='conditional_state'
                session.budget.spend('states')
                candidate_path=path.with_states(frontier.begin())
                measured=[];decision=None
                try:
                    stage='conditional_edges'
                    for i,j in path.incident_pair_indices:
                        a,b=candidate_path[i],candidate_path[j]
                        row=read(a,b,selected_policy);measured.append(row)
                        if not row.get('available',False):
                            frontier.remember_infeasible_pair((i,j),a,b,
                                str(row.get('reason') or 'observed incident comparison unavailable'))
                            break
                    if len(measured)!=len(path.incident_pair_indices) or any(
                            not row.get('available',False) for row in measured):
                        attempts.append(dict(state_keys=[s.key for s in candidate_path],
                            status='OBSERVED_PAIR_REJECT',observed_incident_rows=measured))
                        frontier.finish('OBSERVED_PAIR_REJECT');continue
                    decision=global_editorial_decision(known,measured)
                    if not decision['admitted']:
                        attempts.append(dict(state_keys=[s.key for s in candidate_path],status='NO_GAIN_OR_OBSERVED_REJECT',
                            editorial_decision=decision))
                        frontier.finish('NO_GAIN_OR_OBSERVED_REJECT');continue
                    stage='conditional_dense'
                    session.budget.spend('dense_paths')
                    dense=sensors.dense(candidate_path)
                    preserved=preserve_incumbent_required_parts(sensors,baseline,candidate_path,dense)
                except SceneHypothesisRejection as exc:
                    attempts.append(dict(state_keys=[s.key for s in candidate_path],stage=stage,
                        status='OBSERVED_HYPOTHESIS_REJECT',error=str(exc),error_type=type(exc).__name__,
                        editorial_decision=decision,original_duty_programme_sha256=original_duties['program_sha256']))
                    frontier.finish('OBSERVED_HYPOTHESIS_REJECT');continue
                # Completion is outside the hypothesis-local handler. Once a
                # dense-valid conditional programme is selected it is completed
                # ONCE; contract/identity/budget failures here remain terminal.
                frontier.finish('DENSE_VALID_SELECTED')
                program=dict(contract=GLOBAL_CONTRACT,quality_approved=False,
                    global_execution_scope=candidate_path.to_dict(),
                    selected_states=[s.to_dict() for s in candidate_path],
                    baseline_states=[s.to_dict() for s in baseline],
                    baseline_hypothesis_keys=[[s.editorial_contract_sha256 for s in group] for group in groups],
                    original_role_programme_sha256=role_refs,original_duty_program=original_duties,
                    analysis_rolls=[None],composition_window=any(s.role==GroupOutputRole.COMPOSITION for s in path))
                search=dict(scene_replay_program=program,composition_reference=reference,
                    baseline_components=known,editorial_decision=decision,incumbent_known_parts=preserved,
                    numeric_preflight=numeric_preflight,
                    original_duty_program=original_duties,prior_complete_rejections=attempts,
                    conditional_frontier=frontier.to_dict(),conditional_outcome='SELECTED_DENSE_VALID',
                    atomic_numeric_programmes_completed=1,local_complete_bundles_concatenated=False,
                    automatic_whole_song_discovery_implemented=False,quality_approved=False)
                stage='completion'
                return source.complete_source_candidate(context,candidate_path,sensors,session,assets,dense,measured,search)
        last_error=next((row['error'] for row in reversed(attempts) if 'error' in row),'')
        return family.PerformerFamilyRejection('preselected global programme has no observed preserved gain'
            +(': '+last_error if last_error else ''),dict(attempts=attempts,
                outcome=frontier.outcome,conditional_frontier=frontier.to_dict(),
                scene_observations=sensors.archive(),native_work=session.budget.to_dict(),
                normal_global_execution_performed=True,fallback_work_restart_allowed=False,quality_approved=False))
    except JointGroupBudgetExhausted as exc:
        return family.PerformerFamilyRejection(str(exc),dict(attempts=attempts,stage=stage,
            error_type=type(exc).__name__,outcome='BUDGET_PARTIAL',
            conditional_frontier=None if frontier is None else frontier.to_dict(),
            scene_observations=None if sensors is None else sensors.archive(),
            native_work=None if session is None else session.budget.to_dict(),
            normal_global_execution_performed=session is not None,
            unexecuted_measurements_are_unknown=True,fallback_work_restart_allowed=False,quality_approved=False))
    except (ValueError,RuntimeError,OSError) as exc:
        return family.PerformerFamilyRejection(str(exc),dict(attempts=attempts,stage=stage,
            outcome='NUMERIC_PREFLIGHT_REJECTED' if stage=='numeric_preflight' else 'TERMINAL_FAILURE',
            error_type=type(exc).__name__,conditional_frontier=None if frontier is None else frontier.to_dict(),
            scene_observations=None if sensors is None else sensors.archive(),
            native_work=None if session is None else session.budget.to_dict(),
            normal_global_execution_performed=session is not None,
            fallback_work_restart_allowed=False,quality_approved=False))
