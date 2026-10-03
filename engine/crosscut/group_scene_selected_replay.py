"""Independent selected scene-program replay, never another donor search.

Original baseline hypotheses and selected source/camera hypotheses are freshly
derived from original pixels. Unselected search rows are preserved as an old
archive, not represented as freshly measured or as an optimality certificate.
Descriptors, look, faces, dense paths and the complete bundle are rebuilt once.
This does not establish a hard end-to-end completion reserve or media quality.
"""
from time import monotonic

from .optional_native_worker import isolated_optional
from .group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget
from .group_joint_observations import GroupJointObservationSession
from .group_joint_state_factory import fixed_neighbor_state_from_render_plan
from .group_joint_source_camera import joint_state_from_render_plan
from . import group_performer_family_owner as family
from . import group_source_changing_owner as source

CONTRACT='frozen_scene_source_camera_program_v4'


def _path(plan,domain,rows):
    if not isinstance(rows,list) or len(rows) not in (3,4):
        raise ValueError('selected scene program needs three/four complete states')
    states=[]
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or not isinstance(row.get('key_preimage'),dict):
            raise ValueError('selected scene state preimage missing')
        raw=row['key_preimage']
        builder=fixed_neighbor_state_from_render_plan if index in (0,len(rows)-1) else joint_state_from_render_plan
        state=builder(plan=plan,domain=domain,slot=index,first_output_frame=raw['first'],
            end_output_frame=raw['end'],role=GroupOutputRole(raw['role']),
            editorial_contract_sha256=raw.get('editorial_contract_sha256'),
            raw_shot_family=row.get('raw_shot_family'))
        if family.digest(state.to_dict())!=family.digest(row):
            raise ValueError('selected scene state differs from its complete renderer preimage')
        states.append(state)
    return tuple(states)


def validate_program(program,dense):
    from .group_native_head_orientation import SUPPORTED_ANALYSIS_ROLLS
    if not isinstance(program,dict):
        raise ValueError('scene receipt has no frozen selected program')
    payload=dict(program);sha=payload.pop('program_sha256',None)
    if (payload.get('contract')!=CONTRACT or payload.get('quality_approved') is not False
            or sha!=family.digest(payload) or type(payload.get('composition_window')) is not bool
            or not isinstance(payload.get('numeric_plan'),dict)):
        raise ValueError('selected scene program hash/authority invalid')
    states=payload.get('selected_states');baseline=payload.get('baseline_states')
    groups=payload.get('baseline_hypothesis_keys');rolls=payload.get('analysis_rolls')
    if (not isinstance(states,list) or len(states) not in (3,4)
            or not isinstance(baseline,list) or len(baseline)!=len(states)
            or not isinstance(groups,list) or len(groups)!=len(states)
            or any(not isinstance(group,list) or any(not isinstance(key,str) or len(key)!=64
                or any(c not in '0123456789abcdef' for c in key) for key in group)
                or len(set(group))!=len(group) for group in groups)
            or not isinstance(rolls,list) or not rolls
            or any(roll is not None and (type(roll) not in (int,float)
                or roll not in SUPPORTED_ANALYSIS_ROLLS) for roll in rolls)
            or len(set(rolls))!=len(rolls)):
        raise ValueError('selected scene baseline/program scope is malformed')
    try:
        if ([row['key_preimage'].get('editorial_contract_sha256') for row in states]!=dense['hypothesis_keys']
                or [[row['key_preimage']['first'],row['key_preimage']['end']] for row in states]!=dense['window_bounds']
                or [row['key_preimage']['role'] for row in states]!=dense['roles']
                or any((a['key_preimage']['first'],a['key_preimage']['end'],a['key_preimage']['role']) !=
                    (b['key_preimage']['first'],b['key_preimage']['end'],b['key_preimage']['role'])
                    for a,b in zip(states,baseline,strict=True))
                or payload['composition_window']!=(GroupOutputRole.COMPOSITION.value in dense['roles'])):
            raise ValueError('selected scene program changed bounds, role or hypothesis ownership')
    except (KeyError,TypeError) as exc:
        raise ValueError('selected scene state/role inventory missing') from exc
    from .group_scene_archive import validate_archive
    validate_archive(program, dense)
    from .group_scene_original_duties import validate_original_duty_program
    original_duties=payload.get('original_duty_program')
    if original_duties is None or original_duties!=dense.get('original_duty_program'):
        raise ValueError('selected scene lost its presearch original duty programme')
    validate_original_duty_program(original_duties,raw_inventory={family.digest(row):row
        for row in dense['scene_discovery']['native_rows']})
    units=original_duties['units']
    if len(units)!=len(baseline):
        raise ValueError('selected scene original duty programme has another window')
    for old,group,unit in zip(baseline,groups,units,strict=True):
        if [b['original_subject_hypothesis_id'] for b in unit['branches']]!=group:
            raise ValueError('selected scene original baseline hypotheses changed duty ownership')
        for branch in unit['branches']:
            received=dict(branch['original_state']['key_preimage'])
            received.pop('editorial_contract_sha256',None)
            expected=dict(old['key_preimage']);expected.pop('editorial_contract_sha256',None)
            if family.digest(received)!=family.digest(expected):
                raise ValueError('selected scene original duty unit changed the original full camera')
    from .render_plan_serialization import render_plan_from_json
    from .group_scene_dependency_support import validate_support
    try:
        original=render_plan_from_json(payload['original_numeric_plan'])
        if original_duties['original_plan_identity']!=family.digest(family.plan_payload(original)):
            raise ValueError('selected scene original duties changed their original plan identity')
        numeric=render_plan_from_json(payload['numeric_plan'])
        support=payload['dependency_support']
        edges=tuple(support['retained_applied_look_edges'])
        validate_support(support,original,numeric,retained_applied_look_edges=edges)
        if support['numeric_write_cells']!=dense['changed_cells']:
            raise ValueError('scene dependency numeric writes differ from completed dense scope')
    except (KeyError,TypeError) as exc:
        raise ValueError('scene program lost its original/dependency preimage') from exc


@isolated_optional()
def replay_selected_scene(context,candidate):
    """Fresh independent worker validates one frozen program, not its rivals."""
    from . import group_scene_source_owner as scene
    from .group_scene_observations import SceneSensors,compare_scene_edge,coherent_baseline,preserves_known,has_editorial_gain,scene_anchor_ticks
    from .group_scene_intent import declared_shot_intent
    from .group_scene_editorial_policy import coherent_path_reference,evaluate_composition_path
    from .render_plan_serialization import render_plan_from_json
    from .source_derivative_diversity import DERIVATIVE_DISTINCT
    started=monotonic();family._fresh(context)
    source.validate_binding(candidate)
    receipt=family.plain(candidate.performer_family_review)
    dense=receipt['source_camera_refresh'];program=receipt['selected_scene_program']
    validate_program(program,dense)
    numeric=render_plan_from_json(program['numeric_plan'])
    if render_plan_from_json(program['original_numeric_plan'])!=context.incumbent.render_plan:
        raise ValueError('scene dependency original plan differs from the external incumbent')
    retained_look_edges=tuple(sorted(d.boundary_index for d in context.incumbent.boundary_look_match.decisions
        if d.applied))
    from .group_scene_dependency_support import validate_support
    validate_support(program['dependency_support'],context.incumbent.render_plan,numeric,
        retained_applied_look_edges=retained_look_edges)
    path=_path(numeric,context.domain,program['selected_states'])
    baseline=_path(context.incumbent.render_plan,context.domain,program['baseline_states'])
    assembled,_=source._assembler(path)(plan=context.incumbent.render_plan,domain=context.domain,path=path)
    if assembled!=numeric:
        raise ValueError('selected scene numeric program changed an unrelated clock/camera/transition')
    assets=source.installed_native_assets(context)
    if source.asset_payload(assets)!=dense['native_assets']:
        raise ValueError('selected scene native assets changed')
    with GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=context.deadline,**source.WORK_LIMITS),maximum_raw_cache_bytes=32*1024*1024) as session:
        sensors=SceneSensors(session,assets)
        shots={(s.first_output_frame,s.end_output_frame):s
            for s in source.visible_group_shots(context.incumbent.render_plan,context.domain)}
        def scope(index):
            return 'exit' if index==0 else 'entry' if index==len(path)-1 else 'whole'
        def derive(state,index):
            shot=shots.get((state.first_output_frame,state.end_output_frame))
            if shot is None:
                raise ValueError('selected scene state is not an original output-owned shot')
            intent=declared_shot_intent(context,shot,endpoint_scope=scope(index))
            if intent['role']!=state.role.value:
                raise ValueError('selected scene state weakened its original declared duty')
            sensors.shot_intents[state.first_output_frame,state.end_output_frame,scope(index)]=intent
            return tuple(sensors.bind(state,h) for h in sensors.discover(state,
                rolls=tuple(program['analysis_rolls']),endpoint_scope=scope(index)))
        groups=[derive(state,index) for index,state in enumerate(baseline)]
        if [[s.editorial_contract_sha256 for s in group] for group in groups]!=program['baseline_hypothesis_keys']:
            raise ValueError('selected scene original baseline hypotheses changed')
        from .group_scene_original_duties import freeze_original_duty_program
        original_duties=freeze_original_duty_program(sensors,groups,
            original_plan_identity=family.digest(family.plan_payload(context.incumbent.render_plan)))
        if original_duties!=program['original_duty_program']:
            raise ValueError('selected scene fresh original required duty lifecycle changed')
        sensors.install_original_duty_program(original_duties)
        for index,state in enumerate(path):
            alternatives=derive(state,index)
            if state.key not in {s.key for s in alternatives}:
                raise ValueError('selected scene source/camera hypothesis is not freshly observed')
        edges={}
        def read(a,b):
            key=a.key,b.key
            if key not in edges:
                session.budget.spend('edges')
                value=compare_scene_edge(sensors,a,b)
                value['source_relation']=scene.relation(context.source_derivative_binding,
                    a.eligible_source_index,b.eligible_source_index)
                if value['source_relation']!=DERIVATIVE_DISTINCT:value['available']=False
                edges[key]=value
            return edges[key]
        known={};reference=None
        if program['composition_window']:
            reference=coherent_path_reference(groups,read)
            known={i:row if row is not None else dict(available=False,metrics=None)
                for i,row in enumerate(reference['edges'])}
        else:
            from itertools import islice,product
            for index in range(len(baseline)-1):
                values=[read(a,b) for a,b in islice(product(groups[index],groups[index+1]),4)]
                unique=True
                for old_index in (index,index+1):
                    state=baseline[old_index]
                    try:
                        for tick in scene_anchor_ticks(state.first_output_frame,state.end_output_frame,scope(old_index)):
                            source._dominant(sensors.raw(state,tick,None),state.role)
                    except ValueError:unique=False
                known[index]=coherent_baseline(values,focal_assignment_unique=unique)
        measured=[read(a,b) for a,b in zip(path,path[1:])]
        decision=evaluate_composition_path(reference,measured) if program['composition_window'] else None
        if (decision is not None and not decision['admitted']
                or decision is None and (not all(preserves_known(known[i],row) for i,row in enumerate(measured))
                    or not any(has_editorial_gain(known[i],row) for i,row in enumerate(measured)))):
            raise ValueError('selected scene program no longer beats its actual incumbent reference')
        search=receipt['search']
        if (family.digest(reference)!=family.digest(search['composition_reference'])
                or family.digest(known)!=family.digest(search['baseline_components'])
                or family.digest(decision)!=family.digest(search['editorial_decision'])):
            raise ValueError('selected scene original reference/editorial comparison changed')
        session.budget.spend('dense_paths')
        rows=sensors.dense(path)
        preserved=scene.preserve_incumbent_required_parts(sensors,baseline,path,rows)
        if family.digest(preserved)!=family.digest(search['incumbent_known_parts']):
            raise ValueError('selected scene per-tick incumbent required evidence changed')
        rebuilt=source.complete_source_candidate(context,path,sensors,session,assets,rows,measured,search,
            frozen_scene_discovery=dense['scene_discovery'])
        if (family.digest(family.metadata_payload(rebuilt))!=family.digest(family.metadata_payload(candidate))
                or rebuilt.render_plan!=candidate.render_plan):
            raise ValueError('selected scene independent complete family replay disagrees')
        actual=family.plain(rebuilt.performer_family_review);expected=dict(receipt)
        actual.pop('native_work',None);expected.pop('native_work',None)
        if family.digest(actual)!=family.digest(expected):
            raise ValueError('selected scene fresh descriptor/look/receipt replay disagrees')
        work=session.budget.to_dict()
        observed_sources=sorted({row.request_source_index for row in sensors.raw_rows.values()})
    family._fresh(context)
    return dict(contract='independent_selected_scene_program_replay_v2',
        selected_program_sha256=program['program_sha256'],fresh_complete_family_equal=True,
        selected_and_original_hypotheses_reobserved=True,unselected_donor_search_repeated=False,
        unselected_discovery_archive_preserved_not_reobserved=True,
        referenced_discovery_archive_preserved=True,
        renderer_dependency_support_independently_rederived=True,
        selected_dependency_support_sha256=program['dependency_support']['support_sha256'],
        received_scene_discovery_sha256=program['discovery_archive_binding']['archive_sha256'],
        archive_history_authority='received_candidate_reference_closure_only',
        pre_receipt_history_completeness_certified=False,
        raw_scene_observed_request_source_indices=observed_sources,
        native_work=work,elapsed_seconds=monotonic()-started,quality_approved=False)
