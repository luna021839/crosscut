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

CONTRACT='frozen_scene_source_camera_program_v1'


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


@isolated_optional()
def replay_selected_scene(context,candidate):
    """Fresh independent worker validates one frozen program, not its rivals."""
    from . import group_scene_source_owner as scene
    from .group_scene_observations import SceneSensors,compare_scene_edge,coherent_baseline,preserves_known,scene_anchor_ticks
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
                    or not any(not known[i]['available'] or any(v<known[i]['metrics'][k]-1e-6
                        for k,v in row['metrics'].items() if k in known[i]['metrics']) for i,row in enumerate(measured)))):
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
    return dict(contract='independent_selected_scene_program_replay_v1',
        selected_program_sha256=program['program_sha256'],fresh_complete_family_equal=True,
        selected_and_original_hypotheses_reobserved=True,unselected_donor_search_repeated=False,
        unselected_discovery_archive_preserved_not_reobserved=True,
        raw_scene_observed_request_source_indices=observed_sources,
        native_work=work,elapsed_seconds=monotonic()-started,quality_approved=False)
