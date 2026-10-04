"""Original-native focal duties frozen before donor discovery.

Each original hypothesis owns a separate anonymous, source-local lifecycle.
An observed part stays required through later missing/unresolved measurements.
The union of required part NAMES is a conservative candidate constraint, NOT
a cast count or resolved original/cross-camera actor correspondence. Composition
and formation keep their existing inventory policy; mixed-role/global programme,
hard phase reserves and coherent original hypothesis selection remain separate.
"""
from collections.abc import Mapping
from copy import deepcopy
import json

from . import group_performer_family_owner as family
from .group_joint_source_camera import GroupOutputRole
from .group_scene_intent import validate_intent
from .group_scene_role_obligations import ARM_PARTS

CONTRACT = 'original_native_focal_duty_lifecycle_v1'
GLOBAL_CONTRACT = 'explicit_global_original_native_focal_duty_lifecycle_v1'
FOCAL = (GroupOutputRole.FOCAL_HEAD.value, GroupOutputRole.FOCAL_HEAD_ARMS.value)
MAX_ORIGINAL_NATIVE_SAMPLE_TICKS = 16
MAX_ORIGINAL_HEAD_LINK_NATIVE_GAP = 12


def original_sample_ticks(first,end,anchors):
    """Bound original NN scope, not candidate dense admission or confidence.

    Unrequested original ticks are explicit not_requested observations. A
    previously known required part remains required across them. This is not
    an all-original-frames NN/identity certificate or a hard phase lease.
    """
    if (type(first) is not int or type(end) is not int or first>=end
            or len(set(anchors))>MAX_ORIGINAL_NATIVE_SAMPLE_TICKS
            or any(type(t) is not int or not first<=t<end for t in anchors)):
        raise ValueError('original sample plan has invalid interval/anchors')
    if end-first<=MAX_ORIGINAL_NATIVE_SAMPLE_TICKS:return tuple(range(first,end))
    # Plan the complete temporal grid before honoring exact endpoint anchors.
    # Replacing the nearest nonanchor preserves the later half of the grid;
    # filling from its front until a cap did not.
    selected={first+round((end-first-1)*i/(MAX_ORIGINAL_NATIVE_SAMPLE_TICKS-1))
        for i in range(MAX_ORIGINAL_NATIVE_SAMPLE_TICKS)}
    anchored=set(anchors)
    for tick in sorted(anchored-selected):
        displaced=min(selected-anchored,key=lambda n:(abs(n-tick),n))
        selected.remove(displaced);selected.add(tick)
    return tuple(sorted(selected))


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _raw_dict(raw):
    return raw.to_dict() if hasattr(raw, 'to_dict') else raw


def _branch(*, state, hypothesis, raw_at, head_link):
    preimage = state['key_preimage']
    role = preimage['role']
    intent = hypothesis['shot_intent']
    validate_intent(intent, role=role, first=preimage['first'], end=preimage['end'],
        endpoint_scope=hypothesis['endpoint_scope'])
    first, end = intent['required_output_interval']
    sampled=original_sample_ticks(first,end,hypothesis['anchor_output_ticks'])
    selected = hypothesis['anchor_head_ids'][0] if role in FOCAL else None
    previous = None
    rows, links, latched = [], [], {}
    explicit = role == GroupOutputRole.FOCAL_HEAD_ARMS.value
    for tick in range(first, end):
        if tick not in sampled:
            # No decode/NN/LK call, no negative detection, no interpolated
            # original actor or pose. Retain already known/explicit duties.
            parts={name:dict(obligation='required' if role in FOCAL and (explicit or name in latched) else 'unknown',
                observation='not_requested',
                became_required_at_output_tick=first if explicit else latched.get(name,{}).get('output_tick'),
                original_required_reason='explicit_original_output_role' if explicit else
                    'observed_selected_original_hypothesis_latched' if name in latched else 'original_measurement_unknown',
                original_first_observed_reference=latched.get(name)) for name in ARM_PARTS}
            rows.append(dict(output_tick=tick,native_frame_index=preimage['native'][tick-preimage['first']],
                native_observation_sha256=None,selected_frame_local_head_id=None,
                original_person_observation_id=None,head_obligation='required' if role in FOCAL else 'not_requested',
                head_observation='not_requested',parts=parts,raw_context_retained=True))
            # No inferred row is created. Keep only the last ACTUAL sampled
            # raw/conditional owner so the next sample must earn a measured
            # native head link. Unknown links never reacquire a largest face.
            continue
        raw = _raw_dict(raw_at(tick))
        sha = family.digest(raw)
        native = preimage['native'][tick-preimage['first']]
        if (raw['native_frame_index'] != native or raw['source_clock_identity'] != preimage['clock']
                or raw['native_pixel_sha256'] is None):
            raise ValueError('original duty raw source/native clock/raster changed')
        if previous is not None and native-previous['native_frame_index']>MAX_ORIGINAL_HEAD_LINK_NATIVE_GAP:
            previous=None;selected=None
        if tick in hypothesis['anchor_output_ticks']:
            anchor = hypothesis['anchor_output_ticks'].index(tick)
            if sha != hypothesis['raw_observation_sha256'][anchor]:
                raise ValueError('original duty observation differs from its original hypothesis anchor')
            if previous is None:
                # Separate conditional endpoint of this original hypothesis,
                # not a measured actor bridge across the unobserved middle.
                selected=hypothesis['anchor_head_ids'][anchor] if role in FOCAL else None
        if previous is not None and role in FOCAL and selected is not None:
            link = head_link(previous, raw)
            if link.get('raw_pair') != [family.digest(previous), sha]:
                raise ValueError('original duty head link borrowed another native pair')
            links.append(deepcopy(link))
            choices = [b for a,b in link['unique_links'] if a == selected]
            selected = choices[0] if len(choices) == 1 else None
        head_ids = [head['observation_id'] for head in raw['heads']]
        if selected not in head_ids:
            selected = None
        if role in FOCAL and tick in hypothesis['anchor_output_ticks'] and selected is not None:
            anchor = hypothesis['anchor_output_ticks'].index(tick)
            if selected != hypothesis['anchor_head_ids'][anchor]:
                selected = None  # Keep known duties; never switch to the endpoint's other actor.
        owners = [p for p in raw['people'] if p.get('unique_head_observation_id') == selected] if selected is not None else []
        owner = owners[0] if len(owners) == 1 else None
        parts = {}
        for name in ARM_PARTS:
            if role not in FOCAL:
                observation = 'not_requested'
            elif raw.get('pose_state') == 'not_requested':
                observation = 'not_requested'
            elif selected is None or len(owners) > 1:
                observation = 'unresolved'
            elif owner is None:
                observation = 'missing'
            else:
                status = dict(owner.get('part_states', ())).get(name)
                observation = 'observed' if status == 'observed' else 'unresolved' if status == 'ambiguous' else 'missing'
            if observation == 'observed' and name not in latched:
                latched[name] = dict(output_tick=tick, native_observation_sha256=sha,
                    original_person_observation_id=owner['observation_id'])
            required = role in FOCAL and (explicit or name in latched)
            parts[name] = dict(obligation='required' if required else 'unknown',
                observation=observation,
                became_required_at_output_tick=first if explicit else latched.get(name, {}).get('output_tick'),
                original_required_reason='explicit_original_output_role' if explicit else
                    'observed_selected_original_hypothesis_latched' if name in latched else 'original_measurement_unknown',
                original_first_observed_reference=latched.get(name))
        rows.append(dict(output_tick=tick, native_frame_index=native,
            native_observation_sha256=sha, selected_frame_local_head_id=selected,
            original_person_observation_id=None if owner is None else owner['observation_id'],
            head_obligation='required' if role in FOCAL else 'not_requested',
            head_observation='not_requested' if role not in FOCAL else
                'observed' if selected is not None else 'missing' if not head_ids else 'unresolved',
            parts=parts, raw_context_retained=True))
        previous = raw
    return dict(original_subject_hypothesis_id=family.digest(hypothesis),
        original_state=state, original_hypothesis=hypothesis, rows=rows, head_pixel_links=links,
        original_requested_output_ticks=list(sampled),
        every_original_native_tick_observed=len(sampled)==end-first,
        unobserved_actor_bridges_certified=False,
        named_identity_certified=False, cross_camera_identity_certified=False)


def _unit(branches, eligible_request_indices):
    if not branches:
        raise ValueError('original duties need explicit original hypotheses before donor search')
    first = branches[0]
    state = first['original_state']['key_preimage']
    interval = first['original_hypothesis']['shot_intent']['required_output_interval']
    for branch in branches:
        p = branch['original_state']['key_preimage']
        if any(p[k] != state[k] for k in ('first','end','source','role','native','clock','slot')):
            raise ValueError('original duty alternatives changed original source/camera unit')
        if branch['original_hypothesis']['shot_intent'] != first['original_hypothesis']['shot_intent']:
            raise ValueError('original duty alternatives changed external original intent')
        request = eligible_request_indices[p['source']]
        for row in branch['rows']:
            # Source ownership is checked against the canonical raw inventory by the validator.
            row['original_request_source_index'] = request
    rows = []
    for offset, tick in enumerate(range(*interval)):
        required = sorted({name for branch in branches for name,part in branch['rows'][offset]['parts'].items()
            if part['obligation'] == 'required'})
        rows.append(dict(output_tick=tick, required_part_names=required,
            branch_row_sha256=[family.digest(branch['rows'][offset]) for branch in branches]))
    identity = family.digest(dict(original_source=state['source'], clock=state['clock'],
        bounds=[state['first'],state['end']], role=state['role'], intent=first['original_hypothesis']['shot_intent']))
    return dict(original_unit_id=identity, first_output_frame=state['first'], end_output_frame=state['end'],
        role=state['role'], required_output_interval=interval, branches=branches, rows=rows,
        original_hypothesis_ambiguity_retained=len(branches)>1, part_names_are_not_cast_counts=True,
        lifecycle_policy='separate_original_focal_hypotheses_latched_part_names' if state['role'] in FOCAL
            else 'existing_native_inventory_policy_not_original_part_lifecycle',
        original_native_part_lifecycle_frozen=state['role'] in FOCAL,
        candidate_admitted=False, quality_approved=False)


def _global_scope_inventory(scope):
    """Received scope closure only; parent still verifies the external original.

    Gaps are legal only when the explicit original allocation and its actual
    neighbour inventory own them. The legacy continuous-window contract does
    not gain an option to delete inconvenient original duty units.
    """
    from .group_scene_execution_scope import (CONTRACT as SCOPE_CONTRACT,
        MAX_MUTABLE_UNITS,MAX_MUTABLE_FRAMES,MAX_REFRESHED_CELLS,_unit_indices)
    from .group_scene_owned_program import OwnedSceneProgram,OwnedSceneUnit,own_scene_intervals
    raw=scope['original_owned_program']
    dump=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    program=OwnedSceneProgram(raw['original_plan_identity'],dump(raw['original_plan_header']),
        raw['source_domain_sha256'],tuple(raw['global_cell_frame_counts']),
        tuple(OwnedSceneUnit(u['unit_id'],u['first_output_frame'],u['end_output_frame'],
            dump(u['original_role_program'])) for u in raw['units']))
    allocation=own_scene_intervals(program,mutable_unit_ids=tuple(scope['allocation']['mutable_unit_ids']),
        maximum_mutable_units=scope['allocation']['maximum_mutable_units'],
        maximum_mutable_frames=scope['allocation']['maximum_mutable_frames'])
    bounds=[(u.first_output_frame,u.end_output_frame) for u in program.units]
    if (program.to_dict()!=raw or allocation.to_dict()!=scope['allocation']
            or family.digest(scope['source_domain'])!=program.source_domain_sha256
            or allocation.maximum_mutable_units>MAX_MUTABLE_UNITS
            or allocation.maximum_mutable_frames>MAX_MUTABLE_FRAMES
            or len({u.unit_id for u in program.units})!=len(program.units)
            or not bounds or bounds[0][0]!=0
            or any(type(v) is not int for pair in bounds for v in pair)
            or any(first>=end for first,end in bounds)
            or any(a[1]!=b[0] for a,b in zip(bounds,bounds[1:]))
            or bounds[-1][1]!=sum(program.global_cell_frame_counts)):
        raise ValueError('global original duties lost their explicit complete original allocation')
    units=tuple(program.units[i] for i in _unit_indices(program,allocation))
    mutable=tuple(i for i,u in enumerate(units) if u.unit_id in allocation.mutable_unit_ids)
    pairs=tuple((i,i+1) for i,(a,b) in enumerate(zip(units,units[1:]))
        if a.end_output_frame==b.first_output_frame and (i in mutable or i+1 in mutable))
    expected=dict(contract=SCOPE_CONTRACT,original_owned_program=raw,source_domain=scope['source_domain'],
        allocation=allocation.to_dict(),scoped_unit_ids=[u.unit_id for u in units],
        unit_intervals=[[u.first_output_frame,u.end_output_frame] for u in units],
        mutable_state_indices=list(mutable),incident_state_pair_indices=[list(p) for p in pairs],
        endpoint_scopes=['whole']*len(units),maximum_refreshed_cells=MAX_REFRESHED_CELLS,
        positional_fixed_endpoints_assumed=False,whole_camera_restarts_at_role_epochs=False,
        original_native_duties_frozen_by_this_numeric_scope=False,
        whole_request_phase_reserves_implemented=False,quality_approved=False)
    if dump(scope)!=dump(expected):
        raise ValueError('global original duties changed explicit intervals/neighbours/authority')
    return units


def _program(units, plan_identity, eligible, global_execution_scope=None):
    value = dict(contract=CONTRACT if global_execution_scope is None else GLOBAL_CONTRACT,
        original_plan_identity=plan_identity,
        eligible_request_indices=list(eligible), units=units,
        frozen_before_donor_discovery=True, donor_observations_define_duties=False,
        original_known_part_loss_is_not_a_waiver=True, raw_observations_referenced_not_duplicated=True,
        coherent_original_programme_selected=False,
        coherent_cross_camera_actor_correspondence_certified=False,
        mixed_role_global_programme_implemented=False, hard_original_phase_reserve_implemented=False,
        quality_approved=False)
    if global_execution_scope is not None:
        owned=_global_scope_inventory(global_execution_scope)
        # Family duty identity intentionally excludes output relocation;
        # numeric ownership identity includes the original complete plan.
        # Both are independently rederived by the selected parent/final path.
        if (eligible!=tuple(global_execution_scope['source_domain']['eligible_request_indices'])
                or len(owned)!=len(units)):
            raise ValueError('global original duties changed their whole external programme/source domain')
        for original,unit in zip(owned,units,strict=True):
            bounds=[original.first_output_frame,original.end_output_frame]
            roles={epoch['role'] for epoch in original.role_program['epochs']}
            if (unit['required_output_interval']!=bounds
                    or [unit['first_output_frame'],unit['end_output_frame']]!=bounds
                    or roles!={unit['role']}):
                raise ValueError('global original duties changed their original whole-unit role/interval')
        value['global_execution_scope']=global_execution_scope
        value['original_owned_programme_plan_sha256']=global_execution_scope['original_owned_program']['original_plan_identity']
        value['original_unit_bindings']=[dict(numeric_original_unit_id=original.unit_id,
            native_original_unit_id=unit['original_unit_id'],
            original_plan_identity=original.role_program['original_plan_identity'],
            original_role_program=original.role_program,
            first_output_frame=original.first_output_frame,end_output_frame=original.end_output_frame,
            original_hypothesis_branches=[b['original_subject_hypothesis_id'] for b in unit['branches']])
            for original,unit in zip(owned,units,strict=True)]
    value['program_sha256'] = family.digest(value)
    # Canonical transport types at the producer, not equality weakened after
    # worker/JSON replay. Tuple/list conversion cannot fabricate a new failure.
    return json.loads(json.dumps(family.plain(value),sort_keys=True,allow_nan=False))


def freeze_original_duty_program(sensors, baseline_groups, *, original_plan_identity,global_execution_scope=None):
    """Read ORIGINAL native units only; do not inspect selected/unselected donors."""
    eligible = sensors.session.domain.eligible_indices
    units = []
    if any(not group for group in baseline_groups):
        # Preflight the entire window before any extra original dense sensing.
        # This is unresolved scope, not an empty/waived duty or negative donor.
        raise ValueError('original duties need explicit original hypotheses before donor search')
    for group in baseline_groups:
        branches = []
        for state in group:
            hypothesis = sensors.hypothesis(state)
            original_rows = {}
            def raw_at(tick):
                raw = sensors.raw(state,tick,hypothesis['analysis_roll'])
                original_rows[raw.sha256] = raw
                return raw
            def head_link(a,b):
                return sensors._head_links(original_rows[family.digest(a)],original_rows[family.digest(b)])
            branches.append(_branch(state=state.to_dict(), hypothesis=deepcopy(hypothesis),
                raw_at=raw_at, head_link=head_link))
        units.append(_unit(branches, eligible))
    program = _program(units, original_plan_identity, eligible,global_execution_scope)
    sensors.session.budget.spend('pose_evidence_bytes', len(__import__('json').dumps(
        program,sort_keys=True,allow_nan=False).encode()))
    return program


def validate_original_duty_program(program, *, raw_inventory):
    """Re-derive every lifecycle from received raw references and head-link paths.

    This is reference closure, not controller-owned whole-history authority.
    Independent parent/final original-pixel remeasurement is still mandatory.
    """
    if not isinstance(program,dict) or program.get('contract') not in (CONTRACT,GLOBAL_CONTRACT):
        raise ValueError('scene original duty programme is missing')
    payload = dict(program); sha = payload.pop('program_sha256',None)
    if sha != family.digest(payload):
        raise ValueError('original duty programme hash changed')
    try:
        eligible = program['eligible_request_indices']
        if (not isinstance(eligible,list) or not eligible or len(set(eligible)) != len(eligible)
                or any(type(i) is not int or i<0 for i in eligible)):
            raise ValueError('original duty programme source domain invalid')
        units = []
        for unit in program['units']:
            branches = []
            for received in unit['branches']:
                state = received['original_state']; h = received['original_hypothesis']
                if (family.digest(h) != received['original_subject_hypothesis_id']
                        or state['key_preimage'].get('editorial_contract_sha256') != family.digest(h)
                        or state['state_key'] != family.digest(state['key_preimage'])):
                    raise ValueError('original duty subject differs from its original state/hypothesis')
                rows = received['rows']
                first,end = h['shot_intent']['required_output_interval']
                if [r['output_tick'] for r in rows] != list(range(first,end)):
                    raise ValueError('original duty programme omitted a scheduled original tick')
                refs = {r['output_tick']:r['native_observation_sha256'] for r in rows}
                def raw_at(tick):
                    raw = raw_inventory[refs[tick]]
                    if (family.digest(raw) != refs[tick] or
                            raw['request_source_index'] != eligible[state['key_preimage']['source']]):
                        raise ValueError('original duty raw reference changed original source')
                    return raw
                links = {tuple(link['raw_pair']):link for link in received['head_pixel_links']}
                if len(links) != len(received['head_pixel_links']):
                    # Repeated native aliases may repeat the same measured pair.
                    if any(link != links[tuple(link['raw_pair'])] for link in received['head_pixel_links']):
                        raise ValueError('original duty repeated native link changed')
                def head_link(a,b):
                    return links[family.digest(a),family.digest(b)]
                branches.append(_branch(state=state,hypothesis=h,raw_at=raw_at,head_link=head_link))
            units.append(_unit(branches,eligible))
        scope=program.get('global_execution_scope')
        if (program['contract']==GLOBAL_CONTRACT)!=(scope is not None):
            raise ValueError('original duty programme cannot relabel its original ownership contract')
        expected = _program(units, program['original_plan_identity'],tuple(eligible),scope)
        if family.digest(expected) != family.digest(program):
            raise ValueError('original required duty lifecycle changed after original observations')
        bounds = [(u['first_output_frame'],u['end_output_frame']) for u in units]
        if len(set(bounds)) != len(bounds) or any(a[1]>b[0] if scope is not None else a[1]!=b[0]
                for a,b in zip(bounds,bounds[1:])):
            raise ValueError('original duty programme units are duplicated/discontinuous')
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('original duty programme lost native/hypothesis reference closure') from exc
    return program


class NativeOriginalDutyResolver:
    """One validated numeric-unit/native-duty mapping, not a donor role claim.

    Native lifecycle production still supports uniform whole units only. This
    resolver does not manufacture mixed temporal evidence or actor identity.
    Its owner validates and freezes the entire native programme before use.
    """
    def __init__(self,program):
        self.program=program
        self.bindings={b['numeric_original_unit_id']:b for b in program.get('original_unit_bindings',())}
        if len(self.bindings)!=len(program.get('original_unit_bindings',())):
            raise ValueError('original numeric/native duty unit mapping is duplicated')

    def require_query_origin(self,state):
        from .group_scene_camera_queries import GroupSceneRoleQuery
        if isinstance(state,GroupSceneRoleQuery):
            identity,epoch,first,end,role=(state.original_unit_id,state.epoch_index,state.first_output_frame,
                state.end_output_frame,None if state.role is None else state.role.value)
            claimed=state.original_role_program
        elif isinstance(state,dict):
            p=state.get('key_preimage',state)
            if 'original_unit_id' not in p and 'whole_camera_sha256' not in p:return
            identity,epoch,first,end,role=(p.get('original_unit_id'),p.get('original_epoch_index'),
                p.get('first'),p.get('end'),p.get('role'))
            claimed=state.get('original_role_program')
        else:return
        binding=self.bindings.get(identity) if isinstance(identity,str) else None
        if (binding is None or self.program.get('contract')!=GLOBAL_CONTRACT
                or binding['original_plan_identity']!=self.program['original_owned_programme_plan_sha256']
                or claimed is not None and family.digest(claimed)!=family.digest(binding['original_role_program'])
                or type(epoch) is not int or not 0<=epoch<len(binding['original_role_program']['epochs'])):
            raise ValueError('query original role programme is not bound to this external native duty original')
        row=binding['original_role_program']['epochs'][epoch]
        if (type(first) is not int or type(end) is not int
                or not row['first_output_frame']<=first<end<=row['end_output_frame'] or role!=row['role']):
            raise ValueError('query original epoch address changed its external native duty scope')


def _require_query_origin(program,state,resolver):
    if resolver is None:resolver=NativeOriginalDutyResolver(program)
    if resolver.program is not program:
        raise ValueError('original duty resolver belongs to a different native programme')
    resolver.require_query_origin(state)


def _query_address(state):
    if hasattr(state,'role'):
        return state.first_output_frame,state.end_output_frame,state.role.value
    p=state.get('key_preimage',state)
    return p['first'],p['end'],p['role']


def original_units_for_query(program,state,*,resolver=None):
    """Dependency closure on ORIGINAL units, never candidate-owned lifecycles.

    A camera/query may be a strict subspan or cross several original units of
    the same role. This does not create/reset a branch or waive a different
    original role. Full programme/raw closure is validated by its owner first.
    """
    _require_query_origin(program,state,resolver)
    first,end,role = _query_address(state)
    if type(first) is not int or type(end) is not int or not 0<=first<end:
        raise ValueError('original duty query has invalid candidate bounds')
    units=[u for u in program['units'] if u['first_output_frame']<end and u['end_output_frame']>first]
    if (not units or units[0]['first_output_frame']>first or units[-1]['end_output_frame']<end
            or any(a['end_output_frame']!=b['first_output_frame'] for a,b in zip(units,units[1:]))):
        raise ValueError('candidate query has no complete externally frozen original duty scope')
    if any(u['role']!=role for u in units):
        raise ValueError('candidate query changed an externally frozen original role')
    return units


def duty_at(program, state, tick,*,resolver=None):
    if program is None:
        return None
    _require_query_origin(program,state,resolver)
    first,end,role = _query_address(state)
    if (type(first) is not int or type(end) is not int or type(tick) is not int
            or not 0<=first<=tick<end):
        raise ValueError('candidate frame borrowed a nonincident original duty tick')
    units = [u for u in program['units'] if u['first_output_frame']<=tick<u['end_output_frame']]
    if len(units) != 1:
        raise ValueError('candidate frame has no externally frozen original duty unit')
    unit = units[0]; low,high = unit['required_output_interval']
    if unit['role']!=role:
        raise ValueError('candidate frame changed its externally frozen original role')
    if type(tick) is not int or not low <= tick < high:
        raise ValueError('candidate frame borrowed a nonincident original duty tick')
    row = unit['rows'][tick-low]
    return dict(program_sha256=program['program_sha256'],external_original_id=program['original_plan_identity'],
        original_unit_id=unit['original_unit_id'],original_hypothesis_branches=[
            b['original_subject_hypothesis_id'] for b in unit['branches']],
        output_tick=tick, row_sha256=family.digest(row), required_part_names=row['required_part_names'])


def bind_row_obligations(program,state,tick,*,resolver=None):
    ref = duty_at(program,state,tick,resolver=resolver)
    # Unchanged formation/composition policy remains explicit, not claimed solved.
    names = None if ref is None or state.role.value not in FOCAL else tuple(ref['required_part_names'])
    return ref,names
