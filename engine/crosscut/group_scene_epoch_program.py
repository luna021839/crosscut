"""Original role sidecars over one complete source/camera programme.

Roles can change inside a continuous camera without becoming new edit cuts.
Readonly role queries use the full selected renderer without a cell-splitting
scaffold or a fresh per-epoch camera origin. This compiler neither senses
original body lifecycles nor admits donors, completes bundles or grants global
parent/final authority. Those remain separate obligations of the owner.
"""
from dataclasses import dataclass, fields
import json

from .group_actual_shots import (VisibleGroupShot,assert_native_cut_events_preserved,
    visible_group_cut_inventory,visible_group_shots)
from .group_joint_native_preparation import _plan_identity
from .group_joint_source_camera import GroupOutputRole
from .group_scene_camera_queries import (GroupWholeCameraObservation,GroupSceneRoleQuery,
    PreparedGroupCameraQueries)
from .group_scene_intent import CONTRACT as INTENT_CONTRACT,role_from_cells,validate_intent
from .render import RenderPlan,segment_frame_counts,_base_crop_values,_boundary_look_values
from .render_boundary_clock import native_frame_clocks_for_output

CONTRACT='original_output_role_epoch_program_v1'


def _freeze(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _cell_offsets(plan):
    offsets=[0]
    for count in segment_frame_counts(plan):offsets.append(offsets[-1]+count)
    return offsets


def _role_epochs(cells,first,end):
    epochs=[]
    for cell in cells:
        try:role=role_from_cells([cell]).value;reason=None
        except ValueError as exc:role=None;reason=str(exc)
        low=max(cell['first_output_frame'],first);high=min(cell['end_output_frame'],end)
        if epochs and (epochs[-1]['role'],epochs[-1]['unresolved_reason'])==(role,reason):
            epochs[-1]['end_output_frame']=high
            epochs[-1]['original_segment_indices'].append(cell['segment_index'])
        else:
            epochs.append(dict(first_output_frame=low,end_output_frame=high,role=role,
                unresolved_reason=reason,original_segment_indices=[cell['segment_index']]))
    return epochs


def _original_role_program(identity,shot,cells):
    return dict(contract=CONTRACT,original_plan_identity=identity,original_shot=shot,
        original_output_cells=cells,epochs=_role_epochs(cells,shot['first_output_frame'],shot['end_output_frame']),
        original_output_cell_roles_frozen=True,original_native_part_lifecycle_frozen=False,
        role_is_not_physical_cast_or_identity=True,frozen_before_source_camera_search=True,
        epoch_boundaries_are_not_edit_cuts=True,candidate_admitted=False,quality_approved=False)


def validate_original_role_epochs(program,*,original_plan=None):
    """Received role-policy closure, not fresh observation of original pixels.

    Original declared labels remain input evidence, not detected cast/duties.
    A selected parent must separately rederive the exact sidecar from its
    external incumbent. Unresolved roles cannot be promoted or discarded.
    """
    try:
        identity=program['original_plan_identity'];shot=program['original_shot'];cells=program['original_output_cells']
        if (not isinstance(identity,str) or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity)
                or not isinstance(shot,dict) or set(shot)!={'first_output_frame','end_output_frame','frame_count',
                    'source_index','first_segment_index','last_segment_index'}
                or any(type(value) is not int for value in shot.values())
                or not 0<=shot['first_output_frame']<shot['end_output_frame']
                or shot['frame_count']!=shot['end_output_frame']-shot['first_output_frame']
                or shot['source_index']<0 or not 0<=shot['first_segment_index']<=shot['last_segment_index']
                or not isinstance(cells,list) or not cells
                or any(not isinstance(cell,dict) or set(cell)!={'segment_index','first_output_frame','end_output_frame',
                    'shot_type','alignment_subject','face_applied'} for cell in cells)
                or [cell['segment_index'] for cell in cells]!=list(range(shot['first_segment_index'],shot['last_segment_index']+1))
                or any(type(cell[key]) is not int for cell in cells for key in
                    ('segment_index','first_output_frame','end_output_frame'))
                or any(not 0<=cell['first_output_frame']<cell['end_output_frame'] for cell in cells)
                or any(a['end_output_frame']!=b['first_output_frame'] for a,b in zip(cells,cells[1:]))
                or cells[0]['first_output_frame']>shot['first_output_frame']
                or cells[-1]['end_output_frame']<shot['end_output_frame']
                or any(cell['first_output_frame']>=shot['end_output_frame'] or
                    cell['end_output_frame']<=shot['first_output_frame'] for cell in cells)):
            raise ValueError('original role programme lost its external output-cell scope')
        if _freeze(program)!=_freeze(_original_role_program(identity,shot,cells)):
            raise ValueError('original role programme changed role epochs or promoted authority')
        if original_plan is not None:
            if not isinstance(original_plan,RenderPlan) or identity!=_plan_identity(original_plan):
                raise ValueError('original role programme changed its original renderer identity')
            offsets=_cell_offsets(original_plan)
            if shot['last_segment_index']>=len(original_plan.segments):
                raise ValueError('original role programme borrowed an absent output cell')
            if any((cell['first_output_frame'],cell['end_output_frame'])!=tuple(offsets[cell['segment_index']:cell['segment_index']+2])
                    or original_plan.segments[cell['segment_index']].source_index!=shot['source_index'] for cell in cells):
                raise ValueError('original role programme changed its original source/cell clock')
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('original role programme lost its received schema') from exc
    return program


def freeze_original_role_epochs(context,shot):
    """Read external original declarations before any donor observations.

    Unresolved declarations survive as unresolved. Required/context native
    parts are deliberately NOT inferred from these output-cell role labels.
    A subsequent original sensor programme must freeze those lifecycles.
    """
    bundle=context.incumbent;plan=bundle.render_plan
    if (not isinstance(plan,RenderPlan) or not isinstance(shot,VisibleGroupShot)
            or not 0<=shot.first_output_frame<shot.end_output_frame<=plan.output_frame_count
            or len(bundle.visual_timeline.choices)!=len(plan.segments)
            or len(bundle.face_match_plan.segments)!=len(plan.segments)):
        raise ValueError('original role epochs need the complete external output inventory')
    offsets=_cell_offsets(plan)
    indices=tuple(i for i in range(len(plan.segments))
        if offsets[i]<shot.end_output_frame and offsets[i+1]>shot.first_output_frame)
    if (not indices or (indices[0],indices[-1])!=(shot.first_segment_index,shot.last_segment_index)
            or any(plan.segments[i].source_index!=shot.source_index for i in indices)):
        raise ValueError('original role epoch shot differs from its external output cells')
    cells=[]
    for index in indices:
        choice=bundle.visual_timeline.choices[index];face=bundle.face_match_plan.segments[index]
        cell=dict(segment_index=index,first_output_frame=offsets[index],end_output_frame=offsets[index+1],
            shot_type=getattr(choice.shot_type,'value',choice.shot_type),
            alignment_subject=face.alignment_subject,face_applied=face.applied)
        cells.append(cell)
    return _original_role_program(_plan_identity(plan),shot.to_dict(),cells)


def _pixel_parameters(segment,time):
    camera=segment.editorial_transform
    key=camera.evaluate(time) if camera is not None else None
    return (*_base_crop_values(segment,time),*_boundary_look_values(segment.boundary_look_track,time),
        segment.look_adjustment,
        None if key is None else (key.zoom,key.center_x,key.center_y,key.roll_degrees),
        None if camera is None else camera.effective_sampling_canvas_zoom,
        segment.outgoing_overlap_transform)


def _verify_candidate(original,candidate,domain,shot):
    # Output filename is not a pixel/audio parameter. Every other plan header,
    # original cell bounds and complete hard-cut cadence remain externally owned.
    if (not isinstance(candidate,RenderPlan)
            or any(getattr(original,f.name)!=getattr(candidate,f.name) for f in fields(RenderPlan)
                if f.name not in ('segments','transitions','output'))
            or candidate.output_frame_count!=original.output_frame_count
            or len(candidate.segments)!=len(original.segments)
            or segment_frame_counts(candidate)!=segment_frame_counts(original)
            or any((a.master_start,a.duration)!=(b.master_start,b.duration)
                for a,b in zip(original.segments,candidate.segments,strict=True))):
        raise ValueError('role epoch candidate changed the original output clock/canvas/cell inventory')
    before=visible_group_cut_inventory(original,domain);after=visible_group_cut_inventory(candidate,domain)
    if tuple((e.first_incoming_output_frame,e.kind) for e in before)!=tuple(
            (e.first_incoming_output_frame,e.kind) for e in after):
        raise ValueError('role epoch boundary cannot change the actual cut inventory')
    assert_native_cut_events_preserved(original,candidate,domain)
    starts=_cell_offsets(original)
    outside=tuple(t for t in range(original.output_frame_count)
        if not shot.first_output_frame<=t<shot.end_output_frame)
    old_samples=native_frame_clocks_for_output(original,domain.clocks_by_request,outside)
    new_samples=native_frame_clocks_for_output(candidate,domain.clocks_by_request,outside)
    index=0
    for tick in outside:
        while starts[index+1]<=tick:index+=1
        a,b=old_samples[tick],new_samples[tick]
        if (a.source_index!=b.source_index or a.native_frame_index!=b.native_frame_index
                or a.terminal_clone!=b.terminal_clone
                or abs(a.source_request_seconds-b.source_request_seconds)>1e-9):
            raise ValueError('role epoch candidate changed a source occurrence outside the mutable shot')
        time=(tick-starts[index])/original.fps
        if _pixel_parameters(original.segments[index],time)!=_pixel_parameters(candidate.segments[index],time):
            raise ValueError('role epoch candidate changed pixels outside the mutable shot')
    # These immutable full shots are observation sidecars, not additional
    # authority for the old three/four-state positional repair assembler.
    return candidate


@dataclass(frozen=True,slots=True)
class GroupRoleEpochView:
    state: GroupSceneRoleQuery
    intent_json: str

    @property
    def intent(self):return json.loads(self.intent_json)

    def to_dict(self):
        return dict(state=self.state.to_dict(),intent=self.intent)


@dataclass(frozen=True,slots=True)
class GroupRoleEpochCompilation:
    plan: RenderPlan
    camera: GroupWholeCameraObservation
    views: tuple[GroupRoleEpochView,...]
    original_program_json: str

    def to_dict(self):
        return dict(contract='compiled_original_role_camera_queries_v2',
            original_role_program=json.loads(self.original_program_json),
            selected_plan_identity=_plan_identity(self.plan),
            camera=self.camera.to_dict(),views=[v.to_dict() for v in self.views],
            render_cells_split_for_observation=False,
            epoch_boundaries_are_not_edit_cuts=True,selected_render_topology_unchanged=True,
            original_native_part_lifecycle_frozen=False,global_mutable_program_authority=False,
            candidate_admitted=False,full_bundle_created=False,quality_approved=False)


def compile_role_epoch_views(*,context,domain,shot,original_role_program,candidate_plan=None):
    """Use exact original roles with the selected whole camera's global phase.

    The candidate may change source/camera only inside the externally declared
    original actual shot. Role changes cannot create extra source switches.
    Scoped model sensing and completion are not performed by this compiler.
    """
    original=context.incumbent.render_plan
    expected=freeze_original_role_epochs(context,shot)
    if _freeze(original_role_program)!=_freeze(expected):
        raise ValueError('frozen original role programme differs from external incumbent')
    if shot not in visible_group_shots(original,domain):
        raise ValueError('original role programme is not an externally observed actual shot')
    if any(e['role'] is None for e in expected['epochs']):
        raise ValueError('unresolved original role cannot become a donor-selected context duty')
    selected=_verify_candidate(original,original if candidate_plan is None else candidate_plan,domain,shot)
    actual=next(s for s in visible_group_shots(selected,domain)
        if (s.first_output_frame,s.end_output_frame)==(shot.first_output_frame,shot.end_output_frame))
    camera=GroupWholeCameraObservation.from_plan(plan=selected,domain=domain,physical_shot=actual)
    retained=expected['original_output_cells']
    views=[]
    for epoch_index,epoch in enumerate(expected['epochs']):
        first,end=epoch['first_output_frame'],epoch['end_output_frame']
        intent=dict(contract=INTENT_CONTRACT,role=epoch['role'],first_output_frame=first,end_output_frame=end,
            cells=[c for c in retained if c['first_output_frame']<end and c['end_output_frame']>first],
            retained_output_cells=retained,endpoint_scope='whole',required_output_interval=[first,end],
            fixed_nonincident_pixels_unchanged=False,frozen_before_source_camera_search=True,
            incident_boundary_does_not_own_whole_shot=True,quality_approved=False,
            epoch_fragment_origin=dict(contract=CONTRACT,original_plan_identity=expected['original_plan_identity'],
                original_output_interval=[shot.first_output_frame,shot.end_output_frame],
                original_output_cells=expected['original_output_cells'],
                original_native_part_lifecycle_frozen=False,epoch_boundaries_are_not_edit_cuts=True))
        validate_intent(intent,role=epoch['role'],first=first,end=end)
        state=GroupSceneRoleQuery(camera,_freeze(expected),first,end,epoch_index,GroupOutputRole(epoch['role']))
        views.append(GroupRoleEpochView(state,_freeze(intent)))
    return GroupRoleEpochCompilation(selected,camera,tuple(views),_freeze(expected))


@dataclass(frozen=True,slots=True)
class GroupCommonClockCameraQueries:
    """Original roles and selected physical cameras need not share topology.

    Readonly common-clock queries, not an accepted edit, a mutable allocation
    or a native motion receipt. Unknown original roles are retained verbatim.
    """
    plan: RenderPlan
    cameras: tuple[GroupWholeCameraObservation,...]
    queries: tuple[GroupSceneRoleQuery,...]
    original_program_json: str
    output_intervals: tuple[tuple[int,int],...]

    def to_dict(self):
        return dict(contract='common_clock_original_role_physical_camera_queries_v1',
            original_owned_program=json.loads(self.original_program_json),
            selected_plan_identity=_plan_identity(self.plan),
            output_intervals=[list(span) for span in self.output_intervals],
            cameras=[camera.to_dict() for camera in self.cameras],
            queries=[dict(key=query.key,key_preimage=query.key_preimage()) for query in self.queries],
            role_epochs_are_not_physical_cuts=True,render_topology_materialized=False,
            original_native_lifecycle_frozen=False,whole_request_phase_reserves_implemented=False,
            candidate_admitted=False,quality_approved=False)


def compile_common_clock_camera_queries(*,context,domain,original_program,candidate_plan,output_intervals):
    """Intersect original roles with selected cameras without manufacturing cuts.

    One selected camera may cover several original units; a moved selected
    cut may split one original role epoch. Every query still names its ORIGINAL
    unit and epoch. Learned local source clocks are sampled by the normal full
    renderer clock helper, not replaced by zero alignment offsets. Their full
    learned authority and mutable/protected admission remain the owner work.
    """
    from .group_scene_owned_program import validate_owned_scene_program
    validate_owned_scene_program(original_program.to_dict(),context,domain)
    original=context.incumbent.render_plan
    # The source-clock authorization payload can differ in a numerical learned
    # proposal. Observing its geometry is NOT approval of that authority.
    if (not isinstance(candidate_plan,RenderPlan)
            or candidate_plan.output_frame_count!=original.output_frame_count
            or any(getattr(original,f.name)!=getattr(candidate_plan,f.name) for f in fields(RenderPlan)
                if f.name not in ('segments','transitions','output','authorized_shot_video_clocks'))):
        raise ValueError('common-clock query changed external audio/output/canvas/source headers')
    previous=-1
    if not isinstance(output_intervals,tuple) or not output_intervals or len(output_intervals)>256:
        raise ValueError('common-clock queries need an explicit bounded output scope')
    for span in output_intervals:
        if (not isinstance(span,tuple) or len(span)!=2 or any(type(t) is not int for t in span)
                or not 0<=span[0]<span[1]<=original.output_frame_count or span[0]<previous):
            raise ValueError('common-clock query scope must have ordered disjoint exact output ticks')
        previous=span[1]
    prepared=PreparedGroupCameraQueries(candidate_plan,domain)
    cameras=tuple(prepared.camera_for(shot) for shot in prepared.shots if any(
        shot.first_output_frame<end and shot.end_output_frame>first for first,end in output_intervals))
    queries=[]
    for first,end in output_intervals:
        for camera in cameras:
            for unit in original_program.units:
                if unit.first_output_frame>=min(end,camera.end_output_frame) or unit.end_output_frame<=max(first,camera.first_output_frame):
                    continue
                program=unit.role_program
                for index,epoch in enumerate(program['epochs']):
                    low=max(first,camera.first_output_frame,epoch['first_output_frame'])
                    high=min(end,camera.end_output_frame,epoch['end_output_frame'])
                    if low>=high:continue
                    queries.append(GroupSceneRoleQuery(camera,unit.original_role_program_json,low,high,index,
                        None if epoch['role'] is None else GroupOutputRole(epoch['role'])))
    queries=tuple(sorted(queries,key=lambda q:q.first_output_frame))
    for first,end in output_intervals:
        rows=[q for q in queries if first<=q.first_output_frame<q.end_output_frame<=end]
        if (not rows or rows[0].first_output_frame!=first or rows[-1].end_output_frame!=end
                or any(a.end_output_frame!=b.first_output_frame for a,b in zip(rows,rows[1:]))):
            raise ValueError('common-clock queries omitted or duplicated an original output duty tick')
    return GroupCommonClockCameraQueries(candidate_plan,cameras,queries,_freeze(original_program.to_dict()),output_intervals)
