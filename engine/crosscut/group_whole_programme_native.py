"""One frozen whole-song graph with actual actor paths and camera recipes.

The numeric A/B control and actor/action selection use the SAME inventory.
No model runs here, UNKNOWN is a finite cost, and camera fitting ends before
search. This is an editorial proposal, never professional-quality approval.
"""
from dataclasses import dataclass, replace, asdict
import json
import math
from time import monotonic

from .group_cached_programme_inventory import (
    CachedProgrammeInventory, cached_composition_edge_cost,
    cached_programme_physical_compatibility,
)
from .group_native_motion_camera_producer import (
    NativeMotionCameraCandidate, _support,
)
from .group_native_motion_programme import realize_native_motion_programme
from .group_performer_family_owner import digest
from .group_review_projected_action import project_observed_review_face, project_observed_review_joints
from .group_review_two_shot_camera import measure_review_focal_incident, realize_review_contextual_focal_shot
from .group_scene_epoch_program import compile_common_clock_camera_queries
from .group_scene_owned_program import freeze_owned_scene_program
from .group_source_changing_owner import NativeMotionViews
from .group_whole_programme_search import ProgrammeSearchBudget, select_whole_programme
from .render import EditorialTransformKeyframe, EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT


@dataclass(frozen=True, slots=True)
class WholeNativeRecipe:
    base: object
    run: object
    side: object
    actor_path: dict | None
    actor_receipt: dict | None
    entry: tuple
    exit: tuple
    hold_uncertainty: float
    camera_work_cost: float


@dataclass(frozen=True, slots=True)
class WholeNativeInventory:
    original: object
    domain: object
    opportunities: tuple
    recipes: tuple[WholeNativeRecipe, ...]
    protected_spans: tuple
    observed_recipe_count: int
    actor_seed_truncations: int
    equivalent_recipe_aliases: int = 0
    projected_base_recipe_ids: frozenset[str] = frozenset()
    projection_budget_exhausted: bool = False
    attempted_base_recipe_ids: tuple = ()
    completed_base_recipe_ids: tuple = ()

    def evidence(self):
        return dict(contract='whole_native_actor_camera_inventory_v1',
            candidate_count=len(self.recipes), observed_recipe_count=self.observed_recipe_count,
            actor_seed_truncations=self.actor_seed_truncations,
            equivalent_recipe_aliases=self.equivalent_recipe_aliases,
            projected_base_recipe_count=len(self.projected_base_recipe_ids),
            attempted_base_recipe_projection_count=len(self.attempted_base_recipe_ids),
            completed_base_recipe_projection_count=len(self.completed_base_recipe_ids),
            actor_projection_basis='exact-native-availability-source-time-duration-strata',
            projection_budget_exhausted=self.projection_budget_exhausted,
            projection_budget_exclusion_is_actor_unsuitability=False,
            connected_actor_paths=sum(r.actor_receipt is not None for r in self.recipes),
            authored_cameras=sum(r.side is not None and r.side.camera is not None for r in self.recipes),
            camera_fit_before_selection=True, model_calls_in_graph_construction=0,
            missing_identity_is_not_candidate_veto=True, source_only_state_merging=False,
            professional_quality_approved=False)


def _queries(context, domain, original_program, plan, first, end):
    compiled=compile_common_clock_camera_queries(context=context,domain=domain,
        original_program=original_program,candidate_plan=plan,output_intervals=((first,end),))
    return {t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}


class RecipeProjectionUnavailable(ValueError):
    """Missing numeric context, not evidence that the source/run is bad."""


def _recipe_projection_plan(original, domain, base, side):
    """Readonly full-renderer camera query, NOT a publishable edit proposal.

    A new run is independent of obsolete original cell boundaries. A solo
    observation embedding may leave a one-frame ORIGINAL neighbour. Use a
    separate, globally aligned source for the UNQUERIED numerical exterior,
    without copying the old cut grid or certifying its pixels. A different
    source also prevents renderer coalescing from changing the recipe clock.
    Final joint programme construction keeps its existing two-frame contract
    and independently verifies every selected native address.
    """
    from .group_native_motion_programme import _aligned_parts, ProgrammeNativeMotionSide
    from .group_review_interval_compile import assemble_review_intervals
    if side is None:return original
    if side.phase_native_frames or base.aligned_segment is None:
        raise ValueError('readonly camera embedding needs its frozen phase-zero physical segment')
    a,b=side.first_output_frame,side.end_output_frame
    outside=next((i for i in domain.eligible_indices if i!=side.request_source_index),None)
    if outside is None or a==1 or original.output_frame_count-b==1:
        raise RecipeProjectionUnavailable('readonly recipe lacks a two-frame independent exterior context')
    rows=[]
    if a:rows.append((0,a,_aligned_parts(original,domain,ProgrammeNativeMotionSide(0,a,outside))))
    rows.append((a,b,(replace(base.aligned_segment,editorial_transform=side.camera),)))
    if b<original.output_frame_count:
        rows.append((b,original.output_frame_count,
            _aligned_parts(original,domain,ProgrammeNativeMotionSide(b,original.output_frame_count,outside))))
    return assemble_review_intervals(original,tuple(rows),maximum_segments=512)


def _flank(path, queries, ticks, views):
    return tuple((t,project_observed_review_face(path[t],queries[t],t),
        project_observed_review_joints(path[t],queries[t],t),
        None if path[t] is None else views.native_signature(queries[t],t),
        None if path[t] is None else path[t]['vector'],
        bool(path[t] is not None and path[t].get('reacquisition_ambiguous',False))) for t in ticks)


def _unknown_flank(first, end, entry):
    ticks=range(first,min(end,first+3)) if entry else range(max(first,end-3),end)
    return tuple((t,None,{},None,None,True) for t in ticks)


def _camera_key(face, target_height):
    """A declared framing target, not a fabricated observed face."""
    if face is None or face['landmarks_xy'] is None:
        return None
    zoom=min(2.,max(1.,target_height/face['height']))
    center=tuple(sum(p[k] for p in face['landmarks_xy'])/5 for k in (0,1))
    # Use the real renderer's center domain as well as its sampling canvas.
    # Off-centre heads cannot create an unrenderable key that aborts the
    # entire programme; the full support fit still tests this bounded key.
    low=max(.25,.5/zoom);high=min(.75,1-.5/zoom)
    x=min(high,max(low,center[0]))
    y=min(high,max(low,center[1]+(.5-.43)/zoom))
    distance=math.hypot(x-.5,y-.5)
    if distance>EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT:
        ratio=EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT/distance
        x=.5+(x-.5)*ratio;y=.5+(y-.5)*ratio
    return EditorialTransformKeyframe(0.,zoom,x,y,0.)


def freeze_native_projection_frontier(cached, *, available_native_addresses, fps,
        maximum_recipes=64, check=None):
    """Bound projection work, not native reuse by a detector-shot identity.

    Existing exact observations may support many different run durations.
    Availability, source/time/duration strata and immutable IDs determine this
    frontier before any actor/action/camera scoring. No new NN addresses are
    requested here. Every other physical base run stays in the whole graph.
    """
    from collections import Counter
    if (not isinstance(available_native_addresses,(set,frozenset))
            or any(not isinstance(k,tuple) or len(k)!=2 or any(type(n) is not int or n<0 for n in k)
                for k in available_native_addresses)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or type(maximum_recipes) is not int or not 1<=maximum_recipes<=512
            or check is not None and not callable(check)):
        raise ValueError('native projection needs finite exact address/clock/work bounds')
    groups={};targets=(.75,1.15,1.6,2.2,2.8,3.4,4.8)
    for i,recipe in enumerate(cached.recipes):
        if check is not None and i%128==0:check()
        run=recipe.run;grid=recipe.native_frame_indices
        if run.protected_reference or not 2<=len(grid)<=512:continue
        known=tuple((run.request_source_index,n) in available_native_addresses for n in grid)
        if not any(known):continue
        a,b=sum(known[:3]),sum(known[-3:])
        duration=(run.end_output_frame-run.first_output_frame)/fps
        stratum=(run.request_source_index,run.native_camera_run_index,
            min(range(len(targets)),key=lambda n:(abs(duration-targets[n]),n)),
            int(run.first_output_frame/(fps*8.)))
        rank=(-min(a,b),-(a+b),-sum(known)/len(known),run.identity)
        if stratum not in groups or rank<groups[stratum][0]:groups[stratum]=(rank,run.identity)
    counts=Counter();selected=[]
    while groups and len(selected)<maximum_recipes:
        if check is not None:check()
        stratum=min(groups,key=lambda s:(counts['source',s[0]],counts['time',s[3]],
            counts['duration',s[2]],counts['unit',s[0],s[1]],groups[s][0],s))
        _,identity=groups.pop(stratum);selected.append(identity)
        for key in (('source',stratum[0]),('time',stratum[3]),('duration',stratum[2]),
                ('unit',stratum[0],stratum[1])):counts[key]+=1
    return tuple(selected)


class _NativeProjectionTimeExhausted(TimeoutError):
    """Local extra-recipe budget, not a renewed/exhausted parent lease."""


def ordered_whole_actor_seeds(candidates,*,maximum_actor_seeds):
    """Declared GROUP focal hypotheses, never detector-ID or match-score rank.

    A visible foreground head is a focal-duty hypothesis, NOT a named member
    verdict. Keep up to six IVE occurrences by default, including the third
    performer. Exact geometry/content breaks ties equivariantly when detector
    IDs are permuted; no donor pose/action score spends this work allowance.
    """
    if (not isinstance(candidates,tuple) or len(candidates)>24
            or type(maximum_actor_seeds) is not int or not 1<=maximum_actor_seeds<=8):
        raise ValueError('whole GROUP seeds need their predeclared finite occurrence policy')
    def key(view):
        d=view['descriptor']
        return (-(d.face_height*d.face_width),d.head_center,
            digest(dict(vector=view['vector'],height=d.face_height,width=d.face_width,
                center=d.head_center,shape=d.pose_shape)))
    return tuple(sorted(candidates,key=key)[:maximum_actor_seeds])


def freeze_whole_native_inventory(*, context, domain, cached, views,
        protected_spans=(), maximum_actor_seeds=6, target_face_heights=(.24,.30),
        maximum_runs=32768, maximum_seconds=45., work_port=None, sensed_recipe_ids=None,
        completed_joint_recipe_ids=()):
    """Consume one completed native census, retaining missing/adverse facts.

    Raw/body/vector bindings are checked by NativeMotionViews. A legacy
    dominant-body descriptor NEVER becomes the actor behind a cached face.
    A connected path supplies both endpoints and the complete hold; cameras
    cannot reselect the actor, interpolate anatomy, or discard known support.
    """
    if (not isinstance(cached,CachedProgrammeInventory) or not isinstance(views,NativeMotionViews)
            or views.domain!=domain or context.incumbent.render_plan.sources!=domain.request_sources
            or type(maximum_actor_seeds) is not int or not 1<=maximum_actor_seeds<=8
            or not isinstance(target_face_heights,tuple) or len(target_face_heights)>3
            or any(type(h) not in (int,float) or not math.isfinite(h) or not .1<=h<=.45 for h in target_face_heights)
            or type(maximum_runs) is not int or not len(cached.recipes)<=maximum_runs<=32768
            or type(maximum_seconds) not in (int,float) or not math.isfinite(maximum_seconds)
            or not 0<maximum_seconds<=60):
        raise ValueError('whole native graph needs its frozen cache/native/source bounds')
    original=context.incumbent.render_plan
    program=freeze_owned_scene_program(context,domain)
    deadline=monotonic()+maximum_seconds
    result=[];by_identity={};observed=truncated=aliases=0
    available={(s.demand.source_index,s.demand.native_frame_index) for s in views.supplies.values()
        if s.demand.analysis_roll is None and s.observation is not None and s.appearance is not None
        and s.state=='NATIVE_ROWS_AVAILABLE'}
    # Sensing recipes allocate NN work only. They are not an eligibility
    # gate for different-duration recipes sharing actual native observations.
    projection_order=(freeze_native_projection_frontier(cached,available_native_addresses=available,
        fps=original.fps,check=views.sensors.session.budget.check) if sensed_recipe_ids is None
        else tuple(sorted(sensed_recipe_ids)))
    if (not isinstance(completed_joint_recipe_ids,tuple) or len(completed_joint_recipe_ids)>192
            or len(set(completed_joint_recipe_ids))!=len(completed_joint_recipe_ids)
            or not set(completed_joint_recipe_ids)<={r.run.identity for r in cached.recipes}
            or completed_joint_recipe_ids and sensed_recipe_ids is not None):
        raise ValueError('completed joint projection differs from the frozen physical graph')
    if completed_joint_recipe_ids:
        from .group_whole_programme_owner import _whole_native_sample_indices
        by_base={r.run.identity:r for r in cached.recipes}
        for identity in completed_joint_recipe_ids:
            recipe=by_base[identity]
            required={(recipe.run.request_source_index,recipe.native_frame_indices[i])
                for i in _whole_native_sample_indices(recipe)}
            if not required<=available:
                raise ValueError('completed joint projection has unobserved original native requirements')
    # Work just spent on BOTH cut flanks and its hold comparator must reach
    # their actor/camera projections together. Availability is not an actor
    # quality score and never erases another physical run from the graph.
    projection_order=tuple(dict.fromkeys((*completed_joint_recipe_ids,*projection_order)))[:64]
    projection_ids=frozenset(projection_order)
    projection_cutoff=False;attempted_projections=[];completed_projections=[]
    if work_port is not None:work_port.charge('states',len(cached.recipes))

    def check():
        views.sensors.session.budget.check()
        if work_port is not None:work_port._check()
        if monotonic()>=deadline:raise _NativeProjectionTimeExhausted('native extra-recipe budget reached')

    def append(recipe, *, base=False):
        nonlocal aliases
        previous=by_identity.get(recipe.run.identity)
        if previous is not None:
            # Distinct declared face-height targets can realize the EXACT
            # same support-limited camera. Preserve one physical recipe, not
            # two identical graph IDs; never pick a cheaper conflicting row.
            if base or previous!=recipe:
                raise ValueError('whole native recipe identity has conflicting physical content')
            aliases+=1
            return
        if len(result)>=maximum_runs:raise ValueError('whole native graph exceeded its frozen run allowance')
        if work_port is not None and not base:work_port.charge('states')
        by_identity[recipe.run.identity]=recipe
        result.append(recipe)

    def enrich(base):
        nonlocal observed,truncated,aliases
        check()
        run=base.run;a,b=run.first_output_frame,run.end_output_frame
        ticks=tuple(range(a,b))
        known=[t for t,n in zip(ticks,base.native_frame_indices,strict=True)
            if (run.request_source_index,n) in available]
        if not known or b-a>512:
            return
        try:plan=_recipe_projection_plan(original,domain,base,base.side)
        except RecipeProjectionUnavailable:return  # The physical base remains UNKNOWN, not vetoed.
        queries=_queries(context,domain,program,plan,a,b)
        if any(queries[t].sample_at(t)[2]!=n for t,n in zip(ticks,base.native_frame_indices,strict=True)):
            raise ValueError('native graph changed a frozen candidate sampler before scoring')
        # A headless early raster remains a NULL slot, not a permanent veto
        # on actors actually observed later in this exact native-camera unit.
        # The first available-occurrence start rule is declared before NN.
        seeds=()
        for tick in known:
            check()
            seeds=views.candidates(queries[tick],tick,association_ready=True)
            if seeds:break
        truncated+=len(seeds)>maximum_actor_seeds
        for seed in ordered_whole_actor_seeds(seeds,maximum_actor_seeds=maximum_actor_seeds):
            check()
            path,receipt=views.connected_actor_views(queries,ticks,seed)
            if receipt.get('state')!='CONNECTED_SOURCE_LOCAL_HYPOTHESIS':
                continue
            from .group_review_actor_path import validate_connected_actor_path_receipt
            validate_connected_actor_path_receipt(path,receipt)
            observed+=1
            count=sum(v is not None and not v.get('reacquisition_ambiguous',False) for v in path.values())
            unknown=1-count/len(ticks)
            cameras=[(base.side,None)]
            if base.side is not None and b-a<=160:
                faces={t:project_observed_review_face(path[t],queries[t],t) for t in (a,b-1)}
                support=tuple(_support(path[t],queries[t],t) for t in ticks)
                for height in target_face_heights:
                    check()
                    entry,exit=(_camera_key(faces[t],height) for t in (a,b-1))
                    if entry is None or exit is None:continue
                    track,measurement=realize_review_contextual_focal_shot(
                        entry_key=entry,exit_key=exit,support=support,fps=original.fps)
                    if track is not None:cameras.append((replace(base.side,camera=track),measurement))
            camera_ids=set()
            for side,camera in cameras:
                check()
                camera_payload=None if side is None or side.camera is None else asdict(side.camera)
                identity=digest(dict(base=run.identity,actor=receipt['hypothesis_sha256'],camera=camera_payload))
                if identity in camera_ids:
                    aliases+=1
                    continue  # Identical realized camera; no repeated query/reprojection.
                camera_ids.add(identity)
                actual_queries=queries
                actual_path=path
                camera_cost=0.
                if camera is not None:
                    authored=_recipe_projection_plan(original,domain,base,side)
                    actual_queries=_queries(context,domain,program,authored,a,b)
                    actual_path=views.reproject_selected_actor_path(actual_queries,path)
                    camera_cost=.10*max(k.zoom-1 for k in side.camera.keyframes)+(
                        3.*camera['maximum_camera_vector_step']+camera['maximum_camera_vector_second_difference'])
                selected_run=replace(run,identity=identity,actor_path_identity=receipt['hypothesis_sha256'],
                    camera_recipe_identity=digest(camera_payload),
                    unary_cost=run.unary_cost+camera_cost)
                append(WholeNativeRecipe(base,selected_run,side,actual_path,receipt,
                    _flank(actual_path,actual_queries,ticks[:3],views),
                    _flank(actual_path,actual_queries,ticks[-3:],views),unknown,camera_cost))
    with views.native_signature_scope():
        # Construct the COMPLETE physical graph before optional projections.
        # A local camera/association cutoff must not discard every settled
        # observation, omit later sources, or reset/retry the request.
        for base in cached.recipes:
            views.sensors.session.budget.check()
            if work_port is not None:work_port._check()
            run=base.run;a,b=run.first_output_frame,run.end_output_frame
            append(WholeNativeRecipe(base,run,base.side,None,None,
                _unknown_flank(a,b,True),_unknown_flank(a,b,False),1.,0.),base=True)
        by_base={r.run.identity:r for r in cached.recipes}
        try:
            for identity in projection_order:
                check();attempted_projections.append(identity)
                enrich(by_base[identity]);completed_projections.append(identity)
        except _NativeProjectionTimeExhausted:
            # Complete, verified alternatives already appended are retained;
            # unfinished ones remain UNKNOWN. Strict parent/storage/source
            # failures are not caught and still invalidate the entire result.
            projection_cutoff=True
    return WholeNativeInventory(original,domain,cached.opportunities,tuple(result),
        protected_spans,observed,truncated,aliases,projection_ids,projection_cutoff,
        tuple(attempted_projections),tuple(completed_projections))


def whole_native_edge_cost(inventory, *, actor_action_enabled):
    """Identical uncertainty/geometry/cadence, differing ONLY observed losses.

    SFace is an uncalibrated anonymous relation, not a member probability.
    Motion uses the exact selected, owned joints and physical tangents. A
    missing face/wrist/vector stays missing at its declared request address.
    """
    if not isinstance(inventory,WholeNativeInventory) or type(actor_action_enabled) is not bool:
        raise ValueError('whole native costs need one explicit frozen A/B condition')
    by_id={r.run.identity:r for r in inventory.recipes}
    common=cached_composition_edge_cost(tuple({r.base.run.identity:r.base for r in inventory.recipes}.values()))
    from functools import lru_cache
    @lru_cache(maxsize=32768)
    def incident(left_id,right_id):
        a,b=by_id[left_id],by_id[right_id]
        left,right=a.exit,b.entry
        if min(len(left),len(right))<2:
            return 1.,1.,1.
        # A wholly UNKNOWN flank has no face, physical tangent or identity
        # pair. Avoid rebuilding the same six-joint numeric measurement for
        # hundreds of thousands of these seams; its upper costs are EXACTLY
        # the same constants, not a faster substitute for observed evidence.
        if any(all(r[1] is None and not r[2] and r[4] is None for r in flank)
                for flank in (left,right)):
            return 1.,1.,1.
        measured=measure_review_focal_incident(left_faces=tuple(r[1] for r in left),
            right_faces=tuple(r[1] for r in right),left_points=tuple(r[2] for r in left),
            right_points=tuple(r[2] for r in right),left_ticks=tuple(r[0] for r in left),
            right_ticks=tuple(r[0] for r in right),fps=inventory.original.fps,
            left_native=tuple(r[3] for r in left),right_native=tuple(r[3] for r in right))
        av,bv=left[-1][4],right[0][4]
        identity=1.
        if av is not None and bv is not None and not left[-1][5] and not right[0][5]:
            from .group_review_native_window import cosine
            similarity=cosine(av,bv)
            identity=1. if similarity is None else (1-similarity)/2
        return measured['face_cost_interval'][1],sum(v[1] for v in measured['action_cost_intervals'])/3,identity

    def edge(history, incoming):
        b=by_id[incoming.identity]
        cost=common(tuple(by_id[r.identity].base.run for r in history),b.base.run)
        # Both conditions carry the SAME missing-evidence upper cost. Risk is
        # a unary term on every run, not an extra B-only penalty or a free
        # opening. All-UNKNOWN controls must therefore have identical plans.
        cost+=.35*b.hold_uncertainty*(
            b.run.end_output_frame-b.run.first_output_frame)/inventory.original.fps
        if not history:return cost
        face,action,identity=incident(history[-1].identity,incoming.identity)
        # Common cached geometry is a weak source-composition prior. This
        # extra term describes the actual prefit cameras, for BOTH conditions.
        cost+=face
        # A uses the predeclared upper losses at the same requested seams;
        # B may replace them ONLY with the bounded observed losses. Missing
        # joints, an ambiguous identity or absent physical tangents already
        # have upper loss 1 in incident(), for both A and B. No reward is
        # obtained by hiding a difficult measurement or deleting a bad joint.
        cost+=2.*(action if actor_action_enabled else 1.)
        cost+=2.*(identity if actor_action_enabled else 1.)
        return cost
    return edge


def select_whole_native_programme(*, inventory, work_port=None, beam_width=8, maximum_seconds=30.):
    by_id={r.run.identity:r for r in inventory.recipes}
    base_physics=cached_programme_physical_compatibility(original=inventory.original,domain=inventory.domain,
        recipes=tuple({r.base.run.identity:r.base for r in inventory.recipes}.values()))
    def compatible(history,incoming):
        return base_physics(tuple(by_id[r.identity].base.run for r in history),by_id[incoming.identity].base.run)
    budget=ProgrammeSearchBudget(beam_width=beam_width,maximum_expansions=1000000,
        maximum_seconds=maximum_seconds)
    args=dict(opportunities=inventory.opportunities,runs=tuple(r.run for r in inventory.recipes),
        protected_spans=inventory.protected_spans,budget=budget,physically_compatible=compatible,work_port=work_port)
    control=select_whole_programme(**args,transition_cost=whole_native_edge_cost(inventory,actor_action_enabled=False))
    selected=select_whole_programme(**args,transition_cost=whole_native_edge_cost(inventory,actor_action_enabled=True))
    return control,selected


def materialize_whole_native_selection(*,context,inventory,selection,views):
    """Exact selected source/native/actor/camera -> complete normal producer.

    No reconstruction from choices, no post-search camera fit, no salience
    fallback and no second native completion. The final full sampler must
    equal EVERY frozen selected run, including native-unit coalescing.
    """
    from .group_joint_bundle_metadata import SelectedNativeActorFrame,SelectedNativeMotionShot
    from .group_native_motion_metadata import produce_complete_native_motion_metadata
    from .group_review_interval_compile import output_offsets
    from .render_boundary_clock import native_frame_clocks_for_output
    by_id={r.run.identity:r for r in inventory.recipes}
    selected=tuple(by_id[r.identity] for r in selection.runs)
    sides=tuple(r.side for r in selected if r.side is not None)
    if not sides:raise ValueError('whole native selection has no mutable source programme')
    result=realize_native_motion_programme(original=inventory.original,domain=inventory.domain,sides=sides,
        protected_output_intervals=inventory.protected_spans,selection_scope='variable_programme')
    offsets=output_offsets(result.render_plan)
    clocks=native_frame_clocks_for_output(result.render_plan,inventory.domain.clocks_by_request,
        range(result.render_plan.output_frame_count))
    shots=[]
    for recipe in selected:
        run=recipe.run;a,b=run.first_output_frame,run.end_output_frame
        if tuple(clocks[t].native_frame_index for t in range(a,b))!=recipe.base.native_frame_indices or any(
                clocks[t].source_index!=run.request_source_index or clocks[t].terminal_clone for t in range(a,b)):
            raise ValueError('whole native selected renderer differs from the frozen graph sampler')
        if recipe.side is None:continue  # Exact protected reference, never silently made mutable.
        frames=[]
        for tick in range(a,b):
            view=None if recipe.actor_path is None else recipe.actor_path[tick]
            frames.append(SelectedNativeActorFrame(tick,run.request_source_index,clocks[tick].native_frame_index,
                None if view is None else view['descriptor'].frame_local_head_id,
                None if view is None else digest(view['row']['native_observation'])))
        clock=inventory.domain.clocks_by_request[run.request_source_index]
        cuts=tuple(math.ceil(t*float(clock.frame_rate)-1e-9)
            for t in inventory.domain.cuts_for_request(run.request_source_index))
        unit=(0,*cuts,clock.frame_count)
        shots.append(SelectedNativeMotionShot(a,b,run.request_source_index,
            (unit[run.native_camera_run_index],unit[run.native_camera_run_index+1]),recipe.side.proposal_role,
            recipe.side.phase_native_frames,run.actor_path_identity,tuple(frames),
            tuple(range(offsets.index(a),offsets.index(b)))))
    proposal=NativeMotionCameraCandidate(result.render_plan,result.shot_clock_context,tuple(shots),
        json.dumps(dict(contract='whole_native_programme_selection_v1',inventory=inventory.evidence(),
            search=selection.to_dict(),camera_refit_after_selection=False,quality_approved=False),allow_nan=False),
        selection_scope='variable_programme')
    return produce_complete_native_motion_metadata(context=context,selected=proposal,domain=inventory.domain,
        views=views,protected_output_intervals=inventory.protected_spans)
