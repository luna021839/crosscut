"""Normal GROUP whole-programme owner, on the installed controller lease.

This is not a second app/render pipeline. Frozen musical/native opportunities,
one native completion, paired actor/action selection and complete metadata all
belong to the request's existing selected_completion phase. No quality stamp
is manufactured; the normal final renderer and decoded QA remain required.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from time import monotonic

from .group_cached_native_observations import CachedNativeInventory,load_cached_native_inventory
from .group_cached_programme_inventory import build_cached_programme_inventory
from .group_performer_family_owner import installed_pose_model
from .group_source_domain import GroupSourceDomain
from .group_whole_programme_search import programme_opportunities


@dataclass(frozen=True,slots=True)
class WholeProgrammeContext:
    incumbent: object
    domain: GroupSourceDomain
    token: object
    deadline: float


def freeze_whole_sensing_recipes(cached, *, fps, maximum_recipes=64):
    """A declared whole-song sensing frontier, not an old-cut patch scope.

    Keep ALL physical base runs in search. This finite frontier determines
    only where extra NN/actor/camera work can be spent in the existing lease.
    Source-balanced bit reversal spreads probes over the complete song.
    """
    from collections import defaultdict
    if (type(maximum_recipes) is not int or not 1<=maximum_recipes<=512
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('whole sensing frontier needs its explicit work bound')
    by_source=defaultdict(dict)
    for recipe in cached.recipes:
        run=recipe.run
        if run.protected_reference:continue
        group=by_source[run.request_source_index]
        previous=group.get(run.first_output_frame)
        rank=(abs((run.end_output_frame-run.first_output_frame)/fps-2.2),run.identity)
        if previous is None or rank<previous[0]:group[run.first_output_frame]=(rank,recipe)
    ordered={}
    for source,rows in by_source.items():
        values=[v[1] for _,v in sorted(rows.items())]
        width=max(1,(len(values)-1).bit_length())
        ordered[source]=[v for _,v in sorted(enumerate(values),
            key=lambda p:(int(format(p[0],f'0{width}b')[::-1],2),p[0]))]
    selected=[];identities=set();source_counts=defaultdict(int)
    def add(recipe):
        if recipe.run.identity in identities or len(selected)>=maximum_recipes:return
        identities.add(recipe.run.identity);selected.append(recipe)
        source_counts[recipe.run.request_source_index]+=1
    # Retain source coverage before spending the rest on coherent incidents.
    for source,rows in sorted(ordered.items()):
        if rows:add(rows[0])
    by_span=defaultdict(list)
    for recipe in cached.recipes:
        if not recipe.run.protected_reference:
            by_span[recipe.run.first_output_frame,recipe.run.end_output_frame].append(recipe)
    by_first=defaultdict(list)
    for (a,b),rows in by_span.items():
        if len({r.run.request_source_index for r in rows})>=2:
            by_first[a].append((b,rows))
    anchors=sorted(by_first)
    width=max(1,(len(anchors)-1).bit_length())
    anchors=[v for _,v in sorted(enumerate(anchors),
        key=lambda p:(int(format(p[0],f'0{width}b')[::-1],2),p[0]))]
    for anchor in anchors:
        cursor=anchor
        # Three adjacent spans expose entry/return seams across broadcasters.
        # Independent time-spread shots rarely have two observed sides of any
        # selectable cut, even when their individual holds are well observed.
        for _ in range(3):
            choices=by_first.get(cursor,())
            if not choices or len(selected)>=maximum_recipes:break
            end,rows=min(choices,key=lambda p:(abs((p[0]-cursor)/fps-2.2),
                -len({r.run.request_source_index for r in p[1]}),p[0]))
            for recipe in sorted(rows,key=lambda r:(source_counts[r.run.request_source_index],
                    r.run.request_source_index,r.run.identity)):
                add(recipe)
            cursor=end
        if len(selected)>=maximum_recipes:break
    for i in range(max(map(len,ordered.values()),default=0)):
        for source,rows in sorted(ordered.items()):
            if i<len(rows):add(rows[i])
    return tuple(selected)


def freeze_whole_native_demands(cached, *, maximum_demands=16384, sensing_recipes=None):
    """Outcome-independent finite source/time census, BEFORE the first NN.

    Each recipe requests both three-frame flanks, its midpoint and exact
    frozen-grid bridge addresses with at most 12 native frames between them.
    Sparse endpoints cannot become a connected path across an unobserved gap.
    When the
    physical union is too large, a deterministic source-balanced/time-spread
    subset is declared once. Unrequested addresses remain UNKNOWN, not bad
    media. No new inputs are picked after inspecting appearance or action.
    """
    from collections import defaultdict
    from .group_source_changing_owner import NativeMotionDemand
    if type(maximum_demands) is not int or not 1<=maximum_demands<=16384:
        raise ValueError('whole native input census needs a finite declared bound')
    physical={}
    for recipe in cached.recipes if sensing_recipes is None else sensing_recipes:
        run=recipe.run;n=len(recipe.native_frame_indices)
        selected_indices={0,min(1,n-1),min(2,n-1),n//2,max(0,n-3),max(0,n-2),n-1}
        anchor=0
        for i,native in enumerate(recipe.native_frame_indices):
            if native-recipe.native_frame_indices[anchor]>12:
                # Admit an actual frozen output sample, never a made-up raw
                # interpolation. An inherently wider sampler remains UNKNOWN.
                selected_indices.add(max(anchor+1,i-1))
                anchor=max(anchor+1,i-1)
        indices=sorted(selected_indices)
        for i in indices:
            key=(run.request_source_index,recipe.native_frame_indices[i],None)
            demand=NativeMotionDemand(key[0],key[1],run.first_output_frame+i,
                f'whole-source-{key[0]}','moving_closeup',require_appearance=True)
            prior=physical.get(key)
            if prior is None or demand.output_tick<prior.output_tick:physical[key]=demand
    by_source=defaultdict(list)
    for key in sorted(physical):by_source[key[0]].append(physical[key])
    def order(rows):
        width=max(1,(len(rows)-1).bit_length())
        return sorted(enumerate(rows),key=lambda p:(int(format(p[0],f'0{width}b')[::-1],2),p[0]))
    ordered={source:order(rows) for source,rows in by_source.items()}
    selected=[]
    for i in range(max(map(len,ordered.values()),default=0)):
        for source,rows in sorted(ordered.items()):
            if i<len(rows) and len(selected)<maximum_demands:selected.append(rows[i][1])
    return tuple(selected),dict(physical_union_count=len(physical),declared_native_inputs=len(selected),
        unrequested_native_inputs=len(physical)-len(selected),
        truncation_policy='pre-observation-source-balanced-bit-reversed-time-order',
        selection_scores_used=False)


def freeze_whole_completion_packages(sensing, demands):
    """Keep each predeclared actor hold coherent in one consumed NN pass."""
    from .group_native_motion_completion_schedule import NativeMotionCompletionPackage
    available={d.native_key for d in demands}
    packages=[]
    for recipe in sensing:
        source=recipe.run.request_source_index
        keys=tuple(dict.fromkeys((source,n,None) for n in recipe.native_frame_indices
            if (source,n,None) in available))
        if keys:packages.append(NativeMotionCompletionPackage(recipe.run.identity,keys))
    if len(packages)>64:raise ValueError('whole native holds exceeded their frozen completion-package bound')
    return tuple(packages)


def _opportunities(original, domain, rhythm, section_plan, protected):
    from .group_review_interval_compile import output_offsets
    events=set()
    for source in domain.eligible_indices:
        local=domain.eligible_index(source);warp=domain.source_timewarps[local]
        for seconds in domain.cuts_for_request(source):
            master=(seconds-domain.audio_offsets[local] if warp is None else warp.source_to_master(seconds))
            tick=round((master-original.master_audio_start)*original.fps)
            events.update(t for t in (tick-1,tick,tick+1) if 0<t<original.output_frame_count)
    common=original.master_audio_start
    return programme_opportunities(output_frame_count=original.output_frame_count,fps=original.fps,
        beat_seconds=tuple(t-common for t in rhythm.beat_times_seconds),
        phrase_seconds=tuple(p.start_seconds-common for p in rhythm.phrase_candidates),
        section_seconds=tuple(s.start_seconds-common for s in section_plan.sections),
        native_cut_ticks=tuple(sorted(events)),protected_spans=protected,
        protected_recipe_boundaries=tuple(t for t in output_offsets(original) if any(a<t<b for a,b in protected)))


def produce_whole_programme(*,controller,request,eligible_indices,offsets,timewarps,scene_cut_times,
        rhythm,section_plan,domain=None,cached_inventories=None,assets=None,
        recorded_index=None,recorded_bindings=None,maximum_new_native_inputs=512):
    """Called only inside the SAME normal selected_completion phase.

    Explicit inputs are also usable by a real-media diagnostic; they do not
    replace normal validation, raw-parent binding or the controller itself.
    """
    from types import SimpleNamespace
    from .request_work_lease import current_request_work_phase
    from .group_joint_observations import GroupJointObservationSession,_file_identity
    from .group_joint_source_camera import JointGroupWorkBudget
    from .group_source_changing_owner import NativeFamilySensors,NativeMotionViews,installed_native_assets
    from .group_review_actor_appearance import register_review_sface
    from .face_identity import _default_model_path
    from .render_frame_schedule import probe_render_frame_clock
    from .group_whole_programme_native import (
        freeze_whole_native_inventory,select_whole_native_programme,materialize_whole_native_selection,
    )
    port=current_request_work_phase()
    if type(maximum_new_native_inputs) is not int or not 0<=maximum_new_native_inputs<=512:
        raise ValueError('whole programme new native inputs need the original finite allowance')
    if port is None or port.name!='selected_completion' or controller._request_work_lease.active is not port:
        raise ValueError('whole GROUP production needs its existing live selected_completion lease')
    def remaining():
        port._check()
        return max(.01,port.deadline-float(controller.runtime._clock()))
    old=controller.incumbent
    started=stage_started=monotonic();stage_seconds={}
    from .pipeline import _validate_review_first_group_bundle
    _validate_review_first_group_bundle(old,eligible_indices=eligible_indices,offsets=offsets,timewarps=timewarps,
        common_start=old.render_plan.master_audio_start,phrase_times=old.phrase_times)
    token=controller.transaction_token()
    identities=tuple(_file_identity(old.render_plan.sources[i]) for i in eligible_indices)
    if domain is None:
        clocks=[]
        for source in eligible_indices:
            port.charge('source_probes')
            clocks.append(probe_render_frame_clock(old.render_plan.sources[source],
                timeout_seconds=min(10.,remaining())))
        domain=GroupSourceDomain(old.render_plan.sources,tuple(eligible_indices),tuple(offsets),tuple(timewarps),
            tuple(tuple(c) for c in scene_cut_times),tuple(clocks))
    if (domain.request_sources!=old.render_plan.sources or domain.eligible_indices!=tuple(eligible_indices)
            or domain.audio_offsets!=tuple(offsets) or domain.source_timewarps!=tuple(timewarps)
            or domain.scene_cut_times!=tuple(tuple(c) for c in scene_cut_times)):
        raise ValueError('whole GROUP producer changed its external alignment/source authority')
    external=controller.external_original_reference
    protected=() if external is None else external.protected_output_intervals
    if external is not None:external.validate_source_domain(domain)
    context=WholeProgrammeContext(old,domain,token,port.deadline)
    if cached_inventories is None:
        cached_inventories={}
        root=getattr(request,'source_cache_directory',None)
        derived=getattr(request,'visual_feature_derived_cache_directory',None)
        if root is None and derived is None:
            cached_inventories={source:CachedNativeInventory((),0,0,0,0) for source in eligible_indices}
        else:
            from .visual_score import _visual_analyzer_fingerprint
            from .visual_feature_cache import PersistentVisualFeatureCache
            analyzer=_visual_analyzer_fingerprint()
            for source in eligible_indices:
                cache=PersistentVisualFeatureCache(source_cache_directory=root or derived,
                    source_path=old.render_plan.sources[source],analyzer=analyzer,derived_cache_directory=derived)
                cached_inventories[source]=load_cached_native_inventory(cache,source_index=source,
                    source_clock=domain.clocks_by_request[source],deadline_seconds=min(60.,remaining()))
    opportunities=_opportunities(old.render_plan,domain,rhythm,section_plan,protected)
    cached=build_cached_programme_inventory(original=old.render_plan,domain=domain,inventories=cached_inventories,
        opportunities=opportunities,protected_spans=protected,maximum_seconds=min(30.,remaining()))
    sensing=freeze_whole_sensing_recipes(cached,fps=old.render_plan.fps)
    demands,census=freeze_whole_native_demands(cached,sensing_recipes=sensing)
    packages=freeze_whole_completion_packages(sensing,demands)
    census['sensed_recipe_count']=len(sensing)
    census['all_base_run_count']=len(cached.recipes)
    census['sensing_frontier_basis']='source-balanced-shared-seam-chains-before-observation'
    stage_seconds['physical_inventory']=monotonic()-stage_started;stage_started=monotonic()
    if assets is None:
        from .body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256
        model=installed_pose_model(getattr(request,'source_cache_directory',None))
        assets=installed_native_assets(SimpleNamespace(model_path=model,model_sha256=RTMPOSE_BODY7_MODEL_SHA256))
    # Reserve real metadata rasters, not a second detector completion after
    # search. NN/decode/evidence/time allowances never reset between stages.
    budget=JointGroupWorkBudget(absolute_deadline=port.deadline,maximum_new_decodes=1024,
        maximum_model_calls=8192,maximum_projected_cache_bytes=32*1024**2)
    with GroupJointObservationSession(domain=domain,budget=budget) as session:
        sensors=NativeFamilySensors(session,assets)
        register_review_sface(session,_default_model_path())
        fact_cache=None;cache_hit=False;cache_saved=False
        if recorded_index is not None:
            if recorded_bindings is None:raise ValueError('recorded native input omitted its actual source manifest')
            sensors.bind_recorded_motion_index(recorded_index,
                source_identities=recorded_bindings['source_identities'],
                source_content_sha256=recorded_bindings['source_content_sha256'])
        elif getattr(request,'source_cache_directory',None) is not None:
            from .group_whole_native_cache import WholeNativeFactCache
            fact_cache=WholeNativeFactCache(root=request.source_cache_directory,sensors=sensors)
            cache_hit=fact_cache.bind_existing()
        inventory=sensors.motion_inventory(demands)
        # Freeze this allocation before NN outcomes. Leave time for the SAME
        # request's whole graph, selection and real metadata pixel reads.
        # These reserves are not a calibrated latency or quality guarantee.
        downstream_reserve=min(45.,remaining()*.55)
        downstream_identity_reserve=min(4096,math.ceil(
            port.remaining_capacity(('identity_reads',))['identity_reads']*.55))
        def preserve_downstream():
            return (remaining()<=downstream_reserve or
                port.remaining_capacity(('identity_reads',))['identity_reads']<=downstream_identity_reserve)
        settled_chunks=0
        def preserve_settled(inventory):
            nonlocal settled_chunks
            settled_chunks+=1
            # Bounded checkpoints, not a quadratic full-archive rewrite on
            # every seven-input settlement. Always persist the final snapshot.
            if settled_chunks%4==0:fact_cache.preserve_completed(inventory)
        completed=sensors.complete_motion_inventory_once(inventory,maximum_new_observations=maximum_new_native_inputs,
            maximum_new_raw_observations=maximum_new_native_inputs,
            completion_packages=packages,stop_new_work=preserve_downstream,
            on_chunk_settled=None if fact_cache is None else preserve_settled)
        if fact_cache is not None:cache_saved=fact_cache.preserve_completed(completed)
        stage_seconds['cache_and_native_completion']=monotonic()-stage_started;stage_started=monotonic()
        views=NativeMotionViews(sensors,completed)
        bank=freeze_whole_native_inventory(context=context,domain=domain,cached=cached,views=views,
            protected_spans=protected,work_port=port,maximum_seconds=min(30.,remaining()),
            sensed_recipe_ids=frozenset(r.run.identity for r in sensing))
        stage_seconds['actor_paths_and_prefit_cameras']=monotonic()-stage_started;stage_started=monotonic()
        control,selected=select_whole_native_programme(inventory=bank,work_port=port,
            maximum_seconds=min(20.,remaining()))
        stage_seconds['paired_whole_selection']=monotonic()-stage_started;stage_started=monotonic()
        candidate=materialize_whole_native_selection(context=context,inventory=bank,selection=selected,views=views)
        _validate_review_first_group_bundle(candidate,eligible_indices=eligible_indices,offsets=offsets,timewarps=timewarps,
            common_start=old.render_plan.master_audio_start,phrase_times=old.phrase_times)
        port._check()
        if identities!=tuple(_file_identity(old.render_plan.sources[i]) for i in eligible_indices):
            raise ValueError('whole GROUP source changed before atomic owner publication')
        controller.promote(candidate,plan=candidate.render_plan,authority=candidate.candidate_origin,_token=token)
        stage_seconds['full_native_metadata_and_publication']=monotonic()-stage_started
        by_id={r.run.identity:r for r in bank.recipes}
        selected_recipes=tuple(by_id[r.identity] for r in selected.runs)
        complete_keys={s.demand.native_key for s in completed.supplies if s.state=='NATIVE_ROWS_AVAILABLE'}
        sensing_ids={r.run.identity for r in sensing}
        from collections import Counter
        return dict(state='COMPLETE_WHOLE_PROGRAMME_SELECTED_FOR_REVIEW',census=census,
            inventory=bank.evidence(),control=control.to_dict(),selected=selected.to_dict(),
            actor_action_changed_plan=tuple(r.identity for r in control.runs)!=tuple(r.identity for r in selected.runs),
            native_completion_status=completed.status,native_work=budget.to_dict(),
            declared_downstream_reserve_seconds=downstream_reserve,
            declared_downstream_identity_reads=downstream_identity_reserve,
            native_supply_states=dict(Counter(s.state for s in completed.supplies)),
            fully_observed_declared_recipes=sum(set(p.native_keys)<=complete_keys for p in packages),
            selected_actor_runs=sum(r.actor_path is not None for r in selected_recipes),
            selected_authored_camera_runs=sum(r.side is not None and r.side.camera is not None for r in selected_recipes),
            selected_sensed_recipe_runs=sum(r.base.run.identity in sensing_ids for r in selected_recipes),
            selected_native_observed_output_frames=sum((r.run.request_source_index,n,None) in complete_keys
                for r in selected_recipes for n in r.base.native_frame_indices),
            selected_output_frames=sum(r.run.end_output_frame-r.run.first_output_frame for r in selected_recipes),
            stage_seconds=stage_seconds,elapsed_seconds=monotonic()-started,
            native_fact_cache_hit=cache_hit,native_fact_cache_preserved=cache_saved,
            native_fact_cache_checkpoint_stride=4,
            normal_controller_revision=controller.transaction_token().revision,
            rendered_movie=False,professional_quality_approved=False)


def attempt_whole_programme(*,controller,work,request,reporter=None,**inputs):
    """One normal entry; an unavailable model is not repeated legacy work."""
    old=controller.incumbent
    if old.native_motion_production is not None:return None
    # Decline unsupported active owners instead of throwing their learned,
    # look or previous native evidence away to manufacture a passing result.
    unsupported=tuple(name for name in ('dynamic_crop_application','cut_local_sync','splice_graph_authorization',
        'output_geometry_deferral','stage_edit_dynamic_crop_version','stage_edit_dynamic_crop_v3_decodes',
        'dynamic_crop_motion_timeline_v3','normal_interval_review','performer_family_review','boundary_look_match')
        if getattr(old,name) is not None)
    if unsupported:
        if reporter is not None:reporter.log('전곡 native 선택 보류: 기존 활성 소유자 보존 필요 · '+', '.join(unsupported))
        return None
    if inputs.get('assets') is None:
        root=getattr(request,'source_cache_directory',None)
        if root is None or not (Path(root)/'models/source-first/rtmpose_body7.onnx').is_file():
            if reporter is not None:reporter.log('전곡 native 선택: 설치된 RTMPose 입력 없음 · 추가 다운로드/재분석 안 함')
            return None
    from .request_work_lease import RequestWorkExhausted
    from .group_joint_source_camera import JointGroupBudgetExhausted
    try:
        return work.run('selected_completion',lambda:controller.run_optional('whole_native_programme',
            lambda:produce_whole_programme(controller=controller,request=request,**inputs)))
    except (RequestWorkExhausted,JointGroupBudgetExhausted,TimeoutError) as exc:
        if reporter is not None:reporter.log('전곡 native 선택 예산 종료 · 완전한 incumbent 유지 · '+str(exc))
        return None
