"""Real observed two-sided cameras and their complete incident/hold evidence.

The caller owns the finite source/phase/actor/cut frontier and one completed
native inventory. Cameras are produced BEFORE comparison, on the exact native
sampler. This stage does not provide a professional-quality certificate or
replace the full normal timeline/motion metadata producer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .render import RenderPlan
    from .shot_video_clock import FrozenShotVideoClockContext

from .group_native_motion_programme import NativeMotionSide, realize_native_motion_programme
from .group_scene_epoch_program import compile_common_clock_camera_queries
from .group_scene_owned_program import freeze_owned_scene_program
from .group_performer_family_owner import digest
from .group_review_projected_action import project_observed_review_face, project_observed_review_joints
from .group_review_two_shot_camera import (fit_review_fixed_context_face_key,
    fit_review_coupled_boundary_keys, realize_review_contextual_focal_shot,
    measure_review_focal_incident, compare_review_focal_incident_inventory,
    measure_review_partial_joint_trajectory)
from .group_review_projected_action import measure_review_partial_screen_trajectory
from .group_joint_bundle_metadata import SelectedNativeActorFrame, SelectedNativeMotionShot
from .render import segment_frame_counts


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _queries(context, domain, original, plan, first, end):
    compiled=compile_common_clock_camera_queries(context=context,domain=domain,
        original_program=original,candidate_plan=plan,output_intervals=((first,end),))
    return {t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}


def _sample_ticks(first,end,count=7):
    return tuple(sorted({first+round(i*(end-first-1)/(count-1)) for i in range(count)}))


def _support(view,query,tick):
    face=project_observed_review_face(view,query,tick)
    if face is None or face['landmarks_xy'] is None:
        return None
    x,y=face['center'];w,h=face['width'],face['height']
    # A detector box can straddle the raster although all five landmarks are
    # actually visible. Its clipped raster extent is support, not a fabricated
    # facial measurement. Landmark scale still comes only from observed points.
    corners=tuple((min(1.,max(0.,a)),min(1.,max(0.,b))) for a,b in
        ((x-w/2,y-h/2),(x+w/2,y-h/2),(x-w/2,y+h/2),(x+w/2,y+h/2)))
    joints=project_observed_review_joints(view,query,tick)
    return tuple((*face['landmarks_xy'],*corners,*(p for p in joints.values()
        if all(isfinite(v) and 0<=v<=1 for v in p))))


def _native_joints(view,query,tick):
    # Validate the exact raw parent/ROI receipt through the registered consumer;
    # then consume its native coordinates, never camera-projected articulation.
    observed=project_observed_review_joints(view,query,tick)
    if view is None:return {},9/16
    raw=view['row']['native_observation']
    width,height=raw['frame_size_wh']
    return {j:tuple(view['review_joint_provenance'][j]['xy_native']) for j in observed},height/width


def _measure_programme(views,queries,selected,first,cut,end,fps):
    incidents=[]
    for tick in (first,cut,end):
        left=tuple(range(tick-3,tick));right=tuple(range(tick,tick+3))
        def faces(ticks):
            return tuple(project_observed_review_face(selected[t],queries[t],t) for t in ticks)
        def points(ticks):
            return tuple(project_observed_review_joints(selected[t],queries[t],t) for t in ticks)
        incidents.append(measure_review_focal_incident(left_faces=faces(left),right_faces=faces(right),
            left_points=points(left),right_points=points(right),left_ticks=left,right_ticks=right,fps=fps,
            left_native=tuple(views.native_signature(queries[t],t) if selected[t] is not None else None for t in left),
            right_native=tuple(views.native_signature(queries[t],t) if selected[t] is not None else None for t in right)))
    return tuple(incidents)


def _measure_holds(before_queries,before,after_queries,after,sides,fps):
    result=[]
    for side in sides:
        ticks=_sample_ticks(side.first_output_frame,side.end_output_frame)
        before_faces=[project_observed_review_face(before[t],before_queries[t],t) for t in ticks]
        after_faces=[project_observed_review_face(after[t],after_queries[t],t) for t in ticks]
        a=[project_observed_review_joints(before[t],before_queries[t],t) for t in ticks]
        b=[project_observed_review_joints(after[t],after_queries[t],t) for t in ticks]
        native_a=[_native_joints(before[t],before_queries[t],t) for t in ticks]
        native_b=[_native_joints(after[t],after_queries[t],t) for t in ticks]
        lost=[];unknown=[]
        for t,fa,fb,pa,pb in zip(ticks,before_faces,after_faces,a,b,strict=True):
            if fa is not None and fa['landmarks_xy'] is not None and (fb is None or fb['landmarks_xy'] is None):
                lost.append((t,'known_facial_landmarks'))
            if fa is None or fa['landmarks_xy'] is None:unknown.append((t,'original_facial_landmarks'))
            for joint in pa:
                if joint not in pb or (all(0<=v<=1 for v in pa[joint])
                        and not all(0<=v<=1 for v in pb[joint])):
                    lost.append((t,f'known_joint_{joint}'))
            unknown.extend((t,f'original_joint_{j}') for j in range(6) if j not in pa)
        result.append(dict(requested_ticks=ticks,known_observation_losses=lost,
            original_unknown_requests=unknown,
            native_articulation=measure_review_partial_joint_trajectory(
                [p for p,_ in native_a],[p for p,_ in native_b],ticks=ticks,fps=fps,
                left_aspects=tuple(r for _,r in native_a),right_aspects=tuple(r for _,r in native_b)),
            actual_screen_motion=measure_review_partial_screen_trajectory(a,b,ticks=ticks,fps=fps),
            quality_approved=False))
    return tuple(result)


def _appearance_incidents(selected,first,cut,end):
    from .group_review_native_window import cosine
    from .ive_local_arcface_gallery import local_member_hint_conflict
    result=[]
    for tick in (first,cut,end):
        left,right=selected[tick-1],selected[tick]
        a=None if left is None else left.get('member_hint')
        b=None if right is None else right.get('member_hint')
        measured=(None if left is None or right is None or left['vector'] is None or right['vector'] is None
            else cosine(left['vector'],right['vector']))
        result.append(dict(boundary_tick=tick,sface_cosine=measured,
            local_member_hint_conflict=local_member_hint_conflict(a,b),
            left_local_member=None if a is None or a.get('state')!='LIKELY_MEMBER' else a['member'],
            right_local_member=None if b is None or b.get('state')!='LIKELY_MEMBER' else b['member'],
            local_gallery_stage_video_thresholds_calibrated=False,
            named_actor_certified=False,quality_approved=False))
    return tuple(result)


@dataclass(frozen=True,slots=True)
class NativeMotionCameraCandidate:
    render_plan: RenderPlan
    shot_clock_context: FrozenShotVideoClockContext
    selected_shots: tuple[SelectedNativeMotionShot,SelectedNativeMotionShot]
    evidence_json: str

    @property
    def evidence(self):
        return json.loads(self.evidence_json)

    def to_dict(self):
        return dict(contract='actual_two_sided_native_motion_camera_proposal_v1',
            selected_plan_identity=digest(self.render_plan.to_json()),
            selected_shot_count=len(self.selected_shots),evidence=self.evidence,
            full_normal_metadata_constructed=False,quality_approved=False)


def _selected_shots(result,paths,receipts,domain):
    offsets=[0]
    for count in segment_frame_counts(result.render_plan):offsets.append(offsets[-1]+count)
    shots=[]
    for side,path,receipt,run in zip(result.sides,paths,receipts,
            result.preservation['physical_native_runs'],strict=True):
        frames=[]
        from .render_boundary_clock import native_frame_clocks_for_output
        native_schedule=native_frame_clocks_for_output(result.render_plan,domain.clocks_by_request,
            range(side.first_output_frame,side.end_output_frame))
        for tick in range(side.first_output_frame,side.end_output_frame):
            view=path[tick]
            if view is None:
                native=native_schedule[tick].native_frame_index
                frames.append(SelectedNativeActorFrame(tick,side.request_source_index,native,None,None))
            else:
                raw=view['row']['native_observation']
                frames.append(SelectedNativeActorFrame(tick,side.request_source_index,raw['native_frame_index'],
                    view['descriptor'].frame_local_head_id,digest(raw)))
        shots.append(SelectedNativeMotionShot(side.first_output_frame,side.end_output_frame,side.request_source_index,
            tuple(run),side.proposal_role,side.phase_native_frames,receipt.get('hypothesis_sha256'),tuple(frames),
            tuple(range(offsets.index(side.first_output_frame),offsets.index(side.end_output_frame)))))
    return tuple(shots)


def produce_two_sided_native_cameras(*,context,domain,views,original_program,
        original_actors,original_actor_receipt,sides,side_paths,side_path_receipts,
        protected_output_intervals=(),maximum_camera_candidates=2):
    """One joint source/phase/actor/cut tuple; actual BOTH whole-shot cameras.

    Input actor paths must already be selected on the proposed native sampler.
    This function never changes them to fit a crop. Original intent and actors
    are independently frozen before the finite caller starts donor scoring.
    Unknown terms survive in the measured fixed-inventory bounds. No average
    score can override a known incident/hold observation loss.
    """
    from .group_source_changing_owner import NativeMotionViews
    if (not isinstance(views,NativeMotionViews) or views.domain!=domain
            or original_program!=freeze_owned_scene_program(context,domain)
            or original_actor_receipt.get('original_program_identity')!=original_program.program_sha256
            or original_actor_receipt.get('donor_costs_used') is not False
            or not isinstance(sides,tuple) or len(sides)!=2
            or any(not isinstance(s,NativeMotionSide) or s.camera is not None for s in sides)
            or sides[0].end_output_frame!=sides[1].first_output_frame
            or any(s.end_output_frame-s.first_output_frame<4 for s in sides)
            or not isinstance(side_paths,tuple) or len(side_paths)!=2
            or not isinstance(side_path_receipts,tuple) or len(side_path_receipts)!=2
            or type(maximum_camera_candidates) is not int or not 1<=maximum_camera_candidates<=8):
        raise ValueError('two-sided camera production needs actual independent original and native actor paths')
    first,cut,end=sides[0].first_output_frame,sides[0].end_output_frame,sides[1].end_output_frame
    if not 3<=first<cut<end<=context.incumbent.render_plan.output_frame_count-3:
        raise ValueError('two-sided camera production needs all three real incident flanks')
    halo=tuple(range(first-3,end+3))
    if not set(halo)<=set(original_actors):
        raise ValueError('original actor inventory omitted a declared incident or hold')
    for side,path,receipt in zip(sides,side_paths,side_path_receipts,strict=True):
        if (set(path)!=set(range(side.first_output_frame,side.end_output_frame))
                or receipt.get('state')!='CONNECTED_SOURCE_LOCAL_HYPOTHESIS'
                or any(v is not None and v.get('source_actor_path_identity')!=receipt.get('hypothesis_sha256')
                    for v in path.values())):
            raise ValueError('camera proposal borrowed an unconnected or partially addressed actor path')
    original_plan=context.incumbent.render_plan
    base_queries=_queries(context,domain,original_program,original_plan,first-3,end+3)
    # Registered query scopes can differ despite identical source/native/
    # camera rasters. Rebind exact original occurrences, never reselect them.
    original=views.reproject_selected_actor_path(base_queries,{t:original_actors[t] for t in halo})
    # KEEP has real measured incidents on the SAME request basis, not zero.
    keep_incidents=_measure_programme(views,base_queries,original,first,cut,end,original_plan.fps)
    keep_holds=_measure_holds(base_queries,original,base_queries,original,sides,original_plan.fps)
    numeric=realize_native_motion_programme(original=original_plan,domain=domain,sides=sides,
        protected_output_intervals=protected_output_intervals)
    neutral_queries=_queries(context,domain,original_program,numeric.render_plan,first-3,end+3)
    paths=tuple(views.reproject_selected_actor_path(neutral_queries,p) for p in side_paths)
    missing=[t for t in (first,cut-1,cut,end-1) if (paths[0] if t<cut else paths[1])[t] is None]
    source_faces={t:project_observed_review_face((paths[0] if t<cut else paths[1])[t],neutral_queries[t],t)
        for t in (first,cut-1,cut,end-1)}
    fixed_faces={t:project_observed_review_face(original[t],base_queries[t],t) for t in (first-1,end)}
    if (missing or any(f is None or f['landmarks_xy'] is None for f in (*source_faces.values(),*fixed_faces.values()))):
        return (),dict(state='UNOBSERVED_ENDPOINT_ANATOMY',missing_actor_ticks=missing,
            missing_landmark_ticks=tuple(t for t,f in (*source_faces.items(),*fixed_faces.items())
                if f is None or f['landmarks_xy'] is None),
            fixed_context_inputs=tuple(views.observation_status(base_queries[t],t,selected=original[t])
                for t in (first-1,end)),
            measured_keep_incidents=keep_incidents,quality_approved=False)
    entry=fit_review_fixed_context_face_key(source_faces[first],fixed_faces[first-1],
        required_visible_points=_support(paths[0][first],neutral_queries[first],first) or ())
    exit=fit_review_fixed_context_face_key(source_faces[end-1],fixed_faces[end],
        required_visible_points=_support(paths[1][end-1],neutral_queries[end-1],end-1) or ())
    if entry is None or exit is None:
        return (),dict(state='UNREALIZABLE_FIXED_CONTEXT_CAMERA',entry_available=entry is not None,
            exit_available=exit is not None,measured_keep_incidents=keep_incidents,quality_approved=False)
    # An upper-frame face is not a failed centre-frame composition. Fixed
    # decorative destinations alone can make an EXACT shared face fit vanish
    # even when both observed cameras already admit it without cropping. Keep
    # both measured face anchors and their common centre as joint proposals;
    # complete renderer/support validation still applies to every whole track.
    centers=[]
    for tick in (cut-1,cut):
        points=source_faces[tick]['landmarks_xy']
        centers.append(tuple(sum(p[axis] for p in points)/5 for axis in (0,1)))
    centers.append(tuple((centers[0][i]+centers[1][i])/2 for i in (0,1)))
    targets=tuple(dict.fromkeys((*centers,(.5,.45),(.42,.45),(.58,.45),(.5,.5),
        (.5,.525),(.42,.6),(.58,.6))))
    pairs=fit_review_coupled_boundary_keys(left_head=source_faces[cut-1],right_head=source_faces[cut],
        left_joint_points=project_observed_review_joints(paths[0][cut-1],neutral_queries[cut-1],cut-1),
        right_joint_points=project_observed_review_joints(paths[1][cut],neutral_queries[cut],cut),
        left_required_visible_points=_support(paths[0][cut-1],neutral_queries[cut-1],cut-1) or (),
        right_required_visible_points=_support(paths[1][cut],neutral_queries[cut],cut) or (),
        maximum_candidates=maximum_camera_candidates,target_centers=targets)
    candidates=[];rejected=[]
    for outgoing,incoming,pair_receipt in pairs:
        port=views.sensors.session.budget.request_work_phase
        if port is not None:port.charge('states')
        views.sensors.session.budget.check()
        tracks=[];track_receipts=[]
        for side,path,a,b in zip(sides,paths,(entry,incoming),(outgoing,exit),strict=True):
            support=tuple(_support(path[t],neutral_queries[t],t) for t in range(side.first_output_frame,side.end_output_frame))
            track,receipt=realize_review_contextual_focal_shot(entry_key=a,exit_key=b,support=support,fps=original_plan.fps)
            tracks.append(track);track_receipts.append(receipt)
        if any(track is None for track in tracks):
            rejected.append(dict(state='UNREALIZABLE_WHOLE_CAMERA',track_receipts=track_receipts));continue
        authored=realize_native_motion_programme(original=original_plan,domain=domain,
            sides=tuple(replace(s,camera=t) for s,t in zip(sides,tracks,strict=True)),
            protected_output_intervals=protected_output_intervals)
        actual_queries=_queries(context,domain,original_program,authored.render_plan,first-3,end+3)
        actual_paths=tuple(views.reproject_selected_actor_path(actual_queries,p) for p in paths)
        exterior={t:original[t] for t in halo if t<first or t>=end}
        selected=views.reproject_selected_actor_path(actual_queries,exterior)
        selected.update(actual_paths[0]);selected.update(actual_paths[1])
        incidents=_measure_programme(views,actual_queries,selected,first,cut,end,original_plan.fps)
        comparison=compare_review_focal_incident_inventory(keep_incidents,incidents)
        holds=_measure_holds(base_queries,original,actual_queries,selected,sides,original_plan.fps)
        appearance=_appearance_incidents(selected,first,cut,end)
        losses=[loss for hold in holds for loss in hold['known_observation_losses']]
        conflicts=[row['boundary_tick'] for row in appearance if row['local_member_hint_conflict']]
        evidence=dict(state='REALIZED_FOR_REVIEW' if comparison['eligible_for_further_review'] and not losses and not conflicts
                else 'KNOWN_INCIDENT_OR_HOLD_REGRESSION',
            original_program_identity=original_program.program_sha256,
            original_actor_inventory_identity=original_actor_receipt['inventory_identity'],
            source_actor_path_identities=[r['hypothesis_sha256'] for r in side_path_receipts],
            measured_keep_incidents=keep_incidents,measured_keep_holds=keep_holds,
            actual_incidents=incidents,incident_comparison=comparison,
            actual_holds=holds,known_hold_observation_losses=losses,
            actual_appearance_incidents=appearance,local_member_hint_conflict_ticks=conflicts,
            coupled_landmark_fit=pair_receipt,whole_camera_receipts=track_receipts,
            exact_sampler_preservation=authored.preservation,
            measured_keep_score=sum(i['exploration_score'] for i in keep_incidents)+sum(
                h['native_articulation']['exploratory_cost_interval'][1]
                +h['actual_screen_motion']['exploratory_cost_interval'][1]
                for h in keep_holds),
            score=sum(i['exploration_score'] for i in incidents)+sum(
                h['native_articulation']['exploratory_cost_interval'][1]
                +h['actual_screen_motion']['exploratory_cost_interval'][1] for h in holds),
            actor_reselected_after_camera_fit=False,camera_refit_after_selection=False,
            quality_approved=False)
        candidates.append(NativeMotionCameraCandidate(authored.render_plan,authored.shot_clock_context,
            _selected_shots(authored,actual_paths,side_path_receipts,domain),_dump(evidence)))
    return tuple(candidates),dict(state='REALIZED' if candidates else 'NO_REALIZABLE_JOINT_CAMERA',
        realized_camera_count=len(candidates),rejected=rejected,measured_keep_incidents=keep_incidents,
        finite_source_actor_phase_cut_tuple_only=True,full_normal_producer_complete=False,quality_approved=False)
