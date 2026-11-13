"""Finite continuous-shot review scopes and joint same-camera comparisons.

These consumers do not certify expert quality. Physical runs precede actor
costs; actual interval duties and both original-context incidents stay distinct.
An explicit KEEP_ORIGINAL remains a result, never a forced technical winner.
"""
from dataclasses import dataclass
import math
from itertools import combinations
import numpy as np
from .group_performer_family_owner import digest
from .render import EditorialTransformKeyframe
from .group_reference_interval import compare_group_reference_interval


@dataclass(frozen=True,slots=True)
class ContinuousReviewShotScope:
    entry_tick:int
    return_tick:int
    output_frame_count:int
    halo_radius:int=2
    context_frames:int=12
    minimum_hold_frames:int=18

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.entry_tick,self.return_tick,self.output_frame_count,
                self.halo_radius,self.context_frames,self.minimum_hold_frames))
                or not 2<=self.halo_radius<=6 or not 2<=self.context_frames<=60
                or not 2<=self.minimum_hold_frames<=300
                or not self.context_frames<=self.entry_tick<self.return_tick<=self.output_frame_count-self.context_frames
                or self.return_tick-self.entry_tick<self.minimum_hold_frames):
            raise ValueError('continuous review shot needs explicit hold and both original contexts')

    @property
    def changed_ticks(self):return tuple(range(self.entry_tick,self.return_tick))
    @property
    def incidents(self):return (self.entry_tick,self.return_tick)
    @property
    def measurement_ticks(self):
        return tuple(sorted({t for b in self.incidents for t in range(b-self.halo_radius,b+self.halo_radius+1)}))
    @property
    def comparison_context(self):return (self.entry_tick-self.context_frames,self.return_tick+self.context_frames)
    @property
    def identity(self):return digest(self.to_dict())
    def to_dict(self):
        return dict(contract='predeclared_continuous_single_donor_review_scope_v1',
            entry_tick=self.entry_tick,return_tick=self.return_tick,output_frame_count=self.output_frame_count,
            halo_radius=self.halo_radius,context_frames=self.context_frames,minimum_hold_frames=self.minimum_hold_frames,
            duration_is_quality_certificate=False,unchanged_original_context_is_donor_duty=False)


def continuous_review_native_runs(native_signatures):
    """Physical source/clock/file/unit runs, with NO actor/body prefilter."""
    if (not isinstance(native_signatures,dict) or not 1<=len(native_signatures)<=36000
            or any(type(t) is not int or t<0 for t in native_signatures)):
        raise ValueError('physical native runs need a finite exact output inventory')
    rows=[]
    for t,s in sorted(native_signatures.items()):
        if s is not None and (not isinstance(s,(tuple,list)) or len(s)!=5
                or type(s[0]) is not int or type(s[3]) is not int or type(s[4]) is not int):
            raise ValueError('physical native run changed its source/clock/file/unit/frame address')
        if s is None:continue
        if (not rows or t!=rows[-1]['end_tick'] or tuple(s[:4])!=tuple(rows[-1]['physical_unit'])
                or not 0<=s[4]-rows[-1]['last_native_frame']<=12):
            rows.append(dict(first_tick=t,end_tick=t+1,physical_unit=list(s[:4]),first_native_frame=s[4],last_native_frame=s[4]))
        else:rows[-1].update(end_tick=t+1,last_native_frame=s[4])
    return tuple(dict(row,actor_correspondence_certified=False,known_joint_duties_waived=False,quality_approved=False) for row in rows)


def enumerate_continuous_review_intervals(*,runs_by_source,independent_boundaries,output_frame_count,
        minimum_hold_frames=18,maximum_returned=56,context_frames=12):
    """Physical interval catalogue BEFORE actor, anatomy and camera costs.

    Native edges are real alternative edit boundaries, not a reason to lose an
    entire source. Every hypothesis owns BOTH new incidents. Budget exhaustion
    returns deferred hypotheses without erasing already available hypotheses.
    Including the original source index is intentional: KEEP is a recipe, not
    a camera index. No interval here is an editorial/actor quality certificate.
    """
    boundaries=tuple(independent_boundaries)
    if (not 2<=len(boundaries)<=32 or boundaries!=tuple(sorted(set(boundaries)))
            or any(type(t) is not int or t<0 for t in boundaries)
            or type(maximum_returned) is not int or not 1<=maximum_returned<=224
            or not isinstance(runs_by_source,dict) or not 1<=len(runs_by_source)<=16):
        raise ValueError('finite independent physical interval catalogue and budget required')
    if (type(output_frame_count) is not int or type(minimum_hold_frames) is not int
            or type(context_frames) is not int or not 2<=minimum_hold_frames<=300
            or not 2<=context_frames<=60 or output_frame_count<=2*context_frames):
        raise ValueError('physical interval catalogue needs exact hold and original contexts')
    low,high=boundaries[0],boundaries[-1];groups={}
    for source,runs in sorted(runs_by_source.items()):
        if type(source) is not int or source<0 or not isinstance(runs,(list,tuple)) or len(runs)>64:
            raise ValueError('physical interval source/run inventory changed')
        candidates={}
        for run in runs:
            a,b=run['first_tick'],run['end_tick'];unit=run['physical_unit']
            if (type(a) is not int or type(b) is not int or not 0<=a<b<=output_frame_count
                    or len(unit)!=4 or unit[0]!=source):raise ValueError('physical run address changed')
            root=digest(run)
            anchors=sorted({t for t in (*boundaries,a,b)
                if max(low,a,context_frames)<=t<=min(high,b,output_frame_count-context_frames)})
            for first,end in combinations(anchors,2):
                if end-first<minimum_hold_frames:continue
                scope=ContinuousReviewShotScope(first,end,output_frame_count,
                    context_frames=context_frames,minimum_hold_frames=minimum_hold_frames)
                candidates[(first,end,root)]=dict(source=source,scope=scope.to_dict(),native_run_identity=root,
                    physical_unit=unit,actor_correspondence_certified=False,known_joint_duties_waived=False,quality_approved=False)
        groups[source]=[candidates[k] for k in sorted(candidates)]
    ordered=[]
    for index in range(max((len(v) for v in groups.values()),default=0)):
        for group in groups.values():
            if index<len(group):ordered.append(group[index])
    return dict(contract='physical_before_actor_continuous_interval_catalogue_v1',
        state='COMPLETE' if len(ordered)<=maximum_returned else 'BUDGET_LIMITED',
        independent_boundaries=boundaries,hypotheses=ordered[:maximum_returned],deferred=ordered[maximum_returned:],
        total_physical_hypotheses=len(ordered),pose_costs_used_to_choose_boundaries=False,
        quality_approved=False)


class ReviewCameraRealizationBudget:
    """Reserve a unique recipe BEFORE expensive realization; schemas share it."""
    def __init__(self,maximum_recipes):
        if type(maximum_recipes) is not int or not 1<=maximum_recipes<=224:
            raise ValueError('camera realization needs its independently declared finite budget')
        self.maximum_recipes=maximum_recipes;self.reserved={};self.deferred=[]

    def reserve(self,recipe):
        identity=digest(recipe)
        if identity in self.reserved:return True
        if len(self.reserved)>=self.maximum_recipes:
            self.deferred.append(identity);return False
        self.reserved[identity]=recipe;return True

    def receipt(self):
        return dict(state='BUDGET_LIMITED' if self.deferred else 'COMPLETE',maximum_recipes=self.maximum_recipes,
            reserved_recipe_identities=list(self.reserved),deferred_recipe_identities=list(dict.fromkeys(self.deferred)),
            pre_existing_candidates_erased=False,quality_approved=False)


def classify_review_continuation_incident(*,candidate,continuation,source_clocks,
        candidate_native_signatures,continuation_native_signatures,boundary_tick,halo_radius=2):
    """Distinguish removed cuts from NEW cross-performance comparisons.

    ``continuation`` is the explicitly declared outgoing physical source
    carried across this incident. It is not the old edited incoming source.
    A same-source/native-camera run alone is insufficient: every requested
    actual native address and renderer framing must agree. Grade changes and
    added framing interventions remain separately visible. This is structural
    evidence only; no anatomical/actor/duty/adoption status is promoted.
    """
    if (type(boundary_tick) is not int or type(halo_radius) is not int
            or not 2<=halo_radius<=6 or boundary_tick<halo_radius):
        raise ValueError('continuation incident needs its explicit exact output halo')
    ticks=tuple(range(boundary_tick-halo_radius,boundary_tick+halo_radius+1))
    if (set(candidate_native_signatures)!=set(ticks)
            or set(continuation_native_signatures)!=set(ticks)):
        raise ValueError('continuation incident omitted an actual requested native sample')
    physical=continuous_review_native_runs(candidate_native_signatures)
    exact_native=(len(physical)==1 and physical[0]['first_tick']==ticks[0]
        and physical[0]['end_tick']==ticks[-1]+1
        and all(candidate_native_signatures[t] is not None
            and tuple(candidate_native_signatures[t])==tuple(continuation_native_signatures[t]) for t in ticks))
    comparison=compare_group_reference_interval(reference=continuation,candidate=candidate,
        reference_source_clocks=source_clocks,candidate_source_clocks=source_clocks,
        output_intervals=((ticks[0],ticks[-1]+1),),maximum_output_frames=13).to_dict()
    physical_changed=bool(comparison['terminal_clone_output_frames'] or any(
        r['source_path_changed'] or r['native_sample_changed'] or r['source_clock_evidence_changed']
        for r in comparison['changes']))
    framing_changed=any(r['framing_parameters_changed'] for r in comparison['changes'])
    look_changed=any(r['look_parameters_changed'] for r in comparison['changes'])
    kind=('NEW_NATIVE_COMPARISON_REQUIRED' if not exact_native or physical_changed else
        'SAME_NATIVE_WITH_NEW_FRAMING' if framing_changed else
        'SAME_NATIVE_FRAMING_WITH_LOOK_CHANGE' if look_changed else 'EXACT_NATIVE_RENDER_CONTINUATION')
    return dict(contract='actual_native_and_render_continuation_incident_v1',boundary_tick=boundary_tick,
        requested_ticks=ticks,kind=kind,new_cross_performance_action_comparison_required=kind=='NEW_NATIVE_COMPARISON_REQUIRED',
        projected_layout_comparison_required=kind in ('NEW_NATIVE_COMPARISON_REQUIRED','SAME_NATIVE_WITH_NEW_FRAMING'),
        look_change_review_required=look_changed,physical_runs=physical,actual_renderer_comparison=comparison,
        physical_input_identity=digest(dict(candidate=candidate_native_signatures,continuation=continuation_native_signatures)),
        actor_correspondence_certified=False,known_joint_duties_waived=False,
        anatomical_measurement_state='NOT_PROMOTED',raw_pixel_replay_certified=False,
        candidate_admitted=False,quality_approved=False)


def measure_review_added_camera_motion(samples,*,fps,boundary_ticks):
    """Rates within shots AND transform discontinuities at every incident.

    Input is the ACTUAL added editorial transform for every declared tick,
    including fixed original cameras before entry and after return. This does
    not measure body velocity or substitute for observed projected layout.
    """
    if (not isinstance(samples,dict) or not 3<=len(samples)<=36000
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or not isinstance(boundary_ticks,tuple) or not boundary_ticks
            or tuple(sorted(set(boundary_ticks)))!=boundary_ticks):
        raise ValueError('camera rates need exact output clock and all actual incidents')
    ticks=tuple(sorted(samples))
    if any(type(t) is not int or not isinstance(samples[t],EditorialTransformKeyframe) for t in ticks):
        raise ValueError('camera rates require actual renderer transform samples')
    if any(b not in samples or b-1 not in samples for b in boundary_ticks):
        raise ValueError('camera motion omitted an actual entry or original-camera return')
    coordinates=np.asarray([(math.log(samples[t].zoom),samples[t].center_x,samples[t].center_y,
        math.radians(samples[t].roll_degrees)) for t in ticks])
    dt=np.diff(ticks)/fps;delta=np.diff(coordinates,axis=0);velocity=delta/dt[:,None]
    edges=[(a,b) for a,b in zip(ticks,ticks[1:])];within=[i for i,(a,b) in enumerate(edges)
        if not any(a<x<=b for x in boundary_ticks)]
    accelerations=[]
    for i in range(1,len(velocity)):
        if i-1 in within and i in within:
            accelerations.append(float(np.linalg.norm((velocity[i]-velocity[i-1])/((dt[i]+dt[i-1])/2))))
    boundaries=[]
    for b in boundary_ticks:
        i=ticks.index(b);change=coordinates[i]-coordinates[i-1]
        boundaries.append(dict(boundary_tick=b,log_zoom_delta=float(change[0]),
            pan_delta=change[1:3].tolist(),roll_delta_radians=float(change[3]),transform_change_norm=float(np.linalg.norm(change)),
            projected_layout_measurement_required=True))
    return dict(contract='actual_entry_return_added_transform_rates_v1',state='MEASURED',
        actual_input_identity=digest(dict(ticks=ticks,fps=fps,samples=coordinates.tolist(),boundary_ticks=boundary_ticks)),
        boundaries=boundaries,maximum_within_shot_velocity=max((float(np.linalg.norm(velocity[i])) for i in within),default=0.),
        maximum_within_shot_acceleration=max(accelerations,default=0.),
        maximum_boundary_transform_change=max(x['transform_change_norm'] for x in boundaries),
        body_motion_velocity_penalized=False,projected_layout_certified=False,quality_approved=False)


def joint_review_shot_measurements(*,scope,program_identity,incidents,camera_motion,known_joint_state,actor_state):
    """Consume native, head and joint evidence under the SAME actual cameras."""
    if not isinstance(scope,ContinuousReviewShotScope) or len(incidents)!=2:
        raise ValueError('joint continuous shot comparison requires both declared incidents')
    if (tuple(x['boundary_tick'] for x in incidents)!=scope.incidents
            or any(x.get('program_identity')!=program_identity for x in incidents)
            or tuple(x['boundary_tick'] for x in camera_motion['boundaries'])!=scope.incidents):
        raise ValueError('continuous shot evidence borrowed a camera, program or return boundary')
    rows=[];states=[known_joint_state,actor_state]
    for x in incidents:
        n,p,h=x['native_action'],x['projected_action'],x['head_connection']
        if not(n['route_identity']==p['route_identity']==h['route_identity']):
            raise ValueError('joint shot evidence borrowed another native comparison route')
        pinputs={(r['side'],r['tick']):r['registered_query_identity'] for r in p.get('actual_projected_inputs',[])}
        hinputs={(r['side'],r['tick']):r.get('registered_query_identity',
            None if r['face'] is None else r['face']['registered_query_identity']) for r in h.get('actual_inputs',[])}
        if pinputs!=hinputs:raise ValueError('head and anatomical action used different actual cameras')
        def occurrences(rows,head):
            result={}
            for row in rows:
                key=(row['side'],row['tick'])
                if key in result:raise ValueError('joint shot evidence reused an actual occurrence request')
                value=row.get('face') if head else row
                occurrence=None if value is None else tuple(value.get(k) for k in
                    ('native_parent','selected_head','native_head_identity'))
                result[key]=None if occurrence is None or all(k is None for k in occurrence) else occurrence
            return result
        pa=occurrences(p.get('actual_projected_inputs',[]),False)
        ha=occurrences(h.get('actual_inputs',[]),True)
        if set(pa)!=set(ha) or any(pa[k] is not None and ha[k] is not None and pa[k]!=ha[k] for k in pa):
            raise ValueError('head and anatomical action used different actual actor/raw occurrences')
        measured=bool(n.get('same_time_action_supported') and p.get('state')=='MEASURED' and h.get('state')=='MEASURED')
        if measured and (not pa or any(v is None or any(k is None for k in v) for v in (*pa.values(),*ha.values()))):
            raise ValueError('joint shot measurement lacks actual actor/raw occurrence bindings')
        states.append('PASS' if measured else 'UNKNOWN')
        rows.append(dict(boundary_tick=x['boundary_tick'],native_action=n.get('exploration_cost'),
            projected_position=p.get('projected_position_error'),projected_velocity=p.get('projected_velocity_error'),
            head_center=h.get('face_center_position_error'),head_scale=h.get('face_scale_log_error'),
            head_velocity=h.get('face_center_velocity_error'),complete_numeric_comparison=measured))
    if not set(states)<={'PASS','FAIL','UNKNOWN'}:raise ValueError('shot duties require explicit three-state receipts')
    return dict(contract='joint_actual_camera_continuous_shot_comparison_v1',scope_identity=scope.identity,
        program_identity=program_identity,state='FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'MEASURED',
        incidents=rows,camera_motion=camera_motion,actor_state=actor_state,known_joint_state=known_joint_state,
        unknown_metrics_dropped=False,professional_quality_certified=False,quality_approved=False)


def select_continuous_review_shot(candidates,*,independent_acceptable_limits,limits_calibrated):
    """Finite normalized joint objective; KEEP is not a self-motion zero score.

    Limits/metric units come from independent controls, never a donor's best
    score. Raw uncalibrated comparisons are useful, but cannot force adoption.
    Each candidate carries editorial benefit declared independently of costs.
    """
    if not isinstance(candidates,(list,tuple)) or len(candidates)>224 or type(limits_calibrated) is not bool:
        raise ValueError('continuous review selection exceeded its declared finite realization batch')
    reasons=[];eligible=[]
    required=('native_action','projected_position','projected_velocity','head_center','head_scale','head_velocity')
    camera_metrics=dict(added_camera_velocity='maximum_within_shot_velocity',
        added_camera_acceleration='maximum_within_shot_acceleration',boundary_transform_change='maximum_boundary_transform_change')
    if limits_calibrated and (set(independent_acceptable_limits)!=set(required)|set(camera_metrics)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in independent_acceptable_limits.values())):
        raise ValueError('joint shot selection requires independent positive same-unit limits')
    for c in candidates:
        m=c['measurement'];reason=None
        if m['state']!='MEASURED':reason='actual_incident_or_scope_duty_unresolved'
        elif not limits_calibrated:reason='independent_match_controls_unresolved'
        elif not c.get('predeclared_editorial_benefit'):reason='no_independent_editorial_need'
        else:
            values=[r[k] for r in m['incidents'] for k in required]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                reason='joint_measurement_incomplete'
            elif any(r[k]>independent_acceptable_limits[k] for r in m['incidents'] for k in required):
                reason='independent_acceptable_match_exceeded'
            else:
                motion=m.get('camera_motion',{})
                changes={k:motion.get(v) for k,v in camera_metrics.items()}
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in changes.values()):
                    reason='actual_added_camera_rates_incomplete'
                elif any(changes[k]>independent_acceptable_limits[k] for k in camera_metrics):
                    reason='independent_added_camera_limit_exceeded'
                else:
                    match=max(sum(r[k]/independent_acceptable_limits[k] for k in required)/len(required) for r in m['incidents'])
                    intervention=sum(changes[k]/independent_acceptable_limits[k] for k in camera_metrics)/len(camera_metrics)
                    eligible.append((match+intervention,m['program_identity'],c))
        reasons.append(dict(program_identity=m['program_identity'],reason=reason))
    selected=min(eligible,key=lambda p:(p[0],p[1]))[2] if eligible else None
    return dict(result='KEEP_ORIGINAL' if selected is None else 'SELECTED_REVIEW_HYPOTHESIS',
        selected=selected,dropout_audit=reasons,forced_technical_winner=False,quality_approved=False)
