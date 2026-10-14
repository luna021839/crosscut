"""Opt-in coherent event-second programme, separate from rejected C-W v18.

Integrated bounded visibility/framing loss + measured intrinsic camera events
+ real entry K events. No duration target/maximum, forced source diversity,
preselected reference timestamps or source/unknown veto. Camera paths scored
here are the paths realized, without post-selection re-fitting/retiming.
"""
from dataclasses import replace
import math
from time import monotonic
import numpy as np
from crosscut.render import EditorialTransformKeyframe,EditorialTransformTrack,RenderSegment
from .fixed_holds import _curve_at
from .event_states import build
from .event_ports import Ports,rho
from .opportunity_dp import choose_opportunities


def native_events(nodes,states,data,ports,*,fps):
    """Actual native-camera face flanks; missing is a finite neutral prior.

    These caches contain faces, not arm motion. Their benefit is not fabricated
    from a body observation made at another time. No old arbitrary .25/1.25
    native-cut fees are carried into the event-second objective.
    """
    interval=np.zeros((len(states),len(nodes)-1));through=np.zeros((len(states),len(nodes)))
    for i,(state,row) in enumerate(zip(states,data)):
        bank=ports[state.source]
        for event,frame in enumerate(bank['frames']):
            heads=bank['heads'][event,:,0]
            curves=_curve_at(row['ticks'],row['curve'],np.array([frame-1,frame]))
            observed=(heads[:,2]>.019)&(heads[:,3]>=.75)
            risk=.5
            # A native largest face is not an arbitrary actor hypothesis.
            # Without an exact association keep a finite unknown event.
            associated=state.actor is None and bool(bank.get('observed',np.ones(len(bank['frames']),bool))[event])
            if observed.all() and associated:
                centers=.5+curves[:,:1]*(heads[:,:2]-curves[:,1:])
                heights=heads[:,2]*curves[:,0]
                position=np.linalg.norm((centers[0]-centers[1])*[16/9,1.])/max(.005,float(heights.mean()))
                risk=float(rho(max(position,abs(math.log(heights[0]/heights[1])))))
            value=6/fps*risk;at=int(np.searchsorted(nodes,frame))
            if at<len(nodes) and nodes[at]==frame:through[i,at]+=value
            elif 0<at<len(nodes):interval[i,at-1]+=value
    return interval,through


def fragment_costs(nodes,source_events,*,fps,support_seconds):
    """Only lost temporal support in real native fragments, in seconds."""
    sources=[]
    for values in source_events:
        boundaries=np.unique(np.r_[0,np.asarray(values,int),nodes[-1]])
        gaps=np.diff(boundaries)/fps
        prefix=np.r_[0.,np.cumsum(np.maximum(0.,support_seconds-gaps))]
        sources.append((boundaries,prefix))
    def cost(start,end):
        a,b=int(nodes[start]),int(nodes[end]);result=[]
        for boundaries,prefix in sources:
            left=int(np.searchsorted(boundaries,a,side='right')-1)
            right=int(np.searchsorted(boundaries,b-1,side='right')-1)
            if left==right:result.append(max(0.,support_seconds-(b-a)/fps));continue
            first=(boundaries[left+1]-a)/fps;last=(b-boundaries[right])/fps
            middle=prefix[right]-prefix[left+1]
            result.append(max(0.,support_seconds-first)+middle+max(0.,support_seconds-last))
        return np.asarray(result)
    return cost


def select(nodes,banks,poses,styles,*,fps,source_events,body_states,native_ports,music,check=lambda:None,
           maximum_seconds=180.,identity_hints=None,part_observer=None,maximum_preparation_seconds=240.,
           preference_model=None,preference_workspace=None):
    started=monotonic();next_check=started;phase_started=started
    phase_limit=maximum_seconds if part_observer is None else maximum_preparation_seconds
    def live():
        nonlocal next_check
        now=monotonic()
        if now-phase_started>=phase_limit:raise TimeoutError('single coherent event preparation/selection allowance exhausted')
        if now>=next_check:check();next_check=now+.1
    states,data,receipt=build(nodes,banks,poses,styles,fps=fps,source_events=source_events,
        body_states=body_states,check=live,identity_hints=identity_hints,part_observer=part_observer)
    measured=Ports(nodes,states,data,fps=fps,music=music)
    prepared=monotonic();phase_started=prepared;phase_limit=maximum_seconds
    native_interval,through=native_events(nodes,states,data,native_ports,fps=fps)
    unary=np.asarray([d['event_loss_rate'] for d in data])+native_interval/(np.diff(nodes)/fps)
    count=len(states);matrix_bytes=count*count*4
    if matrix_bytes*32>200*1024**2:raise ValueError('one transient vectorized port workspace exceeds 200MiB')
    built=0
    def ports(n):
        nonlocal built
        live();result=measured.matrix(n);built+=1
        if built%40==0:
            print(__import__('json').dumps(dict(phase='coherent_event_ports',nodes=built,total_nodes=len(nodes),
                states=count,elapsed_seconds=monotonic()-started)),flush=True)
        return result
    support=12/fps;fragments=fragment_costs(nodes,source_events,fps=fps,support_seconds=support)
    preference_parts=None;preference_receipt=None;secondary=None
    if preference_model is None:
        runs,score,secondary=choose_opportunities(nodes,fps=fps,unary=unary,ports=ports,
            state_sources=np.asarray([s.source for s in states]),source_events=source_events,
            same_show=np.zeros((count,count),np.float32),through=through,check=live,minimum_hold=support,
            event_seconds=True,native_cost=fragments)
    else:
        import tempfile
        from pathlib import Path
        from .preference_losses import prepare
        from .native_preference import prepare as prepare_native
        from .preference_dp import choose
        if preference_workspace is None:raise ValueError('explicit transient preference workspace required')
        Path(preference_workspace).mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.event-preference-',dir=preference_workspace) as temporary:
            H,K,preference_receipt=prepare(measured,preference_model,workspace=temporary,check=live)
            native_preference,native_preference_lower,native_receipt=prepare_native(measured,preference_model,
                native_ports=native_ports,source_events=source_events,check=live)
            preference_receipt['native_evaluation']=native_receipt
            runs,objective=choose(nodes,fps=fps,unary=unary,through=through,preference_holds=H,
                preference_intervals=native_preference,preference_ports=K,
                state_sources=np.asarray([s.source for s in states]),source_events=source_events,
                minimum_hold=support,native_cost=fragments,check=live)
            score,secondary=objective;preference_parts=[];selected={}
            for i,(first,end,index) in enumerate(runs):
                entry=0. if not i else float(K(first)[runs[i-1][2],index])
                if i:selected[first,runs[i-1][2],index]=entry
                preference_parts.append(dict(internal_event_preferences=float(H[index,first+1:end].sum()),entry_preference=entry,
                    evaluation_native_preference_upper=float(native_preference[index,first:end].sum())))
            preference_receipt['selected_evaluation_native_preference_lower']=float(sum(
                native_preference_lower[index,first:end].sum() for first,end,index in runs))
            measured.selected_preference_costs=selected
        preference_receipt['objective_vector']=[score,secondary]
        preference_receipt['model_sha256']=preference_model.artifact['content_sha256']
        preference_receipt['supervision_kinds']=preference_model.artifact['supervision_kinds']
        preference_receipt['professional_quality_approved']=False
    parts=[];total=0.
    for i,(first,end,index) in enumerate(runs):
        # Decomposition uses the frozen chosen pair, not another all-state
        # search-sized matrix. Preserve the DP's float32 port rounding.
        entry=0. if not i else (1/fps if preference_parts is not None else
            float(np.float32(measured.selected(first,runs[i-1][2],index)['cost']))+1/fps)
        terms=dict(integrated_visibility_framing=float(unary[index,first:end]@(np.diff(nodes)[first:end]/fps)),
            internal_native_events=float(through[index,first+1:end].sum()),
            native_fragment_support=float(fragments(first,end)[states[index].source]),entry_event=entry,
            short_final_support=max(0.,support-(nodes[end]-nodes[first])/fps))
        total+=sum(terms.values());parts.append(dict(first_frame=int(nodes[first]),final_frame=int(nodes[end]),
            source=states[index].source,actor_hypothesis=states[index].actor,camera_family=states[index].family,
            components=terms,cumulative_cost=total))
    if not math.isclose(total,secondary if preference_parts is not None else score,rel_tol=1e-7,abs_tol=1e-6):
        raise AssertionError('coherent event decomposition differs from exact programme objective')
    if preference_parts is not None:
        pref_total=sum(sum(row.values()) for row in preference_parts)
        if not math.isclose(pref_total,score,rel_tol=1e-7,abs_tol=1e-6):raise AssertionError('preference H/K accounting differs')
        receipt['event_preference']=preference_receipt;receipt['selected_event_preferences']=preference_parts
    receipt.update(objective_contract='bounded_joint_motion_style_event_seconds_v1' if preference_parts is None else
        'lexicographic_same_event_preference_visibility_v1',experimental=True,
        selected_hold_cost_decomposition=parts,selection_seconds=monotonic()-prepared,preparation_seconds=prepared-started,
        source_neural_calls_during_selection=0,event_strength_nodes=int(np.count_nonzero(measured.strength)),
        physical_observation_window_seconds=6/fps,minimum_hold_seconds=support,
        port_matrix_nodes_built=built,whole_song_quadratic_tensor_allocated=False,
        camera_paths_scored_and_realized_without_refit=True,legacy_duration_and_risk_units_consumed=False,
        force_source_use_or_maximum_hold=False,normal_app_path_executed=False,
        limb_ownership_certified=False,professional_quality_approved=False)
    return runs,states,data,float(score),receipt,measured


def realize(original,warps,nodes,runs,states,data,*,output,measured):
    segments=[];seams=[]
    for i,(start,end,index) in enumerate(runs):
        state,row=states[index],data[index];first,final=int(nodes[start]),int(nodes[end]);last=final-1
        knots=sorted({first,last,*[int(t) for t in row['ticks'] if first<t<last]})
        values=_curve_at(row['ticks'],row['curve'],knots)
        transform=None if np.all(values==[1.,.5,.5]) else EditorialTransformTrack(tuple(
            EditorialTransformKeyframe((tick-first)/original.fps,float(z),float(x),float(y))
            for tick,(z,x,y) in zip(knots,values)),reason='source_shot_constant_measured_match_candidates')
        a=warps[state.source].master_to_source(original.master_audio_start+first/original.fps)
        b=warps[state.source].master_to_source(original.master_audio_start+final/original.fps)
        segments.append(RenderSegment(state.source,float(a),float((final-first)/original.fps),float(first/original.fps),
            editorial_transform=transform,source_duration=float(b-a)))
        if i:
            old=runs[i-1][2];details=measured.selected(start,old,index)
            if not math.isfinite(details['appearance_cosine']):details['appearance_cosine']=None
            if hasattr(measured,'selected_preference_costs'):
                details['primary_same_event_preference_loss']=measured.selected_preference_costs[start,old,index]
                details['legacy_cost_is_diagnostic_not_optimized']=True
            seams.append(dict(details,output_frame=first,seconds=first/original.fps,left_source=states[old].source,
                right_source=state.source,left_head_index=int(data[old]['indices'][start,0]),
                right_head_index=int(row['indices'][start,1]),left_camera_family=states[old].family,
                right_camera_family=state.family,role='focal' if not(row['formation'][start,1] or data[old]['formation'][start,0]) else 'formation',
                source_local_pose_is_not_verified_limb_ownership=True,post_selection_retimed=False))
    return replace(original,segments=tuple(segments),output=output,transitions=(),authorized_shot_video_clocks=(),
        member_pose_dissolve_requested=None,member_pose_dissolve_effective=None,member_pose_dissolve_count=None),seams


def check_selected_clock(plan,nodes,runs,states,banks,*,check=lambda:None):
    """Report real renderer ordinal drift, without silently retiming ports."""
    from crosscut.render_frame_schedule import probe_render_frame_clock,sampler_frame_schedule
    clocks=[]
    for path in plan.sources:
        check();clocks.append(probe_render_frame_clock(path))
    if any(c is None for c in clocks):
        return dict(all_source_clocks_verified=False,source_observation_ordinals_exact=False,
            maximum_native_ordinal_difference=None,post_selection_retimed=False)
    differences=[];exposed=[];total=0
    for segment,(first,end,index) in zip(plan.segments,runs):
        check();source=states[index].source;clock=clocks[source];size=int(nodes[end]-nodes[first])
        schedule=sampler_frame_schedule(source_start=segment.source_start,source_playback_rate=segment.source_playback_rate,
            native_fps=float(clock.frame_rate),output_fps=plan.fps,frame_count=size,source_frame_count=clock.frame_count)
        expected=banks[source]['native_indices']
        for n in range(first,end+1):
            for side,offset in ((0,-1),(1,0)):
                tick=int(nodes[n]+offset-nodes[first])
                if not 0<=tick<size:continue
                error=int(schedule.source_frame_indices[tick]-expected[n,side]);total+=1
                if error:differences.append(dict(source=source,output_frame=int(nodes[first]+tick),delta=error))
                if tick in (0,size-1):exposed.append(abs(error))
    maximum=max((abs(d['delta']) for d in differences),default=0)
    return dict(all_source_clocks_verified=True,source_observation_ordinals_exact=not differences,
        maximum_native_ordinal_difference=maximum,maximum_exposed_ordinal_difference=max(exposed,default=0),
        measured_grid_inputs_checked=total,differing_inputs=differences,post_selection_retimed=False)
