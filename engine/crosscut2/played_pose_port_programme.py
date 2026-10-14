"""Explicit E trial: one physical focal, played pose, two-ended camera DAG.

Untrained engineering ranking, not professional/editorial acceptance. This
module reuses source caches; it does not acquire, train or reanalyse a source.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
from time import monotonic
import numpy as np
from .fixed_holds import State,_curve_at,make_states
from .shot_camera_paths import make_paths
from .physical_display_observations import focal_metadata
from .display_source_catalogue import SourceCatalogue
from .boundary_port_domain import make
from .boundary_arc_costs import ArcCosts
from .boundary_port_dp import solve
from .played_event_support import PlayedEventSupport
from .played_partial_pose import features,matrix


class PlayedPosePortPrior:
    port_camera_prior=True
    def __init__(self):
        files=['played_pose_port_programme.py','played_partial_pose.py','display_source_catalogue.py',
            'physical_display_observations.py','boundary_port_domain.py','boundary_arc_costs.py',
            'boundary_port_dp.py','boundary_camera_arc.py']
        implementation=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in files)).hexdigest()
        self.artifact=dict(contract='played_partial_pose_physical_focal_boundary_port_E2_v2',
            source_code_sha256=implementation,supervision_kinds=[],new_fit=False,weak_models_used=False,
            authored_cost='played_pair_support*(D_face_rate+mean_fixed_semantic_played_pose_channels)',
            native_cost='physical_support*(same_uncapped_played_face_geometry+unmeasured_partial_pose_.5)',
            native_geometry_not_flattened_or_visibility_attenuated=True,
            native_missing_pose_not_fabricated_as_measured=True,
            own_activity_bound_to_same_displayed_face_sample=True,
            maximum_measured_face_height_ratio=1.5,maximum_measured_displacement_in_face_heights=1.,
            unknown_pose_residual=.5,model_only_shrinkage=.5,shrinkage_is_not_anatomical_probability=True,
            measured_body_action_cost_used=True,pair_conditioned_cameras_added=True,
            physical_GROUP_focal_owns_content_and_pose=True,anatomical_ownership_certified=False,
            paired_proposal_global_limit_per_node=8,roll_degrees=0.,
            digital_camera_cannot_create_own_motion=True,
            content_integration='nearest cached face/body within6 outputframes and same native shot; not a dense track',
            new_source_neural_calls=0,new_source_analysis_decode=0,
            professional_quality_approved=False,normal_app_activation_approved=False)
        self.artifact['content_sha256']=hashlib.sha256(json.dumps(self.artifact,sort_keys=True).encode()).hexdigest()


class SelectedPorts:
    def __init__(self,nodes,fps):self.nodes=nodes;self.fps=fps;self.details={}
    def selected(self,n,left,right):return dict(self.details[n,left,right])


def select(nodes,banks,poses,styles,*,fps,source_events,body_states,native_ports,music,prior,
           identity_hints=None,part_observer=None,check=lambda:None,maximum_preparation_seconds=240.,maximum_seconds=180.):
    from .event_holds import fragment_costs
    started=monotonic();phase='preparation';selection_start=None
    def live():
        check()
        origin,limit=(started,maximum_preparation_seconds) if phase=='preparation' else (selection_start,maximum_seconds)
        if monotonic()-origin>=limit:raise TimeoutError('one shared E '+phase+' allowance exhausted')
    # E consumes raw per-face observations and four physical source paths,
    # not the legacy actor x camera projected reward rows. Do not allocate
    # those416 discarded rows just to recover104raw native observations.
    states,rows,counts=make_states(nodes,banks,poses,fps=fps,source_events=source_events,
        body_states=body_states,camera_families=('native',),check=live)
    part_receipts=[]
    if part_observer is not None:
        rows,part_receipts=(part_observer(states,rows,check=live) if 'check' in inspect.signature(part_observer).parameters
            else part_observer(states,rows))
    paths,camera_receipt=make_paths(nodes,banks,source_events,check=live)
    receipt=dict(source_local_appearance_clusters=counts,raw_actor_states_prepared_once=len(states),
        native_part_corroboration=part_receipts,camera_domain=camera_receipt,
        unused_actor_camera_projection_rows_allocated=False,source_local_member_hints_consumed=False)
    cats=[]
    for s in range(len(banks)):
        live();raw=[d for state,d in zip(states,rows) if state.source==s and state.family=='native']
        cats.append(SourceCatalogue(nodes,banks[s],None if poses is None else poses[s],raw,styles[s],source_events[s],fps=fps))
    del states,rows,raw
    entries,exits,dense,domain=make(nodes,cats,paths,fps=fps,check=live)
    events=[sorted(set(map(int,xs))|set(map(int,bank['frames']))) for xs,bank in zip(source_events,native_ports)]
    support=PlayedEventSupport(events,final=int(nodes[-1]),fps=fps)
    fragments=fragment_costs(nodes,source_events,fps=fps,support_seconds=12/fps)
    arcs=ArcCosts(nodes,entries,exits,cats,dense,native_ports,support,fragments,fps=fps,check=live)
    cut_cache={};cut_statistics=[]
    def cut(n):
        live()
        if n not in cut_cache:
            left=features([p.observation for p in exits[n]],fps=fps)
            right=features([p.observation for p in entries[n]],fps=fps)
            rates,admitted=matrix(left,right)
            a=np.array([p.source for p in exits[n]]);b=np.array([p.source for p in entries[n]])
            weights=support.cut_weights(int(nodes[n]),np.arange(len(cats)))[a[:,None],b[None,:]]
            admitted&=a[:,None]!=b[None,:]
            cut_cache[n]=(rates*weights,admitted)
            cut_statistics.append(dict(frame=int(nodes[n]),exit_ports=len(a),entry_ports=len(b),admitted_pairs=int(admitted.sum())))
        return cut_cache[n]
    # This node-only potential changes no complete-programme objective. It
    # removes exact common content prefix cancellation and does not add edits.
    potential=np.min(np.concatenate(arcs.prefix,axis=0),axis=0)[np.asarray(nodes,int)]
    def holds(i,j):
        costs,allowed,codes=arcs(i,j)
        return costs-(potential[j]-potential[i]),allowed,codes
    live();prepared=monotonic();phase='selection';selection_start=prepared
    port_runs,objective,dp=solve(nodes,[[p.source for p in ps] for ps in entries],[[p.source for p in ps] for ps in exits],
        start_cost=np.zeros(len(entries[0])),hold_cost=holds,cut_cost=cut,minimum_frames=12,check=live)
    score=float(objective[0]+potential[-1]-potential[0]);selected=SelectedPorts(nodes,fps)
    runs=[];chosen_states=[];chosen_rows=[];parts=[];frozen=[];cumulative=0.
    for k,((i,j,p,q),code) in enumerate(zip(port_runs,dp['selected_arc_codes'])):
        live();source=entries[i][p].source;first,final=map(int,np.asarray(nodes)[[i,j]])
        # Replay ONLY the code saved at the actual DP improvement. No argmin,
        # camera refit, boundary shift or source choice is performed here.
        curve=arcs.replay(i,j,p,q,code);ticks=np.arange(first,final)
        cost=float(arcs.direct_cost(source,first,curve[None])[0]+fragments(i,j)[source])
        incoming=0. if not k else float(cut(i)[0][port_runs[k-1][3],p])
        cumulative+=cost+incoming
        grid_camera=_curve_at(ticks,curve,np.clip(np.asarray(nodes)[:,None]+[-1,0],first,final-1))
        heads,indices,ambiguity,scene=focal_metadata(banks[source]['heads'],grid_camera)
        chosen_states.append(State(source,None,'boundary_arc_'+str(code)))
        # Exact collinear-run removal, not an approximate camera re-fit.
        bends=np.flatnonzero(np.any(np.diff(curve,n=2,axis=0)!=0.,axis=1))+1
        knots=np.array(sorted({0,len(curve)-1,*map(int,bends)}),int)
        chosen_rows.append(dict(ticks=ticks[knots],curve=curve[knots],heads=heads,indices=indices,formation=scene==2,
            scored_camera_frames=ticks,scored_camera_values=curve,
            cameras=grid_camera,displayed_focal_heads=heads,displayed_focal_indices=indices,
            focal_ambiguity=ambiguity,scene_kind=scene,pose_cost_used=True,
            camera_path_reason='frozen_two_ended_port_arc_no_post_selection_fit'))
        runs.append((i,j,k))
        digest=hashlib.sha256(curve.tobytes()).hexdigest()
        frozen.append(dict(first_frame=first,final_frame=final,source=source,entry_port=p,exit_port=q,
            entry_geometry_hash=entries[i][p].geometry_hash,exit_geometry_hash=exits[j][q].geometry_hash,
            camera_arc_code=int(code),camera_path_sha256=digest,per_frame_camera_count=len(curve)))
        parts.append(dict(first_frame=first,final_frame=final,source=source,
            components=dict(realized_hold_content_camera_native_fragment=cost,entry_played_face_pose=incoming),cumulative_cost=cumulative))
        if k:
            previous=port_runs[k-1];a=exits[i][previous[3]];b=entries[i][p]
            detail=matrix(features([a.observation],fps=fps),features([b.observation],fps=fps),details=True)
            if not detail['allowed'][0,0]:raise AssertionError('selected cut violates unchanged D face gate')
            observed=[a.observation,b.observation]
            predicted=[]
            for side,obs in enumerate(observed):
                head=obs['head'];cam=obs['camera_strip']
                for at in (0,-1):
                    c=cam[at];center=.5+c[0]*(head[:2]-c[1:])
                    predicted.append(dict(center=list(map(float,center)),height=float(head[2]*c[0]),confidence=float(head[3]),
                        source_face_index=obs['head_index'],face_native_sample_is_actual_endpoint=at==(-1 if side==0 else 0)))
            selected.details[i,k-1,k]=dict(cost=incoming,appearance_cosine=float('nan'),
                played_face_rate=float(detail['face_rate'][0,0]),played_partial_pose_rate=float(detail['pose_rate'][0,0]),
                fixed_semantic_pose_rates=detail['fixed_semantic_channel_rates'][0,0].tolist(),
                played_face_measured=bool(detail['measured_face'][0,0]),
                actual_pose_native_input_frames=[int(v) for o in observed for v in o['native_indices']],
                actual_camera_strips=[o['camera_strip'].tolist() for o in observed],
                expected_output_focal=predicted,
                partial_pose_cost_used=True,ownership_certified=False,unmeasured_pose_is_not_perfect=True)
    if not math.isclose(cumulative,score,rel_tol=1e-8,abs_tol=1e-7):
        raise AssertionError('E selected actual camera/pose decomposition differs from DP: '+str((cumulative,score)))
    live();receipt.update(objective_contract=prior.artifact['contract'],fixed_states=len(cats)*4,
        physical_group_focal_bound=True,actor_alias_changes_cost=False,
        selected_hold_cost_decomposition=parts,frozen_selected_camera_arcs=frozen,
        preparation_seconds=prepared-started,selection_seconds=monotonic()-prepared,
        boundary_port_domain=domain,boundary_port_DP=dp,
        event_preference=dict(fixed_connection_prior=prior.artifact,model_sha256=prior.artifact['content_sha256'],
            supervision_kinds=[],objective_vector=[score,int(objective[1])]),
        port_pair_statistics=cut_statistics,authored_cuts=int(objective[1]),
        long_prefix_combinations=arcs.long_combinations,short_direct_paths=arcs.short_paths,
        camera_patch_batch_bytes_lower_bound=arcs.peak_batch_bytes,
        content_observation_scope='nearest cached sample <=6 frames same native shot; no dense tracking claim',
        source_NN_and_analysis_decode_during_new_selection=0,body_observations_loaded=poses is not None,
        played_pose_cost_used=True,ownership_certified=False,
        native_unknown_body_motion_not_fabricated=True,lower_body_semantic_observations_used=False,
        camera_paths_scored_and_realized_without_refit=True,post_selection_retimed=False,
        force_source_use_or_maximum_hold=False,stage_edit_transformer_used=False,
        normal_app_path_executed=False,professional_quality_approved=False)
    return tuple(runs),tuple(chosen_states),chosen_rows,score,receipt,selected
