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
from .focal_appearance import cosine as focal_appearance_cosine


class PlayedPosePortPrior:
    port_camera_prior=True
    def __init__(self,*,person_extents=False,editorial_opportunities=False,scene_readout=False,played_interval_utility=False,
                 shot_duration_model=None,recycle_static_capacity=False):
        if scene_readout and not editorial_opportunities:
            raise ValueError('SceneReadout requires the fixed editorial opportunity path')
        if played_interval_utility and (not editorial_opportunities or scene_readout):
            raise ValueError('played interval utility requires fixed events without the rejected readout objective')
        if shot_duration_model is not None:
            from .visible_shot_duration import Model as DurationModel
            if not isinstance(shot_duration_model,DurationModel) or not played_interval_utility:
                raise ValueError('an explicit TRAIN duration model requires the actual played-interval path')
        self.shot_duration_model=shot_duration_model
        if recycle_static_capacity and not played_interval_utility:
            raise ValueError('recovered camera capacity requires actual interval camera scoring')
        self.recycle_static_capacity=bool(recycle_static_capacity)
        files=['played_pose_port_programme.py','played_partial_pose.py','display_source_catalogue.py','fixed_holds.py',
            'physical_display_observations.py','boundary_port_domain.py','boundary_arc_costs.py',
            'boundary_port_dp.py','boundary_camera_arc.py','focal_appearance.py','person_extent.py','source_clock_action.py',
            'featured_member_state.py','terminal_stencil.py','dense_subject_catalogue.py','dense_source_action.py',
            'subject_window_bank.py','subject_action_match.py','subject_cut_moments.py','editorial_opportunities.py',
            'programme_framing.py','programme_source_clock.py','subject_opportunity_probe.py','editorial_roles.py',
            'scene_readout.py','readout_port_dp.py','scene_targets.py','sequence_context.py',
            'sequence_preference_model.py','played_context.py','context_port_dp.py','sequence_context_kernel.py',
            'playback_intent.py','playback_blocks.py','ordered_context.py','ordered_preference_model.py','block_played_context.py',
            'context_long_relax.py','held_playback_inputs.py','playback_input_batch.py',
            'played_interval_utility.py','observed_reframing.py','matched_entry_delivery.py','visible_shot_duration.py',
            'source_member_scores.py']
        implementation=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in files)).hexdigest()
        self.artifact=dict(contract=('played_partial_pose_person_visible_extent_boundary_port_E3_v1' if person_extents
            else 'played_partial_pose_physical_focal_boundary_port_E2_v2'),
            source_code_sha256=implementation,supervision_kinds=[],new_fit=False,weak_models_used=False,
            authored_cost='played_pair_support*(D_face_rate+mean_fixed_semantic_played_pose_channels+anonymous_appearance_risk)',
            native_cost='physical_support*(same_uncapped_played_face_geometry+unmeasured_partial_pose_.5+same_anonymous_appearance_risk)',
            native_geometry_not_flattened_or_visibility_attenuated=True,
            native_missing_pose_not_fabricated_as_measured=True,
            own_activity_bound_to_same_displayed_face_sample=True,
            maximum_measured_face_height_ratio=1.5,maximum_measured_displacement_in_face_heights=1.,
            unknown_pose_residual=.5,model_only_shrinkage=.5,shrinkage_is_not_anatomical_probability=True,
            measured_body_action_cost_used=True,pair_conditioned_cameras_added=True,
            paired_subject_pinned_before_camera=True,camera_cannot_silently_reassign_pair_subject=True,
            physical_GROUP_focal_owns_content_and_pose=True,anatomical_ownership_certified=False,
            actual_focal_anonymous_appearance_reported=True,appearance_used_in_objective=True,
            appearance_is_soft_risk_not_identity_gate=True,unknown_appearance_risk=.5,
            sparse_same_clock_source_action_consumed=True,same_clock_context_is_not_dense_actor_certification=True,
            optional_contemporary_dense_subject_context_consumed_before_source_selection=True,
            dense_context_missing_body_never_recovers_stale_normal_pose=True,
            matched_action_change_credit='realized_incoming_hold_seconds_up_to_two_observed_beats * measured_change_density',
            change_density_uses='weakest_observed(common_clock_action,anonymous_appearance,Hellinger_style_change,native_quality,played_seam_shape,actual_played_advantage_over_UNKNOWN)',
            change_cues_are_not_multiplied_as_independent_probabilities=True,
            known_different_member_gets_no_same_actor_change_credit=True,
            ending_pixel_constraint='only_final_hold; measured_new_opaque_stencil_avoided_if_an_observed_clean_tail_exists',
            unmeasured_tail_never_excludes_source=True,all_questionable_tails_preserve_coverage=True,
            paired_credit_requires_its_actual_outgoing_proposal=True,unmeasured_or_static_change_credit=0.,
            change_credit_is_unfitted_engineering_not_human_preference_calibration=True,
            recurrent_member_hints_are_soft_not_certified=True,featured_state_uses_original_native_views_not_proposed_crops=True,
            person_visible_extent_fallback_used=bool(person_extents),
            person_extent_is_not_face_size_identity_or_camera_cause=True,
            paired_proposal_global_limit_per_node=8,roll_degrees=0.,
            digital_camera_cannot_create_own_motion=True,
            content_integration='nearest cached face/body within6 outputframes and same native shot; not a dense track',
            new_source_neural_calls=0,new_source_analysis_decode=0,
            professional_quality_approved=False,normal_app_activation_approved=False)
        self.artifact['fixed_editorial_opportunities']=bool(editorial_opportunities)
        if editorial_opportunities:
            self.artifact.update(contract='fixed_episode_typed_scene_point_boundary_port_E4_v2',
                matched_action_change_credit='once_per_fixed_music_episode * actually_observed_incoming_exposure',
                content_integration='target_ROI_component_deficits_UNKNOWN_.5_role_crop_no_background_black_reward',
                proposal_quality_contract='same_target_ROI_native_detail_visibility_role_no_background_v1',
                quality_validity_preserved_separately=True,digital_zoom_creates_no_native_ROI_quality=True,
                transition_integration='six_frame_instantaneous_geometry_appearance_plus_observed_window_pose',
                role_handoff='original_family_pre_camera_soft_interpretation_shared_with_native',
                ordinary_scene_changes='unpaired_paths_pay_finite_full_discrepancy_without_claiming_face_match',
                maximum_measured_face_height_ratio=None,maximum_measured_displacement_in_face_heights=None,
                paired_match_face_height_ratio_gate=1.5,paired_match_displacement_in_face_heights_gate=1.,
                opportunity_units_are_unfitted_engineering=True)
        self.artifact['scene_readout']=bool(scene_readout)
        if scene_readout:
            self.artifact.update(contract='scene_readout_observed_net_burden_boundary_port_E5_v2',
                readout_state='cost_and_residual_debt_frontier_per_port_and_paid_bit',
                exact_global_optimum_claimed=False,readout_label_limit=8,
                tau='fixed_median_preselection_music_episode_seconds_not_human_cognition',
                raw_head_observations_not_dense_track_or_camera_cause=True,
                observed_net_changes_not_discarded_for_wide_sample_spacing=True,
                known_original_faces_clipped_by_view_are_not_missing_sensor_evidence=True,
                unknown_exposure_recovers_no_readout_burden=True)
        self.artifact['played_interval_utility']=bool(played_interval_utility)
        if played_interval_utility:
            self.artifact.update(contract='pair_bound_actual_interval_action_composed_image_boundary_port_E6_v2',
                opportunity_credit='actual_pair_density_times_observed_entry_target_delivery_once_per_fixed_episode',
                source_continuation_change_credit=False,
                seven_frame_entry_credit_used=False,entry_density_times_hold_duration=False,
                complete_arc_camera_argmin_after_actual_utility=True,
                refiner_articulated_seam_observation_used_before_source_cut_selection=True,
                observed_image_motion_is_not_actor_or_camera_cause=True)
        if shot_duration_model is not None:
            self.artifact.update(visible_shot_duration_model=shot_duration_model.artifact,
                contract='pair_bound_actual_interval_soft_visible_duration_E6_v3')
        if recycle_static_capacity:
            self.artifact.update(recycle_identical_static_capacity=True,
                same_total_port_bound=True,all_old_physical_candidates_retained=True,
                paired_proposal_global_limit_per_node=None,baseline_paired_limit_per_node=8,
                global_port_limit_per_side='four_per_source_plus_eight_unchanged')
        self.artifact['content_sha256']=hashlib.sha256(json.dumps(self.artifact,sort_keys=True).encode()).hexdigest()


class SelectedPorts:
    def __init__(self,nodes,fps):self.nodes=nodes;self.fps=fps;self.details={}
    def selected(self,n,left,right):return dict(self.details[n,left,right])


def select(nodes,banks,poses,styles,*,fps,source_events,body_states,native_ports,music,prior,
           identity_hints=None,part_observer=None,person_observer=None,subjects=None,opportunity_observer=None,
           check=lambda:None,maximum_preparation_seconds=240.,maximum_seconds=180.,
           sequence_preference_model=None,sequence_source_bindings=None,sequence_context_kernel='numpy'):
    from .event_holds import fragment_costs
    started=monotonic();phase='preparation';selection_start=None
    person_extents=bool(prior.artifact.get('person_visible_extent_fallback_used',False))
    editorial=bool(prior.artifact.get('fixed_editorial_opportunities',False))
    readout=bool(prior.artifact.get('scene_readout',False))
    played=bool(prior.artifact.get('played_interval_utility',False))
    sequence=sequence_preference_model is not None
    if sequence_context_kernel not in ('numpy','numba') or (not sequence and sequence_context_kernel!='numpy'):
        raise ValueError('explicit sequence model required for a changed context kernel')
    if sequence:
        from .sequence_preference_model import Model as SequenceModel
        if not isinstance(sequence_preference_model,SequenceModel):
            raise ValueError('an explicit new fixed-time sequence model is required; no same-event checkpoint')
        if not editorial or readout:
            raise ValueError('sequence model replaces rejected E5 and requires the E4 ROI/action base')
        if played:raise ValueError('the explicit pointwise interval trial does not replace learned sequence semantics')
        sequence_preference_model.verify()
        if sequence_source_bindings is None:raise ValueError('sequence model needs actual physical source bindings')
    elif sequence_source_bindings is not None:
        raise ValueError('source sequence bindings supplied without a sequence model')
    def live():
        check()
        origin,limit=(started,maximum_preparation_seconds) if phase=='preparation' else (selection_start,maximum_seconds)
        if monotonic()-origin>=limit:raise TimeoutError('one shared E '+phase+' allowance exhausted')
    if hasattr(check,'input_lifecycle'):live.input_lifecycle=check.input_lifecycle
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
        terminal_stencil_observations=[b.get('terminal_stencil_observation') for b in banks],
        unused_actor_camera_projection_rows_allocated=False,source_local_member_hints_consumed=False)
    cats=[]
    if subjects is not None and len(subjects)!=len(banks):raise ValueError('one dense subject binding per physical source required')
    face_members=[None]*len(banks);featured=[None]*len(banks);roles=None
    if identity_hints is not None:
        from .featured_member_state import bind_hints,consensus
        face_members,native_ports=bind_hints(banks,identity_hints,native_ports)
        featured,featured_receipt,role_evidence=consensus(banks,face_members,return_evidence=True)
        if editorial:
            from .editorial_roles import FeatureRoles
            roles=FeatureRoles(nodes,role_evidence)
        receipt.update(source_local_member_hints_consumed=True,featured_member_state=featured_receipt)
        receipt['individual_member_score_fields']=[dict(source=s,
            gallery_identity=bank.get('member_score_gallery_identity',''),
            supported_face_occurrences=int((bank.get('member_score_support',np.zeros(0))>0).sum()),
            raw_scores_are_not_probabilities=True,forced_member_labels=False,
            raw_scores_or_embeddings_exported=False) for s,bank in enumerate(banks)]
    framing=None
    if editorial:
        from .programme_framing import evidence as framing_evidence
        framing,receipt['original_programme_framing']=framing_evidence(banks)
    for s in range(len(banks)):
        live();raw=[d for state,d in zip(states,rows) if state.source==s and state.family=='native']
        cats.append(SourceCatalogue(nodes,banks[s],None if poses is None else poses[s],raw,styles[s],source_events[s],fps=fps,
            member_hints=face_members[s],featured_factor=featured[s],subjects=None if subjects is None else subjects[s],
            editorial_content=editorial,framing_evidence=framing))
    del states,rows,raw
    music=np.zeros(len(nodes)) if music is None else np.asarray(music,float)
    if music.shape!=(len(nodes),) or not np.isfinite(music).all() or np.any((music<0)|(music>1)):
        raise ValueError('observed per-node musical strengths required')
    opportunities=None
    if editorial:
        from .editorial_opportunities import Opportunities
        opportunities=Opportunities(nodes,music,fps=fps,
            families=[b.get('consensus_family',('source',s)) for s,b in enumerate(banks)])
    entries,exits,dense,domain=make(nodes,cats,paths,fps=fps,check=live,opportunities=opportunities,roles=roles,
        coalesce_static_ports=sequence,recycle_static_capacity=getattr(prior,'recycle_static_capacity',False))
    if opportunity_observer is not None:
        if opportunities is None:raise ValueError('pre-selection subject probe requires fixed editorial opportunities')
        receipt['opportunity_subject_observations']=opportunity_observer(catalogues=cats,opportunities=opportunities,check=live)
        # The final camera/actor pruning must see the NEW raw observations.
        # Nothing has been selected yet; do not patch an old chosen movie.
        from .editorial_opportunities import Opportunities
        opportunities=Opportunities(nodes,music,fps=fps,
            families=[b.get('consensus_family',('source',s)) for s,b in enumerate(banks)])
        entries,exits,dense,domain=make(nodes,cats,paths,fps=fps,check=live,opportunities=opportunities,roles=roles,
            coalesce_static_ports=sequence,recycle_static_capacity=getattr(prior,'recycle_static_capacity',False))
    if person_observer is not None:
        # Freeze additive raw evidence BEFORE any DP cost is constructed.
        # The callback may not alter the finite camera/port domain or fit
        # utility. Selection below consumes only the resulting numeric bank.
        receipt['person_rectangle_observations']=person_observer(nodes=nodes,catalogues=cats,
            entries=entries,exits=exits,dense=dense,native_ports=native_ports,check=live)
    events=[sorted(set(map(int,xs))|set(map(int,bank['frames']))) for xs,bank in zip(source_events,native_ports)]
    support=PlayedEventSupport(events,final=int(nodes[-1]),fps=fps)
    fragments=fragment_costs(nodes,source_events,fps=fps,support_seconds=12/fps)
    duration_model=getattr(prior,'shot_duration_model',None)
    duration_costs=None if duration_model is None else duration_model.table(nodes,source_events,fps=fps)
    duration_sources=[np.asarray([p.source for p in ports],int) for ports in entries]
    interval_utility=None
    if played:
        from .played_interval_utility import PlayedIntervalUtility
        interval_utility=PlayedIntervalUtility.from_catalogues(cats,opportunities,check=live,delivery_in_base=False)
    arcs=ArcCosts(nodes,entries,exits,cats,dense,native_ports,support,fragments,fps=fps,check=live,
        person_extents=person_extents,opportunity_episodes=None if opportunities is None else opportunities.episodes,
        roles=roles,point_geometry=editorial,
        readout_tau=None if not readout else float(np.median(np.diff(opportunities.boundaries))/fps),
        preserve_camera_alternatives=sequence or played,played_interval_utility=interval_utility)
    matched_delivery=None
    if played:
        from .matched_entry_delivery import MatchedEntryDelivery
        matched_delivery=MatchedEntryDelivery(arcs,opportunities)
    contextual=None
    if sequence:
        contextual=sequence_preference_model.prepare(nodes=nodes,entries=entries,exits=exits,catalogues=cats,arcs=arcs,
            native_ports=native_ports,fps=fps,source_bindings=sequence_source_bindings,check=live,kernel=sequence_context_kernel)
    cut_cache={};cut_statistics=[];entry_density={};cut_transfers={}
    beat_frames=np.asarray(nodes)[music>0]
    exposure_seconds=0. if len(beat_frames)<3 else float(2.*np.median(np.diff(beat_frames))/fps)
    def cut(n):
        live()
        if n not in cut_cache:
            left=features([p.observation for p in exits[n]],fps=fps,person_extents=person_extents)
            right=features([p.observation for p in entries[n]],fps=fps,person_extents=person_extents)
            detail=matrix(left,right,details=True)
            if played:
                from .played_interval_utility import articulated_seam_evidence
                detail=articulated_seam_evidence(detail,exits[n],entries[n])
            a=np.array([p.source for p in exits[n]]);b=np.array([p.source for p in entries[n]])
            from .editorial_roles import apply,event_energy,ordinary_pairs
            if roles is not None:
                detail=apply(detail,roles.at(int(nodes[n]),left['member_hints'][:,None],right['member_hints'][None,:]))
            admitted=detail['allowed'].copy()
            if editorial:admitted|=ordinary_pairs(exits[n],entries[n])
            weights=support.cut_weights(int(nodes[n]),np.arange(len(cats)))[a[:,None],b[None,:]]
            admitted&=a[:,None]!=b[None,:]
            density=np.zeros(len(b))
            for right_index,port in enumerate(entries[n]):
                ids={p for p in port.proposal_ids if p[0]=='paired'}
                if not ids:continue
                matching=np.array([bool(ids.intersection(left_port.proposal_ids)) for left_port in exits[n]])
                # A duration-dependent novelty benefit belongs to this actual
                # actor/camera pair, never to an unrelated preceding source.
                admitted[:,right_index]&=matching
                for left_index in np.flatnonzero(matching&admitted[:,right_index]):
                    density[right_index]=max(density[right_index],
                        float(detail['matched_action_change_density'][left_index,right_index]*music[n]) if opportunities is None else
                        opportunities.density(n,exits[n][left_index].observation,port.observation,
                            a[left_index],port.source,detail['matched_action_change_density'][left_index,right_index]))
            entry_density[n]=density
            cut_cache[n]=(event_energy(detail,weights,fps=fps,point_geometry=editorial),admitted)
            if readout:
                from .scene_readout import change
                cut_transfers[n]=change(detail['face_rate'],unit=6./fps)
            cut_statistics.append(dict(frame=int(nodes[n]),exit_ports=len(a),entry_ports=len(b),admitted_pairs=int(admitted.sum()),
                visible_person_extent_comparable_port_pairs=int((detail['person_extent_measured']&(a[:,None]!=b[None,:])).sum()),
                unknown_face_port_pairs_with_person_extent=int((detail['person_extent_measured']&~detail['measured_face']&(a[:,None]!=b[None,:])).sum())))
        return cut_cache[n]
    # This node-only potential changes no complete-programme objective. It
    # removes exact common content prefix cancellation and does not add edits.
    potential=np.min(np.concatenate(arcs.prefix,axis=0),axis=0)[np.asarray(nodes,int)]
    def holds(i,j):
        if sequence or played:costs,allowed=arcs.camera_alternatives(i,j)
        elif readout:costs,allowed,transfer=arcs.alternatives(i,j)
        else:costs,allowed,codes=arcs(i,j)
        if i and i not in entry_density:cut(i)
        density=entry_density.get(i,np.zeros(len(entries[i])))
        reward=(min(float(nodes[j]-nodes[i])/fps,exposure_seconds)*density[:,None]
            if opportunities is None else 0.)
        if duration_costs is not None:
            costs=costs+duration_costs[i,j,duration_sources[i]][:,None,None]
        if sequence:return costs-(potential[j]-potential[i]),allowed
        if played:
            codes=np.argmin(np.where(allowed,costs,np.inf),axis=-1)
            return (np.take_along_axis(costs,codes[...,None],axis=-1).squeeze(-1)
                -(potential[j]-potential[i]),allowed.any(-1),codes)
        return costs-reward-(potential[j]-potential[i]),allowed,transfer if readout else codes
    def claimed_holds(i,j):
        # The ordinary callback has just built ALL legal camera alternatives.
        # A claimed edit chooses again only after actual pair-bound delivery
        # has been scored, instead of reusing the cheapest isolated camera.
        if arcs.last_camera_alternatives[:2]!=(i,j):
            raise AssertionError('claimed hold must consume this exact candidate camera domain')
        costs,allowed=arcs.last_camera_alternatives[2:]
        if duration_costs is not None:
            costs=costs+duration_costs[i,j,duration_sources[i]][:,None,None]
        density=entry_density.get(i,np.zeros(len(entries[i])))
        value,valid,codes,benefit=matched_delivery.choose(i,j,density,costs,allowed)
        return value-(potential[j]-potential[i]),valid,codes,benefit
    observed_exposure={}
    def credit(i,j):
        if not i:return np.zeros(len(entries[0]))
        if i not in entry_density:cut(i)
        if i not in observed_exposure:
            observed_exposure[i]=np.asarray([opportunities.exposure(i,p.observation) for p in entries[i]])
        exposure=np.minimum(observed_exposure[i],float(nodes[j]-nodes[i])/fps)
        return opportunities.credit(i,entry_density[i],exposure)
    live();prepared=monotonic();phase='selection';selection_start=prepared
    if sequence:
        from .context_port_dp import solve as context_solve
        port_runs,objective,dp=context_solve(nodes,[[p.source for p in ps] for ps in entries],[[p.source for p in ps] for ps in exits],
            start_cost=np.zeros(len(entries[0])),hold_cost=holds,cut_cost=cut,context_cost=contextual.incremental,
            history_key=contextual.equivalent_key,history_frames=contextual.history_frames,minimum_frames=12,check=live,
            context_lower_bound=contextual.lower_bound,
            long_context_batch=getattr(contextual,'long_candidates',None),
            episodes=opportunities.episodes,credit=credit,workspace_limit_bytes=200*2**20-contextual.workspace_reserved_bytes)
    elif readout:
        from .readout_port_dp import solve as readout_solve
        port_runs,objective,dp=readout_solve(nodes,[[p.source for p in ps] for ps in entries],[[p.source for p in ps] for ps in exits],
            start_cost=np.zeros(len(entries[0])),hold_cost=holds,cut_cost=cut,cut_transfer=lambda n:cut_transfers[n],
            minimum_frames=12,check=live,episodes=opportunities.episodes,credit=credit)
    else:
        port_runs,objective,dp=solve(nodes,[[p.source for p in ps] for ps in entries],[[p.source for p in ps] for ps in exits],
            start_cost=np.zeros(len(entries[0])),hold_cost=holds,cut_cost=cut,minimum_frames=12,check=live,
            opportunity_episodes=None if opportunities is None else opportunities.episodes,
            hold_credit=None if opportunities is None or played else credit,
            claim_hold_cost=claimed_holds if played else None)
    score=float(objective[0]+potential[-1]-potential[0]);selected=SelectedPorts(nodes,fps)
    context_parts=None
    if contextual is not None:
        from .context_port_dp import ArcRef
        references=[ArcRef(*run,int(code)) for run,code in zip(port_runs,dp['selected_arc_codes'])]
        context_parts,full_context=contextual.replay(references)
        if not np.allclose(context_parts,dp['selected_context_costs'],rtol=1e-9,atol=1e-9):
            raise AssertionError('selected fixed-time context differs from independent whole-film replay')
        if not math.isclose(sum(context_parts),full_context,rel_tol=1e-9,abs_tol=1e-9):
            raise AssertionError('fixed-time context windows were lost or paid twice at cuts')
    runs=[];chosen_states=[];chosen_rows=[];parts=[];frozen=[];cumulative=0.;debt=0.
    for k,((i,j,p,q),code) in enumerate(zip(port_runs,dp['selected_arc_codes'])):
        live();source=entries[i][p].source;first,final=map(int,np.asarray(nodes)[[i,j]])
        # Replay ONLY the code saved at the actual DP improvement. No argmin,
        # camera refit, boundary shift or source choice is performed here.
        curve=arcs.replay(i,j,p,q,code);ticks=np.arange(first,final)
        cost=float(arcs.direct_cost(source,first,curve[None])[0]+fragments(i,j)[source])
        pacing=0. if duration_costs is None else float(duration_costs[i,j,source])
        effect=(dict(observed_action_delivery_credit=0.,composed_image_motion_delta=0.,adjustment=0.)
            if interval_utility is None else interval_utility.replay(source,first,final,curve))
        density=entry_density.get(i,np.zeros(len(entries[i])))
        if opportunities is None:change_credit=min(float(final-first)/fps,exposure_seconds)*float(density[p])
        else:change_credit=dp['opportunity_claims'][k]['credit']
        if played and dp['opportunity_claims'][k]['claimed']:
            actual_credit=matched_delivery.replay(i,j,p,curve,float(density[p]))
            if not math.isclose(change_credit,actual_credit,rel_tol=1e-9,abs_tol=1e-10):
                raise AssertionError('claimed action benefit differs from actual selected target/camera replay')
        incoming=0. if not k else float(cut(i)[0][port_runs[k-1][3],p])
        sequence_cost=0. if context_parts is None else context_parts[k]
        if readout:
            from .scene_readout import apply
            if k:
                debt,extra=apply(cut_transfers[i][port_runs[k-1][3],p],debt);sequence_cost+=float(extra)
            if not math.isclose(debt,dp['selected_readout_states'][k]['entry_debt'],rel_tol=1e-8,abs_tol=1e-8):
                raise AssertionError('selected readout entry state differs from actual sequence')
            transfer=arcs.readouts[source].summary(first,curve[None],hold_first=first)[0]
            debt,extra=apply(transfer,debt);sequence_cost+=float(extra)
            if not math.isclose(debt,dp['selected_readout_states'][k]['exit_debt'],rel_tol=1e-8,abs_tol=1e-8):
                raise AssertionError('selected readout exit state differs from exact camera replay')
        cumulative+=cost+pacing-change_credit+incoming+sequence_cost+effect['adjustment']
        grid_camera=_curve_at(ticks,curve,np.clip(np.asarray(nodes)[:,None]+[-1,0],first,final-1))
        heads,indices,ambiguity,scene=focal_metadata(banks[source]['heads'],grid_camera)
        chosen_states.append(State(source,None,'boundary_arc_'+str(code)))
        boundary_contexts=[]
        for n,port in ((i,entries[i][p]),(j,exits[j][q])):
            context=port.observation.get('dense_subject_context')
            if context is not None and 0<n<len(nodes)-1:
                boundary_contexts.append(dict(output_frame=int(nodes[n]),
                    offsets=context['clock'].output_frames.tolist(),native_indices=context['clock'].native_indices.tolist()))
        # Exact collinear-run removal, not an approximate camera re-fit.
        bends=np.flatnonzero(np.any(np.diff(curve,n=2,axis=0)!=0.,axis=1))+1
        knots=np.array(sorted({0,len(curve)-1,*map(int,bends)}),int)
        chosen_rows.append(dict(ticks=ticks[knots],curve=curve[knots],heads=heads,indices=indices,formation=scene==2,
            scored_camera_frames=ticks,scored_camera_values=curve,
            cameras=grid_camera,displayed_focal_heads=heads,displayed_focal_indices=indices,
            focal_ambiguity=ambiguity,scene_kind=scene,pose_cost_used=True,
            dense_boundary_contexts=boundary_contexts,
            camera_path_reason='frozen_two_ended_port_arc_no_post_selection_fit'))
        runs.append((i,j,k))
        digest=hashlib.sha256(curve.tobytes()).hexdigest()
        frozen.append(dict(first_frame=first,final_frame=final,source=source,entry_port=p,exit_port=q,
            entry_geometry_hash=entries[i][p].geometry_hash,exit_geometry_hash=exits[j][q].geometry_hash,
            camera_arc_code=int(code),camera_path_sha256=digest,per_frame_camera_count=len(curve)))
        parts.append(dict(first_frame=first,final_frame=final,source=source,
            components=dict(realized_hold_content_camera_native_fragment=cost,entry_played_face_pose=incoming,
                visible_shot_duration_prior=pacing,
                measured_matched_action_change_credit=-change_credit,
                observed_played_action_delivery_credit=(-effect['observed_action_delivery_credit']
                    if interval_utility is not None and interval_utility.delivery_in_base else 0.),
                composed_image_motion_replacement=effect['composed_image_motion_delta'],
                residual_readout_interaction=sequence_cost if readout else 0.,
                fixed_time_sequence_preference=sequence_cost if sequence else 0.),cumulative_cost=cumulative))
        if k:
            previous=port_runs[k-1];a=exits[i][previous[3]];b=entries[i][p]
            detail=matrix(features([a.observation],fps=fps,person_extents=person_extents),
                features([b.observation],fps=fps,person_extents=person_extents),details=True)
            if played:
                from .played_interval_utility import articulated_seam_evidence
                detail=articulated_seam_evidence(detail,[a],[b])
            if roles is not None:
                from .editorial_roles import apply
                detail=apply(detail,roles.at(int(nodes[i]),a.observation['recurrent_member_hint'],b.observation['recurrent_member_hint']))
            if not cut(i)[1][previous[3],p]:raise AssertionError('selected cut violates its declared transition admission')
            from .editorial_roles import ordinary_pairs
            ordinary=bool(editorial and ordinary_pairs([a],[b])[0,0])
            observed=[a.observation,b.observation]
            predicted=[]
            for side,obs in enumerate(observed):
                head=obs['head'];cam=obs['camera_strip']
                for at in (0,-1):
                    c=cam[at];center=.5+c[0]*(head[:2]-c[1:])
                    predicted.append(dict(center=list(map(float,center)),height=float(head[2]*c[0]),confidence=float(head[3]),
                        source_face_index=obs['head_index'],face_native_sample_is_actual_endpoint=at==(-1 if side==0 else 0)))
            selected.details[i,k-1,k]=dict(cost=incoming,appearance_cosine=focal_appearance_cosine(a.observation,b.observation),
                appearance_used_in_objective=True,anonymous_appearance_is_not_member_identity=True,
                anonymous_appearance_rate=float(detail['appearance_rate'][0,0]),
                anonymous_appearance_measured=bool(detail['appearance_measured'][0,0]),
                individual_member_score_influence=float(detail['individual_member_score_influence'][0,0]),
                feature_handoff_strength=float(detail.get('feature_handoff_strength',np.zeros((1,1)))[0,0]),
                editorial_transition_kind='SCENE_CHANGE' if ordinary else 'MATCH_ACTION_CANDIDATE',
                successful_face_match_claimed=not ordinary and bool(detail['allowed'][0,0]),
                recurrent_member_hints=[int(o['recurrent_member_hint']) for o in (a.observation,b.observation)],
                same_clock_source_action_used=bool(detail['same_clock_source_action_used'][0,0]),
                contemporary_dense_action_used=bool(detail['contemporary_dense_action_used'][0,0]),
                contemporary_dense_action_observed=bool(detail['contemporary_dense_action_observed'][0,0]),
                matched_action_change_density=float(detail['matched_action_change_density'][0,0]),
                actual_played_match_advantage=float(detail['actual_played_match_advantage'][0,0]),
                realized_incoming_change_credit=change_credit,
                played_face_rate=float(detail['face_rate'][0,0]),played_partial_pose_rate=float(detail['pose_rate'][0,0]),
                actual_arm_direction_used=bool(detail.get('articulated_arm_direction_measured',np.zeros((1,1),bool))[0,0]),
                actual_arm_direction_residual=(float(detail['articulated_arm_direction'][0,0])
                    if detail.get('articulated_arm_direction_measured',np.zeros((1,1),bool))[0,0] else None),
                played_geometry_rate=float(detail['geometry_rate'][0,0]),
                person_visible_extent_gap=float(detail['person_visible_extent_gap'][0,0]),
                person_visible_extent_measured=bool(detail['person_extent_measured'][0,0]),
                person_extent_fallback_used=bool(person_extents and not detail['measured_face'][0,0]
                    and detail['person_extent_measured'][0,0]),
                person_extent_is_not_face_size_or_identity=True,
                fixed_semantic_pose_rates=detail['fixed_semantic_channel_rates'][0,0].tolist(),
                played_face_measured=bool(detail['measured_face'][0,0]),
                actual_pose_native_input_frames=[int(v) for o in observed for v in o['native_indices']],
                actual_dense_context_native_inputs=[None if o.get('dense_subject_context') is None
                    else o['dense_subject_context']['clock'].native_indices.tolist() for o in observed],
                actual_camera_strips=[o['camera_strip'].tolist() for o in observed],
                expected_output_focal=predicted,
                partial_pose_cost_used=True,ownership_certified=False,unmeasured_pose_is_not_perfect=True)
    if not math.isclose(cumulative,score,rel_tol=1e-8,abs_tol=1e-7):
        raise AssertionError('E selected actual camera/pose decomposition differs from DP: '+str((cumulative,score)))
    live();receipt.update(objective_contract=prior.artifact['contract'],fixed_states=len(cats)*4,
        scene_readout_consumed=readout,
        scene_readout_observations=[] if not readout else [r.receipt() for r in arcs.readouts],
        pre_DP_scalar_camera_argmin_used=not (readout or sequence or played),
        conditional_unpaid_or_paid_camera_argmin_used=played,
        camera_argmin_includes_actual_played_interval_utility=played,
        played_interval_utility=None if interval_utility is None else interval_utility.receipt(),
        matched_entry_action_delivery=None if matched_delivery is None else matched_delivery.receipt(),
        visible_shot_duration_prior=None if duration_model is None else dict(
            model_sha256=duration_model.artifact['content_sha256'],
            references=duration_model.artifact['references'],
            preferred_log_seconds_mean=duration_model.mean,preferred_log_seconds_scale=duration_model.scale,
            includes_retained_native_cuts=True,table_bytes=duration_costs.nbytes,
            hard_minimum_or_maximum_added=False,aesthetic_weight_fitted=False),
        sequence_preference_consumed=sequence,
        sequence_preference_model_sha256=None if not sequence else sequence_preference_model.artifact['content_sha256'],
        sequence_preference_context=None if contextual is None else contextual.receipt(),
        original_feature_roles=None if roles is None else roles.receipt(),
        instantaneous_geometry_support_not_shortened_by_adjacent_native_cuts=editorial,
        editorial_opportunities=None if opportunities is None else opportunities.receipt(),
        actual_opportunity_supported_entry_ports=sum(int((observed_exposure[n]>0).sum()) for n in observed_exposure),
        actual_opportunity_useful_supported_entry_ports=sum(int(((observed_exposure[n]>0)&(entry_density[n]>0)).sum())
            for n in observed_exposure),
        dense_subject_observations=[] if subjects is None else [s.receipt() for s in subjects],
        physical_group_focal_bound=True,actor_alias_changes_cost=False,
        selected_hold_cost_decomposition=parts,frozen_selected_camera_arcs=frozen,
        preparation_seconds=prepared-started,selection_seconds=monotonic()-prepared,
        boundary_port_domain=domain,boundary_port_DP=dp,
        change_credit_exposure_seconds=exposure_seconds,
        event_preference=dict(fixed_connection_prior=prior.artifact,model_sha256=prior.artifact['content_sha256'],
            supervision_kinds=[] if not sequence else ['human_av_context_preference'],
            sequence_model=None if not sequence else sequence_preference_model.artifact,
            objective_vector=[score,int(objective[1])]),
        port_pair_statistics=cut_statistics,authored_cuts=int(objective[1]),
        long_prefix_combinations=arcs.long_combinations,short_direct_paths=arcs.short_paths,
        whole_viewport_paths_scored_before_source_selection=arcs.whole_viewport_paths_scored,
        camera_patch_batch_bytes_lower_bound=arcs.peak_batch_bytes,
        content_observation_scope='nearest cached sample <=6 frames same native shot; no dense tracking claim',
        source_NN_and_analysis_decode_during_new_selection=0,body_observations_loaded=poses is not None,
        played_pose_cost_used=True,ownership_certified=False,
        native_unknown_body_motion_not_fabricated=True,lower_body_semantic_observations_used=False,
        camera_paths_scored_and_realized_without_refit=True,post_selection_retimed=False,
        force_source_use_or_maximum_hold=False,stage_edit_transformer_used=False,
        normal_app_path_executed=False,professional_quality_approved=False)
    return tuple(runs),tuple(chosen_states),chosen_rows,score,receipt,selected
