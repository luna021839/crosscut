"""Automatic selected-edge registration using the existing source-first session."""
from dataclasses import replace
import hashlib
import math

from .direct_member_editorial_motion import DirectMemberEditorialMotionConfig
from .direct_member_target_face import FOCAL_FACE_BASIS
from .direct_member_transform import map_box_to_portrait_fill
from .focal_pixel_containment_support import measure_focal_pixel_containment
from .joint_dissolve_registration import (OverlapFaceObservation,fit_joint_dissolve_registration,
    materialize_joint_dissolve_registration,review_joint_dissolve_containment,shorten_joint_incoming_return,
    joint_pose_anchor_observations)
from .joint_dissolve_pose import source_focal_overlap_pose
from .render import TransitionKind,DIRECT_MEMBER_POSE_DISSOLVE_REASON,quantized_segment_frame_counts


def register_automatic_member_dissolves(*, render_segments, transitions, visual_selection,
        eligible_request_source_indices, common_master_start, focal_provider,
        fps=29.97, config=None, registration_cache=None, progress_callback=None):
    """Register selected effects only; keep failed proposals for ordinary pixel QA.

No source/partition search, downloader, model construction or alternate render
is started here. The same focal provider owns both selection and registration.
All measurement calls use the actual source-local clocks of the render plan.
    """
    settings=config or DirectMemberEditorialMotionConfig()
    if isinstance(fps,bool) or not math.isfinite(fps) or fps<=0:
        raise ValueError('automatic joint camera fps must be finite and positive')
    segments=list(render_segments);effects=list(transitions)
    eligible=tuple(eligible_request_source_indices);selected=tuple(visual_selection.source_indices)
    if len(segments)!=len(selected) or len(effects)!=len(segments)-1 or len(set(eligible))!=len(eligible):
        raise ValueError('automatic joint camera source/timeline domain is invalid')
    geometries=tuple(visual_selection.source_frame_geometries)
    if len(geometries)!=len(eligible):raise ValueError('automatic joint camera needs every source geometry')
    reverse={request:compact for compact,request in enumerate(eligible)}
    geometry_by_request={request:geometries[compact] for request,compact in reverse.items()}
    for segment,compact in zip(segments,selected):
        if not 0<=compact<len(eligible) or segment.source_index!=eligible[compact]:
            raise ValueError('automatic joint camera selected source differs from render source')
    counts=quantized_segment_frame_counts(segments,fps)
    samples={};pixel_cache={};rows=[];owned_incoming=set()
    solves=registration_cache if registration_cache is not None else {}
    def sample(request,local):
        key=(request,local)
        if key not in samples:samples[key]=focal_provider.sample(reverse[request],local)
        return samples[key]
    def box(value):
        return value.bbox_xywh if value is not None and value.face_count==1 and value.alignment_geometry_basis==FOCAL_FACE_BASIS else None
    for index,effect in enumerate(effects):
        if effect.kind is not TransitionKind.FACE_DISSOLVE or effect.reason!=DIRECT_MEMBER_POSE_DISSOLVE_REASON:
            continue
        row=dict(boundary_index=index,applied=False,quality_approval=False)
        rows.append(row)
        pair=tuple(segments[index:index+2])
        compose_previous=index in owned_incoming
        if pair[0].source_index==pair[1].source_index:
            row['reason']='same_source_boundary';continue
        if effect.frame_count<2 or effect.frame_count>64:
            row['reason']='unsupported_overlap_frame_count';continue
        if any(s.crop is not None or s.pan_crop is not None or s.outgoing_overlap_transform is not None
               or (s.editorial_transform is not None and not (side==0 and compose_previous)
                   and any(abs(p.zoom-1)>1e-9 or abs(p.center_x-.5)>1e-9 or abs(p.center_y-.5)>1e-9 or abs(p.roll_degrees)>1e-9 for p in s.editorial_transform.keyframes)) for side,s in enumerate(pair)):
            row['reason']='existing_camera_ownership_preserved';continue
        pair=tuple(s if side==0 and compose_previous else replace(s,editorial_transform=None)
                   for side,s in enumerate(pair))
        cut=common_master_start+pair[1].master_start
        native={s.source_index:tuple(sorted({float(t) for span in visual_selection.span_validity[index:index+2]
            for entry in span.sources if entry.eligible_source_index==reverse[s.source_index]
            for t in entry.native_scene_cut_source_seconds})) for s in pair}
        rates={s.source_index:s.source_playback_rate for s in pair}
        if any(any((s.source_end if side==0 else s.source_start)<t<=
                   (s.source_end if side==0 else s.source_start)
                   +(effect.frame_count-1)/fps*s.source_playback_rate
                   for t in native[s.source_index]) for side,s in enumerate(pair)):
            row['reason']='native_cut_in_overlap';continue
        def pixels(request,local):
            key=(request,local,rates[request],native[request])
            if key in pixel_cache:return pixel_cache[key]
            step=rates[request]/fps
            times=tuple(local+n*step for n in range(-2,3))
            values=tuple(box(sample(request,t)) if t>=0 else None for t in times)
            before=next((i for i in (1,0) if values[i] is not None),None)
            after=next((i for i in (3,4) if values[i] is not None),None)
            support=None
            if before is not None and after is not None:
                local_times=times[before:after+1]
                frames=tuple(focal_provider.faces._samplers[reverse[request]].frame_at(t) for t in local_times)
                support=measure_focal_pixel_containment(source_index=request,frames=frames,
                    source_times=local_times,focal_boxes=values[before:after+1],query_index=2-before,
                    fps=fps,native_scene_cut_seconds=native[request])
            pixel_cache[key]=support
            return support
        observations=[];failure=None
        for side,segment in enumerate(pair):
            local_start=segment.source_end if side==0 else segment.source_start
            locals=tuple(local_start+n/fps*segment.source_playback_rate for n in range(effect.frame_count))
            if any(locals[0]<t<=locals[-1] for t in native[segment.source_index]):
                failure='native_cut_in_overlap';break
            seq=[]
            for n,local in enumerate(locals):
                observed=box(sample(segment.source_index,local));basis='observed_focal_face'
                if observed is None:
                    support=pixels(segment.source_index,local)
                    if support is None:failure='focal_overlap_not_measured';break
                    observed=support.bbox_xywh;basis='bidirectionally_tracked_focal_face'
                try:
                    seq.append(OverlapFaceObservation(segment.source_index,cut+n/fps,local,
                        map_box_to_portrait_fill(observed,geometry_by_request[segment.source_index]),basis))
                except ValueError:
                    failure='focal_face_outside_portrait_fill';break
            if failure:break
            observations.append(tuple(seq))
        pose_resolver=getattr(focal_provider,'selector_geometry',None)
        pose_observations=None
        if failure == 'focal_overlap_not_measured' and pose_resolver is not None:
            # A turned/occluded face is not an absent tracked performer. Keep
            # every exact target joint and use ONE common anatomical basis on
            # both sides; never mix a face box with a body box or invent a face.
            pose_observations=tuple(tuple(source_focal_overlap_pose(
                pose_resolver(reverse[segment.source_index],local),
                source_index=segment.source_index,
                expected_geometry_source_index=reverse[segment.source_index],
                master_seconds=cut+n/fps,source_seconds=local,
                frame_geometry=geometry_by_request[segment.source_index])
                for n,local in enumerate(tuple(
                    (segment.source_end if side==0 else segment.source_start)
                    +n/fps*segment.source_playback_rate for n in range(effect.frame_count))))
                for side,segment in enumerate(pair))
            anchors,anchor_audit=joint_pose_anchor_observations(pose_observations)
            row['pose_anchor_observations']=anchor_audit
            if anchors is not None:
                observations=anchors;failure=None
        if failure:
            row['reason']=failure;continue
        limits=tuple(min(settings.maximum_boundary_zoom,geometry_by_request[s.source_index].lossless_zoom_capacity*settings.maximum_boundary_upscale_ratio) for s in pair)
        if min(limits)<1:
            row['reason']='output_exceeds_source_camera_capacity';continue
        if pose_resolver is not None and pose_observations is None:
            # sample() above already resolved these exact source frames in the
            # shared session; this retrieves their source-focal screen joints.
            pose_observations=tuple(tuple(source_focal_overlap_pose(
                pose_resolver(reverse[obs.source_index],obs.source_seconds),
                source_index=obs.source_index,
                expected_geometry_source_index=reverse[obs.source_index],
                master_seconds=obs.master_seconds,source_seconds=obs.source_seconds,
                frame_geometry=geometry_by_request[obs.source_index])
                for obs in seq) for seq in observations)
        key=hashlib.sha256(repr((observations,pose_observations,fps,limits,settings.maximum_center_shift,
            settings.maximum_face_residual_position_mismatch,settings.maximum_face_residual_scale_mismatch)).encode()).hexdigest()
        row['registration_cache_hit']=key in solves
        if key not in solves:
            solves[key]=fit_joint_dissolve_registration(*observations,fps=fps,maximum_zooms=limits,
                maximum_center_shift=settings.maximum_center_shift,
                maximum_position_mismatch=settings.maximum_face_residual_position_mismatch,
                maximum_scale_mismatch=settings.maximum_face_residual_scale_mismatch,
                pose_observations=pose_observations)
        tracks,fit=solves[key];row['registration']=fit
        if tracks is None:
            row['reason']='joint_solver_no_verified_solution';continue
        try:
            a,b,registered=materialize_joint_dissolve_registration(left_segment=pair[0],right_segment=pair[1],
                left_observations=observations[0],right_observations=observations[1],tracks=tracks,
                transition=effect,common_master_start=common_master_start,fps=fps,ramp_seconds=settings.ramp_seconds,
                incoming_transition=effects[index-1] if compose_previous else None)
        except ValueError as error:
            row.update(reason='camera_composition_unavailable',detail=str(error));continue
        containment=review_joint_dissolve_containment(segments=(a,b),transitions=(registered,),
            source_geometries=geometry_by_request,geometry_provider=sample,pixel_containment_provider=pixels,
            fps=fps,frame_counts=counts[index:index+2],minimum_face_visible=settings.minimum_dense_face_visible_fraction,
            minimum_body_visible=settings.minimum_face_associated_subject_visible_fraction,
            legacy_face_only_maximum_zoom=settings.legacy_face_only_maximum_zoom,
            legacy_face_only_maximum_center_shift=settings.legacy_face_only_maximum_center_shift)
        row['containment']=containment
        if not containment['passed']:
            alternate,return_audit=shorten_joint_incoming_return(b.editorial_transform,registered,containment,fps=fps)
            row['visibility_timed_return']=return_audit
            if alternate is not None:
                row['initial_containment']=containment
                alternate_b=replace(b,editorial_transform=alternate)
                revised=review_joint_dissolve_containment(segments=(a,alternate_b),transitions=(registered,),
                    source_geometries=geometry_by_request,geometry_provider=sample,pixel_containment_provider=pixels,
                    fps=fps,frame_counts=counts[index:index+2],minimum_face_visible=settings.minimum_dense_face_visible_fraction,
                    minimum_body_visible=settings.minimum_face_associated_subject_visible_fraction,
                    legacy_face_only_maximum_zoom=settings.legacy_face_only_maximum_zoom,
                    legacy_face_only_maximum_center_shift=settings.legacy_face_only_maximum_center_shift)
                row['containment']=revised
                containment=revised
                if revised['passed']:b=alternate_b
        if not containment['passed']:
            row['reason']='joint_camera_containment_not_verified';continue
        segments[index:index+2]=(a,b);effects[index]=registered
        owned_incoming.add(index+1)
        row.update(applied=True,reason='joint_camera_registered_pending_decoded_qa',
            previous_overlap_camera_preserved=compose_previous,
            nominal_camera_composed=compose_previous)
        if progress_callback:progress_callback(dict(boundary_index=index,applied=True))
    return tuple(segments),tuple(effects),dict(contract='automatic_member_joint_dissolve_v1',
        boundaries=rows,applied_count=sum(r['applied'] for r in rows),source_measurement_queries=len(samples),
        additional_render_count=0,source_selection_changed=False,quality_approval=False)
