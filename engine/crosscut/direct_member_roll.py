"""Transactional observed-roll overlay after the direct MEMBER dense solve."""
from dataclasses import replace
import math
import numpy as np

from .direct_member_target_face import FOCAL_FACE_BASIS
from .direct_member_transform import map_box_to_portrait_fill
from .focal_selector_geometry import FocalSelectorGeometry
from .joint_dissolve_pose import source_focal_overlap_pose, paired_overlap_pose_arrays
from .member_roll_geometry import (estimate_pair_roll, project_points, project_box, compose_roll_state,
    roll_ramp_safe_zoom, box_visible_fraction as rotated_visible_fraction)
from .render import EditorialTransformKeyframe, EditorialTransformTrack, minimum_roll_safe_zoom


def bind_registered_member_roll(motion, segments, camera_audit):
    """Report the actual post-registration tracks, including overlap ownership."""
    rolled=tuple(i for i,s in enumerate(segments) if any(track and any(abs(p.roll_degrees)>1e-9
        for p in track.keyframes) for track in (s.editorial_transform,s.outgoing_overlap_transform)))
    if not rolled:
        return motion
    rows=[row for row in motion.member_roll_audit if row.get('reason')!='joint_observed_roll_registered']
    for row in camera_audit['boundaries']:
        evidence=row.get('registration',{}).get('roll',{})
        if row.get('applied') is True and evidence.get('applied') is True:
            index=row['boundary_index']
            rows.append(dict(boundary_index=index,accepted=True,reason='joint_observed_roll_registered',
                evidence=evidence,segment_indices=[index,index+1],
                request_source_indices=[segments[index].source_index,segments[index+1].source_index],
                examined_frame_count=row['containment']['examined_frame_count'],quality_approval=False))
    return replace(motion,segments=tuple(replace(item,track=segments[i].editorial_transform)
        for i,item in enumerate(motion.segments)),roll_event_count=len(rolled),member_roll_audit=tuple(rows))


def _native_cuts(selection, compact, segment_indices):
    spans = tuple(getattr(selection,'span_validity',()))
    if not spans or max(segment_indices)>=len(spans):
        raise ValueError('roll requires source-native cut coverage')
    entries = [entry for index in segment_indices for entry in spans[index].sources
               if entry.eligible_source_index==compact]
    if len(entries)!=len(segment_indices):
        raise ValueError('roll source-native cut domain is incomplete')
    return tuple(sorted({float(t) for entry in entries for t in entry.native_scene_cut_source_seconds}))


def apply_direct_member_roll(base, *, visual_selection, render_segments, transitions,
        source_frame_geometries, pose_geometry_provider, common_master_start, fps, config):
    """Keep the existing camera unless both observed roll tracks pass every gate.

    The provider uses compact source indices and source-local seconds. The
    renderer's selected interval/rate, not a second timewarp or cached caller
    time, determines every query. Dense following ran before this overlay.
    """
    if not callable(pose_geometry_provider):
        return base
    from .direct_member_editorial_motion import DirectMemberFaceSample, _rendered_attention_mismatch
    from .joint_dissolve_registration import review_joint_dissolve_containment
    from .render import quantized_segment_frame_counts
    renders=tuple(render_segments); path=tuple(visual_selection.source_indices)
    counts=quantized_segment_frame_counts(renders,fps)
    tracks=[item.track for item in base.segments]
    rows=[];accepted=[];residuals=dict((i,(p,s)) for i,p,s in base.boundary_residual_attention_mismatches)
    cache={}
    def focal(compact,seconds):
        key=(compact,float(seconds))
        if key not in cache:
            value=pose_geometry_provider(compact,seconds)
            if value is not None and (not isinstance(value,FocalSelectorGeometry)
                    or value.source_index!=compact or abs(value.source_seconds-seconds)>1e-9):
                raise ValueError('roll provider returned another source or query clock')
            cache[key]=value
        return cache[key]
    for boundary,effect in enumerate(transitions):
        if path[boundary]==path[boundary+1] or effect.kind.uses_overlap:
            continue
        row=dict(boundary_index=boundary,accepted=False,quality_approval=False)
        rows.append(row)
        # The overlap planner owns its entire nominal ramp. Do not compose a
        # second rotation through an already registered adjacent effect.
        affected=(boundary,boundary+1)
        if any(transitions[j].kind.uses_overlap for j in (boundary-1,boundary+1)
               if 0<=j<len(transitions)):
            row['reason']='adjacent_overlap_camera_owned';continue
        if any(tracks[i] is not None and any(abs(p.roll_degrees)>1e-9 for p in tracks[i].keyframes)
               for i in affected):
            row['reason']='adjacent_roll_camera_owned';continue
        try:
            geometries=tuple(source_frame_geometries[path[i]] for i in affected)
            if any(g is None for g in geometries):
                raise ValueError('roll requires verified source resolution')
            size=(geometries[0].output_width,geometries[0].output_height)
            if any((g.output_width,g.output_height)!=size for g in geometries):
                raise ValueError('roll output aspect differs between sources')
            cut=common_master_start+renders[boundary+1].master_start
            poses=[]
            cuts=[]
            for side,i in enumerate(affected):
                segment=renders[i];compact=path[i]
                native=_native_cuts(visual_selection,compact,affected);cuts.append(native)
                origin=segment.source_end if side==0 else segment.source_start
                seconds=tuple(origin+n/fps*segment.source_playback_rate for n in (-1,0,1))
                if seconds[0]<0 or any(seconds[0]<t<=seconds[-1] for t in native):
                    raise ValueError('roll pose window crosses a native cut or source start')
                poses.append(tuple(source_focal_overlap_pose(focal(compact,t),source_index=segment.source_index,
                    expected_geometry_source_index=compact,master_seconds=cut+n/fps,source_seconds=t,
                    frame_geometry=geometries[side]) for n,t in zip((-1,0,1),seconds)))
            angle,evidence=estimate_pair_roll(poses,fps=fps,output_size=size)
            row['evidence']=evidence
            if angle is None:
                row['reason']=evidence['reason'];continue
            points,_=paired_overlap_pose_arrays(poses,poses)
            rolls=(-angle/2,angle/2)
            previous=[tracks[i] or EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),)) for i in affected]
            endpoints=((counts[boundary]-1)/fps,0.)
            # Same extra magnification on both sides preserves matched scale.
            multiplier=max(1.,*(roll_ramp_safe_zoom(rolls[s])/previous[s].evaluate(endpoints[s]).zoom
                               for s in (0,1)))
            proposed=[]
            for side,i in enumerate(affected):
                segment=renders[i];compact=path[i];geometry=geometries[side]
                last=(counts[i]-1)/fps
                if last<config.ramp_seconds:
                    raise ValueError('roll lacks a full bounded return ramp')
                ramp=config.ramp_seconds
                times={0.,segment.duration,*[p.time_seconds for p in previous[side].keyframes]}
                times.update(n/fps for n in range(counts[i])
                             if (n/fps>=last-ramp if side==0 else n/fps<=ramp))
                times.add(last-ramp if side==0 else ramp)
                keyframes=[]
                for time in sorted(times):
                    weight=min(1.,max(0.,(time-(last-ramp))/ramp if side==0 else 1-time/ramp))
                    original=previous[side].evaluate(time)
                    if weight<=1e-12:
                        keyframes.append(replace(original,time_seconds=time));continue
                    seconds=segment.source_start+time*segment.source_playback_rate
                    # Exact current focal geometry supplies the pivot. A fixed
                    # box from another frame must not steer a moving performer.
                    observation=source_focal_overlap_pose(focal(compact,seconds),source_index=segment.source_index,
                        expected_geometry_source_index=compact,master_seconds=common_master_start+segment.master_start+time,
                        source_seconds=seconds,frame_geometry=geometry)
                    if observation is None:
                        raise ValueError('dense roll focal pose missing')
                    visible=tuple(observation.points_xyc[j] for j in evidence['common_joint_indices'])
                    if any(p[2]<.5 or not 0<=p[0]<=1 or not 0<=p[1]<=1 for p in visible):
                        raise ValueError('dense roll joints are insufficient')
                    pivot=np.asarray(visible)[:,:2].mean(axis=0)
                    scale=max(1+(multiplier-1)*weight,minimum_roll_safe_zoom(rolls[side]*weight)/original.zoom)
                    point=compose_roll_state(original,rolls[side]*weight,pivot=pivot,zoom_multiplier=scale,
                        output_size=size,time_seconds=time)
                    if (point.zoom>min(config.maximum_boundary_zoom,geometry.lossless_zoom_capacity*config.maximum_boundary_upscale_ratio)+1e-9
                            or math.hypot(point.center_x-.5,point.center_y-.5)>config.maximum_center_shift+1e-9):
                        raise ValueError('dense roll exceeds source capacity')
                    keyframes.append(point)
                proposed.append(EditorialTransformTrack(tuple(keyframes),reason='observed_member_roll_after_dense_camera_v1'))
                active=[p.time_seconds for p in keyframes if abs(p.roll_degrees)>1e-9]
                if active and any(segment.source_start+min(active)*segment.source_playback_rate<t<=
                    segment.source_start+max(active)*segment.source_playback_rate for t in cuts[side]):
                    raise ValueError('roll ramp crosses native source cut')
            before=[];after=[]
            for n in range(3):
                original=[project_points(points[s,n],previous[s].evaluate(endpoints[s]),output_size=size) for s in (0,1)]
                changed=[project_points(points[s,n],proposed[s].evaluate(endpoints[s]),output_size=size) for s in (0,1)]
                before.append(float(np.mean(np.sum((original[0]-original[1])**2,axis=-1))))
                after.append(float(np.mean(np.sum((changed[0]-changed[1])**2,axis=-1))))
            if any(a>b+1e-9 for a,b in zip(after,before)) or sum(after)>.8*sum(before):
                raise ValueError('roll does not improve the existing camera pose')
            reverse={renders[i].source_index:path[i] for i in affected}
            def sample(source,seconds):
                value=focal(reverse[source],seconds)
                if value is None or value.person is None:
                    return None
                body=value.person.bbox_xywh
                if value.face_bbox_xywh is None:
                    if value.body_track is None:
                        return None
                    return DirectMemberFaceSample(body,0,'none',body,value.person.detection_confidence,
                        'source_local_focal_body_track','frozen_source_focal_body_track_geometry')
                return DirectMemberFaceSample(value.face_bbox_xywh,1,'yunet',body,value.person.detection_confidence,
                    'source_local_focal_body_track',FOCAL_FACE_BASIS)
            candidate=tuple(replace(renders[i],editorial_transform=proposed[s]) for s,i in enumerate(affected))
            containment=review_joint_dissolve_containment(segments=candidate,transitions=(effect,),
                source_geometries={renders[i].source_index:geometries[s] for s,i in enumerate(affected)},
                geometry_provider=sample,fps=fps,frame_counts=tuple(counts[i] for i in affected),
                minimum_face_visible=config.minimum_dense_face_visible_fraction,
                minimum_body_visible=config.minimum_face_associated_subject_visible_fraction)
            if not containment['passed']:
                raise ValueError('rotated dense face/body containment failed')
            boxes=[]
            for s,i in enumerate(affected):
                seconds=renders[i].source_start+endpoints[s]*renders[i].source_playback_rate
                value=focal(path[i],seconds)
                if value is None or value.face_bbox_xywh is None:
                    raise ValueError('roll boundary face evidence missing')
                mapped=map_box_to_portrait_fill(value.face_bbox_xywh,geometries[s])
                state=proposed[s].evaluate(endpoints[s])
                if rotated_visible_fraction(mapped,state,output_size=size)+1e-9<config.minimum_face_visible_fraction:
                    raise ValueError('rotated boundary exceeds existing face visibility limit')
                boxes.append(project_box(mapped,state,output_size=size))
            position,scale=_rendered_attention_mismatch(*boxes)
            if position>config.maximum_face_residual_position_mismatch+1e-9 or scale>config.maximum_face_residual_scale_mismatch+1e-9:
                raise ValueError('rotated boundary exceeds existing face residual limits')
        except (ValueError,OSError) as error:
            row['reason']=str(error)
            continue
        for s,i in enumerate(affected):
            tracks[i]=proposed[s]
        residuals[boundary]=(position,scale)
        accepted.append(boundary)
        row.update(accepted=True,reason='observed_roll_dense_validated',
            camera_roll_degrees=list(rolls),request_source_indices=[renders[i].source_index for i in affected],
            segment_indices=list(affected),examined_frame_count=containment['examined_frame_count'],
            pose_rms_before=math.sqrt(sum(before)/3),pose_rms_after=math.sqrt(sum(after)/3),
            minimum_face_visible=min(r['face_visible_fraction'] for r in containment['frames'] if r.get('face_visible_fraction') is not None),
            minimum_body_visible=min(r['body_visible_fraction'] for r in containment['frames'] if r.get('body_visible_fraction') is not None))
    if not rows:
        return base
    return replace(base,segments=tuple(replace(item,track=tracks[i],
        reasons=item.reasons+(('observed_relative_screen_roll',) if tracks[i] is not item.track else ()))
        for i,item in enumerate(base.segments)),
        roll_event_count=sum(t is not None and any(abs(p.roll_degrees)>1e-9 for p in t.keyframes) for t in tracks),
        boundary_residual_attention_mismatches=tuple((i,*values) for i,values in sorted(residuals.items())),
        member_roll_audit=tuple(rows))
