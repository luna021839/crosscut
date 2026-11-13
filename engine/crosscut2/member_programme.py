"""Complete numeric MEMBER programme stages, independent of audit scripts.

Selection, paired moving cameras and the original audio-warp clock remain
distinct operations. None opens a decoder/model or reads a previous movie.
"""
from dataclasses import replace
import math

import numpy as np

from crosscut.render import (NormalizedCrop, PanCropKeyframe, PanCropTrack,
    EditorialTransformKeyframe, EditorialTransformTrack, _base_crop_values,
    segment_frame_counts)
from crosscut.render_frame_schedule import sampler_frame_schedule
from .aspect_viewport import aspect_crop
from .member_pair_camera import fit
from .member_viewport_observations import project_box, project_points


def camera_path(shot, start, end):
    """Smooth fitted endpoints plus the original member-follow residual."""
    count=shot['count'];ticks=np.unique(np.r_[np.arange(0,count,5),count-1])
    u=ticks/max(1,count-1);smooth=u*u*(3.-2.*u)
    values=(1.-smooth[:,None])*start+smooth[:,None]*end
    baseline=(1.-u[:,None])*shot['old_camera'][0]+u[:,None]*shot['old_camera'][-1]
    values+=shot['old_camera'][ticks]-baseline
    values[:,0]=np.clip(values[:,0],1.,2.4)
    for k,(zoom,x,y) in enumerate(values):
        delta=np.array([x,y])-.5;radius=np.linalg.norm(delta)
        center=.5+delta*min(1.,.25/max(1e-12,radius))
        values[k,1:]=np.clip(center,.5/zoom,1.-.5/zoom)
    return ticks,values


def fit_cameras(plan, sources, observations, *, context_penalty=None, check=lambda:None):
    """Fit both views at every measured seam without reselecting any cut.

    Exact native flank observations supply face/arm geometry. Missing flanks
    keep the original cameras; they do not veto a source or erase a cut.
    `context_penalty(shot,start=...,end=...)` is optional observed hold context.
    """
    if (tuple(s.path for s in sources)!=plan.sources or len(observations)!=len(sources)
            or plan.width>=plan.height):
        raise ValueError('bound MEMBER sources, observations and portrait programme required')
    shots=[];absolute=0;fps=plan.fps;size=(plan.width,plan.height)
    for segment,count in zip(plan.segments,segment_frame_counts(plan),strict=True):
        check();source=sources[segment.source_index]
        native=sampler_frame_schedule(source_start=segment.source_start,source_playback_rate=segment.source_playback_rate,
            native_fps=source.fps,output_fps=fps,frame_count=count,source_frame_count=source.frame_count).source_frame_indices
        old=np.array([_base_crop_values(segment,f/fps) for f in range(count)])
        if segment.editorial_transform is not None:
            raise ValueError('fit the selected physical view before adding a paired editorial layer')
        centers=old[:,:2]+old[:,2:]/2
        full=aspect_crop((source.width,source.height),size)
        full_size=np.array([full.width,full.height]);chosen=None
        for fraction in np.linspace(1.,float(np.max(old[:,3]/full.height)),21):
            extent=full_size*fraction;center=np.clip(centers,extent/2,1.-extent/2)
            bases=np.c_[center-extent/2,np.tile(extent,(count,1))]
            zoom=bases[:,2]/old[:,2];pan=(centers-bases[:,:2])/bases[:,2:]
            if np.all((zoom>=1.-1e-9)&(zoom<=2.4)) and np.max(np.linalg.norm(pan-.5,axis=1))<=.25+1e-10:
                chosen=(bases,np.c_[zoom,pan],fraction);break
        if chosen is None:raise ValueError('selected member camera cannot retain its physical coverage')
        bases,camera,fraction=chosen
        def observation(index,base):
            row=observations[segment.source_index].get(int(index));pose=np.zeros((8,3))
            if row is None or row['face_bbox_xywh'] is None:return None,pose
            crop=NormalizedCrop(*map(float,base));face=np.array(project_box(row['face_bbox_xywh'],crop))
            if row.get('pose_screen_xyc'):
                pose=np.array(row['pose_screen_xyc'],float);pose[:,:2]=project_points(pose[:,:2],crop)
            return face,pose
        capacity=min(bases[0,2]*source.width/plan.width,bases[0,3]*source.height/plan.height)
        shots.append(dict(segment=segment,source=segment.source_index,count=count,native=native,old=old,base=bases,
            old_camera=camera,new_start=camera[0].copy(),new_end=camera[-1].copy(),
            first=observation(native[0],bases[0]),last=observation(native[-1],bases[-1]),
            sampling=capacity,cap=min(2.4,2.4*fraction),start=absolute))
        absolute+=count
    fits=[]
    for i,(left,right) in enumerate(zip(shots,shots[1:])):
        check();fa,pa=left['last'];fb,pb=right['first']
        if fa is None or fb is None:
            fits.append(dict(cut=i+1,measured_face_pair=False,camera_fit_claimed=False));continue
        penalty=None
        if context_penalty is not None:
            def penalty(cameras):
                return (context_penalty(left,start=left['new_start'],end=cameras[0])
                    +context_penalty(right,start=cameras[1],end=right['new_end']))
        cameras,receipt=fit(fa,fb,pa,pb,existing=np.array([left['old_camera'][-1],right['old_camera'][0]]),
            output_size=size,caps=(left['cap'],right['cap']),sampling=(left['sampling'],right['sampling']),
            context_penalty=penalty)
        left['new_end']=cameras[0];right['new_start']=cameras[1]
        fits.append(dict(cut=i+1,seconds=right['start']/fps,measured_face_pair=True,**receipt))
    segments=[];changes=0;max_zoom=1.;max_speed=0.
    for shot in shots:
        check();ticks,cameras=camera_path(shot,shot['new_start'],shot['new_end'])
        base_keys=tuple(PanCropKeyframe(int(t)/fps,NormalizedCrop(*map(float,shot['base'][t]))) for t in ticks)
        edit_keys=tuple(EditorialTransformKeyframe(int(t)/fps,*map(float,v)) for t,v in zip(ticks,cameras,strict=True))
        track=EditorialTransformTrack(edit_keys,reason='actual_member_two_sided_viewport_face_and_arm_fit')
        if not track.covers_aspect_ratio(plan.aspect_ratio):raise ValueError('paired camera reveals a source edge')
        segments.append(replace(shot['segment'],crop=None,pan_crop=PanCropTrack(base_keys),editorial_transform=track))
        max_zoom=max(max_zoom,float(cameras[:,0].max()))
        if len(ticks)>1:max_speed=max(max_speed,float((np.abs(np.diff(np.log(cameras[:,0])))/(np.diff(ticks)/fps)).max()))
        changes+=int(np.any(np.abs(cameras-shot['old_camera'][ticks])>1e-6,axis=1).sum())
    return replace(plan,segments=tuple(segments)),dict(source_and_cut_sequence_unchanged=True,
        all_native_source_frame_ordinals_unchanged=True,face_pair_fits=fits,edited_camera_keyframes=changes,
        maximum_editorial_zoom=max_zoom,maximum_added_log_zoom_speed_per_second=max_speed,
        source_NN=0,source_rasters=0,professional_quality_approved=False)


def replay_audio_warp(plan, sources, warps, *, master_start, check=lambda:None):
    """Replay every original quantized source ordinal, not just hold endpoints.

    This operation solves the complete affine-rate interval for each hold.
    Non-affine trajectories are reported explicitly, never relabelled exact.
    """
    if tuple(s.path for s in sources)!=plan.sources or len(warps)!=len(sources) or not math.isfinite(master_start):
        raise ValueError('bound source clocks and original audio warps required')
    segments=[];repairs=[];checked=0
    for i,(segment,count) in enumerate(zip(plan.segments,segment_frame_counts(plan),strict=True)):
        check();source=sources[segment.source_index];start=round(segment.master_start*plan.fps)
        wanted=np.array([math.floor(warps[segment.source_index].master_to_source(master_start+(start+k)/plan.fps)*source.fps+.5)
            for k in range(count)])
        def schedule(value):
            return np.array(sampler_frame_schedule(source_start=value.source_start,source_playback_rate=value.source_playback_rate,
                native_fps=source.fps,output_fps=plan.fps,frame_count=count,source_frame_count=source.frame_count).source_frame_indices)
        actual=schedule(segment)
        if not np.array_equal(actual,wanted):
            if count<2 or actual[0]!=wanted[0]:raise ValueError('selected member start does not match its original warp')
            k=np.arange(1,count)
            low=np.max(((wanted[1:]-.5)/source.fps-segment.source_start)*plan.fps/k)
            high=np.min(((wanted[1:]+.5)/source.fps-segment.source_start)*plan.fps/k)
            if not low<high:raise ValueError('original quantized member hold needs a piecewise, not affine, video clock')
            rate=float((low+high)/2)
            fixed=replace(segment,source_duration=rate*segment.duration)
            if not np.array_equal(schedule(fixed),wanted):raise ValueError('exact affine interval did not replay the physical source frames')
            repairs.append(dict(segment=i,source=segment.source_index,old_rate=segment.source_playback_rate,
                exact_quantized_rate=rate,previous_interior_differences=int(np.sum(actual!=wanted))))
            segment=fixed
        segments.append(segment);checked+=count
    return replace(plan,segments=tuple(segments)),dict(all_nominal_native_ordinals_checked=checked,
        all_nominal_native_ordinals_exact_to_audio_timewarp=True,affine_quantization_repairs=repairs,
        source_NN=0,source_rasters=0,source_or_cut_sequence_changed=False)


def build(sources, observations, *, warps, master_start, duration_frames, fps, beat_times,
        output, output_size=(1080,1920), reference_source=0, duration_model=None, body_context=None,
        member=None, current_faces=None, original_geometry=None, effects_enabled=False, effect_frames=3,
        check=lambda:None, progress=None):
    """One ordered complete programme build; callbacks never hide a new solve."""
    from .member_stage_mix import select
    from .member_film_effects import windows,evaluate,apply
    plan,selection=select(sources,observations,warps=warps,master_start=master_start,
        duration_frames=duration_frames,fps=fps,beat_times=beat_times,output=output,output_size=output_size,
        reference_source=reference_source,duration_model=duration_model,
        content_loss=None if body_context is None else body_context.content,check=check,progress=progress)
    plan,cameras=fit_cameras(plan,sources,observations,
        context_penalty=None if body_context is None else body_context.penalty,check=check)
    plan,clock=replay_audio_warp(plan,sources,warps,master_start=master_start,check=check)
    if effects_enabled:
        if current_faces is None or member is None:raise ValueError('enabled member effects need current observed target faces')
        request=windows(plan,sources,frame_count=effect_frames,current_faces=current_faces)
        geometry=observations if original_geometry is None else original_geometry
        reports=[evaluate(w,current_faces=current_faces,original_geometry=geometry,sources=sources,
            member=member,output_size=output_size) for w in request['boundaries']]
        plan,effects=apply(plan,reports,frame_count=effect_frames)
        effects['missing_native_effect_observations']=request['requested_native_rasters']
    else:
        plan,effects=apply(plan,[],frame_count=effect_frames,enabled=False)
    return plan,dict(selection=selection,cameras=cameras,clock=clock,effects=effects,
        source_NN=0,source_rasters=0,whole_programme_selection_performed=True,
        previous_movie_or_cut_list_used_as_input=False,professional_quality_approved=False)
