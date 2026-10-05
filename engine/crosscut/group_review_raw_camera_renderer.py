"""Selected GROUP raw raster renderer, including actual production color look.

This renders only frozen changed ticks from exact native pixels. It does not
fit, choose actors, infer pose, grant adoption, or certify perceptual quality.
An independent caller/session must replay the returned provider before delivery.
"""
import hashlib
import json
import subprocess
import numpy as np

from .group_performer_family_owner import digest
from .group_scene_camera_queries import GroupSceneRoleQuery
from .group_review_interval_compile import output_offsets
from .group_retained_raster_stream import FrozenRetainedRasterProvider
from .render import _video_post_clock_filters


def render_selected_review_rasters(*,plan,queries,changed_intervals,session,native_row_for_query,
        expected_program_identity,maximum_frames=64,timeout_seconds=45.):
    if (digest(plan.to_json())!=expected_program_identity or not isinstance(changed_intervals,tuple)
            or not 1<=maximum_frames<=1200 or not 0<timeout_seconds<=180.):
        raise ValueError('raw renderer changed its independently selected program or bounded scope')
    wanted=tuple(t for a,b in changed_intervals for t in range(a,b))
    if (not wanted or len(wanted)>maximum_frames or wanted!=tuple(sorted(set(wanted)))
            or any(not 0<=t<plan.output_frame_count for t in wanted)
            or set(queries)!=set(wanted)):
        raise ValueError('raw renderer omitted or changed its exact output tick inventory')
    starts=output_offsets(plan);frames={};bindings=[]
    for i,segment in enumerate(plan.segments):
        ticks=tuple(t for t in wanted if starts[i]<=t<starts[i+1])
        if not ticks:continue
        # Changed intervals were materialized before selection. Rendering a
        # partial authored cell here would silently reset its camera clock.
        if ticks!=tuple(range(starts[i],starts[i+1])):
            raise ValueError('raw renderer cannot post-select slice a selected camera cell')
        native=[]
        for tick in ticks:
            query=queries[tick]
            if not isinstance(query,GroupSceneRoleQuery):raise ValueError('raw renderer needs complete registered cameras')
            observed_segment,phase,index=query.sample_at(tick)
            if observed_segment!=segment or abs(phase-(tick-starts[i])/plan.fps)>1e-9:
                raise ValueError('raw renderer query changed the exact selected camera phase')
            source=session.domain.request_index(query.eligible_source_index)
            if source!=segment.source_index:raise ValueError('raw renderer query borrowed another selected source')
            row=native_row_for_query(query,tick)
            raw=row['native_observation']
            if (row['output_tick']!=tick or raw['request_source_index']!=source
                    or raw['native_frame_index']!=index or raw['source_clock_identity']!=query.source_clock_identity
                    or tuple(raw['source_file_identity'])!=tuple(session._assert_source_identity(source))):
                raise ValueError('raw renderer observation borrowed another actual source/frame/file/clock')
            pixels=session.native_pixels(source,index)
            if pixels is None or hashlib.sha256(memoryview(pixels)).hexdigest()!=raw['native_pixel_sha256']:
                raise ValueError('raw renderer actual native raster differs from its measured parent')
            native.append(pixels)
            bindings.append(dict(tick=tick,source=source,native=index,raw_parent=digest(raw),
                registered_query_identity=query.key,camera_identity=query.camera.key,
                native_pixel_sha256=raw['native_pixel_sha256']))
        shape=native[0].shape
        if any(p.shape!=shape or p.dtype!=np.uint8 for p in native):
            raise ValueError('raw renderer native canvas changed inside a selected cell')
        filters=_video_post_clock_filters(crop=segment.crop,pan_crop=segment.pan_crop,
            editorial_transform=segment.editorial_transform,look_adjustment=segment.look_adjustment,
            boundary_look_track=segment.boundary_look_track,width=plan.width,height=plan.height,
            fps=plan.fps,frame_count=len(ticks),input_clock_is_canonical=True)
        command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','bgr24',
            '-video_size',f'{shape[1]}x{shape[0]}','-framerate',str(plan.fps),'-i','pipe:0',
            '-vf',','.join(filters),'-frames:v',str(len(ticks)),'-threads','2',
            '-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
        completed=subprocess.run(command,input=b''.join(memoryview(p).cast('B') for p in native),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout_seconds,check=False)
        if completed.returncode:raise RuntimeError('selected production camera failed: '+completed.stderr.decode()[-2000:])
        expected=len(ticks)*plan.width*plan.height*3
        if len(completed.stdout)!=expected:raise ValueError('selected raw camera omitted or added output rasters')
        output=np.frombuffer(completed.stdout,dtype=np.uint8).reshape(len(ticks),plan.height,plan.width,3)
        for tick,pixels in zip(ticks,output):
            pixels.setflags(write=False);frames[tick]=pixels
    if set(frames)!=set(wanted):raise ValueError('selected raw renderer incomplete after actual camera replay')
    native_root=digest(bindings)
    # Binding the renderer's source keeps a camera or color-path revision from
    # being mistaken for replay of a previously selected provider.
    import inspect
    renderer_identity=digest(dict(contract='production_post_clock_camera_and_look_renderer_v1',
        canonical_raster_sync='passthrough_v1',
        source=inspect.getsource(_video_post_clock_filters),selected_segments=[s for s in plan.to_json()['segments']],
        output_size=[plan.width,plan.height],fps=plan.fps))
    token=dict(contract='selected_raw_raster_provider_v1',frame_count=plan.output_frame_count,
        frame_size_wh=[plan.width,plan.height],changed_intervals=[list(s) for s in changed_intervals],
        selected_program_identity=expected_program_identity,native_evidence_root=native_root,
        renderer_identity=renderer_identity,post_selection_camera_refit=False)
    provider=FrozenRetainedRasterProvider(json.dumps(token,sort_keys=True,allow_nan=False),frames.__getitem__)
    return provider,dict(contract='actual_selected_raw_camera_raster_receipt_v1',provider_identity=provider.identity,
        selected_program_identity=expected_program_identity,native_bindings=bindings,
        rendered_output_pixel_sha256={str(t):hashlib.sha256(memoryview(p)).hexdigest() for t,p in frames.items()},
        output_frames=len(frames),new_NN_calls=0,post_selection_camera_refit=False,quality_approved=False)
