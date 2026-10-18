"""Reuse wardrobe handles only on the same freshly selected physical seam.

The cache contains rendered, measured effect handles, not a chosen cut list.
Every source ordinal and viewport is rebuilt from the current programme before
an effect can match. A changed camera/clock/source is a cache miss, not an old
effect pasted onto a new pose. Missing handles leave ordinary editing intact.
"""
import hashlib
import json
from pathlib import Path

from crosscut.render_frame_schedule import sampler_frame_schedule
from .member_film_effects import view
from .member_geometry_inputs import identity


def binding(plan,sources,boundary,*,frame_count=25,lead_frames=12):
    if (type(boundary) is not int or not 0<=boundary<len(plan.segments)-1
            or type(frame_count) is not int or not 3<=frame_count<=60
            or type(lead_frames) is not int or not 0<=lead_frames<frame_count):
        raise ValueError('bounded wardrobe seam required')
    left,right=plan.segments[boundary:boundary+2]
    if round(left.duration*plan.fps)<=lead_frames or round(right.duration*plan.fps)<=frame_count-lead_frames:
        raise ValueError('wardrobe window crosses another editorial cut')
    sides=[]
    for side,segment in enumerate((left,right)):
        source=sources[segment.source_index]
        center=segment.source_end if side==0 else segment.source_start
        native=sampler_frame_schedule(source_start=center-lead_frames/plan.fps*segment.source_playback_rate,
            source_playback_rate=segment.source_playback_rate,native_fps=source.fps,
            output_fps=plan.fps,frame_count=frame_count,source_frame_count=source.frame_count).source_frame_indices
        crops=[]
        for j in range(frame_count):
            seconds=(j-lead_frames)/plan.fps
            crop=view(segment,segment.duration+seconds if side==0 else max(0.,seconds))
            crops.append([crop.x,crop.y,crop.width,crop.height])
        sides.append(dict(source=str(source.path),identity=identity(source.path),
            native_frames=list(native),crops=crops))
    return dict(contract='measured_member_wardrobe_handle_v1',fps=plan.fps,
        output_size=[plan.width,plan.height],frame_count=frame_count,lead_frames=lead_frames,sides=sides)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def match(plan,sources,candidates):
    selected=[];misses=0;used=set()
    for candidate in candidates:
        wanted=candidate['binding']
        if candidate['binding_sha256']!=signature(wanted):raise ValueError('wardrobe handle binding changed')
        frames,lead=wanted['frame_count'],wanted['lead_frames']
        found=None
        for i in range(len(plan.segments)-1):
            if i in used:continue
            try:current=binding(plan,sources,i,frame_count=frames,lead_frames=lead)
            except ValueError:continue
            if signature(current)==candidate['binding_sha256']:
                found=i;break
        if found is None:misses+=1;continue
        path=Path(candidate['movie'])
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=candidate['movie_sha256']:raise ValueError('measured wardrobe handle movie changed')
        first=round(plan.segments[found+1].master_start*plan.fps)-lead
        selected.append(dict(movie=str(path),first_frame=first,frame_count=frames,
            edge_blend_frames=candidate.get('edge_blend_frames',2),binding_sha256=candidate['binding_sha256']))
        used.add(found)
    return sorted(selected,key=lambda e:e['first_frame']),dict(matched_windows=len(selected),
        unmatched_windows=misses,source_NN=0,source_selection_repeated=False)


def validate(rendering,plan,wardrobe_input=None):
    """Validate the explicit post-effect chain, keeping the nominal plan honest."""
    path=rendering.get('wardrobe_receipt')
    if path is None:
        if rendering.get('base_video_path') is not None:raise ValueError('wardrobe output lacks its effect receipt')
        return Path(plan['output'])
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=rendering['wardrobe_receipt_sha256']:
        raise ValueError('wardrobe post-effect receipt changed')
    value=json.loads(raw)
    if wardrobe_input is None:raise ValueError('wardrobe output lacks its measured handle input')
    inputs=Path(wardrobe_input['path']).read_bytes()
    if hashlib.sha256(inputs).hexdigest()!=wardrobe_input['sha256']:
        raise ValueError('wardrobe handle inputs changed')
    handles=json.loads(inputs)['handles']
    if any(not any(e['movie']==h['movie'] and e['sha256']==h['movie_sha256']
            and e['binding_sha256']==h['binding_sha256'] for h in handles) for e in value.get('effects',[])):
        raise ValueError('finished wardrobe pictures do not bind the measured handles')
    if (value.get('contract')!='member_spatial_garment_first_post_effect_v1'
            or value.get('base_movie')!=plan['output'] or rendering.get('base_video_path')!=plan['output']
            or value.get('base_movie_sha256')!=rendering.get('base_movie_sha256')
            or value.get('output')!=rendering.get('video_path')
            or value.get('output_sha256')!=rendering.get('movie_sha256')
            or value.get('output_frames')!=plan['output_frame_count']
            or value.get('audio_packet_payloads_and_timestamps_preserved') is not True
            or value.get('full_AV_decode_exit_code')!=0 or not value.get('effects')):
        raise ValueError('wardrobe post-effect chain does not describe this real film')
    with Path(plan['output']).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=value['base_movie_sha256']:raise ValueError('wardrobe base film changed')
    return Path(plan['output'])
