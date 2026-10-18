from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from crosscut.ive import AspectRatio
from crosscut.render import (RenderPlan,RenderSegment,RenderTransition,TransitionKind,FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE,
                            PanCropTrack,PanCropKeyframe,NormalizedCrop)
from crosscut.render_reference_clock import load_render_reference_clock
from crosscut2.bounded_plan_render import render as original_render
from crosscut2.encoded_programme_reuse import render,shot_choices


def plans(tmp_path):
    sources=tuple(tmp_path/f'source-{i}.mp4' for i in range(2))
    crop=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(0.,0.,1.,1.)),
                      PanCropKeyframe(.9,NormalizedCrop(0.,0.,1.,1.))))
    base=RenderPlan(sources,tuple(RenderSegment(i%2,float(i//2),1.,float(i),pan_crop=crop) for i in range(4)),
        0,AspectRatio.LANDSCAPE,tmp_path/'base.mp4',width=160,height=90,fps=30.)
    effect=RenderTransition(TransitionKind.FACE_DISSOLVE,8,reason='explicit_reuse_fixture',
        face_dissolve_curve=FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE)
    draft=replace(base,output=tmp_path/'draft.mp4',transitions=(effect,RenderTransition(),effect))
    target=replace(draft,output=tmp_path/'result.mp4',transitions=(effect,RenderTransition(),RenderTransition()))
    return base,draft,target


def test_choices_change_only_effects_not_nominal_cameras_cuts_or_clocks(tmp_path):
    base,draft,target=plans(tmp_path)
    assert shot_choices(target,[(draft.output,draft),(base.output,base)])==[0,0,0,1]
    shifted=replace(target,segments=(replace(target.segments[0],source_start=.01),*target.segments[1:]))
    with pytest.raises(ValueError,match='exact same nominal'):
        shot_choices(shifted,[(draft.output,draft),(base.output,base)])
    missing=replace(target,transitions=(replace(target.transitions[0],frame_count=7),*target.transitions[1:]))
    with pytest.raises(ValueError,match='no existing encoded shot'):
        shot_choices(missing,[(draft.output,draft),(base.output,base)])


def test_real_packet_reuse_removes_only_requested_effect_without_reencoding(tmp_path):
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:pytest.skip('local FFmpeg required')
    base,draft,target=plans(tmp_path)
    for path,color in zip(base.sources,('red','blue'),strict=True):
        subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',f'color={color}:s=160x90:r=30:d=5',
            '-f','lavfi','-i','sine=frequency=440:duration=5','-c:v','libx264','-bf','0','-g','15',
            '-pix_fmt','yuv420p','-c:a','aac','-video_track_timescale','30000','-shortest','-n',str(path)],
            check=True,capture_output=True,timeout=15.)
    for plan in (base,draft):
        original_render(plan,ffmpeg=ffmpeg,maximum_seconds=45.,reserve_bytes=256*2**20,video_codec='libx264')
    result=render(target,[(draft.output,draft),(base.output,base)],ffmpeg=ffmpeg,maximum_seconds=45.)
    assert result['output_frames']==120 and result['video_encoder_calls']==0
    assert result['source_NN']==0 and result['source_rasters']==0
    assert result['full_AV_decode_exit_code']==0 and result['all_video_packet_payloads_preserved']
    assert result['audio_packet_payloads_and_timestamps_preserved']
    assert len(result['encoded_ranges'])==2 and not list(tmp_path.glob('.crosscut-encoded-reuse-*'))
    def pixels(path):
        data=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-an','-pix_fmt','rgb24',
            '-f','rawvideo','pipe:1'],check=True,capture_output=True,timeout=10.)
        return np.frombuffer(data.stdout,np.uint8).reshape(120,90,160,3)
    actual,first,second=pixels(target.output),pixels(draft.output),pixels(base.output)
    np.testing.assert_array_equal(actual[:90],first[:90]);np.testing.assert_array_equal(actual[90:],second[90:])
    import json
    payload=json.loads(json.dumps(target.to_json()))
    clock=load_render_reference_clock(target.output,payload,[29,38,60,90])
    assert clock.frames[90].source_index==1
    with pytest.raises(ValueError,match='composited'):
        load_render_reference_clock(target.output,payload,[34])
    with pytest.raises(FileExistsError,match='never overwrites'):
        render(target,[(draft.output,draft),(base.output,base)],ffmpeg=ffmpeg)
    # A content-bound sidecar alone cannot relabel a160px movie as320px.
    forged=replace(base,output=tmp_path/'forged.mp4',width=320,height=180)
    shutil.copyfile(base.output,forged.output)
    from crosscut.render_clock_artifact import render_clock_path
    sidecar=json.loads(render_clock_path(base.output).read_bytes());sidecar['plan']=forged.to_json()
    render_clock_path(forged.output).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='candidate raster'):
        render(replace(forged,output=tmp_path/'wrong-size-result.mp4'),[(forged.output,forged)],ffmpeg=ffmpeg)
