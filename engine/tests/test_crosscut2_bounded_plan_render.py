from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from crosscut.ive import AspectRatio
from crosscut.render import (RenderPlan,RenderSegment,RenderTransition,TransitionKind,
    PanCropTrack,PanCropKeyframe,NormalizedCrop,EditorialTransformTrack,EditorialTransformKeyframe)
from crosscut.rendered_endpoint_component import incoming_dissolve_weight
from crosscut2.bounded_plan_render import outgoing_handle,blend_graph,render


def test_outgoing_overlap_uses_the_real_boundary_clock_and_final_camera_only():
    base=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(0.,0.,.8,.8)),
        PanCropKeyframe(1.9,NormalizedCrop(.1,.1,.8,.8))))
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.,.5,.5),
        EditorialTransformKeyframe(1.9,1.2,.5,.5)),reason='measured_camera')
    segment=RenderSegment(2,10.,2.,7.,pan_crop=base,editorial_transform=camera,source_duration=2.02)
    handle=outgoing_handle(segment,frame_count=6,fps=30.)
    assert handle.source_start==segment.source_end
    assert handle.source_playback_rate==pytest.approx(segment.source_playback_rate)
    np.testing.assert_allclose([handle.crop.x,handle.crop.y,handle.crop.width,handle.crop.height],[.1,.1,.8,.8],atol=1e-15)
    assert handle.pan_crop is None
    assert handle.editorial_transform.evaluate(0.).zoom==1.2
    assert handle.editorial_transform.evaluate(.1).zoom==1.2
    assert handle.boundary_look_track is None and handle.outgoing_overlap_transform is None


def test_explicit_registered_handle_is_not_silently_replaced_by_a_held_camera():
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.,.5,.5),
        EditorialTransformKeyframe(.2,1.05,.5,.5)),reason='measured_future_camera')
    segment=RenderSegment(0,0.,2.,0.,outgoing_overlap_transform=camera)
    assert outgoing_handle(segment,frame_count=6,fps=30.).editorial_transform is camera


def test_blend_uses_elapsed_canonical_time_not_an_ambiguous_frame_counter():
    graph=blend_graph(30000/1001)
    assert 'settb=1001/30000,setpts=N' in graph and 'T*30000/1001' in graph
    assert '__H__' in graph and 'P-' not in graph


def _ffmpeg():
    value=shutil.which('ffmpeg')
    if value is None:pytest.skip('local FFmpeg unavailable')
    return value


def test_real_logistic_blend_pixels_follow_the_production_alpha_curve(tmp_path):
    ffmpeg=_ffmpeg();h=6;path=tmp_path/'blend.mkv'
    graph=blend_graph(30.).replace('__H__',str(h))
    subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','color=black:s=160x90:r=30:d=1',
        '-f','lavfi','-i','color=white:s=160x90:r=30:d=1','-filter_complex',graph,
        '-map','[v]','-frames:v',str(h),'-c:v','ffv1','-n',str(path)],check=True,capture_output=True,timeout=10.)
    decoded=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-pix_fmt','gray',
        '-f','rawvideo','pipe:1'],check=True,capture_output=True,timeout=10.)
    frames=np.frombuffer(decoded.stdout,np.uint8).reshape(h,90,160)
    actual=frames.mean((1,2))/255
    expected=np.array([incoming_dissolve_weight(i,h) for i in range(h)])
    np.testing.assert_allclose(actual,expected,atol=.012)
    assert actual[0]==0 and np.all(np.diff(actual)>0)


def test_real_sequential_render_keeps_overlap_duration_audio_and_exact_packet_grid(tmp_path):
    ffmpeg=_ffmpeg();sources=[]
    for i,color in enumerate(('red','blue')):
        path=tmp_path/f'source-{i}.mp4';sources.append(path)
        subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',f'color={color}:s=160x90:r=30000/1001:d=3',
            '-f','lavfi','-i','sine=frequency=440:duration=3','-c:v','libx264','-preset','ultrafast',
            '-g','15','-c:a','aac','-video_track_timescale','30000','-shortest','-n',str(path)],
            check=True,capture_output=True,timeout=15.)
    fps=30000/1001;n=30;duration=n/fps
    plan=RenderPlan(tuple(sources),(RenderSegment(0,.1,duration,0.),
        RenderSegment(1,.2,duration,duration)),0,AspectRatio.LANDSCAPE,tmp_path/'film.mp4',
        transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,6,reason='explicit_test_overlap'),),
        width=160,height=90,fps=fps)
    receipt=render(plan,ffmpeg=ffmpeg,maximum_seconds=45.,reserve_bytes=256*2**20,video_codec='libx264')
    assert receipt['output_frames']==60 and receipt['full_AV_decode_exit_code']==0
    assert receipt['source_NN']==0 and receipt['own_transient_blocks_removed']
    assert not receipt['professional_quality_approved']
    assert all(p.exists() for p in sources) and not list(tmp_path.glob('.crosscut-bounded-render-*'))
    assert [v['output_frames'] for v in receipt['blocks']]==[30,6,24]
    with pytest.raises(FileExistsError,match='never overwrites'):
        render(plan,ffmpeg=ffmpeg,video_codec='libx264')
    result=subprocess.run([ffmpeg,'-v','error','-i',str(plan.output),'-an','-pix_fmt','rgb24',
        '-f','rawvideo','pipe:1'],check=True,capture_output=True,timeout=10.)
    pixels=np.frombuffer(result.stdout,np.uint8).reshape(60,90,160,3).mean((1,2))
    assert pixels[29,0]>240 and pixels[30,0]>240 and pixels[30,2]<12
    assert pixels[36,2]>240 and pixels[36,0]<12


def test_unsupported_effect_is_rejected_before_source_access_or_decoding(tmp_path):
    # Avoid manufacturing bridge authorizations to test the early renderer
    # refusal: the supported-effect check is independent of plan authoring.
    from types import SimpleNamespace
    plan=SimpleNamespace(transitions=[SimpleNamespace(kind=TransitionKind.MEMBER_ZOOM_BRIDGE)])
    with pytest.raises(ValueError,match='only explicit hard cuts and face dissolves'):
        render(plan)
