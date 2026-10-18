import importlib
import json
from dataclasses import fields
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut.render import RenderPlan,RenderSegment
from crosscut.ive import AspectRatio
from crosscut2 import final_render_backend as backend


def plan(tmp_path):
    return RenderPlan((tmp_path/'a.mp4',tmp_path/'b.mp4'),(
        RenderSegment(0,0.,1.,0.),RenderSegment(1,0.,1.,1.)),0,
        AspectRatio.LANDSCAPE,tmp_path/'film.mp4',width=160,height=90,fps=30.)


def test_app_final_keeps_whole_film_progress_until_audio_and_decode_finish(tmp_path,monkeypatch):
    film=plan(tmp_path);events=[]
    def observed(p,**kwargs):
        cb=kwargs['progress_callback']
        cb(dict(pid=11,phase='end',frame=30))
        cb(dict(phase='bounded_segment_complete',completed=1))
        cb(dict(pid=12,phase='end',frame=30))
        cb(dict(phase='bounded_segment_complete',completed=2))
        assert [v['frame'] for v in events]==[30,59]
        assert all(v['phase']=='continue' for v in events)
        return dict(output_frames=60)
    monkeypatch.setattr(backend.bounded_plan_render,'render',observed)
    codec=backend.render(film,ffmpeg='ffmpeg',video_codec='h264_videotoolbox',
        maximum_runtime_seconds=120.,progress_callback=events.append)
    assert codec=='h264_videotoolbox' and events[-1]['frame']==60 and events[-1]['phase']=='end'
    assert [v['pid'] for v in events]==[11,12,12]


def test_bounded_failure_is_not_retried_with_another_codec_or_filter_graph(tmp_path,monkeypatch):
    render=importlib.import_module('crosscut.render');calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise RuntimeError('one intentional encoder failure')
    monkeypatch.setattr(backend.bounded_plan_render,'render',fail)
    with pytest.raises(RuntimeError,match='intentional encoder failure'):
        render.render(plan(tmp_path),execution_backend='sequential_when_supported')
    assert len(calls)==1 and calls[0]['video_codec']=='h264_videotoolbox'


@pytest.mark.parametrize('options',[
    dict(use_verified_frame_schedule=False),dict(video_bitrate='1M'),dict(decoder_threads=4)])
def test_explicit_sequential_path_cannot_silently_drop_custom_encoding_or_clock_options(tmp_path,options):
    render=importlib.import_module('crosscut.render')
    with pytest.raises(ValueError,match='standard encoding contract'):
        render.render(plan(tmp_path),execution_backend='sequential_when_supported',**options)


def test_unsupported_effect_keeps_the_existing_renderer_not_a_degraded_hard_cut(tmp_path,monkeypatch):
    render=importlib.import_module('crosscut.render')
    original=plan(tmp_path)
    movie=SimpleNamespace(**{field.name:getattr(original,field.name) for field in fields(original)})
    movie.transitions=[SimpleNamespace(kind='unrecognized_effect')]
    monkeypatch.setattr(render,'build_filter_graph',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('existing effect renderer entered')))
    with pytest.raises(RuntimeError,match='existing effect renderer entered'):
        render.render(movie,execution_backend='sequential_when_supported',use_verified_frame_schedule=False)


def test_real_app_render_entry_uses_bounded_native_clock_film_and_standard_qa_sidecar(tmp_path):
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:pytest.skip('local FFmpeg unavailable')
    render=importlib.import_module('crosscut.render');film=plan(tmp_path)
    for source,color in zip(film.sources,('red','blue'),strict=True):
        subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',f'color={color}:s=160x90:r=30:d=2',
            '-f','lavfi','-i','sine=frequency=440:duration=2','-c:v','libx264','-preset','ultrafast',
            '-g','15','-c:a','aac','-shortest','-n',str(source)],check=True,capture_output=True,timeout=15.)
    events=[]
    assert render.render(film,ffmpeg=ffmpeg,video_codec='libx264',execution_backend='sequential_when_supported',
        maximum_runtime_seconds=45.,progress_callback=events.append)=='libx264'
    assert [v['frame'] for v in events]==[30,59,60]
    sidecar=json.loads(Path(str(film.output)+'.frame-clock.json').read_bytes())
    assert sidecar['backend']=='sequential_native_clock_blocks_v1'
    assert sidecar['resource_policy']['decoder_concurrency']==1 and not sidecar['quality_approval']
    decoded=subprocess.run([ffmpeg,'-v','error','-i',str(film.output),'-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],
        check=True,capture_output=True,timeout=10.)
    pixels=np.frombuffer(decoded.stdout,np.uint8).reshape(60,90,160,3).mean((1,2))
    assert pixels[29,0]>240 and pixels[30,2]>240
