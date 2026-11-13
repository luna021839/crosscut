import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut import render_frame_schedule
from crosscut.render import AspectRatio,RenderPlan,RenderSegment
from crosscut2 import source_cut_refinement as refine
from crosscut2.grid_perception import file_identity


def frames():
    rng=np.random.default_rng(412)
    a=rng.integers(10,110,(180,320,3),np.uint8)
    b=rng.integers(20,200,(180,320,3),np.uint8)
    # Large-scale spatial features survive the declared32x18 quadrature.
    a[20:100,30:120]=230;b[70:150,190:270]=5
    return [a.copy() for _ in range(3)]+[b.copy() for _ in range(3)]


def test_true_persistent_cut_is_distinct_from_a_brightness_flash_and_blank():
    values=frames()
    assert refine.persistent_structure_cut(values)
    a=values[0];bright=np.clip(a.astype(float)*.6+70,0,255).astype(np.uint8)
    assert not refine.persistent_structure_cut([a,a,a,bright,bright,bright])
    assert not refine.persistent_structure_cut([np.zeros_like(a)]*6)


def fixture(tmp_path,monkeypatch):
    source=tmp_path/'source.mp4';source.write_bytes(b'raw source fixture')
    movie=tmp_path/'movie.mp4';movie.write_bytes(b'output fixture')
    plan=RenderPlan((source,),(RenderSegment(0,0.,6.,0.),),0,AspectRatio.LANDSCAPE,movie,fps=30.)
    record=dict(render_plan=plan.to_json(),sparse_perception=[dict(source_file_identity=list(file_identity(source)))],
        bound_source_cut_events=[dict(fps=30.,frame_count=300)])
    critic=dict(source_file_identity=list(file_identity(movie)),boundary_coverage=dict(unexplained_detections=[15]))
    monkeypatch.setattr(refine.shutil,'which',lambda name:sys.executable)
    monkeypatch.setattr(render_frame_schedule,'probe_render_frame_clock',lambda p:SimpleNamespace(frame_rate=30.,frame_count=300))
    monkeypatch.setattr(refine.sparse_native_frames,'probe',lambda *a,**kw:SimpleNamespace(rate=30.,count=300))
    pixels=frames()
    def read(path,needed,**kw):
        kw['receipt']['strategy']='verified_original_PTS_bounded_native_seek_v1'
        yield from zip(needed,pixels)
    monkeypatch.setattr(refine.sparse_native_frames,'frames',read)
    return dict(movie=movie,record=record,critic=critic,cache=tmp_path/'cuts',check=lambda:None),source


def test_confirmed_cut_becomes_source_evidence_without_changing_the_input_plan(tmp_path,monkeypatch):
    kw,source=fixture(tmp_path,monkeypatch);before=copy.deepcopy(kw['record'])
    result=refine.discover_from_output(**kw)
    assert result['confirmed_source_events']==[(0,15)]
    assert result['decoded_native_frames']==6 and result['new_neural_calls']==0
    assert kw['record']==before and not result['edited_existing_plan']
    assert refine.observed_native_cuts([source],kw['cache'])==[[15]]


def test_completed_warm_cut_evidence_does_not_probe_decode_or_rewrite(tmp_path,monkeypatch):
    kw,source=fixture(tmp_path,monkeypatch);refine.discover_from_output(**kw)
    files={p.name:p.read_bytes() for p in kw['cache'].glob('*.json')}
    def forbidden(*a,**kw):raise AssertionError('warm source-cut reuse must not probe/decode')
    monkeypatch.setattr(refine.sparse_native_frames,'probe',forbidden)
    monkeypatch.setattr(refine.sparse_native_frames,'frames',forbidden)
    result=refine.discover_from_output(**kw)
    assert result['decoded_native_frames']==0 and result['proposals'][0]['reused']
    assert files=={p.name:p.read_bytes() for p in kw['cache'].glob('*.json')}


def test_incomplete_or_failed_native_window_never_saves_source_cut(tmp_path,monkeypatch):
    kw,_=fixture(tmp_path,monkeypatch)
    def broken(path,needed,**kw):
        yield needed[0],frames()[0]
        raise OSError('corrupt source window')
    monkeypatch.setattr(refine.sparse_native_frames,'frames',broken)
    with pytest.raises(OSError,match='corrupt source'):refine.discover_from_output(**kw)
    assert not list(kw['cache'].glob('*.json'))


def test_output_binding_and_declared_finite_budget_cannot_be_bypassed(tmp_path,monkeypatch):
    kw,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='allowance'):refine.discover_from_output(**kw,maximum_native_frames=5)
    kw['critic']['source_file_identity'][2]+=1
    with pytest.raises(ValueError,match='binding'):refine.discover_from_output(**kw)


def test_source_version_or_reader_change_cannot_reuse_the_old_cut(tmp_path,monkeypatch):
    kw,source=fixture(tmp_path,monkeypatch);refine.discover_from_output(**kw)
    source.write_bytes(b'different original source bytes')
    assert refine.observed_native_cuts([source],kw['cache'])==[[]]


def test_no_authored_boundary_is_promoted_to_an_original_source_cut(tmp_path,monkeypatch):
    kw,source=fixture(tmp_path,monkeypatch)
    plan=RenderPlan((source,),(RenderSegment(0,0.,.5,0.),RenderSegment(0,.5,5.5,.5)),
        0,AspectRatio.LANDSCAPE,kw['movie'],fps=30.)
    kw['record']['render_plan']=plan.to_json()
    result=refine.discover_from_output(**kw)
    assert not result['confirmed_source_events'] and result['decoded_native_frames']==0
