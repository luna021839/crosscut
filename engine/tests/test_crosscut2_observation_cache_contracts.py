import inspect
import numpy as np
import pytest
from crosscut2 import source_cuts,part_support_bank as parts,part_hypotheses,native_ports
from crosscut2.observation_cache_contract import function_text
from crosscut2 import sparse_native_frames as sparse


def test_unchanged_cut_algorithm_reuses_historical_source_key_despite_output_QA_edits():
    assert source_cuts.cache_implementation()=='ae6a48d0d9abad25ab659487877fff6988a4bd615e26e2e3c1f5e2c24cfb9a9e'


def test_unchanged_part_algorithm_keeps_raw_source_snapshots_and_consumer_agrees():
    assert parts.cache_implementation()=='447432eab91d837e40edf01d482a3ef7302c210e24efa59a189266c55aba4abb'
    assert 'part_support_bank.cache_implementation()' in inspect.getsource(part_hypotheses.retain_model_hypotheses)


def test_actual_algorithm_source_change_does_not_receive_historical_key(monkeypatch):
    def changed(path,name):
        source=function_text(path,name)
        return source+'\n# changed real detector contract' if name=='signature' else source
    monkeypatch.setattr(source_cuts,'function_text',changed)
    assert source_cuts.cache_implementation()!='ae6a48d0d9abad25ab659487877fff6988a4bd615e26e2e3c1f5e2c24cfb9a9e'


def test_native_flank_cache_does_not_depend_on_unrelated_cut_or_output_code(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('warm cache must not decode')
    monkeypatch.setattr(native_ports,'native_frames',forbidden)
    assert native_ports.cache_implementation()=='e36db4fa915de0a9f616f1dac6c1f15ff729f62bff443fed8a2f91cf0b25adc8'


def test_strict_part_queue_never_silently_returns_to_whole_source_scan(monkeypatch):
    def unsupported(*a,**kw):raise sparse.UnsupportedSeek('unproved packet clock')
    def forbidden(*a,**kw):raise AssertionError('strict sparse queue must not scan full source')
    monkeypatch.setattr(sparse,'probe',unsupported);monkeypatch.setattr(parts,'_sequential_rasters',forbidden)
    with pytest.raises(sparse.UnsupportedSeek,match='refuses full-source fallback'):
        list(parts._rasters('unused.mp4',{2,3},check=lambda:None,allow_sequential_fallback=False))


def test_sparse_part_consumer_uses_verified_native_reader(monkeypatch):
    marker=object();got=[]
    monkeypatch.setattr(sparse,'probe',lambda *a,**kw:marker)
    monkeypatch.setattr(sparse,'windows',lambda values,c:got.append((values,c)))
    def frames(path,values,**kwargs):
        assert kwargs['clock'] is marker and kwargs['maximum_seconds']==12.
        kwargs['receipt']['strategy']='verified_sparse'
        yield 5,np.zeros((2,2,3),np.uint8)
    monkeypatch.setattr(sparse,'frames',frames)
    receipt={};result=list(parts._rasters('unused.mp4',{5},check=lambda:None,maximum_seconds=12.,receipt=receipt))
    assert got==[([5],marker)] and result[0][0]==5 and receipt['strategy']=='verified_sparse'


def test_real_part_sparse_cold_rasters_match_original_global_indices(tmp_path):
    import subprocess
    source=tmp_path/'part-native-clock.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30',
        '-frames:v','120','-an','-c:v','libx264','-bf','3','-g','30','-pix_fmt','yuv420p',str(source)],check=True)
    wanted=[2,3,4,5,80,81,82,119];receipt={}
    actual=dict(parts._rasters(source,wanted,check=lambda:None,allow_sequential_fallback=False,receipt=receipt))
    expected=dict(parts._sequential_rasters(source,wanted,check=lambda:None))
    assert actual.keys()==expected.keys()
    for tick in wanted:np.testing.assert_array_equal(actual[tick],expected[tick])
    assert receipt['strategy']=='verified_original_PTS_bounded_native_seek_v1'
    assert not receipt['whole_source_sequential_strategy_requested']
