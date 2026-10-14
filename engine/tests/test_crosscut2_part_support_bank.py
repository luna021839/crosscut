from pathlib import Path
import numpy as np
import pytest
from crosscut2.fixed_holds import State
from crosscut2 import part_support_bank as bank
from crosscut2.temporal_parts import PartPaths


def setup(tmp_path):
    source=tmp_path/'source.mp4';source.write_bytes(b'fake-source-clock')
    points=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
        [.3,.55,.9],[.7,.55,.9],[.43,.7,.9],[.57,.7,.9]])
    bodies=np.tile(points,(2,4,1,1));bodies[:,3,:,1]+=.02
    row=dict(body=bodies,heads=np.tile([.5,.3,.25,.95],(2,2,1)),formation=np.zeros((2,2),bool),
        pose_native_indices=np.tile([1,7,8,14],(2,1)))
    kw=dict(sources=[source],states=[State(0,0,'native')],data=[row],census=[dict(frames=[],fps=30.,frame_count=20)],
        cache=tmp_path/'cache',check=lambda:None)
    return kw,bodies


def fake_result():
    mask=np.array([True,False,False,False,True,False,False,False])
    return PartPaths(mask,mask.copy(),np.zeros(8,bool),np.where(mask,0.,np.nan),True)


def test_atomic_physical_interval_cache_reused_without_decode_and_raw_unchanged(tmp_path,monkeypatch):
    kw,raw=setup(tmp_path);old=raw.copy();calls=[]
    def rasters(path,needed,check,**kw):
        kw['receipt']['strategy']='verified_original_PTS_bounded_native_seek_v1'
        calls.append(1)
        for f in sorted(needed):yield f,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(bank,'_rasters',rasters);monkeypatch.setattr(bank.temporal_parts,'observe',lambda *a,**k:fake_result())
    rows,reports=bank.corroborate(**kw)
    assert len(calls)==1 and reports[0]['new_physical_intervals']==3
    assert reports[0]['new_neural_calls']==0 and np.array_equal(raw,old)
    assert np.array_equal(rows[0]['body'][...,:2],raw[...,:2])
    assert (rows[0]['body'][:,:,4,2]>.3).all() and (rows[0]['body'][:,:,5,2]==0).all()
    monkeypatch.setattr(bank,'_rasters',lambda *a,**k:pytest.fail('warm cache must not decode'))
    again,second=bank.corroborate(**kw,maximum_new_intervals=0)
    assert second[0]['new_physical_intervals']==0 and second[0]['reused_physical_intervals']==3
    assert np.array_equal(rows[0]['body'],again[0]['body'])


def test_native_cut_disallows_only_that_interval_without_false_zero_motion(tmp_path,monkeypatch):
    kw,_=setup(tmp_path);kw['census'][0]['frames']=[8]
    def rasters(path,needed,check,**kw):
        kw['receipt']['strategy']='verified_original_PTS_bounded_native_seek_v1'
        for f in sorted(needed):yield f,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(bank,'_rasters',rasters);monkeypatch.setattr(bank.temporal_parts,'observe',lambda *a,**k:fake_result())
    rows,reports=bank.corroborate(**kw)
    assert not rows[0]['part_pair_support'][:,1].any()
    assert rows[0]['part_pair_support'][:,0,4].all() and rows[0]['part_pair_support'][:,2,4].all()
    assert reports[0]['new_physical_intervals']==2


def test_partial_completion_retained_after_interruption_and_explicit_retry_reuses_it(tmp_path,monkeypatch):
    kw,_=setup(tmp_path);calls=[]
    def rasters(path,needed,check,**kw):
        kw['receipt']['strategy']='verified_original_PTS_bounded_native_seek_v1'
        for f in sorted(needed):yield f,np.zeros((24,40,3),np.uint8)
    def interrupted(*a,**k):
        calls.append(1)
        if len(calls)==2:raise TimeoutError('fixture interruption')
        return fake_result()
    monkeypatch.setattr(bank,'_rasters',rasters);monkeypatch.setattr(bank.temporal_parts,'observe',interrupted)
    with pytest.raises(TimeoutError):bank.corroborate(**kw)
    assert len(list(kw['cache'].glob('*.npz')))==1
    monkeypatch.setattr(bank.temporal_parts,'observe',lambda *a,**k:fake_result())
    _,reports=bank.corroborate(**kw)
    assert reports[0]['new_physical_intervals']==2 and reports[0]['reused_physical_intervals']==1
