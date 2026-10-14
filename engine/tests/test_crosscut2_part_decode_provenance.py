import numpy as np
import pytest
from crosscut2 import part_support_bank as parts,sparse_native_frames as sparse
from crosscut2.observation_cache_contract import function_text
from test_crosscut2_part_support_bank import setup,fake_result


@pytest.mark.parametrize('changed',['_rasters','render_frame_clock_from_probe'])
def test_changed_production_reader_contract_cannot_inherit_legacy_part_key(monkeypatch,changed):
    original=function_text
    def changed_source(path,name):
        text=original(path,name)
        return text+'\n# changed exact native raster execution' if name==changed else text
    monkeypatch.setattr(parts,'function_text',changed_source)
    assert parts.cache_implementation()!='447432eab91d837e40edf01d482a3ef7302c210e24efa59a189266c55aba4abb'


def many_intervals(tmp_path):
    kw,_=setup(tmp_path);row=kw['data'][0];N=12
    row['body']=np.repeat(row['body'][:1],N,axis=0)
    row['heads']=np.repeat(row['heads'][:1],N,axis=0)
    row['formation']=np.zeros((N,2),bool)
    row['pose_native_indices']=np.arange(N)[:,None]*20+np.array([1,7,8,14])
    kw['census'][0]['frame_count']=300
    return kw


def test_failed_sequential_fallback_cannot_commit_even32_completed_intervals(tmp_path,monkeypatch):
    kw=many_intervals(tmp_path)
    def unsupported(*a,**kw):raise sparse.UnsupportedSeek('unproven sparse clock')
    def corrupt(path,needed,*,check):
        for tick in sorted(needed):yield tick,np.zeros((24,40,3),np.uint8)
        raise OSError('FFmpeg concealed frames then reported corrupt decode')
    monkeypatch.setattr(sparse,'probe',unsupported)
    monkeypatch.setattr(parts,'_sequential_rasters',corrupt)
    monkeypatch.setattr(parts.temporal_parts,'observe',lambda *a,**kw:fake_result())
    with pytest.raises(OSError,match='corrupt decode'):parts.corroborate(**kw,allow_sequential_fallback=True)
    assert not list(kw['cache'].glob('*.npz'))


def test_sparse_module_bytes_change_cannot_inherit_qualified_part_key(monkeypatch):
    from pathlib import Path
    original=Path.read_bytes
    def changed(path):
        raw=original(path)
        return raw+b'\n# changed exact PTS selection' if path==Path(sparse.__file__) else raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert parts.cache_implementation()!='447432eab91d837e40edf01d482a3ef7302c210e24efa59a189266c55aba4abb'


def test_clean_sequential_fallback_commits_only_after_valid_EOF(tmp_path,monkeypatch):
    kw=many_intervals(tmp_path)
    def unsupported(*a,**kw):raise sparse.UnsupportedSeek('unproven sparse clock')
    def clean(path,needed,*,check):
        for tick in sorted(needed):
            assert not list(kw['cache'].glob('*.npz'))
            yield tick,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(sparse,'probe',unsupported);monkeypatch.setattr(parts,'_sequential_rasters',clean)
    monkeypatch.setattr(parts.temporal_parts,'observe',lambda *a,**kw:fake_result())
    rows,receipt=parts.corroborate(**kw,allow_sequential_fallback=True)
    with np.load(next(kw['cache'].glob('*.npz')),allow_pickle=False) as saved:
        assert len(saved['keys'])==36 and saved['decode_verified'].all()
    assert rows[0]['part_pair_decode_verified'].all()
    assert receipt[0]['native_raster_execution']['strategy']=='explicit_legacy_sequential_fallback'


def test_legacy_cells_reused_without_decode_but_never_retroactively_certified(tmp_path,monkeypatch):
    import hashlib
    from crosscut2.part_hypotheses import retain_model_hypotheses
    kw,_=setup(tmp_path)
    def verified(path,needed,*,check,receipt,**kw):
        receipt['strategy']='verified_original_PTS_bounded_native_seek_v1'
        for tick in sorted(needed):yield tick,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(parts,'_rasters',verified);monkeypatch.setattr(parts.temporal_parts,'observe',lambda *a,**kw:fake_result())
    parts.corroborate(**kw);target=next(kw['cache'].glob('*.npz'))
    with np.load(target,allow_pickle=False) as saved:data={k:saved[k].copy() for k in saved.files if k!='decode_verified'}
    with target.open('wb') as stream:np.savez_compressed(stream,**data)
    before=hashlib.sha256(target.read_bytes()).hexdigest()
    def forbidden(*a,**kw):raise AssertionError('legacy warm reuse must not decode')
    monkeypatch.setattr(parts,'_rasters',forbidden)
    rows,receipt=parts.corroborate(**kw,maximum_new_intervals=0)
    assert receipt[0]['legacy_unverified_intervals_reused']==3 and not rows[0]['part_pair_decode_verified'].any()
    result=retain_model_hypotheses(rows=rows,states=kw['states'],sources=kw['sources'],cache=kw['cache'])
    assert result[0]['body_reliability'].min()==result[0]['body_reliability'].max()==.5
    assert np.array_equal(result[0]['body'],rows[0]['raw_body_hypotheses'])
    assert hashlib.sha256(target.read_bytes()).hexdigest()==before


def test_old_runtime_snapshot_cannot_conflict_with_current_part_consumer(tmp_path,monkeypatch):
    import json
    from crosscut2.part_hypotheses import retain_model_hypotheses
    kw,_=setup(tmp_path)
    def verified(path,needed,*,check,receipt,**kw):
        receipt['strategy']='verified_original_PTS_bounded_native_seek_v1'
        for tick in sorted(needed):yield tick,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(parts,'_rasters',verified);monkeypatch.setattr(parts.temporal_parts,'observe',lambda *a,**kw:fake_result())
    rows,_=parts.corroborate(**kw);target=next(kw['cache'].glob('*.npz'))
    with np.load(target,allow_pickle=False) as saved:data={k:saved[k].copy() for k in saved.files}
    binding=json.loads(str(data['binding']));binding['decoder'][1]+=1
    data['binding']=json.dumps(binding,sort_keys=True)
    with (kw['cache']/('0'*64+'.npz')).open('wb') as stream:np.savez_compressed(stream,**data)
    result=retain_model_hypotheses(rows=rows,states=kw['states'],sources=kw['sources'],cache=kw['cache'])
    assert result[0]['body_reliability'][:,:,4].min()==1.


def test_production_part_queue_cannot_silently_enter_whole_source_fallback(tmp_path,monkeypatch):
    kw,_=setup(tmp_path)
    def unsupported(*a,**kw):raise sparse.UnsupportedSeek('unproven clock')
    def forbidden(*a,**kw):raise AssertionError('strict production queue must not decode full source')
    monkeypatch.setattr(sparse,'probe',unsupported);monkeypatch.setattr(parts,'_sequential_rasters',forbidden)
    with pytest.raises(sparse.UnsupportedSeek,match='refuses full-source fallback'):parts.corroborate(**kw)
    assert not list(kw['cache'].glob('*.npz'))
    import ast,inspect
    from crosscut2 import stage_mix
    calls=[node for node in ast.walk(ast.parse(inspect.getsource(stage_mix.main)))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='corroborate']
    assert len(calls)==1
    assert any(k.arg=='allow_sequential_fallback' and isinstance(k.value,ast.Constant) and k.value.value is False
        for k in calls[0].keywords)
