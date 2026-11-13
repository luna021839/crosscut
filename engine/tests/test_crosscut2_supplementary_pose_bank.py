from types import SimpleNamespace
import json
import numpy as np
import pytest
from crosscut2 import supplementary_pose_bank as bank


def args(tmp_path):
    source=tmp_path/'source.bin';source.write_bytes(b'bound synthetic source')
    return dict(path=source,requests=[(2,np.array([.5,.3,.2,.95]))],
        cache=tmp_path/'cache',source_clock=dict(fps=30.,frame_count=10),check=lambda:None)


class Sensor:
    def __init__(self):self.calls=0
    def describe(self,*a,**k):
        self.calls+=1;raw=np.zeros((133,3),np.float32);raw[:,2]=4.2
        return SimpleNamespace(wholebody_xyc=raw)


def test_duplicate_physical_inputs_and_new_grid_reuse_without_model_or_decode(tmp_path,monkeypatch):
    p=args(tmp_path);p['requests']*=2
    monkeypatch.setattr(bank,'_frames',lambda path,natives,check:iter((n,np.zeros((540,960,3),np.uint8)) for n in natives))
    sensor=Sensor();raw,cold=bank.observe_requests(sensor=sensor,**p)
    assert cold['new_neural_calls']==1 and len(raw)==2 and sensor.calls==1
    def forbidden(*a,**k):raise AssertionError('warm supplementary bank reran perception')
    monkeypatch.setattr(bank,'_frames',forbidden)
    p['requests']=p['requests'][:1]
    raw,warm=bank.observe_requests(sensor=SimpleNamespace(describe=forbidden),**p)
    assert warm['new_neural_calls']==0 and warm['reused_neural_inputs']==1
    assert (raw[0][:,2]==np.float32(4.2)).all() and len(list(p['cache'].glob('*.npz')))==1
    assert warm['native_raster_execution']['strategy']=='warm_raw_cache_only'


def test_interrupted_source_retains_completed_cells_without_fake_success(tmp_path,monkeypatch):
    p=args(tmp_path);p['requests'].append((3,np.array([.5,.3,.2,.95])))
    def stop_after_first(path,natives,check):
        yield natives[0],np.zeros((540,960,3),np.uint8)
        raise TimeoutError('actual later interruption')
    monkeypatch.setattr(bank,'_frames',stop_after_first);sensor=Sensor()
    with pytest.raises(TimeoutError,match='later interruption'):bank.observe_requests(sensor=sensor,**p)
    assert sensor.calls==1
    p['requests']=p['requests'][:1]
    def forbidden(*a,**k):raise AssertionError('completed input lost after interruption')
    monkeypatch.setattr(bank,'_frames',forbidden)
    _,warm=bank.observe_requests(sensor=SimpleNamespace(describe=forbidden),**p)
    assert warm['new_neural_calls']==0


def test_bad_native_range_or_excess_work_does_not_start_network(tmp_path,monkeypatch):
    p=args(tmp_path);sensor=Sensor();p['requests']=[(12,p['requests'][0][1])]
    with pytest.raises(ValueError,match='actual native'):bank.observe_requests(sensor=sensor,**p)
    assert sensor.calls==0


def test_preexisting_completed_raw_inputs_do_not_probe_or_rewrite_when_sparse_reader_is_added(tmp_path,monkeypatch):
    from crosscut2 import sparse_native_frames as sparse
    import hashlib
    p=args(tmp_path)
    monkeypatch.setattr(bank,'_frames',lambda path,natives,check:iter((n,np.zeros((540,960,3),np.uint8)) for n in natives))
    _,cold=bank.observe_requests(sensor=Sensor(),**p)
    saved=next(p['cache'].glob('*.npz'));before=hashlib.sha256(saved.read_bytes()).hexdigest()
    def forbidden(*a,**kw):raise AssertionError('warm cache must not probe/decode/infer')
    monkeypatch.setattr(sparse,'probe',forbidden);monkeypatch.setattr(sparse,'frames',forbidden)
    monkeypatch.setattr(bank,'_frames',forbidden)
    _,warm=bank.observe_requests(sensor=type('NoNN',(),{'describe':forbidden})(),**p)
    assert warm['new_neural_calls']==0 and warm['written_checkpoints']==0
    assert hashlib.sha256(saved.read_bytes()).hexdigest()==before
    assert warm['cache_file']==cold['cache_file']


def test_failed_grid_window_uses_one_model_on_all_four_flanks_not_a_blend(tmp_path,monkeypatch):
    nodes=np.array([0,20,40]);heads=np.zeros((3,2,8,4),np.float32);heads[:,:,0]=[.5,.3,.2,.95]
    p=np.zeros((3,4,3,8,3),np.float32)
    p[:,:,0]=[[.4,.44,.9],[.6,.44,.9],[.3,.55,.9],[.7,.55,.9],
                [.25,.3,.9],[.75,.3,.9],[.42,.8,.9],[.58,.8,.9]]
    states=np.zeros((3,4,3),np.uint8);states[:,:,0]=1;states[1,1,0]=2
    before=p.copy();state_before=states.copy();seen=[]
    raw=np.zeros((133,3),np.float32);raw[0]=[.5,.3,4.]
    raw[[5,6,7,8,9,10],:]=[[.42,.44,4.],[.58,.44,4.],[.35,.6,4.],
        [.65,.6,4.],[.3,.39,4.],[.7,.39,4.]]
    def observed(path,requests,**kwargs):
        seen.extend(n for n,h in requests)
        return [raw.copy() for _ in requests],dict(new_neural_calls=4,reused_neural_inputs=0)
    monkeypatch.setattr(bank,'observe_requests',observed)
    revised,receipts=bank.augment_missing_grid(sources=[tmp_path/'source'],banks=[dict(heads=heads)],
        poses=[p],body_states=[states],warps=[SimpleNamespace(master_to_source=lambda t:t)],nodes=nodes,
        master_start=0.,fps=10.,camera_census=[dict(fps=10.,frame_count=50)],model_path=tmp_path/'model',
        cache=tmp_path/'cache',check=lambda:None,node_scope=[1])
    assert seen==[13,19,20,26] and receipts[0]['same_model_for_all_four_temporal_samples']
    assert all(np.allclose(revised[0][1,phase,0,:6,:2],raw[[5,6,7,8,9,10],:2]) for phase in range(4))
    assert not revised[0][1,:,0,6:].any()  # no hip/visibility invention
    assert np.array_equal(p,before) and np.array_equal(states,state_before)
    assert np.array_equal(revised[0][[0,2]],p[[0,2]])


def temporal_fixture(tmp_path,monkeypatch,*,height=.2,rejected=False):
    from crosscut2.temporal_face_chain import FaceChain,BoundWindowFaces
    source=tmp_path/'temporal-source';source.write_bytes(b'one explicit synthetic source')
    nodes=np.array([0,20,40]);heads=np.zeros((3,2,8,4),np.float32);heads[:,:,0]=[.5,.3,height,.95]
    p=np.zeros((3,4,3,8,3),np.float32);states=np.zeros((3,4,3),np.uint8)
    current=np.array([.35 if height>.12 else .47,.3,height,.95]);seen=[]
    def observed(path,requests,**kwargs):
        seen.extend(requests);result=[]
        for i,_ in enumerate(requests):
            raw=np.zeros((133,3),np.float32);raw[:,2]=.9
            raw[0,:2]=([.5,.3] if rejected or i else current[:2]);raw[5:11,:2]=[.5,.5]
            result.append(raw)
        return result,dict(new_neural_calls=0,reused_neural_inputs=len(requests))
    monkeypatch.setattr(bank,'observe_requests',observed)
    chain=FaceChain(not rejected,'synthetic_rejected' if rejected else 'synthetic_supported',current,np.zeros((0,4)),
        tuple([0]*7),tuple(dict(from_native=n,to_native=n-1) for n in range(19,13,-1)))
    bound=BoundWindowFaces(bank.file_identity(source),(13,19,20,26),heads[1,:,0],{0:chain})
    kwargs=dict(sources=[source],banks=[dict(heads=heads)],poses=[p],body_states=[states],
        warps=[SimpleNamespace(master_to_source=lambda t:t)],nodes=nodes,master_start=0.,fps=10.,
        camera_census=[dict(fps=10.,frame_count=50)],model_path=tmp_path/'model',cache=tmp_path/'cache',
        check=lambda:None,node_scope=[1],maximum_calls=0)
    return kwargs,bound,seen


@pytest.mark.parametrize('height',[.2,.08])
def test_opt_in_producer_uses_bound_contemporary_face_not_stale_box_and_preserves_other_phases(tmp_path,monkeypatch,height):
    p,bound,seen=temporal_fixture(tmp_path,monkeypatch,height=height)
    legacy,old_receipts=bank.augment_missing_grid(**p)
    revised,receipts=bank.augment_missing_grid(**p,temporal_faces={(0,1,0):bound})
    assert not legacy[0][1,0,0].any() and (revised[0][1,0,0,:6,2]>.3).all()
    assert np.array_equal(p['poses'][0],np.zeros_like(p['poses'][0]))
    assert not revised[0][[0,2]].any()
    if height>.12:assert np.array_equal(revised[0][1,1:,0],legacy[0][1,1:,0])
    assert receipts[0]['opt_in_contemporary_face_reanchors'][0]['supported_model_joints']==6
    json.dumps(receipts,allow_nan=False)  # same export surface as stage_mix's plan receipt
    assert all('opt_in_contemporary_face_reanchors' not in r for r in old_receipts)


def test_rejected_temporal_chain_does_not_keep_intruder_in_the_old_face_box(tmp_path,monkeypatch):
    p,bound,seen=temporal_fixture(tmp_path,monkeypatch,rejected=True)
    legacy,_=bank.augment_missing_grid(**p)
    revised,receipts=bank.augment_missing_grid(**p,temporal_faces={(0,1,0):bound})
    assert (legacy[0][1,0,0,:6,2]>.3).all() and not revised[0][1,0,0].any()
    assert 'chain_unmeasured' in receipts[0]['opt_in_contemporary_face_reanchors'][0]['reason']


@pytest.mark.parametrize('wrong',['source','native','endpoint_head'])
def test_temporal_source_native_or_face_misbinding_is_rejected_before_model_requests(tmp_path,monkeypatch,wrong):
    from dataclasses import replace
    p,bound,seen=temporal_fixture(tmp_path,monkeypatch,rejected=True)
    if wrong=='source':bound=replace(bound,source_file_identity=(1,2,3,4))
    elif wrong=='native':bound=replace(bound,native_indices=(12,19,20,26))
    else:bound=replace(bound,endpoint_heads=bound.endpoint_heads+[.01,0,0,0])
    with pytest.raises(ValueError,match='physical source/window/endpoint binding'):
        bank.augment_missing_grid(**p,temporal_faces={(0,1,0):bound})
    assert seen==[]
