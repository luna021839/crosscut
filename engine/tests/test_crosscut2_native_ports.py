from types import SimpleNamespace

import numpy as np
import pytest

from crosscut2 import native_ports
from crosscut2.fixed_holds import select_fixed_holds
from tests.test_crosscut2_fixed_holds import bank_fixture


def cut_bank(frames=(15,), before=.20, after=.02):
    heads=np.zeros((len(frames),2,8,4),np.float32)
    heads[:,:,0]=[.5,.3,before,.95]
    heads[:,1,0,2]=after
    vectors=np.zeros((len(frames),2,3,128),np.float32)
    vectors[:,:,0,0]=1.
    return dict(frames=np.array(frames,np.int64),heads=heads,vectors=vectors)


def test_retained_native_close_to_extreme_wide_is_not_a_free_cut():
    bank=cut_bank()
    native=np.tile([1.,.5,.5],(2,1))
    jump=native_ports.native_geometry_cost(bank['heads'][0],bank['vectors'][0],native)
    same=cut_bank(before=.20,after=.20)
    clean=native_ports.native_geometry_cost(same['heads'][0],same['vectors'][0],native)
    assert jump>8 and clean==0.


def test_realized_fixed_camera_is_consumed_by_native_cut_geometry():
    bank=cut_bank(before=.10,after=.20)
    native=np.tile([1.,.5,.5],(2,1))
    fitted=np.array([[2.,.5,.4],[1.,.5,.5]])
    before=native_ports.native_geometry_cost(bank['heads'][0],bank['vectors'][0],native)
    after=native_ports.native_geometry_cost(bank['heads'][0],bank['vectors'][0],fitted)
    assert after<1e-5 and before>2.


def test_native_geometry_is_charged_once_at_interval_or_interior_node():
    nodes=np.array([0,10,20,30])
    bank=cut_bank((10,15))
    interval,through=native_ports.native_geometry_events(nodes,bank,np.array([0,29]),
        np.tile([1.,.5,.5],(2,1)))
    assert interval[1]>8 and through[1]>8
    assert np.count_nonzero(interval)==np.count_nonzero(through)==1
    assert sum(interval)==pytest.approx(sum(through))


def test_unknown_native_faces_do_not_invent_a_mismatch_or_forbid_source():
    bank=cut_bank();bank['heads'][:]=0
    cost=native_ports.native_geometry_cost(bank['heads'][0],bank['vectors'][0],np.tile([1.,.5,.5],(2,1)))
    assert cost==0.


def test_unobserved_native_cut_is_finite_uncertainty_not_a_free_match():
    from crosscut2.native_port_costs import native_port_cost_events
    from crosscut2.fixed_holds import MISSING_PORT_COST
    nodes=np.array([0,10,20,30]);bank=cut_bank();bank['heads'][:]=0
    interval,through=native_port_cost_events(nodes,bank,np.array([0,29]),np.tile([1.,.5,.5],(2,1)))
    assert interval.sum()==MISSING_PORT_COST and through.sum()==0.
    banks=[bank_fixture(),bank_fixture()]
    _,_,_,_,receipt=select_fixed_holds(nodes,banks,None,fps=10,
        source_events=[np.array([15]),np.array([15])],broadcast_slots=None,check=lambda:None,
        native_ports=[bank,cut_bank(before=.2,after=.2)])
    assert all(row['source']==1 for row in receipt['selected_hold_cost_decomposition'])


def test_same_centroid_native_formation_still_consumes_layout_change():
    bank=cut_bank();bank['heads'][:]=0
    for side,xs in enumerate(([.4,.5,.6],[.15,.5,.85])):
        bank['heads'][0,side,:3]=[[x,.4,.05,.95] for x in xs]
    assert native_ports.native_geometry_cost(bank['heads'][0],bank['vectors'][0],np.tile([1.,.5,.5],(2,1)))>1.


def test_native_port_sensor_uses_only_exact_flanks_and_its_own_cache(monkeypatch,tmp_path):
    observed=[]
    def sensor(path,frames,record,**kwargs):
        kwargs['frames']=frames
        observed.append(kwargs)
        bank=dict(heads=np.zeros((2,2,8,4)),vectors=np.zeros((2,2,3,128)))
        return bank,dict(reused=True,source_file_identity=[1,2,3,4],native_inputs=4,decoder='fixture')
    monkeypatch.setattr(native_ports,'_measure_flanks',sensor)
    banks,receipts=native_ports.measure_native_ports(sources=['source.mp4'],
        census=[dict(frames=[10,20,35],frame_count=40,fps=10.)],
        warps=[SimpleNamespace(source_to_master=lambda t:t)],master_start=1.,fps=10.,final_frame=30,
        cache=tmp_path/'native-only',models=tmp_path,check=lambda:None)
    assert observed[0]['frames'].tolist()==[20,35]
    assert observed[0]['cache']==tmp_path/'native-only'
    assert banks[0]['frames'].tolist()==[10,25]
    assert banks[0]['heads'].shape==(2,2,8,4)
    assert receipts[0]['native_input_frames']==4


def test_dp_consumes_native_cut_geometry_with_exact_cost_decomposition():
    nodes=np.array([0,10,20,30]);banks=[bank_fixture(),bank_fixture()]
    kwargs=dict(fps=10,source_events=[np.array([15]),np.array([15])],
        broadcast_slots=None,check=lambda:None)
    _,states,_,_,plain=select_fixed_holds(nodes,banks,None,**kwargs)
    assert plain['selected_hold_cost_decomposition'][0]['source']==0
    _,states,_,score,measured=select_fixed_holds(nodes,banks,None,
        native_ports=[cut_bank(),cut_bank(before=.20,after=.20)],**kwargs)
    assert all(row['source']==1 for row in measured['selected_hold_cost_decomposition'])
    assert measured['selected_hold_cost_decomposition'][-1]['cumulative_cost']==pytest.approx(score)
    assert measured['measured_native_cut_geometry_consumed'] is True


def _fake_flank_environment(monkeypatch,tmp_path):
    source=tmp_path/'source.mp4';source.write_bytes(b'fixture')
    models=tmp_path/'models';models.mkdir()
    for name in ('face_detection_yunet_2023mar.onnx','face_recognition_sface_2021dec.onnx'):
        (models/name).write_bytes(b'fake model for mocked observer')
    class Cap:
        def isOpened(self):return True
        def get(self,key):return {native_ports.cv2.CAP_PROP_FRAME_COUNT:30,
            native_ports.cv2.CAP_PROP_FRAME_WIDTH:12,native_ports.cv2.CAP_PROP_FRAME_HEIGHT:8}[key]
        def release(self):pass
    monkeypatch.setattr(native_ports.cv2,'VideoCapture',lambda path:Cap())
    monkeypatch.setattr(native_ports.cv2,'FaceDetectorYN',SimpleNamespace(
        create=lambda *args:SimpleNamespace(detect=lambda pixels:(None,None))))
    monkeypatch.setattr(native_ports.cv2,'FaceRecognizerSF',SimpleNamespace(create=lambda *args:None))
    return source,models


def test_sequential_native_flank_cache_is_reused_without_decoder(monkeypatch,tmp_path):
    source,models=_fake_flank_environment(monkeypatch,tmp_path)
    calls=[]
    def frames(path,ticks,**kwargs):
        calls.append(ticks)
        return iter((tick,np.zeros((8,12,3),np.uint8)) for tick in ticks)
    monkeypatch.setattr(native_ports,'native_frames',frames)
    kwargs=dict(cache=tmp_path/'cache',models=models,check=lambda:None)
    bank,cold=native_ports._measure_flanks(source,np.array([10]),dict(fps=10.,frame_count=30),**kwargs)
    other,warm=native_ports._measure_flanks(source,np.array([10]),dict(fps=10.,frame_count=30),**kwargs)
    assert calls==[[9,10]]
    assert not cold['reused'] and warm['reused']
    assert np.array_equal(bank['heads'],other['heads'])


def test_corrupt_native_flank_decoder_never_writes_partial_cache(monkeypatch,tmp_path):
    source,models=_fake_flank_environment(monkeypatch,tmp_path)
    def frames(*args,**kwargs):
        yield 9,np.zeros((8,12,3),np.uint8)
        raise OSError('corrupt native decoder')
    monkeypatch.setattr(native_ports,'native_frames',frames)
    with pytest.raises(OSError,match='corrupt native decoder'):
        native_ports._measure_flanks(source,np.array([10]),dict(fps=10.,frame_count=30),
            cache=tmp_path/'cache',models=models,check=lambda:None)
    assert not list((tmp_path/'cache').glob('*.npz'))
