import cv2
import numpy as np
import pytest
from crosscut2.native_readability import face_readability,measure_native_readability,native_frames


def image():
    rng=np.random.default_rng(41)
    gray=rng.integers(40,210,(120,160),np.uint8)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


def test_native_motion_blur_changes_edge_proxy_without_inventing_identity():
    pixels=image();head=[.5,.5,.6,.95]
    a,av=face_readability(pixels,head)
    b,bv=face_readability(cv2.GaussianBlur(pixels,(15,15),4),head)
    assert av and bv and a[1]>b[1]+.2
    assert a[3]==b[3] and np.isfinite(b).all()


def test_missing_or_tiny_unreadable_crop_is_not_a_perfect_measured_face():
    for head in ([.5,.5,0,.95],[.5,.5,.025,.95],[.5,.5,.3,.5]):
        values,valid=face_readability(image(),head)
        assert not valid and not values.any()


def test_saturation_is_a_scalar_observation_not_a_source_ban():
    values,valid=face_readability(np.full((120,160,3),255,np.uint8),[.5,.5,.6,.95])
    assert valid and values[2]==1 and np.isfinite(values).all()


def fixture(tmp_path):
    path=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
    assert writer.isOpened()
    for _ in range(4):writer.write(image())
    writer.release()
    heads=np.zeros((3,2,8,4),np.float32);heads[:,:,0]=[.5,.5,.6,.95]
    return path,dict(heads=heads,native_indices=np.array([[0,0],[1,2],[3,3]]),zoom_cap=1.35)


def test_exact_input_cache_reuses_without_any_more_decoding_or_neural_work(tmp_path,monkeypatch):
    path,bank=fixture(tmp_path)
    result,receipts=measure_native_readability(sources=[path],banks=[bank],cache=tmp_path/'cache',check=lambda:None)
    assert receipts[0]['decoded_native_inputs']==4 and receipts[0]['new_neural_calls']==0
    def forbidden(*a,**kw):raise AssertionError('warm scalar cache must not decode or infer')
    monkeypatch.setattr(cv2,'VideoCapture',forbidden)
    warm,again=measure_native_readability(sources=[path],banks=[bank],cache=tmp_path/'cache',check=lambda:None)
    assert again[0]['reused'] and again[0]['decoded_native_inputs']==0
    assert np.array_equal(result[0]['native_readability'],warm[0]['native_readability'])
    assert not result[0]['native_readability_valid'][:,:,1:].any()


def test_head_rectangle_change_cannot_reuse_an_old_quality_binding(tmp_path):
    path,bank=fixture(tmp_path);cache=tmp_path/'cache'
    _,first=measure_native_readability(sources=[path],banks=[bank],cache=cache,check=lambda:None)
    changed=dict(bank,heads=bank['heads'].copy());changed['heads'][:,:,0,0]=.7
    _,second=measure_native_readability(sources=[path],banks=[changed],cache=cache,check=lambda:None)
    assert not first[0]['reused'] and not second[0]['reused']
    assert len(list(cache.glob('*.npz')))==2


def test_native_frame_outside_source_is_not_silently_clamped(tmp_path):
    path,bank=fixture(tmp_path);bank['native_indices'][1,1]=4
    with pytest.raises(ValueError,match='exceeds source'):
        measure_native_readability(sources=[path],banks=[bank],cache=tmp_path/'cache',check=lambda:None)


def test_streaming_filter_preserves_exact_selected_ordinals_and_native_size(tmp_path):
    path=tmp_path/'ordinals.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
    assert writer.isOpened()
    for level in (40,80,120,160):writer.write(np.full((120,160,3),level,np.uint8))
    writer.release()
    decoded=list(native_frames(path,[0,2,3],width=160,height=120,check=lambda:None))
    assert [tick for tick,_ in decoded]==[0,2,3]
    assert all(pixels.shape==(120,160,3) for _,pixels in decoded)
    assert [float(pixels.mean()) for _,pixels in decoded]==pytest.approx([40,120,160],abs=1)


def test_incomplete_declared_frame_set_cannot_be_silently_bound_to_another_frame(tmp_path):
    path,_=fixture(tmp_path)
    with pytest.raises(OSError,match='incomplete'):
        list(native_frames(path,[0,4],width=160,height=120,check=lambda:None))
