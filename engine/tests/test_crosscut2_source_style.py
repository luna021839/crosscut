from pathlib import Path
import numpy as np
import pytest
from crosscut2 import source_style as s


def test_native_pixel_colour_not_source_name_or_camera_identity():
    red=np.full((180,320,3),[0,0,200],np.uint8)
    blue=np.full_like(red,[200,0,0]);heads=np.zeros((8,4));heads[0]=[.5,.3,.2,.95]
    a,av,p=s.describe(red,heads);b,bv,_=s.describe(blue,heads)
    assert av[0] and bv[0] and not av[1:].any()
    assert np.allclose(a[0].sum(1),1.) and np.allclose(b[0].sum(1),1.)
    assert not np.array_equal(a[0],b[0])
    assert np.array_equal(s.describe(red.copy(),heads)[0],a)
    assert p[0]==0 and s.describe(np.zeros_like(red),heads)[2][0]==1


def test_completed_colour_source_reuses_without_decode_and_is_bound_to_native_inputs(tmp_path,monkeypatch):
    path=tmp_path/'source';path.write_bytes(b'not a movie; fake decoder below')
    heads=np.zeros((2,2,8,4),np.float32);heads[:,:,0]=[.5,.3,.2,.95]
    bank=dict(heads=heads,native_indices=np.array([[10,11],[10,12]]))
    calls=[]
    def decode(path,natives,check):
        calls.append(natives)
        for n in natives:yield n,np.full((180,320,3),[0,0,n],np.uint8)
    monkeypatch.setattr(s,'frames',decode)
    a,ar=s.observe(sources=[path],banks=[bank],cache=tmp_path/'cache',check=lambda:None)
    assert calls==[[10,11,12]] and ar[0]['decoded_native_inputs']==3
    monkeypatch.setattr(s,'frames',lambda *a,**k:pytest.fail('warm cache decoded source'))
    b,br=s.observe(sources=[path],banks=[bank],cache=tmp_path/'cache',check=lambda:None)
    assert br[0]['reused'] and np.array_equal(a[0]['colour'],b[0]['colour'])
    assert np.array_equal(a[0]['colour'][0,0],a[0]['colour'][1,0])
