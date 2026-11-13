import hashlib
import json

import numpy as np
import pytest

from crosscut2 import person_rectangle_bank as people
from crosscut2.person_attention import competition
from crosscut2.display_source_catalogue import SourceCatalogue
from test_crosscut2_person_rectangle_bank import world, run, forbid_cold
from test_crosscut2_display_source_catalogue import prepared


def test_large_headless_competing_body_is_not_ignored_by_the_face_proxy():
    boxes=np.array([[.43,.19,.57,.68,.8],[.65,.1,1.,1.,.9]])
    value=competition(boxes,[.5,.25,.1,.99],[1.,.5,.5])
    assert value['measured'] and value['deficit']>.75
    assert value['compatible_rectangles']==1
    tight=competition(boxes,[.5,.25,.1,.99],[2.,.5,.5])
    assert tight['measured'] and tight['deficit']<value['deficit']


def test_ambiguous_large_rectangle_gives_an_optimistic_bound_not_false_occlusion():
    boxes=np.array([[.43,.19,.57,.68,.8],[.1,.1,.9,1.,.9]])
    value=competition(boxes,[.5,.25,.1,.99],[1.,.5,.5])
    assert value['compatible_rectangles']==2 and value['measured']
    assert value['deficit']==0.


def test_observed_original_body_cannot_borrow_a_larger_foreground_person_area():
    boxes=np.array([[.43,.19,.57,.68,.8],[.1,.1,.9,1.,.9]])
    head=[.5,.25,.1,.99];camera=[1.,.5,.5]
    assert competition(boxes,head,camera)['deficit']==0.
    measured=competition(boxes,head,camera,target_index=0)
    assert measured['measured'] and measured['deficit']>.85
    missing=competition(boxes,head,camera,target_index=-1)
    assert not missing['measured'] and missing['deficit']==.5
    for invalid in (True,0.,2,-2):
        with pytest.raises(ValueError,match='person-instance'):
            competition(boxes,head,camera,target_index=invalid)


def test_original_instance_indices_broadcast_over_real_candidate_viewports():
    boxes=np.array([[[.43,.19,.57,.68,.8],[.1,.1,.9,1.,.9]]]*3)
    heads=np.array([[.5,.25,.1,.99]]*3)
    cameras=np.array([[[1.,.5,.5]]*3,[[2.,.5,.5]]*3])
    indices=np.array([0,1,-1]);batch=competition(boxes,heads,cameras,target_index=indices)
    for k,n in np.ndindex(2,3):
        scalar=competition(boxes[n],heads[n],cameras[k,n],target_index=int(indices[n]))
        assert batch['deficit'][k,n]==scalar['deficit']
    empty=competition(np.empty((0,5)),heads[0],cameras[0,0],target_index=-1)
    assert not empty['measured'] and empty['deficit']==.5


@pytest.mark.parametrize('boxes,head',[
    (np.zeros((0,5)),[.5,.25,.1,.99]),
    (np.zeros((3,5)),[.5,.25,.1,.99]),
    (np.array([[.65,.1,1.,1.,.9]]),[.5,.25,.1,.99]),
    (np.array([[.1,.1,.9,1.,.9]]),[0.,0.,0.,0.]),
])
def test_empty_unanchored_and_missing_face_are_unknown_not_clean(boxes,head):
    value=competition(boxes,head,[1.,.5,.5])
    assert not value['measured'] and value['deficit']==.5


def test_batched_actual_crops_match_scalar_and_score_does_not_rank_people():
    boxes=np.array([[[.43,.19,.57,.68,.8],[.65,.1,1.,1.,.9]]]*2)
    heads=np.array([[.5,.25,.1,.99]]*2)
    cameras=np.array([[[1.,.5,.5]]*2,[[2.,.5,.5]]*2])
    batch=competition(boxes,heads,cameras)
    for b,n in np.ndindex(2,2):
        scalar=competition(boxes[n],heads[n],cameras[b,n])
        assert batch['deficit'][b,n]==scalar['deficit']
    boxes[...,4]=.36
    np.testing.assert_array_equal(batch['deficit'],competition(boxes,heads,cameras)['deficit'])
    boxes[0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite normalized'):
        competition(boxes,heads,cameras)


def test_catalogue_uses_the_actual_displayed_person_and_keeps_unknown_gaps():
    _,cats,_=prepared();original=cats[0]
    original.bank['heads'][:]=0.;original.bank['heads'][:,:,0]=[.5,.25,.1,.99]
    boxes=np.array([[.43,.19,.57,.68,.8],[.65,.1,1.,1.,.9]],np.float32)
    original.bank['person_rectangles_by_native']={int(f):boxes for f in original.bank['native_indices'].ravel()}
    args=(original.nodes,original.bank,original.body,[],original.style,[])
    old=SourceCatalogue(*args,fps=30.,editorial_content=True)
    new=SourceCatalogue(*args,fps=30.,editorial_content=True,person_attention=True)
    frames=np.arange(54);camera=np.tile([1.,.5,.5],(54,1))
    delta=new.rates(frames,camera)-old.rates(frames,camera)
    assert delta[new.content_scope].min()>.75
    np.testing.assert_allclose(delta[~new.content_scope],.5)
    sample=new.observation(1,0,np.tile([1.,.5,.5],(7,1)))
    assert sample['person_competition_measured'] and sample['person_competition_deficit']>.75
    np.testing.assert_allclose(new.rates(frames,np.stack([camera,camera])),
        np.stack([new.rates(frames,camera)]*2))


def test_paired_proposal_quality_uses_its_actual_crop_not_native_person_competition():
    from crosscut2.boundary_port_domain import proposal_display_quality
    _,cats,_=prepared();original=cats[0]
    original.bank['heads'][:]=0.;original.bank['heads'][:,:,0]=[.5,.25,.1,.99]
    boxes=np.array([[.43,.19,.57,.68,.8],[.65,.1,1.,1.,.9]],np.float32)
    original.bank['person_rectangles_by_native']={int(f):boxes for f in original.bank['native_indices'].ravel()}
    cat=SourceCatalogue(original.nodes,original.bank,original.body,[],original.style,[],
        fps=30.,editorial_content=True,person_attention=True,output_height=1080)
    native=cat.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    values=[]
    for camera in ([1.,.5,.5],[2.,.5,.5]):
        actual=cat.observation(1,0,np.tile(camera,(7,1)),target_head_index=0)
        proposed=proposal_display_quality(native,camera)
        np.testing.assert_allclose(proposed,actual['displayed_native_quality_proxy'])
        values.append(proposed)
    assert values[1]>values[0]  # A real crop removes part of the competing body.
    missing=dict(native,person_rectangles=None,person_competition_measured=False)
    disabled={k:v for k,v in native.items() if not k.startswith('person_')}
    for camera in ([1.,.5,.5],[2.,.5,.5]):
        assert proposal_display_quality(missing,camera)==proposal_display_quality(disabled,camera)


def cached_world(tmp_path,monkeypatch):
    w=world(tmp_path,monkeypatch)
    w.bound[0]['source_sha256']=hashlib.sha256(w.paths[0].read_bytes()).hexdigest()
    run(w,[{20:0,21:0}])
    monkeypatch.setattr(people,'_runtime_metadata',lambda:dict(test=w.runtime))
    return w


def recover(w,**kw):
    return people.recover_existing_rectangles(sources=w.paths,bound=w.bound,requests=[{20,21}],
        cache=w.cache,check=lambda:None,**kw)


def test_raw_recovery_requires_no_model_asset_decode_or_new_session(tmp_path,monkeypatch):
    w=cached_world(tmp_path,monkeypatch);before={p.name:p.read_bytes() for p in w.cache.glob('*.npz')}
    w.model.unlink();forbid_cold(monkeypatch)
    output,receipt=recover(w)
    assert set(output[0])=={20,21} and receipt['new_person_forwards']==receipt['source_rasters']==0
    assert receipt['source_contents_rehashed'] and not receipt['model_file_required']
    assert before=={p.name:p.read_bytes() for p in w.cache.glob('*.npz')}


def test_remount_accepts_only_device_change_and_does_not_rewrite_originals(tmp_path,monkeypatch):
    w=cached_world(tmp_path,monkeypatch)
    for p in list(w.cache.glob('*.npz')):
        with np.load(p,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
        binding=json.loads(str(arrays['binding']));binding['source'][0]+=99
        encoded=json.dumps(binding,sort_keys=True);arrays['binding']=np.array(encoded)
        destination=w.cache/(hashlib.sha256(encoded.encode()).hexdigest()+'.npz')
        np.savez_compressed(destination,**arrays);p.unlink()
    before={p.name:p.read_bytes() for p in w.cache.glob('*.npz')}
    output,receipt=recover(w)
    assert set(output[0])=={20,21} and receipt['remounted_records']==2
    assert before=={p.name:p.read_bytes() for p in w.cache.glob('*.npz')}
    w.runtime='different-producer'
    assert recover(w)[0]==[{}]


def test_recovery_rejects_changed_source_content_before_reading_old_boxes(tmp_path,monkeypatch):
    w=cached_world(tmp_path,monkeypatch);w.paths[0].write_bytes(b'changed source')
    w.bound[0]['source_file_identity']=list(people.file_identity(w.paths[0]))
    with pytest.raises(ValueError,match='source content changed'):recover(w)


def test_historical_raw_arrays_are_read_as_old_not_current_runtime_inference(tmp_path,monkeypatch):
    w=world(tmp_path,monkeypatch)
    w.bound[0]['source_sha256']=hashlib.sha256(w.paths[0].read_bytes()).hexdigest()
    old=dict(test='same-pixel-code',numpy='2.2.6',opencv='4.10.0',onnxruntime='1.30.0')
    monkeypatch.setattr(people,'_runtime',lambda p:(old,people.file_identity(p)))
    run(w,[{20:0,21:0}])
    current=dict(old,numpy='2.4.6',opencv='5.0.0')
    monkeypatch.setattr(people,'_runtime_metadata',lambda:current)
    output,receipt=recover(w)
    assert set(output[0])=={20,21}
    assert receipt['historical_runtime_records_read_as_original']==2
    assert not receipt['old_runtime_relabelled']
    current['test']='different-pixel-code'
    assert recover(w)[0]==[{}]
