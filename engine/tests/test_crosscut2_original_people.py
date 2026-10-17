"""Whole-image people remain independent of which face a cut later chooses."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.multi_person_pose import BODY8_COCO17,MODEL_SHA256
from crosscut2.original_people_bank import FAMILY,save_chunk,load_chunk,load_reference
from crosscut2.original_people_subjects import OriginalPeople,link_people
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.detailed_pose import describe
from test_crosscut2_source_clock_action import catalogue


def observed_catalogue(*,missing=False,cut=()):
    old=catalogue(cut=cut);bank=dict(old.bank);bank['heads']=old.bank['heads'].copy()
    bank['heads'][:,:,3]=[.2,.3,.12,.99]
    points=np.zeros((3,17,3),np.float32)
    for i,h in enumerate((0,1,3)):
        head=bank['heads'][1,0,h];points[i,:,:2]=head[:2];points[i,:,2]=.9
        points[i,list(BODY8_COCO17)]=old.body[1,1,0]+[head[0]-.45,0,0]
    rows={}
    for tick in sorted(set(bank['pose_native_indices'].ravel())):
        order=[1,0] if missing else [1,2,0]
        if tick%2:order=order[::-1]
        boxes=np.array([[p[0,0]-.12,.2,p[0,0]+.12,.85] for p in points[order]],np.float32)
        rows[int(tick)]=dict(keypoints17=points[order].copy(),scores=np.ones(len(order))*.9,boxes=boxes)
    raw=dict(by_native=rows,raster=[1280,720],family=FAMILY,source=0,reference_sha256='a'*64)
    bank['original_people_observations']=raw
    return SourceCatalogue(old.nodes,bank,old.body,[],old.style,[],fps=30.)


def test_all_people_can_supply_the_fourth_face_without_borrowing_face_zero():
    cat=observed_catalogue();view=np.tile([1.2,.2,.4],(7,1))
    sample=cat.observation(1,0,view,target_head_index=3)
    assert cat.body_capacity==8 and sample['original_full_image_people_used']
    assert sample['evidence'].min()==1 and sample['source_clock_subject_associated']
    assert sample['source_clock_subject_key']==(20,3,3)
    np.testing.assert_allclose(sample['body'][1,:,0],cat.body[1,1,0,:,0]-.25)
    assert np.isfinite(cat.rates(np.arange(39),np.tile([1.2,.2,.4],(39,1)))).all()


def test_missing_body_does_not_reuse_old_face_conditioned_pose_or_remove_the_face():
    cat=observed_catalogue(missing=True)
    cat.subjects=SimpleNamespace(sample=lambda *a:None,augment=lambda *a:pytest.fail('old body override'))
    sample=cat.observation(1,1,np.tile([1.2,.2,.4],(7,1)),target_head_index=3)
    assert sample['head_index']==3 and sample['head'][2]>.1
    assert not sample['evidence'].any() and not sample['body'].any()


def test_native_camera_cut_breaks_cross_side_and_far_body_hypotheses_only():
    cat=observed_catalogue(cut=[18,20])
    assert not cat.original_people.associated[1,0].any()
    assert cat.original_people.associated[1,1,3]
    assert cat.original_people.endpoint_matches[1,3]==-1


def test_reordered_person_outputs_link_geometrically_and_missing_heads_do_not_force_a_swap():
    cat=observed_catalogue();rows=cat.original_people.raw['by_native']
    a,b=rows[19],rows[20];links=link_people(a,b,aspect=16/9)
    for i,j in enumerate(links):np.testing.assert_allclose(a['keypoints17'][i],b['keypoints17'][j])
    assert len(set(links))==len(links)


def test_coco17_head_points_never_fabricate_finger_landmarks():
    cat=observed_catalogue();sample=cat.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=3)
    detail=describe(sample)
    assert detail['known'][:5].all() and not detail['known'][5:].any()


@pytest.mark.parametrize('wrong_wrist',[False,True])
def test_old_fingers_require_a_root_matching_the_actual_whole_frame_wrist(tmp_path,wrong_wrist):
    from test_crosscut2_detailed_pose import fixture,observation
    from crosscut2.detailed_pose import attach
    window,path,_,_=fixture(tmp_path);window=attach(window,path)
    previous=observation(window,0);body=window.body[[0,6]].copy()
    if wrong_wrist:body[:,4,0]+=.2
    current=dict(previous,original_full_image_people_used=True,original_people_associated=True,
        original_people_head_points=window.detailed_points[6,:5],head=window.heads[6],body=body,
        evidence=np.where(body[...,2]>=.3,1,0),native_indices=np.array([20,26]),
        detailed_subject_context=previous['dense_subject_context'])
    detail=describe(current)
    assert detail['known'][26:].all()
    assert detail['known'][5:26].all()==(not wrong_wrist)
    np.testing.assert_array_equal(detail['known'][:5],True)
    current['native_indices']=np.array([20,27])
    with pytest.raises(ValueError,match='endpoint clock'):describe(current)


def row(native,people=1):
    return dict(native=native,boxes=np.zeros((people,4),np.float32),scores=np.ones(people,np.float32)*.8,
        keypoints17=np.zeros((people,17,3),np.float32),raster_sha256='a'*64)


def test_raw_bank_preserves_empty_frames_and_exact_native_bindings(tmp_path):
    path=tmp_path/'raw.npz';binding=dict(source='a'*64,raster=[1280,720])
    save_chunk(path,[row(7,0),row(9,2)],binding=binding)
    loaded=load_chunk(path,binding=binding)
    assert list(loaded)==[7,9] and loaded[7]['keypoints17'].shape==(0,17,3)
    assert loaded[9]['keypoints17'].shape==(2,17,3)
    with pytest.raises(ValueError,match='binding changed'):load_chunk(path,binding={**binding,'source':'b'*64})
    with pytest.raises(FileExistsError):save_chunk(path,[row(7)],binding=binding)


def test_reference_reuse_is_model_free_but_requires_actual_source_schedule(tmp_path,monkeypatch):
    from crosscut2.grid_perception import file_identity
    import onnxruntime as ort
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:pytest.fail('cache read ran a model'))
    source=tmp_path/'source.mp4';source.write_bytes(b'fixture')
    chunk=tmp_path/'raw.npz';binding=dict(source_sha256='b'*64,raster=[1280,720],family=FAMILY,model_sha256=MODEL_SHA256)
    save_chunk(chunk,[row(7),row(9)],binding=binding)
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(family=FAMILY,sources=[str(source)],source_sha256=['b'*64],
        source_records=[dict(source_identity=list(file_identity(source)),binding=binding,
            chunks=[dict(path=str(chunk),sha256=hashlib.sha256(chunk.read_bytes()).hexdigest())])])))
    kw=dict(sources=[source],source_hashes=['b'*64],authorized_root=tmp_path)
    actual=load_reference(reference,banks=[dict(pose_native_indices=np.array([[7,7,9,9]]))],**kw)
    assert list(actual[0]['by_native'])==[7,9]
    with pytest.raises(ValueError,match='schedule was not observed'):
        load_reference(reference,banks=[dict(pose_native_indices=np.array([[7,8,9,9]]))],**kw)
    record=json.loads(reference.read_text());record['source_records'][0]['binding']['source_sha256']='c'*64
    reference.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='source/model/raster family'):
        load_reference(reference,banks=[dict(pose_native_indices=np.array([[7,7,9,9]]))],**kw)
