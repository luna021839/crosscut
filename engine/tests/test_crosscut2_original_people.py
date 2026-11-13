"""Whole-image people remain independent of which face a cut later chooses."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.multi_person_pose import BODY8_COCO17,MODEL_SHA256
from crosscut2.original_people_bank import FAMILY,save_chunk,load_chunk,load_reference
from crosscut2.original_people_subjects import OriginalPeople,link_people,independently_supported_dense
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.detailed_pose import describe
from test_crosscut2_source_clock_action import catalogue


def observed_catalogue(*,missing=False,cut=()):
    old=catalogue(cut=cut);bank=dict(old.bank);bank['heads']=old.bank['heads'].copy()
    bank['native_indices']=bank['pose_native_indices'][:,[1,2]].copy()
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


def test_original_instance_competition_is_shared_by_content_and_prepruning_views():
    from crosscut2.boundary_port_domain import proposal_display_quality
    cat=observed_catalogue();bank=cat.bank
    for value in bank['original_people_observations']['by_native'].values():
        value['keypoints17']=np.concatenate([value['keypoints17'],np.zeros((1,17,3),np.float32)])
        value['boxes']=np.r_[value['boxes'],[[.01,.15,.99,1.]]]
        value['scores']=np.r_[value['scores'],[.9]]
    actual=SourceCatalogue(cat.nodes,bank,None,[],cat.style,[],fps=30.,editorial_content=True,
        person_attention=True,original_person_attention=True,output_height=1080)
    native=actual.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    assert native['person_competition_measured'] and native['person_competition_deficit']>.75
    for camera in ([1.,.5,.5],[2.,.45,.4]):
        proposed=proposal_display_quality(native,camera)
        final=actual.observation(1,0,np.tile(camera,(7,1)),target_head_index=0)
        np.testing.assert_allclose(proposed,final['displayed_native_quality_proxy'])
    frames=np.arange(39);camera=np.tile([1.,.5,.5],(39,1))
    np.testing.assert_allclose(actual.rates(frames,np.stack([camera,camera])),
        np.stack([actual.rates(frames,camera)]*2))


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


def precise_fixture(tmp_path):
    from test_crosscut2_detailed_pose import fixture,observation
    from crosscut2.detailed_pose import attach
    window,path,_,_=fixture(tmp_path);window=attach(window,path)
    context=observation(window,0)['dense_subject_context'];slots=[0,6,7,13]
    value=dict(side=0,source_clock_body=window.body[slots].copy(),
        source_clock_evidence=np.where(window.support[slots],1,0),
        source_clock_native_indices=window.ordinals[slots].copy())
    return window,path,context,value


def test_precise_gestures_are_retained_without_rewriting_or_relabelling_old_windows(tmp_path):
    window,path,context,value=precise_fixture(tmp_path);original=path.read_bytes()
    actual=independently_supported_dense(value,context,source=0,binding=window.binding,aspect=16/9)
    assert actual['independent_full_image_body_validation']['confirmed_arms']==[True,True]
    np.testing.assert_array_equal(actual['window'].body,window.body)
    np.testing.assert_array_equal(actual['supported'],context['supported'])
    assert path.read_bytes()==original and actual['window'] is not window


def test_one_contradicted_dense_arm_cannot_override_a_correct_original_person(tmp_path):
    window,_,context,value=precise_fixture(tmp_path)
    value['source_clock_body'][:,[0,2,4],0]+=.4
    actual=independently_supported_dense(value,context,source=0,binding=window.binding,aspect=16/9)
    assert actual['independent_full_image_body_validation']['confirmed_arms']==[False,True]
    assert not actual['window'].support[[0,6,7,13]][:,[0,2,4]].any()
    assert actual['window'].support[[0,6,7,13]][:,[1,3,5]].all()
    assert window.support[[0,6,7,13]][:,[0,2,4]].all()  # old cache remains immutable
    value['source_clock_body'][:,[1,3,5],0]+=.4
    assert independently_supported_dense(value,context,source=0,binding=window.binding,aspect=16/9) is None


def test_a_precise_arm_needs_actual_independent_support_not_just_missing_contradictions(tmp_path):
    window,_,context,value=precise_fixture(tmp_path)
    value['source_clock_evidence'][:]=0
    assert independently_supported_dense(value,context,source=0,binding=window.binding,aspect=16/9) is None
    value['source_clock_native_indices'][1]+=1
    with pytest.raises(ValueError,match='physical clock/source'):
        independently_supported_dense(value,context,source=0,binding=window.binding,aspect=16/9)


@pytest.mark.parametrize('independent',[False,True])
def test_unusable_supplemental_dense_evidence_does_not_erase_new_independent_source_pose(tmp_path,monkeypatch,independent):
    from crosscut2 import dense_source_action
    window,_,context,_=precise_fixture(tmp_path)
    monkeypatch.setattr(dense_source_action,'_seam_action',lambda *a,**k:dict(seam_action_supported=False))
    side=dict(dense_subject_contexts=[context],endpoint_cameras=np.array([[1.,.5,.5]]),
        independent_full_image_body_context=np.array([independent]))
    rates,available,basis,observed,used=dense_source_action.paired(side,side,
        rates=np.full((1,1,2),.3),available=np.ones((1,1),bool),basis=np.full((1,1),.2))
    np.testing.assert_allclose(rates,.3 if independent else .5)
    assert available[0,0]==independent and basis[0,0]==(.2 if independent else 0.)
    assert observed[0,0] and not used[0,0]


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
