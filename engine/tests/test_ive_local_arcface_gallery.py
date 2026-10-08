"""Synthetic reference vectors exercise the private local adapter, not accuracy."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from crosscut.ive import IVEMember
from crosscut.ive_local_arcface_gallery import (LocalIVEArcFaceGallery,
    LocalIVEArcFaceFeatureCache, CONTRACT, local_member_hint_conflict, observe_recorded_member_features,
    validate_recorded_member_features)
from crosscut.native_face_descriptor import (NativeFaceDescriptorSpace,
    ARCFACE_MODEL_NAME, ARCFACE_R50_SHA256, SFACE_MODEL_NAME)
from crosscut.face_identity import SFACE_SHA256
from crosscut.group_performer_family_owner import digest
from crosscut.group_source_changing_owner import NativeMotionDemand,NativeMotionSupply
from crosscut.group_scene_focal_pose_motion import restore_native_row
from tests.test_group_review_recorded_index import native_record


def vector(index):
    result=[0.]*512;result[index]=1.
    return tuple(result)


def gallery():
    return LocalIVEArcFaceGallery(tuple((member,(vector(i),)) for i,member in enumerate(IVEMember)),('a'*64,))


SPACE=NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME,512,ARCFACE_R50_SHA256,'b'*64)


def cached_features(raw,model,index=1):
    return dict(contract=CONTRACT,**{k:raw[k] for k in ('request_source_index','native_frame_index',
        'source_clock_identity','source_file_identity','native_pixel_sha256')},
        recorded_native_observation_sha256=digest(raw),descriptor_space=SPACE.to_dict(),
        gallery_identity=model.identity,named_member_identity_certified=False,
        persistent_actor_identity_certified=False,quality_approved=False,
        heads=[dict(observation_id=h['observation_id'],embedding=vector(index),
            member_hint=model.classify(vector(index),descriptor_space=SPACE)) for h in raw['heads']])


def test_native_member_features_roundtrip_and_parent_specific_reuse():
    raw,_=native_record();model=gallery();features=cached_features(raw,model)
    cache=LocalIVEArcFaceFeatureCache(model,SPACE)
    cache.add(raw,features)
    restored=LocalIVEArcFaceFeatureCache.from_dict(json.loads(json.dumps(cache.to_dict())),
        gallery=model,descriptor_space=SPACE)
    assert restored.get(raw)==json.loads(json.dumps(features))
    assert restored.get(dict(raw,native_frame_index=81)) is None
    assert restored.to_dict()['photo_references_copied'] is False
    assert cache.stored_bytes<40000


def test_native_member_feature_conflicts_and_limits_are_atomic():
    raw,_=native_record();model=gallery();features=cached_features(raw,model)
    cache=LocalIVEArcFaceFeatureCache(model,SPACE,maximum_rows=1)
    cache.add(raw,features);before=cache.to_dict()
    conflict=cached_features(raw,model,2)
    with pytest.raises(ValueError,match='conflicting'):
        cache.import_features([conflict])
    assert cache.to_dict()==before
    other=dict(raw,native_frame_index=81)
    with pytest.raises(ValueError,match='row/byte budget'):
        cache.import_features([cached_features(other,model)])
    assert cache.to_dict()==before
    tiny=LocalIVEArcFaceFeatureCache(model,SPACE,maximum_bytes=1)
    with pytest.raises(ValueError,match='row/byte budget'):
        tiny.add(raw,features)
    assert not tiny.rows and tiny.stored_bytes==0


def test_member_feature_archive_does_not_promote_identity_or_wrong_gallery():
    raw,_=native_record();model=gallery();features=cached_features(raw,model)
    cache=LocalIVEArcFaceFeatureCache(model,SPACE)
    with pytest.raises(ValueError,match='authority'):
        cache.add(raw,dict(features,named_member_identity_certified=True))
    wrong=replace(model,minimum_similarity=.6)
    value=dict(cache.to_dict(),gallery_identity=wrong.identity)
    with pytest.raises(ValueError,match='bindings'):
        LocalIVEArcFaceFeatureCache.from_dict(value,gallery=model,descriptor_space=SPACE)


def test_confident_member_conflict_adds_evidence_cost_not_an_automatic_track_pass():
    from crosscut.group_review_actor_path import _link
    from tests.test_group_review_actor_path import view
    model=gallery();a=view(0,0);b=view(2,0)
    baseline=_link(a,b,maximum_native_gap=12,minimum_cosine=.5)
    a['member_hint']=model.classify(vector(0),descriptor_space=SPACE)
    b['member_hint']=model.classify(vector(1),descriptor_space=SPACE)
    conflict=_link(a,b,maximum_native_gap=12,minimum_cosine=.5)
    assert conflict['local_photo_member_hint_conflict']
    assert conflict['cost']==pytest.approx(baseline['cost']+2.)
    b['vector']=[0.,1.]+[0.]*126
    assert _link(a,b,maximum_native_gap=12,minimum_cosine=.5) is None
    b['vector']=a['vector'];b['member_hint']=model.classify(None,descriptor_space=SPACE)
    assert _link(a,b,maximum_native_gap=12,minimum_cosine=.5)['cost']==pytest.approx(baseline['cost'])


def test_six_member_private_reference_scores_and_unknown_margin():
    model=gallery();members=tuple(IVEMember)
    hint=model.classify(vector(1),descriptor_space=SPACE)
    assert hint['member']==members[1].value and hint['state']=='LIKELY_MEMBER'
    assert hint['named_member_identity_certified'] is False
    assert hint['stage_video_thresholds_calibrated'] is False
    assert model.classify(vector(300),descriptor_space=SPACE)['state']=='UNKNOWN'
    tied=np.asarray(vector(0))+np.asarray(vector(1))
    assert model.classify(tied.tolist(),descriptor_space=SPACE)['state']=='UNKNOWN'
    assert model.classify(None,descriptor_space=SPACE)['member'] is None
    assert 'embedding' not in model.summary()
    assert model.summary()['references_copied'] is False


def test_sface_coordinates_cannot_enter_the_arcface_gallery():
    wrong=NativeFaceDescriptorSpace(SFACE_MODEL_NAME,128,SFACE_SHA256,'c'*64)
    with pytest.raises(ValueError,match='SFace and ArcFace'):
        gallery().classify(vector(0),descriptor_space=wrong)


def test_explicit_photo_bank_import_retains_only_original_core_references(tmp_path):
    model=gallery();members=tuple(IVEMember)
    alias={'장원영':'WONYOUNG'}
    content={'galleries':{alias.get(m.value,m.name):[
        dict(embedding=vector(i),reference_core=True),
        dict(embedding=vector(400),reference_core=False)]
        for i,m in enumerate(members) if m is not IVEMember.ANYUJIN}}
    a=tmp_path/'members.json';b=tmp_path/'yujin.json'
    a.write_text(json.dumps(content))
    b.write_text(json.dumps(dict(model_sha256=ARCFACE_R50_SHA256,references=[
        dict(embedding=vector(0),reference_core=True)])))
    actual=LocalIVEArcFaceGallery.from_existing_photo_banks(a,b)
    assert set(actual.summary()['member_reference_counts'].values())=={1}
    assert actual.classify(vector(400),descriptor_space=SPACE)['member'] is None
    a.write_text(json.dumps(dict(content,model_sha256='d'*64)))
    with pytest.raises(ValueError,match='another ArcFace'):
        LocalIVEArcFaceGallery.from_existing_photo_banks(a,b)
    a.write_text(json.dumps(content))
    b.write_text(json.dumps(dict(model_sha256='d'*64,references=[])))
    with pytest.raises(ValueError,match='another ArcFace'):
        LocalIVEArcFaceGallery.from_existing_photo_banks(a,b)


@pytest.mark.parametrize('bad',[(0.,)*512,(1.,)*128,(float('nan'),)+(0.,)*511,(True,)+(0.,)*511])
def test_invalid_query_vectors_are_rejected_not_normalized_into_a_member(bad):
    with pytest.raises(ValueError):gallery().classify(bad,descriptor_space=SPACE)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),1.1])
def test_invalid_calculated_cosines_are_never_clamped_into_a_member(monkeypatch,invalid):
    monkeypatch.setattr(np,'einsum',lambda *a,**k:np.asarray([invalid]))
    with pytest.raises(ValueError,match='cosine calculation'):
        gallery().classify(vector(0),descriptor_space=SPACE)


def test_large_finite_numbers_do_not_overflow_vector_normalization():
    actual=gallery().classify([1e308]*512,descriptor_space=SPACE)
    assert actual['state']=='UNKNOWN'
    assert all(abs(s-1/np.sqrt(512))<1e-12 for s in actual['member_scores'].values())


def test_member_hints_penalize_only_same_gallery_strong_conflicts():
    model=gallery();a=model.classify(vector(0),descriptor_space=SPACE)
    b=model.classify(vector(1),descriptor_space=SPACE)
    assert local_member_hint_conflict(a,b)
    assert not local_member_hint_conflict(a,a)
    assert not local_member_hint_conflict(a,None)
    assert not local_member_hint_conflict(a,model.classify(vector(300),descriptor_space=SPACE))
    assert not local_member_hint_conflict(a,dict(b,gallery_identity='d'*64))
    with pytest.raises(ValueError,match='authority'):
        local_member_hint_conflict(a,dict(b,quality_approved=True))


def test_raw_parent_and_native_pixel_hash_are_kept_exact_during_member_completion():
    import hashlib
    raw,_=native_record();raw=restore_native_row(raw)
    pixels=np.zeros((100,160,3),dtype=np.uint8)
    clock=SimpleNamespace(to_dict=lambda:{'frame_rate':30,'frame_count':100})
    raw=replace(raw,native_pixel_sha256=hashlib.sha256(memoryview(pixels)).hexdigest(),
        source_clock_identity=digest(clock.to_dict()))
    calls=[];model=gallery()
    session=SimpleNamespace(closed=False,domain=SimpleNamespace(clocks_by_request={raw.request_source_index:clock}),
        models={ARCFACE_MODEL_NAME:(None,None,ARCFACE_R50_SHA256)},model_adapters={ARCFACE_MODEL_NAME:'b'*64},
        _assert_model_assets=lambda name:None,_assert_source_identity=lambda source:raw.source_file_identity,
        native_pixels=lambda source,native:pixels,
        observe_native=lambda source,native,**kwargs:(calls.append((source,native,kwargs['model'])) or vector(1)))
    demand=NativeMotionDemand(raw.request_source_index,raw.native_frame_index,0,'source-0','other',
        require_member_features=True)
    supply=NativeMotionSupply(demand,raw)
    features=observe_recorded_member_features(session,supply,model)
    payload=json.loads(json.dumps(raw.to_dict()))
    assert features['recorded_native_observation_sha256']==digest(payload)
    assert features['heads'][0]['observation_id']==raw.heads[0].observation_id
    assert features['heads'][0]['member_hint']['member']==IVEMember.GAEUL.value
    assert len(calls)==len(raw.heads) and all(c[2]==ARCFACE_MODEL_NAME for c in calls)
    validate_recorded_member_features(payload,features,gallery=model,expected_space=SPACE)
    changed=dict(features,native_pixel_sha256='d'*64)
    with pytest.raises(ValueError,match='raw parent'):
        validate_recorded_member_features(payload,changed,gallery=model,expected_space=SPACE)
    session.native_pixels=lambda source,native:np.ones_like(pixels)
    with pytest.raises(ValueError,match='native raster'):
        observe_recorded_member_features(session,supply,model)
