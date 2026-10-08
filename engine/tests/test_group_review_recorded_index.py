from copy import deepcopy
from hashlib import sha256
import json
import pytest

from crosscut.group_performer_family_owner import digest
from crosscut.group_review_recorded_index import (RecordedGroupReviewIndex,verify_recorded_source_file,
    native_analysis_variant)
from crosscut.group_joint_observations import _file_identity


def test_remount_requires_complete_content_hash_and_preserves_recorded_address(tmp_path):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'actual complete source bytes')
    current = _file_identity(path)
    recorded = [current[0] + 1, *current[1:]]
    original = list(recorded)
    result = verify_recorded_source_file(path, recorded,
        content_sha256=sha256(path.read_bytes()).hexdigest())
    assert result['state'] == 'CONTENT_VERIFIED_DEVICE_REMOUNT'
    assert result['recorded_file_identity'] == recorded == original
    assert result['current_file_identity'] == list(current)
    assert not result['archived_observations_rewritten'] and not result['quality_approved']
    for missing_or_wrong in (None, '0' * 64):
        with pytest.raises(ValueError):
            verify_recorded_source_file(path, recorded, content_sha256=missing_or_wrong)


def test_remount_does_not_accept_other_file_changes_even_with_supplied_digest(tmp_path):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'unchanged source')
    current = _file_identity(path)
    for index in (1, 2, 3, 4):
        recorded = [current[0] + 1, *current[1:]]
        recorded[index] += 1
        with pytest.raises(ValueError, match='beyond'):
            verify_recorded_source_file(path, recorded,
                content_sha256=sha256(path.read_bytes()).hexdigest())


def test_exact_identity_does_not_hash_source_again(tmp_path, monkeypatch):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'current source')
    def forbidden_hash(*args, **kwargs):
        raise AssertionError('unchanged identity must not repeat whole-file hashing')
    monkeypatch.setattr('crosscut.group_review_recorded_index.file_digest', forbidden_hash)
    result = verify_recorded_source_file(path, _file_identity(path))
    assert result['state'] == 'EXACT_FILE_IDENTITY' and result['content_hash_bytes'] == 0


def test_remount_rejects_change_during_hash_and_charges_io(tmp_path, monkeypatch):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'source')
    current = _file_identity(path)
    calls = []
    class Port:
        def charge(self, resource, amount=1):
            calls.append((resource, amount))
    identities = iter((current, (*current[:2], current[2] + 1, *current[3:])))
    monkeypatch.setattr('crosscut.group_review_recorded_index._file_identity', lambda path: next(identities))
    with pytest.raises(ValueError, match='during'):
        verify_recorded_source_file(path, [current[0] + 1, *current[1:]],
            content_sha256=sha256(path.read_bytes()).hexdigest(), work_port=Port())
    assert calls == [('identity_reads', 1), ('model_hash_bytes', current[2]), ('identity_reads', 1)]


def capture(tick=8, native=80, *, face=True, appearance=True):
    raw = dict(request_source_index=2, native_frame_index=native,
        source_clock_identity='c'*64, source_file_identity=[1, 2, 3, 4, 5],
        native_pixel_sha256='a'*64, heads=[dict(observation_id=7)] if face else [],
        people=[], quality_approved=False)
    app = dict(request_source_index=2, native_frame_index=native,
        source_clock_identity='c'*64, source_file_identity=[1, 2, 3, 4, 5],
        native_pixel_sha256='a'*64, recorded_native_observation_sha256=digest(raw),
        heads=[dict(observation_id=7, embedding=None)] if face else [], quality_approved=False)
    return dict(rows={str(tick):dict(output_tick=tick, native_observation=raw)},
        patches={str(tick):{'7':dict(detail_risk=.2)} if face else {}},
        appearance={str(tick):app} if appearance else {}, quality_approved=False)


def lookup(index, native=80, **changes):
    args=dict(source=2, native=native, clock='c'*64, identity=[1, 2, 3, 4, 5])
    return index.lookup(**dict(args, **changes))


def test_same_native_at_different_output_ticks_reuses_without_rewriting_saved_rows():
    index=RecordedGroupReviewIndex()
    a,b=capture(),capture(tick=20)
    original=deepcopy((a,b))
    index.add_capture(a, provenance='original')
    index.add_capture(b, provenance='candidate')
    result=lookup(index)
    assert result['state']=='RECORDED_NATIVE_REVIEW_INPUT'
    assert [r['output_tick'] for r in result['occurrences']]==[8,20]
    assert len(result['raw_variant_sha256'])==1
    assert (a,b)==original and not result['live_native_authority']


def test_adverse_native_result_cannot_be_replaced_by_favorable_capture():
    index=RecordedGroupReviewIndex()
    index.add_capture(capture(face=False), provenance='adverse')
    index.add_capture(capture(face=True), provenance='favorable')
    result=lookup(index)
    assert result['state']=='CONFLICTING_NATIVE_RECORDS'
    assert result['raw'] is None and len(result['raw_variant_sha256'])==2


def test_missing_appearance_does_not_erase_actual_visible_heads():
    index=RecordedGroupReviewIndex()
    index.add_capture(capture(appearance=False), provenance='missing-appearance')
    result=lookup(index)
    assert result['raw']['heads'] and result['appearance'] is None
    assert result['patches']['7']['detail_risk']==.2


def test_conflicting_feature_does_not_grant_best_embedding_or_delete_native_visibility():
    index=RecordedGroupReviewIndex()
    a,b=capture(),capture(tick=9)
    b['appearance']['9']['heads'][0]['embedding']=[1.]+[0.]*127
    index.add_capture(a, provenance='missing-vector')
    index.add_capture(b, provenance='observed-vector')
    result=lookup(index)
    assert result['state']=='CONFLICTING_FEATURE_RECORDS'
    assert result['appearance'] is None and result['raw']['heads']


def test_address_is_not_only_source_number_and_native_frame():
    index=RecordedGroupReviewIndex()
    index.add_capture(capture(), provenance='original')
    for changes in (dict(source=3),dict(native=81),dict(clock='b'*64),dict(identity=[1,2,3,4,6])):
        assert lookup(index, **changes)['state']=='MISSING'


def test_wrong_appearance_parent_or_output_tick_is_rejected_atomically():
    index=RecordedGroupReviewIndex()
    for kind in ('parent','output'):
        data=capture()
        if kind=='parent':data['appearance']['8']['recorded_native_observation_sha256']='b'*64
        else:data['rows']['8']['output_tick']=9
        with pytest.raises(ValueError):index.add_capture(data, provenance='incorrect')
        assert index.diagnostics()['imported_rows']==0


def test_row_and_byte_limits_precede_any_import():
    for index in (RecordedGroupReviewIndex(maximum_rows=1),RecordedGroupReviewIndex(maximum_bytes=1)):
        data=capture()
        if index.maximum_rows==1:data['rows']['9']=dict(data['rows']['8'],output_tick=9)
        with pytest.raises(ValueError,match='budget'):index.add_capture(data, provenance='too-large')
        assert index.diagnostics()['physical_addresses']==0


def test_returned_payload_cannot_mutate_recorded_index_or_gain_authority():
    index=RecordedGroupReviewIndex()
    index.add_capture(capture(), provenance='original')
    result=lookup(index)
    result['raw']['heads'].clear()
    assert lookup(index)['raw']['heads']
    assert not result['quality_approved'] and not result['fresh_inference_performed']


def native_record(*,adapter='d'*64,reason='observed'):
    from crosscut.group_joint_native_observer import NativeGroupObservation,NativeGroupHead
    from crosscut.native_face_descriptor import NativeFaceDescriptorSpace,SFACE_MODEL_NAME
    from crosscut.face_identity import SFACE_SHA256
    names=('nanodet_people','rtmpose_body7','yunet_heads')
    head=NativeGroupHead(7,(.4,.2,.1,.2),((.41,.25),(.48,.25),(.45,.3),(.42,.35),(.47,.35)),.9)
    raw=NativeGroupObservation(2,80,(1920,1080),'c'*64,(1,2,3,4,5),'a'*64,
        tuple((n,'b'*64) for n in names),(),(head,),reason,
        model_adapter_fingerprints=tuple((n,adapter) for n in names)).to_dict()
    raw=json.loads(json.dumps(raw))
    appearance={k:deepcopy(raw[k]) for k in ('request_source_index','native_frame_index',
        'source_clock_identity','source_file_identity','native_pixel_sha256')}
    appearance.update(recorded_native_observation_sha256=digest(raw),quality_approved=False,
        descriptor_space=NativeFaceDescriptorSpace(SFACE_MODEL_NAME,128,SFACE_SHA256,'f'*64).to_dict(),
        heads=[dict(observation_id=7,embedding=[1.]+[0.]*127)])
    return raw,appearance


def test_raw_native_import_needs_no_fabricated_output_capture_and_keeps_missing_appearance():
    index=RecordedGroupReviewIndex();raw,appearance=native_record();saved=deepcopy(raw)
    index.add_native_record(raw,provenance='normalized-native-packet')
    result=lookup(index,analysis_variant_sha256=native_analysis_variant(raw))
    assert result['raw']==saved and result['appearance'] is None
    assert result['occurrences'][0]['output_tick'] is None
    index.add_native_record(raw,appearance,provenance='appearance-completion')
    assert lookup(index)['appearance']==appearance and raw==saved


def decode_failure_record():
    raw,_=native_record(reason='exact_native_decode_unavailable')
    return dict(raw,native_pixel_sha256=None,frame_size_wh=[0,0],heads=[],people=[])


def test_exact_decode_failure_survives_an_explicit_native_cache_reload(tmp_path):
    raw=decode_failure_record();saved=deepcopy(raw)
    index=RecordedGroupReviewIndex()
    cache_path=tmp_path/'explicit-native-input-cache.json'
    cache_path.write_text(json.dumps(dict(rows=[],decode_failures=[raw])),encoding='utf-8')
    persisted=json.loads(cache_path.read_text(encoding='utf-8'))
    index.add_native_completion_records(persisted['rows'],decode_failures=persisted['decode_failures'],
        provenance='persisted exact decode attempt')
    result=lookup(index,analysis_variant_sha256=native_analysis_variant(raw))
    assert result['state']=='RECORDED_EXACT_DECODE_UNAVAILABLE'
    assert result['raw']==saved and result['appearance'] is None and result['patches'] is None
    assert result['occurrences'][0]['output_tick'] is None
    assert not result['fresh_inference_performed'] and not result['live_native_authority']
    assert raw==saved
    for changes in (dict(native=81),dict(clock='e'*64),dict(identity=[1,2,3,4,6])):
        assert lookup(index,**changes)['state']=='MISSING'


@pytest.mark.parametrize('failure_first',(False,True))
def test_success_cannot_hide_a_retained_exact_decode_failure(failure_first):
    raw,app=native_record();failure=decode_failure_record();index=RecordedGroupReviewIndex()
    operations=(lambda:index.add_native_decode_failure(failure,provenance='exact failed read'),
        lambda:index.add_native_record(raw,app,provenance='different successful measurement'))
    for operation in operations if failure_first else operations[::-1]:operation()
    result=lookup(index,analysis_variant_sha256=native_analysis_variant(raw))
    assert result['state']=='CONFLICTING_NATIVE_RECORDS' and result['raw'] is None
    assert len(result['raw_variant_sha256'])==2


@pytest.mark.parametrize('damage',('reason','head','dimensions','appearance','approval'))
def test_failure_import_cannot_smuggle_successful_features_or_an_unobserved_miss(damage):
    raw,app=native_record();failure=decode_failure_record();index=RecordedGroupReviewIndex()
    if damage=='reason':failure['reason']='unknown_not_observed'
    if damage=='head':failure['heads']=raw['heads']
    if damage=='dimensions':failure['frame_size_wh']=[1920,1080]
    if damage=='approval':failure['quality_approved']=True
    good=dict(native_observation=raw,appearance=app)
    if damage=='appearance':good=dict(native_observation=failure,appearance=app)
    with pytest.raises(ValueError):
        index.add_native_completion_records([good],decode_failures=[failure],provenance='invalid mixed input')
    assert index.imported_rows==0 and index.stored_bytes==0 and not index.rows


def test_observed_failure_does_not_block_a_different_declared_model_variant():
    failure=decode_failure_record();changed,_=native_record(adapter='e'*64)
    index=RecordedGroupReviewIndex();index.add_native_decode_failure(failure,provenance='old analysis variant')
    assert lookup(index,analysis_variant_sha256=native_analysis_variant(changed))['state']=='MISSING_DECLARED_ANALYSIS_VARIANT'
    index.add_native_record(changed,provenance='explicit different declared analysis')
    assert lookup(index,analysis_variant_sha256=native_analysis_variant(changed))['raw']==changed
    assert lookup(index,analysis_variant_sha256=native_analysis_variant(failure))['state']=='RECORDED_EXACT_DECODE_UNAVAILABLE'


def test_predeclared_analysis_variants_keep_their_own_raw_parent_features():
    index=RecordedGroupReviewIndex();a,aa=native_record();b,bb=native_record(adapter='e'*64)
    bb['heads'][0]['embedding']=[0.,1.]+[0.]*126
    for raw,app in ((a,aa),(b,bb)):index.add_native_record(raw,app,provenance='declared-variant')
    assert lookup(index)['state']=='CONFLICTING_NATIVE_RECORDS'
    for raw,app in ((a,aa),(b,bb)):
        found=lookup(index,analysis_variant_sha256=native_analysis_variant(raw))
        assert found['raw']==raw and found['appearance']==app
        assert found['state']=='RECORDED_NATIVE_REVIEW_INPUT'
    assert lookup(index,analysis_variant_sha256='0'*64)['state']=='MISSING_DECLARED_ANALYSIS_VARIANT'
    bad,badapp=native_record(reason='different output within same variant')
    index.add_native_record(bad,badapp,provenance='conflict')
    assert lookup(index,analysis_variant_sha256=native_analysis_variant(a))['state']=='CONFLICTING_NATIVE_RECORDS'
    assert lookup(index,analysis_variant_sha256=native_analysis_variant(b))['raw']==b


@pytest.mark.parametrize('kind',('wrong_parent','dimension','duplicate','foreign_head','zero','nonfinite','adapter','missing_head'))
def test_native_import_rejects_invalid_appearance_without_partial_cache_mutation(kind):
    raw,app=native_record();index=RecordedGroupReviewIndex()
    if kind=='wrong_parent':app['recorded_native_observation_sha256']='0'*64
    elif kind=='dimension':app['heads'][0]['embedding']*=4
    elif kind=='duplicate':app['heads'].append(deepcopy(app['heads'][0]))
    elif kind=='foreign_head':app['heads'][0]['observation_id']=8
    elif kind=='zero':app['heads'][0]['embedding']=[0.]*128
    elif kind=='nonfinite':app['heads'][0]['embedding'][0]=float('nan')
    elif kind=='adapter':app['descriptor_space']['adapter_sha256']='unknown'
    elif kind=='missing_head':app['heads']=[]
    with pytest.raises(ValueError):index.add_native_record(raw,app,provenance='invalid')
    assert not index.rows and index.imported_rows==0
