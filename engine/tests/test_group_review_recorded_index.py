from copy import deepcopy
import pytest

from crosscut.group_performer_family_owner import digest
from crosscut.group_review_recorded_index import RecordedGroupReviewIndex


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
