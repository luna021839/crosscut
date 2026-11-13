from dataclasses import replace
import hashlib
import json
import pytest

from test_stage_edit_face_cues import inputs
from crosscut.stage_edit_face_cues import augment_snapshot, schema_with_face_cues
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_native_face_completion import CONTRACT, GEOMETRY, missing_queries, augment_missing_native_faces


def fixture():
    base, face = inputs()
    face = replace(face, records=())
    snapshot = augment_snapshot(base, face)
    request = dict(sources=[dict(source_id='source-a', media_content_sha256='5'*64,
        frame_rate=30., clock=dict(offset_seconds=1., rate=1.))])
    # Use the exact public source-clock mapping schema.
    from crosscut.reference_evidence_builder import ReferenceSourceClock
    request['sources'][0]['clock'] = ReferenceSourceClock(1., 1.).to_dict()
    raw = json.dumps(request).encode()
    report = dict(contract=CONTRACT, request_file_sha256=hashlib.sha256(raw).hexdigest(),
        input_snapshot_sha256=[snapshot.snapshot_sha256], source_media_sha256={'source-a': '5'*64},
        gallery_identity=face.gallery_identity, descriptor_space=face.descriptor_space.to_dict(),
        current_source_contents_verified_against_request=True,
        selection='all_available_snapshot_candidates_with_absent_face_inputs', teacher_labels_consumed=False,
        rows=[dict(source_id='source-a', native_frame=30, media_content_sha256='5'*64, state='OBSERVED',
            face_bbox_xywh=[.3, .2, .2, .3], member_cosines=[.1, .2, .3, .4, .5, -.1])])
    return snapshot, face, raw, report


def test_real_inputs_gain_only_measured_primary_face_geometry_and_scores():
    old, face, request, report = fixture()
    original = old.to_dict()
    new = augment_missing_native_faces(old, face, report, request)
    assert old.to_dict() == original
    a, b = old.sequence.tokens[0], new.sequence.tokens[0]
    assert replace(b, source_candidates=a.source_candidates) == a
    aa, bb = a.source_candidates[0], b.source_candidates[0]
    assert replace(bb, features=aa.features, missing_mask=aa.missing_mask) == aa
    schema = schema_with_face_cues()
    indices = [schema.source_feature_names.index(n) for n in GEOMETRY]
    assert [bb.features[i] for i in indices] == [.4, .35, .3]
    assert all(not bb.missing_mask[i] for i in indices)
    assert bb.features[-6:] == (.1, .2, .3, .4, .5, -.1)
    for name in ('target_center_x', 'target_center_y'):
        assert bb.missing_mask[schema.source_feature_names.index(name)]
    assert new.causal_sensing_binding == old.causal_sensing_binding
    assert StageEditFeatureSnapshot.from_dict(new.to_dict(), schema=schema) == new


@pytest.mark.parametrize('state', ['NO_FACE', 'LANDMARKS_UNAVAILABLE', 'FRAME_UNAVAILABLE'])
def test_negative_or_partial_landmarks_do_not_fabricate_inputs_or_exclude_sources(state):
    old, face, raw, report = fixture()
    report['rows'][0] = {k:v for k,v in report['rows'][0].items() if k not in ('face_bbox_xywh', 'member_cosines')}
    report['rows'][0]['state'] = state
    new = augment_missing_native_faces(old, face, report, raw)
    assert new.sequence == old.sequence


def test_wrong_native_frame_cannot_be_filled_by_nearest_time():
    old, face, raw, report = fixture()
    report['rows'][0]['native_frame'] = 31
    assert augment_missing_native_faces(old, face, report, raw).sequence == old.sequence


@pytest.mark.parametrize('field,value', [('teacher_labels_consumed', True), ('gallery_identity', '9'*64),
    ('current_source_contents_verified_against_request', False), ('input_snapshot_sha256', ['9'*64])])
def test_input_domain_and_producer_binding_are_required(field, value):
    old, face, raw, report = fixture()
    report[field] = value
    with pytest.raises(ValueError, match='do not bind'):
        augment_missing_native_faces(old, face, report, raw)


@pytest.mark.parametrize('change', ['source', 'media', 'duplicate', 'nan', 'offscreen'])
def test_invalid_observation_is_not_made_plausible(change):
    old, face, raw, report = fixture()
    row = report['rows'][0]
    if change == 'source':row['source_id'] = 'foreign'
    elif change == 'media':row['media_content_sha256'] = '8'*64
    elif change == 'duplicate':report['rows'].append(dict(row))
    elif change == 'nan':row['member_cosines'][0] = float('nan')
    elif change == 'offscreen':row['face_bbox_xywh'][0] = .95
    with pytest.raises(ValueError):augment_missing_native_faces(old, face, report, raw)


def test_availability_known_measurements_and_other_times_are_preserved():
    old, face, raw, report = fixture()
    schema = schema_with_face_cues(); first = old.sequence.tokens[0]
    candidate = first.source_candidates[0]
    values, masks = list(candidate.features), list(candidate.missing_mask)
    i = schema.source_feature_names.index('face_center_x'); values[i] = .61; masks[i] = False
    known = replace(candidate, features=tuple(values), missing_mask=tuple(masks))
    # Different master time, same exact native pixel; known row must NOT be overwritten.
    tokens = (first, replace(first, master_time_seconds=.001, source_candidates=(known,)),
        replace(first, master_time_seconds=.01, source_candidates=(replace(candidate, available=False),)))
    old = StageEditFeatureSnapshot.build(replace(old.sequence, tokens=tokens), schema=schema,
        provenance=old.provenance, causal_sensing_binding=old.causal_sensing_binding)
    report['input_snapshot_sha256'] = [old.snapshot_sha256]
    assert list(missing_queries((old,), json.loads(raw))) == [('source-a', 30)]
    new = augment_missing_native_faces(old, face, report, raw)
    assert new.sequence.tokens[1:] == old.sequence.tokens[1:]


def test_completion_cannot_be_applied_twice():
    old, face, raw, report = fixture()
    new = augment_missing_native_faces(old, face, report, raw)
    with pytest.raises(ValueError, match='already applied'):
        augment_missing_native_faces(new, face, report, raw)
