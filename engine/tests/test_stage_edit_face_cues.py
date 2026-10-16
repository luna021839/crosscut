from dataclasses import replace

import numpy as np
import pytest
import json
from pathlib import Path

from test_stage_edit_performer_refresh import fixture
from crosscut.native_face_descriptor import NativeFaceDescriptorSpace, ARCFACE_MODEL_NAME, ARCFACE_R50_SHA256
from crosscut.stage_edit_face_cues import FaceCue, FaceCueTable, augment_snapshot, schema_with_face_cues, FEATURE_NAMES
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA, StageEditTransformerConfig, tensorize_stage_edit_batch
from materialize_stageedit_face_cues import build_table


def inputs():
    base, _, _ = fixture()
    space = NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME, 512, ARCFACE_R50_SHA256, '4'*64)
    cue = FaceCue('source-a', 0., 1., 30, (.1, .2, .3, .4, .5, -.1))
    table = FaceCueTable(base.snapshot_sha256, (('source-a', '5'*64),), '6'*64, space, (cue,))
    return base, table


def test_extend_only_six_columns_and_keep_legacy_schema_and_missingness():
    base, table = inputs()
    old_payload = base.to_dict()
    schema = schema_with_face_cues()
    new = augment_snapshot(base, table)
    assert base.to_dict() == old_payload
    assert schema.source_feature_names[:-6] == DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
    assert schema.source_feature_names[-6:] == FEATURE_NAMES
    assert schema_with_face_cues(schema) is schema
    a, b = base.sequence.tokens[0], new.sequence.tokens[0]
    assert replace(b, source_candidates=a.source_candidates) == a
    ac, bc = a.source_candidates[0], b.source_candidates[0]
    assert replace(bc, features=ac.features, missing_mask=ac.missing_mask) == ac
    assert bc.features == ac.features + table.records[0].member_cosines
    assert bc.missing_mask == ac.missing_mask + (False,)*6
    assert new.causal_sensing_binding == base.causal_sensing_binding
    assert dict(new.provenance.upstream_sha256)['face_member_cues'] == table.content_sha256
    assert StageEditFeatureSnapshot.from_dict(new.to_dict(), schema=schema) == new
    with pytest.raises(ValueError):
        StageEditFeatureSnapshot.from_dict(new.to_dict())


def test_no_face_is_missing_not_an_equal_member_vote_or_forced_unknown_label():
    base, table = inputs()
    new = augment_snapshot(base, replace(table, records=()))
    c = new.sequence.tokens[0].source_candidates[0]
    assert c.features[-6:] == (0.,)*6
    assert c.missing_mask[-6:] == (True,)*6


def test_table_roundtrip_retains_negative_scores_and_no_query_vectors():
    _, table = inputs()
    assert FaceCueTable.from_dict(table.to_dict()) == table
    assert len(table.to_dict()['records'][0]['member_cosines']) == 6
    bad = table.to_dict()
    bad['member_order'].reverse()
    with pytest.raises(ValueError):
        FaceCueTable.from_dict(bad)


def test_wrong_snapshot_source_time_or_duplicate_cue_rejected():
    base, table = inputs()
    with pytest.raises(ValueError, match='different target-free snapshot'):
        augment_snapshot(base, replace(table, base_snapshot_sha256='9'*64))
    with pytest.raises(ValueError, match='source domain'):
        augment_snapshot(base, replace(table, source_media_sha256=table.source_media_sha256 + (('source-b', '8'*64),)))
    with pytest.raises(ValueError, match='exact candidate'):
        augment_snapshot(base, replace(table, records=(replace(table.records[0], master_seconds=.001),)))
    with pytest.raises(ValueError, match='repeated'):
        replace(table, records=table.records*2)
    with pytest.raises(ValueError):
        augment_snapshot(augment_snapshot(base, table), table)


@pytest.mark.parametrize('scores', [(1.,)*5, (float('nan'),)*6, (1.01,)*6, (True,)*6])
def test_invalid_similarity_profile_not_clamped_into_a_plausible_input(scores):
    with pytest.raises(ValueError):
        FaceCue('a', 0., 0., 0, scores)


def test_real_tensorization_consumes_cosines_and_missing_bits_without_changing_source_gates():
    base, table = inputs()
    new = augment_snapshot(base, table)
    schema = schema_with_face_cues()
    config = StageEditTransformerConfig.from_schema(schema, model_dimension=24, num_heads=2,
        num_layers=1, feed_forward_dimension=48, max_sequence_length=8, max_sources=2)
    batch = tensorize_stage_edit_batch((new.sequence,), schema, config)
    values = np.asarray(batch.source_features)[0, 0, 0]
    assert len(values) == schema.encoded_source_dimension
    # The tensor format concatenates all values and then all missing flags.
    assert np.allclose(values[schema.raw_source_dimension-6:schema.raw_source_dimension], table.records[0].member_cosines)
    assert np.array_equal(values[-6:], np.zeros(6))


def test_actual_inference_materializer_keeps_extended_inputs_and_serialized_evidence(tmp_path, monkeypatch):
    import crosscut.stage_edit_inference_snapshot as inference
    import test_stage_edit_inference_snapshot as fixtures
    sources = (tmp_path/'a.mp4', tmp_path/'b.mp4')
    for source in sources:
        source.write_bytes(b'fixture')
    times = inference.build_stage_edit_inference_decision_times(timing_draft=fixtures._draft(),
        rhythm=fixtures._rhythm(), section_plan=fixtures._sections(), maximum_sequence_length=128)
    sensing = fixtures._causal_sensing(tmp_path/'sensing', source_ids=(
        'source-00-1111111111111111', 'source-01-2222222222222222'), decision_times=times)
    monkeypatch.setattr(inference, '_VideoSampler', fixtures._FakeSampler)
    monkeypatch.setattr(inference, 'visual_analyzer_fingerprint_sha256', lambda: 'a'*64)
    fixtures._FakeSampler.next_index = 0
    calls = []
    def provider(base):
        calls.append(base)
        ids = sorted(c.source_id for c in base.sequence.tokens[0].source_candidates)
        return FaceCueTable(base.snapshot_sha256, tuple(zip(ids, ('1'*64,'2'*64))), '6'*64,
            inputs()[1].descriptor_space, (FaceCue(ids[0], times[0], times[0], 0, (.1,)*6),))
    kwargs = dict(song_id='unseen song', sources=sources, alignment=fixtures._alignment(sources),
        alignment_hard_gates=(True,True), source_timewarps=(None,None), timing_draft=fixtures._draft(),
        rhythm=fixtures._rhythm(), section_plan=fixtures._sections(), output_directory=tmp_path/'outputs',
        feature_cache_directory=tmp_path/'cache', maximum_sequence_length=128, maximum_sources=8,
        causal_sensing=sensing, schema=schema_with_face_cues())
    with pytest.raises(inference.StageEditInferenceSnapshotError, match='requires.*face cue provider'):
        inference.materialize_stage_edit_inference_snapshot(**kwargs)
    assert fixtures._FakeSampler.next_index == 0
    result = inference.materialize_stage_edit_inference_snapshot(**kwargs, face_cue_provider=provider)
    assert len(calls) == 1
    assert result.snapshot.token_schema_fingerprint == schema_with_face_cues().fingerprint
    result.snapshot.require_production_causal_sensing(edit_mode='group', schema=schema_with_face_cues())
    manifest = json.loads(result.manifest_path.read_bytes())
    evidence = manifest['face_member_cues']
    table = FaceCueTable.from_dict(json.loads(Path(evidence['path']).read_bytes()))
    assert table.content_sha256 == evidence['content_sha256']
    assert result.snapshot.sequence.tokens[0].source_candidates[0].features[-6:] == (.1,)*6
    assert result.snapshot.sequence.tokens[0].source_candidates[1].missing_mask[-6:] == (True,)*6


def _actual_report_inputs():
    from crosscut.ive import IVEMember
    from crosscut.reference_evidence_builder import ReferenceSourceClock
    base, table = inputs()
    row = table.records[0]
    request = dict(sources=[dict(source_id=row.source_id, media_content_sha256='5'*64,
        clock=ReferenceSourceClock.scalar(1.).to_dict(), frame_rate=30.)])
    hint = dict(descriptor_space=table.descriptor_space.to_dict(), gallery_identity=table.gallery_identity,
        member=None, state='UNKNOWN', member_scores={member.value: score for member, score in
            zip(IVEMember, row.member_cosines)})
    report = dict(request_file_sha256='7'*64,
        graph_content_sha256=base.causal_sensing_binding.member_state_graph_content_sha256,
        current_source_contents_verified_against_request=True, raw_query_or_reference_vectors_serialized=False,
        rows=[dict(source_id=row.source_id, master_seconds=row.master_seconds, source_seconds=row.source_seconds,
            native_frame=row.native_frame, media_content_sha256='5'*64, state='OBSERVED', arcface=hint)])
    return base, request, report, table


def test_completed_report_keeps_all_six_scores_from_rejected_identity_decision():
    base, request, report, table = _actual_report_inputs()
    actual = build_table(base, report, request, request_file_sha256='7'*64)
    assert actual == table
    assert augment_snapshot(base, actual).sequence.tokens[0].source_candidates[0].missing_mask[-6:] == (False,)*6


@pytest.mark.parametrize('field,value', [('source_seconds', 1.01), ('native_frame', 31),
    ('media_content_sha256', '8'*64)])
def test_completed_report_cannot_borrow_a_nearby_frame_or_source(field, value):
    base, request, report, _ = _actual_report_inputs()
    report['rows'][0][field] = value
    with pytest.raises(ValueError, match='clock'):
        build_table(base, report, request, request_file_sha256='7'*64)
