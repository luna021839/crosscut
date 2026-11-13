"""One normalized blob, distinct camera references and unchanged observations.

Explicit nonhuman model inputs; no human inference or editorial approval.
The production frame cache, frozen evidence and transaction publisher are used.
"""
from copy import deepcopy
from dataclasses import replace
import json
from time import monotonic
from unittest.mock import patch

import pytest

from crosscut import group_performer_family_owner as family
from crosscut import group_scene_native_storage as wire
from crosscut.group_joint_primitive_journal import encoded
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from engine.tests.test_scene_native_wire_storage import capture
from engine.tests.test_scene_focal_pose_motion import scene_world
from engine.tests import test_performer_native_wire_publication as publication


def blob_api():
    from crosscut import group_scene_native_blob_storage as blobs
    return blobs


def repeated_evidence():
    sensors, state, _, rows, _ = scene_world()
    raw = rows[0].to_dict()
    return sensors, state, rows, dict(native_rows=[raw],
        camera_views=[dict(camera_index=i, native_observation=raw,
            required_parts=((0, 'left_arm'),), quality_approved=False) for i in range(20)],
        unrelated=dict(contract='explicit_nonhuman_unchanged_context', number=3))


def test_shared_blob_store_reconstructs_exactly_one_native_payload_for_many_cameras():
    blobs = blob_api()
    _, _, _, evidence = repeated_evidence()
    stored = blobs.pack_shared_native_evidence(evidence)
    table = stored[blobs.STORE_KEY]
    assert len(table['views']) == 1
    assert all(row['native_observation']['contract'] == blobs.REFERENCE_CONTRACT
        for row in stored['camera_views'])
    assert family.plain(stored) == family.plain(evidence)
    assert family.digest(stored) == family.digest(evidence)
    assert len(encoded(stored)) < len(encoded(wire.pack_native_evidence(evidence)))/3
    assert table['controller_owned'] is False
    assert table['raw_nn_or_adapter_returns_retained'] is False
    assert table['pre_receipt_history_certified'] is table['quality_approved'] is False
    assert stored['unrelated'] == evidence['unrelated']
    assert stored['camera_views'][0]['required_parts'] == ((0, 'left_arm'),)


@pytest.mark.parametrize('mutation', ['drop_blob', 'wrong_ref', 'wrong_key', 'payload',
    'unreferenced_blob', 'authority', 'reference_authority', 'unknown_field'])
def test_received_blob_table_cannot_borrow_erase_or_promote_an_observation(mutation):
    blobs = blob_api()
    _, _, rows, evidence = repeated_evidence()
    stored = deepcopy(blobs.pack_shared_native_evidence(evidence))
    table = stored[blobs.STORE_KEY]
    ref = next(iter(table['views']))
    if mutation == 'drop_blob': table['views'].pop(ref)
    elif mutation == 'wrong_ref': stored['camera_views'][0]['native_observation']['observation_sha256'] = 'b'*64
    elif mutation == 'wrong_key': table['views']['b'*64] = table['views'].pop(ref)
    elif mutation == 'payload': table['views'][ref]['lossless_json']['decoded_byte_count'] += 1
    elif mutation == 'unreferenced_blob':
        packet = wire.pack_native_observation(rows[1].to_dict())
        table['views'][packet['observation_sha256']] = packet
    elif mutation == 'authority': table['controller_owned'] = True
    elif mutation == 'reference_authority': stored['camera_views'][0]['native_observation']['quality_approved'] = True
    else: table['self_certified_original_actor'] = True
    with pytest.raises(ValueError): family.plain(stored)


def test_reference_without_its_complete_owned_table_is_not_a_native_observation():
    blobs = blob_api()
    _, _, rows, _ = repeated_evidence()
    reference = blobs.native_reference(wire.pack_native_observation(rows[0].to_dict()))
    with pytest.raises(ValueError): family.plain(dict(native_observation=reference))


def test_nested_shared_stores_repack_to_one_canonical_table_without_sensor_work():
    blobs = blob_api()
    _, _, _, evidence = repeated_evidence()
    one = blobs.pack_shared_native_evidence(evidence)
    combined = blobs.pack_shared_native_evidence(dict(first=one, second=one))
    assert len(combined[blobs.STORE_KEY]['views']) == 1
    assert blobs.STORE_KEY not in combined['first']
    assert family.plain(combined) == dict(first=family.plain(evidence), second=family.plain(evidence))
    assert blobs.pack_shared_native_evidence(combined) == combined


def test_two_cameras_share_owned_wire_bytes_but_keep_projection_and_attempt_charges():
    blobs = blob_api()
    sensors, state, _, rows, _ = scene_world()
    budget = sensors.session.budget
    capture(budget, rows[0])
    packet = budget.normalized_native_views.views[rows[0].sha256]
    before = dict(budget.counts)
    first = sensors.frame(state, 0)
    after = dict(budget.counts)
    from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
    camera = replace(state, segments=(replace(state.segments[0], editorial_transform=
        EditorialTransformTrack((EditorialTransformKeyframe(0., 1.02),))),))
    second = sensors.frame(camera, 0)
    assert isinstance(first['native_observation'], blobs.SharedNativeObservationView)
    assert first['native_observation']._packet is packet
    assert second['native_observation']._packet is packet
    assert first['projected_face_roi_xywh'] != second['projected_face_roi_xywh']
    assert family.digest(first['native_observation']) == rows[0].sha256
    assert budget.counts['model_calls'] == before['model_calls']
    assert len(budget.normalized_native_views.attempts) == 1
    for row, increment in ((first, after['pose_evidence_bytes']-before['pose_evidence_bytes']),
            (second, budget.counts['pose_evidence_bytes']-after['pose_evidence_bytes'])):
        accounted = wire.pack_native_evidence(row)
        accounted['native_observation'] = blobs.native_reference(packet)
        assert increment == len(encoded(accounted))
        assert increment < len(encoded(wire.pack_native_evidence(row)))
    stored = blobs.pack_shared_native_evidence(dict(rows=[first, second], archive=sensors.archive()))
    # The fixture's discovery already retained three OTHER raw anchors. They
    # must not disappear just because the two new cameras share the first one.
    expected = {family.digest(row) for row in sensors.archive()['native_rows']}
    assert set(stored[blobs.STORE_KEY]['views']) == {rows[0].sha256}
    assert {family.digest(row) for row in family.plain(stored)['archive']['native_rows']} == expected
    wire.validate_normalized_view_records(family.plain(stored)['archive']['normalized_native_view_records'],
        {rows[0].sha256: rows[0].to_dict()})


def test_frame_without_a_preexisting_native_commit_keeps_full_wire_charge():
    blobs = blob_api()
    sensors, state, _, _, _ = scene_world()
    before = sensors.session.budget.counts['pose_evidence_bytes']
    row = sensors.frame(state, 0)
    assert not isinstance(row['native_observation'], blobs.SharedNativeObservationView)
    assert sensors.session.budget.counts['pose_evidence_bytes']-before == len(encoded(wire.pack_native_evidence(row)))


def test_same_bytes_do_not_refund_fresh_model_execution_or_drop_terminal_records():
    blobs = blob_api()
    _, _, _, rows, _ = scene_world()
    budget = JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    def execute():
        budget.spend('model_calls', model='explicit_nonhuman_model_fixture')
        return rows[0]
    capture(budget, rows[0], execute)
    capture(budget, rows[0], execute)
    record = budget.normalized_native_views.to_dict()
    stored = blobs.pack_shared_native_evidence(dict(records=record, views=[rows[0].to_dict()]*2))
    assert budget.counts['model_calls'] == 2
    assert len(family.plain(stored)['records']['attempts']) == 2
    assert len(stored[blobs.STORE_KEY]['views']) == 1


def test_frozen_motion_shares_committed_native_bytes_but_public_json_keeps_exact_schema():
    blobs = blob_api()
    from crosscut.editorial_motion import EditorialMotionPlan
    sensors, state, _, rows, _ = scene_world()
    capture(sensors.session.budget, rows[0])
    refresh = dict(rows=[sensors.frame(state, 0)], scene_discovery=sensors.archive())
    motion = EditorialMotionPlan((), 0, 0, 0, 0, 0, source_camera_refresh=refresh)
    assert blobs.STORE_KEY in motion.source_camera_refresh
    assert motion.to_dict()['source_camera_refresh'] == family.plain(refresh)


def test_real_partial_publication_stores_blob_once_and_reconstructs_all_cameras(tmp_path):
    blobs = blob_api()
    raw, controller, context = publication.fixture()
    packet = wire.pack_native_observation(raw)
    view = blobs.SharedNativeObservationView(packet)
    evidence = dict(outcome='BUDGET_PARTIAL', raw_prefix_completeness_certified=False,
        rows=[dict(native_observation=view, camera=i) for i in range(20)], quality_approved=False)
    path = tmp_path/'shared-partial.json'
    rejection = family.PerformerFamilyRejection('EXPLICIT_SHARED_NONHUMAN_PARTIAL', evidence)
    with patch.object(family, 'freeze_context', return_value=context), \
            patch.object(family, 'propose_performer_family', return_value=rejection), \
            pytest.raises(ValueError, match='EXPLICIT_SHARED_NONHUMAN_PARTIAL'):
        publication.attempt(controller, path)
    saved = json.loads(path.read_text())
    assert len(saved[blobs.STORE_KEY]['views']) == 1
    assert saved['producer_rejection_evidence']['rows'][0]['native_observation']['contract'] == blobs.REFERENCE_CONTRACT
    assert family.plain(saved)['producer_rejection_evidence'] == family.plain(evidence)
    assert saved['status'] == 'rejected_incumbent_preserved'
    assert saved['quality_approved'] is False


def test_one_use_native_observation_retains_existing_wire_without_table_overhead():
    blobs = blob_api()
    _, _, rows, _ = repeated_evidence()
    evidence = dict(native_observation=rows[0].to_dict(), unrelated=3)
    stored = blobs.pack_shared_native_evidence(evidence)
    assert blobs.STORE_KEY not in stored
    assert stored == wire.pack_native_evidence(evidence)
    assert family.plain(stored) == family.plain(evidence)


def test_unshared_native_context_is_kept_inline_not_dropped_by_repeated_camera_blob():
    blobs = blob_api()
    _, _, rows, evidence = repeated_evidence()
    evidence['other_native'] = rows[1].to_dict()
    stored = blobs.pack_shared_native_evidence(evidence)
    assert len(stored[blobs.STORE_KEY]['views']) == 1
    assert stored['other_native']['contract'] == wire.CONTRACT
    assert family.plain(stored) == family.plain(evidence)


def test_expanded_camera_rows_do_not_gain_mutable_alias_authority_over_each_other():
    blobs = blob_api()
    _, _, _, evidence = repeated_evidence()
    saved = blobs.pack_shared_native_evidence(evidence)
    expanded = family.plain(saved)
    original = deepcopy(expanded['camera_views'][1]['native_observation'])
    expanded['camera_views'][0]['native_observation']['quality_approved'] = True
    assert expanded['camera_views'][1]['native_observation'] == original
    assert family.plain(saved)['camera_views'][0]['native_observation']['quality_approved'] is False


def test_existing_archive_byte_cap_still_blocks_shared_publication_without_losing_predecessor(tmp_path):
    blobs = blob_api()
    raw, controller, context = publication.fixture()
    view = blobs.SharedNativeObservationView(wire.pack_native_observation(raw))
    rejection = family.PerformerFamilyRejection('EXPLICIT_SHARED_CAP', dict(rows=[dict(native_observation=view)]))
    path = tmp_path/'shared-cap.json'
    with patch.object(family, 'freeze_context', return_value=context), \
            patch.object(family, 'propose_performer_family', return_value=rejection), \
            patch.object(family, 'MAX_ARCHIVE_BYTES', 512), \
            pytest.raises(ValueError, match='archive exceeds explicit byte budget'):
        publication.attempt(controller, path)
    saved = json.loads(path.read_text())
    assert saved['status'] == 'preparing'
    assert 'producer_rejection_evidence' not in saved
    assert saved['quality_approved'] is False
