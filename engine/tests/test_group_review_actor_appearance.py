"""Substituted nonhuman rows/embedding outputs, never real identity proof."""
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from crosscut.face_identity import SFACE_SHA256
from crosscut.group_performer_family_owner import digest
from crosscut.group_review_actor_appearance import observe_recorded_head_appearances,register_review_sface
from crosscut.native_face_descriptor import SFACE_MODEL_NAME
from engine.tests.test_registered_camera_queries import world


def fixture():
    domain,compiled=world();query=compiled.views[0].state
    source=domain.request_index(query.eligible_source_index);native=query.native_frame_indices[0]
    pixels=np.zeros((24,40,3),np.uint8);identity=(1,2,3,4,5)
    raw=dict(request_source_index=source,native_frame_index=native,source_file_identity=list(identity),
        source_clock_identity=digest(domain.render_frame_clocks[query.eligible_source_index].to_dict()),
        native_pixel_sha256=sha256(memoryview(pixels)).hexdigest(),heads=[dict(observation_id=0,
            box_xywh=[.3,.2,.2,.3],landmarks_xy=[[.35,.3],[.45,.3],[.4,.35],[.35,.4],[.45,.4]])])
    session=SimpleNamespace(domain=domain,closed=False,native_pixels=Mock(return_value=pixels),
        _assert_source_identity=Mock(return_value=identity),_assert_model_assets=Mock(),
        models={SFACE_MODEL_NAME:(None,None,SFACE_SHA256,None)},
        model_adapters={SFACE_MODEL_NAME:'c'*64},observe_native=Mock(return_value=(1.,)+(0.,)*127))
    return session,raw


def test_actual_feature_invocation_is_not_new_detection_or_identity_approval():
    session,raw=fixture();result=observe_recorded_head_appearances(session,raw)
    assert result['heads'][0]['embedding']==(1.,)+(0.,)*127
    assert result['head_detection_reexecuted'] is False and result['detector_input_is_recorded'] is True
    for key in ('named_member_identity_certified','persistent_actor_identity_certified',
            'motion_certified','candidate_admitted','quality_approved'):
        assert result[key] is False
    session.observe_native.assert_called_once()


def test_changed_raster_cannot_borrow_the_recorded_face_box():
    session,raw=fixture();raw['native_pixel_sha256']='f'*64
    with pytest.raises(ValueError,match='raster changed'):observe_recorded_head_appearances(session,raw)
    session.observe_native.assert_not_called()


def test_changed_clock_rejected_before_decode():
    session,raw=fixture();raw['source_clock_identity']='f'*64
    with pytest.raises(ValueError,match='clock'):observe_recorded_head_appearances(session,raw)
    session.native_pixels.assert_not_called();session.observe_native.assert_not_called()


def test_geometry_outside_model_input_stays_unknown_not_clamped():
    session,raw=fixture();raw['heads'][0]['box_xywh'][0]=-.1
    result=observe_recorded_head_appearances(session,raw)
    assert result['heads'][0]['embedding'] is None
    session.observe_native.assert_not_called()


def test_closed_owner_is_not_a_saved_live_embedding_receipt():
    session,raw=fixture();session.closed=True
    with pytest.raises(ValueError,match='live'):observe_recorded_head_appearances(session,raw)
    session.native_pixels.assert_not_called()


def test_unpinned_asset_is_not_loaded(tmp_path):
    path=tmp_path/'nonmodel.onnx';path.write_bytes(b'not a pinned model')
    session=SimpleNamespace(_identity=lambda p:(1,2,p.stat().st_size,4,5),
        budget=SimpleNamespace(request_work_phase=None))
    with pytest.raises(ValueError,match='pinned'):register_review_sface(session,path)
