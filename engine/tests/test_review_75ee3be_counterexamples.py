"""Reviewer regressions: synthetic rasters/ORT schema only, no recognition/quality claim."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import sys
from unittest.mock import Mock, patch
import pytest

from crosscut.group_joint_focal_frontier import (
    NativeFocalLinkGraph, NativeFaceAppearance, match_focal_trajectory, _same_source_links,
    resolve_inherited_focal_reference,
)
from crosscut.native_face_descriptor import NativeArcFaceR50
from crosscut.group_joint_observations import GroupJointObservationSession, _file_identity
from engine.tests import test_group_joint_focal_frontier as fixtures
from engine.tests.test_group_joint_focal_frontier import appearance


def values(n):
    return tuple(NativeFaceAppearance(i, appearance(axis=i).embedding, ()) for i in range(n))


def test_five_sparse_persistent_faces_are_not_125_feasible_paths():
    v = values(5)
    pair = Mock(side_effect=lambda layer,a,b: a.face_observation_id == b.face_observation_id)
    selected, result = match_focal_trajectory((v[0],)*3,(v,)*3,
        adjacent_available=pair, path_available=lambda path: True)
    assert selected == (v[0],)*3
    assert len(result['hypotheses']) == 5
    assert pair.call_count == 50
    assert result['identity_certified'] is False


def test_dense_graph_still_returns_unknown_instead_of_truncating_to_best_64():
    v = values(5)
    selected,result = match_focal_trajectory((v[0],)*3,(v,)*3,
        adjacent_available=lambda *args: True,path_available=lambda path: True)
    assert selected is None and result['hypotheses'] == []
    assert result['hypothesis_count_lower_bound'] == 65


def test_pair_cap_checked_before_any_lookup():
    v = values(9); pair=Mock(return_value=True)
    selected,result=match_focal_trajectory((v[0],)*3,(v,)*3,
        adjacent_available=pair,path_available=lambda path: True)
    assert selected is None and result['pair_query_count'] == 162
    pair.assert_not_called()


def test_inherited_seed_filter_is_applied_before_feasible_path_cap():
    v = values(5)
    selected,result=match_focal_trajectory((v[0],)*3,(v,)*3,
        adjacent_available=lambda *args: True,
        path_available=lambda path: path[0].face_observation_id == 0)
    assert selected is not None and len(result['hypotheses']) == 25
    assert all(h['face_observation_ids'][0] == 0 for h in result['hypotheses'])


def test_same_native_alias_never_links_another_face_or_measures_velocity():
    session,rows,vs=fixtures.FocalFrontierTests().graph_fixture((100,50))
    graph=NativeFocalLinkGraph(session)
    graph.add_pair(rows[0],rows[0],vs[0],vs[0])
    graph.add_pair(rows[0],rows[0],vs[0],vs[0])  # same pair in [0,0,0]
    assert graph.path_available((rows[0],)*3,(vs[0][0],)*3)
    assert not graph.path_available((rows[0],)*2,(vs[0][0],vs[0][1]))
    assert graph.to_dict()['unique_pair_observations'] == 0
    assert graph.to_dict()['identity_alias_links'] == 4
    assert graph.to_dict()['aliases_are_velocity_samples'] is False
    assert _same_source_links(session,(rows[0],)*3,(vs[0][0],)*3)
    session.native_pixels.assert_not_called()


def test_same_frame_number_with_changed_raw_hash_is_not_identity_alias():
    session,rows,vs=fixtures.FocalFrontierTests().graph_fixture((100,50))
    changed=replace(rows[0],native_pixel_sha256='a'*64)
    graph=NativeFocalLinkGraph(session)
    graph.add_pair(rows[0],changed,vs[0],vs[0])
    assert not graph.path_available((rows[0],changed),(vs[0][0],)*2)
    session.native_pixels.assert_not_called()


def test_same_raw_row_cannot_change_appearance_without_new_measurement():
    session,rows,vs=fixtures.FocalFrontierTests().graph_fixture((100,50))
    graph=NativeFocalLinkGraph(session)
    with pytest.raises(ValueError,match='inconsistent'):
        graph.add_pair(rows[0],rows[0],vs[0],tuple(reversed(vs[0])))


def test_repeated_alias_registration_and_query_cannot_reuse_changed_descriptor():
    session,rows,vs=fixtures.FocalFrontierTests().graph_fixture((100,50))
    graph=NativeFocalLinkGraph(session)
    graph.add_pair(rows[0],rows[0],vs[0],vs[0])
    changed=(replace(vs[0][0],embedding=vs[0][1].embedding),*vs[0][1:])
    with pytest.raises(ValueError,match='alias changed'):
        graph.add_pair(rows[0],rows[0],changed,changed)
    assert not graph.path_available((rows[0],)*2,(changed[0],)*2)
    assert graph.path_available((rows[0],)*2,(vs[0][0],)*2)
    session.native_pixels.assert_not_called()


def test_cut_is_still_not_an_alias_even_when_face_ids_match():
    session,rows,vs=fixtures.FocalFrontierTests().graph_fixture((100,50),cuts=(1/30.,))
    graph=NativeFocalLinkGraph(session)
    graph.add_pair(rows[0],rows[1],vs[0],vs[1])
    assert not graph.path_available(rows[:2],(vs[0][0],vs[1][0]))
    session.native_pixels.assert_not_called()


def test_inherited_reference_can_use_exact_alias_and_sparse_graph():
    f=fixtures.FocalFrontierTests();session,rows,vs=f.graph_fixture((100,50))
    graph=NativeFocalLinkGraph(session); graph.add_pair(rows[0],rows[0],vs[0],vs[0])
    chosen=resolve_inherited_focal_reference(session,(rows[0],)*3,(vs[0],)*3,
        inherited_reference=f.inherited(rows[0],face=0),
        pixel_path_available=lambda p: graph.path_available((rows[0],)*3,p),
        pixel_pair_available=lambda layer,a,b: graph.path_available((rows[0],)*2,(a,b)))
    assert chosen == (vs[0][0],)*3


def fake_ort(factory):
    return SimpleNamespace(SessionOptions=lambda: SimpleNamespace(),
        InferenceSession=factory, __version__='synthetic-test-no-inference')


def schema():
    return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='input',type='tensor(float)',shape=[None,3,112,112])],
        get_outputs=lambda: [SimpleNamespace(name='output',shape=[None,512])])


def test_arcface_path_swap_cannot_replace_immutable_loaded_bytes():
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'fake.onnx';path.write_bytes(b'original')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        def factory(model_bytes,**kwargs):
            assert model_bytes == b'original'
            path.write_bytes(b'replaced')  # same byte length; metadata must still change
            return schema()
        with patch.dict(sys.modules,{'onnxruntime':fake_ort(factory)}), \
             patch('crosscut.native_face_descriptor.ARCFACE_R50_SHA256',digest):
            model = NativeArcFaceR50(path)
            assert model.asset_sha256 == digest
            assert hashlib.sha256(path.read_bytes()).hexdigest() != model.asset_sha256


def test_model_registration_cannot_bind_later_file_to_loaded_observer():
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'fake.onnx';path.write_bytes(b'original')
        before=_file_identity(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b'replaced')
        session=object.__new__(GroupJointObservationSession)
        session.budget=SimpleNamespace(check=lambda: None)
        session.models={};session.model_adapters={};session.model_asset_dependencies={};session.model_adapter_preimages={}
        with pytest.raises(ValueError,match='differs from the loaded observer'):
            session.register_model('r50',path,lambda *a: None,
                expected_file_identity=before,expected_asset_sha256=digest)
        assert session.models == {} and session.model_adapters == {}


def test_arcface_unchanged_schema_only_loader_still_accepts():
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'fake.onnx';path.write_bytes(b'original')
        with patch.dict(sys.modules,{'onnxruntime':fake_ort(lambda *a,**k: schema())}), \
             patch('crosscut.native_face_descriptor.ARCFACE_R50_SHA256',hashlib.sha256(path.read_bytes()).hexdigest()):
            model=NativeArcFaceR50(path)
            assert model.asset_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_frontier_streams_both_pairs_before_three_anchor_cartesian_growth():
    from crosscut.group_joint_focal_frontier import observe_synchronized_focal_frontier
    f=fixtures.FocalFrontierTests();session,rows,vs=f.graph_fixture((100,50))
    rows=tuple(replace(r,heads=(*r.heads,replace(r.heads[0],observation_id=4))) for r in rows)
    session.retain_native_pixels=Mock()
    graph=Mock(path_available=Mock(return_value=False))
    with patch('crosscut.group_joint_focal_frontier.NativeFocalLinkGraph',return_value=graph), \
         patch('crosscut.group_joint_focal_frontier.observe_native_group',side_effect=rows), \
         patch('crosscut.group_joint_focal_frontier.observe_native_face_appearances',return_value=values(5)), \
         patch('crosscut.group_joint_focal_frontier._native_for_canonical',side_effect=(0,1,2)):
        result=observe_synchronized_focal_frontier(session,canonical_times=(0.,.03,.06),
            reference_source=rows[0].request_source_index,face_roi_only=True,
            inherited_reference=f.inherited(rows[0],face=0))
    assert result['production_admitted'] is False
    assert graph.add_pair.call_count == 2
    assert [len(call.args[2])*len(call.args[3]) for call in graph.add_pair.call_args_list] == [25,25]


@pytest.mark.parametrize('replacement_load',[1,2])
def test_rotated_detector_asset_changed_in_loader_never_registers(replacement_load):
    from crosscut.group_native_head_orientation import register_native_head_orientation_models
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'fake.onnx';path.write_bytes(b'original')
        session=SimpleNamespace(budget=SimpleNamespace(check=lambda:None),register_model=Mock())
        loads=[]
        def factory(*a,**k):
            loads.append(1)
            if len(loads)==replacement_load:
                path.write_bytes(b'replaced')
            return Mock()
        with patch('crosscut.group_native_head_orientation.YUNET_EXPECTED_SHA256',hashlib.sha256(path.read_bytes()).hexdigest()), \
             patch('crosscut.group_native_head_orientation.cv2.FaceDetectorYN.create',side_effect=factory):
            with pytest.raises(ValueError,match='changed during model loading'):
                register_native_head_orientation_models(session,path,angles=(-30.,30.))
        session.register_model.assert_not_called()
