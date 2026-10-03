"""Count the real cache/adapter path with explicit chart pixels and mock NN.

No performer or model quality is inferred from these fixture outputs.
"""
from dataclasses import replace
from time import monotonic
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from crosscut.body_pose_2d import BodyPose2DAnalyzer
from crosscut.group_joint_native_observer import NativeGroupHead,observe_native_group
from crosscut.group_joint_observations import GroupJointObservationSession,_file_identity
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_native_head_orientation import OrientedNativeHeadAlternative,native_head_analysis_spec
from tests import test_body_pose_2d as pose_fixture
from tests import test_group_joint_temporal_pose as clock_fixture


def session_world(tmp_path):
    source=tmp_path/'EXPLICIT_CHART_IDENTITY_NOT_A_VIDEO.bin'
    source.write_bytes(b'Explicit chart bytes, native raster is injected, not decoded.')
    asset=tmp_path/'EXPLICIT_MOCK_MODEL_NOT_REAL_WEIGHTS.bin'
    asset.write_bytes(b'The registered forward is a mock, not a model quality test.')
    domain=replace(clock_fixture.NativeTemporalPoseTests().domain(),request_sources=(source,source))
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10)
    session=GroupJointObservationSession(domain=domain,budget=budget)
    # This is an explicitly injected chart raster/clock, not a fake video
    # offered to a production decoder. Subsequent source-identity checks stay
    # real, and actual model/cache accounting remains the unmodified session.
    session.identities={index:_file_identity(source) for index in (0,1)}
    pixels=np.random.default_rng(2).integers(0,255,(200,200,3),np.uint8)
    session.native_pixels=lambda *args:pixels
    analyzer=BodyPose2DAnalyzer();analyzer._net=Mock()
    tensors=[]
    analyzer._net.setInput.side_effect=lambda tensor:tensors.append(tensor.copy())
    analyzer._net.forward.return_value=pose_fixture.simcc()
    def body(image,*,bbox_xywh,face_bbox_xywh=None):
        return analyzer.describe(image,bbox_xywh=tuple(bbox_xywh),
            face_bbox_xywh=None if face_bbox_xywh is None else tuple(face_bbox_xywh))
    session.register_model('rtmpose_body7',asset,body,adapter_fingerprint=analyzer.fingerprint())
    session.register_model('nanodet_people',asset,lambda image:np.asarray([[50.,20.,150.,180.,.95]]),
        adapter_fingerprint={'test':'explicit_detector_chart'})
    original=NativeGroupHead(0,(.45,.45,.1,.1),((.5,.5),)*5,.95)
    session.register_model('yunet_head_faces',asset,lambda image:(original,),
        adapter_fingerprint={'test':'explicit_original_chart_head'})
    roll=-30.;name,_=native_head_analysis_spec(roll)
    head=replace(original,box_xywh=(.35,.35,.1,.1),landmarks_xy=((.4,.4),)*5)
    affine=cv2.getRotationMatrix2D((99.5,99.5),roll,1.)
    alternative=OrientedNativeHeadAlternative(name,roll,(200,200),
        tuple(tuple(float(v) for v in row) for row in affine),head,
        ((.35,.35),(.45,.35),(.45,.45),(.35,.45)))
    session.register_model(name,asset,lambda image:(alternative,),
        adapter_fingerprint={'test':'explicit_alternative_chart_head','roll':roll})
    return session,analyzer,tensors,pixels,asset


@pytest.mark.parametrize('order',[(None,-30.),(-30.,None)])
def test_native_body_forward_shared_but_face_consistency_recomputed_for_each_variant(tmp_path,order):
    session,analyzer,tensors,_,_=session_world(tmp_path)
    try:
        rows={roll:observe_native_group(session,request_source_index=0,native_frame_index=0,
            head_analysis_roll=roll) for roll in order}
        assert rows[None].people[0].face_anchor_consistent is True
        assert rows[-30.].people[0].face_anchor_consistent is False
        assert rows[None].people[0].raw_pose_xyc==rows[-30.].people[0].raw_pose_xyc
        assert analyzer._net.forward.call_count==1
        assert len(tensors)==1
        assert session.budget.model_calls['rtmpose_body7']==1
        assert session.diagnostics()['reused_model_observations']>=2
    finally:session.close()


@pytest.mark.parametrize('change',['body','frame','source','adapter'])
def test_different_actual_native_body_inputs_do_not_borrow_shared_forward(tmp_path,change):
    session,analyzer,_,_,_=session_world(tmp_path)
    try:
        session.observe_native(0,0,model='rtmpose_body7',arguments={'bbox_xywh':list(pose_fixture.BOX)})
        source,native,box=0,0,list(pose_fixture.BOX)
        if change=='body':box[0]=.2
        elif change=='frame':native=1
        elif change=='source':source=1
        else:session.model_adapters['rtmpose_body7']='f'*64
        session.observe_native(source,native,model='rtmpose_body7',arguments={'bbox_xywh':box})
        assert analyzer._net.forward.call_count==2
        assert session.budget.model_calls['rtmpose_body7']==2
    finally:session.close()


def test_model_asset_change_rejects_before_any_cached_pose_is_returned(tmp_path):
    session,analyzer,_,_,asset=session_world(tmp_path)
    try:
        observe_native_group(session,request_source_index=0,native_frame_index=0)
        asset.write_bytes(b'Changed disposable fixture asset only.')
        with pytest.raises(ValueError,match='model identity changed'):
            observe_native_group(session,request_source_index=0,native_frame_index=0)
        assert analyzer._net.forward.call_count==1
    finally:session.close()


def test_pure_face_view_does_not_mutate_cached_model_coordinates():
    from crosscut.body_pose_2d import pose_face_anchor_consistency
    analyzer=BodyPose2DAnalyzer();analyzer._net=Mock()
    analyzer._net.forward.return_value=pose_fixture.simcc()
    raw=analyzer.describe(pose_fixture.IMAGE,bbox_xywh=pose_fixture.BOX)
    assert raw.face_anchor_consistent is None
    assert pose_face_anchor_consistency(raw,(.45,.45,.1,.1)) is True
    assert pose_face_anchor_consistency(raw,(.35,.35,.1,.1)) is False
    assert raw.face_anchor_consistent is None
    assert analyzer._net.forward.call_count==1
    with pytest.raises(ValueError):pose_face_anchor_consistency(raw,(-.1,.4,.1,.1))
