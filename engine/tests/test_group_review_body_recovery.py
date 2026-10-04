from copy import deepcopy
from dataclasses import replace
import pytest

from crosscut.group_review_body_recovery import (freeze_review_body_window,
    apply_review_body_receipt,apply_review_primary_joint_observation,
    head_fingerprint,native_binding,consume_head_conditioned_review_support)
from crosscut.body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256
from crosscut.group_performer_family_owner import digest
from crosscut.group_review_camera_repair import FramingDescriptor


def raw(n=10):
    return dict(request_source_index=2,native_frame_index=n,source_clock_identity='clock',
        source_file_identity=[1,2,3,4,5],native_pixel_sha256=str(n),frame_size_wh=[1920,1080],
        heads=[dict(observation_id=7,box_xywh=[.4,.2,.1,.2],confidence=.9,
            landmarks_xy=[[.41,.25],[.48,.25],[.45,.3],[.42,.35],[.47,.35]],
            extent_basis='face_roi_not_whole_head',candidate_body_observation_ids=[0,1])],
        people=[dict(observation_id=i,box_xywh=[.2+i*.05,.1,.5,.9],
            face_anchor_consistent=True,pose_reason='observed',unique_head_observation_id=None,
            part_states=[['shoulder_anchor','ambiguous']]) for i in range(2)])


def fixture():
    raws=[raw(10),raw(11),raw(12)]
    frozen=freeze_review_body_window(raws,native_camera_unit=0)
    points=[[.4,.5,.8],[.6,.5,.8],[.3,.6,.8],[.7,.6,.8],[.2,.4,.8],[.8,.4,.8],
        [.4,.9,.4],[.6,.9,.4]]
    raw_points=[[.45,.3,.8] for _ in range(17)]
    for i,j in enumerate((5,6,7,8,9,10,11,12)):raw_points[j]=points[i]
    receipt=dict(contract='actual_fixed_roi_body_review_receipt_v1',window_identity=frozen.identity,
        model_content_sha256=RTMPOSE_BODY7_MODEL_SHA256,binding=native_binding(raws[1]),
        raw_parent_sha256=digest(raws[1]),
        head_fingerprint=head_fingerprint(raws[1]['heads'][0]),pose=dict(reason='observed',
            face_anchor_consistent=True,bbox_xywh=frozen.payload['requests'][1]['roi'],
            upper_body_xyc=points,raw_keypoints_xyc=raw_points))
    d=FramingDescriptor((.45,.3),.2,.1,None,7,False)
    view=dict(row=dict(output_tick=11,native_observation=raws[1]),descriptor=d)
    return frozen,view,receipt


def test_freeze_keeps_all_prior_duplicate_boxes_and_does_not_claim_owner():
    originals=[raw(10),raw(11)];saved=deepcopy(originals)
    result=freeze_review_body_window(originals,native_camera_unit=3)
    assert len(result.payload['prior_body_roi_alternatives'])==4
    assert not result.payload['body_ownership_certified']
    assert not result.payload['roi_selection_inspected_new_pose'] and originals==saved


def test_actual_alternate_pose_can_restore_shape_without_rewriting_native_owner():
    frozen,view,receipt=fixture();saved=deepcopy(view['row'])
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert len(result['descriptor'].pose_shape)==6 and result['row']==saved
    assert not result['descriptor'].owner_known and not result['auxiliary_body_is_production_ownership']


def test_low_confidence_wrist_is_not_completed_or_hidden():
    frozen,view,receipt=fixture();receipt['pose']['upper_body_xyc'][4][2]=.1
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert [p[0] for p in result['descriptor'].pose_shape]==[0,1,2,3,5]


def test_primary_adverse_body_observations_cannot_be_replaced_with_a_good_alternate():
    frozen,view,receipt=fixture()
    view['descriptor']=replace(view['descriptor'],pose_shape=((0,-.5,0.),(1,.5,0.),(4,9.,9.)))
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert result['descriptor']==view['descriptor']
    assert result['review_body_recovery_state']=='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY'


def complementary_fixture():
    frozen,view,receipt=fixture();raws=[raw(10),view['row']['native_observation'],raw(12)]
    points=deepcopy(receipt['pose']['upper_body_xyc'])
    for i in (2,3,5):points[i][2]=.1
    raws[1]['people'][0]['upper_body_xyc']=points
    raws[1]['people'][0]['part_states']=[['shoulder_anchor','observed'],['left_arm','unknown'],['right_arm','unknown']]
    frozen=freeze_review_body_window(raws,native_camera_unit=0)
    receipt['window_identity']=frozen.identity;receipt['raw_parent_sha256']=digest(raws[1])
    receipt['pose']['bbox_xywh']=frozen.payload['requests'][1]['roi']
    view['row']['projection']=dict(people=[dict(observation_id=0,unique_head_observation_id=7)])
    view['descriptor']=replace(view['descriptor'],owner_known=True,
        pose_shape=((0,-.5,0.),(1,.5,0.),(4,-1.5,-.28125)))
    return frozen,view,receipt


def test_complementary_actual_joints_are_unioned_in_native_coordinates_not_row_replaced():
    frozen,view,receipt=complementary_fixture();saved=deepcopy(view['row'])
    receipt['pose']['upper_body_xyc'][4]=[.9,.9,.1]
    receipt['pose']['raw_keypoints_xyc'][9]=[.9,.9,.1]
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert [p[0] for p in result['descriptor'].pose_shape]==[0,1,2,3,4,5]
    assert result['review_body_native_xyc'][4]==(.2,.4,.8)
    assert result['review_joint_provenance'][4]['basis']=='PRIMARY_NATIVE_MODEL'
    assert result['review_joint_provenance'][2]['basis']=='FIXED_ROI_NATIVE_MODEL'
    assert result['row']==saved and result['descriptor'].owner_known
    assert not result['auxiliary_body_is_production_ownership']


def test_primary_shoulders_normalize_all_added_joints_once_and_keep_native_conflict():
    frozen,view,receipt=complementary_fixture()
    for i,j in enumerate((5,6)):
        point=list(receipt['pose']['upper_body_xyc'][i]);point[0]+=.01
        receipt['pose']['upper_body_xyc'][i]=point;receipt['pose']['raw_keypoints_xyc'][j]=point
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert dict((p[0],p[1:]) for p in result['descriptor'].pose_shape)[2]==pytest.approx((-.9999999999999999,.28125))
    receipt['pose']['upper_body_xyc'][4]=[.9,.9,.8];receipt['pose']['raw_keypoints_xyc'][9]=[.9,.9,.8]
    conflicted=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert conflicted['descriptor']==view['descriptor']
    assert conflicted['review_body_recovery_state']=='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY'


def test_primary_joints_do_not_require_a_successful_auxiliary_model_to_survive():
    _,view,_=complementary_fixture();saved=deepcopy(view)
    primary=view['row']['native_observation']['people'][0]['upper_body_xyc']
    primary[2][2]=.7
    value=apply_review_primary_joint_observation(view,project_point=lambda p:p)
    assert {p[0] for p in value['descriptor'].pose_shape}=={0,1,2,4}
    assert value['review_joint_provenance'][2]['basis']=='PRIMARY_NATIVE_MODEL'
    assert 3 not in value['review_joint_provenance'] and 5 not in value['review_joint_provenance']
    assert value['primary_whole_arm_ownership_unchanged'] and value['row']==view['row']
    assert 'review_body_receipt_sha256' not in value
    assert view['descriptor']==saved['descriptor']


@pytest.mark.parametrize('adverse',['ambiguous','fail'])
def test_primary_only_resolver_does_not_hide_adverse_part_ownership(adverse):
    _,view,_=complementary_fixture()
    view['row']['native_observation']['people'][0]['part_states']=[
        ['shoulder_anchor','observed'],['left_arm',adverse],['right_arm','unknown']]
    assert apply_review_primary_joint_observation(view,project_point=lambda p:p)==view


def test_primary_only_missing_owner_or_cropped_shoulder_cannot_manufacture_body():
    _,view,_=complementary_fixture()
    assert apply_review_primary_joint_observation(view,project_point=lambda p:(p[0]+2,p[1]))==view
    view['row']['projection']['people']=[]
    assert apply_review_primary_joint_observation(view,project_point=lambda p:p)==view


def test_known_normalized_primary_without_actual_native_binding_is_not_inverted_to_make_a_union():
    frozen,view,receipt=complementary_fixture();view['row']['projection']['people']=[]
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert result['descriptor']==view['descriptor']
    assert 'review_joint_provenance' not in result


def test_individually_observed_primary_wrist_survives_unknown_whole_arm_without_fabricating_missing_elbow():
    frozen,view,receipt=complementary_fixture()
    native=view['row']['native_observation'];native['people'][0]['upper_body_xyc'][2][2]=.1
    frozen=freeze_review_body_window([raw(10),native,raw(12)],native_camera_unit=0)
    receipt['window_identity']=frozen.identity;receipt['raw_parent_sha256']=digest(native)
    receipt['pose']['upper_body_xyc'][4]=[.9,.9,.1];receipt['pose']['raw_keypoints_xyc'][9]=[.9,.9,.1]
    actual=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert actual['review_joint_provenance'][2]['basis']=='FIXED_ROI_NATIVE_MODEL'
    assert actual['review_joint_provenance'][4]['basis']=='PRIMARY_NATIVE_MODEL'
    native['people'][0]['part_states']=[['shoulder_anchor','observed'],['left_arm','ambiguous'],['right_arm','observed']]
    frozen=freeze_review_body_window([raw(10),native,raw(12)],native_camera_unit=0)
    receipt['window_identity']=frozen.identity;receipt['raw_parent_sha256']=digest(native)
    blocked=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert blocked['descriptor']==view['descriptor']
    assert blocked['review_body_recovery_state']=='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY'


@pytest.mark.parametrize('field', ['binding','head_fingerprint','window_identity','model_content_sha256','raw_parent_sha256'])
def test_wrong_physical_actor_window_or_model_is_rejected(field):
    frozen,view,receipt=fixture();receipt[field]='wrong'
    with pytest.raises(ValueError,match='borrowed'):
        apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)


def test_new_inference_cannot_select_an_easier_roi():
    frozen,view,receipt=fixture();receipt['pose']['bbox_xywh']=[0.,0.,1.,1.]
    with pytest.raises(ValueError,match='predeclared ROI'):
        apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)


def test_crop_outside_shoulder_does_not_become_measured_body():
    frozen,view,receipt=fixture()
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:(p[0]+2.,p[1]),expected_window=frozen)
    assert result['descriptor'].pose_shape is None
    assert result['review_body_recovery_state']=='SHOULDER_OUTSIDE_ACTUAL_CAMERA'


def test_multiple_head_window_or_clock_gap_is_not_a_single_actor_recovery():
    a,b=raw(10),raw(11);b['heads'].append(deepcopy(b['heads'][0]))
    with pytest.raises(ValueError,match='single-face'):freeze_review_body_window([a,b],native_camera_unit=0)
    with pytest.raises(ValueError,match='bounded source-local'):
        freeze_review_body_window([raw(10),raw(30)],native_camera_unit=0)


def test_explicit_prior_actor_allows_background_faces_but_not_salience_reselection():
    a,b=raw(10),raw(11)
    other=deepcopy(a['heads'][0]);other['observation_id']=99;other['box_xywh']=[.01,.01,.8,.8]
    for row in (a,b):row['heads'].append(deepcopy(other))
    window=freeze_review_body_window([a,b],native_camera_unit=0,head_ids=(7,7))
    assert window.payload['head_selection_scope']=='explicit_prior_selected_occurrences'
    assert [v['head_id'] for v in window.payload['requests']]==[7,7]
    assert not window.payload['body_ownership_certified']
    with pytest.raises(ValueError,match='independently selected'):
        freeze_review_body_window([a,b],native_camera_unit=0,head_ids=(7,))
    with pytest.raises(ValueError,match='absent, repeated'):
        freeze_review_body_window([a,b],native_camera_unit=0,head_ids=(7,100))


def test_auxiliary_raw_joint_mapping_cannot_be_replaced_by_favorable_upper_joints():
    frozen,view,receipt=fixture();receipt['pose']['raw_keypoints_xyc'][9]=[.9,.9,.9]
    with pytest.raises(ValueError,match='raw joint mapping'):
        apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)


def test_actual_simcc_raw_score_is_clipped_only_in_public_upper_confidence():
    frozen,view,receipt=fixture()
    receipt['pose']['raw_keypoints_xyc'][7]=[.3,.6,1.0356090068817139]
    receipt['pose']['upper_body_xyc'][2]=[.3,.6,1.]
    result=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    assert len(result['descriptor'].pose_shape)==6
    assert receipt['pose']['raw_keypoints_xyc'][7][2]>1.
    receipt['pose']['upper_body_xyc'][2][2]=.9
    with pytest.raises(ValueError,match='raw joint mapping'):
        apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)


def support_fixture():
    frozen,view,receipt=fixture()
    view['row']['projection']=dict(heads=[dict(observation_id=7,projected_box_xywh=[.4,.2,.1,.2],
        visible_fraction=1.)],people=[])
    duties=dict(contract='frozen_original_review_part_duties_v1',rows={'11':dict(head_required=True,
        parts={'left_arm':None,'right_arm':None})})
    return frozen,view,receipt,duties


def test_review_body_camera_and_support_share_verified_evidence_without_native_ownership_rewrite():
    from crosscut.render import EditorialTransformKeyframe
    frozen,view,receipt,duties=support_fixture();saved=deepcopy(view['row'])
    actual=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    verdict=consume_head_conditioned_review_support(duties,actual,key=EditorialTransformKeyframe(0.,1.))
    assert verdict['state']=='PASS' and verdict['part_states']['left_arm']=='PASS'
    assert not verdict['auxiliary_body_is_production_ownership'] and actual['row']==saved
    cropped=consume_head_conditioned_review_support(duties,actual,key=EditorialTransformKeyframe(0.,1.75))
    assert cropped['state']=='FAIL' and cropped['part_states']['left_arm']=='FAIL'


def test_low_confidence_required_wrist_stays_unknown_in_shared_support_adapter():
    from crosscut.render import EditorialTransformKeyframe
    frozen,view,receipt,duties=support_fixture();receipt['pose']['upper_body_xyc'][4][2]=.1
    actual=apply_review_body_receipt(view,receipt,project_point=lambda p:p,expected_window=frozen)
    verdict=consume_head_conditioned_review_support(duties,actual,key=EditorialTransformKeyframe(0.,1.))
    assert verdict['part_states']['left_arm']=='UNKNOWN' and verdict['part_states']['right_arm']=='PASS'
