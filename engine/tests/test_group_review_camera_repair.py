"""Synthetic numeric fixtures, not actual performer or editing-quality proof."""
from copy import deepcopy
from crosscut.group_review_camera_repair import (FramingDescriptor, framing_descriptor,
    geometry_cost, compare_two_sided_geometry, fit_whole_camera_track,
    measured_native_support_preserved,effective_face_detail_pixels,transformed_descriptor)
from crosscut.group_review_camera_repair import relax_measured_camera_track
from crosscut.group_review_camera_repair import ReviewActorSelection
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack
import pytest

def face(h=.1, x=.45, y=.3):
    box=[x,y,h/2,h]
    return dict(projection=dict(heads=[dict(observation_id=0,projected_box_xywh=box,
        visible_fraction=1.)],people=[]),native_observation=dict(
        heads=[dict(observation_id=0,box_xywh=box)],people=[],frame_size_wh=[40,24]))

def descriptor(h=.1, x=.5):
    return FramingDescriptor((x,.4),h,h/2,None,0,False)


def test_unmatched_actor_choice_cannot_become_largest_face_salience():
    row=face()
    assert framing_descriptor(row,actor_selection=ReviewActorSelection('DEFAULT_SALIENCE')) is not None
    assert framing_descriptor(row,actor_selection=ReviewActorSelection.occurrence(None)) is None
    assert framing_descriptor(row,actor_selection=ReviewActorSelection.occurrence(0)).frame_local_head_id==0
    assert framing_descriptor(row,actor_selection=ReviewActorSelection.occurrence(99)) is None

def test_unknown_is_not_zero_or_geometry_gain():
    assert geometry_cost(None,descriptor()) is None
    row=compare_two_sided_geometry(left=None,right=descriptor(),original_first=descriptor(),
        original_last=descriptor(),candidate_first=descriptor(),candidate_last=descriptor())
    assert not row['review_geometry_gain'] and row['original_costs'][0] is None

def test_better_entry_does_not_pay_for_worse_exit():
    row=compare_two_sided_geometry(left=descriptor(.2),right=descriptor(.1),
        original_first=descriptor(.05),original_last=descriptor(.1),
        candidate_first=descriptor(.2),candidate_last=descriptor(.15))
    assert not row['review_geometry_gain']

def test_two_sided_geometry_improvement_is_review_not_admission():
    row=compare_two_sided_geometry(left=descriptor(.2),right=descriptor(.2),
        original_first=descriptor(.05),original_last=descriptor(.05),
        candidate_first=descriptor(.15),candidate_last=descriptor(.15))
    assert row['review_geometry_gain']
    assert row['candidate_admitted'] is False and row['motion_nonregression_certified'] is False

def test_measured_visible_face_cannot_be_cut_to_fit_neighbor():
    row=face(.1,.03)
    assert measured_native_support_preserved(row,EditorialTransformKeyframe(0.,1.))
    assert not measured_native_support_preserved(row,EditorialTransformKeyframe(0.,1.75))

def test_partial_measured_arm_retains_visible_bone_not_outside_vertex():
    row=face(); joints=[[.45,.4,1.],[.55,.4,1.],[.45,1.2,1.],[.55,.6,1.],
        [.45,.8,1.],[.55,.8,1.],[.45,.7,1.],[.55,.7,1.]]
    row['native_observation']['people']=[dict(upper_body_xyc=joints,part_states={'left_arm':'observed'})]
    assert measured_native_support_preserved(row,EditorialTransformKeyframe(0.,1.))
    assert not measured_native_support_preserved(row,EditorialTransformKeyframe(0.,1.3))

def test_body_box_is_not_a_semantic_cast():
    row=face(); row['projection']['people']=[dict(observation_id=0,projected_box_xywh=[0.,0.,1.,1.],
        unique_head_observation_id=None)]
    assert framing_descriptor(row).owner_known is False
    assert measured_native_support_preserved(row,EditorialTransformKeyframe(0.,1.3))

def test_continuous_whole_camera_no_fake_role_edit_or_motion():
    rows=[face() for _ in range(3)]
    frozen=deepcopy(rows)
    track=fit_whole_camera_track(rows,descriptor(.15),descriptor(.15),duration=1.)
    assert track is not None and track.evaluate(.5).zoom>=1.
    assert rows==frozen and framing_descriptor(rows[0]).to_dict()['motion_observed'] is False

def test_body_geometry_does_not_invent_a_face_or_cast():
    row=face();row['projection']['heads']=[];row['native_observation']['heads']=[]
    row['projection']['people']=[dict(observation_id=0,projected_box_xywh=[.4,.2,.2,.5],
        visible_fraction=1.,unique_head_observation_id=None)]
    row['native_observation']['people']=[dict(observation_id=0,confidence=.9)]
    value=framing_descriptor(row)
    assert value.face_height is None and value.head_center is None and value.body_box is not None
    assert value.to_dict()['body_box_is_required_cast'] is False

def test_added_face_coverage_alone_is_not_editorial_gain():
    body=FramingDescriptor(None,None,None,None,None,False,(.4,.2,.2,.5))
    added=FramingDescriptor((.5,.3),.1,.05,None,0,False,(.4,.2,.2,.5))
    row=compare_two_sided_geometry(left=body,right=body,original_first=body,original_last=body,
        candidate_first=added,candidate_last=added)
    assert not row['review_geometry_gain']

def test_legal_identity_not_discarded_for_bad_endpoint_optimum():
    rows=[face(.1,.25),face(.1,.001),face(.1,.55)]
    track=fit_whole_camera_track(rows,descriptor(.15),descriptor(.15),duration=1.)
    assert track is not None
    assert measured_native_support_preserved(rows[1],track.evaluate(.5))


def test_face_body_geometry_does_not_borrow_a_larger_other_person():
    row=face(); row['projection']['people']=[
        dict(observation_id=0,projected_box_xywh=[.4,.2,.2,.5],visible_fraction=1.,
            unique_head_observation_id=0,projected_upper_body_xyc=[],part_states={}),
        dict(observation_id=1,projected_box_xywh=[.1,.1,.8,.8],visible_fraction=1.,
            unique_head_observation_id=None,projected_upper_body_xyc=[],part_states={})]
    row['native_observation']['people']=[dict(observation_id=0,confidence=.9),dict(observation_id=1,confidence=.9)]
    assert framing_descriptor(row).body_box==(.4,.2,.2,.5)


def test_selected_frame_local_head_cannot_silently_switch_to_largest():
    row=face()
    assert framing_descriptor(row,head_id=0).frame_local_head_id==0
    assert framing_descriptor(row,head_id=1) is None


def test_native_resolution_cannot_hide_small_output_face_or_invent_zoom_detail():
    row=face(.03);row['native_observation']['frame_size_wh']=[3840,2160]
    row['query']=dict(camera=dict(output_size=[1920,1080]))
    value=framing_descriptor(row)
    assert value.native_face_pixels==64.8
    assert effective_face_detail_pixels(value)==32.4
    zoomed=transformed_descriptor(value,EditorialTransformKeyframe(0.,2.))
    assert effective_face_detail_pixels(zoomed)==64.8
    assert effective_face_detail_pixels(transformed_descriptor(zoomed,EditorialTransformKeyframe(0.,2.)))==64.8


def test_unmeasured_output_inventory_is_not_guessed_for_legibility():
    assert effective_face_detail_pixels(framing_descriptor(face())) is None


def test_dense_correction_retains_largest_legal_whole_camera_not_tickwise_zoom():
    rows=[face(.1,.25),face(.1,.001),face(.1,.55)]
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.4),EditorialTransformKeyframe(1.,1.4)))
    fixed,report=relax_measured_camera_track(track,rows,sample_times=(0.,.5,1.),
        required_support_check=measured_native_support_preserved)
    assert fixed is not None and fixed.evaluate(.5).zoom==1.
    assert report['measured_ticks']==3 and not report['fresh_inference_performed']
    assert all(measured_native_support_preserved(row,fixed.evaluate(t)) for row,t in zip(rows,(0.,.5,1.)))


def test_impossible_native_duty_is_not_hidden_by_camera_relaxation():
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),EditorialTransformKeyframe(1.,1.2)))
    fixed,report=relax_measured_camera_track(track,[face()],sample_times=(0.,),
        required_support_check=lambda row,key:False)
    assert fixed is None and report['retained_intervention_fraction'] is None


def test_dense_correction_rejects_sample_index_or_unbounded_search():
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),EditorialTransformKeyframe(1.,1.2)))
    with pytest.raises(ValueError,match='bounded ordered'):
        relax_measured_camera_track(track,[face(),face()],sample_times=(.5,.4),
            required_support_check=measured_native_support_preserved)
    with pytest.raises(ValueError,match='bounded ordered'):
        relax_measured_camera_track(track,[face()],sample_times=(0.,),maximum_trials=100000,
            required_support_check=measured_native_support_preserved)


def test_context_face_does_not_override_explicit_original_focal_support_consumer():
    rows=[face(.1,.45) for _ in range(3)]
    for row in rows:
        row['native_observation']['heads'].append(dict(observation_id=9,box_xywh=[.002,.25,.03,.06]))
    conservative=fit_whole_camera_track(rows,descriptor(.15),descriptor(.15),duration=1.)
    calls=[]
    def focal_support(row,key):
        calls.append(key.zoom)
        focal=deepcopy(row)
        focal['native_observation']['heads']=focal['native_observation']['heads'][:1]
        return measured_native_support_preserved(focal,key)
    explicit=fit_whole_camera_track(rows,descriptor(.15),descriptor(.15),duration=1.,
        head_ids=(0,0,0),required_support_check=focal_support)
    assert conservative.evaluate(.5).zoom==1.
    assert explicit.evaluate(.5).zoom>1.4 and calls


def test_explicit_required_support_failure_cannot_be_ignored_in_primary_camera_fit():
    assert fit_whole_camera_track([face() for _ in range(3)],descriptor(.15),descriptor(.15),
        duration=1.,required_support_check=lambda row,key:False) is None
    with pytest.raises(ValueError,match='explicit boolean'):
        fit_whole_camera_track([face() for _ in range(3)],descriptor(.15),descriptor(.15),
            duration=1.,required_support_check=lambda row,key:'UNKNOWN')


def test_offscreen_projected_wrist_cannot_be_observed_pose_support():
    row=face();joints=[[.4,.4,.9],[.6,.4,.9],[.4,.6,.9],[.6,.6,.9],
        [.4,1.1,.9],[.6,.8,.9],[.4,.7,.9],[.6,.7,.9]]
    row['projection']['people']=[dict(observation_id=0,unique_head_observation_id=0,
        projected_box_xywh=[.3,.2,.4,.8],visible_fraction=1.,projected_upper_body_xyc=joints,
        part_states=dict(shoulder_anchor='observed',left_arm='observed',right_arm='observed'))]
    row['native_observation']['people']=[dict(observation_id=0,confidence=.9)]
    shape=framing_descriptor(row).pose_shape
    assert 4 not in {p[0] for p in shape}
    assert {0,1,2,3,5}<={p[0] for p in shape}
