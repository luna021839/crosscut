from dataclasses import replace
import numpy as np
import pytest
from crosscut.render import (RenderPlan,RenderSegment,RenderTransition,TransitionKind,
    EditorialTransformTrack,EditorialTransformKeyframe,DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    CACHED_MEMBER_SHORT_DISSOLVE_REASON,MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY)
from crosscut.ive import AspectRatio
from crosscut2.member_short_dissolve import assess


def _input():
    faces=np.tile([.4,.2,.12,.12],(3,2,1));known=np.ones((3,2),bool);pose=np.zeros((3,2,8,3))
    pose[:,:,:,0]=np.linspace(.3,.7,8);pose[:,:,:,1]=np.linspace(.4,.9,8);pose[:,:,:,2]=.9
    return faces,known,pose


def test_current_matched_overlap_preserves_actual_pose_confidence():
    faces,known,pose=_input();before=pose.copy()
    result=assess(faces,known,pose,output_size=(1080,1920))
    assert result['eligible'] and result['measured_future_bone_pairs']==12
    assert result['maximum_visible_arm_direction_mismatch']==0
    np.testing.assert_array_equal(before,pose)


def test_a_future_wrong_arm_or_face_scale_is_not_hidden_by_matching_endpoints():
    faces,known,pose=_input();pose[2,1,4,:2]=[.1,.1]
    assert not assess(faces,known,pose,output_size=(1080,1920))['eligible']
    faces,known,pose=_input();faces[2,1,3]*=1.8
    assert not assess(faces,known,pose,output_size=(1080,1920))['eligible']


def test_missing_future_body_remains_unknown_and_missing_current_face_declines_effect_only():
    faces,known,pose=_input();pose[:,:,:,2]=0
    result=assess(faces,known,pose,output_size=(1080,1920))
    assert result['eligible'] and result['maximum_visible_arm_direction_mismatch'] is None
    assert result['score']==.5 and result['measured_future_bone_pairs']==0
    known[1,0]=False
    assert not assess(faces,known,pose,output_size=(1080,1920))['eligible']


def test_current_face_overlap_authority_cannot_claim_the_old_joint_registration_contract(tmp_path):
    with pytest.raises(ValueError,match='explicitly reviewed or source-measured'):
        RenderTransition(TransitionKind.FACE_DISSOLVE,6,reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
            transform_overlap_authority=MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY)
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2,.5,.5),),reason='measured_face_camera')
    handle=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2,.5,.5),
        EditorialTransformKeyframe(5/30,1.2,.5,.5)),reason='measured_face_future_handle')
    transition=RenderTransition(TransitionKind.FACE_DISSOLVE,6,reason=CACHED_MEMBER_SHORT_DISSOLVE_REASON,
        transform_overlap_authority=MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY)
    segments=(RenderSegment(0,0.,2.,0.,editorial_transform=camera,outgoing_overlap_transform=handle),
        RenderSegment(1,0.,2.,2.,editorial_transform=camera))
    plan=RenderPlan((tmp_path/'a.mp4',tmp_path/'b.mp4'),segments,0,AspectRatio.PORTRAIT,tmp_path/'film.mp4',
        transitions=(transition,),width=1080,height=1920,fps=30.,member_pose_dissolve_requested=True,
        member_pose_dissolve_effective=True,member_pose_dissolve_count=1)
    assert plan.transitions[0].to_dict(30.)['transform_overlap_authority']==MEMBER_SOURCE_MEASURED_FACE_OVERLAP_AUTHORITY
    from crosscut.render_plan_serialization import render_plan_from_json
    recovered=render_plan_from_json(plan.to_json())
    assert recovered.to_json()==plan.to_json()
    with pytest.raises(ValueError,match='disabled member pose dissolve cannot author an effect'):
        replace(plan,member_pose_dissolve_requested=False)
    with pytest.raises(ValueError,match='registration needs both explicit overlap cameras'):
        replace(plan,segments=(replace(segments[0],outgoing_overlap_transform=None),segments[1]))
