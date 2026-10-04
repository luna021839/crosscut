from dataclasses import replace
import json
import pytest
from crosscut.group_review_camera_repair import (FramingDescriptor,fit_whole_camera_track,
    transformed_descriptor,geometry_cost)
from crosscut.group_review_two_shot_camera import (realize_review_two_shot_camera_pair,
    FrozenRealizedTwoShotCandidate,select_review_two_shot_diagnostic)


def view(t,x,h):
    box=[x-h/4,.35-h/2,h/2,h]
    row=dict(output_tick=t,native_observation=dict(heads=[dict(observation_id=0,box_xywh=box)],people=[],
        frame_size_wh=[1920,1080]),
        projection=dict(heads=[dict(observation_id=0,projected_box_xywh=box,visible_fraction=1.,
            projected_landmarks_xy=[])],people=[]),query=dict(camera=dict(output_size=[1920,1080])))
    return dict(row=row,descriptor=FramingDescriptor((x,.35),h,h/2,None,0,False))


def test_actual_donor_joint_fit_beats_two_donors_independently_fitted_to_old_neighbours():
    a=[view(t,.42,.15) for t in range(10,16)];b=[view(t,.60,.18) for t in range(16,22)]
    old=replace(a[0]['descriptor'],head_center=(.5,.45),face_height=.08,face_width=.04)
    la,lb,receipt=realize_review_two_shot_camera_pair(a,b,first=10,boundary=16,end=22,fps=30.,
        original_entry=old,original_exit=old,required_support_check=lambda side,v,key:'PASS')
    def oldfit(values):
        return fit_whole_camera_track([v['row'] for v in values],old,old,duration=5/30.,
            sample_times=tuple(i/30. for i in range(6)),head_ids=(0,)*6,
            descriptors=tuple(v['descriptor'] for v in values),required_support_check=lambda row,key:True)
    aa,bb=oldfit(a),oldfit(b)
    before=geometry_cost(transformed_descriptor(a[-1]['descriptor'],aa.evaluate(5/30.)),
        transformed_descriptor(b[0]['descriptor'],bb.evaluate(0.)))
    assert receipt['state']=='REALIZED' and receipt['geometry_costs']['internal']<before
    assert receipt['camera_fitted_before_selection'] and not receipt['post_selection_refit_permitted']


def candidate(boundary,motion,geometry=0.,state='PASS',closeups=1.):
    return FrozenRealizedTwoShotCandidate(json.dumps(dict(contract='sealed_realized_two_shot_candidate_v1',
        boundary=boundary,technical_state=state,changed_frames=48,camera_fitted_before_selection=True,
        post_selection_camera_refit=False,native_motion_error=motion,projected_geometry_error=geometry,
        camera_instability=0.,closeup_fraction=closeups)))


def test_predeclared_boundary_alternative_and_motion_take_priority_over_closeup_reward():
    a=candidate(3665,2.,closeups=1.);b=candidate(3671,.1,closeups=.1)
    assert select_review_two_shot_diagnostic([a,b])==b
    assert select_review_two_shot_diagnostic([candidate(3677,0.,state='UNKNOWN')]) is None


def test_selected_camera_cannot_be_changed_after_the_realization_was_ranked():
    a=candidate(3671,.1);p=a.payload;p['camera_instability']=100.
    with pytest.raises(ValueError,match='after sealing'):a.assert_unchanged(p)


def test_joint_fit_does_not_claim_unknown_required_support_is_pass():
    a=[view(t,.42,.15) for t in range(10,16)];b=[view(t,.60,.18) for t in range(16,22)]
    _,_,r=realize_review_two_shot_camera_pair(a,b,first=10,boundary=16,end=22,fps=30.,
        original_entry=a[0]['descriptor'],original_exit=b[-1]['descriptor'],
        required_support_check=lambda side,v,key:'UNKNOWN')
    assert r['state']=='REALIZED' and r['actual_support_state']=='UNKNOWN'
