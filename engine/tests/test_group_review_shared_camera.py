"""Shared-camera adapter contract; synthetic joints are not media-quality proof."""
from copy import deepcopy
import pytest
from crosscut.group_review_shared_camera import fit_shared_head_body_review_camera
from crosscut.group_review_body_recovery import freeze_review_body_window,apply_review_body_receipt
from crosscut.group_review_camera_repair import framing_descriptor,FramingDescriptor
from crosscut.group_performer_family_owner import digest
from crosscut.group_review_body_recovery import native_binding,head_fingerprint
from crosscut.body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256
from engine.tests.test_group_review_body_recovery import raw


def views_and_duties():
    raws=[raw(t) for t in (10,11,12)]
    window=freeze_review_body_window(raws,native_camera_unit=0)
    points=[[.4,.5,.8],[.6,.5,.8],[.3,.6,.8],[.7,.6,.8],[.2,.4,.8],[.8,.4,.8],[.4,.9,.4],[.6,.9,.4]]
    views=[]
    for current in raws:
        row=dict(output_tick=current['native_frame_index'],native_observation=current,
            projection=dict(heads=[dict(observation_id=7,projected_box_xywh=[.4,.2,.1,.2],
                visible_fraction=1.)],people=[]))
        raw17=[[.45,.3,.8] for _ in range(17)]
        for i,j in enumerate((5,6,7,8,9,10,11,12)):raw17[j]=points[i]
        request=next(r for r in window.payload['requests'] if r['binding']==native_binding(current))
        receipt=dict(contract='actual_fixed_roi_body_review_receipt_v1',window_identity=window.identity,
            model_content_sha256=RTMPOSE_BODY7_MODEL_SHA256,binding=native_binding(current),
            raw_parent_sha256=digest(current),head_fingerprint=head_fingerprint(current['heads'][0]),
            pose=dict(reason='observed',face_anchor_consistent=True,bbox_xywh=request['roi'],
                upper_body_xyc=points,raw_keypoints_xyc=raw17))
        views.append(apply_review_body_receipt(dict(row=row,descriptor=framing_descriptor(row,head_id=7)),
            receipt,project_point=lambda p:p,expected_window=window))
    duties=dict(contract='frozen_original_review_part_duties_v1',rows={str(t):dict(
        head_required=True,parts={'left_arm':None,'right_arm':None}) for t in (10,11,12)})
    return views,duties


def test_actual_merged_body_descriptor_and_same_support_consumer_reach_camera(monkeypatch):
    import crosscut.group_review_shared_camera as module
    views,duties=views_and_duties();saved=deepcopy(views);observed=[]
    real=module.fit_whole_camera_track
    def witness(rows,left,right,**kwargs):
        observed.extend(kwargs['descriptors'])
        return real(rows,left,right,**kwargs)
    monkeypatch.setattr(module,'fit_whole_camera_track',witness)
    target=FramingDescriptor((.45,.3),.25,.125,None,7,False)
    track,receipt=fit_shared_head_body_review_camera(views,target,target,duration=2/30.,
        sample_times=(0.,1/30.,2/30.),original_part_duties=duties)
    assert track is not None and observed==[v['descriptor'] for v in views]
    assert all(len(d.pose_shape)==6 for d in observed)
    assert any(r['state']=='PASS' and r.get('review_head_conditioned_parts') for r in receipt['support_receipts'])
    assert not receipt['unknown_support_certified_pass'] and views==saved


def test_primary_face_fail_cannot_be_hidden_by_the_auxiliary_body():
    views,duties=views_and_duties();views[1]['row']['projection']['heads'][0]['visible_fraction']=.5
    track,receipt=fit_shared_head_body_review_camera(views,None,None,duration=2/30.,
        sample_times=(0.,1/30.,2/30.),original_part_duties=duties)
    assert track is None and any(r['state']=='FAIL' for r in receipt['support_receipts'])


def test_missing_actor_or_duplicate_occurrence_does_not_change_the_camera_sample_inventory():
    views,duties=views_and_duties()
    for rows in ([views[0],None,views[2]],[views[0],views[0],views[2]]):
        with pytest.raises(ValueError):fit_shared_head_body_review_camera(rows,None,None,duration=2/30.,
            sample_times=(0.,1/30.,2/30.),original_part_duties=duties)
