from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.temporal_face_chain import bind,reanchor_existing_raw
from crosscut2.partial_pose import face_rectangle,face_guided_hypothesis


def box(head):
    xywh=face_rectangle(head);return np.r_[xywh[:2],xywh[:2]+xywh[2:]]


def example():
    heads=[np.array([.5+.05*i,.35,.2,.95]) for i in range(3)]
    frames=[np.full((80,144,3),i,np.uint8) for i in range(3)]
    detections=[np.array([head,[.1,.2,.1,.95]]) for head in heads]
    def flow(first,last,current):return box(heads[int(last[0,0,0])]),dict(flow_points=12,inliers=10)
    kw=dict(ordinals=np.array([20,21,22]),endpoint_head=heads[0],endpoint_at=0,_flow=flow)
    return frames,detections,heads,kw


def test_unique_local_chain_moves_with_actor_not_largest_or_hypothetical_identity():
    frames,found,heads,kw=example();found[-1]=found[-1][::-1].copy()
    result=bind(frames,found,**kw)
    assert result.accepted and result.detection_indices==(0,0,1)
    assert np.array_equal(result.head,heads[-1])
    assert np.array_equal(result.others,found[-1][[0]])
    assert not result.head.flags.writeable and not result.named_identity_certified


def test_backward_A_window_binds_contemporary_head_and_others():
    frames,found,heads,kw=example();kw.update(endpoint_at=2,endpoint_head=heads[-1])
    result=bind(frames,found,**kw)
    assert result.accepted and np.array_equal(result.head,heads[0])
    assert [s['to_native'] for s in result.steps]==[21,20]


def test_moving_face_restores_only_model_hypothesis_not_intruding_stale_box_nose():
    frames,found,heads,kw=example();result=bind(frames,found,**kw)
    raw=np.zeros((17,3));raw[:,2]=.9;raw[0,:2]=heads[-1][:2];raw[5:11,:2]=[.6,.5]
    evidence=SimpleNamespace(reason='observed',raw_keypoints_xyc=raw)
    stale=face_guided_hypothesis(evidence,heads[0],np.zeros((0,4)))
    current=face_guided_hypothesis(evidence,result.head,result.others)
    assert not stale.supported.any() and current.supported.sum()==6
    # Another actor can enter the stale box and pass the old check. The
    # contemporary selected head does NOT make that intruder this actor.
    raw[0,:2]=heads[0][:2]
    assert face_guided_hypothesis(evidence,heads[0],np.zeros((0,4))).supported.any()
    assert not face_guided_hypothesis(evidence,result.head,result.others).supported.any()


def test_crossing_duplicate_or_missing_detection_cannot_choose_nearest_head():
    for missing in (False,True):
        frames,found,heads,kw=example()
        found[1]=np.zeros((0,4)) if missing else np.array([heads[1],heads[1]+[.005,0,0,0]])
        result=bind(frames,found,**kw)
        assert not result.accepted and 'ambiguous' in result.reason
        assert not result.head.any() and len(result.detection_indices)==1


def test_failed_pixel_link_never_falls_back_to_current_detections():
    frames,found,heads,kw=example();kw['_flow']=lambda *a:(None,dict(flow_points=0))
    result=bind(frames,found,**kw)
    assert not result.accepted and result.reason=='temporal_face_pixel_path_unmeasured'


def test_missing_ambiguous_endpoint_and_native_cut_are_explicitly_unmeasured():
    frames,found,heads,kw=example();found[0]=np.vstack((found[0],heads[0]))
    assert bind(frames,found,**kw).reason=='endpoint_redetection_missing_or_ambiguous'
    frames,found,heads,kw=example()
    assert bind(frames,found,**kw,native_cuts=[21]).reason=='native_camera_cut_inside_window'


@pytest.mark.parametrize('ordinals',[np.array([20,20,22]),np.array([20,22,23]),np.array([22,21,20]),np.array([20.,21.,22.])])
def test_frozen_or_skipped_or_non_native_clocks_cannot_fabricate_chain(ordinals):
    frames,found,heads,kw=example();kw['ordinals']=ordinals
    with pytest.raises(ValueError,match='native face window'):bind(frames,found,**kw)


def test_real_flow_on_textureless_frames_does_not_certify_stationary_face():
    frames,found,heads,kw=example();kw.pop('_flow');frames=[np.zeros_like(f) for f in frames]
    assert bind(frames,found,**kw).reason=='temporal_face_pixel_path_unmeasured'


def test_existing_raw_reanchors_without_coordinate_refit_or_stale_face_fallback():
    frames,found,heads,kw=example();chain=bind(frames,found,**kw)
    raw=np.zeros((17,3));raw[:,2]=.9;raw[0,:2]=heads[-1][:2];raw[5:11,:2]=[.6,.5]
    before=raw.copy();evidence=SimpleNamespace(reason='observed',raw_keypoints_xyc=raw)
    partial=reanchor_existing_raw(evidence,chain,raw_roi_xywh=[.1,.1,.8,.8])
    assert partial.supported.sum()==6 and np.array_equal(raw,before)
    assert np.array_equal(partial.points[:6],raw[5:11].astype(np.float32))
    outside=reanchor_existing_raw(evidence,chain,raw_roi_xywh=[.1,.1,.2,.2])
    assert not outside.supported.any() and 'requires_new_observation' in outside.reason
    kw['_flow']=lambda *args:(None,{})
    rejected=reanchor_existing_raw(evidence,bind(frames,found,**kw),raw_roi_xywh=[0,0,1,1])
    assert not rejected.supported.any() and 'chain_unmeasured' in rejected.reason


@pytest.mark.parametrize('roi',[[0,0,0,1],[-.1,0,1,1],[0,0,2,1],[0,0,np.nan,1]])
def test_raw_roi_cannot_be_relabelled_or_expanded_by_adapter(roi):
    frames,found,heads,kw=example();chain=bind(frames,found,**kw)
    with pytest.raises(ValueError,match='raw ROI'):reanchor_existing_raw(None,chain,raw_roi_xywh=roi)
