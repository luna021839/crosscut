"""Real OpenCV on synthetic rasters, not person/video quality certification."""
from copy import deepcopy
from hashlib import sha256
from time import monotonic
from types import SimpleNamespace
import math
import cv2
import numpy as np
import pytest
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_review_patch_flow import measure_patch_head_links,PatchReviewTracklet
from crosscut.group_performer_family_owner import digest

def scene():
    rng=np.random.default_rng(773)
    a=np.zeros((192,256,3),np.uint8);a[50:130,70:150]=rng.integers(30,230,(80,80,3),np.uint8)
    b=cv2.warpAffine(a,np.asarray([[1,0,4],[0,1,-3]],float),(256,192))
    def raw(pixels,n,identity,x,y):
        return dict(request_source_index=0,native_frame_index=n,source_clock_identity='a'*64,
            source_file_identity=[1,2,3,4,5],frame_size_wh=[256,192],
            native_pixel_sha256=sha256(memoryview(pixels)).hexdigest(),
            heads=[dict(observation_id=identity,box_xywh=[x/256,y/192,80/256,80/192],
                landmarks_xy=[[.4,.4]]*5)])
    before,after=raw(a,10,3,70,50),raw(b,11,9,74,47)
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+20.,maximum_region_flow_feature_updates=1000)
    session=SimpleNamespace(budget=budget,domain=None,closed=False,identities={0:(1,2,3,4,5)},
        native_pixels=lambda src,n:{10:a,11:b}[n])
    return session,before,after

def test_real_patch_motion_ignores_detector_landmark_jitter_but_not_box_binding():
    session,a,b=scene();b['heads'][0]['landmarks_xy']=[[.9,.1]]*5
    result=measure_patch_head_links(session,a,b)
    assert result['unique_links']==[[3,9]] and result['motion_inliers']>=6
    assert result['median_translation_pixels']==pytest.approx([4,-3],abs=.15)
    assert len(result['primitive_refs'])==4 and session.budget.to_dict()['counts']['region_flow_feature_updates']>0
    assert not result['named_identity_certified'] and not result['motion_nonregression_certified']

def test_identical_competing_boxes_remain_ambiguous():
    session,a,b=scene();other=deepcopy(b['heads'][0]);other['observation_id']=2;b['heads'].append(other)
    value=measure_patch_head_links(session,a,b)
    assert value['unique_links']==[] and value['reason']=='independent_detector_patch_ambiguous'

def test_detector_order_does_not_change_selected_hypothesis():
    session,a,b=scene();other=deepcopy(b['heads'][0]);other['observation_id']=2;other['box_xywh']=[.7,.05,.15,.25]
    b['heads'].append(other);first=measure_patch_head_links(session,a,b)
    b['heads'].reverse();second=measure_patch_head_links(session,a,b)
    assert first['unique_links']==second['unique_links']==[[3,9]]

def test_native_gap_does_not_grow_to_make_sparse_endpoints_match():
    session,a,b=scene();b['native_frame_index']=23
    result=measure_patch_head_links(session,a,b)
    assert result['reason']=='native_patch_gap_unrequested' and result['primitive_refs']==[]

def test_native_cut_prevents_flow_before_raster_read():
    session,a,b=scene();clock=SimpleNamespace(frame_rate=30.,to_dict=lambda:dict(fps=30))
    a['source_clock_identity']=b['source_clock_identity']=digest(clock.to_dict())
    session.domain=SimpleNamespace(clocks_by_request={0:clock},cuts_for_request=lambda src:[11/30])
    session.native_pixels=lambda *args:pytest.fail('cut must stop before native raster decode')
    assert measure_patch_head_links(session,a,b)['reason']=='native_broadcaster_cut'

def test_changed_native_raster_and_file_are_not_unmatched_style():
    session,a,b=scene();b['native_pixel_sha256']='b'*64
    with pytest.raises(ValueError,match='raster'):measure_patch_head_links(session,a,b)
    session,a,b=scene();session.identities[0]=(5,4,3,2,1)
    with pytest.raises(ValueError,match='file'):measure_patch_head_links(session,a,b)

def test_tracklet_keeps_alias_and_never_reacquires_largest_face():
    session,a,b=scene();track=PatchReviewTracklet(3)
    assert track.observe(session,a,100)==3 and track.observe(session,a,101)==3
    assert track.observe(session,b,102)==9
    fake=deepcopy(b);fake['heads'][0]['observation_id']=21
    with pytest.raises(ValueError,match='alias'):track.observe(session,fake,103)
    value=track.to_dict()
    assert value['normal_landmark_contract_unchanged'] and not value['largest_face_reacquisition']

def test_no_feature_texture_is_unknown_not_a_different_face():
    session,a,b=scene();blank=np.zeros((192,256,3),np.uint8)
    session.native_pixels=lambda src,n:blank
    a['native_pixel_sha256']=b['native_pixel_sha256']=sha256(memoryview(blank)).hexdigest()
    result=measure_patch_head_links(session,a,b)
    assert result['reason']=='no_native_face_patch_features' and result['unique_links']==[]


@pytest.mark.parametrize('degrees,scale',[(8.,1.08),(-8.,.94)])
def test_real_patch_rotation_and_zoom_use_similarity_not_a_looser_translation_limit(degrees,scale):
    session,a,b=scene()
    before=session.native_pixels(0,10)
    matrix=cv2.getRotationMatrix2D((110.,90.),degrees,scale)
    matrix[:,2]+=np.asarray((3.,-2.))
    after=cv2.warpAffine(before,matrix,(256,192))
    corners=np.asarray(((70.,50.),(150.,50.),(70.,130.),(150.,130.)))
    moved=corners@matrix[:,:2].T+matrix[:,2]
    low,high=moved.min(axis=0),moved.max(axis=0)
    b['heads'][0]['box_xywh']=[low[0]/256,low[1]/192,(high[0]-low[0])/256,(high[1]-low[1])/192]
    b['native_pixel_sha256']=sha256(memoryview(after)).hexdigest()
    session.native_pixels=lambda src,n:{10:before,11:after}[n]
    value=measure_patch_head_links(session,a,b)
    assert value['unique_links']==[[3,9]] and value['motion_inliers']>=24
    assert value['similarity_scale']==pytest.approx(scale,abs=.02)
    assert value['similarity_roll_radians']==pytest.approx(math.radians(-degrees),abs=.02)
    assert not value['named_identity_certified'] and not value['motion_nonregression_certified']
