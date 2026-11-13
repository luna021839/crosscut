"""Separate anonymous review hypothesis from actual native face-patch features.

This does not relax the normal landmark-flow policy or certify actor identity.
Independent detector boxes constrain a robust feature similarity; conflicting
boxes stay unmatched. All OpenCV sensor returns have explicit primitive work.
"""
from hashlib import sha256
import math
import cv2
import numpy as np

from .group_performer_family_owner import digest
from .group_scene_head_pixel_flow import _check_pair, POLICY as LANDMARK_POLICY
from .group_review_native_window import NativeReviewTracklet

CONTRACT='native_face_patch_feature_anonymous_review_hypothesis_v2'
POLICY=dict(contract=CONTRACT,maximum_native_gap=12,maximum_features=48,minimum_features=6,
    maximum_forward_backward_pixels=1.25,quality_level=.01,minimum_point_distance=3.,
    window_size=[21,21],maximum_pyramid_level=4,maximum_iterations=30,epsilon=.01,
    minimum_motion_inlier_fraction=.5,maximum_similarity_residual_pixels=2.,
    similarity_estimator='opencv_lmeds_partial_affine',maximum_fit_iterations=128,
    maximum_refine_iterations=10,minimum_similarity_scale=.5,maximum_similarity_scale=2.,
    minimum_predicted_box_iou=.2,maximum_normalized_center_error=.6,
    minimum_target_area_ratio=.4,maximum_target_area_ratio=2.5,association_ambiguity_margin=.15)

def _hash(array):return sha256(memoryview(np.ascontiguousarray(array))).hexdigest()

def _box(head,size):
    box=np.asarray(head['box_xywh'],float)*np.asarray((*size,*size),float)
    x,y,w,h=box
    if not np.isfinite(box).all() or not 0<=x<x+w<=size[0] or not 0<=y<y+h<=size[1]:return None
    return box

def _iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1])
    area=max(0.,min(a[0]+a[2],b[0]+b[2])-x)*max(0.,min(a[1]+a[3],b[1]+b[3])-y)
    return area/(a[2]*a[3]+b[2]*b[3]-area)

def _arrays(result,count):
    if type(result) is not tuple or len(result)!=3:return None
    points,status,_=result
    if (not isinstance(points,np.ndarray) or points.shape!=(count,1,2) or points.dtype.kind!='f'
            or not np.isfinite(points).all() or not isinstance(status,np.ndarray) or status.shape!=(count,1)
            or status.dtype.kind not in 'bui'):return None
    return points,status[:,0].astype(bool)

def measure_patch_head_links(session,a,b):
    _check_pair(a,b)
    if getattr(session,'closed',False):raise ValueError('patch review requires a live native raster session')
    for raw in (a,b):
        if type(raw['request_source_index']) is not int or type(raw['native_frame_index']) is not int:
            raise ValueError('patch native source/frame must be exact integers')
        ids=[head['observation_id'] for head in raw['heads']]
        if any(type(i) is not int for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('patch head inventory has invalid frame-local IDs')
    if len(a['heads'])!=1:raise ValueError('patch review needs one explicit seed, not a largest-face chooser')
    first,last=a['native_frame_index'],b['native_frame_index']
    result=dict(contract=CONTRACT,raw_pair=[digest(a),digest(b)],policy_sha256=digest(POLICY),
        native_frame_indices=[first,last],unique_links=[],alternatives=[],primitive_refs=[],
        feature_count=0,forward_backward_inliers=0,motion_inliers=0,median_translation_pixels=None,
        similarity_matrix=None,similarity_scale=None,similarity_roll_radians=None,
        reason=None,cross_camera_identity_certified=False,named_identity_certified=False,
        motion_nonregression_certified=False,quality_approved=False)
    if last-first>POLICY['maximum_native_gap']:
        result['reason']='native_patch_gap_unrequested';return result
    domain=getattr(session,'domain',None)
    if domain is not None:
        clock=domain.clocks_by_request[a['request_source_index']]
        if digest(clock.to_dict())!=a['source_clock_identity']:raise ValueError('patch source clock changed')
        if any(first/float(clock.frame_rate)<cut<=last/float(clock.frame_rate)
                for cut in domain.cuts_for_request(a['request_source_index'])):
            result['reason']='native_broadcaster_cut';return result
    grays=[]
    for raw in (a,b):
        pixels=session.native_pixels(raw['request_source_index'],raw['native_frame_index'])
        if (not isinstance(pixels,np.ndarray) or pixels.dtype!=np.uint8
                or pixels.shape!=(a['frame_size_wh'][1],a['frame_size_wh'][0],3)
                or _hash(pixels)!=raw['native_pixel_sha256']):
            raise ValueError('patch native raster changed its recorded binding')
        identities=getattr(session,'identities',None)
        if isinstance(identities,dict) and identities.get(a['request_source_index'])!=tuple(raw['source_file_identity']):
            raise ValueError('patch native file identity changed')
        grays.append(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY))
    head=a['heads'][0];box=_box(head,a['frame_size_wh'])
    if box is None:result['reason']='seed_patch_outside_native_raster';return result
    x,y,w,h=box;x0,y0=math.ceil(x),math.ceil(y);x1,y1=math.floor(x+w),math.floor(y+h)
    if x1-x0<3 or y1-y0<3:result['reason']='seed_patch_too_small';return result
    binding=dict(contract=CONTRACT,raw_pair=result['raw_pair'],source=a['request_source_index'],
        native_frames=[first,last],source_clock_identity=a['source_clock_identity'],
        source_file_identity=a['source_file_identity'],seed_head_id=head['observation_id'],
        source_pixel_sha256=a['native_pixel_sha256'],target_pixel_sha256=b['native_pixel_sha256'],
        seed_native_box=box.tolist(),policy_sha256=result['policy_sha256'])
    def primitive(label,inputs,work,producer):
        journal=session.budget.primitive_journal;sequence=len(journal.attempts)
        value=session.budget.measure_primitive(label=label,input_binding=dict(binding,**inputs),
            work=dict(region_flow_feature_updates=work),maximum_result_bytes=8192,producer=producer)
        result['primitive_refs'].append(journal.attempts[sequence]['measurement_ref_sha256'])
        return value
    points=primitive('review_native_face_patch_good_features',dict(native_roi=[x0,y0,x1,y1]),
        POLICY['maximum_features'],lambda:cv2.goodFeaturesToTrack(grays[0][y0:y1,x0:x1],
            maxCorners=POLICY['maximum_features'],qualityLevel=POLICY['quality_level'],
            minDistance=POLICY['minimum_point_distance']))
    if points is None:result['reason']='no_native_face_patch_features';return result
    if (not isinstance(points,np.ndarray) or points.ndim!=3 or points.shape[1:]!=(1,2)
            or not np.isfinite(points).all() or len(points)>POLICY['maximum_features']):
        raise ValueError('invalid native patch feature producer return')
    points=points+np.asarray((x0,y0),np.float32);result['feature_count']=len(points)
    if len(points)<POLICY['minimum_features']:result['reason']='insufficient_native_patch_features';return result
    options=dict(winSize=tuple(POLICY['window_size']),maxLevel=POLICY['maximum_pyramid_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,POLICY['maximum_iterations'],POLICY['epsilon']))
    forward=primitive('review_native_face_patch_lk_forward',dict(point_sha256=_hash(points)),len(points),
        lambda:cv2.calcOpticalFlowPyrLK(grays[0],grays[1],points,None,**options))
    parsed=_arrays(forward,len(points))
    if parsed is None:result['reason']='invalid_forward_patch_flow';return result
    after,fs=parsed
    backward=primitive('review_native_face_patch_lk_backward',dict(point_sha256=_hash(after)),len(points),
        lambda:cv2.calcOpticalFlowPyrLK(grays[1],grays[0],after,None,**options))
    parsed=_arrays(backward,len(points))
    if parsed is None:result['reason']='invalid_backward_patch_flow';return result
    back,bs=parsed
    width,height=a['frame_size_wh']
    alive=fs&bs&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<=POLICY['maximum_forward_backward_pixels'])
    alive &= (after[:,0,0]>=0)&(after[:,0,0]<width)&(after[:,0,1]>=0)&(after[:,0,1]<height)
    result['forward_backward_inliers']=int(alive.sum())
    if alive.sum()<POLICY['minimum_features']:result['reason']='insufficient_forward_backward_support';return result
    # A moving/rotating face is not a translation-only patch. Preserve the
    # original pixel FB/residual/inlier requirements and fit the motion model
    # they constrain instead of relaxing those limits to turn UNKNOWN green.
    motion=primitive('review_native_face_patch_similarity_fit',
        dict(source_point_sha256=_hash(points[alive]),target_point_sha256=_hash(after[alive])),
        int(alive.sum()),lambda:cv2.estimateAffinePartial2D(points[alive,0],after[alive,0],
            method=cv2.LMEDS,maxIters=POLICY['maximum_fit_iterations'],
            confidence=.99,refineIters=POLICY['maximum_refine_iterations']))
    if (type(motion) is not tuple or len(motion)!=2 or not isinstance(motion[0],np.ndarray)
            or motion[0].shape!=(2,3) or not np.isfinite(motion[0]).all()):
        result['reason']='invalid_native_patch_similarity';return result
    matrix=motion[0]
    a0,b0=matrix[0,0],matrix[1,0];scale=math.hypot(a0,b0)
    if (abs(matrix[0,1]+b0)>1e-6 or abs(matrix[1,1]-a0)>1e-6
            or not POLICY['minimum_similarity_scale']<=scale<=POLICY['maximum_similarity_scale']):
        result['reason']='nonphysical_native_patch_similarity';return result
    transformed=points[:,0]@matrix[:,:2].T+matrix[:,2]
    stable=alive&(np.linalg.norm(after[:,0]-transformed,axis=1)<=POLICY['maximum_similarity_residual_pixels'])
    result['motion_inliers']=int(stable.sum())
    if stable.sum()<max(POLICY['minimum_features'],math.ceil(len(points)*POLICY['minimum_motion_inlier_fraction'])):
        result['reason']='incoherent_patch_motion';return result
    delta=after[:,0]-points[:,0]
    result['median_translation_pixels']=np.median(delta[stable],axis=0).tolist()
    result['similarity_matrix']=matrix.tolist();result['similarity_scale']=scale
    result['similarity_roll_radians']=math.atan2(b0,a0)
    corners=np.asarray(((x,y),(x+w,y),(x,y+h),(x+w,y+h)),float)
    moved=corners@matrix[:,:2].T+matrix[:,2]
    low,high=moved.min(axis=0),moved.max(axis=0);predicted=np.asarray((*low,*(high-low)))
    session.budget.spend('pose_comparisons',len(b['heads']))
    eligible=[]
    for target in b['heads']:
        target_box=_box(target,b['frame_size_wh'])
        if target_box is None:continue
        iou=_iou(predicted,target_box);ratio=target_box[2]*target_box[3]/(box[2]*box[3])
        center_error=float(np.linalg.norm(target_box[:2]+target_box[2:]/2-predicted[:2]-predicted[2:]/2)/np.linalg.norm(box[2:]))
        cost=center_error+1-iou+.35*abs(math.log(ratio))
        candidate=dict(target_head_id=target['observation_id'],predicted_box_iou=iou,
            target_area_ratio=ratio,normalized_center_error=center_error,cost=cost)
        result['alternatives'].append(candidate)
        if (iou>=POLICY['minimum_predicted_box_iou'] and center_error<=POLICY['maximum_normalized_center_error']
                and POLICY['minimum_target_area_ratio']<=ratio<=POLICY['maximum_target_area_ratio']):eligible.append(candidate)
    ordered=sorted(eligible,key=lambda value:(value['cost'],value['target_head_id']))
    if not ordered:result['reason']='independent_detector_patch_unmatched';return result
    if len(ordered)>1 and ordered[1]['cost']-ordered[0]['cost']<POLICY['association_ambiguity_margin']:
        result['reason']='independent_detector_patch_ambiguous';return result
    result['unique_links']=[[head['observation_id'],ordered[0]['target_head_id']]]
    result['reason']='measured_anonymous_patch_box_hypothesis'
    return result

class PatchReviewTracklet(NativeReviewTracklet):
    def _measure_link(self,session,view,raw):return measure_patch_head_links(session,view,raw)

    def to_dict(self):
        result=super().to_dict()
        if POLICY['maximum_native_gap']!=LANDMARK_POLICY['maximum_native_gap']:
            raise ValueError('review patch policy must not silently enlarge native gap')
        return dict(result,contract=CONTRACT,native_gap_policy=dict(POLICY),
            normal_landmark_contract_unchanged=True,detector_landmark_agreement_certified=False)
