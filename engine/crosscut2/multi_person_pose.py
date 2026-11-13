"""Original full-frame RTMO/COCO17 numerics, independent of an edited actor.

The official body7-L pipeline uses BGR0..255, top-left114 padding,640 square,
and IoU.65 suppression. No face crops, identity labels or certified anatomy.
References: open-mmlab/mmpose/projects/rtmo and the author's rtmlib RTMO API.
This module loads no models, reads no pixels from disk and acquires nothing.
"""
import cv2
import numpy as np

MODEL_SHA256='090096ca90f29163cc4f67137dcc0cd4b2ee95ea0af11764fbfda88dd2ae1140'
BODY8_COCO17=(5,6,7,8,9,10,11,12)


def tensor(image):
    """Keep the actual source aspect; return the exact inverse resize ratio."""
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[-1]!=3 or min(image.shape[:2])<2):
        raise ValueError('original BGR8 image required')
    h,w=image.shape[:2];ratio=min(640/h,640/w)
    resized=cv2.resize(image,(max(1,int(w*ratio)),max(1,int(h*ratio))),interpolation=cv2.INTER_LINEAR)
    padded=np.full((640,640,3),114,np.uint8)
    padded[:resized.shape[0],:resized.shape[1]]=resized
    return np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32),ratio


def decode(outputs,*,ratio,raster,minimum_score=.35,nms_iou=.65,maximum_people=32):
    """Separate real body instances before associating ANY proposed face.

    The exported model may return multiple hypotheses for one person. Keep
    the strongest overlapping box, not several body poses for one face.
    Coordinates remain in original normalized units, including out-of-raster
    joints; downstream evidence masks must decide visibility, not clamp them.
    """
    if (len(outputs)!=2 or not np.isfinite(ratio) or ratio<=0
            or len(raster)!=2 or any(type(v) is not int or v<2 for v in raster)
            or not 0<=minimum_score<=1 or not 0<nms_iou<=1
            or type(maximum_people) is not int or not 1<=maximum_people<=64):
        raise ValueError('explicit finite original multi-person geometry/configuration required')
    det,pose=map(np.asarray,outputs)
    if (det.ndim!=3 or det.shape[0]!=1 or det.shape[-1]!=5
            or pose.shape!=det.shape[:2]+(17,3) or not np.isfinite(det).all()
            or not np.isfinite(pose).all() or np.any((det[...,4]<0)|(det[...,4]>1))
            or np.any((pose[...,2]<0)|(pose[...,2]>1))):
        raise ValueError('original RTMO dets/COCO17 tensor family changed')
    boxes=det[0,:,:4];scores=det[0,:,4]
    area=np.prod(np.maximum(0.,boxes[:,2:]-boxes[:,:2]),axis=-1)
    order=np.flatnonzero((scores>=minimum_score)&(area>0))
    order=order[np.argsort(-scores[order],kind='stable')];kept=[]
    while len(order) and len(kept)<maximum_people:
        chosen=int(order[0]);kept.append(chosen);others=order[1:]
        overlap=np.prod(np.maximum(0.,np.minimum(boxes[chosen,2:],boxes[others,2:])-
            np.maximum(boxes[chosen,:2],boxes[others,:2])),axis=-1)
        iou=overlap/np.maximum(1e-12,area[chosen]+area[others]-overlap)
        order=others[iou<=nms_iou]
    width,height=raster;indices=np.asarray(kept,np.int64)
    people=pose[0,indices].astype(np.float32,copy=True)
    people[...,:2]/=ratio*np.array([width,height],np.float32)
    return dict(boxes=(boxes[indices]/ratio/[width,height,width,height]).astype(np.float32),
        scores=scores[indices].copy(),keypoints17=people,raw_detection_indices=indices,
        above_score_before_nms=int(((scores>=minimum_score)&(area>0)).sum()),
        evidence='MODEL_ONLY',named_identity_certified=False,body8_indices=BODY8_COCO17)


def bind_faces(heads,people,*,source_aspect,maximum_face_distance=1.25):
    """One-to-one original-face/body association BEFORE a crop or edit.

    Nose/eye anchors belong to a full-image body instance, not a pose chosen
    from a face ROI containing several performers. A missing anchor remains
    unassociated; one body cannot silently supply two faces' arm motions.
    The distance is geometric support, not a person-identity probability.
    """
    from scipy.optimize import linear_sum_assignment
    heads=np.asarray(heads,float);points=np.asarray(people['keypoints17'],float)
    scores=np.asarray(people['scores'],float)
    if (heads.ndim!=2 or heads.shape[1]!=4 or points.ndim!=3 or points.shape[1:]!=(17,3)
            or scores.shape!=(len(points),) or not np.isfinite(heads).all() or not np.isfinite(points).all()
            or not np.isfinite(source_aspect) or source_aspect<=0
            or not np.isfinite(maximum_face_distance) or maximum_face_distance<=0):
        raise ValueError('actual original face/body geometry and aspect required')
    assigned=np.full(len(heads),-1,np.int64);distance=np.full(len(heads),np.nan)
    body=np.zeros((len(heads),8,3),np.float32)
    active=(heads[:,2]>.019)&(heads[:,3]>=.75)
    # The visible nose is normally central; the eyes also support an anchor
    # when the nose is hidden. Ears are intentionally not a back-of-head ID.
    q=points[:,:3];valid=(q[...,2]>=.3)&(q[...,:2]>=0).all(-1)&(q[...,:2]<=1).all(-1)
    weight=np.where(valid,q[...,2]*[2.,1.,1.],0.);mass=weight.sum(-1)
    anchors=(q[...,:2]*weight[...,None]).sum(1)/np.maximum(1e-12,mass[:,None])
    person_ids=np.flatnonzero(mass>0);face_ids=np.flatnonzero(active)
    if len(person_ids) and len(face_ids):
        physical=(heads[face_ids,None,:2]-anchors[person_ids][None])*[source_aspect,1.]
        gap=np.linalg.norm(physical,axis=-1)/heads[face_ids,None,2]
        cost=gap+.05*(1.-scores[person_ids])[None]
        cost=np.where(gap<=maximum_face_distance,cost,maximum_face_distance+1.)
        # Per-face UNKNOWN alternatives prevent a bad mandatory assignment.
        costs=np.c_[cost,np.full((len(face_ids),len(face_ids)),maximum_face_distance+.1)]
        rows,columns=linear_sum_assignment(costs)
        for row,col in zip(rows,columns):
            if col>=len(person_ids) or gap[row,col]>maximum_face_distance:continue
            face,person=int(face_ids[row]),int(person_ids[col])
            assigned[face]=person;distance[face]=gap[row,col]
            body[face]=points[person,list(BODY8_COCO17)]
    supported=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    supported&=(assigned>=0)[:,None]
    return dict(person_index=assigned,body8=body,supported=supported,head_anchor_distance=distance,
        anatomical_ownership_certified=False,named_identity_certified=False)
