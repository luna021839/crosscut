"""Associate all-face candidates to already observed full-image bodies.

Four physical native samples remain sparse MODEL_ONLY hypotheses, not a
fourteen-frame track. Face/DP changes do not request more model observations.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .multi_person_pose import bind_faces,BODY8_COCO17


def link_people(a,b,*,aspect):
    """One-to-one short same-shot correspondence, with UNKNOWN alternatives."""
    pa,pb=np.asarray(a['keypoints17']),np.asarray(b['keypoints17'])
    result=np.full(len(pa),-1,int)
    if not len(pa) or not len(pb):return result
    ba,bb=np.asarray(a['boxes']),np.asarray(b['boxes'])
    scale=np.maximum(.025,(ba[:,3]-ba[:,1])[:,None]+(bb[:,3]-bb[:,1])[None,:])/2
    seen_a=(pa[...,2]>=.3)&(pa[...,:2]>=0).all(-1)&(pa[...,:2]<=1).all(-1)
    seen_b=(pb[...,2]>=.3)&(pb[...,:2]>=0).all(-1)&(pb[...,:2]<=1).all(-1)
    known=seen_a[:,None]&seen_b[None,:]
    delta=np.linalg.norm((pa[:,None,:,:2]-pb[None,:,:,:2])*[aspect,1.],axis=-1)
    # Median corresponding landmarks resist one occluded/invented wrist.
    distance=np.ma.median(np.ma.array(delta,mask=~known),axis=-1).filled(np.inf)/scale
    shared=known.sum(-1)>=3
    ratio=np.abs(np.log(np.maximum(.01,ba[:,3]-ba[:,1])[:,None]/np.maximum(.01,bb[:,3]-bb[:,1])[None]))
    cost=distance+.15*ratio
    valid=shared&(cost<=.75)
    extended=np.c_[np.where(valid,cost,2.),np.full((len(pa),len(pa)),.85)]
    rows,columns=linear_sum_assignment(extended)
    for i,j in zip(rows,columns):
        if j>=len(pb) or not valid[i,j]:continue
        # Crossing indistinguishable people stays unknown, not a forced swap.
        if min(np.delete(cost[i],j).min(initial=np.inf),np.delete(cost[:,j],i).min(initial=np.inf))-cost[i,j]<.04:continue
        result[i]=j
    return result


class OriginalPeople:
    def __init__(self,bank):
        raw=bank['original_people_observations'];self.raw=raw
        self.aspect=raw['raster'][0]/raw['raster'][1]
        heads=np.asarray(bank['heads']);native=np.asarray(bank['pose_native_indices'])
        N,_,H,_=heads.shape
        self.body=np.zeros((N,4,H,8,3),np.float32)
        self.head_points=np.zeros((N,4,H,5,3),np.float32)
        self.associated=np.zeros((N,4,H),bool)
        self.endpoint_matches=np.full((N,H),-1,int)
        self.person_indices=np.full((N,4,H),-1,int)
        rows=raw['by_native'];cuts=np.asarray(bank.get('native_cut_frames',[]),int)
        for n in range(N):
            frames=[rows[int(v)] for v in native[n]];seed=[]
            for side,endpoint,far in ((0,1,0),(1,2,3)):
                bound=bind_faces(heads[n,side],frames[endpoint],source_aspect=self.aspect)
                people=bound['person_index'];seed.append(people)
                self.person_indices[n,endpoint]=people
                self.body[n,endpoint]=bound['body8'];self.associated[n,endpoint]=people>=0
                present=people>=0
                self.head_points[n,endpoint,present]=frames[endpoint]['keypoints17'][people[present],:5]
                if np.any((cuts>min(native[n,endpoint],native[n,far]))&(cuts<=max(native[n,endpoint],native[n,far]))):continue
                links=(np.arange(len(frames[endpoint]['scores'])) if native[n,endpoint]==native[n,far]
                    else link_people(frames[endpoint],frames[far],aspect=self.aspect))
                selected=np.full(H,-1,int);selected[present]=links[people[present]]
                self.person_indices[n,far]=selected;good=selected>=0
                self.body[n,far,good]=frames[far]['keypoints17'][selected[good]][:,list(BODY8_COCO17)]
                self.head_points[n,far,good]=frames[far]['keypoints17'][selected[good],:5]
                self.associated[n,far]=good
            if np.any((cuts>native[n,1])&(cuts<=native[n,2])):continue
            links=(np.arange(len(frames[1]['scores'])) if native[n,1]==native[n,2]
                else link_people(frames[1],frames[2],aspect=self.aspect))
            for h,person in enumerate(seed[0]):
                if person<0 or links[person]<0:continue
                candidates=np.flatnonzero(seed[1]==links[person])
                if len(candidates)==1:self.endpoint_matches[n,h]=candidates[0]
        self.supported=(self.body[...,2]>=.3)&(self.body[...,:2]>=0).all(-1)&(self.body[...,:2]<=1).all(-1)
        self.supported&=self.associated[...,None]

    def attach_context(self,result,n,side,h,subjects):
        """Keep valid old FACE motion separate from replaced ROI body poses."""
        result=dict(result,original_full_image_people_used=True,
            original_people_family=self.raw['family'],
            original_people_reference_sha256=self.raw['reference_sha256'])
        if 0<=h<self.body.shape[2]:
            endpoint=1 if side==0 else 2
            result['original_people_head_points']=self.head_points[n,endpoint,h]
            result['original_people_associated']=bool(self.associated[n,endpoint,h])
        if subjects is not None:
            context=subjects.sample(n,side,h)
            if context is not None:
                result['focal_image_motion_context']=context
                result['detailed_subject_context']=context
        return result

    def receipt(self):
        return dict(family=self.raw['family'],source=self.raw['source'],
            reference_sha256=self.raw['reference_sha256'],
            associated_endpoint_faces=int(self.associated[:,[1,2]].sum()),
            associated_four_sample_faces=int(self.associated.sum()),
            face_capacity=self.body.shape[2],new_neural_calls=0,
            four_samples_are_not_a_dense_track=True,anatomical_ownership_certified=False)
