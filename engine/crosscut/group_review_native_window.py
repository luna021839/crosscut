"""Native pixel detail and a declared anonymous review tracklet.

The complete NN inventory remains untouched. A one-head LK seed view is
explicitly linked to its complete measured parent, never a new NN result.
Links are camera-local and gap<=12. Missing links do not select another maximum
face, infer member identity or manufacture a motion certificate.
"""
from types import SimpleNamespace
import hashlib
import math

import cv2
import numpy as np

from .group_performer_family_owner import digest
from .group_scene_head_pixel_flow import measure_head_pixel_links, POLICY

CONTRACT='actual_native_pixel_readability_anonymous_track_review_v1'


def measure_head_patch_details(session,raw):
    pixels=session.native_pixels(raw['request_source_index'],raw['native_frame_index'])
    if pixels is None or hashlib.sha256(memoryview(pixels)).hexdigest()!=raw['native_pixel_sha256']:
        raise ValueError('review patch features changed their actual native observation raster')
    width,height=raw['frame_size_wh'];features={}
    # Three aggregate feature updates per measured head; this is not an
    # all-pixel operation/raster-storage or occlusion-mask certificate.
    session.budget.spend('region_flow_feature_updates',3*len(raw['heads']))
    for head in raw['heads']:
        x,y,w,h=head['box_xywh'];x0,y0=math.floor(x*width),math.floor(y*height)
        x1,y1=math.ceil((x+w)*width),math.ceil((y+h)*height)
        identity=head['observation_id']
        if not 0<=x0<x1<=width or not 0<=y0<y1<=height:
            features[identity]=None;continue
        patch=cv2.cvtColor(pixels[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        patch=cv2.resize(patch,(64,64),interpolation=cv2.INTER_AREA)
        contrast=float(patch.std())/255.
        detail=float(cv2.Laplacian(patch,cv2.CV_32F).var())
        eyes=np.asarray(head['landmarks_xy'][:2],float)*np.asarray((width,height))
        risk=.65*max(0.,1.-detail/100.)+.35*max(0.,1.-contrast/.12)
        features[identity]=dict(native_pixel_sha256=raw['native_pixel_sha256'],
            native_frame_index=raw['native_frame_index'],head_observation_id=identity,
            patch_detail_variance=detail,local_contrast=contrast,
            native_eye_span_pixels=float(np.linalg.norm(eyes[0]-eyes[1])),detail_risk=risk,
            true_resolution_certified=False,occlusion_state='unknown',
            black_color_is_rejection=False,quality_approved=False)
    return features


class NativeReviewTracklet:
    """One explicitly selected seed, no largest-face reacquisition."""
    def __init__(self,head_id):
        if type(head_id) is not int:raise ValueError('review LK seed must be a frame-local actual head ID')
        self.seed=head_id;self.head_id=head_id;self.previous=None
        self.rows=[];self.links=[];self.broken=False

    def _measure_link(self,session,view,raw):
        return measure_head_pixel_links(SimpleNamespace(session=session),view,raw)

    def observe(self,session,raw,tick):
        if self.previous is None:
            if self.head_id not in [h['observation_id'] for h in raw['heads']]:self.head_id=None
        elif raw['request_source_index']!=self.previous['request_source_index']:
            raise ValueError('source-local review track cannot cross cameras')
        elif raw['native_frame_index']==self.previous['native_frame_index']:
            if digest(raw)!=digest(self.previous):raise ValueError('native alias changed its raw observation')
        elif not self.broken:
            gap=raw['native_frame_index']-self.previous['native_frame_index']
            if not 0<gap<=POLICY['maximum_native_gap']:
                self.head_id=None;self.broken=True
            else:
                seed_head=next((h for h in self.previous['heads'] if h['observation_id']==self.head_id),None)
                if seed_head is None:self.head_id=None;self.broken=True
                else:
                    session.budget.spend('association_nodes',1)
                    session.budget.spend('association_edges',len(raw['heads']))
                    view=dict(self.previous,heads=[seed_head])
                    link=self._measure_link(session,view,raw)
                    record=dict(complete_source_native_parent_sha256=digest(self.previous),
                        target_source_native_parent_sha256=digest(raw),
                        seed_view_is_not_new_detector_authority=True,link=link)
                    self.links.append(record)
                    choices=[b for a,b in link['unique_links'] if a==self.head_id]
                    if len(choices)==1:
                        self.head_id=choices[0]
                    else:
                        # Keep the last actually linked seed for a bounded
                        # later bridge; do NOT silently choose today's max.
                        self.rows.append(dict(output_tick=tick,head_id=None,
                            native_frame_index=raw['native_frame_index'],state='unmatched_native_track'))
                        return None
        value=None if self.broken else self.head_id
        self.rows.append(dict(output_tick=tick,head_id=value,native_frame_index=raw['native_frame_index'],
            state='observed_camera_local' if value is not None else 'missing'))
        if value is not None:self.previous=raw
        return value

    def to_dict(self):
        return dict(contract=CONTRACT,seed_frame_local_head_id=self.seed,rows=self.rows,
            measured_source_local_links=self.links,broken=self.broken,
            native_gap_policy=dict(POLICY),largest_face_reacquisition=False,
            cross_source_actor_identity_certified=False,named_member_identity_certified=False,
            whole_window_motion_nonregression_certified=False,quality_approved=False)


def cosine(a,b):
    if a is None or b is None:return None
    aa=np.asarray(a,float);bb=np.asarray(b,float)
    if aa.shape!=(128,) or bb.shape!=(128,) or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError('review appearance space differs from pinned 128d vectors')
    norm=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return None if norm<1e-12 else float(aa@bb/norm)


def one_to_one_appearance_hypotheses(left,right,*,objective,budget=None):
    """Mutual unique best hypotheses with unmatched and competing alternatives.

    Appearance alone is NOT actor certification. The caller adds source-local
    tracks and measured pose; ambiguity survives the resulting review score.
    """
    a={h['observation_id']:h['embedding'] for h in left['heads']}
    b={h['observation_id']:h['embedding'] for h in right['heads']}
    if budget is not None:
        budget.spend('association_nodes',len(a)+len(b))
        budget.spend('association_edges',len(a)*len(b))
    costs={(i,j):cosine(v,w) for i,v in a.items() for j,w in b.items()}
    alternatives={};unique=[]
    for i in sorted(a):
        choices=sorted(((value,j) for (ii,j),value in costs.items() if ii==i and value is not None),reverse=True)
        alternatives[i]=[dict(target_head_id=j,cosine=value) for value,j in choices]
        if not choices or choices[0][0]<objective.minimum_appearance_cosine:continue
        value,j=choices[0]
        if len(choices)>1 and value-choices[1][0]<objective.appearance_ambiguity_margin:continue
        reverse=sorted(((v,ii) for (ii,jj),v in costs.items() if jj==j and v is not None),reverse=True)
        if reverse[0][1]!=i or len(reverse)>1 and reverse[0][0]-reverse[1][0]<objective.appearance_ambiguity_margin:continue
        unique.append((i,j))
    return dict(unique_review_hypotheses=unique,alternatives=alternatives,
        unmatched_source_head_ids=[i for i in a if i not in [x for x,_ in unique]],
        unmatched_target_head_ids=[j for j in b if j not in [y for _,y in unique]],
        pose_motion_evidence_still_required=True,actor_identity_certified=False,
        identity_threshold_is_professional_approval=False,quality_approved=False)
