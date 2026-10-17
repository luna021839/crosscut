"""Read existing RTMW facial/hand landmarks without repeating perception.

COCO WholeBody indices follow MMPose's primary dataset metadata:
https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py
Raw peaks are not probabilities. These remain face-anchored model hypotheses.
"""
from dataclasses import dataclass, fields
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .subject_window_bank import SubjectWindow
from .partial_pose import face_proposals,face_guided_hypothesis
from .rtmw_pose import MODEL_SHA256

LANDMARKS=np.r_[np.arange(5),np.arange(91,133)]
GROUPS=(np.arange(5),np.arange(5,26),np.arange(26,47))
HEAD_EDGES=np.array(((0,1),(0,2),(1,2),(1,3),(2,4)))
HAND_EDGES=np.array([(0,1+4*f) for f in range(5)]+[(1+4*f+j,2+4*f+j) for f in range(5) for j in range(3)])
LANDMARKS.setflags(write=False)


@dataclass(frozen=True)
class DetailedWindow(SubjectWindow):
    detailed_points: object = None
    detailed_support: object = None

    def __post_init__(self):
        super().__post_init__()
        n=len(self.ordinals);points=np.asarray(self.detailed_points);known=np.asarray(self.detailed_support)
        if (points.shape!=(n,47,3) or known.shape!=(n,47) or known.dtype!=bool
                or not np.isfinite(points).all() or np.any(known&~self.associated[:,None])):
            raise ValueError('Exact native facial/hand points and separate observed support required')
        for name,value in (('detailed_points',points),('detailed_support',known)):
            value=value.copy();value.setflags(write=False);object.__setattr__(self,name,value)


def attach(window,path):
    """Direct sibling only; missing sidecar never triggers a sensor or decoder.

    Keep the historical producer (not today's file hashes). Verify its match
    to the original window, actual ordinal and exact face-proposed ROI. The
    body hypotheses used to bind it are not replaced or reserialized.
    """
    path=Path(path);raw_path=path.with_name(path.stem+'-raw133.npz')
    if not raw_path.exists():return window
    if raw_path.is_symlink():raise ValueError('Detailed pose sidecar must be an actual local artifact')
    with np.load(path,allow_pickle=False) as z:provenance=json.loads(str(z['provenance']))
    with np.load(raw_path,allow_pickle=False) as z:
        if 'producer' in z:
            if str(z['producer'])!=provenance.get('producer'):raise ValueError('Raw133 historical window producer changed')
        elif provenance.get('model')!=MODEL_SHA256:
            return window
        raw=z['raw133'];native=z['native'];rois=z['rois'] if 'rois' in z else z['roi'];rasters=z['raster_sha256']
    if (native.ndim!=1 or native.dtype.kind not in 'iu' or raw.shape!=(len(native),133,3)
            or rois.shape!=(len(native),4) or rasters.shape!=(len(native),)
            or not np.isfinite(raw).all() or not np.isfinite(rois).all()
            or any(len(str(h))!=64 or any(c not in '0123456789abcdef' for c in str(h)) for h in rasters)):
        raise ValueError('Actual raw133 native/ROI/raster observations required')
    points=np.zeros((len(window.ordinals),47,3),np.float32)
    known=np.zeros(points.shape[:2],bool);seen=set()
    for p,n,roi in zip(raw,native,rois):
        at=int(n)-int(window.ordinals[0])
        if not 0<=at<len(window.ordinals) or int(window.ordinals[at])!=int(n):
            raise ValueError('Raw133 observation outside its source window')
        if int(n) in seen:raise ValueError('Ambiguous repeated raw133 native observation')
        seen.add(int(n))
        if not window.associated[at]:continue
        proposals=face_proposals(window.heads[at])
        if not proposals or not np.allclose(roi,proposals[0],rtol=0,atol=1e-8):
            raise ValueError('Raw133 ROI changed its physical face proposal')
        evidence=SimpleNamespace(reason='observed',raw_keypoints_xyc=p[:17])
        anchored=face_guided_hypothesis(evidence,window.heads[at],window.competitors[at])
        if anchored.reason!='single_face_guided_raw_hypothesis':continue
        common=window.support[at,:6]&anchored.supported[:6]
        if not np.array_equal(window.body[at,:6,:2][common],anchored.points[:6,:2][common]):
            raise ValueError('Raw133 body no longer matches the original subject observation')
        selected=p[LANDMARKS];valid=(selected[:,2]>=.3)&(selected[:,:2]>=0).all(-1)&(selected[:,:2]<=1).all(-1)
        if not valid[0]:valid[:5]=False
        for wrist,root,group in ((4,91,GROUPS[1]),(5,112,GROUPS[2])):
            joint=9 if wrist==4 else 10
            root_bound=(anchored.supported[wrist] and p[root,2]>=.3
                and np.linalg.norm((p[root,:2]-p[joint,:2])*[16/9,1.])<=.3*window.heads[at,2])
            if not root_bound:valid[group]=False
        points[at]=selected;known[at]=valid
    return DetailedWindow(**{f.name:getattr(window,f.name) for f in fields(SubjectWindow)},
        detailed_points=points,detailed_support=known)


def describe(observation):
    """Exactly the actual -1/0 played endpoint, not a fitted facial pose."""
    result=dict(xy=np.zeros((47,2)),relative=np.zeros((47,2)),known=np.zeros(47,bool),visible=np.zeros(47,bool),height=0.)
    context=observation.get('dense_subject_context')
    if context is None or 'detailed_points' not in context:return result
    endpoint=6 if observation['side']==0 else 7
    points=np.asarray(context['detailed_points']);support=np.asarray(context['detailed_support'])
    if points.shape!=(14,47,3) or support.shape!=(14,47) or support.dtype!=bool:
        raise ValueError('Fourteen common-clock detailed source observations required')
    from .subject_action_match import _sample
    if (not np.array_equal(_sample(context['window'],context['clock']),context['at'])
            or not np.array_equal(context['clock'].output_frames,np.arange(-7,7))):
        raise ValueError('Detailed landmarks changed their physical native clock')
    camera=np.asarray(observation['camera_strip'])[-1 if observation['side']==0 else 0]
    head=context['window'].heads[context['at'][endpoint]]
    source=points[endpoint,:,:2]
    xy=(.5+camera[0]*(source-camera[1:]))*[16/9,1.]
    relative=(source-head[:2])*[16/9,1.]/max(.005,head[2])
    known=support[endpoint]&context['same_shot'][endpoint]
    visible=(xy[:,0]>=0)&(xy[:,0]<=16/9)&(xy[:,1]>=0)&(xy[:,1]<=1)
    return dict(xy=xy,relative=relative,known=known,visible=visible,height=float(head[2]*camera[0]))


def matrix(left,right,*,configuration_only=False):
    """Three fixed channels: head, left hand and right hand; UNKNOWN=.5.

    Compare actual rendered placement and head-relative configuration. The
    proposal-only form excludes native screen displacement before a crop is
    constructed. The worst fixed channel prevents a correctly matched face
    from diluting an unmatched hand; an absent channel stays neutral. A
    hidden hand does not become a perfect match. Model-only residuals shrink
    halfway toward UNKNOWN, as with the existing partial body measurement.
    """
    channels=[];observed=[]
    for index,group in enumerate(GROUPS):
        a=np.asarray([v['xy'][group] for v in left]);b=np.asarray([v['xy'][group] for v in right])
        ra=np.asarray([v['relative'][group] for v in left]);rb=np.asarray([v['relative'][group] for v in right])
        ka=np.asarray([v['known'][group] for v in left]);kb=np.asarray([v['known'][group] for v in right])
        va=np.asarray([v['visible'][group] for v in left]);vb=np.asarray([v['visible'][group] for v in right])
        known=ka[:,None]&kb[None]&(va[:,None]|vb[None])
        height=np.maximum(.005,(np.asarray([v['height'] for v in left])[:,None]
            +np.asarray([v['height'] for v in right])[None])/2.)
        placement=np.linalg.norm(a[:,None]-b[None],axis=-1)/height[...,None]
        configuration=np.linalg.norm(ra[:,None]-rb[None],axis=-1)
        raw=configuration if configuration_only else np.maximum(placement,configuration)
        bounded=raw/(1.+raw)
        bounded=np.where(va[:,None]!=vb[None],1.,bounded)
        count=known.sum(-1);seen=count>=3
        phase=(bounded*known).sum(-1)/np.maximum(1,count)
        edges=HEAD_EDGES if index==0 else HAND_EDGES
        ea=a[:,edges[:,1]]-a[:,edges[:,0]];eb=b[:,edges[:,1]]-b[:,edges[:,0]]
        la=np.linalg.norm(ea,axis=-1);lb=np.linalg.norm(eb,axis=-1)
        edge_known=known[...,edges].all(-1)&(la[:,None]>.003)&(lb[None]>.003)
        unit_a=ea/np.maximum(1e-12,la[...,None]);unit_b=eb/np.maximum(1e-12,lb[...,None])
        angular=(1.-np.clip((unit_a[:,None]*unit_b[None]).sum(-1),-1.,1.))/2.
        angle=(angular*edge_known).sum(-1)/np.maximum(1,edge_known.sum(-1))
        risk=.5+.5*(np.maximum(phase,angle)-.5)
        channels.append(np.where(seen,risk,.5));observed.append(seen)
    channels=np.stack(channels,axis=-1);observed=np.stack(observed,axis=-1)
    return channels.max(-1),observed.any(-1),channels,observed


def bound_connection_credit(detail):
    """A contradictory measured hand cannot retain a body-only match bonus.

    Undo only the documented model-only shrinkage of POSITION residuals,
    not a probability calibration. A known perfect configuration caps at1;
    a configuration no better than UNKNOWN caps at0. Missing channels cap nothing. Never
    add credit, veto a source, or change cut admission.
    """
    if 'detailed_pose_channels' not in detail:return detail
    channels=np.asarray(detail['detailed_pose_channels'],float)
    known=np.asarray(detail['detailed_pose_channel_measured'],bool)
    if (channels.shape!=known.shape or channels.shape[-1:]!=(3,)
            or not np.isfinite(channels).all()):raise ValueError('Three actual measured detailed-pose channels required')
    raw=np.clip(2.*channels-.5,0.,1.)
    cap=np.maximum(0.,1.-2.*np.where(known,raw,0.).max(-1))
    result=dict(detail,detailed_pose_action_credit_cap=cap)
    if 'matched_action_change_density' in result:
        result['matched_action_change_density']=np.minimum(result['matched_action_change_density'],cap)
    if 'actual_played_match_advantage' in result:
        result['actual_played_match_advantage']=np.minimum(result['actual_played_match_advantage'],cap)
    return result
