"""Pre-camera, original-family evidence for a change of featured person.

This is not a singer/identity certificate. Weak or absent evidence keeps the
ordinary same-subject costs. Native and authored changes consume the SAME
soft interpretation; no cut bonus or new source exclusion is issued here.
"""
import numpy as np


def geometry_importance(value):
    """Explicit editorial importance; not evidence confidence or admission."""
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
            or not np.isfinite(value) or not 1.<=value<=8.):
        raise ValueError('Explicit finite focal geometry importance must be in1..8')
    return float(value)


class FeatureRoles:
    def __init__(self,nodes,evidence):
        self.nodes=np.asarray(nodes,int)
        self.posterior=np.asarray(evidence['posterior'],float)
        self.confidence=np.asarray(evidence['confidence'],float)
        votes=np.asarray(evidence['family_votes'],float)
        shape=(len(self.nodes),2)
        if (self.posterior.shape!=shape+(6,) or self.confidence.shape!=shape
                or votes.ndim!=4 or votes.shape[1:]!=shape+(6,)
                or not all(np.isfinite(v).all() for v in (self.posterior,self.confidence,votes))
                or np.any((self.confidence<0)|(self.confidence>1))
                or np.any(self.posterior<0) or np.any(votes<0)
                or not np.allclose(self.posterior.sum(-1),1.)):
            raise ValueError('original-family featured posterior and confidence required')
        self.members=self.posterior.argmax(-1)
        self.support=(votes>0).sum(0)
        supported=np.take_along_axis(self.support,self.members[...,None],axis=-1)[...,0]>=2
        # Decide the KIND from raw independent views, never from the crop or
        # which interpretation happens to make a proposed cut cheaper.
        changed=(self.members[:,0]!=self.members[:,1])&supported.all(-1)
        self.strength=np.where(changed,self.confidence.min(-1),0.)

    def at(self,frames,left_members,right_members):
        frames,left,right=np.broadcast_arrays(np.asarray(frames,int),np.asarray(left_members,int),np.asarray(right_members,int))
        n=np.searchsorted(self.nodes,frames)
        exact=(n<len(self.nodes))&(self.nodes[np.minimum(n,len(self.nodes)-1)]==frames)
        n=np.minimum(n,len(self.nodes)-1)
        matches=(left==self.members[n,0])&(right==self.members[n,1])&(left>=0)&(right>=0)
        return np.where(exact&matches,self.strength[n],0.)

    def receipt(self):
        at=np.flatnonzero(self.strength>0)
        return dict(contract='original_family_feature_handoff_soft_v1',
            events=[dict(frame=int(self.nodes[n]),kind='FEATURE_HANDOFF',
                members=self.members[n].tolist(),strength=float(self.strength[n]),
                independent_family_support=[int(self.support[n,side,self.members[n,side]]) for side in (0,1)]) for n in at],
            inferred_featured_not_verified_singer=True,named_identity_certified=False,
            kind_fixed_before_camera=True,native_and_authored_same_interpretation=True,
            authored_cut_bonus=0.,native_fulfillment='same displayed-role deficit integrated over observed content',
            unobserved_or_unchanged_role_gets_no_handoff_discount=True)


def apply(detail,strength):
    """Finite mixture of same-person match and supported role handoff.

    A handoff still pays frame-scale/position and motion-flow risk. It does
    not pay for two genuinely different people looking different, nor force
    their different faces to the identical location via a large digital crop.
    """
    strength=np.broadcast_to(np.asarray(strength,float),np.shape(detail['geometry_rate']))
    if not np.isfinite(strength).all() or np.any((strength<0)|(strength>1)):
        raise ValueError('bounded original-view role strength required')
    scale=np.asarray(detail['face_log_scale'])/np.log(1.5)
    position=np.asarray(detail['screen_displacement'])/np.hypot(16/9,1.)
    framing=(scale/(1.+scale)+position)/2.
    framing=np.where(detail['measured_face'],framing,1.)
    result=dict(detail)
    result['geometry_rate']=(1.-strength)*detail['geometry_rate']+strength*framing
    result['pose_rate']=(1.-strength)*detail['pose_rate']+strength*detail['motion_flow_rate']
    result['appearance_rate']=(1.-strength)*detail['appearance_rate']
    result['allowed']=np.asarray(detail['allowed'])|((1.-strength)*detail['face_rate']<=1.)
    result['feature_handoff_strength']=strength
    return result


def event_energy(detail,window_seconds,*,fps,point_geometry,geometry_weight=1.):
    """A shorter observed strip cannot erase an instantaneous visual jump.

    The common six-frame perceptual unit is shared by native and authored
    boundaries. Only the window-dependent body evidence uses clipped support.
    Legacy experiments explicitly retain their previous integration.
    """
    focal=detail.get('focal_image_motion_relative_cost',0.)+detail.get('detailed_pose_relative_cost',0.)
    geometry=geometry_importance(geometry_weight)*detail['geometry_rate']
    if not point_geometry:
        return window_seconds*(geometry+detail['pose_rate']+detail['appearance_rate']+focal)
    return 6./fps*(geometry+detail['appearance_rate']+focal)+window_seconds*detail['pose_rate']


def ordinary_pairs(left,right):
    """Unpaired, already-declared camera paths are ordinary scene changes.

    A failed match must not REQUIRE an unmeasured-face/legs insert to get
    around a face-match gate. Ordinary cuts pay the full finite discrepancy;
    they get neither a successful-match claim nor a paired action benefit.
    The interpretation is fixed by the proposal origin, not its fitted error.
    """
    a=np.asarray([not any(i[0]=='paired' for i in p.proposal_ids) for p in left])
    b=np.asarray([not any(i[0]=='paired' for i in p.proposal_ids) for p in right])
    return a[:,None]&b[None,:]
