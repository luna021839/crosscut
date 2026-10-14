"""One declared, UNTRAINED monotone face/action prior for a frozen B trial.

Weak source attribution is not a hold/cut aesthetic preference. This prior is
handcrafted, not calibrated human preference, anatomy or professional quality.
Residuals are never attenuated by evidence; caps affect positive activity only.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .event_ports import rho
from .preference_features import FEATURE_NAMES,SCHEMA
from .fixed_holds import _curve_at
from .native_preference import _focal,_physical_ids


def quality(*,position,scale,phase,motion,crop,visibility,own,connection,music,style,identity_conflict=False):
    """All residual/visibility arguments are bounded raw values, not weights."""
    G=(1-position)*(1-scale)*(1-phase)*(1-motion)*(1-crop)
    bonus=np.where(identity_conflict,0.,own+music*style*connection)
    return visibility*G*(1+G*bonus)


def physical_weights(events,*,final,fps):
    events=np.asarray(sorted(set(int(t) for t in events if 0<t<final)),int)
    if not len(events):return {}
    left=np.r_[events[0],np.diff(events)/2.]
    right=np.r_[np.diff(events)/2.,final-events[-1]]
    widths=np.minimum(6.,np.minimum(left,right))/fps
    return {int(t):float(w) for t,w in zip(events,widths)}


def activity(p,n):
    """Own motion and matching SAME-primitive/window positive evidence.

    A stable background wrist can still be the wrong anatomy. Pixel agreement
    is a cap1, not ownership certification; geometry-only cap.5 is a prior.
    """
    q=len(p.states);body=p.body[:,n]
    valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    geometry=valid[...,p.pairs].all(-1)
    physical=np.asarray([r['physical_window_valid'][n] for r in p.data],bool)
    scope=np.asarray([r.get('pixel_part_scope',np.zeros(len(p.nodes),bool))[n] for r in p.data],bool)
    own=[];paired=np.zeros((q,q))
    for window,(early,late,phase) in enumerate(((0,1,1),(2,3,2))):
        good=geometry[:,early]&geometry[:,late]&physical[:,window,None]
        angular=rho(np.linalg.norm(p.velocity[:,n,window],axis=-1))
        delta=(p.projected[:,n,late,p.destinations,:2]-p.projected[:,n,early,p.destinations,:2])*[16/9,1.]
        height=np.maximum(.005,p.head[:,n,window,2]*p.camera[:,n,window,0])
        screen=rho(np.linalg.norm(delta,axis=-1)/height[:,None]/(6/p.fps))
        raw=np.maximum(angular,screen)
        evidence=np.minimum(p.phase_certainty[:,n,early],p.phase_certainty[:,n,late])
        evidence=np.minimum(evidence,p.motion_certainty[:,n,window])
        cap=np.where(evidence<=0.,0.,np.where(scope[:,None]&(evidence==1.),1.,.5))
        own.append(np.where(good,raw*cap,0.).max(-1))
        for k in range(len(p.pairs)):
            common=good[:,k,None]&good[:,k][None,:]&geometry[:,phase,k,None]&geometry[:,phase,k][None,:]
            common&=p.continuous[:,n][None,:] if window==0 else p.continuous[:,n][:,None]
            phase_evidence=np.minimum(p.phase_certainty[:,n,phase,k,None],p.phase_certainty[:,n,phase,k][None,:])
            motion_evidence=np.minimum(p.motion_certainty[:,n,window,k,None],p.motion_certainty[:,n,window,k][None,:])
            evidence=np.minimum(phase_evidence,motion_evidence)
            cap=np.where(evidence<=0.,0.,np.where(scope[:,None]&scope[None,:]&(evidence==1.),1.,.5))
            paired=np.maximum(paired,np.where(common,np.minimum(raw[:,k,None],raw[:,k][None,:])*cap,0.))
    return np.minimum(own[0][:,None],own[1][None,:]),paired


class ConnectionPrior:
    def __init__(self):
        self.artifact=dict(contract='declared_monotone_connection_prior_B_v1',feature_schema=SCHEMA,
            formula='V*G*(1+G*(Aown+m*S*Aconnection)); G=product(1-raw_residual)',
            residuals=['face_position','face_scale','phase','motion','crop'],
            missing_whole_channel_prior=.5,unmeasured_activity_cap=.5,pixel_agreed_activity_cap=1.,
            contradicted_activity_cap=0.,recurrent_identity_conflict_removes_positive_bonus_only=True,
            event_support_frames=6,fragment_support_frames=12,all_primary_coefficients=1.,
            native_cost='event_weight*(1-Q_native); local own central events only',
            native_unmeasured_body_motion_and_pixel_quality_prior=.5,
            weak_models_used=False,new_fit=False,supervision_kinds=[],
            source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            professional_quality_approved=False,normal_app_activation_approved=False)
        self.artifact['content_sha256']=hashlib.sha256(json.dumps(self.artifact,sort_keys=True).encode()).hexdigest()

    def bind(self,p,*,native_ports,source_events):
        events={int(t) for t,m in zip(p.nodes,p.strength) if m>0}
        events.update(int(t) for row in source_events for t in row)
        events.update(int(t) for row in native_ports for t in row['frames'])
        self.weights=physical_weights(events,final=int(p.nodes[-1]),fps=p.fps)
        self.hold_weight=np.array([self.weights.get(int(t),0.) for t in p.nodes])
        self.cut_weight=np.array([self.weights.get(int(t),physical_weights(events|{int(t)},
            final=int(p.nodes[-1]),fps=p.fps).get(int(t),0.)) for t in p.nodes])
        from . import preference_features,connection_observations
        physical=sorted(set(_physical_ids(p))) if hasattr(p,'data') else []
        return dict(physical_event_frames=sorted(self.weights),physical_event_weight_seconds=self.weights,
            grid_sha256=hashlib.sha256(p.nodes.tobytes()).hexdigest(),
            physical_camera_domain_sha256=hashlib.sha256(json.dumps(physical).encode()).hexdigest(),
            raw_feature_implementation_sha256=hashlib.sha256(Path(preference_features.__file__).read_bytes()).hexdigest(),
            exact_projection_implementation_sha256=hashlib.sha256(Path(connection_observations.__file__).read_bytes()).hexdigest(),
            neutral_authoring_nodes=int(np.sum(self.hold_weight[1:-1]==0)),
            same_event_support_shared_by_all_options=True,handcrafted_not_preference_fitted=True)

    def scores_at(self,p,n,features):
        f=features;at={k:i for i,k in enumerate(FEATURE_NAMES)};q=len(p.states)
        phase_seen=sum(f[:,:,at[k]] for k in FEATURE_NAMES[32:38])>0
        motion_seen=sum(f[:,:,at[k]] for k in FEATURE_NAMES[38:41])>0
        phase=np.where(phase_seen,np.maximum(f[:,:,2],f[:,:,3]),.5)
        motion=np.where(motion_seen,f[:,:,4],.5)
        face=f[:,:,at['face_comparison_observed']]>0
        position=np.where(face,f[:,:,0],.5);scale=np.where(face,f[:,:,1],.5)
        central=np.asarray([r['known_center_native_cut'][n] for r in p.data])
        uninterrupted=np.eye(q,dtype=bool)&~central[:,None]
        position=np.where(uninterrupted,0.,position);scale=np.where(uninterrupted,0.,scale)
        phase=np.where(uninterrupted,0.,phase);motion=np.where(uninterrupted,0.,motion)
        own,connection=activity(p,n)
        scores=quality(position=position,scale=scale,phase=phase,motion=motion,crop=f[:,:,12],
            visibility=f[:,:,11],own=own,connection=connection,music=p.strength[n],style=f[:,:,14],
            identity_conflict=f[:,:,18]<0)
        return scores

    def losses_at(self,p,n,features):
        return self._losses(p,n,self.scores_at(p,n,features))

    def scores_from_ports(self,p,n):
        from .connection_observations import extract
        values=extract(p,n);own,connection=activity(p,n)
        return quality(**values,own=own,connection=connection,music=p.strength[n])

    def losses_from_ports(self,p,n):
        return self._losses(p,n,self.scores_from_ports(p,n))

    def _losses(self,p,n,scores):
        q=len(p.states)
        sources=np.asarray([s.source for s in p.states]);allowed=(sources[:,None]!=sources[None,:])|np.eye(q,dtype=bool)
        best=float(scores[allowed].max());loss=np.maximum(0.,best-scores)
        return np.diag(loss)*self.hold_weight[n],loss*self.cut_weight[n]

    def native_intervals(self,p,*,native_ports,source_events,check):
        """Only a played source's own central event; no cross-option boxes."""
        result=np.zeros((len(p.states),len(p.nodes)-1));physical_ids=_physical_ids(p)
        cache={};authored=set(map(int,p.nodes));observed=unknown=0
        for i,(state,row,key) in enumerate(zip(p.states,p.data,physical_ids)):
            check()
            if key in cache:result[i]=cache[key];continue
            bank=native_ports[state.source];lookup={int(t):j for j,t in enumerate(bank['frames'])}
            for tick in sorted(set(map(int,source_events[state.source]))|set(lookup)):
                if not 0<tick<p.nodes[-1] or tick in authored:continue
                position=scale=.5;visibility=.5;crop=0.;found=lookup.get(tick)
                if found is not None and bool(bank.get('observed',np.ones(len(bank['frames']),bool))[found]):
                    cameras=_curve_at(row['ticks'],row['curve'],np.array([tick-1,tick]))
                    flanks=[_focal(bank['heads'][found,side],cameras[side]) for side in (0,1)]
                    if all(v is not None for v in flanks):
                        a,b=flanks;_,ca,ha,_=a;_,cb,hb,_=b
                        position=float(rho(np.linalg.norm((ca-cb)*[16/9,1.])/max(.005,(ha+hb)/2.)))
                        scale=float(rho(abs(np.log(ha/hb))))
                        fractions=[]
                        for center,height in ((ca,ha),(cb,hb)):
                            half=height*np.array([.45*9/16,.5])
                            extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
                            fractions.append(float(np.prod(extent/np.maximum(1e-9,2*half))))
                        # Native flanks have exact face pixels/geometry but no
                        # measured sharpness here. Retain explicit neutral.5.
                        visibility=.5*min(1.,ha/.22,hb/.22)*min(fractions);observed+=1
                    else:unknown+=1
                else:unknown+=1
                score=quality(position=position,scale=scale,phase=.5,motion=.5,crop=crop,
                    visibility=visibility,own=0.,connection=0.,music=0.,style=0.)
                cell=int(np.searchsorted(p.nodes,tick)-1)
                result[i,cell]+=self.weights[tick]*(1.-float(score))
            cache[key]=result[i].copy()
        return result,dict(policy='local_own_central_native_cost_v1',weak_linear_worst_regret_used=False,
            exact_displayed_face_proxy_events=observed,unmeasured_face_proxy_events=unknown,
            native_body_motion_and_sharpness_not_measured=True,physical_actor_aliases_share_pixels=True,
            authored_overlap_charged_only_by_H_K=True,continuous_other_source_event_cost_zero=True,
            source_NN=0,source_decode=0,professional_quality_approved=False)
