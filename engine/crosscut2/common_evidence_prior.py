"""Explicit untrained C trial: local admission + true played-event support.

Continuous H is not event regret. Content/own-motion integrate in time;
locally inadmissible K cannot be purchased by later content utility. This
declared editorial rule is not a claim about global dominance or human taste.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .connection_prior import ConnectionPrior,quality
from .common_evidence_connection import scores
from .played_event_support import PlayedEventSupport,NativeHoldCosts
from .observed_motion_utility import integrals
from .native_preference import _focal,_physical_ids
from .fixed_holds import _curve_at
from .event_ports import rho


class CommonEvidencePrior(ConnectionPrior):
    unified_content=True

    def __init__(self):
        super().__init__()
        self.artifact.update(contract='declared_common_evidence_signed_connection_C_v1',
            continuous_H_event_cost=0.,admission='Q_E(K)>=min(Q_E(H_A),Q_E(H_B)); explicit editorial prior, NOT dominance',
            unsupported_pose_prior_symmetric=.5,raw_model_only_or_contradicted_pose_is_not_hard_veto=True,
            authored_cost='played_source_pair_weight*(min(common_H_A,common_H_B)-common_K)',
            hold_objective='time-integral(content-V*capped_own_activity_max_union)+actual_fragments+played_native',
            native_cost='state-sized hold evaluator with first/last actual event support corrections',
            self_style_event_reward=0.,same_clock_raw_encoder_preserved=True,
            post_selection_retiming_or_refitting=False)
        paths=[Path(__file__),Path(__file__).with_name('common_evidence_connection.py'),
               Path(__file__).with_name('played_event_support.py'),Path(__file__).with_name('observed_motion_utility.py')]
        self.artifact['implementation_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
        self.artifact.pop('content_sha256',None)
        self.artifact['content_sha256']=hashlib.sha256(json.dumps(self.artifact,sort_keys=True).encode()).hexdigest()
        self.statistics=[]

    def bind(self,p,*,native_ports,source_events):
        workspace=len(p.states)**2*8*64+len(p.states)*int(p.nodes[-1])*8
        if workspace>200*1024**2:raise ValueError('declared common-view scratch workspace exceeds200MiB')
        events=[sorted(set(map(int,row))|set(map(int,bank['frames']))) for row,bank in zip(source_events,native_ports)]
        self.support=PlayedEventSupport(events,final=int(p.nodes[-1]),fps=p.fps)
        self.sources=np.asarray([s.source for s in p.states]);self.nodes=p.nodes
        return dict(source_native_event_counts=list(map(len,self.support.events)),
            support_conditioned_on_actual_played_sources=True,unplayed_source_can_change_fixed_path_cost=False,
            minimum_authored_hold_frames=12,support_cap_frames=6,
            declared_scratch_workspace_bytes=workspace,
            grid_sha256=hashlib.sha256(p.nodes.tobytes()).hexdigest(),
            same_event_support_shared_by_all_options=False,handcrafted_not_preference_fitted=True)

    def costs_and_mask(self,p,n):
        K,base,evidence=scores(p,n);different=self.sources[:,None]!=self.sources[None,:]
        allowed=different&(K>=base)
        self.statistics.append(dict(frame=int(p.nodes[n]),allowed_state_pairs=int(allowed.sum()),
            common_phase_pairs=int((different&evidence['common_phase']).sum()),
            common_motion_pairs=int((different&evidence['common_motion']).sum())))
        costs=self.support.cut_weights(int(p.nodes[n]),self.sources)*(base-K)
        return costs,allowed

    def content_reward(self,p,*,check):return integrals(p,check=check)

    def native_holds(self,p,*,native_ports,check):
        physical=_physical_ids(p);cache={};rates=[];observed=unknown=0
        for state,row,key in zip(p.states,p.data,physical):
            check()
            if key in cache:rates.append(cache[key]);continue
            bank=native_ports[state.source];lookup={int(t):i for i,t in enumerate(bank['frames'])};values=[]
            for tick in self.support.events[state.source]:
                position=scale=visibility=.5;found=lookup.get(int(tick))
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
                        visibility=.5*min(1.,ha/.22,hb/.22)*min(fractions);observed+=1
                    else:unknown+=1
                else:unknown+=1
                Q=quality(position=position,scale=scale,phase=.5,motion=.5,crop=0.,visibility=visibility,
                    own=0.,connection=0.,music=0.,style=0.)
                values.append(1.-float(Q))
            cache[key]=np.asarray(values);rates.append(cache[key])
        evaluator=NativeHoldCosts(self.support,self.sources,rates)
        def hold(first,final):return evaluator(int(p.nodes[first]),int(p.nodes[final]))
        return hold,dict(contract='played_state_native_hold_endpoint_corrected_v1',
            events_on_or_off_observation_grid_use_same_evaluator=True,strictly_internal_native_events_only=True,
            exact_displayed_face_proxy_events=observed,unmeasured_face_proxy_events=unknown,
            native_body_motion_and_sharpness_not_measured=True,source_NN=0,source_decode=0,
            actor_or_anatomy_certified=False,professional_quality_approved=False)
