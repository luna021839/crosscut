"""Evaluation-only H at the union's exact physical event timestamps.

No new authoring nodes, source inference, nearby pose substitution or actor
certification. Native faces are a displayed-pixel focal proxy, shared by
identical source/camera choices, NOT head0 borrowed as a named/local actor.
Unmeasured geometry has finite learned-score bounds, not perfect residual0.
The upper regret is explicitly pessimistic, NOT measured perceptual badness.
"""
import hashlib
import numpy as np

from .event_ports import rho
from .fixed_holds import _curve_at
from .preference_features import FEATURE_NAMES,camera_context_at_tick
from .preference_model import regret_bounds


def _physical_ids(p):
    return [str(s.source)+':'+hashlib.sha256(row['ticks'].tobytes()+row['curve'].tobytes()).hexdigest()
        for s,row in zip(p.states,p.data)]


def _focal(heads,camera):
    centers=.5+camera[0]*(heads[:,:2]-camera[1:]);height=heads[:,2]*camera[0]
    half=height[:,None]*[.45*9/16,.5]
    extent=np.maximum(0.,np.minimum(1.,centers+half)-np.maximum(0.,centers-half))
    area=np.prod(extent,axis=1);salience=area*np.exp(-((centers[:,0]-.5)/.4)**2)
    visible=(heads[:,2]>.019)&(heads[:,3]>=.75)&(area>0)
    if not visible.any():return None
    salience=np.where(visible,salience,0.);at=int(salience.argmax())
    sizes=height[visible]
    formation=len(sizes)>=3 and sizes.max()<1.6*np.sort(sizes)[-2]
    return at,centers[at],height[at],formation


def prepare(p,model,*,native_ports,source_events,check):
    q,N=len(p.states),len(p.nodes);F=len(FEATURE_NAMES);index={k:i for i,k in enumerate(FEATURE_NAMES)}
    physical_ids=_physical_ids(p);representatives={}
    for i,key in enumerate(physical_ids):representatives.setdefault(key,i)
    raw_events=sorted({int(t) for bank in native_ports for t in bank['frames']}
        |{int(t) for events in source_events for t in events})
    authored=set(map(int,p.nodes));events=[t for t in raw_events if 0<t<p.nodes[-1] and t not in authored]
    lower_interval=np.zeros((q,N-1));upper_interval=np.zeros_like(lower_interval)
    known=0;lookups=[]
    for bank in native_ports:
        lookup={}
        for i,t in enumerate(bank['frames']):lookup.setdefault(int(t),[]).append(i)
        lookups.append(lookup)
    for tick in events:
        check();lo=np.zeros((q,F));hi=np.ones((q,F));lo[:,[16,18]]=-1.
        def lock(name,value):lo[:,index[name]]=value;hi[:,index[name]]=value
        for name,value in (('phase_unmeasured',1.),('phase_contradicted',0.),('motion_unmeasured',1.),
                ('motion_contradicted',0.),('pixel_agreed_active_novelty',0.),('appearance_unmeasured',1.),
                ('recurrent_member_hint_agreement',0.),('face_comparison_observed',0.)):
            lock(name,value)
        for name in FEATURE_NAMES:
            if name.startswith(('pixel_agreed_','pixel_contradicted_')):lock(name,0.)
        context=camera_context_at_tick(p,tick)
        for j,name in enumerate(FEATURE_NAMES[53:59]):lock(name,context[:,j])
        by_physical={}
        for key,i in representatives.items():
            state,row=p.states[i],p.data[i];s=state.source
            low,high=lo[i].copy(),hi[i].copy()
            def exact(name,value):low[index[name]]=high[index[name]]=value
            cuts=np.asarray(source_events[s]);central=bool(np.any(cuts==tick) or tick in lookups[s])
            exact('outgoing_known_native_cut',float(central));exact('incoming_known_native_cut',float(central))
            if not central:
                # Exact relation of the SAME-CLOCK self-H encoder, not a
                # measurement or imputation of unseen faces/limbs. Geometric
                # comparison is A[t] versus itself; exposed-face fallback is
                # allowed only at a known central native cut. Actor continuity
                # may still be unknown. Body presence, identity, activity and
                # visibility stay bounded, not certified as good motion.
                for name in ('face_position','face_scale','arm_phase','torso_phase'):
                    exact(name,0.)
                for name in ('unmeasured_arm_phase','unmeasured_torso_phase'):
                    exact(name,0.)
            # Motion compares each source's SAME physical velocity window to
            # itself, even if a separate native cut hides another window.
            # The encoder keeps actual pre/post movement in distinct channels.
            for name in ('motion','observed_opposition','unmeasured_motion'):
                exact(name,0.)
            for side,(a,b) in enumerate(((tick-7,tick-1),(tick,tick+6))):
                if np.any((cuts>a)&(cuts<=b)):exact('played_pre_physical_window' if side==0 else 'played_post_physical_window',0.)
            cameras=_curve_at(row['ticks'],row['curve'],np.array([tick-1,tick]))
            if np.array_equal(cameras,np.tile([1.,.5,.5],(2,1))):exact('added_crop',0.)
            found=lookups[s].get(tick,[]);bank=native_ports[s]
            observed=(len(found)==1 and bool(bank.get('observed',np.ones(len(bank['frames']),bool))[found[0]]))
            if observed:
                at=found[0];heads=np.asarray(bank['heads'][at]);a,b=(_focal(heads[side],cameras[side]) for side in (0,1))
                if a is not None and b is not None:
                    ia,ca,ha,ga=a;ib,cb,hb,gb=b;known+=1
                    exact('face_position',float(rho(np.linalg.norm((ca-cb)*[16/9,1.])/max(.005,(ha+hb)/2.))))
                    exact('face_scale',float(rho(abs(np.log(ha/hb)))))
                    exact('face_comparison_observed',1.);exact('featured_pair',float(not(ga or gb)))
                    high[index['local_availability']]=min(1.,ha/.22,hb/.22)
                    vectors=bank.get('vectors')
                    if vectors is not None and max(ia,ib)<3:
                        va,vb=vectors[at,0,ia],vectors[at,1,ib]
                        if min(np.linalg.norm(va),np.linalg.norm(vb))>=.5:
                            exact('anonymous_appearance_cosine',float(np.clip(va@vb,-1.,1.)))
                            exact('appearance_unmeasured',0.)
            by_physical[key]=(low,high)
        for i,key in enumerate(physical_ids):lo[i],hi[i]=by_physical[key]
        a,b=model.score_bounds(lo,hi)
        lower,upper=regret_bounds(a,b,physical_ids=physical_ids)
        cell=int(np.searchsorted(p.nodes,tick)-1)
        lower_interval[:,cell]+=lower;upper_interval[:,cell]+=upper
    return upper_interval,lower_interval,dict(evaluation_only_native_events=events,
        authored_native_events_not_double_counted=sum(0<t<p.nodes[-1] and t in authored for t in raw_events),
        unique_physical_source_camera_options=len(representatives),evaluated_states=q,
        exact_displayed_face_proxy_choices=known,source_neural_calls=0,source_decode=0,
        uncertainty_policy='finite_learned_linear_worst_regret_upper_bound_v1',
        self_regret_zero_and_physical_copies_correlated=True,
        no_central_native_self_H_geometric_error_is_encoder_identity=True,
        unmeasured_actor_body_identity_or_visibility_not_imputed=True,
        physical_face_proxy_is_not_actor_identity_binding=True,
        unseen_motion_not_measured_as_good_or_bad=True,full_perceptual_priority_certified=False,
        authoring_grid_and_camera_domain_expanded=False,
        upper_prefix_bytes=upper_interval.nbytes,lower_prefix_bytes=lower_interval.nbytes)
