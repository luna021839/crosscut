"""Uncalibrated raw observations for SAME-event hold/cut preferences.

No fitted aesthetic coefficients, names, private vectors or probabilities.
Geometry is never multiplied by a reliability number. Evidence category and
physical continuity remain separate inputs; a valid pixel path is not anatomy.
"""
import numpy as np
from .event_ports import rho

FEATURE_NAMES=('face_position','face_scale','arm_phase','torso_phase','motion',
    'observed_opposition','phase_unmeasured','phase_contradicted','motion_unmeasured',
    'motion_contradicted','pixel_agreed_active_novelty','local_availability',
    'added_crop','featured_pair','style_novelty','event_strength',
    'anonymous_appearance_cosine','appearance_unmeasured','recurrent_member_hint_agreement',
    'raw_model_angular_activity','raw_model_screen_activity',
    'raw_model_angular_style','raw_model_screen_style')+tuple(
    f'{evidence}_{kind}' for kind in ('arm_phase','torso_phase','motion')
    for evidence in ('unmeasured','pixel_contradicted','pixel_agreed'))+tuple(
    f'{evidence}_{kind}_geometry' for kind in ('arm_phase','torso_phase','motion')
    for evidence in ('unmeasured','pixel_contradicted','pixel_agreed'))+(
    'played_pre_angular_activity','played_post_angular_activity',
    'played_pre_screen_activity','played_post_screen_activity',
    'played_pre_motion_geometry','played_post_motion_geometry','face_comparison_observed',
    'played_pre_physical_window','played_post_physical_window',
    'outgoing_known_native_cut','incoming_known_native_cut','event_active_style',
    'pre_added_camera_zoom_travel','pre_added_camera_x_travel','pre_added_camera_y_travel',
    'post_added_camera_zoom_travel','post_added_camera_x_travel','post_added_camera_y_travel')
SCHEMA='raw_same_event_preference_features_v4'


def camera_context(p,n):
    """Known actual added-camera curves, not invented intervening actor poses.

    The1s half-windows are those of the rendered annotation comparisons.
    Original broadcaster camera motion/actor occlusion between observations
    remains unmeasured. Equal physical curves share their computation.
    """
    return camera_context_at_tick(p,int(p.nodes[n]))


def camera_context_at_tick(p,tick):
    cached={};result=[];handle=p.fps
    for state,row in zip(p.states,p.data):
        key=(state.source,row['ticks'].tobytes(),row['curve'].tobytes())
        if key not in cached:
            halves=[];ticks=row['ticks'];curve=row['curve']
            for a,b in ((max(0.,tick-handle),tick),(tick,min(p.nodes[-1]-1.,tick+handle))):
                query=np.unique(np.r_[a,ticks[(ticks>a)&(ticks<b)],b])
                values=np.stack([np.interp(query,ticks,curve[:,j]) for j in range(3)],axis=-1)
                values[:,0]=np.log(values[:,0])
                halves.append(rho(np.abs(np.diff(values,axis=0)).sum(0)))
            cached[key]=np.r_[halves[0],halves[1]]
        result.append(cached[key])
    return np.asarray(result)


def extract(ports,n):
    p=ports;q=len(p.states);shape=(q,q);P=len(p.pairs)
    # Only THIS event is requested. Rebuilding the whole-song geometry mask
    # at every node makes sparse fitting/selection accidentally quadratic.
    body=p.body[:,n]
    raw_valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    geometry=raw_valid[...,p.pairs].all(-1)
    physical=np.asarray([d['physical_window_valid'][n] for d in p.data],bool)
    scope=np.asarray([d.get('pixel_part_scope',np.zeros(len(p.nodes),bool))[n] for d in p.data],bool)
    result=np.zeros(shape+(len(FEATURE_NAMES),),np.float32)
    phase_seen=np.zeros(shape+(2,P),bool);phase_evidence=np.ones(shape+(2,P),np.int8)
    motion_seen=np.zeros_like(phase_seen);motion_evidence=np.ones_like(phase_evidence)
    face_seen=np.zeros(shape,bool)
    active_agreed=np.zeros(shape,bool);motion_worst=np.zeros(shape);opposition=np.zeros(shape)
    angular_activity=np.zeros(shape);screen_activity=np.zeros(shape)
    def categories(weight,common,scoped):
        return np.where(common&scoped,np.where(weight==1.,3,np.where(weight==0.,2,1)),1)
    def retain_error(kind,residual,evidence,common):
        # A global worst residual + global evidence fractions loses WHICH
        # primitive's error was agreed/contradicted/unknown. Keep every raw
        # residual in its own discrete-evidence channel, never attenuate it.
        for category in (1,2,3):
            k=23+3*(kind-2)+category-1
            result[:,:,k]=np.maximum(result[:,:,k],np.where(common&(evidence==category),residual,0.))
    for side,phase in ((0,1),(1,2)):
        head=p.head[:,n,side];camera=p.camera[:,n,side]
        center=.5+camera[:,:1]*(head[:,:2]-camera[:,1:]);size=head[:,2]*camera[:,0]
        scale=np.maximum(.005,(size[:,None]+size[None,:])/2.)
        face=head[:,2]>.019;usable=face[:,None]&face[None,:]
        usable&=p.continuous[:,n][None,:] if side==0 else p.continuous[:,n][:,None]
        position=rho(np.linalg.norm((center[:,None]-center[None,:])*[16/9,1.],axis=-1)/scale)
        ratio=rho(np.abs(np.log(np.maximum(.001,size[:,None])/np.maximum(.001,size[None,:]))))
        result[:,:,0]=np.maximum(result[:,:,0],np.where(usable,position,0.))
        result[:,:,1]=np.maximum(result[:,:,1],np.where(usable,ratio,0.))
        face_seen|=usable
        for k,(a,b) in enumerate(p.pairs):
            common=usable&geometry[:,phase,k,None]&geometry[:,phase,k][None,:]
            point=p.projected[:,n,phase,b,:2]
            screen=np.linalg.norm((point[:,None]-point[None,:])*[16/9,1.],axis=-1)/scale
            angle=np.linalg.norm(p.bones[:,None,n,phase,k]-p.bones[None,:,n,phase,k],axis=-1)/2.
            residual=rho(np.maximum(screen,angle));target=2 if max(a,b)<6 else 3
            result[:,:,target]=np.maximum(result[:,:,target],np.where(common,residual,0.))
            phase_seen[:,:,side,k]=common
            weight=np.minimum(p.phase_certainty[:,n,phase,k,None],p.phase_certainty[:,n,phase,k][None,:])
            phase_evidence[:,:,side,k]=categories(weight,common,scope[:,None]&scope[None,:])
            retain_error(target,residual,phase_evidence[:,:,side,k],common)
    # At a native cut the counterfactual same-clock source is unavailable.
    # Keep actual exposed endpoints and unknown motion, not invented poses.
    no_phase=~phase_seen.any((-1,-2))
    native_break=np.asarray([d.get('known_center_native_cut',np.zeros(len(p.nodes),bool))[n] for d in p.data],bool)
    exposed_face_fallback=~face_seen&(native_break[:,None]|native_break[None,:])
    h1,h2=p.head[:,n,0],p.head[:,n,1];c1,c2=p.camera[:,n,0],p.camera[:,n,1]
    size1,size2=h1[:,2]*c1[:,0],h2[:,2]*c2[:,0]
    scale=np.maximum(.005,(size1[:,None]+size2[None,:])/2.)
    exposed=(h1[:,2]>.019)[:,None]&(h2[:,2]>.019)[None,:]
    center1=.5+c1[:,:1]*(h1[:,:2]-c1[:,1:]);center2=.5+c2[:,:1]*(h2[:,:2]-c2[:,1:])
    result[:,:,0]=np.maximum(result[:,:,0],np.where(exposed_face_fallback&exposed,
        rho(np.linalg.norm((center1[:,None]-center2[None,:])*[16/9,1.],axis=-1)/scale),0.))
    result[:,:,1]=np.maximum(result[:,:,1],np.where(exposed_face_fallback&exposed,
        rho(np.abs(np.log(np.maximum(.001,size1[:,None])/np.maximum(.001,size2[None,:])))),0.))
    for k,(a,b) in enumerate(p.pairs):
        common=no_phase&exposed&geometry[:,1,k,None]&geometry[:,2,k][None,:]
        pa,pb=p.projected[:,n,1,b,:2],p.projected[:,n,2,b,:2]
        screen=np.linalg.norm((pa[:,None]-pb[None,:])*[16/9,1.],axis=-1)/scale
        angle=np.linalg.norm(p.bones[:,None,n,1,k]-p.bones[None,:,n,2,k],axis=-1)/2.
        target=2 if max(a,b)<6 else 3
        result[:,:,target]=np.maximum(result[:,:,target],np.where(common,rho(np.maximum(screen,angle)),0.))
        weight=np.minimum(p.phase_certainty[:,n,1,k,None],p.phase_certainty[:,n,2,k][None,:])
        category=categories(weight,common,scope[:,None]&scope[None,:])
        phase_evidence[:,:,0,k]=np.where(common,category,phase_evidence[:,:,0,k])
        phase_seen[:,:,0,k]|=common
        retain_error(target,rho(np.maximum(screen,angle)),category,common)
    for window in (0,1):
        early,late=(0,1) if window==0 else (2,3)
        h=p.head[:,n,window,2]*p.camera[:,n,window,0]
        scale=np.maximum(.005,(h[:,None]+h[None,:])/2.)
        screen=(p.projected[:,n,late,:,:2]-p.projected[:,n,early,:,:2])*[16/9,1.]
        for k,joint in enumerate(p.destinations):
            valid=geometry[:,early,k]&geometry[:,late,k]&physical[:,window]
            common=valid[:,None]&valid[None,:]
            common&=p.continuous[:,n][None,:] if window==0 else p.continuous[:,n][:,None]
            a,b=p.velocity[:,None,n,window,k],p.velocity[None,:,n,window,k]
            na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
            moving=np.maximum(na,nb)>=.3;both=np.minimum(na,nb)>=.15
            direction=np.where(both,(1.-np.clip((a*b).sum(-1)/np.maximum(1e-9,na*nb),-1,1))/2.,1.)
            direction=np.where(moving,direction,0.)
            speed=rho(np.abs(np.log((na+.15)/(nb+.15))))
            camera_motion=rho(np.linalg.norm(screen[:,None,joint]-screen[None,:,joint],axis=-1)/scale)
            residual=np.maximum(np.maximum(direction,speed),camera_motion)
            motion_worst=np.maximum(motion_worst,np.where(common,residual,0.))
            # Two equally moving estimates have zero *mismatch*, just like
            # two stationary estimates. Preserve raw activity on THIS common
            # primitive/window separately. It is not pixel/ownership credit.
            angular_activity=np.maximum(angular_activity,np.where(common,rho(np.minimum(na,nb)),0.))
            screen_speed=np.linalg.norm(screen[:,joint],axis=-1)/(6/p.fps)
            screen_activity=np.maximum(screen_activity,np.where(common,
                rho(np.minimum(screen_speed[:,None],screen_speed[None,:])/scale),0.))
            weight=np.minimum(p.motion_certainty[:,n,window,k,None],p.motion_certainty[:,n,window,k][None,:])
            evidence=categories(weight,common,scope[:,None]&scope[None,:]);motion_seen[:,:,window,k]=common
            motion_evidence[:,:,window,k]=evidence
            retain_error(4,residual,evidence,common)
            opposition=np.maximum(opposition,np.where(common&(evidence==3)&moving,direction,0.))
            displayed=np.maximum(np.linalg.norm(screen[:,None,joint],axis=-1),
                np.linalg.norm(screen[None,:,joint],axis=-1))/scale>=.3
            active_agreed|=common&(moving|displayed)&(evidence==3)&(phase_evidence[:,:,window,k]==3)
    result[:,:,4]=motion_worst;result[:,:,5]=opposition
    result[:,:,6]=(phase_evidence==1).mean((-1,-2));result[:,:,7]=(phase_evidence==2).mean((-1,-2))
    result[:,:,8]=(motion_evidence==1).mean((-1,-2));result[:,:,9]=(motion_evidence==2).mean((-1,-2))
    # An unknown, actually measured torso and an absent torso must not share
    # the same zero-residual representation. Keep geometry/evidence together.
    for kind in (2,3,4):
        included=np.asarray([max(a,b)<6 if kind==2 else max(a,b)>=6 for a,b in p.pairs]) if kind!=4 else np.ones(P,bool)
        seen,evidence=(motion_seen,motion_evidence) if kind==4 else (phase_seen,phase_evidence)
        for category in (1,2,3):
            at=32+3*(kind-2)+category-1
            result[:,:,at]=((seen&(evidence==category))[:,:,:,included].mean((-1,-2)) if included.any() else 0.)
    ca,cb=p.colour[:,n,0],p.colour[:,n,1];style=np.zeros(shape)
    for region in (0,1):
        style+=.5*np.sqrt(np.maximum(0.,1.-np.einsum('ik,jk->ij',ca[:,region],cb[:,region],optimize=False)))
    style=np.where(p.colour_valid[:,n,0,None]&p.colour_valid[:,n,1][None,:],style,0.)
    result[:,:,10]=active_agreed*style
    result[:,:,11]=np.minimum(p.availability[:,n,0,None],p.availability[:,n,1][None,:])
    result[:,:,12]=np.maximum(p.crop[:,n,0,None],p.crop[:,n,1][None,:])
    result[:,:,13]=p.dominant[:,n,0,None]&p.dominant[:,n,1][None,:]
    result[:,:,14]=style;result[:,:,15]=p.strength[n]
    va,vb=p.vectors[:,n,0],p.vectors[:,n,1]
    appearance=(np.linalg.norm(va,axis=-1)>=.5)[:,None]&(np.linalg.norm(vb,axis=-1)>=.5)[None,:]
    result[:,:,16]=np.where(appearance,np.clip(np.einsum('ik,jk->ij',va,vb,optimize=False),-1.,1.),0.)
    result[:,:,17]=~appearance
    ma,mb=p.member_hint[:,n,0],p.member_hint[:,n,1]
    hinted=(ma[:,None]>=0)&(mb[None,:]>=0)
    result[:,:,18]=np.where(hinted,np.where(ma[:,None]==mb[None,:],1.,-1.),0.)
    result[:,:,19]=angular_activity;result[:,:,20]=screen_activity
    # Explicit raw interactions let the small linear ranker represent a
    # moving style change, without requiring scarce pixel-agreed anatomy or
    # treating every clothing/background change as a moving connection.
    result[:,:,21]=angular_activity*style;result[:,:,22]=screen_activity*style
    for side,window in ((0,0),(1,1)):
        early,late=(0,1) if window==0 else (2,3)
        valid=geometry[:,early]&geometry[:,late]&physical[:,window,None]
        angular=np.where(valid,np.linalg.norm(p.velocity[:,n,window],axis=-1),0.).max(-1)
        delta=(p.projected[:,n,late,p.destinations,:2]-p.projected[:,n,early,p.destinations,:2])*[16/9,1.]
        height=np.maximum(.005,p.head[:,n,window,2]*p.camera[:,n,window,0])
        screen=np.where(valid,np.linalg.norm(delta,axis=-1)/height[:,None]/(6/p.fps),0.).max(-1)
        expand=lambda a:a[:,None] if side==0 else a[None,:]
        result[:,:,41+side]=expand(rho(angular));result[:,:,43+side]=expand(rho(screen))
        result[:,:,45+side]=expand(valid.mean(-1));result[:,:,48+side]=expand(physical[:,window])
        result[:,:,50+side]=expand(native_break)
    result[:,:,47]=face_seen|(exposed_face_fallback&exposed)
    result[:,:,52]=p.strength[n]*angular_activity*style
    context=camera_context(p,n)
    result[:,:,53:56]=context[:,None,:3];result[:,:,56:59]=context[None,:,3:]
    if not np.isfinite(result).all():raise ValueError('finite raw preference observations required')
    return result
