"""Same-clock face AND moving-limb evidence, in bounded event loss seconds.

No call into the legacy complete-body/screen-extrapolation success gate. One
observed arm is useful; an actually opposed limb/window remains the worst
residual. Missing is finite uncertainty, never stationary motion or a veto.
"""
import numpy as np


def rho(value):
    value=np.maximum(0.,value)
    return value/(1.+value)


def event_widths(nodes,strength,*,fps,window_frames=6):
    nodes=np.asarray(nodes);strength=np.asarray(strength,float)
    if strength.shape!=nodes.shape or not np.isfinite(strength).all() or np.any((strength<0)|(strength>1)):
        raise ValueError('finite physical-event strengths required')
    events=np.flatnonzero(strength>0);width=np.zeros(len(nodes))
    for i,n in enumerate(events):
        left=nodes[n] if i==0 else (nodes[n]-nodes[events[i-1]])/2
        right=nodes[-1]-nodes[n] if i==len(events)-1 else (nodes[events[i+1]]-nodes[n])/2
        width[n]=min(window_frames,left,right)/fps
    width[[0,-1]]=0.
    return width


class Ports:
    def __init__(self,nodes,states,data,*,fps,music):
        self.nodes=np.asarray(nodes);self.fps=fps;self.states=states;self.data=data
        self.head=np.asarray([d['heads'] for d in data],np.float32)
        self.camera=np.asarray([d['cameras'] for d in data],np.float32)
        self.body=np.asarray([d['body'] for d in data],np.float32)
        self.projected=np.asarray([d['projected'] for d in data],np.float32)
        self.vectors=np.asarray([d['vectors'] for d in data],np.float32)
        self.continuous=np.asarray([d['continuous'] for d in data],bool)
        self.formation=np.asarray([d['formation'] for d in data],bool)
        self.availability=np.asarray([d['availability'] for d in data],np.float32)
        self.colour=np.sqrt(np.asarray([d['colour'] for d in data],np.float32))
        self.colour_valid=np.asarray([d['colour_valid'] for d in data],bool)
        self.crop=np.asarray([d['added_crop'] for d in data],np.float32)
        self.dominant=np.asarray([d['displayed_attention_dominant'] for d in data],bool)
        self.member_hint=np.asarray([d['local_member_hint'] for d in data],np.int16)
        q,N=len(states),len(nodes)
        if self.body.shape!=(q,N,4,8,3) or self.head.shape!=(q,N,2,4):
            raise ValueError('same-clock fixed action/camera states required')
        if (not all(np.isfinite(v).all() for v in (self.body,self.projected,self.head,self.camera,
                self.vectors,self.availability,self.colour,self.crop)) or np.any(self.camera[:,:,:,0]<1)
                or np.any((self.availability<0)|(self.availability>1))):
            raise ValueError('finite physical action/style/camera inputs required')
        valid=(self.body[...,2]>=.3)&(self.body[...,:2]>=0).all(-1)&(self.body[...,:2]<=1).all(-1)
        pixel_parts=any('part_pair_support' in d for d in data)
        # Pixel-corroborated shoulder/wrist endpoints remain useful when the
        # elbow has no texture. This chord is NOT an invented elbow/bone.
        pairs=np.array([[0,2],[2,4],[1,3],[3,5]]+([[0,4],[1,5]] if pixel_parts else []))
        self.pairs=pairs;self.destinations=tuple(map(int,pairs[:,1]))
        certainty=np.asarray([d.get('body_reliability',np.ones((N,4,8))) for d in data],np.float32)
        movement_certainty=np.asarray([d.get('motion_reliability',np.ones((N,2,8))) for d in data],np.float32)
        if certainty.shape!=(q,N,4,8) or movement_certainty.shape!=(q,N,2,8) or any(
                not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in (certainty,movement_certainty)):
            raise ValueError('finite separate body/trajectory reliability in[0,1] required')
        self.phase_certainty=certainty[...,pairs].min(-1)
        self.motion_certainty=movement_certainty[...,pairs].min(-1)
        xy=self.body[...,:2]*[16/9,1.]
        bones=xy[...,pairs[:,1],:]-xy[...,pairs[:,0],:];length=np.linalg.norm(bones,axis=-1)
        supported=valid[...,pairs].all(-1)&(length>=.025)
        self.arm_valid=supported if pixel_parts else np.repeat(supported.reshape(q,N,4,2,2).all(-1),2,axis=-1)
        self.bones=bones/np.maximum(1e-9,length[...,None])
        self.velocity=(self.bones[:,:,[1,3]]-self.bones[:,:,[0,2]])/(6/fps)
        self.velocity_valid=self.arm_valid[:,:,[1,3]]&self.arm_valid[:,:,[0,2]]
        # Native cuts/duplicate physical samples invalidate ONLY that window.
        for i,d in enumerate(data):
            legacy=d['arm_valid'][:,:,2:6]
            if pixel_parts:
                fallback=np.concatenate((legacy,legacy[:,:,[0,2]]),axis=-1)
                scope=np.asarray(d.get('pixel_part_scope',np.zeros(N,bool)))
                pair_support=np.asarray(d.get('part_pair_support',np.ones((N,3,8),bool)))
                actual=(self.motion_certainty[i]>0) if 'motion_reliability' in d else pair_support[:,[0,2]][:,:,pairs].all(-1)
                self.velocity_valid[i]&=np.where(scope[:,None,None],actual,fallback)
                # Uncorroborated legacy rows preserve their complete-arm
                # basis. A new partial endpoint cannot bypass an old missing
                # window merely because another row enabled the pixel bank.
                complete=np.repeat(supported[i,:,:,:4].reshape(N,4,2,2).all(-1),2,axis=-1)
                complete=np.concatenate((complete,complete[:,:,[0,2]]),axis=-1)
                self.arm_valid[i]&=np.where(scope[:,None,None],True,complete)
                self.velocity_valid[i]&=self.arm_valid[i][:,[1,3]]&self.arm_valid[i][:,[0,2]]
            else:self.velocity_valid[i]&=legacy
            self.arm_valid[i]&=self.phase_certainty[i]>0
            self.velocity_valid[i]&=self.motion_certainty[i]>0
            clock=d.get('pose_native_indices')
            if clock is None:self.velocity_valid[i]=False
            else:
                self.velocity_valid[i,:,0]&=(clock[:,0]!=clock[:,1])[:,None]
                self.velocity_valid[i,:,1]&=(clock[:,2]!=clock[:,3])[:,None]
        gesture=np.zeros((q,N))
        for window in (0,1):
            speed=np.linalg.norm(self.velocity[:,:,window],axis=-1)
            if window==0:before=speed
            else:after=speed
        both=self.velocity_valid[:,:,0]&self.velocity_valid[:,:,1]
        dot=(self.velocity[:,:,0]*self.velocity[:,:,1]).sum(-1)
        cosine=dot/np.maximum(1e-8,before*after)
        reversal=np.clip(-cosine,0,1)
        deceleration=1.-np.minimum(before,after)/np.maximum(.3,np.maximum(before,after))
        observed=both&(np.maximum(before,after)>=.3)&self.continuous[:,:,None]
        gesture=np.where(observed,np.maximum(reversal,deceleration),0.).max(-1)
        # A gesture event must be a local peak in physical time, not every
        # high-motion sample. Source/camera duplicates cannot add events.
        g=gesture.max(0);peaks=np.zeros(N)
        for n in range(1,N-1):
            near=np.flatnonzero(np.abs(nodes-nodes[n])<=6)
            if g[n]>0 and n==min(near,key=lambda k:(-g[k],int(nodes[k]))):peaks[n]=g[n]
        self.strength=np.maximum(np.asarray(music,float),peaks)
        self.strength[[0,-1]]=0.
        self.width=event_widths(nodes,self.strength,fps=fps)
        # A non-musical cut still exposes an observation window and pays its
        # real risk. Neutral grid nodes add no new event REWARD or steal an
        # existing physical event's duration.
        self.width=np.where(self.strength>0,self.width,6/fps);self.width[[0,-1]]=0.

    def exposed_partial(self,n):
        """Real outgoing/incoming poses when a native cut hides counterfactuals.

        An actually measured one-frame step envelope removes only plausible
        natural movement, not arbitrary pose error. Direction compatibility
        across pre-hit/post-hit windows is UNKNOWN, not stationary/opposed.
        This basis can earn at most half connection credit, never certification.
        """
        a,b=self.head[:,n,0],self.head[:,n,1];ca,cb=self.camera[:,n,0],self.camera[:,n,1]
        center_a=.5+ca[:,:1]*(a[:,:2]-ca[:,1:]);center_b=.5+cb[:,:1]*(b[:,:2]-cb[:,1:])
        size_a,size_b=a[:,2]*ca[:,0],b[:,2]*cb[:,0]
        scale=np.maximum(.005,(size_a[:,None]+size_b[None,:])/2.)
        face_seen=((a[:,2]>.019)&~self.formation[:,n,0])[:,None]&((b[:,2]>.019)&~self.formation[:,n,1])[None,:]
        face=rho(np.maximum(np.linalg.norm((center_a[:,None]-center_b[None,:])*[16/9,1.],axis=-1)/scale,
            np.abs(np.log(np.maximum(.001,size_a[:,None])/np.maximum(.001,size_b[None,:])))))
        shape=face.shape;phase=np.zeros(shape);raw=np.zeros(shape);observed=np.zeros(shape,bool);active=np.zeros(shape,bool)
        confidence=np.zeros(shape)
        for bone,joint in enumerate(self.destinations):
            common=face_seen&self.arm_valid[:,n,1,bone,None]&self.arm_valid[:,n,2,bone][None,:]
            va,vb=self.velocity_valid[:,n,0,bone],self.velocity_valid[:,n,1,bone]
            measured=va[:,None]|vb[None,:]
            angular_a=np.linalg.norm(self.velocity[:,n,0,bone],axis=-1)
            angular_b=np.linalg.norm(self.velocity[:,n,1,bone],axis=-1)
            angular_step=np.maximum(np.where(va,angular_a,0.)[:,None],np.where(vb,angular_b,0.)[None,:])/self.fps
            bone_jump=np.linalg.norm(self.bones[:,None,n,1,bone]-self.bones[None,:,n,2,bone],axis=-1)
            pa,pb=self.projected[:,n,1,joint,:2],self.projected[:,n,2,joint,:2]
            screen_jump=np.linalg.norm((pa[:,None]-pb[None,:])*[16/9,1.],axis=-1)
            da=np.linalg.norm((self.projected[:,n,1,joint,:2]-self.projected[:,n,0,joint,:2])*[16/9,1.],axis=-1)
            db=np.linalg.norm((self.projected[:,n,3,joint,:2]-self.projected[:,n,2,joint,:2])*[16/9,1.],axis=-1)
            screen_step=np.maximum(np.where(va,da,0.)[:,None],np.where(vb,db,0.)[None,:])/6.
            residual=rho(np.maximum(np.maximum(0.,bone_jump-angular_step)/2.,np.maximum(0.,screen_jump-screen_step)/scale))
            raw_residual=rho(np.maximum(bone_jump/2.,screen_jump/scale))
            weight=np.minimum(self.phase_certainty[:,n,1,bone,None],self.phase_certainty[:,n,2,bone][None,:])
            phase=np.maximum(phase,np.where(common,residual*weight,0.));raw=np.maximum(raw,np.where(common,raw_residual*weight,0.))
            confidence=np.maximum(confidence,np.where(common,weight,0.))
            observed|=common
            moving=(np.maximum(angular_a[:,None],angular_b[None,:])>=.3)|(
                np.maximum(np.where(va,da,0.)[:,None],np.where(vb,db,0.)[None,:])/scale>=.3)
            active|=common&measured&moving
        return dict(face=face,face_seen=face_seen,phase=phase,raw_phase=raw,phase_seen=observed,active=active,confidence=confidence)

    def matrix(self,n,*,details=False):
        q=len(self.states);shape=(q,q)
        worst=np.zeros(shape,np.float32);seen=np.zeros(shape+(4,),bool)
        phase_seen=np.zeros(shape,bool);motion_seen=np.zeros(shape,bool);active=np.zeros(shape,bool)
        opposed=np.zeros(shape,bool);phase_max=np.zeros(shape);motion_max=np.zeros(shape)
        phase_confidence=np.zeros(shape);motion_confidence=np.zeros(shape)
        phase_basis=np.zeros(shape+(len(self.destinations),),np.float32)
        motion_basis=np.zeros_like(phase_basis)
        # Each physical clock is compared to itself across sources. A natural
        # hand movement from frame -1 to 0 is NOT a pose discontinuity.
        for side,phase in ((0,1),(1,2)):
            heads=self.head[:,n,side];cameras=self.camera[:,n,side]
            projected=.5+cameras[:,:1]*(heads[:,:2]-cameras[:,1:])
            size=heads[:,2]*cameras[:,0]
            observed=(heads[:,2]>.019)&~self.formation[:,n,side]
            usable=observed[:,None]&observed[None,:]
            usable&=self.continuous[:,n][None,:] if side==0 else self.continuous[:,n][:,None]
            norm=np.maximum(.005,(size[:,None]+size[None,:])/2)
            position=np.linalg.norm((projected[:,None]-projected[None,:])*[16/9,1.],axis=-1)/norm
            scale=np.abs(np.log(np.maximum(.001,size[:,None])/np.maximum(.001,size[None,:])))
            face=rho(np.maximum(position,scale));worst=np.maximum(worst,np.where(usable,face,0.));seen[:,:,0]|=usable
            joints=self.projected[:,n,phase]
            bones=self.bones[:,n,phase];valid=self.arm_valid[:,n,phase]
            for bone,joint in enumerate(self.destinations):
                common=usable&valid[:,bone,None]&valid[None,:,bone]
                screen=np.linalg.norm((joints[:,None,joint,:2]-joints[None,:,joint,:2])*[16/9,1.],axis=-1)/norm
                angle=np.linalg.norm(bones[:,None,bone]-bones[None,:,bone],axis=-1)/2.
                residual=rho(np.maximum(screen,angle))
                weight=np.minimum(self.phase_certainty[:,n,phase,bone,None],self.phase_certainty[:,n,phase,bone][None,:])
                phase_max=np.maximum(phase_max,np.where(common,residual*weight,0.));phase_seen|=common
                phase_confidence=np.maximum(phase_confidence,np.where(common,weight,0.))
                phase_basis[:,:,bone]=np.maximum(phase_basis[:,:,bone],np.where(common,weight,0.))
        worst=np.maximum(worst,phase_max);seen[:,:,1]=phase_seen
        for window in (0,1):
            v=self.velocity[:,n,window];valid=self.velocity_valid[:,n,window]
            early,late=(0,1) if window==0 else (2,3)
            screen=(self.projected[:,n,late,:,:2]-self.projected[:,n,early,:,:2])*[16/9,1.]
            head=self.head[:,n,window,2]*self.camera[:,n,window,0]
            scale=np.maximum(.005,(head[:,None]+head[None,:])/2.)
            for bone in range(len(self.destinations)):
                common=valid[:,bone,None]&valid[None,:,bone]
                common&=self.continuous[:,n][None,:] if window==0 else self.continuous[:,n][:,None]
                a,b=v[:,None,bone],v[None,:,bone];na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
                moving=np.maximum(na,nb)>=.3;both=np.minimum(na,nb)>=.15
                direction=np.where(both,(1.-np.clip((a*b).sum(-1)/np.maximum(1e-9,na*nb),-1,1))/2.,1.)
                direction=np.where(moving,direction,0.)
                speed=rho(np.abs(np.log((na+.15)/(nb+.15))))
                residual=np.maximum(direction,speed)
                joint=self.destinations[bone]
                # Actual displayed movement includes native camera pan/zoom.
                # Directed-bone angular motion alone removes that mismatch.
                camera_motion=rho(np.linalg.norm(screen[:,None,joint]-screen[None,:,joint],axis=-1)/scale)
                residual=np.maximum(residual,camera_motion)
                weight=np.minimum(self.motion_certainty[:,n,window,bone,None],self.motion_certainty[:,n,window,bone][None,:])
                motion_max=np.maximum(motion_max,np.where(common,residual*weight,0.));motion_seen|=common
                motion_confidence=np.maximum(motion_confidence,np.where(common,weight,0.))
                motion_basis[:,:,bone]=np.maximum(motion_basis[:,:,bone],np.where(common,weight,0.))
                displayed_moving=np.maximum(np.linalg.norm(screen[:,None,joint],axis=-1),
                    np.linalg.norm(screen[None,:,joint],axis=-1))/scale>=.3
                active|=common&(moving|displayed_moving);opposed|=common&moving&(direction>=.5)&(weight>=1.)
        worst=np.maximum(worst,motion_max);seen[:,:,2]=motion_seen
        exposed=self.exposed_partial(n)
        # Prefer actual same-clock evidence whenever available. Fill absent
        # phase/face channels with real exposed endpoints; trajectory direction
        # remains unmeasured, so its finite uncertainty is retained below.
        worst=np.maximum(worst,np.where(~seen[:,:,0]&exposed['face_seen'],exposed['face'],0.))
        seen[:,:,0]|=exposed['face_seen']
        phase_max=np.maximum(phase_max,np.where(~phase_seen&exposed['phase_seen'],exposed['phase'],0.))
        phase_seen|=exposed['phase_seen'];seen[:,:,1]=phase_seen
        phase_confidence=np.maximum(phase_confidence,exposed['confidence'])
        worst=np.maximum(worst,phase_max)
        a,b=self.vectors[:,n,0],self.vectors[:,n,1]
        known=(np.linalg.norm(a,axis=-1)>=.5)[:,None]&(np.linalg.norm(b,axis=-1)>=.5)[None,:]
        cosine=np.clip(np.einsum('ik,jk->ij',a,b,optimize=False),-1,1)
        identity=rho(np.maximum(0.,(.65-cosine)/.3))
        member_a,member_b=self.member_hint[:,n,0],self.member_hint[:,n,1]
        named_known=(member_a[:,None]>=0)&(member_b[None,:]>=0)
        member_agreement=named_known&(member_a[:,None]==member_b[None,:])
        member_conflict=named_known&~member_agreement
        # A single SFace cosine below .55 is NOT known different identity.
        # Actual same-Rei across costumes measured .42 in v19; use independent
        # recurrent local hints when available, never reinterpret128D as512D.
        appearance_disagreement=member_conflict
        identity=np.where(member_agreement,0.,np.where(member_conflict,1.,identity))
        worst=np.maximum(worst,np.where(known|named_known,identity,0.));seen[:,:,3]=known|named_known
        crop=np.maximum(self.crop[:,n,0,None],self.crop[:,n,1][None,:])
        worst=np.maximum(worst,crop)
        evidence=seen.astype(np.float32);evidence[:,:,1]=phase_confidence;evidence[:,:,2]=motion_confidence
        missing=1.-evidence.mean(-1);loss=worst+.5*missing
        ra,rb=self.availability[:,n,0],self.availability[:,n,1]
        colours_a,colours_b=self.colour[:,n,0],self.colour[:,n,1]
        style=np.zeros(shape)
        for region in (0,1):
            affinity=np.einsum('ik,jk->ij',colours_a[:,region],colours_b[:,region],optimize=False)
            style+=.5*np.sqrt(np.maximum(0.,1.-affinity))
        style_known=self.colour_valid[:,n,0,None]&self.colour_valid[:,n,1][None,:]
        style=np.where(style_known,style,0.)
        # Novelty is measured from original framing/colour, never source IDs,
        # extra zoom, number of available sources, or a desired cut count.
        h1,h2=self.head[:,n,0],self.head[:,n,1]
        framing=rho(np.abs(np.log(np.maximum(.005,h1[:,2,None])/np.maximum(.005,h2[None,:,2])))+
            np.linalg.norm((h1[:,None,:2]-h2[None,:,:2])*[16/9,1.],axis=-1))
        novelty=1.-(1.-style)*(1.-framing)
        # One whole displayed face-height of residual or an equally large
        # observed motion disagreement is not a matched-action opportunity.
        # This only removes the optional reward; its finite risk/cut/source
        # remain in the graph. Small measured errors remain continuous costs.
        focal=self.dominant[:,n,0,None]&self.dominant[:,n,1][None,:]
        candidate=phase_seen&motion_seen&(known|member_agreement)&active&~opposed&~appearance_disagreement&(phase_max<.5)&(motion_max<.5)&focal
        same_primitive=np.minimum(phase_basis,motion_basis).max(-1)
        connected=candidate&(same_primitive>=1.)
        partial=(~motion_seen)&phase_seen&exposed['active']&(known|member_agreement)&~opposed&~appearance_disagreement&focal&(phase_max<.5)&(exposed['raw_phase']<.5)
        # Visibility gain is already integrated over the incoming hold. An
        # extra (R_b-R_a)+ at a cut rewards waiting until A zooms out/loses
        # its face, exactly the late wide-flash regression in real v19.
        # Only an actually supported joint/appearance connection earns change
        # credit; no source or unsupported/deliberate cut is removed.
        connection_weight=np.where(candidate,same_primitive,.5*partial*phase_confidence)
        benefit=np.minimum(ra[:,None],rb[None,:])*novelty*connection_weight
        cost=self.width[n]*(loss-self.strength[n]*benefit)
        if not details:return cost.astype(np.float32)
        return dict(cost=cost,loss=loss,benefit=benefit,missing_fraction=missing,
            worst_phase_loss=phase_max,worst_motion_loss=motion_max,opposed_motion=opposed,
            anonymous_appearance_disagreement=appearance_disagreement,
            source_local_member_hint_agreement=member_agreement,source_local_member_hint_conflict=member_conflict,
            chosen_faces_are_displayed_attention_maxima=focal,
            exposed_partial_connection_supported=partial,exposed_phase_raw_loss=exposed['raw_phase'],
            appearance_cosine=np.where(known,cosine,np.nan),style_distance=style,
            moving_connection_supported=connected,width_seconds=float(self.width[n]),
            provisional_model_connection_candidate=candidate&~connected,
            phase_evidence_reliability=phase_confidence,motion_evidence_reliability=motion_confidence,
            same_clock_motion_observed=motion_seen,
            observed_event_strength=float(self.strength[n]))

    def selected(self,n,a,b):
        # The DP already chose these two states. Rebuilding a416x416 matrix
        # for every report row repeats the entire search's pair work. Keep
        # the ORIGINAL physical event strengths/widths and slice only states;
        # never construct a new Ports object that redetects gesture events.
        pair=np.array([a,b]);view=object.__new__(Ports);view.__dict__.update(self.__dict__)
        view.states=tuple(self.states[i] for i in pair)
        view.data=[self.data[i] for i in pair]
        for key in ('head','camera','body','projected','vectors','continuous','formation','availability',
                'colour','colour_valid','crop','dominant','member_hint','phase_certainty','motion_certainty',
                'arm_valid','bones','velocity','velocity_valid'):
            setattr(view,key,getattr(self,key)[pair])
        values=view.matrix(n,details=True)
        return {k:(float(v[0,1]) if np.asarray(v).dtype.kind!='b' else bool(v[0,1]))
            if np.ndim(v)==2 else v for k,v in values.items()}
