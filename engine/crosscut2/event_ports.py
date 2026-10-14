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
        q,N=len(states),len(nodes)
        if self.body.shape!=(q,N,4,8,3) or self.head.shape!=(q,N,2,4):
            raise ValueError('same-clock fixed action/camera states required')
        if (not all(np.isfinite(v).all() for v in (self.body,self.projected,self.head,self.camera,
                self.vectors,self.availability,self.colour,self.crop)) or np.any(self.camera[:,:,:,0]<1)
                or np.any((self.availability<0)|(self.availability>1))):
            raise ValueError('finite physical action/style/camera inputs required')
        valid=(self.body[...,2]>=.3)&(self.body[...,:2]>=0).all(-1)&(self.body[...,:2]<=1).all(-1)
        pairs=np.array([[0,2],[2,4],[1,3],[3,5]])
        xy=self.body[...,:2]*[16/9,1.]
        bones=xy[...,pairs[:,1],:]-xy[...,pairs[:,0],:];length=np.linalg.norm(bones,axis=-1)
        supported=valid[...,pairs].all(-1)&(length>=.025)
        self.arm_valid=np.repeat(supported.reshape(q,N,4,2,2).all(-1),2,axis=-1)
        self.bones=bones/np.maximum(1e-9,length[...,None])
        self.velocity=(self.bones[:,:,[1,3]]-self.bones[:,:,[0,2]])/(6/fps)
        self.velocity_valid=self.arm_valid[:,:,[1,3]]&self.arm_valid[:,:,[0,2]]
        # Native cuts/duplicate physical samples invalidate ONLY that window.
        for i,d in enumerate(data):
            self.velocity_valid[i]&=d['arm_valid'][:,:,2:6]
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

    def matrix(self,n,*,details=False):
        q=len(self.states);shape=(q,q)
        worst=np.zeros(shape,np.float32);seen=np.zeros(shape+(4,),bool)
        phase_seen=np.zeros(shape,bool);motion_seen=np.zeros(shape,bool);active=np.zeros(shape,bool)
        opposed=np.zeros(shape,bool);phase_max=np.zeros(shape);motion_max=np.zeros(shape)
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
            for bone,joint in enumerate((2,4,3,5)):
                common=usable&valid[:,bone,None]&valid[None,:,bone]
                screen=np.linalg.norm((joints[:,None,joint,:2]-joints[None,:,joint,:2])*[16/9,1.],axis=-1)/norm
                angle=np.linalg.norm(bones[:,None,bone]-bones[None,:,bone],axis=-1)/2.
                residual=rho(np.maximum(screen,angle))
                phase_max=np.maximum(phase_max,np.where(common,residual,0.));phase_seen|=common
        worst=np.maximum(worst,phase_max);seen[:,:,1]=phase_seen
        for window in (0,1):
            v=self.velocity[:,n,window];valid=self.velocity_valid[:,n,window]
            for bone in range(4):
                common=valid[:,bone,None]&valid[None,:,bone]
                common&=self.continuous[:,n][None,:] if window==0 else self.continuous[:,n][:,None]
                a,b=v[:,None,bone],v[None,:,bone];na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
                moving=np.maximum(na,nb)>=.3;both=np.minimum(na,nb)>=.15
                direction=np.where(both,(1.-np.clip((a*b).sum(-1)/np.maximum(1e-9,na*nb),-1,1))/2.,1.)
                direction=np.where(moving,direction,0.)
                speed=rho(np.abs(np.log((na+.15)/(nb+.15))))
                residual=np.maximum(direction,speed)
                motion_max=np.maximum(motion_max,np.where(common,residual,0.));motion_seen|=common
                active|=common&moving;opposed|=common&moving&(direction>=.5)
        worst=np.maximum(worst,motion_max);seen[:,:,2]=motion_seen
        a,b=self.vectors[:,n,0],self.vectors[:,n,1]
        known=(np.linalg.norm(a,axis=-1)>=.5)[:,None]&(np.linalg.norm(b,axis=-1)>=.5)[None,:]
        cosine=np.clip(np.einsum('ik,jk->ij',a,b,optimize=False),-1,1)
        identity=rho(np.maximum(0.,(.65-cosine)/.3))
        worst=np.maximum(worst,np.where(known,identity,0.));seen[:,:,3]=known
        crop=np.maximum(self.crop[:,n,0,None],self.crop[:,n,1][None,:])
        worst=np.maximum(worst,crop)
        missing=1.-seen.mean(-1);loss=worst+.5*missing
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
        connected=phase_seen&motion_seen&known&active&~opposed
        benefit=np.maximum(0.,rb[None,:]-ra[:,None])+np.minimum(ra[:,None],rb[None,:])*novelty*connected
        benefit=np.where(opposed,0.,benefit)
        cost=self.width[n]*(loss-self.strength[n]*benefit)
        if not details:return cost.astype(np.float32)
        return dict(cost=cost,loss=loss,benefit=benefit,missing_fraction=missing,
            worst_phase_loss=phase_max,worst_motion_loss=motion_max,opposed_motion=opposed,
            appearance_cosine=np.where(known,cosine,np.nan),style_distance=style,
            moving_connection_supported=connected,width_seconds=float(self.width[n]),
            observed_event_strength=float(self.strength[n]))

    def selected(self,n,a,b):
        values=self.matrix(n,details=True)
        return {k:(float(v[a,b]) if np.asarray(v).dtype.kind!='b' else bool(v[a,b]))
            if np.ndim(v)==2 else v for k,v in values.items()}
