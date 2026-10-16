"""Exact cost of a declared finite, realizable camera-arc family.

Long holds reuse four frozen interiors plus independently bounded19-frame
endpoint patches. Short holds expose either an exactly compatible baseline
or one joint Hermite bridge. No camera fit/retiming occurs after the DP.
"""
import numpy as np
from .connection_prior import physical_weights
from .native_fragment_integrity import NativeFragmentIntegrity
from .played_partial_pose import face_geometry
from .focal_appearance import discrepancy as appearance_discrepancy
from .terminal_stencil import eligible as ending_eligible


def _hermite(a,b,da,db,length):
    t=np.linspace(0.,1.,length)
    return ((2*t**3-3*t**2+1)[None,:,None]*a[:,None]
        +(t**3-2*t**2+t)[None,:,None]*(length-1)*da[:,None]
        +(-2*t**3+3*t**2)[None,:,None]*b[:,None]
        +(t**3-t**2)[None,:,None]*(length-1)*db[:,None])


def _hermite_samples(a,b,da,db,length,at):
    # Index the SAME linspace and retain operation order. Merely dividing by
    # length-1 can differ by an ULP and flip a body point exactly on a border.
    t=np.linspace(0.,1.,length)[at]
    return ((2*t**3-3*t**2+1)[:,None]*a
        +(t**3-2*t**2+t)[:,None]*(length-1)*da
        +(-2*t**3+3*t**2)[:,None]*b+(t**3-t**2)[:,None]*(length-1)*db)


def camera_samples(entry,exit,baseline,offsets,*,code):
    """Exact sampled values of an ALREADY admissible frozen camera code.

    This neither changes its legality nor fits a new path. It avoids copying
    an entire long hold for each four-second history/fingerprint evaluation.
    """
    entry=np.asarray(entry,float);exit=np.asarray(exit,float);baseline=np.asarray(baseline,float)
    at=np.asarray(offsets);L=len(baseline)
    if (entry.shape!=(7,3) or exit.shape!=entry.shape or baseline.shape!=(L,3) or L<1
            or at.ndim!=1 or at.dtype.kind not in 'iu' or np.any((at<0)|(at>=L))
            or type(code) is not int or not 0<=code<=5):
        raise ValueError('actual source arc, camera code and in-range integer samples required')
    result=baseline[at].copy()
    if code<4 and L<38:return result
    if code==5:
        if L<14:raise ValueError('whole viewport needs two nonoverlapping strips')
        a=entry.copy();b=exit.copy();a[:,0]=1/a[:,0];b[:,0]=1/b[:,0]
        middle=(at>=7)&(at<L-7)
        value=_hermite_samples(a[-1],b[0],a[-1]-a[-2],b[1]-b[0],L-12,at[middle]-6)
        if np.any(value[:,0]<=0):raise ValueError('inadmissible inverse-zoom arc')
        value[:,0]=1/value[:,0];result[middle]=value
    elif L>=14:
        din=entry[-1]-baseline[6];dout=exit[0]-baseline[-7]
        vin=entry[-1]-entry[-2]-(baseline[6]-baseline[5])
        vout=exit[1]-exit[0]-(baseline[-6]-baseline[-7])
        if L<38 or code==4:
            middle=(at>=7)&(at<L-7)
            result[middle]+=_hermite_samples(din,dout,vin,vout,L-12,at[middle]-6)
        else:
            first=(at>=7)&(at<18);last=(at>=L-18)&(at<L-7);zero=np.zeros(3)
            result[first]+=_hermite_samples(din,zero,vin,zero,13,at[first]-6)
            result[last]+=_hermite_samples(zero,dout,zero,vout,13,at[last]-(L-19))
    k=min(7,L);first=at<k;last=at>=L-k
    result[first]=entry[at[first]]
    result[last]=exit[at[last]-(L-k)+7-k]
    return result


def camera_samples_many(entry,exit,baseline,offsets,*,code):
    """Batch exact sparse samples of existing codes, not a new camera family.

    Only already-admitted alternatives are passed by the interval consumer.
    Arithmetic and linspace indexing mirror camera_samples bit for bit.
    """
    entry=np.asarray(entry,float);exit=np.asarray(exit,float);base=np.asarray(baseline,float)
    at=np.asarray(offsets);L=len(base)
    if (entry.ndim!=3 or entry.shape[1:]!=(7,3) or exit.shape!=entry.shape
            or base.shape!=(L,3) or L<1 or at.ndim!=1 or at.dtype.kind not in 'iu'
            or np.any((at<0)|(at>=L)) or type(code) is not int or not 0<=code<6):
        raise ValueError('batch actual strips, baseline and sparse integer samples required')
    result=np.broadcast_to(base[at],(len(entry),len(at),3)).copy()
    if code<4 and L<38:return result
    def hermite(a,b,da,db,length,index):
        t=np.linspace(0.,1.,length)[index][None,:,None]
        return ((2*t**3-3*t**2+1)*a[:,None]
            +(t**3-2*t**2+t)*(length-1)*da[:,None]
            +(-2*t**3+3*t**2)*b[:,None]+(t**3-t**2)*(length-1)*db[:,None])
    if code==5:
        if L<14:raise ValueError('whole viewport needs two strips')
        a=entry.copy();b=exit.copy();a[:,:,0]=1/a[:,:,0];b[:,:,0]=1/b[:,:,0]
        mid=(at>=7)&(at<L-7)
        value=hermite(a[:,-1],b[:,0],a[:,-1]-a[:,-2],b[:,1]-b[:,0],L-12,at[mid]-6)
        if np.any(value[:,:,0]<=0):raise ValueError('inadmissible inverse-zoom arc')
        value[:,:,0]=1/value[:,:,0];result[:,mid]=value
    elif L>=14:
        din=entry[:,-1]-base[6];dout=exit[:,0]-base[-7]
        vin=entry[:,-1]-entry[:,-2]-(base[6]-base[5])
        vout=exit[:,1]-exit[:,0]-(base[-6]-base[-7])
        if L<38 or code==4:
            mid=(at>=7)&(at<L-7)
            result[:,mid]+=hermite(din,dout,vin,vout,L-12,at[mid]-6)
        else:
            early=(at>=7)&(at<18);late=(at>=L-18)&(at<L-7);zero=np.zeros_like(din)
            result[:,early]+=hermite(din,zero,vin,zero,13,at[early]-6)
            result[:,late]+=hermite(zero,dout,zero,vout,13,at[late]-(L-19))
    k=min(7,L);early=at<k;late=at>=L-k
    result[:,early]=entry[:,at[early]]
    result[:,late]=exit[:,at[late]-(L-k)+7-k]
    return result


def cameras(entry,exit,baseline,*,cap,bridge_only=False):
    """Batch (... flattened B) integer replay paths, plus feasibility mask."""
    entry=np.asarray(entry,float);exit=np.asarray(exit,float);baseline=np.asarray(baseline,float)
    B=len(entry);L=len(baseline)
    if entry.shape!=(B,7,3) or exit.shape!=entry.shape or baseline.shape!=(L,3) or L<1:
        raise ValueError('seven-frame exposed strips and one finite baseline required')
    values=np.broadcast_to(baseline,(B,L,3)).copy();valid=np.ones(B,bool);k=min(7,L)
    if L<14:
        values[:,:k]=entry[:,:k];values[:,-k:]=exit[:,-k:]
        overlap=max(0,2*k-L)
        if overlap:valid&=(entry[:,k-overlap:k]==(exit[:,:overlap] if L>=7 else exit[:,-L:])).all((1,2))
    elif L<38 or bridge_only:
        values[:,:7]=entry;values[:,-7:]=exit
        din=entry[:,-1]-baseline[6];dout=exit[:,0]-baseline[-7]
        vin=entry[:,-1]-entry[:,-2]-(baseline[6]-baseline[5])
        vout=exit[:,1]-exit[:,0]-(baseline[-6]-baseline[-7])
        values[:,7:L-7]+=_hermite(din,dout,vin,vout,L-12)[:,1:-1]
    else:
        values[:,:7]=entry;values[:,-7:]=exit
        zero=np.zeros((B,3));din=entry[:,-1]-baseline[6];dout=exit[:,0]-baseline[-7]
        vin=entry[:,-1]-entry[:,-2]-(baseline[6]-baseline[5])
        vout=exit[:,1]-exit[:,0]-(baseline[-6]-baseline[-7])
        values[:,7:18]+=_hermite(din,zero,vin,zero,13)[:,1:-1]
        values[:,-18:-7]+=_hermite(zero,dout,zero,vout,13)[:,1:-1]
    z=values[...,0];center=values[...,1:]
    valid&=np.isfinite(values).all((1,2))&(z>=1.-1e-9).all(1)&(z<=cap+1e-9).all(1)
    valid&=(np.linalg.norm(center-.5,axis=-1)<=.25+1e-9).all(1)
    valid&=(center>=.5/z[...,None]-1e-9).all((1,2))&(center<=1.-.5/z[...,None]+1e-9).all((1,2))
    return values,valid


def whole_viewports(entry,exit,*,length,cap):
    """Batch of exact inverse-zoom whole-hold bridges; no neutral rewind."""
    entry=np.asarray(entry,float);exit=np.asarray(exit,float);B=len(entry)
    if entry.shape!=(B,7,3) or exit.shape!=entry.shape or length<14:
        raise ValueError('whole viewport requires two actual7frame strips')
    a=entry.copy();b=exit.copy();a[:,:,0]=1/a[:,:,0];b[:,:,0]=1/b[:,:,0]
    middle=_hermite(a[:,-1],b[:,0],a[:,-1]-a[:,-2],b[:,1]-b[:,0],length-12)[:,1:-1]
    positive=(middle[:,:,0]>0).all(1);middle[:,:,0]=1/np.where(middle[:,:,0]>0,middle[:,:,0],1.)
    path=np.concatenate((entry,middle,exit),axis=1)
    z=path[:,:,0];center=path[:,:,1:]
    valid=positive&np.isfinite(path).all((1,2))&(z>=1.-1e-9).all(1)&(z<=cap+1e-9).all(1)
    valid&=(np.linalg.norm(center-.5,axis=-1)<=.25+1e-9).all(1)
    valid&=(center>=.5/z[:,:,None]-1e-9).all((1,2))&(center<=1.-.5/z[:,:,None]+1e-9).all((1,2))
    return path,valid


class ArcCosts:
    def __init__(self,nodes,entries,exits,catalogues,dense,native_ports,support,fragment_cost,*,fps,check=lambda:None,
                 person_extents=False,opportunity_episodes=None,roles=None,point_geometry=False,readout_tau=None,
                 preserve_camera_alternatives=False,capture_playback_inputs=False,played_interval_utility=None):
        self.nodes=np.asarray(nodes,int);self.entries=entries;self.exits=exits;self.cats=catalogues
        self.dense=dense;self.native=native_ports;self.support=support;self.fragments=fragment_cost
        self.fps=fps;self.check=check;self.integrity=NativeFragmentIntegrity(support,minimum_frames=12)
        self.prefix=[];self.native_tables=[];self.left=[];self.right=[];self.calls=0;self.last_codes=None
        self.long_combinations=0;self.short_paths=0;self.peak_batch_bytes=0;self.person_extents=bool(person_extents)
        self.opportunity_episodes=None if opportunity_episodes is None else np.asarray(opportunity_episodes,int)
        self.roles=roles;self.point_geometry=bool(point_geometry);self.native_points=[]
        self.readouts=None;self.readout_left=[];self.readout_right=[];self.last_alternatives=None
        self.played_interval_utility=played_interval_utility
        self.preserve_camera_alternatives=bool(preserve_camera_alternatives or readout_tau is not None
            or played_interval_utility is not None)
        self.last_camera_alternatives=None
        if capture_playback_inputs and not self.preserve_camera_alternatives:
            raise ValueError('captured playback requires explicit all-camera alternatives')
        self.capture_playback_inputs=bool(capture_playback_inputs);self.last_playback_inputs=None
        self.captured_baseline_bytes=0
        if self.capture_playback_inputs:
            from .held_playback_inputs import frozen
            self.dense=tuple(frozen(paths) for paths in dense);dense=self.dense
            self.captured_baseline_bytes=sum(value.nbytes for value in dense)
        if readout_tau is not None:
            from .scene_readout import SceneReadout
            self.readouts=[SceneReadout(cat,native_ports[s],support.events[s],dense[s],tau=readout_tau)
                for s,cat in enumerate(catalogues)]
        self.whole_viewport_paths_scored=0
        for s,paths in enumerate(dense):
            check();frames=np.arange(int(nodes[-1]));values=np.asarray(paths)
            rates=catalogues[s].rates(frames,values)/fps
            movement=self.movement(s,frames,values)
            self.prefix.append(np.pad(np.cumsum(rates,axis=1),((0,0),(1,0)))+np.pad(np.cumsum(movement,axis=1),((0,0),(2,0))))
            xs=support.events[s];rates_native,points=self.native_rates(s,xs,values[:,np.maximum(0,xs-1)],values[:,xs],components=True)
            weights=physical_weights(xs,final=support.final,fps=fps)
            w=np.array([weights[int(t)] for t in xs])
            self.native_tables.append((xs,rates_native,w,np.pad(np.cumsum(rates_native*w,axis=1),((0,0),(1,0)))))
            self.native_points.append((points,np.pad(np.cumsum(points,axis=1),((0,0),(1,0)))))
        # Prefix movement edge ending at first is not in a newly cut hold.
        self._baseline_edge=[self.movement(s,np.arange(int(nodes[-1])),np.asarray(paths)) for s,paths in enumerate(dense)]
        for n,tick in enumerate(nodes):
            check();left=np.zeros((len(entries[n]),4));right=np.zeros((len(exits[n]),4))
            left_ok=np.zeros_like(left,bool);right_ok=np.zeros_like(right,bool)
            if self.readouts is not None:
                from .scene_readout import IDENTITY
                rl=np.broadcast_to(IDENTITY,(len(entries[n]),4,4)).copy()
                rr=np.broadcast_to(IDENTITY,(len(exits[n]),4,4)).copy()
            if tick+38<=nodes[-1]:
                for s in range(len(catalogues)):
                    ids=np.array([i for i,p in enumerate(entries[n]) if p.source==s],int)
                    if not len(ids):continue
                    strips=np.asarray([entries[n][i].strip for i in ids]);frames=np.arange(tick,tick+38)
                    for g in range(4):
                        base=dense[s][g][frames];other=np.broadcast_to(base[-7:],strips.shape)
                        path,ok=cameras(strips,other,base,cap=catalogues[s].bank['zoom_cap'])
                        new=path[:,:19];old=np.broadcast_to(base[:19],new.shape);used=frames[:19]
                        new=np.where(ok[:,None,None],new,old)
                        left[ids,g]=self.patch_delta(s,used,new,old,first=int(tick),final=int(nodes[-1]))
                        left_ok[ids,g]=ok
                        if self.readouts is not None:
                            context=self.readouts[s].context_frames
                            if tick+context<=nodes[-1]:
                                rb=dense[s][g][tick:tick+context]
                                rp,rok=cameras(strips,np.broadcast_to(rb[-7:],strips.shape),rb,cap=catalogues[s].bank['zoom_cap'])
                                safe=np.where(rok[:,None,None],rp,rb[None])
                                rl[ids,g]=self.readouts[s].summary(int(tick),safe,hold_first=int(tick),baseline=dense[s][g])
            if tick>=38:
                for s in range(len(catalogues)):
                    ids=np.array([i for i,p in enumerate(exits[n]) if p.source==s],int)
                    if not len(ids):continue
                    strips=np.asarray([exits[n][i].strip for i in ids]);frames=np.arange(tick-38,tick)
                    for g in range(4):
                        base=dense[s][g][frames];other=np.broadcast_to(base[:7],strips.shape)
                        path,ok=cameras(other,strips,base,cap=catalogues[s].bank['zoom_cap'])
                        new=path[:,-19:];old=np.broadcast_to(base[-19:],new.shape);used=frames[-19:]
                        new=np.where(ok[:,None,None],new,old)
                        right[ids,g]=self.patch_delta(s,used,new,old,first=0,final=int(tick))
                        right_ok[ids,g]=ok
                        if self.readouts is not None:
                            context=self.readouts[s].context_frames
                            if tick>=context:
                                rb=dense[s][g][tick-context:tick]
                                rp,rok=cameras(np.broadcast_to(rb[:7],strips.shape),strips,rb,cap=catalogues[s].bank['zoom_cap'])
                                safe=np.where(rok[:,None,None],rp,rb[None])
                                rr[ids,g]=self.readouts[s].summary(int(tick)-context,safe,hold_first=0,baseline=dense[s][g])
            self.left.append((left,left_ok));self.right.append((right,right_ok))
            if self.readouts is not None:self.readout_left.append(rl);self.readout_right.append(rr)
        if self.played_interval_utility is not None:self.played_interval_utility.prepare(self)

    def movement(self,s,frames,values):
        """Quadratic digital velocity integral; native source jumps are not pans."""
        change=np.diff(values[...,1:],axis=-2)*[16/9,1.]
        zoom=np.diff(np.log(values[...,0]),axis=-1)
        result=self.fps*((change**2).sum(-1)+zoom**2)
        if len(frames)>1:result[...,np.isin(frames[1:],self.support.events[s])]=0.
        return result

    def native_rates(self,s,xs,before,after,*,components=False):
        B=len(before);E=len(xs)
        if not E:
            value=np.zeros((B,0))
            return (value,value.copy()) if components else value
        bank=self.native[s];lookup={int(x):i for i,x in enumerate(bank['frames'])}
        heads=np.zeros((E,2,8,4));seen=np.zeros(E,bool)
        for e,x in enumerate(xs):
            if int(x) in lookup:
                k=lookup[int(x)];heads[e]=bank['heads'][k]
                seen[e]=bool(bank.get('observed',np.ones(len(bank['frames']),bool))[k])
        camera=np.stack((before,after),axis=2)
        center=.5+camera[...,None,:1]*(heads[...,:2]-camera[...,None,1:]);height=heads[...,2]*camera[...,None,0]
        half=height[...,None]*[.45*9/16,.5]
        extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
        scores=np.prod(extent,-1)*np.exp(-((center[...,0]-.5)/.4)**2)
        scores=np.where((heads[...,2]>.019)&(heads[...,3]>=.75)&seen[:,None,None],scores,0.)
        at=scores.argmax(-1);idx=np.broadcast_to(at[...,None,None],(B,E,2,1,4))
        head=np.take_along_axis(np.broadcast_to(heads,(B,)+heads.shape),idx,axis=3).squeeze(3)
        vectors=np.zeros((E,2,3,128),np.float32)
        for e,x in enumerate(xs):
            if int(x) in lookup and 'vectors' in bank:vectors[e]=bank['vectors'][lookup[int(x)]]
        own=np.take_along_axis(np.broadcast_to(vectors,(B,)+vectors.shape),
            np.broadcast_to(np.minimum(at,2)[...,None,None],(B,E,2,1,128)),axis=3).squeeze(3)
        own=np.where((at<3)[...,None]&seen[None,:,None,None],own,0.)
        member=np.full((B,E,2),-1,int)
        if 'member_hints' in bank:
            all_hints=np.full((E,2,8),-1,int)
            for e,x in enumerate(xs):
                if int(x) in lookup:all_hints[e]=bank['member_hints'][lookup[int(x)]]
            member=np.take_along_axis(np.broadcast_to(all_hints,(B,)+all_hints.shape),at[...,None],axis=3)[...,0]
            member=np.where(seen[None,:,None],member,-1)
        appearance,_=appearance_discrepancy(own[:,:,0],own[:,:,1],member_a=member[:,:,0],member_b=member[:,:,1])
        c=.5+camera[...,:1]*(head[...,:2]-camera[...,1:]);h=head[...,2]*camera[...,0]
        valid=(scores.max(-1)>0).all(-1)
        face,displacement,scale=face_geometry(c[:,:,0]*[16/9,1.],h[:,:,0],c[:,:,1]*[16/9,1.],h[:,:,1],valid)
        face_rate=face.copy()
        if getattr(self,'person_extents',False):
            from .person_extent import height_interval,interval_gap,used_face_risk
            pairs=bank.get('person_rectangles',{})
            for e,x in enumerate(xs):
                pair=pairs.get(int(x))
                if pair is None or valid[:,e].all():continue
                gap,known=interval_gap(height_interval(pair[0],before[:,e]),height_interval(pair[1],after[:,e]))
                face[:,e]=used_face_risk(face[:,e],valid[:,e],gap,known)
        # Existing native ports contain exact endpoint faces, not the exposed
        # body windows. Use the SAME missing-pose convention as authored K.
        # A known bad native event stays finite and playable; it is not gated
        # out or laundered through a near-constant confidence-like1-Q rate.
        detail=dict(geometry_rate=face,face_rate=face_rate,pose_rate=np.full_like(face,.5),appearance_rate=appearance,
            face_log_scale=scale,screen_displacement=np.linalg.norm((c[:,:,0]-c[:,:,1])*[16/9,1.],axis=-1),
            measured_face=valid,motion_flow_rate=np.full_like(face,.5),allowed=np.ones_like(valid))
        if getattr(self,'roles',None) is not None:
            from .editorial_roles import apply
            detail=apply(detail,self.roles.at(np.asarray(xs)[None,:],member[:,:,0],member[:,:,1]))
        rate=detail['geometry_rate']+detail['pose_rate']+detail['appearance_rate']
        if not components:return rate
        if getattr(self,'point_geometry',False):
            return detail['pose_rate'],6./self.fps*(detail['geometry_rate']+detail['appearance_rate'])
        return rate,np.zeros_like(rate)

    def native_weights(self,s,first,final,xs):
        all_xs,_,weights,_=self.native_tables[s]
        at=np.searchsorted(all_xs,xs);w=weights[at].copy()
        if first:w=np.minimum(w,(xs-first)/2./self.fps)
        if final<self.support.final:w=np.minimum(w,(final-xs)/2./self.fps)
        return w

    def patch_delta(self,s,frames,new,old,*,first,final):
        value=(self.cats[s].rates(frames,new)-self.cats[s].rates(frames,old)).sum(-1)/self.fps
        value+=self.movement(s,frames,new).sum(-1)-self.movement(s,frames,old).sum(-1)
        xs=self.support.events[s];xs=xs[(xs>frames[0])&(xs<=frames[-1])&(xs>first)&(xs<final)]
        if len(xs):
            at=xs-frames[0]
            new_rate,new_point=self.native_rates(s,xs,new[:,at-1],new[:,at],components=True)
            old_rate,old_point=self.native_rates(s,xs,old[:,at-1],old[:,at],components=True)
            value+=(new_rate-old_rate)@self.native_weights(s,first,final,xs)+(new_point-old_point).sum(-1)
        self.peak_batch_bytes=max(self.peak_batch_bytes,new.nbytes+old.nbytes)
        return value

    def baseline_cost(self,s,first,final):
        result=self.prefix[s][:,final]-self.prefix[s][:,first]
        if first:result-=self._baseline_edge[s][:,first-1]
        xs,rates,w,prefix=self.native_tables[s];a=np.searchsorted(xs,first,side='right');b=np.searchsorted(xs,final,side='left')
        if a<b:
            weights=self.native_weights(s,first,final,xs[a:b])
            result+=prefix[:,b]-prefix[:,a]+rates[:,a:b]@(weights-w[a:b])
            _,point_prefix=self.native_points[s]
            result+=point_prefix[:,b]-point_prefix[:,a]
        return result

    def direct_cost(self,s,first,path,*,with_readout=False):
        B,L,_=path.shape;final=first+L;frames=np.arange(first,final)
        content_rates=self.cats[s].rates(frames,path)
        result=content_rates.sum(-1)/self.fps+self.movement(s,frames,path).sum(-1)
        xs=self.support.events[s];xs=xs[(xs>first)&(xs<final)]
        if len(xs):
            at=xs-first
            rates,points=self.native_rates(s,xs,path[:,at-1],path[:,at],components=True)
            result+=rates@self.native_weights(s,first,final,xs)+points.sum(-1)
        if with_readout:return result,self.readouts[s].summary(first,path,hold_first=first,content_rates=content_rates)
        return result

    def __call__(self,i,j):
        self.check();self.calls+=1;first,final=map(int,self.nodes[[i,j]]);L=final-first
        shape=(len(self.entries[i]),len(self.exits[j]));cost=np.zeros(shape);allowed=np.zeros(shape,bool);codes=np.full(shape,-1,np.int8)
        if self.preserve_camera_alternatives:
            alternatives=np.zeros(shape+(6,));alternative_ok=np.zeros_like(alternatives,bool)
        capture=None
        if self.capture_playback_inputs:
            from .held_playback_inputs import HoldRecorder
            capture=HoldRecorder(self,i,j)
        if self.readouts is not None:
            from .scene_readout import IDENTITY,then
            transfers=np.broadcast_to(IDENTITY,shape+(6,4)).copy()
        fragments=np.asarray(self.fragments(i,j));valid_source=self.integrity.allowed(first,final,np.arange(len(self.cats)))
        if final==self.nodes[-1]:valid_source&=ending_eligible([c.bank for c in self.cats])
        for s in range(len(self.cats)):
            if not valid_source[s]:continue
            p=np.array([k for k,v in enumerate(self.entries[i]) if v.source==s],int)
            q=np.array([k for k,v in enumerate(self.exits[j]) if v.source==s],int)
            if not len(p) or not len(q):continue
            from .programme_source_clock import admit as clock_admit
            clock_ok=clock_admit(self.cats[s].bank,self.nodes,i,j,
                [self.entries[i][k] for k in p],[self.exits[j][k] for k in q],fps=self.fps)
            if not clock_ok.any():continue
            base=self.baseline_cost(s,first,final)
            if self.preserve_camera_alternatives:
                ac=np.zeros((len(p),len(q),6));av=np.zeros_like(ac,bool)
            if L>=38:
                left,lv=self.left[i];right,rv=self.right[j]
                values=base[None,None,:]+left[p,None,:]+right[None,q,:]
                valid=lv[p,None,:]&rv[None,q,:]
                if self.preserve_camera_alternatives:ac[:,:,:4]=values;av[:,:,:4]=valid
                g=np.argmin(np.where(valid,values,np.inf),axis=-1);ok=valid.any(-1)
                v=np.take_along_axis(values,g[...,None],axis=-1).squeeze(-1)
                self.long_combinations+=len(p)*len(q)*4
            else:
                ps,qs=np.meshgrid(p,q,indexing='ij');entry=np.asarray([self.entries[i][k].strip for k in ps.ravel()]);exit=np.asarray([self.exits[j][k].strip for k in qs.ravel()])
                path,valid=cameras(entry,exit,self.dense[s][0][first:final],cap=self.cats[s].bank['zoom_cap'],bridge_only=True)
                path=np.where(valid[:,None,None],path,self.dense[s][0][None,first:final])
                if self.readouts is not None:direct,bridge_transfer=self.direct_cost(s,first,path,with_readout=True)
                else:direct=self.direct_cost(s,first,path)
                v=direct.reshape(len(p),len(q));ok=valid.reshape(v.shape);g=np.full(v.shape,4,np.int8)
                if capture is not None:capture.record_direct(s,4,p,q,path)
                if self.preserve_camera_alternatives:
                    ac[:,:,4]=v;av[:,:,4]=ok
                if self.readouts is not None:
                    transfers[np.ix_(p,q,[4])]=bridge_transfer.reshape(len(p),len(q),1,4)
                self.short_paths+=len(path)
                for family in range(4):
                    curve=self.dense[s][family][first:final];k=min(7,L)
                    compatible=np.isclose(entry[:,:k],curve[:k],rtol=0.,atol=1e-12).all((1,2))&np.isclose(exit[:,-k:],curve[-k:],rtol=0.,atol=1e-12).all((1,2))
                    compatible=compatible.reshape(v.shape)
                    if self.preserve_camera_alternatives:ac[:,:,family]=base[family];av[:,:,family]=compatible
                    better=compatible&(~ok|(base[family]<v));v=np.where(better,base[family],v);g=np.where(better,family,g);ok|=compatible
            # Add the non-baseline family only on the adjacent-opportunity
            # DAG, not all quadratic long holds. Long baseline alternatives
            # remain legal: this is NOT a maximum duration or cut quota.
            adjacent=(self.opportunity_episodes is not None and
                self.opportunity_episodes[j]-self.opportunity_episodes[i]<=2)
            if adjacent and L>=14 and not np.any((self.support.events[s]>first)&(self.support.events[s]<final)):
                ps,qs=np.meshgrid(p,q,indexing='ij')
                entry=np.asarray([self.entries[i][k].strip for k in ps.ravel()])
                exit=np.asarray([self.exits[j][k].strip for k in qs.ravel()])
                path,valid=whole_viewports(entry,exit,length=L,cap=self.cats[s].bank['zoom_cap'])
                safe=np.where(valid[:,None,None],path,self.dense[s][0][None,first:final])
                if self.readouts is not None:direct,whole_transfer=self.direct_cost(s,first,safe,with_readout=True)
                else:direct=self.direct_cost(s,first,safe)
                actual=direct.reshape(len(p),len(q));valid=valid.reshape(v.shape)
                if capture is not None:capture.record_direct(s,5,p,q,safe)
                if self.preserve_camera_alternatives:
                    ac[:,:,5]=actual;av[:,:,5]=valid
                if self.readouts is not None:
                    transfers[np.ix_(p,q,[5])]=whole_transfer.reshape(len(p),len(q),1,4)
                better=valid&(~ok|(actual<v));v=np.where(better,actual,v);g=np.where(better,5,g);ok|=valid
                self.whole_viewport_paths_scored+=len(path)
                self.peak_batch_bytes=max(self.peak_batch_bytes,path.nbytes+safe.nbytes)
            if self.played_interval_utility is not None:
                av&=clock_ok[:,:,None]
                ac=self.played_interval_utility.adjust(self,i,j,s,p,q,ac,av)
                if not np.isfinite(ac).all():raise ValueError('finite played interval candidate costs required')
                g=np.argmin(np.where(av,ac,np.inf),axis=-1);ok=av.any(-1)
                v=np.take_along_axis(ac,g[...,None],axis=-1).squeeze(-1)
            ix=np.ix_(p,q);cost[ix]=v+fragments[s];allowed[ix]=ok&clock_ok;codes[ix]=g
            if self.preserve_camera_alternatives:
                alternatives[ix]=ac+fragments[s];alternative_ok[ix]=av&clock_ok[:,:,None]
            if self.readouts is not None:
                context=self.readouts[s].context_frames
                if L>=2*context:
                    middle=self.readouts[s].base(first+context,final-context)
                    transfer=then(then(self.readout_left[i][p,None],middle[None,None]),self.readout_right[j][None,q])
                    transfers[np.ix_(p,q,np.arange(4))]=transfer
                else:
                    ps,qs=np.meshgrid(p,q,indexing='ij')
                    entry=np.asarray([self.entries[i][k].strip for k in ps.ravel()])
                    exit=np.asarray([self.exits[j][k].strip for k in qs.ravel()])
                    baseline_transfers=self.readouts[s].base_hold(first,final) if L<38 else None
                    for family in range(4):
                        if not av[:,:,family].any():continue
                        if L<38:
                            transfers[np.ix_(p,q,[family])]=baseline_transfers[family]
                            continue
                        baseline=self.dense[s][family]
                        path,_=cameras(entry,exit,baseline[first:final],cap=self.cats[s].bank['zoom_cap'])
                        summary=self.readouts[s].summary(first,path,hold_first=first,baseline=baseline)
                        transfers[np.ix_(p,q,[family])]=summary.reshape(len(p),len(q),1,4)
        self.last_codes=(i,j,codes)
        if self.preserve_camera_alternatives:self.last_camera_alternatives=(i,j,alternatives,alternative_ok)
        if capture is not None:self.last_playback_inputs=capture.seal(alternative_ok)
        if self.readouts is not None:self.last_alternatives=(alternatives,alternative_ok,transfers)
        return cost,allowed,codes

    def camera_alternatives(self,i,j):
        """Retain all legal camera codes without enabling rejected E5 debt.

        The normal scalar E4 interface is unchanged. A contextual consumer
        must choose its camera jointly with prefix context, not inherit the
        cheapest isolated-hold argmin. Native/content/clock costs are shared.
        """
        if not self.preserve_camera_alternatives:
            raise ValueError('explicit preserved camera domain required')
        self(i,j)
        return self.last_camera_alternatives[2:]

    def sample_camera(self,i,j,p,q,code,frames):
        first,final=map(int,self.nodes[[i,j]]);source=self.entries[i][p].source
        baseline=self.dense[source][0 if code in (4,5) else code][first:final]
        return camera_samples(self.entries[i][p].strip,self.exits[j][q].strip,baseline,
            np.asarray(frames)-first,code=int(code))

    def alternatives(self,i,j):
        """Preserve all finite camera codes before future burden is known."""
        if self.readouts is None:raise ValueError('SceneReadout must be enabled before scoring')
        self(i,j)
        return self.last_alternatives

    def replay(self,i,j,p,q,code):
        first,final=map(int,self.nodes[[i,j]]);source=self.entries[i][p].source
        entry=self.entries[i][p].strip;exit=self.exits[j][q].strip
        if code==5:
            path,valid=whole_viewports(entry[None],exit[None],length=final-first,cap=self.cats[source].bank['zoom_cap'])
            if not valid[0]:raise AssertionError('selected whole viewport is infeasible')
            from .subject_moment_sequence import hold_curve
            reference=hold_curve(first,final,([first,final-1],[[1.,.5,.5]]*2),entry,exit,
                cap=self.cats[source].bank['zoom_cap'],family='whole_viewport')
            if reference is None or not np.allclose(reference.values,path[0],rtol=0.,atol=2e-14):
                raise AssertionError('selected whole viewport differs from scored scalar replay')
            return path[0]
        base=self.dense[source][0 if code==4 else code][first:final]
        if final-first<38 and code<4:path=base[None].copy();valid=np.array([True])
        else:path,valid=cameras(entry[None],exit[None],base,cap=self.cats[source].bank['zoom_cap'],bridge_only=code==4)
        if not valid[0]:raise AssertionError('selected frozen camera arc is not feasible')
        # Independent scalar Opus implementation checks the vectorized path;
        # it does not optimize, clamp or change the replayed values.
        from .boundary_camera_arc import build
        k=min(7,final-first)
        reference=build(entry[:k],exit[-k:],base,cap=self.cats[source].bank['zoom_cap'])
        if reference is None or not np.allclose(reference.values,path[0],rtol=0.,atol=2e-14):
            raise AssertionError('frozen vectorized camera differs from independent scalar replay')
        return path[0]
