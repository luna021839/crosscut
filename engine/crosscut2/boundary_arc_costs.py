"""Exact cost of a declared finite, realizable camera-arc family.

Long holds reuse four frozen interiors plus independently bounded19-frame
endpoint patches. Short holds expose either an exactly compatible baseline
or one joint Hermite bridge. No camera fit/retiming occurs after the DP.
"""
import numpy as np
from .connection_prior import physical_weights
from .native_fragment_integrity import NativeFragmentIntegrity
from .played_partial_pose import face_geometry


def _hermite(a,b,da,db,length):
    t=np.linspace(0.,1.,length)
    return ((2*t**3-3*t**2+1)[None,:,None]*a[:,None]
        +(t**3-2*t**2+t)[None,:,None]*(length-1)*da[:,None]
        +(-2*t**3+3*t**2)[None,:,None]*b[:,None]
        +(t**3-t**2)[None,:,None]*(length-1)*db[:,None])


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


class ArcCosts:
    def __init__(self,nodes,entries,exits,catalogues,dense,native_ports,support,fragment_cost,*,fps,check=lambda:None,
                 person_extents=False):
        self.nodes=np.asarray(nodes,int);self.entries=entries;self.exits=exits;self.cats=catalogues
        self.dense=dense;self.native=native_ports;self.support=support;self.fragments=fragment_cost
        self.fps=fps;self.check=check;self.integrity=NativeFragmentIntegrity(support,minimum_frames=12)
        self.prefix=[];self.native_tables=[];self.left=[];self.right=[];self.calls=0;self.last_codes=None
        self.long_combinations=0;self.short_paths=0;self.peak_batch_bytes=0;self.person_extents=bool(person_extents)
        for s,paths in enumerate(dense):
            check();frames=np.arange(int(nodes[-1]));values=np.asarray(paths)
            rates=catalogues[s].rates(frames,values)/fps
            movement=self.movement(s,frames,values)
            self.prefix.append(np.pad(np.cumsum(rates,axis=1),((0,0),(1,0)))+np.pad(np.cumsum(movement,axis=1),((0,0),(2,0))))
            xs=support.events[s];rates_native=self.native_rates(s,xs,values[:,np.maximum(0,xs-1)],values[:,xs])
            weights=physical_weights(xs,final=support.final,fps=fps)
            w=np.array([weights[int(t)] for t in xs])
            self.native_tables.append((xs,rates_native,w,np.pad(np.cumsum(rates_native*w,axis=1),((0,0),(1,0)))))
        # Prefix movement edge ending at first is not in a newly cut hold.
        self._baseline_edge=[self.movement(s,np.arange(int(nodes[-1])),np.asarray(paths)) for s,paths in enumerate(dense)]
        for n,tick in enumerate(nodes):
            check();left=np.zeros((len(entries[n]),4));right=np.zeros((len(exits[n]),4))
            left_ok=np.zeros_like(left,bool);right_ok=np.zeros_like(right,bool)
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
            self.left.append((left,left_ok));self.right.append((right,right_ok))

    def movement(self,s,frames,values):
        """Quadratic digital velocity integral; native source jumps are not pans."""
        change=np.diff(values[...,1:],axis=-2)*[16/9,1.]
        zoom=np.diff(np.log(values[...,0]),axis=-1)
        result=self.fps*((change**2).sum(-1)+zoom**2)
        if len(frames)>1:result[...,np.isin(frames[1:],self.support.events[s])]=0.
        return result

    def native_rates(self,s,xs,before,after):
        B=len(before);E=len(xs)
        if not E:return np.zeros((B,0))
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
        c=.5+camera[...,:1]*(head[...,:2]-camera[...,1:]);h=head[...,2]*camera[...,0]
        valid=(scores.max(-1)>0).all(-1)
        face,_,_=face_geometry(c[:,:,0]*[16/9,1.],h[:,:,0],c[:,:,1]*[16/9,1.],h[:,:,1],valid)
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
        return face+.5

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
            changed=self.native_rates(s,xs,new[:,at-1],new[:,at])-self.native_rates(s,xs,old[:,at-1],old[:,at])
            value+=changed@self.native_weights(s,first,final,xs)
        self.peak_batch_bytes=max(self.peak_batch_bytes,new.nbytes+old.nbytes)
        return value

    def baseline_cost(self,s,first,final):
        result=self.prefix[s][:,final]-self.prefix[s][:,first]
        if first:result-=self._baseline_edge[s][:,first-1]
        xs,rates,w,prefix=self.native_tables[s];a=np.searchsorted(xs,first,side='right');b=np.searchsorted(xs,final,side='left')
        if a<b:
            weights=self.native_weights(s,first,final,xs[a:b])
            result+=prefix[:,b]-prefix[:,a]+rates[:,a:b]@(weights-w[a:b])
        return result

    def direct_cost(self,s,first,path):
        B,L,_=path.shape;final=first+L;frames=np.arange(first,final)
        result=self.cats[s].rates(frames,path).sum(-1)/self.fps+self.movement(s,frames,path).sum(-1)
        xs=self.support.events[s];xs=xs[(xs>first)&(xs<final)]
        if len(xs):
            at=xs-first
            result+=self.native_rates(s,xs,path[:,at-1],path[:,at])@self.native_weights(s,first,final,xs)
        return result

    def __call__(self,i,j):
        self.check();self.calls+=1;first,final=map(int,self.nodes[[i,j]]);L=final-first
        shape=(len(self.entries[i]),len(self.exits[j]));cost=np.zeros(shape);allowed=np.zeros(shape,bool);codes=np.full(shape,-1,np.int8)
        fragments=np.asarray(self.fragments(i,j));valid_source=self.integrity.allowed(first,final,np.arange(len(self.cats)))
        for s in range(len(self.cats)):
            if not valid_source[s]:continue
            p=np.array([k for k,v in enumerate(self.entries[i]) if v.source==s],int)
            q=np.array([k for k,v in enumerate(self.exits[j]) if v.source==s],int)
            if not len(p) or not len(q):continue
            base=self.baseline_cost(s,first,final)
            if L>=38:
                left,lv=self.left[i];right,rv=self.right[j]
                values=base[None,None,:]+left[p,None,:]+right[None,q,:]
                valid=lv[p,None,:]&rv[None,q,:]
                g=np.argmin(np.where(valid,values,np.inf),axis=-1);ok=valid.any(-1)
                v=np.take_along_axis(values,g[...,None],axis=-1).squeeze(-1)
                self.long_combinations+=len(p)*len(q)*4
            else:
                ps,qs=np.meshgrid(p,q,indexing='ij');entry=np.asarray([self.entries[i][k].strip for k in ps.ravel()]);exit=np.asarray([self.exits[j][k].strip for k in qs.ravel()])
                path,valid=cameras(entry,exit,self.dense[s][0][first:final],cap=self.cats[s].bank['zoom_cap'],bridge_only=True)
                path=np.where(valid[:,None,None],path,self.dense[s][0][None,first:final])
                v=self.direct_cost(s,first,path).reshape(len(p),len(q));ok=valid.reshape(v.shape);g=np.full(v.shape,4,np.int8)
                self.short_paths+=len(path)
                for family in range(4):
                    curve=self.dense[s][family][first:final];k=min(7,L)
                    compatible=np.isclose(entry[:,:k],curve[:k],rtol=0.,atol=1e-12).all((1,2))&np.isclose(exit[:,-k:],curve[-k:],rtol=0.,atol=1e-12).all((1,2))
                    compatible=compatible.reshape(v.shape)
                    better=compatible&(~ok|(base[family]<v));v=np.where(better,base[family],v);g=np.where(better,family,g);ok|=compatible
            ix=np.ix_(p,q);cost[ix]=v+fragments[s];allowed[ix]=ok;codes[ix]=g
        self.last_codes=(i,j,codes)
        return cost,allowed,codes

    def replay(self,i,j,p,q,code):
        first,final=map(int,self.nodes[[i,j]]);source=self.entries[i][p].source
        entry=self.entries[i][p].strip;exit=self.exits[j][q].strip
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
