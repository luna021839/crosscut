"""Pointwise observed action delivery and composed IMAGE change before DP.

Targets belong to fixed music episodes and original source observations,
not a crop's winning face. No entry strip is copied through a hold, no pose
is interpolated, and no extra credit is created by splitting a played frame.
This is an unfitted engineering objective, not learned preference or AV truth.
"""
from dataclasses import dataclass
import json
import numpy as np
from .programme_source_clock import indices
from .observed_reframing import displayed_energy_many


def articulated_seam_evidence(detail,left,right):
    """Reuse the cut refiner's actual played-arm observation BEFORE DP.

    Contemporary source windows and the proposed physical -1/0 cameras are
    required. A complete observed arm supplies its actual configuration,
    placement and common-music motion instead of only being allowed to make
    the sparse score worse. Missing arms retain the finite partial-pose cost.
    """
    from .subject_cut_moments import _seam_action
    rate=np.asarray(detail['pose_rate']).copy();direction=np.zeros_like(rate)
    measured=np.zeros(rate.shape,bool);components=np.zeros(rate.shape+(3,))
    for a,port_a in enumerate(left):
        observation_a=getattr(port_a,'observation',port_a)
        ca=observation_a.get('dense_subject_context')
        if ca is None:continue
        for b,port_b in enumerate(right):
            observation_b=getattr(port_b,'observation',port_b)
            cb=observation_b.get('dense_subject_context')
            if cb is None or not np.array_equal(ca['clock'].output_frames,cb['clock'].output_frames):continue
            geometry=dict(common_source_shot_sample_mask=ca['same_shot']&cb['same_shot'],
                outgoing_camera=observation_a['camera_strip'][-1],
                incoming_camera=observation_b['camera_strip'][0])
            value=_seam_action(ca['window'],cb['window'],ca['clock'],cb['clock'],geometry)
            if not value['seam_action_supported']:continue
            measured[a,b]=True;direction[a,b]=value['seam_arm_direction_residual']
            residual=np.asarray([value[k] for k in
                ('seam_pose_residual','seam_screen_residual','seam_motion_residual')])
            components[a,b]=residual/(1.+residual)
            # Score the actually visible articulated connection. A missing
            # hip/second arm must not add a neutral floor to a matching arm;
            # an opposed arm must not disappear in an average of shoulders.
            # These are unfitted discrepancies, not identity probabilities.
            rate[a,b]=max(direction[a,b],float(components[a,b].mean()))
    result=dict(detail,pose_rate=rate,articulated_arm_direction=direction,
        articulated_arm_direction_measured=measured,
        articulated_configuration_screen_motion=components)
    if 'matched_action_change_density' in detail:
        # Recompute both directions: an opposed arm loses the old benefit,
        # and a genuinely matched arm is no longer buried by UNKNOWN hips.
        # Other appearance/style/quality/activity evidence remains necessary.
        advantage=np.maximum(0.,1.-2.*rate)
        result['actual_played_match_advantage']=advantage
        basis=detail.get('matched_action_evidence_density',detail['matched_action_change_density'])
        result['matched_action_change_density']=np.minimum(basis,advantage)
    return result


@dataclass(frozen=True)
class DeliverySamples:
    frames: np.ndarray
    native: np.ndarray
    heads: np.ndarray
    body: np.ndarray
    support: np.ndarray
    competitors: tuple
    weights: np.ndarray

    def __post_init__(self):
        n=len(self.frames)
        if (self.frames.shape!=(n,) or self.frames.dtype.kind not in 'iu'
                or np.any(np.diff(self.frames)<=0) or np.any(self.frames<0)
                or self.native.shape!=(n,) or self.native.dtype.kind not in 'iu'
                or np.any(self.native<0) or self.heads.shape!=(n,4)
                or self.body.shape!=(n,8,3) or self.support.shape!=(n,8)
                or self.support.dtype!=bool or self.weights.shape!=(n,)
                or len(self.competitors)!=n
                or not all(np.isfinite(v).all() for v in (self.heads,self.body,self.weights))
                or np.any(self.weights<0) or np.any(self.weights>1)):
            raise ValueError('fixed actual source-native delivery samples required')
        for name in ('frames','native','heads','body','support','weights'):
            value=np.array(getattr(self,name),copy=True);value.setflags(write=False)
            object.__setattr__(self,name,value)
        others=[]
        for raw in self.competitors:
            value=np.asarray(raw,float).copy()
            if value.ndim!=2 or value.shape[1]!=4 or not np.isfinite(value).all():
                raise ValueError('actual contemporary competing faces required')
            value.setflags(write=False);others.append(value)
        object.__setattr__(self,'competitors',tuple(others))


def delivery_amount(samples,positions,cameras):
    """One amount per actual camera/sample; unsupported action earns nothing.

    Keeping only an elbow can still show partial action. Missing wrists do
    not veto a source, nor do shoulders alone certify an articulated match.
    No identity, performer-motion or singer-role success is inferred here.
    """
    at=np.asarray(positions,int);camera=np.asarray(cameras,float)
    if camera.ndim!=3 or camera.shape[1:]!=(len(at),3):
        raise ValueError('batch actual cameras on the selected observation clock required')
    heads=samples.heads[at];B=len(camera);T=len(at)
    maximum=max((len(samples.competitors[k]) for k in at),default=0)+1
    raw=np.zeros((T,maximum,4));raw[:,0]=heads
    for row,k in enumerate(at):raw[row,1:1+len(samples.competitors[k])]=samples.competitors[k]
    center=.5+camera[:,:,None,:1]*(raw[None,:,:,:2]-camera[:,:,None,1:])
    height=raw[None,:,:,2]*camera[:,:,None,0]
    half=height[...,None]*[.45*9/16,.5]
    extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
    score=extent.prod(-1)*np.exp(-((center[...,0]-.5)/.4)**2)
    valid=(raw[...,2]>.019)&(raw[...,3]>=.75)
    score=np.where(valid[None],score,0.)
    fraction=np.clip(np.prod(extent[:,:,0]/np.maximum(1e-12,2*half[:,:,0]),axis=-1),0.,1.)
    attention=np.clip(score[:,:,0]/np.maximum(1e-12,score.max(-1)),0.,1.)
    body=samples.body[at,2:6];known=samples.support[at,2:6].copy()
    known&=(body[:,:,2]>=.3)&(body[:,:,:2]>=0).all(-1)&(body[:,:,:2]<=1).all(-1)
    points=.5+camera[:,:,None,:1]*(body[None,:,:,:2]-camera[:,:,None,1:])
    shown=((points>=0)&(points<=1)).all(-1)&known[None]
    action=shown.sum(-1)/np.maximum(1,known.sum(-1))[None]
    return fraction*attention*action*valid[None,:,0]*samples.weights[at][None]


class PlayedIntervalUtility:
    def __init__(self,*,nodes,clocks,delivery,image_records,fps,targets=(),delivery_in_base=True):
        self.nodes=np.asarray(nodes,int);self.delivery=tuple(delivery)
        self.fps=float(fps);self.targets=tuple(targets);self.delivery_in_base=bool(delivery_in_base)
        if len(clocks)!=len(delivery) or not np.isfinite(fps) or fps<=0:
            raise ValueError('one actual renderer clock per physical source required')
        self.clocks=[]
        for clock in clocks:
            times=np.asarray(clock['source_times'],float).copy()
            if (times.shape!=self.nodes.shape or not np.isfinite(times).all()
                    or not np.isfinite(clock['fps']) or clock['fps']<=0 or clock['count']<1):
                raise ValueError('actual finite source-clock endpoint timestamps required')
            times.setflags(write=False);self.clocks.append(dict(clock,source_times=times))
        self.images=[];self.conflicts=0
        for records in image_records:
            by_tick={};bad=set()
            for item in records:
                tick=int(item['output_frames'][1])
                if item['output_frames']!=[tick-1,tick]:
                    raise ValueError('only adjacent actually sampled image pairs are consumed')
                if tick in bad:continue
                key=json.dumps(item,sort_keys=True)
                if tick in by_tick and by_tick[tick][0]!=key:
                    by_tick.pop(tick);bad.add(tick);self.conflicts+=1;continue
                by_tick[tick]=(key,json.loads(key))
            self.images.append(tuple(by_tick[t][1] for t in sorted(by_tick)))
        if len(self.images)!=len(delivery):raise ValueError('one image domain per source required')
        self.calls=0;self.sampled_camera_rows=0
        self._prepared_for=None;self.prefix_interval_uses=0;self.clock_correction_rows=0

    @classmethod
    def from_catalogues(cls,catalogues,opportunities,*,check=lambda:None,delivery_in_base=True):
        """Freeze one source-local target/episode BEFORE any proposed crop.

        Source timestamp interpolation merely requests original native rows;
        body/face coordinates are never interpolated. Each candidate later
        checks these native ordinals against its own exact renderer schedule.
        """
        nodes=opportunities.nodes;fps=opportunities.fps
        if any(not np.array_equal(cat.nodes,nodes) or cat.fps!=fps for cat in catalogues):
            raise ValueError('fixed music event and original catalogue clocks differ')
        clocks=[cat.bank.get('programme_render_clock') for cat in catalogues]
        if any(c is None for c in clocks):raise ValueError('actual renderer clocks required for interval utility')
        candidates={}
        for key,strength in opportunities.bindings.items():
            n,a,b,ha,hb=key;e=int(opportunities.episodes[n])
            for source,side,head in ((a,0,ha),(b,1,hb)):
                candidates.setdefault((source,e),[]).append((-float(strength),n,side,head))
        grouped=[[] for _ in catalogues];targets=[]
        for (source,e),choices in sorted(candidates.items()):
            check();cat=catalogues[source]
            if cat.subjects is None:continue
            selected=None
            for _,n,side,head in sorted(set(choices)):
                context=cat.subjects.sample(n,side,head)
                if context is not None:
                    selected=(n,side,head,context);break
            if selected is None:continue
            n,side,head,context=selected;window=context['window'];clock=clocks[source]
            frames=np.arange(int(opportunities.boundaries[e]),int(opportunities.boundaries[e+1]))
            times=np.interp(frames,nodes,clock['source_times'])
            native=(((times*1000.)/1000.)*clock['fps']+.5).astype(np.int64)
            at=native-int(window.ordinals[0]);inside=(at>=0)&(at<len(window.ordinals))
            frames,native,at=(v[inside] for v in (frames,native,at))
            anchor=int(cat.bank['native_indices'][n,side])
            same=np.searchsorted(window.native_cuts,native,side='right')==np.searchsorted(window.native_cuts,anchor,side='right')
            known=window.associated[at]&same
            frames,native,at=(v[known] for v in (frames,native,at))
            if not len(frames):continue
            grouped[source].append((frames,native,window,at,float(opportunities.utility[e])))
            targets.append(dict(source=source,episode=e,node=n,head_index=head,track=window.track,
                observed_output_samples=len(frames),
                actual_action_samples_after_entry_strip=int(((frames>=nodes[n]+7)
                    &window.support[at,2:6].any(-1)).sum()),
                selected_before_camera=True,role_or_identity_certified=False))
        delivery=[]
        for groups in grouped:
            frames=[];native=[];heads=[];body=[];support=[];others=[];weights=[]
            for f,ns,w,at,weight in groups:
                frames.extend(f);native.extend(ns);heads.extend(w.heads[at]);body.extend(w.body[at])
                support.extend(w.support[at]);others.extend(w.competitors[k] for k in at);weights.extend([weight]*len(f))
            order=np.argsort(frames,kind='stable')
            delivery.append(DeliverySamples(np.asarray(frames,np.int64)[order],np.asarray(native,np.int64)[order],
                np.asarray(heads,float).reshape(-1,4)[order],np.asarray(body,float).reshape(-1,8,3)[order],
                np.asarray(support,bool).reshape(-1,8)[order],tuple(others[k] for k in order),np.asarray(weights,float)[order]))
        image_records=[]
        for cat in catalogues:
            rows=getattr(cat,'observed_image_records',())
            if any(cat.subjects is None or row.get('source_sha256')!=cat.subjects.binding for row in rows):
                raise ValueError('observed image evidence physical source binding changed')
            image_records.append(rows)
        return cls(nodes=nodes,clocks=clocks,delivery=delivery,fps=fps,targets=targets,image_records=image_records,
            delivery_in_base=delivery_in_base)

    def native(self,source,first,final,frames):
        i,j=np.searchsorted(self.nodes,[first,final]);clock=self.clocks[source]
        if i>=len(self.nodes) or j>=len(self.nodes) or self.nodes[i]!=first or self.nodes[j]!=final:
            raise ValueError('interval endpoints must be the actual canonical source-clock nodes')
        a,b=clock['source_times'][[i,j]];rate=float((b-a)/((final-first)/self.fps))
        return indices(a,rate,np.asarray(frames)-first,fps=self.fps,native_fps=clock['fps'])

    def frames(self,source,first,final):
        samples=self.delivery[source]
        rows=samples.frames[(samples.frames>=first)&(samples.frames<final)]
        images=[r for r in self.images[source] if first<r['output_frames'][1]<final]
        return np.unique(np.r_[rows,*[r['output_frames'] for r in images]]).astype(np.int64)

    def effects(self,source,frames,native,cameras):
        """Actual pointwise utility and motion replacement, never extrapolated."""
        return self._effects(source,frames,np.asarray(native,int),cameras)

    def _effects(self,source,frames,native,cameras):
        frames=np.asarray(frames,int);cameras=np.asarray(cameras,float)
        B=len(cameras);delivery=np.zeros(B);motion=np.zeros(B)
        if (cameras.shape!=(B,len(frames),3) or (native is not None and native.shape!=frames.shape)
                or not np.isfinite(cameras).all() or np.any(cameras[:,:,0]<1.-1e-9)
                or np.any(np.diff(frames)<=0) or (native is not None and np.any(native<0))):
            raise ValueError('actual sampled camera/native arrays required')
        samples=self.delivery[source];at=np.searchsorted(samples.frames,frames)
        safe=np.minimum(at,max(0,len(samples.frames)-1))
        if len(samples.frames):
            hit=(at<len(samples.frames))&(samples.frames[safe]==frames)
            if native is not None:hit&=samples.native[safe]==native
            if hit.any():delivery=delivery_amount(samples,at[hit],cameras[:,hit]).sum(-1)/self.fps
        for row in self.images[source]:
            pair=np.asarray(row['output_frames'],int);where=np.searchsorted(frames,pair)
            if np.any(where>=len(frames)) or not np.array_equal(frames[where],pair):continue
            if native is not None and not np.array_equal(native[where],row['native_pair']):continue
            first,last=cameras[:,where[0]],cameras[:,where[1]]
            energy,supported=displayed_energy_many(row['observation'],first,last,seconds=1./self.fps)
            digital=self.fps*(np.sum(((last[:,1:]-first[:,1:])*[16/9,1.])**2,axis=-1)
                +np.log(last[:,0]/first[:,0])**2)
            motion+=np.where(supported,energy-digital,0.)
        return dict(observed_action_delivery_credit=delivery,composed_image_motion_delta=motion,
            adjustment=motion-delivery if self.delivery_in_base else motion)

    def direct_adjust(self,arcs,i,j,source,ps,qs,costs,allowed,*,frames=None,correction=False):
        """Independent direct oracle; also handles short/nonseparable paths."""
        from .boundary_arc_costs import camera_samples_many
        first,final=map(int,self.nodes[[i,j]])
        if frames is None:frames=self.frames(source,first,final)
        if not len(frames):return costs
        native=self.native(source,first,final,frames);result=costs.copy();self.calls+=1
        for code in range(6):
            pi,qi=np.nonzero(allowed[:,:,code])
            if not len(pi):continue
            baseline=arcs.dense[source][0 if code>=4 else code][first:final]
            # Batch workspace, NOT candidate/beam pruning. Every admitted
            # camera is still evaluated; no full hold is copied per history.
            rivals=max((len(v) for v in self.delivery[source].competitors),default=0)+1
            per_camera=max(4096,len(frames)*(256+96*rivals))
            batch=max(1,(32*2**20)//per_camera)
            for first_row in range(0,len(pi),batch):
                pp,qq=pi[first_row:first_row+batch],qi[first_row:first_row+batch]
                entry=np.asarray([arcs.entries[i][ps[k]].strip for k in pp])
                leave=np.asarray([arcs.exits[j][qs[k]].strip for k in qq])
                cameras=camera_samples_many(entry,leave,baseline,frames-first,code=code)
                effect=self.effects(source,frames,native,cameras)
                delta=effect['adjustment']
                if correction:delta=delta-self._effects(source,frames,None,cameras)['adjustment']
                result[pp,qq,code]+=delta;self.sampled_camera_rows+=len(pp)*len(frames)
        return result

    def prepare(self,arcs):
        """Compile the existing four fixed interiors and19-frame patches once.

        These are provisional geometry integrals, not native-clock authority.
        Each admitted interval still checks ALL consumed native ordinals and
        corrects every mismatching sample with its actual camera values.
        """
        import weakref
        from .boundary_arc_costs import cameras
        if self._prepared_for is not None and self._prepared_for() is arcs:return
        final=int(self.nodes[-1]);self._prefix=[];self._motion_at_first=[]
        for source,paths in enumerate(arcs.dense):
            paths=np.asarray(paths)
            arcs.check();cost=np.zeros((4,final));motion=np.zeros_like(cost)
            table=self.delivery[source]
            if self.delivery_in_base and len(table.frames):
                cost[:,table.frames]-=delivery_amount(table,np.arange(len(table.frames)),paths[:,table.frames])/self.fps
            for row in self.images[source]:
                a,b=row['output_frames']
                if not 0<=a<b<final:continue
                first,last=paths[:,a],paths[:,b]
                energy,known=displayed_energy_many(row['observation'],first,last,seconds=1/self.fps)
                digital=self.fps*(np.sum(((last[:,1:]-first[:,1:])*[16/9,1.])**2,axis=-1)
                    +np.log(last[:,0]/first[:,0])**2)
                motion[:,b]+=np.where(known,energy-digital,0.)
            self._prefix.append(np.pad(np.cumsum(cost+motion,axis=1),((0,0),(1,0))))
            self._motion_at_first.append(motion)
        self._entry=[np.zeros((len(v),4)) for v in arcs.entries]
        self._exit=[np.zeros((len(v),4)) for v in arcs.exits]
        for n,tick in enumerate(self.nodes):
            arcs.check()
            for source,cat in enumerate(arcs.cats):
                for side,ports,output in ((0,arcs.entries[n],self._entry[n]),(1,arcs.exits[n],self._exit[n])):
                    first=int(tick) if side==0 else int(tick)-38
                    if first<0 or first+38>final:continue
                    ids=np.array([k for k,p in enumerate(ports) if p.source==source],int)
                    if not len(ids):continue
                    strips=np.asarray([ports[k].strip for k in ids])
                    for code in range(4):
                        baseline=arcs.dense[source][code][first:first+38]
                        if side==0:
                            path,valid=cameras(strips,np.broadcast_to(baseline[-7:],strips.shape),baseline,cap=cat.bank['zoom_cap'])
                            frames=np.arange(first,first+19);new=path[:,:19];old=np.broadcast_to(baseline[:19],new.shape)
                        else:
                            path,valid=cameras(np.broadcast_to(baseline[:7],strips.shape),strips,baseline,cap=cat.bank['zoom_cap'])
                            frames=np.arange(first+19,first+38);new=path[:,-19:];old=np.broadcast_to(baseline[-19:],new.shape)
                        safe=np.where(valid[:,None,None],new,old)
                        output[ids,code]=(self._effects(source,frames,None,safe)['adjustment']
                            -self._effects(source,frames,None,old)['adjustment'])
        self._prepared_for=weakref.ref(arcs)

    def _clock_mismatches(self,source,first,final):
        frames=self.frames(source,first,final)
        if not len(frames):return frames
        actual=self.native(source,first,final,frames);table=self.delivery[source];bad=[]
        if len(table.frames):
            at=np.searchsorted(table.frames,frames);safe=np.minimum(at,len(table.frames)-1)
            hit=(at<len(table.frames))&(table.frames[safe]==frames)
            bad.extend(frames[hit&(table.native[safe]!=actual)])
        for row in self.images[source]:
            pair=np.asarray(row['output_frames']);where=np.searchsorted(frames,pair)
            if np.any(where>=len(frames)) or not np.array_equal(frames[where],pair):continue
            if not np.array_equal(actual[where],row['native_pair']):bad.extend(pair)
        return np.unique(bad).astype(np.int64)

    def adjust(self,arcs,i,j,source,ps,qs,costs,allowed):
        first,final=map(int,self.nodes[[i,j]])
        if final-first<38:return self.direct_adjust(arcs,i,j,source,ps,qs,costs,allowed)
        self.prepare(arcs);result=costs.copy()
        baseline=self._prefix[source][:,final]-self._prefix[source][:,first]-self._motion_at_first[source][:,first]
        result[:,:,:4]+=baseline[None,None,:]+self._entry[i][ps,None,:]+self._exit[j][None,qs,:]
        self.prefix_interval_uses+=1
        bad=self._clock_mismatches(source,first,final)
        if len(bad):
            mask=allowed.copy();mask[:,:,4:]=False
            result=self.direct_adjust(arcs,i,j,source,ps,qs,result,mask,frames=bad,correction=True)
            self.clock_correction_rows+=len(bad)
        direct=allowed.copy();direct[:,:,:4]=False
        if direct.any():result=self.direct_adjust(arcs,i,j,source,ps,qs,result,direct)
        return result

    def replay(self,source,first,final,curve):
        frames=self.frames(source,first,final)
        if not len(frames):return dict(observed_action_delivery_credit=0.,composed_image_motion_delta=0.,adjustment=0.)
        native=self.native(source,first,final,frames)
        return {k:float(v[0]) for k,v in self.effects(source,frames,native,np.asarray(curve)[None,frames-first]).items()}

    def receipt(self):
        return dict(contract='fixed_episode_actual_source_native_action_delivery_composed_image_v1',
            targets=list(self.targets),observed_samples=[len(s.frames) for s in self.delivery],
            image_pairs=[len(v) for v in self.images],conflicting_image_pairs=self.conflicts,
            candidate_calls=self.calls,sampled_camera_rows=self.sampled_camera_rows,
            compiled_prefix_interval_uses=self.prefix_interval_uses,exact_native_clock_correction_rows=self.clock_correction_rows,
            entry_density_times_hold_duration=False,seven_frame_credit_used=False,
            utility_is_pointwise_and_segment_split_invariant=True,unknown_exposure_interpolated=False,
            actual_candidate_native_ordinals_checked=True,source_NN=0,source_decode=0,
            new_camera_families=False,new_cut_quota=False,preference_model_fitted=False,
            source_content_delivery_reward_in_base=self.delivery_in_base,
            professional_quality_approved=False)
