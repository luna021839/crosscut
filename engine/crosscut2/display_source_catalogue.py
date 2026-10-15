"""Source-local physical observation catalogue and declared content quadrature.

K always reads exact cached native inputs. Content integration uses nearest
cached face/body rasters within six output frames and the same native shot;
it does NOT claim a dense person track or new all-frame observations.
"""
import numpy as np
from .physical_display_observations import (focal_metadata, duplicate_wrists,
    UNMEASURED, MODEL_ONLY, PIXEL_SUPPORTED, CONTRADICTED)
from .played_partial_pose import source_activity
from .source_clock_action import endpoint_correspondence


class SourceCatalogue:
    def __init__(self, nodes, bank, poses, rows, style, events, *, fps,member_hints=None,featured_factor=None,subjects=None,
                 editorial_content=False,framing_evidence=None):
        self.nodes=np.asarray(nodes,int);self.bank=bank;self.fps=fps;self.final=int(nodes[-1]);self.style=style
        self.subjects=subjects
        self.editorial_content=bool(editorial_content)
        self.framing_evidence=framing_evidence
        N=len(nodes);self.body=np.zeros((N,4,3,8,3),np.float32) if poses is None else np.asarray(poses,np.float32).copy()
        self.member_hints=np.full(bank['heads'].shape[:3],-1,int) if member_hints is None else np.asarray(member_hints,int)
        self.featured_factor=np.ones(bank['heads'].shape[:3]) if featured_factor is None else np.asarray(featured_factor,float)
        if (self.member_hints.shape!=bank['heads'].shape[:3] or self.featured_factor.shape!=self.member_hints.shape
                or not np.isfinite(self.featured_factor).all() or np.any((self.featured_factor<0)|(self.featured_factor>1.+1e-9))):
            raise ValueError('source/head-bound soft featured-member evidence required')
        self.evidence=np.where(self.body[...,2]>=.3,MODEL_ONLY,UNMEASURED).astype(np.uint8)
        self.salience=np.full((N,2,8),.85,np.float32)
        for row in rows:
            raw=row.get('raw_body_hypotheses',row['body'])
            for n,side in zip(*np.nonzero((row['indices']>=0)&(row['indices']<3))):
                h=int(row['indices'][n,side]);sl=slice(side*2,side*2+2)
                self.body[n,sl,h]=raw[n,sl]
                reliability=row.get('body_reliability')
                if reliability is not None and row.get('pixel_part_scope',np.zeros(N,bool))[n]:
                    kind=np.where(reliability[n,sl]==1.,PIXEL_SUPPORTED,
                        np.where(reliability[n,sl]==0.,CONTRADICTED,MODEL_ONLY))
                    self.evidence[n,sl,h]=np.where(raw[n,sl,:,2]>=.3,kind,UNMEASURED)
                if 'utility_parts' in row:self.salience[n,side,h]=np.exp(-row['utility_parts']['salience'][n,side]/1.1)
        self.ambiguity=duplicate_wrists(self.body)
        self.endpoint_matches=endpoint_correspondence(bank['heads'],bank['vectors']) if 'vectors' in bank else np.full((N,3),-1,int)
        self.physical=np.ones((N,2),bool)
        self.native=np.asarray(bank['pose_native_indices'],int)
        native_cuts=np.asarray(bank.get('native_cut_frames',[]),int)
        for side in (0,1):
            a,b=self.native[:,2*side],self.native[:,2*side+1]
            self.physical[:,side]=b>a
            for n,(first,last) in enumerate(zip(a,b)):
                self.physical[n,side]&=not np.any((native_cuts>first)&(native_cuts<=last))
            self.physical[:,side]&=(self.nodes-7>=0) if side==0 else (self.nodes+6<self.final)
        self.quality=np.full(bank['heads'].shape[:3],.5,np.float32)
        if 'native_readability_valid' in bank:
            contrast,edge,clipped,height=np.moveaxis(bank['native_readability'],-1,0)
            q=np.minimum(1.,height/120.)*np.minimum(1.,edge/.5)*np.minimum(1.,contrast/.075)*(1.-clipped)
            self.quality=np.where(bank['native_readability_valid'],q,self.quality)
        self.black=np.clip((style['presence'][...,0]-.8)/.2,0,1)
        from .scene_readout import roi_readability
        self.roi_readability,self.roi_readability_valid=roi_readability(bank)
        # Explicit numerical integration domain; never used as native K proof.
        ticks=np.clip(self.nodes[:,None]+[-1,0],0,self.final-1).ravel()
        order=np.argsort(ticks,kind='stable');ordered=ticks[order]
        frames=np.arange(self.final);right=np.searchsorted(ordered,frames)
        a=np.maximum(0,right-1);b=np.minimum(len(ordered)-1,right)
        chosen=np.where(frames-ordered[a]<ordered[b]-frames,a,b)
        original=order[chosen];self.near_n=original//2;self.near_side=original%2
        closest=ordered[chosen];cuts=np.asarray(events,int)
        self.content_scope=(np.abs(closest-frames)<=6)&(np.searchsorted(cuts,closest,side='right')==np.searchsorted(cuts,frames,side='right'))
        # Activity belongs to the SAME physical face sample used by rates(),
        # not to the source or the face dominant in an unrelated native crop.
        # Pose is unavailable for face3..7: those faces cannot borrow face0's
        # motion. This is bounded sample quadrature, not a dense actor track.
        self.own=np.zeros(bank['heads'].shape[:3])
        for n in range(N):
            for side in (0,1):
                sl=slice(2*side,2*side+2)
                for h in range(3):
                    observation=dict(head=bank['heads'][n,side,h],body=self.body[n,sl,h],
                        evidence=self.evidence[n,sl,h],ownership_ambiguity=self.ambiguity[n,sl,h],
                        physical_window_valid=bool(self.physical[n,side]),native_indices=self.native[n,sl])
                    self.own[n,side,h]=source_activity(observation,fps=fps)

    def target_readability(self,n,side,head_index):
        """Same original-detail ROI units for content and proposal quality.

        UNKNOWN retains its explicit neutral.5 and separate validity mask.
        Neither added zoom nor global background brightness creates detail.
        """
        height=self.bank['heads'][n,side,head_index,2]
        return self.roi_readability[n,side,head_index]*np.minimum(1.,height/.12)

    def observation(self,n,side,camera_strip,*,target_head_index=None,include_subjects=True):
        camera_strip=np.asarray(camera_strip,float)
        camera=camera_strip[-1 if side==0 else 0]
        heads=self.bank['heads'][n:n+1,side:side+1]
        selected,index,ambiguity,scene=focal_metadata(heads,camera.reshape(1,1,3))
        displayed=int(index[0,0]);h=displayed;sl=slice(side*2,side*2+2)
        target_attention=1. if displayed>=0 else 0.
        if target_head_index is not None:
            if (type(target_head_index) is not int or not 0<=target_head_index<heads.shape[2]
                    or heads[0,0,target_head_index,2]<=.019 or heads[0,0,target_head_index,3]<.75):
                raise ValueError('pinned target must be an actually observed face at this endpoint')
            h=target_head_index
            selected=heads[:,:,h].copy()
            center=.5+camera[0]*(heads[0,0,:,:2]-camera[1:])
            size=heads[0,0,:,2]*camera[0]
            half=size[:,None]*[.45*9/16,.5]
            extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
            score=extent.prod(-1)*np.exp(-((center[:,0]-.5)/.4)**2)
            score=np.where((heads[0,0,:,2]>.019)&(heads[0,0,:,3]>=.75),score,0.)
            target_attention=float(score[h]/max(1e-12,float(score.max())))
        body=np.zeros((2,8,3),np.float32) if not 0<=h<3 else self.body[n,sl,h]
        evidence=np.zeros((2,8),np.uint8) if not 0<=h<3 else self.evidence[n,sl,h]
        ownership=np.zeros((2,8),bool) if not 0<=h<3 else self.ambiguity[n,sl,h]
        appearance=np.zeros(128,np.float32)
        if 0<=h<3 and 'vectors' in self.bank:
            appearance=np.asarray(self.bank['vectors'][n,side,h],np.float32).copy()
        appearance.setflags(write=False)
        endpoint_native=int(self.native[n,1 if side==0 else 2])
        before=h if side==0 else next((a for a,b in enumerate(self.endpoint_matches[n]) if b==h and b>=0),-1)
        after=int(self.endpoint_matches[n,before]) if 0<=before<3 else -1
        if side==1:after=h
        cuts=np.asarray(self.bank.get('native_cut_frames',[]),int)
        if np.any((cuts>self.native[n,1])&(cuts<=self.native[n,2])):
            if side==0:after=-1
            else:before=-1
        common=0<=h<3
        common_body=np.zeros((4,8,3),np.float32);common_evidence=np.zeros((4,8),np.uint8);common_ambiguity=np.zeros((4,8),bool)
        common_heads=np.zeros((4,4),np.float32);sample_mask=np.zeros(4,bool)
        common_readability=np.full(4,.5);common_readability_valid=np.zeros(4,bool)
        if common:
            for source_side,target in enumerate((before,after)):
                if not 0<=target<3:continue
                endpoint=1 if source_side==0 else 2
                slots=[endpoint]
                if self.physical[n,source_side]:slots.append(0 if source_side==0 else 3)
                for slot in slots:
                    common_body[slot]=self.body[n,slot,target];common_evidence[slot]=self.evidence[n,slot,target]
                    common_ambiguity[slot]=self.ambiguity[n,slot,target]
                    common_heads[slot]=self.bank['heads'][n,source_side,target];sample_mask[slot]=True
                common_readability[endpoint]=self.roi_readability[n,source_side,target]
                common_readability_valid[endpoint]=self.roi_readability_valid[n,source_side,target]
        elif h>=0:
            # A fourth/later face has no cached body/appearance chain, but
            # its actually pinned endpoint face is still observed. Do not
            # discard that face or attach face0's body to fill the gap.
            endpoint=1 if side==0 else 2
            common_heads[endpoint]=self.bank['heads'][n,side,h];sample_mask[endpoint]=True
            common_readability[endpoint]=self.roi_readability[n,side,h]
            common_readability_valid[endpoint]=self.roi_readability_valid[n,side,h]
        style_valid=bool(h>=0 and 'colour_valid' in self.style and self.style['colour_valid'][n,side,h])
        colour=self.style['colour'][n,side,h].copy() if style_valid else np.zeros((2,40))
        original_heads=np.zeros((4,8,4),np.float32)
        original_heads[[1,2]]=self.bank['heads'][n]
        visible_quality=0.
        if h>=0:
            head=selected[0,0];center=.5+camera[0]*(head[:2]-camera[1:])
            half=head[2]*camera[0]*np.array([.45*9/16,.5])
            extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
            fraction=float(np.prod(extent/np.maximum(1e-9,2*half)))
            if self.editorial_content:
                visible_quality=float(self.target_readability(n,side,h)*fraction*self.featured_factor[n,side,h])
            else:
                visible_quality=float(np.sqrt(self.quality[n,side,h])*min(1.,head[2]*camera[0]/.22)
                    *np.sqrt(self.salience[n,side,h])*fraction*(1.-self.black[n,side])*self.featured_factor[n,side,h])
        result=dict(head=selected[0,0],head_index=h,body=body,evidence=evidence,
            ownership_ambiguity=ownership,physical_window_valid=bool(self.physical[n,side]),
            native_indices=self.native[n,sl],camera_strip=camera_strip,side=side,
            focal_ambiguity=float(ambiguity[0,0]),scene_kind=int(scene[0,0]),
            target_pinned_before_camera=target_head_index is not None,
            displayed_focal_head_index=displayed,target_is_displayed_focal=h==displayed,
            target_attention_ratio=target_attention,
            source_clock_body=common_body,source_clock_evidence=common_evidence,
            source_clock_ambiguity=common_ambiguity,source_clock_subject_associated=common,
            source_clock_endpoint_head_proxies=common_heads,
            source_clock_observed_sample_mask=sample_mask,
            source_clock_canonical_frame=int(self.nodes[n]),
            source_clock_final_frame=self.final,
            source_clock_subject_key=(int(self.nodes[n]),int(before),int(after)),
            source_clock_output_frames=np.clip(int(self.nodes[n])+np.array([-7,-1,0,6]),0,self.final-1),
            source_clock_native_indices=self.native[n].copy(),
            source_clock_original_heads=original_heads,
            source_clock_native_roi_readability=common_readability,
            source_clock_native_roi_measured=common_readability_valid,
            source_colour_histogram=colour,source_colour_valid=style_valid,
            displayed_native_quality_proxy=visible_quality,
            displayed_native_quality_measured=bool(h>=0 and self.roi_readability_valid[n,side,h]),
            displayed_native_quality_contract=('target_ROI_native_detail_visibility_original_role_v1'
                if self.editorial_content else 'legacy_display_quality_v1'),
            recurrent_member_hint=-1 if h<0 else int(self.member_hints[n,side,h]),
            source_clock_context_is_sparse_model_hypothesis=True,
            appearance_vector=appearance,appearance_is_not_named_identity=True,
            person_native_ordinal=endpoint_native,
            person_rectangles=self.bank.get('person_rectangles_by_native',{}).get(endpoint_native))
        return result if self.subjects is None or not include_subjects else self.subjects.augment(result,n,side,h)

    def rates(self,frames,cameras):
        """Vectorized (...,len(frames),3) replay camera content rates."""
        frames=np.asarray(frames,int);camera=np.asarray(cameras,float)
        if camera.shape[-2:]!=(len(frames),3) or frames.min(initial=0)<0 or frames.max(initial=0)>=self.final:
            raise ValueError('actual finite source-local replay frame/camera domain required')
        n,side=self.near_n[frames],self.near_side[frames]
        heads=self.bank['heads'][n,side]
        center=.5+camera[...,None,:1]*(heads[...,:2]-camera[...,None,1:])
        size=heads[...,2]*camera[...,None,0]
        half=size[...,None]*[.45*9/16,.5]
        extent=np.maximum(0.,np.minimum(1.,center+half)-np.maximum(0.,center-half))
        scores=np.prod(extent,-1)*np.exp(-((center[...,0]-.5)/.4)**2)
        seen=(heads[...,2]>.019)&(heads[...,3]>=.75)&self.content_scope[frames,None]
        scores=np.where(seen,scores,0.);index=scores.argmax(-1)
        h=heads[np.arange(len(frames)),index]
        displayed_height=h[...,2]*camera[...,0]
        focal_center=.5+camera[...,:1]*(h[...,:2]-camera[...,1:])
        focal_half=displayed_height[...,None]*[.45*9/16,.5]
        focal_extent=np.maximum(0.,np.minimum(1.,focal_center+focal_half)-np.maximum(0.,focal_center-focal_half))
        fraction=np.prod(focal_extent/np.maximum(1e-8,2*focal_half),axis=-1)
        quality=self.quality[n,side,index];salience=self.salience[n,side,index]
        visibility=np.sqrt(quality)*np.minimum(1.,displayed_height/.22)*np.sqrt(salience)*fraction*(1.-self.black[n,side])*self.featured_factor[n,side,index]
        visibility=np.where(scores.max(-1)>0,np.clip(visibility,0,1),0.)
        body=self.body[n,np.where(side==0,1,2),np.minimum(index,2)]
        valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
        valid&=(index<3)[...,None]&self.content_scope[frames,None]
        projected=.5+camera[...,None,:1]*(body[...,:2]-camera[...,None,1:])
        crop=(valid&((projected<0).any(-1)|(projected>1).any(-1))).sum(-1)/np.maximum(1,valid.sum(-1))
        if self.editorial_content:
            # Native ROI deficits remain explicit; digital enlargement cannot
            # create detail. The native .12 sampling size and item-wise units
            # are engineering conventions, not fitted aesthetic thresholds.
            # ROI deficits and UNKNOWN stay distinct. Global black background
            # is style, not a recurring defect in an equally readable face.
            readable=self.target_readability(n,side,index)
            shown=np.where(scores.max(-1)>0,readable*fraction,0.)
            role=1.-self.featured_factor[n,side,index]
            # Reuse joint_cameras' oversized-head effort, applying it only to
            # EXTRA enlargement. An original tight shot is not called missing.
            tight=6.*np.maximum(0.,displayed_height-np.maximum(.24,h[...,2]))
            framing=0.
            if self.framing_evidence is not None:
                evidence=self.framing_evidence
                target=evidence['height'][n,side];confidence=evidence['confidence'][n,side]
                # A detector-observed EMPTY/occluded viewport is not an
                # unobserved frame. It must not receive a free role-deficit
                # exemption that makes legs-only footage beat readable faces.
                # Unsupported temporal gaps remain unknown; no source ban.
                known=self.content_scope[frames]
                size_deficit=np.maximum(0.,np.log(np.maximum(.019,target)/np.maximum(.019,displayed_height)))/np.log(1.5)
                # Formation evidence asks for people actually inside THIS
                # viewport. It is not a hard requirement to show all6 always.
                in_view=seen&(center[...,0]>=0)&(center[...,0]<=1)&(center[...,1]>=0)&(center[...,1]<=1)
                count=(in_view&(size>=.5*np.maximum(.019,displayed_height)[...,None])).sum(-1)
                wanted=evidence['people'][n,side]
                coverage=np.maximum(0.,wanted-count)/np.maximum(1.,wanted)
                framing=np.where(known,confidence*np.where(evidence['formation'][n,side],coverage,size_deficit),0.)
            return 1.-shown+crop+shown*role+tight+framing+.1*np.log(camera[...,0])/np.log(2.)
        return 1.-visibility+crop+.1*np.log(camera[...,0])/np.log(2.)-visibility*self.own[n,side,index]
