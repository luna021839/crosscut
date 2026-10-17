"""Put source-exact fine subject moments BEFORE whole-programme selection.

No finished edit supplies cut times. Immutable source windows propose common
music instants; the ordinary global source/camera DAG chooses the programme.
Unknown observations remain unknown. Old grid rows are preserved exactly.
"""
import hashlib
import math
import numpy as np

from .subject_action_match import SubjectClock
from .subject_cut_moments import search
from .fixed_holds import local_actor_hypotheses
from .source_member_scores import MemberScoreSeeds, bank_signature


def schedule(warp, clock, ticks, *, master_start, fps, final):
    return np.asarray([round(warp.master_to_source(master_start+max(0,min(final-1,int(t)))/fps)
        * clock['fps']) for t in np.asarray(ticks).ravel()], np.int64).reshape(np.asarray(ticks).shape)


def propose(nodes, windows, warps, clocks, caps, *, master_start, fps,
            maximum_new_nodes=96, additional_detailed_nodes=0, check=lambda:None):
    nodes=np.asarray(nodes,int);final=int(nodes[-1]);moments={};detailed={};tested=0
    if (type(maximum_new_nodes) is not int or not 0<=maximum_new_nodes<=256
            or type(additional_detailed_nodes) is not int or not 0<=additional_detailed_nodes<=96
            or not 1<=maximum_new_nodes+additional_detailed_nodes<=256
            or not len(windows)==len(warps)==len(clocks)==len(caps)):
        raise ValueError('one explicit bounded fine-grid proposal domain is required')
    spans=[]
    for s,group in enumerate(windows):
        row=[]
        for window in group:
            if window.source!=s:raise ValueError('fine window borrowed another source')
            first=(warps[s].source_to_master(window.ordinals[0]/clocks[s]['fps'])-master_start)*fps
            last=(warps[s].source_to_master(window.ordinals[-1]/clocks[s]['fps'])-master_start)*fps
            row.append((window,max(0,int(math.ceil(first))),min(final-1,int(math.floor(last)))))
        spans.append(row)
    for a in range(len(windows)):
        for b in range(a+1,len(windows)):
            for left,la,lb in spans[a]:
                for right,ra,rb in spans[b]:
                    check();first,last=max(la,ra),min(lb,rb)
                    if last-first<12:continue
                    ticks=np.arange(first,last+1,dtype=np.int64)
                    ca=schedule(warps[a],clocks[a],ticks,master_start=master_start,fps=fps,final=final)
                    cb=schedule(warps[b],clocks[b],ticks,master_start=master_start,fps=fps,final=final)
                    valid=(ca>=left.ordinals[0])&(ca<=left.ordinals[-1])&(cb>=right.ordinals[0])&(cb<=right.ordinals[-1])
                    ticks=ticks[valid];ca=ca[valid];cb=cb[valid]
                    if len(ticks)<13 or np.any(np.diff(ticks)!=1):continue
                    signature=lambda w,t:hashlib.sha256(w.binding.encode()+ticks.tobytes()+t.tobytes()).hexdigest()
                    lc=SubjectClock(a,left.binding,ticks,ca,0,fps,signature(left,ca))
                    rc=SubjectClock(b,right.binding,ticks,cb,0,fps,signature(right,cb))
                    candidates=list(map(int,ticks[6:-6]));tested+=len(candidates)
                    if tested>20000:raise ValueError('fine pair/moment workspace exceeds the declared input batch')
                    result=search(left,right,lc,rc,candidate_frames=candidates,
                        outgoing_cap=caps[a],incoming_cap=caps[b],radius=6)
                    for row in result['supported']:
                        tick=int(row['cut_offset_frames'])
                        rank=(row['seam_arm_direction_residual'],row['cut_moment_score'],tick)
                        if tick not in moments or rank<moments[tick][0]:
                            moments[tick]=(rank,(a,b))
                        if additional_detailed_nodes:
                            from .detailed_pose import describe_moment,matrix
                            descriptions=[describe_moment(w,c,tick,side,row[key]) for w,c,side,key in
                                ((left,lc,0,'outgoing_camera'),(right,rc,1,'incoming_camera'))]
                            risk,known,channels,measured=matrix(descriptions[:1],descriptions[1:])
                            if known[0,0]:
                                raw=np.clip(2.*channels[0,0]-.5,0.,1.)
                                worst=float(np.where(measured[0,0],raw,0.).max())
                                rich=(max(rank[0],worst),row['cut_moment_score']+float(risk[0,0]-.5),tick)
                                if tick not in detailed or rich<detailed[tick][0]:detailed[tick]=(rich,(a,b))
    # Every disjoint observed time region gets a representative BEFORE extra
    # local alternatives. A particularly easy chorus cannot consume the batch.
    xs=sorted(moments);regions=[]
    for tick in xs:
        if not regions or tick>regions[-1][-1]+1:regions.append([])
        regions[-1].append(tick)
    ranked=[sorted(region,key=lambda tick:moments[tick][0]) for region in regions]
    old=set(map(int,nodes));chosen=[]
    if maximum_new_nodes:
        for depth in range(max(map(len,ranked),default=0)):
            for region in ranked:
                if depth<len(region) and region[depth] not in old:
                    chosen.append(region[depth])
                    if len(chosen)==maximum_new_nodes:break
            if len(chosen)==maximum_new_nodes:break
    # Retain EVERY old/body-only fine node. Add a finite number of genuinely
    # better detailed instants; never replace good baseline choices with a
    # greedy hand score. Full-song source/camera selection still follows.
    retained=old|set(chosen);improvements=[]
    for region in regions:
        current=[detailed[t][0] for t in region if t in retained and t in detailed]
        options=[t for t in region if t not in retained and t in detailed]
        if not current or not options:continue
        before=min(current);tick=min(options,key=lambda t:detailed[t][0]);after=detailed[tick][0]
        if after>=before:continue
        improvements.append((before[0]-after[0],before[1]-after[1],tick,before,after))
    additions=sorted(improvements,key=lambda r:(-r[0],-r[1],r[2]))[:additional_detailed_nodes]
    chosen.extend(row[2] for row in additions)
    new=np.unique(np.r_[nodes,chosen]).astype(np.int64)
    return new,dict(original_nodes=len(nodes),new_nodes=len(new)-len(nodes),
        supported_fine_instants=len(moments),observed_regions=len(regions),tested_pair_moments=tested,
        proposal_capacity=maximum_new_nodes,proposals_from_source_windows_not_finished_edit=True,
        additional_detailed_capacity=additional_detailed_nodes,additional_detailed_nodes=len(additions),
        detailed_supported_instants=len(detailed),old_fine_proposals_retained=True,
        detailed_improvements=[dict(frame=r[2],before=list(r[3]),after=list(r[4])) for r in additions],
        all_original_nodes_retained=True,new_neural_calls=0,new_rasters=0,
        source_retimed=False,global_sources_cuts_and_cameras_not_yet_selected=True)


def _frame_rows(windows):
    frames={}
    for window in windows:
        for at,native in enumerate(window.ordinals):
            if window.associated[at]:frames.setdefault(int(native),[]).append((window,at))
    return frames


def expand(nodes,new_nodes,banks,poses,styles,states,windows,warps,clocks,*,master_start,fps,check=lambda:None):
    """Project only real native detections; never nearest-time face interpolation."""
    nodes,new_nodes=np.asarray(nodes,int),np.asarray(new_nodes,int)
    if np.any(np.diff(new_nodes)<=0) or not np.isin(nodes,new_nodes).all() or new_nodes[[0,-1]].tolist()!=nodes[[0,-1]].tolist():
        raise ValueError('fine grid must preserve the complete old output domain')
    old_at=np.searchsorted(new_nodes,nodes);extra=np.flatnonzero(~np.isin(new_nodes,nodes));final=int(nodes[-1])
    outputs=[[],[],[],[]];receipts=[]
    for s,(bank,pose,style,state,group,warp,clock) in enumerate(zip(banks,poses,styles,states,windows,warps,clocks,strict=True)):
        check();frames=_frame_rows(group);N=len(new_nodes)
        def projected(array):
            value=np.asarray(array);result=np.zeros((N,)+value.shape[1:],value.dtype);result[old_at]=value;return result
        new=dict(bank)
        new['fine_grid_old_indices']=old_at.copy()
        new['face_observation_valid']=np.zeros((N,2),bool)
        new['face_observation_valid'][old_at]=bank.get('face_observation_valid',np.ones((len(nodes),2),bool))
        for key in ('heads','vectors','native_readability','native_readability_valid'):
            if key in new:new[key]=projected(bank[key])
        new['native_indices']=schedule(warp,clock,new_nodes[:,None]+[-1,0],master_start=master_start,fps=fps,final=final)
        new['pose_native_indices']=schedule(warp,clock,new_nodes[:,None]+[-7,-1,0,6],master_start=master_start,fps=fps,final=final)
        for key in ('native_indices','pose_native_indices'):
            if not np.array_equal(new[key][old_at],bank[key]):
                raise ValueError('fine projection changed an original source/native clock')
        p=projected(pose);st=projected(state)
        sty={key:projected(value) for key,value in style.items()}
        observations=0;conflicts=0
        for n in extra:
            for side in (0,1):
                native=int(new['native_indices'][n,side]);raw={};bad=set()
                actual=frames.get(native,[])
                new['face_observation_valid'][n,side]=bool(actual)
                for window,at in actual:
                    head=window.heads[at];key=tuple(map(float,head))
                    if key in bad:continue
                    if key in raw and raw[key][2] is not None and not np.array_equal(raw[key][1],window.vectors[at]):
                        raw[key]=(head,np.zeros(128),None,None);conflicts+=1
                        bad.add(key)
                    else:raw[key]=(head,window.vectors[at],window,at)
                    for competitor in window.competitors[at]:
                        k=tuple(map(float,competitor))
                        raw.setdefault(k,(competitor,np.zeros(128),None,None))
                values=sorted(raw.values(),key=lambda row:(-row[0][2],tuple(row[0])))[:8]
                for h,(head,vector,window,at) in enumerate(values):
                    new['heads'][n,side,h]=head;observations+=1
                    if h>=3:continue
                    new['vectors'][n,side,h]=vector
                    if window is None:continue
                    wanted=new['pose_native_indices'][n,side*2:side*2+2]-window.ordinals[0]
                    if wanted.min()<0 or wanted.max()>=len(window.ordinals):continue
                    cuts=np.asarray(window.native_cuts,int)
                    actual=new['pose_native_indices'][n,side*2:side*2+2]
                    if (not window.associated[wanted.min():wanted.max()+1].all()
                            or np.any((cuts>actual[0])&(cuts<=actual[1]))):continue
                    ambiguous=False
                    for other,oi in frames.get(native,[]):
                        if not np.array_equal(other.heads[oi],head):continue
                        positions=actual-other.ordinals[0]
                        if positions.min()<0 or positions.max()>=len(other.ordinals):continue
                        common=window.support[wanted]&other.support[positions]
                        if not np.array_equal(window.body[wanted][common],other.body[positions][common]):
                            ambiguous=True;break
                    if ambiguous:continue
                    body=window.body[wanted].copy();body[...,2]=np.where(window.support[wanted],body[...,2],0.)
                    p[n,side*2:side*2+2,h]=body
                    # State0 is deliberately UNKNOWN. Dense source windows
                    # supply their MODEL_ONLY action evidence separately.
        for old_value,new_value in ((pose,p),(state,st),*( (style[k],sty[k]) for k in style)):
            if not np.array_equal(new_value[old_at],old_value):raise AssertionError('original numeric observation changed')
        for target,value in zip(outputs,(new,p,sty,st),strict=True):target.append(value)
        receipts.append(dict(source=s,fine_observed_faces=observations,conflicting_descriptors=conflicts,
            fine_style_or_readability_fabricated=False,old_rows_unchanged=True))
    return (*outputs,dict(sources=receipts,new_neural_calls=0,new_rasters=0))


def rebind_hints(old_banks,new_banks,hints):
    """Retain original physical query profiles after adding measured occurrences."""
    result=[]
    for old,new,hint in zip(old_banks,new_banks,hints,strict=True):
        if isinstance(hint,MemberScoreSeeds):hint.verify(old)
        old_values=np.asarray(hint.cluster_hints if isinstance(hint,MemberScoreSeeds) else hint)
        old_labels,old_count=local_actor_hypotheses(old);labels,count=local_actor_hypotheses(new)
        if len(old_values)!=old_count:raise ValueError('old member hypotheses changed before fine-grid reuse')
        at=np.asarray(new.get('fine_grid_old_indices',[]))
        if at.shape!=(len(old_labels),) or at.dtype.kind not in 'iu':
            raise ValueError('fine member reuse requires exact preserved coarse-row addresses')
        votes=[set() for _ in range(count)]
        for n in range(len(old_labels)):
            j=int(at[n])
            if (not np.array_equal(new['native_indices'][j],old['native_indices'][n])
                    or not np.array_equal(new['heads'][j],old['heads'][n]) or not np.array_equal(new['vectors'][j],old['vectors'][n])):
                raise ValueError('fine member reuse lost an original physical occurrence')
            for side,h in np.ndindex(old_labels.shape[1:]):
                a,b=int(old_labels[n,side,h]),int(labels[j,side,h])
                if a>=0 and b>=0 and old_values[a]>=0:votes[b].add(int(old_values[a]))
        values=np.asarray([next(iter(v)) if len(v)==1 else -1 for v in votes],np.int16)
        result.append(MemberScoreSeeds(values,hint.vectors,hint.scores,bank_signature(new),hint.gallery_identity,
            hint.minimum_similarity,hint.minimum_member_margin) if isinstance(hint,MemberScoreSeeds) else values)
    return result


def observe_style(sources,banks,styles,*,cache,check=lambda:None,maximum_seconds=180.):
    """Measure only new, actually observed face rows; preserve coarse pixels.

    Missing colour must not masquerade as a measured zero costume change.
    The existing source-native decoder supplies real chest/background colour
    without new face/pose inference, frame storage, or old cache rewrites.
    """
    from .source_style import observe
    if not len(sources)==len(banks)==len(styles):
        raise ValueError('one physical source and style bank per fine grid required')
    selected=[];sub_banks=[];sub_sources=[]
    for s,(path,bank) in enumerate(zip(sources,banks,strict=True)):
        old=np.asarray(bank['fine_grid_old_indices'],int)
        extra=np.ones(len(bank['heads']),bool);extra[old]=False
        seen=((bank['heads'][...,2]>.019)&(bank['heads'][...,3]>=.75)).any((1,2))
        at=np.flatnonzero(extra&seen)
        if not len(at):continue
        selected.append((s,at));sub_sources.append(path)
        sub_banks.append(dict(heads=bank['heads'][at],native_indices=bank['native_indices'][at]))
    output=[{k:np.array(v,copy=True) for k,v in style.items()} for style in styles]
    if not selected:return output,dict(sources=[],new_neural_calls=0,decoded_native_inputs=0)
    values,receipts=observe(sources=sub_sources,banks=sub_banks,cache=cache,check=check,
        maximum_seconds=maximum_seconds)
    for (s,at),value,row in zip(selected,values,receipts,strict=True):
        for key,array in value.items():output[s][key][at]=array
        row['source']=s;row['fine_grid_rows']=len(at)
    for bank,before,after in zip(banks,styles,output,strict=True):
        old=bank['fine_grid_old_indices']
        if any(not np.array_equal(before[k][old],after[k][old]) for k in before):
            raise AssertionError('fine colour changed an original measured row')
    return output,dict(sources=receipts,new_neural_calls=0,
        decoded_native_inputs=sum(r['decoded_native_inputs'] for r in receipts),
        missing_style_fabricated=False,original_measured_rows_unchanged=True)
