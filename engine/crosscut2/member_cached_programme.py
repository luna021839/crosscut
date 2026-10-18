"""Small portrait-aware cached MEMBER baseline, independent of GROUP E6.

Camera interpolation is a proposed trajectory, never a measured pose/face.
Cut costs consume exact source-native flanks; sparse gaps keep a finite
unknown cost. No source-count quota, named identity claim, or model calls.
"""
import numpy as np


def follow_crops(boxes,known,*,source_size,output_size,zoom,fps):
    """A fixed-size view with a smooth center; no detector-size zoom breathing."""
    from .aspect_viewport import aspect_crop
    boxes=np.asarray(boxes,float);known=np.asarray(known,bool)
    if (boxes.ndim!=2 or boxes.shape[1]!=4 or known.shape!=boxes.shape[:1]
            or not np.isfinite(boxes).all() or not len(boxes) or fps<=0):
        raise ValueError('finite source boxes and explicit availability required')
    base=aspect_crop(source_size,output_size,zoom=zoom)
    size=np.array([base.width,base.height]);t=np.arange(len(boxes));center=np.full((len(t),2),.5)
    if known.any():
        observed=boxes[:,:2]+boxes[:,2:]/2
        # Face near the upper third, leaving space for the actual arm action.
        observed=observed+(.5-np.array([.5,.28]))*size
        for axis in (0,1):center[:,axis]=np.interp(t,t[known],observed[known,axis])
        window=max(1,round(.6*fps)//2*2+1);radius=window//2
        for axis in (0,1):
            center[:,axis]=np.convolve(np.pad(center[:,axis],(radius,radius),mode='edge'),
                np.ones(window)/window,mode='valid')
    center=np.clip(center,size/2,1.-size/2)
    return np.c_[center-size/2,np.tile(size,(len(t),1))]


def projected(boxes,poses,crops,*,output_size):
    """Realized image-plane coordinates and original confidence; no clamping."""
    boxes=np.asarray(boxes,float);poses=np.asarray(poses,float);crops=np.asarray(crops,float)
    if boxes.shape!=crops.shape or boxes.shape[-1]!=4 or poses.shape!=boxes.shape[:-1]+(8,3):
        raise ValueError('one actual crop, box and screen-pose row required')
    face=boxes.copy();face[...,:2]=(boxes[...,:2]-crops[...,:2])/crops[...,2:]
    face[...,2:]=boxes[...,2:]/crops[...,2:]
    pose=poses.copy();pose[...,:2]=(poses[...,:2]-crops[...,None,:2])/crops[...,None,2:]
    return face,pose


def pair_cost(left_box,right_box,left_pose,right_pose,left_known,right_known,*,output_size,unknown_geometry=2.):
    """All physical state pairs, using the requested output's isotropic metric."""
    a,b=np.asarray(left_box,float),np.asarray(right_box,float)
    pa,pb=np.asarray(left_pose,float),np.asarray(right_pose,float)
    ka,kb=np.asarray(left_known,bool),np.asarray(right_known,bool)
    if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=4 or b.shape[1]!=4 or pa.shape!=(len(a),8,3) or pb.shape!=(len(b),8,3)
            or ka.shape!=(len(a),) or kb.shape!=(len(b),) or type(unknown_geometry) not in (int,float)
            or not np.isfinite(unknown_geometry) or not 0<=unknown_geometry<=4):
        raise ValueError('explicit source-native face/pose flanks required')
    aspect=output_size[0]/output_size[1];metric=np.array([aspect,1.])
    ca=a[:,:2]+a[:,2:]/2;cb=b[:,:2]+b[:,2:]/2
    height=np.sqrt(np.maximum(.0001,a[:,None,3]*b[None,:,3]))
    distance=np.linalg.norm((ca[:,None]-cb[None,:])*metric,axis=-1)/height
    scale=np.abs(np.log(np.maximum(.001,a[:,None,3])/np.maximum(.001,b[None,:,3])))
    # Explicit editorial risk for an unmeasured flank, never a measured
    # contradiction or a source exclusion. Keep the original2 by default.
    geometry=np.where(ka[:,None]&kb[None,:],np.minimum(4.,distance+scale/.3),unknown_geometry)
    bones=np.array([(0,2),(2,4),(1,3),(3,5)])
    va=(pa[:,bones[:,1],:2]-pa[:,bones[:,0],:2])*metric
    vb=(pb[:,bones[:,1],:2]-pb[:,bones[:,0],:2])*metric
    la=np.linalg.norm(va,axis=-1);lb=np.linalg.norm(vb,axis=-1)
    gooda=(pa[:,bones,2]>=.3).all(-1)&(la>.005)&ka[:,None]
    goodb=(pb[:,bones,2]>=.3).all(-1)&(lb>.005)&kb[:,None]
    good=gooda[:,None]&goodb[None,:]
    angle=np.linalg.norm(va[:,None]/np.maximum(.001,la[:,None,:,None])-
        vb[None,:]/np.maximum(.001,lb[None,:,:,None]),axis=-1)/2
    # Worst visible arm prevents three matching torso points from concealing
    # one opposed wrist. Missing limbs are not a hard source veto.
    pose=np.where(good,angle,.5).max(-1)
    return .15+.35*geometry+.65*pose


def solve(nodes,loss,source_ids,transition,*,fps,maximum_seconds=5.,minimum_seconds=1.1,
          preferred_seconds=2.6,cut_music_penalty=None):
    """Whole-film semi-Markov DP; camera/state is retained over each hold."""
    nodes=np.asarray(nodes);loss=np.asarray(loss,float);sources=np.asarray(source_ids)
    if (nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or nodes[0]!=0 or np.any(np.diff(nodes)<=0)
            or loss.shape!=(len(sources),int(nodes[-1])) or not np.isfinite(loss).all()
            or not 0<minimum_seconds<preferred_seconds<maximum_seconds or fps<=0):
        raise ValueError('finite complete output-frame/state domain required')
    K=len(sources);M=len(nodes);prefix=np.c_[np.zeros(K),loss.cumsum(1)]/fps
    dp=np.full((M,K),np.inf);entry=np.full_like(dp,np.inf);entry[0]=0.
    parent=np.full((M,K),-1,int);start=np.full((M,K),-1,int)
    choose=np.full((M,K),-1,int)
    for end in range(1,M):
        candidates=np.flatnonzero((nodes[end]-nodes[:end]<=maximum_seconds*fps)&
            ((nodes[end]-nodes[:end]>=minimum_seconds*fps)|(end==M-1)))
        if not len(candidates):continue
        d=(nodes[end]-nodes[candidates])/fps
        cost=entry[candidates]+(prefix[:,nodes[end],None]-prefix[:,nodes[candidates]]).T
        cost+=.65*(d[:,None]-preferred_seconds)**2
        at=cost.argmin(0);dp[end]=cost[at,np.arange(K)];start[end]=candidates[at]
        parent[end]=choose[candidates[at],np.arange(K)]
        if end<M-1:
            pair=np.asarray(transition(int(nodes[end])),float)
            if pair.shape!=(K,K) or np.isnan(pair).any():raise ValueError('one finite-or-infinite actual state-pair matrix required')
            pair=np.where(sources[:,None]==sources[None,:],np.inf,pair)
            total=dp[end,:,None]+pair
            if cut_music_penalty is not None:total+=float(cut_music_penalty[end])
            choose[end]=total.argmin(0);entry[end]=total[choose[end],np.arange(K)]
    state=int(dp[-1].argmin());value=float(dp[-1,state])
    if not np.isfinite(value):raise ValueError('No complete finite member programme in the declared domain')
    holds=[];end=M-1
    while end:
        beginning=int(start[end,state]);holds.append((int(nodes[beginning]),int(nodes[end]),state))
        state=int(parent[end,state]);end=beginning
    return tuple(reversed(holds)),value
