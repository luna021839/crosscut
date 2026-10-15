"""Observed scene readability and composable residual readout burden.

Engineering units, NOT a cognitive/camera-cause/identity certificate. Original
ROI measurements stay distinct from UNKNOWN and background colour. Transfers
carry D_out=a*D_in+b and extra_seconds=c*D_in+d; no frame arrays per DP label.
"""
import numpy as np


IDENTITY = np.array([1., 0., 0., 0.])


def roi_readability(bank):
    """Item-wise native measurements; magnification cannot manufacture detail.

    The existing native_readability_cost units are retained, with native pixel
    height explicit. UNKNOWN=.5 is a neutral engineering value, not a measured
    passing face. Only valid, actual ROI measurements can support recovery.
    """
    shape = bank['heads'].shape[:3]
    valid = np.asarray(bank.get('native_readability_valid', np.zeros(shape, bool)))
    if valid.shape != shape or valid.dtype != bool:
        raise ValueError('face-bound native readability validity required')
    values = np.asarray(bank.get('native_readability', np.zeros(shape + (4,))), float)
    if (values.shape != shape + (4,) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values[..., 2] > 1)):
        raise ValueError('finite actual contrast/edge/clipping/native-height required')
    contrast, edge, clipped, height = np.moveaxis(values, -1, 0)
    deficit = (1.4 * np.maximum(0., 1. - edge / .5)
        + .3 * np.maximum(0., 1. - contrast / .075) + .5 * clipped
        + np.maximum(0., 1. - height / 120.))
    return np.where(valid, 1. / (1. + deficit), .5), valid


def then(first, second):
    """Apply first, then second. Associative, including nonzero incoming D."""
    x,y=np.asarray(first,float),np.asarray(second,float)
    if x.shape[-1]!=4 or y.shape[-1]!=4:
        raise ValueError('four affine transfer coefficients required')
    a,b,c,d=(x[...,i] for i in range(4));e,f,g,h=(y[...,i] for i in range(4))
    return np.stack((e*a, e*b+f, c+g*a, d+g*b+h), axis=-1)


def change(amount, *, unit):
    """One attention-framing impulse, origin-blind native/authored units.

    Base instantaneous face/pose/appearance cost is paid elsewhere. This adds
    ONLY interaction with existing burden, not a duplicate isolated-cut fee.
    """
    amount = np.asarray(amount, float)
    if not np.isfinite(amount).all() or np.any(amount < 0) or not np.isfinite(unit) or unit <= 0:
        raise ValueError('finite nonnegative change and positive event unit required')
    return np.stack((np.ones_like(amount), amount, unit*amount, np.zeros_like(amount)), axis=-1)


def recovery(exposure_seconds, *, tau):
    exposure, tau = np.broadcast_arrays(np.asarray(exposure_seconds, float), np.asarray(tau, float))
    if (not np.isfinite(exposure).all() or not np.isfinite(tau).all()
            or np.any(exposure < 0) or np.any(tau <= 0)):
        raise ValueError('observed nonnegative exposure and fixed positive music time required')
    z = np.zeros_like(exposure)
    return np.stack((np.exp(-exposure/tau), z, z, z), axis=-1)


def apply(transfer, debt):
    a, b, c, d = np.moveaxis(np.asarray(transfer, float), -1, 0)
    return a*debt+b, c*debt+d


class TransferTree:
    """Stable range composition without division by a tiny long-film prefix."""
    def __init__(self, steps):
        steps = np.asarray(steps, float)
        if steps.ndim != 3 or steps.shape[-1] != 4 or not np.isfinite(steps).all():
            raise ValueError('batch x chronological frame x4 transfers required')
        self.length = steps.shape[1]
        self.size = 1 << max(0, (self.length-1).bit_length())
        self.values = np.broadcast_to(IDENTITY, (len(steps), 2*self.size, 4)).copy()
        self.values[:, self.size:self.size+self.length] = steps
        for level in range(self.size.bit_length()-2, -1, -1):
            a, b = 1 << level, 1 << (level+1)
            self.values[:, a:b] = then(self.values[:, 2*a:2*b:2], self.values[:, 2*a+1:2*b:2])

    def query(self, first, final):
        if not 0 <= first <= final <= self.length:
            raise ValueError('transfer range outside observed film')
        a = np.broadcast_to(IDENTITY, (len(self.values), 4)).copy(); b = a.copy()
        first += self.size; final += self.size
        while first < final:
            if first & 1: a = then(a, self.values[:, first]); first += 1
            if final & 1: final -= 1; b = then(self.values[:, final], b)
            first //= 2; final //= 2
        return then(a, b)


def compose_steps(steps):
    values=np.asarray(steps,float)
    if not values.shape[-2]:return np.broadcast_to(IDENTITY,values.shape[:-2]+(4,)).copy()
    while values.shape[-2]>1:
        paired=then(values[...,0:-1:2,:],values[...,1::2,:])
        values=np.concatenate((paired,values[...,-1:,:]),axis=-2) if values.shape[-2]%2 else paired
    return values[...,0,:]


class SceneReadout:
    """Exact observed head samples, not interpolated dense camera/actor tracks.

    Grid/native flanks are physical observations. Within-shot changes between
    two samples at most12 output frames apart are sparse ATTENTION-reframing
    impulses, not inferred camera velocities or one person's trajectory.
    UNKNOWN neither certifies stillness nor provides recovery exposure. The
    background-only motion case is deliberately not certified by this field.
    """
    def __init__(self,cat,native,events,dense,*,tau):
        self.cat=cat;self.fps=cat.fps;self.tau=float(tau);self.final=cat.final
        self.heads=np.zeros((cat.final,8,4));self.valid=np.zeros(cat.final,bool)
        self.roi_known=np.zeros((cat.final,8),bool);self.conflicted=set()
        for n,tick in enumerate(cat.nodes):
            for side,frame in enumerate(np.clip(tick+np.array([-1,0]),0,cat.final-1)):
                frame=int(frame);heads=cat.bank['heads'][n,side]
                if frame in self.conflicted:continue
                if self.valid[frame] and not np.array_equal(self.heads[frame],heads):
                    self.valid[frame]=False;self.heads[frame]=0.;self.roi_known[frame]=False
                    self.conflicted.add(frame);continue
                self.heads[frame]=heads;self.valid[frame]=True
                self.roi_known[frame]=cat.roi_readability_valid[n,side]
        # Additional original-cut endpoints are actual observed faces, but no
        # unmeasured ROI detail is fabricated for their extra recovery exposure.
        for n,tick in enumerate(native['frames']):
            if not native.get('observed',np.ones(len(native['frames']),bool))[n]:continue
            for side,frame in enumerate((int(tick)-1,int(tick))):
                if 0<=frame<self.final and not self.valid[frame] and frame not in self.conflicted:
                    self.heads[frame]=native['heads'][n,side];self.valid[frame]=True
        self.events=np.asarray(events,int)
        actual=np.flatnonzero(self.valid);self.previous=np.full(self.final,-1,int)
        self.previous[actual[1:]]=actual[:-1]
        self.dense=np.asarray(dense);self.tree=TransferTree(self.steps(0,self.dense,hold_first=0))

    def view(self,frames,cameras):
        cameras=np.asarray(cameras,float);heads=self.heads[frames]
        centers=.5+cameras[...,None,:1]*(heads[...,:2]-cameras[...,None,1:])
        heights=heads[...,2]*cameras[...,None,0]
        half=heights[...,None]*[.45*9/16,.5]
        extent=np.maximum(0.,np.minimum(1.,centers+half)-np.maximum(0.,centers-half))
        scores=extent.prod(-1)*np.exp(-((centers[...,0]-.5)/.4)**2)
        seen=(heads[...,2]>.019)&(heads[...,3]>=.75)&self.valid[frames,None]
        scores=np.where(seen,scores,0.);index=scores.argmax(-1)
        center=np.take_along_axis(centers,index[...,None,None],axis=-2)[...,0,:]*[16/9,1.]
        height=np.take_along_axis(heights,index[...,None],axis=-1)[...,0]
        known=np.take_along_axis(np.broadcast_to(self.roi_known[frames],scores.shape),index[...,None],axis=-1)[...,0]
        return center,height,scores.max(-1)>0,known

    def steps(self,first,paths,*,hold_first,baseline=None,observation_offsets=None,content_rates=None):
        paths=np.asarray(paths,float);L=paths.shape[-2]
        offsets=np.arange(L) if observation_offsets is None else np.asarray(observation_offsets,int)
        frames=first+offsets;cameras=paths[:,offsets]
        if paths.ndim!=3 or paths.shape[-1]!=3 or first<0 or first+L>self.final:
            raise ValueError('actual batch camera paths inside source clock required')
        center,height,seen,known=self.view(frames,cameras)
        prior=self.previous[frames].copy();native=np.isin(frames,self.events)
        prior=np.where(native,frames-1,prior)
        usable=(prior>=hold_first)&(prior>=0)&(frames-prior<=12)
        # A sparse pair crossing an original cut is NOT a smooth head motion.
        usable&=native|(np.searchsorted(self.events,np.maximum(0,prior),side='right')==
            np.searchsorted(self.events,frames,side='right'))
        safe=np.maximum(0,prior);inside=(safe>=first)&(safe<first+L)
        previous_camera=paths[:,np.clip(safe-first,0,L-1)].copy()
        if baseline is not None:previous_camera[:,~inside]=np.asarray(baseline)[safe[~inside]]
        usable&=inside if baseline is None else True
        previous_center,previous_height,previous_seen,_=self.view(safe,previous_camera)
        from .played_partial_pose import face_geometry
        geometry=face_geometry(previous_center,previous_height,center,height,previous_seen&seen)[0]
        # Native UNKNOWN uses the SAME explicit finite1 convention as its
        # existing point geometry; sparse missing evidence emits no observation.
        geometry=np.where(usable&(native[None]|(previous_seen&seen)),geometry,0.)
        rates=self.cat.rates(frames,cameras) if content_rates is None else np.asarray(content_rates)[:,offsets]
        exposure=np.clip(1.-rates,0.,1.)*known*seen
        return then(change(geometry,unit=6./self.fps),recovery(exposure/self.fps,tau=self.tau))

    def summary(self,first,paths,*,hold_first,baseline=None,content_rates=None):
        frames=np.arange(first,first+paths.shape[-2])
        # Unobserved non-native frames are EXACT identity transfers, not
        # comfort/recovery measurements. Omit their redundant arithmetic.
        used=np.flatnonzero(self.valid[frames]|np.isin(frames,self.events))
        if not len(used):return np.broadcast_to(IDENTITY,(len(paths),4)).copy()
        return compose_steps(self.steps(first,paths,hold_first=hold_first,baseline=baseline,
            observation_offsets=used,content_rates=content_rates))

    def base(self,first,final):
        return self.tree.query(first,final)

    def base_hold(self,first,final):
        """Baseline range with the first physical pair clipped to THIS hold."""
        observed=np.flatnonzero(self.valid[first:final])
        if not len(observed):return self.base(first,final)
        tick=first+int(observed[0])
        if self.previous[tick]>=first:return self.base(first,final)
        corrected=self.summary(tick,self.dense[:,tick:tick+1],hold_first=first)
        return then(then(self.base(first,tick),corrected),self.base(tick+1,final))

    def receipt(self):
        return dict(observed_head_frames=int(self.valid.sum()),conflicting_grid_samples=len(self.conflicted),
            sparse_pair_maximum_output_frames=12,recovery_requires_measured_ROI=True,
            dense_head_track_or_true_camera_cause_certified=False,
            target_changes_are_attention_impulses_not_actor_velocity=True,
            background_only_pan_observed=False,tau_seconds=self.tau,
            tau_is_fixed_music_engineering_not_human_cognition=True,
            source_NN=0,pixel_reads=0,baseline_transfer_bytes=int(self.tree.values.nbytes))
