"""Reuse contemporary subject windows inside the actual programme selector.

The caller binds a physical source hash and an exact output-to-native schedule.
An old grid face may retrieve a window, but cannot rename its track or body ROI.
Conflicting tracks/observations stay unavailable. No neural inference here.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .subject_action_match import SubjectClock
from .subject_window_bank import SubjectWindow
from .temporal_face_chain import _bounds, _iou

OFFSETS=np.arange(-7,7,dtype=np.int64)
OFFSETS.setflags(write=False)


class DenseSubjects:
    def __init__(self,*,source,binding,indices,windows,bank,fps):
        indices=np.asarray(indices)
        if (type(source) is not int or source<0 or not binding or not np.isfinite(fps) or fps<=0
                or indices.shape!=(len(bank['heads']),len(OFFSETS)) or indices.dtype.kind not in 'iu'
                or np.any(indices<0) or np.any(np.diff(indices,axis=1)<0)
                or not np.array_equal(indices[:,[0,6,7,13]],bank['pose_native_indices'])
                or not np.array_equal(indices[:,[6,7]],bank['native_indices'])):
            raise ValueError('dense context must bind the actual grid endpoint/flank native schedule')
        self.source=source;self.binding=binding;self.indices=indices.copy();self.indices.setflags(write=False)
        self.bank=bank;self.fps=float(fps);self.by_native={};self.cache={};self.used=set();self.conflicts=0
        for window in windows:
            if not isinstance(window,SubjectWindow) or window.source!=source or window.binding!=binding:
                raise ValueError('dense subject source/hash differs from the programme source')
            for at,native in enumerate(window.ordinals):
                if window.associated[at]:self.by_native.setdefault(int(native),[]).append((window,at))

    def sample(self,n,side,h):
        key=(int(n),int(side),int(h))
        if key in self.cache:return self.cache[key]
        self.cache[key]=None
        if not 0<=h<self.bank['heads'].shape[2]:return None
        head=self.bank['heads'][n,side,h];box=_bounds(head)
        if box is None:return None
        schedule=self.indices[n];endpoint=6 if side==0 else 7;native=int(schedule[endpoint])
        candidates=[]
        vector=self.bank['vectors'][n,side,h] if h<3 else np.zeros(128)
        for window,at in self.by_native.get(native,[]):
            if _iou(box,_bounds(window.heads[at]))<.7:continue
            # Retrieval needs BOTH physical overlap and compatible observed
            # source appearance when available; never best-cosine assignment.
            other=window.vectors[at]
            if min(np.linalg.norm(vector),np.linalg.norm(other))>.5 and vector@other<.15:continue
            start=schedule-window.ordinals[0]
            if start.min()<0 or start.max()>=len(window.ordinals):continue
            candidates.append((window,np.asarray(start,int)))
        if not candidates:return None
        candidates.sort(key=lambda v:(-int(v[0].support[v[1]].sum()),v[0].track))
        window,at=candidates[0]
        # Seed duplicates are harmless only when their actual observations
        # coincide; a convenient conflicting pose cannot win by input order.
        for other,positions in candidates[1:]:
            common=window.associated[at]&other.associated[positions]
            joints=window.support[at]&other.support[positions]&common[:,None]
            if (not all(np.array_equal(getattr(window,k)[at][common],getattr(other,k)[positions][common])
                        for k in ('heads','vectors'))
                    or not np.array_equal(window.body[at][joints],other.body[positions][joints])):
                self.conflicts+=1;return None
        cuts=np.asarray(window.native_cuts,int)
        same=np.searchsorted(cuts,schedule,side='right')==np.searchsorted(cuts,native,side='right')
        supported=window.support[at]&window.associated[at,None]&same[:,None]
        # No track can jump a missing native detection between two output
        # samples (important when a60fps source skips a native frame).
        contiguous=np.ones(len(at)-1,bool)
        for i,(a,b) in enumerate(zip(at[:-1],at[1:])):
            contiguous[i]=bool(window.associated[a:b+1].all() and same[i] and same[i+1])
        signature=hashlib.sha256(self.binding.encode()+schedule.tobytes()+OFFSETS.tobytes()).hexdigest()
        clock=SubjectClock(self.source,self.binding,OFFSETS,schedule,0,self.fps,signature)
        result=dict(window=window,clock=clock,at=at,supported=supported,
            same_shot=same,contiguous=contiguous)
        self.cache[key]=result;self.used.add(key)
        return result

    def augment(self,observation,n,side,h):
        sample=self.sample(n,side,h)
        if sample is None:return observation
        window,at=sample['window'],sample['at'];slots=np.array([0,6]) if side==0 else np.array([7,13])
        body=window.body[at[slots]].copy();support=sample['supported'][slots]
        # Contemporary NN/body association is MODEL_ONLY, never silently
        # promoted to pixel/anatomical certification by a successful track.
        from .physical_display_observations import MODEL_ONLY,UNMEASURED
        evidence=np.where(support,MODEL_ONLY,UNMEASURED).astype(np.uint8)
        endpoint=6 if side==0 else 7
        result=dict(observation,body=body,evidence=evidence,
            ownership_ambiguity=np.zeros((2,8),bool),
            physical_window_valid=bool(sample['contiguous'][slice(0,6) if side==0 else slice(7,13)].all()),
            dense_subject_context=sample,
            source_clock_context_is_sparse_model_hypothesis=False,
            contemporary_subject_window_used=True)
        # Sparse four-sample context must not survive a corrected/occluded
        # window merely because the old sensor marked it "normal".
        for name in ('source_clock_body','source_clock_evidence','source_clock_ambiguity',
                     'source_clock_endpoint_head_proxies','source_clock_observed_sample_mask'):
            result.pop(name,None)
        result['source_clock_subject_associated']=False
        return result

    def receipt(self):
        return dict(source=self.source,window_bound_actor_endpoints=len(self.used),
            conflicting_contexts=self.conflicts,new_neural_calls=0,new_pixel_rasters=0,
            actual_native_schedule_used=True,old_grid_or_window_cache_rewritten=False,
            ownership_certified=False)


def load_reused(reference,*,sources,source_hashes,banks,nodes,warps,clocks,master_start,fps,
                authorized_root,check=lambda:None):
    """Load only window paths actually retained by one bound observation pass.

    A missing source window is permitted; a changed source/clock is not. This
    never scans all experiment folders, runs a sensor, or rewrites old caches.
    """
    from .grid_perception import file_identity
    from .subject_window_bank import load
    check();root=Path(authorized_root).resolve(strict=True);reference=Path(reference)
    if not reference.resolve(strict=True).is_relative_to(root):raise ValueError('subject evidence outside authorized project')
    record=json.loads(reference.read_text());plan=record['render_plan']
    if (list(map(str,sources))!=plan['sources'] or list(source_hashes)!=record['source_sha256']
            or [list(file_identity(p)) for p in sources]!=record['source_file_identities']):
        raise ValueError('subject evidence physical source domain changed')
    if float(plan['fps'])!=fps or float(plan['master_audio_start'])!=master_start:
        raise ValueError('subject evidence music clock changed')
    paths=record['subject_moment_refinement_receipt']['window_receipts'];groups=[[] for _ in sources];seen=set()
    for receipt in paths:
        check()
        if 'path' not in receipt:continue  # an explicitly unsupported seed
        path=Path(receipt['path']);s=receipt['source']
        if type(s) is not int or not 0<=s<len(sources):raise ValueError('subject evidence source index changed')
        if path in seen:continue
        seen.add(path)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('subject window outside authorized project or symlink')
        groups[s].append(load(path,binding=source_hashes[s]))
    result=[]
    for s,(bank,warp,clock) in enumerate(zip(banks,warps,clocks,strict=True)):
        check()
        native=np.array([[round(warp.master_to_source(master_start+
            max(0,min(int(nodes[-1])-1,int(tick)+int(offset)))/fps)*clock['fps'])
            for offset in OFFSETS] for tick in nodes],np.int64)
        if native.min()<0 or native.max()>=clock['frame_count']:raise ValueError('dense source schedule escapes native domain')
        result.append(DenseSubjects(source=s,binding=source_hashes[s],indices=native,
            windows=groups[s],bank=bank,fps=fps))
    return result
