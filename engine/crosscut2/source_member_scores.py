"""Source-local transfer of individual cached gallery score profiles.

The six numbers are cosine likenesses, NOT member probabilities or labels.
Weak raw queries remain weak instead of disappearing in cluster consensus.
Original hard hints are retained separately. No gallery/reference vectors,
new inference, forced member assignment or source exclusion is introduced.
"""
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
import numpy as np


def bank_signature(bank):
    digest=hashlib.sha256()
    for key in ('heads','vectors','native_indices'):
        a=np.asarray(bank[key])
        digest.update(key.encode()+a.dtype.str.encode()+repr(a.shape).encode()+a.tobytes())
    return digest.hexdigest()


def programme_endpoint_requests(nodes,banks,seams,*,maximum_centres=128,maximum_requests=2048):
    """One native focal per available view at automatic draft boundaries.

    The draft chooses WHERE to acquire, not the final source/camera/cut.
    Every alternative source receives the same two-flank opportunity. Keep
    original face addresses and deduplicate them before any model is called.
    """
    from .physical_display_observations import focal_metadata
    nodes=np.asarray(nodes)
    if (nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or len(nodes)<2 or np.any(np.diff(nodes)<=0)
            or type(maximum_centres) is not int or not 1<=maximum_centres<=128
            or type(maximum_requests) is not int or not 0<=maximum_requests<=2048):
        raise ValueError('bounded exact decision grid and current-face request allowance required')
    indices=set()
    for seam in seams:
        tick=seam['output_frame'];n=int(np.searchsorted(nodes,tick))
        if type(tick) is not int or not 0<n<len(nodes)-1 or nodes[n]!=tick:
            raise ValueError('draft cut outside the actual source decision grid')
        indices.add(n)
    if len(indices)>maximum_centres:raise ValueError('complete draft endpoint centre allowance exceeded')
    result=[];count=0
    for bank in banks:
        heads=np.asarray(bank['heads']);native=np.asarray(bank['native_indices'])
        if (heads.shape[:2]!=(len(nodes),2) or native.shape!=(len(nodes),2)
                or native.dtype.kind not in 'iu' or np.any(native<0)):
            raise ValueError('draft requests require source-bound native frame addresses')
        _,focal,_,_=focal_metadata(heads,np.tile([1.,.5,.5],heads.shape[:2]+(1,)))
        physical={}
        for n in sorted(indices):
            for side in (0,1):
                h=int(focal[n,side])
                if h<0:continue
                head=heads[n,side,h]
                if head[2]<.055 or head[3]<.75:continue
                key=(int(native[n,side]),*map(float,head))
                physical.setdefault(key,(key[0],head.copy()))
        values=[physical[key] for key in sorted(physical)];result.append(values);count+=len(values)
    if count>maximum_requests:
        raise ValueError(f'Complete current-face endpoints need {count} queries; allowance is {maximum_requests}')
    return result,dict(policy='all_source_native_focals_at_automatic_provisional_endpoints_v1',
        provisional_centres=len(indices),physical_face_requests=count,per_source_requests=list(map(len,result)),
        maximum_requests=maximum_requests,final_cuts_or_sources_frozen=False,
        manually_chosen_bad_timestamps_used=False,small_or_missing_faces_exclude_no_source=True)


def load_programme_endpoint_requests(path,*,nodes,banks,sources,source_hashes,master_start,fps,authorized_root,
                                    maximum_requests=2048):
    """Validate a source-bound automatic preliminary plan before acquisition."""
    from .grid_perception import file_identity
    path=Path(path);root=Path(authorized_root).resolve(strict=True)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('provisional member programme outside authorized source project')
    raw=path.read_bytes();record=json.loads(raw);plan=record.get('render_plan',{})
    if (record.get('source_sha256')!=list(source_hashes)
            or record.get('source_file_identities')!=[list(file_identity(p)) for p in sources]
            or plan.get('sources')!=list(map(str,sources)) or plan.get('fps')!=fps
            or plan.get('master_audio_start')!=master_start):
        raise ValueError('provisional member programme physical source/clock domain differs')
    requests,receipt=programme_endpoint_requests(nodes,banks,record['seams'],maximum_requests=maximum_requests)
    return requests,dict(receipt,provisional_plan_sha256=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class MemberScoreSeeds:
    cluster_hints: np.ndarray
    vectors: np.ndarray=field(repr=False)
    scores: np.ndarray=field(repr=False)
    source_signature: str
    gallery_identity: str
    minimum_similarity: float=.45
    minimum_member_margin: float=.08
    direct_queries: tuple=field(default=(),repr=False)

    def __post_init__(self):
        x=np.asarray(self.vectors);q=np.asarray(self.scores);h=np.asarray(self.cluster_hints)
        if (x.ndim!=2 or x.shape[1]!=128 or q.shape!=(len(x),6)
                or not np.isfinite(x).all() or not np.isfinite(q).all()
                or np.any(np.abs(np.linalg.norm(x,axis=1)-1.)>.001)
                or np.any((q < -1.) | (q > 1.)) or h.ndim!=1 or h.dtype.kind not in 'iu'
                or np.any((h < -1) | (h >= 6)) or not self.gallery_identity
                or len(self.source_signature)!=64
                or not 0 < self.minimum_similarity <= 1 or not 0 < self.minimum_member_margin <= 1):
            raise ValueError('source-bound normalized queries, raw six-member scores and original hints required')
        for name,value in (('vectors',x),('scores',q),('cluster_hints',h)):
            value=value.copy();value.setflags(write=False);object.__setattr__(self,name,value)
        queries=[];seen=set()
        for native,head,scores in self.direct_queries:
            head=np.asarray(head,float);scores=np.asarray(scores,float)
            if (type(native) is not int or native<0 or head.shape!=(4,) or scores.shape!=(6,)
                    or not np.isfinite(head).all() or not np.isfinite(scores).all()
                    or np.any((scores < -1.) | (scores > 1.))):
                raise ValueError('exact native face addresses and actual gallery score profiles required')
            key=(native,*map(float,head))
            if key in seen:raise ValueError('one direct gallery query per physical face required')
            seen.add(key);queries.append((native,tuple(map(float,head)),tuple(map(float,scores))))
        object.__setattr__(self,'direct_queries',tuple(queries))

    def verify(self,bank):
        if bank_signature(bank)!=self.source_signature:
            raise ValueError('individual member scores changed their physical source observations')

    def project(self,vectors):
        """Nearest original query profile with explicit bounded support.

        Similarity/margin units reuse the existing source-native SFace .55/.03
        retrieval conventions. Gallery units remain .45/.08. These are soft
        influence measures, not calibrated confidence or new label thresholds.
        A tie across different best-member profiles receives zero influence.
        """
        x=np.asarray(vectors,float)
        if x.shape[-1:]!=(128,) or not np.isfinite(x).all():
            raise ValueError('actual finite source-native SFace occurrences required')
        shape=x.shape[:-1];result=np.zeros(shape+(6,));strength=np.zeros(shape)
        if not len(self.vectors):return result,strength
        similarity=np.einsum('...d,kd->...k',x,self.vectors,optimize=True)
        at=similarity.argmax(-1);best=np.take_along_axis(similarity,at[...,None],axis=-1)[...,0]
        result=self.scores[at].copy()
        classes=self.scores.argmax(-1);chosen=classes[at]
        other=np.where(classes!=chosen[...,None],similarity,-1.).max(-1)
        ordered=np.sort(result,axis=-1)
        retrieval=np.clip((best-.55)/.45,0.,1.)
        separation=np.clip((best-other)/.03,0.,1.)
        detail=np.clip(ordered[...,-1]/self.minimum_similarity,0.,1.)
        discrimination=np.clip((ordered[...,-1]-ordered[...,-2])/self.minimum_member_margin,0.,1.)
        strength=np.minimum.reduce((retrieval,separation,detail,discrimination))
        strength=np.where(np.abs(np.linalg.norm(x,axis=-1)-1.)<=.001,strength,0.)
        return np.where((strength>0)[...,None],result,0.),strength

    def fields(self,heads,vectors,*,native_indices=None):
        """Prefer a current-face query over a descriptor-neighbour transfer.

        Nearest SFace retrieval remains a soft fallback for other frames.
        Its ambiguity must not erase an ArcFace observation of this EXACT
        source/native rectangle. A matching vector alone is not its address.
        Direct score support retains the original gallery likeness/margin
        conventions; this neither creates a label nor changes admission.
        """
        heads=np.asarray(heads);vectors=None if vectors is None else np.asarray(vectors)
        shape=heads.shape[:-1]
        if heads.ndim<2 or heads.shape[-1]!=4 or not np.isfinite(heads).all():
            raise ValueError('original finite face rectangles required')
        raw=np.zeros(shape+(6,));support=np.zeros(shape);direct=np.zeros(shape,bool)
        if vectors is not None:
            if vectors.shape[:-2]!=shape[:-1] or vectors.shape[-2]>shape[-1]:
                raise ValueError('source-bound descriptor/face occurrence shapes differ')
            profile,weight=self.project(vectors);width=profile.shape[-2]
            raw[...,:width,:]=profile;support[...,:width]=weight
        if native_indices is not None:
            native=np.asarray(native_indices)
            if native.shape!=shape[:-1] or native.dtype.kind not in 'iu' or np.any(native<0):
                raise ValueError('exact native ordinal per face frame required')
            lookup={(n,*head):np.asarray(scores) for n,head,scores in self.direct_queries}
            for address in np.ndindex(native.shape):
                for h,head in enumerate(heads[address]):
                    scores=lookup.get((int(native[address]),*map(float,head)))
                    if scores is None:continue
                    ordered=np.sort(scores)
                    weight=float(min(np.clip(ordered[-1]/self.minimum_similarity,0.,1.),
                        np.clip((ordered[-1]-ordered[-2])/self.minimum_member_margin,0.,1.)))
                    at=address+(h,);raw[at]=scores;support[at]=weight;direct[at]=True
        return raw,support,direct


def from_queries(bank,requests,votes,cluster_hints,*,gallery_identity,member_names,
                 minimum_similarity=.45,minimum_member_margin=.08):
    """Recover source/native queries; SFace transfer is optional, not proof."""
    lookup={};conflicts=set()
    for n,side,h in np.ndindex(bank['heads'].shape[:3]):
        key=(int(bank['native_indices'][n,side]),*map(float,bank['heads'][n,side,h]))
        value=(bank['vectors'][n,side,h] if h<bank['vectors'].shape[-2] else np.zeros(128))
        if key in lookup and not np.array_equal(lookup[key],value):conflicts.add(key)
        else:lookup[key]=value
    rows={};direct={}
    for (native,head),vote in zip(requests,votes,strict=True):
        key=(int(native),*map(float,head));scores={} if vote is None else vote.get('member_scores',{})
        if key in conflicts or key not in lookup or set(scores)!=set(member_names):continue
        profile=np.asarray([scores[name] for name in member_names],float)
        if key in direct and not np.array_equal(direct[key],profile):
            raise ValueError('one physical member query has contradictory original scores')
        direct[key]=profile
        vector=np.asarray(lookup[key],float)
        if abs(np.linalg.norm(vector)-1.)>.001:continue
        if key in rows and (not np.array_equal(rows[key][0],vector) or not np.array_equal(rows[key][1],profile)):
            raise ValueError('one physical member query has contradictory original scores')
        rows[key]=(vector,profile)
    # Canonical physical order prevents input listing order from choosing a
    # different observation when two original SFace vectors are identical.
    direct=tuple((key[0],tuple(key[1:]),tuple(map(float,direct[key]))) for key in sorted(direct))
    rows=[rows[key] for key in sorted(rows)]
    return MemberScoreSeeds(np.asarray(cluster_hints,np.int16),
        np.asarray([r[0] for r in rows]).reshape(-1,128),np.asarray([r[1] for r in rows]).reshape(-1,6),
        bank_signature(bank),gallery_identity,minimum_similarity,minimum_member_margin,direct)


def pair_risk(first,second,weight_a,weight_b,*,compatible=True):
    """Compare score-profile shape; a flat/unbound profile gives no credit."""
    a,b=np.asarray(first,float),np.asarray(second,float)
    wa,wb=np.asarray(weight_a,float),np.asarray(weight_b,float)
    if (a.shape[-1:]!=(6,) or b.shape[-1:]!=(6,) or not all(np.isfinite(v).all() for v in (a,b,wa,wb))
            or np.any((wa<0)|(wa>1)) or np.any((wb<0)|(wb>1))):
        raise ValueError('six finite score cues and bounded source-local support required')
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
    cosine=np.sum(a*b,axis=-1)/np.maximum(1e-12,na*nb)
    influence=np.where((na>1e-12)&(nb>1e-12)&np.asarray(compatible,bool),np.minimum(wa,wb),0.)
    return (1.-np.clip(cosine,-1.,1.))/2.,influence
