"""Source-local transfer of individual cached gallery score profiles.

The six numbers are cosine likenesses, NOT member probabilities or labels.
Weak raw queries remain weak instead of disappearing in cluster consensus.
Original hard hints are retained separately. No gallery/reference vectors,
new inference, forced member assignment or source exclusion is introduced.
"""
from dataclasses import dataclass,field
import hashlib
import numpy as np


def bank_signature(bank):
    digest=hashlib.sha256()
    for key in ('heads','vectors','native_indices'):
        a=np.asarray(bank[key])
        digest.update(key.encode()+a.dtype.str.encode()+repr(a.shape).encode()+a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MemberScoreSeeds:
    cluster_hints: np.ndarray
    vectors: np.ndarray=field(repr=False)
    scores: np.ndarray=field(repr=False)
    source_signature: str
    gallery_identity: str
    minimum_similarity: float=.45
    minimum_member_margin: float=.08

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


def from_queries(bank,requests,votes,cluster_hints,*,gallery_identity,member_names,
                 minimum_similarity=.45,minimum_member_margin=.08):
    """Recover only SFace addresses belonging to the actual ArcFace queries."""
    lookup={};conflicts=set()
    for n,side,h in np.ndindex(bank['vectors'].shape[:3]):
        key=(int(bank['native_indices'][n,side]),*map(float,bank['heads'][n,side,h]))
        value=bank['vectors'][n,side,h]
        if key in lookup and not np.array_equal(lookup[key],value):conflicts.add(key)
        else:lookup[key]=value
    rows={}
    for (native,head),vote in zip(requests,votes,strict=True):
        key=(int(native),*map(float,head));scores={} if vote is None else vote.get('member_scores',{})
        if key in conflicts or key not in lookup or set(scores)!=set(member_names):continue
        vector=np.asarray(lookup[key],float)
        if abs(np.linalg.norm(vector)-1.)>.001:continue
        profile=np.asarray([scores[name] for name in member_names],float)
        if key in rows and (not np.array_equal(rows[key][0],vector) or not np.array_equal(rows[key][1],profile)):
            raise ValueError('one physical member query has contradictory original scores')
        rows[key]=(vector,profile)
    # Canonical physical order prevents input listing order from choosing a
    # different observation when two original SFace vectors are identical.
    rows=[rows[key] for key in sorted(rows)]
    return MemberScoreSeeds(np.asarray(cluster_hints,np.int16),
        np.asarray([r[0] for r in rows]).reshape(-1,128),np.asarray([r[1] for r in rows]).reshape(-1,6),
        bank_signature(bank),gallery_identity,minimum_similarity,minimum_member_margin)


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
