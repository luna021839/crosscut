"""Soft featured-member evidence from original, independent source views.

Recurrent photo-derived hints are not certified identity or active-vocal labels.
Duplicate performance families get one vote budget, not one vote per crop.
No reference photographs, embeddings, or named hard source exclusions persist.
"""
import numpy as np
from .fixed_holds import local_actor_hypotheses
from .physical_display_observations import focal_metadata


def bind_hints(banks,hints,native_ports):
    if len(hints)!=len(banks):raise ValueError('one recurrent-hint array per actual source required')
    faces=[];native=[]
    for bank,values,ports in zip(banks,hints,native_ports):
        labels,count=local_actor_hypotheses(bank);values=np.asarray(values)
        if (values.shape!=(count,) or values.dtype.kind not in 'iu'
                or np.any((values<-1)|(values>=6))):raise ValueError('source-local hint/cluster domain changed')
        members=np.full(bank['heads'].shape[:3],-1,int)
        n,side,h=np.nonzero(labels>=0);members[n,side,h]=values[labels[n,side,h]]
        faces.append(members)
        prototypes=[];names=[]
        for actor in range(count):
            if values[actor]<0:continue
            vectors=bank['vectors'][labels==actor]
            if not len(vectors):continue
            vector=vectors.mean(0);norm=np.linalg.norm(vector)
            if norm>.5:prototypes.append(vector/norm);names.append(int(values[actor]))
        result=np.full(ports['heads'].shape[:3],-1,int)
        query=ports.get('vectors')
        if query is not None and prototypes:
            similarity=np.einsum('...d,ad->...a',query,np.asarray(prototypes),optimize=False)
            by_member=np.full(similarity.shape[:-1]+(6,),-1.)
            for m in range(6):
                at=np.flatnonzero(np.asarray(names)==m)
                if len(at):by_member[...,m]=similarity[...,at].max(-1)
            order=np.argsort(-by_member,axis=-1);best=np.take_along_axis(by_member,order[...,:1],axis=-1)[...,0]
            second=np.take_along_axis(by_member,order[...,1:2],axis=-1)[...,0]
            known=(np.linalg.norm(query,axis=-1)>.5)&(best>=.55)&(best-second>=.03)
            result[...,:3]=np.where(known,order[...,0],-1)
        native.append(dict(ports,member_hints=result))
    return faces,native


def consensus(banks,members,*,return_evidence=False):
    shape=banks[0]['heads'].shape[:2]
    families=[bank.get('consensus_family',('unverified_source',s)) for s,bank in enumerate(banks)]
    groups={key:i for i,key in enumerate(dict.fromkeys(families))}
    votes=np.zeros((len(groups),)+shape+(6,));possible=np.zeros((len(groups),)+shape)
    for bank,names,family in zip(banks,members,families):
        heads,index,ambiguity,_=focal_metadata(bank['heads'],np.tile([1.,.5,.5],shape+(1,)))
        quality=np.full(shape,.5)
        if 'native_readability' in bank:
            for n,side in zip(*np.nonzero(index>=0)):
                h=int(index[n,side])
                if bank['native_readability_valid'][n,side,h]:
                    contrast,edge,clipped,height=bank['native_readability'][n,side,h]
                    quality[n,side]=min(1.,height/120.)*min(1.,edge/.5)*min(1.,contrast/.075)*(1.-clipped)
        weight=np.sqrt(quality)*np.minimum(1.,heads[...,2]/.22)*np.maximum(0.,1.-ambiguity)
        weight=np.where(index>=0,weight,0.);g=groups[family]
        possible[g]=np.maximum(possible[g],weight)
        selected=np.take_along_axis(names,np.maximum(0,index)[...,None],axis=2)[...,0]
        for member in range(6):votes[g,...,member]=np.maximum(votes[g,...,member],np.where(selected==member,weight,0.))
    # One family cannot gain extra authority by offering several derivatives.
    mass=votes.sum(-1);votes*=np.minimum(1.,possible/np.maximum(1e-12,mass))[...,None]
    known=votes.sum((0,-1));total=possible.sum(0);summed=votes.sum(0)
    posterior=(summed+np.maximum(0.,total-known)[...,None]/6.)/np.maximum(1e-12,total[...,None])
    posterior=np.where((total>0)[...,None],posterior,1./6.)
    entropy=-(posterior*np.log(np.maximum(1e-12,posterior))).sum(-1)/np.log(6.)
    supporting=(votes.sum(-1)>0).sum(0)
    confidence=np.clip(known/np.maximum(1e-12,total)*(1.-entropy)*total/max(1,len(groups)),0.,1.)
    confidence=np.where(supporting>=2,confidence,0.)
    factors=[]
    maximum=np.maximum(1./6.,posterior.max(-1))
    for names in members:
        selected=np.take_along_axis(posterior,np.maximum(0,names),axis=2)
        probability=np.where(names>=0,selected,1./6.)
        factors.append(1.-confidence[...,None]+confidence[...,None]*probability/maximum[...,None])
    receipt=dict(source_families=len(groups),framing_consensus_nodes=int((confidence>0).sum()),
        mean_confidence=float(confidence.mean()),maximum_confidence=float(confidence.max()),
        inferred_featured_not_verified_active_singer=True,named_identity_certified=False,
        unknown_is_not_forced_to_a_member=True,source_exclusion=False)
    if return_evidence:
        return factors,receipt,dict(posterior=posterior,confidence=confidence,family_votes=votes)
    return factors,receipt
