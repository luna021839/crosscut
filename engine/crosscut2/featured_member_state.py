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
        from .source_member_scores import MemberScoreSeeds
        individual=values if isinstance(values,MemberScoreSeeds) else None
        if individual is not None:
            individual.verify(bank);values=individual.cluster_hints
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
        augmented=dict(ports,member_hints=result)
        if individual is not None:
            for target,query in ((bank,bank['vectors']),(augmented,ports.get('vectors'))):
                raw,support,direct=individual.fields(target['heads'],query,
                    native_indices=target.get('native_indices'))
                raw.setflags(write=False);support.setflags(write=False);direct.setflags(write=False)
                target.update(member_score_profiles=raw,member_score_support=support,
                    member_score_directly_observed=direct,
                    member_score_gallery_identity=individual.gallery_identity)
        native.append(augmented)
    return faces,native


def consensus(banks,members,*,return_evidence=False,source_cohort=None):
    shape=banks[0]['heads'].shape[:2]
    if len(members)!=len(banks):raise ValueError('one member field per physical source required')
    active=tuple(range(len(banks))) if source_cohort is None else tuple(source_cohort)
    if (not active or any(type(s) is not int or not 0<=s<len(banks) for s in active)
            or len(set(active))!=len(active)):
        raise ValueError('distinct physical featured-vote source indices required')
    families=[bank.get('consensus_family',('unverified_source',s)) for s,bank in enumerate(banks)]
    groups={key:i for i,key in enumerate(dict.fromkeys(families[s] for s in active))}
    votes=np.zeros((len(groups),)+shape+(6,));possible=np.zeros((len(groups),)+shape)
    face_votes=[];profile_weights=[];profile_sources=[]
    for s,(bank,names) in enumerate(zip(banks,members,strict=True)):
        names=np.asarray(names);value=np.zeros(names.shape+(6,))
        profile_weight=np.zeros(names.shape)
        for m in range(6):value[...,m]=names==m
        if 'member_score_profiles' in bank and bank.get('member_score_gallery_identity',''):
            scores=np.asarray(bank['member_score_profiles'],float);support=np.asarray(bank['member_score_support'],float)
            if (scores.shape!=value.shape or support.shape!=names.shape or not np.isfinite(scores).all()
                    or not np.isfinite(support).all() or np.any((support<0)|(support>1))):
                raise ValueError('source-bound six-member score fields and separate support required')
            # Positive, centred gallery-score votes, not calibrated identity
            # probabilities. Flat scores contribute nothing. Current query
            # support blends out the old propagated one-hot hint ONCE.
            positive=np.maximum(0.,scores-scores.mean(-1,keepdims=True));total=positive.sum(-1)
            weight=np.where(total>1e-12,support,0.)
            profile_weight=weight
            value=(1.-weight[...,None])*value+weight[...,None]*positive/np.maximum(1e-12,total[...,None])
            profile_sources.append(s)
        face_votes.append(value)
        profile_weights.append(profile_weight)
    current_possible=np.zeros((len(groups),)+shape)
    for s in active:
        bank,names,family=banks[s],members[s],families[s]
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
        selected=np.take_along_axis(face_votes[s],np.broadcast_to(np.maximum(0,index)[...,None,None],
            shape+(1,6)),axis=2)[...,0,:]
        votes[g]=np.maximum(votes[g],selected*weight[...,None])
        current=np.take_along_axis(profile_weights[s],np.maximum(0,index)[...,None],axis=2)[...,0]
        current_possible[g]=np.maximum(current_possible[g],current*weight)
    # One family cannot gain extra authority by offering several derivatives.
    mass=votes.sum(-1);votes*=np.minimum(1.,possible/np.maximum(1e-12,mass))[...,None]
    known=votes.sum((0,-1));total=possible.sum(0);summed=votes.sum(0)
    posterior=(summed+np.maximum(0.,total-known)[...,None]/6.)/np.maximum(1e-12,total[...,None])
    posterior=np.where((total>0)[...,None],posterior,1./6.)
    entropy=-(posterior*np.log(np.maximum(1e-12,posterior))).sum(-1)/np.log(6.)
    supporting=(votes.sum(-1)>0).sum(0)
    confidence=np.clip(known/np.maximum(1e-12,total)*(1.-entropy)*total/max(1,len(groups)),0.,1.)
    confidence=np.where(supporting>=2,confidence,0.)
    current_supported=(current_possible>0).sum(0)>=2
    factors=[]
    maximum=np.maximum(1./6.,posterior.max(-1))
    for value in face_votes:
        mass=value.sum(-1)
        probability=np.sum(value*posterior[...,None,:],axis=-1)+(1.-mass)/6.
        relative=np.clip(probability/maximum[...,None],0.,1.)
        legacy=1.-confidence[...,None]+confidence[...,None]*relative
        # Current query support, missing mass, family weights and disagreement
        # already entered this posterior. Multiplying it by entropy/confidence
        # AGAIN nearly erases the actual featured-person preference. Keep the
        # legacy path where fewer than two independent current profiles exist.
        factors.append(np.where(current_supported[...,None],relative,legacy))
    receipt=dict(source_families=len(groups),voting_source_indices=list(active),physical_source_domain_count=len(banks),
        framing_consensus_nodes=int((confidence>0).sum()),
        mean_confidence=float(confidence.mean()),maximum_confidence=float(confidence.max()),
        current_gallery_score_fields_used_before_camera=bool(profile_sources),
        gallery_profile_sources=profile_sources,score_vote_coefficients_are_not_calibrated_identity_probabilities=True,
        current_profile_supported_flanks=int(current_supported.sum()),
        current_profile_uncertainty_not_applied_twice_to_role_factors=True,
        inferred_featured_not_verified_active_singer=True,named_identity_certified=False,
        unknown_is_not_forced_to_a_member=True,source_exclusion=False)
    if return_evidence:
        return factors,receipt,dict(posterior=posterior,confidence=confidence,family_votes=votes)
    return factors,receipt
