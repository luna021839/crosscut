"""Sparse contemporary action context; no new frames, fit, or ownership proof.

The four samples are [-7,-1,0,+6] on EACH source's music clock. Compare both
sources over the same intervals, not A's pre-cut velocity with B's post-cut
velocity. Endpoint association is a source-local hypothesis, not a dense track.
"""
from itertools import permutations
import numpy as np


def endpoint_correspondence(heads, vectors):
    """One-to-one BEFORE/AFTER hypotheses, with ambiguous/missing left unknown."""
    result=np.full((len(heads),3),-1,int)
    for n in range(len(heads)):
        cost=np.full((3,3),10.)
        for a in range(3):
            ha=heads[n,0,a];va=vectors[n,0,a]
            for b in range(3):
                hb=heads[n,1,b];vb=vectors[n,1,b]
                if min(ha[2],hb[2])<=.019 or min(ha[3],hb[3])<.75:continue
                if min(np.linalg.norm(va),np.linalg.norm(vb))<.5:continue
                distance=np.linalg.norm((ha[:2]-hb[:2])*[16/9,1.])/max(ha[2],hb[2])
                scale=abs(np.log(ha[2]/hb[2]));appearance=float(va@vb)
                if distance>1.5 or scale>np.log(2.) or appearance<.15:continue
                cost[a,b]=distance+.5*scale+.75*(1.-appearance)
        assignment=min(permutations(range(3)),key=lambda order:sum(cost[a,b] for a,b in enumerate(order)))
        for a,b in enumerate(assignment):
            if cost[a,b]>=1.8:continue
            if (np.delete(cost[a],b).min()-cost[a,b]<.12
                    or np.delete(cost[:,b],a).min()-cost[a,b]<.12):continue
            result[n,a]=b
    return result


def describe(observations, pairs, *, fps):
    count=len(observations)
    body=np.asarray([o.get('source_clock_body',np.zeros((4,8,3))) for o in observations],float)
    kind=np.asarray([o.get('source_clock_evidence',np.zeros((4,8),np.uint8)) for o in observations],np.uint8)
    ambiguous=np.asarray([o.get('source_clock_ambiguity',np.zeros((4,8),bool)) for o in observations],bool)
    available=np.asarray([o.get('source_clock_subject_associated',False) for o in observations],bool)
    heads=np.asarray([o.get('source_clock_endpoint_head_proxies',np.zeros((4,4))) for o in observations],float)
    if body.shape!=(count,4,8,3) or kind.shape!=(count,4,8) or not np.isfinite(body).all():
        raise ValueError('four original finite source-clock body hypotheses required')
    xy=body[...,:2]*[16/9,1.]
    valid=(body[...,2]>=.3)&(body[...,:2]>=0).all(-1)&(body[...,:2]<=1).all(-1)
    bone=xy[:,:,pairs[:,1]]-xy[:,:,pairs[:,0]]
    length=np.linalg.norm(bone,axis=-1)
    good=valid[...,pairs].all(-1)&(length>.005)
    from .physical_display_observations import UNMEASURED,MODEL_ONLY,PIXEL_SUPPORTED,CONTRADICTED
    kinds=kind[...,pairs]
    evidence=np.where(~good|(kinds==UNMEASURED).any(-1),UNMEASURED,
        np.where((kinds==CONTRADICTED).any(-1),CONTRADICTED,
            np.where((kinds==PIXEL_SUPPORTED).all(-1),PIXEL_SUPPORTED,MODEL_ONLY)))
    evidence=np.where(available[:,None,None],evidence,UNMEASURED)
    return dict(unit=bone/np.maximum(1e-9,length[...,None]),kind=evidence,
                relative=bone/np.maximum(.02,heads[:,:,None,2,None]),
                ambiguous=ambiguous[...,pairs].any(-1),available=available,fps=fps)


def paired(left,right,*,channels,tentative):
    """Two contemporary phase samples and two contemporary motion intervals."""
    from .physical_display_observations import UNMEASURED,MODEL_ONLY,PIXEL_SUPPORTED,CONTRADICTED
    def combine(a,b):
        return np.where((a==UNMEASURED)|(b==UNMEASURED),UNMEASURED,
            np.where((a==CONTRADICTED)|(b==CONTRADICTED),CONTRADICTED,
                np.where((a==PIXEL_SUPPORTED)&(b==PIXEL_SUPPORTED),PIXEL_SUPPORTED,MODEL_ONLY)))
    def across_times(kind,indices):
        result=kind[:,indices[0]]
        for at in indices[1:]:result=combine(result,kind[:,at])
        return result
    phases=[];raw_terms=[];term_known=[]
    for at in (1,2):
        angular=np.linalg.norm(left['unit'][:,None,at]-right['unit'][None,:,at],axis=-1)/2.
        delta=np.linalg.norm(left['relative'][:,None,at]-right['relative'][None,:,at],axis=-1)
        raw=.5*(angular+delta/(1.+delta))
        raw_terms.append(raw)
        kind=combine(left['kind'][:,None,at],right['kind'][None,:,at])
        term_known.append((kind==MODEL_ONLY)|(kind==PIXEL_SUPPORTED))
        ambiguity=left['ambiguous'][:,None,at]|right['ambiguous'][None,:,at]
        phases.append(tentative(raw,kind,ambiguity))
    motions=[]
    for early,late in ((0,1),(2,3)):
        va=left['unit'][:,late]-left['unit'][:,early]
        vb=right['unit'][:,late]-right['unit'][:,early]
        raw=np.linalg.norm(va[:,None]-vb[None,:],axis=-1)/4.
        raw_terms.append(np.minimum(1.,raw))
        kind=combine(across_times(left['kind'],(early,late))[:,None],
                     across_times(right['kind'],(early,late))[None,:])
        term_known.append((kind==MODEL_ONLY)|(kind==PIXEL_SUPPORTED))
        ambiguity=left['ambiguous'][:,(early,late)].any(1)[:,None]|right['ambiguous'][:,(early,late)].any(1)[None,:]
        motions.append(tentative(np.minimum(1.,raw),kind,ambiguity))
    terms=(phases[0]+phases[1]+motions[0]+motions[1])/4.
    rates=np.stack([terms[...,list(c)].mean(-1) for c in channels],axis=-1)
    # A fresh incoming close-up may have no corresponding PRE-cut camera.
    # Keep a measured common AFTER interval (or vice versa), not a fictitious
    # track across the source cut and not a veto of the entire candidate.
    seam_kind=combine(left['kind'][:,None,1],right['kind'][None,:,2])
    seam_known=(seam_kind==MODEL_ONLY)|(seam_kind==PIXEL_SUPPORTED)
    included=np.zeros_like(seam_known)
    for bones in ((0,1,2),(3,4,5)):
        complete=seam_known[...,list(bones)].all(-1)&(
            term_known[2][...,list(bones)].all(-1)|term_known[3][...,list(bones)].all(-1))
        included[...,list(bones)]=complete[...,None]
    available=left['available'][:,None]&right['available'][None,:]&included.any(-1)
    worst=np.maximum.reduce([np.where(known,raw,0.) for raw,known in zip(raw_terms,term_known)])
    configuration=np.where(available,1.-np.where(included,worst,0.).max(-1),0.)
    movement=np.zeros_like(configuration)
    for window,(early,late) in enumerate(((0,1),(2,3))):
        va=np.linalg.norm(left['unit'][:,late]-left['unit'][:,early],axis=-1)/(6./left['fps'])
        vb=np.linalg.norm(right['unit'][:,late]-right['unit'][:,early],axis=-1)/(6./right['fps'])
        active=np.minimum(va[:,None]/(1.+va[:,None]),vb[None,:]/(1.+vb[None,:]))
        movement=np.maximum(movement,np.where(included&term_known[window+2],active,0.).max(-1))
    ka,kb=across_times(left['kind'],(0,1,2,3)),across_times(right['kind'],(0,1,2,3))
    pixel=(ka[:,None]==PIXEL_SUPPORTED)&(kb[None,:]==PIXEL_SUPPORTED)
    ambiguous=left['ambiguous'].any(1)[:,None]|right['ambiguous'].any(1)[None,:]
    fully_pixel=(~included|(pixel&~ambiguous)).all(-1)
    # Evidence caps are declared engineering limits, not probabilities of
    # anatomical ownership or calibrated human preference.
    cap=np.where(fully_pixel,1.,.5)
    # These are bounded engineering cues, not independent probabilities.
    # Multiplication repeatedly discounts the same model uncertainty and
    # makes an otherwise useful moving connection almost worthless. The
    # weakest observed cue limits credit; no strong limb can cancel it.
    # Apply the evidence shrinkage ONCE, after the action bottleneck. Merely
    # clipping at .5 would hide a measured worsening from .9 to .6.
    basis=np.where(available,np.minimum(configuration,movement)*cap,0.)
    return rates,available,basis
