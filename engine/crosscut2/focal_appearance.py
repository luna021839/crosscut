"""Soft appearance risk of the bound displayed face, never identity proof."""
import numpy as np


def discrepancy(first, second,*,member_a=None,member_b=None,
                profile_a=None,profile_b=None,support_a=None,support_b=None,profile_compatible=True,details=False):
    """Broadcastable anonymous appearance risk; unknown is finite .5.

    This is a retrieval cue, not an identity verdict or admission threshold.
    The same function is used at authored and retained broadcaster boundaries.
    """
    a,b=np.asarray(first,float),np.asarray(second,float)
    if (a.shape[-1:]!=(128,) or b.shape[-1:]!=(128,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('original finite128D SFace vectors required')
    measured=(np.abs(np.linalg.norm(a,axis=-1)-1.)<=.001)&(np.abs(np.linalg.norm(b,axis=-1)-1.)<=.001)
    cosine=np.clip(np.sum(a*b,axis=-1),-1.,1.)
    risk=np.where(measured,(1.-cosine)/2.,.5)
    anonymous_risk=risk.copy();anonymous_measured=measured.copy()
    known=np.zeros_like(risk,bool)
    if member_a is not None and member_b is not None:
        ma,mb=np.asarray(member_a),np.asarray(member_b)
        if ma.dtype.kind not in 'iu' or mb.dtype.kind not in 'iu' or np.any((ma<-1)|(ma>=6)) or np.any((mb<-1)|(mb>=6)):
            raise ValueError('global recurrent IVE hint ids, not source-local cluster ids required')
        known=(ma>=0)&(mb>=0)
        # These are photo-derived hints, not certified actor labels. Blend
        # their agreement with anonymous evidence; never hard-veto a source.
        risk=np.where(known,.5*risk+.5*(ma!=mb),risk);measured=measured|known
    influence=np.zeros_like(risk)
    provided=(profile_a is not None,profile_b is not None,support_a is not None,support_b is not None)
    if any(provided):
        if not all(provided):raise ValueError('both bound member score profiles and support values required')
        from .source_member_scores import pair_risk
        score_risk,influence=pair_risk(profile_a,profile_b,support_a,support_b,compatible=profile_compatible)
        # A propagated cluster hint is not an independent current-face query.
        # When compatible per-face profiles exist, use their actual support
        # ONCE instead of discarding it and imposing a blanket50% hint vote.
        # A missing/weak current profile cannot halve a real disagreement or
        # turn a missing descriptor into a measured favourable appearance.
        weight=.5*influence
        compatible=np.asarray(profile_compatible,bool)
        risk=np.where(compatible,(1.-weight)*anonymous_risk+weight*score_risk,risk)
        measured=np.where(compatible,anonymous_measured|(influence>0),measured)
    return (risk,measured,influence) if details else (risk,measured)


def cosine(left, right):
    """UNKNOWN is NaN for existing report normalization, never a perfect match.

    No threshold, named-person label, admission or editorial reward is added.
    Missing/fourth-face descriptors must not borrow another face's vector.
    """
    values = []
    for observation in (left, right):
        value = np.asarray(observation.get('appearance_vector', np.zeros(128)), float)
        if (value.shape != (128,) or not np.isfinite(value).all()
                or abs(float(np.linalg.norm(value)) - 1.) > .001):
            return float('nan')
        values.append(value)
    return float(np.clip(values[0] @ values[1], -1., 1.))
