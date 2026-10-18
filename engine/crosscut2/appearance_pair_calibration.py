"""Explicit local reference-pair cosine risk, independent of song/edit choice.

The photo-domain result is a soft retrieval cue, not an actor identity or
professional-editing certificate. Unknown queries retain the old neutral.
The opt-in context is task/thread local and always restored after a solve.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from crosscut.native_face_descriptor import ARCFACE_R50_SHA256

_ACTIVE=ContextVar('crosscut_reference_pair_calibration',default=None)


@dataclass(frozen=True)
class PairCalibration:
    intercept: float
    slope: float
    gallery_identity: str
    reference_artifact_sha256: str
    strength: float=.5
    model_sha256: str=ARCFACE_R50_SHA256

    def __post_init__(self):
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in (self.intercept,self.slope,self.strength))
                or not 0<self.slope<=100 or abs(self.intercept)>100
                or not 0<=self.strength<=.5 or not 0<=-self.intercept/self.slope<=1
                or self.model_sha256!=ARCFACE_R50_SHA256
                or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                    for v in (self.gallery_identity,self.reference_artifact_sha256))):
            raise ValueError('bounded explicit R50 reference-pair calibration required')

    def risk(self,cosine):
        value=np.asarray(cosine,float)
        if not np.isfinite(value).all() or np.any((value< -1)|(value>1)):
            raise ValueError('actual normalized direct face cosine required')
        reference=1./(1.+np.exp(np.clip(self.intercept+self.slope*value,-50,50)))
        return (1.-self.strength)*(1.-value)/2.+self.strength*reference

    def to_dict(self):
        return dict(contract='explicit_reference_only_R50_pair_risk_v1',intercept=self.intercept,slope=self.slope,
            strength=self.strength,gallery_identity=self.gallery_identity,model_sha256=self.model_sha256,
            reference_artifact_sha256=self.reference_artifact_sha256,reference_only=True,
            song_or_source_choice_training=False,named_identity_certified=False,source_or_cut_veto=False,
            stage_video_generalization_proven=False,professional_quality_approved=False)


def active():return _ACTIVE.get()


@contextmanager
def use(calibration):
    if not isinstance(calibration,PairCalibration):raise ValueError('explicit validated pair calibration required')
    token=_ACTIVE.set(calibration)
    try:yield calibration
    finally:_ACTIVE.reset(token)
