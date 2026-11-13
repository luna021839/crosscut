"""Explicit fixed-time context model; not a renamed same-event checkpoint.

Loaded parameters must bind the actual context schema/window contract and
declare separate relative/absolute human AV evidence. Integrity/self-report
metadata is not independent proof of listening, calibration, or film quality.
No default weights, fake labels, fitting loop or normal-app approval is given.
"""
import hashlib
import json
import re
import numpy as np
from .sequence_context import SCHEMA_SHA256,FEATURE_NAMES

CONTRACT='fixed_time_target_sequence_preference_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Model:
    def __init__(self,artifact):
        payload=dict(artifact);expected=payload.pop('content_sha256',None)
        config=payload.get('window_config',{});training=payload.get('training',{})
        groups=[training.get(name,[]) for name in ('train_physical_groups','dev_physical_groups')]
        if (expected!=digest(payload) or payload.get('contract')!=CONTRACT
                or payload.get('output_kind')!='group' or payload.get('fitted') is not True
                or payload.get('feature_schema_sha256')!=SCHEMA_SHA256
                or payload.get('feature_names')!=list(FEATURE_NAMES)
                or config!={'schema_sha256':SCHEMA_SHA256,'interval_frames':6,'context_seconds':4.}
                or payload.get('window_contract_sha256')!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                or training.get('supervision_kind')!='human_av_context_preference'
                or training.get('self_reported_normal_av') is not True
                or training.get('relative_and_absolute_labels_separate') is not True
                or not re.fullmatch('[0-9a-f]{64}',training.get('examples_sha256',''))
                or any(not isinstance(g,list) or not g or any(not isinstance(x,str) or not x for x in g) for g in groups)
                or set(groups[0])&set(groups[1])
                or type(training.get('absolute_positive_training_options')) is not int
                or type(training.get('absolute_negative_training_options')) is not int
                or training['absolute_positive_training_options']<1 or training['absolute_negative_training_options']<1
                or payload.get('normal_app_activation_approved') is not False
                or payload.get('professional_quality_approved') is not False):
            raise ValueError('new fitted sequence schema and nonfabricated relative/absolute AV provenance required')
        self.center=np.asarray(payload.get('center'),float);self.scale=np.asarray(payload.get('scale'),float)
        self.weights=np.asarray(payload.get('weights'),float);self.bias=float(payload.get('bias',np.nan))
        if (any(v.shape!=(len(FEATURE_NAMES),) or not np.isfinite(v).all() for v in (self.center,self.scale,self.weights))
                or np.any(self.scale<=0) or not np.isfinite(self.bias)):
            raise ValueError('finite fitted sequence parameters required')
        # Preserve the exact model receipt against accidental caller mutation.
        self.artifact=json.loads(json.dumps(artifact,allow_nan=False))
        self.bound_content_sha256=expected
        for value in (self.center,self.scale,self.weights):value.setflags(write=False)
        self.sequence_feature_schema=SCHEMA_SHA256
        self.sequence_window_contract=payload['window_contract_sha256']
        self.context_seconds=4.;self.interval_frames=6

    def costs(self,features):
        values=np.asarray(features)
        if (values.dtype.kind not in 'iuf' or values.shape[-1:]!=(len(FEATURE_NAMES),)
                or not np.isfinite(values).all()):raise ValueError('bound finite sequence context features required')
        logit=np.einsum('...f,f->...',values,self.weights/self.scale,optimize=False,dtype=np.float64)
        logit+=self.bias-float(self.center@(self.weights/self.scale))
        # Nonnegative uncalibrated loss rate, not a human probability or a
        # per-cut fee. The same fixed-time integral scores split/unsplit films.
        return np.logaddexp(0.,logit)

    def verify(self):
        payload=dict(self.artifact);expected=payload.pop('content_sha256',None)
        if (expected!=self.bound_content_sha256 or expected!=digest(payload)
                or self.context_seconds!=4. or self.interval_frames!=6
                or self.sequence_feature_schema!=SCHEMA_SHA256
                or self.sequence_window_contract!=payload.get('window_contract_sha256')
                or any(not np.array_equal(np.asarray(payload.get(name)),value)
                for name,value in (('center',self.center),('scale',self.scale),('weights',self.weights)))
                or payload.get('bias')!=self.bias):
            raise ValueError('sequence model parameters changed after their verified receipt')

    def prepare(self,*,nodes,entries,exits,catalogues,arcs,native_ports,fps,source_bindings,check,kernel='numpy'):
        self.verify()
        from .played_context import PlayedContext
        return PlayedContext(self,nodes=nodes,entries=entries,exits=exits,catalogues=catalogues,
            arcs=arcs,native_ports=native_ports,fps=fps,source_bindings=source_bindings,check=check,kernel=kernel)
