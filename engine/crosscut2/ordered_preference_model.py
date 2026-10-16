"""Explicit ordered finite-context head; no default or fabricated checkpoint.

Per-frame masked geometry is encoded once, then a causal finite FIR reads
every encoded frame in order. This small preference head is NOT the product's
qualified Dynamic Crop Transformer and does not certify AV supervision.
"""
from functools import lru_cache
import json
import re
import numpy as np
from .ordered_context import NAMES,PERSON_NAMES,SCHEMA_SHA256,OrderedEvidence
from .sequence_preference_model import Model as ScalarModel,digest

CONTRACT='finite_context_ordered_target_preference_v1'
ARCHITECTURE='masked_tanh_ragged_person_set_and_scene_causal_FIR_softplus_v1'


def _logits(frames,values,ends,temporal,bias):
    B,T,D=values.shape;H=len(temporal);out=np.zeros((B,len(ends)))
    for e in range(len(ends)):
        end=int(ends[e]);a=np.searchsorted(frames,max(0,end-H));z=np.searchsorted(frames,end)
        for b in range(B):
            total=bias
            # Same chronological scalar operation order in the independent
            # reference and strict compiled implementation. No fast-math or
            # history/current floating-point reassociation changes near ties.
            for t in range(a,z):
                age=end-1-int(frames[t])
                for d in range(D):total+=values[b,t,d]*temporal[age,d]
            out[b,e]=total
    return out


@lru_cache(maxsize=1)
def _compiled():
    from numba import njit
    return njit(cache=False,nogil=True,fastmath=False,parallel=False)(_logits)


class OrderedModel(ScalarModel):
    pointwise_frame_encoder=True
    def __init__(self,artifact):
        payload=dict(artifact);expected=payload.pop('content_sha256',None)
        config=payload.get('window_config',{});training=payload.get('training',{})
        fps=config.get('clock_fps',0.)
        if not isinstance(fps,(int,float)) or not np.isfinite(fps) or not 1<=fps<=120:
            raise ValueError('explicit fitted output clock required for ordered context')
        context_frames=int(np.ceil(4.*fps));D=payload.get('embedding_width')
        groups=[training.get(n,[]) for n in ('train_physical_groups','dev_physical_groups')]
        if (expected!=digest(payload) or payload.get('contract')!=CONTRACT
                or payload.get('architecture')!=ARCHITECTURE or payload.get('output_kind')!='group'
                or payload.get('fitted') is not True or payload.get('feature_names')!=dict(scene=list(NAMES),person=list(PERSON_NAMES))
                or payload.get('feature_schema_sha256')!=SCHEMA_SHA256
                or config!=dict(schema_sha256=SCHEMA_SHA256,interval_frames=6,context_seconds=4.,
                    clock_fps=fps,context_frames=context_frames)
                or payload.get('window_contract_sha256')!=digest(config)
                or type(D) is not int or not 1<=D<=16
                or training.get('supervision_kind')!='human_av_context_preference'
                or training.get('self_reported_normal_av') is not True
                or training.get('relative_and_absolute_labels_separate') is not True
                or training.get('physical_context_overlap_grouped') is not True
                or training.get('sealed_evaluation_excluded_from_fit') is not True
                or not re.fullmatch('[0-9a-f]{64}',training.get('examples_sha256',''))
                or any(not isinstance(g,list) or not g or any(not isinstance(x,str) or not x for x in g) for g in groups)
                or set(groups[0])&set(groups[1])
                or any(type(training.get(n)) is not int or training[n]<1 for n in
                    ('absolute_positive_training_options','absolute_negative_training_options'))
                or payload.get('normal_app_activation_approved') is not False
                or payload.get('professional_quality_approved') is not False):
            raise ValueError('new fitted ordered schema and separate genuine grouped AV provenance required')
        specs=dict(center=(len(NAMES),),scale=(len(NAMES),),encoder_values=(len(NAMES),D),
            encoder_known=(len(NAMES),D),encoder_conflicted=(len(NAMES),D),encoder_bias=(D,),
            person_center=(len(PERSON_NAMES),),person_scale=(len(PERSON_NAMES),),person_values=(len(PERSON_NAMES),D),
            person_known=(len(PERSON_NAMES),D),person_conflicted=(len(PERSON_NAMES),D),person_bias=(D,),set_weights=(D,D),
            temporal_weights=(context_frames,D))
        self.parameters={}
        for name,shape in specs.items():
            value=np.asarray(payload.get(name),float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('finite fitted ordered parameters required: '+name)
            if name in ('scale','person_scale') and np.any(value<=0):raise ValueError('positive ordered feature scales required')
            value=value.copy();value.setflags(write=False);self.parameters[name]=value
        self.bias=float(payload.get('bias',np.nan))
        if not np.isfinite(self.bias):raise ValueError('finite fitted ordered bias required')
        self.artifact=json.loads(json.dumps(artifact,allow_nan=False));self.bound_content_sha256=expected
        self.context_seconds=4.;self.interval_frames=6;self.clock_fps=float(fps);self.context_frames=context_frames
        self.sequence_feature_schema=SCHEMA_SHA256;self.sequence_window_contract=digest(config);self.embedding_width=D

    def encode(self,evidence):
        if not isinstance(evidence,OrderedEvidence):raise ValueError('ordered target-bound observations required')
        p=self.parameters;values=np.where(evidence.observed,(evidence.values-p['center'])/p['scale'],0.)
        result=np.einsum('tf,fd->td',values,p['encoder_values'],optimize=False,dtype=np.float64)
        result+=np.einsum('tf,fd->td',evidence.observed,p['encoder_known'],optimize=False,dtype=np.float64)
        result+=np.einsum('tf,fd->td',evidence.conflicted,p['encoder_conflicted'],optimize=False,dtype=np.float64)
        person=np.where(evidence.people_observed,(evidence.people-p['person_center'])/p['person_scale'],0.)
        encoded=np.einsum('pf,fd->pd',person,p['person_values'],optimize=False,dtype=np.float64)
        encoded+=np.einsum('pf,fd->pd',evidence.people_observed,p['person_known'],optimize=False,dtype=np.float64)
        encoded+=np.einsum('pf,fd->pd',evidence.people_conflicted,p['person_conflicted'],optimize=False,dtype=np.float64)
        encoded=np.tanh(encoded+p['person_bias'])
        count=np.diff(evidence.person_splits);pooled=np.zeros((len(evidence.frames),self.embedding_width))
        np.add.at(pooled,np.repeat(np.arange(len(count)),count),encoded)
        pooled/=np.maximum(1,count)[:,None]
        result+=np.einsum('td,df->tf',pooled,p['set_weights'],optimize=False,dtype=np.float64)
        return np.tanh(result+p['encoder_bias'])

    def rates(self,frames,encoded,ends,*,kernel='numpy'):
        frames=np.asarray(frames,np.int64);encoded=np.asarray(encoded,float);ends=np.asarray(ends,np.int64)
        if (frames.ndim!=1 or np.any(np.diff(frames)<=0) or frames.min(initial=0)<0
                or encoded.ndim!=3 or encoded.shape[1:]!=(len(frames),self.embedding_width)
                or not np.isfinite(encoded).all() or ends.ndim!=1 or np.any(np.diff(ends)<=0)
                or ends.min(initial=1)<1 or kernel not in ('numpy','numba')):
            raise ValueError('ordered exact output ticks and bounded causal embeddings required')
        function=_compiled() if kernel=='numba' else _logits
        logits=function(frames,encoded,ends,self.parameters['temporal_weights'],self.bias)
        return np.logaddexp(0.,logits)

    def costs(self,features):
        raise ValueError('189 pooled statistics cannot enter an ordered sequence model')

    def verify(self):
        payload=dict(self.artifact);expected=payload.pop('content_sha256',None)
        if (expected!=self.bound_content_sha256 or expected!=digest(payload)
                or self.sequence_feature_schema!=SCHEMA_SHA256 or self.context_seconds!=4. or self.interval_frames!=6
                or self.clock_fps!=payload['window_config']['clock_fps']
                or self.context_frames!=payload['window_config']['context_frames']
                or self.sequence_window_contract!=digest(payload['window_config'])
                or self.embedding_width!=payload['embedding_width'] or self.bias!=payload['bias']
                or any(not np.array_equal(np.asarray(payload[n]),v) for n,v in self.parameters.items())):
            raise ValueError('ordered model parameters changed after verified receipt')

    def prepare(self,**kwargs):
        self.verify()
        from .block_played_context import BlockPlayedContext
        return BlockPlayedContext(self,**kwargs)


def load_model(artifact):
    return OrderedModel(artifact) if artifact.get('contract')==CONTRACT else ScalarModel(artifact)
