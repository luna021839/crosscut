"""Loss attribution and measured update instruments, never release approval."""
from dataclasses import replace
import hashlib
import numpy as np


def norm(tree):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    leaves=[mx.sum(value*value) for _,value in tree_flatten(tree)]
    result=mx.sqrt(sum(leaves)) if leaves else mx.array(0.)
    mx.eval(result)
    value=float(result.item())
    if not np.isfinite(value):raise ValueError('nonfinite diagnostic gradient/delta')
    return value


def feature_groups(tree):
    return {name:norm(tree['base'][name]) for name in ('encoder','source_projection')}


def row_supervision(count,*,motion_masked=False,decoded_curve=False):
    from .stage_edit_dynamic_crop_head_v3 import DYNAMIC_CROP_V3_TIME_BLOCK_SLICES
    if count not in range(1,9):raise ValueError('one to eight supervised keyframes required')
    mask=np.zeros(76,bool);mask[:count*4]=True;mask[32:40]=True
    first,final=DYNAMIC_CROP_V3_TIME_BLOCK_SLICES.get(count,(0,0))
    mask[40+first:40+final]=True;mask[75]=not motion_masked
    # Marginal played-curve supervision can reach every count's actual
    # prefix and time branch. It does not create labels for other sources.
    if decoded_curve:mask[:75]=True
    return mask


def head_row_norms(gradients):
    import mlx.core as mx
    terms=[]
    names=('dynamic_crop_context_head','dynamic_crop_source_head')
    if 'dynamic_crop_interaction_head' in gradients:
        names+=('dynamic_crop_interaction_head',)
    for name in names:
        for value in gradients[name].values():
            squared=mx.square(value)
            terms.append(squared if value.ndim==1 else squared.sum(axis=tuple(range(1,value.ndim))))
    result=mx.sqrt(sum(terms));mx.eval(result)
    values=np.asarray(result)
    if values.shape!=(76,) or not np.isfinite(values).all():
        raise ValueError('finite per-row head gradients required')
    return values


def separated_loss_gradients(model,batch,base,dynamic,contract,*,check=lambda:None,record=lambda evidence:None):
    """Two extra backwards and one actual combined backward, matched RNG.

    Keep only one gradient tree live at a time. The returned combined gradient
    is used for the actual update, not thrown away for another stochastic draw.
    The caller's process-wide budget covers every forward/backward here.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from .stage_edit_transformer import stage_edit_loss
    from .stage_edit_dynamic_crop_head_v3 import stage_edit_dynamic_crop_v3_loss
    state=[mx.array(np.array(key,copy=True)) for key in mx.random.state]
    digest=hashlib.sha256(b''.join(np.asarray(key).tobytes() for key in state)).hexdigest()
    evidence={}
    for kind in ('base_only','crop_only','combined'):
        check();mx.random.state=state
        if kind=='base_only':
            objective=lambda m:stage_edit_loss(m,batch,base,contract.base)[0]*contract.base_loss_weight
        else:
            selected=replace(contract,base_loss_weight=0.) if kind=='crop_only' else contract
            objective=lambda m:stage_edit_dynamic_crop_v3_loss(m,batch,base,dynamic,selected)[0]
        loss,gradients=nn.value_and_grad(model,objective)(model)
        mx.eval(loss,gradients)
        evidence[kind]=dict(loss=float(loss.item()),feature_gradient_l2=feature_groups(gradients),
            matched_rng_sha256=digest)
        if kind!='combined':del gradients,loss
        check()
    evidence['crop_reaches_features']=all(value>0 for value in evidence['crop_only']['feature_gradient_l2'].values())
    record(evidence)
    if not evidence['crop_reaches_features']:
        raise AssertionError('crop-only loss does not reach encoder/source projection')
    return loss,gradients,evidence


def measured_feature_delta(model,before):
    from mlx.utils import tree_map
    current=model.base.parameters()
    return {name:tree_map(lambda after,old:after-old,current[name],before[name])
        for name in ('encoder','source_projection')}


def require_actual_feature_update(gradients,deltas):
    for name in ('encoder','source_projection'):
        if gradients[name]<=0 or deltas[name]<=0:
            raise AssertionError(f'joint feature gradient/update missing: {name}')
