"""Predeclared train-only release probes; not editorial hard-cut policy.

Dropout-only changes are measured separately from deterministic serving drift.
Only the latter is a post-update stop gate. No thresholds are fit to a run.
"""
import numpy as np

LIMITS=dict(max_raw76_step_drift=1.,max_normalized_value_step_drift=.25,
    max_projected_train_sigma_step_drift=10.)


def capture_sources(model,probes,projections,normalizer,*,check=lambda:None):
    import mlx.core as mx
    from .stage_edit_adoption_checks import native_source_raw
    from .stage_edit_dynamic_crop_head_v3 import renderer_safe_values_from_raw_v3,renderer_range_normalized_values_v3
    rows=[]
    for probe in probes:
        check()
        raw,context,source,_=native_source_raw(model,probe['batch'],probe['token_index'],probe['source_id'])
        value=renderer_range_normalized_values_v3(renderer_safe_values_from_raw_v3(raw[:32].reshape(8,4)))
        zc=(projections[0]@context-normalizer.context_mean)/normalizer.context_scale
        zs=(projections[1]@source-normalizer.source_mean)/normalizer.source_scale
        mx.eval(value,zc,zs)
        rows.append(dict(source_id=probe['source_id'],token_index=probe['token_index'],
            raw=np.array(raw),value=np.array(value),features=np.concatenate((np.asarray(zc),np.asarray(zs)))))
    return rows


def compare_drift(before,after):
    if not before or len(before)!=len(after):raise ValueError('complete source-bound drift census required')
    rows=[]
    for a,b in zip(before,after,strict=True):
        if (a['source_id'],a['token_index'])!=(b['source_id'],b['token_index']):
            raise ValueError('release probe source/token changed')
        if not all(np.isfinite(b[k]).all() for k in ('raw','value','features')):
            raise ValueError('nonfinite release output/features')
        rows.append(dict(source_id=a['source_id'],original_token_index=a['token_index'],
            raw76_step_drift=float(np.max(np.abs(a['raw']-b['raw']))),
            normalized_value_step_drift=float(np.max(np.abs(a['value']-b['value']))),
            projected_train_sigma_step_drift=float(np.max(np.abs(a['features']-b['features']))),
            count_changed=bool(np.argmax(a['raw'][32:40])!=np.argmax(b['raw'][32:40])),
            motion_sign_changed=bool((a['raw'][75]>=0)!=(b['raw'][75]>=0))))
    passed=all(row['raw76_step_drift']<=LIMITS['max_raw76_step_drift']
        and row['normalized_value_step_drift']<=LIMITS['max_normalized_value_step_drift']
        and row['projected_train_sigma_step_drift']<=LIMITS['max_projected_train_sigma_step_drift'] for row in rows)
    return dict(passed=passed,limits=dict(LIMITS),track_count=len(rows),tracks=rows)


def snapshot_transaction(model,optimizer):
    import mlx.core as mx
    from mlx.utils import tree_map
    # MLX leaves are immutable; copy tree containers, not 106M arrays to disk.
    return dict(parameters=tree_map(lambda v:v,model.parameters()),
        optimizer=tree_map(lambda v:v,optimizer.state),rng=[mx.array(np.array(k,copy=True)) for k in mx.random.state])


def restore_transaction(model,optimizer,snapshot):
    import mlx.core as mx
    from .stage_edit_production_training import _parameter_tree_sha256
    model.update(snapshot['parameters']);optimizer.state=snapshot['optimizer'];mx.random.state=snapshot['rng']
    mx.eval(model.parameters(),optimizer.state)
    if (_parameter_tree_sha256(model.parameters())!=_parameter_tree_sha256(snapshot['parameters'])
            or _parameter_tree_sha256(optimizer.state)!=_parameter_tree_sha256(snapshot['optimizer'])):
        raise AssertionError('release rollback did not restore parameter/optimizer state')
