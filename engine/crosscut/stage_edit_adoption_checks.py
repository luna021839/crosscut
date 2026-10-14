"""Actual native-rank/public-output checks, not model or media qualification.

No checkpoint activation, source acquisition, split changes, or fitting. A
caller supplies independent expected raw76 values and exact source/clock IDs.
Complete per-probe evidence is collected before a failed adoption is raised.
"""
from __future__ import annotations

import numpy as np

RAW_ATOL, RAW_RTOL = 2e-4, 2e-5


def native_source_raw(model, batch, token_index, source_id):
    import mlx.core as mx
    slot = batch.source_ids[0][token_index].index(source_id)
    hidden, _, source, mask = model.base._hidden_and_sources(batch)
    if not bool(mask[0,token_index,slot].item()):
        raise ValueError('requested adoption source is absent or hard-gated')
    # Do NOT turn these into 1D matrix-vector products before applying heads.
    raw = (model.dynamic_crop_context_head(hidden)[...,None,:]
           + model.dynamic_crop_source_head(source))[0,token_index,slot]
    mx.eval(raw,hidden,source)
    if not np.isfinite(np.asarray(raw)).all():
        raise ValueError('nonfinite native raw76')
    return raw, hidden[0,token_index], source[0,token_index,slot], slot


def compare_public_source(model, probe):
    import mlx.core as mx
    from .stage_edit_dynamic_crop_head_v3 import (
        DYNAMIC_CROP_V3_OUTPUT_CONTRACT,renderer_safe_values_from_raw_v3)
    from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
        DynamicCropHeadPredictionV3,dynamic_crop_prediction_from_model_source_v3)

    batch,token,source_id = probe['batch'],probe['token_index'],probe['source_id']
    expected = np.asarray(probe['expected_raw'],np.float32)
    if expected.shape!=(76,) or not np.isfinite(expected).all():
        raise ValueError('independent finite raw76 reference required')
    slot = batch.source_ids[0][token].index(source_id)
    output = model(batch)  # Public __call__, not another standalone head.
    if output.output_contract!=DYNAMIC_CROP_V3_OUTPUT_CONTRACT or not bool(output.effective_source_mask[0,token,slot].item()):
        raise ValueError('public output source/contract differs from adoption probe')
    expected_values=np.asarray(renderer_safe_values_from_raw_v3(mx.array(expected[:32]).reshape(8,4)))
    pairs={
        'values':(np.asarray(output.dynamic_crop_values[0,token,slot]),expected_values),
        'count':(np.asarray(output.dynamic_crop_count_logits[0,token,slot]),expected[32:40]),
        'time':(np.asarray(output.dynamic_crop_time_interval_logits[0,token,slot]),expected[40:75]),
        'motion':(np.asarray(output.dynamic_crop_motion_logits[0,token,slot]),expected[75]),
    }
    groups={}
    for name,(actual,target) in pairs.items():
        same_shape=actual.shape==target.shape
        finite=bool(np.isfinite(actual).all())
        groups[name]=dict(passed=bool(same_shape and finite and np.allclose(actual,target,atol=RAW_ATOL,rtol=RAW_RTOL)),
            max_absolute_error=float(np.max(np.abs(actual-target))) if same_shape and finite else None)
    prediction=dynamic_crop_prediction_from_model_source_v3(output,batch,
        batch_index=0,token_index=token,source_index=slot)
    reference=DynamicCropHeadPredictionV3(source_id,slot,tuple(map(tuple,expected_values.tolist())),
        tuple(expected[32:40].tolist()),tuple(expected[40:75].tolist()),float(expected[75]))
    actual_decoded=prediction.decode(probe['duration_seconds'])
    expected_decoded=reference.decode(probe['duration_seconds'])
    semantic=dict(source_bound=prediction.source_id==source_id and prediction.source_index==slot,
        count_equal=prediction.predicted_keyframe_count==reference.predicted_keyframe_count,
        motion_sign_equal=(prediction.motion_logit>=0)==(reference.motion_logit>=0),
        both_eligible=actual_decoded.eligible and expected_decoded.eligible,
        renderer_values_equal=False,decoded_times_equal=False)
    if semantic['both_eligible']:
        a,b=actual_decoded.track,expected_decoded.track
        semantic['decoded_times_equal']=len(a.keyframes)==len(b.keyframes) and np.allclose(
            [k.time_seconds for k in a.keyframes],[k.time_seconds for k in b.keyframes],atol=RAW_ATOL,rtol=RAW_RTOL)
        times=probe.get('observed_times',tuple(k.time_seconds for k in b.keyframes))
        def values(track):
            return np.asarray([[k.zoom,k.center_x,k.center_y,k.roll_degrees]
                for k in (track.evaluate(float(t)) for t in times)])
        semantic['renderer_values_equal']=bool(np.allclose(values(a),values(b),atol=RAW_ATOL,rtol=RAW_RTOL))
    semantic={key:bool(value) for key,value in semantic.items()}
    return dict(source_id=source_id,original_token_index=token,source_slot=slot,
        groups=groups,semantics=semantic,
        passed=all(v['passed'] for v in groups.values()) and all(semantic.values()))


def verify_public_outputs(model, probes, *, check=lambda:None):
    model.eval()
    rows=[]
    for probe in probes:
        check()
        try:
            rows.append(compare_public_source(model,probe))
        except (ValueError,IndexError,AttributeError) as error:
            rows.append(dict(source_id=probe['source_id'],original_token_index=probe['token_index'],
                passed=False,error=f'{type(error).__name__}: {error}'))
    return dict(contract='native_public_source_adoption_v1',track_count=len(rows),
        atol=RAW_ATOL,rtol=RAW_RTOL,passed=bool(rows) and all(r['passed'] for r in rows),tracks=rows)


def adopt_and_verify(model,snapshot,probes,*,check=lambda:None,record=lambda evidence:None,assign=None):
    if assign is None:
        from .stage_edit_production_training import _assign_dynamic_crop_v3_prior_head_snapshot
        assign=_assign_dynamic_crop_v3_prior_head_snapshot
    assign(model,snapshot)
    evidence=verify_public_outputs(model,probes,check=check)
    record(evidence)
    if not evidence['passed']:
        raise AssertionError('actual public-output candidate adoption failed')
    return evidence
