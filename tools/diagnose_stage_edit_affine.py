#!/usr/bin/env python3
"""Zero-update numerical decomposition of an already saved train-only prior.

Never constructs a Transformer, runs an optimizer, consumes a validation track,
or writes/activates a model. The frozen fixture and original tolerance remain
unchanged. Private feature arrays stay in the local input file, not the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ATOL, RTOL = 2e-4, 2e-5


def decompose(state, tracks):
    import mlx.core as mx
    import mlx.nn as nn

    def d(name):
        return np.asarray(state[name], dtype=np.float64)

    def q(value):
        return np.asarray(value, dtype=np.float32).astype(np.float64)

    def mm(left, right):
        # Avoid platform BLAS status-flag leakage and use a deterministic FP64
        # reduction for the independent algebra/quantization reference.
        return np.einsum('...i,ij->...j', left, right, optimize=False)

    h, e = d('contexts'), d('sources')
    w0, b0, v0 = d('initial_context_weight'), d('initial_context_bias'), d('initial_source_weight')
    old_hc = d('initial_context_center') if 'initial_context_center' in state else np.zeros(h.shape[1])
    old_es = d('initial_source_center') if 'initial_source_center' in state else np.zeros(e.shape[1])
    ac, ab, ass = d('adapter_context_weight'), d('adapter_context_bias'), d('adapter_source_weight')
    pc, ps, mc, sc = d('context_projection'), d('source_projection'), d('context_mean'), d('context_scale')
    ms, ss = d('source_mean'), d('source_scale')
    if h.shape != e.shape or len(h) != len(tracks) or w0.shape != (76, h.shape[1]):
        raise ValueError('saved state and original fixture domains differ')
    if any(not np.isfinite(np.asarray(v)).all() for v in state.values()) or np.any(sc <= 0) or np.any(ss <= 0):
        raise ValueError('affine state must be finite with positive scales')
    expected = mm(h-old_hc, w0.T) + b0 + mm(e-old_es, v0.T) + mm((mm(h, pc.T) - mc) / sc, ac.T) + ab + mm((mm(e, ps.T) - ms) / ss, ass.T)
    wc = w0 + mm(ac / sc, pc)
    ws = v0 + mm(ass / ss, ps)
    bias = b0 + ab - mm(old_hc, w0.T) - mm(old_es, v0.T) - mm(mc / sc, ac.T) - mm(ms / ss, ass.T)
    # Quantize the deployed centers first. Re-promote those exact stored values
    # to evaluate the old factorized function at the center in double precision.
    hc, es = q(h.mean(axis=0)), q(e.mean(axis=0))
    center_bias = mm(hc-old_hc, w0.T) + b0 + mm(es-old_es, v0.T) + mm((mm(hc, pc.T) - mc) / sc, ac.T) + ab + mm((mm(es, ps.T) - ms) / ss, ass.T)

    def native(weight_c, offset, weight_s, center_c, center_s, *, ranked):
        context_head = nn.Linear(h.shape[1], 76)
        source_head = nn.Linear(h.shape[1], 76, bias=False)
        context_head.weight, context_head.bias = mx.array(weight_c.astype(np.float32)), mx.array(offset.astype(np.float32))
        source_head.weight = mx.array(weight_s.astype(np.float32))
        hh, ee = mx.array(h.astype(np.float32)), mx.array(e.astype(np.float32))
        cc, cs = mx.array(center_c.astype(np.float32)), mx.array(center_s.astype(np.float32))
        if ranked:
            values = context_head(hh[None] - cc)[..., None, :] + source_head(ee[None, :, None, :] - cs)
            mx.eval(values)
            return np.asarray(values[0, :, 0, :], dtype=np.float64)
        values = mx.stack([context_head(x - cc) + source_head(y - cs) for x, y in zip(hh, ee)])
        mx.eval(values)
        return np.asarray(values, dtype=np.float64)

    zero = np.zeros(h.shape[1])
    # This reference is deliberately independently expressed in the original
    # factorized float32 operation order, including its own cancellation error.
    factorized32 = []
    for hh, ee in zip(h.astype(np.float32), e.astype(np.float32)):
        hh, ee = mx.array(hh), mx.array(ee)
        base = (mx.matmul(mx.array(w0.astype(np.float32)), hh-mx.array(old_hc.astype(np.float32)))
                + mx.array(b0.astype(np.float32))
                + mx.matmul(mx.array(v0.astype(np.float32)), ee-mx.array(old_es.astype(np.float32))))
        ch = (mx.matmul(mx.array(pc.astype(np.float32)), hh) - mx.array(mc.astype(np.float32))) / mx.array(sc.astype(np.float32))
        se = (mx.matmul(mx.array(ps.astype(np.float32)), ee) - mx.array(ms.astype(np.float32))) / mx.array(ss.astype(np.float32))
        residual = (mx.addmm(mx.array(ab.astype(np.float32)), ch, mx.array(ac.astype(np.float32)).T)
                    + mx.matmul(se, mx.array(ass.astype(np.float32)).T))
        value = base + residual
        mx.eval(value)
        factorized32.append(np.asarray(value, dtype=np.float64))
    variants = {
        'fp64_unfused_reference': expected,
        'fp64_merged': mm(h, wc.T) + bias + mm(e, ws.T),
        'quantized_coefficients_evaluated_fp64': mm(h, q(wc).T) + q(bias) + mm(e, q(ws).T),
        'factorized_native_float32': np.stack(factorized32),
        'uncentered_native_1d': native(wc, bias, ws, zero, zero, ranked=False),
        'uncentered_native_ranked': native(wc, bias, ws, zero, zero, ranked=True),
        'centered_quantized_coefficients_evaluated_fp64': mm(h - hc, q(wc).T) + q(center_bias) + mm(e - es, q(ws).T),
        'centered_native_1d': native(wc, center_bias, ws, hc, es, ranked=False),
        'centered_native_ranked': native(wc, center_bias, ws, hc, es, ranked=True),
    }
    # Exercise the actual production merge and fixed-centered Module at native
    # tensor rank as well as the separately expressed numerical references.
    # This remains affine-only: there is no encoder, optimizer or model replay.
    from crosscut import stage_edit_production_training as production
    adapter = production._DynamicCropV3StructuredPriorAdapterModule(pc.shape[0])
    adapter.context_adapter.weight, adapter.context_adapter.bias = mx.array(ac.astype(np.float32)), mx.array(ab.astype(np.float32))
    adapter.source_adapter.weight = mx.array(ass.astype(np.float32))
    initial = dict(context_weight=mx.array(w0.astype(np.float32)), context_bias=mx.array(b0.astype(np.float32)),
        source_weight=mx.array(v0.astype(np.float32)), context_center=mx.array(old_hc.astype(np.float32)),
        source_center=mx.array(old_es.astype(np.float32)))
    normalizer = production._DynamicCropV3PriorProjectionNormalizer(
        context_mean=mx.array(mc.astype(np.float32)), context_scale=mx.array(sc.astype(np.float32)),
        source_mean=mx.array(ms.astype(np.float32)), source_scale=mx.array(ss.astype(np.float32)),
        context_center=mx.array(hc.astype(np.float32)), source_center=mx.array(es.astype(np.float32)),
        evidence={"input": "previously_saved_train_only_features"})
    snapshot = production._dynamic_crop_v3_structured_prior_head_snapshot(
        adapter, initial, mx.array(pc.astype(np.float32)), mx.array(ps.astype(np.float32)), normalizer)
    deployed = production._DynamicCropV3PriorHeadModule(
        snapshot['context_weight'], snapshot['context_bias'], snapshot['source_weight'],
        context_center=snapshot['context_center'], source_center=snapshot['source_center'])
    raw = deployed.context_head(mx.array(h.astype(np.float32))[None])[..., None, :] + deployed.source_head(
        mx.array(e.astype(np.float32))[None, :, None, :])
    mx.eval(raw)
    variants['production_centered_native_ranked'] = np.asarray(raw[0, :, 0], dtype=np.float64)
    results = {}
    for name, actual in variants.items():
        if not np.isfinite(actual).all():
            raise RuntimeError(f'nonfinite comparison path: {name}')
        error = np.abs(actual - expected)
        failed = error > ATOL + RTOL * np.abs(expected)
        records = []
        for i, track in enumerate(tracks):
            bad_rows = np.flatnonzero(failed[i])
            count = actual[i, 32:40]
            reference_count = expected[i, 32:40]
            ordered = np.sort(count)
            records.append(dict(source_id=track['source_id'], original_token_index=track['original_token_index'],
                max_absolute_error=float(error[i].max()), failed_rows=bad_rows.tolist(),
                failed_absolute_errors=error[i, bad_rows].tolist(),
                count_margin=float(ordered[-1] - ordered[-2]),
                count_changed=bool(np.argmax(count) != np.argmax(reference_count)),
                motion_sign_changed=bool((actual[i, 75] >= 0) != (expected[i, 75] >= 0))))
        results[name] = dict(max_absolute_error=float(error.max()), violated_elements=int(failed.sum()),
            passed_original_tolerance=not bool(failed.any()), tracks=records)
    native_vs_native = np.abs(variants['centered_native_ranked'] - variants['factorized_native_float32'])
    return dict(contract='saved_train_affine_zero_update_decomposition_v1', optimizer_updates=0,
        transformer_constructed=False, validation_forwards=0, heldout_forwards=0,
        production_activation_authorized=False, private_arrays_written_to_report=False,
        track_count=len(h), native_shapes=dict(context=[1, len(h), h.shape[1]], source=[1, len(h), 1, h.shape[1]]),
        tolerance=dict(atol=ATOL, rtol=RTOL, reference='independently evaluated unfused float64 expression'),
        variants=results, centered_vs_factorized_native_max_error=float(native_vs_native.max()),
        coefficient_l2=dict(context=float(np.linalg.norm(wc)), source=float(np.linalg.norm(ws)),
            uncentered_bias=float(np.linalg.norm(bias)), centered_bias=float(np.linalg.norm(center_bias))),
        decoded_renderer_or_model_adoption_verified=False,
        notes=['Native-rank check covers the affine head only, not real prefix/full encoder equivalence.',
               'Saved affine state is not a full model, optimizer, or RNG checkpoint.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', required=True, type=int)
    args = parser.parse_args()
    root = Path('/Volumes/CrossCutMedia/CrossCut-Training/diagnostics').resolve()
    if root.stat().st_dev != args.device or any(not p.resolve().is_relative_to(root) for p in (args.state, args.fixture, args.out)):
        parser.error('inputs and new output must be in the authorized SSD diagnostics directory')
    started = time.monotonic()
    with np.load(args.state, allow_pickle=False) as archive:
        state = {k: archive[k] for k in archive.files}
    fixture = json.loads(args.fixture.read_text())
    report = decompose(state, fixture['tracks'])
    report.update(elapsed_seconds=time.monotonic() - started,
        state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(optimizer_updates=0, elapsed_seconds=report['elapsed_seconds'],
        variants={k: {n: v[n] for n in ('max_absolute_error', 'violated_elements', 'passed_original_tolerance')}
                  for k, v in report['variants'].items()})), flush=True)


if __name__ == '__main__':
    main()
