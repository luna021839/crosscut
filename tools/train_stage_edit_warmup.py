#!/usr/bin/env python3
"""Cache real full-size cut/source warmup on the two bound TRAIN references.

This is reusable preparation, not a qualified production checkpoint. It does
not inspect validation/held-out references, edit a registry, render, or acquire
media. The existing diagnostic's immutable input binding is preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import diagnose_stage_edit_training as diagnostic
from crosscut.stage_edit_classifier_warmup import (
    CLASSIFIER_WARMUP_CONTRACT, PHASE_MODULES,
    begin_classifier_warmup, finish_classifier_warmup,
)


VOLUME = Path("/Volumes/CrossCutMedia")
STEPS = 500
SEED = 1731
PROTOCOL = dict(
    contract="full_production_shape_train_only_preserved_classifier_warmup_v2",
    classifier_warmup_contract=CLASSIFIER_WARMUP_CONTRACT,
    steps_per_phase=STEPS, window_length=32, peak_learning_rate=3e-4,
    warmup_steps=50, minimum_learning_rate_ratio=.1,
    weight_decay=.01, gradient_clip_norm=1., seed=SEED,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_volume(out, device):
    if not VOLUME.is_mount() or VOLUME.stat().st_dev != device or out.stat().st_dev != device:
        raise RuntimeError("original SSD is no longer mounted; no internal fallback")
    if shutil.disk_usage(VOLUME).free < 1100 * 2**20:
        raise RuntimeError("less than 1100 MiB available for the two warmup checkpoints")


def run_phase(model, corpus, config, *, phase, steps, budget, completed):
    """Same balanced windows, tensorization, losses and AdamW as production."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from crosscut import stage_edit_production_training as production
    from crosscut.stage_edit_transformer import (
        DEFAULT_STAGE_EDIT_SCHEMA as schema, StageEditLossContract,
        tensorize_stage_edit_batch, tensorize_stage_edit_supervision, stage_edit_loss,
    )

    if phase not in ("cut", "source") or steps < 1:
        raise ValueError("a positive cut/source warmup is required")
    base = StageEditLossContract(
        transition_authority=config.transition_authority,
        transition_weight=0., source_objective="eligible_all_hybrid",
    )
    if phase == "cut":
        positive, negative = production._balanced_cut_windows(
            corpus.examples, schema=schema, window_length=32,
        )
        samplers = tuple(production._HierarchicalCutWindowSampler(group, seed=SEED ^ 0x5A17E)
                         for group in (positive, negative))
        contract = production._cut_only_pretraining_loss_contract(base)
        sample = lambda: tuple(sampler.sample() for sampler in samplers)
    else:
        inventory = production._source_pretraining_inventory(
            corpus.examples, schema=schema, window_length=32,
        )
        sampler = production._HierarchicalSourceWindowSampler(inventory.windows, seed=SEED ^ 0x50A7CE)
        contract = production._source_only_pretraining_loss_contract(base)
        sample = lambda: (sampler.sample(),)
    schedule = production._warmup_cosine_schedule(
        peak_learning_rate=PROTOCOL["peak_learning_rate"], total_steps=steps,
        warmup_steps=min(PROTOCOL["warmup_steps"], steps - 1),
        minimum_learning_rate_ratio=PROTOCOL["minimum_learning_rate_ratio"],
    )
    scope = begin_classifier_warmup(model, phase)
    try:
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=PROTOCOL["weight_decay"])
        loss_grad = nn.value_and_grad(model, lambda trained, batch, targets:
                                     stage_edit_loss(trained, batch, targets, contract)[0])
        for step in range(1, steps + 1):
            budget.check()
            windows = sample()
            batch = tensorize_stage_edit_batch(tuple(window.sequence for window in windows), schema, config)
            targets = tensorize_stage_edit_supervision(tuple(window.targets for window in windows), batch, schema)
            model.train()
            loss, gradients = loss_grad(model, batch, targets)
            gradients, norm = optim.clip_grad_norm(gradients, PROTOCOL["gradient_clip_norm"])
            optimizer.update(model, gradients)
            mx.eval(loss, norm, model.parameters(), optimizer.state)
            if not float("-inf") < float(loss.item()) < float("inf"):
                raise RuntimeError("nonfinite warmup loss")
            completed(dict(phase=phase, step=step, total_steps=steps, loss=float(loss.item()),
                           gradient_norm=float(norm.item()), **budget.check()))
    finally:
        classifier_evidence = finish_classifier_warmup(model, scope)
    if int(optimizer.state["step"].item()) != steps:
        raise RuntimeError("warmup AdamW step count differs")
    if phase == "cut":
        sampling = dict(positive=samplers[0].diagnostics(), negative=samplers[1].diagnostics())
    else:
        sampling = dict(per_song_sampled_counts=sampler.per_song_sampled_counts,
                        per_reference_sampled_counts=sampler.per_reference_sampled_counts)
    return dict(optimizer_steps=steps, loss_contract=asdict(contract), sampling=sampling,
                classifier_warmup=classifier_evidence)


def _preserved_classifier_stage(stage, phase):
    value = stage.get('classifier_warmup', {})
    before, after = value.get('frozen_parameter_sha256_before'), value.get('frozen_parameter_sha256_after')
    return (value.get('contract') == CLASSIFIER_WARMUP_CONTRACT
        and value.get('phase') == phase and value.get('trainable_modules') == list(PHASE_MODULES[phase])
        and value.get('all_nonclassifier_parameters_byte_identical') is True
        and value.get('classifier_parameters_changed') is True
        and value.get('full_joint_trainability_restored') is True
        and value.get('encoder_updated_during_classifier_warmup') is False
        and isinstance(before, str) and len(before) == 64 and set(before) <= set('0123456789abcdef')
        and before == after)


def restore_cached_warmup(model, cache, fixture_binding, *, check=lambda: None):
    """Restore a completed, input-bound warmup, never partial/foreign weights."""
    import mlx.core as mx
    import numpy as np
    from crosscut.stage_edit_production_training import _parameter_tree_sha256
    from crosscut.stage_edit_dynamic_crop_head_v3 import stage_edit_dynamic_crop_v3_parameter_count

    root = Path(cache).resolve()
    report_path = root / "warmup-report.json"
    report, report_digest = diagnostic.read_bound_json(report_path)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    if (
        report.get("status") != "train_only_warmup_cached_not_qualified"
        or report.get("protocol") != PROTOCOL
        or report.get("validation_forwards") != 0
        or report.get("heldout_forwards") != 0
        or report.get("production_activation_authorized") is not False
        or canonical(report.get("fixture")) != canonical(fixture_binding)
        or report.get("parameter_count") != stage_edit_dynamic_crop_v3_parameter_count(
            model.base_config, model.dynamic_crop_config)
        or sha256(root / "warmup-runner-source.py") != report.get("runner_sha256")
        or any(report.get("stages", {}).get(phase, {}).get("optimizer_steps") != STEPS
               or not _preserved_classifier_stage(report.get("stages", {}).get(phase, {}), phase)
               for phase in ("cut", "source"))
    ):
        raise ValueError("warmup cache protocol, completed phases or TRAIN inputs differ")
    stage = report["stages"]["source"]
    weights = root / "after-source.safetensors"
    if weights.is_symlink() or str(weights) != stage.get("weights") or sha256(weights) != stage.get("weight_sha256"):
        raise ValueError("warmup cache weight binding differs")
    # Warmup starts from fresh heads and never calibrates a fixed affine
    # center. Save/reload below concerns the parameter tree, not a prior head.
    for name in ("dynamic_crop_context_head", "dynamic_crop_source_head"):
        if np.any(np.asarray(getattr(model, name).input_center)):
            raise ValueError("warmup restoration requires fresh zero-centered heads")
    check()
    model.load_weights(str(weights), strict=True)
    mx.eval(model.parameters())
    check()
    if _parameter_tree_sha256(model.parameters()) != stage.get("parameter_tree_sha256"):
        raise ValueError("restored warmup parameter tree differs")
    # A fresh next-phase RNG is explicit: this cache does not claim to restore
    # the old Adam moments, or the dropout stream consumed by the warmup.
    mx.random.seed(SEED)
    return dict(report=str(report_path), report_sha256=report_digest,
                weight_sha256=stage["weight_sha256"], optimizer_steps_reused=2 * STEPS,
                warmup_optimizer_steps_repeated=0, next_phase_rng_seed=SEED,
                optimizer_moments_restored=False, prior_or_joint_training_already_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.absolute()
    if not VOLUME.is_mount() or VOLUME.resolve() not in out.parents or out.is_symlink():
        raise ValueError("an unused output directory on the mounted SSD is required")
    device = VOLUME.stat().st_dev
    out.mkdir(parents=False, exist_ok=False)
    require_volume(out, device)
    budget = diagnostic.DiagnosticBudget(out)
    watcher = threading.Thread(target=budget.watch, daemon=True)
    watcher.start()
    report = dict(protocol=PROTOCOL, diagnostic_only=True, production_ready=False,
                  validation_forwards=0, heldout_forwards=0, production_activation_authorized=False,
                  stages={}, progress=[], status="preparing_inputs", pid=os.getpid())
    report_path = out / "warmup-report.json"

    def write():
        require_volume(out, device)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))

    def completed(record):
        if record["step"] == 1 or record["step"] % 25 == 0 or record["step"] == record["total_steps"]:
            report["progress"].append(record)
            write()
            print(json.dumps(record), flush=True)

    try:
        fixture = diagnostic.prepare_fixture()
        corpus, _, config, _, _, binding = fixture
        report["fixture"] = binding
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["runner_sha256"] = sha256(__file__)
        write()
        import mlx.core as mx
        from crosscut import stage_edit_production_training as production
        from crosscut.stage_edit_dynamic_crop_head_v3 import (
            StageEditTransformerDynamicCropV3, stage_edit_dynamic_crop_v3_parameter_count,
        )
        mx.set_cache_limit(128 * 2**20)
        mx.set_memory_limit(budget.mlx_bytes)
        mx.random.seed(SEED)
        model = StageEditTransformerDynamicCropV3(config)
        production._enable_dynamic_crop_v3_activation_checkpointing(model)
        mx.eval(model.parameters())
        report["parameter_count"] = stage_edit_dynamic_crop_v3_parameter_count(config)
        for phase in ("cut", "source"):
            report["status"] = f"training_{phase}"
            write()
            stage = run_phase(model, corpus, config, phase=phase, steps=STEPS,
                              budget=budget, completed=completed)
            model.eval()
            budget.check()
            require_volume(out, device)
            weights = out / f"after-{phase}.safetensors"
            model.save_weights(str(weights))
            mx.eval(model.parameters())
            stage.update(weights=str(weights), weight_sha256=sha256(weights),
                         parameter_tree_sha256=production._parameter_tree_sha256(model.parameters()))
            report["stages"][phase] = stage
            write()
            gc.collect()
            mx.clear_cache()
        report.update(status="train_only_warmup_cached_not_qualified", **budget.check())
        write()
    except Exception as error:
        report.update(status="failed_not_qualified", error=f"{type(error).__name__}: {error}")
        if VOLUME.is_mount() and VOLUME.stat().st_dev == device and out.exists():
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        raise
    finally:
        budget.done.set()
        watcher.join(timeout=1.)


if __name__ == "__main__":
    main()
