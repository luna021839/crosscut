#!/usr/bin/env python3
"""Bounded train-only diagnostic. Never qualifies or activates a checkpoint.

Read only the historical train references explicitly named below. The frozen
validation references, held-out song and final blind song are not data inputs.
Labels stratify the fixture before creating a model; predictions cannot change
it. Reuses production tensorization, losses, adapter merge and renderer decoder.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

TRAIN_REFERENCES = {
    "IVE-BANG-BANG": "0O-bHFSeYxQ",
    "IVE-BLACKHOLE": "8yVatqZOCls",
}
PLAN = Path("/Volumes/CrossCutMedia/CrossCut-Training/four-song-loso-v3-full-head-v3/plans/"
    "stageedit-four-song-loso-plan-a46afbdca134c4c51dccd0239429ca84b551385429b83078d521b19ed49b45ab.json")
CONFIG_REPORT = Path("/Volumes/CrossCutMedia/CrossCut-Training/current-four-song-temporal-input-v7-20260913/"
    "stageedit-production-training-failure-report.json")
HISTORICAL_BINDING = Path(__file__).with_name("fixtures") / "stage_edit_train_only_inputs_v1.json"


def read_bound_json(path, expected=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"immutable input digest mismatch: {path}")
    return json.loads(data), digest


def read_historical_input(path, *, kind, reference_id, binding):
    matches = [row for row in binding["direct_inputs"]
        if row["kind"] == kind and row["reference_id"] == reference_id]
    if len(matches) != 1 or matches[0]["path"] != str(Path(path)):
        raise ValueError("historical diagnostic input path/reference binding mismatch")
    return read_bound_json(path, matches[0]["sha256"])


def require_historical_fixture_identity(evidence, binding):
    for key in ("plan_sha256", "config_report_sha256", "train_reference_ids", "data_id"):
        if evidence[key] != binding[key]:
            raise ValueError(f"historical diagnostic fixture identity mismatch: {key}")
    digest = hashlib.sha256(json.dumps(evidence["tracks"],sort_keys=True,
        separators=(",", ":"),allow_nan=False).encode()).hexdigest()
    if digest != binding["selected_tracks_sha256"]:
        raise ValueError("historical diagnostic fixture target/selection mismatch")


def select_fixture(tracks, *, per_song=6):
    """First chronological label strata, then fill; no prediction-based search."""
    from crosscut.stage_edit_training_preflight import require_trajectory_selection_coverage

    selected = []
    for song_id in sorted(TRAIN_REFERENCES):
        candidates = sorted((t for t in tracks if t.song_id == song_id),
            key=lambda t: (t.token_index, t.source_id))
        chosen = []
        # Include both observed motion classes and every available count class
        # before filling chronologically. Fail rather than change the split.
        for label in ("static", "dynamic"):
            match = next((t for t in candidates if not t.motion_label_loss_masked
                and t.motion_label == label), None)
            if match is not None and match not in chosen:
                chosen.append(match)
        for count in sorted({len(t.values) for t in candidates}):
            match = next(t for t in candidates if len(t.values) == count)
            if match not in chosen:
                chosen.append(match)
        if len(chosen) > per_song:
            raise ValueError("predeclared label strata exceed the fixture budget")
        for track in candidates:
            if len(chosen) >= per_song:
                break
            if track not in chosen:
                chosen.append(track)
        if len(chosen) != per_song:
            raise ValueError(f"insufficient real train trajectories: {song_id}")
        selected.extend(sorted(chosen, key=lambda t: t.token_index))
    require_trajectory_selection_coverage(selected, partition="train")
    if not 8 <= len(selected) <= 16 or len({t.song_id for t in selected}) < 2:
        raise ValueError("real train-only fixture coverage is insufficient")
    return tuple(selected)


def prepare_fixture():
    from crosscut.ive import EditMode
    from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
    from crosscut.stage_edit_reference_adapter import load_reference_training_corpus
    from crosscut.stage_edit_dynamic_crop_training import build_stage_edit_dynamic_crop_training_corpus_v2
    from crosscut.stage_edit_dynamic_crop_training_v3 import build_stage_edit_dynamic_crop_training_corpus_v3
    from crosscut.stage_edit_production_training import (
        _dynamic_crop_v3_prior_inventory, prepare_stage_edit_production_input_sequences)
    from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA as schema
    from crosscut.stage_edit_transformer import StageEditTransformerConfig

    # These expected values were extracted from the saved, PRE-EXISTING third
    # diagnostic receipt, not from whichever inputs happen to be present now.
    # There is deliberately no refresh/accept-current-hashes command or fallback.
    binding, binding_hash = read_bound_json(HISTORICAL_BINDING)
    if (binding.get("binding_contract") != "historical_train_fixture_binding_v1"
            or binding.get("train_reference_ids") != sorted(TRAIN_REFERENCES.values())):
        raise ValueError("historical diagnostic binding contract/reference set mismatch")
    plan, plan_hash = read_bound_json(PLAN,binding["plan_sha256"])
    old_report, report_hash = read_bound_json(CONFIG_REPORT,binding["config_report_sha256"])
    # Partition authority is the frozen train list, not a guessed song name.
    train_ids = set(old_report["training_reference_ids"])
    validation_ids = set(old_report["validation_reference_ids"])
    if not set(TRAIN_REFERENCES.values()) <= train_ids or set(TRAIN_REFERENCES.values()) & validation_ids:
        raise ValueError("fixture references are not exclusively historical split.train")
    config = StageEditTransformerConfig.from_dict(old_report["model_config"])
    if (config.model_dimension, config.num_layers, config.num_heads,
            config.feed_forward_dimension, config.max_sequence_length, config.max_sources) != (768, 14, 12, 3072, 2048, 32):
        raise ValueError("diagnostic must use the actual full production configuration")
    snapshots, songs, paths, artifacts, inputs = {}, {}, [], {}, []
    for record in plan["songs"]:
        song_id = record["training_song_id"]
        if song_id not in TRAIN_REFERENCES:
            continue
        ref = TRAIN_REFERENCES[song_id]
        snapshot_path = Path(record["feature_snapshot_paths_by_reference"][ref])
        snapshot, snapshot_hash = read_historical_input(snapshot_path,kind="snapshot",reference_id=ref,binding=binding)
        snapshots[ref] = StageEditFeatureSnapshot.from_dict(snapshot, schema=schema)
        songs[ref] = song_id
        path = Path(record["supervision_paths_by_reference"][ref])
        _, supervision_hash = read_historical_input(path,kind="supervision",reference_id=ref,binding=binding)
        paths.append(path)
        manifest_item = record["dynamic_crop_manifest"]
        manifest, _ = read_bound_json(manifest_item["path"], manifest_item["file_sha256"])
        artifacts[ref] = []
        for item in manifest["artifacts"]:
            if item["reference_id"] != ref or not item["eligible"]:
                continue
            bound = item["artifact"]
            _, digest = read_bound_json(bound["path"], bound["file_sha256"])
            artifacts[ref].append(Path(bound["path"]))
            inputs.append(dict(kind="crop_artifact", reference_id=ref, path=bound["path"], sha256=digest))
        inputs.extend([
            dict(kind="snapshot", reference_id=ref, path=str(snapshot_path), sha256=snapshot_hash),
            dict(kind="supervision", reference_id=ref, path=str(path), sha256=supervision_hash),
        ])
    corpus = load_reference_training_corpus(paths, schema=schema,
        feature_snapshot_by_reference=snapshots, song_id_by_reference=songs,
        negative_samples_per_hold=2, max_sources=32, production_edit_mode=EditMode.GROUP)
    # These hashes include original videos but never decode or run perception.
    v2 = build_stage_edit_dynamic_crop_training_corpus_v2(corpus.examples,
        artifact_paths_by_reference=artifacts, verify_input_files=True)
    v3 = build_stage_edit_dynamic_crop_training_corpus_v3(v2)
    inventory = _dynamic_crop_v3_prior_inventory(corpus.examples, v3)
    tracks = select_fixture(inventory.tracks)
    prepared = prepare_stage_edit_production_input_sequences(corpus.examples, schema=schema)
    sequences = {s.sequence_id: s for s in prepared}
    evidence = dict(diagnostic_only=True, production_activation_authorized=False,
        validation_forwards=0, heldout_forwards=0, original_inputs_sha256_verified=True,
        plan_sha256=plan_hash, config_report_sha256=report_hash,
        model_config=config.to_dict(), train_reference_ids=sorted(songs),
        data_id=v3.data_id, input_files=inputs,
        tracks=[dict(song_id=t.song_id, reference_id=t.reference_id,
            sequence_id=t.sequence_id, original_token_index=t.token_index,
            prefix_length=t.token_index+1, source_id=t.source_id,
            count=len(t.values), motion=t.motion_label, motion_masked=t.motion_label_loss_masked,
            normalized_times=t.normalized_times, values=t.values) for t in tracks])
    require_historical_fixture_identity(evidence,binding)
    evidence["historical_binding_sha256"] = binding_hash
    evidence["historical_binding_source_receipt_sha256"] = binding["source_receipt_sha256"]
    return corpus, v3, config, sequences, replace(inventory, tracks=tracks), evidence


class DiagnosticBudget:
    """Watch only our process. A hung native call cannot overrun silently."""
    seconds = 600
    rss_bytes = 5 * 2**30
    mlx_bytes = int(4.5 * 2**30)

    def __init__(self, out):
        import psutil
        self.out = out
        self.process = psutil.Process()
        # Include interpreter/import/argument-validation time, not just the
        # later model phase. The same clock also covers --prepare-only.
        self.start = time.monotonic() - max(0., time.time() - self.process.create_time())
        self.done = threading.Event()

    def check(self):
        elapsed = time.monotonic() - self.start
        # Never import/allocate an MLX runtime merely to watch preparation.
        # Once production imports it, include its peak without resetting away
        # any preparation-phase allocation. None means unmeasured, not zero.
        runtime = sys.modules.get("mlx.core")
        rss = self.process.memory_info().rss
        peak = None if runtime is None else int(runtime.get_peak_memory())
        if elapsed > self.seconds or rss > self.rss_bytes or (peak is not None and peak > self.mlx_bytes):
            raise RuntimeError(f"diagnostic budget exceeded: seconds={elapsed}, rss={rss}, mlx_peak={peak}")
        return dict(elapsed_seconds=elapsed, rss_bytes=rss, mlx_peak_bytes=peak)

    def watch(self):
        import os
        while not self.done.wait(.5):
            try:
                self.check()
            except Exception as exc:
                (self.out / "budget-stop.json").write_text(json.dumps(dict(
                    diagnostic_only=True, error=str(exc), own_pid=os.getpid())))
                os._exit(75)


def tree_norm(tree):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    terms = [mx.sum(x*x) for _, x in tree_flatten(tree)]
    result = mx.sqrt(sum(terms)) if terms else mx.array(0.)
    mx.eval(result)
    value = float(result.item())
    if not np.isfinite(value):
        raise RuntimeError("nonfinite gradient or parameter delta")
    return value


def gradient_evidence(gradients, *, adapter=False):
    groups = {}
    for name, start, stop in (("values",0,32),("count",32,40),("time",40,75),("motion",75,76)):
        selected = {}
        names = ("context_adapter", "source_adapter") if adapter else (
            "dynamic_crop_context_head", "dynamic_crop_source_head")
        for key in names:
            for leaf, values in gradients[key].items():
                selected[f"{key}.{leaf}"] = values[start:stop]
        groups[name] = tree_norm(selected)
    if not adapter:
        groups["encoder"] = tree_norm(gradients["base"]["encoder"])
        groups["source_projection"] = tree_norm(gradients["base"]["source_projection"])
    return groups


def run_diagnostic(fixture, out, *, budget):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from crosscut import stage_edit_production_training as production
    from crosscut import stage_edit_dynamic_crop_head_v3 as h3
    from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3
    from crosscut.stage_edit_decoded_curve_metrics import decoded_observation_error, summarize_decoded_observations
    from crosscut.stage_edit_transformer import (
        DEFAULT_STAGE_EDIT_SCHEMA as schema, StageEditLossContract,
        StageEditTargetSequence, tensorize_stage_edit_batch, tensorize_stage_edit_supervision,
        stage_edit_loss)

    corpus, v3, config, sequences, inventory, _ = fixture
    mx.set_cache_limit(128 * 2**20)
    mx.set_memory_limit(budget.mlx_bytes)
    budget.check()
    report = dict(diagnostic_only=True, production_activation_authorized=False,
        production_ready=False, validation_forwards=0, heldout_forwards=0,
        max_optimizer_updates=80, limits=dict(seconds=budget.seconds,
        rss_bytes=budget.rss_bytes, mlx_bytes=budget.mlx_bytes),
        native_affine_tolerance=dict(atol=2e-4, rtol=2e-5),
        model_config=config.to_dict(), steps=[], evaluations={})
    examples = {e.sequence.sequence_id: e for e in corpus.examples}
    tracks = inventory.tracks
    def write_report():
        (out / "diagnostic-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    def prefix_batch(track):
        sequence = sequences[track.sequence_id]
        return tensorize_stage_edit_batch((replace(sequence,
            tokens=sequence.tokens[:track.token_index+1]),), schema, config)

    def target_supervision(track, batch):
        original = examples[track.sequence_id].targets
        base = tensorize_stage_edit_supervision((StageEditTargetSequence(
            original.sequence_id, original.targets[:track.token_index+1]),), batch, schema)
        all_targets = v3.sequence_for(track.sequence_id).targets
        targets = (None,)*track.token_index + (all_targets[track.token_index],)
        dynamic = h3.tensorize_stage_edit_dynamic_crop_supervision_v3((
            h3.StageEditDynamicCropTargetSequenceV3(track.sequence_id, targets),), batch)
        return base, dynamic

    def raw_at(batch, track):
        hidden, _, source, mask = model.base._hidden_and_sources(batch)
        slot = batch.source_ids[0][track.token_index].index(track.source_id)
        mx.eval(mask)
        if not bool(mask[0,track.token_index,slot].item()):
            raise RuntimeError("diagnostic target source lost its native hard gate")
        # Use the actual Linear/addmm operators and tensor ranks of __call__.
        raw = model.dynamic_crop_context_head(hidden)[0,track.token_index] + model.dynamic_crop_source_head(
            source)[0,track.token_index,slot]
        mx.eval(raw)
        if not np.isfinite(np.asarray(raw)).all():
            raise RuntimeError("nonfinite raw76 output")
        return raw, slot

    def evaluate(name):
        model.eval()
        rows, dynamic_rows, outputs = [], [], []
        for track in tracks:
            budget.check()
            raw, slot = raw_at(prefix_batch(track), track)
            raw_np = np.asarray(raw)
            values = np.asarray(h3.renderer_safe_values_from_raw_v3(raw[:32].reshape(8,4)))
            prediction = DynamicCropHeadPredictionV3(track.source_id, slot,
                tuple(map(tuple,values.tolist())), tuple(raw_np[32:40].tolist()),
                tuple(raw_np[40:75].tolist()), float(raw_np[75]))
            target = v3.sequence_for(track.sequence_id).targets[track.token_index]
            decoded = prediction.decode(target.shot_duration_seconds)
            row = decoded_observation_error(decoded, target.original_observed_keyframes,
                expected_source_id=track.source_id)
            rows.append(row)
            if track.motion_label == "dynamic" and not track.motion_label_loss_masked:
                dynamic_rows.append(row)
            outputs.append(dict(source_id=track.source_id, original_token_index=track.token_index,
                predicted_count=decoded.predicted_keyframe_count,
                motion_logit=float(raw_np[75]), raw76=raw_np.tolist(),
                decoded_times=[point.time_seconds for point in decoded.track.keyframes]))
        metric = summarize_decoded_observations(rows, expected_tracks=len(rows), source_mismatches=0)
        metric["dynamic_subset"] = summarize_decoded_observations(dynamic_rows,
            expected_tracks=len(dynamic_rows), source_mismatches=0)
        metric["predictions"] = outputs
        report["evaluations"][name] = metric
        write_report()
        print(json.dumps(dict(phase=name, mae=metric["normalized_mae"], **budget.check())), flush=True)
        return metric

    def update(loss, gradients, optimizer, trained, phase, local_step, *, adapter=False):
        budget.check()
        evidence = gradient_evidence(gradients, adapter=adapter)
        keys = ("context_adapter","source_adapter") if adapter else (
            "dynamic_crop_context_head","dynamic_crop_source_head")
        before = {key:dict(getattr(trained,key).parameters()) for key in keys}
        gradients, norm = optim.clip_grad_norm(gradients, 1.)
        optimizer.update(trained, gradients)
        mx.eval(loss, norm, trained.parameters(), optimizer.state)
        deltas = {key:{leaf:values-before[key][leaf]
            for leaf,values in getattr(trained,key).parameters().items()} for key in keys}
        # Retention is measured again after its byte-for-byte restoration below.
        if not adapter:
            deltas["base"]={"encoder":{},"source_projection":{}}
        parameter_deltas=gradient_evidence(deltas,adapter=adapter)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise RuntimeError("nonfinite training loss")
        report["steps"].append(dict(phase=phase, phase_step=local_step,
            optimizer_step=int(optimizer.state["step"].item()), loss=loss_value,
            gradient_norms=evidence, head_parameter_delta_l2=parameter_deltas,
            head_delta_before_retention_restore=phase=="retention",
            learning_rate=float(optimizer.learning_rate.item()),
            **budget.check()))
        write_report()
        if local_step == 1 or local_step % 4 == 0:
            print(json.dumps(report["steps"][-1]), flush=True)

    try:
        mx.random.seed(1731)
        model = h3.StageEditTransformerDynamicCropV3(config)
        production._enable_dynamic_crop_v3_activation_checkpointing(model)
        mx.eval(model.parameters())
        report["parameter_count"] = h3.stage_edit_dynamic_crop_v3_parameter_count(config)
        base_contract = StageEditLossContract(transition_authority=config.transition_authority,
            transition_weight=0., source_objective="eligible_all_hybrid")
        crop_contract = h3.DynamicCropV3LossContract(base=base_contract)
        evaluate("initial")
        # Exactly eight genuine full-model updates for each production loss.
        for phase, contract in (("cut", production._cut_only_pretraining_loss_contract(base_contract)),
                ("source", production._source_only_pretraining_loss_contract(base_contract))):
            optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=.01)
            grad_fn = nn.value_and_grad(model, lambda m,b,s: stage_edit_loss(m,b,s,contract)[0])
            for step in range(1,9):
                track = tracks[(step-1)%len(tracks)]
                batch = prefix_batch(track)
                base, _ = target_supervision(track,batch)
                model.train()
                loss, gradients = grad_fn(model,batch,base)
                update(loss,gradients,optimizer,model,phase,step)
        del optimizer, grad_fn, gradients
        gc.collect(); mx.clear_cache()
        model.eval()
        before_base = production._parameter_tree_sha256(model.base.parameters())
        inventory = production._materialize_dynamic_crop_v3_prior_features(model, corpus.examples,
            inventory, schema=schema, config=config, device=mx.default_device())
        projections = production._dynamic_crop_v3_structured_prior_projections(config.model_dimension, seed=1731)
        projection_hash = production._parameter_tree_sha256(dict(context=projections[0],source=projections[1]))
        normalizer = production._dynamic_crop_v3_structured_prior_projection_normalizer(inventory,*projections)
        report["normalizer"] = dict(normalizer.evidence)
        balance = production._dynamic_crop_v3_prior_class_balance_evidence(inventory)
        report["class_balance"] = balance
        initial_head = dict(context_weight=model.dynamic_crop_context_head.weight,
            context_bias=model.dynamic_crop_context_head.bias, source_weight=model.dynamic_crop_source_head.weight,
            context_center=model.dynamic_crop_context_head.input_center,
            source_center=model.dynamic_crop_source_head.input_center)
        adapter = production._DynamicCropV3StructuredPriorAdapterModule(16)
        optimizer = optim.AdamW(learning_rate=.01,weight_decay=0.)
        sampler = production._HierarchicalDynamicCropV3PriorSampler(inventory.tracks,seed=1731)
        def prior_loss(module, track):
            raw = production._dynamic_crop_v3_structured_prior_raw(module,initial_head,*projections,
                normalizer,mx.array(track.context_embedding),mx.array(track.source_embedding))
            return production._dynamic_crop_v3_prior_loss_from_raw(raw,mx.array(track.values),
                mx.array(track.normalized_times),len(track.values),float(track.motion_label=="dynamic"),
                not track.motion_label_loss_masked,crop_contract,
                balance["count_class_weights_1_to_8"][len(track.values)-1],
                balance["motion_class_weights_static_dynamic"][int(track.motion_label=="dynamic")])[0]
        grad_fn = nn.value_and_grad(adapter,prior_loss)
        for step in range(1,33):
            track = sampler.sample()
            loss, gradients = grad_fn(adapter,track)
            update(loss,gradients,optimizer,adapter,"detached_prior",step,adapter=True)
        merged = production._dynamic_crop_v3_structured_prior_head_snapshot(adapter,initial_head,*projections,normalizer)
        merged_module=production._DynamicCropV3PriorHeadModule(
            merged["context_weight"],merged["context_bias"],merged["source_weight"],
            context_center=merged["context_center"],source_center=merged["source_center"])
        mx.savez(str(out/"prior-affine-state.npz"),
            **{f"initial_{k}":v for k,v in initial_head.items()},
            adapter_context_weight=adapter.context_adapter.weight,adapter_context_bias=adapter.context_adapter.bias,
            adapter_source_weight=adapter.source_adapter.weight,
            context_projection=projections[0],source_projection=projections[1],
            context_mean=normalizer.context_mean,context_scale=normalizer.context_scale,
            source_mean=normalizer.source_mean,source_scale=normalizer.source_scale,
            contexts=mx.array(np.stack([t.context_embedding for t in inventory.tracks])),
            sources=mx.array(np.stack([t.source_embedding for t in inventory.tracks])))
        max_error = 0.
        for track in inventory.tracks:
            context, source = mx.array(track.context_embedding),mx.array(track.source_embedding)
            expected = production._dynamic_crop_v3_structured_prior_raw(adapter,initial_head,*projections,
                normalizer,context,source)
            actual = merged_module(context,source)
            mx.eval(expected,actual)
            np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),atol=2e-4,rtol=2e-5)
            max_error=max(max_error,float(mx.max(mx.abs(actual-expected)).item()))
        if before_base != production._parameter_tree_sha256(model.base.parameters()):
            raise RuntimeError("detached prior changed the base model")
        if projection_hash != production._parameter_tree_sha256(dict(context=projections[0],source=projections[1])):
            raise RuntimeError("fixed detached projections changed")
        report["detached_prior"] = dict(base_unchanged=True, projections_unchanged=True,
            base_sha256=before_base, projection_sha256=projection_hash, merge_max_raw76_error=max_error,
            per_song_sampled_counts=sampler.per_song_sampled_counts)
        production._assign_dynamic_crop_v3_prior_head_snapshot(model,merged)
        evaluate("after_prior")
        del optimizer, grad_fn, adapter, gradients, initial_head, merged
        gc.collect(); mx.clear_cache()
        features = {k:v for k,v in model.base.parameters().items() if k in production.DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES}
        feature_hash = production._parameter_tree_sha256(features)
        head = dict(context_weight=model.dynamic_crop_context_head.weight,
            context_bias=model.dynamic_crop_context_head.bias,source_weight=model.dynamic_crop_source_head.weight,
            context_center=model.dynamic_crop_context_head.input_center,
            source_center=model.dynamic_crop_source_head.input_center)
        head_hash = production._parameter_tree_sha256(head)
        sentinel_batch=prefix_batch(tracks[-1])
        raw_before=np.array(raw_at(sentinel_batch,tracks[-1])[0])
        schedule=production._warmup_cosine_schedule(peak_learning_rate=1e-5,total_steps=32,warmup_steps=2,
            minimum_learning_rate_ratio=.1)
        optimizer=optim.AdamW(learning_rate=schedule,weight_decay=.01)
        grad_fn=nn.value_and_grad(model,lambda m,b,s,d: h3.stage_edit_dynamic_crop_v3_loss(m,b,s,d,crop_contract)[0])
        reset_count=0
        for step in range(1,33):
            budget.check()
            if step==5:
                if int(optimizer.state["step"].item())!=4:
                    raise RuntimeError("retention optimizer did not execute exactly four updates")
                optimizer=optim.AdamW(learning_rate=production._offset_learning_rate_schedule(schedule,4),weight_decay=.01)
                if int(optimizer.state["step"].item())!=0:
                    raise RuntimeError("release did not start a fresh optimizer")
                reset_count+=1
            track=tracks[(step-1)%len(tracks)]
            batch=prefix_batch(track); base,dynamic=target_supervision(track,batch)
            model.train()
            if step<=4:
                for name in (*production.DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES,"input_dropout"):
                    getattr(model.base,name).eval()
            loss,gradients=grad_fn(model,batch,base,dynamic)
            if step<=4:
                gradients=dict(gradients)
                for name in ("dynamic_crop_context_head","dynamic_crop_source_head"):
                    gradients[name]=production._zero_gradient_tree(gradients[name])
                gradients["base"]=dict(gradients["base"])
                for name in production.DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES:
                    gradients["base"][name]=production._zero_gradient_tree(gradients["base"][name])
            update(loss,gradients,optimizer,model,"retention" if step<=4 else "released_joint",step if step<=4 else step-4)
            if step<=4:
                model.base.update(features)
                production._assign_dynamic_crop_v3_prior_head_snapshot(model,head)
                if step==4:
                    if feature_hash != production._parameter_tree_sha256({k:model.base.parameters()[k] for k in features}):
                        raise RuntimeError("feature parameters drifted during retention")
                    model.eval()
                    np.testing.assert_array_equal(raw_before,np.asarray(raw_at(sentinel_batch,tracks[-1])[0]))
                    report["retention"] = dict(feature_sha256=feature_hash,head_sha256=head_hash,
                        output_bit_identical=True,restore_steps=4)
        report["release"] = dict(reset_count=reset_count, schedule_offset_steps=4,
            final_optimizer_step=int(optimizer.state["step"].item()),
            features_changed=feature_hash != production._parameter_tree_sha256(
                {k:model.base.parameters()[k] for k in features}),
            heads_changed=head_hash != production._parameter_tree_sha256(dict(
                context_weight=model.dynamic_crop_context_head.weight,
                context_bias=model.dynamic_crop_context_head.bias,source_weight=model.dynamic_crop_source_head.weight,
                context_center=model.dynamic_crop_context_head.input_center,
                source_center=model.dynamic_crop_source_head.input_center)))
        if reset_count!=1 or report["release"]["final_optimizer_step"]!=28 or not all(
                report["release"][key] for key in ("features_changed","heads_changed")):
            raise RuntimeError("genuine post-retention joint release failed")
        final=evaluate("after_joint")
        del optimizer,gradients,grad_fn,features,head
        gc.collect(); mx.clear_cache()
        # Production weights are never registered. Diagnostic save/reload only.
        metadata=h3.new_stage_edit_dynamic_crop_checkpoint_metadata_v3(out.name,schema,config,
            training_steps=80,weights_trained=True,training_data_id=v3.data_id)
        h3.save_stage_edit_dynamic_crop_checkpoint_v3(out/"unqualified-checkpoint",model,metadata)
        model.eval(); expected=np.asarray(raw_at(sentinel_batch,tracks[-1])[0]).copy()
        del model; gc.collect(); mx.clear_cache()
        model,_=h3.load_stage_edit_dynamic_crop_checkpoint_v3(out/"unqualified-checkpoint",expected_schema=schema)
        model.eval(); actual=np.asarray(raw_at(sentinel_batch,tracks[-1])[0])
        np.testing.assert_array_equal(expected,actual)
        report["reload_raw76_bit_identical"]=True
        sequence=sequences[tracks[-1].sequence_id]
        permuted=replace(sequence,tokens=tuple(replace(t,source_candidates=tuple(reversed(t.source_candidates)))
            for t in sequence.tokens[:tracks[-1].token_index+1]))
        permutation=tensorize_stage_edit_batch((permuted,),schema,config)
        permuted_raw=np.asarray(raw_at(permutation,tracks[-1])[0])
        np.testing.assert_allclose(actual,permuted_raw,atol=2e-4,rtol=2e-5)
        if bool(np.asarray(sentinel_batch.source_candidate_mask[:,:,-1]).any()):
            raise RuntimeError("masked-source probe has no genuinely masked padding slot")
        perturbed=replace(sentinel_batch,source_features=sentinel_batch.source_features.at[:,:,-1,:].set(12345.),
            temporal_source_features=sentinel_batch.temporal_source_features.at[:,:,-1,:].set(12345.))
        isolated=np.asarray(raw_at(perturbed,tracks[-1])[0])
        np.testing.assert_array_equal(actual,isolated)
        report["source_invariance"]=dict(permutation_max_raw76_error=float(np.max(np.abs(actual-permuted_raw))),
            masked_source_raw76_bit_identical=True)
        released=[row for row in report["steps"] if row["phase"]=="released_joint"]
        prior=[row for row in report["steps"] if row["phase"]=="detached_prior"]
        for phase_rows,groups in ((prior,("values","count","time","motion")),
                (released,("values","count","time","motion","encoder","source_projection"))):
            for group in groups:
                if not any(row["gradient_norms"][group]>0 for row in phase_rows):
                    raise RuntimeError(f"persistent zero gradient in supervised group: {group}")
            for group in ("values","count","time","motion"):
                if not any(row["head_parameter_delta_l2"][group]>0 for row in phase_rows):
                    raise RuntimeError(f"supervised head parameters never updated: {group}")
        initial=report["evaluations"]["initial"]["normalized_mae"]
        reduction=(initial-final["normalized_mae"])/max(initial,1e-12)
        count_classes={p["predicted_count"] for p in final["predictions"]}
        motion_classes={p["motion_logit"]>=0 for p in final["predictions"]}
        report["train_fit"]=dict(relative_curve_error_reduction=reduction,
            dynamic_beats_static=final["dynamic_subset"]["improvement_over_static_prior"]>0,
            predicted_count_classes=sorted(count_classes),predicted_motion_classes=sorted(motion_classes))
        report["status"]="diagnostic_pass" if reduction>=.2 and report["train_fit"]["dynamic_beats_static"] \
            and len(count_classes)>1 and len(motion_classes)>1 else "stop_investigate_no_long_rerun"
        report["resources"]=budget.check()
        report["optimizer_update_count"]=len(report["steps"])
        if report["optimizer_update_count"]!=80:
            raise RuntimeError("diagnostic did not execute exactly the predeclared 80 updates")
        write_report()
        print(json.dumps(dict(status=report["status"],train_fit=report["train_fit"],**report["resources"])),flush=True)
    except Exception as exc:
        report["status"]="diagnostic_failed"
        report["error"]=f"{type(exc).__name__}: {exc}"
        write_report()
        raise


def execute_diagnostic(out, *, prepare_only=False):
    """One process-age budget/watchdog around preparation AND execution."""
    budget = DiagnosticBudget(out)
    watcher = threading.Thread(target=budget.watch, daemon=True)
    watcher.start()
    try:
        budget.check()
        fixture = prepare_fixture()
        resources = budget.check()
        (out / "fixture.json").write_text(json.dumps(fixture[-1], indent=2))
        preparation = dict(status="fixture_verified", diagnostic_only=True,
            production_activation_authorized=False, prepare_only=prepare_only,
            tracks=len(fixture[-2].tracks), optimizer_updates=0,
            watchdog_started_before_prepare=True, clock_origin="process_creation",
            limits=dict(seconds=budget.seconds,rss_bytes=budget.rss_bytes,mlx_bytes=budget.mlx_bytes),
            **resources)
        (out / "preparation-report.json").write_text(json.dumps(preparation,indent=2))
        print(json.dumps(preparation), flush=True)
        if not prepare_only:
            run_diagnostic(fixture,out,budget=budget)
    except Exception as exc:
        try:
            binding_digest=hashlib.sha256(HISTORICAL_BINDING.read_bytes()).hexdigest()
        except OSError:
            binding_digest=None
        (out / "failure.json").write_text(json.dumps(dict(diagnostic_only=True,
            production_activation_authorized=False, error=f"{type(exc).__name__}: {exc}",
            historical_binding_sha256=binding_digest,
            elapsed_seconds=time.monotonic()-budget.start), indent=2))
        raise
    finally:
        budget.done.set()
        watcher.join(timeout=1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path("/Volumes/CrossCutMedia/CrossCut-Training/diagnostics")
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(root):
        parser.error("output must be a new project diagnostics directory on SSD")
    if root.parent.stat().st_dev != args.device:
        parser.error("authorized SSD device does not match")
    args.out.mkdir(parents=True, exist_ok=False)
    execute_diagnostic(args.out,prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
