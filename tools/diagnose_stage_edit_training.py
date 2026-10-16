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
        if not adapter and "dynamic_crop_interaction_head" in gradients:
            names += ("dynamic_crop_interaction_head",)
        for key in names:
            for leaf, values in gradients[key].items():
                selected[f"{key}.{leaf}"] = values[start:stop]
        groups[name] = tree_norm(selected)
    if not adapter:
        groups["encoder"] = tree_norm(gradients["base"]["encoder"])
        groups["source_projection"] = tree_norm(gradients["base"]["source_projection"])
    return groups


def run_diagnostic(fixture, out, *, budget, interaction_release_ab=False):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from crosscut import stage_edit_production_training as production
    from crosscut import stage_edit_dynamic_crop_head_v3 as h3
    from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3
    from crosscut.stage_edit_decoded_curve_metrics import decoded_observation_error, summarize_decoded_observations
    from crosscut.stage_edit_adoption_checks import native_source_raw,adopt_and_verify,verify_public_outputs
    from crosscut.stage_edit_gradient_checks import (separated_loss_gradients,head_row_norms,row_supervision,
        measured_feature_delta,require_actual_feature_update)
    from crosscut.stage_edit_train_fit_checks import label_fit_checks
    from crosscut.stage_edit_release_checks import (LIMITS as release_limits,capture_sources,
        compare_drift,snapshot_transaction,restore_transaction)
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
        release_step_drift_limits=dict(release_limits),
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
        raw,_,_,slot=native_source_raw(model,batch,track.token_index,track.source_id)
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
                teacher_forced_count_times=np.asarray(h3.dynamic_crop_v3_times_by_count(raw[40:75]))[len(track.values)-1,:len(track.values)].tolist(),
                decoded_normalized_times=[] if decoded.track is None else
                    [point.time_seconds/target.shot_duration_seconds for point in decoded.track.keyframes],
                decoded_times=[] if decoded.track is None else [point.time_seconds for point in decoded.track.keyframes],
                decoded_values=[] if decoded.track is None else
                    [[p.zoom,p.center_x,p.center_y,p.roll_degrees] for p in decoded.track.keyframes],
                decode_eligible=bool(decoded.eligible),renderer_projection_count=decoded.minimum_safe_zoom_projection_count))
        metric = summarize_decoded_observations(rows, expected_tracks=len(rows), source_mismatches=0)
        metric["dynamic_subset"] = summarize_decoded_observations(dynamic_rows,
            expected_tracks=len(dynamic_rows), source_mismatches=0)
        metric["predictions"] = outputs
        metric['label_and_decode_fit']=label_fit_checks(tracks,outputs)
        report["evaluations"][name] = metric
        write_report()
        print(json.dumps(dict(phase=name, mae=metric["normalized_mae"], **budget.check())), flush=True)
        return metric

    def update(loss, gradients, optimizer, trained, phase, local_step, *, adapter=False):
        from mlx.utils import tree_map
        budget.check()
        evidence = gradient_evidence(gradients, adapter=adapter)
        keys = ("context_adapter","source_adapter") if adapter else (
            "dynamic_crop_context_head","dynamic_crop_source_head")
        if not adapter and trained.dynamic_crop_config.uses_context_source_interaction:
            keys += ("dynamic_crop_interaction_head",)
        before = {key:dict(getattr(trained,key).parameters()) for key in keys}
        before_base=None if adapter else {name:tree_map(lambda value:value,trained.base.parameters()[name])
            for name in ('encoder','source_projection')}
        gradients, norm = optim.clip_grad_norm(gradients, 1.)
        report['optimizer_update_attempt_count']=report.get('optimizer_update_attempt_count',0)+1
        report['last_attempt']=dict(phase=phase,phase_step=local_step)
        write_report()
        optimizer.update(trained, gradients)
        mx.eval(loss, norm, trained.parameters(), optimizer.state)
        deltas = {key:{leaf:values-before[key][leaf]
            for leaf,values in getattr(trained,key).parameters().items()} for key in keys}
        # Retention is measured again after its byte-for-byte restoration below.
        if not adapter:
            deltas["base"]=measured_feature_delta(trained,before_base)
        parameter_deltas=gradient_evidence(deltas,adapter=adapter)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise RuntimeError("nonfinite training loss")
        report["steps"].append(dict(phase=phase, phase_step=local_step,
            optimizer_step=int(optimizer.state["step"].item()), loss=loss_value,
            gradient_norms=evidence, head_parameter_delta_l2=parameter_deltas,
            encoder_and_source_delta_measured=not adapter,
            head_delta_before_retention_restore=phase=="retention",
            learning_rate=float(optimizer.learning_rate.item()),
            **budget.check()))
        write_report()
        if phase=='released_joint':require_actual_feature_update(evidence,parameter_deltas)
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
        max_error = 0.;merge_rows=[];adoption_probes=[]
        model.eval()
        for track in inventory.tracks:
            budget.check()
            context, source = mx.array(track.context_embedding),mx.array(track.source_embedding)
            expected = production._dynamic_crop_v3_structured_prior_raw(adapter,initial_head,*projections,
                normalizer,context,source)
            actual = merged_module(context,source)
            mx.eval(expected,actual)
            max_error=max(max_error,float(mx.max(mx.abs(actual-expected)).item()))
            batch=prefix_batch(track)
            _,prefix_context,prefix_source,_=native_source_raw(model,batch,track.token_index,track.source_id)
            prefix_expected=production._dynamic_crop_v3_structured_prior_raw(adapter,initial_head,*projections,
                normalizer,prefix_context,prefix_source)
            mx.eval(prefix_expected)
            cached,prefix=np.asarray(expected),np.asarray(prefix_expected)
            merge_rows.append(dict(source_id=track.source_id,original_token_index=track.token_index,
                standalone_passed=bool(np.allclose(np.asarray(actual),cached,atol=2e-4,rtol=2e-5)),
                cached_to_prefix_passed=bool(np.allclose(prefix,cached,atol=2e-4,rtol=2e-5)),
                cached_to_prefix_raw76_max_error=float(np.max(np.abs(prefix-cached))),
                context_feature_max_error=float(np.max(np.abs(np.asarray(prefix_context)-track.context_embedding))),
                source_feature_max_error=float(np.max(np.abs(np.asarray(prefix_source)-track.source_embedding)))))
            target=v3.sequence_for(track.sequence_id).targets[track.token_index]
            adoption_probes.append(dict(batch=batch,token_index=track.token_index,source_id=track.source_id,
                expected_raw=np.array(prefix,copy=True),duration_seconds=target.shot_duration_seconds,
                observed_times=tuple(k.time_seconds for k in target.original_observed_keyframes)))
        report['native_affine_prefix_checks']=merge_rows
        write_report()
        if not all(r['standalone_passed'] and r['cached_to_prefix_passed'] for r in merge_rows):
            raise AssertionError('native affine cached/prefix comparison failed; complete rows saved')
        if before_base != production._parameter_tree_sha256(model.base.parameters()):
            raise RuntimeError("detached prior changed the base model")
        if projection_hash != production._parameter_tree_sha256(dict(context=projections[0],source=projections[1])):
            raise RuntimeError("fixed detached projections changed")
        report["detached_prior"] = dict(base_unchanged=True, projections_unchanged=True,
            base_sha256=before_base, projection_sha256=projection_hash, merge_max_raw76_error=max_error,
            per_song_sampled_counts=sampler.per_song_sampled_counts)
        def record_adoption(evidence):
            report['actual_public_adoption']=evidence;write_report()
        adopt_and_verify(model,merged,adoption_probes,check=budget.check,record=record_adoption)
        evaluate("after_prior")
        del optimizer, grad_fn, adapter, gradients, initial_head, merged, adoption_probes
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
        release_probes=[dict(batch=prefix_batch(t),token_index=t.token_index,source_id=t.source_id) for t in tracks]
        reset_count=0
        for step in range(1,5 if interaction_release_ab else 33):
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
            transaction=None;before_release=None
            if step in (5,6):
                model.eval()
                before_release=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                if step==5:
                    repeated=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                    repeat=compare_drift(before_release,repeated)
                    if any(row['raw76_step_drift']!=0 for row in repeat['tracks']):
                        raise AssertionError('eval/eval release repeat is not deterministic')
                    rng=[mx.array(np.array(k,copy=True)) for k in mx.random.state]
                    model.train()
                    stochastic=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                    mx.random.state=rng
                    report['release_dropout_no_update']=dict(eval_repeat=repeat,
                        dropout_only=compare_drift(before_release,stochastic),optimizer_updates=0,
                        stochastic_drift_is_advisory_not_serving_output=True)
                    write_report()
                transaction=snapshot_transaction(model,optimizer)
            model.train()
            if step==5:model.eval()  # first real joint update WITHOUT dropout
            if step<=4:
                for name in (*production.DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES,"input_dropout"):
                    getattr(model.base,name).eval()
            if step in (5,6):
                def attribution(evidence):
                    report.setdefault('released_loss_attribution',[]).append(dict(
                        phase_step=step-4,dropout_active=step==6,evidence=evidence));write_report()
                loss,gradients,_=separated_loss_gradients(model,batch,base,dynamic,crop_contract,
                    check=budget.check,record=attribution)
            else:loss,gradients=grad_fn(model,batch,base,dynamic)
            if step>=5:
                rows=head_row_norms(gradients);supported=row_supervision(len(track.values),motion_masked=track.motion_label_loss_masked)
                report.setdefault('released_row_gradients',[]).append(dict(phase_step=step-4,
                    count_target=len(track.values),supervised_rows=np.flatnonzero(supported).tolist(),
                    row_l2=rows.tolist(),unsupported_nonzero_rows=np.flatnonzero((~supported)&(rows>0)).tolist()))
                write_report()
                if np.any(rows[~supported]>0):raise AssertionError('gradient in unsupervised trajectory rows')
            if step<=4:
                gradients=dict(gradients)
                for name in ("dynamic_crop_context_head","dynamic_crop_source_head"):
                    gradients[name]=production._zero_gradient_tree(gradients[name])
                gradients["base"]=dict(gradients["base"])
                for name in production.DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES:
                    gradients["base"][name]=production._zero_gradient_tree(gradients["base"][name])
            update(loss,gradients,optimizer,model,"retention" if step<=4 else "released_joint",step if step<=4 else step-4)
            if transaction is not None:
                model.eval()
                after_release=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                drift=compare_drift(before_release,after_release)
                result=dict(phase_step=step-4,dropout_during_update=step==6,deterministic_serving_drift=drift,
                    rollback_performed=False)
                report.setdefault('release_step_probes',[]).append(result)
                if not drift['passed']:
                    restore_transaction(model,optimizer,transaction)
                    restored=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                    restored_drift=compare_drift(before_release,restored)
                    if any(r['raw76_step_drift']!=0 for r in restored_drift['tracks']):
                        raise AssertionError('serving outputs not restored after rollback')
                    result.update(rollback_performed=True,optimizer_state_restored=True,
                        restored_serving_raw76_bit_identical=True)
                    report['steps'][-1]['rolled_back']=True
                    write_report()
                    raise AssertionError('predeclared release step drift exceeded; rolled back and stopped')
                write_report();del transaction,before_release,after_release
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
        if interaction_release_ab:
            from mlx.utils import tree_map
            # Actual prior diagnostics now show first-update failure. Check
            # that prerequisite before spending the proposed 14+14 updates.
            # Both branches share all 52 updates, exact starting weights/RNG,
            # original losses, first released sample and unchanged stop gates.
            if len(report['steps']) != 52:
                raise AssertionError('interaction A/B requires exactly 52 common updates')
            model.eval()
            common = tree_map(lambda value:value,model.parameters())
            common_rng = [mx.array(np.array(key,copy=True)) for key in mx.random.state]
            expected = [np.array(raw_at(p['batch'],t)[0],copy=True)
                for p,t in zip(release_probes,tracks,strict=True)]
            branches=[]
            for interaction in (False,True):
                if interaction:
                    del model,optimizer,grad_fn,gradients
                    gc.collect();mx.clear_cache()
                    cfg=h3.DynamicCropHeadV3Config(source_conditioning=h3.DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING)
                    model=h3.StageEditTransformerDynamicCropV3(config,cfg)
                    production._enable_dynamic_crop_v3_activation_checkpointing(model)
                    model.update(common)
                    production._assign_dynamic_crop_v3_prior_head_snapshot(model,dict(
                        head,interaction_weight=mx.zeros_like(model.dynamic_crop_interaction_head.weight)))
                model.eval();mx.random.state=common_rng
                probes=[dict(p,expected_raw=r,duration_seconds=v3.sequence_for(t.sequence_id).targets[t.token_index].shot_duration_seconds)
                    for p,t,r in zip(release_probes,tracks,expected,strict=True)]
                adoption=verify_public_outputs(model,probes,check=budget.check)
                if not adoption['passed']:
                    raise AssertionError('A/B zero-init shared public starting point differs')
                initial=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                optimizer=optim.AdamW(learning_rate=production._offset_learning_rate_schedule(schedule,4),weight_decay=.01)
                transaction=snapshot_transaction(model,optimizer)
                # Original first released update is deterministic eval mode;
                # dropout is not silently made a second changed variable.
                track=tracks[4%len(tracks)];batch=prefix_batch(track);base,dynamic=target_supervision(track,batch)
                grad_fn=nn.value_and_grad(model,lambda m,b,s,d:h3.stage_edit_dynamic_crop_v3_loss(m,b,s,d,crop_contract)[0])
                loss,gradients=grad_fn(model,batch,base,dynamic)
                update(loss,gradients,optimizer,model,'released_joint',1)
                after=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                drift=compare_drift(initial,after)
                name='context_source_product' if interaction else 'legacy_affine'
                report['steps'][-1]['ab_branch']=name
                restore_transaction(model,optimizer,transaction)
                restored=capture_sources(model,release_probes,projections,normalizer,check=budget.check)
                rollback=compare_drift(initial,restored)
                if any(row['raw76_step_drift']!=0 for row in rollback['tracks']):
                    raise AssertionError('A/B diagnostic did not restore exact source outputs')
                branches.append(dict(name=name,source_conditioning=model.dynamic_crop_config.source_conditioning,
                    parameter_count=h3.stage_edit_dynamic_crop_v3_parameter_count(config,model.dynamic_crop_config),
                    zero_init_public_parity=adoption,first_update_drift=drift,
                    exact_parameter_optimizer_rng_rollback=True,restored_serving_raw76_bit_identical=True,
                    first_update_passed_existing_gate=drift['passed'],candidate_adopted=False))
                del transaction,initial,after,restored
            report.update(status='interaction_release_ab_complete_not_qualified',
                optimizer_update_count=len(report['steps']),interaction_release_ab=dict(
                    common_updates=52,branch_updates=1,max_total_updates=54,branches=branches,
                    original_14_plus_14_fit_completed=False,production_activation_authorized=False),resources=budget.check())
            if len(report['steps'])!=54:
                raise AssertionError('interaction release A/B exceeded its declared 54 updates')
            write_report()
            print(json.dumps(dict(status=report['status'],updates=len(report['steps']),
                branches=[dict(name=b['name'],passed=b['first_update_passed_existing_gate']) for b in branches])),flush=True)
            return
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
        reload_probes=[]
        for track in tracks:
            budget.check();batch=prefix_batch(track)
            target=v3.sequence_for(track.sequence_id).targets[track.token_index]
            reload_probes.append(dict(batch=batch,token_index=track.token_index,source_id=track.source_id,
                expected_raw=np.array(raw_at(batch,track)[0]),duration_seconds=target.shot_duration_seconds,
                observed_times=tuple(k.time_seconds for k in target.original_observed_keyframes)))
        del model; gc.collect(); mx.clear_cache()
        model,_=h3.load_stage_edit_dynamic_crop_checkpoint_v3(out/"unqualified-checkpoint",expected_schema=schema)
        model.eval(); actual=np.asarray(raw_at(sentinel_batch,tracks[-1])[0])
        np.testing.assert_array_equal(expected,actual)
        report["reload_raw76_bit_identical"]=True
        report['actual_public_reload']=verify_public_outputs(model,reload_probes,check=budget.check)
        for probe in reload_probes:
            budget.check()
            restored_raw,*_=native_source_raw(model,probe['batch'],probe['token_index'],probe['source_id'])
            np.testing.assert_array_equal(np.asarray(restored_raw),probe['expected_raw'])
        report['reload_all_raw76_bit_identical']=True
        write_report()
        if not report['actual_public_reload']['passed']:
            raise AssertionError('full public-output reload failed')
        sequence=sequences[tracks[-1].sequence_id]
        permuted=replace(sequence,tokens=tuple(replace(t,source_candidates=tuple(reversed(t.source_candidates)))
            for t in sequence.tokens[:tracks[-1].token_index+1]))
        permutation=tensorize_stage_edit_batch((permuted,),schema,config)
        permuted_raw=np.asarray(raw_at(permutation,tracks[-1])[0])
        np.testing.assert_allclose(actual,permuted_raw,atol=2e-4,rtol=2e-5)
        if bool(np.asarray(sentinel_batch.source_candidate_mask[:,:,-1]).any()):
            raise RuntimeError("masked-source probe has no genuinely masked padding slot")
        padding=~sentinel_batch.source_candidate_mask
        def perturb(batch,mask):
            return replace(batch,source_features=mx.where(mask[...,None],12345.,batch.source_features),
                temporal_source_features=None if batch.temporal_source_features is None else
                    mx.where(mask[...,None],12345.,batch.temporal_source_features))
        perturbed=perturb(sentinel_batch,padding)
        isolated=np.asarray(raw_at(perturbed,tracks[-1])[0])
        np.testing.assert_array_equal(actual,isolated)
        gated=sentinel_batch.source_candidate_mask & ~sentinel_batch.hard_source_gate_mask
        if not bool(mx.any(gated).item()):
            # Real present distractor, explicitly hard-gated in BOTH controls.
            # Do not compare against the original differently eligible set.
            mask=np.zeros(sentinel_batch.source_candidate_mask.shape,bool)
            for token in range(mask.shape[1]):
                choices=[i for i,s in enumerate(sentinel_batch.source_ids[0][token])
                    if s is not None and s!=tracks[-1].source_id and bool(sentinel_batch.source_candidate_mask[0,token,i].item())]
                if choices:mask[0,token,choices[0]]=True
            gated=mx.array(mask)
            sentinel_control=replace(sentinel_batch,hard_source_gate_mask=sentinel_batch.hard_source_gate_mask & ~gated)
        else:sentinel_control=sentinel_batch
        if not bool(mx.any(gated).item()):
            raise AssertionError('no present distractor for hard-gated-source probe')
        gated_before=np.asarray(raw_at(sentinel_control,tracks[-1])[0])
        gated_after=np.asarray(raw_at(perturb(sentinel_control,gated),tracks[-1])[0])
        np.testing.assert_array_equal(gated_before,gated_after)
        report["source_invariance"]=dict(permutation_max_raw76_error=float(np.max(np.abs(actual-permuted_raw))),
            masked_source_raw76_bit_identical=True,present_hard_gated_source_raw76_bit_identical=True)
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
            and len(count_classes)>1 and len(motion_classes)>1 and final['label_and_decode_fit']['passed'] \
            else "stop_investigate_no_long_rerun"
        report["resources"]=budget.check()
        report["optimizer_update_count"]=len(report["steps"])
        if report["optimizer_update_count"]!=80:
            raise RuntimeError("diagnostic did not execute exactly the predeclared 80 updates")
        write_report()
        print(json.dumps(dict(status=report["status"],train_fit=report["train_fit"],**report["resources"])),flush=True)
    except Exception as exc:
        report["status"]="diagnostic_failed"
        report['optimizer_update_count']=len(report['steps'])
        report["error"]=f"{type(exc).__name__}: {exc}"
        write_report()
        raise


def execute_diagnostic(out, *, prepare_only=False, interaction_release_ab=False):
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
            if interaction_release_ab:
                run_diagnostic(fixture,out,budget=budget,interaction_release_ab=True)
            else:
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
    parser.add_argument("--interaction-release-ab", action="store_true",
        help="54-update train-only first-release prerequisite A/B; never qualifies or activates a model")
    args = parser.parse_args()
    root = Path("/Volumes/CrossCutMedia/CrossCut-Training/diagnostics")
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(root):
        parser.error("output must be a new project diagnostics directory on SSD")
    if root.parent.stat().st_dev != args.device:
        parser.error("authorized SSD device does not match")
    args.out.mkdir(parents=True, exist_ok=False)
    execute_diagnostic(args.out,prepare_only=args.prepare_only,interaction_release_ab=args.interaction_release_ab)


if __name__ == "__main__":
    main()
