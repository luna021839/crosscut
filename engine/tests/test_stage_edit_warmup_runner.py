"""Small numeric tests of the reusable warmup; not real-video quality proof."""
from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from train_stage_edit_warmup import run_phase, restore_cached_warmup, PROTOCOL, sha256
from tests.test_stage_edit_production_training import _training_example
from crosscut.stage_edit_reference_adapter import ReferenceTrainingCorpus
from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3, stage_edit_dynamic_crop_v3_parameter_count
from crosscut.stage_edit_production_training import _parameter_tree_sha256, DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES
from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA, StageEditTransformerConfig


@pytest.mark.parametrize("phase", ("cut", "source"))
def test_real_optimizer_updates_and_all_train_references_are_counted(phase):
    examples = tuple(_training_example(song, reference)
                     for song, reference in (("A", "a"), ("B", "b")))
    corpus = ReferenceTrainingCorpus(examples=examples, data_id="numeric-fixture", source_paths=(), statistics={})
    config = StageEditTransformerConfig.from_schema(
        DEFAULT_STAGE_EDIT_SCHEMA, model_dimension=16, num_heads=4, num_layers=1,
        feed_forward_dimension=32, max_sequence_length=8, max_sources=2, dropout=0.,
    )
    model = StageEditTransformerDynamicCropV3(config)
    before = _parameter_tree_sha256(model.parameters())
    features = lambda: {k:v for k,v in model.base.parameters().items() if k in DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES}
    features_before = _parameter_tree_sha256(features())
    records = []
    result = run_phase(model, corpus, config, phase=phase, steps=2,
                       budget=SimpleNamespace(check=lambda: {}), completed=records.append)
    assert result["optimizer_steps"] == 2
    assert _parameter_tree_sha256(model.parameters()) != before
    assert _parameter_tree_sha256(features()) == features_before
    assert result['classifier_warmup']['all_nonclassifier_parameters_byte_identical']
    assert result['classifier_warmup']['full_joint_trainability_restored']
    from mlx.utils import tree_flatten
    assert {k for k,_ in tree_flatten(model.trainable_parameters())} == {k for k,_ in tree_flatten(model.parameters())}
    assert [row["step"] for row in records] == [1, 2]
    assert all(row["phase"] == phase for row in records)
    sampling = result["sampling"]
    counts = sampling["positive"] if phase == "cut" else sampling
    assert counts["per_reference_sampled_counts"] == {"a": 1, "b": 1}


def test_cached_parameter_tree_restores_exactly_and_foreign_or_partial_cache_is_rejected(tmp_path):
    config = StageEditTransformerConfig.from_schema(
        DEFAULT_STAGE_EDIT_SCHEMA, model_dimension=16, num_heads=4, num_layers=1,
        feed_forward_dimension=32, max_sequence_length=8, max_sources=2, dropout=0.,
    )
    model = StageEditTransformerDynamicCropV3(config)
    examples=tuple(_training_example(song, reference) for song,reference in (("A","a"),("B","b")))
    corpus=ReferenceTrainingCorpus(examples=examples,data_id='numeric-cache-fixture',source_paths=(),statistics={})
    warmup={phase:run_phase(model,corpus,config,phase=phase,steps=2,
        budget=SimpleNamespace(check=lambda:{}),completed=lambda _:None) for phase in ('cut','source')}
    weights = tmp_path / 'after-source.safetensors'
    model.save_weights(str(weights))
    runner = tmp_path / 'warmup-runner-source.py'
    runner.write_text('# numeric cache fixture, not real training evidence\n')
    binding = dict(train_reference_ids=['a', 'b'], model_config=config.to_dict())
    report = dict(status='train_only_warmup_cached_not_qualified', protocol=PROTOCOL,
                  validation_forwards=0, heldout_forwards=0, production_activation_authorized=False,
                  fixture=binding, parameter_count=stage_edit_dynamic_crop_v3_parameter_count(config),
                  runner_sha256=sha256(runner), stages=dict(cut=dict(warmup['cut'],optimizer_steps=500), source=dict(
                      warmup['source'],optimizer_steps=500, weights=str(weights), weight_sha256=sha256(weights),
                      parameter_tree_sha256=_parameter_tree_sha256(model.parameters()))))
    path = tmp_path / 'warmup-report.json'
    path.write_text(json.dumps(report))
    restored = StageEditTransformerDynamicCropV3(config)
    receipt = restore_cached_warmup(restored, tmp_path, binding)
    assert _parameter_tree_sha256(restored.parameters()) == _parameter_tree_sha256(model.parameters())
    assert receipt['warmup_optimizer_steps_repeated'] == 0
    assert not receipt['optimizer_moments_restored']
    mutations = (
        lambda value: value.update(status='training_source'),
        lambda value: value.update(heldout_forwards=1),
        lambda value: value['fixture'].update(train_reference_ids=['validation']),
        lambda value: value['stages']['source'].update(optimizer_steps=499),
        lambda value: value['stages']['source'].update(weight_sha256='0' * 64),
        lambda value: value['stages']['source'].update(parameter_tree_sha256='0' * 64),
        lambda value: value['protocol'].update(contract='full_production_shape_train_only_balanced_cut_source_warmup_v1'),
        lambda value: value['stages']['source'].pop('classifier_warmup'),
        lambda value: value['stages']['cut']['classifier_warmup'].update(encoder_updated_during_classifier_warmup=True),
        lambda value: value['stages']['source']['classifier_warmup'].update(frozen_parameter_sha256_after='0'*64),
        lambda value: value['stages']['source']['classifier_warmup'].update(full_joint_trainability_restored=False),
    )
    for mutate in mutations:
        changed = copy.deepcopy(report)
        mutate(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            restore_cached_warmup(restored, tmp_path, binding)


def test_failed_callback_does_not_leave_the_model_frozen_for_joint_training():
    from mlx.utils import tree_flatten
    examples=tuple(_training_example(song, reference) for song,reference in (("A","a"),("B","b")))
    corpus=ReferenceTrainingCorpus(examples=examples,data_id='numeric-failure-fixture',source_paths=(),statistics={})
    config=StageEditTransformerConfig.from_schema(DEFAULT_STAGE_EDIT_SCHEMA,model_dimension=16,num_heads=4,
        num_layers=1,feed_forward_dimension=32,max_sequence_length=8,max_sources=2,dropout=0.)
    model=StageEditTransformerDynamicCropV3(config)
    def fail(_):raise RuntimeError('injected diagnostic callback failure')
    with pytest.raises(RuntimeError,match='injected diagnostic callback failure'):
        run_phase(model,corpus,config,phase='source',steps=2,budget=SimpleNamespace(check=lambda:{}),completed=fail)
    assert {k for k,_ in tree_flatten(model.trainable_parameters())}=={k for k,_ in tree_flatten(model.parameters())}
