import json
from pathlib import Path

import pytest
from dataclasses import asdict

from train_stageedit_face_corpus import training_arguments, checkpoint_layout, checkpoint_working_space
from crosscut.stage_edit_face_cues import schema_with_face_cues
from crosscut.stage_edit_transformer import production_stage_edit_config
from crosscut.stage_edit_transformer import StageEditLossContract
from crosscut.stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract


def test_face_run_keeps_full_capacity_objectives_and_holdout_partition():
    old = production_stage_edit_config()
    args = ['--held-out-song','unseen', '--required-training-song','seen',
        '--feature-snapshot','ref=old.json','--checkpoint-dir','old-checkpoint',
        '--model-config-json',json.dumps(old.to_dict()), '--dynamic-crop-v3-prior-pretrain-steps','500']
    original = list(args)
    schema = schema_with_face_cues()
    result, new = training_arguments(args, dict(feature_snapshot_by_reference={'ref':'new.json'}), schema, Path('/new'))
    assert args == original
    assert new.model_dimension == old.model_dimension and new.num_layers == old.num_layers
    assert new.num_heads == old.num_heads and new.max_sequence_length == old.max_sequence_length
    assert new.max_sources == old.max_sources and new.temporal_input_mode == old.temporal_input_mode
    assert new.source_input_dimension == old.source_input_dimension+12
    assert new.token_schema_fingerprint == schema.fingerprint
    assert result[:4] == original[:4]
    assert result[result.index('--dynamic-crop-v3-prior-pretrain-steps')+1] == '500'
    assert result[result.index('--checkpoint-dir')+1] == '/new/checkpoint'
    assert result[result.index('--feature-snapshot')+1] == 'ref=new.json'


@pytest.mark.parametrize('snapshots', [{}, {'other':'new.json'}, {'ref':'new.json','other':'more.json'}])
def test_face_run_cannot_change_the_frozen_reference_set(snapshots):
    args = ['--feature-snapshot','ref=old.json','--model-config-json',json.dumps(production_stage_edit_config().to_dict())]
    with pytest.raises(ValueError):
        training_arguments(args, dict(feature_snapshot_by_reference=snapshots), schema_with_face_cues(), Path('/new'))


def test_new_played_curve_objective_requires_an_explicit_override_and_preserves_the_base():
    base = StageEditLossContract()
    old_loss = asdict(DynamicCropV3LossContract(base=base))
    old_loss.pop('decoded_curve_weight')
    old_loss['contract'] = 'renderer_range_normalized_values_count_times_motion_loss_v3'
    args = ['--feature-snapshot', 'ref=old.json', '--model-config-json',
        json.dumps(production_stage_edit_config().to_dict()), '--dynamic-crop-version', '3',
        '--base-loss-contract-json', json.dumps(asdict(base)),
        '--dynamic-crop-loss-contract-json', json.dumps(old_loss)]
    original = list(args)
    prepared = dict(feature_snapshot_by_reference={'ref': 'new.json'})
    unchanged, _ = training_arguments(args, prepared, schema_with_face_cues(), Path('/new'))
    flag = '--dynamic-crop-loss-contract-json'
    assert json.loads(unchanged[unchanged.index(flag)+1]) == old_loss
    override = asdict(DynamicCropV3LossContract(base=base))
    changed, _ = training_arguments(args, prepared, schema_with_face_cues(), Path('/new'),
        dynamic_loss_override=override)
    assert json.loads(changed[changed.index(flag)+1]) == override
    assert args == original
    invalid = dict(override, base=asdict(StageEditLossContract(source_listwise_weight=.3)))
    with pytest.raises(ValueError, match='frozen base'):
        training_arguments(args, prepared, schema_with_face_cues(), Path('/new'), dynamic_loss_override=invalid)
    with pytest.raises(ValueError, match='explicit v3'):
        training_arguments([('2' if item == '3' else item) for item in args], prepared,
            schema_with_face_cues(), Path('/new'), dynamic_loss_override=override)


@pytest.mark.parametrize('version',[2,3])
def test_checkpoint_space_is_full_float32_weights_plus_metadata_and_explicit_reserve(version):
    config=production_stage_edit_config(schema_with_face_cues())
    count,name=checkpoint_layout(config,version)
    assert 'v'+str(version) in name
    initial=checkpoint_working_space(config,version)
    assert initial==2**30+64*2**20+count*4
    assert checkpoint_working_space(config,version,written_weight_bytes=1000)==initial-1000
    assert checkpoint_working_space(config,version,written_weight_bytes=count*4)==2**30+64*2**20
    assert checkpoint_working_space(config,version,reserve_bytes=256*2**20)==initial-768*2**20
    assert checkpoint_working_space(config,version,followup_payload_bytes=700_000_000)==initial+700_000_000
    with pytest.raises(ValueError):checkpoint_working_space(config,version,followup_payload_bytes=-1)


@pytest.mark.parametrize('reserve,written',[(0,0),(255*2**20,0),(True,0),(2**30,-1),(2**30,True)])
def test_checkpoint_space_cannot_erase_reserve_or_invent_progress(reserve,written):
    with pytest.raises(ValueError):
        checkpoint_working_space(production_stage_edit_config(),reserve_bytes=reserve,written_weight_bytes=written)
