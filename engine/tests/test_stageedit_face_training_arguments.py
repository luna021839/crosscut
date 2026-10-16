import json
from pathlib import Path

import pytest

from train_stageedit_face_corpus import training_arguments
from crosscut.stage_edit_face_cues import schema_with_face_cues
from crosscut.stage_edit_transformer import production_stage_edit_config


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
