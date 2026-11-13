from types import SimpleNamespace
from dataclasses import replace

import numpy as np

import pytest

from crosscut.stage_edit_production_training import (
    _DynamicCropV3PriorTrack, _HierarchicalDynamicCropV3PriorSampler,
    _dynamic_crop_v3_prior_class_balance_evidence,
)
from crosscut.stage_edit_training_preflight import (
    require_releasable_training_plan, require_trajectory_selection_coverage,
)


def track(song, reference, count, motion, *, masked=False):
    return _DynamicCropV3PriorTrack(song,reference,f'{song}:{reference}',0,'source',
        tuple((1.02+.02*i,.5,.5,0.) for i in range(count)),
        tuple(.8*i/max(1,count-1) for i in range(count)),motion,masked)


def test_prior_class_weights_match_actual_hierarchical_exposure_not_raw_counts():
    tracks=tuple([track('A','many',1,'static') for _ in range(100)]+[track('B','few',2,'dynamic')])
    evidence=_dynamic_crop_v3_prior_class_balance_evidence(SimpleNamespace(tracks=tracks))
    sampler=_HierarchicalDynamicCropV3PriorSampler(tracks,seed=17)
    mass=[0.,0.]
    for _ in range(200):
        sample=sampler.sample();kind=int(sample.motion_label=='dynamic')
        mass[kind]+=evidence['motion_class_weights_static_dynamic'][kind]
    assert mass[0]==pytest.approx(mass[1])
    assert sum(mass)/200==pytest.approx(1.)
    assert evidence['count_class_weights_1_to_8'][:2]==pytest.approx([1.,1.])


def test_sampler_weights_respect_unequal_reference_sizes_and_masked_motion():
    tracks=tuple([track('A','many',1,'static') for _ in range(100)]
        +[track('A','other',2,'dynamic') for _ in range(2)]
        +[track('B','masked',3,'dynamic',masked=True)])
    evidence=_dynamic_crop_v3_prior_class_balance_evidence(SimpleNamespace(tracks=tracks))
    assert evidence['count_sampler_mass_1_to_8'][:3]==pytest.approx([.25,.25,.5])
    assert evidence['motion_sampler_mass_static_dynamic']==pytest.approx([.25,.25])
    assert evidence['motion_class_weights_static_dynamic']==pytest.approx([1.,1.])
    mass=np.array(evidence['count_sampler_mass_1_to_8'])
    weight=np.array(evidence['count_class_weights_1_to_8'])
    assert mass@weight==pytest.approx(1.)
    assert (mass*weight)[:3]==pytest.approx([1/3]*3)
    assert evidence['validation_labels_consumed'] is evidence['heldout_labels_consumed'] is False


@pytest.mark.parametrize('kind', ['single_count','missing_static','missing_dynamic','masked_only','zero_range'])
def test_impossible_label_coverage_is_rejected_without_forward(kind):
    values=[track('A','a',1,'static'),track('B','b',2,'dynamic')]
    if kind=='single_count':values=[replace(t,values=t.values[:1]) for t in values]
    elif kind=='missing_static':values=[replace(t,motion_label='dynamic') for t in values]
    elif kind=='missing_dynamic':values=[replace(t,motion_label='static') for t in values]
    elif kind=='masked_only':values=[replace(t,motion_label_loss_masked=True) for t in values]
    else:values=[replace(t,values=(t.values[0],)*len(t.values)) for t in values]
    with pytest.raises(ValueError,match='data_contract_unsatisfiable'):
        require_trajectory_selection_coverage(values,partition='validation')


def test_coverage_preflight_rejects_opening_heldout_partition():
    with pytest.raises(ValueError,match='must not inspect heldout'):
        require_trajectory_selection_coverage([],partition='heldout')


def test_releasable_plan_reserves_real_joint_updates_after_epoch_rounding():
    kwargs=dict(epochs=30,batches_per_epoch=4,minimum_finetune_steps=120,
        checkpoint_burn_in_steps=120,prior_may_be_retained=True,minimum_joint_steps_after_release=1)
    with pytest.raises(ValueError,match='insufficient_release_budget'):
        require_releasable_training_plan(**kwargs)
    valid=require_releasable_training_plan(**dict(kwargs,epochs=31))
    assert valid['maximum_retention_steps']==120
    assert valid['minimum_stop_step_when_retained']==121
    assert valid['planned_steps']==124
    with pytest.raises(ValueError,match='insufficient_release_budget'):
        require_releasable_training_plan(**dict(kwargs,batches_per_epoch=120,epochs=1,checkpoint_burn_in_steps=60))


def _actual_training_args(tmp_path):
    from crosscut.stage_edit_reference_adapter import ReferenceTrainingCorpus
    from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA, StageEditTransformerConfig
    from tests.test_stage_edit_production_training import _training_example, _dynamic_crop_corpus_v3
    examples=tuple(_training_example(song,f'{song}-{i}') for song in ('BANG','BLACKHOLE','XOXZ') for i in range(2))
    corpus=ReferenceTrainingCorpus(examples=examples,data_id='a'*64,source_paths=(),
        statistics={'crop_coordinate_supervision_count':24,'transition_supervision_count':6})
    config=StageEditTransformerConfig.from_schema(DEFAULT_STAGE_EDIT_SCHEMA,model_dimension=16,
        num_heads=4,num_layers=1,feed_forward_dimension=32,max_sequence_length=8,max_sources=2,dropout=0.)
    return dict(corpus=corpus,checkpoint_directory=tmp_path/'never-created',held_out_song_ids=('XOXZ',),
        model_config=config,epochs=2,batch_size=2,cut_pretrain_steps=0,source_pretrain_steps=0,
        minimum_finetune_optimizer_steps=1,minimum_checkpoint_selection_finetune_steps=1,
        dynamic_crop_corpus=_dynamic_crop_corpus_v3(examples))


@pytest.mark.parametrize('partition',['train','validation'])
@pytest.mark.parametrize('kind',['single_count','missing_static','missing_dynamic','masked_only','zero_range','missing_original'])
def test_trainer_rejects_unsatisfiable_fixed_labels_before_any_optimizer(monkeypatch,tmp_path,partition,kind):
    import crosscut.stage_edit_production_training as trainer
    args=_actual_training_args(tmp_path)
    dynamic=args['dynamic_crop_corpus']
    split=trainer.production_song_split(args['corpus'],held_out_song_ids=args['held_out_song_ids'])
    chosen={example.sequence.sequence_id for example in getattr(split,partition)}
    def invalid(target):
        if target is None:return None
        if kind=='missing_original':return replace(target,original_observed_keyframes=())
        if kind=='single_count':return replace(target,keyframes=target.keyframes[:1],retained_original_indices=(0,))
        if kind=='missing_static':return replace(target,motion_label='dynamic')
        if kind=='missing_dynamic':return replace(target,motion_label='static')
        if kind=='masked_only':return replace(target,motion_label_loss_masked=True)
        first=target.keyframes[0]
        def fixed(point):return replace(point,zoom=first.zoom,center_x=first.center_x,
            center_y=first.center_y,roll_degrees=first.roll_degrees)
        return replace(target,keyframes=tuple(fixed(point) for point in target.keyframes),
            original_observed_keyframes=tuple(fixed(point) for point in target.original_observed_keyframes))
    args['dynamic_crop_corpus']=replace(dynamic,sequences={key:replace(sequence,targets=tuple(
        invalid(target) for target in sequence.targets)) if key in chosen else sequence
        for key,sequence in dynamic.sequences.items()})
    def forbidden(*args,**kwargs):raise AssertionError('optimizer must not be constructed')
    monkeypatch.setattr(trainer.optim,'AdamW',forbidden)
    with pytest.raises(ValueError,match=f'data_contract_unsatisfiable: partition={partition}'):
        trainer.train_production_stage_edit(**args)
    assert not args['checkpoint_directory'].exists()


def test_trainer_rejects_exhausted_retention_before_any_pretraining(monkeypatch,tmp_path):
    import crosscut.stage_edit_production_training as trainer
    args=_actual_training_args(tmp_path)
    args.update(epochs=120,batch_size=2,minimum_finetune_optimizer_steps=120,
        minimum_checkpoint_selection_finetune_steps=120,dynamic_crop_v3_prior_pretrain_steps=500)
    def forbidden(*args,**kwargs):raise AssertionError('optimizer must not be constructed')
    monkeypatch.setattr(trainer.optim,'AdamW',forbidden)
    with pytest.raises(ValueError,match='insufficient_release_budget'):
        trainer.train_production_stage_edit(**args)
    assert not args['checkpoint_directory'].exists()
