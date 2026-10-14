from dataclasses import replace

import numpy as np
import pytest

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_decoded_curve_metrics import decoded_observation_error
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3
from crosscut.stage_edit_dynamic_crop_head_v2 import DynamicCropTrajectoryTargetV2
from crosscut.stage_edit_dynamic_crop_target_v3 import simplify_dynamic_crop_trajectory_v3


OBSERVED=tuple(EditorialTransformKeyframe(t,z,.5,.5,0.) for t,z in ((0.,1.04),(.4,1.08),(.8,1.12)))


def prediction(*,time=.8,count=2,unused_zoom=1.04):
    logits=np.full(8,-10.);logits[count-1]=10.
    times=np.zeros(35);times[:2]=np.log([time,1-time])
    return DynamicCropHeadPredictionV3('source-A',0,
        ((1.04,.5,.5,0.),(1.12,.5,.5,0.))+((unused_zoom,.5,.5,0.),)*6,
        tuple(logits),tuple(times),1.)


def error(model):
    decoded=model.decode(1.)
    assert decoded.eligible
    return decoded_observation_error(decoded,OBSERVED,expected_source_id='source-A')


def test_actual_predicted_times_change_error_on_original_observed_clock():
    good=error(prediction());bad=error(prediction(time=.2))
    assert good['normalized_absolute_error'].mean()<1e-4
    assert bad['normalized_absolute_error'].mean()>.01


def test_predicted_count_one_is_static_despite_perfect_unused_value_rows():
    static=error(prediction(count=1));moving=error(prediction())
    assert np.ptp(static['predicted'][:,0])==0
    assert static['normalized_absolute_error'].mean()>moving['normalized_absolute_error'].mean()


def test_unapplied_rows_do_not_improve_or_change_actual_curve_score():
    first=error(prediction(unused_zoom=1.04));other=error(prediction(unused_zoom=1.18))
    assert np.array_equal(first['normalized_absolute_error'],other['normalized_absolute_error'])


def test_actual_decoder_source_mismatch_is_rejected():
    decoded=replace(prediction().decode(1.),source_id='source-B')
    with pytest.raises(ValueError,match='decoded source identity mismatch'):
        decoded_observation_error(decoded,OBSERVED,expected_source_id='source-A')


def test_selection_never_accepts_legacy_teacher_forced_only_score():
    from crosscut.stage_edit_production_training import _dynamic_crop_v3_validation_selection_evidence
    from tests.test_stage_edit_production_training import _passing_dynamic_v3_metrics
    legacy=_passing_dynamic_v3_metrics();legacy['trajectory'].pop('decoded_observed_time')
    rejected=_dynamic_crop_v3_validation_selection_evidence(legacy)
    assert not rejected['selectable']
    assert 'dynamic_crop_v3_validation_actual_decoded_observation_metrics_missing' in rejected['reasons']
    bad=_passing_dynamic_v3_metrics()
    bad['trajectory']['decoded_observed_time']['improvement_over_static_prior']=-.1
    bad['trajectory']['aggregate_normalized_improvement_over_static_prior']=.9
    assert not _dynamic_crop_v3_validation_selection_evidence(bad)['selectable']


def test_simplification_preserves_removed_real_observations_for_scoring():
    target=DynamicCropTrajectoryTargetV2(0,'source-A',1.,OBSERVED,'dynamic',False,'0'*64,'1'*64)
    simplified=simplify_dynamic_crop_trajectory_v3(target)
    assert len(simplified.keyframes)==2
    assert simplified.original_observed_keyframes==OBSERVED
    assert len(simplified.to_dict()['original_observed_keyframes'])==3


def test_actual_mlx_evaluator_scores_predicted_clock_not_teacher_forced_rows(monkeypatch):
    mx=pytest.importorskip('mlx.core')
    import crosscut.stage_edit_dynamic_crop_head_v3 as head
    from crosscut.stage_edit_dynamic_crop_head_v2 import StageEditDynamicCropTargetSequenceV2
    from crosscut.stage_edit_transformer import (smoke_stage_edit_config,tensorize_stage_edit_batch,
        tensorize_stage_edit_supervision)
    from tests.test_stage_edit_transformer import SCHEMA,_sequence,_targets
    sequence=_sequence();config=smoke_stage_edit_config(SCHEMA)
    batch=tensorize_stage_edit_batch((sequence,),SCHEMA,config,device=mx.cpu)
    base=tensorize_stage_edit_supervision((_targets(sequence.sequence_id),),batch,SCHEMA,device=mx.cpu)
    target=DynamicCropTrajectoryTargetV2(0,'C',1.,OBSERVED,'dynamic',False,'0'*64,'1'*64)
    dynamic=head.tensorize_stage_edit_dynamic_crop_supervision_v3(
        (StageEditDynamicCropTargetSequenceV2(sequence.sequence_id,(target,)+(None,)*(len(sequence.tokens)-1)),),
        batch,device=mx.cpu)
    model=head.StageEditTransformerDynamicCropV3(config);template=model(batch)
    calls=[]
    class CountedModel:
        dynamic_crop_config=model.dynamic_crop_config
        def eval(self):model.eval()
        def __call__(self,batch):calls.append(1);return model(batch)
    head.evaluate_stage_edit_dynamic_crop_v3(CountedModel(),batch,base,dynamic)
    assert len(calls)==1
    source=int(np.asarray(dynamic.target_source_indices)[0,0])
    assert len(dynamic.original_observed_keyframes[0,0,source])==3
    assert int(np.asarray(dynamic.keyframe_mask)[0,0,source].sum())==2
    def evaluate(*,time=.8,count=2,metadata=dynamic.original_observed_keyframes):
        fixture=prediction(time=time,count=count)
        values=np.array(template.dynamic_crop_values);values[0,0,source]=fixture.values
        counts=np.array(template.dynamic_crop_count_logits);counts[0,0,source]=fixture.count_logits
        times=np.array(template.dynamic_crop_time_interval_logits);times[0,0,source]=fixture.time_interval_logits
        output=replace(template,dynamic_crop_values=mx.array(values),dynamic_crop_count_logits=mx.array(counts),
            dynamic_crop_time_interval_logits=mx.array(times))
        class FixedOutput:
            dynamic_crop_config=model.dynamic_crop_config
            def eval(self):pass
            def __call__(self,batch):return output
        # Isolate evaluation from fitting: production decoder and renderer are
        # real; only the unrelated loss total and frozen output are controlled.
        monkeypatch.setattr(head,'stage_edit_dynamic_crop_v3_loss',
            lambda *args,**kwargs:(mx.array(0.),{'total':mx.array(0.)}))
        return head.evaluate_stage_edit_dynamic_crop_v3(FixedOutput(),batch,base,
            replace(dynamic,original_observed_keyframes=metadata))['trajectory']
    good=evaluate();bad=evaluate(time=.2);static=evaluate(count=1);missing=evaluate(metadata=None)
    assert good['aggregate_renderer_range_normalized_mae']<1e-4
    assert bad['aggregate_renderer_range_normalized_mae']>.01
    assert good['teacher_forced_value_diagnostics']['aggregate_renderer_range_normalized_mae']==pytest.approx(
        bad['teacher_forced_value_diagnostics']['aggregate_renderer_range_normalized_mae'])
    assert static['curve_delta_and_range']['zoom']['predicted_mean_range']==0.
    assert not missing['decoded_observed_time']['complete']
    assert missing['aggregate_renderer_range_normalized_mae'] is None
    assert good['decoded_observed_time']['observed_keyframe_count']==3
    import crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 as adapter
    decode=adapter.decode_dynamic_crop_model_source_v3
    monkeypatch.setattr(adapter,'decode_dynamic_crop_model_source_v3',
        lambda *args,**kwargs:replace(decode(*args,**kwargs),source_id='wrong-source'))
    wrong_source=evaluate()
    assert wrong_source['decoded_observed_time']['source_identity_mismatch_count']==1
    assert not wrong_source['decoded_observed_time']['complete']
    assert wrong_source['renderer_decode']['eligible_count']==0
