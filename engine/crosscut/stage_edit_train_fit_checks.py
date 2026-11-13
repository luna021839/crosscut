"""Train-only label/decoder diagnostics, never heldout or media acceptance."""
import numpy as np


def label_fit_checks(tracks,predictions):
    from .stage_edit_production_training import ProductionQualityThresholds
    limits=ProductionQualityThresholds()
    if not tracks or len(tracks)!=len(predictions):raise ValueError('complete train prediction census required')
    target_hist=np.zeros(8,int);pred_hist=np.zeros(8,int);confusion=np.zeros((2,2),int)
    correct=0;time_errors=[];matched_errors=[];target_ranges=[];pred_ranges=[]
    eligible=0;projection=0;source_mismatches=0
    for track,p in zip(tracks,predictions,strict=True):
        count=len(track.values);predicted=p['predicted_count']
        if count not in range(1,9) or predicted not in range(1,9):raise ValueError('invalid count diagnostic')
        source_mismatches+=int((track.source_id,track.token_index)!=(p['source_id'],p['original_token_index']))
        target_hist[count-1]+=1;pred_hist[predicted-1]+=1;correct+=count==predicted
        if not track.motion_label_loss_masked:
            confusion[int(track.motion_label=='dynamic'),int(p['motion_logit']>=0)]+=1
        expected=np.asarray(track.normalized_times,float)
        forced=np.asarray(p['teacher_forced_count_times'],float)
        if forced.shape!=expected.shape or not np.isfinite(forced).all():raise ValueError('missing train time measurement')
        time_errors.extend(np.abs(forced-expected).tolist())
        if count==predicted:
            actual=np.asarray(p['decoded_normalized_times'],float)
            if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('missing predicted-count time measurement')
            matched_errors.extend(np.abs(actual-expected).tolist())
        eligible+=p['decode_eligible'] is True
        projection+=p['renderer_projection_count']
        target_ranges.append(np.ptp(np.asarray(track.values,float),axis=0))
        values=np.asarray(p['decoded_values'],float)
        if values.shape==(predicted,4) and np.isfinite(values).all():pred_ranges.append(np.ptp(values,axis=0))
    recall=[float(confusion[i,i]/confusion[i].sum()) if confusion[i].sum() else None for i in (0,1)]
    accuracy=correct/len(tracks);balanced=None if None in recall else sum(recall)/2
    time_mae=float(np.mean(time_errors));matched_mae=float(np.mean(matched_errors)) if matched_errors else None
    target_range=np.mean(target_ranges,axis=0);pred_range=np.mean(pred_ranges,axis=0) if pred_ranges else None
    range_complete=len(pred_ranges)==len(tracks)
    target_active=target_range>1e-8
    range_preserved=bool(range_complete and target_active.any() and np.all(pred_range[target_active]>1e-8))
    reasons=[]
    if (pred_hist>0).sum()<min(3,int((target_hist>0).sum())):reasons.append('count_prediction_collapsed')
    if accuracy<limits.minimum_dynamic_crop_keyframe_count_accuracy:reasons.append('count_accuracy_below_existing_gate')
    if balanced is None or balanced<limits.minimum_dynamic_crop_motion_accuracy or any(v is None or v<=0 for v in recall):
        reasons.append('motion_labels_collapsed_or_incorrect')
    if time_mae>limits.maximum_dynamic_crop_normalized_time_mae:reasons.append('teacher_forced_time_error')
    if matched_mae is None or matched_mae>limits.maximum_dynamic_crop_normalized_time_mae:reasons.append('predicted_count_matched_time_error')
    if eligible!=len(tracks) or projection!=0 or not range_complete:reasons.append('renderer_decode_or_projection_failed')
    if source_mismatches:reasons.append('source_binding_mismatch')
    if not range_preserved:reasons.append('target_range_collapsed')
    return dict(contract='train_only_label_and_decode_fit_v1',passed=not reasons,reasons=reasons,
        track_count=len(tracks),count_exact_accuracy=accuracy,target_count_histogram=target_hist.tolist(),
        predicted_count_histogram=pred_hist.tolist(),motion_confusion_static_dynamic=confusion.tolist(),
        motion_recall_static_dynamic=recall,motion_balanced_accuracy=balanced,
        teacher_forced_count_time_mae=time_mae,predicted_count_matched_time_mae=matched_mae,
        renderer_eligible_count=eligible,renderer_projection_count=projection,source_mismatch_count=source_mismatches,
        target_mean_range=target_range.tolist(),predicted_mean_range=None if pred_range is None else pred_range.tolist(),
        range_measurement_complete=range_complete,range_preserved=range_preserved,
        thresholds=dict(count_accuracy=limits.minimum_dynamic_crop_keyframe_count_accuracy,
            motion_balanced_accuracy=limits.minimum_dynamic_crop_motion_accuracy,
            normalized_time_mae=limits.maximum_dynamic_crop_normalized_time_mae),
        validation_forwards=0,heldout_forwards=0,production_activation_authorized=False)
