import copy
from types import SimpleNamespace as Item
import pytest
from crosscut.stage_edit_train_fit_checks import label_fit_checks


def fixture():
    tracks=[];predictions=[]
    for count in range(1,7):
        for label in ('static','dynamic'):
            values=[(1.02+i*.001,.5,.5,0.) for i in range(count)]
            times=[i/count for i in range(count)];index=len(tracks)
            tracks.append(Item(source_id=str(index),token_index=index,values=values,normalized_times=times,
                motion_label=label,motion_label_loss_masked=False))
            predictions.append(dict(source_id=str(index),original_token_index=index,predicted_count=count,
                motion_logit=1. if label=='dynamic' else -1.,teacher_forced_count_times=times,
                decoded_normalized_times=times,decode_eligible=True,renderer_projection_count=0,decoded_values=values))
    return tracks,predictions


def test_correct_labels_and_actual_decode_pass_existing_structural_limits():
    tracks,predictions=fixture();report=label_fit_checks(tracks,predictions)
    assert report['passed'] and report['motion_recall_static_dynamic']==[1.,1.]
    assert report['count_exact_accuracy']==1 and report['renderer_projection_count']==0


@pytest.mark.parametrize('mutation',['reverse_motion','two_counts','projection','wrong_source','collapsed_range','wrong_time'])
def test_good_curves_or_diverse_predictions_cannot_hide_label_and_decode_failures(mutation):
    tracks,predictions=fixture();predictions=copy.deepcopy(predictions)
    for p in predictions:
        if mutation=='reverse_motion':p['motion_logit']*=-1
        elif mutation=='two_counts':
            count=min(2,p['predicted_count']);p['predicted_count']=count
            p['decoded_values']=p['decoded_values'][:count];p['decoded_normalized_times']=[i/count for i in range(count)]
        elif mutation=='projection':p['renderer_projection_count']=1
        elif mutation=='wrong_source':p['source_id']='wrong'
        elif mutation=='collapsed_range':p['decoded_values']=[(1.02,.5,.5,0.)]*p['predicted_count']
        elif mutation=='wrong_time':p['teacher_forced_count_times']=[min(.99,t+.2) for t in p['teacher_forced_count_times']]
    report=label_fit_checks(tracks,predictions)
    assert not report['passed'] and report['reasons']
