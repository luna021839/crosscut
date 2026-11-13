import copy
import numpy as np
import pytest

from crosscut2.preference_annotations import materialize
from crosscut2.preference_features import SCHEMA, FEATURE_NAMES
from crosscut2.preference_model import fit, Model


def fixture():
    digest='a'*64
    values=np.zeros((3,len(FEATURE_NAMES)));values[1,2]=1.
    features=dict(contract='unlabelled_same_event_preference_features_v1',packet_sha256=digest,
        feature_schema=SCHEMA,feature_names=list(FEATURE_NAMES),events=[dict(id=e,
            physical_group='same-phrase',option_order=['hold_A','cut_A_B','hold_B'],features=values.tolist())
            for e in ('E01','E02','E03','E04')])
    labels=dict(contract='same_event_user_preference_annotations_v1',packet_sha256=digest,
        labels=[dict(id='E01',choice='hold_A',provenance='user_browser_choice',self_reported_normal_av=True)])
    return digest,features,labels


def test_actual_choice_mapping_is_bound_and_reaches_small_fitter_without_relabelling():
    digest,features,labels=fixture()
    rows,report=materialize(labels,features,packet_sha256=digest,split_by_group={'same-phrase':'train'})
    assert rows[0]['chosen']==0 and rows[0]['supervision_kind']=='human_av_preference'
    assert report['imported_strict_preferences']==1 and len(report['unanswered_event_ids'])==3
    model=Model(fit(rows))
    assert model.scores(rows[0]['features'])[0]>model.scores(rows[0]['features'])[1]


def test_ties_all_bad_uncertainty_and_sealed_reference_never_become_fake_winners():
    digest,features,labels=fixture()
    labels['labels']=[dict(id=e,choice=c,provenance='user_browser_choice',self_reported_normal_av=True)
        for e,c in zip(('E01','E02','E03','E04'),('tie','all_bad','uncertain','cut_A_B'))]
    rows,report=materialize(labels,features,packet_sha256=digest,split_by_group={},regression_event_ids={'E04'})
    assert rows==[] and len(report['skipped'])==4
    assert report['skipped'][-1]['reason']=='sealed_regression_not_training'
    with pytest.raises(ValueError,match='no training'):fit(rows)


def test_no_fake_AV_or_reused_packet_old_schema_or_duplicate_label():
    digest,features,labels=fixture()
    labels['labels'][0]['self_reported_normal_av']=False
    rows,report=materialize(labels,features,packet_sha256=digest,split_by_group={})
    assert rows==[] and 'self_reported' in report['skipped'][0]['reason']
    for key,value in (('packet_sha256','b'*64),('feature_schema','raw_same_event_preference_features_v1')):
        old=copy.deepcopy(features);old[key]=value
        with pytest.raises(ValueError,match='binding'):materialize(labels,old,packet_sha256=digest,split_by_group={})
    labels['labels']*=2
    with pytest.raises(ValueError,match='duplicate'):materialize(labels,features,packet_sha256=digest,split_by_group={})


def test_explicit_physical_split_and_actual_option_order_cannot_be_silently_changed():
    digest,features,labels=fixture()
    with pytest.raises(ValueError,match='physical-group split'):
        materialize(labels,features,packet_sha256=digest,split_by_group={})
    features['events'][0]['option_order']=['cut_A_B','hold_A','hold_B']
    with pytest.raises(ValueError,match='option order'):
        materialize(labels,features,packet_sha256=digest,split_by_group={'same-phrase':'train'})
