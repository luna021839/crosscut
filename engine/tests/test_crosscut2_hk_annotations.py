import copy
import hashlib
import json
import pytest
from crosscut2.hk_annotations import import_annotations


def fixture():
    packet=json.dumps(dict(contract='unlabelled_same_event_hk_av_pairs_v2',
        actual_human_preferences_obtained=False,events=[dict(id=e,clips=dict(A=e+'-A.mp4',B=e+'-B.mp4'))
            for e in ('E01','E02','E03','E04','E05','E06')])).encode()
    grounding=json.dumps(dict(events=[dict(id=e,global_frame=i*300,
        variant_map=dict(A='K' if i%2 else 'H',B='H' if i%2 else 'K'))
        for i,e in enumerate(('E01','E02','E03','E04','E05','E06'))])).encode()
    annotations=dict(contract='same_event_user_hk_annotations_v2',packet_sha256=hashlib.sha256(packet).hexdigest(),
        labels=[dict(id='E01',choice='B',provenance='user_browser_choice',self_reported_normal_av=True)])
    kw=dict(manifest_bytes=packet,grounding_bytes=grounding,
        expected_grounding_sha256=hashlib.sha256(grounding).hexdigest())
    return annotations,kw


def test_exact_blinded_choice_is_mapped_without_fitting_or_numeric_features():
    labels,kw=fixture();labels['labels'].append(dict(labels['labels'][0],id='E02',choice='A'))
    rows,report=import_annotations(labels,**kw)
    assert [r['chosen_action'] for r in rows]==['K','K']
    assert [r['choice'] for r in rows]==['B','A']
    assert all(r['supervision_kind']=='human_av_hk_pair_annotation_only' for r in rows)
    assert all('features' not in r and 'chosen' not in r for r in rows)
    assert report['imported_strict_preferences']==2 and report['preference_fit_executed'] is False
    assert len(report['unanswered_event_ids'])==4


def test_tie_bad_uncertain_unanswered_and_sealed_are_distinct_not_fake_winners():
    labels,kw=fixture();base=labels['labels'][0]
    labels['labels']=[dict(base,id=e,choice=choice) for e,choice in zip(
        ('E01','E02','E03','E04','E05'),('tie','all_bad','uncertain','','A'))]
    rows,report=import_annotations(labels,**kw,regression_event_ids=['E05'])
    assert rows==[]
    assert [r['choice'] for r in report['non_strict_or_ineligible']]==['tie','all_bad','uncertain','','A']
    assert report['unanswered_event_ids']==['E04','E06']
    assert report['non_strict_or_ineligible'][-1]['reason']=='sealed_regression_not_training'


@pytest.mark.parametrize('change',[dict(self_reported_normal_av=False),dict(provenance='model_still_review'),
    dict(provenance='agent_browser_playback'),dict(self_reported_normal_av=1)])
def test_playback_telemetry_or_model_advice_cannot_supply_missing_human_provenance(change):
    labels,kw=fixture();labels['labels'][0].update(change)
    labels['playback_events_advisory_only']={'E01':{'A':{'ended':True},'B':{'ended':True}}}
    rows,report=import_annotations(labels,**kw)
    assert rows==[] and report['non_strict_or_ineligible'][0]['reason'].startswith('missing_actual')


def test_different_packet_private_mapping_or_contract_is_rejected():
    labels,kw=fixture()
    for override in (dict(manifest_bytes=kw['manifest_bytes']+b' '),dict(expected_grounding_sha256='a'*64),
        dict(grounding_bytes=kw['grounding_bytes']+b' ')):
        with pytest.raises(ValueError):import_annotations(labels,**dict(kw,**override))
    labels['contract']='same_event_user_preference_annotations_v1'
    with pytest.raises(ValueError):import_annotations(labels,**kw)


@pytest.mark.parametrize('bad',['duplicate','unknown','invalid_choice','duplicate_mapping','bad_mapping','missing_mapping'])
def test_duplicate_unknown_or_non_bijective_choices_are_rejected(bad):
    labels,kw=fixture()
    if bad=='duplicate':labels['labels']*=2
    elif bad=='unknown':labels['labels'][0]['id']='not-in-packet'
    elif bad=='invalid_choice':labels['labels'][0]['choice']='hold_A'
    else:
        mapping=json.loads(kw['grounding_bytes'])
        if bad=='duplicate_mapping':mapping['events'].append(copy.deepcopy(mapping['events'][0]))
        elif bad=='bad_mapping':mapping['events'][0]['variant_map']['B']='H'
        else:mapping['events'].pop()
        kw['grounding_bytes']=json.dumps(mapping).encode()
        kw['expected_grounding_sha256']=hashlib.sha256(kw['grounding_bytes']).hexdigest()
    with pytest.raises(ValueError):import_annotations(labels,**kw)


def test_empty_actual_export_imports_no_labels():
    labels,kw=fixture();labels['labels']=[]
    rows,report=import_annotations(labels,**kw)
    assert rows==[] and report['imported_strict_preferences']==0 and len(report['unanswered_event_ids'])==6
