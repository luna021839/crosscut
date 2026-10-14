import copy
import numpy as np
import pytest
from crosscut2.preference_features import FEATURE_NAMES
from crosscut2.preference_model import fit,Model,partial_choice


def example(group='phrase-a',split='train',kind='human_av_preference'):
    values=np.zeros((3,len(FEATURE_NAMES)));values[1,2]=1.;values[2,2]=.5
    return dict(physical_group=group,split=split,supervision_kind=kind,self_reported_normal_av=True,
        features=values.tolist(),chosen=0)


def test_only_actual_provenance_can_train_and_normalization_never_uses_held_out():
    a=example();b=example('different-song','held_out');b['features'][2][3]=1e6
    artifact=fit([a,b]);assert artifact['center'][3]==0
    assert artifact['report']['held_out']['pairs']==2
    assert not artifact['professional_quality_approved']
    assert Model(artifact).scores(np.array(a['features']))[0]>Model(artifact).scores(np.array(a['features']))[1]
    with pytest.raises(ValueError,match='no training'):fit([])


def test_no_phrase_leakage_or_advisory_frames_masquerading_as_human_av():
    with pytest.raises(ValueError,match='leaks'):fit([example(),example(split='dev')])
    with pytest.raises(ValueError,match='advisory'):fit([example(kind='opus_silent_still_advisory')])
    row=example();row['self_reported_normal_av']=False
    with pytest.raises(ValueError,match='self-report'):fit([row])


def test_legacy_weights_or_tampered_schema_do_not_silently_enter_new_objective():
    a=fit([example()]);bad=copy.deepcopy(a);bad['weights'][0]+=.1
    with pytest.raises(ValueError,match='binding'):Model(bad)


def test_event_wide_max_hold_and_cut_losses_are_invariant_to_option_duplication():
    m=Model(fit([example()]));f=np.zeros((2,2,len(FEATURE_NAMES)));f[0,1,2]=1.
    h,k=m.losses(f,state_sources=[0,1]);dup=f[[0,0,1]][:,[0,0,1]]
    hh,kk=m.losses(dup,state_sources=[0,0,1])
    assert np.array_equal(hh,h[[0,0,1]]) and np.array_equal(kk,k[[0,0,1]][:,[0,0,1]])


def weak():
    row=example(kind='reference_editor_choice_weak_imitation');row.pop('chosen')
    values=np.zeros((3,len(FEATURE_NAMES)));values[0,2]=1.
    row.update(features=values.tolist(),compatible_options=[1,2],option_ids=['Ahold','AtoB','Bhold'])
    return row


def test_weak_incoming_B_choice_does_not_invent_cut_over_B_hold_and_deduplicates_physical_options():
    row=weak();model=Model(fit([row]));scores=model.scores(row['features'])
    assert scores[1]==scores[2] and scores[1]>scores[0]
    duplicate=copy.deepcopy(row);duplicate['features'].append(row['features'][1]);duplicate['option_ids'].append('AtoB')
    duplicate['compatible_options'].append(3)
    again=Model(fit([duplicate]))
    np.testing.assert_array_equal(again.weights,model.weights)
    assert model.artifact['report']['train']['pairs']==0
    assert model.artifact['report']['train']['weak_events']==1
    row['chosen']=1
    with pytest.raises(ValueError,match='strict H/K'):fit([row])


def test_partial_marginal_gradient_matches_finite_difference_and_preserves_no_preference_inside_set():
    score=np.array([.3,-.2,.7]);yes=[1,2];loss,gradient=partial_choice(score,yes)
    numeric=[]
    for i in range(3):
        delta=np.eye(3)[i]*1e-6
        numeric.append((partial_choice(score+delta,yes)[0]-partial_choice(score-delta,yes)[0])/2e-6)
    np.testing.assert_allclose(gradient,numeric,atol=1e-9)
    assert loss>0 and abs(gradient.sum())<1e-12


def test_repeated_reference_of_same_physical_group_cannot_double_its_training_weight():
    a=weak();b=example(group='different-phrase');base=Model(fit([a,b]))
    again=Model(fit([a,copy.deepcopy(a),b]))
    np.testing.assert_allclose(base.center,again.center,atol=1e-15)
    np.testing.assert_allclose(base.weights,again.weights,atol=1e-14)
