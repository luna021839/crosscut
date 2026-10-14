import copy
import numpy as np
import pytest
from crosscut2.preference_features import FEATURE_NAMES
from crosscut2.preference_model import fit,Model


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
