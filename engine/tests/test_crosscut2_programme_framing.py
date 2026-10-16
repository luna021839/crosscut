import copy
import numpy as np
import pytest
from crosscut2.programme_framing import evidence
from test_crosscut2_source_clock_action import catalogue


def native_bank(height,family):
    bank=copy.deepcopy(catalogue().bank);bank['heads'][:]=0.
    bank['heads'][:,:,0]=[.5,.3,height,.99];bank['consensus_family']=family
    return bank


def test_native_family_median_is_fixed_before_cameras_and_clone_does_not_change_it():
    banks=[native_bank(.1,'a'),native_bank(.2,'b'),native_bank(.3,'c')]
    state,receipt=evidence(banks)
    np.testing.assert_allclose(state['height'],.2)
    more,again=evidence([*banks,copy.deepcopy(banks[2]),copy.deepcopy(banks[2])])
    for key in state:np.testing.assert_array_equal(more[key],state[key])
    assert receipt['original_source_families']==again['original_source_families']==3
    assert not state['formation'].any() and state['confidence'].min()>0.


def test_unknown_single_original_family_does_not_impose_a_framing_role():
    state,_=evidence([native_bank(.3,'a'),native_bank(.3,'a')])
    assert not state['confidence'].any()


def test_readable_but_distant_shot_has_role_deficit_not_a_mandatory_cut():
    cat=catalogue();cat.editorial_content=True
    cat.bank['heads'][:]=0.;cat.bank['heads'][:,:,0]=[.5,.3,.12,.99]
    framing,_=evidence([native_bank(.22,'a'),native_bank(.24,'b'),native_bank(.26,'c')])
    cat.framing_evidence=framing
    frames=np.array([19,20]);native=np.tile([1.,.5,.5],(2,1))
    old=cat.rates(frames,native)
    cat.framing_evidence=None;without=cat.rates(frames,native)
    assert (old>without).all()
    # The same evidence does not keep rewarding a face once its role size is
    # met. Oversized-camera and detail costs still act in the other direction.
    cat.framing_evidence=framing;cat.bank['heads'][:,:,0,2]=.26
    roomy=cat.rates(frames,native);zoomed=np.tile([1.35,.5,.5],(2,1))
    assert (cat.rates(frames,zoomed)>roomy).all()


def test_original_formation_evidence_requires_people_actually_inside_viewport():
    banks=[native_bank(.1,f) for f in ('a','b','c')]
    for bank in banks:
        for i,x in enumerate(np.linspace(.15,.85,5)):bank['heads'][:,:,i]=[x,.4,.1,.99]
    framing,_=evidence(banks)
    assert framing['formation'].all() and framing['people'].min()==5
    cat=catalogue();cat.editorial_content=True;cat.framing_evidence=framing;cat.bank['heads']=banks[0]['heads'].copy()
    frames=np.array([19,20]);native=np.tile([1.,.5,.5],(2,1));crop=np.tile([2.,.5,.5],(2,1))
    assert (cat.rates(frames,crop)>cat.rates(frames,native)).all()


def test_observed_no_face_cannot_dodge_the_required_closeup_deficit():
    cat=catalogue();cat.editorial_content=True
    cat.framing_evidence=evidence([native_bank(.22,'a'),native_bank(.24,'b'),native_bank(.26,'c')])[0]
    frames=np.array([19,20]);camera=np.tile([1.,.5,.5],(2,1))
    cat.bank['heads'][:]=0.;cat.content_scope[:]=True
    known_empty=cat.rates(frames,camera)
    cat.content_scope[:]=False;unsupported_gap=cat.rates(frames,camera)
    assert (known_empty>unsupported_gap).all()
    assert np.isfinite(known_empty).all() and np.isfinite(unsupported_gap).all()


def test_output_oversize_effort_is_origin_blind_and_keeps_ordinary_closeups():
    cat=catalogue();cat.editorial_content=True;cat.body[:]=0.
    cat.bank['heads'][:]=0.;cat.bank['heads'][:,:,0]=[.5,.5,.2,.99]
    frames=np.array([19,20]);native=np.tile([1.,.5,.5],(2,1))
    roomy=cat.rates(frames,native)
    cat.bank['heads'][:,:,0,2]=.27
    np.testing.assert_allclose(cat.rates(frames,native),roomy)
    cat.bank['heads'][:,:,0,2]=.4
    original_tight=cat.rates(frames,native)
    np.testing.assert_allclose(original_tight-roomy,.6,atol=1e-6)
    cat.bank['heads'][:,:,0,2]=.2
    added_tight=cat.rates(frames,np.tile([2.,.5,.5],(2,1)))
    # Equal output size pays equal oversize effort. Only the already existing
    # digital enlargement/detail cost distinguishes how the shot was made.
    np.testing.assert_allclose(added_tight-original_tight,.1,atol=1e-6)


def test_unobserved_temporal_content_keeps_original_proxy_without_fabricated_face_or_free_crop():
    cat=catalogue();cat.editorial_content=True;cat.content_scope[:]=False
    frames=np.array([19,20]);native=np.tile([1.,.5,.5],(2,1))
    np.testing.assert_allclose(cat.rates(frames,native),1.)
    np.testing.assert_allclose(cat.rates(frames,np.tile([2.,.5,.5],(2,1))),1.1)
    cat.bank['heads'][:]=0.;cat.content_scope[:]=True
    assert np.isfinite(cat.rates(frames,native)).all()
