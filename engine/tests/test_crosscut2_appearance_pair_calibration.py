from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from crosscut2.appearance_pair_calibration import PairCalibration,active,use
from crosscut2.focal_appearance import direct_discrepancy,discrepancy
from crosscut2.played_pose_port_programme import PlayedPosePortPrior


def calibration(**kwargs):
    return PairCalibration(**dict(intercept=-5.6,slope=18.2,gallery_identity='a'*64,
        reference_artifact_sha256='b'*64,**kwargs))


def vector(cosine):
    a=np.eye(512)[0];b=np.zeros(512);b[:2]=[cosine,np.sqrt(1.-cosine**2)]
    return a,b


def test_weak_positive_cosine_is_not_assumed_better_than_reference_unknown():
    a,b=vector(.13);old,_,raw=direct_discrepancy(a,b,True,True)
    with use(calibration()):new,known,current=direct_discrepancy(a,b,True,True)
    assert old<.5<new<1 and known and raw==current
    assert active() is None and direct_discrepancy(a,b,True,True)[0]==old


def test_strong_same_face_retains_a_bounded_useful_score_and_unknown_stays_exactly_neutral():
    a,b=vector(.7)
    with use(calibration()):
        new,known,_=direct_discrepancy(a,b,True,True)
        missing,measured,raw=direct_discrepancy(a,b,True,False)
    assert known and 0<new<(1.-.7)/2.
    assert missing==.5 and not measured and np.isnan(raw)


def test_reference_calibration_does_not_reinterpret_the_independent128D_SFace():
    sf=np.eye(128)[0];a,b=vector(.13)
    baseline=discrepancy(sf,sf)[0]
    with use(calibration()):
        assert discrepancy(sf,sf)[0]==baseline
        combined=discrepancy(sf,sf,direct_a=a,direct_b=b,direct_valid_a=True,direct_valid_b=True)[0]
        direct=direct_discrepancy(a,b,True,True)[0]
    assert combined==pytest.approx((baseline+direct)/2.)


def test_explicit_context_restores_on_failure_and_does_not_change_other_threads():
    original=calibration()
    with pytest.raises(RuntimeError):
        with use(original):
            assert active() is original
            with ThreadPoolExecutor(1) as executor:assert executor.submit(active).result() is None
            raise RuntimeError('intentional interrupted solve')
    assert active() is None


def test_experiment_provenance_is_bound_to_the_actual_prior_not_a_hidden_global_weight():
    options=dict(editorial_opportunities=True,played_interval_utility=True,direct_face_appearance=True)
    ordinary=PlayedPosePortPrior(**options)
    with use(calibration()):experimental=PlayedPosePortPrior(**options)
    assert ordinary.artifact['direct_face_pair_score_calibration'] is None
    record=experimental.artifact['direct_face_pair_score_calibration']
    assert record['reference_only'] and not record['professional_quality_approved']
    assert ordinary.artifact['content_sha256']!=experimental.artifact['content_sha256']


@pytest.mark.parametrize('kwargs',[dict(strength=1.),dict(slope=0.),dict(slope=True),dict(gallery_identity='bad'),
    dict(model_sha256='c'*64),dict(intercept=float('nan'))])
def test_invalid_or_unbounded_reference_calibration_is_rejected(kwargs):
    options=dict(intercept=-5.6,slope=18.2,gallery_identity='a'*64,reference_artifact_sha256='b'*64)
    options.update(kwargs)
    with pytest.raises(ValueError):PairCalibration(**options)
