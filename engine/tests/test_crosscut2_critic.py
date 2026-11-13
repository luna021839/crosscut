"""Numeric geometry/flash counterexamples, not fabricated detector results."""
import numpy as np
import pytest

from crosscut2.critic import classify,looks_like_flash,pair_features,signature


def face(x=.5,y=.4,h=.2):return dict(center=(x,y),height=h)


def test_missing_face_is_not_a_zero_cost_match():
    result=pair_features((),(face(),))
    assert result['heuristic_discontinuity'] is None
    assert not result['geometry_measured'] and not result['body_action_measured']


def test_focal_scale_and_position_are_measured_independently():
    same=pair_features((face(),),(face(),))
    scale=pair_features((face(),),(face(h=.4),))
    pan=pair_features((face(),),(face(x=.8),))
    assert same['heuristic_discontinuity']==0
    assert scale['face_height_ratio']==2 and scale['face_center_displacement']==0
    assert pan['face_height_ratio']==1 and pan['face_center_displacement']==pytest.approx(.3)
    assert min(scale['heuristic_discontinuity'],pan['heuristic_discontinuity'])>0


def test_group_is_not_relabelled_as_one_focal_member():
    left=tuple(face(x=x,h=.08) for x in (.3,.5,.7))
    right=tuple(reversed(tuple(face(x=x+.1,h=.08) for x in (.3,.5,.7))))
    result=pair_features(left,right)
    assert classify(left)=='formation' and result['role']=='formation'
    assert result['group_center_displacement']==pytest.approx(.1)
    assert result['anonymous_group_layout_residual']==pytest.approx(0.)
    assert not result['named_member_identity_measured']


def test_close_to_extreme_wide_is_retained_as_a_separate_problem():
    result=pair_features((face(h=.25),),(face(h=.025),))
    assert result['close_to_extreme_wide'] and result['face_height_ratio']==10


def test_photometric_flash_can_return_to_the_same_structure():
    rng=np.random.default_rng(5)
    image=rng.integers(0,160,(180,320,3),dtype=np.uint8)
    recovered=np.clip(image.astype(float)*1.1+20,0,255).astype(np.uint8)
    different=rng.integers(0,200,(180,320,3),dtype=np.uint8)
    assert looks_like_flash(signature(image),signature(recovered))
    assert not looks_like_flash(signature(image),signature(different))


def test_featureless_frames_cannot_certify_a_flash():
    blank=np.zeros((90,160,3),dtype=np.uint8)
    assert not looks_like_flash(signature(blank),signature(blank))
