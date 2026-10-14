import numpy as np
import pytest
from crosscut2.motion_direction import compare
from crosscut2.preference_features import extract,FEATURE_NAMES,SCHEMA
from test_crosscut2_preference_features import setup


def test_one_stationary_vector_is_unknown_direction_not_measured_opposition():
    direction,moving,observed=compare(np.array([1.,0.]),np.array([0.,0.]))
    assert direction==.5 and moving and not observed
    assert compare(np.array([1.,0.]),np.array([-1.,0.]))[0]==1.
    assert compare(np.array([1.,0.]),np.array([1.,0.]))[0]==0.
    assert compare(np.zeros(2),np.zeros(2))[0]==0.


def test_producer_preserves_speed_error_and_separate_unknown_direction_without_fake_opposition():
    p,a,b=setup();p.velocity[:,1]=0.
    p.velocity[a,1,:]=[1.,0.]
    p.phase_certainty[:,1]=1.;p.motion_certainty[:,1]=1.
    for row in p.data:row['pixel_part_scope']=np.ones(len(p.nodes),bool)
    f=extract(p,1)[a,b]
    assert .5<=f[FEATURE_NAMES.index('motion')]<1.
    assert f[FEATURE_NAMES.index('observed_opposition')]==0.
    assert f[FEATURE_NAMES.index('motion_direction_unmeasured_geometry')]>0.
    assert SCHEMA.endswith('_v6')
    # Unknown direction never erases a genuinely opposite measured vector.
    p.velocity[b,1,:]=[-1.,0.]
    f=extract(p,1)[a,b]
    assert f[FEATURE_NAMES.index('motion')]==pytest.approx(1.)
    assert f[FEATURE_NAMES.index('observed_opposition')]==pytest.approx(1.)
    assert f[FEATURE_NAMES.index('motion_direction_unmeasured_geometry')]==0.
