import numpy as np
from crosscut2.played_partial_pose import features,matrix
from test_crosscut2_source_clock_action import catalogue,obs,set_style


def test_common_bone_motion_cannot_erase_three_quarter_face_height_body_shift():
    a,b=catalogue(),catalogue()
    set_style(a,0);set_style(b,1)
    def compare():return matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    same=compare();b.body[...,1]+=.15;shifted=compare()
    assert same['same_clock_source_action_used'][0,0] and shifted['same_clock_source_action_used'][0,0]
    np.testing.assert_allclose(same['common_clock_action_channels'],shifted['common_clock_action_channels'],atol=1e-7)
    assert shifted['pose_rate'][0,0]>same['pose_rate'][0,0]+.03
    assert shifted['played_body_placement_channels'].mean()>same['played_body_placement_channels'].mean()+.03
    assert same['face_rate'][0,0]==shifted['face_rate'][0,0]==0.


def test_synchronous_reversal_does_not_restore_the_unrelated_velocity_penalty():
    a,b=catalogue(),catalogue()
    value=matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    assert value['same_clock_source_action_used'][0,0]
    assert value['velocity_raw'].mean()>.1
    expected=(2*value['played_body_placement_channels']+value['common_clock_action_channels'])/3.
    np.testing.assert_allclose(value['fixed_semantic_channel_rates'],expected)
