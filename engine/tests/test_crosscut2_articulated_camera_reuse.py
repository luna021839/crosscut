import numpy as np
import pytest

from crosscut2 import subject_cut_moments as moments
from crosscut2.played_interval_utility import articulated_seam_evidence
from crosscut2.played_partial_pose import features,matrix
from test_crosscut2_projected_articulated_rank import observations


@pytest.mark.parametrize('missing',[False,True])
def test_camera_alternatives_reuse_only_source_terms_and_preserve_every_actual_pair(monkeypatch,missing):
    a,b=observations(missing=missing)
    cameras_a=[[1.,.5,.5],[1.3,.45,.48],[1.6,.52,.55]]
    cameras_b=[[1.,.5,.5],[2.,.6,.45],[1.8,.57,.5],[1.4,.48,.49]]
    left=[dict(a,camera_strip=np.tile(c,(7,1))) for c in cameras_a]
    right=[dict(b,camera_strip=np.tile(c,(7,1))) for c in cameras_b]
    raw=matrix(features(left,fps=30.),features(right,fps=30.),details=True)
    original=moments._seam_action;calls=[]
    def counted(*args,**kwargs):
        calls.append((args,kwargs))
        return original(*args,**kwargs)
    monkeypatch.setattr(moments,'_seam_action',counted)
    actual=articulated_seam_evidence(raw,left,right,static_camera_terms=True)
    assert len(calls)==1
    ca,cb=a['dense_subject_context'],b['dense_subject_context']
    for i,camera_a in enumerate(cameras_a):
        for j,camera_b in enumerate(cameras_b):
            value=original(ca['window'],cb['window'],ca['clock'],cb['clock'],dict(
                common_source_shot_sample_mask=ca['same_shot']&cb['same_shot'],
                outgoing_camera=camera_a,incoming_camera=camera_b))
            assert actual['articulated_arm_direction_measured'][i,j]==value['seam_action_supported']
            if not value['seam_action_supported']:
                assert actual['pose_rate'][i,j]==raw['pose_rate'][i,j]
                continue
            residual=np.array([value[k] for k in ('seam_pose_residual','seam_screen_residual','seam_motion_residual')])
            np.testing.assert_array_equal(actual['articulated_configuration_screen_motion'][i,j],residual/(1.+residual))
            assert actual['pose_rate'][i,j]==max(value['seam_arm_direction_residual'],float((residual/(1.+residual)).mean()))
    # A later invocation cannot reuse stale geometry after a caller changes
    # a context; neither a process-wide cache nor invented support is used.
    articulated_seam_evidence(raw,left,right)
    assert len(calls)==2


def test_distinct_same_window_contexts_do_not_share_different_native_shot_masks(monkeypatch):
    a,b=observations();context=b['dense_subject_context']
    unsupported=dict(context,same_shot=np.zeros_like(context['same_shot']))
    right=[b,dict(b,dense_subject_context=unsupported)]
    raw=matrix(features([a],fps=30.),features(right,fps=30.),details=True)
    original=moments._seam_action;calls=[]
    def counted(*args,**kwargs):
        calls.append(None)
        return original(*args,**kwargs)
    monkeypatch.setattr(moments,'_seam_action',counted)
    actual=articulated_seam_evidence(raw,[a],right)
    assert len(calls)==2
    assert actual['articulated_arm_direction_measured'][0].tolist()==[True,False]
    assert actual['pose_rate'][0,1]==raw['pose_rate'][0,1]
