import numpy as np
from copy import deepcopy
from crosscut.group_review_camera_repair import FramingDescriptor
from crosscut.group_review_measured_features import pair_risk,local_motion


def view(tick, *, mouth=.41):
    vector=[1.]+[0.]*127
    descriptor=FramingDescriptor((.5,.35),.3,.15,None,7,False)
    points=[[.45,.3],[.55,.3],[.5,.36],[.46,mouth],[.54,mouth]]
    return dict(row=dict(output_tick=tick,projection=dict(camera_sha256='a'*64,
        heads=[dict(observation_id=7,projected_landmarks_xy=points)])),
        descriptor=descriptor,vector=vector,near=1.,
        appearance=dict(heads=[dict(observation_id=7,embedding=vector)]))


def test_actual_numpy_kinetic_result_crosses_strict_graph_scalar_boundary():
    a,b=view(0),view(2,mouth=.412)
    motion=local_motion(a,b,30.)
    assert isinstance(motion['vector'],np.ndarray)
    result=pair_risk(a,b,motion,motion)
    assert type(result['risk']) is float and type(result['moving_match']) is float
    assert 0 < result['moving_match'] <= 1
    assert not result['same_actor_certified']


def test_calm_pose_or_missing_motion_cannot_be_renamed_moving_crosscut():
    a,b=view(0),view(2)
    m=local_motion(a,b,30.)
    assert pair_risk(a,b,m,m)['moving_match']==0.
    assert pair_risk(a,b,None,m)['moving_match']==0.


def test_source_camera_change_or_output_alias_does_not_provide_local_motion():
    a,b=view(0),view(2,mouth=.42)
    b['row']['projection']['camera_sha256']='b'*64
    assert local_motion(a,b,30.) is None
    assert local_motion(a,view(0),30.) is None


def test_competing_appearance_hypotheses_cannot_earn_a_match_reward():
    a,b=view(0),view(2,mouth=.42)
    b['appearance']['heads'].append(dict(observation_id=9,embedding=b['vector']))
    m=local_motion(a,b,30.)
    assert pair_risk(a,b,m,m)['moving_match']==0.
