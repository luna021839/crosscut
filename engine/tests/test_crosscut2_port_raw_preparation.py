import numpy as np
from crosscut2 import event_states
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from crosscut2.event_holds import select
from test_crosscut2_preference_integration import producer


def test_port_pipeline_does_not_build_unused_actor_cross_camera_reward_rows(tmp_path,monkeypatch):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    def forbidden(*a,**kw):raise AssertionError('E must not build discarded416-style projected rows')
    monkeypatch.setattr(event_states,'build',forbidden)
    runs,states,data,_,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=PlayedPosePortPrior(),preference_workspace=tmp_path)
    assert runs and not receipt['unused_actor_camera_projection_rows_allocated']
    assert receipt['raw_actor_states_prepared_once']<=receipt['fixed_states']
    assert all(s.actor is None for s in states)
    assert all(np.isfinite(d['curve']).all() for d in data)
