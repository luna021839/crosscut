from types import SimpleNamespace
import numpy as np
import pytest
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut2.programme_source_clock import indices,admit,attach


def test_vectorized_clock_keeps_exact_production_millisecond_arithmetic():
    for start in (0.,.5/29.97,1.25,62.3156489816483):
        for rate in (1.,1.000000000000001,.9994):
            expected=sampler_frame_schedule(source_start=start,source_playback_rate=rate,
                native_fps=29.97,output_fps=29.97,frame_count=240,source_frame_count=10000)
            np.testing.assert_array_equal(indices(start,rate,np.arange(240),fps=29.97,native_fps=29.97),
                expected.source_frame_indices)


def fixture():
    nodes=np.array([0,15,30]);bank={};warp=SimpleNamespace(master_to_source=lambda t:t)
    attach([bank],[warp],[dict(fps=30.,frame_count=100)],nodes,master_start=0.,fps=30.)
    bank['pose_native_indices']=np.clip(nodes[:,None]+[-7,-1,0,6],0,29)
    bank['native_indices']=np.clip(nodes[:,None]+[-1,0],0,29)
    port=SimpleNamespace(observation={})
    return bank,nodes,[port],[port]


def test_mismatched_old_input_masks_only_that_hold_before_DP_not_the_source():
    bank,nodes,left,right=fixture()
    assert admit(bank,nodes,0,2,left,right,fps=30.).all()
    bank['pose_native_indices'][1,3]+=1
    assert not admit(bank,nodes,0,2,left,right,fps=30.).any()
    assert admit(bank,nodes,0,1,left,right,fps=30.).all()  # mismatching input not played


def test_dense_unplayed_context_is_bound_too_not_only_the_cut_endpoints():
    bank,nodes,left,right=fixture();native=np.arange(8,22)
    context=dict(clock=SimpleNamespace(output_frames=np.arange(-7,7),native_indices=native))
    right=[SimpleNamespace(observation=dict(dense_subject_context=context))]
    assert admit(bank,nodes,0,1,left,right,fps=30.).all()
    native[8]+=1
    assert not admit(bank,nodes,0,1,left,right,fps=30.).any()


def test_absent_clock_is_not_fabricated_as_verified():
    bank,nodes,left,right=fixture();bank.pop('programme_render_clock')
    assert admit(bank,nodes,0,2,left,right,fps=30.).all()  # legacy domain, final verifier still required


def test_stale_grid_is_rejected_before_any_actor_camera_preparation(monkeypatch):
    from crosscut2 import played_pose_port_programme as programme
    bank,nodes,_,_=fixture();bank['programme_render_clock']['source_times']=np.zeros(2)
    def forbidden(*a,**k):raise AssertionError('costly actor/camera preparation started before clock validation')
    monkeypatch.setattr(programme,'make_states',forbidden)
    prior=programme.PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True)
    with pytest.raises(ValueError,match='candidate grid differs'):
        programme.select(nodes,[bank],None,[{}],fps=30.,source_events=[[]],body_states=None,
            native_ports=[],music=np.zeros(len(nodes)),prior=prior)
