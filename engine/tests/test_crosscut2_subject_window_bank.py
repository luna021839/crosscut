from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.subject_window_bank import ProbeBudget,associate,build,save,load,face_observation
from crosscut2.temporal_face_chain import _bounds


def example(n=7):
    frames=[np.full((80,144,3),i,np.uint8) for i in range(n)]
    heads=np.array([[.12,.23,.12,.99],[.35,.23,.11,.99],[.58,.23,.10,.99],[.82,.23,.08,.99]])
    detections=[];vectors=[]
    for i in range(n):
        order=[3,1,0,2] if i%2 else [0,1,2,3]
        detections.append(heads[order]);vectors.append(np.eye(128,dtype=np.float32)[:4][order])
    def flow(*args):return _bounds(args[-1]),dict(flow_points=12,inliers=12)
    return frames,detections,vectors,heads,flow


class Sensor:
    def __init__(self):self.calls=0
    def describe(self,pixels,*,bbox_xywh):
        self.calls+=1;raw=np.zeros((17,3));raw[:,2]=.9;raw[0,:2]=[.82,.23]
        raw[5:13,:2]=[[.78,.35],[.86,.35],[.75,.45],[.89,.45],[.72,.5],[.92,.5],[.78,.6],[.86,.6]]
        return SimpleNamespace(reason='observed',raw_keypoints_xyc=raw)


def test_fourth_face_reordered_track_is_pinned_before_pose_and_uses_contemporary_roi():
    frames,heads,vectors,seed,flow=example();sensor=Sensor();budget=ProbeBudget()
    result,receipt=build(source=3,binding='pixels-model-clock',frames=frames,detections=heads,vectors=vectors,
        ordinals=np.arange(20,27),seed_native=23,seed_head=seed[3],pose_natives=[20,23,26],
        sensor=sensor,budget=budget,flow=flow)
    assert result is not None and result.associated.all()
    assert np.all(result.heads[:,0]==.82) and result.support.sum()==18
    assert sensor.calls==budget.used['pose']==3
    assert not receipt['named_identity_certified'] and not receipt['anatomical_ownership_certified']


def test_missing_frame_does_not_adopt_a_same_position_different_appearance_intruder():
    frames,heads,vectors,seed,flow=example();heads[3]=np.array([seed[3]]);vectors[3]=-np.eye(128,dtype=np.float32)[[3]]
    tracks=associate(frames,heads,vectors,np.arange(20,27),flow=flow)
    target=next(t for t in tracks if t['indices'][0]==3)
    assert target['indices'][3]==-1


def test_native_camera_cut_splits_tracks_instead_of_interpolating_actor_across_shots():
    frames,heads,vectors,seed,flow=example();result,_=build(source=0,binding='test',frames=frames,
        detections=heads,vectors=vectors,ordinals=np.arange(20,27),seed_native=25,seed_head=seed[3],
        pose_natives=[20,23,26],sensor=Sensor(),budget=ProbeBudget(),native_cuts=(24,),flow=flow)
    assert not result.associated[:4].any() and result.associated[4:].all()
    assert not result.support[:4].any()


def test_flow_failure_with_no_appearance_does_not_fall_back_to_nearest_face():
    frames,heads,vectors,seed,flow=example();vectors=[np.zeros_like(v) for v in vectors]
    tracks=associate(frames,heads,vectors,np.arange(20,27),flow=lambda *a:(None,{}))
    assert all(np.count_nonzero(t['indices']>=0)==1 for t in tracks)


def test_rotating_face_can_use_multi_cue_contemporary_appearance_without_best_cosine_only():
    frames,heads,vectors,seed,flow=example()
    tracks=associate(frames,heads,vectors,np.arange(20,27),flow=lambda *a:(None,{}))
    target=next(t for t in tracks if t['indices'][0]==3)
    assert (target['indices']>=0).all()


def test_successful_face_path_cannot_borrow_the_intruders_raw_body():
    frames,heads,vectors,seed,flow=example()
    class Intruder(Sensor):
        def describe(self,*args,**kwargs):
            value=super().describe(*args,**kwargs);value.raw_keypoints_xyc[0,:2]=seed[0,:2]
            return value
    result,_=build(source=0,binding='test',frames=frames,detections=heads,vectors=vectors,
        ordinals=np.arange(20,27),seed_native=23,seed_head=seed[3],pose_natives=[20,23,26],
        sensor=Intruder(),budget=ProbeBudget(),flow=flow)
    assert result.associated.all() and not result.support.any()


def test_exact_shared_last_pose_reservation_precedes_nn_and_is_not_reset_per_source():
    budget=ProbeBudget(pose_limit=3);sensor=Sensor();frames,heads,vectors,seed,flow=example()
    kwargs=dict(binding='test',frames=frames,detections=heads,vectors=vectors,ordinals=np.arange(20,27),
        seed_native=23,seed_head=seed[3],pose_natives=[20,23,26],sensor=sensor,budget=budget,flow=flow)
    build(source=0,**kwargs)
    with pytest.raises(TimeoutError,match='pose allowance'):build(source=1,**kwargs)
    assert sensor.calls==budget.used['pose']==3


def test_warm_bank_replays_original_arrays_without_nn_or_decoder(tmp_path):
    frames,heads,vectors,seed,flow=example();sensor=Sensor()
    result,_=build(source=0,binding='bound',frames=frames,detections=heads,vectors=vectors,
        ordinals=np.arange(20,27),seed_native=23,seed_head=seed[3],pose_natives=[20,23,26],
        sensor=sensor,budget=ProbeBudget(),flow=flow)
    path=tmp_path/'bank.npz';save(result,path,provenance={'not_identity_certification':True})
    before=path.read_bytes();warm=load(path,binding='bound')
    np.testing.assert_array_equal(warm.body,result.body);np.testing.assert_array_equal(warm.heads,result.heads)
    assert path.read_bytes()==before and sensor.calls==3
    with pytest.raises(ValueError,match='binding changed'):load(path,binding='other-source-or-phase')


def test_subject_decoder_budget_counts_every_returned_raster_and_buffer_before_new_work():
    b=ProbeBudget(raster_limit=2,buffer_limit=10);b.reserve('raster',2)
    with pytest.raises(TimeoutError):b.reserve('raster')
    assert b.used['raster']==2
    with pytest.raises(MemoryError):b.arrays(11)
