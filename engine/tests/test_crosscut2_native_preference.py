from types import SimpleNamespace
import numpy as np

from crosscut2.native_preference import prepare
from crosscut2.preference_model import regret_bounds
from crosscut2.event_holds import select
from test_crosscut2_preference_integration import producer,fitted_toy_model
from crosscut2.preference_features import FEATURE_NAMES


def native_bank(frame):
    head=np.zeros((1,2,8,4),np.float32)
    head[0,:,0]=[[.2,.2,.12,.95],[.8,.6,.4,.95]]
    return dict(frames=np.array([frame]),heads=head,observed=np.ones(1,bool),vectors=np.zeros((1,2,3,128)))


def test_actual_producer_off_grid_native_H_enters_primary_once_without_new_authoring_nodes(tmp_path):
    args,kw,native,p,a,b=producer(opposed=False)
    model=fitted_toy_model()
    _,_,_,_,baseline,_=select(*args,**kw,native_ports=native,music=[0.,1.,1.,0.],
        preference_model=model,preference_workspace=tmp_path)
    native=list(native);native[1]=native_bank(90)
    runs,states,data,score,receipt,_=select(*args,**kw,native_ports=native,music=[0.,1.,1.,0.],
        preference_model=model,preference_workspace=tmp_path)
    evidence=receipt['event_preference'];nr=evidence['native_evaluation']
    assert nr['evaluation_only_native_events']==[90]
    assert score>baseline['event_preference']['objective_vector'][0]
    assert all(args[0][first]!=90 and args[0][end]!=90 for first,end,_ in runs)
    assert sum(r['evaluation_native_preference_upper'] for r in receipt['selected_event_preferences'])>0
    assert nr['source_neural_calls']==nr['source_decode']==0
    assert not nr['full_perceptual_priority_certified']


def test_native_authored_overlap_is_replaced_by_H_K_not_double_charged():
    _,_,native,p,_,_=producer(opposed=False);native=list(native);native[1]=native_bank(60)
    upper,lower,receipt=prepare(p,fitted_toy_model(),native_ports=native,
        source_events=[np.array([],int),np.array([60])],check=lambda:None)
    assert not upper.any() and not lower.any()
    assert receipt['authored_native_events_not_double_counted']==1
    assert receipt['evaluation_only_native_events']==[]


def test_all_states_are_evaluated_at_one_shared_time_and_unknown_is_finite_not_free():
    _,_,native,p,_,_=producer(opposed=False);native=list(native);native[1]=native_bank(90)
    upper,lower,receipt=prepare(p,fitted_toy_model(),native_ports=native,
        source_events=[np.array([],int),np.array([90])],check=lambda:None)
    assert np.isfinite(upper).all() and np.all(lower<=upper)
    assert (upper[:,1]>0).all() and not upper[:,[0,2]].any()
    assert receipt['unique_physical_source_camera_options']<len(p.states)
    # More local actor labels do not create new rendered camera options.
    copied=SimpleNamespace(states=(*p.states,p.states[0]),data=(*p.data,p.data[0]),fps=p.fps,nodes=p.nodes)
    u,_,_=prepare(copied,fitted_toy_model(),native_ports=native,
        source_events=[np.array([],int),np.array([90])],check=lambda:None)
    np.testing.assert_array_equal(u[:-1],upper)
    np.testing.assert_array_equal(u[-1],upper[0])


def test_finite_score_interval_self_regret_and_physical_duplicate_correlation():
    lo,hi=regret_bounds([-100.],[100.],physical_ids=['one-physical-hold'])
    assert lo[0]==hi[0]==0.
    a,b=regret_bounds([-2.,1.],[3.,2.],physical_ids=['A','B'])
    aa,bb=regret_bounds([-2.,-2.,1.],[3.,3.,2.],physical_ids=['A','A','B'])
    np.testing.assert_array_equal(aa,a[[0,0,1]])
    np.testing.assert_array_equal(bb,b[[0,0,1]])


def test_no_native_self_H_has_exact_zero_face_geometry_even_if_actor_continuity_is_unknown():
    # Derive the relation from the real encoder, not a favourable assumption
    # about unobserved video. Same-clock self comparison is identically0;
    # exposed-face fallback is explicitly restricted to known central cuts.
    _,_,_,p,a,_=producer(opposed=False);original_body=p.body.copy()
    for continuous in (True,False):
        for body_present in (True,False):
            p.continuous[:,1]=continuous
            p.data[a]['known_center_native_cut'][1]=False
            p.head[a,1,1,:3]=[.8,.6,.42]
            p.body[:]=original_body if body_present else 0.
            values=p.preference_features(1)[a,a]
            assert values[FEATURE_NAMES.index('face_position')]==0.
            assert values[FEATURE_NAMES.index('face_scale')]==0.


def test_off_grid_no_native_self_H_bounds_lock_only_provable_face_errors():
    _,_,native,p,_,_=producer(opposed=False);native=list(native);native[1]=native_bank(90)
    class CaptureBounds:
        def __init__(self):self.lower=self.upper=None
        def score_bounds(self,lower,upper):
            self.lower,self.upper=lower.copy(),upper.copy()
            return lower.sum(-1),upper.sum(-1)
    model=CaptureBounds()
    prepare(p,model,native_ports=native,source_events=[np.array([],int),np.array([90])],check=lambda:None)
    a=np.array([s.source==0 for s in p.states]);b=~a
    for key in ('face_position','face_scale','arm_phase','torso_phase'):
        at=FEATURE_NAMES.index(key)
        assert not model.lower[a,at].any() and not model.upper[a,at].any()
        if key.startswith('face_'):assert np.any(model.upper[b,at]>0)
    # Raw geometric self identity does NOT imply anatomical/actor continuity.
    assert (model.upper[a,FEATURE_NAMES.index('unmeasured_arm_phase_geometry')]==1.).all()
    assert (model.upper[a,FEATURE_NAMES.index('played_pre_angular_activity')]==1.).all()
    for key in ('motion','observed_opposition','unmeasured_motion'):
        assert not model.upper[:,FEATURE_NAMES.index(key)].any()
