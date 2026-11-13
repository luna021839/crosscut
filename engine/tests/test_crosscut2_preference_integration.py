"""Mechanical producer tests; toy votes are NOT actual human media labels."""
import numpy as np
from crosscut2.event_states import build
from crosscut2.event_ports import Ports
from crosscut2.event_holds import select
from crosscut2.preference_model import fit,Model
from crosscut2.fixed_holds import State
from test_crosscut2_event_holds import inputs


def producer(*,opposed,tail=180,one_arm=False):
    _,banks,poses,styles=inputs(opposed_source=1 if opposed else None,one_arm=one_arm)
    nodes=np.array([0,60,120,tail])
    for bank in banks:
        bank['native_indices']=np.clip(nodes[:,None]+[-1,0],0,tail-1)
        bank['pose_native_indices']=np.clip(nodes[:,None]+[-7,-1,0,6],0,tail-1)
    bad=[.003,.8,0.,200.]
    banks[0]['native_readability'][1,1:]=bad;banks[0]['native_readability'][2:]=bad
    banks[1]['native_readability'][0]=bad;banks[1]['native_readability'][1,0]=bad
    kw=dict(fps=30.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2,identity_hints=[np.array([0]),np.array([0])])
    def parts(states,rows):
        result=[]
        for row in rows:
            d=dict(row);d['pixel_part_scope']=np.ones(4,bool)
            d['part_pair_support']=np.ones((4,3,8),bool)
            d['body_reliability']=np.ones((4,4,8));d['motion_reliability']=np.ones((4,2,8));result.append(d)
        return result,[]
    kw['part_observer']=parts
    states,data,_=build(nodes,banks,poses,styles,**kw)
    p=Ports(nodes,states,data,fps=30.,music=[0.,1.,1.,0.])
    a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4)))]*2
    return (nodes,banks,poses,styles),kw,native,p,a,b


def fitted_toy_model():
    examples=[]
    for opposed in (False,True):
        _,_,_,p,a,b=producer(opposed=opposed);f=p.preference_features(1)
        examples.append(dict(physical_group='toy-mechanics-not-media',split='train',supervision_kind='human_av_preference',
            self_reported_normal_av=True,features=[f[a,a].tolist(),f[a,b].tolist(),f[b,b].tolist()],chosen=0 if opposed else 1))
    return Model(fit(examples))


def test_actual_producer_lexicographic_path_cannot_purchase_known_inferior_motion_by_long_visibility(tmp_path):
    model=fitted_toy_model()
    for tail in (180,1800):
        args,kw,native,p,a,b=producer(opposed=True,tail=tail)
        assert p.selected(1,a,b)['opposed_motion']
        runs,_,_,_,receipt,_=select(*args,**kw,native_ports=native,music=[0.,1.,1.,0.],
            preference_model=model,preference_workspace=tmp_path)
        assert len(runs)==1
        assert receipt['objective_contract']=='lexicographic_same_event_preference_visibility_v1'
        assert not receipt['event_preference']['professional_quality_approved']
        assert list(tmp_path.iterdir())==[]  # no accumulating song-sized score files


def test_actual_good_partial_arm_can_beat_blind_hold_in_fitted_preference_path(tmp_path):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    runs,_,data,_,receipt,_=select(*args,**kw,native_ports=native,music=[0.,1.,1.,0.],
        preference_model=fitted_toy_model(),preference_workspace=tmp_path)
    assert len(runs)>1
    assert all(not row['body'][:,:,[1,3,5,6,7],2].any() for row in data)
    assert receipt['event_preference']['source_neural_calls']==0
