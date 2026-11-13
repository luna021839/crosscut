from types import SimpleNamespace
import numpy as np
from crosscut2 import person_pair_observer as observer


def world():
    a=SimpleNamespace(source=0,observation=dict(head=np.zeros(4),side=0,native_indices=np.array([4,10])))
    b=SimpleNamespace(source=1,observation=dict(head=np.array([.5,.3,.2,.99]),side=1,native_indices=np.array([11,17])))
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4)),observed=np.array([],bool)) for _ in range(2)]
    bound=[dict(source=s,frame_count=100,native_event_rows=[]) for s in range(2)]
    dense=[[np.tile([1.,.5,.5],(40,1)) for _ in range(4)] for _ in range(2)]
    entries=[(),(b,),()];exits=[(),(a,),()]
    cats=[SimpleNamespace(bank={}) for _ in range(2)]
    return a,b,native,bound,dense,entries,exits,cats


def test_raw_missing_face_pair_requests_known_opposite_person_frame():
    a,b,native,bound,dense,entries,exits,cats=world()
    pairs,count=observer.comparison_pairs(entries,exits,native,bound,dense)
    assert pairs==[((0,10),(1,11))] and count==0
    # A raw source face could be known but cropped out. Port head is UNKNOWN;
    # it must not be replaced with source-native face counts.
    a.observation['raw_source_face_was_known']=True
    assert observer.comparison_pairs(entries,exits,native,bound,dense)[0]==pairs


def test_native_pair_uses_exposed_not_assumed_adjacent_ordinals():
    a,b,native,bound,dense,entries,exits,cats=world()
    native[0]=dict(frames=np.array([15]),heads=np.zeros((1,2,8,4)),observed=np.array([False]))
    bound[0]['native_event_rows']=[dict(output_frame=15,exposed_native_ordinals=[12,15])]
    pairs,count=observer.comparison_pairs(entries,exits,native,bound,dense)
    assert pairs[0]==((0,12),(0,15)) and count==1 and ((0,14),(0,15)) not in pairs


def test_observation_reserves_missing_counterpart_and_binds_exact_frame(monkeypatch):
    a,b,native,bound,dense,entries,exits,cats=world();calls=[]
    boxes=np.array([[.1,.1,.6,.9,.9]],np.float32)
    def measure(**kw):
        calls.append(kw)
        result=[{10:boxes},{}] if len(calls)==1 else [{},{11:boxes}]
        receipts=[dict(reused_frames=int(len(calls)==1 and s==0),new_person_forwards=int(len(calls)==2 and s==1)) for s in range(2)]
        return result,receipts
    monkeypatch.setattr(observer,'observe_person_rectangles',measure)
    run=observer.prepare_pair_observer(sources=['a','b'],bound=bound,cache='unused',model='unused',maximum_new_frames=1)
    receipt=run(nodes=[0,18,36],catalogues=cats,entries=entries,exits=exits,dense=dense,native_ports=native,check=lambda:None)
    assert calls[0]['maximum_new_frames']==0 and calls[1]['requests']==[{}, {11:0}]
    assert receipt['new_person_forwards']==1 and receipt['closed_endpoint_attempt_pairs']==1
    assert a.observation['person_rectangles'] is boxes and b.observation['person_rectangles'] is boxes
    assert cats[0].bank['person_rectangles_by_native'][10] is boxes


def test_failed_other_endpoint_stays_incomplete_not_fake_face_or_extent(monkeypatch):
    a,b,native,bound,dense,entries,exits,cats=world()
    boxes=np.array([[.1,.1,.6,.9,.9]],np.float32)
    def measure(**kw):
        return [{10:boxes},{}],[dict(reused_frames=0,new_person_forwards=0)]*2
    monkeypatch.setattr(observer,'observe_person_rectangles',measure)
    run=observer.prepare_pair_observer(sources=['a','b'],bound=bound,cache='unused',model='unused',maximum_new_frames=1)
    receipt=run(nodes=[0,18,36],catalogues=cats,entries=entries,exits=exits,dense=dense,native_ports=native,check=lambda:None)
    assert receipt['closed_endpoint_attempt_pairs']==0 and not receipt['all_targeted_pairs_attempted']
    assert b.observation['person_rectangles'] is None and receipt['nonempty_raw_rectangle_pairs']==0
