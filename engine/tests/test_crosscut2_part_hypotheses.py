import numpy as np
import pytest
from crosscut2.part_hypotheses import reliabilities
from crosscut2.event_states import build
from crosscut2.event_ports import Ports
from crosscut2.fixed_holds import State
from test_crosscut2_event_holds import inputs


def test_unmeasured_is_not_erased_or_relabelled_as_corroborated():
    positive=np.zeros((2,3,8),bool);negative=positive.copy()
    positive[0,1,4]=True;negative[0,0,5]=True
    phase,motion=reliabilities(positive,negative,scope=np.array([True,False]))
    assert phase[0,0,4]==.5 and phase[0,1,4]==phase[0,2,4]==1.
    assert phase[0,0,5]==0. and motion[0,0,5]==0.
    assert phase[1].min()==1. and motion[1].min()==1.


def test_raw_partial_geometry_can_be_provisional_not_a_certified_connection():
    nodes,banks,poses,styles=inputs(one_arm=True)
    def observer(states,rows):
        result=[]
        for original in rows:
            d=dict(original);d['body_reliability']=np.full((4,4,8),.5)
            d['motion_reliability']=np.full((4,2,8),.5);d['pixel_part_scope']=np.ones(4,bool)
            d['part_pair_support']=np.zeros((4,3,8),bool);result.append(d)
        return result,[]
    states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2,part_observer=observer)
    ports=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.])
    a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
    detail=ports.selected(1,a,b)
    assert detail['provisional_model_connection_candidate'] and not detail['moving_connection_supported']
    assert detail['benefit']>0 and detail['missing_fraction']==pytest.approx(.25)
    assert detail['loss']>0 and detail['phase_evidence_reliability']==.5
    assert np.array_equal(ports.body[a],np.asarray(data[a]['body'],np.float32))


def test_positive_and_contradicted_cannot_be_the_same_measurement():
    support=np.ones((1,3,8),bool)
    with pytest.raises(ValueError):reliabilities(support,support,scope=np.array([True]))


def test_exfat_appledouble_metadata_is_not_unpickled_as_an_observation(tmp_path,monkeypatch):
    from crosscut2 import part_support_bank as bank
    from crosscut2.part_hypotheses import retain_model_hypotheses
    from test_crosscut2_part_support_bank import setup,fake_result
    kw,_=setup(tmp_path)
    def rasters(path,needed,check):
        for f in sorted(needed):yield f,np.zeros((24,40,3),np.uint8)
    monkeypatch.setattr(bank,'_rasters',rasters);monkeypatch.setattr(bank.temporal_parts,'observe',lambda *a,**k:fake_result())
    rows,_=bank.corroborate(**kw)
    (kw['cache']/('._'+'a'*64+'.npz')).write_bytes(b'not a numpy observation')
    result=retain_model_hypotheses(rows=rows,states=kw['states'],sources=kw['sources'],cache=kw['cache'])
    assert np.array_equal(result[0]['body'],rows[0]['raw_body_hypotheses'])
    assert result[0]['body_reliability'][:,:,4].min()==1.
    assert result[0]['body_reliability'][:,:,5].max()==.5
