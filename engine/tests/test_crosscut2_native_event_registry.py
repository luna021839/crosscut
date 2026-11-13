from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.source_event_union import attach_existing_ports
from crosscut2.event_holds import native_events,fragment_costs
from crosscut2.fixed_holds import State


def context(native=60,output=59,observed_native=60):
    event=dict(output_frame=output,source_native_first_incoming=native,
        exposed_native_ordinals=[native-1,native])
    h=np.zeros((1,2,8,4),np.float32);h[0,0,0]=[.2,.3,.2,.95];h[0,1,0]=[.8,.3,.2,.95]
    return dict(bound=[dict(native_event_rows=[event,event.copy()])],
        census=[dict(frames=[observed_native],fps=30.)],
        ports=[dict(frames=np.array([observed_native]),heads=h,vectors=np.zeros((1,2,3,128),np.float32))],
        warps=[SimpleNamespace(source_to_master=lambda x:x)],master_start=0.,fps=30.,final_frame=180)


def row():return dict(ticks=np.array([0,179]),curve=np.array([[1.,.5,.5],[1.,.5,.5]]))


def test_exact_native_ordinal_join_ignores_timestamp_shift_and_deduplicates_event():
    args=context();registry,receipt=attach_existing_ports(**args)
    assert registry[0]['frames'].tolist()==[59]
    assert registry[0]['native_event_keys']==[[60]]
    assert registry[0]['observed'].tolist()==[True]
    assert np.array_equal(registry[0]['heads'],args['ports'][0]['heads'])
    assert receipt[0]['exact_existing_flanks']==1 and receipt[0]['source_nn_calls']==0


def test_transnet_only_event_is_finite_unknown_not_free_or_nearby_substitution():
    registry,receipt=attach_existing_ports(**context(native=60,output=60,observed_native=61))
    assert registry[0]['observed'].tolist()==[False]
    assert not registry[0]['heads'].any() and receipt[0]['explicit_unmeasured_events']==1
    nodes=np.array([0,60,120,180])
    interval,through=native_events(nodes,[State(0,None,'native')],[row()],registry,fps=30.)
    assert interval.sum()==0 and through[0,1]==pytest.approx(.1)
    assert fragment_costs(nodes,[np.array([60])],fps=30.,support_seconds=.4)(0,3)[0]==0
    # A cut at this node replaces the through event, rather than consuming H
    # again inside the outgoing or incoming hold.
    assert through[0,1:1].sum()+through[0,2:3].sum()==0


def test_different_exposed_native_ordinals_cannot_reuse_measured_flanks():
    args=context();args['bound'][0]['native_event_rows'][0]['exposed_native_ordinals']=[58,60]
    args['bound'][0]['native_event_rows']=args['bound'][0]['native_event_rows'][:1]
    registry,_=attach_existing_ports(**args)
    assert not registry[0]['observed'].any()


def test_primary_native_face_is_not_borrowed_for_a_different_actor_hypothesis():
    registry,_=attach_existing_ports(**context(output=60))
    _,through=native_events(np.array([0,60,120,180]),[State(0,None,'native'),State(0,0,'native')],
        [row(),row()],registry,fps=30.)
    assert through[0,1]>.1 and through[1,1]==pytest.approx(.1)
