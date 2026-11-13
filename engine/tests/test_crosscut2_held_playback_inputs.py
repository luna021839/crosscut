"""Compile-before-history camera/native producer; no timing/AV claims."""
import numpy as np
import pytest
from crosscut2.programme_source_clock import attach,indices
from crosscut2.held_playback_inputs import HoldRecorder
from test_crosscut2_boundary_arc_costs import arc_fixture


def captured(with_native=False):
    nodes,entry,exit,arcs,fragments=arc_fixture(with_native,
        preserve_camera_alternatives=True,capture_playback_inputs=True)
    class Warp:
        def master_to_source(self,t):return t
    attach([c.bank for c in arcs.cats],[Warp()]*len(arcs.cats),
        [dict(fps=30.,frame_count=54)]*len(arcs.cats),nodes,master_start=0.,fps=30.)
    arcs.opportunity_episodes=np.array([0,0,1,2])
    return nodes,entry,exit,arcs


@pytest.mark.parametrize('with_native',[False,True])
def test_all_legal_codes_native_clock_and_camera_values_equal_actual_scored_replay(with_native):
    nodes,entry,exit,arcs=captured(with_native);total=0;codes=set()
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            _,valid=arcs.camera_alternatives(i,j);arena=arcs.last_playback_inputs
            assert (arena.first_node,arena.final_node)==(i,j)
            assert np.array_equal(valid,arena.allowed)
            frames=np.arange(nodes[i],nodes[j])
            for p,q,code in zip(*np.nonzero(valid)):
                native,actual=arena.sample(int(p),int(q),int(code),frames)
                expected=arcs.replay(i,j,int(p),int(q),int(code))
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(native,frames)
                total+=1;codes.add(int(code))
    assert total>0
    if not with_native:assert {0,1,2,3,4,5}<=codes


def test_capture_does_not_recompute_clock_or_camera_for_each_history(monkeypatch):
    nodes,entry,exit,arcs=captured()
    arcs.camera_alternatives(0,3);arena=arcs.last_playback_inputs
    monkeypatch.setattr(arcs,'sample_camera',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('camera rebuilt')))
    monkeypatch.setattr('crosscut2.held_playback_inputs.indices',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('clock rebuilt')))
    for p,q,code in zip(*np.nonzero(arena.allowed)):
        first=arena.sample(int(p),int(q),int(code),np.arange(2,35))
        for _ in range(8):
            another=arena.sample(int(p),int(q),int(code),np.arange(2,35))
            assert all(np.array_equal(a,b) for a,b in zip(first,another))
    # This tests operation ownership; it is not a full-domain speed proof.


def test_frozen_arena_survives_next_hold_without_aliasing_mutable_baselines():
    nodes,_,_,arcs=captured();arcs.camera_alternatives(0,3);old=arcs.last_playback_inputs
    p,q,code=map(int,np.argwhere(old.allowed)[0]);frames=np.arange(54)
    expected=old.sample(p,q,code,frames)
    with pytest.raises(ValueError):arcs.dense[0][0,0,0]=1.2
    with pytest.raises(ValueError):old.allowed[0,0,0]=False
    with pytest.raises(TypeError):old.native[0]=np.zeros(54,int)
    arcs.camera_alternatives(1,2)
    assert old is not arcs.last_playback_inputs
    assert all(np.array_equal(a,b) for a,b in zip(expected,old.sample(p,q,code,frames)))
    with pytest.raises(ValueError,match='already admitted'):old.sample(p,q,code,np.array([54]))


def test_nonlinear_source_clock_keeps_actual_affine_hold_schedule_not_global_reference():
    nodes,_,_,arcs=captured();clock=arcs.cats[0].bank['programme_render_clock']
    clock['source_times']=np.array([.01,.61,1.19,1.76]);clock['count']=100
    # Producer-only mask, not a claim that these changed sensor clocks pass
    # ArcCosts admission. The actual hold schedule is independently asserted.
    mask=np.zeros((len(arcs.entries[0]),len(arcs.exits[3]),6),bool)
    p=next(k for k,v in enumerate(arcs.entries[0]) if v.source==0)
    q=next(k for k,v in enumerate(arcs.exits[3]) if v.source==0);mask[p,q,0]=True
    arena=HoldRecorder(arcs,0,3).seal(mask);frames=np.arange(54)
    native,_=arena.sample(p,q,0,frames)
    exact=indices(.01,(1.76-.01)/(54/30.),frames,fps=30.,native_fps=30.)
    reference=(np.interp(frames,nodes,clock['source_times'])*30.+.5).astype(int)
    assert np.array_equal(native,exact) and not np.array_equal(native,reference)


def test_no_implicit_all_camera_activation_or_guessed_native_clock():
    with pytest.raises(ValueError,match='explicit all-camera'):arc_fixture(False,capture_playback_inputs=True)
    _,_,_,arcs,_=arc_fixture(False,preserve_camera_alternatives=True,capture_playback_inputs=True)
    with pytest.raises(ValueError,match='actual renderer clock'):arcs.camera_alternatives(0,3)


def test_moving_baselines_endpoint_patches_and_partial_cells_keep_exact_camera_bits():
    from test_crosscut2_sequence_preference_consumer import prepared
    from crosscut2.boundary_port_domain import make
    from crosscut2.boundary_arc_costs import ArcCosts
    from crosscut2.played_event_support import PlayedEventSupport
    from crosscut2.event_holds import fragment_costs
    nodes=np.array([0,13,37,38,80,120]);cats=prepared(nodes)
    paths=[]
    for s in range(len(cats)):
        family=[]
        for g in range(4):
            curve=np.stack((1.18+.04*g+.01*np.sin(nodes/17.+s),
                .5+.008*np.sin(nodes/13.+g),.5+.007*np.cos(nodes/19.+s)),axis=1)
            family.append((nodes,curve))
        paths.append(family)
    entry,exit,dense,_=make(nodes,cats,paths,fps=30.,coalesce_static_ports=True)
    events=[np.zeros(0,int)]*len(cats)
    native=[dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[]) for _ in cats]
    arcs=ArcCosts(nodes,entry,exit,cats,dense,native,PlayedEventSupport(events,final=120,fps=30.),
        fragment_costs(nodes,events,fps=30.,support_seconds=.4),fps=30.,opportunity_episodes=np.zeros(len(nodes),int),
        preserve_camera_alternatives=True,capture_playback_inputs=True)
    checked=0
    for i,j in ((0,2),(0,3),(0,5),(1,4),(2,5),(3,4)):
        _,allowed=arcs.camera_alternatives(i,j);arena=arcs.last_playback_inputs
        frames=np.arange(nodes[i],nodes[j]);sparse=frames[(frames%6==1)|(frames%6==4)]
        for p,q,code in zip(*np.nonzero(allowed)):
            p,q,code=map(int,(p,q,code));expected=arcs.replay(i,j,p,q,code)
            _,actual=arena.sample(p,q,code,frames);np.testing.assert_array_equal(actual,expected)
            _,part=arena.sample(p,q,code,sparse);np.testing.assert_array_equal(part,expected[sparse-nodes[i]])
            checked+=1
    assert checked>100


def test_capture_keeps_float32_input_costs_but_returns_actual_float64_replay_values():
    from crosscut2.boundary_arc_costs import ArcCosts
    nodes,entry,exit,old=captured()
    dense=np.asarray(old.dense,np.float32)
    kwargs=dict(fps=30.,preserve_camera_alternatives=True)
    plain=ArcCosts(nodes,entry,exit,old.cats,dense,old.native,old.support,old.fragments,**kwargs)
    cap=ArcCosts(nodes,entry,exit,old.cats,dense,old.native,old.support,old.fragments,
        capture_playback_inputs=True,**kwargs)
    a,aa=plain.camera_alternatives(0,3);b,bb=cap.camera_alternatives(0,3)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(aa,bb)
    for p,q,code in zip(*np.nonzero(bb)):
        p,q,code=map(int,(p,q,code))
        _,camera=cap.last_playback_inputs.sample(p,q,code,np.arange(54))
        assert camera.dtype==np.float64
        np.testing.assert_array_equal(camera,cap.replay(0,3,p,q,code))
    assert dense.flags.writeable  # caller-owned inputs were not frozen/mutated


def test_cross_source_or_wrong_shape_mask_cannot_be_laundered_into_captured_inputs():
    _,entry,exit,arcs=captured()
    bad=np.zeros((len(entry[0]),len(exit[3]),6),bool)
    p=next(k for k,v in enumerate(entry[0]) if v.source==0)
    q=next(k for k,v in enumerate(exit[3]) if v.source==1);bad[p,q,0]=True
    with pytest.raises(ValueError,match='same-source'):HoldRecorder(arcs,0,3).seal(bad)
    with pytest.raises(ValueError,match='same-source'):HoldRecorder(arcs,0,3).seal(bad[:,:,0])


@pytest.mark.parametrize('length',[12,13,14,37,38,157,158])
def test_boundary_lengths_preserve_every_admitted_cost_mask_and_played_camera_bit(length):
    from test_crosscut2_sequence_preference_consumer import prepared
    from crosscut2.boundary_port_domain import make
    from crosscut2.boundary_arc_costs import ArcCosts
    from crosscut2.played_event_support import PlayedEventSupport
    from crosscut2.event_holds import fragment_costs
    nodes=np.array([0,length]);cats=prepared(nodes)
    values=np.stack((1.12+.03*np.sin(nodes/17.),.5+.01*np.sin(nodes/13.),
        .5+.008*np.cos(nodes/19.)),axis=1)
    paths=[[(nodes,values.copy()) for _ in range(4)] for _ in cats]
    entry,exit,dense,_=make(nodes,cats,paths,fps=30.,coalesce_static_ports=True)
    events=[np.zeros(0,int)]*len(cats)
    native=[dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[]) for _ in cats]
    arguments=(nodes,entry,exit,cats,dense,native,PlayedEventSupport(events,final=length,fps=30.),
        fragment_costs(nodes,events,fps=30.,support_seconds=.4))
    plain=ArcCosts(*arguments,fps=30.,opportunity_episodes=np.zeros(2,int),preserve_camera_alternatives=True)
    cap=ArcCosts(*arguments,fps=30.,opportunity_episodes=np.zeros(2,int),
        preserve_camera_alternatives=True,capture_playback_inputs=True)
    before,admitted=plain.camera_alternatives(0,1);after,allowed=cap.camera_alternatives(0,1)
    np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(admitted,allowed)
    np.testing.assert_array_equal(cap.last_playback_inputs.allowed,admitted)
    frames=np.arange(length)
    assert allowed.any()
    for p,q,code in zip(*np.nonzero(allowed)):
        p,q,code=map(int,(p,q,code));ni,actual=cap.last_playback_inputs.sample(p,q,code,frames)
        np.testing.assert_array_equal(ni,frames)
        # Compare bits, not tolerance: even a one-ULP camera difference matters.
        np.testing.assert_array_equal(actual.view(np.uint64),plain.replay(0,1,p,q,code).view(np.uint64))
