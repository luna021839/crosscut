import numpy as np
import pytest
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.native_fragment_integrity import NativeFragmentIntegrity
from crosscut2.preference_dp import choose


def guard(events,final=100):
    return NativeFragmentIntegrity(PlayedEventSupport(events,final=final,fps=30.),minimum_frames=3)


def test_native_then_authored_one_or_two_frame_flash_is_inadmissible_before_selection():
    g=guard([[44],[],[87]])
    for final in (45,46):
        np.testing.assert_array_equal(g.allowed(12,final,np.array([0,0,1,2])),[False,False,True,True])
    assert g.allowed(12,47,np.array([0]))[0]


def test_authored_then_native_tiny_fragment_is_also_blocked():
    g=guard([[44]])
    assert not g.allowed(42,90,np.array([0]))[0]
    assert not g.allowed(43,90,np.array([0]))[0]
    assert g.allowed(41,90,np.array([0]))[0]


def test_coincident_outside_and_file_endpoint_events_do_not_create_fake_fragments():
    g=guard([[1,44,99]])
    for first,final in ((0,100),(0,44),(44,100),(44,99),(45,98)):
        assert g.allowed(first,final,np.array([0]))[0]
    # This source already contains a short native/native shot. It is not an
    # authored truncation, so the rule does not ban the whole source.
    assert guard([[20,21]]).allowed(0,100,np.array([0]))[0]


def test_unused_sources_and_identical_physical_state_copies_do_not_change_integrity():
    a,b=guard([[44],[]]),guard([[44],[45,46,47]])
    for first,final in ((0,45),(43,70),(44,100)):
        np.testing.assert_array_equal(a.allowed(first,final,np.array([0,0])),b.allowed(first,final,np.array([0,0])))


def test_integrity_is_a_semi_markov_hold_mask_not_an_infinite_or_postretimed_cost():
    nodes=np.array([0,20,45,60]);g=guard([[44],[]],final=60)
    # Without integrity the low-content-loss programme cuts from source0 at45,
    # playing its known native44 for exactly one frame.
    kw=dict(nodes=nodes,fps=30.,unary=np.array([[0.,0.,10.],[10.,10.,0.]]),
        through=np.zeros((2,4)),preference_holds=np.zeros((2,4)),preference_intervals=np.zeros((2,3)),
        preference_ports=lambda n:np.zeros((2,2)),state_sources=np.array([0,1]),source_events=[[44],[]],
        minimum_hold=.4,native_cost=lambda a,b:np.zeros(2),unified_content=True)
    before,_=choose(**kw);assert before==((0,2,0),(2,3,1))
    after,_=choose(**kw,hold_allowed=lambda a,b:g.allowed(int(nodes[a]),int(nodes[b]),kw['state_sources']))
    assert after!=before
    assert all(g.allowed(int(nodes[a]),int(nodes[b]),np.array([s]))[0] for a,b,s in after)


def test_hold_integrity_shape_and_type_are_validated():
    from test_crosscut2_preference_dp import args
    for bad in (np.ones(2),np.ones(3,bool)):
        with pytest.raises(ValueError,match='hold integrity'):choose(**args(),hold_allowed=lambda a,b:bad)


def test_existing_twelve_frame_authored_support_applies_to_new_native_truncations_not_original_file_tails():
    support=PlayedEventSupport([[1,744,799]],final=800,fps=30.)
    g=NativeFragmentIntegrity(support,minimum_frames=12);s=np.array([0])
    assert not g.allowed(700,745,s)[0]
    assert g.allowed(700,744,s)[0] and g.allowed(700,756,s)[0]
    assert not g.allowed(743,780,s)[0]
    assert g.allowed(0,800,s)[0]  # No authored edit made the source's file-edge fragments.


def test_masked_semi_markov_optimum_matches_exhaustive_legal_hold_paths():
    rng=np.random.default_rng(175)
    for _ in range(20):
        nodes=np.arange(5)*12;sources=np.array([0,1,0]);q=3
        g=NativeFragmentIntegrity(PlayedEventSupport([[23,38],[17]],final=48,fps=30.),minimum_frames=12)
        unary=rng.uniform(0,1,(q,4));K=rng.uniform(0,.15,(5,q,q));found=[]
        def walk(first,path,value,tie):
            if first==4:found.append(((value,tie),tuple(path)));return
            for end in range(first+1,5):
                mask=g.allowed(int(nodes[first]),int(nodes[end]),sources)
                for state in range(q):
                    if not mask[state] or (path and sources[path[-1][2]]==sources[state]):continue
                    entry=K[first,path[-1][2],state] if path else 0.
                    walk(end,[*path,(first,end,state)],value+entry+float(unary[state,first:end].sum()*.4),tie+(1/30 if path else 0.))
        walk(0,[],0.,0.)
        runs,score=choose(nodes,fps=30.,unary=unary,through=np.zeros((q,5)),
            preference_holds=np.zeros((q,5)),preference_intervals=np.zeros((q,4)),
            preference_ports=lambda n:K[n],state_sources=sources,source_events=[[23,38],[17]],
            minimum_hold=.4,native_cost=lambda a,b:np.zeros(2),unified_content=True,
            hold_allowed=lambda a,b:g.allowed(int(nodes[a]),int(nodes[b]),sources))
        expected=min(found)
        assert score==pytest.approx(expected[0],abs=1e-12) and runs==expected[1]
