import numpy as np
import pytest
from crosscut2.played_event_support import PlayedEventSupport,NativeHoldCosts
from crosscut2.connection_prior import physical_weights


def test_unused_source_cannot_discount_actual_cut_but_played_short_native_fragment_is_real():
    for unused in ([],[58,59,61,62]):
        support=PlayedEventSupport([[],[],unused],final=120,fps=30.)
        assert support.cut_weights(60,[0,1,2])[0,1]==.2
    support=PlayedEventSupport([[58],[]],final=120,fps=30.)
    assert support.cut_weights(60,[0,1])[0,1]==pytest.approx(1/30)
    costs=NativeHoldCosts(support,[0,1],[[1.],[]])
    assert costs(0,60)[0]==pytest.approx(1/30)


def test_native_single_event_both_end_corrections_exact_overlap_and_state_not_source_domain():
    support=PlayedEventSupport([[58]],final=120,fps=30.)
    costs=NativeHoldCosts(support,[0,0],[[1.],[.5]])
    np.testing.assert_allclose(costs(56,60),[1/30,.5/30])
    assert not costs(0,58).any() and not costs(58,120).any()


def test_source_pair_and_native_hold_decomposition_equal_direct_selected_path():
    rng=np.random.default_rng(174)
    for _ in range(40):
        final=180;points=[30,70,110,150];sources=[0,1,0,2,1]
        runs=list(zip([0,*points],[*points,final],sources))
        events=[sorted(rng.choice(np.arange(1,final),size=15,replace=False).tolist()) for _ in range(3)]
        support=PlayedEventSupport(events,final=final,fps=30.)
        rates=[rng.uniform(0,1,len(xs)) for xs in support.events]
        held=NativeHoldCosts(support,[0,1,2],rates)
        actual=sorted(set(points+[t for a,b,s in runs for t in events[s] if a<t<b]))
        weights=physical_weights(actual,final=final,fps=30.)
        direct=0.;decomposed=0.
        for a,b,s in runs:
            direct+=sum(weights[t]*rates[s][np.searchsorted(support.events[s],t)] for t in events[s] if a<t<b)
            decomposed+=held(a,b)[s]
        assert decomposed==pytest.approx(direct,abs=1e-13)
        for i,t in enumerate(points):
            assert support.cut_weights(t,[0,1,2])[sources[i],sources[i+1]]==pytest.approx(weights[t])
