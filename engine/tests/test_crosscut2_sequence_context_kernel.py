"""Real compiled arithmetic tests; not model fitting or AV quality evidence."""
import numpy as np
import pytest
pytest.importorskip('numba')
from crosscut2.sequence_context import Evidence,PRIMITIVES,FEATURE_NAMES,windows,batched_window_features
from crosscut2.sequence_context_kernel import features,receipt


@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('history',[1,120])
def test_strict_compiled_kernel_matches_independent_reference_with_missing_ticks(sparse,history):
    rng=np.random.default_rng(106);frames=np.arange(241)
    if sparse:frames=frames[rng.uniform(size=len(frames))>.3]
    values=rng.normal(size=(8,len(frames),len(PRIMITIVES)))
    known=rng.uniform(size=values.shape)>.25;ends=np.r_[np.arange(6,241,6),241]
    actual=features(frames,values,known,ends,fps=29.97,context_frames=history)
    for b in range(len(values)):
        expected=windows(Evidence(frames,values[b],known[b]),final_frame=241,fps=29.97,
            context_seconds=history/29.97)['features']
        np.testing.assert_allclose(actual[b],expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(actual,batched_window_features(frames,values,known,ends,fps=29.97,
        context_frames=history),rtol=2e-13,atol=2e-13)
    assert receipt()['nopython_specializations']>0 and not receipt()['fastmath']


def test_empty_unknown_windows_and_empty_query_do_not_fabricate_comfort():
    raw=np.zeros((2,0,len(PRIMITIVES)));known=np.zeros(raw.shape,bool)
    result=features(np.zeros(0,int),raw,known,np.array([6,12]),fps=30.,context_frames=120)
    shaped=result.reshape(2,2,len(PRIMITIVES),9)
    assert (shaped[:,:,:,5]==1.).all()
    shaped=shaped.copy();shaped[:,:,:,5]=0.
    assert not shaped.any()
    assert features(np.zeros(0,int),raw,known,np.zeros(0,int),fps=30.,
        context_frames=120).shape==(2,0,len(FEATURE_NAMES))


def test_invalid_data_never_silently_coerces_or_uses_object_mode():
    raw=np.zeros((1,3,len(PRIMITIVES)));known=np.ones(raw.shape,bool)
    for frames in (np.array([0,0,1]),np.array([2,1,0],dtype=np.uint64),np.array([-1,0,1])):
        with pytest.raises(ValueError):features(frames,raw,known,np.array([3]),fps=30.,context_frames=120)
    bad=raw.copy();bad[0,0,0]=np.nan
    with pytest.raises(ValueError):features(np.arange(3),bad,known,np.array([3]),fps=30.,context_frames=120)
    with pytest.raises(ValueError):features(np.arange(3),raw,known.astype(float),np.array([3]),fps=30.,context_frames=120)


def test_actual_frozen_native_camera_input_matches_reference_without_mutating_it():
    from test_crosscut2_sequence_preference_consumer import context_fixture
    from crosscut2.context_port_dp import ArcRef
    context,_=context_fixture(np.array([0,30,60,90]))
    p=next(k for k,v in enumerate(context.entries[0]) if v.source==0)
    q=next(k for k,v in enumerate(context.exits[-1]) if v.source==0)
    evidence=context._part(ArcRef(0,3,p,q,4),0,90);before=evidence.values.copy()
    ends=np.arange(6,91,6);known=evidence.observed[None]
    actual=features(evidence.frames,evidence.values[None],known,ends,fps=30.,context_frames=120)[0]
    expected=windows(evidence,final_frame=90,fps=30.)['features']
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_array_equal(evidence.values,before)
    assert not evidence.values.flags.writeable


def test_compiled_incremental_all_camera_costs_match_independent_full_reference_replay():
    from test_crosscut2_sequence_preference_consumer import context_fixture
    from crosscut2.context_port_dp import ArcRef
    context,_=context_fixture(np.arange(0,421,30),kernel='numba',feature_weights={
        'feature_center_x.observed_adjacent_mean_rate':.3,
        'additional_log_zoom.observed_adjacent_maximum_rate':.7})
    def ref(i,j,source,code):
        p=next(k for k,v in enumerate(context.entries[i]) if v.source==source)
        q=next(k for k,v in enumerate(context.exits[j]) if v.source==source)
        return ArcRef(i,j,p,q,code)
    for code in range(6):
        first=ref(0,3,1,2);arc=ref(3,14,0,code)
        cost=float(context.incremental([(first,)],arc)[0]);parts,_=context.replay([first,arc])
        assert cost==pytest.approx(parts[-1],rel=1e-12,abs=1e-12)
        assert cost>=context.lower_bound(arc)


def test_changed_kernel_requires_sequence_model_before_any_runtime_or_source_work(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--experimental-sequence-kernel','numba'])
    with pytest.raises(ValueError,match='kernel requires an explicit sequence model'):main()
