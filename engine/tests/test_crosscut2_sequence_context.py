import numpy as np
import pytest
from crosscut2.scene_targets import from_original_view,UNKNOWN
from crosscut2.sequence_context import (Evidence,PRIMITIVES,SUMMARIES,FEATURE_NAMES,
    SCHEMA_SHA256,project,merge,windows,integrate,batched_window_features)
from test_crosscut2_scene_targets import scene
from test_crosscut2_scene_readout import catalogue


def rows(frames,*,value=.2,feature='feature_height'):
    frames=np.asarray(frames,int);values=np.zeros((len(frames),len(PRIMITIVES)))
    known=np.zeros(values.shape,bool);k=PRIMITIVES.index(feature)
    values[:,k]=value;known[:,k]=True
    return Evidence(frames,values,known)


def column(report,primitive,summary):
    return report['features'][:,FEATURE_NAMES.index(primitive+'.'+summary)]


class ConstantNumericFixture:
    # Only a numeric integration fixture, NOT a fitted or activated model.
    sequence_feature_schema=SCHEMA_SHA256
    sequence_window_contract=windows(rows([]),final_frame=1,fps=30.)['window_contract_sha256']
    def costs(self,values):return np.full(len(values),.75)


def test_native_detail_and_known_crop_are_not_changed_to_a_better_visible_actor():
    original=np.tile([[.3,.35,.12,.99],[.7,.35,.12,.99]],(30,1,1))
    target=scene(original[:,:1],original=original)
    a=project(target,np.tile([1.,.5,.5],(30,1)))
    b=project(target,np.tile([2.,.7,.5],(30,1)))
    q=PRIMITIVES.index('native_ROI_readability');crop=PRIMITIVES.index('known_head_crop_fraction')
    np.testing.assert_array_equal(a.values[:,q],b.values[:,q])
    assert a.observed[:,q].all() and b.observed[:,q].all()
    assert (b.values[:,crop]==1.).all() and b.observed[:,crop].all()
    assert (b.values[:,PRIMITIVES.index('feature_center_x')]<0).all()


def test_unmeasured_background_and_anatomy_are_not_fabricated_by_camera_or_role():
    target=scene(np.tile([[[.5,.35,.12,.99]]],(30,1,1)))
    value=project(target,np.tile([1.,.5,.5],(30,1)))
    for name in ('measured_full_frame_change','measured_background_flow','known_body_crop_fraction'):
        assert not value.observed[:,PRIMITIVES.index(name)].any()
    assert not value.values[:,PRIMITIVES.index('body_observed_fraction')].any()
    report=windows(value,final_frame=30,fps=30.)
    assert not column(report,'measured_background_flow','observed_frame_fraction').any()


def test_native_ambiguous_size_ties_have_no_fabricated_feature_motion_channel():
    cat=catalogue(np.array([0,60,120]));cat.bank['heads'][:,:,0]=[.3,.35,.12,.99]
    cat.bank['heads'][:,:,1]=[.7,.35,.12001,.99]
    target=from_original_view(cat,1,1,source_binding='hash')
    assert target.role==UNKNOWN
    value=project(target,np.tile([1.,.5,.5],(len(target.frames),1)))
    assert not value.observed[:,PRIMITIVES.index('feature_center_x')].any()
    assert value.values[:,PRIMITIVES.index('role_unknown')].all()


def test_global_time_intervals_and_scores_are_invariant_to_hold_chunking_and_duplicates():
    full=rows(np.arange(32));parts=[rows(np.arange(i,j)) for i,j in ((0,7),(7,15),(15,23),(23,32))]
    a=windows(full,final_frame=32,fps=30.)
    for merged in (merge(parts),merge(parts*3)):
        b=windows(merged,final_frame=32,fps=30.)
        np.testing.assert_array_equal(a['features'],b['features'])
        np.testing.assert_array_equal(a['interval_final'],[6,12,18,24,30,32])
        assert integrate(a,ConstantNumericFixture())==integrate(b,ConstantNumericFixture())
    assert integrate(a,ConstantNumericFixture())==pytest.approx(.75*32/30.)
    assert a['context_frames']==120 and not a['cut_origin_or_count_used']


def test_sparse_samples_do_not_fill_gaps_or_claim_dense_recovery_exposure():
    report=windows(rows([0,29]),final_frame=120,fps=30.)
    coverage=column(report,'feature_height','observed_frame_fraction')
    assert coverage[-1]==pytest.approx(2/120.)
    assert coverage[0]==pytest.approx(1/6.)
    assert not report['missing_observations_imputed']
    assert column(report,'feature_height','latest_age_fraction')[-1]==pytest.approx(90/120.)


def test_conflicting_duplicates_make_only_the_conflicting_cues_unknown():
    a=rows([0,6,12]);b=rows([6],value=.9)
    other=rows([6],value=1.,feature='role_feature')
    merged=merge([a,b,other]);k=PRIMITIVES.index('feature_height');role=PRIMITIVES.index('role_feature')
    assert merged.frames.tolist()==[0,6,12]
    assert merged.observed[:,k].tolist()==[True,False,True]
    assert merged.observed[:,role].tolist()==[False,True,False]
    assert merged.values[1,k]==0. and merged.values[1,role]==1.
    np.testing.assert_array_equal(merge([other,b,a]).observed,merged.observed)
    nested=merge([merge([a,b]),other,a])
    np.testing.assert_array_equal(nested.observed,merged.observed)
    np.testing.assert_array_equal(nested.conflicted,merged.conflicted)


def test_equal_numeric_cues_for_unassociated_people_cannot_merge_into_one_target():
    a=rows([0,6]);b=rows([0,6])
    a=Evidence(a.frames,a.values,a.observed,('target-A',)*2)
    b=Evidence(b.frames,b.values,b.observed,('target-B',)*2)
    merged=merge([a,b]);k=PRIMITIVES.index('feature_height')
    assert not merged.observed[:,k].any() and merged.conflicted[:,k].all()
    np.testing.assert_array_equal(merge([merged,a]).observed,merged.observed)
    # A digital camera channel has no competing target ownership.
    camera=rows([0,6],value=.2,feature='additional_log_zoom')
    together=merge([merged,camera]);j=PRIMITIVES.index('additional_log_zoom')
    assert together.observed[:,j].all() and not together.conflicted[:,j].any()


def test_one_conflicting_sensor_does_not_erase_the_same_targets_other_cues_on_nested_merge():
    a=rows([0],value=.2);b=rows([0],value=.9);role=PRIMITIVES.index('role_feature')
    value=a.values.copy();known=a.observed.copy();value[:,role]=1.;known[:,role]=True
    a=Evidence(a.frames,value,known,('same-target',))
    value=b.values.copy();value[:,role]=1.
    b=Evidence(b.frames,value,known,('same-target',))
    flat=merge([a,b,a]);nested=merge([merge([a,b]),a])
    np.testing.assert_array_equal(flat.observed,nested.observed)
    np.testing.assert_array_equal(flat.conflicted,nested.conflicted)
    assert nested.observed[0,role] and nested.values[0,role]==1.


def test_empty_observation_context_is_explicitly_unknown_and_finite():
    report=windows(merge([]),final_frame=120,fps=30.)
    assert np.isfinite(report['features']).all()
    channels=report['features'].reshape(-1,len(PRIMITIVES),len(SUMMARIES))
    assert not channels[:,:,:5].any()
    assert (channels[:,:,5]==1.).all()
    assert not channels[:,:,6:].any()


def test_wrong_schema_old_same_event_weights_or_nonfinite_scores_cannot_sneak_in():
    report=windows(rows([0,6]),final_frame=30,fps=30.)
    with pytest.raises(ValueError,match='sequence preference model'):
        integrate(report,object())
    class Bad(ConstantNumericFixture):
        def costs(self,values):return np.full(len(values),np.nan)
    with pytest.raises(ValueError,match='finite contextual cost'):
        integrate(report,Bad())
    assert len(FEATURE_NAMES)==len(PRIMITIVES)*len(SUMMARIES)
    assert not any(name.startswith(('source_id','song_timestamp','cut_count')) for name in FEATURE_NAMES)
    other_context=windows(rows([0,6]),final_frame=30,fps=30.,context_seconds=2.)
    with pytest.raises(ValueError,match='sequence preference model'):
        integrate(other_context,ConstantNumericFixture())


def test_evidence_arrays_do_not_mutate_the_shared_physical_inputs():
    original=rows([0,6]);values=original.values.copy();known=original.observed.copy()
    evidence=Evidence(original.frames,values,known);values[:]=.9;known[:]=False
    assert evidence.observed.any() and evidence.values.max()==.2
    with pytest.raises(ValueError):evidence.values[:]=0.
    with pytest.raises(ValueError,match='unique chronological'):
        Evidence(np.array([0,0]),np.zeros((2,len(PRIMITIVES))),np.zeros((2,len(PRIMITIVES)),bool))
    unsigned=Evidence(np.array([],dtype=np.uint64),np.zeros((0,len(PRIMITIVES))),np.zeros((0,len(PRIMITIVES)),bool))
    assert unsigned.frames.dtype==np.int64
    assert np.isfinite(windows(unsigned,final_frame=1,fps=30.)['features']).all()


def test_global_interval_subset_is_identical_including_partial_final_interval_and_empty_request():
    evidence=rows(np.arange(43),value=.37);whole=windows(evidence,final_frame=43,fps=29.97)
    for first,last in ((0,12),(12,39),(39,43),(43,43),(5,5)):
        subset=windows(evidence,final_frame=43,fps=29.97,score_after_frame=first,score_until_frame=last)
        mask=(whole['interval_final']>first)&(whole['interval_final']<=last)
        for name in ('features','interval_first','interval_final','interval_seconds'):
            np.testing.assert_array_equal(subset[name],whole[name][mask])
    assert whole['interval_seconds'][-1]==pytest.approx(1/29.97)


def test_disjoint_fast_merge_matches_duplicate_reduction_with_all_masks_scopes_and_conflicts():
    rng=np.random.default_rng(63);parts=[]
    for frames in (np.arange(20,40),np.arange(0,10),np.arange(10,20)):
        values=rng.normal(size=(len(frames),len(PRIMITIVES)))
        known=rng.uniform(size=values.shape)>.2;bad=(rng.uniform(size=known.shape)>.8)&~known
        parts.append(Evidence(frames,values,known,tuple('owner-'+str(k) for k in frames),bad))
    fast=merge(parts);slow=merge(parts*2)
    for name in ('frames','values','observed','conflicted'):
        np.testing.assert_array_equal(getattr(fast,name),getattr(slow,name))
    assert fast.target_scopes==slow.target_scopes
    for a,b in ((merge([fast]),fast),(merge([parts[0]]),merge([parts[0],parts[0]]))):
        for name in ('frames','values','observed','conflicted'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_repeated_camera_oscillation_is_not_aliased_by_equal_means_ranges_and_latest_values():
    frames=np.arange(8);values=np.zeros((8,len(PRIMITIVES)));known=np.zeros(values.shape,bool)
    k=PRIMITIVES.index('additional_log_zoom');known[:,k]=True
    values[:,k]=[0.,0.,0.,0.,.1,.1,.1,.1]
    smooth=windows(Evidence(frames,values,known),final_frame=8,fps=30.,interval_frames=8)
    values[:,k]=[0.,.1,0.,.1,0.,.1,0.,.1]
    repeated=windows(Evidence(frames,values,known),final_frame=8,fps=30.,interval_frames=8)
    for name in ('observed_mean','observed_minimum','observed_maximum',
                 'observed_frame_fraction','latest_observed_value','latest_age_fraction'):
        np.testing.assert_allclose(column(smooth,'additional_log_zoom',name),column(repeated,'additional_log_zoom',name))
    assert not np.array_equal(smooth['features'],repeated['features'])
    assert column(repeated,'additional_log_zoom','observed_adjacent_mean_rate')[0]==pytest.approx(3.)
    assert column(smooth,'additional_log_zoom','observed_adjacent_mean_rate')[0]==pytest.approx(3/7.)


def test_temporal_rates_do_not_bridge_missing_or_conflicting_observations():
    frames=np.array([0,1,10]);values=np.zeros((3,len(PRIMITIVES)));known=np.zeros(values.shape,bool)
    k=PRIMITIVES.index('additional_log_zoom');known[:,k]=True;values[:,k]=[0.,.1,2.]
    report=windows(Evidence(frames,values,known),final_frame=11,fps=30.,interval_frames=11)
    assert column(report,'additional_log_zoom','observed_adjacent_mean_rate')[0]==pytest.approx(3.)
    assert column(report,'additional_log_zoom','observed_adjacent_maximum_rate')[0]==pytest.approx(3.)
    assert column(report,'additional_log_zoom','observed_adjacent_pair_fraction')[0]==pytest.approx(.1)
    known[1,k]=False;bad=np.zeros(known.shape,bool);bad[1,k]=True
    report=windows(Evidence(frames,values,known,conflicted=bad),final_frame=11,fps=30.,interval_frames=11)
    for name in ('observed_adjacent_mean_rate','observed_adjacent_maximum_rate','observed_adjacent_pair_fraction'):
        assert column(report,'additional_log_zoom',name)[0]==0.


@pytest.mark.parametrize('frames',[np.arange(191),np.array([0,3,8,12,54,72,170,185]),np.zeros(0,int)])
def test_batched_actual_prefix_kernel_equals_independent_single_stream_with_sparse_gaps(frames):
    rng=np.random.default_rng(89);values=rng.normal(size=(5,len(frames),len(PRIMITIVES)))
    known=rng.uniform(size=values.shape)>.15;ends=np.r_[np.arange(6,191,6),191]
    actual=batched_window_features(frames,values,known,ends,fps=29.97,context_frames=120)
    for b in range(len(values)):
        full=windows(Evidence(frames,values[b],known[b]),final_frame=191,fps=29.97)
        np.testing.assert_allclose(actual[b],full['features'],rtol=1e-13,atol=1e-13)
    assert batched_window_features(frames,values,known,np.zeros(0,int),fps=29.97,
        context_frames=120).shape==(5,0,len(FEATURE_NAMES))
