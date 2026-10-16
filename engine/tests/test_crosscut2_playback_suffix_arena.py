"""Exact state/lifetime fixtures only; not a new selector or media approval."""
import gc
from itertools import combinations
import numpy as np
import pytest
from crosscut2.playback_input_batch import pack
from crosscut2.playback_workspace import PlaybackWorkspace
from crosscut2.playback_suffix_arena import PlaybackSuffixArena


def arena(*,kernel='numpy',**kwargs):
    if kernel=='numba':pytest.importorskip('numba')
    return PlaybackSuffixArena(namespace=('numeric-model','bound-source-and-observation-registry','bound-intent-registry'),
        embedding_width=2,kernel=kernel,**kwargs)


def descriptor(a,first,final,patch=None):
    ticks=np.arange(max(first,final-a.H),final,dtype=np.int64)
    camera=np.column_stack((1.1+.01*np.sin(ticks/7.),.5+.001*ticks,.5+.001*np.cos(ticks)))
    _,keys=pack(model_binding=('numeric-schema','numeric-model'),frames=ticks,native=ticks,
        cameras=camera,source_observation_handles=np.ones(len(ticks),np.int64),
        intent_handles=np.zeros((len(ticks),2),np.int64))
    if patch is not None:keys=keys.copy();patch(keys)
    values=np.column_stack((ticks/1000.,np.sin(ticks/9.))).astype(np.float64)
    return a.admit(keys,values,first=first)


def extend(a,parent,first,final,patch=None):
    with descriptor(a,first,final,patch) as current:return a.append(parent,current)


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_all_virtual_cell_partitions_have_one_exact_canonical_root(kernel):
    a=arena(kernel=kernel);empty=a.empty();expected=extend(a,empty,0,157)
    # Includes 6k-1/6k/6k+1, H boundaries and a final partial cell.
    cuts=[5,6,7,119,120,121]
    for count in range(len(cuts)+1):
        for split in combinations(cuts,count):
            parent=a.retain(empty);first=0
            for final in (*split,157):
                child=extend(a,parent,first,final);a.release(parent);parent=child;first=final
            assert parent==expected
            a.release(parent)
    with a.decode([expected],end=157) as bank:
        np.testing.assert_array_equal(bank.frames,np.arange(37,157))
        np.testing.assert_array_equal(bank.values[0,:,0],bank.frames/1000.)
    a.release(expected);a.release(empty)
    assert all(t['live']==0 for t in a.receipt()['tables'].values())
    a.close()


def test_whole_4357_frame_partial_cell_and_interleaved_future_ends_are_not_end_local_ids():
    a=arena(kernel='numba');empty=a.empty();whole=extend(a,empty,0,4357);kept=[]
    for final in (119,120,121,4355,4356):
        root=extend(a,empty,0,final);kept.append(root)
        with a.decode([root],end=final) as bank:
            np.testing.assert_array_equal(bank.frames,np.arange(max(0,final-120),final))
    for split in (4235,4236,4237,4355,4356):
        prefix=extend(a,empty,0,split);same=extend(a,prefix,split,4357)
        assert same==whole;a.release(same);a.release(prefix)
    with a.decode([whole],end=4357) as bank:assert bank.frames[-1]==4356
    for root in (*kept,whole,empty):a.release(root)
    a.close()


@pytest.mark.parametrize('column',[1,2,3,4,5])
def test_native_observation_intent_mask_and_one_ulp_camera_are_not_embedding_aliases(column):
    a=arena();empty=a.empty();normal=extend(a,empty,0,120)
    def changed(keys):keys[-1,column]^=np.uint64(1)
    different=extend(a,empty,0,120,changed)
    assert normal!=different  # Even though their synthetic embeddings are identical.
    left=extend(a,normal,120,240);right=extend(a,different,120,240)
    assert left==right  # The actual different frame has now expired from H=120.
    for root in (left,right,normal,different,empty):a.release(root)
    a.close()


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_full_hash_collision_keeps_distinct_inputs_and_releases_zero_reference_keys(kernel):
    a=arena(kernel=kernel,history_frames=13,hash_mask=0,frame_capacity=128,cell_capacity=64,root_capacity=16)
    empty=a.empty();roots=[]
    for variant in range(8):
        root=extend(a,empty,0,13,lambda keys,v=variant:keys.__setitem__((-1,2),v))
        assert root not in roots;roots.append(root)
    for root in reversed(roots):a.release(root)
    assert a.receipt()['tables']['frames']['live']==0
    assert a.receipt()['tables']['cells']['live']==0
    assert a.receipt()['tables']['roots']['live']==1
    a.release(empty);a.close()


def test_arena_namespace_and_reused_slot_generation_cannot_make_old_handles_live():
    a=arena(history_frames=12,root_capacity=2);b=arena(history_frames=12,root_capacity=2)
    empty=a.empty();root=extend(a,empty,0,12);a.release(root)
    replacement=extend(a,empty,0,12)
    assert root!=replacement
    with pytest.raises(ValueError,match='stale'):a.retain(root)
    with pytest.raises(ValueError,match='another'):b.retain(replacement)
    with pytest.raises(ValueError,match='different finalized'):a.decode([replacement],end=11)
    a.release(replacement);a.release(empty);a.close();b.close()


def test_shared_children_and_future_roots_survive_release_of_origin_and_other_roots():
    a=arena();empty=a.empty();past=extend(a,empty,0,120);a.retain(past)
    future=extend(a,past,120,130);a.release(past)
    with a.decode([past],end=120) as bank:assert len(bank.frames)==120
    assert a.receipt()['tables']['frames']['live']==130
    a.release(past)
    assert a.receipt()['tables']['frames']['live']==120
    with a.decode([future],end=130) as bank:
        np.testing.assert_array_equal(bank.frames,np.arange(10,130))
    a.release(future);a.release(empty);a.close()


def test_failed_root_allocation_rolls_back_temporary_cells_and_does_not_drop_existing_beam():
    a=arena(history_frames=12,root_capacity=2,cell_capacity=8,frame_capacity=64)
    empty=a.empty();kept=extend(a,empty,0,12);before=a.receipt()['tables']
    with pytest.raises(MemoryError,match='capacity'):
        extend(a,empty,0,12,lambda keys:keys.__setitem__((-1,2),77))
    after=a.receipt()['tables']
    assert {n:t['live'] for n,t in before.items()}=={n:t['live'] for n,t in after.items()}
    with a.decode([kept],end=12) as bank:assert len(bank.frames)==12
    a.release(kept);a.release(empty);a.close()


def test_frame_admission_failure_is_atomic_and_input_equality_cannot_hide_model_disagreement():
    a=arena(history_frames=12,frame_capacity=13);empty=a.empty();kept=extend(a,empty,0,12)
    with pytest.raises(MemoryError,match='frame capacity'):
        extend(a,empty,0,12,lambda keys:keys.__setitem__((slice(None),2),77))
    before=a.receipt()['tables']['frames']['live']
    with descriptor(a,0,12) as current:
        slot=current._ids
        assert len(slot)==12
    # Same raw key with a different embedding is a producer/model binding error.
    ticks=np.arange(12,dtype=np.int64);camera=np.column_stack((1.1+.01*np.sin(ticks/7.),.5+.001*ticks,.5+.001*np.cos(ticks)))
    _,keys=pack(model_binding=('schema','model'),frames=ticks,native=ticks,cameras=camera,
        source_observation_handles=np.ones(12,np.int64),intent_handles=np.zeros((12,2),np.int64))
    with pytest.raises(ValueError,match='different model embeddings'):a.admit(keys,np.zeros((12,2)))
    assert a.receipt()['tables']['frames']['live']==before
    a.release(kept);a.release(empty);a.close()


def test_small_external_view_stays_charged_until_actual_array_owner_dies():
    workspace=PlaybackWorkspace(2**20)
    a=arena(history_frames=12,frame_capacity=32,cell_capacity=8,root_capacity=4,workspace=workspace)
    empty=a.empty();root=extend(a,empty,0,12);before=workspace.current_bytes
    bank=a.decode([root],end=12);view=bank.values[:,:1];whole_bytes=bank.values.nbytes
    bank.close();gc.collect()
    assert workspace.current_bytes>=before+whole_bytes
    del view;gc.collect();assert workspace.current_bytes==before
    a.release(root);a.release(empty);a.close();gc.collect()
    assert workspace.current_bytes==workspace.bookkeeping_bytes


def test_maximum_40_port_two_bit_eight_label_16_dimensional_prefix_bank_is_one_owned_lease():
    a=PlaybackSuffixArena(namespace=('numeric-maximum-prefix-bank-only',),embedding_width=16,kernel='numba',
        frame_capacity=240,cell_capacity=64,root_capacity=4)
    ticks=np.arange(120,dtype=np.int64)
    _,keys=pack(model_binding=('numeric-schema','numeric-model'),frames=ticks,native=ticks,
        cameras=np.tile([1.,.5,.5],(120,1)),source_observation_handles=np.zeros(120,np.int64),
        intent_handles=np.zeros((120,2),np.int64))
    encoded=np.arange(120*16,dtype=np.float64).reshape(120,16)/10000.
    empty=a.empty()
    with a.admit(keys,encoded) as current:root=a.append(empty,current)
    before=a.workspace.current_bytes
    with a.decode([root]*(40*2*8),end=120) as bank:
        assert bank.values.shape==(640,120,16) and bank.values.nbytes==9830400
        np.testing.assert_array_equal(bank.values[0],encoded)
        np.testing.assert_array_equal(bank.values[-1],encoded)
        assert a.workspace.current_bytes>=before+9830400
    assert a.workspace.current_bytes==before
    a.release(root);a.release(empty);a.close()


def test_budget_refuses_fixed_arena_and_temporary_prefix_before_large_allocation(monkeypatch):
    workspace=PlaybackWorkspace(70*2**10);before=workspace.current_bytes
    with pytest.raises(MemoryError,match='before fixed-array'):
        arena(workspace=workspace,frame_capacity=10000)
    assert workspace.current_bytes==before
    called=[];original=np.empty
    def record(shape,*args,**kwargs):
        if np.prod(shape)>0:called.append(shape)
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(np,'empty',record)
    with pytest.raises(MemoryError,match='before creation'):workspace.empty((100000,),np.float64,owner='too-large')
    assert not called


def test_closed_current_lease_missing_ticks_and_active_arena_close_are_rejected():
    a=arena(history_frames=12);empty=a.empty();current=descriptor(a,0,12);current.close()
    with pytest.raises(ValueError,match='closed arena'):a.append(empty,current)
    with pytest.raises(ValueError,match='live roots'):a.close()
    with pytest.raises(ValueError,match='interval mismatch'):extend(a,empty,1,12)
    a.release(empty);a.close()


@pytest.mark.parametrize('replacements',[2048,4096])
def test_compiled_saturated_beam_replacements_are_bounded_by_live_roots_not_seen_candidates(replacements):
    a=arena(kernel='numba',history_frames=12,frame_capacity=128,cell_capacity=64,root_capacity=16)
    empty=a.empty();beam=[]
    for variant in range(replacements):
        new=extend(a,empty,0,12,lambda keys,v=variant:keys.__setitem__((-1,2),v+1))
        if len(beam)==8:a.release(beam.pop(0))
        beam.append(new)
        if variant%257==0:
            assert a.receipt()['tables']['roots']['live']==len(beam)+1
    receipt=a.receipt()
    assert receipt['tables']['roots']['live']==9 and receipt['tables']['roots']['peak_live']<=10
    assert receipt['tables']['frames']['peak_live']<=21  # Eleven shared + at most ten different final frames.
    assert receipt['tables']['cells']['peak_live']<=11
    for root in beam:a.release(root)
    a.release(empty)
    assert all(t['live']==0 for t in a.receipt()['tables'].values())
    a.close()
